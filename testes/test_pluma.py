"""Testes de propriedades físicas, geometria e integridade dos resultados."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pluma_revap as p


class PromptsAndMetrics(unittest.TestCase):
    def test_normalized_coordinates_non_square(self):
        box, pts, labels = p.prompt_arrays({"box": [0, 0, 1, 1], "positive": [[1, 0.5]], "negative": [[0, 0]]}, (101, 201, 3))
        np.testing.assert_allclose(box, [0, 0, 200, 100])
        np.testing.assert_allclose(pts, [[200, 50], [0, 0]])
        np.testing.assert_array_equal(labels, [1, 0])

    def test_invalid_prompt_rejected(self):
        for spec in [{"box": [1, 0, 0, 1]}, {"box": [0, 0, 1, 1], "positive": [[-0.1, 0]]}]:
            with self.assertRaises(ValueError):
                p.prompt_arrays(spec, (100, 200))

    def test_candidates_use_prompt_consistency(self):
        masks = np.zeros((2, 10, 10), bool)
        masks[0, 0, 0] = True  # maior score, mas viola positivo/negativo
        masks[1, 5, 5] = True
        selected, rows = p.select_candidate(masks, [0.99, 0.7], np.array([[5, 5], [0, 0]]), np.array([1, 0]), np.array([0, 0, 9, 9]))
        self.assertEqual(selected, 1)
        self.assertEqual(rows[1]["consistencia_prompts"], 1)

    def test_cleanup_preserves_seeded_island_and_hole(self):
        mask = np.zeros((30, 30), bool)
        mask[2:15, 2:15] = True
        mask[6:9, 6:9] = False
        mask[25, 25] = True
        mask[22, 22] = True
        clean = p.clean_seeded(mask, np.array([[25, 25]]), np.array([1]))
        self.assertTrue(clean[25, 25])
        self.assertFalse(clean[22, 22])
        self.assertFalse(clean[7, 7])
        self.assertTrue(np.all(clean <= mask))

    def test_metrics_known_confusion(self):
        ref = np.array([[1, 1], [0, 0]], bool)
        mask = np.array([[1, 0], [1, 0]], bool)
        r = p.segmentation_metrics(mask, ref)
        self.assertAlmostEqual(r["iou_ref_aprox"], 1 / 3)
        self.assertAlmostEqual(r["dice_ref_aprox"], 0.5)
        self.assertEqual(r["fp_ref_aprox_px"], 1)
        valid = np.array([[1, 0], [0, 1]], bool)
        self.assertEqual(p.segmentation_metrics(mask, ref, valid)["iou_ref_aprox"], 1)

    def test_geometry_in_pixels(self):
        mask = np.zeros((30, 50), bool)
        mask[10:15, 5:40] = True
        r = p.geometry(mask)
        self.assertEqual(r["area_px"], 175)
        self.assertAlmostEqual(r["orientacao_imagem_graus_mod180"], 0)
        self.assertAlmostEqual(r["eixo_maior_px"], 34)


class WindAndPhysics(unittest.TestCase):
    @staticmethod
    def wind(u=(2., 2.), v=(0., 0.)):
        return pd.DataFrame({"time_utc": pd.date_range("2019-09-29T18:00:00Z", periods=2, freq="h"), "u10": u, "v10": v})

    def test_cardinal_conventions_and_calm(self):
        speed, math_angle, toward, from_ = p.wind_quantities([1, 0, -1, 0, 0], [0, 1, 0, -1, 0])
        np.testing.assert_allclose(math_angle[:4], [0, 90, 180, 270])
        np.testing.assert_allclose(toward[:4], [90, 0, 270, 180])
        np.testing.assert_allclose(from_[:4], [270, 180, 90, 0])
        self.assertTrue(np.isnan(from_[-1]))

    def test_exact_integral_constant_wind(self):
        r = p.integrate_wind(self.wind(), pd.to_datetime(["2019-09-29T18:30:00Z", "2019-09-29T19:00:00Z"]))
        np.testing.assert_allclose(r, [[3600, 0], [7200, 0]])

    def test_exact_integral_linear_wind(self):
        r = p.integrate_wind(self.wind(u=(0., 4.)), pd.to_datetime(["2019-09-29T18:30:00Z", "2019-09-29T19:00:00Z"]))
        np.testing.assert_allclose(r[:, 0], [1800, 7200])

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            p.integrate_wind(self.wind(), pd.to_datetime(["2019-09-29T19:01:00Z"]))

    def test_gaussian_mass_symmetry_and_diffusion(self):
        x = np.linspace(-4000, 4000, 401)
        no_wind = self.wind(u=(0., 0.))
        c, centers, mass = p.gaussian_puffs(no_wind, "2019-09-29T18:00:00Z", "2019-09-29T19:00:00Z", x, x, diffusivity=5., rate=2.)
        measured = np.trapezoid(np.trapezoid(c, x=x, axis=1), x=x)
        self.assertAlmostEqual(measured / mass, 1., places=5)
        np.testing.assert_allclose(c, c[::-1], atol=1e-12)
        np.testing.assert_allclose(centers, 0)
        xx, yy = np.meshgrid(x, x)
        moment = np.trapezoid(np.trapezoid(c * (xx ** 2 + yy ** 2), x=x, axis=1), x=x) / measured
        # Idade média = T/2: <r²> = 2 sigma0² + 2 K T.
        self.assertAlmostEqual(moment, 2 * 100 ** 2 + 2 * 5 * 3600, delta=1)

    def test_received_era5_all_times_preserved(self):
        ds, wind = p.load_wind(p.ROOT / "dados/Vento_ERA5_REVAP_2019.nc")
        self.assertEqual(dict(ds.sizes), {"time": 8, "latitude": 5, "longitude": 5})
        self.assertEqual(len(wind), 16)
        self.assertEqual(wind.time_utc.nunique(), 8)
        self.assertEqual(wind.time_local.dt.hour.tolist()[:8], list(range(15, 23)))
        self.assertTrue(np.isfinite(wind[["u10", "v10"]]).all().all())
        with self.assertRaises(ValueError):
            p.load_wind(p.ROOT / "dados/Vento_ERA5_REVAP_2019.nc", lat=0)


class DeliveredOutputs(unittest.TestCase):
    def test_original_hashes_and_prompt_bounds(self):
        manifest = json.loads((p.ROOT / "dados/manifesto.json").read_text())
        for name, info in manifest["arquivos"].items():
            self.assertEqual(p.sha256(p.ROOT / name), info["sha256"])
        for image_id, item in p.load_config()["imagens"].items():
            rgb = p.read_rgb(p.ROOT / item["arquivo"])
            for spec in p.configurations(item).values():
                p.prompt_arrays(spec, rgb.shape)

    def test_real_sam_export_integrity(self):
        table_path = p.OUT / "metricas_segmentacao.csv"
        if not table_path.exists():
            self.skipTest("Execute primeiro o notebook ou pluma_revap.py para validar resultados reais.")
        table = pd.read_csv(table_path)
        self.assertEqual(len(table), 15)
        self.assertTrue(np.isfinite(table[["iou_ref_aprox", "score_interno_sam"]]).all().all())
        for image_id, item in p.load_config()["imagens"].items():
            rgb = p.read_rgb(p.ROOT / item["arquivo"])
            for name in [*p.configurations(item), "P5_final_limpeza_minima"]:
                mask = np.asarray(Image.open(p.OUT / image_id / f"{name}_binaria.png"))
                self.assertEqual(mask.shape, rgb.shape[:2])
                self.assertTrue(set(np.unique(mask)) <= {0, 255})
                self.assertGreater(np.sum(mask == 255), 100)
                self.assertLess(np.mean(mask == 255), .85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
