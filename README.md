# REVAP: segmentação de fumaça com SAM e vento ERA5

**Eduardo Nascimento — RA 168800 · Rebeka Wolpi — RA 176598**
Modelagem Computacional · Profa. Raquel Lobosco · UNIFESP

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EduNascimento14/ModelagemComputacional/blob/main/Projeto_REVAP_SAM_ERA5.ipynb)

**Arquivo para entrega:** [Projeto_REVAP_SAM_ERA5.ipynb](Projeto_REVAP_SAM_ERA5.ipynb). O notebook contém código, figuras e tabelas de uma execução real em CPU. Abra no Colab e use **Ambiente de execução → Executar tudo**; GPU T4 é opcional. Não exige Drive, chave API ou cadastro no Copernicus. Pesos SAM oficiais (~375 MB) são baixados e verificados automaticamente; não estão versionados aqui.

O estudo usa as três fotografias e o recorte ERA5 enviados com a proposta do incêndio da REVAP de 29/09/2019. O código compara quatro configurações de prompts **para cada foto**, com original, máscara binária e sobreposição lado a lado:

1. Caixa ampla, sem pontos.
2. Caixa ajustada e pontos positivos distribuídos na fumaça.
3. Caixa ajustada, positivos e negativos em céu, vegetação e estruturas.
4. Refinamento iterativo com os logits de P3 e os mesmos pontos.

Uma quinta linha apresenta a máscara final com limpeza mínima de ilhas pequenas; máscaras brutas e candidatas SAM permanecem auditáveis. P2 muda caixa e pontos ao mesmo tempo: o experimento compara configurações completas e não isola causalmente o efeito de cada mudança. O notebook inclui um editor opcional por cliques no Colab, desativado por padrão para não bloquear a execução automática.

## Resultados e avaliação

[Relatório com discussão, tabelas e figuras](resultados/relatorio.md) · [Métricas CSV](resultados/metricas_segmentacao.csv) · [Candidatas SAM](resultados/candidatos_sam.csv)

A caixa ampla pode incluir grande parte do céu, especialmente na segunda foto. Pontos e refinamento corrigem boa parte dessa falha; a terceira foto já funciona bem com caixa e não ganha necessariamente com mais prompts. Há análise explícita de iluminação, contraste, fumaça versus nuvens, oclusões, céu entre filamentos e partes fora do enquadramento.

As referências poligonais são **aproximações por inspeção visual assistida por IA**, traçadas antes da inferência e sem revisão humana independente. Não são ground truth e não são usadas para recortar as previsões ou selecionar candidatas. As métricas ignoram uma faixa de borda de 1,2% da menor dimensão. IoU/Dice medem acordo com essa referência aproximada, não acurácia validada; o score interno SAM também não é acurácia medida. Os prompts foram desenvolvidos nas mesmas três imagens usadas na avaliação, sem conjunto externo de teste.

As medidas são em pixels. Sem escala, orientação/posição da câmera e horário de cada foto não há conversão confiável para metros nem correlação entre o ângulo na imagem e o azimute ERA5. Não se garante máscara perfeita de fumaça translúcida.

## Vento e solução de transporte

O NetCDF recebido contém u10 e v10 juntos, em m/s, com 8 horários e grade 5×5. O código preserva todas as horas, exporta a grade para CSV e compara vizinho mais próximo com interpolação bilinear na localização representativa da refinaria. Distingue ângulo desde leste, azimute de transporte e direção meteorológica; converte UTC para `America/Sao_Paulo` e proíbe extrapolação.

Uma extensão resolve advecção–difusão 2D pela convolução da solução de Green gaussiana, com vento uniforme variável no tempo. A integral do vento linear por partes é exata; a emissão é integrada por ponto médio. Emissão unitária, K=20 m²/s e largura inicial 100 m são **hipóteses ilustrativas**, sem calibração do incêndio. Verificam-se balanço de massa e convergência temporal. O cenário não estima concentração real, ascensão térmica ou CFD 3D, e não é usado para validar as máscaras das fotografias.

## Organização

| Caminho | Conteúdo |
|---|---|
| `Projeto_REVAP_SAM_ERA5.ipynb` | Entrega principal, com saídas executadas e equações explicadas |
| `pluma_revap.py` | Mesmas funções em módulo Python executável |
| `dados/imagens/` | Originais preservados byte a byte, com nomes simplificados |
| `dados/prompts.json` | Caixas, positivos, negativos, referência aproximada e justificativas |
| `dados/manifesto.json` | SHA-256 e tamanhos dos dados de entrada |
| `dados/Proposta_REVAP_2019.pdf` | Proposta recebida |
| `dados/Vento_ERA5_REVAP_2019.nc` | Recorte ERA5 original |
| `resultados/revap_*/` | Comparações, avaliação, máscaras binárias e sobreposições finais |
| `resultados/` | Relatório, CSV, cenário físico e registros de validação |
| `testes/test_pluma.py` | 14 testes de propriedades, dados e resultados reais |

A execução também gera localmente `resultados_revap.zip` com todas as saídas. No final do notebook há a chamada de download opcional.

## Executar localmente

Python 3.12 foi usado na validação. Crie um ambiente virtual e instale primeiro um par compatível de PyTorch e torchvision. Exemplo CPU:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python pluma_revap.py
python -m unittest discover -s testes -v
```

Execute na raiz do repositório. O Colab já traz PyTorch/torchvision; a célula inicial preserva esse par. As funções estão nas próprias células do notebook, sem depender de importar um módulo remoto mutável.

## Validação e limites

- Inferência real do SAM oficial ViT-B nas três fotos, com quatro configurações e três candidatas por configuração.
- 14 testes de coordenadas, métricas, componentes, convenções de vento, integração analítica, massa, simetria, difusão, dados ERA5 e integridade das máscaras.
- Células de código executadas sequencialmente em IPython local/CPU; [registro da execução](resultados/validacao_notebook.json), [ambiente](resultados/ambiente_execucao.json) e [testes](resultados/testes_execucao.txt).
- Editor por cliques é específico do Colab: sua sintaxe foi revisada; a interação no navegador Colab não foi validada nesta execução local. Execução GPU também não foi testada aqui.

## Fontes

O [notebook inicial da professora](https://github.com/rjlobosco/modelagem_computacional/blob/main/Projeto_Pluma_ERA5_Colab.ipynb) orienta a parte ERA5. Segmentação: [SAM oficial](https://github.com/facebookresearch/segment-anything), dependência `segment-anything==1.0`, checkpoint ViT-B com SHA-256 no código. Referência científica: [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643). Dados: [ERA5/C3S](https://doi.org/10.24381/cds.adbb2d47), cópia pública [ARCO-ERA5](https://github.com/google-research/arco-era5), conforme metadados do NetCDF recebido.

A proposta recebida documenta o caso e as fontes jornalísticas. Fotografias fornecidas pelo estudante, sem créditos/URLs/horários verificáveis; sua associação ao caso é a informada pelo estudante. A coordenada representa a refinaria e não um tanque confirmado. Código, prompts e referência aproximada foram elaborados com assistência de IA; a revisão e interpretação acadêmica cabem aos integrantes.
