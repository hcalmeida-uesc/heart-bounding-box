# Heart Bounding Box — Extração Determinística de ROI Cardíaca

Fork do pipeline **Coronary Centerline Detection** (Silva et al., 2026) adaptado para um objetivo
diferente do original: em vez de segmentar as artérias coronárias, este projeto reaproveita as
etapas de pré-processamento desse pipeline para produzir um **bounding box 3D determinístico ao
redor do coração**, usado como etapa de pré-processamento no Capítulo 2 da tese de doutorado de
Hélder Almeida (IPRJ/UERJ).

## 📋 Contexto

A tese trata de segmentação de artérias coronárias em CTA usando uma ROI cardíaca determinística
como pré-processamento (Capítulo 2), dentro de um pipeline maior de reprodução de benchmark
(Capítulo 1) e integração ROI+segmentação (Capítulo 3). O Capítulo 2 se apoia no trabalho de
mestrado do mesmo grupo de pesquisa — "Coronary Artery Segmentation in CCTA Volumes via Aortic
Localization and Modified Vesselness-Driven Region Growing" (Silva et al., 2026, Apêndice A da
tese) — cujo código é a base deste repositório.

O pipeline do artigo segmenta as artérias coronárias. **O objetivo deste projeto é mais restrito**:
apenas isolar o coração e derivar um bounding box 3D fixo, sem segmentar a aorta nem a árvore
arterial. As etapas iniciais dos dois pipelines (redução dimensional, limiarização HU, extração do
maior componente conectado) são as mesmas — por isso o código já implementado aqui é reaproveitado
em vez de reescrito do zero.

Um mapeamento detalhado de onde cada etapa reaproveitável está implementada (arquivo:linha,
assinatura, convenção de eixos, riscos) está em
[docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md).

## 🎯 Objetivo

Implementar, a partir do código herdado, a sequência descrita na Seção 2.3.1 do Capítulo 2:

1. Redução dimensional (fator 2x, bicúbica, preserva N slices) — **já implementado** no pipeline herdado.
2. Limiarização HU adaptativa (−300 HU fixo / percentil 99,7 adaptativo) — **já implementado**.
3. Extração do maior componente conectado (LCC) por fatia axial — **já implementado**.
4. Empilhamento dos slices processados em um volume cardíaco 3D binário — **já implementado**
   (resultado direto das etapas 1-3).
5. Cálculo do centróide geométrico do volume cardíaco (média das coordenadas dos voxels, por eixo) — **a implementar**.
6. A partir das 1000 imagens do ImageCAS, identificação das dimensões máximas observadas nos três
   eixos — bounding box de **tamanho fixo** (envelope conservador) — **a implementar**.
7. Posicionamento do bounding box centralizado no centróide de cada imagem individual — **a implementar**.
8. Aplicação do bounding box sobre a imagem **original em resolução integral** (não a versão
   reduzida usada para o cálculo) — **a implementar**.
9. Validação: cobertura de 100% dos voxels de artéria coronária rotulados dentro do bbox (métrica
   binária por imagem) + cálculo da redução volumétrica média `Rv = 1 − (V_ROI / V_original)` — **a implementar**.

As etapas 5-9 **não segmentam a aorta nem realizam detecção de óstios/artérias** — todo o restante
do pipeline herdado (Hough Circle Transform, Level Set, Frangi/vesselness, Region Growing) existe
no repositório mas é **fora do escopo** deste objetivo e não precisa ser executado para gerar o
bounding box.

## 🏗️ Estrutura do Projeto

```file
heart-bounding-box/
├── pyproject.toml                  # Configuração e dependências do projeto
├── README.md                       # Este arquivo
├── docs/
│   └── CODE_NOTES_bounding_box.md  # Mapeamento detalhado do código reaproveitável + plano de implementação
├── doc/                             # Documentação herdada do pipeline original
├── config/
│   ├── pipeline_config.json        # Config default do pipeline herdado (resolução MID, factor 2x)
│   └── imagecas_splits*.json       # Splits fixos train/val/test do ImageCAS
├── src/
│   ├── segmentation_pipeline.py    # Entry point do pipeline herdado (segmentação arterial completa)
│   ├── main.ipynb                  # Notebook de execução do pipeline herdado
│   ├── eda/                        # Notebooks de análise exploratória
│   ├── experiments/                # Scripts/notebooks de sweep de parâmetros
│   └── utils/
│       ├── processing/
│       │   ├── preprocessing.py    # downscale, threshold, LCC — núcleo reaproveitável p/ bounding box
│       │   ├── frangi.py           # Filtro de Frangi (não usado no bounding box)
│       │   ├── binary_operations.py
│       │   └── gpu_utils.py
│       ├── segmentation/
│       │   ├── pipeline_preprocessing.py  # load_and_preprocess_image() — orquestra etapas 1-4
│       │   ├── lower_threshold.py         # limiar inferior HU (fixed/percentile/...)
│       │   ├── aorta_localization.py      # Hough Circle Transform (não usado no bounding box)
│       │   ├── aorta_segmentation.py      # Level Set (não usado no bounding box)
│       │   ├── ostia_detection.py         # detecção de óstios (não usado no bounding box)
│       │   └── artery_segmentation.py     # region growing (não usado no bounding box)
│       ├── project/
│       │   ├── dataset.py          # get_data_splits() — splits fixos, útil para rodar sobre as 1000 imagens
│       │   ├── config.py
│       │   └── results.py
│       └── utils/
│           ├── nifti_io.py         # I/O NIfTI (nibabel)
│           └── roi.py              # extração de ROI retangular/circular (atenção: eixos nomeados de forma diferente do resto do código)
└── output/                          # Resultados e artefatos de execução (pipeline herdado)
    └── segmentation/
        ├── runs/                   # Execuções completas por resolução/timestamp
        ├── canonical/              # Referências estáveis para EDA/comparações
        └── analysis/               # Comparações, bad cases e exemplos visuais
```

## 🔬 Pipeline herdado (código-base reaproveitado)

As subseções abaixo descrevem o pipeline **original** de segmentação coronária tal como
implementado neste repositório. As etapas 1 (pré-processamento) são a base direta do bounding box;
as etapas 2-7 (vesselness, aorta, óstios, artérias) permanecem no código como referência e para
comparações futuras (Capítulo 3), mas não são executadas para gerar o bounding box do Capítulo 2.

### 1. Pré-processamento (reaproveitado no bounding box)

#### 1.1 Downsampling

- Redução da imagem por fator 2 em x/y (interpolação linear por padrão, cúbica/bicúbica configurável)
- Fatores de redução: `(2, 2, 1)` — reduz x e y, mantém z
- Implementado em `downscale_image_opencv` / `downscale_image_ndi` ([src/utils/processing/preprocessing.py](src/utils/processing/preprocessing.py))

#### 1.2 Threshold Dinâmico Adaptativo

- **Limite inferior**: `-300 HU` fixo (configurável via `resolve_lower_threshold`)
- **Limite superior**: percentil 99.7 da imagem (adaptativo)
- Implementado em `threshold_image_with_offset` ([src/utils/processing/preprocessing.py](src/utils/processing/preprocessing.py))

#### 1.3 Extração do Maior Componente Conectado (LCC)

- Aplicação **slice-by-slice** (2D) para cada fatia axial (`LCC_PER_SLICE=true` por padrão)
- Remove regiões isoladas, pulmões e vasos pulmonares; mantém apenas a maior estrutura conectada (coração e grandes vasos)
- Implementado em `largest_connected_component` ([src/utils/processing/preprocessing.py](src/utils/processing/preprocessing.py))

### 2-7. Vesselness, localização/segmentação da aorta, óstios e artérias (fora do escopo do bounding box)

O restante do pipeline herdado — mapa de vesselness (Frangi), Hough Circle Transform + Level Set
para a aorta, detecção de óstios e Region Growing 3D para as artérias — segue implementado e
funcional no repositório, documentado no README original do pipeline de centerline. Para detalhes
de cada uma dessas etapas (parâmetros, arquivos, assinaturas), ver
[docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md), Seção 2.

## 🚀 Instalação

### Requisitos

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes recomendado)

### Instalação das Dependências

```bash
# Instale as dependências usando uv (recomendado)
uv sync

# Ou usando pip
pip install -e .
```

### Dependências Principais

- `scikit-image`: Processamento de imagens (downscale, threshold, LCC, Frangi, Hough, Level Set)
- `nibabel`: Leitura de arquivos NIfTI
- `opencv-python`: Downsampling / Transformada de Hough
- `scipy`: `ndimage` (zoom, label, morfologia)
- `pandas`: Análise de dados / persistência de resultados
- `scikit-learn`: Divisão train/test

## 💻 Uso

### Pipeline herdado (segmentação coronária completa)

```python
from segmentation_pipeline import process_image, run_pipeline

# Processar uma imagem específica
result = process_image(img_id=1, config=CONFIG, base_path=BASE_PATH, base_save_path=BASE_SAVE_PATH)

# Processar múltiplas imagens em lote
ids = [1, 2, 3, 4, 134, 195]
results = run_pipeline(ids, split_name="train", config=CONFIG, base_path=BASE_PATH, base_save_path=BASE_SAVE_PATH)
```

Ou via CLI:

```bash
python src/segmentation_pipeline.py --split train
```

### Extração de bounding box cardíaco (planejado, ainda não implementado)

A extração de bounding box (etapas 5-9 da Seção "Objetivo") ainda não tem um script/entry point
próprio. O plano de implementação proposto — reaproveitando `downscale_image_opencv/ndi`,
`resolve_lower_threshold`, `threshold_image_with_offset` e `largest_connected_component`, e
adicionando cálculo de centróide, envelope máximo, posicionamento e validação de cobertura/Rv —
está descrito em detalhe em [docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md),
Seção 8.

## 📈 Estado Atual do Projeto

### ✅ Disponível (herdado do pipeline de segmentação coronária)

- [x] Pré-processamento reaproveitável (downsampling, thresholding, LCC por fatia)
- [x] Pipeline completo de segmentação arterial (vesselness, aorta, óstios, region growing) — mantido como referência, fora do escopo do bounding box
- [x] Processamento em lote sobre splits do ImageCAS (`run_pipeline` / `get_data_splits`)
- [x] Mapeamento documentado do código reaproveitável ([docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md))

### 🚧 Próximos Passos (objetivo deste projeto)

- [ ] Extrair função dedicada de pré-processamento (etapas 1-4) desacoplada do label/vesselness/aorta
- [ ] Implementar cálculo do centróide geométrico do volume cardíaco binário
- [ ] Rodar o pré-processamento sobre as 1000 imagens do ImageCAS e derivar o envelope de dimensões máximas
- [ ] Implementar posicionamento do bounding box fixo centrado no centróide, aplicado à imagem em resolução integral
- [ ] Implementar validação de cobertura das anotações coronárias dentro do bbox
- [ ] Calcular a redução volumétrica média `Rv`
- [ ] Persistir resultados agregados (por caso e globais) seguindo o padrão de `output/` já usado pelo pipeline herdado

## 🔍 Observações e Riscos

Riscos identificados durante o estudo do código (detalhados em
[docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md), Seção 9):

- **Convenção de eixos**: os volumes seguem `shape = (Y, X, Z)` de forma consistente no pipeline
  principal, mas `src/utils/utils/roi.py` nomeia os eixos de forma diferente — atenção ao
  reaproveitar essa função para o corte final do bounding box.
- **Paths fixos**: `segmentation_pipeline.py` tem paths absolutos específicos da máquina do
  mestrando (com fallback); scripts novos devem receber `base_path`/`base_save_path` como parâmetro.
- **Sem testes automatizados**: qualquer função nova de pré-processamento/bbox deve vir com
  verificação mínima de shape/dtype/eixos, já que não há suíte de testes existente.

## 📚 Referências

- Silva et al., 2026 — "Coronary Artery Segmentation in CCTA Volumes via Aortic Localization and
  Modified Vesselness-Driven Region Growing" (Apêndice A da tese de doutorado de Hélder Almeida).
- Frangi et al., 1998 — formulação original do filtro de vesselness usada no pipeline herdado.
- Dataset: ImageCAS (1000 volumes CTA, 512×512 por slice).

## 📝 Licença

[Adicionar informação de licença]

## 📧 Contato

Hélder Almeida — hcalmeida@uesc.br

---

**Nota**: Este é um projeto de pesquisa em desenvolvimento. Os resultados devem ser validados antes
de qualquer aplicação clínica.
