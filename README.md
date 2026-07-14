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
tese) — cujo código era a base original deste repositório.

O pipeline do artigo segmenta as artérias coronárias. **O objetivo deste projeto é mais restrito**:
apenas isolar o coração e derivar um bounding box 3D fixo, sem segmentar a aorta nem a árvore
arterial. As etapas iniciais dos dois pipelines (redução dimensional, limiarização HU, extração do
maior componente conectado) são as mesmas — por isso esse trecho do código original foi mantido e
reaproveitado, enquanto todo o restante (vesselness, localização/segmentação da aorta, óstios,
artérias) foi removido do repositório por não ser necessário para este objetivo.

Um mapeamento detalhado de onde cada etapa reaproveitável estava/está implementada (arquivo:linha,
assinatura, convenção de eixos, riscos) está em
[docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md). A justificativa completa,
arquivo a arquivo, de tudo que foi removido está em
[docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md).

## 🎯 Objetivo

Implementar, a partir do código herdado, a sequência descrita na Seção 2.3.1 do Capítulo 2:

1. Redução dimensional (fator 2x, bicúbica, preserva N slices) — **implementado**.
2. Limiarização HU adaptativa (−300 HU fixo / percentil 99,7 adaptativo) — **implementado**.
3. Extração do maior componente conectado (LCC) por fatia axial — **implementado**.
4. Empilhamento dos slices processados em um volume cardíaco 3D binário — **implementado**
   (resultado direto das etapas 1-3, já desacoplado de vesselness/aorta/óstios).
5. Cálculo do centróide geométrico do volume cardíaco (média das coordenadas dos voxels, por eixo) — **implementado** ([docs/step5_centroid.md](docs/step5_centroid.md)).
6. A partir das 1000 imagens do ImageCAS, identificação das dimensões máximas observadas nos três
   eixos — bounding box de **tamanho fixo** (envelope conservador) — **a implementar**.
7. Posicionamento do bounding box centralizado no centróide de cada imagem individual — **a implementar**.
8. Aplicação do bounding box sobre a imagem **original em resolução integral** (não a versão
   reduzida usada para o cálculo) — **a implementar**.
9. Validação: cobertura de 100% dos voxels de artéria coronária rotulados dentro do bbox (métrica
   binária por imagem) + cálculo da redução volumétrica média `Rv = 1 − (V_ROI / V_original)` — **a implementar**.

As etapas 5-9 **não segmentam a aorta nem realizam detecção de óstios/artérias** — todo o restante
do pipeline original (Hough Circle Transform, Level Set, Frangi/vesselness, Region Growing) foi
removido deste fork por estar **fora do escopo** deste objetivo (ver
[docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md)).

## 📝 Acompanhamento por etapa

Cada etapa implementada a partir daqui ganha um arquivo próprio em `docs/`, numerado conforme a
lista acima (ex.: `docs/step5_centroid.md`, `docs/step6_envelope.md`), registrando: o que foi
implementado, decisões tomadas (convenção de eixos, tratamento de borda etc.), arquivo:linha da
função nova e como validar. Documentos já existentes:

- [docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md) — estudo do código herdado e plano de implementação das etapas 5-9.
- [docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md) — o que foi removido do pipeline original e por quê.
- [docs/step5_centroid.md](docs/step5_centroid.md) — centróide geométrico do volume cardíaco (etapa 5).

## 🏗️ Estrutura do Projeto

```file
heart-bounding-box/
├── pyproject.toml                  # Configuração e dependências do projeto
├── README.md                       # Este arquivo
├── docs/
│   ├── CODE_NOTES_bounding_box.md  # Mapeamento do código reaproveitável + plano de implementação
│   ├── CODE_REMOVAL_PLAN.md        # Justificativa arquivo a arquivo do que foi removido
│   └── step5_centroid.md           # Documentação da etapa 5 (centróide geométrico)
├── doc/                             # Documentação herdada do pipeline original (histórico)
├── config/
│   ├── pipeline_config.json        # Config mínima: downscale, threshold HU, LCC
│   └── imagecas_splits*.json       # Splits fixos train/val/test do ImageCAS
├── tests/
│   └── test_bounding_box.py        # Testes da etapa 5 (uv run pytest tests/ -v)
├── src/
│   ├── eda/
│   │   ├── eda_imgs.ipynb                          # EDA genérica de intensidades HU (não amarrada ao pipeline arterial)
│   │   └── check_preprocessing_and_centroid.ipynb  # Testa etapas 1-5 + Rv preliminar numa imagem aleatória do dataset
│   └── utils/
│       ├── processing/
│       │   └── preprocessing.py            # downscale, threshold, LCC — núcleo do bounding box
│       ├── segmentation/
│       │   ├── lower_threshold.py          # limiar inferior HU (fixed/percentile/object_relative_percentile)
│       │   ├── pipeline_preprocessing.py   # load_and_preprocess_image() — orquestra as etapas 1-4
│       │   └── bounding_box.py             # etapas 5+ novas: centróide, envelope, posicionamento, corte, validação
│       ├── project/
│       │   ├── cache.py            # cache genérico em disco (JSON/NPY)
│       │   ├── config.py           # load/save/merge de config JSON
│       │   ├── dataset.py          # get_data_splits() — splits fixos, útil para rodar sobre as 1000 imagens
│       │   ├── notebook_env.py     # bootstrap de notebooks (sys.path, paths do dataset)
│       │   └── results.py          # timing de execuções em lote e diretórios com timestamp
│       ├── utils/
│       │   ├── json_io.py          # I/O JSON genérico
│       │   └── nifti_io.py         # I/O NIfTI (nibabel)
│       ├── visualization/
│       │   ├── image_slices.py     # MIP, grid de fatias
│       │   ├── preprocessing_views.py  # visualiza downscale/threshold/LCC lado a lado
│       │   └── volume.py           # render 3D genérico via k3d (marching cubes)
│       └── groups/                 # namespaces agrupados (io/processing/segmentation/visualization)
└── output/                          # Resultados e artefatos de execução
    └── segmentation/
        ├── runs/                   # Execuções completas por resolução/timestamp
        ├── canonical/              # Referências estáveis para EDA/comparações
        └── analysis/               # Comparações, bad cases e exemplos visuais (execuções antigas)
```

> Nota: `src/segmentation_pipeline.py`, o pipeline completo de segmentação arterial (Frangi, Hough,
> Level Set, óstios, region growing/fuzzy connectedness) e toda a ferragem de comparação/EDA/sweep
> construída em torno dele foram **removidos** nesta limpeza. Detalhes e justificativa de cada
> arquivo em [docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md).

## 🔬 Pré-processamento reaproveitado (base do bounding box)

### 1. Downsampling

- Redução da imagem por fator 2 em x/y (interpolação linear por padrão, cúbica/bicúbica configurável)
- Fatores de redução: `(2, 2, 1)` — reduz x e y, mantém z
- Implementado em `downscale_image_opencv` / `downscale_image_ndi` ([src/utils/processing/preprocessing.py](src/utils/processing/preprocessing.py))

### 2. Threshold Dinâmico Adaptativo

- **Limite inferior**: `-300 HU` fixo (configurável via `resolve_lower_threshold`)
- **Limite superior**: percentil 99.7 da imagem (adaptativo)
- Implementado em `threshold_image_with_offset` ([src/utils/processing/preprocessing.py](src/utils/processing/preprocessing.py))

### 3. Extração do Maior Componente Conectado (LCC)

- Aplicação **slice-by-slice** (2D) para cada fatia axial (`LCC_PER_SLICE=true` por padrão)
- Remove regiões isoladas, pulmões e vasos pulmonares; mantém apenas a maior estrutura conectada (coração e grandes vasos)
- Implementado em `largest_connected_component` ([src/utils/processing/preprocessing.py](src/utils/processing/preprocessing.py))

### 4. Orquestração

- `load_and_preprocess_image(img_id, base_path, config)` encadeia as três etapas acima e devolve o
  volume cardíaco 3D (`lcc_image`) já pronto para o cálculo do centróide (etapa 5)
  ([src/utils/segmentation/pipeline_preprocessing.py](src/utils/segmentation/pipeline_preprocessing.py))

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

- `nibabel`: Leitura de arquivos NIfTI
- `opencv-python`: Downsampling (`cv2.resize`)
- `scipy`: `ndimage` (zoom, label)
- `scikit-image`: usado hoje só por `volume.py` (marching cubes para render 3D via `k3d`)
- `k3d`: visualização 3D opcional de máscaras
- `pandas`: persistência/timing de execuções em lote
- `scikit-learn`: divisão train/test dos splits do ImageCAS

> GPU (`cupy`/`cucim`) e `morphsnakes` foram removidos do projeto junto com o pipeline arterial —
> nenhuma etapa do bounding box (downscale/threshold/LCC) tinha ou precisa de aceleração GPU. Ver
> decisão registrada em [docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md).

## 💻 Uso

### Pré-processamento + centróide (etapas 1-5, disponível hoje)

```python
from utils.segmentation.pipeline_preprocessing import load_and_preprocess_image
from utils.segmentation.bounding_box import compute_geometric_centroid

image_data = load_and_preprocess_image(img_id=1, base_path=BASE_PATH, config=CONFIG)
lcc_image = image_data["lcc_image"]          # volume cardíaco 3D já reduzido/limiarizado/LCC
lcc_mask = image_data["lcc_mask"]            # máscara binária correspondente
label = image_data["label"]                  # label coronário na mesma resolução, para validação futura (etapa 9)
scaled_spacing = image_data["scaled_spacing"]

centroid = compute_geometric_centroid(lcc_mask)  # array([y_c, x_c, z_c]), na resolução reduzida
```

Para uma checagem interativa (visual, com uma imagem aleatória real do dataset local), use
[src/eda/check_preprocessing_and_centroid.ipynb](src/eda/check_preprocessing_and_centroid.ipynb):
sorteia um `IMG_ID`, roda as etapas 1-5 e plota MIP/fatias/centróide sobre o resultado.

### Extração de bounding box cardíaco (etapas 6-9, planejado)

Ainda não há um script/entry point próprio para agregar as etapas 6-9. O plano de implementação
proposto — envelope máximo sobre as 1000 imagens, posicionamento, corte em resolução integral e
validação de cobertura/Rv — está descrito em detalhe em
[docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md), Seção 8, e será documentado
etapa a etapa conforme a convenção da seção "Acompanhamento por etapa" acima.

## 📈 Estado Atual do Projeto

### ✅ Concluído

- [x] Pré-processamento reaproveitável e desacoplado de vesselness/aorta/óstios (downsampling, thresholding, LCC por fatia)
- [x] Processamento em lote sobre splits do ImageCAS (`get_data_splits`)
- [x] Limpeza de código: pipeline arterial completo removido, mantido só o necessário para o bounding box ([docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md))
- [x] Mapeamento documentado do código reaproveitável ([docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md))
- [x] Etapa 5 — cálculo do centróide geométrico do volume cardíaco binário ([docs/step5_centroid.md](docs/step5_centroid.md))

### 🚧 Próximos Passos

- [ ] Rodar o pré-processamento sobre as 1000 imagens do ImageCAS e derivar o envelope de dimensões máximas (etapa 6)
- [ ] Implementar posicionamento do bounding box fixo centrado no centróide, aplicado à imagem em resolução integral (etapas 7-8)
- [ ] Implementar validação de cobertura das anotações coronárias dentro do bbox (etapa 9)
- [ ] Calcular a redução volumétrica média `Rv` (etapa 9)
- [ ] Persistir resultados agregados (por caso e globais) seguindo o padrão de `output/` já usado pelo pipeline herdado

## 🔍 Observações e Riscos

Riscos identificados durante o estudo do código (detalhados em
[docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md), Seção 9):

- **Convenção de eixos**: os volumes seguem `shape = (Y, X, Z)` de forma consistente no código
  mantido. A função `extract_square_region`/`extract_circular_region` (que nomeava eixos de forma
  divergente) foi removida junto com `roi.py` — a função de corte da etapa 8 será escrita do zero,
  já com a convenção correta, em vez de reaproveitar aquele código.
- **Paths fixos**: `resolve_imagecas_base_path`/`resolve_processed_imagecas_path`
  ([src/utils/project/notebook_env.py](src/utils/project/notebook_env.py)) ainda têm candidatos de
  path absolutos específicos da máquina do mestrando (com fallback); scripts novos devem continuar
  recebendo `base_path`/`base_save_path` como parâmetro em vez de depender desses defaults.
- **Testes automatizados ainda mínimos**: `tests/` começou com a etapa 5 (`compute_geometric_centroid`,
  `uv run pytest tests/ -v`); qualquer função nova de pré-processamento/bbox deve continuar vindo
  com verificação de shape/dtype/eixos, já que a cobertura ainda é parcial.

## 📚 Referências

- Silva et al., 2026 — "Coronary Artery Segmentation in CCTA Volumes via Aortic Localization and
  Modified Vesselness-Driven Region Growing" (Apêndice A da tese de doutorado de Hélder Almeida).
- Frangi et al., 1998 — formulação original do filtro de vesselness usada no pipeline original
  (código removido deste fork; citado aqui só como contexto histórico da técnica).
- Dataset: ImageCAS (1000 volumes CTA, 512×512 por slice).

## 📝 Licença

[Adicionar informação de licença]

## 📧 Contato

Hélder Almeida — hcalmeida@uesc.br

---

**Nota**: Este é um projeto de pesquisa em desenvolvimento. Os resultados devem ser validados antes
de qualquer aplicação clínica.
