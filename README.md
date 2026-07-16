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
derivar um bounding box 3D fixo ao redor do coração, sem segmentar a árvore coronária em si. As
etapas iniciais dos dois pipelines (redução dimensional, limiarização HU, extração do maior
componente conectado) são as mesmas — por isso esse trecho do código original foi mantido e
reaproveitado. Vesselness, detecção de óstios, segmentação/crescimento de região das artérias
coronárias e fuzzy connectedness foram removidos por não serem necessários para este objetivo (ver
[docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md)).

**Atualização (2026-07-15):** a localização/segmentação da aorta (Hough Circle Transform + Level
Set), inicialmente também removida, foi **reintroduzida** como base da etapa 6 — não para segmentar
a aorta em si, mas como âncora determinística e mais confiável que o centróide do LCC (ver
[docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md) para o racional completo).

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

   ⚠️ **Achado (2026-07-15):** a máscara resultante das etapas 1-4 não isola o coração — isola o
   tronco inteiro (pele, parede torácica, mediastino), pois não há fronteira de densidade HU entre
   o coração e o tecido adjacente. Por isso a etapa 6 não usa mais essa máscara como âncora — ver
   item 6. Detalhes, evidência visual/quantitativa:
   [docs/step6_envelope.md](docs/step6_envelope.md#achado-bloqueante-lcc-não-isola-o-coração).
5. Cálculo do centróide geométrico do volume cardíaco (média das coordenadas dos voxels, por eixo) — **implementado** ([docs/step5_centroid.md](docs/step5_centroid.md)); não é mais a âncora usada na etapa 6 (ver item 6), mas a função permanece testada e em uso pela etapa 6 v2 (agora aplicada à máscara da aorta, não ao LCC).
6. A partir das 1000 imagens do ImageCAS, identificação das dimensões máximas observadas nos três
   eixos — bounding box de **tamanho fixo** (envelope conservador) — **código implementado (2ª
   versão, âncora pela aorta via Hough+Level Set), lote completo ainda não executado** — ver
   [docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md).
7. Posicionamento do bounding box centralizado na âncora de cada imagem individual — **a implementar** (offsets calibrados já saem prontos da etapa 6 v2; falta aplicá-los).
8. Aplicação do bounding box sobre a imagem **original em resolução integral** (não a versão
   reduzida usada para o cálculo) — **a implementar**.
9. Validação: cobertura de 100% dos voxels de artéria coronária rotulados dentro do bbox (métrica
   binária por imagem) + cálculo da redução volumétrica média `Rv = 1 − (V_ROI / V_original)` —
   uma versão preliminar já existe em `validate_coverage()` (etapa 6 v2, checagem por intervalo sem
   cortar a imagem); a validação completa (com corte real e Rv) — **a implementar**.

As etapas 5-9 **não segmentam a árvore coronária nem realizam detecção de óstios** — vesselness,
óstios, region growing e fuzzy connectedness continuam removidos deste fork por estarem fora do
escopo deste objetivo (ver [docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md)). A localização e
segmentação da aorta (Hough Circle Transform + Level Set) foi reintroduzida como âncora da etapa 6 —
ver [docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md).

## 📝 Acompanhamento por etapa

Cada etapa implementada a partir daqui ganha um arquivo próprio em `docs/`, numerado conforme a
lista acima (ex.: `docs/step5_centroid.md`, `docs/step6_envelope.md`), registrando: o que foi
implementado, decisões tomadas (convenção de eixos, tratamento de borda etc.), arquivo:linha da
função nova e como validar. Documentos já existentes:

- [docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md) — estudo do código herdado e plano de implementação das etapas 5-9.
- [docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md) — o que foi removido do pipeline original e por quê.
- [docs/step5_centroid.md](docs/step5_centroid.md) — centróide geométrico do volume cardíaco (etapa 5).
- [docs/step6_envelope.md](docs/step6_envelope.md) — 1ª tentativa da etapa 6 (âncora = centróide do LCC): **substituída**, mantida como referência histórica do achado que motivou a 2ª tentativa.
- [docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md) — 2ª tentativa da etapa 6 (âncora = centróide da máscara da aorta via Hough+Level Set, offsets calibrados no split de treino): caminho ativo, código pronto e testado, lote completo (1000 imagens) ainda não executado.

## 🏗️ Estrutura do Projeto

```file
heart-bounding-box/
├── pyproject.toml                  # Configuração e dependências do projeto
├── README.md                       # Este arquivo
├── docs/
│   ├── CODE_NOTES_bounding_box.md  # Mapeamento do código reaproveitável + plano de implementação
│   ├── CODE_REMOVAL_PLAN.md        # Justificativa arquivo a arquivo do que foi removido
│   ├── step5_centroid.md           # Documentação da etapa 5 (centróide geométrico)
│   ├── step6_envelope.md           # Etapa 6, 1ª tentativa (âncora=LCC) — substituída, achado bloqueante
│   ├── step6v2_aorta_anchor.md     # Etapa 6, 2ª tentativa (âncora=aorta) — caminho ativo
│   └── img/                        # Evidência visual referenciada pelos docs de etapa
├── doc/                             # Documentação herdada do pipeline original (histórico)
├── config/
│   ├── pipeline_config.json        # downscale, threshold HU, LCC, detecção de círculos, Level Set
│   └── imagecas_splits*.json       # Splits fixos train/val/test do ImageCAS
├── tests/
│   ├── test_bounding_box.py        # Testes das etapas 5-6 (uv run pytest tests/ -v)
│   ├── test_envelope.py            # Testes da etapa 6, 1ª tentativa (âncora=LCC)
│   ├── test_aorta_anchor.py        # Testes da etapa 6, 2ª tentativa (âncora=aorta)
│   └── test_batch.py               # Testes do runner de lote retomável genérico
├── src/
│   ├── run_envelope.py             # Script da etapa 6, 1ª tentativa (substituída)
│   ├── run_aorta_anchor.py         # Script da etapa 6, 2ª tentativa (uv run python src/run_aorta_anchor.py)
│   ├── eda/
│   │   ├── eda_imgs.ipynb                          # EDA genérica de intensidades HU (não amarrada ao pipeline arterial)
│   │   └── check_preprocessing_and_centroid.ipynb  # Testa etapas 1-5 + Rv preliminar numa imagem aleatória do dataset
│   └── utils/
│       ├── processing/
│       │   ├── preprocessing.py            # downscale, threshold, LCC — núcleo do bounding box
│       │   ├── binary_operations.py        # keep_largest_component, binary_opening (pós-proc. da aorta)
│       │   └── gpu_utils.py                # detecção de GPU (cupy/cucim), fallback CPU automático
│       ├── segmentation/
│       │   ├── lower_threshold.py          # limiar inferior HU (fixed/percentile/object_relative_percentile)
│       │   ├── pipeline_preprocessing.py   # load_and_preprocess_image() — orquestra as etapas 1-4
│       │   ├── bounding_box.py             # etapas 5-6: centróide, extensão/limites do bbox mínimo
│       │   ├── envelope.py                 # etapa 6, 1ª tentativa (âncora=LCC) — substituída
│       │   ├── aorta_localization.py       # Hough Circle Transform — localização da aorta (herdado)
│       │   ├── aorta_segmentation.py       # Level Set (MGAC) — segmentação da aorta (herdado)
│       │   └── aorta_anchor.py             # etapa 6, 2ª tentativa: âncora pela aorta + calibração
│       ├── project/
│       │   ├── cache.py            # cache genérico em disco (JSON/NPY)
│       │   ├── config.py           # load/save/merge de config JSON
│       │   ├── dataset.py          # get_data_splits() — splits fixos, útil para rodar sobre as 1000 imagens
│       │   ├── notebook_env.py     # bootstrap de notebooks (sys.path, paths do dataset)
│       │   ├── results.py          # timing de execuções em lote e diretórios com timestamp
│       │   └── batch.py            # run_resumable_batch() — checkpoint incremental genérico por caso
│       ├── utils/
│       │   ├── json_io.py          # I/O JSON genérico
│       │   └── nifti_io.py         # I/O NIfTI (nibabel)
│       ├── visualization/
│       │   ├── image_slices.py     # MIP, grid de fatias
│       │   ├── preprocessing_views.py  # visualiza downscale/threshold/LCC lado a lado
│       │   └── volume.py           # render 3D genérico via k3d (marching cubes)
│       └── groups/                 # namespaces agrupados (io/processing/segmentation/visualization)
└── output/                          # Resultados e artefatos de execução
    ├── bounding_box/
    │   ├── envelope/               # Saída da etapa 6, 1ª tentativa (âncora=LCC)
    │   └── aorta_anchor/           # Saída da etapa 6, 2ª tentativa (âncora=aorta): CSVs por bloco, calibration.json
    └── segmentation/
        ├── runs/                   # Execuções completas por resolução/timestamp (pipeline arterial herdado)
        ├── canonical/              # Referências estáveis para EDA/comparações (pipeline arterial herdado)
        └── analysis/               # Comparações, bad cases e exemplos visuais (execuções antigas)
```

> Nota: `src/segmentation_pipeline.py`, o pipeline completo de segmentação arterial (Frangi,
> detecção de óstios, region growing/fuzzy connectedness) e toda a ferragem de comparação/EDA/sweep
> construída em torno dele foram **removidos** nesta limpeza — seguem removidos. Detalhes e
> justificativa de cada arquivo em [docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md). A
> detecção de círculos (Hough) e o Level Set da aorta foram **reintroduzidos** em 2026-07-15 como
> âncora da etapa 6 (não para segmentar a árvore coronária) — ver
> [docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md).

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
- Remove regiões isoladas (ar/pulmões, osso muito denso), mas **não isola o coração** — o
  componente resultante é o tronco inteiro (pele, parede torácica, mediastino), já que não há
  fronteira de densidade HU entre essas estruturas e o coração. Ver achado em
  [docs/step6_envelope.md](docs/step6_envelope.md#achado-bloqueante-lcc-não-isola-o-coração). Por
  isso deixou de ser usado como âncora do bbox (etapa 6 v2 usa a aorta, ver item 5 abaixo) — mas
  continua sendo a base das etapas 1-4 e da imagem (`lcc_image`) sobre a qual a aorta é detectada.
- Implementado em `largest_connected_component` ([src/utils/processing/preprocessing.py](src/utils/processing/preprocessing.py))

### 4. Orquestração

- `load_and_preprocess_image(img_id, base_path, config)` encadeia as três etapas acima e devolve o
  volume 3D (`lcc_image`) usado tanto pelo cálculo do centróide (etapa 5) quanto pela
  detecção/segmentação da aorta (item 5)
  ([src/utils/segmentation/pipeline_preprocessing.py](src/utils/segmentation/pipeline_preprocessing.py))

### 5. Localização e segmentação da aorta (âncora da etapa 6 v2)

- **Localização**: Transformada de Hough para círculos, fatia a fatia, rastreando a aorta
  ascendente com tolerância de variação de raio/distância em mm — `detect_aorta_circles`
  ([src/utils/segmentation/aorta_localization.py](src/utils/segmentation/aorta_localization.py)).
- **Segmentação**: Level Set 3D (Morphological Geodesic Active Contour) inicializado pelos círculos
  detectados — `level_set_segmentation` + `remove_leaks_morphology`
  ([src/utils/segmentation/aorta_segmentation.py](src/utils/segmentation/aorta_segmentation.py)).
- Código herdado do pipeline de segmentação arterial de Silva et al. (2026), reintroduzido em
  2026-07-15 sem as etapas de vesselness/óstios que o acompanhavam originalmente — não são
  necessárias para usar a aorta como âncora posicional. Orquestrado por
  `segment_aorta()` ([src/utils/segmentation/aorta_anchor.py](src/utils/segmentation/aorta_anchor.py)).
  Ver [docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md).

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
- `scipy`: `ndimage` (zoom, label, distance transform)
- `scikit-image`: `feature.canny`/`transform.hough_circle*` (detecção da aorta),
  `segmentation.morphological_geodesic_active_contour` (Level Set da aorta),
  `morphology`/`draw` (pós-processamento), e `volume.py` (marching cubes para render 3D via `k3d`)
- `k3d`: visualização 3D opcional de máscaras
- `pandas`: persistência/timing de execuções em lote
- `scikit-learn`: divisão train/test dos splits do ImageCAS

> **Atualização (2026-07-15):** a detecção/segmentação da aorta (Hough Circle Transform + Level
> Set) foi **reintroduzida** como âncora da etapa 6 v2 — ver
> [docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md). GPU (`cupy`/`cucim`) continua
> **opcional**: os módulos migrados detectam GPU automaticamente e caem para CPU se
> `cupy`/`cucim` não estiverem instalados (não estão nesta instalação `uv sync`); `config["USE_GPU"]`
> default `false`. `morphsnakes` segue removido (não é usado pelo Level Set migrado, que usa
> `skimage.segmentation.morphological_geodesic_active_contour`, já coberto por `scikit-image`).

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

### Extração de bounding box cardíaco (etapas 6-9)

```python
from utils.segmentation.aorta_anchor import (
    process_case_for_aorta_anchor,  # 1 caso: roda etapas 1-4 + aorta, retorna âncora + limites do label
    calibrate_bbox_offsets,         # agrega vários casos (treino) em offsets fixos por eixo
    validate_coverage,              # checa cobertura em casos fora da calibração (val/test)
)
```

`src/run_aorta_anchor.py` roda/retoma o lote sobre `dataset/` local, calibra os offsets com o split
de treino e valida cobertura em val/test — ver
[docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md):

```bash
uv run python src/run_aorta_anchor.py
```

**Ainda não foi executado sobre as 1000 imagens** — custo estimado ~35s/caso sem GPU (~9-10h para o
dataset inteiro nesta máquina); ver Estado Atual abaixo. Etapas 7-8 (posicionar/cortar o bbox na
imagem em resolução integral) ainda não têm código próprio, mas a calibração da etapa 6 v2 já
produz os offsets necessários. O plano original completo (anterior à âncora pela aorta) segue em
[docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md), Seção 8.

## 📈 Estado Atual do Projeto

### ✅ Concluído

- [x] Pré-processamento reaproveitável e desacoplado de vesselness/óstios/artérias (downsampling, thresholding, LCC por fatia)
- [x] Processamento em lote sobre splits do ImageCAS (`get_data_splits`)
- [x] Limpeza de código: pipeline arterial completo removido, mantido só o necessário para o bounding box ([docs/CODE_REMOVAL_PLAN.md](docs/CODE_REMOVAL_PLAN.md))
- [x] Mapeamento documentado do código reaproveitável ([docs/CODE_NOTES_bounding_box.md](docs/CODE_NOTES_bounding_box.md))
- [x] Etapa 5 — cálculo do centróide geométrico do volume cardíaco binário ([docs/step5_centroid.md](docs/step5_centroid.md))
- [x] Etapa 6, 1ª tentativa (âncora=LCC) — código pronto e testado, **substituída** após achado de que a máscara do LCC não isola o coração ([docs/step6_envelope.md](docs/step6_envelope.md))
- [x] Etapa 6, 2ª tentativa (âncora=aorta via Hough+Level Set, offsets calibrados no split de treino) — código pronto, testado (38 testes) e validado manualmente com dado real; **lote completo (1000 imagens) ainda não executado** ([docs/step6v2_aorta_anchor.md](docs/step6v2_aorta_anchor.md))

### 🔶 Decisões em aberto antes de rodar o lote completo da etapa 6 v2

- **Onde/quando rodar**: ~35s/caso sem GPU nesta máquina → ~9-10h para as 1000 imagens. Os módulos
  já suportam GPU (`config["USE_GPU"] = true`, requer `cupy`/`cucim`) — decidir entre rodar aqui em
  background ou na máquina com GPU compartilhada do grupo.
- **Split de calibração pequeno demais?** `config/imagecas_splits.json` (o split fixo default) tem
  só **30 casos de treino** (270 val, 700 test) — poucos para calibrar offsets "conservadores"
  (mín/máx entre casos) com confiança de que cobrem o pior caso fora da amostra. Existem
  `imagecas_splits_train60.json`/`imagecas_splits_train90.json` com mais casos de treino — decidir
  se a calibração desta etapa deve usar um desses em vez do split default.

### 🚧 Próximos Passos

- [ ] Rodar `src/run_aorta_anchor.py` sobre as 1000 imagens (depois das decisões acima)
- [ ] Implementar posicionamento/corte do bounding box fixo na imagem em resolução integral, a partir dos offsets calibrados (etapas 7-8)
- [ ] Implementar a validação de cobertura completa com corte real + cálculo de `Rv` (etapa 9) — `validate_coverage()` já faz uma checagem preliminar por intervalo, sem cortar a imagem
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
