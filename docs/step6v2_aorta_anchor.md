# Etapa 6 v2 — Âncora pela aorta (substitui a âncora por LCC)

> Parte da sequência do Capítulo 2 (Seção 2.3.1). Documenta a segunda tentativa da etapa 6, depois
> do achado bloqueante da primeira tentativa (via LCC) registrado em
> [step6_envelope.md](step6_envelope.md). Ver também [CODE_NOTES_bounding_box.md](CODE_NOTES_bounding_box.md).

## Por que uma segunda tentativa

A etapa 6 original (`docs/step6_envelope.md`) usava o centróide do `lcc_mask` (etapas 1-4) como
âncora para o bbox fixo. A sessão de 2026-07-15 mostrou que essa máscara isola o tronco inteiro, não
o coração — inviabilizando qualquer envelope calculado sobre ela (bbox ficaria perto do tamanho da
imagem original).

Antes de adotar o LCC, Hélder já havia cogitado usar a posição da aorta (com uma margem em mm) como
base para a ROI cardíaca, e abandonou esse caminho em favor do LCC. Ao revisar o código original do
pipeline de segmentação arterial de Silva et al. (2026) — restaurado temporariamente na branch
`draft-coronary-centerline-detection` para este estudo — ficou claro que:

1. A detecção/segmentação da aorta (Hough Circle Transform + Level Set) já está implementada e
   validada em escala: a execução canônica salva em
   `output/segmentation/canonical/mid_res/test/2026-06-05_20-00-43/` (**não migrada para esta
   branch** — só citada como evidência) mostra 696/700 (99,4%) casos do split de teste com
   aorta/óstios detectados, zero erros de pipeline.
2. Uma margem fixa **só** ao redor da aorta detectada tende a cortar parte da árvore coronária: no
   caso de teste IMG_ID=1, a aorta (Hough+Level Set) foi localizada entre z=94 e z=274 (na
   resolução reduzida), mas o label real de artéria coronária desse mesmo caso vai de z=17 a
   z=182 — a árvore se estende bem abaixo de onde a aorta é rastreada (a aorta ascendente arqueia
   sobre o topo do coração; a árvore coronária desce pela superfície do coração em direção ao
   ápice). Isso é coerente com Hélder ter abandonado esse caminho antes.

## Caminho adotado (Opção A, decidida em 2026-07-15)

Em vez de usar a aorta para "segmentar o coração" com uma margem arbitrária, ou tentar isolar o
coração como etapa intermediária (o problema da tentativa anterior), a nova etapa 6:

1. Usa o **centróide 3D da máscara da aorta** (Hough + Level Set) como **âncora determinística por
   caso** — só depende da imagem, sem label, sem componente aprendido.
2. Calibra o **tamanho fixo do bbox empiricamente**, uma única vez, a partir dos limites do label
   coronário **em relação a essa âncora**, usando só o split de treino — uma agregação min/max
   determinística (não um modelo), da mesma natureza da "envelope conservador" já prevista desde o
   início do capítulo (`docs/CODE_NOTES_bounding_box.md`, Seção 8), só que relativa a uma âncora
   melhor e com offsets assimétricos por direção (não um único "extent" simétrico).
3. Valida a cobertura nos splits de validação/teste (dados fora da calibração) — evita o vazamento
   que o plano original teria (calculava o envelope sobre as 1000 imagens juntas, sem separar
   calibração de validação).

Isso conecta a etapa 6 diretamente à etapa 9 (cobertura de 100%) sem depender de "segmentar o
coração" como etapa intermediária.

## O que foi implementado

Módulos migrados da branch `draft-coronary-centerline-detection` (código herdado, não alterado),
mínimo necessário — sem vesselness nem detecção de óstios, que essa âncora não precisa:

- `src/utils/segmentation/aorta_localization.py` — `detect_aorta_circles` (Hough Circle Transform +
  rastreamento entre fatias).
- `src/utils/segmentation/aorta_segmentation.py` — `level_set_segmentation` (Morphological Geodesic
  Active Contour) + `remove_leaks_morphology`.
- `src/utils/processing/gpu_utils.py`, `src/utils/processing/binary_operations.py` — dependências
  dos dois acima (`keep_largest_component`, `binary_opening`); fallback CPU automático, GPU
  (`cupy`/`cucim`) opcional via `config["USE_GPU"]` (default `false`, sem exigir instalação).
- `config/pipeline_config.json` — blocos `CIRCLE_DETECTION` e `LEVEL_SET` (só esses; nenhum
  parâmetro de vesselness/óstios/region growing/fuzzy foi trazido de volta).

Código novo desta etapa:

- `src/utils/segmentation/bounding_box.py::compute_bounding_box_bounds(mask)` — limites
  mín/máx (inclusivos) por eixo de uma máscara 3D; complementa `compute_bounding_extent` (que só
  retorna o tamanho) quando os limites absolutos são necessários para calcular offsets em relação a
  um ponto de referência externo.
- `src/utils/segmentation/aorta_anchor.py` — módulo principal da etapa:
  - `segment_aorta(lcc_image, scaled_spacing, config)` — encadeia detecção de círculos + Level Set
    + limpeza, retorna `(mask, num_circulos_detectados)`. Máscara vazia (nenhuma aorta detectada)
    é um resultado válido, não uma exceção — tratado como falha um nível acima.
  - `process_case_for_aorta_anchor(img_id, base_path, block, config)` — roda etapas 1-4, segmenta a
    aorta, calcula a âncora (`compute_geometric_centroid` na máscara da aorta) e os limites do
    label coronário (`compute_bounding_box_bounds`). Erros por caso (aorta não detectada, arquivo
    ausente) viram `status="error"`, não interrompem o lote.
  - `process_block`/`run_aorta_anchor_batch` — mesmo padrão retomável por bloco de pasta da etapa 6
    v1, agora compartilhado via `src/utils/project/batch.py::run_resumable_batch` (extraído de
    `segmentation.envelope` nesta sessão para não duplicar a lógica de checkpoint incremental).
  - `calibrate_bbox_offsets(records_df, downscale_factors)` — para cada eixo, offset mínimo =
    o mais conservador (mais negativo) de `label_min - âncora` entre os casos de calibração; offset
    máximo = o mais conservador (mais positivo) de `label_max - âncora`. Garante, por construção,
    que `âncora + offset_min <= label_min` e `label_max <= âncora + offset_max` para todo caso
    usado na calibração.
  - `validate_coverage(records_df, calibration)` — aplica os offsets calibrados a casos **fora** da
    calibração e marca, por caso e por eixo, se o label ficaria coberto pelo bbox resultante.
- `src/run_aorta_anchor.py` — script executável: roda/retoma o lote sobre `dataset/`, calibra com o
  split de treino (`config/imagecas_splits.json`), valida cobertura em val/test, salva
  `calibration.json` + `coverage_val.csv`/`coverage_test.csv`.

## Validação

- `tests/test_bounding_box.py` — 5 testes novos para `compute_bounding_box_bounds` (consistente com
  `compute_bounding_extent`, convenção de eixos, casos de erro).
- `tests/test_batch.py` — 4 testes do runner genérico `run_resumable_batch` (skip de casos `ok`,
  reprocessa tudo com `resume=False`, re-tenta casos `error`, schema do CSV).
- `tests/test_aorta_anchor.py` — 8 testes: caminho `ok` (âncora + limites do label, com volume
  NIfTI sintético real via `nibabel`, `segment_aorta` mockada), caminho `error` (aorta não
  detectada, arquivo ausente), schema do `process_block`, calibração (offsets conservadores por
  eixo, escala para resolução integral), validação de cobertura (caso dentro/fora do bbox
  calibrado).
- `uv run pytest tests/ -v` — 38/38 passando.
- **Validação manual com dado real** (IMG_ID=1, fora dos testes automatizados): âncora
  `(y=82.3, x=154.6, z=185.8)`, 171 círculos detectados, 336651 voxels na máscara da aorta, limites
  do label `y=[53,205] x=[94,229] z=[17,182]` — consistente com a investigação anterior desta
  sessão. ~36s de processamento (detecção + Level Set) nesta máquina sem GPU.

## Não incluído nesta etapa / decisões em aberto

- **Lote completo (1000 imagens) ainda não foi executado.** Custo estimado ~35s/caso sem GPU →
  ~9-10h para o dataset inteiro nesta máquina. Os módulos aceitam `config["USE_GPU"] = true`
  (requer `cupy`/`cucim`, não instalado nesta máquina Windows) — considerar rodar na máquina com
  GPU compartilhada do grupo.
- **Tamanho do split de treino é pequeno**: `config/imagecas_splits.json` define apenas **30**
  casos de treino (270 val, 700 test) — poucos casos para calibrar um offset "conservador" (min/max
  entre casos) com confiança; o valor calibrado pode não cobrir o pior caso real fora da amostra de
  30. Existem `config/imagecas_splits_train60.json` e `imagecas_splits_train90.json` com proporções
  de treino maiores — vale decidir com Hélder se a calibração desta etapa deve usar um desses em vez
  do split default, antes de rodar o lote completo.
- Etapas 7 (posicionamento do bbox) e 8 (corte em resolução integral) ainda não implementadas —
  ficam mais diretas agora: `bbox_min = âncora_resolução_integral + offset_min_full_resolution`,
  `bbox_max = âncora_resolução_integral + offset_max_full_resolution`, por eixo.
