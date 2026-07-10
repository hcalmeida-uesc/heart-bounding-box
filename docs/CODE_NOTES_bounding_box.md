# Notas técnicas — reaproveitamento do pipeline para bounding box cardíaco (Cap. 2)

> Documento de estudo, sem alteração de código. Gerado para apoiar a implementação futura
> da extração determinística de bounding box cardíaco (Capítulo 2 da tese de Hélder),
> reaproveitando o pipeline do artigo do Apêndice A (Silva et al., 2026).

## 1. Mapa do repositório

```
src/
  segmentation_pipeline.py          # Entry point CLI (main()), monta config e roda por split
  eda/                               # Notebooks de análise exploratória (sem .py utilitário)
  experiments/                       # Scripts/notebooks de sweep de parâmetros (aorta+óstios, threshold)
  utils/
    project/
      config.py                     # load/save/scale de config JSON
      dataset.py                    # get_data_splits() — split fixo train/val/test do ImageCAS
      cache.py                      # cache genérico JSON/NPY em disco
      results.py                    # persistência de resultados/metadados por batch
      notebook_env.py
    utils/
      nifti_io.py                   # load_raw_img_and_label, save_nii_image, save_npy_array
      roi.py                        # extract_square_region / extract_circular_region (ROI 2D→3D)
      normalization.py, metrics.py (dice_score), segmentation.py, json_io.py
    processing/
      preprocessing.py              # downscale, threshold, LCC — o núcleo reaproveitável (Cap. 2, etapas 1-4)
      binary_operations.py          # closing/dilation/erosion/label com fallback GPU
      frangi.py                     # get_vesselness / get_modified_vesselness (Frangi normal e modificado)
      gpu_utils.py                  # detecção de GPU (cupy/cucim) e helpers to_gpu/to_cpu
    segmentation/
      pipeline_preprocessing.py     # load_and_preprocess_image() — orquestra etapas 1-4 por caso
      pipeline_detection.py         # aorta (círculos+level set) e óstios, com cache em disco
      pipeline_arteries.py          # segmentação da árvore coronária (region growing / fuzzy connectedness)
      pipeline_orchestration.py     # process_image() [1 caso] e run_pipeline() [lote/split completo]
      pipeline_steps.py             # shim de compatibilidade (reexporta as 3 acima)
      pipeline_cli.py               # argparse da CLI
      pipeline_reporting.py         # prints de resumo por split
      aorta_localization.py         # Hough Circle Transform + rastreamento entre fatias
      aorta_segmentation.py         # Level Set 3D (MGAC) + remoção de vazamentos morfológica
      ostia_detection.py            # extração da superfície da aorta, seleção/validação dos óstios
      artery_segmentation.py        # region growing modificado por vesselness
      lower_threshold.py            # limiar inferior HU (fixed/percentile/object_relative_percentile)
      fuzzy_threshold.py            # variante fuzzy do threshold inicial (alternativa ao "normal")
      fuzzy_connectedness.py        # variante fuzzy da segmentação arterial
      backend_comparison.py
    visualization/                  # plots de debug (Hough, vesselness, slices, casos ruins etc.)
    groups/                         # reexports "agrupados" (lazy) de utils/processing/project para notebooks
    comparison_utils/, experiments/ # utilitários de comparação/sweep entre variantes do pipeline
config/
  pipeline_config.json              # config default (parâmetros em resolução MID, factor 2x)
  imagecas_splits*.json             # splits fixos train/val/test (1000 casos)
output/                              # resultados de execuções (CSV, metadata, cache) — não versionar dados grandes
```

**Ponto de entrada:** `src/segmentation_pipeline.py:main()` (`segmentation_pipeline.py:564`). Fluxo:
CLI args → `build_effective_config()` (aplica resolução mid/high + overrides) → `run_pipeline()`
(`pipeline_orchestration.py:261`) → `process_image()` por caso (`pipeline_orchestration.py:34`), em lotes
salvos incrementalmente.

## 2. Funções relevantes por etapa

### (a) Redução dimensional / downsampling

- `downscale_image_ndi(image, factors, order=3)` — [preprocessing.py:30](../src/utils/processing/preprocessing.py#L30). `scipy.ndimage.zoom` com `zoom = 1/factor` por eixo. Usada para o **label** sempre com `order=0` (nearest, preserva classes) em [pipeline_preprocessing.py:190](../src/utils/segmentation/pipeline_preprocessing.py#L190).
- `downscale_image_opencv(image, factors, interpolation=cv2.INTER_LINEAR)` — [preprocessing.py:38](../src/utils/processing/preprocessing.py#L38). Suporta 2D e 3D; no caso 3D reamostra fatia a fatia em XY e, se `factor_z != 1`, faz uma segunda passada em Z. É o método **default** do pipeline (`DOWNSCALE_METHOD: "opencv"`, `OPENCV_INTERPOLATION: "linear"` em `pipeline_config.json:5-6`) — nota: a tese pede bicúbica; o config atual usa linear (fácil de trocar via `--opencv-interpolation cubic`).
- `downscale_image(image, factors, order, use_opencv, opencv_interpolation)` — [preprocessing.py:87](../src/utils/processing/preprocessing.py#L87). Wrapper único com fallback GPU→CPU; não é o que o pipeline principal chama diretamente (que decide opencv vs ndi inline em `pipeline_preprocessing.py:111-118`), mas é o mais indicado para reaproveitamento direto por ser a interface estável.
- **I/O:** entrada `NDArray` float, shape 3D `(Y, X, Z)`. `factors` é uma sequência `(fx, fy, fz)`; no config é `[2, 2, 1]` — reduz XY, preserva Z (bate com a tese, "eixo axial preservado"). Saída: mesmo dtype, shape reduzido.

### (b) Limiarização por unidades Hounsfield

- `threshold_image_with_offset(image, min_val=-300, max_val=675)` — [preprocessing.py:126](../src/utils/processing/preprocessing.py#L126). Retorna `(thresh_img, thresh_mask, offset)`; aplica offset `abs(min_val)` para manter valores positivos (necessário para o `ndi.label` funcionar bem com LCC depois). É a função "clássica" usada no caminho `threshold_mode == "normal"`.
- `threshold_image(image, min_val=-300, max_val=675)` — [preprocessing.py:117](../src/utils/processing/preprocessing.py#L117). Variante mais simples sem offset (usada em contextos legados/comparação).
- **Limite superior adaptativo:** calculado inline com `np.percentile(down_image, config["MAX_THRESHOLD_PERCENTILE"])` — [pipeline_preprocessing.py:126](../src/utils/segmentation/pipeline_preprocessing.py#L126). Config default `MAX_THRESHOLD_PERCENTILE: 99.7` (bate exatamente com a tese).
- **Limite inferior:** não é mais fixo puro — foi generalizado em `resolve_lower_threshold(volume, config)` — [lower_threshold.py:58](../src/utils/segmentation/lower_threshold.py#L58), com 3 métodos (`fixed` = -300 HU fixo, `percentile`, `object_relative_percentile`). O default de config é `fixed_hu: -300`, que reproduz o comportamento do artigo; os outros métodos são extensões exploratórias do mestrando, não fazem parte do pipeline original.
- Existe também um caminho alternativo **fuzzy threshold** (`fuzzy_threshold.py`, `THRESHOLDING.method == "fuzzy"`) que substitui o threshold binário por pertinência fuzzy — é uma variante experimental, não a usada no artigo publicado (que é o método "normal"/HU fixo+percentil).

### (c) Maior componente conectado (LCC) por fatia

- `largest_connected_component(image, mask)` — [preprocessing.py:143](../src/utils/processing/preprocessing.py#L143). Usa `scipy.ndimage.label` + `_find_largest_component_label` (maior por contagem de voxels, ignorando o fundo). Retorna `(lcc_img, lcc_mask)`.
- Aplicação **por fatia**: laço explícito em [pipeline_preprocessing.py:136-142](../src/utils/segmentation/pipeline_preprocessing.py#L136) — `for z_idx in range(thresh_image.shape[2]): largest_connected_component(thresh_image[:, :, z_idx], thresh_mask[:, :, z_idx])`. Controlado por `config["LCC_PER_SLICE"]` (default `True`, bate com a tese). Se `False`, cai no LCC 3D do volume inteiro (`largest_connected_component` chamado direto no volume).
- `build_lcc_image_from_mask(volume, mask, offset, per_slice=True)` — [preprocessing.py:161](../src/utils/processing/preprocessing.py#L161) — mesma lógica, mas genérica para qualquer máscara candidata (usada no caminho fuzzy). É a versão mais "reutilizável" porque encapsula o loop por fatia com uma flag, em vez de duplicar o loop inline.
- `run_core_preprocessing_pipeline(...)` — [preprocessing.py:205](../src/utils/processing/preprocessing.py#L205) — **já existe uma função que encadeia (a)+(b)+(c) num só pipeline reutilizável**, mas está desacoplada do restante (não usa `resolve_lower_threshold`, não lida com o label, não retorna spacing). Parece código legado/alternativo — vale conferir se ainda é chamada em algum lugar antes de migrar para ela (não encontrei chamadores fora do próprio módulo/testes).

### (d) Detecção/segmentação da aorta (HCT + Level Set)

- `detect_aorta_circles(img_volume, hough_radii, pixel_spacing, ...)` — [aorta_localization.py:540](../src/utils/segmentation/aorta_localization.py#L540). Fatia a fatia, começando do **último slice** (`first_slice_idx = num_slices - 1`) e caminhando para trás (`range(first_slice_idx - 1, -1, -1)`), com Canny (`skimage.feature.canny`) + `hough_circle`/`hough_circle_peaks`. Tolerâncias em mm convertidas para pixels via `pixel_spacing`. Retorna lista de dicts `{slice_index, center_x, center_y, radius, accum, interpolated}`.
- `detect_initial_circle(img_slice, hough_radii, quadrant_offset=(30,30), ...)` — [aorta_localization.py:376](../src/utils/segmentation/aorta_localization.py#L376) — restringe a busca inicial ao quadrante anatômico esperado (`cx > width/2 - qx and cy < height/2 + qy`).
- `level_set_segmentation(volume_ccta, detected_circles, ...)` — [aorta_segmentation.py:152](../src/utils/segmentation/aorta_segmentation.py#L152). MGAC (`skimage.segmentation.morphological_geodesic_active_contour`) inicializado com discos nos círculos detectados (`radius_reduction_factor` default de config = 0.15, não 0.85 como no texto resumido do artigo — conferir contra o manuscrito publicado se for citar o número exato). Roda numa ROI recortada por padrão (`use_roi=True`) para custo computacional menor.
- `remove_leaks_morphology(mask_3d, radius=3)` — [aorta_segmentation.py:294](../src/utils/segmentation/aorta_segmentation.py#L294) — abertura morfológica 3D (`skimage.morphology.ball` + `binary_opening`), pós-processamento citado na tese.
- Wrappers de orquestração com cache em disco: `get_or_detect_aorta_circles()` e `get_or_segment_aorta()` — [pipeline_detection.py:15](../src/utils/segmentation/pipeline_detection.py#L15) e [pipeline_detection.py:84](../src/utils/segmentation/pipeline_detection.py#L84). Note que a segmentação da aorta e a detecção de círculos **não são o alvo do Cap. 2** — no bounding box do capítulo 2 não se segmenta a aorta, então essa etapa toda (d) é dispensável para o objetivo final, exceto como referência de convenção de eixos/spacing.
- **I/O:** `img_volume` é o volume 3D já com LCC aplicado (`lcc_image`), shape `(Y, X, Z)`, `pixel_spacing` em mm/pixel (escalar, média de `dx,dy`).

### (e) Mapas de vesselness (Frangi)

- `get_vesselness(image, sigmas, alpha, beta, gamma, black_ridges, normalization, smooth_sigma, gpu)` — [frangi.py:130](../src/utils/processing/frangi.py#L130). Frangi "clássico" (`skimage.filters.ridges.frangi` na CPU, `cucim.skimage.filters.frangi` na GPU se disponível). Corresponde à formulação original de Frangi et al. 1998 citada na tese.
- `get_modified_vesselness(...)` — [frangi.py:214](../src/utils/processing/frangi.py#L214). Pondera o Frangi clássico por uma razão grayness/gradiente (`Gf/Gd`, funções `get_gf`/`get_gd`) — é a variante "modified" mencionada no título do artigo, mas o **config default usa `method: "normal"`** tanto para `VESSELNESS_AORTA` quanto `VESSELNESS_ARTERY` (`pipeline_config.json:34,62`). Ou seja, apesar do nome do artigo, a execução padrão atual usa Frangi normal, não modificado — confirmar qual foi realmente usado nos resultados publicados antes de reaproveitar os parâmetros default como "os do artigo".
- Dois conjuntos de sigmas em config, batendo com a tese: `VESSELNESS_AORTA.sigmas = [2.5, 3.0]` (óstios/estruturas maiores) e `VESSELNESS_ARTERY.sigmas = [1.5, 2.0, 2.5, 3.0]` (árvore arterial, calibres variados).
- Orquestrado com cache por `get_or_compute_vesselness()` — [pipeline_preprocessing.py:208](../src/utils/segmentation/pipeline_preprocessing.py#L208). **Não é necessário para o Cap. 2** (o vesselness só serve para localizar óstios/artérias).

## 3. Estado de acoplamento: (a)-(c) são reutilizáveis?

**Sim, majoritariamente.** `downscale_image*`, `threshold_image_with_offset` e `largest_connected_component`
em `preprocessing.py` são funções puras (NumPy in/out, sem dependência do resto do pipeline de aorta/artéria).
Também não dependem de cache, disco ou config — recebem parâmetros escalares/arrays diretamente.

O ponto de acoplamento está um nível acima, em `load_and_preprocess_image()`
([pipeline_preprocessing.py:76](../src/utils/segmentation/pipeline_preprocessing.py#L76)), que:
- já orquestra (a)+(b)+(c) corretamente na ordem certa, **mas**
- está hardcoded para carregar um par imagem+label do ImageCAS via `load_raw_img_and_label` (não aceita um
  array já em memória),
- mistura a lógica de bifurcação `threshold_mode` (normal vs. fuzzy) — para o Cap. 2 só interessa o ramo
  `"normal"`,
- retorna também o `label` reamostrado e o vesselness-ready `scaled_spacing`, que não são necessários
  para bounding box (mas não atrapalham).

**Recomendação prática:** para o Cap. 2, não reaproveitar `load_and_preprocess_image()` inteira; extrair
apenas o trecho do ramo `"normal"` (linhas 109-155) como uma nova função menor, ou chamar diretamente
`downscale_image_opencv/ndi` → `resolve_lower_threshold` → `threshold_image_with_offset` →
`largest_connected_component` (per slice) em sequência — exatamente como já está escrito, só que sem o
branch fuzzy e sem depender do carregamento de label. Alternativamente, `run_core_preprocessing_pipeline()`
(preprocessing.py:205) já faz quase isso, mas usa `MIN_THRESHOLD` fixo (sem os métodos adaptativos de
`resolve_lower_threshold`) — seria a base mais rápida de adaptar já que ignora o label e o vesselness desde o início.

## 4. Dependências principais

- **Imagem/array:** `numpy`, `scipy.ndimage` (zoom, label, morfologia CPU), `opencv-python` (`cv2.resize`,
  interpolação), `scikit-image` (`feature.canny`, `transform.hough_circle*`, `segmentation.morphological_geodesic_active_contour`,
  `filters.ridges.frangi`, `morphology.ball/disk`).
- **GPU opcional:** `cupy` + `cucim` (detecção automática em `gpu_utils.py`, fallback silencioso para CPU se
  indisponível ou se a chamada falhar). Não é obrigatório para reproduzir o pipeline.
- **I/O de imagem médica:** `nibabel` (`nib.load(...).get_fdata()` / `Nifti1Image`) — arquivos `.img.nii.gz`
  e `.label.nii.gz` do ImageCAS, path pattern `{base_path}/{img_id}.img.nii.gz`.
- **Dataset/splits:** `sklearn.model_selection.train_test_split` como fallback; splits fixos e versionados em
  `config/imagecas_splits.json` (chave `"splits": {"train": [...], "val": [...], "test": [...]}`).
- **Resultados:** `pandas` (CSV por lote + resumo consolidado), `json` (metadata/config efetiva).

## 5. Convenção de eixos (ponto de atenção crítico para o Cap. 2)

- Volumes carregados via `nib.load(...).get_fdata()` mantêm a ordem de eixos do array NIfTI original — o
  pipeline **não transpõe** o array. A partir daí, o código trata consistentemente
  `volume.shape == (altura, largura, profundidade) == (Y, X, Z)`, isto é:
  - eixo 0 = linha/`y`,
  - eixo 1 = coluna/`x`,
  - eixo 2 = fatia axial/`z` (slices empilhados no último eixo — por isso o downscale usa `factors[2]=1`
    e os loops "por fatia" são sempre `[:, :, z]`).
  - Confirmado explicitamente em `ostia_detection.py:20-30` (`_validate_coordinates`: `y, x, z = coords;
    height, width, depth = volume_shape`) e em `aorta_segmentation.py:120-144` (`height, width = volume_shape[:2]`).
- **Spacing:** `nii_img.header.get_zooms()` é usado diretamente como `spacing[0], spacing[1], spacing[2]`
  multiplicado pelos mesmos `downscale_factors` na mesma ordem ([pipeline_preprocessing.py:192](../src/utils/segmentation/pipeline_preprocessing.py#L192)), então `scaled_spacing = (dx, dy, dz)` está alinhado a
  `(spacing_eixo0, spacing_eixo1, spacing_eixo2)` — ou seja, mesmo chamando de `dx`/`dy`, esses nomes
  correspondem a eixo0/eixo1 do array, não necessariamente a "x físico"/"y físico" do NIfTI. Isso só é seguro
  porque o resto do código é consistente com a mesma convenção nomeada "errado" em todo lugar.
- **Inconsistência a observar:** em `pipeline_steps.py`/`pipeline_orchestration.py:91`,
  `vesselness_spacing = (scaled_spacing[1], scaled_spacing[0], scaled_spacing[2])` **inverte** os dois
  primeiros componentes do spacing antes de repassar para `get_or_compute_vesselness` (usado por
  `get_gd`/`np.gradient`). Ou seja, o próprio pipeline já tem um lugar em que a ordem (eixo0, eixo1) é trocada
  intencionalmente para casar com a API de outra função. **Para a etapa de centróide/bounding box do Cap. 2,
  não há gradiente físico envolvido, então basta manter a convenção simples (eixo0=y, eixo1=x, eixo2=z) do
  volume; não copiar esse swap sem entender por que ele existe.**
- **Outro ponto de atenção:** `src/utils/utils/roi.py` (`extract_square_region`, `extract_circular_region`)
  usa nomes de variável **trocados** em relação ao resto do código: lá, `h, w, _ = image.shape` mas os
  parâmetros se chamam `x_min/x_max` (comparados contra `h`) e `y_min/y_max` (comparados contra `w`) — ou
  seja, na prática `x` aqui indexa o eixo 0 (que em todo o resto do código chamamos de `y`/altura). Se for
  reaproveitar `extract_square_region` para recortar o bounding box final, **renomear ou documentar
  explicitamente** essa inversão para não introduzir um bug de eixos trocados silencioso.

## 6. Caso a caso vs. lote

O pipeline já roda em lote sobre o dataset inteiro:

- `run_pipeline(ids, split_name, config, base_path, base_save_path, output_dir, resume_from_batch=0)` —
  [pipeline_orchestration.py:261](../src/utils/segmentation/pipeline_orchestration.py#L261) — recebe uma
  lista de IDs (tipicamente um split inteiro, até 700 casos de teste), divide em `NUM_BATCHES` lotes
  (default 5), processa cada imagem com `process_image()` e salva CSV incremental por lote + timing/manifests
  (útil para retomar em caso de falha — ver `--resume-batch`/`--resume-dir` em `pipeline_cli.py`).
- `get_data_splits(base_path, ...)` — [dataset.py:73](../src/utils/project/dataset.py#L73) — devolve
  `(train_ids, val_ids, test_ids, all_ids)`; com os parâmetros canônicos usa o split fixo versionado em
  `config/imagecas_splits.json` (930/... conferir splits reais no arquivo) em vez de sortear de novo a cada
  execução — importante para reprodutibilidade.
- **Para o Cap. 2:** já é possível rodar as etapas (a)-(d) sobre `all_ids` (as 1000 imagens) sem modificar a
  orquestração de lotes — bastaria um novo "process_image" mais simples (só até o LCC 3D + centróide) chamado
  dentro do mesmo laço de `run_pipeline`, ou um script separado que reusa `get_data_splits(..., split_config)`
  para iterar sobre os 1000 IDs e acumular as dimensões máximas observadas.
- Não há hoje nenhuma função "resuma estatísticas agregadas entre casos" (como max bounding box); tudo em
  `pipeline_orchestration.py`/`results.py` agrega por linha de CSV (uma linha = um caso), não por
  estatística global entre casos. Isso precisaria ser adicionado.

## 7. Validação/notebooks existentes como possível referência

Não há testes automatizados (`pytest`, etc.) nem scripts de validação de cobertura binária/volumetric-reduction
no repositório. Existem apenas notebooks de EDA em `src/eda/` (ex.: `aorta_circle_slice_analysis.ipynb`,
`cases_analysis.ipynb`, `threshold_pipeline_comparison_analysis.ipynb`) e scripts de sweep de parâmetros em
`src/experiments/` (`aorta_ostia_parameter_sweep.py`, `threshold_parameter_sweep.py`) — estes são bons
modelos de **como o projeto já faz varredura em lote + comparação de métricas por caso**, mas nenhum calcula
cobertura de bounding box ou redução volumétrica. `dice_score()` ([metrics.py](../src/utils/utils/metrics.py))
é o único utilitário de métrica pronto, reaproveitável apenas se a validação de cobertura for expressa como
uma métrica tipo-Dice (não é o caso — cobertura de bbox é mais simples: `label_artery[bbox].sum() ==
label_artery.sum()`, uma checagem binária por caso).

## 8. Plano de implementação proposto para as etapas 5-9 (NÃO implementado)

Sequência sugerida, reaproveitando o que já existe:

1. **Preprocess por caso (reuso direto):** para cada um dos 1000 IDs, chamar em sequência
   `downscale_image_opencv` (ou `_ndi`) → `resolve_lower_threshold` → `threshold_image_with_offset` →
   `largest_connected_component` por fatia — ou extrair o ramo `"normal"` de `load_and_preprocess_image()`
   para uma função dedicada `build_cardiac_volume(img_id, base_path, config) -> lcc_mask_3d` que **não**
   carrega/retorna o label nem calcula vesselness (evita custo desnecessário — Frangi e detecção de aorta são
   as etapas mais caras do pipeline atual e são inteiramente dispensáveis para o Cap. 2).
2. **Centróide geométrico (novo):** `centroid = np.array(np.nonzero(lcc_mask_3d)).mean(axis=1)` — resulta em
   `(y_c, x_c, z_c)` na mesma convenção de eixos do restante do código (Seção 5). Guardar por caso.
3. **Passo em lote sobre as 1000 imagens (novo, mas seguindo o padrão de `run_pipeline`/`get_data_splits`):**
   iterar `all_ids` (não só um split), aplicar passos 1-2, e acumular por caso: `(bbox_extent_y, bbox_extent_x,
   bbox_extent_z)` do maior componente conectado (ou diretamente `mask.shape` das coordenadas não-nulas,
   `max - min + 1` por eixo). Persistir esses extents por caso em um CSV/JSON simples (seguindo o padrão de
   `save_results`/`make_result_dataframe` de `project/results.py`, para manter consistência de formato com o
   resto do projeto).
4. **Envelope conservador (novo):** `fixed_extent = np.max(all_extents, axis=0)` sobre as 1000 medições —
   dimensão fixa do bbox nos 3 eixos, na resolução **reduzida** (mesma resolução do LCC). Escalar de volta
   para resolução integral multiplicando por `downscale_factors` (cuidado: downscale é só em XY, fator Z=1,
   então a extensão em Z já está na resolução original).
5. **Posicionamento por caso (novo):** para cada imagem, projetar o centróide (passo 2) de volta à resolução
   integral (multiplicar coordenadas pelos `downscale_factors`, na mesma ordem de eixos) e centralizar
   `fixed_extent` nesse ponto, clampando nas bordas do volume original (reaproveitar a lógica de
   clamp já usada em `_compute_local_roi_bounds` de `aorta_localization.py:188` ou em `_calculate_roi_bounds`
   de `aorta_segmentation.py:30` como referência de padrão, embora ambas sejam 2D+z-range e precisem de
   adaptação para um bbox 3D simétrico).
6. **Corte na imagem original (reuso parcial):** `extract_square_region` de `roi.py` faz algo próximo, mas
   com nomes de eixo trocados (Seção 5) e sem clamp elegante nos dois lados — melhor escrever uma função nova
   e pequena `crop_fixed_bbox(volume, centroid_full_res, fixed_extent)` do zero, para não herdar a confusão de
   nomenclatura.
7. **Validação de cobertura + Rv (novo):** por caso, com o label original em resolução integral,
   `covered = bool(np.array_equal(label_artery[bbox_slices], label_artery[bbox_slices] * ... ))` — mais
   simples: `covered = np.all(label_artery[full_mask_of_artery_outside_bbox] == 0)`, ou equivalentemente
   `label_artery.sum() == label_artery[y0:y1, x0:x1, z0:z1].sum()`. `Rv = 1 - (V_roi / V_original)` é um
   cálculo direto de volumes (`np.prod(fixed_extent) / np.prod(volume.shape)`), não depende de nenhuma função
   existente.
8. **Persistência dos resultados agregados:** seguir o padrão de `create_timestamped_output_dir` +
   `save_metadata`/`save_results` já usado pelo pipeline principal, para manter os artefatos do Cap. 2 no
   mesmo layout (`output/bounding_box/runs/<timestamp>/...`) e facilitar comparação futura.

## 9. Observações de risco

- **Paths fixos:** `segmentation_pipeline.py:47-49` tem paths absolutos hardcoded (`/media/matheus/HD/...`,
  `/data04/home/mpmaia/...`) específicos da máquina do mestrando, com fallback automático. Qualquer script
  novo para o Cap. 2 deve receber `base_path`/`base_save_path` como parâmetro (como já é feito via CLI) e não
  copiar esses defaults.
- **Dependências pesadas opcionais:** `cupy`/`cucim` são importados de forma defensiva (`try/except`) em
  praticamente todo módulo de processamento — não é necessário ter GPU para reproduzir os passos 1-4, mas o
  código sempre tenta detectar GPU no import (`gpu_utils.py`, `use_gpu()`), então rodar num ambiente sem CUDA
  deve funcionar via fallback, mas vale confirmar que não há import-time hard failure fora desse padrão.
- **Inconsistência de eixos/nomenclatura** já detalhada na Seção 5 — é o maior risco de bug silencioso ao
  portar centróide/bbox, porque o cálculo de centróide/dimensões máximas por eixo é exatamente o tipo de
  operação onde uma troca x↔y não quebra a execução, só produz um bbox descentrado sem erro visível.
- **`LCC_PER_SLICE` como premissa:** o Cap. 2 depende do LCC por fatia (igual ao artigo); o config permite
  desligar isso (`--lcc-per-volume`) para experimentos — se algum script novo herdar um `config` de execuções
  experimentais anteriores, confirmar explicitamente essa flag em vez de assumir o default.
- **Frangi/vesselness "normal" vs "modified" no config default:** como registrado na Seção 2(e), o nome do
  artigo sugere "modified", mas o `pipeline_config.json` atual usa `method: "normal"` para os dois mapas de
  vesselness. Isso não afeta o Cap. 2 (que não usa vesselness), mas é relevante se algum trecho de texto da
  tese comparar números do artigo assumindo a variante errada.
- **`run_core_preprocessing_pipeline` possivelmente órfã:** não encontrei chamadores fora de
  `preprocessing.py`; confirmar antes de adotá-la como base, para não herdar código não testado em produção.
- **Sem testes automatizados:** qualquer refator ou nova função de preprocessing deveria vir acompanhada de um
  teste mínimo (shape/dtype/eixos) já que não há suíte existente para pegar regressões de eixo trocado.
