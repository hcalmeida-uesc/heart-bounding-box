# Plano de Remoção de Código — Foco no Bounding Box Cardíaco

> Relatório de consulta, gerado antes de qualquer remoção. Nenhum arquivo foi apagado ainda.

**Critério usado:** mantém-se apenas o que é necessário para as etapas 1-9 descritas em
[docs/CODE_NOTES_bounding_box.md](CODE_NOTES_bounding_box.md) (downscale → threshold HU → LCC por
fatia → centróide → envelope máximo → posicionamento → corte → validação de cobertura/Rv) e sua
infraestrutura mínima de I/O/config/batch. Tudo que existe apenas para segmentar aorta, óstios ou
artérias coronárias — ou para comparar/depurar essas etapas — é candidato a remoção.

## 1. Módulos de segmentação arterial completa (remover por inteiro)

| Arquivo | Justificativa |
|---|---|
| `src/segmentation_pipeline.py` | Entry point do pipeline completo de segmentação coronária; será substituído por um script novo e menor específico do bounding box. |
| `src/utils/processing/frangi.py` | Filtro de Frangi/vesselness — usado só para localizar óstios e guiar o region growing arterial. |
| `src/utils/segmentation/aorta_localization.py` | Hough Circle Transform para rastrear a aorta — não faz parte do bounding box. |
| `src/utils/segmentation/aorta_segmentation.py` | Level Set 3D (MGAC) para segmentar a aorta — não é necessário para isolar o coração. |
| `src/utils/segmentation/ostia_detection.py` | Detecção/validação de óstios coronários — fora do escopo do Cap. 2. |
| `src/utils/segmentation/artery_segmentation.py` | Region growing guiado por vesselness para a árvore arterial. |
| `src/utils/segmentation/pipeline_arteries.py` | Orquestra segmentação arterial (vesselness + RG/FC + pós-processamento + Dice). |
| `src/utils/segmentation/pipeline_detection.py` | Orquestra detecção de círculos da aorta, Level Set e avaliação de óstios. |
| `src/utils/segmentation/pipeline_orchestration.py` | `process_image`/`run_pipeline` do pipeline completo (vesselness→aorta→óstios→artéria); o bbox precisa de uma orquestração muito mais simples. |
| `src/utils/segmentation/pipeline_steps.py` | Shim de compatibilidade que só reexporta os módulos acima. |
| `src/utils/segmentation/pipeline_cli.py` | Argparse com flags específicas de Hough/Level Set/Region Growing/Fuzzy Connectedness. |
| `src/utils/segmentation/pipeline_reporting.py` | Prints de resumo com métricas de óstios/Dice arterial. |
| `src/utils/segmentation/fuzzy_threshold.py` | Variante experimental de threshold (pertinência fuzzy); a tese usa o threshold HU "normal". |
| `src/utils/segmentation/fuzzy_connectedness.py` | Segmentação arterial alternativa por fuzzy connectedness. |
| `src/utils/segmentation/backend_comparison.py` | Compara execução CPU vs GPU de todo o pipeline arterial (vesselness, círculos, level set, RG). |

## 2. Scripts de experimentos/sweeps (remover por inteiro)

| Arquivo | Justificativa |
|---|---|
| `src/experiments/threshold_parameter_sweep.py` | Varredura de parâmetros de threshold+RG do pipeline arterial. |
| `src/experiments/aorta_ostia_parameter_sweep.py` | Varredura de parâmetros de Hough/óstios. |
| `src/experiments/fuzzy_pipeline_comparison.py` e `.ipynb` | Compara variantes normal vs. fuzzy do threshold/segmentação arterial. |
| `src/experiments/compare_cpu_gpu.py` | CLI fina sobre `backend_comparison.py` (já removido). |
| `src/experiments/runners/` | Scripts auxiliares dos sweeps acima. |
| `src/utils/experiments/sweep_common.py` | Helpers de CLI/paths compartilhados só pelos scripts de sweep acima. |
| `src/utils/experiments/fuzzy_pipeline_comparison.py` | Lógica de alternância normal/fuzzy usada pelo notebook de comparação. |

## 3. Utilitários de comparação/avaliação específicos do pipeline arterial

| Arquivo | Justificativa |
|---|---|
| `src/utils/comparison_utils/bad_cases.py` | Seleção de "casos ruins" de óstios/artéria para EDA. |
| `src/utils/comparison_utils/ia_math.py` | Comparação IA vs. determinístico específica do pipeline de artéria. |
| `src/utils/comparison_utils/io.py` | I/O auxiliar dos notebooks de comparação de segmentação. |
| `src/utils/comparison_utils/metadata.py` | Metadados de comparação de runs de segmentação arterial. |
| `src/utils/comparison_utils/ostia_scenarios.py` | Cenários de avaliação de óstios. |
| `src/utils/utils/metrics.py` (`dice_score`) | Métrica usada só para avaliar a segmentação arterial contra o label; a validação do bbox é cobertura binária, não Dice. |

## 4. Visualizações específicas do pipeline arterial

| Arquivo | Justificativa |
|---|---|
| `src/utils/visualization/hough.py` | Plots de diagnóstico da Transformada de Hough. |
| `src/utils/visualization/vesselness.py` | Plots de mapas de vesselness/Frangi. |
| `src/utils/visualization/segmentation_eda.py` | EDA visual da segmentação arterial/óstios. |
| `src/utils/visualization/bad_cases.py` | Visualização de casos ruins de óstios/artéria. |
| `src/utils/visualization/category.py` | Categorização visual de resultados de óstios/Dice. |
| `src/utils/visualization/comparison.py` | Comparação visual entre variantes do pipeline arterial. |
| `src/utils/visualization/variant_comparison.py` | Idem, versão genérica usada pelos sweeps (Seção 2). |
| `src/utils/visualization/images.py` | Só reexporta `hough.py` (removido) por compatibilidade. |
| `src/utils/visualization/subset.py` | Seleção de subconjuntos de casos para os notebooks de comparação de segmentação. |

*Mantidos* (uso genérico, não amarrado ao pipeline arterial): `image_slices.py`,
`preprocessing_views.py` (inteiros); `volume.py` foi **trimado**, não removido — mantidas só
`visualize_3d_k3d` e `save_k3d_plot_html` (renderização genérica de máscara 3D via marching cubes),
removidas `visualize_aorta_with_ostia`, `visualize_aorta_ostia_artery`,
`visualize_arteries_comparison` e os helpers `_add_mask_mesh`/`_add_ostium_point` (específicos de
aorta/óstios/artéria).

## 5. Notebooks EDA específicos do pipeline arterial

| Arquivo | Justificativa |
|---|---|
| `src/main.ipynb` | Executa o pipeline completo de segmentação coronária. |
| `src/eda/aorta_circle_slice_analysis.ipynb` | Análise dos círculos de Hough da aorta. |
| `src/eda/cases_analysis.ipynb` | Análise de casos por status de óstios/Dice arterial. |
| `src/eda/eda_tests_casos_ruins.ipynb` | Casos ruins de segmentação arterial. |
| `src/eda/eda_tests.ipynb`, `eda_tests_subsets.ipynb` | Comparações entre variantes do pipeline arterial. |
| `src/eda/eda_segmentacao.ipynb` | EDA dos resultados de segmentação arterial. |
| `src/eda/exemplo_operacoes_morfologicas.ipynb` | Exemplos de morfologia usada no pós-processamento arterial. |
| `src/eda/fuzzy_membership_functions.ipynb` | Demonstração do threshold fuzzy (Seção 1). |
| `src/eda/threshold_pipeline_comparison_analysis.ipynb` | Compara threshold normal vs. fuzzy + RG. |

*A revisar manualmente antes de decidir* (conteúdo não inspecionado célula a célula):
`src/eda/eda_imgs.ipynb`, `src/eda/viz_imgs.ipynb`.

## 6. Utilitários genéricos usados só pelos módulos removidos

| Arquivo | Justificativa |
|---|---|
| `src/utils/utils/normalization.py` | `normalize_vesselness`/`robust_normalize` só são usadas por `frangi.py` e `fuzzy_connectedness.py`. |
| `src/utils/utils/segmentation.py` (`segment_by_hu`) | Classificador grosseiro de tecido por HU; não faz parte das 9 etapas do bbox e não é chamado pelo restante do pipeline mantido. |
| `src/utils/utils/roi.py` | `extract_square_region`/`extract_circular_region` têm nomenclatura de eixos divergente do resto do código (ver `docs/CODE_NOTES_bounding_box.md`, Seção 5) e serão substituídas por uma função de corte nova e pequena, com eixos corretos, conforme já recomendado no plano de implementação. |

## 7. Suporte a GPU (remover por inteiro)

| Arquivo | Justificativa |
|---|---|
| `src/utils/processing/gpu_utils.py` | Nenhuma função da cadeia mantida (`downscale_image_ndi/opencv`, `threshold_image_with_offset`, `largest_connected_component`, `resolve_lower_threshold`) usa GPU; a aceleração só importava para Frangi/Hough/Level Set/Region Growing, todos removidos. |
| `src/utils/processing/binary_operations.py` | Closing/dilation/erosion/label com fallback GPU — usados apenas por `ostia_detection.py`, `aorta_segmentation.py` e `pipeline_arteries.py` (todos removidos); nenhuma das 9 etapas do bbox exige morfologia. |

Consequência: `USE_GPU`, `LOAD_CACHE`/`SAVE_CACHE` de vesselness e toda a plumbing de detecção de
GPU deixam de fazer sentido no config/CLI do bbox.

## 8. Arquivos a simplificar (não remover por inteiro)

| Arquivo | O que cortar | Motivo |
|---|---|---|
| `config/pipeline_config.json` | Chaves `VESSELNESS_AORTA`, `VESSELNESS_ARTERY`, `CIRCLE_DETECTION`, `LEVEL_SET`, `OSTIA_DETECTION`, `OSTIA_VALIDATION`, `ARTERY_SEGMENTATION`, `REGION_GROWING`, `FUZZY_CONNECTEDNESS`, `POSTPROCESSING`, `USE_GPU` | Só servem ao pipeline arterial; o bbox precisa apenas de `DOWNSCALE_*`, `MIN_THRESHOLD`, `MAX_THRESHOLD_PERCENTILE`, `LOWER_THRESHOLD`, `LCC_PER_SLICE`. Substituir por um config mínimo dedicado. |
| `src/utils/project/config.py` | Função `scale_config_to_resolution()` inteira; simplificar `normalize_runtime_config()` para não referenciar `CIRCLE_DETECTION`/`VESSELNESS_*` | Toda a lógica de escala é sobre parâmetros de Hough/Level Set/RG/FC em pixels. |
| `src/utils/project/results.py` | `RESULT_COLUMNS`, `READABLE_COLUMN_NAMES`, `build_result_row`, `make_result_dataframe`, `summarize_results_df`, `build_metadata` e afins | Schema inteiro é sobre óstios/Dice/artéria; o bbox precisa de um schema novo e bem menor (centróide, extents, cobertura, Rv). Manter só os helpers genéricos de timing/diretório (`create_timestamped_output_dir`, `duration_breakdown`, `batch_timing_*`). |
| `src/utils/project/notebook_env.py` | `get_default_split_paths`, `get_bad_cases_export_dir`, `get_cases_analysis_output_dir`, `get_cases_analysis_cache_dir` | Apontam para pastas de output do pipeline arterial/EDA antigo. Manter `configure_notebook_environment`, `resolve_imagecas_base_path`, `resolve_processed_imagecas_path` (genéricos). |
| `src/utils/groups/processing.py` | Entradas do dicionário que apontam para `frangi`, `gpu_utils`, `binary_operations` | Só devem sobrar os símbolos de `preprocessing.py`. |
| `src/utils/groups/io.py` | Entradas para `project.results` (schema antigo), `dice_score`, `extract_circular_region`/`extract_square_region`, `segment_by_hu` | Mesma razão das Seções 3, 6 e 8 acima. |
| `src/utils/segmentation/pipeline_preprocessing.py` | Branch `else` (`threshold_mode == "fuzzy"`) de `load_and_preprocess_image()` | Só serve ao threshold fuzzy (`fuzzy_threshold.py`, removido na Seção 1); a tese usa o threshold "normal". Manter só o branch `if threshold_mode == "normal"`. |
| `src/utils/processing/preprocessing.py` | Função `build_lcc_image_from_mask` | Fica órfã assim que o branch fuzzy acima for removido (era a única chamadora). Ver Seção 9.1 — **não** remover `largest_connected_component`, que é a LCC do isolamento cardíaco e continua em uso. |
| `pyproject.toml` | Dependências `cucim-cu12`, `cupy-cuda12x`, `morphsnakes` | Só usadas pelo suporte a GPU e pelo Level Set (`morphsnakes`) do pipeline arterial. **`k3d` foi mantido** — ainda é usado por `visualize_3d_k3d`/`save_k3d_plot_html` em `volume.py`, que permaneceram (ver Seção 4). |

## 9. Índices/lazy-loaders a atualizar junto

`src/utils/segmentation/__init__.py`, `src/utils/visualization/__init__.py`,
`src/utils/utils/__init__.py` e `src/utils/groups/segmentation.py` são dispatchers lazy
(`_SYMBOL_TO_MODULE`) que hoje apontam para os módulos listados acima. Não precisam de análise
individual — devem simplesmente ser podados dos símbolos removidos no mesmo commit, ou removidos
por inteiro se o submódulo inteiro esvaziar (ex.: `groups/segmentation.py` fica sem conteúdo
depois da Seção 1).

## 9.1 Verificação específica: funções de LCC

Conferido explicitamente porque a extração do maior componente conectado (LCC) é a etapa 3 do
bounding box — não pode ser removida por engano. Existem **duas funções distintas** com "LCC" no
nome/propósito, e é fácil confundi-las:

| Função | Onde vive | Papel | Status |
|---|---|---|---|
| `largest_connected_component` (+ `_find_largest_component_label`) | `src/utils/processing/preprocessing.py` | **A LCC do isolamento cardíaco** — usada por `load_and_preprocess_image()` no loop por fatia (etapa 3 da tese). | **Mantida** (Seção 10) — confirmado, não está em nenhuma lista de remoção. |
| `build_lcc_image_from_mask` | `src/utils/processing/preprocessing.py` | Generaliza a mesma lógica de LCC por fatia, mas só é chamada pelo branch `threshold_mode == "fuzzy"` de `load_and_preprocess_image()` (`fuzzy_threshold.py`, que **é removido** na Seção 1). | Fica **órfã** depois que o branch fuzzy for cortado de `pipeline_preprocessing.py` (ver Seção 8) — não remover o arquivo `preprocessing.py`, só essa função específica, no mesmo commit em que o branch fuzzy for retirado. |
| `keep_largest_component` | `src/utils/processing/binary_operations.py` | LCC genérica com fallback GPU, mas usada **apenas** para limpar a máscara da **aorta** pós Level-Set (`pipeline_detection.py:118`), nada a ver com isolamento cardíaco. | Corretamente na Seção 7 (remover junto com `binary_operations.py`) — remover não afeta a LCC do coração. |

**Conclusão:** a LCC que isola o coração (`largest_connected_component`, chamada por fatia em
`load_and_preprocess_image`) está segura e listada em "O que permanece" (Seção 10). O único ajuste
necessário no plano é explicitar, na Seção 8, que `build_lcc_image_from_mask` deve ser removida
junto com o branch fuzzy — não antes.

## 10. O que permanece (referência rápida)

- `src/utils/processing/preprocessing.py` (downscale, threshold, LCC)
- `src/utils/segmentation/lower_threshold.py` (limiar inferior adaptativo)
- `src/utils/segmentation/pipeline_preprocessing.py` — mas só o ramo `threshold_mode == "normal"`
  de `load_and_preprocess_image()`; o restante (branch fuzzy, cálculo de vesselness) some junto com
  a Seção 1.
- `src/utils/utils/nifti_io.py`, `src/utils/utils/json_io.py`
- `src/utils/project/dataset.py` (`get_data_splits`)
- `src/utils/project/cache.py`
- `config/imagecas_splits*.json`
- `src/utils/visualization/image_slices.py`, `preprocessing_views.py`, `volume.py` (uso a confirmar)

## 11. Fora do escopo deste relatório

- `doc/` (documentação Markdown do pipeline original) e `output/segmentation/*` (resultados/dados
  de execuções passadas) não são código-fonte; ficam de fora desta limpeza e podem ser arquivados
  separadamente se desejado.

## 12. Execução — o que foi feito

Executado em 2026-07-12/13, nenhum commit criado (mudanças ficam no working tree para revisão).

- Todas as remoções das Seções 1-7 foram aplicadas (arquivos deletados).
- `src/experiments/` foi removido por inteiro (Seção 2), incluindo `README.md`, que ficou órfão
  depois que todos os scripts que ele documentava foram removidos — não estava listado
  explicitamente no plano original, mas se enquadra no mesmo critério.
- Todos os trims da Seção 8 foram aplicados (`pipeline_preprocessing.py`, `preprocessing.py`,
  `project/config.py`, `project/results.py`, `project/notebook_env.py`, `groups/processing.py`,
  `groups/io.py`, `pipeline_config.json`, `pyproject.toml`).
- `volume.py` foi trimado (não removido) — ver nota corrigida na Seção 4.
- `pyproject.toml`: removidos `cucim-cu12`, `cupy-cuda12x[ctk]`, `morphsnakes`; **`k3d` mantido**
  (nota que corrige um erro do plano original, que o listava para remoção sem justificativa
  correta).
- Todos os índices/lazy-loaders da Seção 9 foram podados para os símbolos que restaram
  (`segmentation/__init__.py`, `visualization/__init__.py`, `processing/__init__.py`,
  `utils/__init__.py` de topo, `utils/utils/__init__.py`, `groups/segmentation.py`,
  `groups/visualization.py`).
- **Achado durante a execução, fora do plano original:** `src/utils/project/__init__.py` fazia
  imports diretos (não lazy) de símbolos já removidos (`scale_config_to_resolution`,
  `get_bad_cases_export_dir`, `get_default_split_paths`, `save_results`, `make_result_dataframe`,
  etc.) — quebraria a importação do pacote inteiro. Corrigido no mesmo commit lógico desta limpeza.
- `src/__init__.py`: docstring desatualizado (falava em "coronary centerline detection") corrigido
  para refletir o objetivo atual do repositório.
- Verificação: todos os `.py` restantes em `src/` (26 arquivos) passam em `ast.parse` sem erros de
  sintaxe. Não foi possível rodar um import real (`numpy`/`cv2`/`nibabel`/`scikit-image` não estão
  instalados no Python usado para a verificação) — recomenda-se rodar `uv sync` e um smoke-test de
  import antes do próximo commit.
- Estrutura final de `src/` após a limpeza:
  `src/__init__.py`, `src/eda/__init__.py`, `src/eda/eda_imgs.ipynb`,
  `src/utils/__init__.py`, `src/utils/groups/*.py`, `src/utils/processing/{__init__.py,preprocessing.py}`,
  `src/utils/project/{__init__.py,cache.py,config.py,dataset.py,notebook_env.py,results.py}`,
  `src/utils/segmentation/{__init__.py,lower_threshold.py,pipeline_preprocessing.py}`,
  `src/utils/utils/{__init__.py,json_io.py,nifti_io.py}`,
  `src/utils/visualization/{__init__.py,image_slices.py,preprocessing_views.py,volume.py}`.
