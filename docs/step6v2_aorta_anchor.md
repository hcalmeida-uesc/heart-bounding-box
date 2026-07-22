# Etapa 6 v2 — Âncora pela aorta (substitui a âncora por LCC)

> Parte da sequência do Capítulo 2 (Seção 2.3.1). Documenta a segunda tentativa da etapa 6, depois
> do achado bloqueante da primeira tentativa (via LCC) registrado em
> [step6_envelope.md](step6_envelope.md). Ver também [CODE_NOTES_bounding_box.md](CODE_NOTES_bounding_box.md).
>
> **Status em 2026-07-21**: a âncora pela aorta (centróide 3D via Hough+Level Set) continua válida
> e em uso. Mas a ideia de um **bbox de tamanho fixo** calibrado a partir dela — o "Caminho
> adotado" original descrito logo abaixo — foi **abandonada** depois de 3 variantes de calibração
> falharem em conciliar cobertura alta com `Rv` bom (ver seção "Decisão (2026-07-21)" mais adiante
> no documento). O pivô atual é um envelope individual por imagem, ainda não implementado.

## Por que uma segunda tentativa

A etapa 6 original (`docs/step6_envelope.md`) usava o centróide do `lcc_mask` (etapas 1-4) como
âncora para o bbox fixo. A sessão de 2026-07-15 mostrou que essa máscara isola o tronco inteiro, não
o coração — inviabilizando qualquer envelope calculado sobre ela (bbox ficaria perto do tamanho da
imagem original). **Confirmado em 2026-07-17** (ver [step6_envelope.md](step6_envelope.md#achado-bloqueante-lcc-não-isola-o-coração)):
isso não é uma limitação da reimplementação — o LCC nunca teve a intenção de isolar o coração no
pipeline original de Silva et al. (2026), sempre foi só limpeza grosseira antes da localização da
aorta. A âncora pela aorta adotada abaixo é, portanto, o mecanismo de localização que o pipeline
original sempre usou, não uma alternativa arbitrária.

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

## Execução do lote completo (1000 imagens) — concluída (2026-07-16), achado bloqueante novo

**Decisão (2026-07-15):** calibrar com o **maior split de treino disponível** entre os já
versionados no repositório — `config/imagecas_splits_train90.json` (90 casos de treino, 210 val,
700 test) — em vez do split default (`config/imagecas_splits.json`, só 30 casos de treino), para
reduzir o risco de um offset "conservador" subestimado por amostra pequena demais (ver ressalva que
motivou essa decisão, registrada abaixo em "decisões em aberto" antes de ser resolvida).
`imagecas_splits_train90.json` foi escolhido sobre `imagecas_splits_train60.json` (60 casos) por
ter mais casos de treino ainda; nenhum split com mais de 90 estava disponível no repo nesta data —
se um for criado depois, reconsiderar.

- **Comando**: `uv run python src/run_aorta_anchor.py --splits config/imagecas_splits_train90.json`
- **GPU**: não usada (`USE_GPU=false`, default) — ver achado de divergência algorítmica na sessão
  anterior (não repetir a configuração de GPU sem pedido explícito).
- **Custo estimado**: ~33-35s/caso em CPU × 1000 casos ≈ 9-10h. O processamento por caso
  (`block_<pasta>.csv` em `output/bounding_box/aorta_anchor/`) independe do split escolhido — só a
  calibração (`calibration_imagecas_splits_train90.json`) e a validação de cobertura
  (`coverage_val_imagecas_splits_train90.csv`, `coverage_test_imagecas_splits_train90.csv`) usam o
  split especificado. Rodando em background nesta sessão a partir de 2026-07-15; **retomável** — se
  interrompido, rodar o mesmo comando de novo pula os casos já processados (`status="ok"` nos CSVs
  de bloco).
- **Como acompanhar**: os CSVs de bloco em `output/bounding_box/aorta_anchor/block_*.csv` são
  atualizados incrementalmente, um caso por vez — dá para inspecionar o progresso a qualquer momento
  sem esperar o lote terminar (contar linhas com `status="ok"`).

A execução foi interrompida uma vez por um reinício não planejado da máquina (~04:42 de
2026-07-16, provavelmente atualização do Windows) com 766/1000 casos já processados; retomada com o
mesmo comando (`process_block`/`run_resumable_batch` pulou os casos já `status="ok"`) e concluída às
~11:00 de 2026-07-16. Tempo total efetivo de processamento: ~10h15min (excluindo o intervalo parado).

### Resultado — processamento por caso

**994/1000 casos processados com sucesso; 6 sem aorta detectada** (2 no split de treino, 0 no val,
4 no test — excluídos da calibração/validação, não substituídos por fallback, conforme decidido em
2026-07-15). Taxa de falha (0,6%) consistente com a execução canônica antiga citada acima (99,4%).

### Resultado — calibração (88 casos de treino válidos, `imagecas_splits_train90.json`)

```json
{
  "offset_min_reduced": {"y": -88.27, "x": -99.32, "z": -218.56},
  "offset_max_reduced": {"y": 151.97, "x": 108.03, "z": 94.02},
  "bbox_extent_reduced": {"y": 240.25, "x": 207.36, "z": 312.58}
}
```

### Resultado — cobertura (val/test, fora da calibração)

| Split | Cobertos | Total | Taxa |
|---|---|---|---|
| val | 203 | 210 | 96,67% |
| test | 659 | 696 | 94,68% |

**Não chegou a 100%** — 44 dos 906 casos de val+test (4,9%) ficam parcialmente fora do bbox
calibrado em pelo menos um eixo, mesmo com o split de treino maior disponível (88 vs. 30 casos).
Falhas espalhadas pelos 3 eixos (y: 2,7%, x: 1,7%, z: 2,2% no test), não concentradas em um só.
Magnitude do excesso (voxels fora do bbox, resolução reduzida) entre os que falham: mediana 6,9,
média 13,7, máximo 66,8 — a maioria é uma margem pequena, mas há casos claramente fora da faixa
vista no treino.

### ⚠️ Achado bloqueante novo: o bbox calibrado é enorme — `Rv` é ruim, às vezes negativo

`bbox_extent_reduced` acima, escalado para resolução integral
(`× downscale_factors = [2, 2, 1]`), dá **480,5 × 414,7 × 312,6 voxels**. Para comparação, as
imagens originais deste dataset têm tipicamente `512×512×(200-300)` voxels. Ou seja, o bbox
calibrado cobre **~94% da altura (y) e ~81% da largura (x) da imagem inteira**, e em **z chega a
ultrapassar o tamanho total** de boa parte dos casos (que têm 200-300 fatias, contra 312,6
necessários).

Calculei `Rv` real (não a estimativa otimista por caso da notebook) numa amostra de 60 casos reais,
usando o espaçamento físico de cada um:

- **Rv médio: 5,0%** (mediana 13,6%, desvio-padrão 11,8%, faixa: **-15,3% a 13,6%**)
- Valores negativos acontecem quando o bbox calibrado (312,6 fatias) excede o número real de
  fatias da imagem original daquele caso — o "bbox" seria fisicamente maior que a imagem inteira.

Isso **não é um bug de execução** — é uma consequência do método de calibração
(`calibrate_bbox_offsets`): o offset mínimo/máximo de cada eixo é tomado **independentemente**
entre os 88 casos de treino. Os casos que definiram os extremos são **pacientes diferentes em cada
eixo** (`argmin_case_per_axis`/`argmax_case_per_axis` no `calibration_*.json`: y_min do caso 934,
y_max do caso 428, x_min do caso 428, x_max do caso 90, z_min do caso 23, z_max do caso 44 — seis
casos distintos). O bbox final precisa satisfazer simultaneamente o extremo de y de um paciente, o
extremo de x de outro, e o extremo de z de um terceiro — uma combinação que nenhum paciente real
individual atinge, mas que o método de agregação por eixo produz de qualquer forma. **Ampliar o
split de treino piora esse efeito, não melhora**: mais casos aumenta a chance de ver um extremo
ainda maior em algum eixo isoladamente, sem necessariamente resolver a cobertura (que falha quando
um caso de fora tem um valor mais extremo que *qualquer* caso de treino num eixo, não uma
combinação).

**Isso ameaça a hipótese central do Cap. 2** (ganho simultâneo em qualidade *e* eficiência
computacional — VRAM, tempo de treino/inferência): um `Rv` de ~5% não representa redução de volume
relevante. Como no achado da etapa 6 v1 (LCC não isola o coração), estou documentando e parando
aqui em vez de escolher um caminho de correção por conta própria — decisão vinculada à validade
metodológica do capítulo.

**Evidência visual** (gerada em 2026-07-16 a partir de `output/bounding_box/aorta_anchor/`):

![Distribuição dos offsets por eixo, 994 casos, com o limite calibrado e o caso extremo que o definiu](img/step6v2_offset_histograms.png)

Para cada eixo, os offsets `label_min − âncora` (claro) e `label_max − âncora` (escuro) formam uma
distribuição razoavelmente concentrada — mas o limite calibrado (linha tracejada), definido pelo
extremo de só 88 casos, fica na cauda distante dessa distribuição mais ampla (994 casos), não perto
do grosso dela. É esperado que casos de fora da amostra de treino ultrapassem esse limite — a cauda
completa (994 casos) claramente vai além do que 88 casos capturaram, principalmente no eixo z.

**Atualizado em 2026-07-17:** o histograma agora também marca, por eixo/lado, os percentis
80/85/90/95/100% da distribuição completa (994 casos) — traços curtos, mais escuros quanto mais
perto de 100%, mostrando visualmente quantos casos ficariam de fora em cada nível de cobertura
marginal (`compute_coverage_percentiles()` em
[plot_aorta_anchor_extremes.py](../src/eda/plot_aorta_anchor_extremes.py), tabela apresentada em
[aorta_anchor_calibration_summary.ipynb](../src/eda/aorta_anchor_calibration_summary.ipynb),
Seção 2.1). Isso quantifica exatamente o que o parágrafo acima descreve: em **todos os 6
eixo/lado**, o limite calibrado com 88 casos já fica aquém do extremo real dos 994 processados —
gap de 10,7 a 24,5 voxels em y, 36,4 a 66,8 em x, 33,4 a 54,0 em z (resolução reduzida).

![20 piores casos de cobertura fora da calibração, por eixo responsável](img/step6v2_worst_coverage_cases.png)

O eixo **z é o maior responsável pelas piores falhas de cobertura** (9 dos 20 piores casos), à
frente de x (6) e y (3) — coerente com o offset em z ter a maior amplitude entre os três
(`bbox_extent_reduced.z = 312,6`, contra 240,2 em y e 207,4 em x). O pior caso individual (IMG_ID
132, split test) fica ~67 voxels fora do bbox calibrado num dos eixos.

Gerado por `src/eda/plot_aorta_anchor_extremes.py` (`uv run python src/eda/plot_aorta_anchor_extremes.py`):
carrega `all_cases.csv` + `calibration_imagecas_splits_train90.json` +
`coverage_{val,test}_imagecas_splits_train90.csv`, calcula `offset_min/max = label_min/max −
âncora` por caso e eixo, plota histogramas com `matplotlib` marcando o limite calibrado e o caso
que o define (via `argmin_case_per_axis`/`argmax_case_per_axis`), e um gráfico de barras dos piores
casos de cobertura por excesso além do bbox. Reaproveitável se a calibração for refeita.

**Atualizado em 2026-07-17** — o histograma agora também marca, por eixo/lado, os percentis
80/85/90/95/100% da distribuição completa (994 casos): traços curtos, mais escuros quanto mais
perto de 100%, mostrando visualmente quantos casos ficariam de fora em cada nível de cobertura
marginal (`compute_coverage_percentiles()`, mesmo módulo). Em **todos os 6 combinações de
eixo/lado**, o limite calibrado com 88 casos já fica aquém do extremo real dos 994 processados —
gap de 10,7 a 24,5 voxels em y, 36,4 a 66,8 em x, 33,4 a 54,0 em z (resolução reduzida). Tabela
completa e comparação numérica em
[aorta_anchor_calibration_summary.ipynb](../src/eda/aorta_anchor_calibration_summary.ipynb),
Seção 2.1.

### Explorado: recalibração por percentil em vez de min/max — não abre um meio-termo viável

Testado em 2026-07-17, sem re-rodar o lote (reaproveita `all_cases.csv` + `coverage_*.csv` já
salvos): em vez de calibrar cada eixo pelo mínimo/máximo absoluto dos 88 casos de treino (nível de
cobertura 100%, o método atual), recalibrar por um percentil menos conservador — aceitando que uma
fração pequena dos próprios casos de treino fique fora do bbox — e medir o efeito na cobertura
**conjunta** (3 eixos simultâneos, não marginal) dos 906 casos de val+test e no `Rv` real (mesma
amostra de 60 casos da Seção 4).

| Cobertura no treino | Extensão do bbox (resolução integral, y×x×z) | Cobertura conjunta fora da calibração (val+test) | `Rv` médio |
|---|---|---|---|
| 100% (método atual) | 480,5 × 414,7 × 312,6 | 95,14% | 5,01% |
| 95% | 407,2 × 357,0 × 253,1 | 75,94% | 43,91% |
| 90% | 386,0 × 334,3 × 236,3 | 59,60% | 53,50% |
| 85% | 373,7 × 318,3 × 226,0 | 45,70% | 59,00% |
| 80% | 361,1 × 307,5 × 217,3 | 33,44% | 63,20% |

**Não existe meio-termo razoável nessa faixa**: já relaxar de 100% para 95% de cobertura no treino
(uma redução pequena, em tese) faz a cobertura conjunta fora da calibração despencar de 95,14% para 75,94% —
uma queda muito mais que proporcional — enquanto o `Rv` melhora de 5,01% para 43,91%. Não há um
ponto onde cobertura fique perto de 100% **e** `Rv` seja relevante ao mesmo tempo: ou a cobertura é
aceitável e o `Rv` é ruim (o achado já documentado acima), ou o `Rv` melhora e a cobertura
desmorona bem abaixo do que o capítulo exige.

Reproduzível via
[aorta_anchor_calibration_summary.ipynb](../src/eda/aorta_anchor_calibration_summary.ipynb),
Seção 4.1 (`calibrate_at_level`/`joint_coverage_rate`/`rv_for_extent` — generalizam
`calibrate_bbox_offsets`/`validate_coverage` para qualquer percentil, sem reprocessar o dataset).

**O que isso descarta e o que não descarta**: descarta a versão simples de "relaxar a agregação por
eixo com um corte de percentil marginal" — a sub-ideia mais direta do item 1 abaixo — como solução
suficiente sozinha; a queda de cobertura fora da calibração é desproporcional demais a qualquer ganho de `Rv`
que valha a pena. **Não descarta** abordagens que respeitem a correlação entre eixos por paciente
(ex.: um envelope conjunto/percentil multivariado, considerando os 3 eixos ao mesmo tempo por
caso, não marginal eixo a eixo) — essa variante mais sofisticada do item 1, e as demais direções
listadas abaixo, continuam em aberto.

### Explorado (2026-07-21): calibração conjunta por paciente único — também não resolve

Testado o restante do item 1 (a variante "que considere os 3 eixos conjuntamente por caso", citada
acima como ainda em aberto): em vez de combinar o extremo de cada eixo de pacientes diferentes
(`calibrate_bbox_offsets`) ou relaxar cada eixo isoladamente por percentil marginal (seção
anterior), usar o extent 3D de **um único paciente real** de treino — um triplete `(y, x, z)`
coerente, vindo do mesmo caso — como o bbox calibrado inteiro. Implementado em
`calibrate_bbox_size_joint()` (`src/utils/segmentation/aorta_anchor.py`, 8 testes em
`tests/test_aorta_anchor.py`), compatível com `validate_coverage()` sem alteração (mesmo schema de
dict de calibração que `calibrate_bbox_offsets()`).

Testados dois critérios de seleção do paciente de referência entre os 88 de treino:

| Critério de seleção | Caso escolhido | Extent (y, x, z), resolução reduzida | Cobertura conjunta (fora da calibração) | `Rv` médio |
|---|---|---|---|---|
| Maior volume (`extent_y × extent_x × extent_z`) | 291 | 176, 136, 224 | **1,99%** | 67,3% |
| Mais equilibrado (maior percentil mínimo entre os 3 eixos) | 570 | 178, 143, 205 | **12,69%** | 68,2% |

Ambos dão `Rv` ótimo, mas cobertura piores que o método marginal atual (95,14%) — não é uma opção
viável como está. Diagnóstico do caso 291 (maior volume): falha em 88% dos casos no eixo y e 95,7%
no eixo x, mas só 26,7% no eixo z — ele é extenso em z (por isso "maior volume"), mas estreito em x
e y. **"Maior volume" não implica "grande nos 3 eixos ao mesmo tempo"** — daí o critério
"equilibrado" (rank percentual mínimo entre os 3 eixos) ter sido testado como alternativa, com
melhora real (1,99% → 12,69%) mas ainda muito aquém do necessário.

**Conclusão**: nenhum paciente individual real, por melhor que seja escolhido, tem porte suficiente
para servir de envelope para os outros ~900 casos. Não é um problema do critério de seleção — é a
variabilidade anatômica entre pacientes (relativa à âncora da aorta) sendo grande demais para
qualquer método baseado em tamanho(s) real(is) observado(s), seja um paciente só ou min/max entre
vários. **Isso esgota, nas variantes testadas, a direção 1** (marginal-percentil e paciente-único
em 2 critérios) — ver decisão abaixo.

Reproduzível via consulta direta a `all_cases.csv` + `calibrate_bbox_size_joint()` +
`validate_coverage()`, sem reprocessar o dataset (mesmo padrão das explorações anteriores).

### Decisão (2026-07-21): abandonar bbox de tamanho fixo, pivotar para envelope individual por imagem

Com a direção 1 esgotada nas variantes testadas, Hélder decidiu **abandonar a ideia de um tamanho
de bbox fixo compartilhado por todas as imagens** (o desenho original da etapa 6 desde o início do
capítulo) e pediu um **envelope individual por imagem** — cada imagem com seu próprio tamanho de
bbox, não um tamanho único aplicado a todas.

**Isso ainda não foi implementado.** Ficou pendente uma decisão metodológica importante antes de
implementar: **de onde deve vir o tamanho do envelope de cada imagem?** Se vier do label verdadeiro
daquela mesma imagem, o resultado tem cobertura 100% trivial e `Rv` ótimo — mas usar o label de
val/test para definir o pré-processamento deles seria vazamento de informação que o pipeline não
teria disponível numa aplicação real (o Cap. 2 existe para localizar a região sem saber de antemão
onde está a artéria). Perguntado a Hélder; ele pediu para não aprofundar por ora e só consolidar o
que foi encontrado hoje (esta seção + commit). **Retomar essa pergunta explicitamente antes de
implementar qualquer envelope individual** — não presumir o critério.

### Possíveis direções (histórico — 1, marginal e paciente-único, esgotada; ver decisão acima para o rumo atual)

1. ~~Repensar a agregação da calibração~~ — testado em duas variantes (percentil marginal,
   2026-07-17; paciente único por 2 critérios, 2026-07-21), nenhuma resolve. Ver seções acima.
2. Revisar se a âncora (centróide da aorta) é o ponto de referência certo — a Opção B cogitada em
   2026-07-15 (ponto médio dos óstios, mais próximo anatomicamente do início da árvore coronária)
   poderia reduzir a dispersão dos offsets por eixo. **Detecção de óstios de verdade continua não
   testada.** Um proxy barato (`z_min` da máscara da aorta, sem detecção de óstios — ver
   "Diagnóstico exploratório (2026-07-22)" acima) foi testado como forma rápida de checar a
   hipótese antes de investir na detecção real: a dispersão em z **piorou**, não melhorou, e
   `z_min` mostrou-se sensível a onde o rastreamento Hough+Level Set termina (não uma referência
   anatômica estável). Isso não descarta a Opção B (óstios reais seriam uma referência anatômica
   precisa, diferente de `z_min`), mas também não a confirma — decisão sobre investir nisso segue
   em aberto.
3. Investigar os casos mais extremos (excesso de 30-67 voxels na cobertura, ou os 6 argmin/argmax
   da calibração) individualmente — podem ser outliers anatômicos genuínos ou indicar problema no
   label/dado de origem. Não testada.
4. **Adotada em 2026-07-21** (ver "Decisão" acima): abandonar bbox de tamanho fixo, envelope
   individual por imagem. Critério de tamanho ainda em aberto.

### Onde estão os resultados

- `output/bounding_box/aorta_anchor/all_cases.csv` — os 1000 casos processados (âncora, limites do
  label, status).
- `output/bounding_box/aorta_anchor/calibration_imagecas_splits_train90.json` — calibração completa.
- `output/bounding_box/aorta_anchor/coverage_val_imagecas_splits_train90.csv` /
  `coverage_test_imagecas_splits_train90.csv` — cobertura por caso e por eixo nos splits de fora.
- `output_run_aorta_anchor.log` (raiz do repo) — log completo da execução, incluindo a interrupção
  e retomada.

## Detalhe técnico confirmado (2026-07-21): qual lado da máscara da aorta é a raiz

Pergunta prática para qualquer posicionamento futuro que precise diferenciar "lado da raiz" (onde
ficam os óstios coronários) de "lado do arco" na máscara da aorta — depende da convenção de
orientação do NIfTI deste dataset, não é assumível sem checar.

**Orientação do NIfTI**: `nibabel.aff2axcodes()` em 5 casos de blocos diferentes do dataset
(`1-200` a `801-1000`) deu consistentemente `('L', 'A', 'S')` — o eixo z (3º eixo do array, o
mesmo `z` da convenção `(Y, X, Z)` usada em todo o projeto) cresce em direção **Superior
(craniana)**. `z_max` = mais próximo da cabeça; `z_min` = mais próximo dos pés.

**Lado da raiz**: verificado empiricamente (não só por dedução anatômica) — segmentando a aorta de
verdade (`segment_aorta`) em 3 casos reais de blocos diferentes e medindo a distância 3D mínima
(mm, com espaçamento físico) entre cada extremo da máscara da aorta (`z_min`, `z_max`) e o label
coronário real do mesmo caso:

| IMG_ID | aorta z=[min,max] (resolução reduzida) | dist. z_min → label | dist. z_max → label |
|---|---|---|---|
| 1   | [94, 274]  | **21,2 mm** | 53,2 mm |
| 250 | [138, 274] | **15,7 mm** | 39,6 mm |
| 500 | [157, 274] | **11,8 mm** | 18,2 mm |

Em todos os 3 casos, `z_min` da máscara da aorta fica 2-3× mais perto do label coronário do que
`z_max`. **Conclusão: `z_min` é o lado da raiz; `z_max` é o lado do arco** (mais distante do
coração).

**Por que faz sentido**: `z_max` bate exatamente em `num_slices − 1` nos 3 casos — não é
coincidência, é onde `detect_aorta_circles` **começa** a busca (última fatia do volume) e rastreia
para trás (z decrescente) até a geometria deixar de ser circular o bastante para o Hough, o que
acontece perto do anel/seios de Valsalva, na raiz — por isso o rastreamento tende a **parar**
perto da raiz (`z_min` observado), não a começar lá.

**Ressalva**: `('L','A','S')` é a orientação confirmada para este dataset/pipeline de carregamento
(`load_raw_img_and_label` via `nibabel`, sem reorientação), não uma garantia universal de todo
arquivo NIfTI — reconfirmar se outro dataset for usado no futuro.

Reproduzível chamando `segment_aorta(lcc_image, scaled_spacing, config)` (de
`aorta_anchor.py`) com a saída de `load_and_preprocess_image` para qualquer `img_id`, depois
`compute_bounding_box_bounds` na máscara resultante e no label (`label == 1`) para obter os
extremos em z e medir a distância até o label como acima.

## Diagnóstico exploratório (2026-07-22): a âncora atual mistura arco+descendente no eixo z?

Antes de decidir se vale reintroduzir a detecção de óstios (mais cara, mais código), testada a
hipótese de que parte da dispersão do offset em z (que motivou o bbox de 480×414×312, ver achado
acima) vem de ruído de captura — a extensão do trecho descendente da aorta dentro do campo de
visão varia por protocolo de aquisição, não por anatomia coronária — em vez de variabilidade
anatômica real entre pacientes. Se assim fosse, trocar o centróide 3D da máscara inteira por uma
âncora baseada só no lado proximal (`z_min`, o lado da raiz confirmado na seção anterior) deveria
reduzir a dispersão do offset em z.

**Método**: amostra estratificada de 50 casos (10 por bloco × 5 blocos, seed=42), reaproveitando
`label_min`/`label_max` e a âncora atual (centróide) já salvos em `all_cases.csv` — único
reprocessamento foi rodar `segment_aorta` de novo em cada caso da amostra, só para extrair `z_min`
via `compute_proximal_z_anchor` (nova função em `bounding_box.py`, wrapper sobre
`compute_bounding_box_bounds`). y/x da âncora candidata usam o mesmo centróide da âncora atual —
controle, não deveriam mudar entre as duas versões. Não altera nenhum artefato de produção:
script em `src/eda/diagnose_z_anchor_dispersion.py`, saída em
`output/bounding_box/aorta_anchor_z_diagnosis/` (pasta nova).

**Checagem de sanidade**: `z_min < z_max` em todos os 50 casos (sem inversão de lado).

**Resultado — dispersão do offset (`label − âncora`), desvio-padrão e amplitude (max−min) entre
os 50 casos, para as duas âncoras:**

| Eixo | Lado | std atual | amplitude atual | std candidata | amplitude candidata | Δ std | Δ amplitude |
|---|---|---|---|---|---|---|---|
| y | mín | 19,86 | 132,84 | 19,86 | 132,84 | 0,0% | 0,0% |
| y | máx | 20,00 | 103,84 | 20,00 | 103,84 | 0,0% | 0,0% |
| x | mín | 12,75 | 62,77 | 12,75 | 62,77 | 0,0% | 0,0% |
| x | máx | 12,20 | 53,27 | 12,20 | 53,27 | 0,0% | 0,0% |
| z | mín | 26,75 | 122,32 | 32,86 | 182,00 | **−22,8%** | **−48,8%** |
| z | máx | 22,07 | 126,30 | 28,34 | 206,00 | **−28,4%** | **−63,1%** |

(Δ negativo = a dispersão da âncora candidata é *maior* que a atual, não menor.) y/x com Δ
exatamente 0,0% nos 50 casos — confirma que o controle funcionou (mesma referência y/x nas duas
âncoras). Em z, a dispersão **piora** com a âncora candidata nos dois lados e nas duas métricas.

**Achado adicional**: `z_min` (âncora candidata) correlaciona com `num_circles_detected`
(r=−0,68) e com o tempo de processamento do caso (r=−0,71) na amostra — quanto mais círculos o
Hough/Level Set detecta (rastreamento mais longo), menor o `z_min` alcançado. Ou seja, `z_min`
reflete principalmente até onde o rastreamento consegue avançar (sensível a contraste, ruído,
parâmetros de tolerância do algoritmo) — não é uma referência anatômica estável, ao contrário do
centróide, que é uma média sobre todos os voxels da máscara e por isso menos sensível a onde
exatamente o rastreamento termina.

**Interpretação**: a hipótese não se sustenta nesta amostra — trocar o centróide por `z_min` como
componente z da âncora aumenta a dispersão do offset em z (22-28% em desvio-padrão, 49-63% em
amplitude), em vez de reduzi-la. A dispersão observada não parece ser, predominantemente, ruído de
FOV do trecho descendente absorvido pelo centróide; parece estar mais relacionada à
variabilidade de onde o rastreamento Hough+Level Set termina (achado adicional acima) e/ou a
variabilidade anatômica real entre pacientes — este diagnóstico não separa as duas. Decisão sobre
reintroduzir detecção de óstios (ou outra direção) não tomada aqui.

Reproduzível: `uv run python src/eda/diagnose_z_anchor_dispersion.py` (retomável via
`candidate_anchor_z.csv`; `--fresh` reprocessa do zero, `--sample-per-block N`/`--seed S` ajustam
a amostra). Dados: `output/bounding_box/aorta_anchor_z_diagnosis/{candidate_anchor_z,
offsets_detail,dispersion_comparison}.csv`.

## Diagnóstico complementar (2026-07-23): `num_circles_detected` explica o offset de calibração?

Extensão do diagnóstico anterior (2026-07-22, amostra de 50 casos) para a **população inteira**
(994 casos) e para o **offset de produção** (`label − anchor_z`, âncora = centróide, não o `z_min`
candidato testado ontem). Sem reprocessar nada: usa só `all_cases.csv`,
`calibration_imagecas_splits_train90.json` e `coverage_{val,test}_imagecas_splits_train90.csv`, já
salvos — `num_circles_detected` já estava persistido em todos os 994 casos `ok` (confirmado antes
de rodar esta análise). Script: `src/eda/analyze_num_circles_offset_correlation.py` (reproduz
localmente o ranking dos 20 piores casos com a mesma lógica de
`plot_aorta_anchor_extremes.plot_worst_coverage_cases`, sem chamar essa função — evita
re-salvar o PNG de produção).

**Correlações (994 casos), offset bruto e desvio absoluto da mediana, vs. `num_circles_detected`:**

| Comparação | Pearson r | Pearson p | Spearman r | Spearman p |
|---|---|---|---|---|
| `offset_min_z` (bruto) | 0,317 | <0,0001 | 0,085 | 0,0075 |
| `offset_max_z` (bruto) | 0,424 | <0,0001 | 0,046 | 0,148 (n.s.) |
| \|`offset_min_z` − mediana\| | 0,078 | 0,013 | 0,042 | 0,182 (n.s.) |
| \|`offset_max_z` − mediana\| | 0,235 | <0,0001 | 0,219 | <0,0001 |

Pearson e Spearman **divergem** bastante em 3 das 4 linhas (Pearson moderado/significativo,
Spearman fraco ou não-significativo) — sinal de que a correlação linear não é robusta a
outliers/não-linearidade, não uma relação monotônica limpa entre as variáveis.

**Sinal — poucos vs. muitos círculos (corte na mediana, 106 círculos):**

| Grupo | n | mediana de círculos | `offset_min_z` médio | `offset_min_z` mediana | `offset_max_z` médio | `offset_max_z` mediana |
|---|---|---|---|---|---|---|
| poucos círculos (≤ mediana) | 510 | 89 | −156,54 | −154,84 | 18,56 | 20,56 |
| muitos círculos (> mediana) | 484 | 126 | −151,95 | −155,30 | 22,98 | 17,79 |

Diferença de mediana entre os grupos é pequena (~0,5 em `offset_min_z`, ~2,8 em `offset_max_z`,
e inconsistente em sinal entre média e mediana) — **na população inteira, dividir por poucos vs.
muitos círculos não separa grupos com offset visivelmente diferente**, ao contrário do que as
correlações de Pearson por si só sugeririam.

**20 piores casos de cobertura (fora da calibração) x `num_circles_detected`:**

| Split de circles | n | Casos (img_id, nº círculos) |
|---|---|---|
| — | — | 132(47) 622(275) 738(11) 790(226) 306(9) 924(2) 777(17) 196(275) 593(7) 321(116) 823(261) 218(24) 73(81) 383(12) 347(23) 719(97) 601(183) 191(231) 202(107) 636(41) |

- População (994 casos): mediana 106, média 107,7, desvio-padrão 32,9.
- 20 piores: mediana **64,0** (bem abaixo da população), média 102,2 (próxima da população).
- 12 dos 20 (60%) ficam abaixo da mediana da população.
- **O mínimo (2 círculos, caso 924) e o máximo (275 círculos, casos 622 e 196) de toda a
  população de 994 casos aparecem entre os 20 piores** — não só a cauda baixa.

A divergência entre mediana (bem abaixo da população) e média (próxima da população) do grupo dos
20 piores já indicava que a distribuição deles não é simplesmente deslocada para baixo — o achado
do min/max confirma: há dois subgrupos nos 20 piores, um com poucos círculos (rastreamento
incompleto: 924, 593, 306, 738, 383, 777, 347, 218, 636, 132 — todos com <50 círculos) e outro com
muitos círculos, alguns nos extremos superiores da população inteira (622, 196, 823, 191, 601 —
todos ≥183 círculos). O eixo `z` é o mais frequente como `worst_axis` em ambos os subgrupos.

**Interpretação**: `num_circles_detected` não tem uma relação linear/monotônica simples e forte
com o offset de produção na população inteira (divergência Pearson/Spearman, grupos por mediana
quase idênticos) — mais fraco do que a correlação vista ontem numa amostra de 50 usando `z_min` da
máscara em vez do offset de produção. Mas o cruzamento com os 20 piores casos de cobertura sugere
que os **extremos** de `num_circles_detected` (rastreamento muito curto **ou** muito longo — não
só curto) estão sobrerrepresentados entre as falhas, incluindo o mínimo e o máximo observados em
toda a população. Isso é consistente com uma relação não-monotônica (pior nos dois extremos,
melhor no meio) — que explicaria por que correlações lineares/de rank ficam fracas mesmo havendo
sinal real nos extremos. Não testado formalmente aqui (fora do escopo desta rodada); decisão sobre
próximos passos não tomada nesta análise.

Reproduzível: `uv run python src/eda/analyze_num_circles_offset_correlation.py`. Dados:
`output/bounding_box/aorta_anchor_z_diagnosis/{num_circles_offset_detail,
num_circles_offset_correlations,num_circles_offset_sign_summary,worst_coverage_num_circles}.csv`.

## Diagnóstico complementar (2026-07-24): falha de cobertura em z por quintil de `num_circles_detected`

Continuação não-paramétrica dos diagnósticos de 2026-07-22/23 (correlação linear/rank, que assume
relação monotônica e por isso pode mascarar um padrão em U). Aqui os 906 casos de val+test são
agrupados em **quintis** de `num_circles_detected` (~181 casos por faixa) e reportamos a taxa de
falha de cobertura por faixa — sem ajustar modelo quadrático nem qualquer outra forma funcional,
propositalmente, para ver o padrão bruto antes de impor uma hipótese.

Sem reprocessar nada: `num_circles_detected` já vem direto em `coverage_{val,test}_imagecas_
splits_train90.csv` (confirmado por join com `all_cases.csv` via `img_id` — os valores batem em
todos os 906 casos, então nenhum merge era de fato necessário para uso, só para checagem da chave).
Script: `src/eda/analyze_num_circles_quintile_coverage.py`.

**Correção a um número citado na solicitação**: os "44 casos" já documentados na seção de
cobertura val/test são a falha **conjunta** (`covered`==False, qualquer eixo), não uma contagem
específica de z. A falha binária **só em z** é **18** de 906 (1,99%); em y é 22 (2,43%); em x é 16
(1,77%).

**Taxa de falha de cobertura por quintil de `num_circles_detected` (906 casos):**

| Quintil | `num_circles_detected` | N | Falhas y (n / %) | Falhas x (n / %) | Falhas z (n / %) |
|---|---|---|---|---|---|
| 1 | 2 – 83 | 182 | 12 / 6,59% | 6 / 3,30% | 11 / 6,04% |
| 2 | 84 – 99 | 186 | 1 / 0,54% | 1 / 0,54% | 0 / 0,00% |
| 3 | 100 – 112 | 176 | 1 / 0,57% | 4 / 2,27% | 0 / 0,00% |
| 4 | 113 – 130 | 187 | 0 / 0,00% | 2 / 1,07% | 0 / 0,00% |
| 5 | 131 – 275 | 175 | 8 / 4,57% | 3 / 1,71% | 7 / 4,00% |

**Padrão em U visível a olho nu em z (e também em y), sem ajuste de curva**: as duas faixas
extremas de `num_circles_detected` (menos de 84 círculos, ou mais de 130) concentram **18 das 18
falhas de z** (11 + 7 = 18 = 100%) e a maior parte das falhas de y (20 de 22 = 91%) — as três
faixas centrais (84–130 círculos, ~55% dos 906 casos) têm falha de z **zero**. O eixo x, usado como
controle, não segue o mesmo padrão de forma tão limpa: tem sua maior taxa na faixa central
(quintil 3, 2,27%) e não zera nas faixas do meio — é mais plano/ruidoso, sem o mesmo formato em U
nítido de y e z.

Isso é consistente com a hipótese de não-monotonicidade levantada no diagnóstico de 2026-07-23
(mínimo e máximo populacional de `num_circles_detected` entre os 20 piores casos de cobertura) —
aqui, agrupando por faixa em vez de correlacionar, o padrão em U fica visualmente muito mais claro
para z e y do que a correlação linear/rank sozinha sugeria.

**5 piores casos por excesso em z (voxels fora do bbox, não binário) entre os 906:**

| img_id | split | `num_circles_detected` | excesso em z (voxels) | `covered_z` |
|---|---|---|---|---|
| 622 | test | 275 | 53,97 | False |
| 790 | val | 226 | 39,84 | False |
| 924 | test | 2 | 33,38 | False |
| 777 | test | 17 | 31,57 | False |
| 196 | test | 275 | 30,62 | False |

Todos os 5 **coincidem** com casos já listados em `worst_coverage_num_circles.csv` (diagnóstico de
2026-07-23) com `worst_axis`=z — consistente, não é um achado novo isolado. Note que os 5 já
cobrem os dois extremos: 2 casos com `num_circles_detected` no teto da população (622, 196 —
275 círculos, o máximo de toda a população de 994), 1 no piso (924 — 2 círculos, o mínimo), e 2 em
valores intermediários mas ainda fora da faixa central sem falha (790 com 226, 777 com 17).

Nenhuma coincidência com os 5 `img_id` citados na seção "Achado bloqueante novo" (934, 428, 90, 23,
44) — aqueles são os casos que definiram os extremos do **split de treino** usado na calibração
(`argmin/argmax_case_per_axis`), não os piores casos de cobertura em val/test; são conjuntos
diferentes por construção e não há razão a priori para coincidirem.

Reproduzível: `uv run python src/eda/analyze_num_circles_quintile_coverage.py`. Dados:
`output/bounding_box/aorta_anchor_z_diagnosis/{num_circles_quintile_coverage,
worst_z_excess_cases}.csv`. Nenhum artefato de produção alterado (não reprocessa, não regrava
`calibration.json` nem os CSVs de cobertura). Interpretação do padrão em U (o que causa a faixa
central ser "boa") e decisão sobre próximos passos não tomadas aqui, propositalmente.

### Passo 1 (extra): os casos que definiram a calibração são outliers de `num_circles_detected`?

Os 5 `img_id` citados na seção "Achado bloqueante novo" (`argmin/argmax_case_per_axis` do
`calibration_imagecas_splits_train90.json`) contra a distribuição de `num_circles_detected` nos
994 casos (mediana 106, média 107,7, std 32,9):

| img_id | eixo definido | `num_circles_detected` | percentil (994 casos) | faixa (cortes do passo anterior: Q1 ≤ 83, Q5 ≥ 131) |
|---|---|---|---|---|
| 934 | y_min | 162 | 95,9% | quintil 5 |
| 428 | y_max / x_min | 172 | 97,2% | quintil 5 |
| 90 | x_max | 139 | 86,7% | quintil 5 |
| 23 | z_min | 28 | 1,7% | quintil 1 |
| 44 | z_max | 186 | 97,9% | quintil 5 |

**Os 5 caem todos numa faixa extrema (quintil 1 ou 5) — nenhum no meio da distribuição.** Reforça
a leitura do padrão em U: os casos que forçaram os extremos do bbox calibrado (e portanto o
`bbox_extent_reduced` de 240×207×313, ver "Achado bloqueante novo") parecem ser, em boa parte,
outliers de rastreamento Hough+Level Set — não necessariamente pacientes com anatomia coronária
mais extrema de verdade. Não implica causalidade nem propõe correção; é só uma correlação a mais
apontando na mesma direção dos passos anteriores.

### Passo 2 (extra): inspeção visual dos 5 piores casos de excesso em z

Reprocessados só estes 5 casos (`load_and_preprocess_image` + `segment_aorta` — os únicos passos
caros; label e demais campos vêm de `all_cases.csv`/coverage, não recomputados). Script:
`src/eda/inspect_worst_z_excess_cases.py`. Cortes sagital e coronal (fixados no centróide da
máscara), com a máscara da aorta e o label coronário sobrepostos e as linhas de `label_z_min/max` e
do bbox calibrado em z marcadas, salvos em
`output/bounding_box/aorta_anchor_z_diagnosis/img/case_<id>_z_inspection.png`.

| img_id | `num_circles_detected` | hipótese esperada | mask z (min–max) | extensão (fatias) | label z (min–max) | observado |
|---|---|---|---|---|---|---|
| 622 | 275 | over-tracking | 0 – 274 | 274 | 26 – 236 | over-tracking, mas como **leak** (ver abaixo) |
| 196 | 275 | over-tracking | 0 – 274 | 274 | 19 – 218 | over-tracking, mesmo padrão de leak |
| 790 | 226 | checar qual padrão | 39 – 274 | 235 | 29 – 265 | over-tracking / leak, não under-tracking |
| 924 | 2 | under-tracking | 263 – 274 | 11 | 17 – 264 | under-tracking confirmado |
| 777 | 17 | under-tracking | 258 – 274 | 16 | 16 – 201 | under-tracking confirmado |

**Under-tracking (924, 777) confirmado visualmente, e mais severo do que o esperado**: em ambos, a
máscara é um fragmento minúsculo (11–16 fatias) preso bem no **topo do volume** (fatias 258–274, a
borda superior do campo de visão), com uma lacuna enorme até o início do label coronário (924:
label começa em z=17, mask só existe a partir de z=263 — quase 250 fatias de distância). Não
parece "parou um pouco cedo": o rastreamento praticamente não encontrou nada além de um resquício
no topo da imagem.

**Over-tracking (622, 196, 790) confirmado, mas com uma correção qualitativa ao esperado**: nos 3
casos a máscara não é um tubo aórtico simplesmente mais longo que o normal — é um **blob único e
contíguo, muito maior e com formato irregular**, que na sobreposição visual parece incluir bem mais
tecido do que a aorta (em 622/196 o blob cobre quase toda a extensão vertical do corte, com contorno
que acompanha o que parece ser a silhueta do coração; em 790 tem um formato arredondado grande
condizente com uma câmara cardíaca, não uma aorta tubular). Isto é uma leitura visual qualitativa,
não confirmada contra nenhuma referência anatômica formal além do label coronário disponível — mas
sugere que a causa provável não é "o algoritmo seguiu a aorta longe demais no eixo z", e sim um
**vazamento de segmentação (level-set) para uma estrutura vizinha muito maior**, que incidentalmente
também é muito mais alto/baixo em z do que uma aorta ascendente real.

Nenhuma correção proposta nesta etapa, conforme solicitado — só o que foi observado.

Reproduzível: `uv run python src/eda/inspect_worst_z_excess_cases.py`. Dados/imagens:
`output/bounding_box/aorta_anchor_z_diagnosis/worst_z_excess_visual_inspection_summary.csv` e
`output/bounding_box/aorta_anchor_z_diagnosis/img/*.png`. Nenhum artefato de produção alterado.

## Diagnóstico complementar (2026-07-25): quantificação do leak (volume + área por fatia), 10 casos

Continuação quantitativa da inspeção visual anterior. Dois grupos de 5 casos, mesmo método para
os dois: volume total da máscara (voxels) comparado à referência de **IMG_ID=1 (336651 voxels,
validado manualmente como caso "normal" — ver "Execução do lote completo")**, e o perfil de área de
seção transversal por fatia axial ao longo do range z da máscara, para localizar saltos abruptos
(assinatura esperada de leak) vs. área pequena e constante (assinatura esperada de truncamento).
Script: `src/eda/quantify_aorta_mask_leak.py` (reaproveita `plot_case` de
`inspect_worst_z_excess_cases.py` para as imagens sagital/coronal). Reprocessa só estes 10 casos
(`load_and_preprocess_image` + `segment_aorta`); volume total já estava em `all_cases.csv`
(`aorta_voxels`) para todos os 10 sem precisar reprocessar, mas o perfil de área por fatia não é um
campo salvo em produção, daí o reprocessamento.

**Grupo 1 — 5 piores casos de excesso em z** (já inspecionados visualmente antes):

| img_id | volume (voxels) | vs. referência (336651) | mask z (min–max) | maior salto (z, Δ, x mediana) | CV da área |
|---|---|---|---|---|---|
| 622 | 1.569.468 | **4,66x** | 0 – 274 | z=1, Δ=1000 (0,16x) | 0,64 |
| 196 | 1.667.441 | **4,95x** | 0 – 274 | z=1, Δ=647 (0,11x) | 0,59 |
| 790 | 724.192 | **2,15x** | 39 – 274 | z=40, Δ=491 (0,15x) | 0,42 |
| 924 | 1.836 | **0,005x** | 263 – 274 | z=274, Δ=46 (0,28x) | 0,35 |
| 777 | 16.986 | **0,05x** | 258 – 274 | z=274, Δ=130 (0,12x) | 0,13 |

**Grupo 2 — 5 casos que definiram os extremos da calibração original** (2026-07-16):

| img_id | eixo | volume (voxels) | vs. referência (336651) | mask z (min–max) | maior salto (z, Δ, x mediana) | CV da área |
|---|---|---|---|---|---|---|
| 934 | y_min | 681.114 | **2,02x** | 103 – 274 | z=104, Δ=484 (0,10x) | 0,43 |
| 428 | y_max/x_min | 536.221 | **1,59x** | 93 – 274 | z=274, Δ=219 (0,07x) | 0,26 |
| 90 | x_max | 489.841 | **1,46x** | 126 – 274 | z=127, Δ=514 (0,23x) | 0,47 |
| 23 | z_min | 66.303 | **0,20x** | 247 – 274 | z=274, Δ=204 (0,08x) | 0,08 |
| 44 | z_max | 1.024.748 | **3,04x** | 10 – 205 | z=11, Δ=763 (0,12x) | 0,44 |

**Achado 1 — nenhum dos 5 casos calibradores tem volume perto da referência**: variam de 0,20x a
3,04x — a mesma assinatura já vista no grupo 1 (piores casos de cobertura), reforçando que os
casos que definiram o tamanho do bbox calibrado (`bbox_extent_reduced` = 240×207×313, `Rv` ~5%) não
parecem ter uma máscara de aorta "normal" em nenhum dos dois sentidos (nem os 4 grandes demais, nem
o 1 pequeno demais).

**Achado 2 — a métrica de "maior salto" não localiza bem o leak**: em 8 dos 10 casos, o maior salto
de área entre fatias consecutivas ocorre exatamente na **primeira ou última fatia da máscara**
(z=z_min+1 ou z=z_max) — ou seja, é só a transição natural de entrar/sair de área zero, não um
salto *interno* que marcaria onde o vazamento começa. Essa métrica, como definida aqui (maior
diferença absoluta entre fatias consecutivas), não separa bem leak de truncamento; o **coeficiente
de variação (CV) da área ao longo da máscara** é mais informativo: os 3 casos de fragmento truncado
(924, 777, 23) têm CV entre 0,08 e 0,35 (777 e 23 claramente baixos; 924 intermediário, não tão
constante quanto o esperado), enquanto os casos de blob/leak variam mais (CV 0,26–0,64, sem
separação limpa de um único corte).

**Achado 3 — todas as 10 máscaras terminam exatamente na última fatia do volume**: `mask_z_max`
coincide com `volume_shape_z − 1` nos 10 casos (274 para os 9 casos com 275 fatias; 205 para o
caso 44, que tem só 206 fatias) — **inclusive os 3 fragmentos truncados** (924, 777, 23), cuja
pequena máscara também está colada na borda superior do FOV, não solta no meio do volume. Não
investigado a fundo aqui, mas sugere que a fatia mais superior do campo de visão desempenha algum
papel sistemático nesses 10 casos-problema (estrutura anatômica real perto do desfiladeiro
torácico/tronco braquiocefálico, artefato de borda do FOV, ou os dois) — não dá para distinguir
causa com os dados já disponíveis.

**Achado 4 — nem todo caso "grande" tem a forma de blob**: na inspeção visual, 622, 196, 790, 934 e
44 mostram um blob arredondado, sem formato tubular, condizente com vazamento para outra estrutura
(provável câmara cardíaca) — mas **428 e 90 têm formato de tubo/arco curvo plausível para uma
aorta**, só que maior/mais longo que a referência e indo até a borda do FOV. Não dá para descartar
que 428 e 90 sejam rastreamento correto de uma aorta genuinamente mais visível no campo de visão
daquele exame (FOV maior), em vez de vazamento — distinção que a leitura visual sozinha não resolve.

Nenhuma correção proposta. Reproduzível: `uv run python src/eda/quantify_aorta_mask_leak.py`.
Dados/imagens: `output/bounding_box/aorta_anchor_z_diagnosis/mask_leak_quantification_summary.csv`,
`output/bounding_box/aorta_anchor_z_diagnosis/img/case_<id>_area_profile.png` e
`case_<id>_z_inspection.png` (regenerado para os 5 já existentes, novo para os 5 calibradores).
Nenhum artefato de produção alterado.

## Diagnóstico complementar (2026-07-26): "colar na borda do FOV" é raro ou universal?

Dois passos de baixo custo, sem reprocessar o lote completo (994 casos).

### Passo 1 — prevalência de `mask_z_max == volume_shape_z - 1` na população

O diagnóstico anterior (2026-07-25) achou essa igualdade nos 10 casos-problema (5 piores cobertura
+ 5 calibradores) e não sabia dizer se isso discrimina esses casos ou é comum. `mask_z_max` **não é
um campo salvo em `all_cases.csv`** para os 994 casos — só existe para a amostra de 50 casos já
reprocessada no diagnóstico de 2026-07-22 (`candidate_anchor_z.csv`, coluna `z_max_mask`).
Reprocessar os 994 casos completos só para obter esse campo estaria fora do escopo pedido (sem
reprocessar o lote completo); em vez disso, reaproveitamos essa amostra de 50, sem processar nada
de novo. Script: `src/eda/check_z_max_edge_prevalence.py`.

**Resultado: 50 de 50 casos da amostra (100%) têm `mask_z_max == volume_shape_z - 1`,  em todos os
5 quintis de `num_circles_detected` (5/5, 7/7, 13/13, 11/11, 14/14) — igual aos 10/10 (100%) dos
casos-problema.** Ou seja, **"colar na borda do FOV" não é raro nem discrimina os casos-problema —
é universal** (dentro da amostra disponível, 50 de 994 casos, não exaustivo). O achado 3 do
diagnóstico anterior (todas as 10 máscaras terminam na borda) precisa ser lido à luz disso: **não é
uma assinatura dos casos-problema, é o comportamento normal de praticamente todo caso** — a aorta
ascendente detectada por Hough+Level Set aparentemente sempre se estende até a última fatia
disponível do exame, problemático ou não. O que diferencia os casos-problema não é *chegar* na
borda, é o que acontece **antes** de chegar lá (blob grande demais vs. fragmento minúsculo).

Dados: `output/bounding_box/aorta_anchor_z_diagnosis/z_max_edge_prevalence_sample50.csv`.

### Passo 2 — inspeção da imagem HU bruta (sem máscara) nas últimas fatias do FOV

Para os 10 casos-problema + IMG_ID=1 (referência): as 5 últimas fatias do FOV (`z_max` até
`z_max - 4`), imagem HU bruta, sem sobreposição de máscara, sem rodar
`load_and_preprocess_image`/`segment_aorta` (só carrega o NIfTI bruto — mais barato que os
diagnósticos anteriores; o eixo z não é reduzido de resolução, `DOWNSCALE_FACTORS[2] == 1`, então o
índice de fatia bate 1:1 com o resto do diagnóstico). Script:
`src/eda/inspect_fov_top_slices_raw.py`.

**Estatísticas agregadas de HU na última fatia não mostram nada atípico**: médias entre -413 e -262
HU, p99 entre 461 e 991 HU, todos os 11 casos (incluindo a referência) na mesma faixa — nenhum
valor isolado se destaca como anômalo.

**Observação visual qualitativa**: em todos os 11 casos, as últimas fatias mostram o mesmo nível
anatômico — próximo ao desfiladeiro torácico/base do pescoço (clavículas visíveis na borda superior
da imagem, traqueia, grandes vasos do arco/tronco braquiocefálico em corte transversal como
estruturas redondas e claras). **Não parece artefato de reconstrução ou truncamento de FOV** — são
estruturas anatômicas coerentes e nítidas, não ruído ou corte abrupto na borda da imagem. A
diferença qualitativa entre a referência (IMG_ID=1, 1 estrutura vascular redonda claramente
dominante nesse nível) e alguns casos-problema (622, 924, 44 — 2 a 4 estruturas redondas de
brilho/tamanho parecido, próximas entre si) é sugestiva de uma região com múltiplos vasos de
calibre semelhante (candidatos ambíguos para o Hough), mas **não é conclusiva** — amostra pequena
(11 casos), sem comparação sistemática contra casos "bons" além da única referência, e a
classificação de "quantos vasos redondos aparecem" foi feita visualmente, não com uma métrica
objetiva.

Nenhuma correção proposta — só o observado. Reproduzível:
`uv run python src/eda/inspect_fov_top_slices_raw.py`. Dados/imagens:
`output/bounding_box/aorta_anchor_z_diagnosis/fov_top_slices_raw_summary.csv` e
`output/bounding_box/aorta_anchor_z_diagnosis/img/case_<id>_fov_top_raw.png` (11 casos, incluindo a
referência `case_1_fov_top_raw.png`). Nenhum artefato de produção alterado.

## Diagnóstico complementar (2026-07-27): tentativa de objetivar "múltiplos vasos ambíguos" — resultado negativo

Tentativa de transformar a observação qualitativa de 2026-07-26 (múltiplas estruturas redondas de
calibre parecido perto do topo do FOV, em casos-problema vs. referência) numa métrica objetiva, em
amostra maior (40 casos: os 10 já conhecidos + 30 de comparação com `aorta_voxels` entre 0,7x-1,3x
da referência, 336651 voxels, amostra estratificada por bloco, sem reprocessar nada para a
seleção). Método: **só a detecção de círculos** (`detect_aorta_circles`/
`_detect_circles_in_slice` de `aorta_localization.py`), sem Level-Set, aplicada nas 15 fatias mais
próximas do topo do FOV — muito mais barato que os diagnósticos anteriores (~5-10s/caso, sem
Level-Set, vs. 40-220s/caso). Script: `src/eda/measure_ambiguous_circle_candidates.py`.

**Métrica**: em cada fatia analisada, roda `detect_aorta_circles` real (mesma config de produção,
janela de 15 fatias) para obter "o círculo principal" rastreado; separadamente, roda
`_detect_circles_in_slice` (sem restrição de ROI, `total_num_peaks_initial`=15 picos) na própria
fatia e nas duas adjacentes para obter todos os candidatos Hough. Um candidato conta como
"ambíguo" se não é o próprio círculo principal, tem raio dentro de ±20% do raio principal, e centro
a menos de 2x o raio médio de distância.

**Resultado — a métrica, como especificada, NÃO confirma a diferença vista visualmente**:

| grupo | n casos | média | mediana | desvio-padrão | mín | máx |
|---|---|---|---|---|---|---|
| problema (10) | 10 | 290,9 | 295,0 | 121,8 | 1 | 419 |
| normal (30) | 30 | 332,4 | 346,0 | 85,8 | 12 | 513 |

As distribuições **se sobrepõem quase inteiramente** (problema: 238–419 excluindo 1 outlier;
normal: 191–513 excluindo 1 outlier) — a mediana do grupo "normal" é até um pouco **maior** que a do
grupo "problema", o oposto do que a hipótese visual sugeria.

**Causa provável — efeito-teto nos parâmetros da métrica, não ausência de sinal real**: com
`total_num_peaks_initial`=15 picos por fatia e faixa de raio de Hough de só 18–30px (12px de
amplitude total, config de produção), uma tolerância de raio de ±20% sobre um raio típico de
~20–25px (±4–5px) já cobre boa parte de toda a faixa de busca permitida — e um limiar de distância
de 2x o raio médio (~40–50px) é uma fração grande de uma fatia de 256×256. Nessas condições, a
maioria dos picos Hough detectados em qualquer fatia (de qualquer caso, problemático ou não)
trivialmente satisfaz os dois critérios simultaneamente — o método super-conta candidatos
"ambíguos" quase universalmente, mascarando qualquer diferença real entre os grupos.

**Duas exceções informativas, mas por razões diferentes**: `img_id=924` (grupo problema, só 1
ambíguo) tem apenas **2 círculos reais rastreados** de 15 fatias (rastreamento falhou cedo — poucos
candidatos principais para comparar, não indica ausência de ambiguidade na imagem). `img_id=190`
(grupo normal, 12 ambíguos) rastreou as 15 fatias **normalmente** e ainda assim teve poucos
candidatos ambíguos — esse sim é um caso genuinamente de baixa ambiguidade dentro da amostra,
mas é o único.

**Conclusão**: com os parâmetros exatamente como especificados no pedido, a métrica objetiva **não
confirma** a diferença observada visualmente em 2026-07-26 — mas a razão mais provável é uma
limitação do desenho da métrica (tolerâncias largas demais relativas à faixa de busca do Hough
configurada em produção), não uma refutação da hipótese original. Recalibrar os limiares (ex.:
tolerância de raio menor, ou normalizar pela faixa de busca em vez de por porcentagem fixa) poderia
mudar o resultado, mas isso seria uma escolha de método nova, não coberta por este diagnóstico.
Nenhuma correção proposta.

Reproduzível: `uv run python src/eda/measure_ambiguous_circle_candidates.py`. Dados:
`output/bounding_box/aorta_anchor_z_diagnosis/ambiguous_circle_candidates_summary.csv`. Nenhum
artefato de produção alterado.

## Não incluído nesta etapa / decisões em aberto

- Etapas 7 (posicionamento do bbox) e 8 (corte em resolução integral) ainda não implementadas.
  **A fórmula `bbox_min/max = âncora + offset_min/max_full_resolution` (por eixo) valia para um
  bbox de tamanho FIXO — não vale mais** desde a decisão de 2026-07-21 de usar um envelope
  individual por imagem (ver seção "Decisão" acima). Não implementar etapas 7-8 até o critério de
  tamanho do envelope individual ser decidido.
