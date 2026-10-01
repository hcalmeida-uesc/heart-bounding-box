# Bbox dinâmico — Passo 0c: bbox fixo por âncora

**Data:** 2026-09-30. **Natureza:** exploratória. Nenhum dataset foi gerado, nenhum artefato de
produção foi alterado e nenhuma segmentação foi reprocessada. Este passo continua o
[Passo 0](dynamic_bbox_step0.md) e o [Passo 0b](dynamic_bbox_step0b.md). No 0b a caixa tinha o
tamanho exato do label (oráculo); aqui ela tem **tamanho único para todos os casos**, como no
método de produção, e só a âncora que a posiciona muda. **Não se recomenda trocar a âncora de
produção.**

## 1. Método

### 1.1 Âncoras (resolução integral, eixos y, x, z)

| Âncora | Definição |
|---|---|
| **A1** (produção) | Centróide da máscara da aorta = âncora de `all_cases.csv` × (2, 2, 1) |
| **A2** | Centro do volume = shape / 2 |
| **A6** (nova) | y e z de A2, x de A1 (o melhor eixo de cada uma, seção 3.1 do Passo 0b) |

### 1.2 Caixa fixa por âncora

1. Offsets do label relativos à âncora, por caso e por eixo: `min_label − âncora` e
   `max_label + 1 − âncora`.
2. A caixa vai de âncora + P₁₀₀₋ₚ(offset inferior) até âncora + Pₚ(offset superior), por eixo.
   O tamanho é único para todos os casos: Pₚ(sup.) − P₁₀₀₋ₚ(inf.).
   - **A centralização em âncora + viés mediano (pedida, "como no 0b") não altera a caixa.** Com
     percentis tomados por lado, o viés é uma constante por eixo que soma nos dois percentis e
     se cancela.
3. `p` é o mesmo nos três eixos. Foi ajustado por bissecção em (50, 100] até a perda média no
   conjunto avaliado igualar a da Opção 1 de produção no mesmo conjunto. Foi escolhido o menor
   `p` com perda média ≤ alvo. Como a perda varia em degraus (limites inteiros), o valor obtido
   pode ficar um pouco abaixo do alvo.
4. As caixas são clampadas ao volume. `Rv` = 1 − V_caixa_efetiva / V_volume. Perda = fração dos
   voxels do label fora da caixa, em resolução integral.
5. **Opção 1 de produção:** os offsets de `calibration_both_options.json` (calibrados no treino,
   com o Critério C), posicionados por A1 via `compute_full_resolution_bbox`. São os mesmos
   índices do corte salvo em `cropped_option1`. As caixas de A1, A2 e A6 são calibradas **dentro
   do conjunto avaliado**.

### 1.3 Conjuntos de avaliação

| Conjunto | Definição | N |
|---|---|---|
| **1** | Os 624 casos do Passo 0 (interseção Opção 1 ∩ Opção 2, menos teste). Favorece A1: a aorta foi bem segmentada em todos | 624 |
| **2** | Todos os casos fora do teste (`split_original.csv`, `split=trainval`), **sem Critério C**, com aorta detectada. Usado por A1, A6, Opção 1 e A2 | **786** |
| **2b** | Todos os casos fora do teste, inclusive os 6 sem aorta detectada (86, 268, 606, 657, 822, 834). Só A2 | **792** |

- Dos 786 casos do conjunto 2, **62** ficam fora da população do Critério C da Opção 1 (tabela
  4.3).
- No conjunto 2b a Opção 1 não é definida para os 6 casos sem aorta. O alvo de perda do 2b é o
  mesmo do conjunto 2.
- **Extensão da extração:** o cache de coordenadas do Passo 0 cobria só os 624 casos. Os 168
  restantes fora do teste foram extraídos com `p0c_extract.py`, que só lê os labels NIfTI, sem
  segmentação.

## 2. Resultados — conjunto 1 (624 casos)

Alvo de perda média = perda da Opção 1 nos 624 = **0,1559%**.

| Âncora | `p` | Caixa (vox) | Caixa (mm*) | `Rv` média | mediana | p5 | p95 |
|---|---|---|---|---|---|---|---|
| **Opção 1 (produção)** | — | 408,5 × 357,2 × 243,1 | 143,6 × 125,6 × 121,5 | **49,18%** | 50,32% | 44,01% | 53,49% |
| A1 aorta | **97,18** | 411,1 × 357,6 × 240,6 | 144,5 × 125,7 × 120,3 | 49,15% | 50,40% | 43,58% | 53,73% |
| A2 centro do volume | **98,72** | 395,0 × 411,0 × 246,0 | 138,9 × 144,5 × 123,0 | 41,52% | 43,59% | 37,45% | 43,59% |
| A6 mista | **97,59** | 389,0 × 360,8 × 237,0 | 136,8 × 126,8 × 118,5 | **51,37%** | 53,39% | 46,14% | 55,04% |

| Âncora | Perda média | mediana | p95 | máx. | N > 0 | N > 1% | N > 5% | N > 20% |
|---|---|---|---|---|---|---|---|---|
| Opção 1 (produção) | 0,156% | 0% | 0,894% | 8,36% | 84 | 30 | 4 | 0 |
| A1 aorta | 0,156% | 0% | 0,808% | 8,36% | 83 | 28 | 5 | 0 |
| A2 centro do volume | 0,144% | 0% | 0,261% | 18,66% | 45 | 21 | 4 | 0 |
| A6 mista | 0,155% | 0% | 0,695% | 11,98% | 74 | 26 | 6 | 0 |

\* Caixa em mm = voxels × spacing mediano do conjunto (0,352 / 0,352 / 0,5 mm). O spacing varia
por caso.

**Maiores perdas por caso** (id: perda, `num_circles_detected`):

- **Opção 1:** 273: 8,4% (146) · 943: 7,9% (111) · 989: 6,4% (113) · 80: 5,7% (165) · 807: 4,4% (115)
- **A1:** 273: 8,4% · 943: 7,9% · 80: 6,5% · 989: 6,1% · 807: 5,3%
- **A2:** 355: 18,7% (79) · 608: 11,3% (82) · 2: 7,4% (82) · 568: 5,6% (92) · 406: 5,0% (115)
- **A6:** 2: 12,0% (82) · 595: 7,1% (71) · 406: 7,1% (115) · 352: 6,7% (167) · 530: 5,7% (72)

## 3. Resultados — conjunto 2 (786 casos fora do teste, sem Critério C)

Alvo de perda média = perda da Opção 1 nos 786 = **2,818%**.

| Âncora | `p` | Caixa (vox) | Caixa (mm*) | `Rv` média | mediana | p5 | p95 |
|---|---|---|---|---|---|---|---|
| **Opção 1 (produção)** | — | 408,5 × 357,2 × 243,1 | 142,8 × 124,9 × 121,5 | **50,08%** | 50,32% | 44,01% | 58,89% |
| A1 aorta | **92,93** | 407,6 × 354,8 × 242,5 | 142,5 × 124,0 × 121,3 | 50,50% | 50,86% | 44,59% | 59,51% |
| A2 centro do volume | **84,84** | 356,0 × 331,0 × 212,0 | 124,5 × 115,7 × 106,0 | **62,42%** | 65,04% | 54,65% | 65,04% |
| A6 mista | **84,97** | 357,0 × 327,0 × 212,0 | 124,8 × 114,3 × 106,0 | **62,92%** | 65,40% | 55,33% | 65,40% |
| A2, conjunto 2b (792) | **84,83** | 356,0 × 331,0 × 211,0 | 124,5 × 115,7 × 105,5 | 62,41% | 65,04% | 54,65% | 65,04% |

| Âncora | Perda média | mediana | p95 | máx. | N > 0 | N > 1% | N > 5% | N > 20% |
|---|---|---|---|---|---|---|---|---|
| Opção 1 (produção) | 2,818% | 0% | 18,99% | 69,46% | 229 | 166 | 91 | **38** |
| A1 aorta | 2,818% | 0% | 18,94% | 70,44% | 237 | 173 | 91 | 38 |
| A2 centro do volume | 2,749% | **0,495%** | 11,22% | 42,01% | **470** | **345** | 153 | **12** |
| A6 mista | 2,806% | 0,367% | 12,26% | 55,51% | 459 | 341 | 152 | 11 |
| A2, conjunto 2b (792) | 2,754% | 0,495% | 11,19% | 42,01% | 473 | 348 | 155 | 12 |

**Maiores perdas por caso** (id: perda, `num_circles_detected`, dentro/fora da população do
Critério C):

- **Opção 1:** 738: 69,5% (11, fora) · 823: 65,3% (261, fora) · 790: 63,7% (226, fora) · 132: 61,7% (47, dentro) · 622: 61,1% (275, fora)
- **A1:** 738: 70,4% · 823: 64,4% · 790: 62,9% · 132: 62,3% · 622: 59,2%
- **A2:** 595: 42,0% (71, dentro) · 355: 33,9% (79, dentro) · 756: 31,7% (66, dentro) · 310: 26,4% (54, dentro) · 380: 24,9% (77, dentro)
- **A6:** 132: 55,5% (47, dentro) · 738: 44,8% (11, fora) · 595: 41,8% (71, dentro) · 306: 27,2% (9, fora) · 321: 25,3% (116, dentro)

**Os 6 casos sem aorta detectada (conjunto 2b, só A2):** perda 86: 10,32% · 268: 5,92% ·
822: 2,97% · 606, 657, 834: 0%. `Rv` entre 54,7% e 65,0%. A Opção 1 e a A1 não processam esses
casos.

### 3.1 Conjunto 2 separado pela população do Critério C

| Âncora | Grupo | N | `Rv` média | Perda média | Perda máx. | N perda > 0 |
|---|---|---|---|---|---|---|
| Opção 1 | dentro do Critério C | 724 | 49,53% | 1,479% | 61,67% | 184 |
| Opção 1 | **fora** do Critério C | 62 | 56,57% | **18,46%** | 69,46% | 45 |
| A1 aorta | dentro | 724 | 49,94% | 1,475% | 62,28% | 192 |
| A1 aorta | **fora** | 62 | 57,08% | **18,51%** | 70,44% | 45 |
| A2 centro do volume | dentro | 724 | 62,44% | 2,752% | 42,01% | 434 |
| A2 centro do volume | **fora** | 62 | 62,21% | **2,71%** | 22,57% | 36 |
| A6 mista | dentro | 724 | 62,92% | 2,632% | 55,51% | 422 |
| A6 mista | **fora** | 62 | 62,87% | 4,84% | 44,76% | 37 |

## 4. Observações de medição

- **A1 recalibrada reproduz a Opção 1 no conjunto 1.** O `Rv` fica em 49,15% contra 49,18%, com a
  mesma perda média, e o `p` calibrado é 97,18. A caixa de produção já fica muito perto do
  melhor tamanho fixo para a A1 com esse orçamento de perda. Isso serve também como checagem de
  consistência do método.
- **O `Rv` da A2 não varia de caso a caso em y/x.** Todos os volumes têm 512 × 512 em y/x e a caixa
  é centrada no volume. O `Rv` só muda com o tamanho em z (206 a 275 fatias). Por isso a mediana e
  o p95 coincidem.
- **A perda da A2 no conjunto 1 (0,144%) ficou abaixo do alvo** (0,156%) por causa do degrau da
  bissecção. Não há `p` que produza exatamente o alvo.

## 5. Interpretação (sem recomendar trocar a âncora de produção)

- **A ordem das âncoras depende do conjunto e do orçamento de perda.**
  - No conjunto 1, curado e com orçamento baixo (0,156%): A6 > Opção 1 ≈ A1 > A2. A6 ganha
    +2,2 pp de `Rv`; A2 perde −7,7 pp.
  - No conjunto 2, sem o Critério C e com orçamento de 2,8%: A6 ≈ A2 (+12,8 e +12,3 pp) ≫ A1 ≈
    Opção 1.
- **Por que a A2 perde no conjunto 1.** Com orçamento baixo, `p` sobe para 98,7 e a caixa passa a
  ser ditada pela cauda do erro em x. No Passo 0b, x é o eixo em que o centro do volume erra mais
  (p95 de 60,9 vox contra 45,5 da aorta). A caixa da A2 fica com 411 vox em x, contra 357 da
  Opção 1. A A6 usa x da aorta e evita essa cauda; nos outros eixos, onde a A2 é melhor, ela
  encolhe a caixa (y 389 e z 237, contra 408,5 e 243,1).
- **No conjunto 2, o orçamento de perda é inflado pelas falhas da própria A1.**
  - Os 62 casos fora do Critério C têm perda média de 18,5% com a Opção 1 e a A1. Com a A2, a
    média é de 2,7%, igual à dos casos dentro do critério.
  - Como a perda média é igualada, A2 e A6 gastam esse orçamento de outro jeito: perdas pequenas
    e espalhadas (mediana 0,5% contra 0%; 345 casos com perda > 1% contra 166). Em troca, têm
    menos perdas severas (12 casos > 20% contra 38; máximo de 42% contra 69%).
  - A caixa resultante é bem menor (356 × 331 × 212 contra 408,5 × 357,2 × 243,1).
  - **Esses perfis de perda não são equivalentes:** muitos casos com perda pequena contra poucos
    casos com perda grande. Qual é aceitável é uma escolha metodológica, que também interage com
    o treino do Capítulo 3 sobre labels recortados. Este passo não decide isso.
- **A A6 herda da A1 a sensibilidade em x à falha de segmentação da aorta.** Fora do Critério C,
  a perda média da A6 é de 4,8%, contra 2,7% da A2. Casos 738 (11 círculos) e 306 (9 círculos).
- **A A2 não depende da aorta.** Ela cobre também os 6 casos sem aorta detectada (perda de 0% a
  10,3%), que a Opção 1 não processa.
- **Seleção dos 624:** como antecipado no pedido, o conjunto 1 favorece a A1. Os 624 já excluem as
  falhas de segmentação da aorta, que são justamente o que pesa contra a A1 no conjunto 2.

## 6. Reprodutibilidade

- Resultados em `output/bounding_box/dynamic_bbox_step0c/`:
  - `p0c_fixed_box_summary.csv`: tabelas das seções 2 e 3, com as 5 maiores perdas.
  - `p0c_percase.csv`: `Rv` e perda por caso, âncora e conjunto, com `num_circles_detected`,
    `aorta_voxels` e pertencimento à população do Critério C.
  - `p0c_extra_cases.csv`: shape, spacing e voxels do label dos 168 casos extraídos agora.
- Scripts em `output/bounding_box/dynamic_bbox_step0c/scripts/`:
  - `p0c_extract.py`: estende o cache `p0_cache/`; só lê labels.
  - `p0c_analyze.py`: calibra `p` e avalia as caixas.
  - Os dois dependem do cache de coordenadas do Passo 0.
- A tabela por faixa de perda (N > 1%, > 5%, > 20%) e a separação pelo Critério C foram
  derivadas de `p0c_percase.csv`.
