# Bbox dinâmico — Passo 0b: comparação de âncoras de posição

**Data:** 2026-09-30. **Natureza:** exploratória. Nenhum dataset foi gerado, nenhum artefato de
produção foi alterado e nenhuma segmentação foi reprocessada. Este passo continua o
[Passo 0](dynamic_bbox_step0.md), que atribuiu cerca de 60% do espaço de ganho de `Rv` à posição
(centro da árvore) e cerca de 7% ao tamanho. Aqui se mede qual âncora **sem rótulo** erra menos o
centro da árvore coronária. Este passo **não** avalia a viabilidade do bbox dinâmico e **não**
recomenda trocar a âncora de produção.

## 1. Casos, insumos e âncoras

- **Casos:** os mesmos 624 do Passo 0 (interseção Opção 1 ∩ Opção 2, menos os 208 de teste).
- **Insumos, só leitura:**
  - `output/bounding_box/dynamic_bbox_step0/p0_cases.csv` (shape, spacing do cabeçalho NIfTI,
    âncora) e `p0_percase.csv` (referências do Passo 0).
  - `output/bounding_box/aorta_anchor/all_cases.csv` (`num_circles_detected`).
  - O cache de coordenadas do label do Passo 0 (`p0_cache/`, regenerável por
    `dynamic_bbox_step0/scripts/p0_extract.py`), para o centro exato da árvore em resolução
    integral, que não está nos CSVs.
- **Convenções**, as mesmas do Passo 0:
  - Resolução integral, eixos `(y, x, z)`.
  - Centro do label = centro da caixa exata do label.
  - Caixas clampadas ao volume.
  - `Rv` = 1 − V_caixa / V_volume.

| Âncora | Definição (posição em voxels de resolução integral) |
|---|---|
| **A1** (referência) | Centróide da máscara da aorta = âncora de `all_cases.csv` × (2, 2, 1) |
| **A2** | Centro geométrico do volume = shape / 2 |
| **A3** | Mediana populacional de (centro do label / shape) por eixo, × shape do caso. Frações: y 0,5010, x 0,5635, z 0,5000 |
| **A4** | y e x de A1; z de A3 |
| **A5** | y e x de A1; z = z mínimo da máscara da aorta (proxy da raiz aórtica) |

**A5 não pôde ser avaliada nos 624 casos.** O z mínimo da máscara da aorta não é salvo em
`all_cases.csv`. Ele só existe em `aorta_anchor_z_diagnosis/candidate_anchor_z.csv`, uma amostra
estratificada por bloco (10 por bloco, seed = 42) do diagnóstico de 2026-07-22, e **30 desses 50
casos estão entre os 624**. Sem reprocessar nada, A5 é comparada às demais âncoras só nesses 30
casos (seção 4.4), como indicação limitada.

**Equivalência A2 ≡ A3 neste dataset.** Em todos os 624 casos o volume tem 512 × 512 em y/x, e a
fração mediana em z é exatamente 0,5. Depois da correção de viés, A3 vira matematicamente A2.
Por isso as duas linhas coincidem nas tabelas; só o viés bruto difere.

## 2. Métrica

- Erro de centro = centro_label − âncora, por eixo, em voxels e em mm (× spacing do caso).
- Correção de viés: subtrai-se a mediana populacional do erro. Isso é feito separadamente em
  voxels e em mm.
- |erro residual|: mediana, p95 e máximo.
- **Caixa (variante 5 do Passo 0 com a âncora trocada):**
  - centro = âncora + viés mediano;
  - tamanho = extensão exata do label do caso;
  - margem por eixo, dos dois lados = p95 (principal) ou p100 (secundária) do |erro residual|.
- Frações do espaço de ganho = (`Rv` − `Rv`_piso) / (`Rv`_teto − `Rv`_piso), com piso = variante 1
  e teto = variante 4 do Passo 0, calculados no mesmo conjunto de casos.
- Uma âncora perfeita (erro zero) reproduz o oráculo completo do Passo 0: `Rv` 78,55%, 97,9% do
  espaço de ganho.

## 3. Resultados — 624 casos

### 3.1 Erro de centro residual (após correção de viés)

| Âncora | Eixo | Viés (vox / mm) | Mediana \|res.\| vox | p95 vox | Máx. vox | Mediana mm | p95 mm | Máx. mm |
|---|---|---|---|---|---|---|---|---|
| A1 aorta | y | +74,1 / +26,1 | 16,2 | 45,8 | 81,1 | 5,75 | 17,55 | 30,52 |
| A1 aorta | x | +9,4 / +3,4 | 15,1 | 45,5 | 73,9 | 5,34 | 15,62 | 24,61 |
| A1 aorta | z | −68,1 / −34,1 | 11,3 | 30,1 | 60,3 | 5,63 | 15,05 | 30,13 |
| A2 centro do volume | y | +0,5 / +0,2 | 11,5 | **32,0** | 57,0 | 3,97 | **11,86** | 20,00 |
| A2 centro do volume | x | +32,5 / +11,4 | 20,0 | 60,9 | 99,0 | 6,97 | 21,22 | 34,09 |
| A2 centro do volume | z | 0,0 / 0,0 | 9,8 | **29,5** | 53,5 | 4,88 | **14,75** | 26,75 |
| A3 fração fixa | y | 0 / 0 | 11,5 | 32,0 | 57,0 | 3,98 | 11,86 | 19,97 |
| A3 fração fixa | x | 0 / 0 | 20,0 | 60,9 | 99,0 | 6,95 | 20,85 | 33,84 |
| A3 fração fixa | z | 0 / 0 | 9,8 | 29,5 | 53,5 | 4,88 | 14,75 | 26,75 |
| A4 mista | y, x | igual a A1 | | | | | | |
| A4 mista | z | igual a A3 | | | | | | |

Menor p95 por eixo: y → A2/A3 (32,0 vox), x → **A1** (45,5 vox), z → A2/A3 (29,5 vox; A1 fica
em 30,1).

### 3.2 `Rv` e perda com a caixa da variante 5 (tamanho exato + margem do erro residual)

| Âncora | Pct. | `Rv` média | `Rv` mediana | Perda média | Perda máx. | N perda > 0 | Margem y/x/z (vox) | Fração do espaço de ganho |
|---|---|---|---|---|---|---|---|---|
| A1 aorta | p95 | **51,40%** | 51,81% | 0,561% | 95,17% | 68 | 45,8 / 45,5 / 30,1 | **7,4%** |
| A2 = A3 | p95 | 50,27% | 50,56% | 0,499% | 55,67% | 75 | 32,0 / 60,9 / 29,5 | 3,6% |
| A4 mista | p95 | 50,93% | 51,28% | 0,546% | 77,47% | 76 | 45,8 / 45,5 / 29,5 | 5,8% |
| A1 aorta | p100 | 28,57% | 28,58% | 0% | 0% | 0 | 81,1 / 73,9 / 60,3 | −68,7% |
| A2 = A3 | p100 | 27,41% | 26,89% | 0% | 0% | 0 | 57,0 / 99,0 / 53,5 | −72,6% |
| A4 mista | p100 | 28,29% | 28,46% | 0% | 0% | 0 | 81,1 / 73,9 / 53,5 | −69,7% |
| *Ref.: piso (variante 1, Opção 1)* | — | *49,18%* | *50,32%* | *0,156%* | *8,36%* | *84* | — | *0%* |
| *Ref.: oráculo de posição (variante 6)* | *p95* | *67,23%* | *69,21%* | *0,251%* | *13,00%* | *80* | — | *60,2%* |
| *Ref.: oráculo de posição (variante 6)* | *p100* | *55,20%* | *57,10%* | *0%* | *0%* | *0* | — | *20,1%* |
| *Ref.: teto (variante 4)* | — | *79,17%* | — | *0,156%* | — | — | — | *100%* |

Maiores perdas em p95 (caso: perda, `num_circles_detected`):
- **A1:** 809: 95,2% (64) · 80: 27,9% (165) · 134: 23,8% (98) · 427: 22,1% (123)
- **A2:** 809: 55,7% · 595: 43,5% (71) · 427: 33,9% · 276: 21,0% (79)
- **A4:** 809: 77,5% · 595: 43,5% · 427: 33,9% · 80: 16,1%

## 4. Checagem de robustez: sem os extremos de `num_circles_detected`

### 4.1 Critério A literal: nenhum caso excluído

Com o critério A a 5% (`select_cases_by_circle_count_percentile`, 2,5% por cauda, limiares
globais sobre os 994 casos `ok`: 45,0 e 178,05), **nenhum dos 624 casos é excluído**. Os 624 já
passaram pelos filtros das Opções 1 e 2, então a checagem literal não muda nada.

### 4.2 Variante informativa: 5% extremos dentro dos 624

Mesma regra (5% total, 2,5% por cauda), aplicada à distribuição dos 624: limiares **66 e 155**,
**28 casos excluídos** (80, 156, 164, 177, 224, 242, 289, 310, 314, 352, 371, 433, 458, 469, 565,
575, 623, 653, 665, 678, 696, 697, 746, 809, 851, 870, 908, 973), **596 restantes**. Viés,
percentis e referências foram recalculados sobre os 596.

| Âncora | Eixo | p95 \|res.\| vox (624 → 596) | Máx. vox (624 → 596) |
|---|---|---|---|
| A1 aorta | y | 45,8 → 45,7 | 81,1 → 71,4 |
| A1 aorta | x | 45,5 → 45,1 | 73,9 → 63,6 |
| A1 aorta | z | 30,1 → 30,2 | 60,3 → 50,6 |
| A2 = A3 | y | 32,0 → 32,0 | 57,0 → 57,0 |
| A2 = A3 | x | 60,9 → 61,0 | 99,0 → 99,5 |
| A2 = A3 | z | 29,5 → 28,1 | 53,5 → 53,5 |

| Âncora (596 casos) | `Rv` p95 | Perda média p95 | Fração do espaço de ganho p95 | `Rv` p100 | Fração p100 |
|---|---|---|---|---|---|
| A1 aorta | 51,52% | 0,387% | 7,9% | **35,12%** | −46,7% |
| A2 = A3 | 50,62% | 0,412% | 4,9% | 27,38% | −72,5% |
| A4 mista | 51,42% | 0,415% | 7,5% | 33,83% | −51,0% |
| *Ref.: piso (variante 1)* | *49,15%* | *0,153%* | *0%* | — | — |
| *Ref.: oráculo de posição* | *67,01%* | *0,235%* | *59,4%* | *55,20%* | *20,1%* |

Perda máxima da A1 em p95 nos 596: 23,8% (caso 134).

### 4.3 Resumo da checagem

- O p95 do erro residual da A1 praticamente não muda sem os extremos (variação ≤ 0,4 vox).
- O que muda é o máximo, que cai 10 vox em cada eixo.
- Por isso a caixa em p95 quase não muda (51,40 → 51,52%), e a caixa em p100, que depende do
  máximo, sobe de 28,57 para 35,12%.

### 4.4 A5 (z mínimo da aorta): amostra de 30 casos

**Amostra pequena.** O p95 de 30 valores fica entre as duas maiores observações. Estes números
indicam uma direção, mas não são comparáveis aos das seções 3 e 4.2. Nenhum desses 30 casos cai
nos 5% extremos de `num_circles_detected` (faixa na amostra: 62 a 149).

| Âncora (30 casos) | Viés z (vox) | Mediana \|res. z\| vox | p95 z | Máx. z | `Rv` p95 | Perda média p95 | `Rv` p100 |
|---|---|---|---|---|---|---|---|
| A1 aorta (centróide) | −67,7 | 16,7 | 27,2 | 29,5 | 55,85% | 0,163% | 50,72% |
| **A5 z mínimo da aorta** | −8,8 | 12,8 | 27,1 | 29,8 | 56,13% | 0,163% | 50,92% |
| A3 fração fixa (z) | 0 | 10,0 | 19,8 | 25,0 | — | — | — |
| A2 = A3 (y, x, z) | — | — | — | — | 58,39% | 0,243% | 49,55% |
| A4 mista | — | — | — | — | 57,51% | 0,167% | 51,48% |
| *Ref.: piso (variante 1)* | | | | | *48,73%* | *0,039%* | |
| *Ref.: oráculo de posição* | | | | | *67,23%* | *0,200%* | *62,81%* |

Em y/x, A5 é idêntica a A1. Na amostra, o p95 de y/x é 37,9 / 38,3 vox para A1 e 24,2 / 47,4 vox
para A2.

## 5. Interpretação (sem concluir sobre viabilidade nem recomendar trocar a âncora)

- **Nenhuma âncora sem rótulo se aproxima do oráculo de posição.** Em p95, as âncoras recuperam
  3,6% a 7,4% do espaço de ganho, contra 60,2% do oráculo de posição. Em p100, todas ficam
  abaixo do piso (bbox fixo da Opção 1). A margem que cada âncora precisa para cobrir o próprio
  erro residual (cerca de 30 a 61 vox por lado em p95, 54 a 99 em p100) consome o que se ganha
  ao conhecer o tamanho exato.
- **A âncora da aorta (A1) quase não bate a âncora trivial.** Depois da correção de viés, o centro
  do volume (A2) erra **menos** que A1 em y (p95 32,0 contra 45,8 vox; 11,9 contra 17,6 mm) e
  praticamente o mesmo em z (29,5 contra 30,1 vox). A1 só é melhor em x (45,5 contra 60,9 vox).
  Na caixa, isso resulta em `Rv` de 51,40% (A1) contra 50,27% (A2), perto do piso de 49,18%.
- **O erro residual típico é de cerca de 4 a 7 mm por eixo (mediana) para todas as âncoras.** O
  que as separa é a cauda (p95 de 12 a 21 mm; máximos de 20 a 34 mm), e é a cauda que define a
  margem.
- **O erro da A1 não vem só de falhas de rastreamento.** Tirar os casos com
  `num_circles_detected` extremo não altera o p95 da A1. Só reduz os máximos, que dependem de
  casos como 809 (under-tracking, 64 círculos) e 80. O grosso do erro residual da A1 aparece
  também em casos com segmentação da aorta "normal" por esse critério.
- **A combinação por eixo muda a ordem.** A âncora com menor p95 não é a mesma em todos os eixos:
  y e z favorecem o centro do volume, x favorece a aorta. Nenhuma das âncoras pedidas combina
  essas escolhas. A4 (y, x da aorta + z fixo) só troca o eixo em que as duas já empatavam.
- **A5 (raiz aórtica pelo z mínimo)**, na amostra de 30: o z mínimo tem viés bruto muito menor que
  o centróide (−8,8 contra −67,7 vox), ou seja, fica perto do z do centro da árvore. Depois da
  correção de viés, porém, a dispersão é a mesma do centróide (p95 27,1 contra 27,2 vox) e maior
  que a da fração fixa (19,8 vox). Isso vale para esta amostra; não é conclusão sobre os 624.
- **Ressalva de comparabilidade (a mesma do Passo 0):** as caixas em p95 não têm a perda média
  igualada entre si nem à do piso (0,50 a 0,56% contra 0,156% do piso e 0,251% do oráculo de
  posição). As frações em p95 são, portanto, otimistas em relação a uma comparação com a mesma
  perda, e as de p100 são pessimistas.

## 6. Reprodutibilidade

- Resultados em `output/bounding_box/dynamic_bbox_step0b/`:
  - `p0b_anchor_errors.csv`: erro residual por âncora, eixo e conjunto de casos.
  - `p0b_anchor_boxes.csv`: `Rv`, perda, margens e fração do espaço de ganho por âncora,
    percentil e conjunto.
  - `p0b_percase.csv`: resíduo por eixo e perda por caso nos 624.
  - `p0b_meta.json`: frações da A3, limiares do critério A, IDs excluídos, IDs da amostra A5 e
    referências.
- Script: `output/bounding_box/dynamic_bbox_step0b/scripts/p0b_analyze.py`. Depende do cache
  `p0_cache/` do Passo 0, regenerável por `dynamic_bbox_step0/scripts/p0_extract.py`.
