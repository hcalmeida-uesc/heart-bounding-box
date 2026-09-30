# Bbox dinâmico — Passo 0: teto, piso e decomposição

**Data:** 2026-09-29. **Natureza:** exploratória. Nenhum dataset foi gerado e nenhum artefato de
produção foi alterado (`D:\Doutorado\dataset\cropped_*`, manifestos, `all_cases.csv` e calibração
ficaram intocados). **Referência principal:** Opção 1 (perda parcial tolerada), que teve o melhor
resultado no Capítulo 3.

> **Os oráculos (variantes 3 a 6) são referências teóricas, não métodos.** Todos usam o label
> coronário verdadeiro do próprio caso para definir tamanho e/ou posição da caixa, informação que
> não existe numa imagem nova. Servem para medir o espaço de ganho e de onde ele vem, não como
> candidatos a pipeline.

## 1. Casos

| Conjunto | N |
|---|---|
| Interseção real Opção 1 ∩ Opção 2 (`manifest_all.csv`) | 832 |
| Teste (`Coronary_Segmentation_deep_learning/data_list/split_original.csv`, `split=test`) | 208 |
| Teste ∩ interseção | 208 (todos os casos de teste estão na interseção) |
| **Casos analisados = interseção − teste** | **624** ✔ |

## 2. Checagem prévia: perda da Opção 1 nos 208 casos de teste

Mede o viés de avaliar a Opção 1 contra o label **recortado** (os voxels cortados somem do
ground truth, então não contam como erro na avaliação).

| Medida | Valor |
|---|---|
| Casos com `option1_loss_pct > 0` | **30 / 208 (14,4%)** |
| Perda média (208 casos) | 0,155% |
| Perda mediana (208 casos) | 0% |
| p95 (208 casos) | 0,656% |
| Mediana entre os 30 com perda > 0 | 0,442% |
| Média entre os 30 com perda > 0 | 1,074% |
| **Máximo** | **12,00% (caso 419)**; seguintes: 333 (3,92%), 800 (2,89%), 221 (1,94%) |

Distribuição por faixa: `=0`: 178 · `(0, 1]`: 23 · `(1, 5]`: 6 · `(5, 20]`: 1 · `>20`: 0.
Na Opção 2 a perda é 0 nos 208 casos de teste, por construção.

## 3. Método

### 3.1 Insumos e desvio em relação ao pedido

- **Labels em resolução integral lidos diretamente dos NIfTI originais**
  (`D:\Doutorado\dataset\original\<bloco>\<id>.label.nii.gz`, voxels `== 1`), **em vez do**
  `label_voxel_cache_all_ok.pkl`. Motivo: esse cache não existe nesta máquina (é gitignored, foi
  gerado em outra) e está em resolução reduzida (2, 2, 1). O pedido é em resolução integral.
- **Validação cruzada:** a perda da variante 1 recalculada aqui é **idêntica** ao
  `option1_loss_pct` do manifesto nos 624 casos (diferença máxima de 9·10⁻¹⁶ pp).
- Spacing: só do cabeçalho NIfTI (`header.get_zooms()`). Nos 624 casos: y/x mediana 0,352 mm
  (p5 0,318, p95 0,406); z = 0,5 mm em todos.

### 3.2 Convenções comuns

- Eixos `(y, x, z)` = eixos 0, 1, 2 do volume, a mesma convenção do resto do repositório.
- Toda caixa é arredondada de forma conservadora (`floor` no mínimo, `ceil` no máximo) e
  **clampada ao volume** antes de medir qualquer coisa.
- `Rv = 1 − V_caixa_efetiva / V_volume`, em voxels de resolução integral. É a caixa efetiva, ou
  seja, o que a rede de fato recebe. Nos docs anteriores `Rv` usava a extensão nominal, sem clamp
  (ver 4.4).
- Perda = fração dos voxels do label fora da caixa.
- Centro do label = centro da caixa exata do label; extensão do label = `max − min + 1` por eixo.
- Âncora em resolução integral = âncora reduzida (`all_cases.csv`) × fatores (2, 2, 1), a mesma
  relação usada em `compute_full_resolution_bbox`.
- Percentis populacionais (variantes 5 e 6) calculados sobre os 624 casos, por eixo.

### 3.3 Variantes

1. **Piso principal:** bbox fixo da Opção 1 (408,5 × 357,2 × 243,1), com os mesmos índices do
   corte de produção (`compute_full_resolution_bbox` + `calibration_both_options.json`).
2. **Piso secundário:** bbox fixo da Opção 2 (436,2 × 379,0 × 257,4).
3. **Oráculo completo:** caixa exata do label de cada caso (perda zero).
4. **Oráculo com orçamento de perda (teto principal):** por caso e por eixo, limites
   `[floor(P_q), floor(P_{100−q}) + 1)` das coordenadas dos voxels do label. `q` é o mesmo para
   todos os casos e foi ajustado por bissecção até a perda média nos 624 casos igualar a da
   variante 1.
5. **Oráculo de tamanho:** tamanho = extensão exata do label do caso; centro = âncora +
   mediana(centro_label − âncora); margem por eixo, somada dos dois lados = percentil p de
   |erro de centro residual| na população. Principal p = 95, secundária p = 100 (5′).
6. **Oráculo de posição:** centro = centro exato do label do caso; tamanho fixo para todos =
   percentil p das extensões do label na população. Principal p = 95, secundária p = 100 (6′).

As definições de 5 e 6 foram reconstruídas nesta sessão e **aprovadas pelo Hélder em 2026-09-29**,
porque a versão anterior do prompt, que as definia, não estava disponível.

## 4. Resultados

### 4.1 `Rv` e perda por variante (624 casos, resolução integral)

| Variante | `Rv` média | mediana | p5 | p95 | Perda média | mediana | p95 | máx. | N perda > 0 |
|---|---|---|---|---|---|---|---|---|---|
| 1 Piso: bbox fixo Opção 1 | **49,18%** | 50,32% | 44,01% | 53,49% | **0,156%** | 0% | 0,894% | 8,36% | 84 |
| 2 Piso sec.: bbox fixo Opção 2 | 40,75% | 40,57% | 36,51% | 46,64% | 0% | 0% | 0% | 0% | 0 |
| 3 Oráculo completo | 78,55% | 79,04% | 69,57% | 85,93% | 0% | 0% | 0% | 0% | 0 |
| **4 Oráculo c/ orçamento (q = 0,0591)** | **79,17%** | 79,65% | 70,65% | 86,36% | **0,156%** | 0,157% | 0,229% | 0,31% | 623 |
| 5 Oráculo de tamanho, p95 | 51,40% | 51,81% | 38,68% | 63,62% | 0,561% | 0% | 1,421% | 95,17% | 68 |
| 6 Oráculo de posição, p95 | 67,23% | 69,21% | 60,12% | 69,46% | 0,251% | 0% | 1,731% | 13,00% | 80 |
| 5′ Oráculo de tamanho, p100 | 28,57% | 28,58% | 16,57% | 40,79% | 0% | 0% | 0% | 0% | 0 |
| 6′ Oráculo de posição, p100 | 55,20% | 57,10% | 48,47% | 58,28% | 0% | 0% | 0% | 0% | 0 |

- **`q` encontrado: 0,0591** (alvo de perda média 0,15594%; obtido 0,15591%).
- Parâmetros populacionais. Margem de posição (variante 5), em voxels y/x/z: p95 = 45,8 / 45,5 /
  30,1; p100 = 81,1 / 73,9 / 60,3. Viés mediano centro_label − âncora = +74,1 / +9,4 / −68,1.
  Tamanho fixo (variante 6): p95 = 355 × 304 × 204; p100 = 395 × 341 × 228. Extensão mediana do
  label: 310 × 264 × 176.

### 4.2 Espaço de ganho e decomposição

**Espaço de ganho = `Rv`(4) − `Rv`(1) = 79,17 − 49,18 = 29,99 pp.**

Fração recuperada = (`Rv`(k) − `Rv`(1)) / 29,99 pp:

| Variante | Fração do espaço de ganho | Perda média associada |
|---|---|---|
| 3 Oráculo completo | 97,9% | 0% |
| **6 Oráculo de posição, p95** | **60,2%** | 0,251% |
| **5 Oráculo de tamanho, p95** | **7,4%** | 0,561% |
| 6′ Oráculo de posição, p100 | 20,1% | 0% |
| 5′ Oráculo de tamanho, p100 | −68,7% | 0% |
| 2 Piso sec. (Opção 2) | −28,1% | 0% |

### 4.3 Espaçamento efetivo após resize para a grade do `--rl 2` (256 × 256 × 128)

Por eixo: (extensão efetiva da caixa em voxels × spacing do caso) / tamanho da grade. A grade é
mapeada eixo a eixo, 0→256, 1→256, 2→128 (`utils/utils.py:reshape_img`, repositório
Coronary_Segmentation_deep_learning).

| Variante | Eixo | Média (mm) | Mediana | p5 | p95 | Spacing nativo (mediana) |
|---|---|---|---|---|---|---|
| 1 | y | 0,568 | 0,563 | 0,509 | 0,651 | 0,352 |
| 1 | x | 0,495 | 0,491 | 0,445 | 0,570 | 0,352 |
| 1 | z | 0,916 | 0,953 | 0,798 | 0,953 | 0,500 |
| 4 | y | 0,422 | 0,423 | 0,356 | 0,489 | 0,352 |
| 4 | x | 0,362 | 0,362 | 0,301 | 0,424 | 0,352 |
| 4 | z | 0,677 | 0,680 | 0,570 | 0,789 | 0,500 |

### 4.4 Observações de medição

- **Clamp da variante 1:** em 347/624 casos a caixa da Opção 1 ultrapassa o volume em pelo menos
  um eixo (z: 300, x: 63, y: 17) e é clampada. Por isso o `Rv` efetivo da variante 1 (média
  49,18%) é maior que o `Rv` nominal, sem clamp, nos mesmos 624 casos (média 47,03%, o número
  comparável ao ≈47% dos docs anteriores).
- **Cauda do oráculo de tamanho p95:** o máximo de 95,2% de perda vem de um único caso, o 809
  (`num_circles_detected` = 64, abaixo do limiar de 5% do critério A, que é 64,3). É
  under-tracking da âncora, que desloca o centro previsto. Sem ele, a perda média da variante 5
  cai de 0,561% para 0,409%. Os seguintes são 80 (27,9%), 134 (23,8%), 427 (22,1%) e 943 (21,7%).
- **Cauda do oráculo de posição p95:** máximo de 13,0% (caso 417), seguido de 706 (7,8%) e 346
  (7,5%).

## 5. Interpretação (sem concluir sobre a viabilidade do bbox dinâmico)

- **O teto é quase todo a caixa exata.** Com `q` = 0,059, o orçamento de perda da variante 1
  compra só 0,6 pp de `Rv` além do oráculo completo (79,17 vs. 78,55). A variante 3 sozinha já
  cobre 97,9% do espaço de ganho. O ganho potencial vem de ajustar a caixa ao caso, não de aparar
  caudas do label.
- **O orçamento de perda se distribui de forma oposta nas duas pontas.** Na variante 1 a perda
  fica concentrada (84 casos com perda > 0, máximo de 8,4%). Na variante 4 a mesma perda média
  fica espalhada em 623 casos, com máximo de 0,31%. É a mesma perda média com perfis de risco
  muito diferentes.
- **Na decomposição, a posição pesa muito mais que o tamanho.** Saber a posição exata (6)
  recupera 60% do espaço de ganho em p95 e 20% em p100. Saber o tamanho exato, mantendo a
  posição pela âncora (5), recupera só 7% em p95 e fica **abaixo do piso** em p100 (−69%). A
  margem necessária para absorver o erro da âncora (30 a 80 voxels por lado, conforme eixo e
  percentil) consome o que se ganha conhecendo o tamanho. As frações não somam 100%: há interação
  entre os dois fatores, e cada oráculo isolado mantém o outro componente no nível populacional.
- **Ressalva de comparabilidade:** as variantes 5 e 6 não têm a perda média igualada à da
  variante 1. Em p95 perdem mais (0,56% e 0,25% contra 0,156%), então suas frações são otimistas
  em relação a uma comparação com a mesma perda. Em p100 perdem 0%, então são pessimistas. Com a
  perda igualada, o valor deve ficar entre os dois.
- **Espaçamento efetivo:** com a caixa da Opção 1, o resize para 256 × 256 × 128 subamostra z em
  ≈1,9× (0,95 mm contra 0,5 nativo) e y/x em ≈1,4 a 1,6×. No teto (variante 4) o fator cai para
  ≈1,35× em z e ≈1,0 a 1,2× em y/x, com x praticamente na resolução nativa. Mesmo o teto não
  chega à resolução nativa em z com 128 fatias.
- **A Opção 2 como piso fica 8,4 pp abaixo da Opção 1** (`Rv` 40,75% contra 49,18%) nos mesmos
  casos, com perda zero, e aparece como −28% do espaço de ganho.
- **Viés de avaliação (seção 2):** avaliar a Opção 1 contra o label recortado deixa de fora, em
  média, 0,155% do label nos casos de teste. O efeito se concentra em 30 dos 208 casos e chega a
  12% num deles (419).

## 6. Reprodutibilidade

- Resultados: `output/bounding_box/dynamic_bbox_step0/`
  - `p0_cases.csv`: por caso, shape, spacing, número de voxels do label, âncora e índices das
    caixas Opção 1/2.
  - `p0_percase.csv`: `Rv` e perda por caso para cada variante.
  - `p0_summary.csv`, `p0_spacing.csv`, `p0_meta.json`: tabelas da seção 4 e parâmetros (`q`,
    margens, tamanhos fixos, viés).
- Scripts ad-hoc: `output/bounding_box/dynamic_bbox_step0/scripts/`
  - `p0_extract.py`: extrai as coordenadas dos labels, com 6 processos em paralelo.
  - `p0_analyze.py`: calcula as variantes, cerca de 1,5 min.
  - O cache de coordenadas (`p0_cache/`, 22 MB) não foi versionado; `p0_extract.py` o regenera.
- Insumos só de leitura: `D:\Doutorado\dataset\original\`, `D:\Doutorado\dataset\cropped_manifests\`
  (`manifest_all.csv`, `calibration_both_options.json`), `output/bounding_box/aorta_anchor/all_cases.csv`,
  `Coronary_Segmentation_deep_learning/data_list/split_original.csv`.
