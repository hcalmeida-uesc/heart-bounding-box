# Bbox centrado no volume — calibração

**Data:** 2026-10-15. **Natureza:** exploratória. Nenhum dataset foi gerado e nenhum artefato de
produção foi alterado. Este documento define e calibra o bbox ancorado no centro do volume (A2),
com a mesma metodologia da Opção 1 (caixa única por percentil dos offsets). **Não escolhe o ponto
final.** Os antecedentes estão nos Passos [0](dynamic_bbox_step0.md), [0b](dynamic_bbox_step0b.md),
[0c](dynamic_bbox_step0c.md) e [0d](dynamic_bbox_step0d.md).

## 1. Método

- **População de calibração:** os 792 casos fora do teste (`split_original.csv`,
  `split=trainval`), **sem Critério C**. Inclui os 6 casos sem aorta detectada, que a A2 não
  precisa.
- **Avaliação:** in-sample nos 792 e out-of-sample nos 208 casos de teste, que não entram na
  calibração.
- **Âncora:** `c = shape / 2` por eixo, em resolução integral.
- **Offsets por caso e eixo:** `off_inf = label_min − c` e `off_sup = label_max + 1 − c`.
- **Caixa:** `[c + P₁₀₀₋ₚ(off_inf), c + Pₚ(off_sup)]` por eixo, com o mesmo `p` nos três eixos.
  Os percentis são calculados sobre os 792. O arredondamento é conservador (`floor` no mínimo,
  `ceil` no máximo), e a caixa é clampada ao volume.
- **Duas variantes de unidade:**
  - **voxels:** offsets e caixa em voxels.
  - **mm:** offset × spacing do caso; a caixa em mm é convertida para voxels em cada caso
    (÷ spacing do caso).
- **Medidas:**
  - `Rv` = 1 − V_caixa_efetiva / V_volume.
  - Perda = fração dos voxels do label fora da caixa.
  - Espaçamento efetivo = extensão da caixa em voxels × spacing do caso / grade do `--rl 2`
    (256 × 256 × 128).
- **Grade de `p`:** 90 a 99,5 com passo 0,5, e 99,55 a 100 com passo 0,05.
- **Referência — Opção 1 de produção:**
  - Nos 792: `Rv` 49,53%, média nos 724 casos que ela cobre (Critério C + aorta detectada).
    Ela não existe para os outros 68.
  - Nos 208 de teste: os mesmos índices do corte salvo em `cropped_option1`. Todos os 208 de teste
    pertencem à interseção curada (Passo 0).
- **Coordenadas do label:** o cache dos Passos 0 e 0c foi estendido aos 208 casos de teste, só
  com leitura dos labels NIfTI.

## 2. Dispersão dos offsets nos 792 casos

| Lado | Eixo | Voxels: p5 / mediana / p95 | DP (vox) | DP em mm (vox × spacing mediano) | mm: p5 / mediana / p95 | DP (mm) |
|---|---|---|---|---|---|---|
| inf | y | −186,0 / −154,0 / −117,6 | 21,1 | 7,38 | −67,0 / −53,8 / −41,2 | **7,89** |
| inf | x | −158,5 / −103,0 / −49,6 | 33,0 | 11,54 | −54,4 / −36,4 / −17,8 | **11,22** |
| inf | z | −120,5 / −89,3 / −54,5 | 19,9 | 9,94 | −60,3 / −44,6 / −27,3 | **9,94** |
| sup | y | 116,6 / 157,0 / 196,0 | 24,5 | 8,56 | 41,6 / 55,3 / 68,0 | **8,10** |
| sup | x | 102,0 / 163,5 / 215,0 | 32,8 | 11,46 | 34,5 / 57,8 / 76,7 | **12,44** |
| sup | z | 63,3 / 87,5 / 113,2 | 15,6 | 7,81 | 31,6 / 43,8 / 56,6 | **7,81** |

O spacing em mm é o mediano: 0,352 / 0,352 / 0,5. Em z o spacing é 0,5 mm em todos os casos, então
as duas unidades coincidem nesse eixo.

## 3. Pontos de destaque

Para cada ponto: `Rv`, perda in-sample (792) e perda out-of-sample (208), e a Opção 1 de produção
nos mesmos 208.

| Ponto | Unid. | `p` | Caixa nominal | `Rv` 792 média / mediana | Perda 792: média / > 0 / > 1% / > 5% / > 20% / máx. | `Rv` 208 média / mediana | Perda 208: média / > 0 / > 1% / > 5% / > 20% / máx. | N 208 > 1% |
|---|---|---|---|---|---|---|---|---|
| **`Rv` = Opção 1 (49,53%)** | vox | **95,83** | 386 × 379 × 235 vox | 49,74 / 52,31% | 0,589% / 21,2% / 12,0% / 3,4% / 0,13% / 24,5% | 49,82 / 52,31% | **0,517%** / 18,3% / **11,1%** / 2,9% / 0% / 16,1% | 23 |
| **`Rv` = Opção 1 (49,53%)** | mm | **95,32** | 135,5 × 131,6 × 117,0 mm | 49,62 / 50,77% | 0,700% / 22,7% / 12,8% / 5,1% / 0,25% / 22,6% | 49,36 / 50,72% | 0,739% / 19,2% / 11,5% / 5,3% / 0% / 17,2% | 24 |
| perda máx. ≤ 1% | vox | 99,85 | 430,9 × 458,4 × 266,4 vox | 25,74 / 26,56% | 0,002% / 0,8% / 0% / 0% / 0% / 0,89% | 25,73 / 26,56% | 0,007% / 1,4% / 0% / 0% / 0% / 0,61% | 0 |
| perda máx. ≤ 1% | mm | 99,95 | 160,0 × 163,9 × 135,6 mm | 20,96 / 19,98% | 0,002% / 0,5% / 0% / 0% / 0% / 0,89% | 20,37 / 20,16% | 0,003% / 0,5% / 0% / 0% / 0% / 0,65% | 0 |
| ≤ 2% com perda > 1% | vox | 99,5 | 420,1 × 436,4 × 262,1 vox | 31,27 / 32,31% | 0,036% / 2,0% / 0,76% / 0,25% / 0% / 11,4% | 31,27 / 32,31% | 0,022% / 2,4% / 0,96% / 0% / 0% / 1,85% | 2 |
| ≤ 2% com perda > 1% | mm | 99,0 | 146,3 × 148,8 × 129,0 mm | 34,91 / 34,78% | 0,075% / 5,2% / 1,64% / 0,51% / 0% / 10,4% | 34,47 / 35,97% | 0,105% / 5,3% / **3,37%** / 0% / 0% / 4,83% | 7 |
| perda média ≤ 0,156% | vox | 98,5 | 401,1 × 408,4 × 250,3 vox | 40,66 / 42,53% | 0,151% / 8,0% / 3,7% / 0,76% / 0% / 18,3% | 40,69 / 42,53% | **0,184%** / 6,7% / 4,3% / 1,4% / 0% / 8,4% | 9 |
| perda média ≤ 0,156% | mm | 98,5 | 143,4 × 145,8 × 125,1 mm | 38,26 / 38,63% | 0,142% / 7,6% / 3,7% / 0,88% / 0% / 14,6% | 37,85 / 38,82% | **0,207%** / 6,3% / 4,8% / 1,4% / 0% / 8,4% | 10 |
| ***Opção 1 de produção*** | — | — | 408,5 × 357,2 × 243,1 vox | *49,53 / 50,32% (724)* | *1,479% / 25,4% / 17,3% / 7,2% / 1,66% / 61,7% (724)* | *48,98 / 50,32%* | *0,155% / 14,4% / 3,4% / 0,5% / 0% / 12,0%* | *7* |

Os pontos de restrição são o **menor `p` da grade** que satisfaz a restrição **in-sample** (792).
O ponto de `Rv` igual foi obtido por bissecção contínua em `p`.

### 3.1 Espaçamento efetivo após resize para 256 × 256 × 128 (mm, mediana [p95] nos 208 de teste)

| Ponto | Unid. | y | x | z |
|---|---|---|---|---|
| `Rv` = Opção 1 | vox | 0,527 [0,620] | 0,518 [0,609] | 0,918 [0,918] |
| `Rv` = Opção 1 | mm | 0,531 [0,531] | 0,516 [0,516] | 0,914 [0,914] |
| perda máx. ≤ 1% | vox | 0,590 [0,694] | 0,627 [0,738] | 1,043 [1,043] |
| perda máx. ≤ 1% | mm | 0,626 [0,627] | 0,630 [0,642] | 1,062 [1,062] |
| ≤ 2% com perda > 1% | vox | 0,576 [0,678] | 0,598 [0,704] | 1,031 [1,031] |
| ≤ 2% com perda > 1% | mm | 0,573 [0,574] | 0,582 [0,583] | 1,008 [1,008] |
| perda média ≤ 0,156% | vox | 0,549 [0,646] | 0,559 [0,657] | 0,984 [0,984] |
| perda média ≤ 0,156% | mm | 0,562 [0,562] | 0,571 [0,572] | 0,984 [0,984] |
| *Opção 1 de produção* | — | *0,560 [0,657]* | *0,489 [0,575]* | *0,953 [0,953]* |

Os valores in-sample (792) são praticamente idênticos e estão em `a2_highlights.csv`.

## 4. Curva completa (resumo)

Valores selecionados da varredura. A tabela inteira está em `a2_sweep.csv`.

| `p` | Unid. | Caixa nominal | `Rv` 792 | Perda média 792 / 208 | > 1% 792 / 208 | Perda máx. 792 / 208 |
|---|---|---|---|---|---|---|
| 90 | vox | 366 × 348 × 221 | 58,31% | 1,66 / 1,66% | 30,3 / 26,0% | 32,9 / 22,3% |
| 92,5 | vox | 373 × 359 × 227 | 55,22% | 1,16 / 1,16% | 21,7 / 18,3% | 29,2 / 19,9% |
| 95 | vox | 382 × 373 × 234 | 51,09% | 0,70 / 0,64% | 13,8 / 12,0% | 25,4 / 17,1% |
| 97 | vox | 392 × 389 × 240 | 46,50% | 0,37 / 0,33% | 8,0 / 7,2% | 22,6 / 13,3% |
| 98 | vox | 398 × 403 × 246 | 42,77% | 0,22 / 0,25% | 5,1 / 4,8% | 20,1 / 11,5% |
| 99 | vox | 410 × 419 × 258 | 36,88% | 0,08 / 0,09% | 2,1 / 3,4% | 16,2 / 3,9% |
| 99,5 | vox | 420 × 436 × 262 | 31,27% | 0,04 / 0,02% | 0,8 / 1,0% | 11,4 / 1,9% |
| 100 | vox | 440 × 465 × 274 | 22,13% | 0 / 0% | 0 / 0% | 0 / 0% |
| 90 | mm | 129,3 × 122,9 × 110,5 | 56,98% | 1,68 / 1,60% | 28,8 / 24,0% | 29,6 / 20,3% |
| 95 | mm | 135,0 × 131,1 × 116,9 | 50,04% | 0,74 / 0,76% | 12,9 / 12,0% | 22,9 / 17,5% |
| 99 | mm | 146,3 × 148,8 × 129,0 | 34,91% | 0,08 / 0,11% | 1,6 / 3,4% | 10,4 / 4,8% |
| 100 | mm | 162,8 × 167,0 × 137,0 | 18,68% | 0 / 0,002% | 0 / 0% | 0 / 0,37% |

## 5. Interpretação (sem escolher o ponto final)

- **Com `Rv` igual, a A2 perde mais que a Opção 1 nos mesmos 208 casos de teste.** Na unidade
  voxels, `p` = 95,83:

  | Medida (208 de teste) | A2, `Rv` = Opção 1 | Opção 1 |
  |---|---|---|
  | Perda média | 0,517% | 0,155% (≈ 3,3× menor) |
  | Casos com perda > 1% | 23 (11,1%) | 7 (3,4%) |
  | Casos com perda > 5% | 6 | 1 |
  | Perda máxima | 16,1% | 12,0% |

  A diferença vem de populações diferentes. Os 208 de teste pertencem à interseção curada, ou seja,
  são casos com aorta bem segmentada, onde a A1 funciona bem. A A2 foi calibrada nos 792 **sem**
  Critério C e por isso carrega a variabilidade de posição da população toda. Nos 724/792, o
  quadro se inverte: com `Rv` parecido, a Opção 1 perde 1,48% em média, contra 0,59% a 0,70% da A2.
- **A A2 generaliza de forma estável.** O `Rv` depende só do shape, e o in-sample e o
  out-of-sample coincidem a menos de 0,6 pp. A perda média nos 208 acompanha a dos 792 em quase
  toda a curva, e a perda máxima é sistematicamente menor nos 208.
- **As restrições calibradas in-sample nem sempre valem fora da amostra.**
  - Perda média ≤ 0,156%: nos 208, a perda fica em 0,184% (voxels) e 0,207% (mm).
  - ≤ 2% dos casos com perda > 1%: nos 208, a unidade mm (`p` 99,0) chega a 3,4% dos casos; a
    unidade voxels (`p` 99,5) fica em 0,96%, dentro da restrição.
  - Perda máxima ≤ 1%: vale nos 208 nas duas unidades.
- **mm contra voxels:**
  - **A dispersão dos offsets não diminui em mm.** O desvio-padrão em mm é praticamente igual ao
    desvio-padrão em voxels × spacing mediano (de −0,5 a +1,0 mm por lado e eixo; idêntico em z,
    onde o spacing é constante). A variação de spacing em y/x (0,32 a 0,41 mm, p5–p95) não explica
    a variabilidade de posição e tamanho da árvore em relação ao centro do volume.
  - Em `Rv` × perda, a unidade mm não é melhor. No ponto de `Rv` igual ela perde um pouco mais
    (0,74% contra 0,52% nos 208), e com perda média ≤ 0,156% dá `Rv` menor (38,3% contra 40,7%).
  - **O que a unidade mm muda é o espaçamento efetivo:** ele fica praticamente constante entre os
    casos (p95 ≈ mediana em y/x). A caixa tem o mesmo tamanho físico e passa ao resize para
    256 × 256 × 128 com a mesma resolução em todos os casos. Na unidade voxels, o espaçamento
    efetivo varia com o spacing do caso (p95 cerca de 18% acima da mediana em y/x).
- **Espaçamento efetivo:** no ponto de `Rv` igual, a A2 dá ≈ 0,52 a 0,53 mm em y/x e 0,91 a
  0,92 mm em z. A Opção 1 dá 0,56 / 0,49 / 0,95 mm: a A2 tem caixa mais quadrada em y/x e um pouco
  mais curta em z.
- **Cobertura:** a A2 cobre os 792 casos fora do teste e os 208 de teste, sem exigir aorta
  detectada nem Critério C. A Opção 1 cobre 724 dos 792.

## 6. Reprodutibilidade

- Resultados em `output/bounding_box/a2_center_bbox/`:
  - `a2_sweep.csv`: varredura completa (unidade × `p`; métricas in e out of sample, espaçamento
    efetivo, caixa).
  - `a2_highlights.csv`: pontos de destaque.
  - `a2_offset_dispersion.csv`: tabela da seção 2.
  - `a2_option1_test.csv`: Opção 1 nos 208 de teste.
  - `a2_test_cases.csv`: shape e spacing dos 208 de teste.
- Scripts em `output/bounding_box/a2_center_bbox/scripts/`:
  - `a2_extract_test.py`: estende o cache `p0_cache/` aos casos de teste; reaproveita `work` de
    `p0c_extract.py`, copiado junto.
  - `a2_calibrate.py`: calibração e avaliação.
