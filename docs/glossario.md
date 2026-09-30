# Glossário — siglas e termos da investigação de ROI (Capítulos 2 e 3)

> Referência rápida para os termos usados em `docs/`. Atualizado em 2026-09-30, após o Passo 0d
> da investigação de bbox dinâmico. Onde uma sigla depende de outra, a outra aparece antes.

## 1. Grandezas básicas

| Termo | O que é |
|---|---|
| **Âncora** | Ponto de referência calculado na imagem, sem rótulo, em torno do qual a caixa (bbox) é posicionada. |
| **Offset** | Distância, por eixo, entre a âncora e o limite do label coronário. `offset_min` é o limite de baixo (negativo), `offset_max` o de cima. |
| **Extensão (`envelope_extent`)** | `offset_max − offset_min`: o tamanho que a caixa precisaria ter, naquele eixo, para caber o label daquele caso. |
| **Rv** | Redução volumétrica: 1 − (volume da caixa / volume da imagem). Rv de 50% = a caixa tem metade do volume original. Rv negativo = a caixa é maior que a imagem. Nos passos 0 a 0d, calculado com a caixa efetiva (cortada nos limites do volume). |
| **Perda** | Fração dos voxels do label coronário que fica fora da caixa, por caso. |
| **p (percentil dos offsets)** | Define o tamanho da caixa fixa. p = 100 usa o pior caso (mínimo e máximo absolutos dos offsets); p = 95 ignora os 5% mais extremos. Quanto menor o p, menor a caixa e maior o risco de perda. Na Opção 1 aparece como `offset_percentile`. |
| **`exclusion_pct`** | Percentual de casos excluídos por um critério de qualidade, dividido igualmente entre as duas pontas da distribuição (5% = 2,5% de cada lado). |
| **Espaço de ganho** | Diferença de Rv entre o piso (bbox fixo) e o teto (oráculo). Mede quanto um bbox ajustado por caso poderia ganhar, no máximo. |
| **Espaçamento efetivo** | Tamanho do voxel, em mm, depois que o recorte é redimensionado para a grade da rede (256 × 256 × 128 no `--rl 2`). Quanto menor, mais resolução o coração recebe. |

## 2. Segmentação da aorta (Capítulo 2)

| Termo | O que é |
|---|---|
| **LCC** | Maior componente conectado após limiar de intensidade. Remove ar e osso, mas isola o tronco inteiro, não o coração. Ficou só como limpeza. |
| **Hough + Level Set** | Pipeline herdado do Silva (Matheus): o Hough detecta círculos (a aorta) fatia a fatia, e o Level Set gera a máscara 3D da aorta. |
| **`num_circles_detected`** | Quantas fatias tiveram um círculo detectado. Mede, na prática, quanto o rastreamento durou. |
| **`aorta_voxels`** | Volume da máscara da aorta, em voxels. Valores extremos indicam falha: muito pequeno = rastreamento interrompido; muito grande = vazamento para outra estrutura. |

## 3. Critérios de exclusão de casos

| Critério | Base | Usa rótulo? | Situação |
|---|---|---|---|
| **Critério A** | Percentis extremos de `num_circles_detected` | Não | Testado; não pega todos os casos problemáticos. |
| **Critério B** | Percentis extremos da extensão do label | **Sim** | Classificado como curadoria de dataset, não serve para imagem nova. |
| **Critério C** | Percentis extremos de `aorta_voxels`, 5% (2,5% por cauda) | Não | **Adotado** nas Opções 1 e 2. Detecta falha de segmentação da aorta, não erro de posição. |

## 4. Datasets gerados (Capítulo 3)

| Nome | Como foi feito | Casos | Caixa (y × x × z, voxels) | Perda |
|---|---|---|---|---|
| **original** | Imagens completas do ImageCAS | 1000 | — | 0 |
| **Opção 1** (`cropped_option1`) | Âncora da aorta (A1) + Critério C calibrado no split provisório de 88 casos + caixa fixa com p = 95 | 932 | 408,5 × 357,2 × 243,1 | Tolerada (maioria 0; alguns casos perdem) |
| **Opção 2** (`cropped_option2`) | A1 + Critério C calibrado na população toda + caixa com p = 98 + exclusão dos casos cujo label não coube inteiro (checagem que **usa o label**) | 846 | 436,2 × 379,0 × 257,4 | Zero por construção |

Todos os recortes são slicing puro: spacing e HU preservados, sem reamostragem.

## 5. Split do Capítulo 3

| Termo | O que é |
|---|---|
| **Interseção (832)** | Casos presentes nos três datasets (original, Opção 1 e Opção 2). As Opções 1 e 2 não são subconjunto uma da outra: 100 casos só na 1, 14 só na 2. |
| **Teste (208)** | 25% da interseção, sorteado uma vez (seed 42), **idêntico nos três datasets**. É o que permite o teste pareado. |
| **Trainval** | O restante de cada dataset (792, 724 e 638 casos), dividido em 4 folds de validação cruzada. |
| **Split provisório (88 casos)** | `imagecas_splits_train90.json`, usado só como semente nas calibrações antigas. Sem papel metodológico. |

## 6. Âncoras (Passos 0b a 0d)

| Âncora | Definição | Observação |
|---|---|---|
| **A1** | Centróide da máscara da aorta | **Âncora de produção** (Opções 1 e 2). |
| **A2** | Centro geométrico do volume (shape / 2) | Não usa segmentação, nunca falha. |
| **A3** | Posição fixa proporcional ao volume (fração mediana da população) | No ImageCAS é idêntica a A2, porque todos os volumes têm 512 × 512 em y/x e a fração em z é 0,5. |
| **A4** | y e x de A1, z de A3 | Mistura pouco útil: pega o eixo y, em que a aorta é pior. |
| **A5** | y e x de A1, z = ponto mais baixo da máscara da aorta (aproxima a raiz aórtica) | Só avaliada em 30 casos. |
| **A6** | y e z de A2, x de A1 | Combina o melhor eixo de cada uma: volume em y e z, aorta em x. |

## 7. Métodos (Passo 0d)

Método = âncora + regra de exclusão.

| Método | Âncora | Exclusão | Casos cobertos (de 792 fora do teste) |
|---|---|---|---|
| **M1** | A1 | Critério C | 724 (≈ Opção 1 recalibrada) |
| **M2** | A1 | Nenhuma | 786 |
| **M3** | A2 (centro do volume) | Nenhuma | 792 (todos) |
| **M4** | A6 | Critério C | 724 |
| **M5** | A6 | Nenhuma (x de A2 onde não há aorta) | 792 |

## 8. Conjuntos de avaliação (Passos 0 a 0d)

| Conjunto | Definição | N | Cuidado |
|---|---|---|---|
| **Conjunto 1** | Interseção (832) menos teste (208) | 624 | Curado indiretamente com o label (via Opção 2). Favorece A1. |
| **Conjunto 2** | Todos fora do teste com aorta detectada, sem Critério C | 786 | Sem curadoria por rótulo. |
| **Conjunto 2b** | Todos fora do teste, inclusive os 6 sem aorta detectada | 792 | Só A2 e métodos que não dependem da aorta. |

## 9. Variantes do Passo 0 (teto, piso e decomposição)

| Variante | Nome | O que representa |
|---|---|---|
| **1** | Piso principal | Caixa fixa da Opção 1. |
| **2** | Piso secundário | Caixa fixa da Opção 2. |
| **3** | Oráculo completo | Caixa exata do label de cada caso. Perda zero. |
| **4** | Oráculo com orçamento de perda (teto) | Caixa exata, aparando uma fração mínima do label (parâmetro **q**) até igualar a perda média da variante 1. |
| **5** | Oráculo de tamanho | Conhece o tamanho exato do label, mas posiciona pela âncora. |
| **6** | Oráculo de posição | Conhece o centro exato do label, mas usa tamanho fixo. |

- **Oráculo** = referência teórica que usa o label verdadeiro. Mede limites, nunca é método.
- **p95 / p100** nas variantes 5 e 6: margem ou tamanho pelo percentil 95 (tolera alguma perda) ou
  100 (cobertura total).
- **Piso / teto**: o piso é o bbox fixo atual; o teto é o melhor bbox por caso possível. O método
  real fica entre os dois.

## 10. Passos da investigação do bbox dinâmico

| Passo | Pergunta | Resposta |
|---|---|---|
| **0** | Quanto um bbox por caso poderia ganhar, e de onde vem o ganho? | Espaço de ~30 pontos de Rv. A posição explica ~60%, o tamanho ~7%. |
| **0b** | Qual âncora sem rótulo erra menos o centro da árvore? | O centro do volume (A2) erra menos em y e z; a aorta (A1), em x. Nenhuma chega perto do oráculo. |
| **0c** | Com caixa de tamanho único, qual âncora dá mais Rv com a mesma perda média? | Depende do conjunto; o conjunto 1 favorecia A1 por curadoria com o label. |
| **0d** | Curvas de Rv contra risco de perda, por método, e simulação de FOV deslocado | M3 (centro do volume) domina a Opção 1 em todas as métricas no ImageCAS. A aorta só vence com FOV deslocado, e essa robustez ainda é suposta, não medida. |

## 11. Estatística do Capítulo 3

| Termo | O que é |
|---|---|
| **Modelo único** | Dice por caso = média dos 4 folds. Estima o desempenho de um modelo. |
| **Votação** | Máscara final por voto majoritário dos 4 folds (≥ 2 de 4). Mede um comitê de 4 modelos. |
| **Wilcoxon pareado** | Testa se o Dice do mesmo caso difere sistematicamente entre dois datasets. Exige os mesmos 208 casos de teste nos dois. α = 0,05. |
