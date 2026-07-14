# Etapa 5 — Centróide geométrico do volume cardíaco

> Parte da sequência do Capítulo 2 (Seção 2.3.1), documentada conforme a convenção definida no
> README ("📝 Acompanhamento por etapa"). Ver também [CODE_NOTES_bounding_box.md](CODE_NOTES_bounding_box.md), Seção 8 (plano original).

## O que foi implementado

- `compute_geometric_centroid(mask)` — [src/utils/segmentation/bounding_box.py:17](../src/utils/segmentation/bounding_box.py#L17).
  Calcula a média aritmética das coordenadas dos voxels não-nulos de uma máscara binária 3D, um
  eixo por vez. Não pondera por intensidade — é o centróide **geométrico** pedido na tese, não um
  centro de massa por HU.
- `load_and_preprocess_image()` passou a retornar também `lcc_mask` (a máscara booleana da maior
  componente conectada, já calculada internamente mas antes descartada) —
  [src/utils/segmentation/pipeline_preprocessing.py:99](../src/utils/segmentation/pipeline_preprocessing.py#L99).
  Essa foi a única mudança na etapa 1-4: sem isso, o centróide teria que ser re-derivado a partir de
  `lcc_image` por um valor de fundo implícito (`-offset`), o que é ambíguo perto do limiar HU
  mínimo. Usar a máscara já computada é direto e sem ambiguidade.

## Uso

```python
from utils.segmentation.pipeline_preprocessing import load_and_preprocess_image
from utils.segmentation.bounding_box import compute_geometric_centroid

image_data = load_and_preprocess_image(img_id=1, base_path=BASE_PATH, config=CONFIG)
centroid = compute_geometric_centroid(image_data["lcc_mask"])
# centroid = array([y_c, x_c, z_c]), em coordenadas de voxel na resolução reduzida
```

## Decisões

- **Convenção de eixos**: entrada e saída seguem `(Y, X, Z)`, a mesma convenção usada em todo o
  pipeline mantido (`shape = (altura, largura, profundidade)`), confirmada em
  `ostia_detection.py`/`aorta_segmentation.py` antes da remoção e documentada em
  `CODE_NOTES_bounding_box.md` (Seção 5). `np.argwhere(mask).mean(axis=0)` preserva essa ordem sem
  nenhuma transposição — a forma mais direta de evitar um bug silencioso de eixo trocado.
- **Erros explícitos em vez de fallback silencioso**: `ValueError` se a máscara não for 3D ou não
  tiver nenhum voxel — situações que não deveriam ocorrer com uma imagem ImageCAS válida, mas que
  indicariam um bug real (threshold/LCC vazio) se acontecessem; preferível estourar a devolver um
  centróide `NaN` sem avisar.
- **Sem ponderação por intensidade**: a tese define o centróide como média das coordenadas dos
  voxels, não centro de massa por HU — por isso a função recebe uma máscara (bool ou não-zero),
  não a imagem de intensidade.

## Validação

- `tests/test_bounding_box.py` — 6 casos: voxel único, cubo simétrico, bloco assimétrico (expõe
  troca de eixos), máscara não-booleana, máscara vazia (deve levantar erro), entrada não-3D (deve
  levantar erro). Rodar com:

  ```bash
  uv run pytest tests/ -v
  ```

  Todos os 6 testes passam (verificado nesta sessão).
- Também foi verificado manualmente que `load_and_preprocess_image`, `compute_geometric_centroid` e
  seus reexports em `utils/__init__.py` e `utils/groups/segmentation.py` importam e resolvem para a
  mesma função, sem quebrar os índices lazy existentes.

## Não incluído nesta etapa

- Agregação do centróide entre as 1000 imagens (isso é a etapa 6 — envelope de dimensões máximas).
- Conversão do centróide para a resolução integral (necessário só na etapa 7, ao posicionar o bbox
  sobre a imagem original).
