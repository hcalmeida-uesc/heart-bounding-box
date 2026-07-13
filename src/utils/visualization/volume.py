import numpy as np
import k3d
from skimage import measure
from pathlib import Path
from typing import Any, Optional, Sequence
from numpy.typing import NDArray


def visualize_3d_k3d(
    mask_3d: NDArray[Any],
    spacing: Sequence[float] = (1, 1, 1),
    color: int = 0xFF0000,
    opacity: float = 0.5,
    use_physical_coords: bool = True,
    save_html_path: Optional[str] = None,
    display_plot: bool = True,
) -> Any:
    """Renderiza uma máscara 3D em k3d usando marching cubes."""
    # Define unidade de coordenada (mm ou pixel).
    if use_physical_coords:
        dy, dx, dz = tuple(float(s) for s in spacing)
        axes_labels = ["Y (mm)", "X (mm)", "Z (mm)"]
    else:
        dy, dx, dz = 1.0, 1.0, 1.0
        axes_labels = ["Y (pixels)", "X (pixels)", "Z (pixels)"]

    # Mensagem curta para indicar a geração da malha.
    print("Gerando mesh 3D...")

    # Extrai malha de superfície da máscara.
    verts, faces, normals, values = measure.marching_cubes(
        mask_3d.astype(float), level=0.5, spacing=(dy, dx, dz)
    )

    # Inicializa cena 3D.
    plot = k3d.plot(
        name="Segmentação 3D", height=800, grid_visible=True, axes=axes_labels
    )

    mesh = k3d.mesh(
        verts.astype(np.float32),
        faces.astype(np.uint32),
        color=color,
        opacity=opacity,
        name="Máscara 3D",
    )

    # Adiciona malha e renderiza.
    plot += mesh
    if save_html_path is not None:
        with open(save_html_path, "w", encoding="utf-8") as html_file:
            html_file.write(plot.get_snapshot())

    if display_plot:
        plot.display()
    return plot


def save_k3d_plot_html(plot: Any, html_path: str) -> None:
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(plot.get_snapshot(), encoding="utf-8")
    print(f"HTML salvo em: {html_path}")
