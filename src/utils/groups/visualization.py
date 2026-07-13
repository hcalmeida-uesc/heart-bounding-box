"""Grouped exports for image/volume visualization helpers."""

from importlib import import_module

_VISUALIZATION_EXPORTS = {
    "plot_mip_projection",
    "plot_preprocessing_grid",
    "plot_stage",
    "plot_slices",
    "visualize_circles_on_slices",
    "save_k3d_plot_html",
    "visualize_3d_k3d",
}


def __getattr__(name):
    parent_pkg = __package__.rpartition(".")[0] or __package__
    if name in _VISUALIZATION_EXPORTS:
        module = import_module(".visualization", parent_pkg)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value
