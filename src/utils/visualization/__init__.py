"""Visualization domain subpackage - all exports loaded on-demand."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # image_slices
    "plot_mip_projection": "image_slices",
    "plot_slices": "image_slices",
    "visualize_circles_on_slices": "image_slices",
    # preprocessing_views
    "plot_preprocessing_grid": "preprocessing_views",
    "plot_stage": "preprocessing_views",
    # volume
    "visualize_3d_k3d": "volume",
    "save_k3d_plot_html": "volume",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        module = import_module(f".{submodule}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
