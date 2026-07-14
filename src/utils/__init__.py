"""Pacote utilitário com exports preguiçosos e compatíveis.

Prefira imports diretos dos submódulos em código novo, por exemplo:

```
from utils.project.config import load_config_json
from utils.processing.preprocessing import downscale_image_ndi
```

Os exports abaixo continuam disponíveis para notebooks e scripts antigos.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    # Config / dataset / results
    "deep_update_dict": ".project.config",
    "load_config_json": ".project.config",
    "normalize_runtime_config": ".project.config",
    "save_config_json": ".project.config",
    "serialize_config_for_json": ".project.config",
    "get_data_splits": ".project.dataset",
    "create_timestamped_output_dir": ".project.results",
    # General utils
    "load_img_and_label": ".utils.nifti_io",
    "load_json_file": ".utils.json_io",
    "load_raw_img_and_label": ".utils.nifti_io",
    "save_json_file": ".utils.json_io",
    "save_nii_image": ".utils.nifti_io",
    "save_npy_array": ".utils.nifti_io",
    # Processing
    "downscale_image_ndi": ".processing.preprocessing",
    "downscale_image_opencv": ".processing.preprocessing",
    "largest_connected_component": ".processing.preprocessing",
    "run_core_preprocessing_pipeline": ".processing.preprocessing",
    "threshold_image": ".processing.preprocessing",
    "threshold_image_with_offset": ".processing.preprocessing",
    # Segmentation
    "get_lower_threshold_config": ".segmentation.lower_threshold",
    "normalize_lower_threshold_method": ".segmentation.lower_threshold",
    "resolve_lower_threshold": ".segmentation.lower_threshold",
    "load_and_preprocess_image": ".segmentation.pipeline_preprocessing",
    "compute_geometric_centroid": ".segmentation.bounding_box",
    # Visualization
    "plot_mip_projection": ".visualization.image_slices",
    "plot_preprocessing_grid": ".visualization.preprocessing_views",
    "plot_stage": ".visualization.preprocessing_views",
    "plot_slices": ".visualization.image_slices",
    "visualize_circles_on_slices": ".visualization.image_slices",
    "visualize_3d_k3d": ".visualization.volume",
    "save_k3d_plot_html": ".visualization.volume",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "groups":
        module = import_module(".groups", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_LAZY_EXPORTS) + ["groups"]
