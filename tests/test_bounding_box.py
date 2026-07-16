"""Testes das etapas 5 (centróide geométrico) e 6 (extensão do bbox) do bounding box cardíaco."""

import numpy as np
import pytest

from utils.segmentation.bounding_box import (
    compute_bounding_box_bounds,
    compute_bounding_extent,
    compute_geometric_centroid,
)


def test_centroid_of_single_voxel_is_its_own_coordinate():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3, 4, 5] = True

    centroid = compute_geometric_centroid(mask)

    np.testing.assert_allclose(centroid, [3.0, 4.0, 5.0])


def test_centroid_of_symmetric_cube_is_its_center():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True  # cubo 3x3x3 nos índices 2,3,4 em cada eixo

    centroid = compute_geometric_centroid(mask)

    np.testing.assert_allclose(centroid, [3.0, 3.0, 3.0])


def test_centroid_respects_yxz_axis_order_for_asymmetric_mask():
    # Bloco não-cúbico para expor qualquer troca acidental entre eixos.
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[0:2, 0:4, 0:6] = True  # y em [0,1], x em [0,3], z em [0,5]

    centroid = compute_geometric_centroid(mask)

    np.testing.assert_allclose(centroid, [0.5, 1.5, 2.5])


def test_accepts_non_boolean_mask_with_nonzero_voxels():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[1, 1, 3] = 1

    centroid = compute_geometric_centroid(mask)

    np.testing.assert_allclose(centroid, [1.0, 1.0, 2.0])


def test_raises_on_empty_mask():
    mask = np.zeros((5, 5, 5), dtype=bool)

    with pytest.raises(ValueError, match="não contém nenhum voxel"):
        compute_geometric_centroid(mask)


def test_raises_on_non_3d_input():
    mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="deve ser 3D"):
        compute_geometric_centroid(mask)


def test_extent_of_single_voxel_is_one_in_every_axis():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3, 4, 5] = True

    extent = compute_bounding_extent(mask)

    np.testing.assert_array_equal(extent, [1, 1, 1])


def test_extent_of_symmetric_cube():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True  # cubo 3x3x3 nos índices 2,3,4 em cada eixo

    extent = compute_bounding_extent(mask)

    np.testing.assert_array_equal(extent, [3, 3, 3])


def test_extent_respects_yxz_axis_order_for_asymmetric_mask():
    # Bloco não-cúbico para expor qualquer troca acidental entre eixos.
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[0:2, 0:4, 0:6] = True  # y em [0,1] (2), x em [0,3] (4), z em [0,5] (6)

    extent = compute_bounding_extent(mask)

    np.testing.assert_array_equal(extent, [2, 4, 6])


def test_extent_ignores_holes_between_extreme_voxels():
    # Dois voxels isolados, longe um do outro: extensão é a distância entre eles,
    # não a contagem de voxels ocupados (bbox mínimo, não volume ocupado).
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[1, 1, 1] = True
    mask[8, 1, 1] = True

    extent = compute_bounding_extent(mask)

    np.testing.assert_array_equal(extent, [8, 1, 1])


def test_extent_accepts_non_boolean_mask_with_nonzero_voxels():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[1, 1, 3] = 1

    extent = compute_bounding_extent(mask)

    np.testing.assert_array_equal(extent, [1, 1, 3])


def test_extent_raises_on_empty_mask():
    mask = np.zeros((5, 5, 5), dtype=bool)

    with pytest.raises(ValueError, match="não contém nenhum voxel"):
        compute_bounding_extent(mask)


def test_extent_raises_on_non_3d_input():
    mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="deve ser 3D"):
        compute_bounding_extent(mask)


def test_bounds_of_single_voxel_min_equals_max():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3, 4, 5] = True

    min_coords, max_coords = compute_bounding_box_bounds(mask)

    np.testing.assert_array_equal(min_coords, [3, 4, 5])
    np.testing.assert_array_equal(max_coords, [3, 4, 5])


def test_bounds_respects_yxz_axis_order_for_asymmetric_mask():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:4, 5:9, 10:16] = True  # y em [2,3], x em [5,8], z em [10,15]

    min_coords, max_coords = compute_bounding_box_bounds(mask)

    np.testing.assert_array_equal(min_coords, [2, 5, 10])
    np.testing.assert_array_equal(max_coords, [3, 8, 15])


def test_bounds_are_consistent_with_extent():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:4, 5:9, 10:16] = True

    min_coords, max_coords = compute_bounding_box_bounds(mask)
    extent = compute_bounding_extent(mask)

    np.testing.assert_array_equal(max_coords - min_coords + 1, extent)


def test_bounds_raises_on_empty_mask():
    mask = np.zeros((5, 5, 5), dtype=bool)

    with pytest.raises(ValueError, match="não contém nenhum voxel"):
        compute_bounding_box_bounds(mask)


def test_bounds_raises_on_non_3d_input():
    mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="deve ser 3D"):
        compute_bounding_box_bounds(mask)
