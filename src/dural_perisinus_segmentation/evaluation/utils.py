from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torchio as tio
from scipy.spatial import cKDTree
from torchio.transforms import Transform

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint


def color_mask_by_region(mask: tio.LabelMap, regions: tio.LabelMap) -> tio.LabelMap:
    """
    Color a binary mask by region.

    Infers the region by finding the nearest one in a reference mapping.

    The output will be labeled from 1 to the number of regions.

    Parameters
    ----------
    mask : tio.LabelMap
        The binary mask to color.
    regions : tio.LabelMap
        The reference mapping.

    Returns
    -------
    tio.LabelMap
        The colored mask.
    """
    try:
        np.testing.assert_allclose(mask.affine, regions.affine)
    except AssertionError as e:
        e.add_note("The two inputs are not in the same space!")

    coord_regions = np.argwhere(regions.data > 0).T
    value_regions = regions.data[regions.data > 0]

    tree = cKDTree(coord_regions)

    coords_pred = np.argwhere(mask.data == 1).T
    _, closest_indices = tree.query(coords_pred, k=1)

    output = np.zeros_like(mask.data)
    output[
        coords_pred[:, 0], coords_pred[:, 1], coords_pred[:, 2], coords_pred[:, 3]
    ] = value_regions[closest_indices]

    return tio.LabelMap(tensor=output, affine=mask.affine)


class ColorMask(Transform):
    def __init__(
        self, mask_key: str, regions_key: str, colored_mask_name: str, **kwargs
    ):
        self.mask_key = mask_key
        self.regions_key = regions_key
        self.colored_mask_name = colored_mask_name
        super().__init__(**kwargs)

    def apply_transform(self, data_point: DataPoint) -> DataPoint:
        colored_mask = color_mask_by_region(
            data_point[self.mask_key], data_point[self.regions_key]
        )
        data_point.add_mask(colored_mask, self.colored_mask_name)

        return data_point
