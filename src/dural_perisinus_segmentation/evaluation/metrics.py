from typing import Optional, Sequence

import numpy as np
import torch
from clinicadl.metrics import Metric
from skimage.morphology import skeletonize


def _cl_dice(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    tprec = _positive_rate(y_pred, skeletonize(y_true))
    tsens = _positive_rate(y_true, skeletonize(y_pred))
    return _f1_score(tprec, tsens)


def _dice(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    tprec = _positive_rate(y_pred, y_true)
    tsens = _positive_rate(y_true, y_pred)
    return _f1_score(tprec, tsens)


def _positive_rate(v: np.ndarray, s: np.ndarray) -> float:
    return np.sum(v * s) / np.sum(s)

def _f1_score(tprec: float, tsens: float) -> float:
    if (tprec == 0) or (tsens == 0):
        return 0
    return (2 * tprec * tsens / (tprec + tsens)).item()


class DiceMetric(Metric):
    """
    Implementation of Dice Score Coefficient compatible with ClinicaDL.
    """

    _metric_computation = staticmethod(_dice)

    def __init__(
        self,
        pred_key: str,
        label_key: str,
        label: int = 1,
    ):
        self.label_key = label_key
        self.pred_key = pred_key
        self.label = label
        super().__init__()

    def _accumulate(self, batch):
        return torch.tensor(
            [
                self._metric_computation(
                    np.where(
                        sample[self.pred_key].tensor == self.label,
                        1,
                        0,
                    ).squeeze(0),
                    np.where(
                        sample[self.label_key].tensor == self.label,
                        1,
                        0,
                    ).squeeze(0),
                )
                for sample in batch
            ],
        )

    def _aggregate(self, data: torch.Tensor):
        return data.mean().item()


class clDiceMetric(DiceMetric):
    """
    Implementation of clDice (https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)
    compatible with ClinicaDL.
    """

    _metric_computation = staticmethod(_cl_dice)


class VolumeMetric(Metric):
    """
    Compute the volume of a segmentation.
    """

    def __init__(
        self,
        image_key: str,
        label: int = 1,
    ):
        self.image_key = image_key
        self.label = label
        super().__init__()

    def _accumulate(self, batch):
        return torch.tensor(
            [
                _get_volume(
                    np.where(
                        sample[self.image_key].tensor == self.label,
                        1,
                        0,
                    ).squeeze(0),
                    tuple(sample[self.image_key].spacing),
                )
                for sample in batch
            ]
        )

    def _aggregate(self, data: torch.Tensor):
        return data.mean().item()


def _get_volume(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return mask.sum() * np.prod(spacing)
