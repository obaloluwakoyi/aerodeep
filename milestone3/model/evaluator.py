"""
milestone3/model/evaluator.py

Post-training evaluation utilities:
  - Threshold optimisation for fault classification
  - Confusion matrix generation
  - SHAP-based node/sensor attribution
  - Calibration analysis for TTF regression
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
)
from loguru import logger


class FaultClassifierEvaluator:
    """
    Evaluation suite for the multi-label fault classification head.
    """

    def __init__(self, fault_classes: List[Dict], threshold: float = 0.45):
        self._classes = fault_classes
        self._threshold = threshold
        self._labels = [
            f"{c['node']}—{c['fault'].replace('_', ' ')}"
            for c in fault_classes
        ]

    def evaluate(
        self,
        y_true: np.ndarray,       # (N_samples, N_classes) binary
        fault_probs: np.ndarray,  # (N_samples, N_classes) [0,1]
    ) -> Dict:
        y_pred = (fault_probs >= self._threshold).astype(int)

        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Macro averages
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            macro_auroc = roc_auc_score(y_true, fault_probs, average="macro")
        except ValueError:
            macro_auroc = float("nan")

        report = {
            "macro_f1": float(macro_f1),
            "macro_auroc": float(macro_auroc),
            "per_class": {
                label: {"f1": float(f1)}
                for label, f1 in zip(self._labels, per_class_f1)
            },
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=self._labels,
                zero_division=0,
            ),
        }

        return report

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        fault_probs: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Grid-search the threshold that maximises macro-F1 on validation data.
        Returns (optimal_threshold, best_f1).
        """
        if thresholds is None:
            thresholds = np.linspace(0.2, 0.8, 25).tolist()

        best_thresh, best_f1 = 0.5, 0.0
        for t in thresholds:
            y_pred = (fault_probs >= t).astype(int)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        logger.info(f"Optimal threshold: {best_thresh:.3f} — macro-F1: {best_f1:.4f}")
        return best_thresh, best_f1


class TTFRegressionEvaluator:
    """Evaluation for the time-to-failure regression head."""

    def evaluate(
        self,
        y_true_hours: np.ndarray,
        y_pred_hours: np.ndarray,
    ) -> Dict:
        mae = mean_absolute_error(y_true_hours, y_pred_hours)
        errors = y_pred_hours - y_true_hours
        bias = float(np.mean(errors))
        within_10h = float(np.mean(np.abs(errors) <= 10.0))
        within_24h = float(np.mean(np.abs(errors) <= 24.0))

        return {
            "mae_hours": float(mae),
            "bias_hours": bias,
            "within_10h_pct": within_10h * 100,
            "within_24h_pct": within_24h * 100,
            "p50_error": float(np.percentile(np.abs(errors), 50)),
            "p90_error": float(np.percentile(np.abs(errors), 90)),
        }


class NodeAttribution:
    """
    Simple gradient-based node attribution: which nodes drove the prediction?
    Uses input × gradient (integrated gradients approximation).
    """

    def __init__(self, model, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        self._model = model
        self._edge_index = edge_index
        self._edge_attr = edge_attr

    def attribute_fault(
        self,
        ts_sequence: Dict[str, torch.Tensor],
        txt_embeddings: Dict[str, torch.Tensor],
        fault_idx: int,
    ) -> Dict[str, float]:
        """
        Returns per-node attribution scores for a specific fault class.
        Higher score = node contributed more to triggering this fault prediction.
        """
        node_ids = list(ts_sequence.keys())

        # Enable gradients for ts_sequence
        ts_grad = {
            nid: ts_sequence[nid].float().requires_grad_(True)
            for nid in node_ids
        }

        self._model.eval()
        _, fault_logits, _, _ = self._model(
            ts_grad, txt_embeddings,
            self._edge_index, self._edge_attr
        )

        # Backprop through fault_idx logit
        fault_logits[fault_idx].backward()

        # Attribution = mean |grad| × |input| per node
        attributions = {}
        for nid in node_ids:
            grad = ts_grad[nid].grad
            if grad is not None:
                attr = float((grad.abs() * ts_grad[nid].abs()).mean().item())
            else:
                attr = 0.0
            attributions[nid] = attr

        # Normalise to [0,1]
        max_attr = max(attributions.values()) + 1e-8
        return {nid: v / max_attr for nid, v in attributions.items()}
