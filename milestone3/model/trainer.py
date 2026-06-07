"""
milestone3/model/trainer.py

PyTorch Lightning training module for AeroDeepDiagnosticModel.
Handles multi-task loss, learning rate scheduling, early stopping,
and Weights & Biases logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC
import wandb
from loguru import logger

from milestone3.model.dual_head import AeroDeepDiagnosticModel
from milestone2.graph.schema import CompressorGraphSchema


class AeroDeepLightningModule(pl.LightningModule):
    """
    Lightning wrapper for training AeroDeepDiagnosticModel.

    Multi-task loss:
      total_loss = alpha * L_ttf + (1 - alpha) * L_fault

    L_ttf:   Huber loss (robust to outlier TTF labels in early-stage data)
    L_fault: Binary cross-entropy with logits (multi-label)
    """

    def __init__(
        self,
        timeseries_dim: int = 128,
        text_dim: int = 768,
        fusion_dim: int = 256,
        hidden_channels: List[int] = None,
        n_fault_classes: int = 18,
        dropout: float = 0.3,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        ttf_loss_weight: float = 0.4,
        fault_loss_weight: float = 0.6,
        warmup_steps: int = 500,
        fault_threshold: float = 0.45,
        pos_weight_fault: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hidden_channels is None:
            hidden_channels = [256, 128, 64]

        self.model = AeroDeepDiagnosticModel(
            timeseries_dim=timeseries_dim,
            text_dim=text_dim,
            fusion_dim=fusion_dim,
            hidden_channels=hidden_channels,
            n_fault_classes=n_fault_classes,
            dropout=dropout,
        )

        self._ttf_w = ttf_loss_weight
        self._fault_w = fault_loss_weight
        self._threshold = fault_threshold

        # pos_weight handles class imbalance in fault labels
        self._pos_weight = pos_weight_fault

        # Metrics
        self.val_mae = MeanAbsoluteError()
        self.val_f1 = MultilabelF1Score(
            num_labels=n_fault_classes, threshold=fault_threshold, average="macro"
        )
        self.val_auroc = MultilabelAUROC(num_labels=n_fault_classes, average="macro")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch):
        return self.model(
            batch["ts_sequence"],
            batch["txt_embeddings"],
            batch["edge_index"],
            batch["edge_attr"],
        )

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        ttf_pred: torch.Tensor,
        fault_logits: torch.Tensor,
        y_ttf: torch.Tensor,
        y_fault: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_ttf = F.huber_loss(ttf_pred.squeeze(-1), y_ttf, delta=5.0)

        pos_weight = self._pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(fault_logits.device)

        l_fault = F.binary_cross_entropy_with_logits(
            fault_logits, y_fault, pos_weight=pos_weight
        )

        total = self._ttf_w * l_ttf + self._fault_w * l_fault
        return total, l_ttf, l_fault

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        ttf_pred, fault_logits, node_risk, _ = self(batch)
        total, l_ttf, l_fault = self._compute_loss(
            ttf_pred, fault_logits, batch["y_ttf"], batch["y_fault"]
        )

        self.log_dict({
            "train/loss": total,
            "train/loss_ttf": l_ttf,
            "train/loss_fault": l_fault,
        }, on_step=True, on_epoch=True, prog_bar=True)

        return total

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        ttf_pred, fault_logits, _, _ = self(batch)
        total, l_ttf, l_fault = self._compute_loss(
            ttf_pred, fault_logits, batch["y_ttf"], batch["y_fault"]
        )

        fault_probs = torch.sigmoid(fault_logits)
        self.val_mae(ttf_pred.squeeze(-1), batch["y_ttf"])
        self.val_f1(fault_probs, batch["y_fault"].long())
        self.val_auroc(fault_probs, batch["y_fault"].long())

        self.log_dict({
            "val/loss": total,
            "val/loss_ttf": l_ttf,
            "val/loss_fault": l_fault,
        }, on_step=False, on_epoch=True)

        return total

    def on_validation_epoch_end(self):
        mae = self.val_mae.compute()
        f1 = self.val_f1.compute()
        auroc = self.val_auroc.compute()

        self.log_dict({
            "val/ttf_mae_hours": mae,
            "val/fault_f1": f1,
            "val/fault_auroc": auroc,
        }, prog_bar=True)

        self.val_mae.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_trainer(cfg: dict, checkpoint_dir: str = "checkpoints/") -> pl.Trainer:
    """Build a configured Lightning Trainer from config dict."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:03d}-{val/fault_f1:.4f}",
            monitor="val/fault_f1",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/fault_f1",
            patience=cfg.get("patience", 15),
            mode="max",
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
    ]

    loggers_list = []
    try:
        wandb_logger = pl.loggers.WandbLogger(
            project=cfg.get("wandb_project", "aerodeep-fault-diag"),
            log_model=True,
        )
        loggers_list.append(wandb_logger)
    except Exception:
        logger.warning("W&B not configured — logging to CSV only")
        loggers_list.append(pl.loggers.CSVLogger("logs/"))

    return pl.Trainer(
        max_epochs=cfg.get("epochs", 120),
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=cfg.get("grad_clip", 1.0),
        callbacks=callbacks,
        logger=loggers_list,
        log_every_n_steps=10,
        deterministic=False,
        enable_progress_bar=True,
    )
