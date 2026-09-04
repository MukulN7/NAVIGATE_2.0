"""
attitude_model.py - AVNet-inspired Attitude Estimation Model for NAVIGATE 2.0.

Estimates vehicle attitude as a normalized unit quaternion [qw, qx, qy, qz]
from a 5-second window of 6-axis IMU data (accel XYZ + gyro XYZ at 10 Hz).

Inspired by QDeepOdo DeepOriModel (CNN + GRUCell) but adapted for IO-VNBD:
  - 10 Hz sampling (window_size=50), no hard-coded 50/200 Hz assumptions.
  - Batched parallel processing (batch-first GRU), not a Python step loop.
  - Full 4-component unit quaternion output via L2 normalization.
  - Lightweight design (~195k params) for smartphone inference.
  - Modular: quaternion output feeds directly into a future IEKF.

Input  : Tensor [B, window_size, 6]  (accel XYZ + gyro XYZ)
Output : quaternion [B, 4]            unit quaternion [qw, qx, qy, qz]
         hx_new    [num_layers, B, H] updated GRU hidden state
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttitudeModel(nn.Module):
    """
    AVNet-inspired CNN + GRU Attitude Estimator for IO-VNBD at 10 Hz.

    Architecture
    ============
    Conv1  : Conv1d(6->64, k=5, pad=2) -> BatchNorm1d -> ReLU -> MaxPool1d(2) -> Dropout1d
    Conv2  : Conv1d(64->128, k=3, pad=1) -> BatchNorm1d -> ReLU -> MaxPool1d(2) -> Dropout1d
    GRU    : 2-layer GRU(128, h=128, batch_first=True) over T'=window_size//4 steps
    Pool   : Temporal mean pooling -> [B, 128]
    Head   : Dropout -> Linear(128,64) -> ReLU -> Linear(64,4) -> L2-normalize

    Parameters
    ----------
    in_channels  : int   IMU input channels (default 6).
    hidden_size  : int   GRU hidden state dimension (default 128).
    window_size  : int   Samples per window (default 50 = 5s x 10Hz).
    dropout_rate : float Dropout probability (default 0.25).
    """

    CONV1_OUT    = 64
    CONV1_KERNEL = 5
    CONV2_OUT    = 128
    CONV2_KERNEL = 3
    POOL         = 2

    def __init__(
        self,
        in_channels: int = 6,
        hidden_size: int = 128,
        window_size: int = 50,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()

        self.in_channels  = in_channels
        self.hidden_size  = hidden_size
        self.window_size  = window_size
        self.dropout_rate = dropout_rate

        # ---- CNN Block 1: 6 -> 64 channels --------------------------------- #
        self.conv1 = nn.Conv1d(in_channels, self.CONV1_OUT, self.CONV1_KERNEL, padding=2)
        self.bn1   = nn.BatchNorm1d(self.CONV1_OUT)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(self.POOL)
        self.drop1 = nn.Dropout1d(p=dropout_rate / 2)

        # ---- CNN Block 2: 64 -> 128 channels -------------------------------- #
        self.conv2 = nn.Conv1d(self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL, padding=1)
        self.bn2   = nn.BatchNorm1d(self.CONV2_OUT)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(self.POOL)
        self.drop2 = nn.Dropout1d(p=dropout_rate / 2)

        # Sequence length after CNN (same-padding keeps T, two MaxPool halve it twice)
        self._seq_len = window_size // (self.POOL * self.POOL)

        # ---- 2-Layer Batch-First GRU ---------------------------------------- #
        self.gru = nn.GRU(
            input_size=self.CONV2_OUT,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
        )

        # ---- Quaternion Output Head ------------------------------------------ #
        # Predicts raw 4-vector; L2 normalization enforces ||q||=1 exactly
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Initialize to identity quaternion [1, 0, 0, 0]
        with torch.no_grad():
            last_linear = self.head[-1]
            nn.init.zeros_(last_linear.weight)
            last_linear.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # -------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x  : Tensor [B, window_size, 6]  IMU windows (normalized).
        hx : Tensor [2, B, H], optional  Initial GRU hidden state.

        Returns
        -------
        quaternion : Tensor [B, 4]  unit quaternion [qw, qx, qy, qz]
        hx_new     : Tensor [2, B, H]  updated GRU hidden state
        """
        # [B, T, C] -> [B, C, T]  (channel-first for Conv1d)
        x = x.permute(0, 2, 1)

        # CNN Block 1
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        # CNN Block 2
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        # x: [B, 128, T'] where T' = window_size // 4

        # [B, 128, T'] -> [B, T', 128]  (batch-first for GRU)
        x = x.permute(0, 2, 1)

        # Temporal GRU
        gru_out, hx_new = self.gru(x, hx)   # [B, T', H]

        # Temporal mean pooling
        context = gru_out.mean(dim=1)        # [B, H]

        # Raw quaternion prediction
        q_raw = self.head(context)           # [B, 4]

        # L2 normalize -> unit quaternion  ||q||_2 = 1
        quaternion = F.normalize(q_raw, p=2, dim=-1)  # [B, 4]

        return quaternion, hx_new

    # -------------------------------------------------------------------- #
    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        """Returns human-readable architecture summary."""
        return "\n".join([
            "AttitudeModel (AVNet-inspired CNN+GRU for IO-VNBD @ 10 Hz)",
            f"  Input  : [B, {self.window_size}, {self.in_channels}]",
            f"  Conv1  : Conv1d({self.in_channels}->{self.CONV1_OUT}, k={self.CONV1_KERNEL},"
            f" pad=2) -> BN -> ReLU -> MaxPool(2) -> Drop1d",
            f"  Conv2  : Conv1d({self.CONV1_OUT}->{self.CONV2_OUT}, k={self.CONV2_KERNEL},"
            f" pad=1) -> BN -> ReLU -> MaxPool(2) -> Drop1d",
            f"  Seq    : [B, {self._seq_len}, {self.CONV2_OUT}]",
            f"  GRU    : 2-Layer GRU(in={self.CONV2_OUT}, h={self.hidden_size},"
            f" drop={self.dropout_rate})",
            f"  Pool   : Temporal Mean -> [B, {self.hidden_size}]",
            f"  Head   : Drop -> Linear({self.hidden_size},64) -> ReLU -> Linear(64,4)"
            f" -> L2-Normalize",
            f"  Output : [B, 4] unit quaternion [qw, qx, qy, qz]",
            f"  Params : {self.count_parameters():,}",
        ])
