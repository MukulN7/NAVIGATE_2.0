"""
VelocityModel — V2 Temporal CNN+GRU Vehicle Speed Estimator for NAVIGATE 2.0.

Designed specifically to improve generalization on the IO-VNBD smartphone dataset.

Key Improvements in V2:
1. Preserves temporal sequence information (T'=10) after CNN feature extraction
   instead of flattening the entire 5-second window.
2. Replaces single GRUCell with multi-layer GRU sequence model over feature time steps.
3. Adds 1D and 2D Dropout layers to prevent co-adaptation and overfitting.
4. Drastically reduces parameter count from ~3M down to ~359k to prevent memorization.

Input
-----
    Tensor [B, 50, 6]  (Batch, 50 samples at 10 Hz, 6 IMU channels)

Output
------
    speed  : Tensor [B, 1] — predicted vehicle speed (in normalized target space)
    hx_new : Tensor [2, B, 128] or None — updated GRU hidden states
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class VelocityModel(nn.Module):
    """
    V2 Temporal CNN+GRU Velocity Estimator for IO-VNBD.

    Parameters
    ----------
    in_channels : int
        Number of IMU input channels (default 6: accel_xyz + gyro_xyz).
    hidden_size : int
        GRU hidden state dimension (default 128).
    window_size : int
        Number of time-steps per window (default 50 = 5 s × 10 Hz).
    dropout_rate : float
        Dropout probability (default 0.2).
    """

    CONV1_OUT_CHANNELS = 128
    CONV1_KERNEL = 5
    CONV2_OUT_CHANNELS = 256
    CONV2_KERNEL = 3
    POOL_SIZE = 2

    def __init__(
        self,
        in_channels: int = 6,
        hidden_size: int = 128,
        window_size: int = 50,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dropout_rate = dropout_rate

        # ------ 1D Convolutional Feature Extractor ---------------------- #
        self.conv1 = nn.Conv1d(in_channels, self.CONV1_OUT_CHANNELS, self.CONV1_KERNEL)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(self.POOL_SIZE)
        self.drop1 = nn.Dropout1d(p=dropout_rate / 2)

        self.conv2 = nn.Conv1d(self.CONV1_OUT_CHANNELS, self.CONV2_OUT_CHANNELS, self.CONV2_KERNEL)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(self.POOL_SIZE)
        self.drop2 = nn.Dropout1d(p=dropout_rate / 2)

        # Sequence length after CNN: L = 10 for window_size = 50
        self._seq_len = self._compute_seq_len(window_size)

        # ------ Recurrent GRU Sequence Layer ----------------------------- #
        # Processes temporal sequence [B, T'=10, C'=256]
        self.gru = nn.GRU(
            input_size=self.CONV2_OUT_CHANNELS,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
        )

        # ------ Dense Output Head with Regularization ------------------- #
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------ #
    def _compute_seq_len(self, window_size: int) -> int:
        """Computes feature sequence length after CNN & pooling layers."""
        L = window_size
        L = math.floor((L - self.CONV1_KERNEL + 1) / self.POOL_SIZE)  # Conv1 + Pool1
        L = math.floor((L - self.CONV2_KERNEL + 1) / self.POOL_SIZE)  # Conv2 + Pool2
        return L

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [B, 50, 6]
            Batch of IMU window samples.
        hx : Tensor [2, B, hidden_size], optional
            Initial GRU hidden state.

        Returns
        -------
        speed : Tensor [B, 1]
            Predicted vehicle speed scalar per window.
        hx_new : Tensor [2, B, hidden_size]
            Updated GRU hidden states.
        """
        B, T, C = x.shape

        # [B, 50, 6] → [B, 6, 50]
        x = x.permute(0, 2, 1)

        # CNN Feature Extractor
        x = self.drop1(self.pool1(self.relu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.relu2(self.conv2(x))))
        # x is now [B, 256, 10]

        # Reshape for GRU: [B, 256, 10] → [B, 10, 256] (Batch, Seq_Len=10, Channels=256)
        x = x.permute(0, 2, 1)

        # Temporal GRU sequence processing
        gru_out, hx_new = self.gru(x, hx)  # gru_out: [B, 10, 128]

        # Mean pooling across temporal sequence dimension T'=10
        context = gru_out.mean(dim=1)  # [B, 128]

        # Output speed prediction
        speed = self.head(context)  # [B, 1]

        return speed, hx_new

    # ------------------------------------------------------------------ #
    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        """Returns human-readable architecture summary string."""
        lines = [
            f"VelocityModel V2 (Temporal CNN+GRU for IO-VNBD @ 10 Hz)",
            f"  Input:     [B, {self.window_size}, {self.in_channels}]  (5 s × 10 Hz, 6-channel IMU)",
            f"  Conv1:     Conv1d({self.in_channels}, {self.CONV1_OUT_CHANNELS}, k={self.CONV1_KERNEL}) → ReLU → MaxPool1d(2) → Drop({self.dropout_rate/2})",
            f"  Conv2:     Conv1d({self.CONV1_OUT_CHANNELS}, {self.CONV2_OUT_CHANNELS}, k={self.CONV2_KERNEL}) → ReLU → MaxPool1d(2) → Drop({self.dropout_rate/2})",
            f"  Seq Reshape: [B, {self._seq_len}, {self.CONV2_OUT_CHANNELS}] (preserving temporal structure)",
            f"  GRU:       2-Layer GRU(in={self.CONV2_OUT_CHANNELS}, hidden={self.hidden_size}, drop={self.dropout_rate})",
            f"  Pooling:   Temporal Mean Pooling → [B, {self.hidden_size}]",
            f"  Head:      Drop({self.dropout_rate}) → Linear({self.hidden_size}, 64) → ReLU → Drop({self.dropout_rate}) → Linear(64, 1)",
            f"  Output:    [B, 1] (scalar vehicle speed)",
            f"  Params:    {self.count_parameters():,}",
        ]
        return "\n".join(lines)
