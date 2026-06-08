"""
milestone1/stream_a/preprocessor.py

Sliding-window preprocessor for raw sensor telemetry.
Generates overlapping time windows and applies FFT + statistical
feature extraction. Produces a fixed-length feature vector per
sensor per window, ready for node-level fusion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from loguru import logger


@dataclass
class SensorWindow:
    """A completed window of raw sensor readings."""
    unit_id: str
    sensor_id: str
    timestamps: np.ndarray          # shape (W,)  UNIX ms
    values: np.ndarray              # shape (W,)
    window_start_ms: int
    window_end_ms: int
    sampling_rate_hz: float


@dataclass
class SensorFeatureVector:
    """
    Dense feature vector for a single sensor window.
    Contains statistical, spectral, and entropy features.
    """
    unit_id: str
    sensor_id: str
    window_start_ms: int
    window_end_ms: int
    features: np.ndarray            # shape (N_FEATURES,)
    feature_names: List[str]


# Total feature count per sensor per window
N_STAT_FEATURES = 10
N_FFT_FEATURES = 32     # top-N FFT magnitude bins
N_BAND_FEATURES = 5     # energy in frequency bands
TOTAL_FEATURES = N_STAT_FEATURES + N_FFT_FEATURES + N_BAND_FEATURES  # = 47


class SlidingWindowBuffer:
    """
    Accumulates incoming sensor readings into a circular buffer,
    yields complete overlapping windows for processing.
    """

    def __init__(
        self,
        window_size_samples: int,
        overlap_ratio: float = 0.5,
    ):
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        self._window_size = window_size_samples
        self._step = max(1, int(window_size_samples * (1.0 - overlap_ratio)))
        self._buffer: Deque[Tuple[int, float]] = deque(maxlen=window_size_samples)
        self._samples_since_last_yield = 0

    def push(self, timestamp_ms: int, value: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Push a single sample. Returns (timestamps, values) arrays when a
        complete window is ready, otherwise None.
        """
        self._buffer.append((timestamp_ms, value))
        self._samples_since_last_yield += 1

        if (
            len(self._buffer) == self._window_size
            and self._samples_since_last_yield >= self._step
        ):
            self._samples_since_last_yield = 0
            buf = list(self._buffer)
            ts = np.array([x[0] for x in buf], dtype=np.float64)
            vals = np.array([x[1] for x in buf], dtype=np.float32)
            return ts, vals

        return None


class SignalPreprocessor:
    """
    Per-sensor signal preprocessor.

    For each incoming window:
      1. Z-score normalise
      2. Apply Hann window (FFT leakage suppression)
      3. Compute statistical features
      4. Compute FFT spectral features
      5. Compute band energy features
    """

    # Frequency band boundaries in Hz for offshore compressor monitoring
    # Based on typical fault frequencies for 1500 RPM compressors
    FREQ_BANDS_HZ = [
        (0.5, 10.0),    # Sub-synchronous: rotor instability, surge
        (10.0, 30.0),   # 1× and harmonics: unbalance, misalignment
        (30.0, 100.0),  # Blade pass, vane pass frequencies
        (100.0, 500.0), # Bearing defect frequencies
        (500.0, 1000.0), # High-freq: cavitation, valve flutter
    ]

    def __init__(self, sampling_rate_hz: float, fft_bins: int = 32):
        self._fs = sampling_rate_hz
        self._fft_bins = fft_bins

    def extract_features(self, window: SensorWindow) -> SensorFeatureVector:
        x = window.values.astype(np.float64)

        # ── 1. Normalise ──────────────────────────────────────────────────────
        mu, sigma = np.mean(x), np.std(x)
        x_norm = (x - mu) / (sigma + 1e-8)

        # ── 2. Statistical features ───────────────────────────────────────────
        stat_feats = self._statistical_features(x, x_norm, mu, sigma)

        # ── 3. FFT features ───────────────────────────────────────────────────
        fft_feats = self._fft_features(x_norm)

        # ── 4. Band energy features ───────────────────────────────────────────
        band_feats = self._band_energy_features(x_norm)

        features = np.concatenate([stat_feats, fft_feats, band_feats])
        assert len(features) == TOTAL_FEATURES, (
            f"Feature count mismatch: got {len(features)}, expected {TOTAL_FEATURES}"
        )

        return SensorFeatureVector(
            unit_id=window.unit_id,
            sensor_id=window.sensor_id,
            window_start_ms=window.window_start_ms,
            window_end_ms=window.window_end_ms,
            features=features.astype(np.float32),
            feature_names=self._feature_names(),
        )

    def _statistical_features(
        self, x_raw: np.ndarray, x_norm: np.ndarray, mu: float, sigma: float
    ) -> np.ndarray:
        """10 statistical time-domain features."""
        rms = np.sqrt(np.mean(x_raw ** 2))
        peak = np.max(np.abs(x_raw))
        crest_factor = peak / (rms + 1e-8)
        shape_factor = rms / (np.mean(np.abs(x_raw)) + 1e-8)
        impulse_factor = peak / (np.mean(np.abs(x_raw)) + 1e-8)

        return np.array([
            mu,                          # mean
            sigma,                       # std
            skew(x_raw),                 # skewness
            kurtosis(x_raw),             # kurtosis (excess)
            rms,                         # RMS
            peak,                        # peak amplitude
            crest_factor,                # crest factor
            shape_factor,                # shape factor
            impulse_factor,              # impulse factor
            np.percentile(x_raw, 95) - np.percentile(x_raw, 5),  # p95-p5 range
        ], dtype=np.float64)

    def _fft_features(self, x_norm: np.ndarray) -> np.ndarray:
        """Top-N FFT magnitude bins after Hann windowing."""
        window_fn = np.hanning(len(x_norm))
        x_windowed = x_norm * window_fn
        fft_mag = np.abs(np.fft.rfft(x_windowed))

        # Normalise by window length and pick top bins
        fft_mag /= len(x_norm)

        # Resample to fixed N bins regardless of window length
        n_fft_half = len(fft_mag)
        indices = np.round(
            np.linspace(0, n_fft_half - 1, self._fft_bins)
        ).astype(int)
        return fft_mag[indices].astype(np.float64)

    def _band_energy_features(self, x_norm: np.ndarray) -> np.ndarray:
        """Energy in each predefined frequency band."""
        n = len(x_norm)
        freqs = np.fft.rfftfreq(n, d=1.0 / self._fs)
        fft_power = np.abs(np.fft.rfft(x_norm)) ** 2

        band_energies = []
        for f_lo, f_hi in self.FREQ_BANDS_HZ:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            energy = np.sum(fft_power[mask]) / (n + 1e-8)
            band_energies.append(energy)

        return np.array(band_energies, dtype=np.float64)

    @staticmethod
    def _feature_names() -> List[str]:
        stat_names = [
            "mean", "std", "skewness", "kurtosis",
            "rms", "peak", "crest_factor", "shape_factor",
            "impulse_factor", "p95p5_range",
        ]
        fft_names = [f"fft_bin_{i}" for i in range(N_FFT_FEATURES)]
        band_names = [
            "energy_subsync_0p5_10hz",
            "energy_sync_10_30hz",
            "energy_blade_30_100hz",
            "energy_bearing_100_500hz",
            "energy_hf_500_1000hz",
        ]
        return stat_names + fft_names + band_names


class MultiSensorPreprocessor:
    """
    Manages per-sensor buffers and preprocessors for a single compressor unit.

    Usage:
        preprocessor = MultiSensorPreprocessor(unit_id, sensor_ids, cfg)
        for ts_ms, sensor_id, value in incoming_readings:
            fv = preprocessor.push(ts_ms, sensor_id, value)
            if fv is not None:
                # feature vector ready for this sensor window
    """

    def __init__(
        self,
        unit_id: str,
        sensor_ids: List[str],
        sampling_rate_hz: float = 1000.0,
        window_size_ms: int = 512,
        overlap_ratio: float = 0.5,
        fft_bins: int = 32,
    ):
        self._unit_id = unit_id
        self._fs = sampling_rate_hz
        window_samples = int(sampling_rate_hz * window_size_ms / 1000)

        self._buffers: Dict[str, SlidingWindowBuffer] = {
            sid: SlidingWindowBuffer(window_samples, overlap_ratio)
            for sid in sensor_ids
        }
        self._preprocessor = SignalPreprocessor(sampling_rate_hz, fft_bins)
        logger.info(
            f"MultiSensorPreprocessor ready — unit={unit_id} "
            f"sensors={len(sensor_ids)} window={window_size_ms}ms"
        )

    def push(
        self, timestamp_ms: int, sensor_id: str, value: float
    ) -> Optional[SensorFeatureVector]:
        """
        Push a single reading. Returns a SensorFeatureVector when the
        window for this sensor is complete; None otherwise.
        """
        buf = self._buffers.get(sensor_id)
        if buf is None:
            return None

        result = buf.push(timestamp_ms, value)
        if result is None:
            return None

        ts_arr, val_arr = result
        window = SensorWindow(
            unit_id=self._unit_id,
            sensor_id=sensor_id,
            timestamps=ts_arr,
            values=val_arr,
            window_start_ms=int(ts_arr[0]),
            window_end_ms=int(ts_arr[-1]),
            sampling_rate_hz=self._fs,
        )
        return self._preprocessor.extract_features(window)
