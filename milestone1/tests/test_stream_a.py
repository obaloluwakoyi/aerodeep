"""
milestone1/tests/test_stream_a.py
Tests for Stream A sensor telemetry preprocessing.
"""

import numpy as np
import pytest

from milestone1.stream_a.preprocessor import (
    SlidingWindowBuffer,
    SignalPreprocessor,
    SensorWindow,
    TOTAL_FEATURES,
)


class TestSlidingWindowBuffer:
    def test_no_output_below_window_size(self):
        buf = SlidingWindowBuffer(window_size_samples=100, overlap_ratio=0.5)
        for i in range(99):
            result = buf.push(i, float(i))
        assert result is None

    def test_yields_on_full_window(self):
        buf = SlidingWindowBuffer(window_size_samples=10, overlap_ratio=0.0)
        result = None
        for i in range(10):
            result = buf.push(i * 10, float(i))
        assert result is not None
        ts, vals = result
        assert len(ts) == 10
        assert len(vals) == 10

    def test_yields_again_after_step(self):
        buf = SlidingWindowBuffer(window_size_samples=10, overlap_ratio=0.5)
        results = []
        for i in range(20):
            r = buf.push(i, float(i))
            if r is not None:
                results.append(r)
        assert len(results) >= 2

    def test_overlap_ratio_zero(self):
        buf = SlidingWindowBuffer(window_size_samples=4, overlap_ratio=0.0)
        results = []
        for i in range(16):
            r = buf.push(i, float(i))
            if r is not None:
                results.append(r)
        assert len(results) == 4


class TestSignalPreprocessor:
    def setup_method(self):
        self.preprocessor = SignalPreprocessor(sampling_rate_hz=1000.0, fft_bins=32)

    def _make_window(self, signal: np.ndarray) -> SensorWindow:
        return SensorWindow(
            unit_id="C1001",
            sensor_id="LP_PRESSURE",
            timestamps=np.arange(len(signal), dtype=np.float64),
            values=signal.astype(np.float32),
            window_start_ms=0,
            window_end_ms=len(signal),
            sampling_rate_hz=1000.0,
        )

    def test_feature_vector_length(self):
        signal = np.random.randn(512)
        window = self._make_window(signal)
        fv = self.preprocessor.extract_features(window)
        assert len(fv.features) == TOTAL_FEATURES

    def test_feature_vector_dtype(self):
        signal = np.random.randn(512)
        window = self._make_window(signal)
        fv = self.preprocessor.extract_features(window)
        assert fv.features.dtype == np.float32

    def test_constant_signal_kurtosis(self):
        # Constant signal → std=0 → kurtosis should be 0 (scipy returns -3 for uniform)
        signal = np.ones(512)
        window = self._make_window(signal)
        fv = self.preprocessor.extract_features(window)
        # Just check no exception and finite values
        assert np.all(np.isfinite(fv.features))

    def test_sinusoidal_signal_fft_peak(self):
        # 50 Hz tone should produce energy in 30-100 Hz band
        t = np.arange(512) / 1000.0
        signal = np.sin(2 * np.pi * 50 * t).astype(np.float32)
        window = self._make_window(signal)
        fv = self.preprocessor.extract_features(window)
        # Band index 2 = 30-100 Hz
        n_stat = 10
        n_fft = 32
        band_start = n_stat + n_fft
        band_30_100 = fv.features[band_start + 2]
        band_0_10 = fv.features[band_start + 0]
        assert band_30_100 > band_0_10

    def test_metadata_preserved(self):
        signal = np.random.randn(512)
        window = self._make_window(signal)
        fv = self.preprocessor.extract_features(window)
        assert fv.unit_id == "C1001"
        assert fv.sensor_id == "LP_PRESSURE"


class TestQualityChecker:
    def test_good_reading(self):
        from milestone1.stream_a.ingestion import QualityChecker
        cfg = [{"id": "LP_PRESSURE", "normal_range": [10.0, 45.0], "critical_threshold": 50.0}]
        qc = QualityChecker(cfg)
        assert qc.check("LP_PRESSURE", 25.0) == 0

    def test_suspect_reading(self):
        from milestone1.stream_a.ingestion import QualityChecker
        cfg = [{"id": "LP_PRESSURE", "normal_range": [10.0, 45.0], "critical_threshold": 50.0}]
        qc = QualityChecker(cfg)
        assert qc.check("LP_PRESSURE", 48.0) == 1

    def test_critical_reading(self):
        from milestone1.stream_a.ingestion import QualityChecker
        cfg = [{"id": "LP_PRESSURE", "normal_range": [10.0, 45.0], "critical_threshold": 50.0}]
        qc = QualityChecker(cfg)
        assert qc.check("LP_PRESSURE", 52.0) == 2
