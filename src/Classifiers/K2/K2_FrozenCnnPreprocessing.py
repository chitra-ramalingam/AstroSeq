from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrozenK2CNNPreprocessConfig:
    """Preprocessing contract used by the January 2026 frozen K2 CNN."""

    use_flatten: bool = True
    flatten_window_length: int = 401
    flatten_polyorder: int = 2
    force_relative_flux: bool = True
    robust_center: bool = True
    use_mad_scale: bool = True
    clip_sigma: float | None = 10.0
    fill_nonfinite_with_zero: bool = True
    window_len: int = 512
    stride: int = 256


class FrozenK2CNNTensorBuilder:
    """Build tensors compatible with k2_nocrop_flux_seed46_split303.best.keras.

    The model was trained on tensors produced before the later MIN_SCALE=0.05
    guard was added to K2SegmentDatasetBuilder.  This class intentionally keeps
    the historical MAD/std normalization and is used only by the Phase 2 frozen
    model path.
    """

    LONG_CADENCE_DAYS = 29.4244 / (60.0 * 24.0)
    SHORT_CADENCE_THRESHOLD_DAYS = 0.01

    def __init__(self, config: FrozenK2CNNPreprocessConfig | None = None) -> None:
        self.config = config or FrozenK2CNNPreprocessConfig()

    @staticmethod
    def clean_time_cadences(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Match Lightkurve quality_bitmask='none': drop invalid TIME, retain flux NaNs."""
        time_value = np.asarray(time, dtype=np.float64).reshape(-1)
        flux_value = np.asarray(flux, dtype=np.float32).reshape(-1)
        if len(time_value) != len(flux_value):
            raise ValueError("invalid light curve: time/flux length mismatch")
        keep = np.isfinite(time_value)
        return time_value[keep], flux_value[keep]

    def normalize_short_cadence(
        self,
        time: np.ndarray,
        flux: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adapt short-cadence inputs to the long-cadence sampling used in training."""
        if len(time) < 2:
            return time, flux
        cadence = float(np.nanmedian(np.diff(np.sort(time))))
        if not np.isfinite(cadence) or cadence >= self.SHORT_CADENCE_THRESHOLD_DAYS:
            return time, flux

        keep = np.isfinite(flux)
        time_finite = np.asarray(time[keep], dtype=np.float64)
        flux_finite = np.asarray(flux[keep], dtype=np.float32)
        if len(time_finite) < self.config.window_len:
            return time_finite, flux_finite

        origin = float(np.min(time_finite))
        bins = np.floor((time_finite - origin) / self.LONG_CADENCE_DAYS).astype(np.int64)
        frame = pd.DataFrame({"bin": bins, "time": time_finite, "flux": flux_finite})
        binned = frame.groupby("bin", sort=True, as_index=False).agg({"time": "median", "flux": "median"})
        return binned["time"].to_numpy(np.float64), binned["flux"].to_numpy(np.float32)

    def flatten_and_relative(
        self,
        time: np.ndarray,
        flux: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Historical flatten and relative-flux conversion, including raw fallback."""
        cfg = self.config
        time_value = np.asarray(time, dtype=np.float64)
        flux_value = np.asarray(flux, dtype=np.float32)

        if cfg.use_flatten:
            try:
                import lightkurve as lk

                light_curve = lk.LightCurve(time=time_value, flux=flux_value)
                flattened = light_curve.flatten(
                    window_length=cfg.flatten_window_length,
                    polyorder=cfg.flatten_polyorder,
                )
                flux_value = np.asarray(flattened.flux.value, dtype=np.float32)
                time_value = np.asarray(flattened.time.value, dtype=np.float64)
            except Exception:
                # The training builder used this exact raw-flux fallback.
                pass

        if cfg.force_relative_flux:
            divisor = np.nanmedian(flux_value)
            if np.isfinite(divisor) and divisor != 0.0:
                flux_value = flux_value / divisor
            else:
                divisor = np.nanmean(flux_value)
                if np.isfinite(divisor) and divisor != 0.0:
                    flux_value = flux_value / divisor

        return time_value, flux_value.astype(np.float32, copy=False)

    def standardize_flux(self, flux_relative: np.ndarray) -> np.ndarray:
        """Historical MAD normalization; deliberately has no positive scale floor."""
        cfg = self.config
        values = np.asarray(flux_relative, dtype=np.float32)

        if cfg.robust_center:
            median = np.nanmedian(values)
            centered = values - median
            if cfg.use_mad_scale:
                mad = np.nanmedian(np.abs(centered))
                scale = (1.4826 * mad) if np.isfinite(mad) else np.nan
                if not np.isfinite(scale) or scale <= 0:
                    scale = np.nanstd(values)
            else:
                scale = np.nanstd(values)
            scale = float(scale) + 1e-8
            values = centered / scale

        if cfg.clip_sigma is not None:
            clip = float(cfg.clip_sigma)
            values = np.clip(values, -clip, clip)

        if cfg.fill_nonfinite_with_zero:
            values = values.astype(np.float32, copy=False)
            values[~np.isfinite(values)] = 0.0

        return values.astype(np.float32, copy=False)

    def preprocess(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        *,
        adapt_short_cadence: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        time_value, flux_value = self.clean_time_cadences(time, flux)
        if adapt_short_cadence:
            time_value, flux_value = self.normalize_short_cadence(time_value, flux_value)
        if len(time_value) < self.config.window_len:
            raise ValueError("insufficient valid TIME cadences for a 512-sample tensor")
        time_value, flux_relative = self.flatten_and_relative(time_value, flux_value)
        flux_standardized = self.standardize_flux(flux_relative)
        return time_value, flux_standardized

    def count_windows(self, point_count: int) -> int:
        if point_count < self.config.window_len:
            return 0
        return 1 + (int(point_count) - self.config.window_len) // self.config.stride

    def segment(
        self,
        time: np.ndarray,
        flux_standardized: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct 512x2 windows; incomplete tails are discarded, never padded."""
        window_count = self.count_windows(len(flux_standardized))
        if window_count == 0:
            empty = np.empty((0, self.config.window_len, 2), dtype=np.float32)
            return empty, np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, float)

        tensors: list[np.ndarray] = []
        starts = np.empty(window_count, dtype=np.int64)
        ends = np.empty(window_count, dtype=np.int64)
        mid_times = np.empty(window_count, dtype=float)
        for window in range(window_count):
            start = window * self.config.stride
            end = start + self.config.window_len
            channel0 = np.asarray(flux_standardized[start:end], dtype=np.float32)
            channel1 = np.diff(channel0, prepend=channel0[:1]).astype(np.float32)
            tensors.append(np.stack([channel0, channel1], axis=-1))
            starts[window] = start
            ends[window] = end
            mid_times[window] = float(np.nanmedian(time[start:end]))
        return np.stack(tensors).astype(np.float32), starts, ends, mid_times

    def build_tensor(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        *,
        adapt_short_cadence: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        time_value, flux_standardized = self.preprocess(
            time,
            flux,
            adapt_short_cadence=adapt_short_cadence,
        )
        return self.segment(time_value, flux_standardized)
