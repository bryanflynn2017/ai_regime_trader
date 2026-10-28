"""
Hidden Markov Model regime classifier.

No-lookahead guarantee (critical):
    predict_series_walk_forward() processes bars in chronological order.
    When computing the regime at bar T, ONLY data from bars 0..T is visible.
    Bars T+1, T+2, ... do not exist yet from the model's perspective.
    The HMM is periodically re-fitted using data[0:T+1]; it is never fitted
    on data that includes the bar being predicted or any future bar.

Fallback behaviour:
    If hmmlearn or scikit-learn is unavailable, every method silently falls
    back to FallbackRegimeClassifier. The caller receives a valid RegimeResult
    either way — the method field distinguishes "hmm" from "fallback".
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from regime.fallback_regime import FallbackRegimeClassifier, RegimeResult
from strategy.signals import RegimeLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation if absent
# ---------------------------------------------------------------------------

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM  # type: ignore
    from sklearn.preprocessing import StandardScaler as _StandardScaler  # type: ignore
    _HMM_AVAILABLE = True
    logger.debug("hmmlearn + sklearn loaded successfully.")
except ImportError:
    _HMM_AVAILABLE = False
    logger.info(
        "hmmlearn/sklearn not available — HMMRegimeEngine will use FallbackRegimeClassifier."
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Features passed to the HMM.  All are stationary / normalised by construction.
HMM_FEATURES: List[str] = [
    "log_returns",
    "rolling_volatility_20",
    "trend_score",
    "volume_ratio",
    "drawdown_from_252_high",
]

#: Minimum clean rows needed before we attempt to fit an HMM.
MIN_TRAINING_ROWS: int = 100

#: Re-fit the HMM every this many bars in walk-forward mode (~1 calendar month).
DEFAULT_REFIT_INTERVAL: int = 21


# ---------------------------------------------------------------------------
# HMMRegimeEngine
# ---------------------------------------------------------------------------

class HMMRegimeEngine:
    """
    Gaussian HMM-based market regime classifier with walk-forward no-lookahead.

    Usage (single-shot):
        engine = HMMRegimeEngine()
        engine.fit(df_train)
        result = engine.predict_latest(df_recent)

    Usage (walk-forward series):
        engine = HMMRegimeEngine()
        df_with_regimes = engine.predict_series_walk_forward(df)
    """

    def __init__(
        self,
        n_components: int = 5,
        training_window: Optional[int] = None,
        refit_interval: int = DEFAULT_REFIT_INTERVAL,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            n_components:     Number of HMM hidden states (should equal number
                              of desired regime labels).
            training_window:  Rolling training window length in bars.
                              None = use all available data from bar 0.
            refit_interval:   Re-fit every this many bars in walk-forward mode.
            random_state:     Seed for reproducibility.
        """
        self.n_components = n_components
        self.training_window = training_window
        self.refit_interval = refit_interval
        self.random_state = random_state

        self._model: object = None
        self._scaler: object = None
        self._state_to_label: Dict[int, RegimeLabel] = {}
        self._fallback = FallbackRegimeClassifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if hmmlearn and scikit-learn are installed."""
        return _HMM_AVAILABLE

    def fit(self, df: pd.DataFrame) -> "HMMRegimeEngine":
        """
        Fit the HMM on df.

        If hmmlearn is unavailable or training fails, logs a warning and
        leaves the model unfitted (future predictions use the fallback).

        Returns self for chaining.
        """
        if not _HMM_AVAILABLE:
            logger.warning("hmmlearn unavailable — fit() is a no-op.")
            return self

        df = self._ensure_features(df.copy())
        X, _ = self._extract_feature_matrix(df)

        if len(X) < MIN_TRAINING_ROWS:
            logger.warning(
                "Only %d clean rows available (need %d). HMM not fitted.",
                len(X), MIN_TRAINING_ROWS,
            )
            return self

        try:
            scaler = _StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = _GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=100,
                tol=1e-4,
                random_state=self.random_state,
            )
            model.fit(X_scaled)

            self._scaler = scaler
            self._model = model
            self._state_to_label = self._map_states_to_labels(model, X_scaled)

            logger.info(
                "HMM fitted: %d states, %d training rows. Label map: %s",
                self.n_components,
                len(X),
                {k: v.value for k, v in self._state_to_label.items()},
            )
        except Exception as exc:
            logger.warning("HMM fitting failed (%s). Predictions will use fallback.", exc)
            self._model = None

        return self

    def predict_latest(self, df: pd.DataFrame) -> RegimeResult:
        """
        Classify the regime as of the last row of df.

        Fits internally using all available data if no model is fitted yet.
        Falls back to FallbackRegimeClassifier on any error.
        """
        df = self._ensure_features(df.copy())

        if not _HMM_AVAILABLE or self._model is None:
            # Attempt a one-shot fit then predict
            if _HMM_AVAILABLE:
                self.fit(df)
            if self._model is None:
                logger.debug("predict_latest → fallback (HMM unavailable or not fitted).")
                return self._fallback.classify_latest(df)

        X, _ = self._extract_feature_matrix(df)
        if len(X) < MIN_TRAINING_ROWS:
            return self._fallback.classify_latest(df)

        try:
            X_scaled = self._scaler.transform(X)
            states = self._model.predict(X_scaled)
            state_t = int(states[-1])
            label = self._state_to_label.get(state_t, RegimeLabel.NEUTRAL)

            proba = self._model.predict_proba(X_scaled)[-1]
            confidence = float(proba[state_t])

            n_stable = settings.REGIME_STABILITY_BARS
            is_stable = (
                len(states) >= n_stable
                and len(set(states[-n_stable:])) == 1
            )
            is_uncertain = confidence < 0.40

            return RegimeResult(
                label=label,
                confidence=confidence,
                timestamp=pd.Timestamp(df.index[-1]),
                method="hmm",
                reason=(
                    f"HMM state {state_t} → {label.value} "
                    f"(posterior probability {confidence:.2f})"
                ),
                features_used=HMM_FEATURES,
                is_stable=is_stable,
                is_uncertain=is_uncertain,
                metadata={
                    "hmm_state": state_t,
                    "n_components": self.n_components,
                    "state_label_map": {k: v.value for k, v in self._state_to_label.items()},
                },
            )
        except Exception as exc:
            logger.warning("predict_latest failed (%s). Using fallback.", exc)
            return self._fallback.classify_latest(df)

    def predict_series_walk_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute regime for every bar in df using strict walk-forward evaluation.

        No-lookahead invariant:
            Prediction at bar T uses ONLY data[0:T+1].
            Adding bars T+1, T+2, ... CANNOT change the prediction at bar T.

        Efficiency:
            The HMM is re-fitted every `refit_interval` bars rather than every
            single bar.  Between refits, the last fitted model is applied.

        If hmmlearn is unavailable, delegates entirely to
        FallbackRegimeClassifier.classify_series() which preserves the same
        no-lookahead guarantee.
        """
        df = self._ensure_features(df.copy())

        if not _HMM_AVAILABLE:
            logger.info(
                "predict_series_walk_forward → FallbackRegimeClassifier "
                "(hmmlearn not available)."
            )
            return self._fallback.classify_series(df)

        n = len(df)
        labels_out   = [RegimeLabel.NEUTRAL.value] * n
        confs_out    = [0.10] * n
        stable_out   = [False] * n
        methods_out  = ["fallback"] * n

        current_model  = None
        current_scaler = None
        current_map: Dict[int, RegimeLabel] = {}
        last_refit_at  = -1

        # Pre-extract the full feature matrix (backward-looking by construction).
        # X_full is a COMPACT array with one row per valid observation.
        # valid_positions[i] is the ORIGINAL df row index of X_full[i].
        # To select the subset visible at df-bar t, use a boolean mask over
        # valid_positions — do NOT use valid_positions values as X_full indices.
        X_full, valid_mask = self._extract_feature_matrix(df)
        valid_positions = np.where(valid_mask)[0]  # df row indices of clean feature rows

        for t in range(MIN_TRAINING_ROWS, n):
            # ---- Decide whether to refit ----
            need_refit = (
                current_model is None
                or (t - last_refit_at) >= self.refit_interval
            )

            if need_refit:
                # Boolean mask over X_full: select all clean rows whose
                # original df index is <= t (no future data leaks in).
                window_bool_t = valid_positions <= t
                X_t = X_full[window_bool_t] if window_bool_t.any() else np.empty(
                    (0, X_full.shape[1] if X_full.ndim == 2 else 0)
                )

                if len(X_t) >= MIN_TRAINING_ROWS:
                    try:
                        scaler = _StandardScaler()
                        X_scaled_t = scaler.fit_transform(X_t)

                        model = _GaussianHMM(
                            n_components=self.n_components,
                            covariance_type="diag",
                            n_iter=100,
                            tol=1e-4,
                            random_state=self.random_state,
                        )
                        model.fit(X_scaled_t)

                        current_model  = model
                        current_scaler = scaler
                        current_map    = self._map_states_to_labels(model, X_scaled_t)
                        last_refit_at  = t
                        logger.debug("HMM re-fitted at bar %d (%d clean rows).", t, len(X_t))
                    except Exception as exc:
                        logger.debug("HMM refit failed at bar %d: %s", t, exc)

            # ---- Predict ----
            if current_model is None:
                # Fall through to per-row fallback
                try:
                    fb = self._fallback.classify_latest(df.iloc[:t + 1])
                    labels_out[t]  = fb.label.value
                    confs_out[t]   = fb.confidence
                    stable_out[t]  = fb.is_stable
                    methods_out[t] = "fallback"
                except Exception:
                    pass
                continue

            # Same boolean mask: X_full rows whose df index is <= t
            window_bool_t = valid_positions <= t
            if not window_bool_t.any():
                continue

            try:
                X_t      = X_full[window_bool_t]
                X_scaled = current_scaler.transform(X_t)
                states   = current_model.predict(X_scaled)
                state_t  = int(states[-1])
                label    = current_map.get(state_t, RegimeLabel.NEUTRAL)

                proba      = current_model.predict_proba(X_scaled)[-1]
                confidence = float(proba[state_t])

                n_stable    = settings.REGIME_STABILITY_BARS
                is_stable_t = (
                    len(states) >= n_stable
                    and len(set(states[-n_stable:])) == 1
                )

                labels_out[t]  = label.value
                confs_out[t]   = confidence
                stable_out[t]  = is_stable_t
                methods_out[t] = "hmm"
            except Exception as exc:
                logger.debug("Walk-forward predict failed at bar %d: %s", t, exc)

        result = df.copy()
        result["regime_label"]       = labels_out
        result["regime_confidence"]  = confs_out
        result["regime_is_stable"]   = stable_out
        result["regime_method"]      = methods_out

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
        if "log_returns" not in df.columns:
            from data.feature_engineering import add_features
            df = add_features(df)
        return df

    @staticmethod
    def _extract_feature_matrix(
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the HMM feature columns as a numpy array.

        Returns:
            X:          shape (n_valid, n_features)  — rows without any NaN
            valid_mask: boolean array of shape (n_rows,) — True where row is clean
        """
        available = [c for c in HMM_FEATURES if c in df.columns]
        if not available:
            empty = np.empty((0, len(HMM_FEATURES)))
            return empty, np.zeros(len(df), dtype=bool)

        X_df = df[available]
        valid_mask = X_df.notna().all(axis=1).values
        X = X_df.values[valid_mask]
        return X, valid_mask

    def _map_states_to_labels(
        self, model: object, X_scaled: np.ndarray
    ) -> Dict[int, RegimeLabel]:
        """
        Map HMM hidden state integers to RegimeLabel by sorting states on
        mean scaled log-return (first feature in HMM_FEATURES).

        State ordering: lowest return → crash, highest → euphoria.
        """
        states = model.predict(X_scaled)

        # Mean of the FIRST feature (log_returns, scaled) per state.
        state_means: Dict[int, float] = {}
        for s in range(model.n_components):
            mask = states == s
            state_means[s] = float(X_scaled[mask, 0].mean()) if mask.sum() > 0 else 0.0

        sorted_states = sorted(state_means, key=lambda s: state_means[s])

        label_seq = [
            RegimeLabel.CRASH,
            RegimeLabel.BEAR,
            RegimeLabel.NEUTRAL,
            RegimeLabel.BULL,
            RegimeLabel.EUPHORIA,
        ]
        n = len(sorted_states)
        mapping: Dict[int, RegimeLabel] = {}
        for rank, state in enumerate(sorted_states):
            # Interpolate index into label_seq
            idx = round(rank * (len(label_seq) - 1) / max(n - 1, 1))
            mapping[state] = label_seq[idx]

        return mapping
