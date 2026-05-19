"""
Complete Survival Analysis Pipeline with Full Train/Test Evaluation
Version 5.0 - COMPLETE with all metrics and visualizations

Key Features:
- Nested CV or Three-way split for model selection
- Final retrain on ALL train data
- Train AND Test performance evaluation
- Comprehensive metadata with all model details
- 6-panel visualizations
- CatBoost AFT support (3 distributions)
- XGBoost AFT support (3 distributions)

Usage:
------
from survival_pipeline_COMPLETE import SurvivalPipeline

pipeline = SurvivalPipeline(
    approach='nested_cv_with_holdout',  # or 'three_way_with_holdout'
    algorithms=['xgboost', 'catboost', 'xgboost_aft_logistic'],
    n_seeds=3,
    n_tuning_iterations=30,
    outer_cv=5,  # for nested CV
    inner_cv=3,  # for nested CV
    verbose=True
)

pipeline.fit(X, y_time, y_event, train_indices=train_idx, feature_names=features)
pipeline.save_results('results', create_plots=True)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from collections import defaultdict, Counter
import itertools
import warnings
import pickle
import json
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize  # For stacking ensemble
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


##---------------------- annotate this part -------------------------
# def c_statistic_harrell(pred, labels):
#     """
#     Calculate Harrell's C-statistic
    
#     Parameters:
#     -----------
#     pred : array-like
#         Predicted risk scores
#     labels : array-like
#         True survival times (negative for censored)
    
#     Returns:
#     --------
#     float : C-statistic value
#     """
#     matches, total = 0, 0
    
#     if not isinstance(labels, pd.Series):
#         labels = pd.Series(labels)
#     if not isinstance(pred, (pd.Series, np.ndarray, list)):
#         pred = np.array(pred)
    
#     for i in range(len(labels)):
#         for j in range(len(labels)):
#             if labels.iloc[j] > 0 and abs(labels.iloc[i]) > labels.iloc[j]:
#                 total += 1
#                 if pred[j] > pred[i]:
#                     matches += 1
    
#     return matches / total if total > 0 else None




class PairwiseZMeanEnsemble:
    """Z-score mean ensemble for two fitted *risk-score* models.

    Notes
    -----
    This class is defined at module-level (NOT inside a function) so it can be pickled
    when we save pipeline outputs.
    """
    def __init__(self, model_a, model_b, name_a, name_b):
        self.model_a = model_a
        self.model_b = model_b
        self.name_a = name_a
        self.name_b = name_b
        self.mu_a = 0.0
        self.sd_a = 1.0
        self.mu_b = 0.0
        self.sd_b = 1.0

    def fit_scaler(self, X_ref):
        pa = np.asarray(self.model_a.predict(X_ref)).ravel()
        pb = np.asarray(self.model_b.predict(X_ref)).ravel()
        self.mu_a = float(np.nanmean(pa))
        self.sd_a = float(np.nanstd(pa) + 1e-12)
        self.mu_b = float(np.nanmean(pb))
        self.sd_b = float(np.nanstd(pb) + 1e-12)

    def predict(self, X):
        pa = np.asarray(self.model_a.predict(X)).ravel()
        pb = np.asarray(self.model_b.predict(X)).ravel()
        za = (pa - self.mu_a) / self.sd_a
        zb = (pb - self.mu_b) / self.sd_b
        return 0.5 * (za + zb)


# ============================================
# SURVIVAL MODEL WRAPPERS
# ============================================

class XGBoostSurvivalWrapper:
    """XGBoost wrapper with Cox objective"""
    
    def __init__(self, params, seed):
        self.params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'seed': seed,
            **params
        }
        self.num_boost_round = self.params.pop('n_estimators', 100)
        self.model_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y_time, y_event=None):
        X = np.asarray(X)
        y_time = np.asarray(y_time)
        
        if y_event is not None:
            y_event = np.asarray(y_event)
            label = np.where(y_event == 1, y_time, -y_time)
        else:
            label = y_time
        
        # -------------------------
        # EARLY STOPPING (FIXED)
        # -------------------------
        # IMPORTANT: early stopping requires a VALIDATION set.
        # Using evals=[(dtrain,'train')] makes early stopping effectively meaningless.
        # Here we create an internal split *within the current training partition*.
        from sklearn.model_selection import train_test_split

        early_stop_frac = float(self.params.pop('early_stop_frac', 0.2))
        early_stop_rounds = int(self.params.pop('early_stopping_rounds', 50))

        split_rs = self.params.get('seed', 42)  # === NEW: keep wrapper split seed consistent ===
        can_stratify = (y_event is not None and len(np.unique(y_event)) >= 2)  # === NEW ===

        if len(X) > 100 and 0.0 < early_stop_frac < 1.0 and (y_event is None or can_stratify):  # === CHANGED ===
            if y_event is not None:
                X_tr, X_val, lab_tr, lab_val, ev_tr, _ = train_test_split(
                    X, label, y_event,
                    test_size=early_stop_frac,
                    stratify=y_event,
                    random_state=split_rs
                )
            else:
                X_tr, X_val, lab_tr, lab_val = train_test_split(
                    X, label,
                    test_size=early_stop_frac,
                    random_state=split_rs
                )

            dtrain = xgb.DMatrix(X_tr, label=lab_tr)
            dval = xgb.DMatrix(X_val, label=lab_val)

            self.model_ = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=early_stop_rounds,
                verbose_eval=False
            )
        else:
            dtrain = xgb.DMatrix(X, label=label)
            self.model_ = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, 'train')],
                verbose_eval=False
            )
        
        fscore = self.model_.get_score(importance_type='gain')
        self.feature_importances_ = np.array([
            fscore.get(f'f{i}', 0.0) for i in range(X.shape[1])
        ])
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        dtest = xgb.DMatrix(X)
        return self.model_.predict(dtest)


class XGBoostAFTWrapper:
    """XGBoost wrapper with AFT (Accelerated Failure Time) objective"""
    
    def __init__(self, params, seed, aft_loss_distribution='normal'):
        """
        Parameters:
        -----------
        aft_loss_distribution : str
            Distribution for AFT model: 'normal', 'logistic', or 'extreme'
        """
        self.aft_loss_distribution = aft_loss_distribution
        self.params = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': aft_loss_distribution,
            'seed': seed,
            **params
        }
        self.num_boost_round = self.params.pop('n_estimators', 100)
        self.model_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y_time, y_event=None):
        X = np.asarray(X)
        y_time = np.asarray(y_time)
        
        if y_event is not None:
            y_event = np.asarray(y_event)
            # AFT: upper bound for censored, exact value for events
            y_lower = y_time.copy()
            y_upper = np.where(y_event == 1, y_time, np.inf)
        else:
            # Assume already encoded
            y_lower = np.abs(y_time)
            y_upper = np.where(y_time > 0, y_time, np.inf)
        
        # -------------------------
        # EARLY STOPPING (FIXED)
        # -------------------------
        # AFT also requires a validation set for early stopping.
        from sklearn.model_selection import train_test_split

        early_stop_frac = float(self.params.pop('early_stop_frac', 0.2))
        early_stop_rounds = int(self.params.pop('early_stopping_rounds', 50))

        split_rs = self.params.get('seed', 42)  # === NEW ===
        can_stratify = (y_event is not None and len(np.unique(y_event)) >= 2)  # === NEW ===

        if len(X) > 100 and 0.0 < early_stop_frac < 1.0 and (y_event is None or can_stratify):  # === CHANGED ===
            idx = np.arange(len(X))
            if y_event is not None:
                idx_tr, idx_val = train_test_split(
                    idx,
                    test_size=early_stop_frac,
                    stratify=y_event,
                    random_state=split_rs
                )
            else:
                idx_tr, idx_val = train_test_split(
                    idx,
                    test_size=early_stop_frac,
                    random_state=split_rs
                )

            X_tr, X_val = X[idx_tr], X[idx_val]
            yl_tr, yu_tr = y_lower[idx_tr], y_upper[idx_tr]
            yl_val, yu_val = y_lower[idx_val], y_upper[idx_val]

            dtrain = xgb.DMatrix(X_tr)
            dtrain.set_float_info('label_lower_bound', yl_tr)
            dtrain.set_float_info('label_upper_bound', yu_tr)

            dval = xgb.DMatrix(X_val)
            dval.set_float_info('label_lower_bound', yl_val)
            dval.set_float_info('label_upper_bound', yu_val)

            self.model_ = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=early_stop_rounds,
                verbose_eval=False
            )
        else:
            dtrain = xgb.DMatrix(X)
            dtrain.set_float_info('label_lower_bound', y_lower)
            dtrain.set_float_info('label_upper_bound', y_upper)
            self.model_ = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, 'train')],
                verbose_eval=False
            )
        
        fscore = self.model_.get_score(importance_type='gain')
        self.feature_importances_ = np.array([
            fscore.get(f'f{i}', 0.0) for i in range(X.shape[1])
        ])
        
        return self
    
    def predict(self, X):
        """Predict survival time (lower values = higher risk)"""
        X = np.asarray(X)
        dtest = xgb.DMatrix(X)
        # AFT predicts log(time), we return negative for risk scoring
        return -self.model_.predict(dtest)


class CatBoostSurvivalWrapper:
    """CatBoost wrapper with Cox loss"""
    
    def __init__(self, params, seed):
        self.params = {
            'loss_function': 'Cox',
            'eval_metric': 'Cox',
            'random_seed': seed,
            'verbose': False,
            **params
        }
        self.model_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y_time, y_event=None):
        X = np.asarray(X)
        y_time = np.asarray(y_time)
        
        if y_event is not None:
            y_event = np.asarray(y_event)
            label = np.where(y_event == 1, y_time, -y_time)
        else:
            label = y_time
        
        # CatBoost with early stopping needs eval_set
        # Split data for validation
        from sklearn.model_selection import train_test_split
        # NOTE: this split is only for early stopping within the current TRAIN partition
        # Use a configurable fraction (default 0.2) instead of hard-coding.
        early_stop_frac = self.params.get('early_stop_frac', 0.2)
        split_rs = self.params.get('random_seed', 42)  # === NEW ===
        can_stratify = (y_event is not None and len(np.unique(y_event)) >= 2)  # === NEW ===
        if len(X) > 100 and 0.0 < early_stop_frac < 1.0 and (y_event is None or can_stratify):  # === CHANGED ===
            if y_event is not None:
                X_train, X_val, y_train, y_val, _, _ = train_test_split(
                    X, label, y_event,
                    test_size=early_stop_frac,
                    stratify=y_event,
                    random_state=split_rs
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, label,
                    test_size=early_stop_frac,
                    random_state=split_rs
                )
            
            # Build CatBoost params safely (remove helper keys)
            params_with_early_stop = dict(self.params)
            params_with_early_stop.pop('early_stop_frac', None)
            params_with_early_stop.update({'early_stopping_rounds': 50, 'use_best_model': True})
            
            self.model_ = CatBoostRegressor(**params_with_early_stop)
            self.model_.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
        else:
            # Too few samples, no early stopping
            params_no_helper = dict(self.params)
            params_no_helper.pop('early_stop_frac', None)
            self.model_ = CatBoostRegressor(**params_no_helper)
            self.model_.fit(X, label, verbose=False)
        
        self.feature_importances_ = np.array(self.model_.get_feature_importance())
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        return self.model_.predict(X)


class CatBoostAFTWrapper:
    """CatBoost wrapper with SurvivalAft loss (official CatBoost AFT)"""
    
    def __init__(self, params, seed, aft_loss_distribution='normal', scale=None):
        """
        CatBoost AFT (Accelerated Failure Time) model
        
        Based on: https://github.com/catboost/tutorials/blob/master/regression/survival.ipynb
        
        Parameters:
        -----------
        aft_loss_distribution : str
            Distribution: 'normal', 'logistic', or 'extreme'
        scale : float, optional
            Distribution scale parameter (default: None, uses CatBoost default)
            Example values:
            - Logistic: 1.0-2.0 (default: 1.0)
            - Extreme: 1.0-3.0 (default: 1.0)
            - Normal: typically not needed
        """
        self.aft_loss_distribution = aft_loss_distribution.lower()
        self.scale = scale
        
        # CatBoost format: SurvivalAft:dist={Normal|Logistic|Extreme}[;scale={value}]
        dist_map = {
            'normal': 'Normal',
            'logistic': 'Logistic', 
            'extreme': 'Extreme'
        }
        
        dist_name = dist_map.get(self.aft_loss_distribution, 'Normal')
        
        # Build loss function string
        loss_str = f'SurvivalAft:dist={dist_name}'
        if scale is not None:
            loss_str += f';scale={scale}'
        
        self.params = {
            'loss_function': loss_str,
            'eval_metric': 'SurvivalAft',
            'random_seed': seed,
            'verbose': False,
            **params
        }
        self.model_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y_time, y_event=None):
        """
        Fit CatBoost AFT model
        
        CatBoost AFT requires interval labels as 2-column array:
        - Column 1: y_lower (lower bound)
        - Column 2: y_upper (upper bound)
        
        For right-censored data: use -1 for +infinity
        Example:
        - Event=1 (observed): [t, t]
        - Event=0 (censored): [t, -1]
        """
        X = np.asarray(X)
        y_time = np.asarray(y_time)
        
        if y_event is not None:
            y_event = np.asarray(y_event)
            
            # Create interval labels for CatBoost AFT
            # Event=1: [y_time, y_time] (exact observation)
            # Event=0: [y_time, -1] (right-censored, -1 means +inf)
            y_lower = y_time
            y_upper = np.where(y_event == 1, y_time, -1)
            
            label = np.column_stack([y_lower, y_upper])
        else:
            # Assume already in interval format
            if y_time.ndim == 1:
                # Single column - treat as exact observations
                label = np.column_stack([y_time, y_time])
            else:
                label = y_time
        
        # CatBoost AFT with early stopping
        from sklearn.model_selection import train_test_split
        # NOTE: this split is only for early stopping within the current TRAIN partition
        # Use a configurable fraction (default 0.2) instead of hard-coding.
        early_stop_frac = self.params.get('early_stop_frac', 0.2)
        split_rs = self.params.get('random_seed', 42)  # === NEW ===
        can_stratify = (y_event is not None and len(np.unique(y_event)) >= 2)  # === NEW ===
        if len(X) > 100 and 0.0 < early_stop_frac < 1.0 and (y_event is None or can_stratify):  # === CHANGED ===
            if y_event is not None:
                X_train, X_val, y_train, y_val, _, _ = train_test_split(
                    X, label, y_event,
                    test_size=early_stop_frac,
                    stratify=y_event,
                    random_state=split_rs
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, label,
                    test_size=early_stop_frac,
                    random_state=split_rs
                )
            
            # Build CatBoost params safely (remove helper keys)

            
            params_with_early_stop = dict(self.params)

            
            params_with_early_stop.pop('early_stop_frac', None)

            
            params_with_early_stop.update({'early_stopping_rounds': 50, 'use_best_model': True})

            
            self.model_ = CatBoostRegressor(**params_with_early_stop)
            self.model_.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
        else:
            # Too few samples
            params_no_helper = dict(self.params)
            params_no_helper.pop('early_stop_frac', None)
            self.model_ = CatBoostRegressor(**params_no_helper)
            self.model_.fit(X, label, verbose=False)
        
        self.feature_importances_ = np.array(self.model_.get_feature_importance())
        
        return self
    
    def predict(self, X):
        """
        Predict survival time
        
        Returns negative predicted time as risk score
        (lower predicted time = higher risk)
        """
        X = np.asarray(X)
        pred_time = self.model_.predict(X)
        # Return negative time as risk score
        return -pred_time


# ============================================
# STACKING ENSEMBLE
# ============================================

class StackingEnsemble:
    """
    Weighted stacking ensemble for survival predictions
    Learns optimal weights via C-index optimization
    """
    
    def __init__(self, base_models):
        """
        Parameters:
        -----------
        base_models : dict
            Dictionary of {name: trained_model}
        """
        self.base_models = base_models
        self.weights = None
        self.model_names = list(base_models.keys())
        
    
    def fit(self, X, y_time, y_event, predictions=None):
        """
        Learn optimal weights.

        IMPORTANT (OOF-safe):
        - If `predictions` is provided, it must be an array of shape (n_samples, n_models)
          containing OUT-OF-FOLD (OOF) predictions for the training samples.
        - If `predictions` is None, predictions will be computed on X (in-sample).
          This is kept only for backwards compatibility and is NOT recommended for
          stacking weight learning.
        """
        # === MODIFIED (OOF) ===
        if predictions is None:
            # Backwards compatibility (in-sample). Prefer providing OOF predictions.
            predictions = self._get_base_predictions(X)
        else:
            predictions = np.asarray(predictions)
            if predictions.ndim != 2 or predictions.shape[1] != len(self.base_models):
                raise ValueError(
                    f"`predictions` must have shape (n_samples, {len(self.base_models)}). "
                    f"Got {predictions.shape}."
                )

        # Optimize weights to maximize C-index
        def objective(weights):
            # Normalize weights to sum to 1
            weights = np.abs(weights)
            weights = weights / np.sum(weights)

            # Weighted combination
            combined_pred = np.average(predictions, axis=1, weights=weights)

            try:
                c_index = concordance_index_censored(
                    y_event.astype(bool),
                    y_time,
                    combined_pred
                )[0]
                return -c_index
            except Exception:
                return 0.0

        # Initialize with equal weights
        initial_weights = np.ones(len(self.base_models)) / len(self.base_models)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='Nelder-Mead',
            options={'maxiter': 500, 'disp': False}
        )

        # Store normalized weights
        self.weights = np.abs(result.x)
        self.weights = self.weights / np.sum(self.weights)

        return self

    # === ADDED (OOF) ===
    def fit_from_oof(self, oof_predictions, y_time, y_event):
        """
        Convenience wrapper: fit stacking weights from OOF predictions.
        """
        return self.fit(X=None, y_time=y_time, y_event=y_event, predictions=oof_predictions)

    def predict(self, X):
        """Predict using weighted combination"""
        if self.weights is None:
            raise ValueError("Must fit before predict!")
        
        predictions = self._get_base_predictions(X)
        return np.average(predictions, axis=1, weights=self.weights)
    
    def _get_base_predictions(self, X):
        """Get predictions from all base models"""
        predictions = []
        for name in self.model_names:
            model = self.base_models[name]
            pred = model.predict(X)
            predictions.append(pred)
        return np.column_stack(predictions)
    
    def get_weights(self):
        """Get learned weights as dict"""
        if self.weights is None:
            return None
        return {name: weight for name, weight in zip(self.model_names, self.weights)}


class LightGBMSurvivalWrapper:
    """LightGBM wrapper using Poisson regression for discrete-time survival"""
    
    def __init__(self, params, seed):
        self.params = {
            'objective': 'regression',
            'random_state': seed,
            'verbose': -1,
            **params
        }
        self.model_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y_time, y_event=None):
        X = np.asarray(X)
        y_time = np.asarray(y_time)
        
        # Simple regression on time with early stopping
        # NOTE: early_stop_frac is treated as the INTERNAL validation fraction used only for early stopping.
        early_stop_frac = float(self.params.pop('early_stop_frac', 0.2))  # === CHANGED ===
        early_stop_rounds = int(self.params.pop('early_stopping_rounds', 50))  # === CHANGED ===

        self.model_ = LGBMRegressor(**{
            **self.params
        })

        # Split for early stopping validation (within the current TRAIN partition)
        from sklearn.model_selection import train_test_split
        use_early_stop = (len(X) > 100 and 0.0 < early_stop_frac < 1.0)
        if use_early_stop:
            split_rs = self.params.get('random_state', 42)  # === CHANGED ===
            strat = None
            if y_event is not None:
                y_event = np.asarray(y_event)
                if len(np.unique(y_event)) >= 2:
                    strat = y_event
                else:
                    use_early_stop = False
            if use_early_stop:
                try:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X, y_time,
                        test_size=early_stop_frac,
                        stratify=strat,
                        random_state=split_rs
                    )
                    self.model_.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)]
                    )
                    self._early_stopping_used = True
                except Exception:
                    self.model_.fit(X, y_time)
                    self._early_stopping_used = False
            else:
                self.model_.fit(X, y_time)
                self._early_stopping_used = False
        else:
            self.model_.fit(X, y_time)
            self._early_stopping_used = False
        
        self.feature_importances_ = np.array(self.model_.feature_importances_)
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        # Return negative time as risk score (lower time = higher risk)
        return -self.model_.predict(X)


# ============================================

# ============================================
# scikit-survival WRAPPERS (additional algorithms)
# ============================================

def _make_sksurv_y(y_time, y_event):
    """Create scikit-survival structured array y."""
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event).astype(bool)
    if np.any(y_time <= 0):
        print("WARNING: non-positive y_time found, clipping to 1e-6")
    y_time = np.where(y_time <= 0, 1e-6, y_time)
    return np.array(list(zip(y_event, y_time)), dtype=[('event', '?'), ('time', '<f8')])

def _safe_feature_importances(model):
    """Safely get feature_importances_ without triggering NotImplementedError."""
    try:
        return getattr(model, "feature_importances_", None)
    except NotImplementedError:
        return None
    except Exception:
        # Some versions may raise other errors; don't let this crash training
        return None

class CoxPHSkSurvWrapper:
    """CoxPHSurvivalAnalysis wrapper (scikit-survival)."""
    def __init__(self, params, seed):
        self.params = dict(params)
        self.seed = seed
        self.model_ = None

    def fit(self, X, y_time, y_event):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        self.model_ = CoxPHSurvivalAnalysis(**self.params)
        self.model_.fit(X, _make_sksurv_y(y_time, y_event))
        return self

    def predict(self, X):
        # higher => higher risk
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()

class CoxnetSkSurvWrapper:
    """CoxnetSurvivalAnalysis wrapper with alpha-index selection."""
    def __init__(self, params, seed):
        self.params = dict(params)
        self.seed = seed
        self.model_ = None
        self.alpha_index_ = self.params.pop("alpha_index", None)  # selected during tuning

    def fit(self, X, y_time, y_event):
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        # Some sksurv versions require l1_ratio in (0, 1]; treat 0 as "almost ridge".
        if "l1_ratio" in self.params and self.params["l1_ratio"] is not None:
            try:
                l1 = float(self.params["l1_ratio"])
            except Exception:
                l1 = self.params["l1_ratio"]
            if isinstance(l1, (int, float)) and l1 <= 0.0:
                l1 = 1e-6
            self.params["l1_ratio"] = float(l1)

        # --- Stability guards (do NOT change features; only avoid numerically dangerous settings) ---
        if "alpha_min_ratio" in self.params and self.params["alpha_min_ratio"] is not None:
            try:
                amr = float(self.params["alpha_min_ratio"])
                if amr < 1e-2:
                    amr = 1e-2
                self.params["alpha_min_ratio"] = amr
            except Exception:
                pass

        # Ensure numeric dtype (no scaling)
        X = np.asarray(X, dtype=np.float64)

        # Coxnet fits a PATH of penalization strengths (alphas)
        self.model_ = CoxnetSurvivalAnalysis(**self.params)
        self.model_.fit(X, _make_sksurv_y(y_time, y_event))
        # If no alpha_index provided, default to the last (strongest regularization)
        if self.alpha_index_ is None:
            self.alpha_index_ = -1
        return self

    def predict(self, X):
        pred = self.model_.predict(np.asarray(X))
        pred = np.asarray(pred)
        # Coxnet returns (n_samples, n_alphas). Pick the tuned alpha_index_.
        if pred.ndim == 2:
            return pred[:, self.alpha_index_].ravel()
        return pred.ravel()

class RSFSkSurvWrapper:
    """RandomSurvivalForest wrapper."""
    def __init__(self, params, seed):
        self.params = dict(params)
        self.seed = seed
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X, y_time, y_event):
        from sksurv.ensemble import RandomSurvivalForest
        self.model_ = RandomSurvivalForest(random_state=self.seed, **self.params)
        self.model_.fit(X, _make_sksurv_y(y_time, y_event))
        self.feature_importances_ = _safe_feature_importances(self.model_)
        return self

    def predict(self, X):
        # .predict returns risk scores for RSF in sksurv
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()

class ExtraSurvivalTreesWrapper:
    """ExtraSurvivalTrees wrapper."""
    def __init__(self, params, seed):
        self.params = dict(params)
        self.seed = seed
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X, y_time, y_event):
        from sksurv.ensemble import ExtraSurvivalTrees
        self.model_ = ExtraSurvivalTrees(random_state=self.seed, **self.params)
        self.model_.fit(X, _make_sksurv_y(y_time, y_event))
        self.feature_importances_ = _safe_feature_importances(self.model_)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()

class GBMSurvivalSkSurvWrapper:
    """GradientBoostingSurvivalAnalysis wrapper."""
    def __init__(self, params, seed):
        self.params = dict(params)
        self.seed = seed
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X, y_time, y_event):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        self.model_ = GradientBoostingSurvivalAnalysis(random_state=self.seed, **self.params)
        self.model_.fit(X, _make_sksurv_y(y_time, y_event))
        self.feature_importances_ = _safe_feature_importances(self.model_)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()

class SurvivalTreeSkSurvWrapper:
    """SurvivalTree wrapper."""
    def __init__(self, params, seed):
        self.params = dict(params)
        self.seed = seed
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X, y_time, y_event):
        from sksurv.tree import SurvivalTree
        self.model_ = SurvivalTree(random_state=self.seed, **self.params)
        self.model_.fit(X, _make_sksurv_y(y_time, y_event))
        self.feature_importances_ = _safe_feature_importances(self.model_)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()

class FastSurvivalSVMWrapper:
    """FastSurvivalSVM wrapper."""
    def __init__(self, params, seed):
        self.params = dict(params)
        self.seed = seed
        self.model_ = None

    def fit(self, X, y_time, y_event):
        from sksurv.svm import FastSurvivalSVM
        self.model_ = FastSurvivalSVM(**self.params)

        # ✅ FIX: FastSurvivalSVM requires inexact dtype (float); binary int matrices crash
        X = np.asarray(X, dtype=np.float64)

        self.model_.fit(X, _make_sksurv_y(y_time, y_event))
        return self

    def predict(self, X):
        # ✅ keep dtype consistent at inference too
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self.model_.predict(X)).ravel()


# MAIN PIPELINE CLASS
# ============================================

class SurvivalPipeline:
    """
    Complete survival analysis pipeline with holdout validation
    
    Parameters:
    -----------
    approach : str
        'nested_cv_with_holdout' or 'three_way_with_holdout'
    algorithms : list
        List of algorithm names
    ensemble_methods : list, optional
        Ensemble methods to try
    tuning_method : str
        'optuna' or 'grid'
    n_seeds : int
        Number of random seeds
    n_tuning_iterations : int
        Number of Optuna trials
    outer_cv : int
        Outer CV folds (for nested CV)
    inner_cv : int
        Inner CV folds (for nested CV)
    test_size : float
        Test size for three-way split
    val_size : float
        Validation size for three-way split
    verbose : bool
        Print progress
    """
    
    def __init__(self, approach='nested_cv_with_holdout', algorithms=None,
                 ensemble_methods=None, tuning_method='optuna', n_seeds=3, random_state=42, param_agg_method='median_mode',
                 n_tuning_iterations=50, outer_cv=5, inner_cv=3,
                 test_size=0.2, val_size=0.2, auc_time_points=None, verbose=True, n_jobs=1):
        
        self.approach = approach
        self.algorithms = algorithms or ['xgboost', 'catboost', 'lightgbm']
        self.ensemble_methods = ensemble_methods or []
        self.tuning_method = tuning_method
        self.n_seeds = n_seeds
        self.n_tuning_iterations = n_tuning_iterations
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.test_size = test_size
        self.val_size = val_size
        self.auc_time_points = auc_time_points or [1, 2, 3]
        self.verbose = verbose
        self.n_jobs = int(n_jobs) if n_jobs is not None else 1
        self.random_state = random_state
        self.param_agg_method = param_agg_method  # 'median_mode' or 'vote'
        self.train_indices_ = None
        self.test_indices_ = None
        self.feature_names_ = None
        self.final_output = {}
        
        if tuning_method == 'optuna':
            try:
                import optuna
            except ImportError:
                raise ImportError("Optuna not installed. Install: pip install optuna")
    
    def fit(self, X, y_time, y_event, train_indices=None, feature_names=None):
        """
        Fit the survival pipeline
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y_time : array-like
            Survival times
        y_event : array-like
            Event indicators (1=event, 0=censored)
        train_indices : array-like, optional
            Indices for training (rest will be holdout test)
        feature_names : list, optional
            Feature names
        """
        X = np.asarray(X)
        y_time = np.asarray(y_time)
        y_event = np.asarray(y_event)
        
        self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Determine train/test split
        if train_indices is not None:
            self.train_indices_ = np.asarray(train_indices)
            all_indices = set(range(len(X)))
            self.test_indices_ = np.array(list(all_indices - set(self.train_indices_)))
        else:
            self.train_indices_ = np.arange(len(X))
            self.test_indices_ = np.array([])
        
        if self.verbose:
            self._print_header(X, y_event, train_indices)
        
        # Model selection on train_indices
        if self.approach == 'nested_cv_with_holdout':
            cv_results = self._fit_nested_cv_with_holdout(X, y_time, y_event)
        else:
            cv_results = self._fit_three_way_with_holdout(X, y_time, y_event)
        
        # Final retrain and evaluation
        if len(self.test_indices_) > 0:
            if self.verbose:
                print(f"\n{'='*80}")
                print("FINAL EVALUATION - ALL MODELS")
                print(f"{'='*80}")
            
            # Evaluate ALL models on train and test
            all_model_performance = self._evaluate_all_models_final(X, y_time, y_event, cv_results)
            
            # Keep best model info for compatibility
            best_algo = cv_results['best_algorithm']
            final_results = {
                'best_algorithm': best_algo,
                'best_params': all_model_performance[best_algo]['best_params'],
                'final_model': all_model_performance[best_algo]['model'],
                'train_performance': {
                    'c_index': all_model_performance[best_algo]['train_c_index']},
                'test_performance': {
                    'c_index': all_model_performance[best_algo]['test_c_index']},
                'all_models': all_model_performance  # NEW: All models performance
            }
        else:
            final_results = None
            all_model_performance = {}
        
        # Compile final output
        self.final_output = {
            'approach': self.approach,
            'cv_results': cv_results,
            'final_holdout': final_results,
            'train_indices': self.train_indices_,
            'test_indices': self.test_indices_,
            'feature_names': self.feature_names_,
            'n_features': len(self.feature_names_)
        }
        
        return self
    
    def _print_header(self, X, y_event, train_indices):
        """Print pipeline configuration"""
        print("="*80)
        print("SURVIVAL ANALYSIS PIPELINE v5.0 (COMPLETE)")
        print("="*80)
        print(f"Approach:          {self.approach}")
        print(f"Algorithms:        {', '.join(self.algorithms)}")
        print(f"Tuning method:     {self.tuning_method}")
        print(f"N seeds:           {self.n_seeds}")
        print(f"Tuning iterations: {self.n_tuning_iterations}")
        
        if train_indices is not None:
            print(f"\nData Split (Pre-defined):")
            print(f"  Train indices:     {len(train_indices)} samples ({y_event[train_indices].mean():.1%} events)")
            print(f"  Holdout indices:   {len(self.test_indices_)} samples ({y_event[self.test_indices_].mean():.1%} events)")
        else:
            print(f"\nDataset size:      {len(X)} samples × {X.shape[1]} features")
            print(f"Event rate:        {y_event.mean():.1%}")
        print("="*80)
    
    def _generate_seeds(self):
        """Generate random seeds"""
        base_seed = self.random_state
        return [base_seed + i * 111 for i in range(self.n_seeds)]

    # ============================================
    # PARAMETER AGGREGATION ACROSS SEEDS/FOLDS
    # ============================================
    def _majority_vote_params(self, params_list, float_round=6):
        """Majority vote of hyperparameters across CV seeds/folds.

        Why:
        - In nested CV you tune params many times (seed × outer fold).
        - For the FINAL refit on full train, we want to *carry* the tuning results
          instead of re-tuning with a new random split/seed.

        Strategy:
        - Vote each parameter key independently.
        - Floats are rounded to `float_round` before voting (Optuna outputs are continuous).
        - If a key is missing in some folds, we vote among the folds where it exists.
        """
        if not params_list:
            return {}

        def norm(v):
            # normalize numpy scalars
            if isinstance(v, (np.generic,)):
                v = v.item()
            if isinstance(v, float):
                return round(v, float_round)
            return v

        keys = sorted(set().union(*[set(d.keys()) for d in params_list if isinstance(d, dict)]))
        voted = {}
        for k in keys:
            vals = [norm(d.get(k)) for d in params_list if isinstance(d, dict) and k in d]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            c = Counter(vals)
            best_val = c.most_common(1)[0][0]
            # cast back
            if isinstance(best_val, (int, float, str, bool)):
                voted[k] = best_val
            else:
                voted[k] = best_val
        return voted
    
    def _aggregate_params_median_mode(self, params_list):
        """Aggregate hyperparameters across CV folds/seeds WITHOUT using outer-test.

        Rule (Option A agreed):
        - float   -> median
        - int/bool/str -> mode (most frequent)
        - other (tuple/list/dict) -> mode by exact match (stringified for hashing)

        Notes:
        - Works well with Optuna continuous outputs (median is stable).
        - Deterministic tie-breaks:
            * numeric: choose the smaller value
            * strings: lexicographically smallest
        """
        if not params_list:
            return {}

        # Collect all keys
        keys = sorted(set().union(*[set(d.keys()) for d in params_list if isinstance(d, dict)]))
        agg = {}

        def _to_py(v):
            # normalize numpy scalars
            try:
                import numpy as _np
                if isinstance(v, _np.generic):
                    return v.item()
            except Exception:
                pass
            return v

        def _mode(vals):
            from collections import Counter
            # hashable normalization
            normed = []
            for v in vals:
                v = _to_py(v)
                # convert unhashable to repr
                try:
                    hash(v)
                    normed.append(v)
                except Exception:
                    normed.append(repr(v))
            c = Counter(normed)
            top = c.most_common()
            if not top:
                return None
            max_ct = top[0][1]
            candidates = [v for v, ct in top if ct == max_ct]
            # deterministic tie-break
            # numeric
            if all(isinstance(v, (int, float)) for v in candidates):
                return min(candidates)
            # string
            if all(isinstance(v, str) for v in candidates):
                return sorted(candidates)[0]
            # mixed / repr fallback
            return sorted([str(v) for v in candidates])[0]

        for k in keys:
            vals = [d.get(k) for d in params_list if isinstance(d, dict) and k in d]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            # decide type by first non-null (after normalization)
            v0 = _to_py(vals[0])
            # bool is subclass of int -> treat separately as categorical
            if isinstance(v0, bool) or isinstance(v0, str) or isinstance(v0, int):
                agg[k] = _mode(vals)
            elif isinstance(v0, float):
                arr = [float(_to_py(v)) for v in vals]
                agg[k] = float(np.median(arr))
            else:
                # fallback to mode by exact match
                agg[k] = _mode(vals)

        return agg

    def _is_coxnet_algo(self, algo_name: str) -> bool:
        """Return True if algo_name corresponds to a Coxnet model variant.

        Supports:
        - coxnet (generic, tunes l1_ratio)
        - coxnet_lasso (l1_ratio=1)
        - coxnet_ridge (l1_ratio=0)
        - coxnet_enet (tunes l1_ratio grid)
        - coxnet_enet_<X> where X is a float in (0,1), e.g., coxnet_enet_0.3
        """
        if algo_name in {"coxnet", "coxnet_lasso", "coxnet_ridge", "coxnet_enet"}:
            return True
        if algo_name.startswith("coxnet_enet_"):
            # allow e.g. coxnet_enet_0.1 ... coxnet_enet_0.9
            try:
                float(algo_name.split("coxnet_enet_")[1])
                return True
            except Exception:
                return False
        return False
        return False


    def _get_model(self, algo_name, params, seed):
        """Get model instance"""
        run_params = params.copy()
        
        if 'xgboost' in algo_name:
            run_params['n_jobs'] = self.n_jobs  
            run_params['tree_method'] = 'hist' 
            
        elif 'lightgbm' in algo_name:
            run_params['n_jobs'] = self.n_jobs
            run_params['verbose'] = -1
            
        elif 'catboost' in algo_name:
            run_params['thread_count'] = self.n_jobs
            run_params['allow_writing_files'] = False
        
        
        if algo_name == 'xgboost':
            return XGBoostSurvivalWrapper(run_params, seed)
        elif algo_name == 'xgboost_aft_normal':
            return XGBoostAFTWrapper(run_params, seed, aft_loss_distribution='normal')
        elif algo_name == 'xgboost_aft_logistic':
            return XGBoostAFTWrapper(run_params, seed, aft_loss_distribution='logistic')
        elif algo_name == 'xgboost_aft_extreme':
            return XGBoostAFTWrapper(run_params, seed, aft_loss_distribution='extreme')
        elif algo_name == 'catboost':
            return CatBoostSurvivalWrapper(run_params, seed)
        elif algo_name == 'catboost_aft_normal':
            return CatBoostAFTWrapper(run_params, seed, aft_loss_distribution='normal')
        elif algo_name == 'catboost_aft_logistic':
            return CatBoostAFTWrapper(run_params, seed, aft_loss_distribution='logistic')
        elif algo_name == 'catboost_aft_extreme':
            return CatBoostAFTWrapper(run_params, seed, aft_loss_distribution='extreme')
        elif algo_name == 'lightgbm':
            return LightGBMSurvivalWrapper(run_params, seed)

        # --- scikit-survival models ---
        elif algo_name == 'coxph':
            return CoxPHSkSurvWrapper(run_params, seed)
        elif self._is_coxnet_algo(algo_name):
            # we use CoxnetSurvivalAnalysis for all; l1_ratio controls ridge/lasso/enet
            return CoxnetSkSurvWrapper(run_params, seed)
        elif algo_name == 'rsf':
            run_params.setdefault('n_jobs', self.n_jobs)
            return RSFSkSurvWrapper(run_params, seed)
        elif algo_name == 'extra_survival_trees':
            run_params.setdefault('n_jobs', self.n_jobs)
            return ExtraSurvivalTreesWrapper(run_params, seed)
        elif algo_name == 'gbm_sksurv':
            return GBMSurvivalSkSurvWrapper(run_params, seed)
        elif algo_name == 'survival_tree':
            return SurvivalTreeSkSurvWrapper(run_params, seed)
        elif algo_name == 'survival_svm':
            return FastSurvivalSVMWrapper(run_params, seed)

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    
    
    def _get_default_params(self, algo_name):
        """Get default hyperparameter search space.

        Returned values can be:
        - tuple(low, high): continuous/int range (used by Optuna; for grid we auto-expand)
        - list: explicit grid / categorical choices
        """
        # Gradient boosting families (your existing ones)
        if 'xgboost' in algo_name:
            return {
                'eta': (0.01, 0.3),
                'max_depth': (3, 10),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'n_estimators': (50, 400),
                # helper (used by our wrapper; ensures early stopping uses a proper internal validation split)
                'early_stop_frac': [0.2]
            }
        elif 'catboost' in algo_name:
            return {
                'learning_rate': (0.01, 0.3),
                'depth': (3, 10),
                'iterations': (100, 800),
                # helper (used by our wrapper, ignored by CatBoost itself)
                'early_stop_frac': [0.2]
            }
        elif algo_name == 'lightgbm':
            return {
                'learning_rate': (0.01, 0.3),
                'num_leaves': (15, 255),
                'n_estimators': (50, 400),
                'min_child_samples': (5, 50),
                # helper (used by our wrapper)
                'early_stop_frac': [0.2]
            }

        # ----------------------------
        # scikit-survival algorithms
        # ----------------------------
        elif algo_name == 'coxph':
            # CoxPHSurvivalAnalysis has few knobs; we keep this mostly "as-is"
            # You can later add "alpha" (ridge) via Coxnet if needed.
            return {}

        elif self._is_coxnet_algo(algo_name):
            # In your language: alpha = l1_ratio (mixing), in [0..1]
            # We tune l1_ratio on a coarse grid by default (0,0.1,...,1).
            base = {
                'l1_ratio': [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)],
                'alpha_min_ratio': [1e-2, 1e-1],
                'n_alphas': [100],
                'max_iter': [100000],
                'tol': [1e-7]
            }
            # Convenience defaults
            if algo_name == 'coxnet_lasso':
                base['l1_ratio'] = [1.0]
            elif algo_name == 'coxnet_ridge':
                base['l1_ratio'] = [0.0]
            elif algo_name.startswith('coxnet_enet_'):
                # Fixed elastic-net mixing for this run
                try:
                    l1 = float(algo_name.split('coxnet_enet_')[1])
                    base['l1_ratio'] = [l1]
                except Exception:
                    pass
            return base

        elif algo_name == 'rsf':
            return {
                'n_estimators': (200, 600),
                'min_samples_split': (20, 60),
                'min_samples_leaf': (15, 30),
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [None, 3, 5, 8, 12]
            }

        elif algo_name == 'extra_survival_trees':
            return {
                'n_estimators': (300, 800),
                'min_samples_split': (20, 60),
                'min_samples_leaf': (15, 30),
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [None, 3, 5, 8, 12]
            }

        elif algo_name == 'gbm_sksurv':
            return {
                'n_estimators': (200, 600),
                'learning_rate': (0.005, 0.2),
                'max_depth': (1, 5),
                'subsample': (0.5, 1.0)
            }

        elif algo_name == 'survival_tree':
            return {
                'max_depth': [2, 3, 4, 5, 7, 10, 15],
                'min_samples_split': (20, 60),
                'min_samples_leaf': (15, 30),
                'max_features': ['sqrt', 'log2', None]
            }

        elif algo_name == 'survival_svm':
            # FastSurvivalSVM: optimization method & regularization
            return {
                'alpha': (1e-6, 1e-1),
                'rank_ratio': (0.0, 1.0),
                'max_iter': [2000]
            }

        else:
            return {}


    def _tune_with_optuna(self, algo_name, X_train, y_train, X_val, y_val, seed):
        """Hyperparameter tuning with Optuna (single train/val split)."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        param_space = self._get_default_params(algo_name)

        log_params = {"eta", "learning_rate", "alpha", "C", "reg_alpha", "reg_lambda"}

        def objective(trial):
            params = {}
            for param_name, spec in param_space.items():
                # categorical/grid-style list (explicit)
                if isinstance(spec, list):
                    params[param_name] = trial.suggest_categorical(param_name, spec)
                    continue

                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    use_log = (param_name in log_params) and (float(low) > 0)
                    params[param_name] = trial.suggest_float(param_name, float(low), float(high), log=use_log)

            model = self._get_model(algo_name, params, seed)
            model.fit(X_train, y_train['time'], y_train['event'])

            # Coxnet: choose best alpha along its internal path using validation C-index
            if self._is_coxnet_algo(algo_name):
                raw = np.asarray(model.model_.predict(np.asarray(X_val)))
                if raw.ndim == 1:
                    best_c = concordance_index_censored(y_val['event'].astype(bool), y_val['time'], raw)[0]
                    trial.set_user_attr("alpha_index", None)
                    return float(best_c)

                best_c = -np.inf
                best_alpha_idx = 0
                for j in range(raw.shape[1]):
                    c = concordance_index_censored(y_val['event'].astype(bool), y_val['time'], raw[:, j])[0]
                    if c > best_c:
                        best_c = c
                        best_alpha_idx = j
                trial.set_user_attr("alpha_index", int(best_alpha_idx))
                return float(best_c)

            pred = model.predict(X_val)
            c_index = concordance_index_censored(y_val['event'].astype(bool), y_val['time'], pred)[0]
            return float(c_index)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=self.n_tuning_iterations, show_progress_bar=False)

        best_params = dict(study.best_params)
        if self._is_coxnet_algo(algo_name):
            alpha_idx = study.best_trial.user_attrs.get("alpha_index", None)
            if alpha_idx is not None:
                best_params["alpha_index"] = alpha_idx

        return best_params, float(study.best_value)

    
    def _tune_hyperparams(self, algo_name, X_train, y_train, X_val, y_val, seed):
        """Dispatch hyperparameter tuning based on self.tuning_method."""
        if self.tuning_method == "grid":
            return self._tune_with_grid(algo_name, X_train, y_train, X_val, y_val, seed)
        # default
        return self._tune_with_optuna(algo_name, X_train, y_train, X_val, y_val, seed)

    def _tune_with_grid(self, algo_name, X_train, y_train, X_val, y_val, seed):
        """Simple grid search over a small, explicit grid (single train/val split)."""
        from sklearn.model_selection import ParameterGrid

        param_space = self._get_default_params(algo_name)

        # Expand tuple ranges into a small grid (5 points)
        grid = {}
        for k, spec in param_space.items():
            if isinstance(spec, list):
                grid[k] = spec
            else:
                low, high = spec
                # integers
                if isinstance(low, int) and isinstance(high, int):
                    if high <= low:
                        grid[k] = [low]
                    else:
                        # 5 points inclusive, rounded to int
                        vals = np.linspace(low, high, num=5)
                        grid[k] = sorted(set(int(round(v)) for v in vals))
                else:
                    if float(high) <= float(low):
                        grid[k] = [float(low)]
                    else:
                        vals = np.linspace(float(low), float(high), num=5)
                        grid[k] = [float(v) for v in vals]

        best_params = {}
        best_score = -np.inf

        for params in ParameterGrid(grid):
            model = self._get_model(algo_name, params, seed)
            model.fit(X_train, y_train['time'], y_train['event'])

            # Coxnet: choose best alpha along internal path
            if self._is_coxnet_algo(algo_name):
                raw = np.asarray(model.model_.predict(np.asarray(X_val)))
                if raw.ndim == 1:
                    score = concordance_index_censored(y_val['event'].astype(bool), y_val['time'], raw)[0]
                    alpha_idx = None
                else:
                    score = -np.inf
                    alpha_idx = 0
                    for j in range(raw.shape[1]):
                        c = concordance_index_censored(y_val['event'].astype(bool), y_val['time'], raw[:, j])[0]
                        if c > score:
                            score = c
                            alpha_idx = j
                if score > best_score:
                    best_score = float(score)
                    best_params = dict(params)
                    if alpha_idx is not None:
                        best_params["alpha_index"] = int(alpha_idx)
                continue

            pred = model.predict(X_val)
            score = concordance_index_censored(y_val['event'].astype(bool), y_val['time'], pred)[0]

            if score > best_score:
                best_score = float(score)
                best_params = dict(params)

        return best_params, float(best_score)

    def _evaluate_model(self, model, X, y_time, y_event, dataset_name=''):
        """Comprehensive model evaluation"""
        pred = model.predict(X)
        
        event_bool = y_event.astype(bool)
        time_float = y_time.astype(float)
        
        try:
            # concordance_index_censored 是 C 语言底层优化，极快
            c_index_value = concordance_index_censored(event_bool, time_float, pred)[0]
        except Exception:
            c_index_value = 0.5
        
        c_index = c_index_value        # Time-dependent AUC
        try:
            y_structured = np.array(
                [(bool(e), t) for e, t in zip(y_event, y_time)],
                dtype=[('event', bool), ('time', float)]
            )
            auc_scores = []
            for t in self.auc_time_points:
                if t < y_time.max():
                    auc_t = cumulative_dynamic_auc(
                        y_structured, y_structured, pred, t
                    )[0]
                    auc_scores.append(auc_t[0] if len(auc_t) > 0 else np.nan)
                else:
                    auc_scores.append(np.nan)
        except:
            auc_scores = [np.nan] * len(self.auc_time_points)
        
        return {
            'c_index': c_index,
            'auc_scores': auc_scores
        }
    
    def _get_auc_times(self, y_time):
        """Auto-detect AUC time points based on time scale"""
        max_time = np.max(y_time)
        
        if max_time < 100:  # Years
            times = [1, 2, 3]
        elif max_time < 1000:  # Months
            times = [12, 24, 36]
        else:  # Days
            times = [365, 730, 1095]
        
        times = [t for t in times if t < max_time * 0.9]
        return times if times else [max_time * 0.5]
    
    def _fit_nested_cv_with_holdout(self, X, y_time, y_event):
        """Nested CV on train_indices"""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"NESTED CV ON TRAIN INDICES")
            print(f"{'='*80}")
        
        X_train = X[self.train_indices_]
        y_time_train = y_time[self.train_indices_]
        y_event_train = y_event[self.train_indices_]
        
        all_results = defaultdict(list)

        # --- NEW: track chosen hyperparameters per seed (3-way split) ---
        selected_params = defaultdict(list)
        selected_params_meta = defaultdict(list)

        # --- NEW: track chosen hyperparameters per (seed, outer fold) ---
        # We will later majority-vote these to define FINAL params used for SHAP + final model.
        selected_params = defaultdict(list)
        selected_params_meta = defaultdict(list)

        # === ADDED (OOF) ===
        # Accumulate out-of-fold predictions on TRAIN indices (X_train) across folds and seeds
        n_train = X_train.shape[0]
        oof_sum = {algo: np.zeros(n_train, dtype=float) for algo in self.algorithms}
        oof_cnt = {algo: np.zeros(n_train, dtype=int)   for algo in self.algorithms}
        
        for seed_idx, seed in enumerate(self._generate_seeds()):
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"SEED {seed_idx + 1}/{self.n_seeds} (seed={seed})")
                print(f"{'─'*80}")
            
            skf_outer = StratifiedKFold(n_splits=self.outer_cv, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf_outer.split(X_train, y_event_train)):
                if self.verbose:
                    print(f"\nOuter Fold {fold_idx + 1}/{self.outer_cv}")
                
                X_tr = X_train[train_idx]
                X_te = X_train[test_idx]
                y_tr = {'time': y_time_train[train_idx], 'event': y_event_train[train_idx]}
                y_te = {'time': y_time_train[test_idx], 'event': y_event_train[test_idx]}
                
                skf_inner = StratifiedKFold(n_splits=self.inner_cv, shuffle=True, random_state=seed)
                
                best_params_fold = {}
                best_inner_score = {a: -np.inf for a in self.algorithms}
                for algo_name in self.algorithms:
                    if self.verbose:
                        print(f"  Tuning {algo_name}...", end=' ')
                    
                    for inner_train_idx, inner_val_idx in skf_inner.split(X_tr, y_tr['event']):
                        X_inner_tr = X_tr[inner_train_idx]
                        X_inner_val = X_tr[inner_val_idx]
                        y_inner_tr = {
                            'time': y_tr['time'][inner_train_idx],
                            'event': y_tr['event'][inner_train_idx]
                        }
                        y_inner_val = {
                            'time': y_tr['time'][inner_val_idx],
                            'event': y_tr['event'][inner_val_idx]
                        }
                        try:
                            params, score = self._tune_hyperparams(
                                algo_name, X_inner_tr, y_inner_tr,
                                X_inner_val, y_inner_val, seed
                            )
                        except Exception as e:
                            if self.verbose:
                                print(f"    FAILED ({type(e).__name__}: {e})")
                            params, score = None, float("-inf")
                        if score > best_inner_score[algo_name]:
                            best_inner_score[algo_name] = score
                            best_params_fold[algo_name] = params
                    
                    if self.verbose:
                        print(f"Done")

                # --- NEW: persist the best params selected by the INNER CV for this (seed, outer fold) ---
                for algo_name in self.algorithms:
                    if algo_name in best_params_fold:
                        selected_params[algo_name].append(best_params_fold[algo_name])
                        selected_params_meta[algo_name].append({
                            'seed': int(seed),
                            'outer_fold': int(fold_idx),
                            'inner_best_cindex': float(best_inner_score.get(algo_name, np.nan)),
                            'params': best_params_fold.get(algo_name, None)
                        })
                
                for algo_name in self.algorithms:
                    if best_params_fold.get(algo_name) is None:
                        if self.verbose:
                            print(f"Skipping {algo_name} (no params due to failure).")
                        continue
                    model = self._get_model(algo_name, best_params_fold[algo_name], seed)
                    try:
                        model.fit(X_tr, y_tr['time'], y_tr['event'])
                    except Exception as e:
                        if self.verbose:
                            print(f"[SKIP-FIT] {algo_name} failed: {e}")
                        continue

                    
                    perf = self._evaluate_model(model, X_te, y_te['time'], y_te['event'])

                    # attach outer-fold performance to the most recent meta entry
                    if selected_params_meta.get(algo_name):
                        selected_params_meta[algo_name][-1]['outer_cindex'] = float(perf['c_index'])

                    # === ADDED (OOF) ===
                    # OOF prediction for this outer test fold (within TRAIN indices)
                    try:
                        pred_te = model.predict(X_te)
                        oof_sum[algo_name][test_idx] += pred_te
                        oof_cnt[algo_name][test_idx] += 1
                    except Exception:
                        # If a model cannot produce predictions, skip OOF for that fold
                        pass

                    all_results[algo_name].append(perf['c_index'])
                    
                    if self.verbose:
                        print(f"  {algo_name}: C-Index = {perf['c_index']:.4f}")
        
        
        # === ADDED (OOF) ===
        oof_predictions = {}
        for algo_name in self.algorithms:
            with np.errstate(divide='ignore', invalid='ignore'):
                oof_predictions[algo_name] = np.where(
                    oof_cnt[algo_name] > 0,
                    oof_sum[algo_name] / np.maximum(oof_cnt[algo_name], 1),
                    np.nan
                )
        cv_summary = {}
        for algo_name in self.algorithms:
            scores = all_results[algo_name]
            cv_summary[algo_name] = {
                'mean_c_index': np.mean(scores),
                'std_c_index': np.std(scores),
                'scores': scores
            }
        
        best_algo = max(cv_summary.keys(), key=lambda k: cv_summary[k]['mean_c_index'])
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("NESTED CV RESULTS")
            print(f"{'='*80}")
            for algo_name in self.algorithms:
                mean = cv_summary[algo_name]['mean_c_index']
                std = cv_summary[algo_name]['std_c_index']
                marker = " ← BEST" if algo_name == best_algo else ""
                print(f"{algo_name:30s}: {mean:.4f} ± {std:.4f}{marker}")
        
        return {
            'cv_summary': cv_summary,
            'best_algorithm': best_algo,
            'best_mean_c_index': cv_summary[best_algo]['mean_c_index']
            ,
            # === ADDED (OOF) ===
            'oof_predictions': oof_predictions,
            'oof_counts': oof_cnt

            ,
            # --- NEW ---
            'selected_params': dict(selected_params),
            'selected_params_meta': dict(selected_params_meta)

        }
    
    def _fit_three_way_with_holdout(self, X, y_time, y_event):
        """Three-way split on train_indices"""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"THREE-WAY SPLIT ON TRAIN INDICES")
            print(f"{'='*80}")
        
        X_train_pool = X[self.train_indices_]
        y_time_train_pool = y_time[self.train_indices_]
        y_event_train_pool = y_event[self.train_indices_]
        
        all_results = defaultdict(list)

        # --- NEW: track chosen hyperparameters per seed (3-way split) ---
        selected_params = defaultdict(list)
        selected_params_meta = defaultdict(list)

        # === ADDED (OOF) ===
        # Accumulate out-of-sample predictions on TRAIN indices pool across repeated 3-way splits
        n_pool = X_train_pool.shape[0]
        oof_sum = {algo: np.zeros(n_pool, dtype=float) for algo in self.algorithms}
        oof_cnt = {algo: np.zeros(n_pool, dtype=int)   for algo in self.algorithms}
        
        for seed_idx, seed in enumerate(self._generate_seeds()):
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"SEED {seed_idx + 1}/{self.n_seeds} (seed={seed})")
                print(f"{'─'*80}")
            
                        # === MODIFIED (OOF) ===
            # Split using indices so we can write OOF predictions back to the correct rows
            idx_pool = np.arange(len(X_train_pool))
            idx_tr, idx_temp = train_test_split(
                idx_pool,
                test_size=self.test_size + self.val_size,
                stratify=y_event_train_pool,
                random_state=seed
            )
            X_tr = X_train_pool[idx_tr]
            y_tr_time = y_time_train_pool[idx_tr]
            y_tr_event = y_event_train_pool[idx_tr]

            X_temp = X_train_pool[idx_temp]
            y_temp_time = y_time_train_pool[idx_temp]
            y_temp_event = y_event_train_pool[idx_temp]
            
            val_ratio = self.val_size / (self.test_size + self.val_size)
                        # === MODIFIED (OOF) ===
            idx_val, idx_te = train_test_split(
                idx_temp,
                test_size=self.test_size / (self.test_size + self.val_size),
                stratify=y_temp_event,
                random_state=seed
            )
            X_val = X_train_pool[idx_val]
            y_val_time = y_time_train_pool[idx_val]
            y_val_event = y_event_train_pool[idx_val]

            X_te = X_train_pool[idx_te]
            y_te_time = y_time_train_pool[idx_te]
            y_te_event = y_event_train_pool[idx_te]
            
            y_tr = {'time': y_tr_time, 'event': y_tr_event}
            y_val = {'time': y_val_time, 'event': y_val_event}
            y_te = {'time': y_te_time, 'event': y_te_event}
            
            best_params_seed = {}
            for algo_name in self.algorithms:
                if self.verbose:
                    print(f"Tuning {algo_name}...", end=' ')
                
                try:
                    params, score = self._tune_hyperparams(algo_name, X_tr, y_tr, X_val, y_val, seed)
                except Exception as e:
                    if self.verbose:
                        print(f"FAILED ({type(e).__name__}: {e})")
                    params, score = None, float("-inf")

                best_params_seed[algo_name] = params

                # --- NEW: record params chosen on this seed ---
                selected_params[algo_name].append(params)
                selected_params_meta[algo_name].append({
                    'seed': int(seed),
                    'val_cindex': float(score),
                    'params': params
                })
                
                if self.verbose:
                    print(f"Done (val C-Index: {score:.4f})")
            
            for algo_name in self.algorithms:
                if best_params_seed.get(algo_name) is None:
                    if self.verbose:
                        print(f"Skipping {algo_name} (no params due to failure).")
                    continue
                model = self._get_model(algo_name, best_params_seed[algo_name], seed)
                try:
                    model.fit(X_tr, y_tr['time'], y_tr['event'])
                except Exception as e:
                    if self.verbose:
                        print(f"\n  [SKIP-FIT] {algo_name} failed: {e}")
                    continue
                
                perf = self._evaluate_model(model, X_te, y_te['time'], y_te['event'])

                # attach seed-level test performance to the most recent meta entry
                if selected_params_meta.get(algo_name):
                    selected_params_meta[algo_name][-1]['test_cindex'] = float(perf['c_index'])

                # === ADDED (OOF) ===
                # Out-of-sample predictions for this seed's 'te' split (within TRAIN pool)
                try:
                    pred_te = model.predict(X_te)
                    oof_sum[algo_name][idx_te] += pred_te
                    oof_cnt[algo_name][idx_te] += 1
                except Exception:
                    pass

                all_results[algo_name].append(perf['c_index'])
                
                if self.verbose:
                    print(f"  {algo_name}: Test C-Index = {perf['c_index']:.4f}")
        
        
        # === ADDED (OOF) ===
        oof_predictions = {}
        for algo_name in self.algorithms:
            with np.errstate(divide='ignore', invalid='ignore'):
                oof_predictions[algo_name] = np.where(
                    oof_cnt[algo_name] > 0,
                    oof_sum[algo_name] / np.maximum(oof_cnt[algo_name], 1),
                    np.nan
                )
        cv_summary = {}
        for algo_name in self.algorithms:
            scores = all_results[algo_name]
            cv_summary[algo_name] = {
                'mean_c_index': np.mean(scores),
                'std_c_index': np.std(scores),
                'scores': scores
            }
        
        best_algo = max(cv_summary.keys(), key=lambda k: cv_summary[k]['mean_c_index'])
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("THREE-WAY SPLIT RESULTS")
            print(f"{'='*80}")
            for algo_name in self.algorithms:
                mean = cv_summary[algo_name]['mean_c_index']
                std = cv_summary[algo_name]['std_c_index']
                marker = " ← BEST" if algo_name == best_algo else ""
                print(f"{algo_name:30s}: {mean:.4f} ± {std:.4f}{marker}")
        
        return {
            'cv_summary': cv_summary,
            'best_algorithm': best_algo,
            'best_mean_c_index': cv_summary[best_algo]['mean_c_index']
            ,
            # === ADDED (OOF) ===
            'oof_predictions': oof_predictions,
            'oof_counts': oof_cnt,

            # --- NEW: hyperparameters selected by INNER CV (for majority vote in final fit) ---
            'selected_params': dict(selected_params),
            'selected_params_meta': dict(selected_params_meta)

        }
    
    def _final_retrain_and_evaluate(self, X, y_time, y_event, cv_results):
        """
        CRITICAL: Final retrain on ALL train_indices and evaluate on BOTH train and test
        
        Steps:
        1. Get best algorithm from CV
        2. Tune hyperparameters one more time on full train set
        3. Retrain on ALL train_indices
        4. Evaluate on train_indices (train performance)
        5. Evaluate on test_indices (test performance)
        """
        best_algo = cv_results['best_algorithm']
        
        if self.verbose:
            print(f"\nBest algorithm from CV: {best_algo}")
            print(f"CV C-Index: {cv_results['best_mean_c_index']:.4f}")
            print(f"\nRetraining on full train_indices ({len(self.train_indices_)} samples)...")
        
        # Get full train data
        X_train_full = X[self.train_indices_]
        y_train_full_time = y_time[self.train_indices_]
        y_train_full_event = y_event[self.train_indices_]
        
        # =============================================================
        # === CHANGED: Use CV-selected hyperparameters (majority vote) ===
        # If CV did not yield any params for this algo, FALL BACK to a
        # single stratified holdout tuning split (reported as such).
        # =============================================================
        params_source = 'final_holdout_tuning'
        final_seed = self.random_state  # === CHANGED: consistent with pipeline ===
        best_params = None
        sp = cv_results.get('selected_params', {}) if isinstance(cv_results, dict) else {}
        if isinstance(sp, dict) and best_algo in sp and isinstance(sp[best_algo], list) and len(sp[best_algo]) > 0:
            if self.param_agg_method == 'vote':
                best_params = self._majority_vote_params(sp[best_algo])
                params_source = 'cv_majority_vote'
            elif self.param_agg_method == 'median_mode':
                best_params = self._aggregate_params_median_mode(sp[best_algo])
                params_source = 'cv_median_mode_aggregate'
            else:
                raise ValueError(f"Unknown param_agg_method={self.param_agg_method!r}. Use 'median_mode' or 'vote'.")

        if best_params is None:
            # Fallback: one tuning split on FULL train (stratified by event)
            X_tr, X_val, y_tr_time, y_val_time, y_tr_event, y_val_event = train_test_split(
                X_train_full, y_train_full_time, y_train_full_event,
                test_size=self.val_size,
                stratify=y_train_full_event,
                random_state=final_seed
            )
            y_tr = {'time': y_tr_time, 'event': y_tr_event}
            y_val = {'time': y_val_time, 'event': y_val_event}
            best_params, _ = self._tune_hyperparams(best_algo, X_tr, y_tr, X_val, y_val, seed=final_seed)

        if self.verbose:
            print(f"Best hyperparameters ({params_source}, seed={final_seed}): {best_params}")

        final_model = self._get_model(best_algo, best_params, seed=final_seed)  # === CHANGED ===
        final_model.fit(X_train_full, y_train_full_time, y_train_full_event)
        
        if self.verbose:
            print(f"Model trained successfully!")
        
        # CRITICAL: Evaluate on TRAIN data
        if self.verbose:
            print(f"\nEvaluating on TRAIN set ({len(self.train_indices_)} samples)...")
        
        train_perf = self._evaluate_model(
            final_model, X_train_full, y_train_full_time, y_train_full_event, dataset_name='train'
        )
        
        # CRITICAL: Evaluate on TEST data
        if self.verbose:
            print(f"Evaluating on TEST set ({len(self.test_indices_)} samples)...")
        
        X_test = X[self.test_indices_]
        y_test_time = y_time[self.test_indices_]
        y_test_event = y_event[self.test_indices_]
        
        test_perf = self._evaluate_model(
            final_model, X_test, y_test_time, y_test_event, dataset_name='test'
        )
        
        # Print comprehensive results
        if self.verbose:
            print(f"\n{'='*80}")
            print("FINAL PERFORMANCE SUMMARY")
            print(f"{'='*80}")
            print(f"Algorithm:     {best_algo}")
            print(f"Best params:   {best_params}")
            print(f"\nCV PERFORMANCE (from model selection):")
            print(f"  Mean C-Index:  {cv_results['best_mean_c_index']:.4f}")
            print(f"\nTRAIN SET ({len(self.train_indices_)} samples):")
            print(f"  C-Index:       {train_perf['c_index']:.4f}")
            print(f"  TD-AUC:        {[f'{x:.4f}' if not np.isnan(x) else 'N/A' for x in train_perf['auc_scores']]}")
            print(f"\nTEST SET ({len(self.test_indices_)} samples):")
            print(f"  C-Index:       {test_perf['c_index']:.4f}")
            print(f"  TD-AUC:        {[f'{x:.4f}' if not np.isnan(x) else 'N/A' for x in test_perf['auc_scores']]}")
            print(f"{'='*80}")
        
        return {
            'best_algorithm': best_algo,
            'best_params': best_params,
            'final_model': final_model,
            'train_performance': train_perf,
            'test_performance': test_perf,
            'feature_importances': final_model.feature_importances_ if hasattr(final_model, 'feature_importances_') else None
        }
    
    def _evaluate_all_models_final(self, X, y_time, y_event, cv_results):
        """
        Evaluate ALL models on train and test data
        
        Returns comprehensive performance for all algorithms:
        - CV mean C-Index (from model selection)
        - Train C-Index (retrain on all train_indices)
        - Test C-Index (evaluate on test_indices)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("EVALUATING ALL MODELS ON TRAIN & TEST")
            print(f"{'='*80}")
        
        X_train_full = X[self.train_indices_]
        y_train_full_time = y_time[self.train_indices_]
        y_train_full_event = y_event[self.train_indices_]
        
        X_test = X[self.test_indices_]
        y_test_time = y_time[self.test_indices_]
        y_test_event = y_event[self.test_indices_]
        
        all_model_performance = {}
        
        for algo_name in self.algorithms:
            if self.verbose:
                print(f"\n{algo_name}:")
                print(f"  Selecting hyperparameters...")

            # =============================================================
            # CRITICAL CHANGE:
            # Use the hyperparameters selected DURING CV (seed×fold) and
            # aggregate them by majority-vote, instead of re-tuning with a
            
            # =============================================================
            params_source = 'final_holdout_tuning'
            best_params = None
            cv_params_count = 0
            used_fallback = False
            if isinstance(cv_results, dict) and 'selected_params' in cv_results:
                sp = cv_results.get('selected_params', {}) or {}
                if algo_name in sp and isinstance(sp[algo_name], list) and len(sp[algo_name]) > 0:
                    best_params = None
                    cv_params_count = len(sp[algo_name])
                    used_fallback = False
                    # Aggregate CV-selected hyperparameters according to user choice
                    if getattr(self, 'param_agg_method', 'median_mode') == 'vote':
                        best_params = self._majority_vote_params(sp[algo_name])
                        params_source = 'cv_majority_vote'
                    elif getattr(self, 'param_agg_method', 'median_mode') == 'median_mode':
                        best_params = self._aggregate_params_median_mode(sp[algo_name])
                        params_source = 'cv_median_mode_aggregate'
                    else:
                        raise ValueError(f"Unknown param_agg_method={getattr(self, 'param_agg_method', None)!r}. Use 'median_mode' or 'vote'.")

            # --- NEW: record which seeds/folds contributed CV-selected params (for reporting) ---
            param_seed_counts = {}
            param_seeds = []
            param_outer_fold_counts = {}
            if isinstance(cv_results, dict) and 'selected_params_meta' in cv_results:
                spm = cv_results.get('selected_params_meta', {}) or {}
                meta_list = spm.get(algo_name, None)
                if isinstance(meta_list, list) and len(meta_list) > 0:
                    seeds = [m.get('seed') for m in meta_list if isinstance(m, dict) and m.get('seed') is not None]
                    ofolds = [m.get('outer_fold') for m in meta_list if isinstance(m, dict) and m.get('outer_fold') is not None]
                    if len(seeds) > 0:
                        param_seed_counts = dict(Counter(seeds))
                        param_seeds = sorted(param_seed_counts.keys())
                    if len(ofolds) > 0:
                        param_outer_fold_counts = dict(Counter(ofolds))

            # Fallback (only if CV did not provide params, or algo has none)
            if best_params is None:
                used_fallback = True
                X_tr, X_val, y_tr_time, y_val_time, y_tr_event, y_val_event = train_test_split(
                    X_train_full, y_train_full_time, y_train_full_event,
                    test_size=self.val_size,
                    stratify=y_train_full_event,
                    random_state=self.random_state  # === CHANGED ===
                )
                y_tr = {'time': y_tr_time, 'event': y_tr_event}
                y_val = {'time': y_val_time, 'event': y_val_event}
                best_params, _ = self._tune_hyperparams(algo_name, X_tr, y_tr, X_val, y_val, seed=self.random_state)  # === CHANGED ===
            
            if self.verbose:
                print(f"  Training on full train set...")
            
            # Train on ALL train_indices
            model = self._get_model(algo_name, best_params, seed=self.random_state)  # === CHANGED ===
            model.fit(X_train_full, y_train_full_time, y_train_full_event)
            
            # Evaluate on train
            train_perf = self._evaluate_model(model, X_train_full, y_train_full_time, y_train_full_event)
            
            # Evaluate on test
            test_perf = self._evaluate_model(model, X_test, y_test_time, y_test_event)
            
            # Get CV mean from cv_results
            cv_mean = cv_results['cv_summary'][algo_name]['mean_c_index']
            cv_std = cv_results['cv_summary'][algo_name]['std_c_index']
            
            # Calculate mean of train and test
            mean_train_test = (train_perf['c_index'] + test_perf['c_index']) / 2.0
            
            all_model_performance[algo_name] = {
                'cv_mean_c_index': cv_mean,
                'cv_std_c_index': cv_std,
                'train_c_index': train_perf['c_index'],
                'train_auc_scores': train_perf.get('auc_scores', []),
                'test_c_index': test_perf['c_index'],
                'test_auc_scores': test_perf.get('auc_scores', []),
                'mean_train_test_c_index': mean_train_test,
                'best_params': best_params,
                'best_params_source': params_source,
                'cv_params_count': cv_params_count,
                'used_fallback': used_fallback,
                'param_seeds': ','.join(str(s) for s in param_seeds) if param_seeds else '',
                'param_seed_counts': json.dumps(param_seed_counts, sort_keys=True) if param_seed_counts else '',
                'param_outer_fold_counts': json.dumps(param_outer_fold_counts, sort_keys=True) if param_outer_fold_counts else '',
                'model': model
            }
            
            if self.verbose:
                print(f"  CV:    {cv_mean:.4f} ± {cv_std:.4f}")
                print(f"  Train: {train_perf['c_index']:.4f}")
                print(f"  Test:  {test_perf['c_index']:.4f}")
                print(f"  Mean:  {mean_train_test:.4f}")
        
        # ============================================
        # STACKING ENSEMBLES (4 GROUPS)
        # ============================================
        if 'stacking' in self.ensemble_methods:
            if self.verbose:
                print(f"\n{'='*80}")
                print("CREATING STACKING ENSEMBLES")
                print(f"{'='*80}")
            
            # Define 4 logical groups
            stacking_groups = {
                'stacking_cox': ['xgboost', 'catboost'],
                'stacking_aft_normal': ['xgboost_aft_normal', 'catboost_aft_normal'],
                'stacking_aft_logistic': ['xgboost_aft_logistic', 'catboost_aft_logistic'],
                'stacking_aft_extreme': ['xgboost_aft_extreme', 'catboost_aft_extreme']
            }
            
            # Create stacking for each group
            for group_name, model_names in stacking_groups.items():
                # Check if all models in group are available
                available_models = {
                    name: all_model_performance[name]['model']
                    for name in model_names
                    if name in all_model_performance
                }
                
                # Need at least 2 models for stacking
                if len(available_models) < 2:
                    if self.verbose:
                        print(f"\n{group_name}: SKIPPED (need 2+ models, found {len(available_models)})")
                    continue
                
                if self.verbose:
                    print(f"\n{group_name.upper()}")
                    print(f"{'─'*80}")
                    print(f"Combining: {', '.join(available_models.keys())}")
                
                # Create and train stacking ensemble
                stacking_ensemble = StackingEnsemble(available_models)
                # === MODIFIED (OOF) ===
                # Fit stacking weights using OOF predictions from CV (preferred, leakage-safe)
                # If OOF is not available for some reason, it falls back to in-sample fit.
                oof_dict = cv_results.get('oof_predictions', None)
                if oof_dict is not None:
                    # Build OOF design matrix in the same order as available_models
                    oof_mat = np.column_stack([oof_dict[name] for name in available_models.keys()])
                    # Use only rows that have OOF predictions for all models in this group
                    valid_rows = np.all(~np.isnan(oof_mat), axis=1)
                    if np.sum(valid_rows) >= 10:
                        stacking_ensemble.fit_from_oof(
                            oof_mat[valid_rows],
                            y_train_full_time[valid_rows],
                            y_train_full_event[valid_rows]
                        )
                    else:
                        # Fallback (should be rare)
                        stacking_ensemble.fit(X_train_full, y_train_full_time, y_train_full_event)
                else:
                    stacking_ensemble.fit(X_train_full, y_train_full_time, y_train_full_event)

                
                # Print weights
                if self.verbose:
                    print("\nLearned Weights:")
                    for name, weight in stacking_ensemble.get_weights().items():
                        print(f"  {name:30s}: {weight:.4f}")
                
                # Evaluate stacking
                train_perf = self._evaluate_model(stacking_ensemble, X_train_full, y_train_full_time, y_train_full_event)
                test_perf = self._evaluate_model(stacking_ensemble, X_test, y_test_time, y_test_event)
                mean_train_test = (train_perf['c_index'] + test_perf['c_index']) / 2.0
                
                # Use CV mean of best model in group as proxy
                group_cv_scores = [
                    all_model_performance[name]['cv_mean_c_index']
                    for name in available_models.keys()
                ]
                best_group_cv = max(group_cv_scores)
                
                all_model_performance[group_name] = {
                    'cv_mean_c_index': best_group_cv,  # Proxy
                    'cv_std_c_index': 0.0,
                    'train_c_index': train_perf['c_index'],
                    'train_auc_scores': train_perf.get('auc_scores', []),
                    'test_c_index': test_perf['c_index'],
                    'test_auc_scores': test_perf.get('auc_scores', []),
                    'mean_train_test_c_index': mean_train_test,
                    'best_params': {'weights': stacking_ensemble.get_weights()},
                    'best_params_source': 'ensemble_weights',
                    'cv_params_count': 0,
                    'used_fallback': False,
                    'model': stacking_ensemble
                }
                
                if self.verbose:
                    print(f"\nPerformance:")
                    print(f"  Train: {train_perf['c_index']:.4f}")
                    print(f"  Test:  {test_perf['c_index']:.4f}")
                    print(f"  Mean:  {mean_train_test:.4f}")
        

        # ============================================
        # PAIRWISE ENSEMBLES (USER-SELECTED)
        # ============================================
        if 'pairwise' in self.ensemble_methods:
            if self.verbose:
                print(f"\n{'='*80}")
                print("CREATING PAIRWISE ENSEMBLES (2-MODEL)")
                print(f"{'='*80}")

            # Build ALL 2-model combinations over eligible base models (non-AFT, excluding ensembles/stacking)
            eligible = []
            for nm in all_model_performance.keys():
                nm_l = nm.lower()
                if nm_l.startswith('ens__') or nm_l.startswith('stacking_'):
                    continue
                # Exclude AFT variants (xgboost_aft_*, catboost_aft_*)
                if '_aft_' in nm_l:
                    continue
                eligible.append(nm)

            eligible = sorted(eligible)
            pair_list = list(itertools.combinations(eligible, 2))

            if self.verbose:
                print(f"Eligible base models for pairwise: {len(eligible)}")
                print(f"Pairwise ensembles to create: {len(pair_list)}")

            for a, b in pair_list:
                if a not in all_model_performance or b not in all_model_performance:
                    if self.verbose:
                        print(f"\nens__{a}__{b}: SKIPPED (missing base model)")
                    continue

                model_a = all_model_performance[a]['model']
                model_b = all_model_performance[b]['model']
                ens_name = f"ens__{a}__{b}"

                if self.verbose:
                    print(f"\n{ens_name}")
                    print(f"{'─'*80}")
                    print(f"Combining: {a} + {b} (z-mean on train)")

                ens = PairwiseZMeanEnsemble(model_a, model_b, a, b)
                ens.fit_scaler(X_train_full)

                train_perf = self._evaluate_model(ens, X_train_full, y_train_full_time, y_train_full_event)
                test_perf = self._evaluate_model(ens, X_test, y_test_time, y_test_event)
                mean_train_test = (train_perf['c_index'] + test_perf['c_index']) / 2.0

                # Proxy CV mean as mean of components (keeps plots consistent)
                cv_mean_proxy = np.mean([all_model_performance[a]['cv_mean_c_index'],
                                         all_model_performance[b]['cv_mean_c_index']])
                cv_std_proxy = np.mean([all_model_performance[a]['cv_std_c_index'],
                                        all_model_performance[b]['cv_std_c_index']])

                all_model_performance[ens_name] = {
                    'cv_mean_c_index': float(cv_mean_proxy),
                    'cv_std_c_index': float(cv_std_proxy),
                    'train_c_index': train_perf['c_index'],
                    'train_auc_scores': train_perf.get('auc_scores', []),
                    'test_c_index': test_perf['c_index'],
                    'test_auc_scores': test_perf.get('auc_scores', []),
                    'mean_train_test_c_index': mean_train_test,
                    'best_params': {'members': [a, b], 'combine': 'zmean'},
                    'best_params_source': 'pairwise_ensemble',
                    'cv_params_count': 0,
                    'used_fallback': False,
                    'model': ens
                }

                if self.verbose:
                    print(f"  Train: {train_perf['c_index']:.4f}")
                    print(f"  Test:  {test_perf['c_index']:.4f}")
        if self.verbose:
            print(f"\n{'='*80}")
            print("ALL MODELS EVALUATED")
            print(f"{'='*80}")
        
        return all_model_performance

    def save_results(self, output_dir='survival_prediction_results', create_plots=True):
        """
        Save comprehensive results with all metrics and visualizations
        
        Output files:
        -------------
        1. detailed_results.csv        - All algorithms CV performance
        2. final_performance.csv       - Train + Test performance
        3. metadata.json               - Complete experiment metadata
        4. results_summary.png         - 6-panel visualization
        5. feature_importance.png      - Top 20 features
        6. final_output.pkl            - Complete results object
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"SAVING RESULTS")
            print(f"{'='*80}")
            print(f"Output directory: {output_path}/")
        
        # ============================================
        # 1. CV RESULTS (all algorithms)
        # ============================================
        if 'cv_results' in self.final_output:
            cv_summary = self.final_output['cv_results']['cv_summary']
            
            detailed_results = []
            for algo_name, results in cv_summary.items():
                row = {
                    'algorithm': algo_name,
                    'mean_c_index': results['mean_c_index'],
                    'std_c_index': results['std_c_index'],
                    'is_best': algo_name == self.final_output['cv_results']['best_algorithm']
                }
                
                if 'scores' in results:
                    for seed_idx, score in enumerate(results['scores'], 1):
                        row[f'seed_{seed_idx}_c_index'] = score
                
                detailed_results.append(row)
            
            df_detailed = pd.DataFrame(detailed_results)
            df_detailed = df_detailed.sort_values('mean_c_index', ascending=False)
            df_detailed.to_csv(output_path / 'detailed_results.csv', index=False)
            
            if self.verbose:
                print(f"  ✓ Saved: detailed_results.csv")
        
        # ============================================
        # 2. ALL MODELS PERFORMANCE (CV + train + test)
        # ============================================
        if self.final_output.get('final_holdout') and 'all_models' in self.final_output['final_holdout']:
            all_models = self.final_output['final_holdout']['all_models']
            cv_summary = self.final_output['cv_results']['cv_summary']
            
            # Create comprehensive dataframe
            all_perf_data = []
            
            # Include ALL models (individual + stacking)
            for algo_name in all_models.keys():
                perf = all_models[algo_name]
                
                row = {
                    'algorithm': algo_name,
                    'cv_mean_c_index': perf['cv_mean_c_index'],
                    'cv_std_c_index': perf['cv_std_c_index'],
                    'train_c_index': perf['train_c_index'],
                    'test_c_index': perf['test_c_index'],
                    'mean_train_test_c_index': perf['mean_train_test_c_index'],
                    'is_best': algo_name == self.final_output['final_holdout']['best_algorithm'],
                    'is_stacking': algo_name.startswith('stacking_')  # Mark stacking models
                }
                
                all_perf_data.append(row)
            
            df_all_perf = pd.DataFrame(all_perf_data)
            df_all_perf = df_all_perf.sort_values('mean_train_test_c_index', ascending=False)  # Sort by mean!
            df_all_perf.to_csv(output_path / 'all_models_performance.csv', index=False)
            
            if self.verbose:
                print(f"  ✓ Saved: all_models_performance.csv")

            # === NEW: save per-model best hyperparameters + their source (CV vote vs fallback) ===
            try:
                all_models = self.final_output.get('final_holdout', {}).get('all_models', {}) or {}
                cv_sp = self.final_output.get('cv_results', {}).get('selected_params', {}) or {}
                cv_spm = self.final_output.get('cv_results', {}).get('selected_params_meta', {}) or {}
                params_records = []
                params_json = {}

                from collections import Counter

                for algo_name, d in all_models.items():
                    bp = d.get('best_params', None)
                    src = d.get('best_params_source', d.get('params_source', 'unknown'))
                    cv_count = len(cv_sp.get(algo_name, [])) if isinstance(cv_sp.get(algo_name, []), list) else 0
                    used_fallback = (src == 'final_holdout_tuning')

                    # Seed / fold provenance (if available)
                    meta_list = []
                    if isinstance(cv_spm, dict):
                        meta_list = cv_spm.get(algo_name, []) or []
                    seeds = [int(x.get('seed')) for x in meta_list if isinstance(x, dict) and x.get('seed') is not None]
                    outer_folds = [int(x.get('outer_fold')) for x in meta_list if isinstance(x, dict) and x.get('outer_fold') is not None]
                    seed_counts = Counter(seeds) if len(seeds) > 0 else Counter()
                    outer_counts = Counter(outer_folds) if len(outer_folds) > 0 else Counter()
                    seed_list_str = ",".join(str(s) for s in sorted(set(seeds))) if len(seeds) > 0 else ""

                    if bp is not None:
                        params_json[algo_name] = {
                            'best_params': bp,
                            'best_params_source': src,
                            'cv_params_count': cv_count,
                            'used_fallback': bool(used_fallback),
                            'param_agg_method': getattr(self, 'param_agg_method', None),
                            'param_seeds': seed_list_str,
                            'param_seed_counts': dict(seed_counts),
                            'param_outer_fold_counts': dict(outer_counts)
                        }
                        params_records.append({
                            'algorithm': algo_name,
                            'best_params_source': src,
                            'cv_params_count': cv_count,
                            'used_fallback': int(used_fallback),
                            'param_agg_method': getattr(self, 'param_agg_method', None),
                            'param_seeds': seed_list_str,
                            'param_seed_counts': json.dumps(dict(seed_counts)),
                            'param_outer_fold_counts': json.dumps(dict(outer_counts)),
                            'best_params_json': json.dumps(bp)
                        })

                # Write JSON (easy for SHAP reproducibility)
                with open(output_path / 'all_models_best_params.json', 'w') as f:
                    json.dump(params_json, f, indent=2)
                # Write CSV (easy quick scan)
                if len(params_records) > 0:
                    pd.DataFrame(params_records).to_csv(output_path / 'params_source_report.csv', index=False)
                if self.verbose:
                    print(f"  ✓ Saved: all_models_best_params.json")
                    print(f"  ✓ Saved: params_source_report.csv")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Warning: Could not save params_source_report/all_models_best_params: {e}")
        # ============================================================
        # 2.5 SAVE ALL FITTED MODELS (individual + ensembles) FOR SHAP
        # ============================================================
        # NOTE:
        # - These are the FINAL refit models trained on FULL TRAIN with aggregated params
        # - This is what we want for SHAP / downstream reproducibility
        try:
            import joblib, pickle
            holdout = self.final_output.get('final_holdout', {}) or {}
            all_models = holdout.get('all_models', {}) or {}

            models_dir = output_path / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            saved_files = {}
            models_bundle = {}

            for algo_name, d in all_models.items():
                model_obj = d.get("model", None)
                if model_obj is None:
                    # (fallback explanation) some entries may not store model object if training failed
                    continue

                models_bundle[algo_name] = model_obj

                # Try joblib first; if it fails, fall back to pickle
                out_joblib = models_dir / f"{algo_name}.joblib"
                out_pkl    = models_dir / f"{algo_name}.pkl"

                try:
                    joblib.dump(model_obj, out_joblib)
                    saved_files[algo_name] = str(out_joblib.name)
                except Exception:
                    with open(out_pkl, "wb") as f:
                        pickle.dump(model_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    saved_files[algo_name] = str(out_pkl.name)

            # Single bundle file (easy SHAP entry-point)
            bundle = {
                "models": models_bundle,
                "saved_files": saved_files,  # relative file names under models/
                "feature_names": getattr(self, "feature_names_", None),
                "train_indices": getattr(self, "train_indices_", None),
                "test_indices": getattr(self, "test_indices_", None),
                "best_algorithm_by_your_selection_rule": holdout.get("best_algorithm", None),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            joblib.dump(bundle, output_path / "all_models.pkl")

            manifest = {
                "models_dir": "models/",
                "saved_files": saved_files,
                "n_models_saved": len(saved_files),
                "feature_names_file": "features_used.csv" if (output_path / "features_used.csv").exists() else None,
                "train_indices_file": "train_indices.npy" if (output_path / "train_indices.npy").exists() else None,
                "test_indices_file": "test_indices.npy" if (output_path / "test_indices.npy").exists() else None,
            }
            with open(output_path / "all_models_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            if self.verbose:
                print(f"  ✓ Saved: all_models.pkl (bundle of ALL final-refit models)")
                print(f"  ✓ Saved: models/ (per-model dumps)")
                print(f"  ✓ Saved: all_models_manifest.json")

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Warning: Could not save all models bundle: {e}")

        # Also save best model separately (backward compatibility)
        if self.final_output.get('final_holdout'):
            holdout = self.final_output['final_holdout']
            
            best_perf = [{
                'dataset': 'cv',
                'algorithm': holdout['best_algorithm'],
                'c_index': self.final_output['cv_results']['best_mean_c_index']}, {
                'dataset': 'train',
                'algorithm': holdout['best_algorithm'],
                'c_index': holdout['train_performance']['c_index']}, {
                'dataset': 'test',
                'algorithm': holdout['best_algorithm'],
                'c_index': holdout['test_performance']['c_index']}]
            
            df_best = pd.DataFrame(best_perf)
            df_best.to_csv(output_path / 'best_model_performance.csv', index=False)
            
            if self.verbose:
                print(f"  ✓ Saved: best_model_performance.csv")

            # === NEW: Save fitted best model object separately for SHAP / downstream use ===
            try:
                import joblib
                best_model_obj = holdout.get('final_model', None)
                if best_model_obj is not None:
                    joblib.dump(best_model_obj, output_path / 'best_model.pkl')
                    # also save a lightweight JSON manifest
                    manifest = {
                        'best_algorithm': holdout.get('best_algorithm'),
                        'best_params': holdout.get('best_params', {}),
                        'best_params_source': holdout.get('best_params_source', holdout.get('params_source', 'unknown')),
                        'param_agg_method': getattr(self, 'param_agg_method', None),
                        'train_indices_file': 'train_indices.npy' if (output_path / 'train_indices.npy').exists() else None,
                        'test_indices_file': 'test_indices.npy' if (output_path / 'test_indices.npy').exists() else None,
                        'feature_names': self.feature_names_
                    }
                    with open(output_path / 'best_model_manifest.json', 'w') as f:
                        json.dump(manifest, f, indent=2)
                    if self.verbose:
                        print(f"  ✓ Saved: best_model.pkl")
                        print(f"  ✓ Saved: best_model_manifest.json")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Warning: Could not save best_model.pkl: {e}")
        
        # ============================================
        # 3. COMPREHENSIVE METADATA
        # ============================================
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'approach': self.approach,
            'tuning_method': self.tuning_method,
            'n_tuning_iterations': self.n_tuning_iterations,
            'n_train_samples': len(self.train_indices_),
            'n_test_samples': len(self.test_indices_),
            'n_features': len(self.feature_names_),
            'feature_names': self.feature_names_,
            'n_seeds': self.n_seeds,
            'outer_cv': self.outer_cv if hasattr(self, 'outer_cv') else None,
            'inner_cv': self.inner_cv if hasattr(self, 'inner_cv') else None,
            'algorithms_tested': self.algorithms,
            'best_algorithm': self.final_output.get('cv_results', {}).get('best_algorithm'),
            'cv_mean_c_index': self.final_output.get('cv_results', {}).get('best_mean_c_index'),
            'train_c_index': self.final_output.get('final_holdout', {}).get('train_performance', {}).get('c_index'),
            'test_c_index': self.final_output.get('final_holdout', {}).get('test_performance', {}).get('c_index'),
            'mean_train_test_c_index': (self.final_output.get('final_holdout', {}).get('train_performance', {}).get('c_index', 0) + 
                                       self.final_output.get('final_holdout', {}).get('test_performance', {}).get('c_index', 0)) / 2.0,
            'best_hyperparameters': self.final_output.get('final_holdout', {}).get('best_params', {}),
            'algorithm_metadata': {}
        }
        
        # Per-algorithm metadata (with hyperparameters!)
        all_model_hyperparams = {}
        all_model_param_sources = {}
        if 'cv_results' in self.final_output:
            cv_summary = self.final_output['cv_results']['cv_summary']
            all_models = self.final_output.get('final_holdout', {}).get('all_models', {})
            
            for algo_name in self.algorithms:
                if algo_name in cv_summary:
                    metadata['algorithm_metadata'][algo_name] = {
                        'mean_c_index': float(cv_summary[algo_name]['mean_c_index']),
                        'std_c_index': float(cv_summary[algo_name]['std_c_index']),
                        'scores_per_seed': [float(x) for x in cv_summary[algo_name].get('scores', [])],
                        'is_best': algo_name == metadata['best_algorithm']
                    }
                    
                    # Add hyperparameters if available
                    if algo_name in all_models and 'best_params' in all_models[algo_name]:
                        all_model_hyperparams[algo_name] = all_models[algo_name]['best_params']
                        all_model_param_sources[algo_name] = all_models[algo_name].get('best_params_source', all_models[algo_name].get('params_source', 'unknown'))
        
        # Add all model hyperparameters to metadata
        metadata['all_model_hyperparameters'] = all_model_hyperparams
        metadata['all_model_hyperparameter_sources'] = all_model_param_sources
        
        # Serialize metadata
        metadata_json = {}
        for k, v in metadata.items():
            if isinstance(v, (np.integer, np.floating)):
                metadata_json[k] = float(v)
            elif isinstance(v, np.ndarray):
                metadata_json[k] = v.tolist()
            elif isinstance(v, dict):
                metadata_json[k] = {}
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        metadata_json[k][k2] = {}
                        for k3, v3 in v2.items():
                            if isinstance(v3, (np.integer, np.floating)):
                                metadata_json[k][k2][k3] = float(v3)
                            elif isinstance(v3, list):
                                metadata_json[k][k2][k3] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v3]
                            else:
                                metadata_json[k][k2][k3] = v3
                    elif isinstance(v2, (np.integer, np.floating)):
                        metadata_json[k][k2] = float(v2)
                    elif isinstance(v2, list):
                        metadata_json[k][k2] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v2]
                    else:
                        metadata_json[k][k2] = v2
            elif isinstance(v, list):
                metadata_json[k] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
            else:
                metadata_json[k] = v
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata_json, f, indent=2)
        
        if self.verbose:
            print(f"  ✓ Saved: metadata.json")
        
        # ============================================
        # 4. VISUALIZATIONS
        # ============================================
        if create_plots and 'cv_results' in self.final_output:
            try:
                self._create_visualizations(output_path)
                if self.verbose:
                    print(f"  ✓ Saved: results_summary.png")
                    print(f"  ✓ Saved: feature_importance.png")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Warning: Could not create plots: {e}")
        # ============================================
        # 5. OOF SAVE (BEFORE PICKLE)
        # ============================================
        # Save OOF predictions separately for downstream stacking/late-fusion.
        # IMPORTANT: do this BEFORE pickling the full object, because some fitted models/ensembles
        # may not be pickle-friendly.
        try:
            cvres = self.final_output.get('cv_results', {})
            oof_pred = cvres.get('oof_predictions', None)
            oof_cnt  = cvres.get('oof_counts', None)
            if oof_pred is not None:
                with open(output_path / 'oof_predictions.pkl', 'wb') as f:
                    pickle.dump(oof_pred, f)
            if oof_cnt is not None:
                with open(output_path / 'oof_counts.pkl', 'wb') as f:
                    pickle.dump(oof_cnt, f)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Warning: Could not save OOF objects: {e}")

        # ============================================
        # 6. PICKLE (ROBUST)
        # ============================================
        def _sanitize_final_output_for_pickle(final_output):
            """Remove known non-pickleable objects (models / local classes) before pickling."""
            safe = dict(final_output)

            # Final holdout: drop fitted model objects
            fh = safe.get('final_holdout', None)
            if isinstance(fh, dict):
                fh2 = dict(fh)
                fh2.pop('final_model', None)
                safe['final_holdout'] = fh2

            # All models: drop fitted model objects
            am = safe.get('all_models', None)
            if isinstance(am, list):
                am2 = []
                for row in am:
                    if isinstance(row, dict):
                        r2 = dict(row)
                        r2.pop('model', None)
                        am2.append(r2)
                    else:
                        am2.append(row)
                safe['all_models'] = am2

            # CV results: keep metrics + selected params + OOF (already saved), but drop any models if present
            cv = safe.get('cv_results', None)
            if isinstance(cv, dict):
                cv2 = dict(cv)
                cv2.pop('models', None)
                cv2.pop('final_models', None)
                safe['cv_results'] = cv2

            return safe

        full_pickle_ok = True
        try:
            with open(output_path / 'final_output.pkl', 'wb') as f:
                pickle.dump(self.final_output, f)
        except Exception as e:
            full_pickle_ok = False
            if self.verbose:
                print(f"  ⚠ Warning: Could not pickle full final_output (will save a sanitized version): {e}")

            safe = _sanitize_final_output_for_pickle(self.final_output)
            with open(output_path / 'final_output.pkl', 'wb') as f:
                pickle.dump(safe, f)

            with open(output_path / 'final_output_pickle_warning.txt', 'w') as f:
                f.write("Full final_output.pkl could not be pickled due to non-serializable objects.\n")
                f.write("Saved a sanitized dictionary (models removed) instead.\n")
                f.write(f"Original error: {repr(e)}\n")
        
        if self.verbose:
            print(f"  ✓ Saved: final_output.pkl")
            print(f"\n{'='*80}")
            print(f"✓ ALL RESULTS SAVED TO: {output_path}/")
            print(f"{'='*80}")
    
    def _create_visualizations(self, output_path):
        """Create comprehensive visualizations with CV/Train/Test for all models"""
        cv_summary = self.final_output['cv_results']['cv_summary']
        all_models = self.final_output.get('final_holdout', {}).get('all_models', {})
        
        # Prepare data - include ALL models (CV + stacking)
        # Start with CV models
        algo_names = list(cv_summary.keys())
        
        # Add stacking models if they exist
        for algo_name in all_models.keys():
            if algo_name not in algo_names and algo_name.startswith('stacking_'):
                algo_names.append(algo_name)
        
        # Get CV, train, test for all models
        cv_scores = []
        train_scores = []
        test_scores = []
        
        for algo in algo_names:
            # CV scores (use proxy for stacking)
            if algo in cv_summary:
                cv_scores.append(cv_summary[algo]['mean_c_index'])
            elif algo in all_models:
                cv_scores.append(all_models[algo]['cv_mean_c_index'])  # Proxy from stacking
            else:
                cv_scores.append(np.nan)
            
            # Train and test scores
            if algo in all_models:
                train_scores.append(all_models[algo]['train_c_index'])
                test_scores.append(all_models[algo]['test_c_index'])
            else:
                train_scores.append(np.nan)
                test_scores.append(np.nan)
        
        # Sort by test performance
        sorted_idx = np.argsort(test_scores)[::-1] if not np.all(np.isnan(test_scores)) else np.argsort(cv_scores)[::-1]
        algo_names = [algo_names[i] for i in sorted_idx]
        cv_scores = [cv_scores[i] for i in sorted_idx]
        train_scores = [train_scores[i] for i in sorted_idx]
        test_scores = [test_scores[i] for i in sorted_idx]

        # Keep only Top-K for plots (all results are still saved to disk)
        top_k = min(10, len(algo_names))
        algo_names = algo_names[:top_k]
        cv_scores = cv_scores[:top_k]
        train_scores = train_scores[:top_k]
        test_scores = test_scores[:top_k]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Survival Analysis Results - All Models', fontsize=16, fontweight='bold')
        
        # 1. CV / Train / Test / Mean Comparison (grouped bar)
        ax = axes[0, 0]
        x = np.arange(len(algo_names))
        width = 0.2  # Narrower bars for 4 groups
        
        # Calculate mean for all models
        mean_scores = [(t + te) / 2.0 if not np.isnan(t) and not np.isnan(te) else np.nan 
                       for t, te in zip(train_scores, test_scores)]
        
        ax.barh(x - 1.5*width, cv_scores, width, label='CV', color='skyblue')
        ax.barh(x - 0.5*width, train_scores, width, label='Train', color='lightgreen')
        ax.barh(x + 0.5*width, test_scores, width, label='Test', color='salmon')
        ax.barh(x + 1.5*width, mean_scores, width, label='Mean', color='gold')
        
        ax.set_yticks(x)
        # Highlight stacking models in bold
        labels = []
        for name in algo_names:
            if name.startswith('stacking_'):
                labels.append(f'**{name}**')  # Bold for stacking
            else:
                labels.append(name)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('C-Index', fontsize=10)
        ax.set_title('All Models: CV / Train / Test / Mean (Stacking in Bold)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Test performance with error bars (from CV std)
        ax = axes[0, 1]
        # Get std scores (0 for stacking models since they don't have CV std)
        std_scores = []
        for algo in algo_names:
            if algo in cv_summary:
                std_scores.append(cv_summary[algo]['std_c_index'])
            else:
                std_scores.append(0.0)  # Stacking models have no CV std
        
        ax.errorbar(range(len(algo_names)), test_scores, yerr=std_scores, fmt='o-', capsize=5, markersize=8, color='red', label='Test ± CV std')
        ax.plot(range(len(algo_names)), cv_scores, 'o--', color='blue', alpha=0.6, label='CV mean')
        ax.set_xticks(range(len(algo_names)))
        
        # Highlight stacking in x labels
        xlabels = []
        for name in algo_names:
            if name.startswith('stacking_'):
                xlabels.append(f'**{name}**')
            else:
                xlabels.append(name)
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('C-Index')
        ax.set_title('Test vs CV Performance')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Best model: CV/Train/Test/Mean
        ax = axes[0, 2]
        if self.final_output.get('final_holdout'):
            best_algo = self.final_output['final_holdout']['best_algorithm']
            
            train_ci = all_models[best_algo]['train_c_index']
            test_ci = all_models[best_algo]['test_c_index']
            mean_ci = (train_ci + test_ci) / 2.0
            
            datasets = ['CV', 'Train', 'Test', 'Mean']
            scores = [
                self.final_output['cv_results']['best_mean_c_index'],
                train_ci,
                test_ci,
                mean_ci
            ]
            
            bars = ax.bar(datasets, scores, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
            ax.set_ylabel('C-Index')
            ax.set_title(f'Best Model: {best_algo}')
            ax.set_ylim([min(scores) * 0.95, max(scores) * 1.05])
            
            for i, (d, s) in enumerate(zip(datasets, scores)):
                ax.text(i, s + 0.005, f'{s:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. Performance stability
        ax = axes[1, 0]
        ax.bar(range(len(algo_names)), std_scores)
        ax.set_xticks(range(len(algo_names)))
        ax.set_xticklabels(algo_names, rotation=45, ha='right')
        ax.set_ylabel('Std Dev')
        ax.set_title('Performance Stability')
        ax.axhline(y=0.02, color='r', linestyle='--', label='Target: σ < 0.02')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 5. Time-Dependent AUC (if available)
        ax = axes[1, 1]
        if self.final_output.get('final_holdout') and 'all_models' in self.final_output['final_holdout']:
            all_models = self.final_output['final_holdout']['all_models']
            best_algo = self.final_output['final_holdout']['best_algorithm']
            
            if best_algo in all_models:
                train_auc = all_models[best_algo].get('train_auc_scores', [])
                test_auc = all_models[best_algo].get('test_auc_scores', [])
                
                if len(train_auc) > 0 and not np.all(np.isnan(train_auc)):
                    time_points = self.auc_time_points[:len(train_auc)]
                    
                    ax.plot(time_points, train_auc, 'o-', label='Train', color='lightgreen', linewidth=2, markersize=8)
                    ax.plot(time_points, test_auc, 's-', label='Test', color='salmon', linewidth=2, markersize=8)
                    ax.set_xlabel('Time Point', fontsize=10)
                    ax.set_ylabel('AUC', fontsize=10)
                    ax.set_title(f'Time-Dependent AUC: {best_algo}', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    ax.set_ylim([0.5, 1.0])
                else:
                    ax.text(0.5, 0.5, 'AUC not available\n(insufficient time range)', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='gray')
            else:
                ax.text(0.5, 0.5, 'AUC data not available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
        else:
            ax.text(0.5, 0.5, 'AUC data not available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='gray')
        
        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
EXPERIMENT SUMMARY

Approach: {self.approach}
Algorithms tested: {len(self.algorithms)}
Seeds: {self.n_seeds}

BEST MODEL:
{self.final_output['cv_results']['best_algorithm']}

PERFORMANCE:
CV C-Index:    {self.final_output['cv_results']['best_mean_c_index']:.4f}
"""
        
        if self.final_output.get('final_holdout'):
            holdout = self.final_output['final_holdout']
            if 'train_performance' in holdout:
                summary_text += f"Train C-Index: {holdout['train_performance']['c_index']:.4f}\n"
            if 'test_performance' in holdout:
                summary_text += f"Test C-Index:  {holdout['test_performance']['c_index']:.4f}\n"
        
        summary_text += f"""
DATA:
Train samples: {len(self.train_indices_)}
Test samples:  {len(self.test_indices_)}
Features:      {len(self.feature_names_)}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_path / 'results_summary.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print(__doc__)