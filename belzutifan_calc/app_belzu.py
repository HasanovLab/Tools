# app_belzu.py
# -------------------------------------------------------------------
# Belzutifan RCC OS Calculator
# best_model: RSF | horizon: 3 years
# -------------------------------------------------------------------

import json
import pickle
import joblib
import numpy as np
import pandas as pd
import shap

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive
from lifelines import CoxPHFitter

# =========================
# Pipeline class stubs
# Required to unpickle all_models.pkl
# Classes mirror survival_pipeline.py exactly so pickle deserialization works
# =========================
import sys as _sys
import types as _types
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from scipy.optimize import minimize
from sksurv.metrics import concordance_index_censored


class PairwiseZMeanEnsemble:
    def __init__(self, model_a, model_b, name_a, name_b):
        self.model_a = model_a; self.model_b = model_b
        self.name_a = name_a; self.name_b = name_b
        self.mu_a = 0.0; self.sd_a = 1.0; self.mu_b = 0.0; self.sd_b = 1.0

    def fit_scaler(self, X_ref):
        pa = np.asarray(self.model_a.predict(X_ref)).ravel()
        pb = np.asarray(self.model_b.predict(X_ref)).ravel()
        self.mu_a = float(np.nanmean(pa)); self.sd_a = float(np.nanstd(pa) + 1e-12)
        self.mu_b = float(np.nanmean(pb)); self.sd_b = float(np.nanstd(pb) + 1e-12)

    def predict(self, X):
        pa = np.asarray(self.model_a.predict(X)).ravel()
        pb = np.asarray(self.model_b.predict(X)).ravel()
        return 0.5 * ((pa - self.mu_a) / self.sd_a + (pb - self.mu_b) / self.sd_b)


class XGBoostSurvivalWrapper:
    def __init__(self, params, seed):
        self.params = {'objective': 'survival:cox', 'eval_metric': 'cox-nloglik', 'seed': seed, **params}
        self.num_boost_round = self.params.pop('n_estimators', 100)
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        dtest = xgb.DMatrix(np.asarray(X))
        return self.model_.predict(dtest)


class XGBoostAFTWrapper:
    def __init__(self, params, seed, aft_loss_distribution='normal'):
        self.aft_loss_distribution = aft_loss_distribution
        self.params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
                       'aft_loss_distribution': aft_loss_distribution, 'seed': seed, **params}
        self.num_boost_round = self.params.pop('n_estimators', 100)
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        dtest = xgb.DMatrix(np.asarray(X))
        return -self.model_.predict(dtest)


class CatBoostSurvivalWrapper:
    def __init__(self, params, seed):
        self.params = {'loss_function': 'Cox', 'eval_metric': 'Cox',
                       'random_seed': seed, 'verbose': False, **params}
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


class CatBoostAFTWrapper:
    def __init__(self, params, seed, aft_loss_distribution='normal', scale=None):
        self.aft_loss_distribution = aft_loss_distribution; self.scale = scale
        dist_map = {'normal': 'Normal', 'logistic': 'Logistic', 'extreme': 'Extreme'}
        loss_str = f"SurvivalAft:dist={dist_map.get(aft_loss_distribution.lower(), 'Normal')}"
        if scale is not None: loss_str += f';scale={scale}'
        self.params = {'loss_function': loss_str, 'eval_metric': 'SurvivalAft',
                       'random_seed': seed, 'verbose': False, **params}
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        return -self.model_.predict(np.asarray(X))


class StackingEnsemble:
    def __init__(self, base_models):
        self.base_models = base_models
        self.weights = None
        self.model_names = list(base_models.keys())

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Must fit before predict!")
        predictions = self._get_base_predictions(X)
        return np.average(predictions, axis=1, weights=self.weights)

    def _get_base_predictions(self, X):
        return np.column_stack([self.base_models[n].predict(X) for n in self.model_names])

    def get_weights(self):
        if self.weights is None: return None
        return {name: w for name, w in zip(self.model_names, self.weights)}


class LightGBMSurvivalWrapper:
    def __init__(self, params, seed):
        self.params = {'objective': 'regression', 'random_state': seed, 'verbose': -1, **params}
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        return -self.model_.predict(np.asarray(X))


class CoxPHSkSurvWrapper:
    def __init__(self, params, seed):
        self.params = dict(params); self.seed = seed; self.model_ = None

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()


class CoxnetSkSurvWrapper:
    def __init__(self, params, seed):
        self.params = dict(params); self.seed = seed; self.model_ = None
        self.alpha_index_ = self.params.pop('alpha_index', None)

    def predict(self, X):
        pred = np.asarray(self.model_.predict(np.asarray(X)))
        if pred.ndim == 2:
            return pred[:, self.alpha_index_].ravel()
        return pred.ravel()


class RSFSkSurvWrapper:
    def __init__(self, params, seed):
        self.params = dict(params); self.seed = seed
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()


class ExtraSurvivalTreesWrapper:
    def __init__(self, params, seed):
        self.params = dict(params); self.seed = seed
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()


class GBMSurvivalSkSurvWrapper:
    def __init__(self, params, seed):
        self.params = dict(params); self.seed = seed
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()


class SurvivalTreeSkSurvWrapper:
    def __init__(self, params, seed):
        self.params = dict(params); self.seed = seed
        self.model_ = None; self.feature_importances_ = None

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()


class FastSurvivalSVMWrapper:
    def __init__(self, params, seed):
        self.params = dict(params); self.seed = seed; self.model_ = None

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float64))).ravel()


# Register all classes in the survival_pipeline module so pickle can find them
_MOD_NAME = "survival_pipeline"
_mod = _types.ModuleType(_MOD_NAME)
_mod.PairwiseZMeanEnsemble = PairwiseZMeanEnsemble
_mod.XGBoostSurvivalWrapper = XGBoostSurvivalWrapper
_mod.XGBoostAFTWrapper = XGBoostAFTWrapper
_mod.CatBoostSurvivalWrapper = CatBoostSurvivalWrapper
_mod.CatBoostAFTWrapper = CatBoostAFTWrapper
_mod.StackingEnsemble = StackingEnsemble
_mod.LightGBMSurvivalWrapper = LightGBMSurvivalWrapper
_mod.CoxPHSkSurvWrapper = CoxPHSkSurvWrapper
_mod.CoxnetSkSurvWrapper = CoxnetSkSurvWrapper
_mod.RSFSkSurvWrapper = RSFSkSurvWrapper
_mod.ExtraSurvivalTreesWrapper = ExtraSurvivalTreesWrapper
_mod.GBMSurvivalSkSurvWrapper = GBMSurvivalSkSurvWrapper
_mod.SurvivalTreeSkSurvWrapper = SurvivalTreeSkSurvWrapper
_mod.FastSurvivalSVMWrapper = FastSurvivalSVMWrapper
_sys.modules[_MOD_NAME] = _mod


# =========================
# 0) Paths (relative)
# =========================
HERE = Path(__file__).resolve().parent

# All files sit in the same directory as app_belzu.py (belzutifan_calc root)
CSV_PATH       = HERE / "train_30_encoded.csv"
ALL_MODELS_PKL = HERE / "all_models.pkl"
EXPLAINER_PKL  = HERE / "shap_explainer.pkl"
METADATA_JSON  = HERE / "metadata.json"

MODEL_KEY = "rsf"
TIME_COL = "os_time_months"
EVENT_COL = "os_event"
TIME_UNIT = "months"
SURV_HORIZON = 36.0  # 3 years in months


# =========================
# 1) Features
# =========================
# All CSV feature column names (order must match training)
FEATURE_COLS = [
    "Alpha-Beta Blockers",
    "age",
    "Hyperuricemia Therapy - Xanthine Oxidase Inhibitors",
    "Antihyperglycemic - Sodium Glucose Cotransporter-2 (SGLT2) Inhibitors",
    "Skeletal Muscle Relaxant - Central Muscle Relaxants",
    "MCH",
    "Antipsychotic - Phenothiazines, Piperazine",
    "MCHC",
    "Antiemetic - Selective Serotonin 5-HT3 Antagonists",
    "Analgesic Opioid Agonists",
    "Sodium",
    "Chloride",
    "ALP",
    "Platelets",
    "RBC",
    "Corrected_Calcium",
    "Hematocrit",
    "Hemoglobin",
    "Albumin",
    "Aldesleukin",
    "Axitinib",
    "Bevacizumab",
    "Cabozantinib",
    "Everolimus",
    "Ipilimumab",
    "Lenvatinib",
    "Nivolumab",
    "Pazopanib",
    "Pembrolizumab",
    "Sunitinib",
    "Temsirolimus",
    "Tivozanib",
]

# Binary features: shown as Yes/No dropdowns
BINARY_COLS = {
    "Alpha-Beta Blockers",
    "Hyperuricemia Therapy - Xanthine Oxidase Inhibitors",
    "Antihyperglycemic - Sodium Glucose Cotransporter-2 (SGLT2) Inhibitors",
    "Skeletal Muscle Relaxant - Central Muscle Relaxants",
    "Antipsychotic - Phenothiazines, Piperazine",
    "Antiemetic - Selective Serotonin 5-HT3 Antagonists",
    "Analgesic Opioid Agonists",
    "Aldesleukin",
    "Axitinib",
    "Bevacizumab",
    "Cabozantinib",
    "Everolimus",
    "Ipilimumab",
    "Lenvatinib",
    "Nivolumab",
    "Pazopanib",
    "Pembrolizumab",
    "Sunitinib",
    "Temsirolimus",
    "Tivozanib",
}

# Numeric feature defaults (median-ish values for initialization)
NUMERIC_DEFAULTS = {
    "age": 62,
    "MCH": 29.0,
    "MCHC": 33.0,
    "Sodium": 139.0,
    "Chloride": 103.0,
    "ALP": 90.0,
    "Platelets": 230.0,
    "RBC": 4.3,
    "Corrected_Calcium": 9.4,
    "Hematocrit": 38.0,
    "Hemoglobin": 12.5,
    "Albumin": 4.0,
}

# Shiny input IDs — special chars not allowed, replace with double underscore
def _to_id(feat: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]", "__", feat)


FEATURE_IDS = [_to_id(f) for f in FEATURE_COLS]
ID_TO_COL = {_to_id(f): f for f in FEATURE_COLS}  # reverse lookup


# =========================
# 2) Pickle-compat helpers
# =========================
def predict_scalar_from_model(model, X_df: pd.DataFrame, feature_cols=None, t_star_years=None, **kwargs) -> np.ndarray:
    if feature_cols is not None and isinstance(X_df, pd.DataFrame):
        X_df = X_df[feature_cols]
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X_df)).reshape(-1).astype(float)
    if hasattr(model, "predict_risk"):
        return np.asarray(model.predict_risk(X_df)).reshape(-1).astype(float)
    raise AttributeError("Model has no predict / predict_risk method.")


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "predict_scalar_from_model":
            return predict_scalar_from_model
        return super().find_class(module, name)


def safe_pickle_load(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        try:
            with open(path, "rb") as f:
                return _SafeUnpickler(f).load()
        except Exception as e_pickle:
            raise RuntimeError(
                f"Failed to load {path}\njoblib: {e_joblib}\npickle: {e_pickle}"
            )


def load_metadata(path: Path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


# =========================
# 3) Load data / model / explainer
# =========================
df_all = pd.read_csv(CSV_PATH)
df_all.columns = df_all.columns.astype(str).str.strip()

missing_cols = [c for c in [TIME_COL, EVENT_COL, *FEATURE_COLS] if c not in df_all.columns]
if missing_cols:
    # Fallback: CSV saved with index column
    df_alt = pd.read_csv(CSV_PATH, index_col=0)
    df_alt.columns = df_alt.columns.astype(str).str.strip()
    missing_alt = [c for c in [TIME_COL, EVENT_COL, *FEATURE_COLS] if c not in df_alt.columns]
    if not missing_alt:
        df_all = df_alt
        missing_cols = []

if missing_cols:
    raise ValueError(f"CSV missing columns: {missing_cols}; available: {list(df_all.columns)[:30]}")

model = safe_pickle_load(ALL_MODELS_PKL)
explainer = safe_pickle_load(EXPLAINER_PKL)
meta = load_metadata(METADATA_JSON)

# Train indices from metadata or npy
train_idx = None
for k in ["train_idx", "train_index", "train_indices"]:
    if k in meta:
        try:
            train_idx = np.array(meta[k], dtype=int)
            if train_idx.size > 0:
                break
        except Exception:
            pass

if train_idx is None or train_idx.size == 0:
    npy_path = HERE / "train_indices.npy"
    if npy_path.exists():
        train_idx = np.load(npy_path)
        print(f"[INFO] train_idx from train_indices.npy: {len(train_idx)}")
    else:
        print("[INFO] No train_idx — using full dataset for calibration")


# =========================
# 4) UI helpers — build input widgets
# =========================
def _make_widget(feat: str):
    sid = _to_id(feat)
    label = feat  # use raw column name as label
    if feat in BINARY_COLS:
        return ui.input_select(
            sid,
            label,
            {"1": "Yes", "0": "No"},
            selected="0",
        )
    else:
        default_val = NUMERIC_DEFAULTS.get(feat, 0)
        return ui.input_numeric(
            sid,
            label,
            value=default_val,
            step=0.1,
        )


input_widgets = [_make_widget(f) for f in FEATURE_COLS]

app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background: #f8f9fa; }
        .shiny-input-container label {
            font-size: 11px !important;
            font-weight: 500 !important;
            margin-bottom: 1px !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .shiny-input-container { margin-bottom: 4px !important; }
        .form-control, .form-select { font-size: 12px !important; padding: 2px 6px !important; height: 28px !important; }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 2px 12px;
        }
        .output-row {
            display: flex;
            gap: 16px;
            margin: 12px 16px 20px 16px;
        }
        .output-row > div { flex: 1; min-width: 0; }
    """),

    ui.h2(
        "Belzutifan RCC — OS Prediction",
        style="text-align:center; font-weight:bold; margin:12px 0 10px 0; font-size:1.4rem;",
    ),

    ui.div(
        ui.card(
            ui.h5("Clinical Features", style="font-weight:bold; margin-bottom:6px; font-size:0.95rem;"),
            ui.div(*input_widgets, class_="feature-grid"),
            ui.div(
                ui.input_action_button(
                    "predict", "Predict",
                    style="background:#2563eb; color:white; border:none; padding:8px 32px; border-radius:6px; font-size:14px; font-weight:600; cursor:pointer;",
                ),
                ui.input_action_button(
                    "reset", "Reset",
                    style="background:#6b7280; color:white; border:none; padding:8px 24px; border-radius:6px; font-size:14px; cursor:pointer;",
                ),
                style="display:flex; justify-content:center; gap:16px; margin-top:12px;",
            ),
            style="padding:14px 16px; box-shadow:0 2px 10px rgba(0,0,0,0.07); border-radius:10px; background:white;",
        ),
        style="margin:0 16px 12px 16px;",
    ),

    ui.div(
        ui.div(
            ui.card(
                ui.h4("Waterfall Plot", style="font-weight:bold; margin-bottom:4px; font-size:1.05rem;"),
                ui.output_plot("shapplot"),
                style="padding:10px; box-shadow:0 2px 8px rgba(0,0,0,0.07); border-radius:10px; background:white;",
            ),
        ),
        ui.div(
            ui.card(
                ui.h4("Survival Curve", style="font-weight:bold; margin-bottom:4px; font-size:1.05rem;"),
                ui.output_plot("survival_curve"),
                style="padding:10px; box-shadow:0 2px 8px rgba(0,0,0,0.07); border-radius:10px; background:white;",
            ),
        ),
        class_="output-row",
    ),
)


# =========================
# 5) Core logic
# =========================
def _get_new_patient_df(input) -> pd.DataFrame:
    row = {}
    for sid, col in ID_TO_COL.items():
        raw = getattr(input, sid)()
        if col in BINARY_COLS:
            row[col] = int(raw)
        else:
            row[col] = float(raw) if raw is not None else NUMERIC_DEFAULTS.get(col, 0.0)
    return pd.DataFrame([row], columns=FEATURE_COLS)


def _predict_risk(m, X_df: pd.DataFrame) -> np.ndarray:
    if hasattr(m, "predict_risk"):
        return np.asarray(m.predict_risk(X_df)).reshape(-1)
    if hasattr(m, "predict"):
        return np.asarray(m.predict(X_df)).reshape(-1)
    raise RuntimeError("Model has no predict_risk or predict method.")


def _compute_risk(X_df: pd.DataFrame) -> np.ndarray:
    return _predict_risk(model, X_df)


def _get_inner_model(mdl):
    """Unwrap our stub wrapper to get the actual fitted sksurv/sklearn model."""
    # Our RSFSkSurvWrapper stores the real model in .model_
    if hasattr(mdl, "model_") and mdl.model_ is not None:
        return mdl.model_
    return mdl


def _compute_survival_curve(mdl, df_train_calib, risk_train, x_one, risk_val):
    times = np.linspace(0.0, SURV_HORIZON, 300)

    # Try direct predict_survival_function on wrapper first
    inner = _get_inner_model(mdl)
    for candidate in [mdl, inner]:
        if hasattr(candidate, "predict_survival_function"):
            try:
                X_arr = x_one.values if isinstance(x_one, pd.DataFrame) else np.asarray(x_one)
                fns = candidate.predict_survival_function(X_arr)
                fn = fns[0]
                surv = np.array([fn(t) for t in times], dtype=float)
                return times, surv
            except Exception:
                pass

    # Cox recalibration fallback
    df_c = pd.DataFrame({
        TIME_COL: df_train_calib[TIME_COL].astype(float).values,
        EVENT_COL: df_train_calib[EVENT_COL].astype(int).values,
        "risk": np.asarray(risk_train, dtype=float).reshape(-1),
    }).dropna()

    cph = CoxPHFitter()
    cph.fit(df_c, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)
    surv_df = cph.predict_survival_function(pd.DataFrame({"risk": [float(risk_val)]}))
    base_t = surv_df.index.values.astype(float)
    base_s = surv_df.iloc[:, 0].values.astype(float)
    surv = np.interp(times, base_t, base_s, left=base_s[0], right=base_s[-1])
    return times, surv


def _make_waterfall_figure(X_new: pd.DataFrame):
    exp = explainer(X_new)
    exp.feature_names = FEATURE_COLS
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(exp[0], max_display=15, show=False)
    plt.subplots_adjust(left=0.45, right=0.97, top=0.95, bottom=0.10)
    return fig


# =========================
# 6) Server
# =========================
predict_clicked = reactive.Value(False)


def server(input, output, session):

    @reactive.Effect
    @reactive.event(input.predict)
    def _on_predict():
        predict_clicked.set(True)

    @reactive.Effect
    @reactive.event(input.reset)
    def _on_reset():
        # Reset binary features to "0" (No)
        for f in FEATURE_COLS:
            sid = _to_id(f)
            if f in BINARY_COLS:
                session.send_input_message(sid, {"value": "0"})
            else:
                session.send_input_message(sid, {"value": NUMERIC_DEFAULTS.get(f, 0)})
        predict_clicked.set(False)

    @reactive.Effect
    @reactive.event(*[getattr(input, sid) for sid in FEATURE_IDS])
    def _on_any_change():
        predict_clicked.set(False)

    def _ref_df():
        if train_idx is not None and isinstance(train_idx, np.ndarray) and train_idx.size > 10:
            idx = train_idx[(train_idx >= 0) & (train_idx < len(df_all))]
            return df_all.iloc[idx].reset_index(drop=True)
        return df_all

    @output
    @render.plot
    def shapplot():
        if not predict_clicked.get():
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            ax.text(
                0.5, 0.5, "Press Predict to see SHAP explanation",
                ha="center", va="center", fontsize=11, color="gray",
            )
            return fig
        try:
            return _make_waterfall_figure(_get_new_patient_df(input))
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, f"Waterfall error:\n{e}", ha="center", va="center", fontsize=9)
            ax.axis("off")
            return fig

    @output
    @render.plot
    def survival_curve():
        if not predict_clicked.get():
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.axis("off")
            ax.text(
                0.5, 0.5, "Press Predict to see survival curve",
                ha="center", va="center", fontsize=11, color="gray",
            )
            return fig
        try:
            X_new = _get_new_patient_df(input)
            r = float(_compute_risk(X_new)[0])
            ref_df = _ref_df()
            risk_tr = _compute_risk(ref_df[FEATURE_COLS].copy())
            t, s = _compute_survival_curve(model, ref_df, risk_tr, X_new, r)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t, s, linewidth=2.5, color="#2563eb")
            ax.set_xlabel(f"Time ({TIME_UNIT})", fontsize=11)
            ax.set_ylabel("Survival probability", fontsize=11)
            ax.set_xlim(0, SURV_HORIZON)
            ax.set_ylim(0, 1.05)
            # Vertical lines at 12, 24, 36 months
            for vline in [12, 24, 36]:
                ax.axvline(vline, color="gray", linestyle="--", alpha=0.4, linewidth=1)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            return fig
        except Exception as e:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, f"Survival curve error:\n{e}", ha="center", va="center", fontsize=9)
            ax.axis("off")
            return fig


app = App(app_ui, server)
