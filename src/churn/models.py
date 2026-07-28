"""Zoo de modelos, tuning con CV agrupado, stacking y red secuencial.

Sobre la elección de modelos
----------------------------
En datos tabulares de banca un GBM bien tuneado casi siempre le gana a un MLP.
La red neuronal aporta valor cuando se le da lo que el GBM no puede ver: la
**secuencia mensual cruda**. Por eso aquí hay dos cosas distintas:

* `build_model_zoo()` — modelos tabulares sobre las features agregadas.
* `SequenceModel` — un GRU sobre las series mensuales [n, ventana, canales],
  que captura la *forma* de la trayectoria (aceleración, inflexiones) en vez
  de resúmenes de ella.

El ensemble interesante es la mezcla de ambos, no cinco GBMs promediados.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler,
)

from .config import Config
from .evaluation import make_cv

warnings.filterwarnings("ignore", category=UserWarning)


# ── Preprocesamiento ────────────────────────────────────────────────────

def _clip_extremes(X):
    """Recorta a ±10 sigmas tras el escalado.

    Una columna casi constante tiene desviación ~1e-160; al escalarla,
    StandardScaler la convierte en valores del orden de 1e160 y el `matmul`
    de la regresión logística desborda a inf. VarianceThreshold quita las
    constantes exactas y este recorte cubre el resto.
    """
    return np.clip(X, -10.0, 10.0)


def make_preprocessor(numeric: list[str], categorical: list[str],
                      for_trees: bool = True) -> ColumnTransformer:
    """Los árboles toleran NaN y prefieren codificación ordinal; los modelos
    lineales y las redes necesitan imputación, escala y one-hot."""
    if for_trees:
        num_tf = SimpleImputer(strategy="median", add_indicator=True)
        cat_tf = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
        ])
    else:
        num_tf = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("var", VarianceThreshold(threshold=1e-10)),
            ("sc", StandardScaler()),
            ("clip", FunctionTransformer(_clip_extremes, feature_names_out="one-to-one")),
        ])
        cat_tf = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("enc", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                  sparse_output=False)),
        ])
    return ColumnTransformer(
        [("num", num_tf, numeric), ("cat", cat_tf, categorical)],
        remainder="drop", verbose_feature_names_out=False,
    )


# ── Zoo ─────────────────────────────────────────────────────────────────

def build_model_zoo(numeric: list[str], categorical: list[str],
                    scale_pos_weight: float = 1.0,
                    random_state: int = 42) -> dict[str, dict[str, Any]]:
    """Devuelve {nombre: {"pipeline": Pipeline, "params": search_space}}.

    Nota sobre paralelismo: todos los estimadores usan **un solo hilo**
    (`n_jobs=1`). El paralelismo va en el `RandomizedSearchCV` de afuera.
    Anidar `n_jobs=-1` dentro y fuera produce un deadlock de joblib/loky
    (el proceso se queda vivo al 0% de CPU) — un clásico difícil de
    diagnosticar si no se sabe de antemano.
    """
    zoo: dict[str, dict[str, Any]] = {}
    pre_lin = make_preprocessor(numeric, categorical, for_trees=False)
    pre_tree = make_preprocessor(numeric, categorical, for_trees=True)

    zoo["logreg"] = {
        "pipeline": Pipeline([
            ("pre", pre_lin),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=random_state)),
        ]),
        "params": {
            # Tope de C deliberadamente bajo. Con clases muy desbalanceadas y
            # datos casi separables, una C grande deja que los coeficientes se
            # disparen (separación cuasi-completa) y el `matmul` desborda a inf.
            # La regularización tiene que morder siempre.
            "clf__C": loguniform(1e-4, 1e1),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
    }

    zoo["random_forest"] = {
        "pipeline": Pipeline([
            ("pre", pre_tree),
            ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                           n_jobs=1, random_state=random_state)),
        ]),
        "params": {
            "clf__n_estimators": randint(200, 700),
            "clf__max_depth": randint(4, 18),
            "clf__min_samples_leaf": randint(5, 80),
            "clf__max_features": uniform(0.2, 0.6),
        },
    }

    try:
        from lightgbm import LGBMClassifier

        zoo["lightgbm"] = {
            "pipeline": Pipeline([
                ("pre", pre_tree),
                ("clf", LGBMClassifier(objective="binary", n_jobs=1, verbose=-1,
                                       scale_pos_weight=scale_pos_weight,
                                       random_state=random_state)),
            ]),
            "params": {
                "clf__n_estimators": randint(200, 900),
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__num_leaves": randint(15, 120),
                "clf__min_child_samples": randint(10, 150),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__subsample_freq": [1],
                "clf__colsample_bytree": uniform(0.5, 0.5),
                "clf__reg_lambda": loguniform(1e-3, 20),
            },
        }
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier

        zoo["xgboost"] = {
            "pipeline": Pipeline([
                ("pre", pre_tree),
                ("clf", XGBClassifier(objective="binary:logistic", n_jobs=1,
                                      eval_metric="aucpr", tree_method="hist",
                                      scale_pos_weight=scale_pos_weight,
                                      random_state=random_state)),
            ]),
            "params": {
                "clf__n_estimators": randint(200, 900),
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__max_depth": randint(3, 10),
                "clf__min_child_weight": randint(1, 20),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.5, 0.5),
                "clf__reg_lambda": loguniform(1e-3, 20),
            },
        }
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        zoo["catboost"] = {
            "pipeline": Pipeline([
                ("pre", pre_tree),
                ("clf", CatBoostClassifier(verbose=0, allow_writing_files=False,
                                           thread_count=1,
                                           loss_function="Logloss",
                                           scale_pos_weight=scale_pos_weight,
                                           random_seed=random_state)),
            ]),
            "params": {
                "clf__iterations": randint(200, 800),
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__depth": randint(4, 9),
                "clf__l2_leaf_reg": loguniform(1, 30),
            },
        }
    except ImportError:
        pass

    from sklearn.neural_network import MLPClassifier

    zoo["mlp"] = {
        "pipeline": Pipeline([
            ("pre", pre_lin),
            ("clf", MLPClassifier(max_iter=400, early_stopping=True,
                                  random_state=random_state)),
        ]),
        "params": {
            "clf__hidden_layer_sizes": [(64,), (128,), (128, 64), (64, 32)],
            "clf__alpha": loguniform(1e-5, 1e-1),
            "clf__learning_rate_init": loguniform(1e-4, 1e-2),
        },
    }
    return zoo


# ── Tuning ──────────────────────────────────────────────────────────────

def tune(name: str, spec: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
         groups: np.ndarray, cfg: Config, verbose: bool = True):
    """RandomizedSearchCV con StratifiedGroupKFold agrupado por CIF."""
    m = cfg.models
    search = RandomizedSearchCV(
        spec["pipeline"], spec["params"],
        n_iter=int(m.get("n_iter", 20)),
        scoring=m.get("scoring", "average_precision"),
        cv=make_cv(cfg),
        n_jobs=int(m.get("n_jobs", -1)),
        random_state=int(cfg.split.get("random_state", 42)),
        refit=True, error_score="raise",
    )
    search.fit(X, y, groups=groups)
    if verbose:
        print(f"   {name:14s} CV {m.get('scoring')} = {search.best_score_:.4f}")
    return search


def tune_all(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
             numeric: list[str], categorical: list[str], cfg: Config,
             verbose: bool = True) -> dict[str, Any]:
    """Tunea todos los modelos habilitados. Devuelve {nombre: search}."""
    pos = float(y.mean())
    spw = (1 - pos) / pos if 0 < pos < 1 else 1.0
    zoo = build_model_zoo(numeric, categorical, scale_pos_weight=spw,
                          random_state=int(cfg.split.get("random_state", 42)))
    enabled = [n for n in cfg.models.get("enabled", list(zoo)) if n in zoo]

    if verbose:
        print(f"\n🔍 Tuning {len(enabled)} modelos "
              f"({cfg.models.get('n_iter')} iteraciones, "
              f"{cfg.split.get('n_folds')} folds agrupados por CIF)")
        print(f"   scale_pos_weight = {spw:.1f}  (tasa base {pos:.2%})")

    out = {}
    for name in enabled:
        try:
            out[name] = tune(name, zoo[name], X, y, groups, cfg, verbose)
        except Exception as e:  # un modelo roto no debe tumbar el pipeline
            print(f"   ⚠️  {name}: falló ({type(e).__name__}: {e})")
    return out


def build_stack(searches: dict[str, Any], top_k: int, cfg: Config):
    """Stacking con meta-modelo logístico sobre los top-k por CV.

    `cv` interno agrupado no es posible en StackingClassifier sin groups, así
    que se usa un CV estratificado simple sobre las predicciones out-of-fold.
    El riesgo de leakage aquí es menor porque los base learners ya fueron
    seleccionados con CV agrupado.
    """
    ranked = sorted(searches.items(), key=lambda kv: kv[1].best_score_, reverse=True)
    chosen = ranked[:top_k]
    if len(chosen) < 2:
        return None, [n for n, _ in chosen]

    estimators = [(n, s.best_estimator_) for n, s in chosen]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=3, n_jobs=1, passthrough=False,
    )
    return stack, [n for n, _ in chosen]


# ── Red secuencial (GRU sobre las series mensuales) ─────────────────────

SEQUENCE_COLS = [
    "owned_balance", "txn_count_total", "loan_balance", "n_deposit_accounts",
    "login_count_mobile_month", "login_count_web_month",
    "has_direct_deposit", "fees_charged",
]


def build_sequences(state: pd.DataFrame, index_df: pd.DataFrame,
                    seq_cols: list[str], window: int) -> np.ndarray:
    """Arma el tensor [n_muestras, ventana, n_canales].

    Para cada fila de `index_df` (cif, as_of_month) toma los `window` meses
    que terminan en ese mes — nunca posteriores.
    """
    cols = [c for c in seq_cols if c in state.columns]
    s = state.sort_values(["cif_id", "as_of_month"]).copy()

    lagged = {}
    g = s.groupby("cif_id")
    for c in cols:
        for lag in range(window):
            lagged[f"{c}__lag{lag}"] = g[c].shift(lag)
    lag_df = pd.DataFrame(lagged, index=s.index)
    lag_df[["cif_id", "as_of_month"]] = s[["cif_id", "as_of_month"]]

    merged = index_df[["cif_id", "as_of_month"]].merge(
        lag_df, on=["cif_id", "as_of_month"], how="left"
    )
    n = len(merged)
    arr = np.zeros((n, window, len(cols)), dtype=np.float32)
    for ci, c in enumerate(cols):
        block = merged[[f"{c}__lag{lag}" for lag in range(window)]].to_numpy(dtype=np.float32)
        # lag0 es el mes actual → invertir para que el eje temporal avance
        arr[:, :, ci] = block[:, ::-1]

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalización por canal (log1p en montos para domar las colas)
    for ci, c in enumerate(cols):
        ch = arr[:, :, ci]
        if "balance" in c or "fees" in c or "amt" in c:
            ch = np.sign(ch) * np.log1p(np.abs(ch))
        mu, sd = ch.mean(), ch.std()
        arr[:, :, ci] = (ch - mu) / (sd if sd > 1e-6 else 1.0)
    return arr


class SequenceModel:
    """GRU binario sobre las series mensuales. API estilo sklearn (fit/predict_proba)."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 256, patience: int = 5, seed: int = 42):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.epochs, self.batch_size, self.patience, self.lr = epochs, batch_size, patience, lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(n_features, hidden_size, num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                    nn.Linear(hidden_size, 32), nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.model = _Net().to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None, verbose=True):
        torch, nn = self.torch, self.nn
        from torch.utils.data import DataLoader, TensorDataset

        pos = float(y.mean())
        pos_weight = torch.tensor([(1 - pos) / pos if 0 < pos < 1 else 1.0],
                                  device=self.device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best, bad, best_state = np.inf, 0, None
        for ep in range(self.epochs):
            self.model.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(xb)
            train_loss = tot / len(ds)

            if X_val is not None:
                from sklearn.metrics import average_precision_score
                p = self.predict_proba(X_val)[:, 1]
                score = -average_precision_score(y_val, p)
            else:
                score = train_loss

            if score < best - 1e-5:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    if verbose:
                        print(f"   early stopping en época {ep + 1}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose:
            metric = "PR-AUC val" if X_val is not None else "loss train"
            print(f"   GRU entrenado — mejor {metric}: {abs(best):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.from_numpy(X[i:i + 4096]).to(self.device)
                outs.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        p = np.concatenate(outs) if outs else np.zeros(len(X))
        return np.column_stack([1 - p, p])
