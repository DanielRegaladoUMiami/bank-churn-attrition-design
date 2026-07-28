"""Splits sin leakage, métricas de negocio y calibración.

Tres cosas que este módulo hace distinto al default de sklearn:

1. **Split out-of-time con embargo.** No basta con cortar por fecha: una fila
   de train con as-of en noviembre y ventana de performance de 6 meses "ve"
   hasta mayo. Si el test empieza en enero, esa fila tiene información del
   periodo de test. El embargo elimina del train toda fila cuya `label_end`
   caiga dentro del periodo de test.

2. **StratifiedGroupKFold agrupado por CIF.** Con panel, el mismo cliente
   aparece en muchas filas. Un KFold normal lo pone en train y validación a la
   vez, y el AUC sube varios puntos sin que el modelo haya mejorado.

3. **Métricas de negocio.** Con 5% de tasa base, accuracy es inútil y ROC-AUC
   engaña. Lo que importa es lift y capture rate en los primeros deciles,
   porque retención solo puede contactar el top 5-10% de la lista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .config import Config


# ── Splits ──────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, cfg: Config, verbose: bool = True):
    """Split out-of-time con embargo. Devuelve (train, test).

    El embargo cuesta `gap + performance` meses de panel: una fila de train con
    as-of en el mes t etiqueta hasta t+gap+perf, así que para no contaminar el
    periodo de test hace falta ese colchón. Es un costo real, no un bug — si
    deja el train vacío, el problema es que el panel es demasiado corto.
    """
    oot_months = int(cfg.split.get("oot_months", 6))
    use_embargo = bool(cfg.split.get("embargo", True))
    months = pd.DatetimeIndex(sorted(df["as_of_month"].unique()))
    if len(months) <= oot_months:
        raise ValueError(
            f"Solo hay {len(months)} meses en el panel; no alcanzan para "
            f"reservar {oot_months} como out-of-time."
        )
    cutoff = months[-oot_months]

    test = df[df["as_of_month"] >= cutoff].copy()
    train_raw = df[df["as_of_month"] < cutoff]

    # Embargo: fuera las filas de train que etiquetan dentro del periodo de test
    if use_embargo and "label_end" in train_raw.columns:
        train = train_raw[train_raw["label_end"] < cutoff].copy()
        embargoed = len(train_raw) - len(train)
    else:
        train, embargoed = train_raw.copy(), 0

    if len(train) == 0:
        gap = int(cfg.panel.get("gap_months", 1))
        perf = int(cfg.panel.get("performance_window_months", 6))
        raise ValueError(
            f"El embargo dejó el train vacío.\n"
            f"  Panel: {len(months)} meses ({months[0].date()} → {months[-1].date()})\n"
            f"  OOT reservado: {oot_months} meses (corte {cutoff.date()})\n"
            f"  Colchón que exige el embargo: gap({gap}) + performance({perf}) "
            f"= {gap + perf} meses\n"
            f"  → Necesitas un panel de al menos "
            f"{oot_months + gap + perf + 3} meses.\n"
            f"  Opciones: más historia, menos oot_months, "
            f"performance_window más corta, o split.embargo=false "
            f"(acepta un sesgo optimista leve)."
        )

    if verbose:
        print(f"\n✂️  Split out-of-time (corte: {cutoff.date()})")
        print(f"   Train: {len(train):>8,} filas  "
              f"({train['as_of_month'].min().date()} → "
              f"{train['as_of_month'].max().date()})  "
              f"churn {train['y'].mean():.2%}")
        print(f"   Test:  {len(test):>8,} filas  "
              f"({test['as_of_month'].min().date()} → "
              f"{test['as_of_month'].max().date()})  "
              f"churn {test['y'].mean():.2%}")
        print(f"   Embargadas por solapamiento de ventana: {embargoed:,}"
              f"{'' if use_embargo else '  (embargo DESACTIVADO)'}")
        overlap = len(set(train['cif_id']) & set(test['cif_id']))
        print(f"   CIFs presentes en ambos: {overlap:,} "
              f"(esperado y correcto en OOT: en producción re-scoreas al mismo cliente)")
    return train, test


def make_cv(cfg: Config) -> StratifiedGroupKFold:
    """CV estratificado Y agrupado por CIF. El `groups` va en `.split()`."""
    return StratifiedGroupKFold(
        n_splits=int(cfg.split.get("n_folds", 5)),
        shuffle=True,
        random_state=int(cfg.split.get("random_state", 42)),
    )


# ── Métricas ────────────────────────────────────────────────────────────

def lift_at_k(y_true, y_prob, k: float = 0.10) -> float:
    """Lift en el top k de la lista ordenada por probabilidad."""
    y_true = np.asarray(y_true)
    n = max(1, int(len(y_true) * k))
    idx = np.argsort(y_prob)[::-1][:n]
    base = y_true.mean()
    return float(y_true[idx].mean() / base) if base > 0 else 0.0


def capture_at_k(y_true, y_prob, k: float = 0.10) -> float:
    """Fracción de todos los churners capturada en el top k."""
    y_true = np.asarray(y_true)
    n = max(1, int(len(y_true) * k))
    idx = np.argsort(y_prob)[::-1][:n]
    total = y_true.sum()
    return float(y_true[idx].sum() / total) if total > 0 else 0.0


def evaluate(y_true, y_prob, prefix: str = "") -> dict[str, float]:
    """Panel de métricas. PR-AUC es la principal; lift/capture son el KPI
    que le importa al negocio."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    m = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "base_rate": float(y_true.mean()),
    }
    for k in (0.05, 0.10, 0.20):
        m[f"lift@{int(k*100)}"] = lift_at_k(y_true, y_prob, k)
        m[f"capture@{int(k*100)}"] = capture_at_k(y_true, y_prob, k)
    return {f"{prefix}{k}": v for k, v in m.items()}


def decile_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de deciles: la forma en que retención va a leer el modelo."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decil"] = (np.arange(len(df)) * n_bins // len(df)) + 1

    base = df["y"].mean()
    out = (df.groupby("decil")
           .agg(n=("y", "size"), churners=("y", "sum"),
                tasa_churn=("y", "mean"), prob_media=("p", "mean"),
                prob_min=("p", "min"))
           .reset_index())
    out["lift"] = out["tasa_churn"] / base if base > 0 else 0
    out["capture_%"] = 100 * out["churners"].cumsum() / max(df["y"].sum(), 1)
    for c in ("tasa_churn", "prob_media", "prob_min", "lift", "capture_%"):
        out[c] = out[c].round(4)
    return out


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Probabilidad predicha vs. observada. Si el modelo alimenta un cálculo
    de valor esperado, esta tabla tiene que estar cerca de la diagonal."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    try:
        df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop", labels=False)
    except ValueError:
        df["bin"] = 0
    out = (df.groupby("bin")
           .agg(n=("y", "size"), prob_predicha=("p", "mean"),
                tasa_observada=("y", "mean")).reset_index())
    out["gap"] = (out["prob_predicha"] - out["tasa_observada"]).round(4)
    return out.round(4)


def calibrate(estimator, X_cal, y_cal, method: str = "isotonic"):
    """Calibración post-hoc sobre datos no usados en entrenamiento.

    El estimador ya está entrenado; aquí solo se ajusta el mapeo de
    probabilidad. En sklearn ≥1.6 eso se expresa con `FrozenEstimator`
    (`cv="prefit"` quedó deprecado y desaparece en 1.8).
    """
    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except ImportError:
        cal = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def summarize(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Ordena los resultados de varios modelos por PR-AUC."""
    df = pd.DataFrame(results).T
    cols = [c for c in ["pr_auc", "roc_auc", "lift@5", "lift@10", "capture@10",
                        "capture@20", "brier", "base_rate"] if c in df.columns]
    return df[cols].sort_values("pr_auc", ascending=False).round(4)
