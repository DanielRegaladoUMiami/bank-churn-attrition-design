"""Explicabilidad: SHAP global/local y contrafactuales accionables.

Por qué SHAP y no LIME
----------------------
LIME reajusta un modelo lineal local con muestreo aleatorio: dos corridas sobre
el mismo cliente dan explicaciones distintas. En un contexto donde la
explicación puede terminar en un expediente de model risk, esa inestabilidad es
inaceptable. `TreeExplainer` de SHAP es exacto y determinista sobre GBMs, y da
lo local y lo global con la misma base axiomática.

Lo que SHAP no da
-----------------
SHAP dice *qué pesó*, no *qué hacer*. Para retención lo accionable es:
"este cliente baja N puntos de riesgo si recupera la nómina". Eso es un
contrafactual, y está en `counterfactual_analysis()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


# ── Utilidades de pipeline ──────────────────────────────────────────────

def _split_pipeline(estimator):
    """Separa (preprocesador, modelo final) de un Pipeline de sklearn."""
    if hasattr(estimator, "named_steps"):
        steps = list(estimator.named_steps.items())
        pre = estimator[:-1] if len(steps) > 1 else None
        model = steps[-1][1]
        return pre, model
    return None, estimator


def _transformed_frame(estimator, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica el preprocesador y devuelve un DataFrame con nombres de columna."""
    pre, _ = _split_pipeline(estimator)
    if pre is None:
        return X
    Xt = pre.transform(X)
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return pd.DataFrame(np.asarray(Xt), columns=names, index=X.index)


# ── SHAP ────────────────────────────────────────────────────────────────

def shap_analysis(estimator, X: pd.DataFrame, cfg: Config | None = None,
                  sample_size: int | None = None, verbose: bool = True):
    """Calcula valores SHAP. Devuelve (shap_values, X_transformado).

    Usa TreeExplainer cuando el modelo final es de árboles (exacto y rápido);
    si no, cae a un explainer genérico con un background pequeño.
    """
    import shap

    n = sample_size or (cfg.explain.get("shap_sample_size", 2000) if cfg else 2000)
    Xs = X.sample(min(n, len(X)), random_state=42) if len(X) > n else X
    Xt = _transformed_frame(estimator, Xs)
    _, model = _split_pipeline(estimator)

    tree_like = any(k in type(model).__name__.lower()
                    for k in ("forest", "lgbm", "xgb", "catboost", "boosting", "tree"))
    if verbose:
        print(f"🔍 SHAP sobre {len(Xt):,} filas "
              f"({'TreeExplainer' if tree_like else 'Explainer genérico'})")

    if tree_like:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
    else:
        bg = shap.sample(Xt, min(100, len(Xt)), random_state=42)
        explainer = shap.Explainer(model.predict_proba, bg)
        sv = explainer(Xt).values

    sv = np.asarray(sv)
    # Normalizar la forma: algunos backends devuelven (n, f, 2) o lista por clase
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]
    return sv, Xt


def global_importance(shap_values: np.ndarray, Xt: pd.DataFrame,
                      top_n: int = 25) -> pd.DataFrame:
    """Importancia global: |SHAP| medio por feature, con la dirección del efecto."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Correlación entre el valor de la feature y su contribución = dirección
    direction = []
    for i, c in enumerate(Xt.columns):
        col = Xt[c].to_numpy(dtype=float)
        if np.std(col) < 1e-9:
            direction.append(0.0)
        else:
            direction.append(float(np.corrcoef(col, shap_values[:, i])[0, 1]))

    out = pd.DataFrame({
        "feature": Xt.columns,
        "shap_abs_mean": mean_abs,
        "direccion": np.round(direction, 3),
    }).sort_values("shap_abs_mean", ascending=False).head(top_n)
    out["efecto"] = np.where(out["direccion"] > 0.05, "↑ sube riesgo",
                     np.where(out["direccion"] < -0.05, "↓ baja riesgo", "no lineal"))
    out["shap_abs_mean"] = out["shap_abs_mean"].round(5)
    return out.reset_index(drop=True)


def explain_customer(estimator, X: pd.DataFrame, row_index, top_n: int = 10,
                     shap_values: np.ndarray | None = None,
                     Xt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Explicación local: los N drivers del score de un cliente concreto."""
    if shap_values is None or Xt is None:
        shap_values, Xt = shap_analysis(estimator, X.loc[[row_index]], verbose=False)
        pos = 0
    else:
        if row_index not in Xt.index:
            raise KeyError(f"{row_index} no está en la muestra SHAP; "
                           "recalcula con esa fila incluida.")
        pos = Xt.index.get_loc(row_index)

    contrib = shap_values[pos]
    out = pd.DataFrame({
        "feature": Xt.columns,
        "valor": Xt.iloc[pos].to_numpy(),
        "contribucion_shap": contrib,
    })
    out["abs"] = out["contribucion_shap"].abs()
    out = out.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")
    out["direccion"] = np.where(out["contribucion_shap"] > 0, "↑ riesgo", "↓ riesgo")
    return out.reset_index(drop=True).round(4)


# ── Contrafactuales ─────────────────────────────────────────────────────

def counterfactual_analysis(estimator, X: pd.DataFrame, cfg: Config,
                            features: list[str] | None = None,
                            top_pct: float = 0.10,
                            verbose: bool = True) -> pd.DataFrame:
    """¿Cuánto baja el riesgo si se mueve una palanca accionable?

    Para cada feature accionable, simula la intervención sobre los clientes de
    mayor riesgo y mide la caída promedio en probabilidad. Es la traducción de
    "el modelo dice X" a "retención puede hacer Y".
    """
    feats = features or cfg.explain.get("counterfactual_features", [])
    feats = [f for f in feats if f in X.columns]
    if not feats:
        return pd.DataFrame(columns=["feature", "intervencion", "delta_riesgo_medio"])

    base_p = estimator.predict_proba(X)[:, 1]
    k = max(1, int(len(X) * top_pct))
    at_risk = np.argsort(base_p)[::-1][:k]
    Xr = X.iloc[at_risk]
    base_r = base_p[at_risk]

    rows = []
    for f in feats:
        col = X[f]
        if pd.api.types.is_bool_dtype(col) or set(col.dropna().unique()) <= {0, 1}:
            target, label = 1.0, f"activar {f}"
        else:
            target = float(col.quantile(0.75))
            label = f"subir {f} al p75 ({target:.1f})"

        Xc = Xr.copy()
        Xc[f] = target
        new_p = estimator.predict_proba(Xc)[:, 1]
        delta = float((new_p - base_r).mean())
        rows.append({
            "feature": f,
            "intervencion": label,
            "riesgo_antes": round(float(base_r.mean()), 4),
            "riesgo_despues": round(float(new_p.mean()), 4),
            "delta_riesgo_medio": round(delta, 4),
            "reduccion_%": round(-100 * delta / max(base_r.mean(), 1e-9), 1),
        })

    out = pd.DataFrame(rows).sort_values("delta_riesgo_medio")
    if verbose:
        print(f"\n🎯 Contrafactuales sobre el top {top_pct:.0%} de riesgo "
              f"({k:,} clientes)")
    return out.reset_index(drop=True)


def sanity_check(importance: pd.DataFrame, expected_top: list[str] | None = None,
                 verbose: bool = True) -> dict[str, bool]:
    """Valida el SHAP contra intuición de negocio.

    Si aparece un identificador o una variable geográfica arriba, casi siempre
    es un proxy de algo — o leakage. Vale más detectarlo aquí que en el comité
    de validación.
    """
    top = importance["feature"].head(10).str.lower().tolist()
    suspicious = [f for f in top
                  if any(k in f for k in ("branch", "zip", "state", "_id",
                                          "cif", "account", "household"))]
    checks = {
        "sin_proxies_geograficos_o_ids_en_top10": len(suspicious) == 0,
    }
    if expected_top:
        found = [e for e in expected_top if any(e.lower() in t for t in top)]
        checks["recupera_señales_esperadas"] = len(found) > 0
        if verbose and found:
            print(f"✓ Señales esperadas en el top-10: {found}")
    if verbose and suspicious:
        print(f"⚠️  Features sospechosas en el top-10 (posible proxy o leakage): "
              f"{suspicious}")
    return checks
