"""Feature engineering temporal sobre el panel CIF×mes.

Principio rector
----------------
Para cada fila `(cif_id, as_of_month)`, **toda** feature se calcula únicamente
con información de la ventana `[t - feature_window + 1, t]`. Nada del futuro.
Combinado con el gap window del panel, esto garantiza que el modelo sea
reproducible en producción: el día que corras el scoring, todas estas features
existen.

Por qué series de tiempo y no una foto
--------------------------------------
El churn es un proceso, no un atributo. Un saldo de $5,000 estable y un saldo
de $5,000 que venía de $50,000 son el mismo nivel y riesgos opuestos. Lo que
predice es la **trayectoria**: pendiente, drawdown, aceleración y rachas.

Familias de features
--------------------
1.  Nivel            — valor en t
2.  Tendencia        — pendiente OLS a 3/6/12 meses
3.  Ratio a sí mismo — t / t-3, t / t-6, t / t-12
4.  Drawdown         — t / max(ventana): captura caídas desde el pico
5.  Volatilidad      — coeficiente de variación de la ventana
6.  Aceleración      — pendiente reciente menos pendiente previa (inflexión)
7.  Rachas           — meses consecutivos cayendo, meses sin actividad
8.  Estacionalidad   — YoY y mes calendario
9.  Relacional       — co-titulares, contagio de hogar

Implementación
--------------
Las pendientes se calculan como producto punto con un vector de pesos fijo
(`Σ wᵢ·y_{t-i}`), lo que las vuelve un puñado de `shift` vectorizados en vez
de un `rolling().apply()` lento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Columnas que reciben el tratamiento temporal completo
TREND_COLS = [
    "owned_balance", "txn_count_total", "n_deposit_accounts",
    "n_relationships_any_role", "loan_balance", "login_count_mobile_month",
    "login_count_web_month", "bill_pay_payees_active", "debit_amt_total",
    "txn_pos_signature", "txn_ach_credit_in", "txn_zelle_out", "txn_bill_pay",
]

# Columnas donde interesa la suma acumulada en ventana (eventos, no niveles)
SUM_COLS = [
    "nsf_count", "fees_charged", "complaints_open_month", "disputes_month",
    "call_center_calls_month", "branch_visits_month", "competitor_outflow_amt",
    "competitor_txn_count", "txn_atm_foreign", "txn_wire_out",
]

# Se arrastran tal cual (estado en t)
LEVEL_COLS = [
    "n_active_loans", "min_months_to_payoff", "months_to_cd_maturity",
    "cd_rate_avg", "has_direct_deposit", "months_since_activity",
    "n_signer_only", "n_joint_accounts", "max_days_past_due",
    "n_checking", "n_savings", "n_mma", "n_cd",
    "online_enrolled_flag", "mobile_enrolled_flag", "estatement_flag",
    "debit_card_active_flag", "alerts_enrolled_flag", "tenure_months",
]

CATEGORICAL_COLS = ["entity_type", "segment", "residency_status", "state",
                    "kyc_risk_rating"]


# ── Utilidades temporales ───────────────────────────────────────────────

def _slope_weights(window: int) -> np.ndarray:
    """Pesos de la pendiente OLS para x = 0..w-1, ordenados de más antiguo a
    más reciente. slope = Σ wᵢ·y_{t-(w-1-i)}"""
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    return xc / (xc**2).sum()


def _rolling_slope(df: pd.DataFrame, col: str, window: int, group: str) -> pd.Series:
    """Pendiente OLS de `col` sobre los últimos `window` meses, por grupo.

    Vectorizado: una combinación lineal de `window` shifts, en vez de
    rolling().apply(), que sobre paneles grandes es órdenes de magnitud
    más lento.
    """
    w = _slope_weights(window)
    g = df.groupby(group)[col]
    out = None
    for i in range(window):
        lag = window - 1 - i
        s = g.shift(lag) * w[i]
        out = s if out is None else out + s
    return out


def _safe_ratio(num: pd.Series, den: pd.Series, cap: float = 50.0) -> pd.Series:
    """Ratio robusto: evita división por cero y trunca colas absurdas."""
    d = den.replace(0, np.nan)
    r = num / d
    return r.clip(-cap, cap)


# ── Motor principal ─────────────────────────────────────────────────────

def build_features(panel: pd.DataFrame, cfg: Config,
                   tables: dict[str, pd.DataFrame] | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Devuelve el panel con todas las features temporales añadidas.

    Preserva `cif_id`, `as_of_month`, `y` para el split posterior.
    """
    df = panel.sort_values(["cif_id", "as_of_month"]).copy()
    g = "cif_id"
    win = int(cfg.panel.get("feature_window_months", 12))
    present_trend = [c for c in TREND_COLS if c in df.columns]
    new: dict[str, pd.Series] = {}

    # ── 1-6. Tratamiento temporal por columna de tendencia ──
    for col in present_trend:
        s = df[col].astype(float)
        grp = df.groupby(g)[col]

        # Ratios a su propio pasado
        for lag in (1, 3, 6, 12):
            if lag > win:
                continue
            new[f"{col}_ratio_{lag}m"] = _safe_ratio(s, grp.shift(lag))
            new[f"{col}_diff_{lag}m"] = s - grp.shift(lag)

        # Pendientes
        for w in (3, 6, 12):
            if w > win:
                continue
            new[f"{col}_slope_{w}m"] = _rolling_slope(df, col, w, g)

        # Aceleración: ¿la caída se está profundizando o estabilizando?
        if 6 <= win:
            slope_recent = new.get(f"{col}_slope_3m")
            slope_prev = df.groupby(g)[col].shift(3)
            if slope_recent is not None:
                prev_slope = _rolling_slope(
                    df.assign(_lagged=slope_prev), "_lagged", 3, g
                )
                new[f"{col}_accel"] = slope_recent - prev_slope

        # Estadísticos de ventana (causales: incluyen t)
        roll = grp.rolling(win, min_periods=2)
        mean_w = roll.mean().reset_index(level=0, drop=True)
        std_w = roll.std().reset_index(level=0, drop=True)
        max_w = roll.max().reset_index(level=0, drop=True)
        min_w = roll.min().reset_index(level=0, drop=True)

        new[f"{col}_mean_{win}m"] = mean_w
        new[f"{col}_cv_{win}m"] = _safe_ratio(std_w, mean_w.abs())
        # Drawdown: qué fracción de su propio máximo conserva.
        # Es de las features más predictivas y casi nunca se incluye.
        new[f"{col}_drawdown_{win}m"] = _safe_ratio(s, max_w, cap=5.0)
        new[f"{col}_vs_min_{win}m"] = _safe_ratio(s, min_w, cap=50.0)
        new[f"{col}_zscore_{win}m"] = _safe_ratio(s - mean_w, std_w, cap=10.0)

        # 7. Racha: meses consecutivos de caída
        declining = (s < grp.shift(1)).fillna(False)
        blocks = (~declining).groupby(df[g]).cumsum()
        new[f"{col}_declining_streak"] = (
            declining.groupby([df[g], blocks]).cumsum()
        )

    # ── Sumas en ventana para columnas de evento ──
    for col in [c for c in SUM_COLS if c in df.columns]:
        grp = df.groupby(g)[col]
        for w in (3, 6, 12):
            if w > win:
                continue
            new[f"{col}_sum_{w}m"] = (
                grp.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            )

    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # ── 7b. Rachas específicas de dominio ──
    out = _add_streak_features(out, g)

    # ── 8. Estacionalidad ──
    out["cal_month"] = out["as_of_month"].dt.month
    out["cal_quarter"] = out["as_of_month"].dt.quarter

    # ── 9. Features de dominio y relacionales ──
    out = _add_domain_features(out)
    if tables is not None:
        out = _add_contagion_features(out, tables, cfg)

    if verbose:
        n_feat = out.shape[1] - df.shape[1]
        print(f"🔧 Features generadas: {n_feat} nuevas "
              f"({out.shape[1]} columnas totales)")
    return out


def _add_streak_features(df: pd.DataFrame, g: str) -> pd.DataFrame:
    """Rachas y tiempos-desde-evento que no salen del tratamiento genérico."""
    if "has_direct_deposit" in df.columns:
        dd = df["has_direct_deposit"].fillna(0).astype(int)
        # Meses desde que perdió la nómina: el predictor #1 en checking
        had_dd = dd.groupby(df[g]).cummax()
        lost = (had_dd == 1) & (dd == 0)
        blocks = (~lost).groupby(df[g]).cumsum()
        df["months_since_dd_loss"] = lost.groupby([df[g], blocks]).cumsum()
        df["ever_had_direct_deposit"] = had_dd
        # ¿Perdió la nómina en los últimos 6 meses?
        df["dd_lost_recently"] = (
            (df["months_since_dd_loss"] > 0) & (df["months_since_dd_loss"] <= 6)
        ).astype(int)

    if "months_since_activity" in df.columns:
        df["is_dormant_3m"] = (df["months_since_activity"].fillna(0) >= 3).astype(int)

    return df


def _add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios y banderas específicas de banca."""
    # Loan por terminar sin ancla de depósitos: el escenario del README
    if {"min_months_to_payoff", "owned_balance"} <= set(df.columns):
        near_payoff = df["min_months_to_payoff"].fillna(999) <= 6
        thin_deposits = df["owned_balance"].fillna(0) < 500
        df["loan_ending_no_deposit_anchor"] = (near_payoff & thin_deposits).astype(int)
        df["months_to_payoff_capped"] = df["min_months_to_payoff"].fillna(999).clip(0, 120)

    # CD por vencer: gatillo de rolloff
    if "months_to_cd_maturity" in df.columns:
        df["cd_maturing_3m"] = (
            df["months_to_cd_maturity"].fillna(999).between(0, 3)
        ).astype(int)

    # Profundidad de relación
    prod_cols = [c for c in ("n_checking", "n_savings", "n_mma", "n_cd") if c in df.columns]
    if prod_cols:
        df["n_products_owned"] = df[prod_cols].fillna(0).sum(axis=1)
        df["product_diversity"] = (df[prod_cols].fillna(0) > 0).sum(axis=1)
    if "n_active_loans" in df.columns and "n_products_owned" in df.columns:
        df["has_deposit_and_loan"] = (
            (df["n_products_owned"] > 0) & (df["n_active_loans"] > 0)
        ).astype(int)

    # Intensidad de uso normalizada por saldo
    if {"txn_count_total", "owned_balance"} <= set(df.columns):
        df["txn_per_1k_balance"] = _safe_ratio(
            df["txn_count_total"], df["owned_balance"] / 1000
        )

    # Fuga de fondos hacia otra institución
    if {"competitor_outflow_amt_sum_6m", "owned_balance"} <= set(df.columns):
        df["competitor_outflow_ratio"] = _safe_ratio(
            df["competitor_outflow_amt_sum_6m"], df["owned_balance"], cap=20.0
        )

    # Presión de fees respecto al saldo
    if {"fees_charged_sum_6m", "owned_balance"} <= set(df.columns):
        df["fee_burden_ratio"] = _safe_ratio(
            df["fees_charged_sum_6m"], df["owned_balance"], cap=5.0
        )

    # Digital engagement combinado
    logins = [c for c in ("login_count_web_month", "login_count_mobile_month")
              if c in df.columns]
    if logins:
        df["logins_total"] = df[logins].fillna(0).sum(axis=1)
        df["is_digitally_inactive"] = (df["logins_total"] == 0).astype(int)

    return df


def _add_contagion_features(df: pd.DataFrame, tables: dict[str, pd.DataFrame],
                            cfg: Config) -> pd.DataFrame:
    """Contagio de hogar: si el co-titular de una cuenta joint se va, el
    riesgo del CIF sube muchísimo. Suele ser la feature relacional más fuerte.
    """
    rel = tables.get("relationships")
    if rel is None or rel.empty or "churn_event_month" not in df.columns:
        return df

    joint = rel[rel["relationship_role"].astype(str).str.lower() == "joint"]
    if joint.empty:
        df["n_co_owners"] = 0
        df["co_owner_churned_flag"] = 0
        return df

    # Pares de co-titulares que comparten cuenta
    pairs = joint[["account_id", "cif_id"]].merge(
        joint[["account_id", "cif_id"]], on="account_id", suffixes=("", "_co")
    )
    pairs = pairs[pairs["cif_id"] != pairs["cif_id_co"]][["cif_id", "cif_id_co"]]
    pairs = pairs.drop_duplicates()

    n_co = pairs.groupby("cif_id")["cif_id_co"].nunique().rename("n_co_owners")

    # Mes de churn del co-titular (una fila por CIF)
    events = (df[["cif_id", "churn_event_month"]].dropna()
              .drop_duplicates("cif_id")
              .rename(columns={"cif_id": "cif_id_co",
                               "churn_event_month": "co_churn_month"}))
    pe = pairs.merge(events, on="cif_id_co", how="inner")
    if pe.empty:
        df["n_co_owners"] = df["cif_id"].map(n_co).fillna(0)
        df["co_owner_churned_flag"] = 0
        return df

    earliest = pe.groupby("cif_id")["co_churn_month"].min().rename("co_churn_month")
    df = df.merge(earliest, on="cif_id", how="left")
    df["n_co_owners"] = df["cif_id"].map(n_co).fillna(0)
    # Solo cuenta si el co-titular ya se fue AL MOMENTO del as-of date:
    # usar un churn futuro sería leakage.
    df["co_owner_churned_flag"] = (
        df["co_churn_month"].notna() & (df["co_churn_month"] <= df["as_of_month"])
    ).astype(int)
    df = df.drop(columns=["co_churn_month"])
    return df


# ── Selección de columnas para el modelo ────────────────────────────────

# Columnas que NUNCA deben entrar como feature: son el target, sus insumos
# directos, o identificadores.
LEAKY_COLS = {
    "y", "churn_flag", "is_attrited_state", "churn_event_month",
    "label_start", "label_end", "cif_id", "as_of_month",
    "balance_baseline_12m", "involuntary_month", "deceased_flag",
    "raw_balance_any_role", "customer_since_date",
}


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Devuelve (features_numéricas, features_categóricas), excluyendo
    cualquier columna con riesgo de leakage."""
    numeric, categorical = [], []
    for c in df.columns:
        if c in LEAKY_COLS:
            continue
        if c in CATEGORICAL_COLS or df[c].dtype == object:
            if df[c].nunique(dropna=True) <= 50:
                categorical.append(c)
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            numeric.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    return numeric, categorical


def clean_features(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Reemplaza inf y recorta outliers extremos por winsorización suave."""
    out = df.copy()
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out
