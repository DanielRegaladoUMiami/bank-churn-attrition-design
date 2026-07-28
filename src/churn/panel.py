"""Construcción del panel CIF×mes, pesos de ownership y etiquetado del target.

Este es el módulo donde se materializan las decisiones de diseño del README:

* Los saldos se atribuyen al CIF **ponderados por rol** — un joint de $50k con
  dos titulares aporta $25k a cada uno, no $50k. Un authorized signer aporta $0
  pero sí cuenta como ancla relacional.
* El target es **compuesto**: cierre formal O (saldo bajo + sin préstamos +
  inactividad), sostenido N meses. Captura la attrition silenciosa.
* Entre el as-of date y la ventana de performance hay un **gap** que hace el
  modelo accionable y evita etiquetar con información ya contemporánea al evento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Roles que cuentan como préstamo (para exposición crediticia)
LOAN_ROLES = {"borrower", "co_borrower", "guarantor"}


# ── Pesos de ownership ──────────────────────────────────────────────────

def attach_ownership_weights(relationships: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Añade `ownership_weight` a cada fila (cif, account, role).

    Peso `null` en config = repartir 1/n_owners entre los titulares, que es lo
    que evita el doble conteo de saldos en cuentas joint.
    """
    rel = relationships.copy()
    weights = cfg.ownership_weights
    rel["relationship_role"] = rel["relationship_role"].astype(str).str.lower().str.strip()

    if "n_owners" not in rel.columns or rel["n_owners"].isna().all():
        # Derivar de los datos si el extracto no lo trae
        owner_roles = {r for r, w in weights.items() if w is None or w > 0}
        counts = (rel[rel["relationship_role"].isin(owner_roles)]
                  .groupby("account_id")["cif_id"].nunique().rename("n_owners"))
        rel = rel.drop(columns=[c for c in ["n_owners"] if c in rel.columns])
        rel = rel.merge(counts, on="account_id", how="left")
    rel["n_owners"] = rel["n_owners"].fillna(1).clip(lower=1)

    explicit = rel["relationship_role"].map(
        {k: v for k, v in weights.items() if v is not None}
    )
    shared = 1.0 / rel["n_owners"]
    # Rol conocido con peso explícito → ese peso.
    # Rol conocido con peso null → 1/n_owners.
    # Rol desconocido → 0 (conservador: no inventamos exposición).
    known_null = rel["relationship_role"].isin(
        [k for k, v in weights.items() if v is None]
    )
    rel["ownership_weight"] = np.where(
        explicit.notna(), explicit.fillna(0.0),
        np.where(known_null, shared, 0.0),
    )
    return rel


def _active_relationships(rel: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Expande relaciones a (cif, account, role, mes) para los meses vigentes."""
    grid = pd.DataFrame({"as_of_month": months, "_k": 1})
    rel = rel.assign(_k=1)
    out = rel.merge(grid, on="_k").drop(columns="_k")
    start_ok = out["relationship_start_date"].isna() | \
               (out["relationship_start_date"] <= out["as_of_month"])
    end_ok = out["relationship_end_date"].isna() | \
             (out["relationship_end_date"] > out["as_of_month"])
    return out[start_ok & end_ok].reset_index(drop=True)


# ── Estado mensual por CIF ──────────────────────────────────────────────

def build_cif_month_state(tables: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Agrega todo a una fila por (cif_id, as_of_month) con el estado de la
    relación: saldos ponderados, productos, préstamos, actividad."""
    deposits = tables["deposits"]
    months = pd.DatetimeIndex(sorted(deposits["as_of_month"].unique()))

    rel = attach_ownership_weights(tables["relationships"], cfg)
    rel_m = _active_relationships(rel, months)

    # ── Depósitos ponderados por ownership ──
    dep_cols = ["account_id", "as_of_month", "eom_balance", "product_family"]
    for c in ("nsf_count_month", "fees_charged_month", "account_status",
              "maturity_date", "interest_rate_paid", "close_reason_code"):
        if c in deposits.columns:
            dep_cols.append(c)
    dep = deposits[dep_cols]

    dm = rel_m[rel_m["product_type"] == "deposit"].merge(
        dep, on=["account_id", "as_of_month"], how="inner"
    )
    dm["owned_balance"] = dm["eom_balance"] * dm["ownership_weight"]
    dm["is_owner"] = dm["ownership_weight"] > 0

    agg = {
        "owned_balance": ("owned_balance", "sum"),
        "raw_balance_any_role": ("eom_balance", "sum"),
        "n_deposit_accounts": ("account_id", "nunique"),
    }
    if "nsf_count_month" in dm.columns:
        agg["nsf_count"] = ("nsf_count_month", "sum")
    if "fees_charged_month" in dm.columns:
        agg["fees_charged"] = ("fees_charged_month", "sum")

    dep_state = dm[dm["is_owner"]].groupby(["cif_id", "as_of_month"]).agg(**agg).reset_index()

    # Conteos por familia de producto (solo donde hay ownership)
    fam = (dm[dm["is_owner"]]
           .pivot_table(index=["cif_id", "as_of_month"], columns="product_family",
                        values="account_id", aggfunc="nunique", fill_value=0)
           .add_prefix("n_").reset_index())
    dep_state = dep_state.merge(fam, on=["cif_id", "as_of_month"], how="left")

    # Relacional: en cuántas cuentas aparece con CUALQUIER rol (incl. signer)
    rel_state = (rel_m.groupby(["cif_id", "as_of_month"])
                 .agg(n_relationships_any_role=("account_id", "nunique"),
                      n_signer_only=("relationship_role",
                                     lambda s: int((s == "authorized_signer").sum())),
                      n_joint_accounts=("relationship_role",
                                        lambda s: int((s == "joint").sum())))
                 .reset_index())

    # ── Espina del panel: el maestro de clientes, NO las relaciones vigentes ──
    # Si se parte de las relaciones, el cliente que cierra todos sus productos
    # simplemente desaparece del panel en vez de aparecer con ceros, y la
    # condición "sin relaciones" del target se vuelve inalcanzable. El maestro
    # de clientes sí conserva la fila mes a mes.
    party_all = tables["party"]
    spine = party_all[["cif_id", "as_of_month"]].drop_duplicates()
    spine = spine[spine["as_of_month"].isin(months)]

    state = (spine
             .merge(rel_state, on=["cif_id", "as_of_month"], how="left")
             .merge(dep_state, on=["cif_id", "as_of_month"], how="left"))

    # CD próximo a vencer: gatillo clásico de rolloff
    if "maturity_date" in dm.columns:
        cd = dm[(dm["is_owner"]) & (dm["product_family"].astype(str) == "cd")].copy()
        if len(cd):
            cd["months_to_cd_maturity"] = (
                (cd["maturity_date"].dt.to_period("M").astype("int64")
                 - cd["as_of_month"].dt.to_period("M").astype("int64"))
            )
            cdg = (cd.groupby(["cif_id", "as_of_month"])
                   .agg(months_to_cd_maturity=("months_to_cd_maturity", "min"),
                        cd_rate_avg=("interest_rate_paid", "mean")).reset_index())
            state = state.merge(cdg, on=["cif_id", "as_of_month"], how="left")

    # ── Préstamos ──
    state = _add_loan_state(state, tables, rel_m, cfg)

    # ── Actividad transaccional ──
    state = _add_activity_state(state, tables, rel_m)

    # ── Digital ──
    digital = tables.get("digital")
    if digital is not None and not digital.empty:
        dcols = [c for c in digital.columns if c not in ("cif_id", "as_of_month")]
        state = state.merge(digital[["cif_id", "as_of_month"] + dcols],
                            on=["cif_id", "as_of_month"], how="left")

    # ── Atributos del cliente ──
    party = tables["party"]
    pcols = [c for c in ["entity_type", "segment", "residency_status", "state",
                         "tenure_months", "employee_flag", "deceased_flag",
                         "kyc_risk_rating", "customer_since_date", "household_id"]
             if c in party.columns]
    state = state.merge(party[["cif_id", "as_of_month"] + pcols],
                        on=["cif_id", "as_of_month"], how="left")

    num = state.select_dtypes(include=[np.number]).columns
    state[num] = state[num].fillna(0)
    return state.sort_values(["cif_id", "as_of_month"]).reset_index(drop=True)


def _add_loan_state(state, tables, rel_m, cfg) -> pd.DataFrame:
    loans = tables.get("loans")
    if loans is None or loans.empty:
        state["n_active_loans"] = 0
        state["loan_balance"] = 0.0
        state["min_months_to_payoff"] = np.nan
        return state

    lm = rel_m[rel_m["relationship_role"].isin(LOAN_ROLES)].merge(
        loans.rename(columns={"loan_id": "account_id"}),
        on=["account_id", "as_of_month"], how="inner",
    )
    if lm.empty:
        state["n_active_loans"] = 0
        state["loan_balance"] = 0.0
        state["min_months_to_payoff"] = np.nan
        return state

    lm["is_active"] = (lm.get("status", pd.Series("active", index=lm.index))
                       .astype(str).eq("active")) & (lm["current_balance"] > 0)
    lm["owned_loan_balance"] = lm["current_balance"] * lm["ownership_weight"].clip(lower=0)

    # Meses hasta liquidar según amortización: el gatillo de fuga que
    # describe el README (loan por terminar + sin depósitos activos).
    if "months_to_maturity" in lm.columns:
        lm["months_to_payoff"] = lm["months_to_maturity"]
    elif "remaining_term_months" in lm.columns:
        lm["months_to_payoff"] = lm["remaining_term_months"]
    else:
        lm["months_to_payoff"] = np.nan

    agg = (lm.groupby(["cif_id", "as_of_month"]).agg(
        n_active_loans=("is_active", "sum"),
        loan_balance=("owned_loan_balance", "sum"),
        min_months_to_payoff=("months_to_payoff", "min"),
        max_days_past_due=("days_past_due", "max") if "days_past_due" in lm.columns
                          else ("is_active", "size"),
    ).reset_index())

    return state.merge(agg, on=["cif_id", "as_of_month"], how="left")


def _add_activity_state(state, tables, rel_m) -> pd.DataFrame:
    txns = tables.get("transactions")
    if txns is None or txns.empty:
        state["txn_count_total"] = np.nan
        state["months_since_activity"] = np.nan
        state["has_direct_deposit"] = np.nan
        state["competitor_outflow_amt"] = np.nan
        return state

    owners = rel_m[rel_m["ownership_weight"] > 0][
        ["cif_id", "account_id", "as_of_month"]
    ].drop_duplicates()
    tx = owners.merge(txns, on=["account_id", "as_of_month"], how="inner")

    aggs = {"txn_count_total": ("txn_count", "sum")}
    if "total_amount_debit" in tx.columns:
        aggs["debit_amt_total"] = ("total_amount_debit", "sum")
    if "total_amount_credit" in tx.columns:
        aggs["credit_amt_total"] = ("total_amount_credit", "sum")
    g = tx.groupby(["cif_id", "as_of_month"]).agg(**aggs).reset_index()

    if "is_direct_deposit" in tx.columns:
        dd = (tx[tx["is_direct_deposit"].astype("boolean").fillna(False)]
              .groupby(["cif_id", "as_of_month"])["txn_count"].sum()
              .rename("dd_txn_count").reset_index())
        g = g.merge(dd, on=["cif_id", "as_of_month"], how="left")
        g["dd_txn_count"] = g["dd_txn_count"].fillna(0)
        g["has_direct_deposit"] = (g["dd_txn_count"] > 0).astype(int)

    if "competitor_transfer_flag" in tx.columns:
        cp = (tx[tx["competitor_transfer_flag"].astype("boolean").fillna(False)]
              .groupby(["cif_id", "as_of_month"])
              .agg(competitor_outflow_amt=("total_amount_debit", "sum"),
                   competitor_txn_count=("txn_count", "sum")).reset_index())
        g = g.merge(cp, on=["cif_id", "as_of_month"], how="left")
        g[["competitor_outflow_amt", "competitor_txn_count"]] = \
            g[["competitor_outflow_amt", "competitor_txn_count"]].fillna(0)

    # Categorías principales en columnas separadas
    top = ["ach_credit_in", "ach_debit_out", "zelle_out", "pos_signature",
           "bill_pay", "atm_foreign", "wire_out", "branch_deposit", "mobile_deposit"]
    piv = (tx[tx["txn_category"].isin(top)]
           .pivot_table(index=["cif_id", "as_of_month"], columns="txn_category",
                        values="txn_count", aggfunc="sum", fill_value=0)
           .add_prefix("txn_").reset_index())
    g = g.merge(piv, on=["cif_id", "as_of_month"], how="left")

    state = state.merge(g, on=["cif_id", "as_of_month"], how="left")
    state["txn_count_total"] = state["txn_count_total"].fillna(0)

    # Meses consecutivos sin actividad iniciada por el cliente
    state = state.sort_values(["cif_id", "as_of_month"])
    active = state["txn_count_total"] > 0
    grp = active.groupby(state["cif_id"]).cumsum()
    state["months_since_activity"] = (
        state.groupby(["cif_id", grp]).cumcount().where(~active, 0)
    )
    return state


# ── Target ──────────────────────────────────────────────────────────────

def label_attrition_state(state: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Marca, mes a mes, si el CIF está en estado de attrition, y deriva el
    mes del evento (primer mes de una racha sostenida)."""
    t = cfg.target
    df = state.sort_values(["cif_id", "as_of_month"]).copy()

    # Baseline propio: promedio de saldo de los 12 meses previos (shift para
    # no usar el mes corriente).
    df["balance_baseline_12m"] = (
        df.groupby("cif_id")["owned_balance"]
          .transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
    )
    thresh = np.maximum(
        float(t.get("low_balance_abs", 100.0)),
        df["balance_baseline_12m"].fillna(0) * float(t.get("low_balance_pct_of_baseline", 0.05)),
    )
    low_balance = df["owned_balance"] < thresh
    no_loans = df.get("n_active_loans", pd.Series(0, index=df.index)).fillna(0) <= 0

    inactivity_months = max(1, int(round(float(t.get("inactivity_days", 90)) / 30)))
    if df["months_since_activity"].notna().any():
        inactive = df["months_since_activity"].fillna(0) >= inactivity_months
    else:
        inactive = pd.Series(True, index=df.index)  # sin datos de txn: no restringe

    no_relationship = df["n_relationships_any_role"].fillna(0) <= 0

    df["is_attrited_state"] = (low_balance & no_loans & inactive) | no_relationship

    # Confirmar con persistencia de N meses consecutivos
    k = int(t.get("persistence_months", 2))
    if k > 1:
        rolled = (df.groupby("cif_id")["is_attrited_state"]
                  .transform(lambda s: s.rolling(k, min_periods=k).sum()))
        confirmed = rolled >= k
        # El evento se fecha en el PRIMER mes de la racha, no en el último
        df["_confirmed"] = confirmed
        df["churn_flag"] = (df.groupby("cif_id")["_confirmed"]
                            .transform(lambda s: s.shift(-(k - 1)).fillna(False)))
        df = df.drop(columns="_confirmed")
    else:
        df["churn_flag"] = df["is_attrited_state"]

    # Mes del evento = primer mes con churn_flag
    ev = (df[df["churn_flag"].astype(bool)]
          .groupby("cif_id")["as_of_month"].min().rename("churn_event_month"))
    df = df.merge(ev, on="cif_id", how="left")
    return df


def _involuntary_months(tables: dict[str, pd.DataFrame], cfg: Config) -> pd.Series:
    """Primer mes con causa de salida involuntaria por CIF (para excluir)."""
    codes = {c.upper() for c in cfg.target.get("involuntary_reason_codes", [])}
    frames = []

    events = tables.get("events")
    if events is not None and not events.empty and "reason_code" in events.columns:
        e = events[events["reason_code"].astype(str).str.upper().isin(codes)]
        if len(e):
            frames.append(e.groupby("cif_id")["event_date"].min())

    party = tables.get("party")
    if party is not None and "deceased_flag" in party.columns:
        d = party[party["deceased_flag"].astype("boolean").fillna(False)]
        if len(d):
            frames.append(d.groupby("cif_id")["as_of_month"].min())

    if not frames:
        return pd.Series(dtype="datetime64[ns]", name="involuntary_month")
    s = pd.concat(frames).groupby(level=0).min()
    s.name = "involuntary_month"
    return s


def build_panel(tables: dict[str, pd.DataFrame], cfg: Config,
                verbose: bool = True) -> pd.DataFrame:
    """Panel final: una fila por (cif_id, as_of_month) elegible, con etiqueta `y`.

    `y = 1` si el CIF tiene su evento de churn dentro de la ventana de
    performance, que empieza `gap_months + 1` meses después del as-of date.
    """
    state = build_cif_month_state(tables, cfg)
    state = label_attrition_state(state, cfg)

    p = cfg.panel
    gap = int(p.get("gap_months", 1))
    perf = int(p.get("performance_window_months", 6))
    feat_win = int(p.get("feature_window_months", 12))
    min_tenure = int(p.get("min_tenure_months", 6))
    # Historia mínima real que exigimos por detrás. Es deliberadamente menor
    # que feature_window: los rolling usan min_periods, así que una ventana
    # parcial sigue produciendo features útiles. Exigir los 12 meses completos
    # tira un año entero de panel sin necesidad.
    min_hist = int(p.get("min_history_months", max(3, feat_win // 2)))

    months = pd.DatetimeIndex(sorted(state["as_of_month"].unique()))
    # No se puede etiquetar sin ventana de performance completa por delante.
    first_valid = months[min(min_hist, len(months) - 1)]
    last_valid_pos = len(months) - (gap + perf) - 1
    if last_valid_pos < min_hist:
        raise ValueError(
            f"Historia insuficiente: {len(months)} meses no alcanzan para "
            f"min_history={min_hist} + gap={gap} + performance={perf}.\n"
            f"  → Necesitas al menos {min_hist + gap + perf + 2} meses, o "
            f"reduce performance_window_months / min_history_months."
        )
    last_valid = months[last_valid_pos]

    panel = state[(state["as_of_month"] >= first_valid) &
                  (state["as_of_month"] <= last_valid)].copy()

    # ── Etiqueta ──
    def _add_months(s: pd.Series, k: int) -> pd.Series:
        return (s.dt.to_period("M") + k).dt.to_timestamp()

    panel["label_start"] = _add_months(panel["as_of_month"], gap + 1)
    panel["label_end"] = _add_months(panel["as_of_month"], gap + perf)
    ev = panel["churn_event_month"]
    panel["y"] = ((ev.notna()) & (ev >= panel["label_start"]) &
                  (ev <= panel["label_end"])).astype(int)

    # ── Elegibilidad ──
    n0 = len(panel)
    # 1. No se puede predecir la fuga de quien ya se fue
    already_gone = ev.notna() & (panel["as_of_month"] >= ev)
    panel = panel[~already_gone]

    # 2. Antigüedad mínima
    if "tenure_months" in panel.columns:
        panel = panel[panel["tenure_months"].fillna(0) >= min_tenure]

    # 3. Debe tener al menos un rol elegible vigente
    panel = panel[panel["n_relationships_any_role"].fillna(0) > 0]

    # 4. Empleados fuera
    if "employee_flag" in panel.columns:
        panel = panel[~panel["employee_flag"].astype("boolean").fillna(False)]

    # 5. Salidas involuntarias: fuera el cliente desde antes del evento, para
    #    que el modelo no aprenda a predecir compliance/fallecimiento
    if cfg.target.get("exclude_involuntary", True):
        inv = _involuntary_months(tables, cfg)
        if len(inv):
            panel = panel.merge(inv, on="cif_id", how="left")
            bad = panel["involuntary_month"].notna() & \
                  (panel["label_end"] >= panel["involuntary_month"])
            panel = panel[~bad].drop(columns="involuntary_month")

    if verbose:
        rate = panel["y"].mean() if len(panel) else 0
        print(f"\n📊 Panel: {len(panel):,} filas  "
              f"({panel['cif_id'].nunique():,} CIFs × "
              f"{panel['as_of_month'].nunique()} meses)")
        print(f"   Descartadas por elegibilidad: {n0 - len(panel):,}")
        print(f"   Tasa de churn: {rate:.2%}  ({int(panel['y'].sum()):,} positivos)")
        print(f"   Ventana: as-of + gap {gap}m → performance {perf}m")

    return panel.reset_index(drop=True)
