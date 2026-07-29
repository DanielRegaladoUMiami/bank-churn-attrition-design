"""Generador de datos sintéticos con el esquema exacto de las 7 queries SQL.

Sirve para dos cosas:

1. Correr y validar el pipeline completo **antes** de tener los extractos reales.
2. Ser un test de regresión: la estructura causal es conocida, así que si el
   modelo no recupera `has_direct_deposit` o la pendiente de saldo entre las
   features top del SHAP, hay un bug en el pipeline — no en los datos.

Dinámica simulada (deliberadamente parecida a la real):

    pérdida de nómina  ──┐
    saldo decayendo    ──┤
    logins cayendo     ──┼──▶  attrition silenciosa  ──▶  cierre (a veces)
    transferencias a   ──┤
      competidor       ──┤
    loan por terminar  ──┘

Las señales aparecen **antes** del evento, con adelanto variable, que es
exactamente lo que el gap window del panel debe respetar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEPOSIT_PRODUCTS = ["checking", "savings", "mma", "cd"]
LOAN_TYPES = ["mortgage", "auto", "personal", "heloc", "credit_card"]
INVOLUNTARY_CODES = ["DECEASED", "AML_EXIT", "FRAUD", "CHARGE_OFF"]


def generate(
    n_customers: int = 3000,
    n_months: int = 36,
    start_month: str = "2022-07-01",
    annual_churn_rate: float = 0.09,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Genera las 7 tablas. Devuelve un dict con las mismas llaves que
    `Config.data['files']`."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(start=start_month, periods=n_months, freq="MS")

    customers = _make_customers(rng, n_customers, months, annual_churn_rate)
    accounts, relationships = _make_accounts(rng, customers, months)

    party = _make_party(customers, months)
    deposits, txns = _make_deposit_history(rng, customers, accounts, months)
    loans = _make_loan_history(rng, customers, accounts, months)
    digital = _make_digital(rng, customers, months)
    events = _make_events(customers, accounts, deposits, loans)
    # Las relaciones se cierran cuando se cierra la cuenta subyacente. Sin
    # esto, un CIF que cerró todo seguiría apareciendo como vinculado y la
    # rama "sin relaciones" del target nunca dispararía.
    relationships = _close_relationships(relationships, deposits, loans)

    return {
        "party": party,
        "relationships": relationships,
        "deposits": deposits,
        "loans": loans,
        "transactions": txns,
        "digital": digital,
        "events": events,
    }


# ── Población ───────────────────────────────────────────────────────────

def _make_customers(rng, n, months, annual_churn_rate) -> pd.DataFrame:
    n_months = len(months)
    cif_ids = [f"CIF{i:07d}" for i in range(n)]

    entity_type = rng.choice(["personal", "business"], n, p=[0.85, 0.15])
    tenure_start = rng.integers(6, 180, n)  # meses de antigüedad al inicio
    segment = rng.choice(["mass", "affluent", "premier", "small_business"], n,
                         p=[0.62, 0.22, 0.06, 0.10])
    residency = rng.choice(["resident", "non_resident"], n, p=[0.82, 0.18])

    # ── Riesgo latente: combina factores estructurales ──
    base_balance = np.exp(rng.normal(8.6, 1.35, n))  # lognormal, mediana ~5.5k
    has_dd_initial = rng.random(n) < np.where(entity_type == "personal", 0.58, 0.25)
    n_products = rng.integers(1, 5, n)
    digital_affinity = rng.beta(2.4, 2.0, n)

    # Menos productos, sin nómina, poco digital y poca antigüedad => más riesgo
    risk = (
        -0.40 * has_dd_initial
        - 0.22 * np.log1p(n_products)
        - 0.30 * digital_affinity
        - 0.12 * np.log1p(tenure_start / 12)
        - 0.10 * np.log1p(base_balance) / 5
        + 0.25 * (residency == "non_resident")
        + rng.normal(0, 0.45, n)
    )
    # Calibrar la tasa de churn del periodo completo al objetivo anual
    horizon_years = n_months / 12
    target_rate = 1 - (1 - annual_churn_rate) ** horizon_years
    thresh = np.quantile(risk, 1 - target_rate)
    will_churn = risk > thresh

    # Mes del evento: uniforme sobre la ventana, dejando margen para señales
    # Margen al final: el evento debe caer dentro de la ventana observada
    churn_month_idx = np.where(
        will_churn,
        rng.integers(10, max(11, n_months - 2), n),
        -1,
    )

    # ~12% de las salidas son involuntarias (se excluyen del modelo)
    involuntary = will_churn & (rng.random(n) < 0.12)
    reason = np.where(
        involuntary,
        rng.choice(INVOLUNTARY_CODES, n),
        np.where(will_churn, "CUSTOMER_REQUEST", None),
    )
    # Solo ~45% de los que se van cierran formalmente: el resto es
    # attrition silenciosa (la trampa que el target compuesto debe capturar)
    formal_close = will_churn & (rng.random(n) < 0.45)

    return pd.DataFrame({
        "cif_id": cif_ids,
        "entity_type": entity_type,
        "tenure_start_months": tenure_start,
        "segment": segment,
        "residency_status": residency,
        "base_balance": base_balance,
        "has_dd_initial": has_dd_initial,
        "n_products": n_products,
        "digital_affinity": digital_affinity,
        "will_churn": will_churn,
        "churn_month_idx": churn_month_idx,
        "involuntary": involuntary,
        "close_reason_code": reason,
        "formal_close": formal_close,
        # Adelanto con que cada señal se degrada antes del evento
        "dd_loss_lead": rng.integers(3, 8, n),
        "balance_decay_lead": rng.integers(4, 10, n),
        "login_decay_lead": rng.integers(2, 7, n),
        "state": rng.choice(["FL", "TX", "NY", "NJ", "CA"], n,
                            p=[0.55, 0.13, 0.12, 0.10, 0.10]),
        "employee_flag": rng.random(n) < 0.008,
    })


def _close_relationships(relationships: pd.DataFrame, deposits: pd.DataFrame,
                         loans: pd.DataFrame) -> pd.DataFrame:
    """Propaga el cierre de cuentas y el payoff de préstamos a la vigencia de
    las relaciones CIF↔cuenta."""
    ends = []
    if len(deposits):
        d = deposits.dropna(subset=["close_date"])
        if len(d):
            ends.append(d.groupby("account_id")["close_date"].min())
    if len(loans) and "payoff_date" in loans.columns:
        lo = loans.dropna(subset=["payoff_date"])
        if len(lo):
            ends.append(lo.groupby("loan_id")["payoff_date"].min()
                        .rename_axis("account_id"))
    if not ends:
        return relationships

    end_map = pd.concat(ends).groupby(level=0).min()
    rel = relationships.copy()
    mapped = rel["account_id"].map(end_map)
    rel["relationship_end_date"] = rel["relationship_end_date"].fillna(mapped)
    return rel


def _make_party(customers: pd.DataFrame, months) -> pd.DataFrame:
    """Q1: una fila por CIF por mes."""
    # El alta es un atributo inmutable del cliente, así que se ancla al primer
    # mes del panel y no se recalcula en cada iteración. Derivarla de `m` la
    # hacía avanzar un mes por cada mes del panel: el mismo CIF aparecía dado
    # de alta en 2017 al principio del panel y en 2020 al final, lo que
    # contradice su propio `tenure_months` y la inutiliza como ancla temporal.
    customer_since = pd.Series(
        months[0] - pd.to_timedelta(
            customers["tenure_start_months"].to_numpy() * 30, unit="D"),
        index=customers.index,
    )
    rows = []
    for i, m in enumerate(months):
        rows.append(pd.DataFrame({
            "cif_id": customers["cif_id"],
            "as_of_month": m,
            "entity_type": customers["entity_type"],
            "customer_since_date": customer_since,
            # Derivada de la fecha de alta, no contada aparte: con el mes de
            # 30 días de la línea anterior, `tenure_start + i` se separaba
            # hasta dos meses de lo que dice `customer_since_date`.
            "tenure_months": ((m.year - customer_since.dt.year) * 12
                              + (m.month - customer_since.dt.month)).clip(lower=0),
            "segment": customers["segment"],
            "residency_status": customers["residency_status"],
            "state": customers["state"],
            "employee_flag": customers["employee_flag"],
            "deceased_flag": (customers["close_reason_code"] == "DECEASED")
                             & (i >= customers["churn_month_idx"]),
            "kyc_risk_rating": np.where(
                customers["residency_status"] == "non_resident", "medium", "low"
            ),
            "status": "active",
        }))
    return pd.concat(rows, ignore_index=True)


# ── Cuentas y relaciones ────────────────────────────────────────────────

def _make_accounts(rng, customers: pd.DataFrame, months):
    """Genera cuentas de depósito y préstamos, con roles de CIF realistas
    (incluyendo joint y authorized signer)."""
    acct_rows, rel_rows = [], []
    n = len(customers)
    acct_seq = 0
    start = months[0]

    for idx in range(n):
        cif = customers.at[idx, "cif_id"]
        n_dep = max(1, min(4, int(customers.at[idx, "n_products"])))
        share = rng.dirichlet(np.ones(n_dep))

        for j in range(n_dep):
            acct_seq += 1
            acct_id = f"ACC{acct_seq:08d}"
            product = "checking" if j == 0 else rng.choice(
                DEPOSIT_PRODUCTS[1:], p=[0.45, 0.25, 0.30]
            )
            is_joint = rng.random() < 0.22
            n_owners = 2 if is_joint else 1

            acct_rows.append({
                "account_id": acct_id,
                "owner_cif": cif,
                "product_family": product,
                "balance_share": share[j],
                "n_owners": n_owners,
                "open_offset": -int(rng.integers(1, 200)),
                "term_months": int(rng.choice([6, 12, 24, 36, 60])) if product == "cd" else None,
                "rate": float(rng.uniform(0.001, 0.045)) if product in ("cd", "mma", "savings")
                        else float(rng.uniform(0.0, 0.002)),
            })
            rel_rows.append({
                "cif_id": cif, "account_id": acct_id, "product_type": "deposit",
                "relationship_role": "joint" if is_joint else "primary",
                "relationship_start_date": start + pd.DateOffset(
                    months=acct_rows[-1]["open_offset"]
                ),
                "relationship_end_date": pd.NaT, "n_owners": n_owners,
            })
            # Co-titular en cuentas joint: otro CIF de la población
            if is_joint:
                other = customers.at[int(rng.integers(0, n)), "cif_id"]
                if other != cif:
                    rel_rows.append({
                        "cif_id": other, "account_id": acct_id,
                        "product_type": "deposit", "relationship_role": "joint",
                        "relationship_start_date": rel_rows[-1]["relationship_start_date"],
                        "relationship_end_date": pd.NaT, "n_owners": n_owners,
                    })
            # Authorized signer: sin peso económico, pero ancla relacional
            if rng.random() < 0.08:
                other = customers.at[int(rng.integers(0, n)), "cif_id"]
                if other != cif:
                    rel_rows.append({
                        "cif_id": other, "account_id": acct_id,
                        "product_type": "deposit",
                        "relationship_role": "authorized_signer",
                        "relationship_start_date": rel_rows[-1]["relationship_start_date"],
                        "relationship_end_date": pd.NaT, "n_owners": n_owners,
                    })

        # Préstamo (~35% de los clientes)
        if rng.random() < 0.35:
            acct_seq += 1
            loan_id = f"LN{acct_seq:08d}"
            ltype = str(rng.choice(LOAN_TYPES, p=[0.18, 0.30, 0.24, 0.10, 0.18]))
            term = int(rng.choice([36, 48, 60, 120, 360]))
            amt = float(np.exp(rng.normal(10.2, 0.9)))
            elapsed = int(rng.integers(1, max(2, min(term - 1, 60))))
            acct_rows.append({
                "account_id": loan_id, "owner_cif": cif, "product_family": "loan",
                "loan_type": ltype, "original_amount": amt,
                "original_term_months": term, "elapsed_at_start": elapsed,
                "rate": float(rng.uniform(0.03, 0.19)),
                "balance_share": 0.0, "n_owners": 1, "open_offset": -elapsed,
            })
            rel_rows.append({
                "cif_id": cif, "account_id": loan_id, "product_type": "loan",
                "relationship_role": "borrower",
                "relationship_start_date": start + pd.DateOffset(months=-elapsed),
                "relationship_end_date": pd.NaT, "n_owners": 1,
            })

    return pd.DataFrame(acct_rows), pd.DataFrame(rel_rows)


# ── Historia de depósitos y transacciones ───────────────────────────────

def _make_deposit_history(rng, customers: pd.DataFrame, accounts: pd.DataFrame, months):
    """Q3 (snapshot mensual de depósitos) y Q5 (transacciones en formato largo)."""
    cust = customers.set_index("cif_id")
    dep_accts = accounts[accounts["product_family"] != "loan"].reset_index(drop=True)
    n_months = len(months)

    dep_rows, txn_rows = [], []

    for r in dep_accts.itertuples(index=False):
        c = cust.loc[r.owner_cif]
        churn_i = int(c.churn_month_idx)
        involuntary = bool(c.involuntary)
        base = float(c.base_balance) * float(r.balance_share)
        trend = rng.normal(0.004, 0.012)          # deriva mensual del saldo
        noise = max(0.04, rng.gamma(2.0, 0.035))  # volatilidad idiosincrática
        open_date = months[0] + pd.DateOffset(months=int(r.open_offset))
        maturity = (open_date + pd.DateOffset(months=int(r.term_months))
                    if r.product_family == "cd" and r.term_months else pd.NaT)

        closed = False
        days_since_txn = int(rng.integers(0, 20))

        for i, m in enumerate(months):
            if closed:
                break

            # ── Trayectoria del saldo ──
            level = base * (1 + trend) ** i
            if churn_i >= 0:
                lead = int(c.balance_decay_lead)
                if i >= churn_i - lead:
                    # Decaimiento exponencial hacia ~0 al llegar al evento
                    progress = np.clip((i - (churn_i - lead)) / max(lead, 1), 0, 2.0)
                    level *= float(np.exp(-3.6 * progress))
            bal = max(0.0, level * float(rng.normal(1.0, noise)))

            # ── Nómina / direct deposit ──
            has_dd = bool(c.has_dd_initial)
            if has_dd and churn_i >= 0 and i >= churn_i - int(c.dd_loss_lead):
                has_dd = False

            # ── Actividad transaccional ──
            # Llega a cero ~3 meses ANTES del evento: así el contador de
            # inactividad alcanza a acumularse y el target compuesto dispara
            # cerca del mes real de fuga, no cinco meses después.
            activity = 1.0
            if churn_i >= 0:
                lead = int(c.balance_decay_lead)
                if i >= churn_i - lead:
                    activity = float(np.clip(
                        1 - (i - (churn_i - lead)) / max(lead - 3, 1), 0.0, 1.0
                    ))

            is_primary_dda = r.product_family == "checking"
            intensity = activity * (2.2 if is_primary_dda else 0.5)

            # ── Cierre formal ──
            close_date, reason, status = pd.NaT, None, "open"
            if churn_i >= 0 and i >= churn_i and bool(c.formal_close):
                close_date, reason, status = m, c.close_reason_code, "closed"
                closed = True
            elif bal < 50 and activity < 0.15:
                status = "dormant"

            nsf = int(rng.poisson(0.55 * (1.9 if (churn_i >= 0 and
                      churn_i - 6 <= i < churn_i) else 1.0))) if is_primary_dda else 0
            fees = float(nsf * 35 + (12.0 if bal < 1500 and is_primary_dda else 0.0))

            dep_rows.append({
                "account_id": r.account_id, "as_of_month": m,
                "product_family": r.product_family, "product_code": f"{r.product_family.upper()}01",
                "open_date": open_date, "close_date": close_date,
                "close_reason_code": reason, "account_status": status,
                "eom_balance": round(bal, 2),
                "avg_daily_balance_month": round(bal * float(rng.uniform(0.9, 1.1)), 2),
                "min_balance_month": round(bal * float(rng.uniform(0.55, 0.95)), 2),
                "interest_rate_paid": r.rate,
                "maturity_date": maturity,
                "term_months": r.term_months,
                "auto_renew_flag": bool(rng.random() < 0.6) if r.product_family == "cd" else None,
                "nsf_count_month": nsf,
                "fees_charged_month": round(fees, 2),
                "overdraft_days_count": int(rng.poisson(0.3)) if is_primary_dda else 0,
                "statement_delivery": "e" if float(c.digital_affinity) > 0.45 else "paper",
                "dormant_flag": status == "dormant",
            })

            # ── Transacciones (formato largo, solo filas con actividad) ──
            month_txn_total = 0
            cats = {
                "ach_credit_in": (2.0 + 1.6 * has_dd) * intensity,
                "ach_debit_out": 2.4 * intensity,
                "pos_signature": 9.0 * intensity if is_primary_dda else 0.4,
                "pos_pin": 4.0 * intensity if is_primary_dda else 0.2,
                "atm_own": 1.4 * intensity,
                "atm_foreign": 0.5 * intensity,
                "zelle_out": 1.7 * intensity,
                "zelle_in": 1.1 * intensity,
                "bill_pay": 2.6 * intensity if is_primary_dda else 0.1,
                "mobile_deposit": 1.0 * intensity * float(c.digital_affinity) * 2,
                "internal_transfer": 0.9 * intensity,
                "wire_out": 0.12 * intensity,
            }
            # Fuga en curso: sube la transferencia saliente a otra institución
            competitor_out = 0
            if churn_i >= 0 and churn_i - 6 <= i < churn_i:
                competitor_out = int(rng.poisson(1.8))

            for cat, lam in cats.items():
                cnt = int(rng.poisson(max(lam, 0)))
                if cnt == 0:
                    continue
                month_txn_total += cnt
                debit = cat.endswith(("_out", "_written", "signature", "pin")) or "atm" in cat
                amt = float(cnt * abs(rng.normal(bal * 0.05 + 60, 45)))
                txn_rows.append({
                    "account_id": r.account_id, "as_of_month": m, "txn_category": cat,
                    "txn_count": cnt,
                    "total_amount_debit": round(amt, 2) if debit else 0.0,
                    "total_amount_credit": 0.0 if debit else round(amt, 2),
                    "is_direct_deposit": bool(has_dd and cat == "ach_credit_in"),
                    "competitor_transfer_flag": False,
                    "days_since_last_customer_txn": days_since_txn,
                })

            if competitor_out:
                # Categoría propia para no romper la llave (cuenta, mes, categoría)
                month_txn_total += competitor_out
                txn_rows.append({
                    "account_id": r.account_id, "as_of_month": m,
                    "txn_category": "ach_debit_out_competitor",
                    "txn_count": competitor_out,
                    "total_amount_debit": round(float(competitor_out) * bal * 0.28 + 200, 2),
                    "total_amount_credit": 0.0, "is_direct_deposit": False,
                    "competitor_transfer_flag": True,
                    "days_since_last_customer_txn": days_since_txn,
                })

            days_since_txn = 0 if month_txn_total > 0 else days_since_txn + 30

            if involuntary and churn_i >= 0 and i >= churn_i:
                closed = True

    deposits = pd.DataFrame(dep_rows)
    txns = pd.DataFrame(txn_rows)
    return deposits, txns


def _make_loan_history(rng, customers, accounts, months) -> pd.DataFrame:
    """Q4: snapshot mensual de préstamos, con payoff natural vs. prepago."""
    cust = customers.set_index("cif_id")
    loan_accts = accounts[accounts["product_family"] == "loan"]
    rows = []

    for r in loan_accts.itertuples(index=False):
        c = cust.loc[r.owner_cif]
        churn_i = int(c.churn_month_idx)
        term = int(r.original_term_months)
        amt = float(r.original_amount)
        rate_m = float(r.rate) / 12
        elapsed0 = int(r.elapsed_at_start)
        orig_date = months[0] + pd.DateOffset(months=-elapsed0)
        maturity = orig_date + pd.DateOffset(months=term)
        pmt = amt * rate_m / (1 - (1 + rate_m) ** -term) if rate_m > 0 else amt / term

        # Prepago: más probable si el cliente va de salida
        prepay_i = -1
        if churn_i >= 0 and rng.random() < 0.45:
            prepay_i = max(0, churn_i - int(rng.integers(0, 4)))

        bal = amt * ((1 + rate_m) ** term - (1 + rate_m) ** elapsed0) / \
              ((1 + rate_m) ** term - 1) if rate_m > 0 else amt * (1 - elapsed0 / term)
        bal = max(bal, 0.0)
        status = "active"

        for i, m in enumerate(months):
            k = elapsed0 + i
            if status != "active":
                break
            months_to_maturity = term - k

            if months_to_maturity <= 0:
                status, payoff = "paid_off", m       # maturity natural
                bal = 0.0
            elif i == prepay_i:
                status, payoff = "paid_off", m       # prepago competitivo
                bal = 0.0
            elif c.close_reason_code == "CHARGE_OFF" and churn_i >= 0 and i >= churn_i:
                status, payoff = "charged_off", m
            else:
                payoff = pd.NaT
                interest = bal * rate_m
                bal = max(0.0, bal - max(pmt - interest, 0))

            dpd_risk = 0.05 + (0.25 if churn_i >= 0 and i >= churn_i - 4 else 0)
            dpd = int(rng.choice([0, 30, 60, 90], p=[1 - dpd_risk, dpd_risk * 0.6,
                                                     dpd_risk * 0.28, dpd_risk * 0.12]))

            rows.append({
                "loan_id": r.account_id, "as_of_month": m, "loan_type": r.loan_type,
                "origination_date": orig_date, "original_amount": round(amt, 2),
                "original_term_months": term, "maturity_date": maturity,
                "current_balance": round(bal, 2), "scheduled_payment": round(pmt, 2),
                "interest_rate": r.rate, "rate_type": "fixed",
                "remaining_term_months": max(months_to_maturity, 0),
                "months_to_maturity": max(months_to_maturity, 0),
                "delinquency_bucket": {0: "current", 30: "30", 60: "60", 90: "90+"}[dpd],
                "days_past_due": dpd, "status": status, "payoff_date": payoff,
                "utilization": (round(1 - bal / amt, 4) if amt > 0 else 0.0),
            })

    if not rows:
        return pd.DataFrame(columns=["loan_id", "as_of_month", "loan_type", "current_balance"])
    return pd.DataFrame(rows)


def _make_digital(rng, customers: pd.DataFrame, months) -> pd.DataFrame:
    """Q6: actividad digital y de contacto por CIF-mes."""
    rows = []
    n = len(customers)
    aff = customers["digital_affinity"].to_numpy()
    churn_idx = customers["churn_month_idx"].to_numpy()
    login_lead = customers["login_decay_lead"].to_numpy()
    enrolled = rng.random(n) < (0.35 + 0.55 * aff)

    for i, m in enumerate(months):
        decay = np.ones(n)
        active_churn = churn_idx >= 0
        start_decay = churn_idx - login_lead
        mask = active_churn & (i >= start_decay)
        decay[mask] = np.clip(
            1 - (i - start_decay[mask]) / np.maximum(login_lead[mask], 1), 0.0, 1.0
        )

        web = rng.poisson(np.maximum(aff * 6 * decay * enrolled, 0.01))
        mob = rng.poisson(np.maximum(aff * 11 * decay * enrolled, 0.01))
        complaint_rate = np.where(mask, 0.10, 0.02)

        rows.append(pd.DataFrame({
            "cif_id": customers["cif_id"], "as_of_month": m,
            "online_enrolled_flag": enrolled,
            "mobile_enrolled_flag": enrolled & (aff > 0.35),
            "login_count_web_month": web, "login_count_mobile_month": mob,
            "alerts_enrolled_flag": enrolled & (aff > 0.55),
            "estatement_flag": aff > 0.45,
            "debit_card_active_flag": rng.random(n) < 0.85,
            "call_center_calls_month": rng.poisson(np.where(mask, 0.9, 0.25)),
            "branch_visits_month": rng.poisson(np.maximum((1 - aff) * 1.4 * decay, 0.01)),
            "complaints_open_month": rng.binomial(1, complaint_rate),
            "disputes_month": rng.binomial(1, 0.02),
            "bill_pay_payees_active": rng.poisson(np.maximum(aff * 4 * decay, 0.01)),
        }))
    return pd.concat(rows, ignore_index=True)


def _make_events(customers, accounts, deposits, loans) -> pd.DataFrame:
    """Q7: log de eventos de producto."""
    rows = []

    opens = accounts[["account_id", "owner_cif"]].copy()
    first = deposits.groupby("account_id", as_index=False)["open_date"].first()
    opens = opens.merge(first, on="account_id", how="inner")
    for r in opens.itertuples(index=False):
        rows.append({"cif_id": r.owner_cif, "account_id": r.account_id,
                     "event_date": r.open_date, "event_type": "account_open",
                     "reason_code": None})

    closes = deposits[deposits["close_date"].notna()]
    owner = accounts.set_index("account_id")["owner_cif"]
    for r in closes.itertuples(index=False):
        rows.append({"cif_id": owner.get(r.account_id), "account_id": r.account_id,
                     "event_date": r.close_date, "event_type": "account_close",
                     "reason_code": r.close_reason_code})

    if len(loans):
        po = loans[loans["payoff_date"].notna()]
        for r in po.itertuples(index=False):
            rows.append({"cif_id": owner.get(r.loan_id), "account_id": r.loan_id,
                         "event_date": r.payoff_date, "event_type": "loan_payoff",
                         "reason_code": r.status})

    ev = pd.DataFrame(rows)
    return ev.dropna(subset=["cif_id"]).reset_index(drop=True)
