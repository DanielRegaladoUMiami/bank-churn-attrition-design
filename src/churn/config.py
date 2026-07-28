"""Carga de configuración y contrato de esquemas de las 7 extracciones SQL.

El contrato (`TABLE_SCHEMAS`) es la frontera entre el agente SQL y el pipeline:
si un extracto cumple el contrato, todo lo demás corre sin cambios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Contrato de esquemas ────────────────────────────────────────────────
# required: sin estas columnas el pipeline no puede correr.
# optional: se usan si existen; si no, las features derivadas se omiten.

TABLE_SCHEMAS: dict[str, dict[str, Any]] = {
    "party": {
        "keys": ["cif_id", "as_of_month"],
        "required": ["cif_id", "as_of_month", "entity_type", "customer_since_date"],
        "optional": [
            "age", "tenure_months", "residency_status", "country_of_residence",
            "state", "zip3", "primary_branch_id", "segment", "employee_flag",
            "kyc_risk_rating", "naics", "preferred_language", "deceased_flag",
            "status", "merged_into_cif", "household_id",
        ],
        "dates": ["as_of_month", "customer_since_date"],
    },
    "relationships": {
        "keys": ["cif_id", "account_id", "relationship_role"],
        "required": [
            "cif_id", "account_id", "product_type", "relationship_role",
            "relationship_start_date",
        ],
        "optional": ["relationship_end_date", "n_owners"],
        "dates": ["relationship_start_date", "relationship_end_date"],
    },
    "deposits": {
        "keys": ["account_id", "as_of_month"],
        "required": ["account_id", "as_of_month", "product_family", "eom_balance"],
        "optional": [
            "product_code", "open_date", "close_date", "close_reason_code",
            "account_status", "avg_daily_balance_month", "min_balance_month",
            "max_balance_month", "interest_rate_paid", "maturity_date",
            "term_months", "auto_renew_flag", "original_deposit_amount",
            "overdraft_days_count", "nsf_count_month", "fees_charged_month",
            "statement_delivery", "dormant_flag",
        ],
        "dates": ["as_of_month", "open_date", "close_date", "maturity_date"],
    },
    "loans": {
        "keys": ["loan_id", "as_of_month"],
        "required": ["loan_id", "as_of_month", "loan_type", "current_balance"],
        "optional": [
            "origination_date", "original_amount", "original_term_months",
            "maturity_date", "scheduled_payment", "interest_rate", "rate_type",
            "remaining_term_months", "months_to_maturity", "extra_principal_paid_month",
            "delinquency_bucket", "days_past_due", "times_30dpd_last_12m",
            "status", "payoff_date", "credit_limit", "utilization", "ltv",
        ],
        "dates": ["as_of_month", "origination_date", "maturity_date", "payoff_date"],
    },
    "transactions": {
        # Formato largo: una fila por (cuenta, mes, categoría)
        "keys": ["account_id", "as_of_month", "txn_category"],
        "required": ["account_id", "as_of_month", "txn_category", "txn_count"],
        "optional": [
            "total_amount_debit", "total_amount_credit",
            "last_customer_initiated_txn_date", "days_since_last_customer_txn",
            "is_direct_deposit", "competitor_transfer_flag",
        ],
        "dates": ["as_of_month", "last_customer_initiated_txn_date"],
    },
    "digital": {
        "keys": ["cif_id", "as_of_month"],
        "required": ["cif_id", "as_of_month"],
        "optional": [
            "online_enrolled_flag", "mobile_enrolled_flag", "login_count_web_month",
            "login_count_mobile_month", "last_login_date", "alerts_enrolled_flag",
            "estatement_flag", "debit_card_active_flag", "call_center_calls_month",
            "branch_visits_month", "complaints_open_month", "disputes_month",
            "bill_pay_payees_active",
        ],
        "dates": ["as_of_month", "last_login_date"],
    },
    "events": {
        "keys": ["cif_id", "account_id", "event_date", "event_type"],
        "required": ["cif_id", "event_date", "event_type"],
        "optional": ["account_id", "event_detail", "reason_code"],
        "dates": ["event_date"],
    },
}

# Categorías de transacción que el pipeline sabe explotar. Cualquier otra
# categoría presente en los datos se agrega igual, solo que sin feature
# de dominio dedicada.
KNOWN_TXN_CATEGORIES = [
    "ach_credit_in", "ach_debit_out", "zelle_in", "zelle_out",
    "wire_in", "wire_out", "pos_signature", "pos_pin", "atm_own",
    "atm_foreign", "check_written", "branch_deposit", "mobile_deposit",
    "bill_pay", "internal_transfer",
]


@dataclass
class Config:
    """Configuración del pipeline, cargada desde config.yaml."""

    data: dict[str, Any] = field(default_factory=dict)
    panel: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    ownership_weights: dict[str, Any] = field(default_factory=dict)
    eligible_roles: list[str] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    explain: dict[str, Any] = field(default_factory=dict)
    synthetic: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})

    # Atajos usados en varios módulos
    @property
    def is_synthetic(self) -> bool:
        return self.data.get("source", "synthetic") == "synthetic"

    @property
    def label_offset_months(self) -> int:
        """Meses entre el as-of date y el inicio de la ventana de performance."""
        return int(self.panel.get("gap_months", 1)) + 1

    def artifacts_path(self) -> Path:
        p = Path(self.data.get("artifacts_dir", "artifacts"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def processed_path(self) -> Path:
        p = Path(self.data.get("processed_dir", "data/processed"))
        p.mkdir(parents=True, exist_ok=True)
        return p
