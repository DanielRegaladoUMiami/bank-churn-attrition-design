#!/usr/bin/env python3
"""Pipeline end-to-end de churn/attrition.

    python run_pipeline.py                 # datos sintéticos, corrida completa
    python run_pipeline.py --quick         # versión rápida (smoke test)
    python run_pipeline.py --source drive  # lee los extractos reales de Drive

Etapas: carga → validación → panel → features → split → tuning → ensemble →
calibración → evaluación → SHAP → contrafactuales.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn.config import Config                                    # noqa: E402
from churn.data import load_tables, mount_drive, profile, validate  # noqa: E402
from churn.evaluation import (                                     # noqa: E402
    calibrate, calibration_table, decile_table, evaluate, make_cv,
    summarize, temporal_split,
)
from churn.explain import (                                        # noqa: E402
    counterfactual_analysis, global_importance, sanity_check, shap_analysis,
)
from churn.features import (                                       # noqa: E402
    build_features, clean_features, select_feature_columns,
)
from churn.models import (                                         # noqa: E402
    SEQUENCE_COLS, SequenceModel, build_sequences, build_stack, tune_all,
)
from churn.panel import build_cif_month_state, build_panel         # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'═' * 70}\n  {text}\n{'═' * 70}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", choices=["synthetic", "drive"], default=None,
                    help="sobrescribe data.source del config")
    ap.add_argument("--quick", action="store_true",
                    help="menos clientes e iteraciones, para probar el pipeline")
    ap.add_argument("--no-sequence", action="store_true", help="omitir el GRU")
    ap.add_argument("--no-shap", action="store_true", help="omitir explicabilidad")
    args = ap.parse_args()

    t0 = time.time()
    cfg = Config.load(args.config)
    if args.source:
        cfg.data["source"] = args.source
    if args.quick:
        cfg.synthetic["n_customers"] = 1200
        cfg.synthetic["n_months"] = 36
        # Ventanas más cortas: el embargo cuesta gap+performance meses de panel
        cfg.panel["min_history_months"] = 4
        cfg.panel["feature_window_months"] = 8
        cfg.panel["performance_window_months"] = 3
        cfg.split["oot_months"] = 5
        cfg.models["n_iter"] = 4
        cfg.models["enabled"] = ["logreg", "lightgbm"]
        cfg.split["n_folds"] = 3
        cfg.models["sequence_model"]["epochs"] = 5
        cfg.explain["shap_sample_size"] = 500
        # En modo quick guardamos localmente, no en Drive
        cfg.data["artifacts_dir"] = "artifacts_quick"
        cfg.data["processed_dir"] = "data_quick/processed"

    # ── 1. Carga ──
    banner("1 · CARGA DE DATOS")
    if not cfg.is_synthetic:
        mount_drive()
    tables = load_tables(cfg)
    print()
    print(profile(tables).to_string(index=False))
    validate(tables)

    # ── 2. Panel y target ──
    banner("2 · PANEL Y TARGET")
    state = build_cif_month_state(tables, cfg)
    panel = build_panel(tables, cfg)
    if panel["y"].sum() < 30:
        print(f"\n❌ Solo {int(panel['y'].sum())} positivos: insuficiente para "
              "entrenar. Revisa los umbrales de `target` en config.yaml.")
        return 1

    # ── 3. Features ──
    banner("3 · FEATURE ENGINEERING")
    feats = build_features(panel, cfg, tables)
    numeric, categorical = select_feature_columns(feats)
    feats = clean_features(feats, numeric)
    print(f"   {len(numeric)} numéricas, {len(categorical)} categóricas")

    # ── 4. Split ──
    banner("4 · SPLIT OUT-OF-TIME")
    train, test = temporal_split(feats, cfg)
    Xtr, ytr = train[numeric + categorical], train["y"].to_numpy()
    Xte, yte = test[numeric + categorical], test["y"].to_numpy()
    groups = train["cif_id"].to_numpy()
    if ytr.sum() < 20 or yte.sum() < 10:
        print(f"\n❌ Positivos insuficientes tras el split "
              f"(train={int(ytr.sum())}, test={int(yte.sum())}).")
        return 1

    # ── 5. Tuning ──
    banner("5 · TUNING CON CV AGRUPADO POR CIF")
    searches = tune_all(Xtr, ytr, groups, numeric, categorical, cfg)
    if not searches:
        print("❌ Ningún modelo entrenó correctamente.")
        return 1

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, object] = {}
    for name, s in searches.items():
        p = s.best_estimator_.predict_proba(Xte)[:, 1]
        results[name] = evaluate(yte, p)
        fitted[name] = s.best_estimator_

    # ── 6. Ensemble ──
    banner("6 · STACKING")
    stack, chosen = build_stack(searches, int(cfg.models.get("stack_top_k", 3)), cfg)
    if stack is not None:
        print(f"   Base learners: {chosen}")
        stack.fit(Xtr, ytr)
        p = stack.predict_proba(Xte)[:, 1]
        results["stacking"] = evaluate(yte, p)
        fitted["stacking"] = stack
    else:
        print("   Menos de 2 modelos disponibles: se omite el stacking.")

    # ── 7. Red secuencial ──
    seq_cfg = cfg.models.get("sequence_model", {})
    if seq_cfg.get("enabled", False) and not args.no_sequence:
        banner("7 · RED SECUENCIAL (GRU)")
        try:
            win = int(cfg.panel.get("feature_window_months", 12))
            Str = build_sequences(state, train, SEQUENCE_COLS, win)
            Ste = build_sequences(state, test, SEQUENCE_COLS, win)
            print(f"   Tensor de entrenamiento: {Str.shape} "
                  f"[muestras, meses, canales]")
            gru = SequenceModel(
                n_features=Str.shape[2],
                hidden_size=int(seq_cfg.get("hidden_size", 64)),
                num_layers=int(seq_cfg.get("num_layers", 1)),
                dropout=float(seq_cfg.get("dropout", 0.2)),
                lr=float(seq_cfg.get("lr", 1e-3)),
                epochs=int(seq_cfg.get("epochs", 30)),
                batch_size=int(seq_cfg.get("batch_size", 256)),
                patience=int(seq_cfg.get("patience", 5)),
            ).fit(Str, ytr, Ste, yte)
            p_gru = gru.predict_proba(Ste)[:, 1]
            results["gru_sequence"] = evaluate(yte, p_gru)

            # Blend con el mejor tabular: es el ensemble que realmente aporta
            best_tab = max(results, key=lambda k: results[k]["pr_auc"]
                           if k != "gru_sequence" else -1)
            p_tab = fitted[best_tab].predict_proba(Xte)[:, 1]
            for w in (0.2, 0.3, 0.4):
                results[f"blend_{best_tab}+gru_{w:.0%}"] = evaluate(
                    yte, (1 - w) * p_tab + w * p_gru
                )
        except Exception as e:
            print(f"   ⚠️  GRU omitido ({type(e).__name__}: {e})")

    # ── 8. Resultados ──
    banner("8 · RESULTADOS (holdout out-of-time)")
    board = summarize(results)
    print(board.to_string())

    best_name = board.index[0]
    print(f"\n🏆 Mejor: {best_name}  (PR-AUC {board.loc[best_name, 'pr_auc']:.4f}, "
          f"lift@10 {board.loc[best_name, 'lift@10']:.2f}x)")

    best_model = fitted.get(best_name if best_name in fitted else
                            max(fitted, key=lambda k: results[k]["pr_auc"]))
    p_best = best_model.predict_proba(Xte)[:, 1]

    print("\n📋 Tabla de deciles:")
    print(decile_table(yte, p_best).to_string(index=False))

    # ── 9. Calibración ──
    if cfg.models.get("calibrate", True):
        banner("9 · CALIBRACIÓN")
        n_cal = len(Xte) // 3
        try:
            cal = calibrate(best_model, Xte.iloc[:n_cal], yte[:n_cal])
            p_cal = cal.predict_proba(Xte.iloc[n_cal:])[:, 1]
            y_ev = yte[n_cal:]
            print("   Antes:")
            print(calibration_table(y_ev, p_best[n_cal:]).to_string(index=False))
            print("   Después (isotónica):")
            print(calibration_table(y_ev, p_cal).to_string(index=False))
            m_before = evaluate(y_ev, p_best[n_cal:])
            m_after = evaluate(y_ev, p_cal)
            print(f"   Brier {m_before['brier']:.5f} → {m_after['brier']:.5f}")
        except Exception as e:
            print(f"   ⚠️  Calibración omitida ({type(e).__name__}: {e})")

    # ── 10. Explicabilidad ──
    if not args.no_shap:
        banner("10 · SHAP Y CONTRAFACTUALES")
        try:
            sv, Xt = shap_analysis(best_model, Xte, cfg)
            imp = global_importance(sv, Xt, int(cfg.explain.get("top_n_features", 25)))
            print(imp.to_string(index=False))
            sanity_check(imp, expected_top=["direct_deposit", "balance", "drawdown",
                                            "slope", "activity"])
            cf = counterfactual_analysis(best_model, Xte, cfg)
            if len(cf):
                print()
                print(cf.to_string(index=False))
        except Exception as e:
            print(f"   ⚠️  Explicabilidad omitida ({type(e).__name__}: {e})")

    # ── 11. Guardar ──
    out = cfg.artifacts_path()
    board.to_csv(out / "leaderboard.csv")
    decile_table(yte, p_best).to_csv(out / "deciles.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)

    banner(f"COMPLETADO en {time.time() - t0:.1f}s  →  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
