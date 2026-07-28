"""Carga de datos: Google Drive (Colab) o generador sintético.

Punto de entrada único: `load_tables(cfg)`. El resto del pipeline no sabe
—ni le importa— de dónde salieron los datos.

Uso en Colab
------------
    from churn.data import mount_drive, load_tables
    mount_drive()
    tables = load_tables(cfg)

Dónde poner los archivos en Drive
---------------------------------
    MyDrive/churn_model/data/raw/
        q1_party_master.csv
        q2_cif_account_relationship.csv
        q3_deposit_monthly.csv
        q4_loan_monthly.csv
        q5_txn_monthly.csv
        q6_digital_contact_monthly.csv
        q7_product_events.csv

Los nombres se configuran en `config.yaml` → `data.files`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import TABLE_SCHEMAS, Config


# ── Google Drive ────────────────────────────────────────────────────────

def in_colab() -> bool:
    return "google.colab" in sys.modules


def mount_drive(mount_point: str = "/content/drive", force: bool = False) -> bool:
    """Monta Google Drive si estamos en Colab. Fuera de Colab no hace nada
    (permite correr el mismo código local sin tocarlo)."""
    if not in_colab():
        print("ℹ️  Fuera de Colab: se omite el montaje de Drive.")
        return False
    if Path(mount_point, "MyDrive").exists() and not force:
        print(f"✓ Drive ya montado en {mount_point}")
        return True
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point, force_remount=force)
    print(f"✓ Drive montado en {mount_point}")
    return True


# ── Carga ───────────────────────────────────────────────────────────────

def load_tables(cfg: Config, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Devuelve las 7 tablas como dict. Las tablas de `data.optional` que
    falten se devuelven vacías, para poder empezar solo con Q1–Q3."""
    if cfg.is_synthetic:
        from .synth import generate

        s = cfg.synthetic
        if verbose:
            print(f"⚙️  Generando datos sintéticos "
                  f"({s.get('n_customers')} clientes × {s.get('n_months')} meses)…")
        tables = generate(
            n_customers=int(s.get("n_customers", 3000)),
            n_months=int(s.get("n_months", 36)),
            start_month=str(s.get("start_month", "2022-07-01")),
            annual_churn_rate=float(s.get("annual_churn_rate", 0.09)),
            seed=int(s.get("seed", 42)),
        )
    else:
        tables = _load_from_disk(cfg, verbose=verbose)

    tables = {k: _coerce_dtypes(k, df) for k, df in tables.items()}
    if verbose:
        _print_summary(tables)
    return tables


def _load_from_disk(cfg: Config, verbose: bool = True) -> dict[str, pd.DataFrame]:
    root = Path(cfg.data["drive_dir"])
    fmt = cfg.data.get("file_format", "csv")
    optional = set(cfg.data.get("optional", []))

    if not root.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de datos: {root}\n"
            "  → ¿Montaste Drive? Corre `mount_drive()` primero.\n"
            "  → ¿La ruta en config.yaml (data.drive_dir) es la correcta?"
        )

    out: dict[str, pd.DataFrame] = {}
    for key, stem in cfg.data["files"].items():
        path = root / f"{stem}.{fmt}"
        if not path.exists():
            # Tolerar .csv.gz y .parquet indistintamente
            alts = list(root.glob(f"{stem}.*"))
            path = alts[0] if alts else path
        if not path.exists():
            if key in optional:
                if verbose:
                    print(f"⚠️  {key:14s} no encontrada ({stem}.{fmt}) — se omite")
                out[key] = pd.DataFrame()
                continue
            raise FileNotFoundError(
                f"Falta la tabla obligatoria '{key}': {path}\n"
                f"  Archivos presentes: {[p.name for p in root.iterdir()][:20]}"
            )
        out[key] = _read(path)
        if verbose:
            print(f"✓ {key:14s} {len(out[key]):>9,} filas  ← {path.name}")
    return out


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _coerce_dtypes(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fecha y normaliza as_of_month a inicio de mes."""
    if df.empty:
        return df
    schema = TABLE_SCHEMAS.get(table, {})
    for col in schema.get("dates", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "as_of_month" in df.columns:
        df["as_of_month"] = df["as_of_month"].dt.to_period("M").dt.to_timestamp()
    return df


# ── Validación y perfilado ──────────────────────────────────────────────

def validate(tables: dict[str, pd.DataFrame], strict: bool = False) -> dict[str, list[str]]:
    """Verifica cada tabla contra el contrato de `TABLE_SCHEMAS`.

    Devuelve {tabla: [problemas]}. Con strict=True lanza excepción si hay
    columnas obligatorias faltantes.
    """
    problems: dict[str, list[str]] = {}

    for name, schema in TABLE_SCHEMAS.items():
        df = tables.get(name)
        if df is None or df.empty:
            continue
        issues = []

        missing = [c for c in schema["required"] if c not in df.columns]
        if missing:
            issues.append(f"faltan columnas obligatorias: {missing}")

        keys = [k for k in schema["keys"] if k in df.columns]
        if keys and len(keys) == len(schema["keys"]):
            dupes = int(df.duplicated(subset=keys).sum())
            if dupes:
                issues.append(f"{dupes:,} filas duplicadas en la llave {keys}")

        for c in schema["required"]:
            if c in df.columns:
                null_pct = float(df[c].isna().mean())
                if null_pct > 0.02:
                    issues.append(f"'{c}' tiene {null_pct:.1%} de nulos")

        if issues:
            problems[name] = issues

    if problems:
        print("\n⚠️  Problemas de validación:")
        for t, iss in problems.items():
            for i in iss:
                print(f"   [{t}] {i}")
        if strict:
            raise ValueError("Validación fallida (strict=True)")
    else:
        print("✓ Validación de esquemas: sin problemas")
    return problems


def profile(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Perfil de calidad por tabla: filas, llaves distintas, rango de fechas,
    % de nulos promedio. Es el primer paso del EDA."""
    rows = []
    for name, df in tables.items():
        if df is None or df.empty:
            rows.append({"tabla": name, "filas": 0, "columnas": 0})
            continue
        schema = TABLE_SCHEMAS.get(name, {})
        rec = {
            "tabla": name,
            "filas": len(df),
            "columnas": df.shape[1],
            "nulos_prom_%": round(float(df.isna().mean().mean()) * 100, 2),
            "mem_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        }
        for k in ("cif_id", "account_id", "loan_id"):
            if k in df.columns:
                rec[f"{k}_distintos"] = int(df[k].nunique())
        dcol = "as_of_month" if "as_of_month" in df.columns else (
            schema.get("dates", [None])[0])
        if dcol and dcol in df.columns and df[dcol].notna().any():
            rec["desde"] = str(df[dcol].min().date())
            rec["hasta"] = str(df[dcol].max().date())
        rows.append(rec)
    return pd.DataFrame(rows)


def _print_summary(tables: dict[str, pd.DataFrame]) -> None:
    total = sum(len(d) for d in tables.values() if d is not None)
    print(f"\n📦 {len([d for d in tables.values() if d is not None and not d.empty])}"
          f" tablas cargadas, {total:,} filas en total")


def save_processed(df: pd.DataFrame, name: str, cfg: Config) -> Path:
    """Guarda un artefacto intermedio (panel, features) en processed_dir."""
    path = cfg.processed_path() / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"💾 Guardado: {path}  ({len(df):,} filas)")
    return path


def load_processed(name: str, cfg: Config) -> pd.DataFrame | None:
    path = cfg.processed_path() / f"{name}.parquet"
    if path.exists():
        print(f"📂 Reusando: {path}")
        return pd.read_parquet(path)
    return None
