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

import re
import sys
from pathlib import Path

import pandas as pd

from .config import KNOWN_TXN_CATEGORIES, TABLE_SCHEMAS, Config


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

    tables = {k: unpivot_month_columns(k, df, verbose=verbose)
              for k, df in tables.items()}
    if "transactions" in tables:
        tables["transactions"] = unpivot_txn_categories(
            tables["transactions"], verbose=verbose)
    tables = {k: _coerce_dtypes(k, df) for k, df in tables.items()}
    tables = _expand_static_party(tables, verbose=verbose)
    if verbose:
        _print_summary(tables)
    return tables


# ── Formato ancho (una columna por mes) ─────────────────────────────────
#
# Bajar el panel en formato largo por ODBC a Excel es inviable en cuanto hay
# decenas de miles de clientes: el tope de Excel son 1.048.576 filas y la
# tabla de transacciones sola se pasa por un orden de magnitud. La salida es
# pivotar los meses a columnas — el tope de columnas es 16.384, que sobra.
#
# El pipeline acepta las dos orientaciones y no hay que declarar cuál es: si
# una tabla trae columnas con un mes en el nombre y no trae `as_of_month`,
# se despivota aquí y el resto del código no se entera.
#
# Nombres reconocidos, con el mes al final:
#     eom_balance_2024_07   eom_balance_2024-07   eom_balance_202407
#
_MONTH_SUFFIX = re.compile(
    r"^(?P<metric>.+?)[_\-](?P<year>20\d{2})[_\-]?(?P<month>0[1-9]|1[0-2])$"
)


def _restore_numeric(df: pd.DataFrame, skip: tuple[str, ...] = ()) -> pd.DataFrame:
    """Devuelve su tipo numérico a las columnas que salen del despivote.

    Fundir todas las métricas en una sola columna de valores obliga a un dtype
    común, y basta una métrica de texto (`account_status`) para que todas
    acaben como `object`. Con ese dtype, `select_dtypes(number)` deja de verlas
    y el relleno de nulos del panel no las alcanza. Se reconvierte solo cuando
    no se pierde ningún valor por el camino.
    """
    for c in df.columns:
        if c in skip or not pd.api.types.is_object_dtype(df[c]):
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() == df[c].notna().sum():
            df[c] = conv
    return df


def _split_month_columns(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """{columna: (métrica, 'YYYY-MM')} para las columnas con mes en el nombre."""
    out = {}
    for c in df.columns:
        m = _MONTH_SUFFIX.match(str(c))
        if m:
            out[c] = (m.group("metric"), f"{m.group('year')}-{m.group('month')}")
    return out


def unpivot_month_columns(table: str, df: pd.DataFrame,
                          verbose: bool = True) -> pd.DataFrame:
    """Convierte un extracto ancho (un mes por columna) a formato largo.

    No toca nada si la tabla ya viene larga o no tiene columnas de mes.
    """
    if df is None or df.empty or "as_of_month" in df.columns:
        return df

    month_cols = _split_month_columns(df)
    if not month_cols:
        return df

    id_cols = [c for c in df.columns if c not in month_cols]
    # El pivote va sobre un índice de fila sintético, no sobre las columnas
    # identificadoras: `pivot_table` descarta las filas con NaN en el índice, y
    # las estáticas nulas son la norma (`close_date` en una cuenta abierta,
    # `maturity_date` en una cuenta corriente). Con las identificadoras en el
    # índice se perdería la mayor parte del extracto en silencio.
    src = df.reset_index(drop=True)
    src["_row"] = range(len(src))

    long = src.melt(id_vars=["_row"], value_vars=list(month_cols),
                    var_name="_col", value_name="_val")
    meta = long["_col"].map(month_cols)
    long["_metric"] = meta.str[0]
    long["as_of_month"] = pd.to_datetime(meta.str[1], format="%Y-%m")
    long = long.drop(columns="_col")

    pivoted = (long.pivot_table(index=["_row", "as_of_month"], columns="_metric",
                               values="_val", aggfunc="first")
               .reset_index().rename_axis(columns=None))
    pivoted = _restore_numeric(pivoted, skip=("_row", "as_of_month"))
    wide = (src[id_cols + ["_row"]].merge(pivoted, on="_row", how="inner")
            .drop(columns="_row"))

    if verbose:
        n_metrics = len({m for m, _ in month_cols.values()})
        n_months = len({d for _, d in month_cols.values()})
        print(f"   ↔️  {table:14s} ancho → largo: {len(df):,} filas × "
              f"{len(month_cols)} columnas de mes "
              f"({n_metrics} métricas × {n_months} meses) → {len(wide):,} filas")
    return wide


# Métricas de la tabla de transacciones que pueden venir con la categoría
# pegada al nombre de la columna.
_TXN_VALUE_COLS = ("txn_count", "total_amount_debit", "total_amount_credit")


def unpivot_txn_categories(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Despivota la categoría de transacción cuando viene en el nombre.

    `transactions` es, con diferencia, la tabla más grande: con 50.000 CIFs
    son ~25M de filas en formato largo. Pivotar solo el mes no alcanza —quedan
    `cuenta × categoría`, todavía por encima del millón—, así que la categoría
    también acaba en el nombre de la columna:

        txn_count_zelle_out_2024_07

    El despivote del mes deja `txn_count_zelle_out`; esta función lo parte en
    `txn_category='zelle_out'` y `txn_count`. Las filas sin actividad se
    descartan: materializar cuenta × mes × categoría reproduciría en memoria
    justo la explosión que el formato ancho evita.
    """
    if df is None or df.empty or "txn_category" in df.columns:
        return df

    # La categoría se deduce quitando el prefijo de la métrica, no contrastando
    # contra una lista cerrada: cualquier categoría que el core emita y no esté
    # en KNOWN_TXN_CATEGORIES se perdería en silencio, y son justo las
    # específicas del banco (p. ej. `ach_debit_out_competitor`) las que más
    # señal traen.
    mapping: dict[str, tuple[str, str]] = {}
    for c in df.columns:
        name = str(c)
        for metric in _TXN_VALUE_COLS:
            if name.startswith(f"{metric}_"):
                mapping[c] = (metric, name[len(metric) + 1:])
                break
    if not mapping:
        return df

    unknown = {cat for _, cat in mapping.values()} - set(KNOWN_TXN_CATEGORIES)
    if unknown and verbose:
        print(f"   ℹ️  categorías fuera del catálogo, se cargan igual: "
              f"{sorted(unknown)}")

    id_cols = [c for c in df.columns if c not in mapping]
    # Mismo cuidado que en el despivote del mes: índice de fila sintético para
    # que las identificadoras nulas no se lleven la fila por delante.
    src = df.reset_index(drop=True)
    src["_row"] = range(len(src))

    long = src.melt(id_vars=["_row"], value_vars=list(mapping),
                    var_name="_col", value_name="_val")
    meta = long["_col"].map(mapping)
    long["_metric"] = meta.str[0]
    long["txn_category"] = meta.str[1]
    long = long.drop(columns="_col")

    pivoted = (long.pivot_table(index=["_row", "txn_category"], columns="_metric",
                               values="_val", aggfunc="first")
               .reset_index().rename_axis(columns=None))
    out = (src[id_cols + ["_row"]].merge(pivoted, on="_row", how="inner")
           .drop(columns="_row"))
    if "txn_count" in out.columns:
        out = out[out["txn_count"].fillna(0) > 0].reset_index(drop=True)

    if verbose:
        n_cat = len({c for _, c in mapping.values()})
        print(f"   ↔️  transactions   categoría en columna → filas: "
              f"{n_cat} categorías → {len(out):,} filas con actividad")
    return out


def _expand_static_party(tables: dict[str, pd.DataFrame],
                         verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Expande un maestro de clientes sin dimensión temporal a CIF×mes.

    Con CIFs que no se reutilizan, pedir una fila por cliente y por mes
    multiplica el extracto por el número de meses sin aportar nada: los
    atributos del maestro no cambian. La única excepción es la antigüedad,
    que aquí se recalcula mes a mes desde `customer_since_date`.

    La rejilla de meses sale de `deposits`, que es la tabla que define el
    calendario del panel.
    """
    party = tables.get("party")
    if party is None or party.empty or "as_of_month" in party.columns:
        return tables

    deposits = tables.get("deposits")
    if deposits is None or deposits.empty or "as_of_month" not in deposits.columns:
        raise ValueError(
            "El maestro de clientes no trae `as_of_month` y no hay tabla de "
            "depósitos de la que sacar el calendario de meses.\n"
            "  → O bajas `party` con una fila por CIF y por mes, o te aseguras "
            "de que `deposits` traiga `as_of_month`."
        )

    months = pd.DatetimeIndex(sorted(deposits["as_of_month"].dropna().unique()))
    grid = party.merge(pd.DataFrame({"as_of_month": months}), how="cross")

    # No inventar historia anterior al alta del cliente
    if "customer_since_date" in grid.columns:
        since = pd.to_datetime(grid["customer_since_date"], errors="coerce")
        grid = grid[since.isna() | (since <= grid["as_of_month"])]
        since = pd.to_datetime(grid["customer_since_date"], errors="coerce")
        # La antigüedad del maestro es la de hoy; mes a mes hay que recalcularla
        grid["tenure_months"] = (
            (grid["as_of_month"].dt.year - since.dt.year) * 12
            + (grid["as_of_month"].dt.month - since.dt.month)
        ).clip(lower=0)

    for col in ("cif_close_date", "closed_date", "relationship_end_date"):
        if col in grid.columns:
            end = pd.to_datetime(grid[col], errors="coerce")
            grid = grid[end.isna() | (end > grid["as_of_month"])]
            break

    grid = grid.reset_index(drop=True)
    if "deceased_flag" in grid.columns and "deceased_date" not in grid.columns:
        print("   ⚠️  El maestro trae `deceased_flag` pero no `deceased_date`. "
              "Una bandera sin fecha marca al cliente como fallecido durante "
              "toda su historia y lo saca del panel entero, no desde el "
              "evento. Pide `deceased_date` en el extracto.")
    if verbose:
        print(f"   ↔️  party          maestro estático → panel: "
              f"{len(party):,} CIFs × {len(months)} meses → {len(grid):,} filas")
    tables = dict(tables)
    tables["party"] = grid
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
