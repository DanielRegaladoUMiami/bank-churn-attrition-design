"""Pipeline de churn/attrition bancario.

Uso típico::

    from churn.config import Config
    from churn.data import mount_drive, load_tables
    from churn.panel import build_panel
    from churn.features import build_features, select_feature_columns

    cfg = Config.load("config.yaml")
    mount_drive()                      # no-op fuera de Colab
    tables = load_tables(cfg)
    panel = build_panel(tables, cfg)
    feats = build_features(panel, cfg, tables)
"""

__version__ = "0.1.0"

from .config import TABLE_SCHEMAS, Config

__all__ = ["Config", "TABLE_SCHEMAS", "__version__"]
