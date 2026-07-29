# Extracción en formato ancho (ODBC → Excel)

Cómo pedirle los datos al agente SQL cuando la bajada pasa por ODBC/Excel y el
formato largo no cabe.

## El problema

El pipeline modela un panel mensual. En formato largo eso son
`clientes × meses × (cuentas, categorías)` filas, y Excel se queda en
**1.048.576 filas**. Medido sobre datos sintéticos, por cada 1.000 clientes y
36 meses:

| Tabla | Filas / cliente | A 50.000 CIFs |
|:--|--:|--:|
| **transactions** | 509 | **25,4M** |
| deposits | 85 | 4,2M |
| party | 36 | 1,8M |
| digital | 36 | 1,8M |
| loans | 8,6 | 430k |
| relationships | 3,7 | 185k |
| events | 3,2 | 160k |

`transactions` es el 75% del volumen. Con 25.000–100.000 CIFs, el formato largo
no es viable para esa tabla ni para `deposits`.

## La solución: los meses van a columnas

El tope de **columnas** de Excel es 16.384, y ahí sí cabe todo. La regla es una
sola:

> Cada métrica mensual se emite como `<metrica>_<AAAA>_<MM>`, una columna por mes.

El pipeline detecta la orientación solo. No hay que declarar nada: si una tabla
trae columnas con mes en el nombre y no trae `as_of_month`, se despivota al
cargar y el resto del código no se entera. Se aceptan tres separadores:

```
eom_balance_2024_07    eom_balance_2024-07    eom_balance_202407
```

## Qué pedir, tabla por tabla

### Q1 · party — **una fila por CIF**

El cambio más rentable y el más fácil. Hoy se pide una fila por CIF **y por
mes**; como los CIF no se reutilizan, ahí no cambia nada salvo la antigüedad,
que el pipeline recalcula. Pásalo a maestro: **50.000 filas en vez de 1,8M**.

Columnas: `cif_id`, `entity_type`, `customer_since_date`, y las estáticas que
tengas (`segment`, `state`, `residency_status`, `employee_flag`,
`kyc_risk_rating`, `household_id`).

> **Pide `deceased_date`, no `deceased_flag`.** Una bandera dice "hoy está
> fallecido" pero no desde cuándo. En un maestro sin dimensión temporal eso
> marca al cliente durante toda su historia y lo saca del panel entero, en vez
> de excluirlo desde el evento. Si solo tienes la bandera, el pipeline avisa.

### Q3 · deposits — ancho por mes, una fila por cuenta

- **Índice**: `account_id` + estáticas de la cuenta (`product_family`,
  `product_code`, `open_date`, `close_date`, `close_reason_code`,
  `maturity_date`, `term_months`).
- **Ancho**: `eom_balance_AAAA_MM` y, si están, `avg_daily_balance_month_*`,
  `nsf_count_month_*`, `fees_charged_month_*`, `overdraft_days_count_*`,
  `account_status_*`, `dormant_flag_*`.

A 50.000 CIFs: ~120.000 filas × ~400 columnas.

### Q5 · transactions — ancho por mes **y por categoría**

Aquí no basta con pivotar el mes: quedarían `cuenta × categoría`, todavía por
encima del millón. La categoría también va en el nombre:

```
txn_count_<categoria>_<AAAA>_<MM>
total_amount_debit_<categoria>_<AAAA>_<MM>
total_amount_credit_<categoria>_<AAAA>_<MM>
```

- **Índice**: `account_id`.
- No hace falta que las categorías estén en el catálogo del pipeline: cualquier
  nombre se carga igual (avisa de las que no reconoce).
- Las celdas sin actividad pueden ir vacías; se descartan al cargar.

**Los dos flags de comportamiento no caben en ese esquema** — son atributos de
la fila, no del par categoría×mes. Emítelos ya agregados, como métricas
mensuales propias a nivel de cuenta:

```
dd_txn_count_<AAAA>_<MM>              transacciones de nómina del mes
competitor_outflow_amt_<AAAA>_<MM>    salida a otra institución
competitor_txn_count_<AAAA>_<MM>
```

Si solo necesitas recortar, pide únicamente las categorías que el pipeline
explota: `ach_credit_in`, `ach_debit_out`, `zelle_out`, `pos_signature`,
`bill_pay`, `atm_foreign`, `wire_out`, `branch_deposit`, `mobile_deposit`.

### Q4 · loans — ancho por mes, una fila por préstamo

- **Índice**: `loan_id` + estáticas (`loan_type`, `origination_date`,
  `original_amount`, `maturity_date`, `payoff_date`).
- **Ancho**: `current_balance_*`, `months_to_maturity_*` (o
  `remaining_term_months_*`), `days_past_due_*`, `status_*`, `utilization_*`.

### Q6 · digital — ancho por mes, una fila por CIF

- **Índice**: `cif_id`.
- **Ancho**: `login_count_web_month_*`, `login_count_mobile_month_*`,
  `bill_pay_payees_active_*`, `call_center_calls_month_*`,
  `branch_visits_month_*`, `complaints_open_month_*`, y las banderas de
  enrolamiento.

### Q2 · relationships y Q7 · events — **sin cambios**

No tienen dimensión temporal: 185k y 160k filas a 50.000 CIFs. Se quedan en
formato largo tal como están hoy.

## Resultado

| Tabla | Antes (largo) | Después (ancho) |
|:--|--:|--:|
| transactions | 25,4M filas | ~120k × ~1.500 cols |
| deposits | 4,2M filas | ~120k × ~400 cols |
| party | 1,8M filas | **50k × ~10 cols** |
| digital | 1,8M filas | 50k × ~290 cols |
| loans | 430k filas | ~30k × ~300 cols |
| relationships | 185k filas | sin cambio |
| events | 160k filas | sin cambio |

Ninguna tabla pasa del millón de filas ni de las 16.384 columnas.

## Verificación

El round-trip está probado: se generan datos sintéticos en formato largo, se
exportan al formato ancho de este documento, se vuelven a cargar y se compara
el panel. Coinciden exactamente — mismas filas, misma tasa de churn, cero
etiquetas distintas, y `owned_balance`, `txn_count_total`, `n_active_loans` y
`months_since_activity` con diferencia máxima 0.

## Aviso de memoria

Resolver la extracción no resuelve el tamaño en Colab. Con 50.000 CIFs y 36
meses, el panel con features ronda 1,1M de filas × ~330 columnas: unos 3 GB en
`float64`, más lo que ocupen las tablas crudas. En Colab gratuito (~12 GB) va
justo. Si aprieta: baja a `float32`, recorta `history_months`, o modela sobre
una muestra estratificada de clientes — la tasa base se conserva y el panel se
divide por el factor de muestreo.
