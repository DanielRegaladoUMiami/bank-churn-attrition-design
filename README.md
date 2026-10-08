# Modelo de Churn / Attrition — Banca Retail y Comercial

Pipeline de predicción de attrition de clientes (CIF), cubriendo depósitos y
préstamos. Metodología genérica de banca — no contiene esquemas, datos ni
información de ninguna institución en particular.

> ⚠️ Este repositorio contiene **solo diseño, metodología y código**. No debe
> contener datos de clientes, extractos, credenciales, nombres de tablas reales ni
> PII. Ver [`.gitignore`](.gitignore).

---

## El repositorio en tres archivos

| Archivo | Qué es |
|:---|:---|
| **`README.md`** | Este documento. La metodología completa: por qué el target es el que es, cómo se tratan los roles del CIF, cómo se evita el leakage, qué pedirle al agente SQL y qué exige Compliance. |
| **`notebooks/churn_pipeline_colab.ipynb`** | **Todo el pipeline, en un solo notebook autocontenido.** No importa nada del repositorio ni clona nada: lleva dentro el contrato de datos, el generador sintético, el panel, las features, los modelos y la explicabilidad. |
| **`notebooks/eda_households_cifs_accounts.ipynb`** | **EDA — Households, CIFs & Accounts.** Toda la historia del banco a los tres niveles de granularidad (household, CIF, cuenta) y los roles que los unen (primary owner, joint, signer, guarantor…), sobre el extracto CIF × cuenta × rol. Ver [abajo](#eda--households-cifs--accounts). |

No hay módulos ni scripts aparte. Cada notebook es la única fuente de verdad de su
código; el README, de las decisiones.

### EDA — Households, CIFs & Accounts

Notebook autocontenido, en el mismo estilo que el EDA de CIFs activos (paleta, helpers,
tablas al lado de cada gráfico, observaciones por sección). Se abre en Colab:

```
https://colab.research.google.com/github/DanielRegaladoUMiami/bank-churn-attrition-design/blob/main/notebooks/eda_households_cifs_accounts.ipynb
```

- **Datos de entrada:** un extracto con una fila por CIF × cuenta × rol (household, CIF,
  tipo de persona, residencia, fechas, segmento, rol, ownership category, cuenta, dominio
  DD / CD / LN, producto, Select, sucursal, oficial, status, fechas de apertura y cierre,
  saldo, país). Los nombres de columna se mapean en una sola celda de configuración
  (`COLS`, basta un prefijo), igual que todas las reglas de negocio.
- **`FILE_PATH = 'DEMO'`** corre todo el notebook sobre datos sintéticos con las mismas
  columnas y los mismos problemas de calidad (fechas de tipos mezclados, households
  fallback, cuentas sin primary owner o con dos, números de cuenta repetidos entre
  sistemas). Sirve para verlo funcionar antes de cargar el extracto real.
- **Reglas que deciden todo lo demás:** el saldo se cuenta **una sola vez, en el primary
  owner**; cliente activo = dueño de al menos una cuenta abierta; CIF *connected* = sin
  cuentas propias pero con un rol en una cuenta abierta; ex-cliente = nada abierto en
  ningún rol. La edad detrás de un business sale del **key person** (beneficial owner →
  guarantor → joint → co-borrower → signer…) de sus propias cuentas.
- **Roadmap en cuatro preguntas:**
  - *Parte 0 — Fundamentos:* modelo de datos y calidad (§3–4), tablas de análisis y grupos especiales (§5), el banco hoy (§6).
  - *Parte I — Quién es el cliente:* CIFs en todas las combinaciones IorB × ForD × Select, value tiers y familias de segmento (§7); la persona detrás de cada business (§8).
  - *Parte II — A qué está conectado:* households (§9); roles y redes de relación (§10).
  - *Parte III — Qué tiene:* cuentas y productos (§11).
  - *Parte IV — Cómo performa:* valor y contribución estimada (§12, con supuestos explícitos hasta tener profitability real); historia año a año (§13); attrition (§14).
  - *Parte V — Síntesis:* tests (§15), perfiles (§16), hallazgos y exports para Power BI (§17), preguntas abiertas (§18).
- **Grupos especiales, no exclusiones:** ejecutivos / empleados, vehículos EB-5, large
  relationships y portafolios institucionales (por oficial o sucursal) se etiquetan en
  `SPECIAL_GROUPS` y se analizan aparte (§6.4); el resto del notebook describe a los
  clientes *core*. Los valores (nombres, IDs, oficiales) se llenan en la copia propia,
  nunca en el repositorio.
- Ningún nombre de tabla fuente ni de la institución vive en el notebook; los outputs
  (que llevan datos de clientes) no se suben al repositorio.

## Arranque rápido

Ábrelo en Colab y ejecútalo de arriba abajo:

```
https://colab.research.google.com/github/DanielRegaladoUMiami/bank-churn-attrition-design/blob/main/notebooks/churn_pipeline_colab.ipynb
```

El notebook viene con `data.source = "synthetic"`, así que corre entero sin datos
reales: genera un panel simulado con dinámica causal y recorre las 22 secciones
hasta producir la lista de retención. Sirve para validar el pipeline y para ver qué
forma tiene cada salida antes de tener los extractos.

Para pasar a datos reales, dos cambios:

1. Sube los extractos a `MyDrive/churn_model/data/raw/` con los nombres de la
   [sección 5](#5-qué-pedirle-al-agente-sql).
2. En la celda de configuración, `"source": "drive"`.

Nada más. Q4–Q7 son opcionales: se puede empezar a modelar con Q1–Q3.

### Estructura del notebook

**Parte I — el pipeline.** Nueve secciones de definiciones; ejecutarlas no calcula
nada. Configuración → contrato de datos → generador sintético → carga y despivote →
panel y target → features → evaluación → modelos → explicabilidad.

**Parte II — el análisis.** Trece secciones que sí ejecutan: carga, EDA, panel,
features, split, tuning, stacking, GRU, resultados, calibración, SHAP,
contrafactuales y lista de scoring.

La única celda que hay que tocar es la de **configuración**, al principio de la
Parte I. De ahí sale todo el comportamiento del pipeline.

---

## Tabla de contenido

1. [Definición del target](#1-definición-del-target--aquí-es-donde-se-gana-o-se-pierde)
2. [El problema de los roles del CIF](#2-el-problema-de-los-roles-del-cif)
3. [Estructura de panel — evitar leakage](#3-estructura-de-panel--evitar-leakage)
4. [Preguntas abiertas sobre la data](#4-preguntas-concretas-sobre-la-data)
5. [Spec para el agente SQL](#5-qué-pedirle-al-agente-sql)
6. [Ajustes al pipeline de modelado](#6-ajustes-al-pipeline)
7. [Riesgo regulatorio](#7-riesgo-regulatorio)

---

## 1. Definición del target — aquí es donde se gana o se pierde

En banca no existe un botón de "cancelar suscripción". Hay que construir el evento, y
hay decisiones que cambian por completo el modelo.

### a) ¿Attrition de relación o de producto?

**Recomendación:** modelar attrition total de relación a nivel CIF como target principal,
y llevar dos targets secundarios (deposit attrition, loan attrition) para diagnóstico.

Razón: un cliente que cierra un CD pero mantiene su checking con nómina no se fue; un
cliente que cierra el checking pero mantiene un CD sí está en fuga. Si se mezclan ambos,
el modelo aprende ruido de rolling de CDs.

### b) Cierre formal vs. attrition silenciosa

La trampa clásica: la mayoría de los clientes que se van **nunca cierran la cuenta**.
La dejan con $12 y la ignoran. Si se entrena solo con `account_close_date`, se entrena
sobre la minoría organizada.

Definición compuesta propuesta a nivel CIF, evaluada mensualmente:

> **Attrited en el mes `t`** = (todas las relaciones activas cerradas)
> **OR** (saldo total de depósitos < umbral, p. ej. $100 o 5% del promedio de sus 12
> meses previos) **AND** (0 transacciones iniciadas por el cliente en 90 días)
> **AND** (0 préstamos activos) — sostenido por 2–3 meses consecutivos para evitar
> falsos positivos por estacionalidad.

Los umbrales se calibran contra la distribución real de saldos y conteo de
transacciones. No inventarlos a ojo.

### c) Voluntario vs. involuntario — hay que excluirlos, no predecirlos

Marcar y sacar de la población (o modelar aparte):

- Fallecimiento
- Cierre por decisión del banco (AML/BSA exit, fraude, overdraft crónico)
- Charge-off
- Cuentas de empleados
- Cuentas internas / GL
- Cuentas dormant transferidas a **escheatment** (unclaimed property al estado)

Si el 15% de los "cierres" son exits de compliance, el modelo aprende a predecir
riesgo AML, no churn.

### d) Loans: pagar el préstamo NO es churn

Hay que partirlo en tres eventos distintos:

| Evento | ¿Es attrition? | Notas |
|---|---|---|
| **Maturity natural** | No | Es el contrato cumpliéndose. Es un momento de *riesgo*, no el evento. |
| **Prepago / payoff anticipado** | Sí | Alta señal de refinanciamiento en otro banco, sobre todo mortgage y auto. |
| **Charge-off / foreclosure** | No (involuntaria) | Excluir. |

Para distinguir prepago de maturity se necesita `maturity_date` vs. `payoff_date` y el
`remaining_term` al momento del payoff. Un payoff con >6 meses de término remanente y
sin refinanciamiento interno = **prepago competitivo**.

El mes en que el balance del loan cruza hacia cero, si el cliente no tiene depósitos con
actividad, es el momento de máxima probabilidad de fuga. Se captura con una feature de
`months_to_payoff` proyectado desde la amortización — probablemente una de las features
más potentes del modelo.

---

## 2. El problema de los roles del CIF

Un CIF puede aparecer como primary, joint, authorized signer, custodian, beneficiary,
guarantor, trustee, POA. Tres consecuencias:

**No se pueden sumar saldos ingenuamente.** Si se hace `SUM(balance) GROUP BY cif`, una
cuenta joint de $50k con dos titulares genera $100k de depósitos en el banco. Rompe
cualquier feature de share-of-wallet.

### Ownership weight propuesto

| Rol | Peso económico | ¿Cuenta como "producto propio"? |
|---|---|---|
| Primary / Sole owner | 1.0 | Sí |
| Joint owner | 1/n titulares (o 0.5) | Sí |
| Custodian (UTMA), Trustee | 1.0 (control) | Sí, pero flag aparte |
| Authorized signer | 0.0 | No — pero **es una feature fuertísima** |
| Beneficiary (POD) | 0.0 | No |
| Guarantor (loan) | 0.0 | Flag aparte |

### Dos vistas de features por CIF

1. **Económica** — saldos ponderados por ownership, para share-of-wallet y tendencias.
2. **Relacional** — conteos sin ponderar: en cuántas cuentas aparece con cualquier rol,
   cuántas personas comparten cuentas con él.

Un authorized signer en el negocio de su cónyuge no tiene saldo propio pero tiene un
ancla relacional enorme.

### El target también depende del rol

Un cliente que solo es beneficiary no puede "irse". Población elegible propuesta:

> CIFs con al menos una relación de ownership (primary/joint/custodian/trustee) **o** un
> préstamo como borrower/co-borrower, en el mes de observación.

### Household / contagio

Si se pueden derivar hogares (dirección + apellido, o joint-account graph), el churn del
co-titular es probablemente la feature individual más predictiva del portafolio.
**Pendiente:** verificar si existe un `household_id` en el core.

---

## 3. Estructura de panel — evitar leakage

No hacer un dataset de "una fila por cliente". Hacer un **panel mensual con ventanas
móviles**:

```
[--- FEATURE WINDOW: 12-18 meses ---][GAP: 1 mes][--- PERFORMANCE WINDOW: 6 meses ---]
                                       ^                    ^
                                  as-of date          ¿churneó aquí?
```

- **Feature window:** 12 meses mínimo (para tendencias YoY y estacionalidad),
  idealmente 18.
- **Gap / latency window de 1–2 meses:** *crítico y casi siempre olvidado.* Si se predice
  churn del mes `t+1` con datos del mes `t`, para cuando el dato llega al warehouse y
  retención llama al cliente, ya se fue. El gap hace el modelo accionable. También
  protege de features que ya reflejan el churn en curso (saldo cayendo a cero el mes
  anterior al cierre no es predicción, es observación).
- **Performance window:** 6 meses como default en banca (3 es demasiado corto para
  capturar la fuga lenta, 12 diluye la señal). Opción: dos modelos, 3m (accionable,
  campañas) y 6–12m (estratégico).
- **Snapshots:** as-of dates mensuales sobre 24–36 meses de historia → cada CIF aparece
  muchas veces.

### Consecuencia directa para el cross-validation

Con panel, `StratifiedKFold` normal produce **leakage** — el mismo cliente en train y
test. Se necesita:

- **`StratifiedGroupKFold` con `groups=cif_id`** para el tuning.
- **Split out-of-time** para el holdout final (train hasta `2024-12`, test `2025-01`+),
  no aleatorio.

Idealmente ambos: OOT para el holdout final, StratifiedGroupKFold dentro del train.

---

## 4. Preguntas concretas sobre la data

### Historia y core

1. ✅ **Parcialmente resuelto.** Hay varios años de historia disponible — suficiente para
   panel mensual de 24–36 meses con feature window de 18.
   **Sigue abierto:** ¿hubo alguna **conversión de core bancario** o migración en ese
   periodo? Una conversión rompe IDs, reinicia fechas de apertura y genera un pico falso
   de "cierres". Diagnóstico rápido: graficar cierres y aperturas por mes sobre toda la
   historia; un pico anómalo aislado casi siempre es una migración, no comportamiento
   de clientes. Si existe, cortar la historia después de la conversión o mapear los IDs.
2. ¿Las tablas de saldos son **snapshots mensuales persistidos** o hay que reconstruirlos
   desde transacciones? ¿Existe historia SCD tipo 2 en el maestro de cuentas, o solo el
   estado actual (y por lo tanto no se puede reconstruir el pasado)?
3. ⚠️ **Abierto — hay que verificarlo antes de construir el panel.** ¿El CIF se
   **reutiliza o se duplica**? ¿Un cliente que se fue y volvió tiene el mismo CIF? ¿Hay
   proceso de deduplicación/merge de CIFs, y queda rastro (`merged_into_cif`)?

   **Por qué importa:** si el CIF se recicla, un "cliente" en el panel puede ser en
   realidad dos personas distintas, y el `groups=cif_id` del CV deja de proteger contra
   leakage. Si se duplica (misma persona, dos CIFs), se subestima la relación total y el
   share-of-wallet queda mal.

   **Cómo averiguarlo sin preguntarle a nadie** — tres queries diagnósticas:

   ```sql
   -- A. ¿Un mismo cif_id tiene más de una fecha de "customer since"?
   --    Más de una casi siempre significa reciclaje o merge.
   SELECT COUNT(*) AS cifs_con_multiples_fechas
   FROM (
       SELECT cif_id
       FROM party_master
       GROUP BY cif_id
       HAVING COUNT(DISTINCT customer_since_date) > 1
   ) t;

   -- B. "Resurrecciones": CIFs que quedaron sin ninguna relación activa por un
   --    periodo largo y luego reaparecen. Puede ser cliente que volvió (legítimo,
   --    y hay que decidir cómo tratarlo) o CIF reciclado (contaminante).
   WITH gaps AS (
       SELECT cif_id,
              relationship_start_date AS inicio,
              LAG(relationship_end_date) OVER (
                  PARTITION BY cif_id ORDER BY relationship_start_date
              ) AS fin_anterior
       FROM party_account_relationship
   )
   SELECT cif_id, fin_anterior, inicio,
          DATEDIFF(month, fin_anterior, inicio) AS meses_gap
   FROM gaps
   WHERE fin_anterior IS NOT NULL
     AND DATEDIFF(month, fin_anterior, inicio) > 12
   ORDER BY meses_gap DESC;

   -- C. Duplicados: distintos cif_id que comparten atributos de identidad.
   --    (Correrlo sobre hashes, nunca sobre PII en claro.)
   SELECT dob_hash, name_hash, COUNT(DISTINCT cif_id) AS n_cifs
   FROM party_master
   GROUP BY dob_hash, name_hash
   HAVING COUNT(DISTINCT cif_id) > 1;
   ```

   **Plan según el resultado:**
   - Si (A) o (B) salen con volumen material → construir una llave surrogate
     `entity_id = cif_id + episodio`, y usar **esa** como `groups` en el CV.
   - Si (C) sale material → resolver identidad antes de agregar saldos, o aceptar y
     documentar que el share-of-wallet está sesgado a la baja.
   - Si los tres salen limpios → usar `cif_id` directo y seguir.

### Sobre el evento

4. ¿Existe una tabla de **motivo de cierre** (`close_reason_code`) y qué tan poblada
   está? Define si se puede separar voluntario de involuntario.
5. ¿Cuál es la **tasa base** de attrition anual actual por segmento? Si es 5–8%, el
   desbalance es manejable; si es <2%, hay que replantear sampling y métricas.
6. ¿Se puede detectar **transferencia interna vs. externa**? Un cliente que cierra un
   savings para abrir un CD en el mismo banco no es churn.

### Sobre las señales

7. ¿Hay **detalle de transacciones** o solo agregados? Mínimo necesario: conteos y montos
   mensuales por categoría de canal.
8. ¿Se puede identificar **direct deposit / nómina ACH** (código de compañía ACH,
   descripción del originador)? La pérdida de la nómina es *el* predictor #1 de churn en
   checking, típicamente 3–6 meses antes del cierre.
9. ¿Hay visibilidad de **ACH/Zelle/wire salientes hacia otras instituciones
   financieras**? Transferencias recurrentes crecientes a otro banco = fuga en curso. Si
   se puede clasificar el receptor por routing number contra una lista de competidores,
   oro puro.
10. ¿Hay data de **canales digitales** (logins online/mobile, enrollment, última sesión)?
    La caída de logins es señal temprana.
11. ¿Existen datos de **servicio/quejas** (tickets, llamadas al call center, disputas,
    reclamos de fraude)? Una disputa mal resuelta predice churn a 90 días.
12. ¿Están las **tasas** que paga el banco por cuenta (sobre todo CDs) y alguna
    referencia de mercado? El diferencial de tasa al vencimiento de un CD explica la
    mayoría del rolloff.
13. ¿Hay historial de **fees cobrados** (NSF, monthly maintenance, overdraft)? Un pico de
    fees precede a cierres.

### Segmentación específica

14. ¿Qué peso tiene la banca **internacional / no residente**? Ese segmento tiene una
    dinámica de attrition completamente distinta (regulatoria, cambiaria, de remesas) y
    probablemente merece su propio modelo o al menos un feature de segmento muy
    explícito.
15. ¿Personal y business banking en el mismo modelo? **Recomendación: separarlos.** Un
    CIF business con signers múltiples tiene una física distinta. Si el volumen no da
    para dos modelos, incluir `entity_type` como feature y estratificar el CV por él.

---

## 5. Qué pedirle al agente SQL

**No pedir un mega-join.** Pedir **tablas largas y limpias, una por dominio**, con llaves
consistentes, y ensamblar el panel en Python (pandas/polars). Así se pueden recalcular
ventanas sin volver a golpear la base.

Llaves canónicas en todas: `cif_id`, `account_id`, `as_of_month` (primer día del mes,
tipo `DATE`).

### Query 1 — Party master (dimensión de cliente, SCD)

```
Una fila por cif_id por mes (as_of_month), últimos 36 meses, con:
cif_id, as_of_month, entity_type (personal/business), date_of_birth o age,
customer_since_date, tenure_months, residency_status (US resident / non-resident alien),
country_of_residence, country_of_citizenship, state, zip3 (no zip completo),
primary_branch_id, segment/tier code, employee_flag, kyc_risk_rating,
occupation/NAICS si es business, preferred_language, deceased_flag,
status (active/dormant/closed), merged_into_cif si existe.
Excluir cuentas internas/GL. No incluir nombre, SSN/ITIN, dirección exacta ni teléfono.
```

### Query 2 — Relación CIF ↔ cuenta (la tabla clave)

```
Una fila por (cif_id, account_id, relationship_role) con vigencia:
cif_id, account_id, product_type (deposit/loan), relationship_role
(primary, joint, authorized_signer, custodian, trustee, beneficiary,
 borrower, co_borrower, guarantor), relationship_start_date,
relationship_end_date, y n_owners = número total de titulares con rol de
ownership en esa cuenta.
Necesito la historia, no solo el estado actual: si una relación terminó,
quiero la fila con su end_date.
```

### Query 3 — Depósitos: maestro + snapshot mensual

```
Una fila por (account_id, as_of_month), últimos 36 meses:
account_id, as_of_month, product_code, product_family
(DDA/checking, savings, MMA, CD, IRA), open_date, close_date, close_reason_code,
account_status (open/dormant/closed/escheat),
eom_balance, avg_daily_balance_month, min_balance_month, max_balance_month,
interest_rate_paid, CD: maturity_date, term_months, auto_renew_flag,
original_deposit_amount,
overdraft_days_count, nsf_count_month, fees_charged_month (desglosado por tipo si se puede),
statement_delivery (paper/e-statement), dormant_flag, dormant_since_date.
```

### Query 4 — Loans: maestro + snapshot mensual

```
Una fila por (loan_id, as_of_month), últimos 36 meses:
loan_id, as_of_month, loan_type (mortgage, HELOC, auto, personal, credit_card,
 commercial RE, C&I, LOC), origination_date, original_amount, original_term_months,
maturity_date, current_balance, scheduled_payment, interest_rate, rate_type (fixed/variable),
remaining_term_months, months_to_maturity,
payments_made_count, extra_principal_paid_month,
delinquency_bucket (current/30/60/90+), days_past_due, times_30dpd_last_12m,
status (active/paid_off/charged_off/foreclosed), payoff_date, payoff_type si existe,
para líneas: credit_limit, utilization,
collateral_type, LTV si aplica, escrow_flag.
```

### Query 5 — Actividad transaccional agregada por mes

```
Una fila por (account_id, as_of_month, channel_or_category), últimos 36 meses.
Formato largo (una fila por categoría) para poder pivotear del lado de Python.
Métricas por celda: txn_count, total_amount_debit, total_amount_credit.

Categorías/canales que necesito separados:
- ACH credit entrante, y de esos, marcar direct_deposit / payroll
  (por SEC code PPD + company name/ID del originador)
- ACH débito saliente (y si es identificable, hacia otra institución financiera:
  incluir el routing number de destino o un flag competitor_transfer)
- Zelle/P2P entrante y saliente
- Wires entrantes/salientes (domestic vs international)
- Débito POS: signature vs PIN
- ATM: propio vs foreign, retiros
- Cheques emitidos, depósitos en sucursal, mobile deposit
- Bill pay (conteo de payees activos — ancla de retención)
- Transferencias internas entre cuentas propias

Además, a nivel account_id-mes: last_customer_initiated_txn_date,
days_since_last_customer_txn.
```

### Query 6 — Actividad digital y de contacto

```
Por (cif_id, as_of_month):
online_banking_enrolled_flag, mobile_enrolled_flag, enrollment_date,
login_count_web_month, login_count_mobile_month, last_login_date,
push/email/SMS alerts enrolled, e-statement flag,
debit_card_active_flag, card_expiry_date, card_reissue_events,
llamadas al call center (conteo mensual, y por tipo si existe),
visitas a sucursal si se registran,
tickets/quejas/disputas: conteo, tipo, fecha de apertura y resolución,
campañas de marketing recibidas y respuesta (si existe historia).
```

### Query 7 — Eventos de producto (formato de log)

```
Un log de eventos, una fila por evento:
cif_id, account_id, event_date, event_type (account_open, account_close,
product_switch, cd_renewal, cd_non_renewal, loan_origination, loan_payoff,
loan_charge_off, address_change, name_change, rate_change, fee_reversal,
overdraft_protection_link, joint_owner_added, joint_owner_removed),
event_detail, reason_code.

Los cambios de dirección fuera del footprint del banco y la remoción de
un co-titular son señales tempranas de fuga.
```

### Instrucciones transversales para el agente SQL

- Filtrar cuentas internas, GL, suspense, y cuentas de empleados — pero **etiquetarlas**,
  no borrarlas silenciosamente. Hay que saber cuántas son.
- Nada de PII directa: sin nombre, SSN/ITIN, número de cuenta real, dirección, teléfono,
  email. `account_id` y `cif_id` deben venir **hasheados/pseudonimizados de forma
  consistente entre todas las tablas**.
- Que devuelva también, para cada tabla, un **perfil de calidad**: row count, distinct
  keys, rango de fechas, % de nulos por columna. Ahorra medio EDA.
- Que documente cualquier filtro aplicado y cualquier columna que no exista en el core.

### Si la bajada pasa por ODBC/Excel: formato ancho

El panel en formato largo no cabe en Excel en cuanto hay decenas de miles de
clientes. El tope son **1.048.576 filas**, y medido sobre datos sintéticos, por cada
1.000 clientes y 36 meses:

| Tabla | Filas / cliente | A 50.000 CIFs |
|:--|--:|--:|
| **transactions** | 509 | **25,4M** |
| deposits | 85 | 4,2M |
| party | 36 | 1,8M |
| digital | 36 | 1,8M |
| loans | 8,6 | 430k |
| relationships | 3,7 | 185k |
| events | 3,2 | 160k |

`transactions` sola es el 75% del volumen. La salida es pivotar los meses a
columnas: el tope de **columnas** es 16.384 y ahí cabe todo. La regla es una:

> Cada métrica mensual se emite como `<metrica>_<AAAA>_<MM>`, una columna por mes.

El notebook detecta la orientación solo. No hay que declarar nada: si una tabla trae
columnas con el mes en el nombre y no trae `as_of_month`, se despivota al cargar. Se
aceptan `eom_balance_2024_07`, `eom_balance_2024-07` y `eom_balance_202407`.

**Q1 · party — una fila por CIF.** El cambio más rentable y el más fácil. Como los
CIF no se reutilizan, en el eje temporal no cambia nada salvo la antigüedad, que el
notebook recalcula desde `customer_since_date`. Pasarlo a maestro son **50.000 filas
en vez de 1,8M**.

> **Pide `deceased_date`, no `deceased_flag`.** La bandera dice "hoy está fallecido"
> pero no desde cuándo. En un maestro sin dimensión temporal eso marca al cliente
> durante toda su historia y lo saca del panel entero, en vez de excluirlo desde el
> evento. Si solo llega la bandera, el notebook avisa.

**Q5 · transactions — ancho por mes _y_ por categoría.** Aquí no basta con pivotar
el mes: quedarían `cuenta × categoría`, todavía por encima del millón. La categoría
va también en el nombre:

```
txn_count_<categoria>_<AAAA>_<MM>
total_amount_debit_<categoria>_<AAAA>_<MM>
total_amount_credit_<categoria>_<AAAA>_<MM>
```

Las categorías no tienen que estar en el catálogo del notebook: cualquier nombre se
carga igual, y avisa de las que no reconoce. Los dos flags de comportamiento no
caben en ese esquema —son atributos de la fila, no del par categoría×mes—, así que
se piden ya agregados como métricas mensuales de la cuenta:

```
dd_txn_count_<AAAA>_<MM>              transacciones de nómina del mes
competitor_outflow_amt_<AAAA>_<MM>    salida a otra institución
competitor_txn_count_<AAAA>_<MM>
```

**Q3, Q4, Q6 — ancho por mes.** Índice `account_id` / `loan_id` / `cif_id` más las
columnas estáticas, y el resto de métricas con sufijo de mes.

**Q2 y Q7 — sin cambios.** No tienen dimensión temporal: 185k y 160k filas a 50.000
CIFs. Se quedan en formato largo.

Resultado: ninguna tabla pasa del millón de filas ni de las 16.384 columnas. El
round-trip está probado — se generan datos en largo, se exportan a ancho, se
recargan y el panel reconstruido coincide exactamente: mismas filas, misma tasa de
churn, cero etiquetas distintas.

> **Resolver la extracción no resuelve la memoria.** Con 50.000 CIFs y 36 meses el
> panel con features ronda 1,1M de filas × ~330 columnas: unos 3 GB en `float64`. En
> Colab gratuito (~12 GB) va justo. Si aprieta: `float32`, menos historia, o una
> muestra estratificada de clientes.

---

## 6. Ajustes al pipeline

Sobre el pipeline base (EDA → feature engineering → preprocessing → split → CV →
ensemble → explicabilidad), cinco ajustes. **Todos están implementados** en el
notebook; entre paréntesis, la sección donde vive cada uno.

### 6.1 Split

Out-of-time para el holdout final (no aleatorio) + `StratifiedGroupKFold(groups=cif_id)`
para el tuning. Con panel mensual, `StratifiedKFold` puro es leakage garantizado.

### 6.2 Métricas

Olvidar accuracy y ROC-AUC como métrica primaria con 5% de base rate. Usar:

- **PR-AUC / average precision** como métrica de optimización.
- **Lift y capture rate en los deciles 1–3** como KPI de negocio — retención solo va a
  poder contactar el top 5–10% de la lista.

### 6.3 Calibración

Si la probabilidad va a alimentar un cálculo de valor esperado en riesgo
(`P(churn) × saldo × margen`), hacen falta probabilidades calibradas →
`CalibratedClassifierCV` (isotónica) sobre el holdout, y revisar la curva de calibración.
El ensemble/stacking suele descalibrar.

### 6.4 Redes neuronales — dónde sí aportan

En datos tabulares de banca, un GBM bien tuneado (LightGBM/XGBoost/CatBoost) casi siempre
le gana a un MLP. Donde la red **sí** aporta valor real es tratando el panel como
secuencia:

> **GRU/LSTM o Temporal Fusion Transformer sobre las series mensuales** de saldo,
> transacciones y logins — captura trayectorias (aceleración de la caída) que las
> features agregadas pierden.

Ese es el ensemble interesante: **GBM sobre features tabulares + red secuencial sobre las
series, blended.** CatBoost además resuelve los categóricos de alta cardinalidad
(`product_code`, `branch`, NAICS) sin one-hot.

### 6.5 Explicabilidad

- **SHAP sí**, con `TreeExplainer` (rápido y exacto en GBM).
- **LIME no** — inestable entre corridas, y con SHAP ya se tiene local + global
  coherentes.
- **En su lugar: explicaciones contrafactuales accionables** ("este cliente baja 18
  puntos de riesgo si recupera direct deposit"), porque eso es lo que el equipo de
  retención puede ejecutar.
- Validar el SHAP contra intuición de negocio antes de presentarlo. Si "número de
  sucursal" sale top-3, hay un proxy de algo.

### 6.6 Qué modelos entran

| Capa | Modelos | Notebook |
|:---|:---|:---|
| Tabulares | `logreg`, `random_forest`, `lightgbm`, `xgboost`, `catboost` (y `mlp`, definido pero desactivado) | §8, se entrenan en §15 |
| Secuencial | GRU de PyTorch sobre `[muestras, meses, canales]` | §8, se entrena en §17 |
| Combinación | Stacking de los top-k por CV, y blend GBM × GRU | §16 y §17 |

Cuáles corren lo decide `models.enabled` en la celda de configuración. Cada uno lleva
su propio espacio de búsqueda y se tunea con `RandomizedSearchCV` sobre
`StratifiedGroupKFold` agrupado por CIF, optimizando PR-AUC.

El desbalance se ataca por dos vías según la familia: `class_weight="balanced"` en
los lineales y de bosque, `scale_pos_weight = (1-p)/p` en los tres GBM. Y el
preprocesamiento es distinto para cada una: los árboles reciben imputación por
mediana y codificación ordinal; los lineales y la red, imputación, escalado, recorte
a ±10σ y one-hot.

Dos detalles que cuestan caro si no se saben de antemano, y que están comentados en
el código:

- Todos los estimadores van con `n_jobs=1`. Anidar paralelismo dentro y fuera del
  `RandomizedSearchCV` produce un deadlock de joblib: el proceso se queda vivo al 0%
  de CPU, sin error.
- El tope de `C` de la logística es deliberadamente bajo. Con clases muy
  desbalanceadas y datos casi separables los coeficientes se disparan (separación
  cuasi-completa) y el `matmul` desborda a infinito.

### 6.7 Dónde se elige cada cosa

Con un holdout out-of-time de por medio es fácil ajustar contra él sin darse cuenta.
La regla del notebook:

| Decisión | Dónde se toma |
|:---|:---|
| Hiperparámetros de cada modelo | CV agrupado por CIF, dentro del train |
| Early stopping del GRU | Partición de validación agrupada, dentro del train |
| Peso del blend GBM × GRU | La misma partición de validación |
| Modelo base del blend | Score de CV, no el test |
| Calibración | Primer tercio del test; se mide en el resto |

El test se toca una sola vez, para reportar.

---

## 7. Riesgo regulatorio

Modelo de banca en EE.UU. Aunque churn no es una decisión de crédito bajo ECOA/Reg B:

- Si el output alimenta ofertas diferenciadas (tasas, waivers de fees), **Fair Lending y
  UDAAP** entran en juego.
- **No incluir** raza, etnia, país de origen, edad ni sexo como features directas.
- Correr un **análisis de proxies** sobre zip y `preferred_language`.
- Va a caer bajo **model risk management (SR 11-7)** para validación y documentación.

Conviene involucrar a Compliance antes de construir, no después.

---

## Próximo paso

El pipeline está construido y corre de punta a punta con datos sintéticos. Lo que
falta es aplicarlo a los datos reales, en este orden:

1. **Manda al agente SQL las Queries 1, 2 y 3.** Con eso ya se puede correr el
   notebook entero: Q4–Q7 son opcionales. Si la bajada va por ODBC/Excel, pídelas en
   el [formato ancho](#si-la-bajada-pasa-por-odbcexcel-formato-ancho).
2. **Corre la EDA (§11) y ajusta dos umbrales** en la celda de configuración:
   `target.low_balance_abs`, leyéndolo de la distribución de saldos en vez de
   inventarlo; y la historia utilizable, mirando el gráfico de aperturas y cierres —
   si hay un pico aislado, casi seguro es una conversión de core y hay que cortar ahí.
3. **Corre la descomposición del target (§12).** Si una sola condición explica casi
   todos los positivos, la definición está desbalanceada y hay que revisarla antes de
   modelar.
4. **Revisa la tasa base por mes.** Si salta, hay un problema de datos, no de modelo.
5. **Antes de presentar:** valida el SHAP contra intuición de negocio y corre el
   análisis de proxies sobre `state` y `preferred_language`
   ([sección 7](#7-riesgo-regulatorio)).

Queda abierto el bloque de preguntas de la
[sección 4](#4-preguntas-concretas-sobre-la-data): las respuestas no bloquean la
primera corrida, pero sí condicionan cuánto hay que fiarse del resultado.
