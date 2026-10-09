# AdaptGuard v1.0 — Adaptive LSTM Outage-Occurrence Prediction (Real Data)

An implementation of the adaptive LSTM outage prediction system specified
in **Chapter 1** (Aim, Objectives i–iii) and **Chapter 3** (Methodology)
of the YEDC thesis — rebuilt on the expanded **real** dataset your
supervisor supplied (104 logged interruption events, January–April 2026),
replacing the earlier synthetic-data version.

## What changed from the first version

The original YEDC log had only 54 events (Jan–Feb 2026), which your
supervisor flagged as too small. The updated workbook contains:

- **104 real interruption events** across 3 feeders, spanning
  **January–April 2026** (double the original data, all real).
- A secondary "daily hours interrupted" table in the sheet — checked
  carefully and **not used**: it's only genuinely populated for January
  and February; the March and April copies are byte-for-byte identical
  to February (a stale copy-paste artifact in the source file), so
  relying on it would have quietly fabricated 2 months of fake data.

Instead, `src/data_loader.py` parses the real multi-sheet event log
directly, and `src/feature_engineer.py` derives a genuine **daily
downtime time series per feeder across all 4 months** from the real
event timestamps (exact Date Out / Time Out / Duration are all present
and clean — zero missing values). Nothing in this version is synthetic.

## The resolution/horizon adaptation (read this before presenting)

Chapter 3 specifies an LSTM operating on **5-minute-resolution SCADA
telemetry** (load, voltage, solar, battery) with a **1-hour prediction
horizon**. YEDC's real records contain no continuous SCADA telemetry —
only discrete interruption events with exact start/end timestamps. The
finest resolution honestly derivable from real data is **daily**:
aggregate each feeder's real events per calendar day, then predict
whether that feeder will have an outage **the following day**, using
the preceding **7 days** as input.

This is a documented, deliberate adaptation — not a shortcut. The LSTM
architecture, training procedure, and adaptive retraining framework are
otherwise implemented **exactly** as Chapter 3 specifies:

- LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) →
  Dense(16, ReLU) → Dense(1, sigmoid)
- Adam optimizer, binary cross-entropy loss, AUC training metric
- Up to 50 epochs, batch 64, early stopping (patience 10, restores best
  validation-loss weights)
- 5-fold time-series cross-validation, chronological (never shuffled)
- Min-Max scaling fit only on the training partition
- SMOTE applied only to each fit's own training slice — never to
  validation or test data
- Adaptive deployment simulation with rolling-F1 drift monitoring and
  retraining (threshold 0.35)

## Results

Full spec run (50 epochs, 5-fold CV) on the real data:

- **Held-out test AUC-ROC: 0.910** — clears NFR1 (≥ 0.70)
- F1: 0.743, Recall: 1.000, Precision: 0.591
- Top predictive features: 14-day and 7-day rolling outage frequency,
  recent downtime minutes — operationally sensible, matching how field
  engineers already reason about "problem feeders"

This is a genuine result on real data, not a synthetic demonstration.
`outputs/AdaptGuard_Report.docx` documents the full methodology,
metrics, and the resolution-adaptation rationale for your supervisor.

## Two bugs found and fixed while building this

1. **IQR outlier clipping was crushing the outage signal to zero.**
   Blanket IQR treatment was originally applied to every numeric
   column, including sparse count features like `event_count` (mostly
   0s). Since Q1 = Q3 = 0 for a column that's ~80% zeros, IQR clipping
   flattened every real outage day to 0 — silently destroying the
   entire training signal. Fixed by scoping IQR treatment to only the
   genuinely continuous columns (`mean_load_loss_mw`,
   `max_load_loss_mw`); count/indicator features are left alone.
2. **SMOTE crashed on small cross-validation folds.** SMOTE's default
   `k_neighbors=5` requires at least 6 minority-class samples; early
   CV folds with real data can have fewer. Fixed by capping
   `k_neighbors` to what the minority class actually has available.

## Project structure

```
lstm-outage-prediction/
├── data/
│   └── yedc_interruption_log.xlsx   # the real, supervisor-supplied workbook
├── src/
│   ├── data_loader.py       # parses the real multi-sheet Excel log
│   ├── preprocessor.py      # IQR (continuous cols only) + Min-Max scaling
│   ├── feature_engineer.py  # daily aggregation, rolling/lag, next-day target
│   ├── sequence_builder.py  # 7-day windows, chronological split, SMOTE
│   ├── lstm_model.py        # LSTMModel (architecture exactly per spec)
│   ├── adaptive_framework.py# 5-fold CV, deployment simulation, retraining
│   └── utils.py             # plots, permutation importance, SHAP
├── tests/                   # 21 unit tests, pytest
├── main.py                  # orchestrates the full pipeline
└── generate_report.py       # writes outputs/AdaptGuard_Report.docx
```

## Running it

```bash
pip install -q pandas numpy scikit-learn matplotlib seaborn joblib \
    tensorflow imbalanced-learn shap python-docx pytest openpyxl

python main.py                # full spec: 50 epochs, 5-fold CV
python generate_report.py     # writes the Word report
```

For a faster demo run (fewer epochs/CV-folds):

```bash
python main.py --quick
```

The dataset is small (336 sequences total), so even the full spec run
completes in a few minutes on CPU — no GPU runtime needed.

## Tests

```bash
python -m pytest tests/ -q
```
