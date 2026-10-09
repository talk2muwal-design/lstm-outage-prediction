"""Generates a full Word (.docx) report for AdaptGuard: the adaptive
LSTM outage-occurrence prediction system built per Chapter 1 (Aim,
Objectives) and Chapter 3 (Methodology), trained on REAL YEDC data.
Reads outputs/results.json and the plots in outputs/, both produced by
main.py -- run main.py first.
"""

import json
import os

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

OUT_DIR = "outputs"
RESULTS_PATH = f"{OUT_DIR}/results.json"
REPORT_PATH = f"{OUT_DIR}/AdaptGuard_Report.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
NOTE = RGBColor(0x2E, 0x5C, 0x2E)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = ACCENT
    return h


def add_note_box(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.color.rgb = NOTE
    return p


def add_metric_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = str(h)
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph()
    return table


def add_image_if_exists(doc, path, caption, width=6.0):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in cap.runs:
            r.italic = True
            r.font.size = Pt(9)
    else:
        doc.add_paragraph(f"[Figure not found: {path}]")


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found -- run `python main.py` first "
            "to generate results and plots."
        )
    with open(RESULTS_PATH) as f:
        r = json.load(f)

    doc = Document()

    title = doc.add_heading(f'{r["model_name"]} — Adaptive LSTM Outage-'
                             'Occurrence Prediction System', level=0)
    for run in title.runs:
        run.font.color.rgb = ACCENT
    sub = doc.add_paragraph(f'Version {r["model_version"]} | Technical Report '
                             f'| Trained on Real YEDC Data')
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.italic = True
    doc.add_paragraph()

    add_note_box(
        doc,
        "DATA NOTE: This model is trained entirely on REAL YEDC feeder "
        "interruption records -- 104 logged events across 3 feeders, "
        f'{r["date_range"][0]} to {r["date_range"][1]} (no synthetic data). '
        "Chapter 3's LSTM design specifies 5-minute-resolution SCADA "
        "telemetry with a 1-hour prediction horizon; no continuous SCADA "
        "readings exist in YEDC's records, so this implementation adapts "
        "the resolution and horizon to what the real event log actually "
        "supports: DAILY resolution, predicting whether a feeder will "
        "have an outage on the FOLLOWING calendar day, using the "
        "preceding 7 days as input. The LSTM architecture, training "
        "procedure, and adaptive retraining framework are otherwise "
        "implemented exactly as Chapter 3 specifies. See Section 8 for "
        "the full rationale."
    )
    doc.add_paragraph()

    add_heading(doc, "1. Executive Summary")
    tm = r["test_metrics"]
    nfr1_text = "met" if r["nfr1_met"] else "not met"
    doc.add_paragraph(
        f'{r["model_name"]} is a stacked two-layer LSTM classifier '
        f'(LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16, ReLU) -> '
        f'Dense(1, sigmoid)) trained to predict, from a feeder\'s '
        f'preceding 7 days of interruption history, whether it will '
        f'experience an outage the following day. On {r["n_events"]} real '
        f'logged YEDC events across {r["n_feeders"]} feeders '
        f'({r["n_feeder_days"]} feeder-days total), the model achieved a '
        f'held-out test AUC-ROC of {tm["AUC-ROC"]:.3f} and an F1-score of '
        f'{tm["F1-Score"]:.3f}, with a Recall of {tm["Recall"]:.3f} -- NFR1 '
        f'(AUC-ROC >= 0.70) was {nfr1_text} on this run.'
    )

    add_heading(doc, "2. Dataset Overview")
    doc.add_paragraph(
        f'The dataset consists of {r["n_events"]} real feeder interruption '
        f'events logged by YEDC between {r["date_range"][0]} and '
        f'{r["date_range"][1]}, more than double the 54-event log used '
        f'in the initial phase of this project. Events were aggregated '
        f'into a daily panel: {r["n_feeder_days"]} feeder-days across '
        f'{r["n_feeders"]} feeders, with zero-filled features on days with '
        f'no logged event. A positive target label is assigned to a '
        f'feeder-day if that feeder logs an outage on the following '
        f'calendar day. The positive rate is '
        f'{r["target_positive_rate"]*100:.1f}%.'
    )
    add_metric_table(
        doc, ["Metric", "Value"],
        [
            ["Real interruption events", r["n_events"]],
            ["Feeders", r["n_feeders"]],
            ["Date range", f'{r["date_range"][0]} to {r["date_range"][1]}'],
            ["Feeder-days (daily panel)", r["n_feeder_days"]],
            ["Sequence window length", f'{r["window"]} days (1 week)'],
            ["Total sequences", r["n_sequences_total"]],
            ["Train sequences (chronological, pre-SMOTE)", r["n_sequences_train"]],
            ["Test sequences (chronological, held out)", r["n_sequences_test"]],
            ["Training sequences after SMOTE (fit slice only)",
             r["n_sequences_train_after_smote"]],
            ["Target positive rate (outage next day)",
             f'{r["target_positive_rate"]*100:.1f}%'],
        ],
    )
    add_image_if_exists(
        doc, f"{OUT_DIR}/01_target_distribution.png",
        "Figure 1. Distribution of the next-day outage target."
    )

    add_heading(doc, "3. Feature Importance")
    doc.add_paragraph(
        "Mutual information scores rank each feature by how much it "
        "reduces uncertainty about the next-day outage target, "
        "independent of the model used (Chapter 3, Section 3.3.4)."
    )
    mi_rows = [[f, f"{s:.4f}"] for f, s in r["mutual_information_top10"]]
    add_metric_table(doc, ["Feature", "Mutual Information"], mi_rows)

    doc.add_paragraph(
        "Permutation importance (shuffle each feature, measure the AUC "
        "drop) was used as a model-agnostic substitute for the "
        "feature_importances_ that tree-based models provide but LSTMs "
        "do not."
    )
    add_image_if_exists(
        doc, f"{OUT_DIR}/04_feature_importance.png",
        "Figure 2. Permutation feature importance (LSTM)."
    )
    add_image_if_exists(
        doc, f"{OUT_DIR}/05_shap_summary.png",
        "Figure 3. SHAP summary plot (feature impact on predictions)."
    )

    add_heading(doc, "4. Model Training")
    doc.add_paragraph(
        "Trained with the Adam optimizer, binary cross-entropy loss, and "
        "AUC as the primary training metric, up to 50 epochs with early "
        "stopping (patience 10, monitoring validation loss, restoring the "
        "best-validation-loss weights). SMOTE was applied only to the "
        "training slice used for gradient updates -- never to the "
        "validation slice used for early stopping and threshold "
        "selection, or to the held-out test set -- so every evaluation in "
        "this report reflects the real (imbalanced) class distribution, "
        "not an artificially balanced one."
    )
    add_image_if_exists(
        doc, f"{OUT_DIR}/02_training_history.png",
        "Figure 4. Training/validation loss and AUC across epochs."
    )

    add_heading(doc, "5. 5-Fold Time-Series Cross-Validation")
    doc.add_paragraph(
        f'Per Chapter 3, Section 3.4.4, performance was additionally '
        f'assessed via time-series cross-validation (chronological folds, '
        f'never shuffled) on the training pool. Mean CV AUC-ROC: '
        f'{r["cv_auc_mean"]:.3f}. Fold-to-fold variance is expected given '
        f'how few sequences fall in the earliest folds.'
    )
    cv_rows = [[m["fold"], f'{m["AUC-ROC"]:.3f}', f'{m["F1-Score"]:.3f}',
                f'{m["Precision"]:.3f}', f'{m["Recall"]:.3f}']
               for m in r["cv_fold_metrics"]]
    add_metric_table(doc, ["Fold", "AUC-ROC", "F1", "Precision", "Recall"], cv_rows)

    add_heading(doc, "6. Held-Out Test Set Performance")
    add_metric_table(
        doc, ["Metric", "Value"],
        [[k, f"{v:.3f}"] for k, v in tm.items()]
    )
    doc.add_paragraph(
        f'NFR1 (AUC-ROC >= 0.70): {"MET" if r["nfr1_met"] else "NOT MET"} '
        f'on this run.'
    )
    add_image_if_exists(
        doc, f"{OUT_DIR}/03_confusion_roc.png",
        "Figure 5. Confusion matrix and ROC curve on the held-out test set."
    )

    add_heading(doc, "7. Adaptive Deployment Simulation")
    doc.add_paragraph(
        f'Per the Activity and Sequence Diagrams (3.2.3.3, 3.2.3.4), the '
        f'held-out test set was processed as sequential deployment '
        f'batches. A retrain is triggered whenever the rolling F1-score '
        f'for a batch falls below {r["drift_threshold"]}. Retraining was '
        f'triggered on batch(es): {r["retrain_batches"] or "none"}.'
    )
    add_image_if_exists(
        doc, f"{OUT_DIR}/06_rolling_f1.png",
        "Figure 6. Rolling F1-score per deployment batch and retrain triggers."
    )

    add_heading(doc, "8. Why Daily Resolution Instead of 5-Minute SCADA Telemetry")
    doc.add_paragraph(
        "Chapter 3 specifies an LSTM operating on 5-minute-resolution "
        "SCADA readings (load, voltage, solar, battery) with a 1-hour "
        "prediction horizon. YEDC's actual records contain no continuous "
        "SCADA telemetry -- only discrete interruption events with exact "
        "start/end timestamps. The finest resolution honestly derivable "
        "from real data is therefore daily: aggregating each feeder's "
        "real events per calendar day and predicting the next day's "
        "outage occurrence. This keeps every number in this report "
        "grounded in real, verifiable YEDC records, at the cost of a "
        "coarser resolution and horizon than Chapter 3's literal wording. "
        "The architecture, training procedure, and adaptive framework "
        "(5-fold time-series CV, SMOTE, early stopping, rolling-F1 "
        "retraining) are otherwise implemented exactly as specified."
    )
    doc.add_paragraph(
        "This is a meaningfully different situation from the project's "
        "earlier phase, where a synthetic dataset had to stand in for "
        "SCADA telemetry entirely. Here, every feature and every label is "
        "computed directly from real, supervisor-supplied records -- the "
        "resolution is coarser than the original spec, but nothing in the "
        "pipeline is fabricated."
    )

    add_heading(doc, "9. Conclusion")
    doc.add_paragraph(
        f'{r["model_name"]} demonstrates that the Chapter 3 architecture '
        f'and adaptive framework, applied to real YEDC interruption '
        f'records at daily resolution, achieves an AUC-ROC of '
        f'{tm["AUC-ROC"]:.3f} on a genuinely held-out (chronologically '
        f'future) test set -- '
        f'{"clearing" if r["nfr1_met"] else "falling short of"} '
        f'the NFR1 target of 0.70. The top predictive features (recent '
        f'14-day and 7-day outage frequency, recent downtime minutes) are '
        f'operationally sensible: feeders with a recent history of '
        f'frequent or lengthy interruptions are more likely to fail again '
        f'soon, consistent with how field engineers already reason about '
        f'"problem feeders." Expanding the historical window beyond four '
        f'months, and adding real weather or loading data if YEDC can '
        f'supply it, would likely further stabilize the cross-validation '
        f'variance seen in Section 5.'
    )

    doc.save(REPORT_PATH)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
