//! `classification_report` — the reference text summary of per-class
//! precision, recall, F1, and support.

use solow_core::{Error, Result};

use crate::classification::{precision_recall_fscore, Average};

/// Per-class row in the classification report.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationRow {
    /// Class label (as an `i64`) or the aggregate row name.
    pub name: String,
    /// Precision `TP / (TP + FP)`.
    pub precision: f64,
    /// Recall `TP / (TP + FN)`.
    pub recall: f64,
    /// F1 = 2·P·R / (P + R).
    pub f1: f64,
    /// Number of samples belonging to this class.
    pub support: usize,
}

/// A full classification report.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationReport {
    /// Per-class rows in `classes` order.
    pub rows: Vec<ClassificationRow>,
    /// Accuracy summary row.
    pub accuracy: f64,
    /// Macro-average precision / recall / F1.
    pub macro_avg: ClassificationRow,
    /// Weighted-average precision / recall / F1.
    pub weighted_avg: ClassificationRow,
}

impl ClassificationReport {
    /// Text summary matching the reference `classification_report(digits=4)`.
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:>15} {:>10} {:>10} {:>10} {:>10}\n",
            "", "precision", "recall", "f1-score", "support"
        ));
        for row in &self.rows {
            out.push_str(&format!(
                "{:>15} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                row.name, row.precision, row.recall, row.f1, row.support
            ));
        }
        out.push('\n');
        out.push_str(&format!(
            "{:>15} {:>10} {:>10} {:>10.4} {:>10}\n",
            "accuracy",
            "",
            "",
            self.accuracy,
            self.rows.iter().map(|r| r.support).sum::<usize>()
        ));
        let total = self.rows.iter().map(|r| r.support).sum::<usize>();
        out.push_str(&format!(
            "{:>15} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "macro avg",
            self.macro_avg.precision,
            self.macro_avg.recall,
            self.macro_avg.f1,
            total
        ));
        out.push_str(&format!(
            "{:>15} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "weighted avg",
            self.weighted_avg.precision,
            self.weighted_avg.recall,
            self.weighted_avg.f1,
            total
        ));
        out
    }
}

/// Compute a classification report.
///
/// `labels` optionally selects which classes to include; if omitted, the
/// sorted union of true/pred labels is used.
pub fn classification_report(
    y_true: &[usize],
    y_pred: &[usize],
    labels: Option<&[usize]>,
) -> Result<ClassificationReport> {
    let n = y_true.len();
    if y_pred.len() != n {
        return Err(Error::Shape("classification_report: length mismatch".into()));
    }
    if n == 0 {
        return Err(Error::Value("classification_report: empty inputs".into()));
    }
    let classes: Vec<usize> = if let Some(l) = labels {
        l.to_vec()
    } else {
        let mut all = y_true.to_vec();
        all.extend(y_pred);
        all.sort();
        all.dedup();
        all
    };
    // Compute per-class metrics via existing precision_recall_fscore.
    let mut rows = Vec::with_capacity(classes.len());
    let per_class = per_class_prf(y_true, y_pred, &classes)?;
    let mut sum_precision = 0.0_f64;
    let mut sum_recall = 0.0_f64;
    let mut sum_f1 = 0.0_f64;
    let mut w_precision = 0.0_f64;
    let mut w_recall = 0.0_f64;
    let mut w_f1 = 0.0_f64;
    let total_support: usize = per_class.iter().map(|r| r.support).sum();
    for row in &per_class {
        sum_precision += row.precision;
        sum_recall += row.recall;
        sum_f1 += row.f1;
        w_precision += row.precision * row.support as f64;
        w_recall += row.recall * row.support as f64;
        w_f1 += row.f1 * row.support as f64;
        rows.push(row.clone());
    }
    let n_classes = per_class.len() as f64;
    let macro_avg = ClassificationRow {
        name: "macro avg".into(),
        precision: sum_precision / n_classes.max(1.0),
        recall: sum_recall / n_classes.max(1.0),
        f1: sum_f1 / n_classes.max(1.0),
        support: total_support,
    };
    let weighted_avg = ClassificationRow {
        name: "weighted avg".into(),
        precision: w_precision / total_support.max(1) as f64,
        recall: w_recall / total_support.max(1) as f64,
        f1: w_f1 / total_support.max(1) as f64,
        support: total_support,
    };
    let accuracy = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t == p)
        .count() as f64
        / n as f64;
    Ok(ClassificationReport {
        rows,
        accuracy,
        macro_avg,
        weighted_avg,
    })
}

fn per_class_prf(
    y_true: &[usize],
    y_pred: &[usize],
    classes: &[usize],
) -> Result<Vec<ClassificationRow>> {
    let prf = precision_recall_fscore(y_true, y_pred, Average::Macro, 1.0, None)?;
    let mut rows = Vec::with_capacity(classes.len());
    for (i, &c) in classes.iter().enumerate() {
        rows.push(ClassificationRow {
            name: c.to_string(),
            precision: prf.precision.get(i).copied().unwrap_or(0.0),
            recall: prf.recall.get(i).copied().unwrap_or(0.0),
            f1: prf.fbeta.get(i).copied().unwrap_or(0.0),
            support: prf.support.get(i).copied().unwrap_or(0),
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_report_returns_all_classes() {
        let y_true = vec![0_usize, 0, 1, 1, 2, 2];
        let y_pred = vec![0_usize, 0, 1, 2, 2, 2];
        let r = classification_report(&y_true, &y_pred, None).unwrap();
        assert_eq!(r.rows.len(), 3);
        assert!((r.accuracy - 5.0 / 6.0).abs() < 1e-12);
        // Every reported metric is finite.
        assert!(r.macro_avg.f1.is_finite());
        assert!(r.weighted_avg.f1.is_finite());
    }

    #[test]
    fn classification_report_renders_as_a_string() {
        let y_true = vec![0_usize, 1, 1, 2];
        let y_pred = vec![0_usize, 1, 0, 2];
        let r = classification_report(&y_true, &y_pred, None).unwrap();
        let s = r.to_string();
        assert!(s.contains("precision"));
        assert!(s.contains("accuracy"));
        assert!(s.contains("weighted avg"));
    }
}
