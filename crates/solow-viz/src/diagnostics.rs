//! Ready-made diagnostic charts for common statistical evaluations.
//!
//! These helpers each take an existing [`crate::Axes`] and paint a
//! recognisable, publication-quality figure on it: an ROC curve with the
//! chance-diagonal reference and an AUC annotation, a precision-recall
//! curve with the no-skill baseline and an AP annotation, a reliability
//! diagram for a probabilistic classifier's calibration, and a residual-
//! versus-fitted scatter for a regression fit.
//!
//! The helpers deliberately take primitive `&[f64]` slices so this crate
//! does not depend on [`solow-metrics`](https://docs.rs/solow-metrics) —
//! the pattern is: compute the diagnostic vectors there, then hand them
//! to a helper here.

use crate::{Axes, Color, LegendLoc, LineStyle, Marker};

const CHANCE_LINE: Color = Color(160, 160, 160);

/// Draw an ROC curve on `ax`: `fpr` vs `tpr` with the `y = x` chance
/// reference and axis labels/limits set. `label` (typically `"AUC = 0.87"`)
/// is added to the legend if not empty.
pub fn plot_roc(ax: &mut Axes, fpr: &[f64], tpr: &[f64], label: &str) {
    assert_eq!(
        fpr.len(),
        tpr.len(),
        "plot_roc: fpr and tpr must have the same length"
    );
    ax.line(
        &[0.0, 1.0],
        &[0.0, 1.0],
        CHANCE_LINE,
        1.0,
        LineStyle::Dashed,
        Marker::None,
        0.0,
        None,
    );
    let legend_label = if label.is_empty() { None } else { Some(label) };
    ax.line(
        fpr,
        tpr,
        Color::BLUE,
        2.0,
        LineStyle::Solid,
        Marker::None,
        0.0,
        legend_label,
    );
    ax.set_xlim(0.0, 1.0)
        .set_ylim(0.0, 1.0)
        .set_xlabel("False positive rate")
        .set_ylabel("True positive rate")
        .set_title("ROC curve");
    if legend_label.is_some() {
        ax.legend(LegendLoc::LowerRight);
    }
}

/// Draw a precision-recall curve on `ax`: `recall` vs `precision`. The
/// no-skill baseline `y = positive_prevalence` is drawn as a dashed
/// reference. `label` (typically `"AP = 0.75"`) is added to the legend
/// if not empty.
pub fn plot_precision_recall(
    ax: &mut Axes,
    recall: &[f64],
    precision: &[f64],
    positive_prevalence: f64,
    label: &str,
) {
    assert_eq!(
        recall.len(),
        precision.len(),
        "plot_precision_recall: recall and precision must have the same length"
    );
    ax.line(
        &[0.0, 1.0],
        &[positive_prevalence, positive_prevalence],
        CHANCE_LINE,
        1.0,
        LineStyle::Dashed,
        Marker::None,
        0.0,
        None,
    );
    let legend_label = if label.is_empty() { None } else { Some(label) };
    ax.line(
        recall,
        precision,
        Color::BLUE,
        2.0,
        LineStyle::Solid,
        Marker::None,
        0.0,
        legend_label,
    );
    ax.set_xlim(0.0, 1.0)
        .set_ylim(0.0, 1.0)
        .set_xlabel("Recall")
        .set_ylabel("Precision")
        .set_title("Precision-recall curve");
    if legend_label.is_some() {
        ax.legend(LegendLoc::LowerLeft);
    }
}

/// Draw a reliability diagram on `ax`: the bin-mean predicted probability
/// against the bin-mean empirical positive rate, with a `y = x` reference
/// for perfect calibration. Pass an empty `label` to omit the legend
/// entry (e.g. when the caller is stacking multiple models on one axes).
pub fn plot_reliability_diagram(
    ax: &mut Axes,
    mean_predicted: &[f64],
    mean_actual: &[f64],
    label: &str,
) {
    assert_eq!(
        mean_predicted.len(),
        mean_actual.len(),
        "plot_reliability_diagram: mean_predicted and mean_actual must have the same length"
    );
    ax.line(
        &[0.0, 1.0],
        &[0.0, 1.0],
        CHANCE_LINE,
        1.0,
        LineStyle::Dashed,
        Marker::None,
        0.0,
        None,
    );
    let legend_label = if label.is_empty() { None } else { Some(label) };
    ax.line(
        mean_predicted,
        mean_actual,
        Color::BLUE,
        2.0,
        LineStyle::Solid,
        Marker::Circle,
        4.0,
        legend_label,
    );
    ax.set_xlim(0.0, 1.0)
        .set_ylim(0.0, 1.0)
        .set_xlabel("Mean predicted probability")
        .set_ylabel("Empirical positive rate")
        .set_title("Reliability diagram");
    if legend_label.is_some() {
        ax.legend(LegendLoc::UpperLeft);
    }
}

/// Draw a residuals-vs-fitted scatter with the horizontal zero-reference
/// line — the standard first regression-diagnostic plot.
pub fn plot_residuals_vs_fitted(ax: &mut Axes, fitted: &[f64], residuals: &[f64], label: &str) {
    assert_eq!(
        fitted.len(),
        residuals.len(),
        "plot_residuals_vs_fitted: fitted and residuals must have the same length"
    );
    if !fitted.is_empty() {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        for &v in fitted {
            if v < xmin {
                xmin = v;
            }
            if v > xmax {
                xmax = v;
            }
        }
        ax.line(
            &[xmin, xmax],
            &[0.0, 0.0],
            CHANCE_LINE,
            1.0,
            LineStyle::Dashed,
            Marker::None,
            0.0,
            None,
        );
    }
    let legend_label = if label.is_empty() { None } else { Some(label) };
    ax.line(
        fitted,
        residuals,
        Color::BLUE,
        0.0,
        LineStyle::Solid,
        Marker::Circle,
        3.0,
        legend_label,
    );
    ax.set_xlabel("Fitted values")
        .set_ylabel("Residuals")
        .set_title("Residuals vs fitted");
    if legend_label.is_some() {
        ax.legend(LegendLoc::UpperRight);
    }
}
