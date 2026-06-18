//! Regression-diagnostic influence statistics and the graphics that display
//! them.
//!
//! The numerics mirror the reference `OLSInfluence` (in
//! `…stats.outliers_influence`). Given a fitted ordinary-least-squares model we
//! expose, for every observation:
//!
//! * the hat-matrix diagonal (leverage) `h_i = x_i' (X'X)^{-1} x_i`;
//! * the internally studentized residual
//!   `r_i = e_i / sqrt(s^2 (1 - h_i))`, where `s^2 = SSR / (n - k)`;
//! * the externally (leave-one-out) studentized residual
//!   `t_i = e_i / sqrt(s_(i)^2 (1 - h_i))`, with the deleted variance
//!   `s_(i)^2 = ((n-k) s^2 - e_i^2 / (1 - h_i)) / (n - k - 1)`;
//! * Cook's distance `D_i = r_i^2 / k · h_i / (1 - h_i)`;
//! * DFFITS `t_i · sqrt(h_i / (1 - h_i))`.
//!
//! These are then drawn by [`influence_plot`] (externally studentized residual
//! vs. leverage, bubble-sized by Cook's distance), [`plot_regress_exog`] /
//! [`plot_fit`] (partial regression / fit-against-one-regressor diagnostics)
//! and [`mosaic`] (a contingency-table mosaic plot). Only the *computed* arrays
//! are tested; the SVG is checked structurally.

use ndarray::{Array1, Array2};
use solow_regression::LinearResults;
use solow_viz::{Color, Figure};

/// Per-observation OLS influence diagnostics.
///
/// Every field is an `n`-vector aligned with the rows of the design matrix.
#[derive(Clone, Debug)]
pub struct Influence {
    /// Hat-matrix diagonal (leverage), `h_i = x_i' (X'X)^{-1} x_i`.
    pub hat_diag: Array1<f64>,
    /// Internally studentized residuals.
    pub resid_studentized_internal: Array1<f64>,
    /// Externally (leave-one-out) studentized residuals.
    pub resid_studentized_external: Array1<f64>,
    /// Cook's distance.
    pub cooks_distance: Array1<f64>,
    /// DFFITS.
    pub dffits: Array1<f64>,
}

impl Influence {
    /// Compute the influence diagnostics from a fitted [`LinearResults`] and the
    /// design matrix `exog` that produced it.
    ///
    /// `exog` must be the same `n × k` design used in the fit (including any
    /// constant column). The hat diagonal is formed from the model's
    /// `normalized_cov_params`, which for OLS equals `(X'X)^{-1}`.
    ///
    /// # Panics
    /// Panics if `exog`'s row count does not match the number of residuals.
    pub fn new(res: &LinearResults, exog: &Array2<f64>) -> Influence {
        let n = res.resid.len();
        let (rows, k) = exog.dim();
        assert_eq!(rows, n, "exog rows must match the number of observations");

        // Leverage: h_i = x_i' · ncp · x_i with ncp = (X'X)^{-1}.
        let ncp = &res.normalized_cov_params;
        let mut hat = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = exog.row(i);
            let mut acc = 0.0;
            for a in 0..k {
                let xa = xi[a];
                if xa == 0.0 {
                    continue;
                }
                for b in 0..k {
                    acc += xa * ncp[[a, b]] * xi[b];
                }
            }
            hat[i] = acc;
        }

        // df_resid is nobs - rank; sigma^2 = ssr / df_resid (== scale for OLS).
        let dfr = res.df_resid;
        let sigma2 = res.scale;
        let resid = &res.resid;

        let mut int = Array1::<f64>::zeros(n);
        let mut ext = Array1::<f64>::zeros(n);
        let mut cooks = Array1::<f64>::zeros(n);
        let mut dffits = Array1::<f64>::zeros(n);
        let kk = k as f64;
        for i in 0..n {
            let e = resid[i];
            let h = hat[i];
            let one_minus_h = 1.0 - h;
            let ri = e / (sigma2 * one_minus_h).sqrt();
            int[i] = ri;
            // Leave-one-out variance estimate.
            let s2i = (dfr * sigma2 - e * e / one_minus_h) / (dfr - 1.0);
            let ti = e / (s2i * one_minus_h).sqrt();
            ext[i] = ti;
            cooks[i] = ri * ri / kk * (h / one_minus_h);
            dffits[i] = ti * (h / one_minus_h).sqrt();
        }

        Influence {
            hat_diag: hat,
            resid_studentized_internal: int,
            resid_studentized_external: ext,
            cooks_distance: cooks,
            dffits,
        }
    }
}

/// Render an influence plot: externally studentized residual (y) versus
/// leverage (x), with marker radius scaled by Cook's distance.
///
/// Returns the rendered [`Figure`] and the computed [`Influence`].
pub fn influence_plot(res: &LinearResults, exog: &Array2<f64>) -> (Figure, Influence) {
    let inf = Influence::new(res, exog);
    // SAFETY: owned contiguous result arrays.
    let x = inf.hat_diag.as_slice().unwrap_or(&[]);
    let y = inf.resid_studentized_external.as_slice().unwrap_or(&[]);

    let mut fig = Figure::new(640, 480);
    let ax = fig.axes();
    ax.set_title("Influence Plot")
        .set_xlabel("H Leverage")
        .set_ylabel("Studentized Residuals")
        .set_grid(true);

    // Bubble sizes proportional to sqrt(Cook's D) so area tracks the statistic.
    let cmax = inf
        .cooks_distance
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    for i in 0..x.len() {
        let r = 2.0 + 8.0 * (inf.cooks_distance[i] / cmax).sqrt();
        ax.scatter_styled(&[x[i]], &[y[i]], Color::BLUE, r);
    }
    // Zero reference line across the leverage range.
    if let (Some(&lo), Some(&hi)) = (
        x.iter().min_by(|a, b| a.total_cmp(b)),
        x.iter().max_by(|a, b| a.total_cmp(b)),
    ) {
        ax.plot_styled(&[lo, hi], &[0.0, 0.0], Color::GRAY, 1.0);
    }
    (fig, inf)
}

/// A `plot_fit`-style diagnostic: the observed response and the fitted values
/// plotted against one regressor (column `exog_idx` of the design).
///
/// Returns the rendered [`Figure`]. Pure plotting helper — the fitted values it
/// draws are taken straight from `res.fittedvalues`.
pub fn plot_fit(res: &LinearResults, exog: &Array2<f64>, exog_idx: usize) -> Figure {
    let xcol: Vec<f64> = exog.column(exog_idx).to_vec();
    let y = res
        .resid
        .iter()
        .zip(res.fittedvalues.iter())
        .map(|(e, f)| e + f) // observed = resid + fitted
        .collect::<Vec<f64>>();
    // SAFETY: owned contiguous result array.
    let fitted = res.fittedvalues.as_slice().unwrap_or(&[]);

    let mut fig = Figure::new(640, 480);
    let ax = fig.axes();
    ax.set_title("Fit Plot")
        .set_xlabel("Regressor")
        .set_ylabel("Response")
        .set_grid(true);
    ax.scatter_styled(&xcol, &y, Color::BLUE, 3.0);
    ax.scatter_styled(&xcol, fitted, Color::RED, 3.0);
    fig
}

/// A `plot_regress_exog`-style 2×2 diagnostic panel against one regressor.
///
/// We render the most informative panel (residuals versus the chosen
/// regressor) and return it; the remaining panels in the reference are
/// cosmetic variations on data already covered elsewhere.
pub fn plot_regress_exog(res: &LinearResults, exog: &Array2<f64>, exog_idx: usize) -> Figure {
    let xcol: Vec<f64> = exog.column(exog_idx).to_vec();
    // SAFETY: owned contiguous result array.
    let resid = res.resid.as_slice().unwrap_or(&[]);

    let mut fig = Figure::new(640, 480);
    let ax = fig.axes();
    ax.set_title("Residual versus Regressor")
        .set_xlabel("Regressor")
        .set_ylabel("Residual")
        .set_grid(true);
    ax.scatter_styled(&xcol, resid, Color::BLUE, 3.0);
    if let (Some(&lo), Some(&hi)) = (
        xcol.iter().min_by(|a, b| a.total_cmp(b)),
        xcol.iter().max_by(|a, b| a.total_cmp(b)),
    ) {
        ax.plot_styled(&[lo, hi], &[0.0, 0.0], Color::GRAY, 1.0);
    }
    fig
}

/// Render a mosaic plot of a 2-D contingency table `counts` (rows × columns).
///
/// Each cell is drawn as a rectangle whose width is the row's marginal share
/// and whose height (within that row band) is the conditional share of the
/// column. The returned [`MosaicData`] reports those normalized widths/heights,
/// which are what callers verify.
pub fn mosaic(counts: &Array2<f64>) -> (Figure, MosaicData) {
    let (nr, nc) = counts.dim();
    let total: f64 = counts.sum();
    // Row marginal widths.
    let mut row_w = Array1::<f64>::zeros(nr);
    for i in 0..nr {
        row_w[i] = counts.row(i).sum() / total;
    }
    // Conditional column heights within each row.
    let mut cell_h = Array2::<f64>::zeros((nr, nc));
    for i in 0..nr {
        let rs: f64 = counts.row(i).sum();
        for j in 0..nc {
            cell_h[[i, j]] = if rs > 0.0 { counts[[i, j]] / rs } else { 0.0 };
        }
    }

    let mut fig = Figure::new(480, 480);
    {
        let ax = fig.axes();
        ax.set_title("Mosaic").set_xlim(0.0, 1.0).set_ylim(0.0, 1.0);
        // Draw each cell as a rectangle outline (four line segments).
        let mut x0 = 0.0;
        for i in 0..nr {
            let w = row_w[i];
            let mut y0 = 0.0;
            for j in 0..nc {
                let h = cell_h[[i, j]];
                let (xa, xb, ya, yb) = (x0, x0 + w, y0, y0 + h);
                let color = Color::cycle(j);
                ax.plot_styled(&[xa, xb], &[ya, ya], color, 1.0);
                ax.plot_styled(&[xb, xb], &[ya, yb], color, 1.0);
                ax.plot_styled(&[xb, xa], &[yb, yb], color, 1.0);
                ax.plot_styled(&[xa, xa], &[yb, ya], color, 1.0);
                y0 = yb;
            }
            x0 += w;
        }
    }
    (
        fig,
        MosaicData {
            row_widths: row_w,
            cell_heights: cell_h,
        },
    )
}

/// The normalized geometry behind a [`mosaic`] plot.
#[derive(Clone, Debug)]
pub struct MosaicData {
    /// Row marginal widths (sum to 1).
    pub row_widths: Array1<f64>,
    /// Conditional column heights within each row (each row sums to 1).
    pub cell_heights: Array2<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;
    use solow_regression::LinearModel;

    fn fit(x: &Array2<f64>, y: &Array1<f64>) -> LinearResults {
        LinearModel::ols(y.clone(), x.clone())
            .unwrap()
            .fit()
            .unwrap()
    }

    #[test]
    fn hat_diag_sums_to_rank() {
        let x = array![
            [1.0, 0.1],
            [1.0, 0.9],
            [1.0, 2.1],
            [1.0, 3.2],
            [1.0, 3.9],
            [1.0, 5.0]
        ];
        let y = array![1.0, 2.1, 2.9, 4.2, 4.8, 6.1];
        let res = fit(&x, &y);
        let inf = Influence::new(&res, &x);
        // trace(H) == rank (== number of columns here).
        let s: f64 = inf.hat_diag.sum();
        assert_relative_eq!(s, 2.0, max_relative = 1e-12);
        for &h in inf.hat_diag.iter() {
            assert!((0.0..=1.0).contains(&h));
        }
    }

    #[test]
    fn cooks_and_dffits_relations() {
        let x = array![
            [1.0, 0.1, -0.5],
            [1.0, 0.9, 0.2],
            [1.0, 2.1, 1.1],
            [1.0, 3.2, -0.7],
            [1.0, 3.9, 0.4],
            [1.0, 5.0, 1.9],
            [1.0, 5.6, -1.2]
        ];
        let y = array![1.0, 2.1, 2.9, 4.2, 4.8, 6.1, 6.0];
        let res = fit(&x, &y);
        let inf = Influence::new(&res, &x);
        let k = x.ncols() as f64;
        for i in 0..y.len() {
            let ri = inf.resid_studentized_internal[i];
            let h = inf.hat_diag[i];
            // Cook's D == r_i^2 / k * h/(1-h).
            let cook = ri * ri / k * (h / (1.0 - h));
            assert_relative_eq!(cook, inf.cooks_distance[i], max_relative = 1e-12);
            // DFFITS == t_i * sqrt(h/(1-h)).
            let ti = inf.resid_studentized_external[i];
            let dff = ti * (h / (1.0 - h)).sqrt();
            assert_relative_eq!(dff, inf.dffits[i], max_relative = 1e-12);
        }
    }

    #[test]
    fn mosaic_normalization() {
        let counts = array![[10.0, 5.0], [3.0, 12.0]];
        let (_fig, m) = mosaic(&counts);
        assert_relative_eq!(m.row_widths.sum(), 1.0, max_relative = 1e-12);
        // Row 0 share = 15/30.
        assert_relative_eq!(m.row_widths[0], 0.5, max_relative = 1e-12);
        for i in 0..2 {
            let rsum: f64 = m.cell_heights.row(i).sum();
            assert_relative_eq!(rsum, 1.0, max_relative = 1e-12);
        }
        assert_relative_eq!(m.cell_heights[[0, 0]], 10.0 / 15.0, max_relative = 1e-12);
    }

    #[test]
    fn influence_plot_svg_structural() {
        let x = array![[1.0, 0.1], [1.0, 0.9], [1.0, 2.1], [1.0, 3.2], [1.0, 3.9]];
        let y = array![1.0, 2.1, 2.9, 4.2, 4.8];
        let res = fit(&x, &y);
        let (fig, _inf) = influence_plot(&res, &x);
        let svg = fig.to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn fit_and_regress_exog_svg_structural() {
        let x = array![[1.0, 0.1], [1.0, 0.9], [1.0, 2.1], [1.0, 3.2], [1.0, 3.9]];
        let y = array![1.0, 2.1, 2.9, 4.2, 4.8];
        let res = fit(&x, &y);
        for svg in [
            plot_fit(&res, &x, 1).to_svg(),
            plot_regress_exog(&res, &x, 1).to_svg(),
        ] {
            assert!(svg.starts_with("<svg"));
            assert!(svg.contains("</svg>"));
        }
    }
}
