//! `arma_order_select_ic`: information-criterion grid search over ARMA orders.
//!
//! For every `(p, q)` with `p <= max_ar` and `q <= max_ma` an ARMA(p, q) model
//! with a constant is estimated by exact Gaussian maximum likelihood (the same
//! Kalman-filter / state-space likelihood used by the reference's modern
//! `ARIMA(order=(p, 0, q), trend="c")` estimator), and the requested
//! information criteria are tabulated. The argmin order over each grid is also
//! reported.
//!
//! The state-space / linear-algebra kernels below use explicit index loops that
//! mirror the underlying matrix recursions, so a couple of index-based clippy
//! lints are silenced module-wide.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_optimize::minimize_bfgs;

/// Information criterion to tabulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoCriterion {
    /// Akaike information criterion.
    Aic,
    /// Bayesian (Schwarz) information criterion.
    Bic,
}

impl InfoCriterion {
    /// Parse a criterion code (`"aic"` or `"bic"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "aic" => Ok(InfoCriterion::Aic),
            "bic" => Ok(InfoCriterion::Bic),
            other => Err(Error::Value(format!("unknown ic '{other}'"))),
        }
    }
}

/// Result of [`arma_order_select_ic`].
#[derive(Debug, Clone)]
pub struct ArmaOrderSelectResult {
    /// The information criterion whose grid is held here.
    pub ic: InfoCriterion,
    /// `(max_ar + 1) × (max_ma + 1)` grid indexed `[p, q]`.
    pub grid: Array2<f64>,
    /// The `(p, q)` order minimising the grid.
    pub min_order: (usize, usize),
}

/// Compute the IC grid(s) for ARMA orders up to `(max_ar, max_ma)`.
///
/// Returns one [`ArmaOrderSelectResult`] per requested criterion.
pub fn arma_order_select_ic(
    y: &Array1<f64>,
    max_ar: usize,
    max_ma: usize,
    ics: &[InfoCriterion],
) -> Result<Vec<ArmaOrderSelectResult>> {
    let nobs = y.len();
    if nobs < 4 {
        return Err(Error::Value("series too short".into()));
    }
    // results[p][q] = (aic, bic).
    let mut aic = Array2::<f64>::from_elem((max_ar + 1, max_ma + 1), f64::NAN);
    let mut bic = Array2::<f64>::from_elem((max_ar + 1, max_ma + 1), f64::NAN);
    for p in 0..=max_ar {
        for q in 0..=max_ma {
            if let Ok(fit) = fit_arma(y, p, q) {
                let k = (p + q + 2) as f64; // ar + ma + const + sigma2
                let n = nobs as f64;
                aic[[p, q]] = -2.0 * fit.llf + 2.0 * k;
                bic[[p, q]] = -2.0 * fit.llf + n.ln() * k;
            }
        }
    }

    let mut out = Vec::with_capacity(ics.len());
    for &ic in ics {
        let grid = match ic {
            InfoCriterion::Aic => aic.clone(),
            InfoCriterion::Bic => bic.clone(),
        };
        let min_order = argmin_grid(&grid);
        out.push(ArmaOrderSelectResult {
            ic,
            grid,
            min_order,
        });
    }
    Ok(out)
}

/// Reproduce the reference argmin: flatten the absolute difference from the
/// minimum and take the first index (row-major), mapping back to `(p, q)`.
fn argmin_grid(grid: &Array2<f64>) -> (usize, usize) {
    let (nr, nc) = grid.dim();
    let mut min_val = f64::INFINITY;
    for v in grid.iter() {
        if v.is_finite() && *v < min_val {
            min_val = *v;
        }
    }
    let mut best = (0usize, 0usize);
    let mut best_delta = f64::INFINITY;
    for p in 0..nr {
        for q in 0..nc {
            let v = grid[[p, q]];
            let delta = if v.is_finite() {
                (v - min_val).abs()
            } else {
                f64::INFINITY
            };
            if delta < best_delta {
                best_delta = delta;
                best = (p, q);
            }
        }
    }
    best
}

struct ArmaFit {
    llf: f64,
}

/// Exact-likelihood ARMA(p, q) fit with a constant, by BFGS over
/// `(const, ar_1..ar_p, ma_1..ma_q, log sigma2)`.
fn fit_arma(y: &Array1<f64>, p: usize, q: usize) -> Result<ArmaFit> {
    let yv = y.to_vec();
    let n = yv.len();
    let mean = yv.iter().sum::<f64>() / n as f64;
    let var = yv.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    // Start parameters following the reference convention: const = mean,
    // sigma2 = sample variance, AR/MA starts from a short Hannan-Rissanen
    // style regression. Simpler robust starts (small coefficients) converge to
    // the same optimum because the likelihood is smooth and these models are
    // identified.
    let (ar0, ma0) = hannan_rissanen_start(&yv, mean, p, q);

    // Parameterisation: [const, ar..., ma..., log_sigma2].
    let np = 1 + p + q + 1;
    let log_var = var.max(1e-8).ln();

    let neg_ll = |x: &Array1<f64>| -> f64 {
        let c = x[0];
        let ar: Vec<f64> = (0..p).map(|i| x[1 + i]).collect();
        let ma: Vec<f64> = (0..q).map(|i| x[1 + p + i]).collect();
        let s2 = x[np - 1].exp();
        // The reference parameterises ARMA in a stationary/invertible region.
        // Add a smooth quadratic barrier whenever the AR/MA characteristic
        // roots approach or cross the unit circle, so the optimiser is guided
        // back to the constrained estimate rather than a non-invertible mirror.
        let rho_ar = max_root_magnitude_ar(&ar);
        let rho_ma = max_root_magnitude_ar(&ma_to_ar(&ma));
        let mut penalty = 0.0;
        let limit = 1.0 - 1e-6;
        if rho_ar > limit {
            penalty += 1e6 * (rho_ar - limit).powi(2) + 1e3 * (rho_ar - limit);
        }
        if rho_ma > limit {
            penalty += 1e6 * (rho_ma - limit).powi(2) + 1e3 * (rho_ma - limit);
        }
        let base = match kalman_loglike(&yv, c, &ar, &ma, s2) {
            Some(ll) => -ll,
            None => 1e12,
        };
        base + penalty
    };

    // Build a panel of starting points. The Hannan-Rissanen estimate is a good
    // start for low orders, but over-parameterised cells can have multiple
    // basins; trying a handful of starts and keeping the best optimum makes the
    // search robust enough to recover the global MLE the reference reports.
    let mut starts: Vec<Array1<f64>> = Vec::new();
    let push_start = |ar: &[f64], ma: &[f64], starts: &mut Vec<Array1<f64>>| {
        let mut x = Array1::<f64>::zeros(np);
        x[0] = mean;
        for i in 0..p {
            x[1 + i] = ar[i];
        }
        for i in 0..q {
            x[1 + p + i] = ma[i];
        }
        x[np - 1] = log_var;
        starts.push(x);
    };
    push_start(&ar0, &ma0, &mut starts);
    // All-zero coefficient start (often the basin of the global optimum).
    push_start(&vec![0.0; p], &vec![0.0; q], &mut starts);
    // Small-coefficient starts of mixed sign.
    push_start(&vec![0.1; p], &vec![0.1; q], &mut starts);
    push_start(&vec![-0.1; p], &vec![-0.1; q], &mut starts);
    if q > 0 {
        // The MA part is the usual source of multiple basins. Combine the HR AR
        // start with a coarse grid of constant MA values so at least one start
        // lands in the global basin.
        for &mv in &[-0.7, -0.4, -0.2, 0.2, 0.4, 0.7] {
            push_start(&ar0, &vec![mv; q], &mut starts);
            push_start(&vec![0.0; p], &vec![mv; q], &mut starts);
        }
    }

    let central = |x: &Array1<f64>| -> Array1<f64> {
        let mut g = Array1::<f64>::zeros(np);
        for i in 0..np {
            let h = 1e-5 * (1.0 + x[i].abs());
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += h;
            xm[i] -= h;
            g[i] = (neg_ll(&xp) - neg_ll(&xm)) / (2.0 * h);
        }
        g
    };
    let fwd = |x: &Array1<f64>| -> Array1<f64> {
        let mut g = Array1::<f64>::zeros(np);
        let f0 = neg_ll(x);
        for i in 0..np {
            let h = 1e-6 * (1.0 + x[i].abs());
            let mut xp = x.clone();
            xp[i] += h;
            g[i] = (neg_ll(&xp) - f0) / h;
        }
        g
    };

    let mut best_fval = f64::INFINITY;
    let mut best_x: Option<Array1<f64>> = None;
    for x0 in &starts {
        // Derivative-free Nelder-Mead is robust on the flat ARMA likelihood;
        // it reliably descends into the basin from each start.
        let (xnm, fnm) = nelder_mead(&neg_ll, x0, 2000, 1e-10);
        let f = if fnm < best_fval { fnm } else { best_fval };
        if fnm < best_fval {
            best_fval = fnm;
            best_x = Some(xnm);
        }
        let _ = f;
    }
    // Polish the overall best with two BFGS passes (forward then central
    // differences) to drive the gradient to zero for a high-accuracy optimum.
    if let Some(x) = best_x {
        if let Ok(r1) = minimize_bfgs(&x, neg_ll, fwd, 500, 1e-8) {
            if r1.fval < best_fval {
                best_fval = r1.fval;
            }
            if let Ok(r2) = minimize_bfgs(&r1.x, neg_ll, central, 500, 1e-10) {
                if r2.fval < best_fval {
                    best_fval = r2.fval;
                }
            }
        }
    }
    if !best_fval.is_finite() {
        return Err(Error::Value("ARMA optimisation failed".into()));
    }
    Ok(ArmaFit { llf: -best_fval })
}

/// Map MA coefficients (sign convention `1 + b1 L + ...`) to the equivalent
/// "AR" coefficients whose companion characteristic polynomial shares roots,
/// so [`max_root_magnitude_ar`] can be reused for invertibility.
fn ma_to_ar(ma: &[f64]) -> Vec<f64> {
    ma.iter().map(|&b| -b).collect()
}

/// Largest magnitude among the roots of `z^p - a1 z^{p-1} - ... - ap`.
/// Returns 0 for an empty coefficient set.
fn max_root_magnitude_ar(ar: &[f64]) -> f64 {
    if ar.is_empty() {
        return 0.0;
    }
    // Polynomial (highest degree first): z^p - a1 z^{p-1} - ... - ap.
    let mut coef = vec![1.0];
    for &a in ar {
        coef.push(-a);
    }
    poly_roots(&coef)
        .iter()
        .map(|(re, im)| (re * re + im * im).sqrt())
        .fold(0.0_f64, f64::max)
}

/// Find all (complex) roots of a real polynomial (highest degree first) via the
/// Durand-Kerner iteration. Returns `(re, im)` pairs.
fn poly_roots(coef: &[f64]) -> Vec<(f64, f64)> {
    // Normalise to monic.
    let lead = coef[0];
    let a: Vec<f64> = coef.iter().map(|&c| c / lead).collect();
    let n = a.len() - 1;
    if n == 0 {
        return vec![];
    }
    // Initial guesses on a spiral.
    let mut roots: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let ang = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64) + 0.4;
            (0.5 * ang.cos(), 0.5 * ang.sin())
        })
        .collect();
    let eval = |x: (f64, f64)| -> (f64, f64) {
        // Horner in complex arithmetic.
        let mut re = a[0];
        let mut im = 0.0;
        for &c in a.iter().skip(1) {
            let nre = re * x.0 - im * x.1 + c;
            let nim = re * x.1 + im * x.0;
            re = nre;
            im = nim;
        }
        (re, im)
    };
    for _ in 0..200 {
        let mut max_delta = 0.0_f64;
        for i in 0..n {
            let xi = roots[i];
            // denominator = prod_{j != i} (xi - xj).
            let mut dre = 1.0;
            let mut dim = 0.0;
            for (j, &xj) in roots.iter().enumerate() {
                if j == i {
                    continue;
                }
                let rre = xi.0 - xj.0;
                let rim = xi.1 - xj.1;
                let nre = dre * rre - dim * rim;
                let nim = dre * rim + dim * rre;
                dre = nre;
                dim = nim;
            }
            let (fre, fim) = eval(xi);
            // delta = f(xi) / denominator (complex division).
            let den = dre * dre + dim * dim;
            if den < 1e-300 {
                continue;
            }
            let qre = (fre * dre + fim * dim) / den;
            let qim = (fim * dre - fre * dim) / den;
            roots[i] = (xi.0 - qre, xi.1 - qim);
            max_delta = max_delta.max((qre * qre + qim * qim).sqrt());
        }
        if max_delta < 1e-12 {
            break;
        }
    }
    roots
}

/// A compact Nelder-Mead simplex minimiser. Returns the best point and value.
fn nelder_mead<F>(f: &F, x0: &Array1<f64>, max_iter: usize, tol: f64) -> (Array1<f64>, f64)
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = x0.len();
    // Build the initial simplex.
    let mut simplex: Vec<Array1<f64>> = Vec::with_capacity(n + 1);
    let mut fvals: Vec<f64> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    fvals.push(f(x0));
    for i in 0..n {
        let mut x = x0.clone();
        let step = if x[i].abs() > 1e-8 {
            0.05 * x[i].abs()
        } else {
            0.05
        };
        x[i] += step;
        let fv = f(&x);
        simplex.push(x);
        fvals.push(fv);
    }

    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    for _ in 0..max_iter {
        // Order by function value.
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ordered: Vec<Array1<f64>> = idx.iter().map(|&i| simplex[i].clone()).collect();
        let ofv: Vec<f64> = idx.iter().map(|&i| fvals[i]).collect();
        simplex = ordered;
        fvals = ofv;

        let best = fvals[0];
        let worst = fvals[n];
        // Convergence: spread of function values is tiny.
        if (worst - best).abs() <= tol * (1.0 + best.abs()) {
            break;
        }

        // Centroid of all but the worst.
        let mut centroid = Array1::<f64>::zeros(n);
        for x in simplex.iter().take(n) {
            centroid = &centroid + x;
        }
        centroid /= n as f64;

        // Reflection.
        let xr = &centroid + &((&centroid - &simplex[n]) * alpha);
        let fr = f(&xr);
        if fr < fvals[0] {
            // Expansion.
            let xe = &centroid + &((&xr - &centroid) * gamma);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contraction.
            let xc = &centroid + &((&simplex[n] - &centroid) * rho);
            let fc = f(&xc);
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // Shrink toward the best.
                let x0 = simplex[0].clone();
                for i in 1..=n {
                    simplex[i] = &x0 + &((&simplex[i] - &x0) * sigma);
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }
    // Return the best vertex.
    let mut bi = 0;
    for i in 1..=n {
        if fvals[i] < fvals[bi] {
            bi = i;
        }
    }
    (simplex[bi].clone(), fvals[bi])
}

/// Hannan-Rissanen starting values. A high-order autoregression first
/// pre-whitens the (demeaned) series to estimate the innovations; the target
/// orders are then obtained by regressing the series on its own `p` lags and
/// the `q` lagged estimated innovations.
fn hannan_rissanen_start(y: &[f64], mean: f64, p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let z: Vec<f64> = y.iter().map(|&v| v - mean).collect();

    // Pre-whitening AR order: generous but bounded by the sample.
    let h = if q > 0 {
        let cand = (p + q + 10).max((n as f64).sqrt() as usize);
        cand.min(n / 2).max(p.max(1))
    } else {
        p
    };

    // Innovations from a long AR(h) regression (conditional residuals).
    let mut resid = vec![0.0; n];
    if h > 0 && n > 2 * h + 1 {
        if let Some(beta) = ols_lags(&z, &[], h, 0) {
            for t in h..n {
                let mut pred = 0.0;
                for (i, &b) in beta.iter().enumerate() {
                    pred += b * z[t - 1 - i];
                }
                resid[t] = z[t] - pred;
            }
        }
    }

    // Second-stage regression of z on its p lags and the q lagged innovations.
    let mut ar = vec![0.0; p];
    let mut ma = vec![0.0; q];
    if p + q > 0 {
        let min_start = if q > 0 { h } else { 0 };
        if let Some(beta) = ols_lags_from(&z, &resid, p, q, min_start) {
            for i in 0..p {
                ar[i] = beta[i];
            }
            for i in 0..q {
                ma[i] = beta[p + i];
            }
        } else if p > 0 {
            // Fall back to a plain AR(p) fit for the AR part.
            if let Some(beta) = ols_lags(&z, &[], p, 0) {
                for i in 0..p {
                    ar[i] = beta[i];
                }
            }
        }
    }
    // Keep starts modest to stay near invertibility/stationarity.
    for a in ar.iter_mut() {
        *a = a.clamp(-0.95, 0.95);
    }
    for m in ma.iter_mut() {
        *m = m.clamp(-0.95, 0.95);
    }
    (ar, ma)
}

/// OLS of `z[t]` on `[z[t-1..t-p], resid[t-1..t-q]]` over `t = start..n`.
///
/// `min_start` lets the caller skip leading rows whose lagged residuals are not
/// yet valid (e.g. before the pre-whitening AR order).
fn ols_lags_from(
    z: &[f64],
    resid: &[f64],
    p: usize,
    q: usize,
    min_start: usize,
) -> Option<Vec<f64>> {
    let n = z.len();
    let k = p + q;
    if k == 0 {
        return None;
    }
    let start = p.max(q).max(min_start);
    if n <= start + k {
        return None;
    }
    let rows = n - start;
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for t in start..n {
        let mut row = vec![0.0; k];
        for i in 0..p {
            row[i] = z[t - 1 - i];
        }
        for i in 0..q {
            row[p + i] = resid[t - 1 - i];
        }
        for a in 0..k {
            for b in 0..k {
                xtx[a][b] += row[a] * row[b];
            }
            xty[a] += row[a] * z[t];
        }
    }
    let _ = rows;
    solve_sym(&mut xtx, &mut xty)
}

/// Convenience wrapper: OLS with no extra start offset.
fn ols_lags(z: &[f64], resid: &[f64], p: usize, q: usize) -> Option<Vec<f64>> {
    ols_lags_from(z, resid, p, q, 0)
}

/// Solve a small symmetric system by Gaussian elimination with partial pivot.
fn solve_sym(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

/// Exact Gaussian log-likelihood of an ARMA(p, q) model with observation
/// intercept `const`, via the Kalman filter in Harvey's state-space form with
/// the stationary (Lyapunov) initial covariance. Returns `None` if the implied
/// state covariance is not positive (non-stationary parameters).
fn kalman_loglike(y: &[f64], constant: f64, ar: &[f64], ma: &[f64], sigma2: f64) -> Option<f64> {
    if sigma2 <= 0.0 || !sigma2.is_finite() {
        return None;
    }
    let p = ar.len();
    let q = ma.len();
    let m = p.max(q + 1);
    let n = y.len();

    // Transition T (companion), selection R, design Z = e_0.
    let mut t = vec![vec![0.0; m]; m];
    for (i, &a) in ar.iter().enumerate() {
        t[i][0] = a;
    }
    for i in 0..(m - 1) {
        t[i][1 + i] = 1.0;
    }
    let mut r = vec![0.0; m];
    r[0] = 1.0;
    for (i, &b) in ma.iter().enumerate() {
        r[1 + i] = b;
    }
    // RQR' = sigma2 * r r'.
    let mut rqr = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            rqr[i][j] = sigma2 * r[i] * r[j];
        }
    }

    // Stationary init: solve (I - T⊗T) vec(P) = vec(RQR').
    let mm = m * m;
    let mut amat = vec![vec![0.0; mm]; mm];
    let mut rhs = vec![0.0; mm];
    for i in 0..m {
        for j in 0..m {
            let row = i * m + j;
            rhs[row] = rqr[i][j];
            for k in 0..m {
                for l in 0..m {
                    let col = k * m + l;
                    let kron = t[i][k] * t[j][l];
                    amat[row][col] = if row == col { 1.0 - kron } else { -kron };
                }
            }
        }
    }
    let pvec = solve_sym(&mut amat, &mut rhs)?;
    let mut pmat = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            pmat[i][j] = pvec[i * m + j];
        }
    }

    let mut a = vec![0.0; m]; // state mean
    let mut ll = 0.0;
    let ln2pi = (2.0 * std::f64::consts::PI).ln();

    for &yt in y.iter().take(n) {
        // v = (y - const) - Z a = (y-const) - a[0]; F = P[0][0].
        let v = (yt - constant) - a[0];
        let f = pmat[0][0];
        if f.is_nan() || f <= 0.0 || f.is_infinite() {
            return None;
        }
        ll += -0.5 * (ln2pi + f.ln() + v * v / f);
        // K = P Z' / F = P[:,0] / F.
        let mut k = vec![0.0; m];
        for i in 0..m {
            k[i] = pmat[i][0] / f;
        }
        // a = a + K v.
        for i in 0..m {
            a[i] += k[i] * v;
        }
        // P = P - K (Z P) = P - K * P[0,:].
        let prow0: Vec<f64> = (0..m).map(|j| pmat[0][j]).collect();
        for i in 0..m {
            for j in 0..m {
                pmat[i][j] -= k[i] * prow0[j];
            }
        }
        // Predict: a = T a.
        let mut anew = vec![0.0; m];
        for i in 0..m {
            let mut s = 0.0;
            for k2 in 0..m {
                s += t[i][k2] * a[k2];
            }
            anew[i] = s;
        }
        a = anew;
        // P = T P T' + RQR'.
        let mut tp = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k2 in 0..m {
                    s += t[i][k2] * pmat[k2][j];
                }
                tp[i][j] = s;
            }
        }
        let mut pnew = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k2 in 0..m {
                    s += tp[i][k2] * t[j][k2];
                }
                pnew[i][j] = s + rqr[i][j];
            }
        }
        pmat = pnew;
    }
    Some(ll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn ar2_series() -> Array1<f64> {
        let n = 100;
        let mut y = vec![0.0; n + 50];
        let mut s = 3.0_f64;
        let mut rnd = || {
            s = (s * 1103515245.0 + 12345.0) % 2147483648.0;
            s / 2147483648.0 - 0.5
        };
        for t in 2..(n + 50) {
            y[t] = 0.5 * y[t - 1] - 0.3 * y[t - 2] + rnd();
        }
        Array1::from_vec(y[50..].to_vec())
    }

    #[test]
    fn grid_has_expected_shape() {
        let y = ar2_series();
        let r = arma_order_select_ic(&y, 2, 2, &[InfoCriterion::Aic, InfoCriterion::Bic]).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].grid.dim(), (3, 3));
        // All entries should be finite for this well-behaved series.
        assert!(r[0].grid.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn white_noise_loglike_matches_closed_form() {
        // For p=q=0 the exact likelihood is the iid Gaussian likelihood.
        let y = Array1::from_vec(vec![0.5, -0.2, 0.1, 0.4, -0.3, 0.2, 0.0, -0.1, 0.3, -0.4]);
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        let ll_cf = -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + var.ln() + 1.0);
        let ll = kalman_loglike(&y.to_vec(), mean, &[], &[], var).unwrap();
        assert!((ll - ll_cf).abs() < 1e-9, "got {ll}, want {ll_cf}");
    }
}
