//! SARIMAX(p, d, q)(P, D, Q, s) estimation by maximum likelihood.
//!
//! The model is cast in the Harvey state-space form and estimated with the
//! exact Gaussian log-likelihood from the [`crate::kalman`] filter. Non-seasonal
//! and seasonal differencing are applied directly to the data (the
//! *simple-differencing* convention), after which the differenced series
//! follows a stationary ARMA whose state space uses the stationary
//! (discrete-Lyapunov) initialization.
//!
//! Stationarity and invertibility are enforced through the Monahan
//! reparametrization so the unconstrained BFGS iterates always map to a
//! stationary AR and invertible MA polynomial; `sigma^2` is mapped through a
//! square so it stays positive.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_sf;
use solow_linalg::{inv, solve};
use solow_optimize::{approx_fprime, approx_hess, minimize_bfgs};

use crate::kalman::StateSpace;

/// Specification of a SARIMAX order `(p, d, q)(P, D, Q, s)`.
#[derive(Clone, Copy, Debug)]
pub struct SarimaxOrder {
    /// Non-seasonal AR order `p`.
    pub p: usize,
    /// Non-seasonal differencing order `d`.
    pub d: usize,
    /// Non-seasonal MA order `q`.
    pub q: usize,
    /// Seasonal AR order `P`.
    pub sp: usize,
    /// Seasonal differencing order `D`.
    pub sd: usize,
    /// Seasonal MA order `Q`.
    pub sq: usize,
    /// Seasonal period `s` (0 when there is no seasonal part).
    pub s: usize,
}

impl SarimaxOrder {
    /// Non-seasonal `(p, d, q)` order with no seasonal component.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        SarimaxOrder {
            p,
            d,
            q,
            sp: 0,
            sd: 0,
            sq: 0,
            s: 0,
        }
    }

    /// Full seasonal order `(p, d, q)(P, D, Q, s)`.
    pub fn seasonal(
        p: usize,
        d: usize,
        q: usize,
        sp: usize,
        sd: usize,
        sq: usize,
        s: usize,
    ) -> Self {
        SarimaxOrder {
            p,
            d,
            q,
            sp,
            sd,
            sq,
            s,
        }
    }

    /// Number of free AR coefficients (`p + P`).
    fn k_ar(&self) -> usize {
        self.p + self.sp
    }

    /// Number of free MA coefficients (`q + Q`).
    fn k_ma(&self) -> usize {
        self.q + self.sq
    }

    /// Total number of estimated parameters (AR + MA + sigma^2).
    fn k_params(&self) -> usize {
        self.k_ar() + self.k_ma() + 1
    }
}

/// A fitted SARIMAX model.
#[derive(Clone, Debug)]
pub struct SarimaxResults {
    /// Estimated parameters, ordered `[ar.L1.., ar.S.L.., ma.L1.., ma.S.L.., sigma2]`.
    pub params: Array1<f64>,
    /// Standard errors from the inverse negative Hessian of the log-likelihood.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub zvalues: Array1<f64>,
    /// Two-sided normal p-values.
    pub pvalues: Array1<f64>,
    /// Maximized log-likelihood.
    pub llf: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,
    /// Hannan-Quinn information criterion.
    pub hqic: f64,
    /// One-step-ahead in-sample predictions on the original scale of the
    /// differenced series (length = number of differenced observations).
    pub fittedvalues: Array1<f64>,
    /// In-sample residuals `y - fittedvalues`.
    pub resid: Array1<f64>,
    /// Whether the optimizer's gradient test was satisfied.
    pub converged: bool,
    /// Number of effective observations (after differencing).
    pub nobs: usize,
}

/// SARIMAX model bound to an observed series.
#[derive(Clone, Debug)]
pub struct Sarimax {
    endog: Array1<f64>,
    order: SarimaxOrder,
}

impl Sarimax {
    /// Build a SARIMAX model for `endog` with the given `order`.
    pub fn new(endog: Array1<f64>, order: SarimaxOrder) -> Result<Self> {
        if order.s == 0 && (order.sp > 0 || order.sd > 0 || order.sq > 0) {
            return Err(Error::Value(
                "seasonal period s must be > 0 when a seasonal order is set".into(),
            ));
        }
        Ok(Sarimax { endog, order })
    }

    /// The differenced series used for estimation (simple differencing).
    pub fn differenced(&self) -> Array1<f64> {
        difference(&self.endog, &self.order)
    }

    /// Fit by maximum likelihood, optimizing `-loglike` with BFGS over the
    /// unconstrained (Monahan-reparametrized) space.
    pub fn fit(&self) -> Result<SarimaxResults> {
        let order = self.order;
        let k = order.k_params();
        let y = self.differenced();
        let nobs = y.len();
        if nobs == 0 {
            return Err(Error::Value("no observations after differencing".into()));
        }

        // Objective in unconstrained space: returns -loglike.
        let neg_ll = |u: &Array1<f64>| -> f64 {
            let p = transform_params(u, &order);
            match loglike(&y, &p, &order) {
                Some(ll) if ll.is_finite() => -ll,
                _ => f64::INFINITY,
            }
        };
        let grad = |u: &Array1<f64>| approx_fprime(u, neg_ll);

        // Start: AR/MA at zero, sigma^2 at the sample variance of the
        // differenced data. In unconstrained space these map to 0 (AR/MA) and
        // sqrt(var) (sigma2 via the square map).
        let var = sample_var(&y).max(1e-8);
        let mut start = Array1::<f64>::zeros(k);
        start[k - 1] = var.sqrt();

        // Restart BFGS in short bursts, stopping once the objective stops
        // improving relative to its scale (an ftol rule). The finite-difference
        // gradient cannot drive the gradient norm down to the optimizer's gtol,
        // so convergence is judged by the function-value change instead. A final
        // Newton polish then sharpens the optimum.
        let mut u_hat = start.clone();
        let mut f_prev = neg_ll(&u_hat);
        let mut converged = false;
        for _ in 0..200 {
            let res = minimize_bfgs(&u_hat, neg_ll, grad, 20, 1e-12)?;
            u_hat = res.x;
            let f_now = res.fval;
            if res.converged || (f_prev - f_now).abs() <= 1e-12 * (1.0 + f_now.abs()) {
                converged = true;
                break;
            }
            f_prev = f_now;
        }
        // Newton polish in the unconstrained space to tighten the gradient.
        u_hat = newton_polish(&u_hat, neg_ll);
        let params = transform_params(&u_hat, &order);

        let llf = loglike(&y, &params, &order)
            .ok_or_else(|| Error::Convergence("log-likelihood undefined at optimum".into()))?;

        // Standard errors: Hessian of the (natural-parameter) log-likelihood.
        let ll_natural =
            |p: &Array1<f64>| -> f64 { loglike(&y, p, &order).unwrap_or(f64::NEG_INFINITY) };
        let h = approx_hess(&params, ll_natural);
        let neg_h = h.mapv(|v| -v);
        let cov = inv(&neg_h)?;
        let bse = Array1::from_iter((0..k).map(|i| {
            let v = cov[[i, i]];
            if v > 0.0 {
                v.sqrt()
            } else {
                f64::NAN
            }
        }));

        let zvalues = Array1::from_iter((0..k).map(|i| params[i] / bse[i]));
        let pvalues = zvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        // Information criteria.
        let kf = k as f64;
        let n = nobs as f64;
        let aic = -2.0 * llf + 2.0 * kf;
        let bic = -2.0 * llf + kf * n.ln();
        let hqic = -2.0 * llf + 2.0 * kf * n.ln().ln();

        // Fitted values: one-step-ahead predictions = y - forecast_error.
        let ss = build_state_space(&params, &order)?;
        let out = ss.filter(&y, 0);
        let fittedvalues = &y - &out.forecast_error;
        let resid = out.forecast_error.clone();

        Ok(SarimaxResults {
            params,
            bse,
            zvalues,
            pvalues,
            llf,
            aic,
            bic,
            hqic,
            fittedvalues,
            resid,
            converged,
            nobs,
        })
    }
}

/// A handful of damped Newton steps on `f` to sharpen an already-good optimum.
///
/// Uses finite-difference gradients and Hessians; each proposed step is only
/// accepted if it decreases the objective, so the polish never moves away from
/// the located minimum even when the Hessian is poorly conditioned.
fn newton_polish<F>(start: &Array1<f64>, mut f: F) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let mut x = start.clone();
    let mut fx = f(&x);
    for _ in 0..25 {
        let g = approx_fprime(&x, &mut f);
        let gnorm = g.dot(&g).sqrt();
        if gnorm < 1e-9 {
            break;
        }
        let h = approx_hess(&x, &mut f);
        let step = match solve(&h, &g) {
            Ok(s) => s,
            Err(_) => break,
        };
        // Try the full Newton step, backtracking if it does not improve.
        let mut alpha = 1.0;
        let mut improved = false;
        for _ in 0..20 {
            let cand = &x - &(&step * alpha);
            let fc = f(&cand);
            if fc < fx {
                x = cand;
                fx = fc;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !improved {
            break;
        }
    }
    x
}

/// Sample variance (population, divided by `n`) used for the sigma^2 start.
fn sample_var(y: &Array1<f64>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let mean = y.sum() / n as f64;
    y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64
}

/// Apply non-seasonal and seasonal differencing to `y`.
fn difference(y: &Array1<f64>, order: &SarimaxOrder) -> Array1<f64> {
    let mut v: Vec<f64> = y.to_vec();
    for _ in 0..order.d {
        v = (1..v.len()).map(|i| v[i] - v[i - 1]).collect();
    }
    if order.s > 0 {
        let s = order.s;
        for _ in 0..order.sd {
            v = (s..v.len()).map(|i| v[i] - v[i - s]).collect();
        }
    }
    Array1::from_vec(v)
}

/// Transform unconstrained optimizer parameters to natural parameters.
///
/// AR and MA blocks each pass through the Monahan stationary map; the trailing
/// `sigma^2` is the square of its unconstrained value.
fn transform_params(u: &Array1<f64>, order: &SarimaxOrder) -> Array1<f64> {
    let mut p = Array1::<f64>::zeros(u.len());
    let mut off = 0;
    let k_ar = order.k_ar();
    let k_ma = order.k_ma();
    if k_ar > 0 {
        let block = u.slice(ndarray::s![off..off + k_ar]).to_owned();
        let c = constrain_stationary(&block);
        for (i, &val) in c.iter().enumerate() {
            p[off + i] = val;
        }
        off += k_ar;
    }
    if k_ma > 0 {
        let block = u.slice(ndarray::s![off..off + k_ma]).to_owned();
        // MA uses the same map but with the opposite overall sign convention so
        // that the natural MA coefficients match the reference parametrization.
        let c = constrain_stationary(&block);
        for (i, &val) in c.iter().enumerate() {
            p[off + i] = -val;
        }
    }
    // sigma^2 = u^2.
    let last = u.len() - 1;
    p[last] = u[last] * u[last];
    p
}

/// Monahan (1984) map from unconstrained reals to stationary AR coefficients.
///
/// Returns the constrained coefficients of `(1 + a_1 L + ... + a_k L^k)`-style
/// polynomials following the reference convention (the result equals
/// `-y[k-1, :]` of the Levinson recursion on the partial autocorrelations
/// `r_i = u_i / sqrt(1 + u_i^2)`).
fn constrain_stationary(u: &Array1<f64>) -> Array1<f64> {
    let n = u.len();
    let r: Vec<f64> = u.iter().map(|&v| v / (1.0 + v * v).sqrt()).collect();
    let mut y = vec![vec![0.0_f64; n]; n];
    for k in 0..n {
        for i in 0..k {
            y[k][i] = y[k - 1][i] + r[k] * y[k - 1][k - i - 1];
        }
        y[k][k] = r[k];
    }
    Array1::from_iter((0..n).map(|i| -y[n - 1][i]))
}

/// Build the full AR / MA lag polynomials from the natural parameters.
///
/// Returns `(ar_full, ma_full)` where each is the list of non-constant
/// coefficients (length `k_ar_full` / `k_ma_full`) such that the AR polynomial
/// is `1 - ar_full[0] L - ar_full[1] L^2 - ...` and the MA polynomial is
/// `1 + ma_full[0] L + ma_full[1] L^2 + ...`.
fn polynomials(params: &Array1<f64>, order: &SarimaxOrder) -> (Vec<f64>, Vec<f64>) {
    let mut idx = 0;
    let ar: Vec<f64> = params.slice(ndarray::s![idx..idx + order.p]).to_vec();
    idx += order.p;
    let sar: Vec<f64> = params.slice(ndarray::s![idx..idx + order.sp]).to_vec();
    idx += order.sp;
    let ma: Vec<f64> = params.slice(ndarray::s![idx..idx + order.q]).to_vec();
    idx += order.q;
    let sma: Vec<f64> = params.slice(ndarray::s![idx..idx + order.sq]).to_vec();

    // Non-seasonal AR polynomial 1 - ar_1 L - ... ; coefficients of L^j (j>=0).
    let mut ar_poly = vec![1.0];
    for &phi in &ar {
        ar_poly.push(-phi);
    }
    // Seasonal AR polynomial 1 - sar_1 L^s - ...
    let mut sar_poly = vec![1.0];
    if order.s > 0 {
        for &phi in &sar {
            sar_poly.resize(sar_poly.len() + order.s, 0.0);
            *sar_poly.last_mut().unwrap() = -phi;
        }
    }
    let full_ar_poly = poly_mul(&ar_poly, &sar_poly);

    // MA polynomials with the +convention.
    let mut ma_poly = vec![1.0];
    for &th in &ma {
        ma_poly.push(th);
    }
    let mut sma_poly = vec![1.0];
    if order.s > 0 {
        for &th in &sma {
            sma_poly.resize(sma_poly.len() + order.s, 0.0);
            *sma_poly.last_mut().unwrap() = th;
        }
    }
    let full_ma_poly = poly_mul(&ma_poly, &sma_poly);

    // ar_full[j] = -coefficient of L^{j+1} (so polynomial is 1 - sum ar_full L).
    let ar_full: Vec<f64> = full_ar_poly[1..].iter().map(|&c| -c).collect();
    // ma_full[j] = coefficient of L^{j+1}.
    let ma_full: Vec<f64> = full_ma_poly[1..].to_vec();
    (ar_full, ma_full)
}

/// Multiply two polynomials given as coefficient vectors (index = power of L).
fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Construct the state space for fixed parameters (test/cross-validation hook).
///
/// Exposes [`build_state_space`] so external tests can verify the fixed-parameter
/// Kalman log-likelihood against reference values.
#[doc(hidden)]
pub fn build_state_space_for_test(
    params: &Array1<f64>,
    order: &SarimaxOrder,
) -> Result<StateSpace> {
    build_state_space(params, order)
}

/// Construct the time-invariant state space for the (already differenced) ARMA.
fn build_state_space(params: &Array1<f64>, order: &SarimaxOrder) -> Result<StateSpace> {
    let (ar_full, ma_full) = polynomials(params, order);
    let k_ar_full = ar_full.len();
    let k_ma_full = ma_full.len();
    let sigma2 = params[params.len() - 1];

    // State dimension: Harvey representation.
    let m = std::cmp::max(k_ar_full, k_ma_full + 1).max(1);

    // Transition: companion form with AR coefficients down the first column and
    // a unit super-diagonal.
    let mut transition = Array2::<f64>::zeros((m, m));
    for (i, &phi) in ar_full.iter().enumerate() {
        transition[[i, 0]] = phi;
    }
    for i in 0..m - 1 {
        transition[[i, i + 1]] = 1.0;
    }

    // Selection: [1, theta_1, ..., theta_{m-1}].
    let mut selection = Array2::<f64>::zeros((m, 1));
    selection[[0, 0]] = 1.0;
    for (i, &theta) in ma_full.iter().enumerate() {
        if i + 1 < m {
            selection[[i + 1, 0]] = theta;
        }
    }

    let state_cov = Array2::from_elem((1, 1), sigma2);

    // Design row: [1, 0, ..., 0].
    let mut design = Array1::<f64>::zeros(m);
    design[0] = 1.0;

    // Stationary initialization via the discrete Lyapunov equation
    //   P = T P T' + R Q R'.
    let rqr = selection.dot(&state_cov).dot(&selection.t());
    let init_cov = solve_discrete_lyapunov(&transition, &rqr)?;
    let init_state = Array1::<f64>::zeros(m);

    Ok(StateSpace {
        transition,
        selection,
        state_cov,
        design,
        obs_cov: 0.0,
        init_state,
        init_cov,
    })
}

/// Solve `P = T P T' + C` for `P` via the vectorized form
/// `vec(P) = (I - T ⊗ T)^{-1} vec(C)`.
fn solve_discrete_lyapunov(t: &Array2<f64>, c: &Array2<f64>) -> Result<Array2<f64>> {
    let m = t.nrows();
    let m2 = m * m;
    // Build I - kron(T, T).
    let mut a = Array2::<f64>::eye(m2);
    for i in 0..m {
        for j in 0..m {
            let tij = t[[i, j]];
            if tij == 0.0 {
                continue;
            }
            for p in 0..m {
                for q in 0..m {
                    // kron(T,T)[(i*m+p),(j*m+q)] = T[i,j]*T[p,q]
                    a[[i * m + p, j * m + q]] -= tij * t[[p, q]];
                }
            }
        }
    }
    // vec(C) in row-major flattening consistent with the kron indexing above.
    let mut vec_c = Array1::<f64>::zeros(m2);
    for i in 0..m {
        for p in 0..m {
            vec_c[i * m + p] = c[[i, p]];
        }
    }
    let vec_p = solve(&a, &vec_c)?;
    let mut p = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            p[[i, j]] = vec_p[i * m + j];
        }
    }
    // Symmetrize.
    for i in 0..m {
        for j in (i + 1)..m {
            let v = 0.5 * (p[[i, j]] + p[[j, i]]);
            p[[i, j]] = v;
            p[[j, i]] = v;
        }
    }
    Ok(p)
}

/// Evaluate the exact Gaussian log-likelihood at the natural parameters.
fn loglike(y: &Array1<f64>, params: &Array1<f64>, order: &SarimaxOrder) -> Option<f64> {
    if params[params.len() - 1] <= 0.0 {
        return None;
    }
    let ss = build_state_space(params, order).ok()?;
    let out = ss.filter(y, 0);
    if out.loglike.is_finite() {
        Some(out.loglike)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn differencing_orders() {
        let y = array![1.0, 3.0, 6.0, 10.0, 15.0];
        let d1 = difference(&y, &SarimaxOrder::new(0, 1, 0));
        assert_eq!(d1.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
        let d2 = difference(&y, &SarimaxOrder::new(0, 2, 0));
        assert_eq!(d2.to_vec(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn seasonal_polynomial_signs() {
        // (1,0,0)x(1,0,0,4): full AR = (1 - 0.5L)(1 - 0.3L^4)
        //   = 1 - 0.5L - 0.3L^4 + 0.15 L^5.
        // ar_full (negated non-constant coeffs) = [0.5, 0, 0, 0.3, -0.15].
        let order = SarimaxOrder::seasonal(1, 0, 0, 1, 0, 0, 4);
        let params = array![0.5, 0.3, 1.0];
        let (ar_full, _ma) = polynomials(&params, &order);
        let expect = [0.5, 0.0, 0.0, 0.3, -0.15];
        assert_eq!(ar_full.len(), 5);
        for (a, b) in ar_full.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn constrain_ar1_matches_known_value() {
        // u = 0.7 -> -0.7/sqrt(1+0.49) = -0.5734623344...
        let c = constrain_stationary(&array![0.7]);
        assert!((c[0] - (-0.573_462_344_363)).abs() < 1e-10, "{}", c[0]);
    }

    #[test]
    fn lyapunov_solves_fixed_point() {
        let t = array![[0.5, 1.0], [0.0, 0.0]];
        let c = array![[1.0, 0.3], [0.3, 0.09]];
        let p = solve_discrete_lyapunov(&t, &c).unwrap();
        let recon = t.dot(&p).dot(&t.t()) + &c;
        for i in 0..2 {
            for j in 0..2 {
                assert!((p[[i, j]] - recon[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
