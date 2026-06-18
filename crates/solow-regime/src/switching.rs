//! Shared machinery for Markov-switching models: parameter transforms, the
//! left-stochastic transition matrix, steady-state initial probabilities, and
//! the fitted-results container.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::{pinv, solve};
use solow_optimize::{approx_fprime, approx_hess, minimize_bfgs};

/// Numerically stable `log(sum(exp(x)))`.
pub(crate) fn logsumexp(x: &[f64]) -> f64 {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let s: f64 = x.iter().map(|&v| (v - m).exp()).sum();
    m + s.ln()
}

/// Build the left-stochastic transition matrix from the transition block of the
/// (constrained) parameter vector.
///
/// `trans` has length `k*(k-1)`. The reference reshapes it row-major into the
/// leading `k-1` rows of the matrix, so entry `(i, j)` for `i < k-1` is
/// `trans[i*k + j]`. The matrix is "left-stochastic": column `j` is the
/// distribution over the next regime given the previous regime was `j`, so
/// `P[i, j] = Pr(S_t = i | S_{t-1} = j)` and each column sums to one. The bottom
/// row is the residual that makes each column sum to one.
pub(crate) fn transition_matrix(trans: &[f64], k: usize) -> Array2<f64> {
    let mut p = Array2::<f64>::zeros((k, k));
    let mut col_sum = vec![0.0f64; k];
    for i in 0..(k - 1) {
        for j in 0..k {
            let v = trans[i * k + j];
            p[[i, j]] = v;
            col_sum[j] += v;
        }
    }
    for j in 0..k {
        p[[k - 1, j]] = 1.0 - col_sum[j];
    }
    p
}

/// Steady-state (ergodic) distribution of the chain with transition matrix `p`.
///
/// Solves `(I - P) pi = 0` subject to `sum(pi) = 1` in the least-squares /
/// pseudo-inverse sense, matching the reference's `pinv(A)[:, -1]` construction.
pub(crate) fn steady_state(p: &Array2<f64>) -> Result<Array1<f64>> {
    let k = p.nrows();
    // The reference builds `A = np.c_[(I - P).T, ones].T`, whose double
    // transpose with the appended ones column reconstructs the rows of `I - P`,
    // giving `A = [[ I - P ]; [ 1 1 ... 1 ]]` of shape `(k+1, k)`.
    let mut a = Array2::<f64>::zeros((k + 1, k));
    for i in 0..k {
        for j in 0..k {
            let id = if i == j { 1.0 } else { 0.0 };
            a[[i, j]] = id - p[[i, j]];
        }
    }
    for j in 0..k {
        a[[k, j]] = 1.0;
    }
    // pi = pinv(A)[:, -1]; the pseudo-inverse has shape (k, k+1).
    let (pinv_a, _sv) = pinv(&a)?;
    let mut pi = Array1::<f64>::zeros(k);
    for i in 0..k {
        pi[i] = pinv_a[[i, k]];
    }
    // Bound away from zero for the log-space filter.
    for v in pi.iter_mut() {
        if *v < 1e-20 {
            *v = 1e-20;
        }
    }
    Ok(pi)
}

/// Logistic transform of one column of unconstrained transition parameters.
///
/// Given `k-1` unconstrained values for column `j`, returns the `k-1`
/// constrained probabilities `exp(u_i) / (1 + sum_l exp(u_l))`. This matches the
/// reference `transform_params`: a softmax over `[0, u_0, ..., u_{k-2}]` keeping
/// the entries past the leading zero.
pub(crate) fn logistic_column(unconstrained: &[f64]) -> Vec<f64> {
    let mut padded = Vec::with_capacity(unconstrained.len() + 1);
    padded.push(0.0);
    padded.extend_from_slice(unconstrained);
    let denom = logsumexp(&padded);
    unconstrained.iter().map(|&u| (u - denom).exp()).collect()
}

/// Inverse of [`logistic_column`] for two regimes (`k - 1 == 1`): the closed-form
/// `u = -log(1/c - 1)`.
pub(crate) fn inv_logistic_scalar(c: f64) -> f64 {
    -((1.0 / c) - 1.0).ln()
}

/// Flat index of the transition parameter for matrix entry `(i, j)`, `i < k-1`.
///
/// The reference reshapes the transition block row-major into the leading `k-1`
/// rows, so the flat index is `i * k + j`.
fn trans_index(i: usize, j: usize, k: usize) -> usize {
    i * k + j
}

/// Human-readable names for the `k*(k-1)` transition parameters, matching the
/// reference order `for i in 0..k-1 { for j in 0..k }` with name `p[j->i]`.
pub(crate) fn transition_param_names(k: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(k * (k - 1));
    for i in 0..(k - 1) {
        for j in 0..k {
            names.push(format!("p[{j}->{i}]"));
        }
    }
    names
}

/// Apply the logistic transform to the transition block of `out` in place,
/// reading unconstrained values from `u`. Done per source column `j`: the `k-1`
/// entries `(0, j), ..., (k-2, j)` form one softmax group.
pub(crate) fn transform_transition(out: &mut Array1<f64>, u: &Array1<f64>, k: usize) {
    for j in 0..k {
        let col: Vec<f64> = (0..(k - 1)).map(|i| u[trans_index(i, j, k)]).collect();
        let probs = logistic_column(&col);
        for (i, &p) in probs.iter().enumerate() {
            out[trans_index(i, j, k)] = p;
        }
    }
}

/// Inverse of [`transform_transition`]: recover unconstrained values into `out`
/// from constrained probabilities `c`.
pub(crate) fn untransform_transition(out: &mut Array1<f64>, c: &Array1<f64>, k: usize) {
    for j in 0..k {
        if k == 2 {
            let idx = trans_index(0, j, k);
            out[idx] = inv_logistic_scalar(c[idx]);
        } else {
            let probs: Vec<f64> = (0..(k - 1)).map(|i| c[trans_index(i, j, k)]).collect();
            let unc = inv_logistic_general(&probs);
            for (i, &v) in unc.iter().enumerate() {
                out[trans_index(i, j, k)] = v;
            }
        }
    }
}

/// Inverse logistic for a general number of regimes (`k - 1 > 1`), recovering
/// unconstrained values from a probability vector. With
/// `c_i = exp(u_i) / (1 + sum_l exp(u_l))` and the implicit leading element 1,
/// `u_i = log(c_i) - log(1 - sum_l c_l)`.
fn inv_logistic_general(probs: &[f64]) -> Vec<f64> {
    let rem: f64 = 1.0 - probs.iter().sum::<f64>();
    let base = rem.max(1e-20).ln();
    probs.iter().map(|&c| c.max(1e-20).ln() - base).collect()
}

/// Results of fitting a Markov-switching model.
#[derive(Clone, Debug)]
pub struct MarkovResults {
    /// Maximum-likelihood parameter estimates in constrained space. See
    /// [`param_names`](Self::param_names) for the per-element layout (transition,
    /// then mean/exog, then variance, then — for the autoregression — the AR
    /// coefficients).
    pub params: Array1<f64>,
    /// Human-readable name for each parameter.
    pub param_names: Vec<String>,
    /// Maximised log-likelihood.
    pub llf: f64,
    /// Number of observations used in estimation (after differencing out lags).
    pub nobs: usize,
    /// Number of free parameters.
    pub k_params: usize,
    /// Akaike information criterion, `2k - 2*llf`.
    pub aic: f64,
    /// Bayesian information criterion, `k*ln(nobs) - 2*llf`.
    pub bic: f64,
    /// Whether the optimiser reported convergence.
    pub converged: bool,
    /// Left-stochastic transition matrix at the estimates, `P[i, j] = Pr(S_t = i | S_{t-1} = j)`.
    pub transition: Array2<f64>,
    /// Steady-state initial regime distribution.
    pub initial_probabilities: Array1<f64>,
    /// Filtered marginal regime probabilities `Pr(S_t = i | Y_t)`, shaped `(nobs, k)`.
    pub filtered_marginal_probabilities: Array2<f64>,
    /// Smoothed marginal regime probabilities `Pr(S_t = i | Y_T)`, shaped `(nobs, k)`.
    pub smoothed_marginal_probabilities: Array2<f64>,
    /// Expected duration of each regime, `1 / (1 - P[i, i])`.
    pub expected_durations: Array1<f64>,
}

impl MarkovResults {
    // Internal aggregating constructor; the fields are independent so they are
    // passed positionally rather than via an intermediate builder struct.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        params: Array1<f64>,
        param_names: Vec<String>,
        llf: f64,
        nobs: usize,
        transition: Array2<f64>,
        initial_probabilities: Array1<f64>,
        filtered: Array2<f64>,
        smoothed: Array2<f64>,
        converged: bool,
    ) -> Self {
        let k_params = params.len();
        let aic = 2.0 * k_params as f64 - 2.0 * llf;
        let bic = (k_params as f64) * (nobs as f64).ln() - 2.0 * llf;
        let k = transition.nrows();
        let mut expected_durations = Array1::<f64>::zeros(k);
        for i in 0..k {
            let stay = transition[[i, i]];
            expected_durations[i] = if (1.0 - stay).abs() < 1e-300 {
                f64::INFINITY
            } else {
                1.0 / (1.0 - stay)
            };
        }
        MarkovResults {
            params,
            param_names,
            llf,
            nobs,
            k_params,
            aic,
            bic,
            converged,
            transition,
            initial_probabilities,
            filtered_marginal_probabilities: filtered,
            smoothed_marginal_probabilities: smoothed,
            expected_durations,
        }
    }
}

/// Ordinary-least-squares coefficients `(X'X)^{-1} X'y` via the pseudo-inverse,
/// matching `np.linalg.pinv(X) @ y`.
pub(crate) fn ols(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
    if x.nrows() != y.len() {
        return Err(Error::Shape("ols: rows mismatch".into()));
    }
    let (pinv_x, _sv) = pinv(x)?;
    Ok(pinv_x.dot(y))
}

/// Population variance (divisor `n`) of a residual vector.
pub(crate) fn var_pop(r: &Array1<f64>) -> f64 {
    let n = r.len() as f64;
    let mean = r.sum() / n;
    r.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n
}

/// Maximise a log-likelihood (by minimising its negative) starting from
/// `start_u` in unconstrained space.
///
/// Strategy: run BFGS to get into the neighbourhood of the optimum, then polish
/// with damped Newton steps (finite-difference gradient and Hessian) which
/// converge quadratically and drive the gradient norm to ~0. This mirrors the
/// reference, whose BFGS terminates essentially at a stationary point.
///
/// Returns `(x, neg_loglike, converged)` where `converged` is true once the
/// gradient norm of the negative log-likelihood falls below `1e-6`.
pub(crate) fn maximize<F>(
    start_u: &Array1<f64>,
    neg_loglike: F,
    maxiter: usize,
) -> (Array1<f64>, f64, bool)
where
    F: Fn(&Array1<f64>) -> f64,
{
    let f = |u: &Array1<f64>| {
        let v = neg_loglike(u);
        if v.is_finite() {
            v
        } else {
            1e10
        }
    };
    let g = |u: &Array1<f64>| approx_fprime(u, |x| f(x));

    // Phase 1: BFGS.
    let mut x = match minimize_bfgs(start_u, |u| f(u), |u| g(u), maxiter, 1e-8) {
        Ok(r) => r.x,
        Err(_) => start_u.clone(),
    };

    // Phase 2: damped Newton polishing.
    let mut fval = f(&x);
    let mut grad = g(&x);
    let mut gnorm = grad.dot(&grad).sqrt();
    for _ in 0..100 {
        if gnorm < 1e-7 {
            break;
        }
        let h = approx_hess(&x, |u| f(u));
        let step = match solve(&h, &grad) {
            Ok(s) => s,
            Err(_) => break,
        };
        // Backtracking to ensure descent of the negative log-likelihood.
        let mut alpha = 1.0;
        let mut improved = false;
        for _ in 0..40 {
            let cand = &x - &(alpha * &step);
            let fc = f(&cand);
            if fc < fval - 1e-12 * (1.0 + fval.abs()) {
                x = cand;
                fval = fc;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !improved {
            break;
        }
        grad = g(&x);
        gnorm = grad.dot(&grad).sqrt();
    }

    let converged = gnorm < 1e-6;
    (x, fval, converged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn transition_matrix_row_major_k3() {
        // For k=3 the leading 2 rows are filled row-major: indices
        // [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)] = [a,b,c,d,e,f]; the bottom row
        // is the column residual.
        let trans = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let p = transition_matrix(&trans, 3);
        assert_eq!(p[[0, 0]], 0.1);
        assert_eq!(p[[0, 1]], 0.2);
        assert_eq!(p[[0, 2]], 0.3);
        assert_eq!(p[[1, 0]], 0.4);
        assert_eq!(p[[1, 1]], 0.5);
        assert_eq!(p[[1, 2]], 0.6);
        // Columns sum to one.
        for j in 0..3 {
            let s: f64 = (0..3).map(|i| p[[i, j]]).sum();
            assert!((s - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn steady_state_is_stationary() {
        let p = array![[0.9, 0.2], [0.1, 0.8]];
        let pi = steady_state(&p).unwrap();
        // P pi == pi.
        let ppi = p.dot(&pi);
        for i in 0..2 {
            assert!((ppi[i] - pi[i]).abs() < 1e-12);
        }
        assert!((pi.sum() - 1.0).abs() < 1e-12);
        // Closed form for a 2-state chain: pi_0 = p10 / (p01 + p10) with
        // p01 = P[0,1] = 0.2, p10 = P[1,0] = 0.1 => pi_0 = 0.2/0.3? Use the
        // standard ergodic formula pi ∝ [P[0,1], P[1,0]] swapped appropriately.
        // Here P[1,0]=0.1 leaving state 0, P[0,1]=0.2 leaving state 1.
        let want0 = 0.2 / (0.1 + 0.2);
        assert!((pi[0] - want0).abs() < 1e-9);
    }

    #[test]
    fn logistic_roundtrip_k2() {
        let u = array![0.7, -1.3];
        let mut c = u.clone();
        transform_transition(&mut c, &u, 2);
        // Each is a probability in (0, 1).
        assert!(c[0] > 0.0 && c[0] < 1.0);
        assert!(c[1] > 0.0 && c[1] < 1.0);
        let mut back = c.clone();
        untransform_transition(&mut back, &c, 2);
        assert!((back[0] - u[0]).abs() < 1e-10);
        assert!((back[1] - u[1]).abs() < 1e-10);
    }

    #[test]
    fn logistic_roundtrip_k3() {
        // Two unconstrained columns of 2 entries each (k-1=2, k=3).
        let u = array![0.3, -0.5, 1.1, -0.2, 0.6, 0.1];
        let mut c = u.clone();
        transform_transition(&mut c, &u, 3);
        // Columns of probabilities must each be < 1 with positive residual.
        for j in 0..3 {
            let s = c[j] + c[3 + j];
            assert!(s < 1.0 && s > 0.0, "column {j} sum {s}");
        }
        let mut back = c.clone();
        untransform_transition(&mut back, &c, 3);
        for i in 0..6 {
            assert!(
                (back[i] - u[i]).abs() < 1e-9,
                "idx {i}: {} vs {}",
                back[i],
                u[i]
            );
        }
    }
}
