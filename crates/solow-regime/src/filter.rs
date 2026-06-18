//! The Hamilton filter and Kim smoother for `k`-regime, `order`-dependent
//! Markov-switching models, evaluated in log space for numerical stability.
//!
//! The joint state at time `t` is the tuple `(S_t, S_{t-1}, ..., S_{t-order})`,
//! enumerated in row-major order over `k^(order+1)` combinations with `S_t` the
//! most-significant index. For `order == 0` this reduces to the textbook
//! Hamilton filter over the `k` current regimes.
//
// The loops here index by joint-state and regime-position integers that are used
// as values (digit decomposition, mixed-radix index arithmetic), not merely to
// walk a slice, so the range-loop form is the clearer one.
#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2};

use crate::switching::logsumexp;

/// Output of the Hamilton filter.
pub(crate) struct FilterOutput {
    /// Total log-likelihood (sum over observations).
    pub llf: f64,
    /// Filtered marginal probabilities `Pr(S_t = i | Y_t)`, shaped `(nobs, k)`.
    pub filtered_marginal: Array2<f64>,
    /// Predicted joint probabilities (log space), shaped `(k^(order+1), nobs)`.
    pub predicted_joint_log: Array2<f64>,
    /// Filtered joint probabilities (log space), shaped `(k^(order+1), nobs)`.
    pub filtered_joint_log: Array2<f64>,
}

/// Enumerated index helpers for the joint state of dimension `order + 1`.
fn n_states(k: usize, order: usize) -> usize {
    k.pow((order + 1) as u32)
}

/// Decompose a joint-state index into its `order + 1` regime digits, most
/// significant first: `digits[0] = S_t`, `digits[order] = S_{t-order}`.
fn digits(mut idx: usize, k: usize, order: usize) -> Vec<usize> {
    let mut d = vec![0usize; order + 1];
    for pos in (0..=order).rev() {
        d[pos] = idx % k;
        idx /= k;
    }
    d
}

/// Run the Hamilton filter.
///
/// * `init` — initial regime distribution `Pr(S = i)` of length `k`.
/// * `transition` — left-stochastic matrix, `P[i, j] = Pr(S_t = i | S_{t-1} = j)`.
/// * `cond_loglik` — conditional log-likelihoods `log f(y_t | S_t, ..., S_{t-order})`,
///   shaped `(k^(order+1), nobs)`, indexed by the joint state.
pub(crate) fn hamilton_filter(
    init: &Array1<f64>,
    transition: &Array2<f64>,
    cond_loglik: &Array2<f64>,
    order: usize,
) -> FilterOutput {
    let k = init.len();
    let ns = n_states(k, order);
    let nobs = cond_loglik.ncols();

    let log_init = init.mapv(|v| v.max(1e-20).ln());
    let log_p = transition.mapv(|v| v.max(1e-20).ln());

    let mut predicted_joint_log = Array2::<f64>::zeros((ns, nobs));
    let mut filtered_joint_log = Array2::<f64>::zeros((ns, nobs));
    let mut filtered_marginal = Array2::<f64>::zeros((nobs, k));
    let mut llf = 0.0;

    // Initial filtered joint distribution before observing y_0, over the
    // `order + 1` regimes. Built as
    //   log f0(s_0, ..., s_{-order}) = log pi(s_{-order})
    //       + sum_{m} log P[s_m, s_{m+1}].
    // For `order == 0` this is simply `log pi`.
    let mut prev_filtered_log = vec![0.0f64; ns];
    for state in 0..ns {
        let d = digits(state, k, order);
        let mut acc = log_init[d[order]];
        for m in 0..order {
            acc += log_p[[d[m], d[m + 1]]];
        }
        prev_filtered_log[state] = acc;
    }

    // Reusable buffer.
    let mut predicted_log = vec![0.0f64; ns];

    for t in 0..nobs {
        // Predicted joint: Pr(S_t, ..., S_{t-order} | Y_{t-1}).
        // Pr(S_t = i, S_{t-1}, ..., S_{t-order})
        //   = P[i, S_{t-1}] * Pr(S_{t-1}, ..., S_{t-order} | Y_{t-1}),
        // where the right factor is the marginal of the previous filtered joint
        // over its first `order` digits (dropping S_{t-1-order}).
        if order == 0 {
            // predicted[i] = logsumexp_j (log P[i, j] + prev_filtered[j])
            for i in 0..k {
                let mut terms = Vec::with_capacity(k);
                for j in 0..k {
                    terms.push(log_p[[i, j]] + prev_filtered_log[j]);
                }
                predicted_log[i] = logsumexp(&terms);
            }
        } else {
            for state in 0..ns {
                let d = digits(state, k, order);
                // The lower `order` digits (d[1..=order]) identify
                // (S_{t-1}, ..., S_{t-order}); these are the leading digits of
                // the previous joint state, whose last digit ranges over
                // S_{t-1-order}. Sum over that trailing digit.
                let mut terms = Vec::with_capacity(k);
                for last in 0..k {
                    // Previous joint state digits: (S_{t-1}, ..., S_{t-order}, last)
                    // = (d[1], ..., d[order], last).
                    let mut prev_idx = 0usize;
                    for pos in 1..=order {
                        prev_idx = prev_idx * k + d[pos];
                    }
                    prev_idx = prev_idx * k + last;
                    terms.push(prev_filtered_log[prev_idx]);
                }
                let marg = logsumexp(&terms);
                // d[0] is S_t and d[1] is S_{t-1} (valid because order >= 1 here).
                predicted_log[state] = log_p[[d[0], d[1]]] + marg;
            }
        }

        // Multiply by conditional density and normalise.
        let mut joint_log = vec![0.0f64; ns];
        for state in 0..ns {
            joint_log[state] = predicted_log[state] + cond_loglik[[state, t]];
        }
        let lik_log = logsumexp(&joint_log);
        llf += lik_log;

        for state in 0..ns {
            predicted_joint_log[[state, t]] = predicted_log[state];
            let f = joint_log[state] - lik_log;
            filtered_joint_log[[state, t]] = f;
        }

        // Filtered marginal Pr(S_t = i | Y_t): sum the joint over all but the
        // most-significant digit.
        for i in 0..k {
            let mut terms = Vec::with_capacity(ns / k);
            for state in 0..ns {
                if digits(state, k, order)[0] == i {
                    terms.push(filtered_joint_log[[state, t]]);
                }
            }
            filtered_marginal[[t, i]] = logsumexp(&terms).exp();
        }

        // Roll forward.
        for state in 0..ns {
            prev_filtered_log[state] = filtered_joint_log[[state, t]];
        }
    }

    FilterOutput {
        llf,
        filtered_marginal,
        predicted_joint_log,
        filtered_joint_log,
    }
}

/// Run the Kim smoother given the log-space filter output.
///
/// Returns smoothed marginal probabilities `Pr(S_t = i | Y_T)`, shaped
/// `(nobs, k)`. The recursion operates on the joint `(order + 1)`-tuple states.
pub(crate) fn kim_smoother(
    transition: &Array2<f64>,
    out: &FilterOutput,
    k: usize,
    order: usize,
) -> Array2<f64> {
    let ns = n_states(k, order);
    let nobs = out.filtered_joint_log.ncols();
    let log_p = transition.mapv(|v| v.max(1e-20).ln());

    let mut smoothed_joint_log = Array2::<f64>::zeros((ns, nobs));
    // Terminal: smoothed joint == filtered joint at t = T-1.
    for state in 0..ns {
        smoothed_joint_log[[state, nobs - 1]] = out.filtered_joint_log[[state, nobs - 1]];
    }

    // Backward recursion. For the joint state at time t with digits
    // (S_t, ..., S_{t-order}) and the joint at t+1 with digits
    // (S_{t+1}, ..., S_{t+1-order}), they overlap on (S_t, ..., S_{t+1-order}).
    // Kim's smoother:
    //   smoothed(s_t..) = filtered(s_t..) * sum_{s_{t+1}}
    //       smoothed(s_{t+1}..) * P[s_{t+1}, s_t] / predicted(s_{t+1}..).
    for t in (0..nobs - 1).rev() {
        for state in 0..ns {
            let d = digits(state, k, order); // (S_t, ..., S_{t-order})
            let mut terms = Vec::with_capacity(k);
            for s_next in 0..k {
                // Next joint state digits: (s_next, S_t, ..., S_{t-order+1})
                // = (s_next, d[0], ..., d[order-1]).
                let mut next_idx = s_next;
                for pos in 0..order {
                    next_idx = next_idx * k + d[pos];
                }
                let trans = log_p[[s_next, d[0]]];
                let num = smoothed_joint_log[[next_idx, t + 1]] + trans;
                let den = out.predicted_joint_log[[next_idx, t + 1]];
                terms.push(num - den);
            }
            let ratio = logsumexp(&terms);
            smoothed_joint_log[[state, t]] = out.filtered_joint_log[[state, t]] + ratio;
        }
    }

    // Marginalise to Pr(S_t = i | Y_T).
    let mut smoothed_marginal = Array2::<f64>::zeros((nobs, k));
    for t in 0..nobs {
        for i in 0..k {
            let mut terms = Vec::with_capacity(ns / k);
            for state in 0..ns {
                if digits(state, k, order)[0] == i {
                    terms.push(smoothed_joint_log[[state, t]]);
                }
            }
            smoothed_marginal[[t, i]] = logsumexp(&terms).exp();
        }
    }
    smoothed_marginal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::switching::{steady_state, transition_matrix};
    use ndarray::array;

    /// A direct order-0 Hamilton filter (in linear space) for cross-checking.
    fn naive_filter(init: &Array1<f64>, p: &Array2<f64>, dens: &Array2<f64>) -> (f64, Array2<f64>) {
        let k = init.len();
        let nobs = dens.ncols();
        let mut filt = Array2::<f64>::zeros((nobs, k));
        let mut prev = init.clone();
        let mut llf = 0.0;
        for t in 0..nobs {
            // predicted = P prev
            let pred = p.dot(&prev);
            let mut joint = vec![0.0; k];
            let mut lik = 0.0;
            for i in 0..k {
                joint[i] = pred[i] * dens[[i, t]];
                lik += joint[i];
            }
            llf += lik.ln();
            for i in 0..k {
                filt[[t, i]] = joint[i] / lik;
                prev[i] = filt[[t, i]];
            }
        }
        (llf, filt)
    }

    #[test]
    fn order0_filter_matches_naive() {
        let p = transition_matrix(&[0.9, 0.2], 2); // [p[0->0], p[1->0]]
        let init = steady_state(&p).unwrap();
        // Conditional densities (linear) for a few observations.
        let dens = array![[0.30, 0.10, 0.25, 0.40], [0.05, 0.35, 0.20, 0.02]];
        let cll = dens.mapv(f64::ln);
        let out = hamilton_filter(&init, &p, &cll, 0);
        let (llf_naive, filt_naive) = naive_filter(&init, &p, &dens);
        assert!(
            (out.llf - llf_naive).abs() < 1e-12,
            "{} vs {}",
            out.llf,
            llf_naive
        );
        for t in 0..4 {
            for i in 0..2 {
                assert!((out.filtered_marginal[[t, i]] - filt_naive[[t, i]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn filtered_and_smoothed_are_distributions() {
        let p = transition_matrix(&[0.85, 0.15], 2);
        let init = steady_state(&p).unwrap();
        let dens = array![[0.4, 0.1, 0.3], [0.1, 0.5, 0.2]];
        let cll = dens.mapv(f64::ln);
        let out = hamilton_filter(&init, &p, &cll, 0);
        let smoothed = kim_smoother(&p, &out, 2, 0);
        for t in 0..3 {
            let fs: f64 = (0..2).map(|i| out.filtered_marginal[[t, i]]).sum();
            let ss: f64 = (0..2).map(|i| smoothed[[t, i]]).sum();
            assert!((fs - 1.0).abs() < 1e-12, "filtered sum {fs}");
            assert!((ss - 1.0).abs() < 1e-12, "smoothed sum {ss}");
        }
        // The last smoothed equals the last filtered.
        for i in 0..2 {
            assert!((smoothed[[2, i]] - out.filtered_marginal[[2, i]]).abs() < 1e-12);
        }
    }

    #[test]
    fn order1_filter_sums_to_one() {
        let p = transition_matrix(&[0.8, 0.3], 2);
        let init = steady_state(&p).unwrap();
        // 4 joint states (S_t, S_{t-1}) over 3 observations.
        let cll = Array2::from_shape_vec(
            (4, 3),
            vec![
                -1.0, -1.2, -0.9, -2.0, -1.5, -1.1, -1.3, -1.0, -1.4, -0.8, -1.1, -1.6,
            ],
        )
        .unwrap();
        let out = hamilton_filter(&init, &p, &cll, 1);
        for t in 0..3 {
            let s: f64 = (0..2).map(|i| out.filtered_marginal[[t, i]]).sum();
            assert!((s - 1.0).abs() < 1e-12, "order-1 filtered sum {s}");
        }
        let smoothed = kim_smoother(&p, &out, 2, 1);
        for t in 0..3 {
            let s: f64 = (0..2).map(|i| smoothed[[t, i]]).sum();
            assert!((s - 1.0).abs() < 1e-12, "order-1 smoothed sum {s}");
        }
    }
}
