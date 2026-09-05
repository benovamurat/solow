//! Feed-forward MLP regressor and classifier.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

fn uniform_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn uniform_symmetric(state: &mut u64, scale: f64) -> f64 {
    (uniform_f64(state) - 0.5) * 2.0 * scale
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// Hidden-layer activation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    /// Identity `f(x) = x`.
    Identity,
    /// Logistic sigmoid.
    Logistic,
    /// Hyperbolic tangent.
    Tanh,
    /// Rectified linear unit.
    Relu,
}

impl Activation {
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Identity => x,
            Activation::Logistic => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
        }
    }

    fn derivative_from_output(&self, y: f64) -> f64 {
        match self {
            Activation::Identity => 1.0,
            Activation::Logistic => y * (1.0 - y),
            Activation::Tanh => 1.0 - y * y,
            Activation::Relu => {
                if y > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Optimiser choice.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Solver {
    /// Stochastic sub-gradient descent with optional momentum.
    Sgd {
        /// Nesterov / classical momentum coefficient (0 = plain SGD).
        momentum: f64,
    },
    /// Kingma-Ba (2015) adaptive moment estimation.
    Adam,
}

// ---------------------------------------------------------------------------
// Layer weights (arena stored)
// ---------------------------------------------------------------------------

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
struct Layer {
    /// Weights `(in, out)`.
    w: Array2<f64>,
    /// Biases `(out,)`.
    b: Array1<f64>,
    /// SGD velocity buffer.
    v_w: Array2<f64>,
    v_b: Array1<f64>,
    /// Adam second-moment buffer.
    m_w: Array2<f64>,
    m_b: Array1<f64>,
}

fn xavier_layer(in_dim: usize, out_dim: usize, state: &mut u64) -> Layer {
    // Glorot uniform init.
    let bound = (6.0 / (in_dim + out_dim) as f64).sqrt();
    let mut w = Array2::<f64>::zeros((in_dim, out_dim));
    for i in 0..in_dim {
        for j in 0..out_dim {
            w[[i, j]] = uniform_symmetric(state, bound);
        }
    }
    let b = Array1::<f64>::zeros(out_dim);
    Layer {
        w: w.clone(),
        v_w: Array2::<f64>::zeros((in_dim, out_dim)),
        m_w: Array2::<f64>::zeros((in_dim, out_dim)),
        b: b.clone(),
        v_b: Array1::<f64>::zeros(out_dim),
        m_b: Array1::<f64>::zeros(out_dim),
    }
}

// ---------------------------------------------------------------------------
// MlpRegressor
// ---------------------------------------------------------------------------

/// Regression MLP with identity output.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MlpRegressor {
    layers: Vec<Layer>,
    /// Hidden activation.
    pub activation: Activation,
    /// Number of epochs actually run.
    pub n_iter: usize,
    /// L² regularisation strength.
    pub alpha: f64,
}

impl MlpRegressor {
    /// Fit with default settings (`hidden = [16]`, `activation = ReLU`,
    /// `alpha = 1e-4`, `max_iter = 200`, `learning_rate = 1e-3`,
    /// `Solver::Adam`).
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, seed: u64) -> Result<Self> {
        Self::fit_with(
            x,
            y,
            &[16],
            Activation::Relu,
            Solver::Adam,
            0.001,
            1e-4,
            200,
            seed,
        )
    }

    /// Full-configuration fit.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        hidden: &[usize],
        activation: Activation,
        solver: Solver,
        lr: f64,
        alpha: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "MlpRegressor::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        let mut state = seed.wrapping_add(0xCAFE_BABE_1357_9BDF);
        let mut sizes: Vec<usize> = vec![x.ncols()];
        sizes.extend_from_slice(hidden);
        sizes.push(1);
        let mut layers: Vec<Layer> = (0..sizes.len() - 1)
            .map(|i| xavier_layer(sizes[i], sizes[i + 1], &mut state))
            .collect();
        train_epochs(
            &mut layers,
            x,
            &y_to_matrix(y),
            activation,
            true,
            solver,
            lr,
            alpha,
            max_iter,
            &mut state,
        );
        Ok(Self {
            layers,
            activation,
            n_iter: max_iter,
            alpha,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let out = forward_batch(&self.layers, x, self.activation, true);
        Ok(out.column(0).to_owned())
    }
}

// ---------------------------------------------------------------------------
// MlpClassifier
// ---------------------------------------------------------------------------

/// Classification MLP with softmax output and cross-entropy loss.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MlpClassifier {
    layers: Vec<Layer>,
    /// Hidden activation.
    pub activation: Activation,
    /// Distinct class count.
    pub n_classes: usize,
    /// Number of epochs actually run.
    pub n_iter: usize,
    /// L² regularisation strength.
    pub alpha: f64,
}

impl MlpClassifier {
    /// Fit with defaults (`hidden = [16]`, `Relu`, `Adam`, `α = 1e-4`,
    /// `lr = 1e-3`, `max_iter = 200`).
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>, seed: u64) -> Result<Self> {
        Self::fit_with(
            x,
            y,
            &[16],
            Activation::Relu,
            Solver::Adam,
            0.001,
            1e-4,
            200,
            seed,
        )
    }

    /// Full-configuration fit.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        hidden: &[usize],
        activation: Activation,
        solver: Solver,
        lr: f64,
        alpha: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 || x.nrows() != y.len() {
            return Err(Error::Shape(format!(
                "MlpClassifier::fit_with: shape mismatch (x: {}×{}, y: {})",
                x.nrows(),
                x.ncols(),
                y.len()
            )));
        }
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1).max(2);
        let mut state = seed.wrapping_add(0xBAD_C0FFEE_0DDF_00D);
        let mut sizes: Vec<usize> = vec![x.ncols()];
        sizes.extend_from_slice(hidden);
        sizes.push(n_classes);
        let mut layers: Vec<Layer> = (0..sizes.len() - 1)
            .map(|i| xavier_layer(sizes[i], sizes[i + 1], &mut state))
            .collect();
        // One-hot targets.
        let mut y_oh = Array2::<f64>::zeros((y.len(), n_classes));
        for i in 0..y.len() {
            y_oh[[i, y[i]]] = 1.0;
        }
        train_epochs(
            &mut layers,
            x,
            &y_oh,
            activation,
            false,
            solver,
            lr,
            alpha,
            max_iter,
            &mut state,
        );
        Ok(Self {
            layers,
            activation,
            n_classes,
            n_iter: max_iter,
            alpha,
        })
    }

    /// Predict class labels.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        let p = self.predict_proba(x)?;
        let mut out = Array1::<usize>::zeros(p.nrows());
        for i in 0..p.nrows() {
            let (mut best_c, mut best) = (0usize, f64::NEG_INFINITY);
            for c in 0..self.n_classes {
                if p[[i, c]] > best {
                    best = p[[i, c]];
                    best_c = c;
                }
            }
            out[i] = best_c;
        }
        Ok(out)
    }

    /// Predict per-class probabilities (row-softmax of the logits).
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let logits = forward_batch(&self.layers, x, self.activation, false);
        let mut out = Array2::<f64>::zeros(logits.dim());
        for i in 0..logits.nrows() {
            let m = logits
                .row(i)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut s = 0.0_f64;
            for c in 0..logits.ncols() {
                let e = (logits[[i, c]] - m).exp();
                out[[i, c]] = e;
                s += e;
            }
            for c in 0..logits.ncols() {
                out[[i, c]] /= s;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Shared forward / backward
// ---------------------------------------------------------------------------

fn y_to_matrix(y: ArrayView1<'_, f64>) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((y.len(), 1));
    for i in 0..y.len() {
        m[[i, 0]] = y[i];
    }
    m
}

fn forward_batch(
    layers: &[Layer],
    x: ArrayView2<'_, f64>,
    activation: Activation,
    identity_output: bool,
) -> Array2<f64> {
    let mut cur = x.to_owned();
    for (idx, layer) in layers.iter().enumerate() {
        let out_dim = layer.b.len();
        let mut next = Array2::<f64>::zeros((cur.nrows(), out_dim));
        for i in 0..cur.nrows() {
            for j in 0..out_dim {
                let mut s = layer.b[j];
                for k in 0..cur.ncols() {
                    s += cur[[i, k]] * layer.w[[k, j]];
                }
                if idx + 1 == layers.len() {
                    next[[i, j]] = if identity_output { s } else { s };
                } else {
                    next[[i, j]] = activation.apply(s);
                }
            }
        }
        cur = next;
    }
    cur
}

#[allow(clippy::too_many_arguments)]
fn train_epochs(
    layers: &mut [Layer],
    x: ArrayView2<'_, f64>,
    y: &Array2<f64>,
    activation: Activation,
    regression: bool,
    solver: Solver,
    lr: f64,
    alpha: f64,
    max_iter: usize,
    state: &mut u64,
) {
    let n = x.nrows();
    let mut t = 0usize;
    for _epoch in 0..max_iter {
        // Shuffle sample order.
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = uniform_index(state, (i + 1) as u64);
            order.swap(i, j);
        }
        for &i in &order {
            t += 1;
            // Forward
            let mut activations: Vec<Array1<f64>> = Vec::with_capacity(layers.len() + 1);
            activations.push(x.row(i).to_owned());
            for (l_idx, layer) in layers.iter().enumerate() {
                let in_dim = layer.w.nrows();
                let out_dim = layer.b.len();
                let prev = &activations[l_idx];
                let mut z = Array1::<f64>::zeros(out_dim);
                for j in 0..out_dim {
                    let mut s = layer.b[j];
                    for k in 0..in_dim {
                        s += prev[k] * layer.w[[k, j]];
                    }
                    z[j] = if l_idx + 1 == layers.len() {
                        // Output layer: identity for regression, logits for softmax.
                        s
                    } else {
                        activation.apply(s)
                    };
                }
                activations.push(z);
            }
            // Loss gradient at the output.
            let last = activations.last().unwrap();
            let mut delta = Array1::<f64>::zeros(last.len());
            if regression {
                for k in 0..last.len() {
                    delta[k] = last[k] - y[[i, k]];
                }
            } else {
                // Softmax + cross-entropy: gradient is softmax(logits) - y_onehot.
                let m = last.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut s = 0.0_f64;
                let mut soft = Array1::<f64>::zeros(last.len());
                for k in 0..last.len() {
                    let e = (last[k] - m).exp();
                    soft[k] = e;
                    s += e;
                }
                for k in 0..last.len() {
                    delta[k] = soft[k] / s - y[[i, k]];
                }
            }
            // Backpropagate.
            for l_idx in (0..layers.len()).rev() {
                let prev = &activations[l_idx];
                let cur_out = &activations[l_idx + 1];
                // Gradient wrt weights + bias.
                let in_dim = layers[l_idx].w.nrows();
                let out_dim = layers[l_idx].b.len();
                let mut grad_w = Array2::<f64>::zeros((in_dim, out_dim));
                let mut grad_b = Array1::<f64>::zeros(out_dim);
                for j in 0..out_dim {
                    grad_b[j] = delta[j];
                    for k in 0..in_dim {
                        grad_w[[k, j]] = prev[k] * delta[j];
                    }
                }
                // Add L2 regularisation on weights (not on bias).
                for j in 0..out_dim {
                    for k in 0..in_dim {
                        grad_w[[k, j]] += alpha * layers[l_idx].w[[k, j]];
                    }
                }
                // Update step.
                apply_update(&mut layers[l_idx], &grad_w, &grad_b, solver, lr, t);
                // Prepare delta for the previous layer.
                if l_idx > 0 {
                    let mut new_delta = Array1::<f64>::zeros(in_dim);
                    for k in 0..in_dim {
                        let mut s = 0.0_f64;
                        for j in 0..out_dim {
                            s += layers[l_idx].w[[k, j]] * delta[j];
                        }
                        // Times the derivative of the previous activation.
                        let prev_out = &activations[l_idx];
                        new_delta[k] = s * activation.derivative_from_output(prev_out[k]);
                    }
                    delta = new_delta;
                }
                let _ = cur_out;
            }
        }
    }
}

fn apply_update(
    layer: &mut Layer,
    grad_w: &Array2<f64>,
    grad_b: &Array1<f64>,
    solver: Solver,
    lr: f64,
    t: usize,
) {
    match solver {
        Solver::Sgd { momentum } => {
            for (v, g) in layer.v_w.iter_mut().zip(grad_w.iter()) {
                *v = momentum * *v + *g;
            }
            for (v, g) in layer.v_b.iter_mut().zip(grad_b.iter()) {
                *v = momentum * *v + *g;
            }
            for (w, v) in layer.w.iter_mut().zip(layer.v_w.iter()) {
                *w -= lr * *v;
            }
            for (b, v) in layer.b.iter_mut().zip(layer.v_b.iter()) {
                *b -= lr * *v;
            }
        }
        Solver::Adam => {
            const B1: f64 = 0.9;
            const B2: f64 = 0.999;
            const EPS: f64 = 1e-8;
            for (m, g) in layer.v_w.iter_mut().zip(grad_w.iter()) {
                *m = B1 * *m + (1.0 - B1) * g;
            }
            for (m, g) in layer.v_b.iter_mut().zip(grad_b.iter()) {
                *m = B1 * *m + (1.0 - B1) * g;
            }
            for (m2, g) in layer.m_w.iter_mut().zip(grad_w.iter()) {
                *m2 = B2 * *m2 + (1.0 - B2) * g * g;
            }
            for (m2, g) in layer.m_b.iter_mut().zip(grad_b.iter()) {
                *m2 = B2 * *m2 + (1.0 - B2) * g * g;
            }
            let bc1 = 1.0 - B1.powi(t as i32);
            let bc2 = 1.0 - B2.powi(t as i32);
            for j in 0..layer.w.ncols() {
                for k in 0..layer.w.nrows() {
                    let m_hat = layer.v_w[[k, j]] / bc1;
                    let v_hat = layer.m_w[[k, j]] / bc2;
                    layer.w[[k, j]] -= lr * m_hat / (v_hat.sqrt() + EPS);
                }
            }
            for j in 0..layer.b.len() {
                let m_hat = layer.v_b[j] / bc1;
                let v_hat = layer.m_b[j] / bc2;
                layer.b[j] -= lr * m_hat / (v_hat.sqrt() + EPS);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mlp_regressor_fits_a_smooth_target() {
        // y = sin(x) + 0.5 · cos(2x); small MLP + Adam should reach a low MSE.
        let n = 60usize;
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64 * 0.1).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v| v.sin() + 0.5 * (2.0 * v).cos());
        let mlp = MlpRegressor::fit_with(
            x.view(),
            y.view(),
            &[16, 16],
            Activation::Tanh,
            Solver::Adam,
            0.02,
            0.0,
            2000,
            7,
        )
        .unwrap();
        let pred = mlp.predict(x.view()).unwrap();
        let mse: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        // A small tanh MLP under Adam should easily beat the trivial constant
        // predictor (whose MSE is the sample variance).
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;
        assert!(
            mse < 0.5 * var_y,
            "MSE = {mse}, sample variance = {var_y} — MLP failed to learn the signal"
        );
    }

    #[test]
    fn mlp_classifier_reaches_perfect_accuracy_on_easy_data() {
        // Two well-separated blobs — a tiny MLP should fit them.
        let x = array![
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [0.05, 0.05],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
            [5.05, 5.05]
        ];
        let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let mlp = MlpClassifier::fit_with(
            x.view(),
            y.view(),
            &[8],
            Activation::Relu,
            Solver::Adam,
            0.05,
            0.0,
            500,
            42,
        )
        .unwrap();
        let p = mlp.predict(x.view()).unwrap();
        assert_eq!(p, y);
    }
}
