# solow-neural

**Neural networks for Rust.** Small feed-forward MLPs for classification and regression, plus a Bernoulli restricted Boltzmann machine, in one memory-safe pure-Rust crate.

Every model is deterministic under a caller seed. Weight initialisation uses Glorot with an MMIX-LCG PRNG so a fixed seed reproduces the fit bit-for-bit across runs and platforms.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-neural = "0.7"
```

## Quick example

```rust
use solow_neural::{MlpClassifier, Activation, Solver};

let mlp = MlpClassifier::builder()
    .hidden_layer_sizes(vec![64, 32])
    .activation(Activation::ReLU)
    .solver(Solver::Adam)
    .max_iter(200)
    .seed(42)
    .fit(&x, &y)?;

let proba = mlp.predict_proba(&x_test)?;
```

## What is inside

| Estimator | What it does |
|---|---|
| `MlpClassifier` | Multi-layer perceptron with softmax output and cross-entropy loss. Any number of hidden layers with configurable widths. |
| `MlpRegressor` | Multi-layer perceptron with linear output and mean squared error loss. |
| `BernoulliRbm` | Bernoulli restricted Boltzmann machine trained with contrastive divergence. |

### Activations

- `Identity` (no non-linearity, for linear regression)
- `Logistic` (sigmoid)
- `Tanh`
- `ReLU`

### Optimizers

- `Sgd` with optional momentum, learning-rate schedule
- `Adam` (Kingma-Ba 2014) with configurable β₁, β₂, ε

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Every fit is deterministic under a caller seed via MMIX-LCG PRNG.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
