# solow-gp

**Gaussian processes for Rust.** GP regression with exact posterior, GP classification via the Laplace approximation, and a full kernel algebra (RBF, Matern, RationalQuadratic, DotProduct, Constant, White, Sum, Product, Exponentiation).

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-gp = "0.7"
```

## Quick example

```rust
use solow_gp::GaussianProcessRegressor;
use solow_gp::kernel::{Rbf, WhiteKernel};

let kernel = Rbf::new(1.0) + WhiteKernel::new(1e-3);
let gp = GaussianProcessRegressor::new(kernel).fit(&x, &y)?;
let (mean, std) = gp.predict_with_std(&x_test)?;
```

## What is inside

### Estimators

| Estimator | What it does |
|---|---|
| `GaussianProcessRegressor` | Exact GP regression with hyperparameter optimization by log-marginal-likelihood maximization. Returns mean and posterior standard deviation. |
| `GaussianProcessClassifier` | Binary and multi-class GP classification via the Laplace approximation. |

### Kernel algebra

| Kernel | What it models |
|---|---|
| `Rbf` (squared-exponential) | Smooth infinitely-differentiable functions. |
| `Matern` | Roughness controlled by the smoothness parameter ν. |
| `RationalQuadratic` | Scale-mixture of RBFs. |
| `DotProduct` | Bayesian linear regression. |
| `ConstantKernel` | Scale multiplier. |
| `WhiteKernel` | Additive iid noise. |
| `Sum`, `Product`, `Exponentiation` | Compositional kernel building blocks. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Cholesky-based exact posterior with jitter for numerical stability.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
