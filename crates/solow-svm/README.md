# solow-svm

**Support vector machines for Rust.** Linear and kernel SVMs for classification and regression, ν-SVM variants, one-class SVM for novelty detection, in one memory-safe pure-Rust crate.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-svm = "0.7"
```

## Quick example

```rust
use solow_svm::{Svc, KernelKind};

let svc = Svc::builder()
    .kernel(KernelKind::Rbf { gamma: 0.5 })
    .c(1.0)
    .seed(42)
    .fit(&x, &y)?;

let pred = svc.predict(&x_test)?;
let proba = svc.predict_proba(&x_test)?;
```

## What is inside

### Linear SVMs (Pegasos-style stochastic sub-gradient)

| Estimator | What it does |
|---|---|
| `LinearSvc` | Binary and one-vs-rest multiclass linear SVC (Shalev-Shwartz-Singer-Srebro 2007) with deterministic MMIX-LCG shuffle. |
| `LinearSvr` | ε-insensitive Vapnik linear regressor. |

### Kernel SVMs (SMO)

| Estimator | What it does |
|---|---|
| `Svc` | Kernel SVC with linear, RBF, polynomial, sigmoid kernels via sequential minimal optimization. Platt scaling for calibrated probabilities. |
| `Svr` | Kernel ε-SVR for regression. |
| `NuSvc` | ν-SVM variant that parametrizes the fraction of support vectors directly. |
| `NuSvr` | ν-SVR regression variant. |

### Novelty detection

| Estimator | What it does |
|---|---|
| `OneClassSvm` | Boundary estimation in feature space for one-class classification and novelty detection. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Linear SVMs are deterministic under a caller seed (bit-identical fit across runs and platforms).
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
