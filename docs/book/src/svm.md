# Support vector machines

`solow-svm` ships linear and kernel SVMs for classification and regression, ν-SVM variants, and one-class SVM for novelty detection.

## Linear SVMs (Pegasos-style)

| Estimator | What it does |
|---|---|
| `LinearSvc` | Binary and one-vs-rest multiclass linear SVC (Shalev-Shwartz-Singer-Srebro 2007) with deterministic MMIX-LCG shuffle. |
| `LinearSvr` | ε-insensitive Vapnik linear regressor with the same stochastic solver. |

## Kernel SVMs (SMO)

| Estimator | What it does |
|---|---|
| `Svc` | Kernel SVC with linear, RBF, polynomial, sigmoid kernels via sequential minimal optimization. Platt scaling for calibrated probabilities. |
| `Svr` | Kernel ε-SVR for regression. |
| `NuSvc` | ν-SVM variant parametrizing the fraction of support vectors directly. |
| `NuSvr` | ν-SVR regression variant. |

## Novelty detection

| Estimator | What it does |
|---|---|
| `OneClassSvm` | Boundary estimation in feature space for one-class classification and novelty detection. |

## Example

```rust
use solow_svm::{Svc, KernelKind};

let svc = Svc::builder()
    .kernel(KernelKind::Rbf { gamma: 0.5 })
    .c(1.0)
    .seed(42)
    .fit(&x, &y)?;

let pred = svc.predict(&x_test)?;
let proba = svc.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2007). *Pegasos: Primal estimated sub-gradient solver for SVM*. ICML.
- Platt, J. C. (1998). *Sequential minimal optimization: A fast algorithm for training support vector machines*. Microsoft Research MSR-TR-98-14.
