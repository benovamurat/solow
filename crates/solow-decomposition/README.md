# solow-decomposition

**Matrix decomposition for machine learning.** PCA and variants (kernel, incremental, sparse), NMF, FastICA, truncated SVD (LSA), dictionary learning, latent Dirichlet allocation, and Gaussian / sparse random projections.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-decomposition = "0.7"
```

## Quick example

```rust
use solow_decomposition::{TruncatedSVD, Nmf};

let svd = TruncatedSVD::new(50).fit(&x)?;
let x_reduced = svd.transform(&x)?;

let nmf = Nmf::new(10).max_iter(300).seed(42).fit(&x)?;
```

## What is inside

### PCA family

| Estimator | What it does |
|---|---|
| `KernelPca` | Non-linear PCA in an RKHS via Gram-matrix centering. Linear, RBF, polynomial kernels. |
| `IncrementalPCA` | Streaming PCA for out-of-core data. |
| `SparsePCA` | PCA with L1 penalty on components for sparse loadings. |

### Independent and non-negative

| Estimator | What it does |
|---|---|
| `FastIca` | Hyvärinen 1999 symmetric decorrelation with logcosh or exp non-Gaussianity contrasts. |
| `Nmf` | Lee-Seung multiplicative updates under the Frobenius objective. |
| `MiniBatchNmf` | Online mini-batch NMF for large data. |

### Truncated SVD and dictionary learning

| Estimator | What it does |
|---|---|
| `TruncatedSVD` | Top-k singular value decomposition, ideal for latent semantic analysis. |
| `DictionaryLearning` | Sparse coding via alternating dictionary and code updates. |
| `MiniBatchDictionaryLearning` | Online variant for streaming or large data. |

### Topic modeling

| Estimator | What it does |
|---|---|
| `LatentDirichletAllocation` | Batch variational Bayes for topic modeling. |

### Random projection

| Estimator | What it does |
|---|---|
| `GaussianRandomProjection` | Johnson-Lindenstrauss projection with a random dense Gaussian matrix. |
| `SparseRandomProjection` | Achlioptas sparse projection with `{-1, 0, +1}` entries. |
| `johnson_lindenstrauss_min_dim` | Analytic lower bound on the target dimension. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Stochastic estimators are deterministic under a caller seed.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
