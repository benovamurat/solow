# Matrix decomposition (PCA, NMF, ICA)

`solow-decomposition` covers PCA variants (kernel, incremental, sparse), NMF (batch and mini-batch), FastICA, truncated SVD (LSA), dictionary learning, latent Dirichlet allocation, and random projections.

## PCA family

| Estimator | What it does |
|---|---|
| `KernelPca` | Non-linear PCA in an RKHS via Gram-matrix centering. Linear, RBF, polynomial kernels. |
| `IncrementalPCA` | Streaming PCA for out-of-core data. |
| `SparsePCA` | PCA with L1 penalty on components for sparse loadings. |

For classical (full-covariance) PCA, factor analysis, and canonical correlation, see [Multivariate analysis](./multivariate.md).

## Independent and non-negative

| Estimator | What it does |
|---|---|
| `FastIca` | Hyvärinen (1999) symmetric decorrelation with `LogCosh` or `Exp` non-Gaussianity contrasts. |
| `Nmf` | Lee-Seung (2001) multiplicative updates under the Frobenius objective. |
| `MiniBatchNmf` | Online mini-batch NMF for large data. |

## Truncated SVD and dictionary learning

| Estimator | What it does |
|---|---|
| `TruncatedSVD` | Top-k singular value decomposition, ideal for latent semantic analysis. |
| `DictionaryLearning` | Sparse coding via alternating dictionary and code updates. |
| `MiniBatchDictionaryLearning` | Online variant for streaming or large data. |

## Topic modeling

| Estimator | What it does |
|---|---|
| `LatentDirichletAllocation` | Batch variational Bayes for topic modeling. |

## Random projection

| Estimator | What it does |
|---|---|
| `GaussianRandomProjection` | Johnson-Lindenstrauss projection with a random dense Gaussian matrix. |
| `SparseRandomProjection` | Achlioptas sparse projection with `{-1, 0, +1}` entries. |
| `johnson_lindenstrauss_min_dim` | Analytic lower bound on the target dimension. |

## Example

```rust
use solow_decomposition::{TruncatedSVD, Nmf};

let svd = TruncatedSVD::new(50).fit(&x)?;
let x_reduced = svd.transform(&x)?;

let nmf = Nmf::new(10).max_iter(300).seed(42).fit(&x)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Hyvärinen, A. (1999). *Fast and robust fixed-point algorithms for independent component analysis*. IEEE Transactions on Neural Networks.
- Lee, D. D., & Seung, H. S. (2001). *Algorithms for non-negative matrix factorization*. NIPS.
- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). *Nonlinear component analysis as a kernel eigenvalue problem*. Neural Computation.
