# solow-manifold

**Manifold learning and non-linear dimensionality reduction for Rust.** Isomap, locally linear embedding, MDS, spectral embedding, and t-SNE.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-manifold = "0.7"
```

## Quick example

```rust
use solow_manifold::{Tsne, Isomap};

let z = Tsne::new(2).perplexity(30.0).seed(42).fit_transform(&x)?;
```

## What is inside

| Estimator | What it does |
|---|---|
| `Isomap` | Tenenbaum-de Silva-Langford 2000 geodesic MDS with a k-NN graph and Floyd-Warshall shortest paths. |
| `LocallyLinearEmbedding` | Roweis-Saul 2000 with regularised local Gram-matrix solves and a `(I − W)ᵀ(I − W)` eigen-embedding. |
| `MDS` | Metric multidimensional scaling via SMACOF or classical torgerson. |
| `SpectralEmbedding` | Laplacian eigenmap embedding with RBF or k-NN affinity. |
| `Tsne` | van der Maaten-Hinton 2008 with perplexity-based Gaussian high-dim kernel and Student-t low-dim kernel. Full O(n²) exact method, deterministic under a caller seed. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Stochastic estimators deterministic under a caller seed.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
