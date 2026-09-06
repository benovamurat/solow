# solow-cluster

**Unsupervised clustering for Rust.** k-means variants, density clusterers, hierarchical clustering, spectral clustering, Gaussian mixtures, in one memory-safe pure-Rust crate. Every stochastic clusterer is deterministic under a caller seed.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-cluster = "0.7"
```

## Quick example

```rust
use solow_cluster::{KMeans, KMeansInit};

let km = KMeans::new(3)
    .init(KMeansInit::KMeansPP)
    .n_init(10)
    .max_iter(300)
    .seed(42)
    .fit(&x)?;

let labels = km.predict(&x)?;
```

## What is inside

### k-means and variants

| Estimator | What it does |
|---|---|
| `KMeans` | Lloyd's k-means with k-means++ initialisation (Arthur-Vassilvitskii 2007), `n_init` restarts, deterministic MMIX-LCG seeding. |
| `MiniBatchKMeans` | Online mini-batch variant for large data, with configurable batch size and momentum. |
| `BisectingKMeans` | Hierarchical bisecting split, always producing a balanced binary tree of clusters. |

### Density-based

| Estimator | What it does |
|---|---|
| `Dbscan` | Ester-Kriegel-Sander-Xu 1996 with per-point `Core` / `Border` / `Noise` roles. |
| `Hdbscan` | Hierarchical density clustering with mutual reachability and condensed cluster tree. |
| `Optics` | Ordering points to identify the clustering structure, robust to varying densities. |

### Mode-seeking and graph-based

| Estimator | What it does |
|---|---|
| `MeanShift` | Kernel density mode-seeking with automatic bandwidth. |
| `AffinityPropagation` | Message-passing exemplar-based clustering with no preset k. |
| `SpectralClustering` | Graph-Laplacian eigenmap clustering with RBF, k-NN, or precomputed affinity. |

### Hierarchical

| Estimator | What it does |
|---|---|
| `AgglomerativeClustering` | Bottom-up merges with single, complete, average, or Ward linkage via Lance-Williams update recurrence. |
| `Birch` | Streaming clustering via the CF-tree summary structure, ideal for one-pass data. |

### Mixture models

| Estimator | What it does |
|---|---|
| `GaussianMixture` | EM-fit Gaussian mixture with full, tied, diagonal, or spherical covariance. |
| `BayesianGaussianMixture` | Variational Bayesian Gaussian mixture with automatic component selection. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- k-means results are deterministic under a caller seed (bit-identical fit across runs and platforms).
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
