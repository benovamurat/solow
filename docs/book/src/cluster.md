# Clustering

`solow-cluster` ships 13 unsupervised clusterers covering centroid, density, hierarchical, spectral, and mixture-model families. Every stochastic clusterer is deterministic under a caller seed.

## k-means and variants

| Estimator | What it does |
|---|---|
| `KMeans` | Lloyd's k-means with k-means++ initialisation (Arthur-Vassilvitskii 2007), `n_init` restarts, deterministic MMIX-LCG seeding. |
| `MiniBatchKMeans` | Online mini-batch variant for large data with configurable batch size and momentum. |
| `BisectingKMeans` | Hierarchical bisecting split producing a balanced binary tree of clusters. |

## Density-based

| Estimator | What it does |
|---|---|
| `Dbscan` | Ester-Kriegel-Sander-Xu (KDD 1996) with per-point `Core` / `Border` / `Noise` roles. |
| `Hdbscan` | Hierarchical density clustering with mutual reachability and condensed cluster tree. |
| `Optics` | Ordering points to identify the clustering structure, robust to varying densities. |

## Mode-seeking and graph-based

| Estimator | What it does |
|---|---|
| `MeanShift` | Kernel density mode-seeking with automatic bandwidth. |
| `AffinityPropagation` | Message-passing exemplar-based clustering with no preset k. |
| `SpectralClustering` | Graph-Laplacian eigenmap clustering with RBF, k-NN, or precomputed affinity. |

## Hierarchical

| Estimator | What it does |
|---|---|
| `AgglomerativeClustering` | Bottom-up merges with single, complete, average, or Ward linkage via Lance-Williams update recurrence. Reports the full dendrogram. |
| `Birch` | Streaming clustering via the CF-tree summary structure. |

## Mixture models

| Estimator | What it does |
|---|---|
| `GaussianMixture` | EM-fit Gaussian mixture with full, tied, diagonal, or spherical covariance. |
| `BayesianGaussianMixture` | Variational Bayesian Gaussian mixture with automatic component selection via a Dirichlet-process prior. |

## Example

```rust
use solow_cluster::{KMeans, KMeansInit};

let km = KMeans::new(3)
    .init(KMeansInit::KMeansPP)
    .n_init(10)
    .max_iter(300)
    .seed(42)
    .fit(&x)?;

let labels = km.predict(&x)?;
# Ok::<_, solow_core::Error>(())
```

## Correctness

Cross-verified against committed golden reference fixtures on every CI run. Fits are bit-for-bit reproducible under a caller seed.

## References

- Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*. SODA '07, 1027-1035.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD-96, 226-231.
