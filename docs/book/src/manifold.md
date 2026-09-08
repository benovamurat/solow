# Manifold learning

`solow-manifold` covers non-linear dimensionality reduction: Isomap, locally linear embedding, MDS, spectral embedding, and t-SNE.

## Estimators

| Estimator | What it does |
|---|---|
| `Isomap` | Tenenbaum-de Silva-Langford (2000) geodesic MDS with a k-NN graph and Floyd-Warshall shortest paths. |
| `LocallyLinearEmbedding` | Roweis-Saul (2000) with regularised local Gram-matrix solves and a `(I − W)ᵀ(I − W)` eigen-embedding. |
| `MDS` | Metric multidimensional scaling via SMACOF or classical Torgerson. |
| `SpectralEmbedding` | Laplacian eigenmap embedding with RBF or k-NN affinity. |
| `Tsne` | van der Maaten-Hinton (2008) with perplexity-based Gaussian high-dim kernel and Student-t low-dim kernel. Full O(n²) exact method, deterministic under a caller seed. |

## Example

```rust
use solow_manifold::Tsne;

let z = Tsne::new(2)
    .perplexity(30.0)
    .learning_rate(200.0)
    .seed(42)
    .fit_transform(&x)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). *A global geometric framework for nonlinear dimensionality reduction*. Science 290, 2319-2323.
- Roweis, S. T., & Saul, L. K. (2000). *Nonlinear dimensionality reduction by locally linear embedding*. Science 290, 2323-2326.
- van der Maaten, L., & Hinton, G. (2008). *Visualizing data using t-SNE*. Journal of Machine Learning Research 9, 2579-2605.
