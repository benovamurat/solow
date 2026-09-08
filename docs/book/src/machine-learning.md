# Machine learning

Solow ships a full machine-learning surface across a set of focused crates. Every estimator has the standard `fit` / `transform` / `predict` shape, every stochastic estimator is deterministic under a caller seed via the portable MMIX-LCG PRNG, and every crate lives under `#![forbid(unsafe_code)]`.

The umbrella `solow::prelude` re-exports every estimator, so a canonical `use solow::prelude::*;` gives you the full toolbox.

## Where to go next

| I need to | Read the chapter |
|---|---|
| Scale, encode, or discretize features | [Preprocessing](./preprocessing.md) |
| Cluster unlabeled data | [Clustering](./cluster.md) |
| Compute nearest-neighbor predictions or densities | [Nearest neighbors](./neighbors.md) |
| Fit a decision tree | [Decision trees](./tree.md) |
| Combine many trees or classifiers | [Ensemble methods](./ensemble.md) |
| Train an SVM classifier or regressor | [Support vector machines](./svm.md) |
| Train a multi-layer perceptron | [Neural networks](./neural.md) |
| Classify with naive-Bayes assumptions | [Naive Bayes](./naive-bayes.md) |
| Use Fisher LDA or QDA | [Discriminant analysis](./discriminant.md) |
| Pick a subset of features | [Feature selection](./feature-selection.md) |
| Chain transformers and tune hyperparameters | [Pipelines & hyperparameter search](./pipeline.md) |
| Calibrate a classifier's probabilities | [Probability calibration](./calibration.md) |
| Train on partially-labeled data | [Semi-supervised learning](./semi-supervised.md) |
| Fit one classifier per class or per output | [Multi-class & multi-output](./multi.md) |
| Vectorize a corpus | [Text feature extraction](./text.md) |
| Generate or load a benchmark dataset | [Datasets and toy loaders](./datasets.md) |
| Reduce dimensionality (PCA, NMF, ICA) | [Matrix decomposition](./decomposition.md) |
| Learn a non-linear manifold (t-SNE, Isomap) | [Manifold learning](./manifold.md) |
| Approximate a kernel with random features | [Kernel approximation](./kernel-approx.md) |

## One-shot pipeline

```rust
use solow::prelude::*;

let (x, y) = load_breast_cancer();
let pipe = Pipeline::new()
    .step("scale", Box::new(StandardScaler::new()))
    .step("clf",   Box::new(LogisticRegressionCV::new().max_iter(200)));

let cv = StratifiedKFold::new(5)?.shuffle(true).seed(42);
let scores = cross_val_score(&pipe, &x, &y, &cv, |m, xt, yt| {
    Ok(accuracy_score(yt, m.predict(xt)?, None)?)
})?;

println!("accuracy: {:.4} +/- {:.4}", scores.mean, scores.std);
# Ok::<_, solow_core::Error>(())
```

## References

- Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*. SODA '07, 1027-1035.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD-96, 226-231.
- Breiman, L. (2001). *Random forests*. Machine Learning 45, 5-32.
- Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics 29(5), 1189-1232.
- Shalev-Shwartz, S., Singer, Y., Srebro, N., Cotter, A. (2007). *Pegasos: Primal estimated sub-gradient solver for SVM*. ICML.
