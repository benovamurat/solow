<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/benovamurat/solow/main/assets/solow-logo-dark.svg">
    <img alt="Solow" src="https://raw.githubusercontent.com/benovamurat/solow/main/assets/solow-logo.svg" width="320">
  </picture>
</p>

<p align="center">
  <strong>The comprehensive statistics and machine learning stack for Rust.</strong><br>
  57 focused crates. Memory safe. Pure Rust. Deterministic.
</p>

<p align="center">
  <a href="https://github.com/benovamurat/solow/actions/workflows/ci.yml"><img src="https://github.com/benovamurat/solow/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://benovamurat.github.io/solow/"><img src="https://img.shields.io/badge/docs-mdBook-success.svg" alt="Docs"></a>
  <a href="https://crates.io/crates/solow"><img src="https://img.shields.io/crates/v/solow.svg" alt="crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-1.80%2B-dea584.svg" alt="Rust 1.80+">
  <img src="https://img.shields.io/badge/unsafe-forbidden-success.svg" alt="unsafe forbidden">
  <img src="https://img.shields.io/badge/tests-1000%2B%20passing-success.svg" alt="tests">
</p>

<p align="center">
  <a href="https://benovamurat.github.io/solow/"><b>Documentation</b></a> ·
  <a href="https://benovamurat.github.io/solow/examples/index.html"><b>Examples</b></a> ·
  <a href="docs/VALIDATION.md"><b>Validation</b></a> ·
  <a href="docs/BENCHMARKS.md"><b>Benchmarks</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/ols.svg" width="270">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/case_forecasting.svg" width="270">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/state_space.svg" width="270">
</p>
<p align="center"><sub>Regression, forecasting with prediction bands, and the Kalman filter. Every figure is rendered by <code>solow-viz</code>, the built-in dependency-light SVG backend.</sub></p>

---

## What Solow gives you

A single Rust workspace covering the full ground of applied statistics and machine learning: linear and generalized linear models, discrete choice, robust regression, time series and state-space models, survival analysis, mixed effects, Bayesian inference, clustering, tree ensembles, support vector machines, neural networks, kernel methods, dimensionality reduction, and change-point detection.

Everything lives under `#![forbid(unsafe_code)]`. Every stochastic estimator is deterministic given a seed. Every classical inference number is verified against committed reference fixtures.

### Beyond the classical stack

Capabilities that few libraries expose as first-class modules.

- **Change-point detection.** CUSUM, PELT (Killick, Fearnhead, Eckley 2012), Binary Segmentation.
- **Volatility models.** `GARCH(1, 1)` with iterated multi-step variance forecast.
- **Extreme value analysis.** `GEV` and `GPD` distributions with `return_level(T)` and peaks-over-threshold fit.
- **Effect sizes.** Cohen's d, Hedges' g, Glass's delta, eta squared, omega squared, Cliff's delta, Cramer's V.
- **Meta-analysis.** Fixed-effect and DerSimonian-Laird random-effects with Cochran's Q, I squared, tau squared.
- **Control charts.** Two-sided CUSUM and EWMA (Roberts 1959) with signed alarm streams.
- **Block bootstrap.** Moving, circular, and stationary variants for time series.

## The correctness model

Correctness is the product. Solow ships a two-layer verification stack.

- **Cross-language reference-fixture pipeline.** A Python driver fits each estimator against a well-known reference implementation, saves the inputs and outputs to JSON, and a Rust replay test asserts numerical agreement inside each fixture directory. Closed-form estimators match bit-wise to `1e-10`. Iterative estimators match to `1e-6` on parameters, or `5e-2` on predictions where reference solvers themselves disagree at that scale.
- **NIST StRD certified cases** re-run on every CI. Worst-case certified relative error across the suite is `2.5e-10`. The ill-conditioned Longley design (cond ~10^10) matches certified coefficients to `~1e-13` because the QR/SVD path never forms `XᵀX`.
- **Committed golden fixtures** re-verify every classical inference number (coefficients, standard errors, t-statistics, R squared, F-statistics, information criteria) on the same design matrix, on every CI run.
- **`#![forbid(unsafe_code)]` across every crate.** No `unsafe`, ever.
- **Deterministic MMIX-LCG PRNG** in every stochastic estimator. A fixed seed reproduces exact results across runs and platforms.

Run the full CI locally with `cargo test --workspace` (1000+ tests).

## A taste

```rust
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

// `x`, `y` are ndarray columns. Here, 50 noisy points of y ≈ 2 + 0.5·x.
let design = add_constant(&x, true, HasConstant::Add)?;
let res = LinearModel::ols(y, design)?.fit()?;

println!("{}", res.summary(Some(&["const", "x"])));
```

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:                       y   R-squared:                     0.976
Model:                             OLS   Adj. R-squared:                0.975
Method:                  Least Squares   F-statistic:                    1924
Date:                 Thu, 18 Jun 2026   Prob (F-statistic):         2.17e-40
Time:                         13:16:20   Log-Likelihood:              -77.253
No. Observations:                   50   AIC:                           158.5
Df Residuals:                       48   BIC:                           162.3
Df Model:                            1
Covariance Type:             nonrobust
==============================================================================
                   coef    std err         t     P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const            2.1421      0.323     6.640     0.000       1.493       2.791
x                0.4977      0.011    43.864     0.000       0.475       0.521
==============================================================================
```

The [examples gallery](https://benovamurat.github.io/solow/examples/index.html) has 20+ runnable end-to-end vignettes.

## What is covered

### Linear models

`LinearRegression`, `Ridge`, `RidgeCV`, `Lasso`, `LassoCV`, `ElasticNet`, `ElasticNetCV`, `BayesianRidge`, `ARDRegression`, `Lars`, `LassoLars`, `LassoLarsIC`, `OrthogonalMatchingPursuit`, `MultiTaskLasso`, `MultiTaskElasticNet`, `HuberRegressor`, `QuantileRegressor`, `RansacRegressor`, `TheilSenRegressor`, `SgdRegressor`, `SgdClassifier`, `Perceptron`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`, `RidgeClassifier`, `RidgeClassifierCV`, `LogisticRegression` (`Logit`), `LogisticRegressionCV`, `DummyRegressor`, `DummyClassifier`, `KernelRidge`.

### Support vector machines

`Svc`, `Svr`, `NuSvc`, `NuSvr`, `LinearSvc`, `LinearSvr`, `OneClassSvm`.

### Trees and ensembles

`DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `BaggingClassifier`, `BaggingRegressor`, `AdaBoostClassifier`, `AdaBoostRegressor`, `VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor`, `IsolationForest`.

### Clustering and mixture models

`KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, `Dbscan`, `Hdbscan`, `Optics`, `MeanShift`, `AffinityPropagation`, `SpectralClustering`, `AgglomerativeClustering`, `Birch`, `GaussianMixture`, `BayesianGaussianMixture`.

### Dimensionality reduction and manifold learning

`Pca`, `KernelPca`, `FastIca`, `Nmf`, `MiniBatchNmf`, `TruncatedSVD`, `IncrementalPCA`, `SparsePCA`, `DictionaryLearning`, `MiniBatchDictionaryLearning`, `LatentDirichletAllocation`, `GaussianRandomProjection`, `SparseRandomProjection`, `Isomap`, `LocallyLinearEmbedding`, `MDS`, `SpectralEmbedding`, `Tsne`.

### Covariance and cross-decomposition

`EmpiricalCovariance`, `ShrunkCovariance`, `LedoitWolf`, `Oas`, `MinCovDet`, `GraphicalLasso`, `EllipticEnvelope`, `PLSRegression`, `PLSCanonical`, `PLSSVD`, `CCA`.

### Discriminant, multi-class, semi-supervised

`LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`, `OneVsRestClassifier`, `OneVsOneClassifier`, `OutputCodeClassifier`, `MultiOutputRegressor`, `MultiOutputClassifier`, `ClassifierChain`, `RegressorChain`, `SelfTrainingClassifier`, `LabelPropagation`, `LabelSpreading`.

### Naive Bayes and neighbors

`GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`, `CategoricalNB`, `KNeighborsClassifier`, `KNeighborsRegressor`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`, `KdTree`, `BallTree`, `NearestCentroid`, `LocalOutlierFactor`, `KernelDensity`.

### Neural networks, kernel approximation, Gaussian processes, calibration

`MlpClassifier`, `MlpRegressor`, `BernoulliRbm`, `RBFSampler`, `Nystroem`, `AdditiveChi2Sampler`, `SkewedChi2Sampler`, `PolynomialCountSketch`, `GaussianProcessRegressor`, `GaussianProcessClassifier`, `CalibratedClassifierCV`.

### Imputation and feature selection

`SimpleImputer`, `KnnImputer`, `IterativeImputer`, `SelectKBest`, `Rfe`, `VarianceThreshold`, `SelectPercentile`, `SelectFpr`, `SelectFdr`, `SelectFwe`, `SequentialFeatureSelector`.

### Pipelines, model selection, resampling

`Pipeline`, `FeatureUnion`, `ColumnTransformer`, `TransformedTargetRegressor`, `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`, `HalvingRandomSearchCV`, `KFold`, `LeaveOneOut`, `LeavePOut`, `ShuffleSplit`, `StratifiedKFold`, `StratifiedGroupKFold`, `GroupKFold`, `TimeSeriesSplit`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `StratifiedShuffleSplit`, `GroupShuffleSplit`, `learning_curve`, `validation_curve`, `permutation_test_score`, `bootstrap_ci`, `moving_block_bootstrap_indices`, `circular_block_bootstrap_indices`, `stationary_bootstrap_indices`.

### Datasets and utilities

`load_iris`, `load_wine`, `load_diabetes`, `load_breast_cancer`, `make_classification`, `make_regression`, `make_blobs`, `make_moons`, `make_circles`, `make_swiss_roll`, `make_low_rank_matrix`, `compute_class_weight`, `compute_sample_weight`.

### Text and preprocessing

`CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`, `FeatureHasher`, `DictVectorizer`, `Binarizer`, `LabelBinarizer`, `MultiLabelBinarizer`, `FunctionTransformer`, `SplineTransformer`, `TargetEncoder`, `PowerTransformer`, `QuantileTransformer`, `Normalizer`, `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `PolynomialFeatures`, `KBinsDiscretizer`.

### Metrics

`IsotonicRegression`, `pairwise_distances`, `rbf_kernel`, `linear_kernel`, `polynomial_kernel`, `sigmoid_kernel`, `laplacian_kernel`, `cosine_similarity`, `silhouette_score`, `adjusted_rand_score`, `normalized_mutual_info_score`, `homogeneity_score`, `completeness_score`, `v_measure_score`, `fowlkes_mallows_score`, `calinski_harabasz_score`, `davies_bouldin_score`, `classification_report`, `partial_dependence`, `permutation_importance`.

### Classical inference

Cointegration, Granger causality, VAR, VECM, Kalman filter and smoother, SARIMAX, Markov switching regression, generalized additive models, generalized estimating equations, mixed effects, empirical likelihood, copulas, and the classical robust HC and HAC covariance battery for ordinary least squares are all first-class Rust APIs.

```rust
use solow::prelude::*;

// SARIMAX with exogenous regressors, then print the classical summary.
let m = Sarimax::builder(y).order(1, 1, 1).seasonal(1, 1, 1, 12).exog(x).fit()?;
println!("{}", m.summary());
```

## Deployment

Solow compiles to a single self-contained binary. No Python runtime. No system LAPACK or BLAS. No C dependencies beyond libc. The numerical core (SVD, eigendecomposition, Cholesky, QR, LU, special functions, distributions) is pure Rust.

- **Polars.** `solow-polars` fits models straight off a `DataFrame`.
- **Python.** `solow-py` exposes the library through PyO3.
- **WASM.** The pure-Rust core compiles to WebAssembly. The classical inference battery runs unchanged in the browser.
- **Reproducibility.** Every stochastic estimator takes a deterministic seed and produces bit-identical output across platforms.

## Getting started

```toml
[dependencies]
solow = "0.7"
```

Or pick individual crates for a leaner build.

```toml
[dependencies]
solow-regression = "0.7"
solow-metrics    = "0.7"
solow-cv         = "0.7"
```

## What is inside

The workspace ships 57 focused crates. See [`docs/book/src/crates.md`](docs/book/src/crates.md) for the full list with per-crate contents.

## Status

`0.7.x`, published on crates.io. Roadmap and per-module state in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## About

Solow is designed and built by **Murat Ova** at **Stochastic Minds**. Design notes and essays live at **Product Philosophy**.

- Company: Stochastic Minds, https://stochasticminds.com
- Writing: Product Philosophy, https://productphilosophy.com

## License

BSD-3-Clause. Copyright (c) 2026, Murat Ova (Stochastic Minds). See [`LICENSE`](LICENSE).
