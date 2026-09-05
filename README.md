<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/benovamurat/solow/main/assets/solow-logo-dark.svg">
    <img alt="Solow" src="https://raw.githubusercontent.com/benovamurat/solow/main/assets/solow-logo.svg" width="320">
  </picture>
</p>

<p align="center">
  <strong>The comprehensive statistics and machine learning stack for Rust.</strong><br>
  <sub>Regression. Time series. Machine learning. Bayesian inference. Change-point detection. One workspace. Zero <code>unsafe</code>.</sub>
</p>

<p align="center">
  <a href="https://github.com/benovamurat/solow/actions/workflows/ci.yml"><img src="https://github.com/benovamurat/solow/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://benovamurat.github.io/solow/"><img src="https://img.shields.io/badge/docs-mdBook-success.svg" alt="Docs"></a>
  <a href="https://crates.io/crates/solow"><img src="https://img.shields.io/crates/v/solow.svg" alt="crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-1.80%2B-dea584.svg" alt="Rust 1.80+">
  <img src="https://img.shields.io/badge/unsafe-forbidden-success.svg" alt="unsafe forbidden">
  <img src="https://img.shields.io/badge/tests-1000%2B%20passing-success.svg" alt="tests">
  <img src="https://img.shields.io/badge/crates-54-informational.svg" alt="54 crates">
</p>

<p align="center">
  <a href="#quick-tour"><b>Quick tour</b></a> ·
  <a href="#the-catalogue"><b>Catalogue</b></a> ·
  <a href="#correctness"><b>Correctness</b></a> ·
  <a href="#deployment"><b>Deployment</b></a> ·
  <a href="https://benovamurat.github.io/solow/"><b>Documentation</b></a> ·
  <a href="https://benovamurat.github.io/solow/examples/index.html"><b>Examples</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/ols.svg" width="270">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/case_forecasting.svg" width="270">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/state_space.svg" width="270">
</p>
<p align="center"><sub>OLS with confidence bands, forecast with prediction intervals, and Kalman filter path. Every figure rendered by <code>solow-viz</code>, the built-in dependency-light SVG backend.</sub></p>

---

## Why Solow

<table>
<tr>
<td width="33%" valign="top">

### Memory safe

Every crate in the workspace lives under `#![forbid(unsafe_code)]`. There is no `unsafe` block anywhere in the library, not even in the pure-Rust linear algebra core.

</td>
<td width="33%" valign="top">

### Deterministic

Every stochastic estimator takes a caller seed and uses a portable MMIX-LCG PRNG. A fixed seed reproduces bit-identical fits across runs, platforms, and CI hosts.

</td>
<td width="33%" valign="top">

### Cross-verified

Every deterministic estimator is checked against committed golden reference fixtures on every CI run. Closed-form solvers agree bit-wise to `1e-10`.

</td>
</tr>
<tr>
<td valign="top">

### Pure Rust

No system LAPACK. No BLAS. No Python. No C beyond libc. The full numerical core (SVD, eigh, Cholesky, QR, LU, pseudoinverse, distributions, special functions) is Rust from the ground up.

</td>
<td valign="top">

### Single binary

Ships as a single self-contained executable. Deploys anywhere Rust runs, including WebAssembly, embedded targets, and containers with no runtime dependencies.

</td>
<td valign="top">

### Full coverage

Linear and generalized linear models, discrete choice, time series, state space, survival, mixed effects, Bayesian, clustering, ensembles, SVM, neural networks, kernel methods, dimensionality reduction, and change-point detection.

</td>
</tr>
</table>

<a id="quick-tour"></a>

## Quick tour

Every workflow below is a complete, runnable Rust program.

### 1. Regression with the classical inference table

```rust
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

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
Prob (F-statistic):           2.17e-40   Log-Likelihood:              -77.253
No. Observations:                   50   AIC:                           158.5
Df Residuals:                       48   BIC:                           162.3
Covariance Type:             nonrobust
==============================================================================
                   coef    std err         t     P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const            2.1421      0.323     6.640     0.000       1.493       2.791
x                0.4977      0.011    43.864     0.000       0.475       0.521
==============================================================================
```

### 2. A leak-safe ML pipeline with cross-validation

```rust
use solow::prelude::*;

let (x, y) = load_breast_cancer();
let pipe = Pipeline::new()
    .step("scale", Box::new(StandardScaler::new()))
    .step("clf", Box::new(LogisticRegressionCV::new().max_iter(200)));

let cv = StratifiedKFold::new(5)?.shuffle(true).seed(42);
let scores = cross_val_score(&pipe, &x, &y, &cv, |m, xt, yt| {
    Ok(accuracy_score(yt, m.predict(xt)?, None)?)
})?;
println!("accuracy: {:.4} +/- {:.4}", scores.mean, scores.std);
```

### 3. Forecasting with SARIMAX

```rust
use solow::prelude::*;

let m = Sarimax::builder(y)
    .order(1, 1, 1)
    .seasonal(1, 1, 1, 12)
    .exog(x)
    .fit()?;
let fc = m.forecast(24)?;
println!("{}", m.summary());
```

### 4. Change-point detection on a stream

```rust
use solow_tsa::{pelt, cusum};

let cp = pelt(&series, 3.0)?;
println!("change points at: {:?}", cp);
```

<a id="the-catalogue"></a>

## The catalogue

Every estimator listed below is a first-class Rust API with `fit` / `transform` / `predict` where applicable, and a first-class committed reference fixture for its numerical output.

### Linear models

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>LinearRegression</code></td><td>Ordinary least squares via QR/SVD. Never forms XᵀX.</td></tr>
<tr><td><code>Ridge</code>, <code>RidgeCV</code></td><td>L2-penalized regression, closed-form Cholesky. CV variant sweeps α with LOO.</td></tr>
<tr><td><code>Lasso</code>, <code>LassoCV</code></td><td>L1-penalized regression via coordinate descent. CV variant follows the full regularization path.</td></tr>
<tr><td><code>ElasticNet</code>, <code>ElasticNetCV</code></td><td>Convex mix of L1 and L2 penalties, with cross-validated α and l1 ratio.</td></tr>
<tr><td><code>Lars</code>, <code>LassoLars</code>, <code>LassoLarsIC</code></td><td>Least-angle regression, LARS solving the Lasso path, and information-criterion model selection.</td></tr>
<tr><td><code>OrthogonalMatchingPursuit</code></td><td>Greedy k-sparse regression.</td></tr>
<tr><td><code>BayesianRidge</code>, <code>ARDRegression</code></td><td>Hierarchical Bayes with iterated α, λ updates. ARD gives per-feature precision.</td></tr>
<tr><td><code>MultiTaskLasso</code>, <code>MultiTaskElasticNet</code></td><td>Joint regression across correlated targets with group-sparse coefficients.</td></tr>
<tr><td><code>HuberRegressor</code>, <code>QuantileRegressor</code></td><td>Robust regression with the Huber loss, and quantile regression via linear programming.</td></tr>
<tr><td><code>RansacRegressor</code>, <code>TheilSenRegressor</code></td><td>Outlier-resistant regression by consensus and by median of slopes.</td></tr>
<tr><td><code>SgdRegressor</code>, <code>SgdClassifier</code>, <code>Perceptron</code>, <code>PassiveAggressive*</code></td><td>Online learners with L2, L1, and elastic-net penalties.</td></tr>
<tr><td><code>PoissonRegressor</code>, <code>GammaRegressor</code>, <code>TweedieRegressor</code></td><td>GLM families for count, positive continuous, and compound Poisson-Gamma targets.</td></tr>
<tr><td><code>RidgeClassifier</code>, <code>RidgeClassifierCV</code></td><td>One-hot regression classifier with ridge regularization.</td></tr>
<tr><td><code>LogisticRegression</code>, <code>LogisticRegressionCV</code></td><td>Newton-Raphson logistic regression with L2 or elastic-net, plus cross-validated regularization.</td></tr>
<tr><td><code>KernelRidge</code></td><td>Kernel ridge regression with RBF, polynomial, sigmoid, cosine kernels.</td></tr>
<tr><td><code>DummyRegressor</code>, <code>DummyClassifier</code></td><td>Baseline models for benchmarking.</td></tr>
</tbody>
</table>

### Support vector machines

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>Svc</code>, <code>Svr</code></td><td>SMO-based kernel SVM with linear, RBF, polynomial, sigmoid kernels for classification and regression.</td></tr>
<tr><td><code>NuSvc</code>, <code>NuSvr</code></td><td>ν-SVM variants that parametrize the fraction of support vectors directly.</td></tr>
<tr><td><code>LinearSvc</code>, <code>LinearSvr</code></td><td>Pegasos-style stochastic sub-gradient descent for large linear problems.</td></tr>
<tr><td><code>OneClassSvm</code></td><td>Unsupervised novelty detection via boundary estimation in feature space.</td></tr>
</tbody>
</table>

### Trees and ensembles

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>DecisionTreeClassifier</code>, <code>DecisionTreeRegressor</code></td><td>CART trees with Gini, entropy, MSE, MAE splitters and deterministic tie-breaking.</td></tr>
<tr><td><code>ExtraTreeClassifier</code>, <code>ExtraTreeRegressor</code></td><td>Extra-random trees with fully randomized split thresholds.</td></tr>
<tr><td><code>RandomForestClassifier</code>, <code>RandomForestRegressor</code></td><td>Breiman 2001 bagged forests with per-tree feature subsampling.</td></tr>
<tr><td><code>ExtraTreesClassifier</code>, <code>ExtraTreesRegressor</code></td><td>Fully-randomized forests, faster to fit than random forests.</td></tr>
<tr><td><code>GradientBoostingClassifier</code>, <code>GradientBoostingRegressor</code></td><td>Friedman 2001 stagewise least-squares boosting with shrinkage and subsampling.</td></tr>
<tr><td><code>HistGradientBoostingClassifier</code>, <code>HistGradientBoostingRegressor</code></td><td>Histogram-binned boosting with per-feature 256-bin quantization for fast fits.</td></tr>
<tr><td><code>BaggingClassifier</code>, <code>BaggingRegressor</code></td><td>Meta-ensembles over any base estimator with feature and sample bootstrap.</td></tr>
<tr><td><code>AdaBoostClassifier</code>, <code>AdaBoostRegressor</code></td><td>Freund-Schapire SAMME for classification, Drucker AdaBoost.R2 for regression.</td></tr>
<tr><td><code>VotingClassifier</code>, <code>VotingRegressor</code></td><td>Soft or hard voting across a set of heterogeneous base learners.</td></tr>
<tr><td><code>StackingClassifier</code>, <code>StackingRegressor</code></td><td>Two-level stacking with an out-of-fold meta-learner.</td></tr>
<tr><td><code>IsolationForest</code></td><td>Liu-Ting-Zhou 2008 unsupervised anomaly scoring with normalised path length.</td></tr>
</tbody>
</table>

### Clustering and mixture models

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>KMeans</code>, <code>MiniBatchKMeans</code>, <code>BisectingKMeans</code></td><td>Lloyd's k-means with k-means++ init, mini-batch variant for large data, and hierarchical bisecting split.</td></tr>
<tr><td><code>Dbscan</code>, <code>Hdbscan</code>, <code>Optics</code></td><td>Density-based clusterers for arbitrary-shape clusters without a preset k.</td></tr>
<tr><td><code>MeanShift</code></td><td>Kernel density mode-seeking with adaptive bandwidth.</td></tr>
<tr><td><code>AffinityPropagation</code></td><td>Message-passing exemplar-based clustering.</td></tr>
<tr><td><code>SpectralClustering</code></td><td>Graph-Laplacian eigenmap clustering with RBF, k-NN, or precomputed affinity.</td></tr>
<tr><td><code>AgglomerativeClustering</code></td><td>Hierarchical clustering with single, complete, average, or Ward linkage via Lance-Williams updates.</td></tr>
<tr><td><code>Birch</code></td><td>Streaming clustering with the CF-tree summary structure.</td></tr>
<tr><td><code>GaussianMixture</code>, <code>BayesianGaussianMixture</code></td><td>EM-fit Gaussian mixtures and Dirichlet-process variational Bayesian mixtures.</td></tr>
</tbody>
</table>

### Dimensionality reduction and manifold learning

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>Pca</code>, <code>KernelPca</code>, <code>IncrementalPCA</code>, <code>SparsePCA</code></td><td>Full, kernel-lifted, streaming, and sparsity-constrained PCA.</td></tr>
<tr><td><code>FastIca</code></td><td>Hyvärinen 1999 symmetric decorrelation with logcosh or exp non-Gaussianity contrasts.</td></tr>
<tr><td><code>Nmf</code>, <code>MiniBatchNmf</code></td><td>Lee-Seung multiplicative updates for non-negative matrix factorization.</td></tr>
<tr><td><code>TruncatedSVD</code></td><td>Top-k singular value decomposition, ideal for latent semantic analysis.</td></tr>
<tr><td><code>DictionaryLearning</code>, <code>MiniBatchDictionaryLearning</code></td><td>Sparse coding via alternating dictionary and code updates.</td></tr>
<tr><td><code>LatentDirichletAllocation</code></td><td>Batch variational Bayes for topic modeling.</td></tr>
<tr><td><code>GaussianRandomProjection</code>, <code>SparseRandomProjection</code></td><td>Johnson-Lindenstrauss projections for cheap distance-preserving dimensionality reduction.</td></tr>
<tr><td><code>Isomap</code></td><td>Tenenbaum 2000 geodesic MDS with a k-NN graph and Floyd-Warshall shortest paths.</td></tr>
<tr><td><code>LocallyLinearEmbedding</code></td><td>Roweis-Saul 2000 local-neighborhood reconstruction.</td></tr>
<tr><td><code>MDS</code>, <code>SpectralEmbedding</code></td><td>Metric multidimensional scaling and graph-Laplacian embedding.</td></tr>
<tr><td><code>Tsne</code></td><td>van der Maaten-Hinton 2008 with perplexity-based high-dim kernel and Student-t low-dim kernel.</td></tr>
</tbody>
</table>

### Neighbors, naive Bayes, discriminant

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>KdTree</code>, <code>BallTree</code></td><td>Efficient exact nearest-neighbor queries in low and moderate dimensions.</td></tr>
<tr><td><code>KNeighborsClassifier</code>, <code>KNeighborsRegressor</code></td><td>Uniform or distance-weighted k-NN prediction with probability output.</td></tr>
<tr><td><code>RadiusNeighborsClassifier</code>, <code>RadiusNeighborsRegressor</code></td><td>Fixed-radius neighborhood prediction.</td></tr>
<tr><td><code>NearestCentroid</code></td><td>Rocchio-style class-mean classifier.</td></tr>
<tr><td><code>LocalOutlierFactor</code></td><td>Breunig-Kriegel 2000 density-based outlier scoring.</td></tr>
<tr><td><code>KernelDensity</code></td><td>Multivariate kernel density estimation.</td></tr>
<tr><td><code>GaussianNB</code>, <code>MultinomialNB</code>, <code>BernoulliNB</code>, <code>ComplementNB</code>, <code>CategoricalNB</code></td><td>Naive Bayes for continuous, count, binary, imbalanced text, and categorical features. All expose log-probability output via log-sum-exp.</td></tr>
<tr><td><code>LinearDiscriminantAnalysis</code>, <code>QuadraticDiscriminantAnalysis</code></td><td>Fisher LDA and QDA with pooled or per-class covariance and Cholesky-with-regularization solves.</td></tr>
</tbody>
</table>

### Neural networks, kernel methods, Gaussian processes, calibration

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>MlpClassifier</code>, <code>MlpRegressor</code></td><td>Feed-forward MLPs with ReLU, tanh, logistic, or identity activations and SGD or Adam optimizers. Glorot init seeded by MMIX-LCG.</td></tr>
<tr><td><code>BernoulliRbm</code></td><td>Contrastive-divergence Bernoulli restricted Boltzmann machine.</td></tr>
<tr><td><code>RBFSampler</code>, <code>Nystroem</code></td><td>Random Fourier features and Nyström approximation of a kernel matrix.</td></tr>
<tr><td><code>AdditiveChi2Sampler</code>, <code>SkewedChi2Sampler</code>, <code>PolynomialCountSketch</code></td><td>Explicit kernel-feature maps for chi-squared, skewed chi-squared, and polynomial kernels.</td></tr>
<tr><td><code>GaussianProcessRegressor</code>, <code>GaussianProcessClassifier</code></td><td>GP regression and Laplace-approximation GP classification with a full kernel algebra (RBF, Matern, RationalQuadratic, DotProduct, Constant, White, Sum, Product, Exponentiation).</td></tr>
<tr><td><code>CalibratedClassifierCV</code></td><td>Post-hoc probability calibration via Platt sigmoid or isotonic regression.</td></tr>
</tbody>
</table>

### Preprocessing, text, imputation, feature selection

<details>
<summary><b>Show full list (48 preprocessors, encoders, transformers)</b></summary>

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>StandardScaler</code>, <code>MinMaxScaler</code>, <code>RobustScaler</code>, <code>MaxAbsScaler</code></td><td>Column-wise scalers. StandardScaler uses one-pass Welford variance.</td></tr>
<tr><td><code>Normalizer</code></td><td>Row-wise L1 or L2 unit normalization.</td></tr>
<tr><td><code>PowerTransformer</code>, <code>QuantileTransformer</code></td><td>Yeo-Johnson and Box-Cox power transforms, and empirical-CDF Gaussianization.</td></tr>
<tr><td><code>KBinsDiscretizer</code></td><td>Binning with uniform, quantile, or k-means strategies.</td></tr>
<tr><td><code>OneHotEncoder</code>, <code>OrdinalEncoder</code>, <code>LabelEncoder</code></td><td>Categorical encoders with drop-first, handle-unknown, and inverse-transform.</td></tr>
<tr><td><code>LabelBinarizer</code>, <code>MultiLabelBinarizer</code></td><td>Binary indicator arrays for single-label and multi-label targets.</td></tr>
<tr><td><code>Binarizer</code></td><td>Threshold-based feature binarization.</td></tr>
<tr><td><code>PolynomialFeatures</code>, <code>SplineTransformer</code></td><td>Graded-lex polynomial expansion and B-spline basis functions.</td></tr>
<tr><td><code>TargetEncoder</code></td><td>Smoothed target-mean encoding of categorical features.</td></tr>
<tr><td><code>FunctionTransformer</code></td><td>Wrap an arbitrary closure as a pipeline transformer.</td></tr>
<tr><td><code>CountVectorizer</code>, <code>TfidfVectorizer</code></td><td>Bag-of-words and TF-IDF text vectorizers with n-gram, min/max-df, and stop-word support.</td></tr>
<tr><td><code>HashingVectorizer</code>, <code>FeatureHasher</code>, <code>DictVectorizer</code></td><td>Feature-hashing vectorizers for streaming text and dict-of-features input.</td></tr>
<tr><td><code>SimpleImputer</code>, <code>KnnImputer</code>, <code>IterativeImputer</code></td><td>Constant / mean / median / most-frequent imputation, KNN imputation, and MICE with Rubin's combining rules.</td></tr>
<tr><td><code>SelectKBest</code>, <code>SelectPercentile</code>, <code>SelectFpr</code>, <code>SelectFdr</code>, <code>SelectFwe</code></td><td>Univariate feature selection with ANOVA F, regression F, or mutual information scoring.</td></tr>
<tr><td><code>VarianceThreshold</code></td><td>Drop near-constant features.</td></tr>
<tr><td><code>Rfe</code>, <code>SequentialFeatureSelector</code></td><td>Recursive feature elimination and forward/backward sequential selection around any ranker.</td></tr>
</tbody>
</table>

</details>

### Pipelines, model selection, resampling

<details>
<summary><b>Show full list (pipelines, hyperparameter search, cross-validation)</b></summary>

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>Pipeline</code>, <code>FeatureUnion</code></td><td>Sequential composition and parallel-concat of transformers/estimators.</td></tr>
<tr><td><code>ColumnTransformer</code></td><td>Apply different transformers to different column subsets in one pass.</td></tr>
<tr><td><code>TransformedTargetRegressor</code></td><td>Wrap a regressor with a target transform and inverse transform.</td></tr>
<tr><td><code>GridSearchCV</code>, <code>RandomizedSearchCV</code></td><td>Full-factorial and Latin hypercube-style hyperparameter search.</td></tr>
<tr><td><code>HalvingGridSearchCV</code>, <code>HalvingRandomSearchCV</code></td><td>Successive-halving search: cheap early filter, expensive final rounds.</td></tr>
<tr><td><code>KFold</code>, <code>StratifiedKFold</code>, <code>RepeatedKFold</code>, <code>RepeatedStratifiedKFold</code></td><td>Random and stratified K-fold with optional repetitions.</td></tr>
<tr><td><code>GroupKFold</code>, <code>StratifiedGroupKFold</code>, <code>GroupShuffleSplit</code></td><td>Group-aware splitters that keep every group in exactly one fold.</td></tr>
<tr><td><code>TimeSeriesSplit</code></td><td>Walk-forward validation with configurable test size, gap, and max train size.</td></tr>
<tr><td><code>PurgedKFold</code>, <code>CombinatorialPurgedKFold</code></td><td>López de Prado (2018) leakage-safe splitters for financial back-tests.</td></tr>
<tr><td><code>LeaveOneOut</code>, <code>LeavePOut</code>, <code>ShuffleSplit</code>, <code>StratifiedShuffleSplit</code></td><td>Exhaustive and random resampling splitters.</td></tr>
<tr><td><code>learning_curve</code>, <code>validation_curve</code></td><td>Training-size and hyperparameter sweeps for diagnosing bias / variance.</td></tr>
<tr><td><code>permutation_test_score</code></td><td>Non-parametric significance test for a model's cross-val score.</td></tr>
<tr><td><code>bootstrap_ci</code>, <code>*_block_bootstrap_indices</code></td><td>Percentile, basic, and BCa confidence intervals, plus moving / circular / stationary block bootstrap for time series.</td></tr>
</tbody>
</table>

</details>

### Metrics

<details>
<summary><b>Show full list (regression / classification / calibration / cluster / forecast)</b></summary>

<table>
<thead><tr><th>Metric</th><th>What it measures</th></tr></thead>
<tbody>
<tr><td>MSE, RMSE, MAE, R², explained variance</td><td>Standard regression losses with Neumaier compensated summation.</td></tr>
<tr><td>Median AE, Max error, MAPE, sMAPE, MSLE, RMSLE, pinball, D² absolute, D² Tweedie</td><td>Robust and asymmetric regression losses.</td></tr>
<tr><td>Mean Tweedie / Poisson / Gamma deviance</td><td>GLM goodness-of-fit scores for fitted Poisson or Gamma models.</td></tr>
<tr><td>accuracy, balanced accuracy, precision, recall, F-beta</td><td>Classification counts with binary, macro, micro, weighted averaging.</td></tr>
<tr><td>Matthews correlation, Cohen kappa, hinge loss, log loss</td><td>Multi-class agreement, hinge, and cross-entropy losses.</td></tr>
<tr><td>ROC-AUC (binary + OvR + Hand-Till OvO), average precision, top-k accuracy</td><td>Rank-based classification scores with tie-corrected mid-ranks.</td></tr>
<tr><td>Brier score, reliability curve, ECE, MCE, top-1 ECE, multiclass Brier, ranked probability score</td><td>Calibration diagnostics and multiclass proper scoring rules.</td></tr>
<tr><td>Focal loss (binary + multiclass), Huber loss, log-cosh loss</td><td>Robust classification and regression losses.</td></tr>
<tr><td>silhouette, ARI, AMI, NMI, homogeneity, completeness, v-measure, Fowlkes-Mallows, Calinski-Harabasz, Davies-Bouldin</td><td>Cluster quality with both external and internal criteria.</td></tr>
<tr><td>pairwise distances, rbf / linear / polynomial / sigmoid / laplacian / cosine / chi² kernels</td><td>Distance and kernel matrices with vectorized inner loops.</td></tr>
<tr><td>Diebold-Mariano (Harvey-Leybourne-Newbold), Giacomini-White</td><td>Forecast-accuracy comparison with HAC standard errors.</td></tr>
<tr><td>MASE, RMSSE, Winkler interval score</td><td>Forecast metrics for M4/M5-style horizon evaluation.</td></tr>
<tr><td>Friedman + Nemenyi + Wilcoxon signed-rank</td><td>Non-parametric model comparison across benchmarks.</td></tr>
<tr><td>WAIC, PSIS-LOO with Pareto-k diagnostic</td><td>Bayesian model comparison from a posterior log-likelihood matrix.</td></tr>
<tr><td>Cohen's d, Hedges' g, Glass's delta, eta², omega², Cliff's delta, Cramer's V</td><td>Effect-size measures for continuous, ordinal, and categorical data.</td></tr>
<tr><td>permutation importance, partial dependence, accumulated local effects</td><td>Model-agnostic interpretability tools.</td></tr>
<tr><td>Platt scaling, isotonic regression, temperature scaling</td><td>Post-hoc probability calibrators.</td></tr>
<tr><td>SplitConformal, JackknifePlus</td><td>Distribution-free prediction intervals under exchangeability.</td></tr>
<tr><td>classification_report</td><td>Per-class precision, recall, F1 in a printable table.</td></tr>
</tbody>
</table>

</details>

### Classical inference (time series, state space, panel, survival)

<details>
<summary><b>Show classical stack (TS, VAR, state space, mixed, GEE, GAM, survival, empirical likelihood, copulas)</b></summary>

<table>
<thead><tr><th>Estimator</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>AutoReg</code>, <code>ARMA</code></td><td>Autoregressive and ARMA models with automatic order selection by AIC / BIC.</td></tr>
<tr><td><code>Sarimax</code></td><td>Seasonal ARIMA with exogenous regressors via maximum likelihood.</td></tr>
<tr><td><code>StateSpace</code>, <code>MvStateSpace</code>, <code>UnobservedComponents</code>, <code>DynamicFactor</code></td><td>General linear state-space, structural time series, and dynamic factor models with Kalman filter and smoother.</td></tr>
<tr><td><code>Var</code>, <code>Vecm</code>, <code>Svar</code></td><td>Vector autoregression, VECM with Johansen cointegration test, and structural VAR with impulse responses.</td></tr>
<tr><td><code>MarkovRegression</code>, <code>MarkovAutoregression</code></td><td>Regime-switching regression with filtered and smoothed regime probabilities.</td></tr>
<tr><td><code>Holt</code>, <code>ExponentialSmoothing</code>, <code>SimpleExpSmoothing</code></td><td>Holt-Winters and simple exponential smoothing.</td></tr>
<tr><td><code>Garch11</code></td><td>Volatility model with iterated multi-step variance forecast.</td></tr>
<tr><td><code>STL</code>, <code>seasonal_decompose</code></td><td>Robust seasonal-trend decomposition.</td></tr>
<tr><td>HP, BK, CF filters</td><td>Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald time-series filters.</td></tr>
<tr><td><code>coint</code>, <code>adfuller</code>, <code>kpss</code>, <code>zivot_andrews</code>, <code>granger_causality</code></td><td>Unit-root, cointegration, structural break, and Granger causality tests.</td></tr>
<tr><td><code>pelt</code>, <code>cusum</code>, <code>binary_segmentation</code></td><td>Change-point detection (Killick-Fearnhead-Eckley 2012).</td></tr>
<tr><td><code>ewma</code>, two-sided <code>cusum</code></td><td>Control charts with signed alarm streams.</td></tr>
<tr><td><code>MixedLm</code></td><td>Linear mixed-effects models by REML with random intercepts and slopes.</td></tr>
<tr><td><code>Gee</code></td><td>Generalized estimating equations for correlated responses.</td></tr>
<tr><td><code>GLMGam</code></td><td>Generalized additive models with penalized B-splines and GCV smoothing.</td></tr>
<tr><td><code>PHReg</code>, <code>SurvfuncRight</code>, log-rank</td><td>Cox proportional hazards with Efron ties, Kaplan-Meier, Nelson-Aalen.</td></tr>
<tr><td><code>Rlm</code></td><td>Robust linear regression via M-estimation (Huber, Tukey biweight, Andrew, Hampel, Ramsay).</td></tr>
<tr><td><code>QuantReg</code>, <code>Glsar</code></td><td>Quantile regression by linear programming and GLS with AR(p) errors.</td></tr>
<tr><td>Bayesian mixed GLM (VB), empirical likelihood, copulas</td><td>Variational Bayes for hierarchical models, empirical-likelihood inference, and Gaussian / Student t / Clayton / Frank / Gumbel / Joe copulas.</td></tr>
<tr><td>Meta-analysis</td><td>Fixed-effect and DerSimonian-Laird random-effects with Cochran Q, I², τ².</td></tr>
<tr><td>Wald, F, LM tests, HC0-HC3, HAC, cluster covariances</td><td>Complete robust-inference battery for linear models.</td></tr>
<tr><td>Levene, Bartlett, Fligner, Shapiro-Wilk, Anderson-Darling, KS, Mann-Whitney U, Kruskal-Wallis, McNemar, chi-squared, Pearson, Spearman, Kendall</td><td>Full battery of parametric and non-parametric statistical tests.</td></tr>
</tbody>
</table>

</details>

<a id="correctness"></a>

## The correctness model

Correctness is the product. Solow ships a multi-layer verification stack.

<table>
<tr>
<td width="50%" valign="top">

**Reference fixtures.** Every deterministic estimator has a committed golden fixture generated from a well-known reference implementation. A Rust replay test re-checks each fixture on every CI run.
- Closed-form solvers agree bit-wise to `1e-10`.
- Iterative solvers agree on parameters to `1e-6` or predictions to `5e-2` where reference solvers themselves disagree at that scale.

</td>
<td width="50%" valign="top">

**NIST StRD certified cases.** Re-run on every CI. Worst-case certified relative error across the suite is `2.5e-10`. The ill-conditioned Longley design (cond ~10¹⁰) matches certified coefficients to `~1e-13` because the QR/SVD path never forms XᵀX.

</td>
</tr>
<tr>
<td valign="top">

**Zero unsafe.** `#![forbid(unsafe_code)]` on every crate. No `unsafe` blocks. Not even in the numerical core.

</td>
<td valign="top">

**Deterministic PRNG.** Every stochastic estimator uses a portable MMIX-LCG. A fixed seed produces bit-identical output across runs, platforms, and CI hosts.

</td>
</tr>
</table>

Run the full CI locally with `cargo test --workspace` (1000+ tests).

<a id="deployment"></a>

## Deployment

Solow compiles to a single self-contained binary. No Python runtime. No system LAPACK or BLAS. No C dependencies beyond libc.

| Target | Support |
|---|---|
| Native Linux, macOS, Windows | first-class |
| WebAssembly | full numerical stack runs unchanged in the browser |
| Embedded / no-std adjacent | pure-Rust core compiles anywhere Rust runs |
| Python | `solow-py` exposes the library via PyO3 |
| Polars | `solow-polars` fits models straight off a `DataFrame` |

## Getting started

The umbrella crate re-exports the full public surface.

```toml
[dependencies]
solow = "0.7"
```

For a leaner build, pick individual crates.

```toml
[dependencies]
solow-regression = "0.7"
solow-metrics    = "0.7"
solow-cv         = "0.7"
```

Then in Rust.

```rust
use solow::prelude::*;
```

The [documentation site](https://benovamurat.github.io/solow/) walks through every module with worked examples. The [gallery](https://benovamurat.github.io/solow/examples/index.html) has 20+ runnable end-to-end vignettes.

## The workspace

Solow ships **54 focused crates**. Foundations, classical models, machine learning, evaluation, presentation. See [`docs/book/src/crates.md`](docs/book/src/crates.md) for the full list with per-crate contents.

## Status

`0.7.x`, published on crates.io. Every subcrate at the same version, released together, verified together. Roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## About

Solow is designed and built by **Murat Ova** at **Stochastic Minds**.

- Company: [Stochastic Minds](https://stochasticminds.com)
- Writing: [Product Philosophy](https://productphilosophy.com)

## License

BSD-3-Clause. Copyright (c) 2026, Murat Ova (Stochastic Minds). See [`LICENSE`](LICENSE).
