# Crate reference

Solow is a Cargo workspace of **57 focused crates**. Depend on the ones you
need, or pull in the umbrella `solow` crate to re-export the full public API.
This page lists the workspace.

## Foundation

| Crate | Purpose |
| --- | --- |
| `solow-core` | Error types, numeric aliases, shared data-handling tools |
| `solow-linalg` | Pure-Rust linear algebra (SVD, eigh, Cholesky, QR, LU, pinv) |
| `solow-distributions` | Special functions, distributions, ECDF, GEV, GPD |
| `solow-optimize` | Newton / BFGS optimizers and numerical differentiation |

## Classical statistical models

| Crate | Models |
| --- | --- |
| `solow-regression` | OLS, WLS, GLS, GLSAR, penalised (Ridge, Lasso, ElasticNet), CV variants, quantile, rolling/recursive, LARS, LassoLars, LassoLarsIC, OMP, HuberRegressor, BayesianRidge, ARDRegression, RANSACRegressor, TheilSenRegressor, MultiTaskLasso/ElasticNet, SGD (regressor + classifier), Perceptron, PassiveAggressive*, RidgeClassifier(CV), Dummy*, KernelRidge, sliced inverse regression, robust covariances |
| `solow-glm` | Generalized linear models (families, links, IRLS) + Tweedie, PoissonRegressor, GammaRegressor, TweedieRegressor |
| `solow-discrete` | Logit, Probit, Poisson, MNLogit, NegativeBinomial, GeneralizedPoisson, OrderedModel, ZeroInflatedPoisson, ConditionalLogit/Poisson, TruncatedLFPoisson, HurdleCountModel, LogisticRegressionCV |
| `solow-robust` | Robust linear models (M-estimation) |
| `solow-nonparametric` | Nonparametric smoothers (lowess, KDE, kernel regression) |
| `solow-multivariate` | PCA, factor analysis and rotation, MANOVA, canonical correlation |
| `solow-duration` | Survival analysis (Kaplan-Meier, Cox PH, log-rank) |
| `solow-mixed` | Linear mixed-effects models (REML) |
| `solow-gee` | Generalized estimating equations (incl. nominal/ordinal) |
| `solow-gam` | Generalized additive models (penalized splines) |
| `solow-impute` | MICE combine + SimpleImputer, KnnImputer, IterativeImputer |
| `solow-regime` | Regime-switching models (Markov switching) |
| `solow-othermod` | Beta regression |
| `solow-copula` | Copulas (Archimedean and elliptical) |
| `solow-bayes` | Bayesian mixed GLM (variational Bayes) |
| `solow-emplike` | Empirical-likelihood inference |

## Time series

| Crate | Contents |
| --- | --- |
| `solow-tsa` | acf/pacf, ADF, KPSS, coint, Granger, AutoReg, ARMA, STL, seasonal_decompose, Holt-Winters, HP/BK/CF filters, zivot_andrews, range_unit_root, breakvar heteroskedasticity, GARCH(1,1), change-point detection (CUSUM, PELT, Binary Segmentation), EWMA control chart |
| `solow-statespace` | Kalman filter/smoother, SARIMAX, unobserved components, dynamic factor |
| `solow-var` | VAR, SVAR, VECM / Johansen |

## Statistics, formulas, and presentation

| Crate | Purpose |
| --- | --- |
| `solow-stats` | ANOVA, Tukey HSD, ARCH, RESET, robust sandwich covariances (HC0-3, HAC, cluster), VIF, mediation, Oaxaca, power, proportion, rates, inter-rater (Cohen's / Fleiss' κ), influence, distance correlation, TOST equivalence, contingency, Pearson / Spearman / Kendall correlations, Mann-Whitney U, Kruskal-Wallis, McNemar, χ² contingency, Levene / Bartlett / Fligner variance tests, Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov, Wald-Wolfowitz runs, meta-analysis (fixed + DerSimonian-Laird random) |
| `solow-formula` | R/patsy-style formula interface (design matrices) |
| `solow-summary` | Labeled results and summary tables |
| `solow-graphics` | Statistical graphics (qqplot, plot_acf, influence) |
| `solow-viz` | General-purpose data-visualization backend (SVG); ROC / PR / reliability / residuals diagnostics |

## Model evaluation & resampling

| Crate | Purpose |
| --- | --- |
| `solow-metrics` | Regression / classification / forecast losses, calibration + post-hoc calibrators (Platt, isotonic, temperature), conformal (split, jackknife+), Diebold-Mariano and Giacomini-White, Friedman / Nemenyi / Wilcoxon, WAIC / PSIS-LOO, permutation importance, partial dependence, ALE, pairwise distances + kernels (RBF / linear / polynomial / sigmoid / laplacian / cosine / χ²), cluster evaluation (silhouette, ARI, MI variants, homogeneity / completeness / v-measure, Fowlkes-Mallows, Calinski-Harabasz, Davies-Bouldin), effect sizes (Cohen's d, Hedges' g, Glass's Δ, η², ω², Cliff's δ, Cramér's V), `classification_report` |
| `solow-cv` | K-fold, stratified, walk-forward, leave-one-out, leave-p-out, shuffle-split, group-aware, purged / combinatorial-purged K-fold; repeated K-fold / stratified K-fold; stratified & group shuffle split; `cross_val_score` scoring; `learning_curve`, `validation_curve`, `permutation_test_score`; bootstrap CIs (percentile, basic, BCa); moving / circular / stationary block bootstrap for time series; optional rayon-parallel fold evaluation |

## Machine learning

| Crate | Purpose |
| --- | --- |
| `solow-preprocessing` | 18 preprocessors: Standard/MinMax/MaxAbs/Robust scalers, Normalizer, LabelEncoder, OrdinalEncoder, OneHotEncoder, PolynomialFeatures, KBinsDiscretizer (uniform / quantile / KMeans), PowerTransformer (Yeo-Johnson / Box-Cox), QuantileTransformer, Binarizer, LabelBinarizer, MultiLabelBinarizer, FunctionTransformer, SplineTransformer, TargetEncoder |
| `solow-cluster` | 11 algorithms: KMeans (+K-means++), MiniBatchKMeans, BisectingKMeans, DBSCAN, HDBSCAN, OPTICS, MeanShift, AffinityPropagation, SpectralClustering, AgglomerativeClustering (single/complete/average/Ward), Birch + 2 mixtures: GaussianMixture, BayesianGaussianMixture |
| `solow-neighbors` | KdTree, BallTree, KNeighborsClassifier / KNeighborsRegressor (uniform / distance weights), RadiusNeighborsClassifier / RadiusNeighborsRegressor, NearestCentroid, LocalOutlierFactor, KernelDensity (multivariate) |
| `solow-tree` | DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor |
| `solow-ensemble` | RandomForest*, ExtraTrees*, GradientBoosting*, HistGradientBoosting*, Bagging*, AdaBoost (classifier + regressor), Voting (classifier + regressor), Stacking (classifier + regressor), IsolationForest |
| `solow-naive-bayes` | GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB |
| `solow-discriminant` | LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis |
| `solow-feature-selection` | SelectKBest, RFE, VarianceThreshold, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, SequentialFeatureSelector |
| `solow-pipeline` | Pipeline, FeatureUnion, ColumnTransformer, TransformedTargetRegressor, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV |
| `solow-svm` | LinearSVC, LinearSVR, SVC, SVR, NuSVC, NuSVR, OneClassSVM (kernels: linear / RBF / polynomial / sigmoid) |
| `solow-neural` | MLPClassifier, MLPRegressor (SGD + Adam), BernoulliRBM |
| `solow-manifold` | Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, t-SNE |
| `solow-decomposition` | Pca, KernelPca, FastIca, Nmf, MiniBatchNmf, TruncatedSVD, IncrementalPCA, SparsePCA, DictionaryLearning, MiniBatchDictionaryLearning, LatentDirichletAllocation, GaussianRandomProjection, SparseRandomProjection |
| `solow-text` | CountVectorizer, TfidfVectorizer, HashingVectorizer, FeatureHasher, DictVectorizer |
| `solow-covariance` | EmpiricalCovariance, ShrunkCovariance, LedoitWolf, Oas, MinCovDet, GraphicalLasso, EllipticEnvelope |
| `solow-cross-decomposition` | PLSRegression, PLSCanonical, PLSSVD, CCA |
| `solow-kernel-approx` | RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler, PolynomialCountSketch |
| `solow-gp` | GaussianProcessRegressor + GaussianProcessClassifier + kernel algebra (RBF, Matern, RationalQuadratic, DotProduct, Constant, White, Sum, Product, Exponentiation) |
| `solow-semi-supervised` | LabelPropagation, LabelSpreading, SelfTrainingClassifier |
| `solow-multi` | OneVsRest, OneVsOne, OutputCode, MultiOutputRegressor, MultiOutputClassifier, ClassifierChain, RegressorChain |
| `solow-calibration` | CalibratedClassifierCV (Platt sigmoid + isotonic) |
| `solow-datasets` | make_classification / regression / blobs / moons / circles / swiss_roll / low_rank_matrix, load_iris / wine / diabetes / breast_cancer, compute_class_weight, compute_sample_weight, resample helpers |

## Umbrella and tooling

| Crate | Purpose |
| --- | --- |
| `solow` | Umbrella crate re-exporting the full public API, with a `prelude` |
| `solow-py` | PyO3 bindings: import `solow` from Python |
| `solow-polars` | Polars `DataFrame` interop |
| `solow-bench` | `criterion` benchmark harness |

Every deterministic estimator is cross-verified against committed
golden reference fixtures. Closed-form solvers agree bit-wise to
`1e-10`. Iterative solvers agree on parameters to `1e-6` or on
predictions to `5e-2` where reference solvers themselves disagree at
that scale.
