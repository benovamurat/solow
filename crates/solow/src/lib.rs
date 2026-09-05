//! # Solow
//!
//! The comprehensive statistics and machine learning stack for Rust.
//! 57 focused crates. Memory safe. Pure Rust. Deterministic.
//!
//! ## What is here
//!
//! Linear and generalized linear models, discrete choice, robust
//! regression, time series and state-space models, survival analysis,
//! mixed effects, Bayesian inference, clustering, tree ensembles,
//! support vector machines, neural networks, kernel methods,
//! dimensionality reduction, and change-point detection.
//!
//! This umbrella crate re-exports the full public API of the workspace,
//! so a consumer can depend on a single crate. Every subcrate is also
//! reachable at `solow::<module>::*`:
//! [`solow::regression`](regression), [`solow::glm`](glm),
//! [`solow::discrete`](discrete), [`solow::tsa`](tsa),
//! [`solow::cluster`](cluster), [`solow::ensemble`](ensemble),
//! [`solow::svm`](svm), [`solow::neural`](neural), and so on.
//!
//! Most users work through the [`prelude`], which brings the common
//! model types, results, and helpers into scope in one glob import.
//!
//! ## Quick start
//!
//! ```
//! use solow::prelude::*;
//! use ndarray::array;
//!
//! let y = array![2.5, 3.4, 4.7, 5.1, 6.5];
//! let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
//! let res = LinearModel::ols(y, x).unwrap().fit().unwrap();
//! assert!(res.rsquared > 0.98);
//! ```
//!
//! ## Beyond the classical stack
//!
//! Capabilities that few libraries expose as first-class modules:
//!
//! * change-point detection: CUSUM, PELT
//!   (Killick, Fearnhead, Eckley 2012), Binary Segmentation,
//! * `GARCH(1, 1)` with iterated multi-step variance forecast,
//! * extreme value analysis: `GEV`, `GPD` with `return_level(T)` and
//!   peaks-over-threshold fit,
//! * effect sizes: Cohen d, Hedges g, Glass delta, eta squared, omega
//!   squared, Cliff delta, Cramer V,
//! * meta-analysis: fixed-effect and DerSimonian-Laird random-effects
//!   with Cochran Q, I squared, tau squared,
//! * two-sided CUSUM and EWMA (Roberts 1959) control charts,
//! * moving, circular, and stationary block bootstrap for time series.
//!
//! ## Correctness
//!
//! Every deterministic estimator is cross-verified against committed
//! golden reference fixtures on every CI run. Closed-form solvers match
//! bit-wise to `1e-10`. Iterative solvers match parameters to `1e-6` or
//! predictions to `5e-2` where reference solvers themselves disagree at
//! that scale. Every crate lives under `#![forbid(unsafe_code)]`.

pub use solow_bayes as bayes;
pub use solow_calibration as calibration;
pub use solow_cluster as cluster;
pub use solow_datasets as datasets;
pub use solow_copula as copula;
pub use solow_core as core;
pub use solow_covariance as covariance;
pub use solow_cross_decomposition as cross_decomposition;
pub use solow_cv as cv;
pub use solow_decomposition as decomposition;
pub use solow_discrete as discrete;
pub use solow_discriminant as discriminant;
pub use solow_distributions as distributions;
pub use solow_duration as duration;
pub use solow_emplike as emplike;
pub use solow_ensemble as ensemble;
pub use solow_feature_selection as feature_selection;
pub use solow_fit as fit;
pub use solow_formula as formula;
pub use solow_gam as gam;
pub use solow_gee as gee;
pub use solow_glm as glm;
pub use solow_gp as gp;
pub use solow_graphics as graphics;
pub use solow_impute as impute;
pub use solow_kernel_approx as kernel_approx;
pub use solow_linalg as linalg;
pub use solow_manifold as manifold;
pub use solow_metrics as metrics;
pub use solow_mixed as mixed;
pub use solow_multi as multi;
pub use solow_multivariate as multivariate;
pub use solow_naive_bayes as naive_bayes;
pub use solow_neighbors as neighbors;
pub use solow_neural as neural;
pub use solow_nonparametric as nonparametric;
pub use solow_optimize as optimize;
pub use solow_othermod as othermod;
pub use solow_pipeline as pipeline;
pub use solow_semi_supervised as semi_supervised;
pub use solow_preprocessing as preprocessing;
pub use solow_regime as regime;
pub use solow_regression as regression;
pub use solow_robust as robust;
pub use solow_statespace as statespace;
pub use solow_stats as stats;
pub use solow_summary as summary;
pub use solow_svm as svm;
pub use solow_text as text;
pub use solow_tree as tree;
pub use solow_tsa as tsa;
pub use solow_var as var;
pub use solow_viz as viz;

pub use ndarray;

/// The recommended glob-import for day-to-day use.
///
/// Brings into scope:
///
/// * the shared error / result / numeric types from [`solow_core`];
/// * the ergonomic formula-driven fit helpers from [`solow_fit`]
///   (`ols`, `wls`, `gls`, `glm`, `logit`, `probit`, `poisson`);
/// * the workhorse estimator types, [`LinearModel`](solow_regression::LinearModel),
///   [`Glm`](solow_glm::Glm), [`Logit`](solow_discrete::Logit),
///   [`Probit`](solow_discrete::Probit), [`Poisson`](solow_discrete::Poisson)
///   and their result structs;
/// * the [`solow_glm`] families and links;
/// * the two most common design-matrix helpers,
///   [`add_constant`](solow_core::tools::add_constant) and
///   [`HasConstant`](solow_core::tools::HasConstant);
/// * the everyday model-evaluation metrics: `mean_squared_error`,
///   `root_mean_squared_error`, `mean_absolute_error`, `r2_score`,
///   `accuracy_score`, `roc_auc_score`, `log_loss`.
pub mod prelude {
    // ndarray essentials.
    pub use ndarray::{array, s, Array1, Array2, ArrayView1, ArrayView2, Axis};

    // Core error, numeric aliases, common tools.
    pub use solow_core::prelude::*;

    // Ergonomic formula-driven fit surface.
    pub use solow_fit::{glm, gls, logit, ols, poisson, probit, wls, DataFrame, NamedFit};

    // Workhorse estimators.
    pub use solow_discrete::{DiscreteResults, Logit, LogisticRegressionCV, Poisson, Probit};
    pub use solow_glm::{
        Family, GammaRegressor, Glm, GlmResults, Link, PoissonRegressor, TweedieRegressor,
    };
    pub use solow_regression::{LinearModel, LinearResults};

    // Formula compilation for the manual path.
    pub use solow_formula::{build, DesignOutput};

    // The most common metrics and the pairwise kernel and distance surface.
    pub use solow_metrics::{
        accumulated_local_effects, accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
        average_precision_score, binary_focal_loss, calinski_harabasz_score, chi2_kernel,
        classification_report, cliffs_delta, cohens_d, completeness_score, cosine_similarity,
        cramers_v,
        davies_bouldin_score, diebold_mariano, eta_squared, expected_calibration_error,
        fowlkes_mallows_score, friedman_test, giacomini_white_test, glass_delta, hedges_g,
        homogeneity_score, huber_loss, laplacian_kernel, linear_kernel, log_cosh_loss, log_loss,
        mean_absolute_error, mean_squared_error, multiclass_brier_score, multiclass_focal_loss,
        nemenyi_critical_difference, normalized_mutual_info_score, omega_squared,
        pairwise_distances, partial_dependence, permutation_importance, polynomial_kernel,
        psis_loo, r2_score, ranked_probability_score, rbf_kernel, roc_auc_score,
        root_mean_squared_error, sigmoid_kernel, silhouette_score, top_label_calibration_error,
        v_measure_score, waic, wilcoxon_signed_rank, AccumulatedLocalEffects, Average,
        BinStrategy, ClassificationReport, ClassificationRow, DmLoss, FeatureImportance,
        FriedmanResult, IsotonicRegression, JackknifePlus, MiAverage, MulticlassAuc,
        PairwiseMetric, PartialDependence, PlattScaling, PredictionInterval, PsisLooResult,
        RegressionReport, SplitConformal, TemperatureScaling, WaicResult, WilcoxonResult,
    };

    // Dataset generators and lightweight utility helpers
    // (class-weight and resampling helpers).
    pub use solow_datasets::{
        compute_class_weight, compute_sample_weight, load_breast_cancer, load_diabetes, load_iris,
        load_wine, make_blobs, make_circles, make_classification, make_low_rank_matrix,
        make_moons, make_regression, make_swiss_roll, resample_indices_no_replace,
        resample_indices_with_replace,
    };

    // Cross-validation, cross-val scoring, and bootstrap, including
    // repeated / leave-p-out splitters and learning / validation /
    // permutation-test curves.
    pub use solow_cv::{
        bootstrap_ci, circular_block_bootstrap_indices, cross_val_score, learning_curve,
        moving_block_bootstrap_indices, permutation_test_score,
        stationary_bootstrap_indices, validation_curve, BootstrapCi, BootstrapMethod,
        CombinatorialPurgedKFold, CrossValScores, GroupKFold, GroupShuffleSplit, KFold,
        LeaveOneOut, LeavePOut, PurgedKFold, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit,
        Split, Splitter, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit,
        TimeSeriesSplit,
    };

    // Time-series analysis, AR, ARMA, cointegration, unit-root, and filters.
    pub use solow_tsa::{
        binary_segmentation, coint, cusum, ewma, kpss, pelt, AutoReg, CusumAlarm, EwmaAlarm,
        EwmaResult, ExponentialSmoothing, Garch11, Holt, SimpleExpSmoothing,
    };

    // Multivariate analysis, PCA, factor analysis, MANOVA, canonical correlation.
    pub use solow_multivariate::{CanCorr, Factor, Manova, Pca};

    // Vector time-series, VAR, VECM, SVAR.
    pub use solow_var::{Svar, Var, Vecm};

    // State-space and structural time series.
    pub use solow_statespace::{
        DynamicFactor, MvStateSpace, Sarimax, StateSpace, UnobservedComponents,
    };

    // Nonparametric, LOWESS, KDE, kernel regression.
    pub use solow_nonparametric::{lowess, KdeUnivariate, KernelReg};

    // Regime-switching models (Markov).
    pub use solow_regime::{MarkovAutoregression, MarkovRegression, MarkovResults};

    // Quantile regression, GLSAR, penalised linear models, Bayesian ridge,
    // ARD, LARS / OMP, robust regressions, multi-task, and stochastic
    // gradient descent-style online learners.
    pub use solow_regression::{
        ARDRegression, BayesianRidge, DummyClassifier, DummyClassifierStrategy, DummyRegressor,
        DummyRegressorStrategy, ElasticNet, ElasticNetCV, Glsar, HuberRegressor,
        InformationCriterion, KernelRidge, Lars, Lasso, LassoCV, LassoLars, LassoLarsIC,
        MultiTaskElasticNet, MultiTaskLasso, OrthogonalMatchingPursuit,
        PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, QuantReg,
        RansacRegressor, Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV, RidgeKernel,
        SgdClassifier, SgdLoss, SgdPenalty, SgdRegressor, TheilSenRegressor,
    };

    // Missing-value imputation.
    pub use solow_impute::{
        IterativeImputer, KnnImputer, SimpleImputer, SimpleStrategy,
    };

    // Preprocessing (scalers, encoders, feature construction, power/quantile,
    // binarizer, label / multilabel binarizer, function transformer, spline, target encoder).
    pub use solow_preprocessing::{
        Binarizer, BinStrategy as PreprocBinStrategy, FunctionTransformer, KBinsDiscretizer,
        LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, MultiLabelBinarizer, NormKind,
        Normalizer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerMethod,
        PowerTransformer, QuantileOutput, QuantileTransformer, RobustScaler, SplineTransformer,
        StandardScaler, TargetEncoder,
    };

    // Unsupervised clustering.
    pub use solow_cluster::{
        AffinityPropagation, AgglomerativeClustering, Birch, BisectingKMeans, CovType, Dbscan,
        DbscanResult, DendrogramNode, GaussianMixture, Hdbscan, KMeans, KMeansInit, KMeansResult,
        Linkage, MeanShift, MiniBatchKMeans, Optics, PointRole, SpectralClustering,
    };

    // Nearest neighbours (KDTree + BallTree + classifiers/regressors + centroid
    // + LOF + multi-variate KernelDensity).
    pub use solow_neighbors::{
        BallTree, KNeighborsClassifier, KNeighborsRegressor, KdTree, KdeKernel, KernelDensity,
        LocalOutlierFactor, NearestCentroid, RadiusNeighborsClassifier,
        RadiusNeighborsRegressor, WeightKind,
    };

    // Decision trees (CART + Extra-Trees).
    pub use solow_tree::{
        ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor,
        ExtraTreeClassifier, ExtraTreeRegressor, RegressionCriterion, TreeParams,
    };

    // Ensembles.
    pub use solow_ensemble::{
        AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
        GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        IsolationForest, RandomForestClassifier, RandomForestRegressor, StackingClassifier,
        StackingRegressor, VotingClassifier, VotingMode, VotingRegressor,
    };

    // Naive Bayes (5 varyant).
    pub use solow_naive_bayes::{
        BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB,
    };

    // Discriminant analysis.
    pub use solow_discriminant::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};

    // Feature selection.
    pub use solow_feature_selection::{
        score_f_classif, score_f_regression, Rfe, SelectFdr, SelectFpr, SelectFwe, SelectKBest,
        SelectPercentile, SequentialFeatureSelector, SfsDirection, VarianceThreshold,
    };

    // Pipelines and hyperparameter search, full-grid, random,
    // successive-halving variants.
    pub use solow_pipeline::{
        ColumnTransformer, ColumnTransformerStep, FeatureUnion, FeatureUnionStep, GridSearchCV,
        HalvingConfig, HalvingGridSearchCV, HalvingRandomSearchCV, ParamGrid, Pipeline,
        RandomizedSearchCV, SearchResult, Step, TransformedTargetRegressor,
    };

    // Support vector machines (linear + kernel + ν-SVM + one-class).
    pub use solow_svm::{
        KernelKind as SvmKernelKind, LinearSvc, LinearSvr, NuSvc, NuSvr, OneClassSvm, Svc, Svr,
    };

    // Neural networks (small MLPs + BernoulliRBM).
    pub use solow_neural::{Activation, BernoulliRbm, MlpClassifier, MlpRegressor, Solver};

    // Manifold learning.
    pub use solow_manifold::{Isomap, LocallyLinearEmbedding, SpectralEmbedding, Tsne, MDS};

    // Decomposition, Kernel PCA / ICA / NMF / MiniBatchNMF / TruncatedSVD /
    // IncrementalPCA / SparsePCA / DictionaryLearning / MiniBatchDictionaryLearning /
    // LDA / Gaussian & Sparse random projections + Johnson-Lindenstrauss helper.
    pub use solow_decomposition::{
        johnson_lindenstrauss_min_dim, DictionaryLearning, FastIca, GaussianRandomProjection,
        IcaFun, IncrementalPCA, KernelKind, KernelPca, LatentDirichletAllocation,
        MiniBatchDictionaryLearning, MiniBatchNmf, Nmf, SparsePCA, SparseRandomProjection,
        TruncatedSVD,
    };

    // Covariance estimators, sample, shrinkage, MCD, graphical lasso, elliptic envelope.
    pub use solow_covariance::{
        EllipticEnvelope, EmpiricalCovariance, GraphicalLasso, LedoitWolf, MinCovDet, Oas,
        ShrunkCovariance,
    };

    // Calibration.
    pub use solow_calibration::{CalibratedClassifierCV, Method as CalibrationMethod};

    // Cross-decomposition, PLS / CCA.
    pub use solow_cross_decomposition::{PLSCanonical, PLSRegression, PLSSVD, CCA};

    // Kernel approximation.
    pub use solow_kernel_approx::{
        AdditiveChi2Sampler, Nystroem, PolynomialCountSketch, RBFSampler, SkewedChi2Sampler,
    };

    // Gaussian processes.
    pub use solow_gp::{GaussianProcessClassifier, GaussianProcessRegressor};

    // Semi-supervised.
    pub use solow_semi_supervised::{LabelPropagation, LabelSpreading, SelfTrainingClassifier};

    // Multi-class / multi-output.
    pub use solow_multi::{
        ClassifierChain, MultiOutputClassifier, MultiOutputRegressor, OneVsOneClassifier,
        OneVsRestClassifier, OutputCodeClassifier, RegressorChain,
    };

    // Text feature extraction.
    pub use solow_text::{
        CountVectorizer, DictVectorizer, FeatureHasher, HashingVectorizer, TfidfVectorizer,
    };

    // Duration / survival.
    pub use solow_duration::{PHReg, PHRegResults, SurvfuncRight};

    // Mixed-effects and GAM.
    pub use solow_mixed::MixedLm;

    // GEE.
    pub use solow_gee::{Gee, GeeResults};

    // Robust regression.
    pub use solow_robust::Rlm;

    // The most common stats tests, the ones a regression analyst reaches for
    // right after inspecting the coefficient table, plus non-parametric
    // two/k-sample tests and the three canonical correlation coefficients.
    pub use solow_stats::{
        anderson_darling, anova_lm, bartlett, chi2_contingency, describe, durbin_watson, f_test,
        fligner, het_white, jarque_bera, kendalltau, ks_2samp, kruskal, levene, mannwhitneyu,
        mcnemar, meta_fixed_effect, meta_random_effects, multipletests, pearsonr, runs_test,
        shapiro_wilk, spearmanr, wald_test, CorrelationResult, DescrStatsW, FTestResult,
        GofResult, LeveneCenter, MetaModel, MetaResult, MetaStudy, NpTestResult,
        VarianceTestResult, WaldResult,
    };
}
