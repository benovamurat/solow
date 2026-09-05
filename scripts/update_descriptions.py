#!/usr/bin/env python3
"""Rewrite every workspace Cargo.toml description with a product-first,
SEO-oriented copy that leads with 'Rust' + primary keywords and lists
the concrete algorithms. No em-dashes. Kept under the crates.io 300-char
cap.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]

DESCS = {
    # Foundation
    "solow-core": "Rust numerical error types, array-shape checks, and design-matrix helpers. Foundation crate for the solow statistics and machine learning stack.",
    "solow-linalg": "Pure Rust linear algebra for statistics and machine learning: SVD, eigendecomposition, Cholesky, QR, LU, matrix pseudoinverse. No system LAPACK or BLAS. Memory safe.",
    "solow-distributions": "Rust statistical distributions and special functions: normal, Student t, chi-squared, F, gamma, beta, Weibull, log-normal, Poisson, negative binomial, GEV, GPD, plus lgamma, digamma, erf, incomplete beta and gamma.",
    "solow-optimize": "Rust unconstrained optimizers for statistical model fitting: Newton, BFGS, line search, and finite-difference gradient and Hessian.",

    # Classical models
    "solow-regression": "Rust linear regression and regularization: OLS, WLS, GLS, GLSAR, Ridge, Lasso, ElasticNet, cross-validated regularization, LARS, LassoLars, LassoLarsIC, OrthogonalMatchingPursuit, Huber, quantile, RANSAC, TheilSen, SGD, Perceptron, Passive Aggressive, Bayesian Ridge, ARD, multi-task, kernel ridge. Deterministic, memory safe.",
    "solow-glm": "Rust generalized linear models by IRLS: Gaussian, Poisson, Binomial, Gamma, and Tweedie families with log, logit, probit, and identity links. Deterministic solvers, memory safe.",
    "solow-discrete": "Rust discrete choice and count regression: logit, probit, Poisson, negative binomial, generalized Poisson, multinomial logit, ordered logit and probit, zero-inflated Poisson, hurdle, conditional logit and Poisson, truncated Poisson, cross-validated logistic regression.",
    "solow-stats": "Rust statistical tests and diagnostics: t-tests, ANOVA, Wald, F-test, Levene, Bartlett, Fligner, Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov, Mann-Whitney U, Kruskal-Wallis, McNemar, chi-squared, Pearson, Spearman, Kendall correlations, meta-analysis, robust HC and HAC covariances, VIF, mediation, power analysis.",
    "solow-tsa": "Rust time series analysis: ACF, PACF, ADF, KPSS, cointegration, Granger causality, AutoReg, ARMA, STL, seasonal decomposition, Holt-Winters, HP, BK, CF filters, Zivot-Andrews, GARCH(1,1), CUSUM, PELT change-point detection, EWMA control chart.",
    "solow-robust": "Rust robust linear regression by M-estimation. Huber, Tukey biweight, Andrew, Hampel, and Ramsay influence functions with iteratively reweighted least squares.",
    "solow-nonparametric": "Rust nonparametric smoothers: LOWESS local regression, kernel density estimation (KDE), Nadaraya-Watson kernel regression.",
    "solow-multivariate": "Rust multivariate analysis: principal component analysis (PCA), factor analysis with varimax rotation, MANOVA, canonical correlation analysis (CCA).",
    "solow-duration": "Rust survival and duration analysis: Kaplan-Meier estimator, Nelson-Aalen cumulative hazard, Cox proportional hazards, log-rank test, survival function estimator.",
    "solow-statespace": "Rust state-space models: Kalman filter, Kalman smoother, SARIMAX, dynamic factor model, unobserved components, multivariate state space.",
    "solow-var": "Rust vector autoregression: VAR, VECM with Johansen cointegration test, structural VAR, impulse response functions, forecast error variance decomposition.",
    "solow-mixed": "Rust linear mixed-effects models: REML estimation, random intercepts, random slopes, crossed and nested random effects.",
    "solow-gee": "Rust generalized estimating equations (GEE): Gaussian, Poisson, Binomial, Gamma families with exchangeable, autoregressive, and unstructured working correlations.",
    "solow-gam": "Rust generalized additive models (GAM): penalized B-splines, cyclic cubic splines, smoothing-parameter selection by generalized cross-validation.",
    "solow-impute": "Rust missing-data imputation: simple imputation (mean, median, most frequent, constant), K-nearest-neighbors imputation, iterative multivariate imputation (MICE), Rubin combining rules.",
    "solow-graphics": "Rust statistical graphics: QQ plot, PP plot, ACF and PACF plots, residual diagnostics, influence plots, regression plots rendered as SVG.",
    "solow-regime": "Rust regime-switching time series: Markov switching regression, Markov switching autoregression, filtered and smoothed regime probabilities.",
    "solow-othermod": "Rust beta regression for continuous data on the unit interval. Mean-precision parameterisation, logit link on the mean, log link on the precision.",
    "solow-copula": "Rust copulas: Gaussian, Student t, Clayton, Frank, Gumbel, Joe. Sampling, log-likelihood, Kendall tau, Spearman rho, tail dependence coefficients.",
    "solow-fit": "Rust ergonomic formula-driven fit surface for the solow statistics stack: ols, wls, gls, glm, logit, probit, poisson from R-style formula strings.",
    "solow-bayes": "Rust Bayesian mixed generalized linear models by variational Bayes. Binomial and Poisson likelihoods, normal random effects, closed-form evidence lower bound.",
    "solow-emplike": "Rust empirical-likelihood inference. Nonparametric tests of the mean and variance, empirical-likelihood confidence intervals via profile log-likelihood.",
    "solow-formula": "Rust formula parser and design-matrix builder for statistical models. Intercept, interactions, treatment contrasts, categorical encoding. R and patsy syntax compatible.",
    "solow-summary": "Rust results tables and summary printers for statistical model output. Coefficient tables, standard errors, confidence intervals, information criteria.",

    # Benchmarks / demos (kept short)
    "solow-bench": "Criterion benchmarks for the solow statistics workspace.",
    "solow-gallery": "Runnable end-to-end examples for the solow statistics workspace.",

    # Visualization
    "solow-viz": "Rust SVG data-visualization backend for the solow statistics stack. Scatter, line, bar, histogram, box, ROC, PR, calibration, and residual plots.",

    # Model evaluation and resampling
    "solow-metrics": "Rust regression, classification, forecast, and cluster evaluation metrics for machine learning: MSE, MAE, R2, RMSE, ROC AUC, average precision, log loss, calibration error, silhouette, adjusted Rand, normalized mutual information, homogeneity, completeness, v-measure, pairwise kernels and distances, effect sizes (Cohen d, Hedges g, Cliff delta, Cramer V), classification report.",
    "solow-cv": "Rust cross-validation and resampling for machine learning: KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut, LeavePOut, ShuffleSplit, GroupKFold, StratifiedGroupKFold, PurgedKFold, CombinatorialPurgedKFold, RepeatedKFold, StratifiedShuffleSplit, GroupShuffleSplit, bootstrap CI, block bootstrap, learning curve, validation curve, permutation test.",

    # preprocessing
    "solow-preprocessing": "Rust feature preprocessing for machine learning: standard, min-max, robust, and max-abs scalers, Normalizer, polynomial features, Yeo-Johnson and Box-Cox power transforms, quantile transformer, K-bin discretizer, one-hot, ordinal, label and multi-label encoders, binarizer, target encoder, spline transformer, function transformer.",

    # cluster
    "solow-cluster": "Rust unsupervised clustering for machine learning: k-means with k-means++, mini-batch k-means, bisecting k-means, DBSCAN, HDBSCAN, OPTICS, mean shift, affinity propagation, spectral clustering, agglomerative clustering with single, complete, average, and Ward linkage, Birch, Gaussian mixture, Bayesian Gaussian mixture.",

    # neighbors
    "solow-neighbors": "Rust nearest-neighbor structures and estimators for machine learning: KD-tree, ball tree, k-nearest-neighbors classifier and regressor, radius-neighbors classifier and regressor, nearest centroid, local outlier factor (LOF), kernel density estimator.",

    # tree
    "solow-tree": "Rust decision trees for machine learning: decision tree classifier and regressor, extra tree classifier and regressor, with Gini, entropy, MSE, and MAE splitters and deterministic tie-breaking.",

    # ensemble
    "solow-ensemble": "Rust ensemble methods for machine learning: random forest classifier and regressor, extra trees classifier and regressor, gradient boosting classifier and regressor, histogram gradient boosting classifier and regressor, bagging classifier and regressor, AdaBoost classifier and regressor, voting classifier and regressor, stacking classifier and regressor, isolation forest.",

    # naive_bayes
    "solow-naive-bayes": "Rust naive Bayes classifiers for machine learning: Gaussian, multinomial, Bernoulli, complement, and categorical naive Bayes with Laplace or Lidstone smoothing.",

    # discriminant_analysis
    "solow-discriminant": "Rust discriminant analysis for machine learning classification: linear discriminant analysis (Fisher 1936) and quadratic discriminant analysis with pooled and per-class covariance.",

    # feature_selection
    "solow-feature-selection": "Rust feature selection for machine learning: select-K-best with ANOVA F, regression F, or mutual information, variance threshold, recursive feature elimination (RFE), percentile, FPR, FDR, FWE, and sequential feature selector.",

    # pipeline + compose
    "solow-pipeline": "Rust machine-learning pipelines and hyperparameter search: pipeline, feature union, column transformer, transformed-target regressor, grid search CV, randomized search CV, halving grid search, halving random search.",

    # svm
    "solow-svm": "Rust support vector machines: linear SVC (Pegasos), linear SVR (epsilon-insensitive), SVC and SVR with linear, RBF, polynomial, and sigmoid kernels via SMO, nu-SVC, nu-SVR, one-class SVM.",

    # neural_network
    "solow-neural": "Rust neural networks: multi-layer perceptron classifier and regressor with ReLU, tanh, logistic, and identity activations and SGD or Adam optimizers, Bernoulli restricted Boltzmann machine.",

    # manifold
    "solow-manifold": "Rust manifold learning and nonlinear dimensionality reduction: Isomap, locally linear embedding (LLE), multidimensional scaling (MDS), spectral embedding, t-SNE.",

    # decomposition + random_projection
    "solow-decomposition": "Rust matrix decomposition for machine learning: PCA, kernel PCA, FastICA, NMF, mini-batch NMF, truncated SVD (LSA), incremental PCA, sparse PCA, dictionary learning, mini-batch dictionary learning, latent Dirichlet allocation, Gaussian and sparse random projection.",

    # feature_extraction.text
    "solow-text": "Rust text feature extraction for NLP: count vectorizer, TF-IDF vectorizer, hashing vectorizer, feature hasher, dict vectorizer with configurable tokenizer, n-grams, min and max document frequency, and stop words.",

    # covariance
    "solow-covariance": "Rust covariance matrix estimators: empirical covariance, shrunk covariance, Ledoit-Wolf, Oracle Approximating Shrinkage (OAS), Rousseeuw FAST-MCD minimum covariance determinant, graphical lasso, elliptic envelope outlier detection.",

    # cross_decomposition
    "solow-cross-decomposition": "Rust partial least squares and canonical correlation for machine learning: PLS regression, PLS canonical, PLS SVD, canonical correlation analysis (CCA) with the NIPALS solver.",

    # kernel_approximation
    "solow-kernel-approx": "Rust kernel approximation via random features: RBF sampler (random Fourier features), Nystroem, additive chi-squared sampler, skewed chi-squared sampler, polynomial count sketch.",

    # gaussian_process
    "solow-gp": "Rust Gaussian process regression and classification: Gaussian process regressor and classifier with RBF, Matern, rational quadratic, dot product, constant, and white kernels and sum, product, exponentiation kernel composition.",

    # semi_supervised
    "solow-semi-supervised": "Rust semi-supervised learning: self-training classifier, label propagation, label spreading.",

    # multiclass + multioutput
    "solow-multi": "Rust multiclass and multi-output meta-estimators: one-vs-rest, one-vs-one, output code, multi-output regressor and classifier, classifier chain, regressor chain.",

    # calibration
    "solow-calibration": "Rust probability calibration for classifiers: calibrated classifier CV with Platt sigmoid and isotonic regression post-hoc calibration.",

    # datasets + utils
    "solow-datasets": "Rust dataset generators and toy loaders for machine learning: make classification, regression, blobs, moons, circles, Swiss roll, and low-rank matrix. Loads iris, wine, diabetes, breast cancer. Class-weight and sample-weight utilities, resample helpers.",

    # Umbrella
    "solow": "The comprehensive statistics and machine learning stack for Rust. Regression, generalized linear models, discrete choice, time series, state space, survival analysis, mixed effects, Bayesian inference, clustering, tree ensembles, SVM, neural networks, kernel methods, dimensionality reduction, and change point detection. 57 focused crates, memory safe, pure Rust.",

    # Excluded / bindings
    "solow-py": "Python bindings for the solow statistics stack via PyO3.",
    "solow-polars": "Polars DataFrame interop for the solow statistics stack. Fit models directly from DataFrame columns.",
}


def update(cargo: pathlib.Path, new_desc: str) -> bool:
    text = cargo.read_text()
    pattern = re.compile(r"^description\s*=\s*\"([^\"]*)\"", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        print(f"SKIP {cargo}: no description line")
        return False
    if m.group(1) == new_desc:
        return False
    new = pattern.sub(f"description = \"{new_desc}\"", text, count=1)
    cargo.write_text(new)
    return True


def main():
    changed = 0
    for cargo in REPO.glob("crates/*/Cargo.toml"):
        name_match = re.search(r"^name\s*=\s*\"([^\"]+)\"", cargo.read_text(), re.MULTILINE)
        if not name_match:
            continue
        name = name_match.group(1)
        if name not in DESCS:
            print(f"MISSING {name}")
            continue
        assert "—" not in DESCS[name], f"em-dash in {name}"
        if update(cargo, DESCS[name]):
            print(f"updated {name}")
            changed += 1
    print(f"\n{changed} crates updated")


if __name__ == "__main__":
    main()
