# Using Solow from Python

Solow ships PyO3 bindings (`solow-py`) that compile to a native CPython
extension module named `solow`. The bindings marshal NumPy `float64` arrays into
Solow's estimators and return NumPy arrays and Python floats. They are verified
to reproduce the reference outputs to roughly `1e-12` from Python.

The `solow-py` crate is its own standalone Cargo workspace (excluded from the
umbrella workspace) so it can be built independently with the
`extension-module` feature.

## The Python API

Every model takes `endog` (a 1-D `float64` array) and `exog` (a 2-D `float64`
array). As with the reference stack, **you supply any intercept column
yourself** — no constant is added automatically.

```python
import numpy as np
import solow

# A design matrix with an explicit constant column.
n = 50
rng = np.random.default_rng(0)
x = rng.standard_normal(n)
X = np.column_stack([np.ones(n), x])           # [const, x]
y = 1.0 + 2.0 * x + 0.1 * rng.standard_normal(n)

# Ordinary least squares.
res = solow.OLS(y, X).fit()
print(res.params)        # ndarray of coefficients
print(res.bse)           # standard errors
print(res.tvalues)       # t-statistics
print(res.pvalues)       # p-values
print(res.conf_int)      # (k, 2) ndarray, 95% (alpha = 0.05)
print(res.rsquared, res.rsquared_adj, res.fvalue, res.f_pvalue)
print(res.aic, res.bic, res.llf, res.nobs, res.df_model, res.df_resid)
print(res.fittedvalues, res.resid)
```

`solow.WLS(y, X, weights)` and `solow.GLS(y, X, sigma)` cover weighted and
generalized least squares; the weights are per-observation (positive,
proportional to the inverse error variance) and `sigma` is the `(n, n)` SPD
error covariance.

### Robust (sandwich) standard errors

For an OLS fit, request heteroskedasticity- or autocorrelation-consistent
standard errors without re-fitting:

```python
import solow

res = solow.OLS(y, X).fit()
res.bse_robust("HC1")                      # HC0 | HC1 | HC2 | HC3
res.bse_robust("HAC", maxlags=4)           # Newey-West (use_correction=False)
res.bse_robust("cluster", groups=g)        # g: int64 array (use_correction=True)
res.cov_params_robust("HC3")               # the full (k, k) covariance matrix
```

The defaults match the reference: no small-sample correction for HAC, the
`G/(G-1) * (n-1)/(n-k)` correction for clusters. Pass `use_correction=...` to
override.

### Generalized linear models

`family` is one of `"gaussian"`, `"binomial"`, `"poisson"`, `"gamma"`, or
`"inversegaussian"`. `link` is optional and defaults to the family's canonical
link.

```python
import solow

res = solow.GLM(y, X, family="poisson").fit()
res = solow.GLM(y, X, family="binomial", link="logit").fit()

print(res.params, res.bse, res.deviance, res.llf, res.aic, res.converged)
```

### Discrete maximum-likelihood models

```python
import solow

res = solow.Logit(y, X).fit()
res = solow.Probit(y, X).fit()
res = solow.Poisson(y, X).fit()

print(res.params, res.bse, res.tvalues, res.pvalues)
print(res.llf, res.aic, res.bic, res.converged)
```

### Time series

```python
import solow

# Autoregression; trend is one of "n" | "c" | "t" | "ct" | "ctt".
res = solow.AutoReg(x, lags=2, trend="c").fit()
print(res.params, res.bse, res.tvalues, res.pvalues)
print(res.sigma2, res.llf, res.aic, res.bic, res.hqic)

# Autocorrelation diagnostics.
a = solow.acf(x, nlags=10)                 # length nlags+1, a[0] == 1
p = solow.pacf(x, nlags=10, method="yw")   # method = "yw" | "ols"
```

Invalid shapes or non-convergence raise `ValueError`.

## Building the extension

`maturin` is the usual tool, but it is optional — a plain Cargo build works.
From the `crates/solow-py` directory (it is a standalone workspace):

```sh
PYO3_PYTHON=$(which python3) cargo build --release
```

The output is a C dynamic library:

- macOS: `target/release/libsolow.dylib`
- Linux: `target/release/libsolow.so`

Copy it next to your code with an extension suffix CPython recognizes. Because
the crate is built with the `abi3-py39` feature, a single `.abi3.so` binary is
compatible with CPython 3.9 and newer:

```sh
mkdir -p pytest_dir
cp target/release/libsolow.dylib pytest_dir/solow.abi3.so   # macOS
PYTHONPATH=pytest_dir python3 -c "import solow; print(solow.__version__)"
```

On macOS the crate's bundled `.cargo/config.toml` passes
`-undefined dynamic_lookup` to the linker so the extension imports cleanly;
building through `maturin` would arrange this automatically.

## Verification from Python

The bindings exist not only for convenience but as an independent check: the
`solow-py` test suite fits each model in both Solow and the authoritative
reference on the same data and asserts agreement to about `1e-12`. If you are
migrating Python code, you can swap `import solow` in for the reference import
on the supported models and expect the same numbers.
