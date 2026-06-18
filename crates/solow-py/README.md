# solow (Python bindings)

Python bindings (PyO3) for the **Solow** statistical-modeling stack. This crate
compiles to a native, abi3 CPython extension module named `solow` and marshals
NumPy `float64` arrays into Solow's pure-Rust estimators, returning NumPy arrays
and Python floats. A single abi3 wheel runs on CPython 3.9+.

It is its own Cargo workspace (excluded from the umbrella workspace) so it can be
built independently with the `extension-module` feature.

## Installation

### From source with maturin (recommended)

[maturin](https://www.maturin.rs/) is the standard build backend for PyO3
extensions and is wired up in `pyproject.toml`.

```sh
pip install maturin

# from this directory (crates/solow-py)
maturin develop --release        # build + install into the active venv/interpreter
```

`maturin develop` compiles the Rust extension and installs it editable into the
current environment, so `import solow` works immediately. It also sets the macOS
`-undefined dynamic_lookup` link flags for you.

### Building a distributable wheel

```sh
# from crates/solow-py
maturin build --release          # produces target/wheels/solow-0.1.0-*.whl (abi3)
pip install target/wheels/solow-*.whl
```

Because the extension is built abi3 (`abi3-py39`), one wheel is forward-compatible
with every CPython from 3.9 up — no per-version rebuild needed.

### Plain Cargo build (no maturin)

maturin is optional. A bare Cargo build works, but you must build **from this
crate directory** so the bundled `.cargo/config.toml` is discovered (Cargo walks
up from the working directory, not the manifest path):

```sh
# from crates/solow-py — NOT from the repo root
PYO3_PYTHON=$(which python3) cargo build --release
```

The output is a C dynamic library:

* macOS: `target/release/libsolow.dylib`
* Linux: `target/release/libsolow.so`

Copy it next to your code with an extension suffix CPython recognizes
(`.abi3.so` works because the crate is built with the `abi3-py39` feature):

```sh
mkdir -p pytest_dir
cp target/release/libsolow.dylib pytest_dir/solow.abi3.so   # macOS
PYTHONPATH=pytest_dir python3 -c "import solow; print(solow.__version__)"
```

#### macOS linker note

A bare `cargo build` of a `cdylib` does not, by itself, leave the CPython
symbols (`Py_*`) undefined for resolution at import time. The bundled
`crates/solow-py/.cargo/config.toml` passes `-undefined dynamic_lookup` to the
linker on `*-apple-darwin` targets so the extension imports cleanly. This config
is only picked up when Cargo is invoked **from inside this crate directory**;
building from the repo root (or with maturin) is the reliable path otherwise.

## Usage

All models take `endog` (1-D `float64`) and `exog` (2-D `float64`). As with the
canonical reference, **you supply any intercept column yourself** — no constant
is added automatically. Invalid shapes or non-convergence raise `ValueError`.

The results object does not ship a `.summary()` method, but every statistic the
reference prints is exposed as an attribute, so a familiar summary table is a few
lines away:

```python
import numpy as np
import solow

rng = np.random.default_rng(0)
n = 200
x1, x2 = rng.normal(size=n), rng.normal(size=n)
X = np.column_stack([np.ones(n), x1, x2])           # explicit intercept column
y = X @ np.array([1.5, -2.0, 0.5]) + rng.normal(size=n)

res = solow.OLS(y, X).fit()
names = ["const", "x1", "x2"]
ci = res.conf_int                                    # (k, 2) array, 95%

print("OLS Regression Results")
print("=" * 64)
print(f"R-squared:      {res.rsquared:.4f}    Adj. R-squared: {res.rsquared_adj:.4f}")
print(f"F-statistic:    {res.fvalue:.2f}   Prob (F):       {res.f_pvalue:.3e}")
print(f"No. Observations: {int(res.nobs)}   AIC: {res.aic:.2f}   BIC: {res.bic:.2f}")
print("-" * 64)
print(f"{'':8}{'coef':>10}{'std err':>10}{'t':>9}{'P>|t|':>9}{'[0.025':>10}{'0.975]':>9}")
for i, nm in enumerate(names):
    print(f"{nm:8}{res.params[i]:10.4f}{res.bse[i]:10.4f}"
          f"{res.tvalues[i]:9.3f}{res.pvalues[i]:9.3f}"
          f"{ci[i, 0]:10.4f}{ci[i, 1]:9.4f}")
print("=" * 64)

# Heteroskedasticity-robust (HC3) standard errors on the same fit
print("HC3 robust SE:", np.round(res.bse_robust("HC3"), 4))
```

Running the snippet above prints:

```
OLS Regression Results
================================================================
R-squared:      0.7980    Adj. R-squared: 0.7960
F-statistic:    389.17   Prob (F):       3.742e-69
No. Observations: 200   AIC: 571.96   BIC: 581.85
----------------------------------------------------------------
              coef   std err        t    P>|t|    [0.025   0.975]
const       1.5093    0.0712   21.191    0.000    1.3689   1.6498
x1         -1.9329    0.0740  -26.127    0.000   -2.0788  -1.7870
x2          0.5595    0.0694    8.067    0.000    0.4227   0.6963
================================================================
HC3 robust SE: [0.0716 0.0639 0.0725]
```

### Generalized linear models

```python
import numpy as np
import solow

rng = np.random.default_rng(0)
n = 200
x1 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1])
counts = rng.poisson(np.exp(0.3 + 0.5 * x1)).astype(float)

#   family = "gaussian" | "binomial" | "poisson" | "gamma" | "inversegaussian"
#   link   = optional; defaults to the family's canonical link
res = solow.GLM(counts, X, family="poisson").fit()
print("params:", np.round(res.params, 4),
      "| deviance:", round(res.deviance, 3),
      "| converged:", res.converged)
# -> params: [0.2803 0.5278] | deviance: 198.495 | converged: True

# logistic Binomial with an explicit link
# res = solow.GLM(y01, X, family="binomial", link="logit").fit()
```

### Discrete and time-series models

```python
res = solow.Logit(y01, X).fit()      # binary logistic MLE
res = solow.Probit(y01, X).fit()     # binary probit MLE
res = solow.Poisson(counts, X).fit() # count MLE
res.params, res.bse, res.tvalues, res.pvalues, res.llf, res.aic, res.bic, res.converged

res = solow.AutoReg(x, lags=2, trend="c").fit()   # trend = n|c|t|ct|ctt
res.params, res.bse, res.tvalues, res.pvalues
res.sigma2, res.llf, res.aic, res.bic, res.hqic, res.nobs, res.df_model

a = solow.acf(x, nlags)                # length nlags+1, a[0] == 1 (adjusted=False)
p = solow.pacf(x, nlags, method="yw")  # method = "yw" | "ols"
```

## API coverage

Everything exported by the module (`import solow; dir(solow)`):

### Linear least squares

| Class | Constructor | `.fit()` returns |
| --- | --- | --- |
| `OLS` | `OLS(endog, exog)` | `OLSResults` |
| `WLS` | `WLS(endog, exog, weights)` | `OLSResults` |
| `GLS` | `GLS(endog, exog, sigma)` | `OLSResults` |

`OLSResults` attributes: `params`, `bse`, `tvalues`, `pvalues`, `conf_int`
(`(k, 2)`, 95% / alpha = 0.05), `fittedvalues`, `resid`, `rsquared`,
`rsquared_adj`, `fvalue`, `f_pvalue`, `aic`, `bic`, `llf`, `scale`, `nobs`,
`df_model`, `df_resid`.

`OLSResults` robust-covariance methods (OLS fits only):

* `bse_robust(cov_type, maxlags=None, groups=None, use_correction=None)`
* `cov_params_robust(cov_type, maxlags=None, groups=None, use_correction=None)` → `(k, k)`

  `cov_type` ∈ `HC0 | HC1 | HC2 | HC3 | HAC | cluster`. HAC requires
  `maxlags`; cluster requires `groups` (int64 array, length nobs).
  `use_correction` defaults match the reference (`False` for HAC, `True` for
  cluster).

### Generalized linear models

| Class | Constructor | `.fit()` returns |
| --- | --- | --- |
| `GLM` | `GLM(endog, exog, family="gaussian", link=None)` | `GLMResults` |

`family` ∈ `gaussian | binomial | poisson | gamma | inversegaussian`.
`link` ∈ `identity | log | logit | probit | cloglog | inverse | inversesquared |
sqrt` (defaults to the family's canonical link).
`GLMResults` attributes: `params`, `bse`, `deviance`, `llf`, `aic`, `converged`.

### Discrete maximum-likelihood models

| Class | Constructor | `.fit()` returns |
| --- | --- | --- |
| `Logit` | `Logit(endog, exog)` | `DiscreteResults` |
| `Probit` | `Probit(endog, exog)` | `DiscreteResults` |
| `Poisson` | `Poisson(endog, exog)` | `DiscreteResults` |

`DiscreteResults` attributes: `params`, `bse`, `tvalues`, `pvalues`, `llf`,
`aic`, `bic`, `converged`.

### Time series

| Name | Signature | Returns |
| --- | --- | --- |
| `AutoReg` | `AutoReg(endog, lags, trend="c")` | `.fit()` → `AutoRegResults` |
| `acf` | `acf(x, nlags, adjusted=False)` | length `nlags+1` array |
| `pacf` | `pacf(x, nlags, method="yw")` | length `nlags+1` array |

`trend` ∈ `n | c | t | ct | ctt`; `pacf` `method` ∈ `yw | ols`.
`AutoRegResults` attributes: `params`, `bse`, `tvalues`, `pvalues`,
`fittedvalues`, `resid`, `sigma2`, `llf`, `aic`, `bic`, `hqic`, `nobs`,
`df_model`.

Module-level: `solow.__version__`.

## Verification

`tools/reference/verify_solow_py.py` builds this module, imports it, and compares
`params` / `bse` (and several other statistics) against an authoritative
reference package across OLS, GLM (Poisson, logistic Binomial, Gaussian), Logit,
and Poisson fits. The reference is loaded indirectly through the
`SOLOW_REFERENCE` environment variable:

```sh
SOLOW_REFERENCE=<reference-package> python3 tools/reference/verify_solow_py.py
```

`tools/reference/verify_solow_py_ext.py` extends this to the rest of the surface
— WLS, GLS, Probit, OLS robust standard errors (HC0–HC3 / HAC / cluster),
AutoReg, and `acf` / `pacf` — loading the reference's `tsa.ar_model` and
`tsa.stattools` submodules indirectly through the same env var:

```sh
SOLOW_REFERENCE=<reference-package> python3 tools/reference/verify_solow_py_ext.py
```

All comparisons agree to well within `1e-8` (observed max absolute difference
~`1e-13`), confirming the bindings faithfully marshal the already-validated
Rust results.
