//! Extended unit-root and heteroskedasticity tests: the Zivot-Andrews test
//! with an endogenous structural break, the range unit-root (RUR) test, and
//! the break-variance (Goldfeld-Quandt style) heteroskedasticity test.
//!
//! These mirror the reference `stattools.zivot_andrews`,
//! `stattools.range_unit_root_test` and
//! `stattools.breakvar_heteroskedasticity_test`.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::{f_cdf, f_sf};
use solow_linalg::inv;

use crate::stattools::{adfuller, AdfRegression, AutoLag};
use crate::tsatools::{lagmat1d, Original, Trim};

// ---------------------------------------------------------------------------
// Zivot-Andrews structural-break unit-root test
// ---------------------------------------------------------------------------

/// Deterministic specification for the Zivot-Andrews regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZaRegression {
    /// Constant only (allow a break in the intercept), `"c"` (the default).
    C,
    /// Trend only (allow a break in the trend), `"t"`.
    T,
    /// Constant and trend (allow a break in both), `"ct"`.
    Ct,
}

impl ZaRegression {
    /// Parse a regression code (`"c"`, `"t"`, `"ct"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "c" => Ok(ZaRegression::C),
            "t" => Ok(ZaRegression::T),
            "ct" => Ok(ZaRegression::Ct),
            other => Err(Error::Value(format!("unknown regression '{other}'"))),
        }
    }
}

/// Result of the Zivot-Andrews structural-break unit-root test.
#[derive(Debug, Clone)]
pub struct ZivotAndrewsResult {
    /// The test statistic (the minimum t-statistic over candidate breakpoints).
    pub stat: f64,
    /// Interpolated p-value.
    pub pvalue: f64,
    /// Critical values at the 1%, 5% and 10% levels.
    pub crit_values: [f64; 3],
    /// Number of lagged differences used (`baselags`).
    pub baselag: usize,
    /// The index of the selected structural break.
    pub breakidx: usize,
}

/// Linear interpolation matching `numpy.interp` (with flat extrapolation).
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    for i in 1..xp.len() {
        if x <= xp[i] {
            let t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
            return fp[i - 1] + t * (fp[i] - fp[i - 1]);
        }
    }
    fp[last]
}

/// The Zivot-Andrews critical-value tables: `(percent, statistic)` rows for the
/// three models. The percentages are quantiles times 100.
fn za_table(reg: ZaRegression) -> &'static [(f64, f64)] {
    const C: &[(f64, f64)] = &[
        (0.001, -6.78442),
        (0.100, -5.83192),
        (0.200, -5.68139),
        (0.300, -5.58461),
        (0.400, -5.51308),
        (0.500, -5.45043),
        (0.600, -5.39924),
        (0.700, -5.36023),
        (0.800, -5.33219),
        (0.900, -5.30294),
        (1.000, -5.27644),
        (2.500, -5.03340),
        (5.000, -4.81067),
        (7.500, -4.67636),
        (10.000, -4.56618),
        (12.500, -4.48130),
        (15.000, -4.40507),
        (17.500, -4.33947),
        (20.000, -4.28155),
        (22.500, -4.22683),
        (25.000, -4.17830),
        (27.500, -4.13101),
        (30.000, -4.08586),
        (32.500, -4.04455),
        (35.000, -4.00380),
        (37.500, -3.96144),
        (40.000, -3.92078),
        (42.500, -3.88178),
        (45.000, -3.84503),
        (47.500, -3.80549),
        (50.000, -3.77031),
        (52.500, -3.73209),
        (55.000, -3.69600),
        (57.500, -3.65985),
        (60.000, -3.62126),
        (65.000, -3.54580),
        (70.000, -3.46848),
        (75.000, -3.38533),
        (80.000, -3.29112),
        (85.000, -3.17832),
        (90.000, -3.04165),
        (92.500, -2.95146),
        (95.000, -2.83179),
        (96.000, -2.76465),
        (97.000, -2.68624),
        (98.000, -2.57884),
        (99.000, -2.40044),
        (99.900, -1.88932),
    ];
    const T: &[(f64, f64)] = &[
        (0.001, -83.9094),
        (0.100, -13.8837),
        (0.200, -9.13205),
        (0.300, -6.32564),
        (0.400, -5.60803),
        (0.500, -5.38794),
        (0.600, -5.26585),
        (0.700, -5.18734),
        (0.800, -5.12756),
        (0.900, -5.07984),
        (1.000, -5.03421),
        (2.500, -4.65634),
        (5.000, -4.40580),
        (7.500, -4.25214),
        (10.000, -4.13678),
        (12.500, -4.03765),
        (15.000, -3.95185),
        (17.500, -3.87945),
        (20.000, -3.81295),
        (22.500, -3.75273),
        (25.000, -3.69836),
        (27.500, -3.64785),
        (30.000, -3.59819),
        (32.500, -3.55146),
        (35.000, -3.50522),
        (37.500, -3.45987),
        (40.000, -3.41672),
        (42.500, -3.37465),
        (45.000, -3.33394),
        (47.500, -3.29393),
        (50.000, -3.25316),
        (52.500, -3.21244),
        (55.000, -3.17124),
        (57.500, -3.13211),
        (60.000, -3.09204),
        (65.000, -3.01135),
        (70.000, -2.92897),
        (75.000, -2.83614),
        (80.000, -2.73893),
        (85.000, -2.62840),
        (90.000, -2.49611),
        (92.500, -2.41337),
        (95.000, -2.30820),
        (96.000, -2.25797),
        (97.000, -2.19648),
        (98.000, -2.11320),
        (99.000, -1.99138),
        (99.900, -1.67466),
    ];
    const CT: &[(f64, f64)] = &[
        (0.001, -38.17800),
        (0.100, -6.43107),
        (0.200, -6.07279),
        (0.300, -5.95496),
        (0.400, -5.86254),
        (0.500, -5.77081),
        (0.600, -5.72541),
        (0.700, -5.68406),
        (0.800, -5.65163),
        (0.900, -5.60419),
        (1.000, -5.57556),
        (2.500, -5.29704),
        (5.000, -5.07332),
        (7.500, -4.93003),
        (10.000, -4.82668),
        (12.500, -4.73711),
        (15.000, -4.66020),
        (17.500, -4.58970),
        (20.000, -4.52855),
        (22.500, -4.47100),
        (25.000, -4.42011),
        (27.500, -4.37387),
        (30.000, -4.32705),
        (32.500, -4.28126),
        (35.000, -4.23793),
        (37.500, -4.19822),
        (40.000, -4.15800),
        (42.500, -4.11946),
        (45.000, -4.08064),
        (47.500, -4.04286),
        (50.000, -4.00489),
        (52.500, -3.96837),
        (55.000, -3.93200),
        (57.500, -3.89496),
        (60.000, -3.85577),
        (65.000, -3.77795),
        (70.000, -3.69794),
        (75.000, -3.61852),
        (80.000, -3.52485),
        (85.000, -3.41665),
        (90.000, -3.28527),
        (92.500, -3.19724),
        (95.000, -3.08769),
        (96.000, -3.03088),
        (97.000, -2.96091),
        (98.000, -2.85581),
        (99.000, -2.71015),
        (99.900, -2.28767),
    ];
    match reg {
        ZaRegression::C => C,
        ZaRegression::T => T,
        ZaRegression::Ct => CT,
    }
}

/// Linear interpolation for the Zivot-Andrews p-value and critical values.
fn za_crit(stat: f64, reg: ZaRegression) -> (f64, [f64; 3]) {
    let table = za_table(reg);
    let pcnts: Vec<f64> = table.iter().map(|&(p, _)| p).collect();
    let stats: Vec<f64> = table.iter().map(|&(_, s)| s).collect();
    // p-value: interp(stat, stats, pcnts) / 100. `stats` is increasing.
    let pvalue = np_interp(stat, &stats, &pcnts) / 100.0;
    // Critical values: interp([1, 5, 10], pcnts, stats).
    let crit = [
        np_interp(1.0, &pcnts, &stats),
        np_interp(5.0, &pcnts, &stats),
        np_interp(10.0, &pcnts, &stats),
    ];
    (pvalue, crit)
}

/// Minimal least-squares t-statistic vector for internal use, mirroring the
/// reference `_quick_ols`: returns `b / sqrt(diag(sigma2 * (X'X)^{-1}))`.
fn quick_ols(endog: &Array1<f64>, exog: &Array2<f64>) -> Result<Array1<f64>> {
    let (nobs, k) = exog.dim();
    let xtx = exog.t().dot(exog);
    let xpxi = inv(&xtx)?;
    let xpy = exog.t().dot(endog);
    let b = xpxi.dot(&xpy);
    let e = endog - &exog.dot(&b);
    let sigma2 = e.dot(&e) / (nobs - k) as f64;
    let mut tvals = Array1::<f64>::zeros(k);
    for j in 0..k {
        tvals[j] = b[j] / (sigma2 * xpxi[[j, j]]).sqrt();
    }
    Ok(tvals)
}

/// Build the auxiliary endog/exog regression data, mirroring the reference
/// `_format_regression_data`. Returns `(endog, exog)` where `endog` is the
/// standardized first difference and `exog` has `cols + lags` columns.
fn format_regression_data(
    series: &Array1<f64>,
    c_const: f64,
    cols: usize,
    lags: usize,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let nobs = series.len();
    // endog = diff(series); standardize.
    let mut endog: Array1<f64> = Array1::from_iter((1..nobs).map(|i| series[i] - series[i - 1]));
    let enorm = endog.dot(&endog).sqrt();
    endog.mapv_inplace(|v| v / enorm);
    // series standardized.
    let snorm = series.dot(series).sqrt();
    let series_std = series.mapv(|v| v / snorm);

    // exog has endog[lags:].shape[0] rows = (nobs - 1 - lags) rows.
    let nrows = endog.len() - lags;
    let mut exog = Array2::<f64>::zeros((nrows, cols + lags));
    // column 0: constant.
    for i in 0..nrows {
        exog[[i, 0]] = c_const;
    }
    // column cols-1: lagged level, series[lags:(nobs-1)].
    for i in 0..nrows {
        exog[[i, cols - 1]] = series_std[lags + i];
    }
    // columns cols..: lagmat(endog, lags, trim="none")[lags : nrows + lags].
    if lags > 0 {
        let (lm, _) = lagmat1d(&endog, lags, Trim::None, Original::Ex)?;
        for i in 0..nrows {
            for j in 0..lags {
                exog[[i, cols + j]] = lm[[lags + i, j]];
            }
        }
    }
    Ok((endog, exog))
}

/// Update the exog array for the breakpoint `period`, mirroring the reference
/// `_update_regression_exog`.
fn update_regression_exog(
    exog: &mut Array2<f64>,
    reg: ZaRegression,
    period: usize,
    c_const: f64,
    t_const: &Array1<f64>,
    lags: usize,
) {
    let cutoff = period - (lags + 1);
    let nrows = exog.nrows();
    if reg != ZaRegression::T {
        // intercept dummy in column 1.
        for i in 0..cutoff.min(nrows) {
            exog[[i, 1]] = 0.0;
        }
        for i in cutoff..nrows {
            exog[[i, 1]] = c_const;
        }
        // trend in column 2: t_const[(lags+2):(nobs+1)].
        for i in 0..nrows {
            exog[[i, 2]] = t_const[lags + 2 + i];
        }
        if reg == ZaRegression::Ct {
            // trend dummy in column 3.
            for i in 0..cutoff.min(nrows) {
                exog[[i, 3]] = 0.0;
            }
            // t_const[1:(nobs-period+1)] placed from row `cutoff`.
            for i in cutoff..nrows {
                exog[[i, 3]] = t_const[1 + (i - cutoff)];
            }
        }
    } else {
        // column 1: trend.
        for i in 0..nrows {
            exog[[i, 1]] = t_const[lags + 2 + i];
        }
        // column 2: trend dummy, zeros up to cutoff-1, then t_const[0:...].
        let c1 = cutoff - 1;
        for i in 0..c1.min(nrows) {
            exog[[i, 2]] = 0.0;
        }
        for i in c1..nrows {
            exog[[i, 2]] = t_const[i - c1];
        }
    }
}

/// Zivot-Andrews structural-break unit-root test.
///
/// Mirrors the reference `zivot_andrews(x, trim, maxlag, regression, autolag)`.
/// The statistic is the minimum t-statistic on the lagged level over all
/// candidate breakpoints; `breakidx` is the corresponding break index.
///
/// `autolag` selects the number of lagged differences via [`adfuller`] with a
/// constant-and-trend regression; pass [`AutoLag::None`] to use exactly
/// `maxlag` lags, or `None` for the Schwert default when `maxlag` is also
/// `None` (here `maxlag` is mandatory, so pass it explicitly).
pub fn zivot_andrews(
    x: &Array1<f64>,
    trim: f64,
    maxlag: Option<usize>,
    regression: ZaRegression,
    autolag: Option<AutoLag>,
) -> Result<ZivotAndrewsResult> {
    if !(0.0..=(1.0 / 3.0)).contains(&trim) {
        return Err(Error::Value(
            "trim value must be a float in range [0, 1/3)".into(),
        ));
    }
    let nobs = x.len();

    // Determine baselags.
    let baselags = match autolag {
        Some(al) if al != AutoLag::None => {
            // adfuller(x, maxlag, regression="ct", autolag).usedlag. When ZA's
            // maxlag is None, the reference passes None to adfuller, which uses
            // its own default `ceil(12 * (nobs/100)^{1/4})` (note: ceil, unlike
            // ZA's own truncated fallback below).
            let ml = maxlag.unwrap_or_else(|| adf_default_maxlag(nobs));
            let res = adfuller(x, ml, AdfRegression::Ct, al)?;
            res.usedlag
        }
        _ => match maxlag {
            Some(ml) => ml,
            None => za_schwert(nobs),
        },
    };

    let trimcnt = (nobs as f64 * trim) as usize;
    let start_period = trimcnt;
    let end_period = nobs - trimcnt;
    let basecols = match regression {
        ZaRegression::Ct => 5,
        _ => 4,
    };

    // Normalized constant and trend terms.
    let c_const = 1.0 / (nobs as f64).sqrt();
    // t_const = arange(1, nobs+2) * sqrt(3) / nobs^(3/2). Length nobs + 1.
    let scale = 3.0_f64.sqrt() / (nobs as f64).powf(1.5);
    let t_const: Array1<f64> = Array1::from_iter((1..=nobs + 1).map(|i| i as f64 * scale));

    let (endog, mut exog) = format_regression_data(x, c_const, basecols, baselags)?;

    // Iterate over candidate breakpoints.
    let mut stats = Array1::from_elem(end_period + 1, f64::INFINITY);
    for bp in (start_period + 1)..=end_period {
        update_regression_exog(&mut exog, regression, bp, c_const, &t_const, baselags);
        let tvals = quick_ols(&endog.slice(s![baselags..]).to_owned(), &exog)?;
        stats[bp] = tvals[basecols - 1];
    }

    // Best (minimum) statistic and its break index.
    let mut zastat = f64::INFINITY;
    let mut argmin = 0usize;
    for (i, &v) in stats.iter().enumerate() {
        if v < zastat {
            zastat = v;
            argmin = i;
        }
    }
    let breakidx = argmin - 1;

    let (pvalue, crit_values) = za_crit(zastat, regression);

    Ok(ZivotAndrewsResult {
        stat: zastat,
        pvalue,
        crit_values,
        baselag: baselags,
        breakidx,
    })
}

/// Schwert (1989) default lag length `12 * (nobs/100)^{1/4}` (truncated),
/// matching the reference Zivot-Andrews fallback when `autolag` is disabled.
fn za_schwert(nobs: usize) -> usize {
    (12.0 * (nobs as f64 / 100.0).powf(0.25)) as usize
}

/// The reference `adfuller` default maxlag `ceil(12 * (nobs/100)^{1/4})`, used
/// when the Zivot-Andrews `autolag` path delegates lag selection to `adfuller`.
fn adf_default_maxlag(nobs: usize) -> usize {
    (12.0 * (nobs as f64 / 100.0).powf(0.25)).ceil() as usize
}

// ---------------------------------------------------------------------------
// Range unit-root (RUR) test
// ---------------------------------------------------------------------------

/// Result of the range unit-root (RUR) test.
#[derive(Debug, Clone)]
pub struct RangeUnitRootResult {
    /// The RUR test statistic.
    pub stat: f64,
    /// The interpolated p-value (a boundary point if outside the table).
    pub pvalue: f64,
    /// Critical values at the 10%, 5%, 2.5% and 1% levels.
    pub crit_values: [f64; 4],
}

/// `scipy.interpolate.interp1d` (linear, no extrapolation) restricted to the
/// table support `[n[0], n[last]]`.
fn interp1d(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    for i in 1..xp.len() {
        if x <= xp[i] {
            let t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
            return fp[i - 1] + t * (fp[i] - fp[i - 1]);
        }
    }
    fp[last]
}

/// Range unit-root test for stationarity.
///
/// Mirrors the reference `range_unit_root_test(x)`. Returns the statistic, the
/// interpolated p-value, and the critical values at the 10%, 5%, 2.5% and 1%
/// levels. The statistic counts the number of records (new running maxima and
/// minima) divided by `sqrt(nobs)`.
pub fn range_unit_root_test(x: &Array1<f64>) -> Result<RangeUnitRootResult> {
    let nobs = x.len();
    if nobs == 0 {
        return Err(Error::Value("x must be non-empty".into()));
    }

    let pvals = [0.01, 0.025, 0.05, 0.10, 0.90, 0.95];
    let n = [
        25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0,
    ];
    let crit: [[f64; 6]; 12] = [
        [0.6626, 0.8126, 0.9192, 1.0712, 2.4863, 2.7312],
        [0.7977, 0.9274, 1.0478, 1.1964, 2.6821, 2.9613],
        [0.9070, 1.0243, 1.1412, 1.2888, 2.8317, 3.1393],
        [0.9543, 1.0768, 1.1869, 1.3294, 2.8915, 3.2049],
        [0.9833, 1.0984, 1.2101, 1.3494, 2.9308, 3.2482],
        [0.9982, 1.1137, 1.2242, 1.3632, 2.9571, 3.2842],
        [1.0494, 1.1643, 1.2712, 1.4076, 3.0207, 3.3584],
        [1.0846, 1.1959, 1.2988, 1.4344, 3.0653, 3.4073],
        [1.1121, 1.2200, 1.3230, 1.4556, 3.0948, 3.4439],
        [1.1204, 1.2295, 1.3303, 1.4656, 3.1054, 3.4632],
        [1.1309, 1.2347, 1.3378, 1.4693, 3.1165, 3.4717],
        [1.1377, 1.2402, 1.3408, 1.4729, 3.1252, 3.4807],
    ];

    // Interpolate each critical-value column at nobs.
    let mut inter_crit = [0.0_f64; 6];
    for (j, ic) in inter_crit.iter_mut().enumerate() {
        let col: Vec<f64> = crit.iter().map(|row| row[j]).collect();
        *ic = interp1d(nobs as f64, &n, &col);
    }

    // RUR statistic: count records (new running max / min), excluding the
    // first observation (the expanding max/min is shifted by one).
    let mut running_max = x[0];
    let mut running_min = x[0];
    let mut count = 0usize;
    for &v in x.iter().skip(1) {
        if v > running_max {
            count += 1;
        }
        if v < running_min {
            count += 1;
        }
        if v > running_max {
            running_max = v;
        }
        if v < running_min {
            running_min = v;
        }
    }
    let stat = count as f64 / (nobs as f64).sqrt();

    // Locate the p-value bucket.
    let mut k = pvals.len() - 1;
    for i in (0..pvals.len()).rev() {
        if stat < inter_crit[i] {
            k = i;
        } else {
            break;
        }
    }
    let pvalue = pvals[k];

    let crit_values = [
        inter_crit[3], // 10%
        inter_crit[2], // 5%
        inter_crit[1], // 2.5%
        inter_crit[0], // 1%
    ];

    Ok(RangeUnitRootResult {
        stat,
        pvalue,
        crit_values,
    })
}

// ---------------------------------------------------------------------------
// Break-variance heteroskedasticity test
// ---------------------------------------------------------------------------

/// Round half to even, matching `numpy.round`.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        // Exactly halfway: round to the nearest even integer.
        let floor = x.floor();
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        r
    }
}

/// Alternative hypothesis for [`breakvar_heteroskedasticity_test`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakvarAlternative {
    /// Variance increases through the sample.
    Increasing,
    /// Variance decreases through the sample.
    Decreasing,
    /// Variance changes through the sample (the default).
    TwoSided,
}

/// Length of the test subsets, the `subset_length` argument of the reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubsetLength {
    /// A fraction in `(0, 1)` of the sample, e.g. `1/3` (the default).
    Fraction(f64),
    /// A fixed integer number of observations.
    Fixed(usize),
}

/// Break-variance heteroskedasticity test.
///
/// Mirrors the reference `breakvar_heteroskedasticity_test(resid,
/// subset_length, alternative, use_f)`. Tests whether the sum-of-squares in the
/// last subset differs from the first subset (analogous to a Goldfeld-Quandt
/// test). Returns `(stat, pvalue)`.
///
/// With `use_f = true` the statistic is compared against an `F(h, h)`
/// distribution; with `use_f = false` `h * H(h)` is compared against a
/// chi-squared distribution with `h` degrees of freedom.
pub fn breakvar_heteroskedasticity_test(
    resid: &Array1<f64>,
    subset_length: SubsetLength,
    alternative: BreakvarAlternative,
    use_f: bool,
) -> Result<(f64, f64)> {
    let nobs = resid.len();
    let h = match subset_length {
        SubsetLength::Fraction(f) => {
            if !(0.0 < f && f < 1.0) {
                return Err(Error::Value(
                    "fractional subset_length must be in (0, 1)".into(),
                ));
            }
            // numpy.round uses round-half-to-even ("banker's rounding").
            round_half_even(nobs as f64 * f) as usize
        }
        SubsetLength::Fixed(k) => {
            if k < 1 {
                return Err(Error::Value("subset_length must be >= 1".into()));
            }
            k
        }
    };
    if h < 2 || h > nobs {
        return Err(Error::Value(
            "subset has too few observations to compute the test".into(),
        ));
    }

    let sq: Array1<f64> = resid.mapv(|v| v * v);
    // numerator: last h squared residuals.
    let numer_sum: f64 = sq.slice(s![nobs - h..]).sum();
    let numer_dof = h as f64;
    // denominator: first h squared residuals.
    let denom_sum: f64 = sq.slice(s![..h]).sum();
    let denom_dof = h as f64;

    let mut test_statistic = numer_sum / denom_sum;

    let (pval_lower, pval_upper): (f64, f64) = if use_f {
        (
            f_cdf(test_statistic, numer_dof, denom_dof),
            f_sf(test_statistic, numer_dof, denom_dof),
        )
    } else {
        (
            solow_distributions::chi2_cdf(numer_dof * test_statistic, denom_dof),
            solow_distributions::chi2_sf(numer_dof * test_statistic, denom_dof),
        )
    };

    let pvalue = match alternative {
        BreakvarAlternative::Increasing => pval_upper,
        BreakvarAlternative::Decreasing => {
            // Reference inverts the statistic and recomputes the upper tail.
            let inv_stat = 1.0 / test_statistic;
            test_statistic = inv_stat;
            if use_f {
                f_sf(inv_stat, numer_dof, denom_dof)
            } else {
                solow_distributions::chi2_sf(numer_dof * inv_stat, denom_dof)
            }
        }
        BreakvarAlternative::TwoSided => 2.0 * pval_lower.min(pval_upper),
    };

    Ok((test_statistic, pvalue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn za_crit_monotone_pvalue() {
        // The p-value increases as the statistic increases (less negative).
        let (p_low, _) = za_crit(-6.0, ZaRegression::C);
        let (p_high, _) = za_crit(-3.0, ZaRegression::C);
        assert!(p_high > p_low);
    }

    #[test]
    fn range_unit_root_runs() {
        let x = Array1::from_vec(vec![
            1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5, 7.0,
        ]);
        let r = range_unit_root_test(&x).unwrap();
        assert!(r.stat >= 0.0);
        assert!((0.01..=0.95).contains(&r.pvalue));
    }

    #[test]
    fn breakvar_two_sided_symmetric_resid() {
        // Constant-variance residuals give a statistic near 1 and a large
        // two-sided p-value.
        let resid = Array1::from_vec(vec![
            0.5, -0.4, 0.3, -0.2, 0.45, -0.35, 0.25, -0.15, 0.4, -0.3, 0.2, -0.1,
        ]);
        let (stat, p) = breakvar_heteroskedasticity_test(
            &resid,
            SubsetLength::Fraction(1.0 / 3.0),
            BreakvarAlternative::TwoSided,
            true,
        )
        .unwrap();
        assert!(stat > 0.0);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn breakvar_subset_too_small_errors() {
        let resid = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(breakvar_heteroskedasticity_test(
            &resid,
            SubsetLength::Fixed(1),
            BreakvarAlternative::TwoSided,
            true
        )
        .is_err());
    }
}
