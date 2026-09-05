//! Seasonal-Trend decomposition using LOESS (STL).
//!
//! A faithful port of the reference `STL(endog, period).fit()` (the Cleveland
//! et al. 1990 algorithm) for the non-robust default configuration. The window
//! lengths follow the reference defaults: `seasonal = 7`, `trend` derived from
//! the period and seasonal window, `low_pass` the smallest odd integer greater
//! than the period; all LOESS degrees are 1 and all jumps are 1. With
//! `robust = false` the fit performs a single outer iteration with five inner
//! iterations and no robustness weights.
//!
//! The loops below deliberately mirror the reference's 1-based, index-driven
//! Fortran/Cython kernel (the loess weighting uses the running index directly),
//! so a handful of index-based clippy lints are silenced module-wide.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_div_ceil)]

use ndarray::Array1;
use solow_core::error::{Error, Result};

/// The three components produced by [`Stl::fit`].
#[derive(Debug, Clone)]
pub struct StlResult {
    /// Estimated trend component.
    pub trend: Array1<f64>,
    /// Estimated seasonal component.
    pub seasonal: Array1<f64>,
    /// Residual component (`endog - trend - seasonal`).
    pub resid: Array1<f64>,
}

/// Seasonal-Trend decomposition using LOESS.
#[derive(Debug, Clone)]
pub struct Stl {
    endog: Vec<f64>,
    period: usize,
    seasonal: usize,
    trend: usize,
    low_pass: usize,
    seasonal_deg: i32,
    trend_deg: i32,
    low_pass_deg: i32,
    seasonal_jump: usize,
    trend_jump: usize,
    low_pass_jump: usize,
}

fn make_odd(mut v: usize) -> usize {
    if v % 2 == 0 {
        v += 1;
    }
    v
}

impl Stl {
    /// Create an STL model for `endog` with the given seasonal `period`, using
    /// the reference default windows and degrees.
    pub fn new(endog: Array1<f64>, period: usize) -> Result<Self> {
        if period < 2 {
            return Err(Error::Value("period must be at least 2".into()));
        }
        let n = endog.len();
        if n < 2 * period {
            return Err(Error::Value(
                "endog must have at least two full periods".into(),
            ));
        }
        solow_core::tools::ensure_all_finite(&endog.view(), "endog")?;
        let seasonal = 7usize; // reference default, odd >= 3
                               // trend = ceil(1.5 * period / (1 - 1.5 / seasonal)) made odd.
        let trend_f = 1.5 * period as f64 / (1.0 - 1.5 / seasonal as f64);
        let trend = make_odd(trend_f.ceil() as usize);
        // low_pass = smallest odd integer > period.
        let low_pass = make_odd(period + 1);
        Ok(Self {
            endog: endog.to_vec(),
            period,
            seasonal,
            trend,
            low_pass,
            seasonal_deg: 1,
            trend_deg: 1,
            low_pass_deg: 1,
            seasonal_jump: 1,
            trend_jump: 1,
            low_pass_jump: 1,
        })
    }

    /// Override the seasonal smoother length (must be odd and `>= 3`).
    pub fn with_seasonal(mut self, seasonal: usize) -> Self {
        self.seasonal = seasonal;
        self
    }

    /// Override the trend smoother length.
    pub fn with_trend(mut self, trend: usize) -> Self {
        self.trend = trend;
        self
    }

    /// Override the low-pass smoother length.
    pub fn with_low_pass(mut self, low_pass: usize) -> Self {
        self.low_pass = low_pass;
        self
    }

    /// Fit the decomposition (non-robust: 5 inner iterations, no outer loop).
    pub fn fit(&self) -> Result<StlResult> {
        let mut inner = StlInner::new(self);
        inner.run(5);
        let n = self.endog.len();
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            resid[i] = self.endog[i] - inner.season[i] - inner.trend[i];
        }
        Ok(StlResult {
            trend: Array1::from(inner.trend.clone()),
            seasonal: Array1::from(inner.season.clone()),
            resid,
        })
    }
}

/// Working state mirroring the reference Cython implementation. Arrays use the
/// same layout: `work[0..=4]` plus the seasonal working buffer.
struct StlInner<'a> {
    cfg: &'a Stl,
    nobs: usize,
    trend: Vec<f64>,
    season: Vec<f64>,
    rw: Vec<f64>,
    // work[5][nobs + 2*period]
    work: [Vec<f64>; 5],
    season_buf: Vec<f64>, // _season scratch used by _est weights
    use_rw: bool,
}

impl<'a> StlInner<'a> {
    fn new(cfg: &'a Stl) -> Self {
        let nobs = cfg.endog.len();
        let np = cfg.period;
        let len = nobs + 2 * np;
        StlInner {
            cfg,
            nobs,
            trend: vec![0.0; nobs],
            season: vec![0.0; nobs],
            rw: vec![1.0; nobs],
            work: [
                vec![0.0; len],
                vec![0.0; len],
                vec![0.0; len],
                vec![0.0; len],
                vec![0.0; len],
            ],
            season_buf: vec![0.0; len],
            use_rw: false,
        }
    }

    fn run(&mut self, inner_iter: usize) {
        // Non-robust: outer_iter = 0, so one onestep call then stop.
        self.use_rw = false;
        for v in self.trend.iter_mut() {
            *v = 0.0;
        }
        for v in self.season.iter_mut() {
            *v = 0.0;
        }
        for v in self.rw.iter_mut() {
            *v = 1.0;
        }
        self.onestp(inner_iter);
    }

    fn onestp(&mut self, inner_iter: usize) {
        let n = self.nobs;
        let np = self.cfg.period;
        for _ in 0..inner_iter {
            // Detrend.
            for i in 0..n {
                self.work[0][i] = self.cfg.endog[i] - self.trend[i];
            }
            // Seasonal smoothing -> writes work[1] (length n + 2*np).
            self.ss();
            // Low-pass filter of work[1] -> work[2].
            self.fts();
            // LOESS of work[2] (length n) with low_pass window -> work[0].
            {
                let src = self.work[2].clone();
                let mut ys = vec![0.0; n];
                let mut res = vec![0.0; self.work[3].len()];
                let rw = self.work[3].clone();
                self.ess(
                    &src,
                    n,
                    self.cfg.low_pass,
                    self.cfg.low_pass_deg,
                    self.cfg.low_pass_jump,
                    false,
                    &rw,
                    &mut ys,
                    &mut res,
                );
                for i in 0..n {
                    self.work[0][i] = ys[i];
                }
            }
            // Deseasonalize: season = work[1][np + i] - work[0][i].
            for i in 0..n {
                self.season[i] = self.work[1][np + i] - self.work[0][i];
                self.work[0][i] = self.cfg.endog[i] - self.season[i];
            }
            // Trend smoothing -> trend.
            {
                let src = self.work[0].clone();
                let mut ys = self.trend.clone();
                let mut res = self.work[2].clone();
                let rw = self.rw.clone();
                self.ess(
                    &src,
                    n,
                    self.cfg.trend,
                    self.cfg.trend_deg,
                    self.cfg.trend_jump,
                    self.use_rw,
                    &rw,
                    &mut ys,
                    &mut res,
                );
                self.trend = ys;
            }
        }
    }

    /// Seasonal smoothing of cycle-subseries (reference `_ss`).
    fn ss(&mut self) {
        let n = self.nobs;
        let np = self.cfg.period;
        let ns = self.cfg.seasonal;
        let isdeg = self.cfg.seasonal_deg;
        let nsjump = self.cfg.seasonal_jump;
        let userw = self.use_rw;

        // y = work[0]; season output = work[1].
        let y = self.work[0].clone();
        let rw = self.rw.clone();

        let mut work1 = vec![0.0; n + 2 * np];
        let mut work2 = vec![0.0; n + 2 * np];
        let mut work3 = vec![0.0; n + 2 * np];
        let mut work4 = self.season_buf.clone();

        for j in 0..np {
            let k = (n - (j + 1)) / np + 1;
            for i in 0..k {
                work1[i] = y[i * np + j];
            }
            if userw {
                for i in 0..k {
                    work3[i] = rw[i * np + j];
                }
            }
            // LOESS smooth subseries into work2[1..].
            {
                let src = work1.clone();
                let mut ys = vec![0.0; k.max(1)];
                let mut res = work4.clone();
                self.ess(&src, k, ns, isdeg, nsjump, userw, &work3, &mut ys, &mut res);
                for (i, &v) in ys.iter().enumerate().take(k) {
                    work2[1 + i] = v;
                }
                work4 = res;
            }
            // Extend by one point before.
            let xs = 0i64;
            let nright = ns.min(k);
            let before = self.est(
                &work1,
                k,
                ns,
                isdeg,
                xs,
                1,
                nright as i64,
                &mut work4,
                userw,
                &work3,
            );
            work2[0] = if before.is_nan() { work2[1] } else { before };
            // Extend by one point after.
            let xs = (k + 1) as i64;
            let nleft = if k >= ns { (k - ns + 1).max(1) } else { 1 };
            let after = self.est(
                &work1,
                k,
                ns,
                isdeg,
                xs,
                nleft as i64,
                k as i64,
                &mut work4,
                userw,
                &work3,
            );
            work2[k + 1] = if after.is_nan() { work2[k] } else { after };
            // Store into season output (work[1]).
            for m in 0..(k + 2) {
                self.work[1][m * np + j] = work2[m];
            }
        }
        self.season_buf = work4;
    }

    /// Low-pass filter (reference `_fts`): three moving averages.
    fn fts(&mut self) {
        let np = self.cfg.period;
        let n = self.nobs + 2 * np;
        // x = work[1], trend = work[2], work = work[0].
        let x = self.work[1].clone();
        let mut trend = vec![0.0; self.work[2].len()];
        let mut work = vec![0.0; self.work[0].len()];
        ma(&x, n, np, &mut trend);
        let tmp = trend.clone();
        ma(&tmp, n - np + 1, np, &mut work);
        let tmp2 = work.clone();
        ma(&tmp2, n - 2 * np + 2, 3, &mut trend);
        self.work[2] = trend;
    }

    /// LOESS smoothing over `n` points (reference `_ess`).
    #[allow(clippy::too_many_arguments)]
    fn ess(
        &self,
        y: &[f64],
        n: usize,
        len_: usize,
        ideg: i32,
        njump: usize,
        userw: bool,
        rw: &[f64],
        ys: &mut [f64],
        res: &mut [f64],
    ) {
        if n < 2 {
            ys[0] = y[0];
            return;
        }
        let newnj = njump.min(n - 1);
        if len_ >= n {
            let nleft = 1i64;
            let nright = n as i64;
            let mut i = 0usize;
            while i < n {
                ys[i] = self.est(
                    y,
                    n,
                    len_,
                    ideg,
                    (i + 1) as i64,
                    nleft,
                    nright,
                    res,
                    userw,
                    rw,
                );
                if ys[i].is_nan() {
                    ys[i] = y[i];
                }
                i += newnj;
            }
        } else if newnj == 1 {
            let nsh = (len_ + 2) / 2;
            let mut nleft = 1i64;
            let mut nright = len_ as i64;
            for i in 0..n {
                if (i + 1) > nsh && nright != n as i64 {
                    nleft += 1;
                    nright += 1;
                }
                ys[i] = self.est(
                    y,
                    n,
                    len_,
                    ideg,
                    (i + 1) as i64,
                    nleft,
                    nright,
                    res,
                    userw,
                    rw,
                );
                if ys[i].is_nan() {
                    ys[i] = y[i];
                }
            }
        } else {
            let nsh = (len_ + 1) / 2;
            let mut nleft;
            let mut nright;
            let mut i = 0usize;
            while i < n {
                if (i + 1) < nsh {
                    nleft = 1i64;
                    nright = len_ as i64;
                } else if (i + 1) >= (n - nsh + 1) {
                    nleft = (n - len_ + 1) as i64;
                    nright = n as i64;
                } else {
                    nleft = (i + 1 - nsh + 1) as i64;
                    nright = (len_ + i + 1 - nsh) as i64;
                }
                ys[i] = self.est(
                    y,
                    n,
                    len_,
                    ideg,
                    (i + 1) as i64,
                    nleft,
                    nright,
                    res,
                    userw,
                    rw,
                );
                if ys[i].is_nan() {
                    ys[i] = y[i];
                }
                i += newnj;
            }
        }

        if newnj == 1 {
            return;
        }

        // Linear interpolation between jumped points.
        let mut i = 0usize;
        while i < n.saturating_sub(newnj) {
            let delta = (ys[i + newnj] - ys[i]) / newnj as f64;
            for j in i..(i + newnj) {
                ys[j] = ys[i] + delta * ((j + 1) as f64 - (i + 1) as f64);
            }
            i += newnj;
        }
        // Fill final segment.
        let k = ((n - 1) / newnj) * newnj + 1;
        if k != n {
            // recompute the final point with the last nleft/nright.
            let (nleft, nright) = last_window(n, len_, newnj);
            ys[n - 1] = self.est(y, n, len_, ideg, n as i64, nleft, nright, res, userw, rw);
            if ys[n - 1].is_nan() {
                ys[n - 1] = y[n - 1];
            }
            if k != (n - 1) {
                let delta = (ys[n - 1] - ys[k - 1]) / (n - k) as f64;
                for j in k..n {
                    ys[j] = ys[k - 1] + delta * ((j + 1) as f64 - k as f64);
                }
            }
        }
    }

    /// LOESS local estimate at position `xs` (reference `_est`).
    #[allow(clippy::too_many_arguments)]
    fn est(
        &self,
        y: &[f64],
        n: usize,
        len_: usize,
        ideg: i32,
        xs: i64,
        nleft: i64,
        nright: i64,
        w: &mut [f64],
        userw: bool,
        rw: &[f64],
    ) -> f64 {
        let rng = n as f64 - 1.0;
        let mut h = ((xs - nleft).max(nright - xs)) as f64;
        if len_ > n {
            h += ((len_ - n) / 2) as f64;
        }
        let h9 = 0.999 * h;
        let h1 = 0.001 * h;
        let mut a = 0.0;
        let lo = (nleft - 1) as usize;
        let hi = nright as usize; // exclusive
        for j in lo..hi {
            w[j] = 0.0;
            let r = ((j as i64 + 1 - xs).abs()) as f64;
            if r <= h9 {
                if r <= h1 {
                    w[j] = 1.0;
                } else {
                    w[j] = (1.0 - (r / h).powi(3)).powi(3);
                }
                if userw {
                    w[j] *= rw[j];
                }
                a += w[j];
            }
        }
        if a <= 0.0 {
            return f64::NAN;
        }
        for j in lo..hi {
            w[j] /= a;
        }
        if h > 0.0 && ideg > 0 {
            let mut amean = 0.0;
            for j in lo..hi {
                amean += w[j] * (j as f64 + 1.0);
            }
            let mut b = xs as f64 - amean;
            let mut c = 0.0;
            for j in lo..hi {
                c += w[j] * (j as f64 + 1.0 - amean).powi(2);
            }
            if c.sqrt() > 0.001 * rng {
                b /= c;
                for j in lo..hi {
                    w[j] *= b * (j as f64 + 1.0 - amean) + 1.0;
                }
            }
        }
        let mut ys = 0.0;
        for j in lo..hi {
            ys += w[j] * y[j];
        }
        ys
    }
}

/// Reproduce the `nleft`/`nright` of the last interior `_est` call so the final
/// segment fill matches the reference (which reuses the loop's final window).
fn last_window(n: usize, len_: usize, newnj: usize) -> (i64, i64) {
    let nsh = (len_ + 1) / 2;
    // Last index visited by the jumping loop strictly below n.
    let mut last_i = 0usize;
    let mut i = 0usize;
    while i < n {
        last_i = i;
        i += newnj;
    }
    let ii = last_i + 1; // 1-based
    if ii < nsh {
        (1, len_ as i64)
    } else if ii >= (n - nsh + 1) {
        ((n - len_ + 1) as i64, n as i64)
    } else {
        ((ii - nsh + 1) as i64, (len_ + ii - nsh) as i64)
    }
}

/// Moving average of length `len_` over the first `n` points (reference `_ma`).
fn ma(x: &[f64], n: usize, len_: usize, ave: &mut [f64]) {
    let newn = n - len_ + 1;
    let flen = len_ as f64;
    let mut v = 0.0;
    for &xi in x.iter().take(len_) {
        v += xi;
    }
    ave[0] = v / flen;
    // Slide the length-`len_` window forward one step at a time: the entering
    // sample is `x[m + len_]` and the leaving sample is `x[m]`.
    for (m, ave_j) in ave.iter_mut().take(newn).enumerate().skip(1) {
        v += x[(m - 1) + len_] - x[m - 1];
        *ave_j = v / flen;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn default_windows_match_reference_rules() {
        let y = Array1::from_shape_fn(48, |i| i as f64);
        let stl = Stl::new(y, 12).unwrap();
        assert_eq!(stl.seasonal, 7);
        assert_eq!(stl.low_pass, 13);
        // trend = ceil(1.5*12/(1-1.5/7)) -> 23.
        assert_eq!(stl.trend, 23);
    }

    #[test]
    fn components_sum_to_series() {
        let n = 60;
        let pattern = [2.0, -1.0, -3.0, 2.0];
        let y = Array1::from_shape_fn(n, |i| 10.0 + 0.3 * i as f64 + pattern[i % 4]);
        let stl = Stl::new(y.clone(), 4).unwrap();
        let r = stl.fit().unwrap();
        for i in 0..n {
            let recon = r.trend[i] + r.seasonal[i] + r.resid[i];
            assert!((recon - y[i]).abs() < 1e-9, "reconstruction at {i}");
        }
    }

    #[test]
    fn rejects_short_series() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(Stl::new(y, 4).is_err());
    }
}
