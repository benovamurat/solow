//! Axis scales (linear / log / symlog), tick locators, and tick formatters.

/// The transform applied along an axis before mapping data to pixels.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Scale {
    /// Identity transform.
    #[default]
    Linear,
    /// Base-10 logarithm. Non-positive data is clamped away from zero.
    Log,
    /// Symmetric log: linear within `[-linthresh, linthresh]`, logarithmic
    /// outside. Handles data that spans zero and both signs.
    Symlog {
        /// Half-width of the central linear region (must be > 0).
        linthresh: f64,
    },
}

impl Scale {
    /// Map a data value into the scale's transformed coordinate.
    pub fn forward(&self, v: f64) -> f64 {
        match *self {
            Scale::Linear => v,
            Scale::Log => {
                let v = v.max(f64::MIN_POSITIVE);
                v.log10()
            }
            Scale::Symlog { linthresh } => {
                let lt = linthresh.abs().max(f64::MIN_POSITIVE);
                if v.abs() <= lt {
                    v / lt
                } else {
                    let sign = v.signum();
                    sign * (1.0 + (v.abs() / lt).log10())
                }
            }
        }
    }

    /// Is this a non-linear scale (so a clamped, positive domain matters)?
    pub fn is_log(&self) -> bool {
        matches!(self, Scale::Log)
    }

    /// Clamp a `[lo, hi]` data range to a domain valid for the scale and
    /// guarantee `lo < hi`.
    pub fn sanitize_range(&self, lo: f64, hi: f64) -> (f64, f64) {
        let (mut lo, mut hi) = (lo.min(hi), lo.max(hi));
        if self.is_log() {
            if hi <= 0.0 {
                hi = 1.0;
            }
            if lo <= 0.0 {
                lo = (hi / 1000.0).min(hi * 0.1).max(f64::MIN_POSITIVE);
            }
        }
        if lo == hi {
            if self.is_log() {
                lo /= 3.16;
                hi *= 3.16;
            } else {
                lo -= 0.5;
                hi += 0.5;
            }
        }
        (lo, hi)
    }

    /// Compute tick locations spanning `[lo, hi]` for this scale.
    pub fn ticks(&self, lo: f64, hi: f64, target: usize) -> Vec<f64> {
        match *self {
            Scale::Linear => nice_ticks(lo, hi, target),
            Scale::Log => log_ticks(lo, hi),
            Scale::Symlog { linthresh } => symlog_ticks(lo, hi, linthresh),
        }
    }

    /// Format a tick value as a label string under this scale.
    pub fn format(&self, v: f64) -> String {
        match self {
            Scale::Log => fmt_log_tick(v),
            _ => fmt_tick(v),
        }
    }
}

/// Round `x` to a "nice" number for axis ticks.
pub fn nice_num(x: f64, round: bool) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let exp = x.log10().floor();
    let f = x / 10f64.powf(exp);
    let nf = if round {
        if f < 1.5 {
            1.0
        } else if f < 3.0 {
            2.0
        } else if f < 7.0 {
            5.0
        } else {
            10.0
        }
    } else if f <= 1.0 {
        1.0
    } else if f <= 2.0 {
        2.0
    } else if f <= 5.0 {
        5.0
    } else {
        10.0
    };
    nf * 10f64.powf(exp)
}

/// Produce up to ~`n` "nice" tick locations spanning `[min, max]`.
pub fn nice_ticks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if !(min.is_finite() && max.is_finite()) || min == max {
        return vec![min];
    }
    let range = nice_num(max - min, false);
    let step = nice_num(range / (n.max(2) - 1) as f64, true);
    if step <= 0.0 {
        return vec![min, max];
    }
    let lo = (min / step).floor() * step;
    let hi = (max / step).ceil() * step;
    let mut t = lo;
    let mut out = Vec::new();
    while t <= hi + 0.5 * step {
        let v = if t.abs() < 1e-12 * step { 0.0 } else { t };
        out.push(v);
        t += step;
        if out.len() > 1000 {
            break;
        }
    }
    out
}

/// Log-spaced ticks at the decade boundaries (10^k) covering `[lo, hi]`.
fn log_ticks(lo: f64, hi: f64) -> Vec<f64> {
    let lo = lo.max(f64::MIN_POSITIVE);
    let hi = hi.max(lo * 10.0);
    let e0 = lo.log10().floor() as i32;
    let e1 = hi.log10().ceil() as i32;
    let mut out = Vec::new();
    for e in e0..=e1 {
        out.push(10f64.powi(e));
    }
    // If we span only one or two decades, add the 2/5 minor decade marks so
    // the axis is not nearly empty.
    if e1 - e0 <= 2 {
        let mut dense = Vec::new();
        for e in e0..=e1 {
            let base = 10f64.powi(e);
            for m in [1.0, 2.0, 5.0] {
                dense.push(base * m);
            }
        }
        dense.retain(|&v| v >= lo * 0.999 && v <= hi * 1.001);
        if dense.len() >= 2 {
            return dense;
        }
    }
    out
}

/// Symlog ticks: decade ticks on each tail plus the linear-region endpoints.
fn symlog_ticks(lo: f64, hi: f64, linthresh: f64) -> Vec<f64> {
    let lt = linthresh.abs().max(f64::MIN_POSITIVE);
    let mut out = vec![0.0];
    if hi > lt {
        let e1 = hi.log10().ceil() as i32;
        let e0 = lt.log10().floor() as i32;
        for e in e0..=e1 {
            let v = 10f64.powi(e);
            if v > lt && v <= hi * 1.001 {
                out.push(v);
            }
        }
        out.push(lt);
    }
    if lo < -lt {
        let e1 = (-lo).log10().ceil() as i32;
        let e0 = lt.log10().floor() as i32;
        for e in e0..=e1 {
            let v = 10f64.powi(e);
            if v > lt && v <= (-lo) * 1.001 {
                out.push(-v);
            }
        }
        out.push(-lt);
    }
    out.sort_by(|a, b| a.total_cmp(b));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    out
}

/// Format a plain tick value.
pub fn fmt_tick(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if !(1e-3..1e4).contains(&a) {
        format!("{v:.1e}")
    } else {
        let s = format!("{v:.4}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    }
}

/// Format a log-scale tick. Pure powers of ten render compactly.
fn fmt_log_tick(v: f64) -> String {
    if v <= 0.0 {
        return fmt_tick(v);
    }
    let e = v.log10();
    let er = e.round();
    if (e - er).abs() < 1e-6 {
        let ei = er as i32;
        if (-3..=4).contains(&ei) {
            return fmt_tick(v);
        }
        return format!("1e{ei}");
    }
    fmt_tick(v)
}
