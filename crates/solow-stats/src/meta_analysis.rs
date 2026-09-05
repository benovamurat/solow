//! Meta-analysis — fixed-effect (inverse-variance weighting) and
//! random-effects (DerSimonian-Laird) pooled estimates with a
//! heterogeneity summary.

use solow_core::{Error, Result};

/// One study contribution to a meta-analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Study {
    /// Point estimate `θᵢ`.
    pub estimate: f64,
    /// Standard error `SEᵢ` (must be > 0).
    pub se: f64,
}

/// Pooled meta-analysis result.
#[derive(Clone, Debug, PartialEq)]
pub struct MetaResult {
    /// Pooled estimate.
    pub estimate: f64,
    /// Pooled standard error.
    pub se: f64,
    /// 95% confidence interval `(lower, upper)`.
    pub ci_95: (f64, f64),
    /// Cochran's Q statistic.
    pub q: f64,
    /// Q's degrees of freedom.
    pub df: usize,
    /// Q's p-value against χ²(df).
    pub q_pvalue: f64,
    /// Higgins' I² — fraction of total variance due to heterogeneity.
    pub i_squared: f64,
    /// Between-study variance τ² (0 under a fixed-effect model).
    pub tau_squared: f64,
    /// Per-study inverse-variance weights.
    pub weights: Vec<f64>,
    /// Model kind.
    pub model: MetaModel,
}

/// Pooling model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetaModel {
    /// Inverse-variance-weighted fixed effect.
    FixedEffect,
    /// DerSimonian-Laird random effects.
    RandomEffects,
}

/// Fixed-effect meta-analysis.
pub fn meta_fixed_effect(studies: &[Study]) -> Result<MetaResult> {
    validate(studies)?;
    let (q, weights, pooled, pooled_se) = fixed_effect_stats(studies);
    let ci = ci_95(pooled, pooled_se);
    let df = studies.len().saturating_sub(1);
    let q_pvalue = chi2_survival(q, df as f64);
    let i2 = i_squared(q, df as f64);
    Ok(MetaResult {
        estimate: pooled,
        se: pooled_se,
        ci_95: ci,
        q,
        df,
        q_pvalue,
        i_squared: i2,
        tau_squared: 0.0,
        weights,
        model: MetaModel::FixedEffect,
    })
}

/// DerSimonian-Laird random-effects meta-analysis.
pub fn meta_random_effects(studies: &[Study]) -> Result<MetaResult> {
    validate(studies)?;
    let (q, weights_fe, pooled_fe, _) = fixed_effect_stats(studies);
    let df = studies.len().saturating_sub(1) as f64;
    // τ² = max(0, (Q − df) / (Σw − Σw² / Σw))
    let sum_w: f64 = weights_fe.iter().sum();
    let sum_w2: f64 = weights_fe.iter().map(|w| w * w).sum();
    let c = sum_w - sum_w2 / sum_w.max(1e-30);
    let tau2 = if c > 0.0 {
        ((q - df) / c).max(0.0)
    } else {
        0.0
    };
    let mut weights_re = Vec::with_capacity(studies.len());
    for s in studies {
        weights_re.push(1.0 / (s.se * s.se + tau2));
    }
    let sum_w_re: f64 = weights_re.iter().sum();
    let mut pooled = 0.0_f64;
    for (i, s) in studies.iter().enumerate() {
        pooled += weights_re[i] * s.estimate;
    }
    pooled /= sum_w_re.max(1e-30);
    let pooled_se = (1.0 / sum_w_re.max(1e-30)).sqrt();
    let ci = ci_95(pooled, pooled_se);
    let q_pvalue = chi2_survival(q, df);
    let i2 = i_squared(q, df);
    Ok(MetaResult {
        estimate: pooled,
        se: pooled_se,
        ci_95: ci,
        q,
        df: df as usize,
        q_pvalue,
        i_squared: i2,
        tau_squared: tau2,
        weights: weights_re,
        model: MetaModel::RandomEffects,
    })
}

fn validate(studies: &[Study]) -> Result<()> {
    if studies.len() < 2 {
        return Err(Error::Value("meta_analysis: need ≥ 2 studies".into()));
    }
    for s in studies {
        if !(s.se > 0.0 && s.se.is_finite()) {
            return Err(Error::Value("meta_analysis: SEs must be finite and > 0".into()));
        }
    }
    Ok(())
}

fn fixed_effect_stats(studies: &[Study]) -> (f64, Vec<f64>, f64, f64) {
    let mut weights = Vec::with_capacity(studies.len());
    let mut sum_w = 0.0_f64;
    let mut sum_wy = 0.0_f64;
    for s in studies {
        let w = 1.0 / (s.se * s.se);
        weights.push(w);
        sum_w += w;
        sum_wy += w * s.estimate;
    }
    let pooled = sum_wy / sum_w;
    let pooled_se = (1.0 / sum_w).sqrt();
    // Cochran's Q.
    let mut q = 0.0_f64;
    for (i, s) in studies.iter().enumerate() {
        q += weights[i] * (s.estimate - pooled).powi(2);
    }
    (q, weights, pooled, pooled_se)
}

fn ci_95(estimate: f64, se: f64) -> (f64, f64) {
    (estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se)
}

fn i_squared(q: f64, df: f64) -> f64 {
    if q <= df || df <= 0.0 {
        return 0.0;
    }
    ((q - df) / q).clamp(0.0, 1.0)
}

fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    1.0 - lower_regularised_gamma(df / 2.0, x / 2.0)
}

fn lower_regularised_gamma(s: f64, x: f64) -> f64 {
    if x < 0.0 || s <= 0.0 {
        return 0.0;
    }
    if x < s + 1.0 {
        gamma_series(s, x)
    } else {
        1.0 - gamma_continued_fraction(s, x)
    }
}

fn gamma_series(s: f64, x: f64) -> f64 {
    let mut sum = 1.0 / s;
    let mut term = sum;
    for n in 1..200 {
        term *= x / (s + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 3e-15 {
            break;
        }
    }
    sum * (-x + s * x.ln() - ln_gamma(s)).exp()
}

fn gamma_continued_fraction(s: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - s;
    let mut c = 1.0 / 1e-300;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - s);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = b + an / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 3e-15 {
            break;
        }
    }
    (-x + s * x.ln() - ln_gamma(s)).exp() * h
}

fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let cof = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_5e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = cof[0];
        let t = x + g + 0.5;
        for (i, &c) in cof.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_effect_returns_the_weighted_average() {
        let studies = vec![
            Study { estimate: 0.5, se: 0.1 },
            Study { estimate: 0.6, se: 0.1 },
            Study { estimate: 0.55, se: 0.15 },
        ];
        let r = meta_fixed_effect(&studies).unwrap();
        assert!(r.estimate > 0.5 && r.estimate < 0.6);
        assert!(r.se > 0.0);
        assert!(r.i_squared >= 0.0 && r.i_squared <= 1.0);
    }

    #[test]
    fn random_effects_widens_ci_versus_fixed_effect_under_heterogeneity() {
        let studies = vec![
            Study { estimate: 0.5, se: 0.05 },
            Study { estimate: 1.5, se: 0.05 },
            Study { estimate: -0.2, se: 0.05 },
        ];
        let fe = meta_fixed_effect(&studies).unwrap();
        let re = meta_random_effects(&studies).unwrap();
        assert!(re.se >= fe.se);
        assert!(re.tau_squared > 0.0);
    }
}
