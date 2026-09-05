//! Standard bivariate-normal CDF via the Drezner-Wesolowsky / Genz scheme.
//!
//! Computes `P(X <= h, Y <= k)` for a standard bivariate normal with
//! correlation `r`. The implementation follows the widely used Fortran
//! routine `BVND` (Genz, after Drezner-Wesolowsky), which evaluates the
//! tail probability `L(h, k; r) = P(X >= h, Y >= k)` by Gauss-Legendre
//! quadrature with the number of nodes adapted to `|r|`. The CDF is then
//! recovered from `Phi(h) + Phi(k) - 1 + L(h, k; r)`.

use solow_distributions::norm_cdf;

const TWO_PI: f64 = std::f64::consts::TAU;

/// 6-point Gauss-Legendre abscissae/weights on the half interval (used for
/// `|r| < 0.3`).
const W6: [f64; 3] = [
    0.171_324_492_379_170_06,
    0.360_761_573_048_138_6,
    0.467_913_934_572_691_3,
];
const X6: [f64; 3] = [
    0.932_469_514_203_152,
    0.661_209_386_466_264_5,
    0.238_619_186_083_196_9,
];

/// 12-point Gauss-Legendre (used for `0.3 <= |r| < 0.75`).
const W12: [f64; 6] = [
    0.047_175_336_386_511_82,
    0.106_939_325_995_318_43,
    0.160_078_328_543_346_2,
    0.203_167_426_723_065_9,
    0.233_492_536_538_354_8,
    0.249_147_045_813_402_8,
];
const X12: [f64; 6] = [
    0.981_560_634_246_719_2,
    0.904_117_256_370_474_9,
    0.769_902_674_194_304_7,
    0.587_317_954_286_617_4,
    0.367_831_498_998_180_2,
    0.125_233_408_511_468_9,
];

/// 20-point Gauss-Legendre (used for `|r| >= 0.75`).
const W20: [f64; 10] = [
    0.017_614_007_139_152_118,
    0.040_601_429_800_386_94,
    0.062_672_048_334_109_06,
    0.083_276_741_576_704_75,
    0.101_930_119_817_240_43,
    0.118_194_531_961_518_42,
    0.131_688_638_449_176_63,
    0.142_096_109_318_382_05,
    0.149_172_986_472_603_75,
    0.152_753_387_130_725_85,
];
const X20: [f64; 10] = [
    0.993_128_599_185_094_9,
    0.963_971_927_277_913_8,
    0.912_234_428_251_326,
    0.839_116_971_822_218_8,
    0.746_331_906_460_150_8,
    0.636_053_680_726_515_2,
    0.510_867_001_950_827_1,
    0.373_706_088_715_419_6,
    0.227_785_851_141_645_1,
    0.076_526_521_133_497_31,
];

/// Tail probability `L(h, k; r) = P(X >= h, Y >= k)` for a standard
/// bivariate normal with correlation `r`. This is Genz's `BVND` routine.
fn bvnu(h: f64, k: f64, r: f64) -> f64 {
    let (w, x): (&[f64], &[f64]) = if r.abs() < 0.3 {
        (&W6, &X6)
    } else if r.abs() < 0.75 {
        (&W12, &X12)
    } else {
        (&W20, &X20)
    };

    let hk = h * k;
    let mut bvn = 0.0_f64;

    if r.abs() < 0.925 {
        // Central region: a single Gauss-Legendre quadrature in the angle.
        let hs = 0.5 * (h * h + k * k);
        let asr = 0.5 * r.asin();
        for i in 0..w.len() {
            for sign in [-1.0_f64, 1.0] {
                let sn = (asr * (1.0 + sign * x[i])).sin();
                bvn += w[i] * ((sn * hk - hs) / (1.0 - sn * sn)).exp();
            }
        }
        bvn = bvn * asr / TWO_PI + norm_cdf(-h) * norm_cdf(-k);
    } else {
        // High-correlation region: Drezner-Wesolowsky expansion plus a
        // correction quadrature (Genz BVND).
        let mut k = k;
        let mut hk = hk;
        if r < 0.0 {
            k = -k;
            hk = -hk;
        }
        if r.abs() < 1.0 {
            let mut a = ((1.0 - r) * (1.0 + r)).sqrt();
            let bs = (h - k) * (h - k);
            let c = (4.0 - hk) / 8.0;
            let d = (12.0 - hk) / 16.0;
            let asr = -(bs / (a * a) + hk) / 2.0;
            if asr > -100.0 {
                bvn = a
                    * asr.exp()
                    * (1.0 - c * (bs - a * a) * (1.0 - d * bs / 5.0) / 3.0
                        + c * d * a.powi(4) / 5.0);
            }
            if -hk < 100.0 {
                let b = bs.sqrt();
                bvn -= (-hk / 2.0).exp()
                    * TWO_PI.sqrt()
                    * norm_cdf(-b / a)
                    * b
                    * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
            }
            a *= 0.5;
            for i in 0..w.len() {
                for sign in [-1.0_f64, 1.0] {
                    let xs = (a * (1.0 + sign * x[i])).powi(2);
                    let rs = (1.0 - xs).sqrt();
                    let asr = -(bs / xs + hk) / 2.0;
                    if asr > -100.0 {
                        bvn += a
                            * w[i]
                            * asr.exp()
                            * ((-hk * xs / (2.0 * (1.0 + rs) * (1.0 + rs))).exp() / rs
                                - (1.0 + c * xs * (1.0 + d * xs)));
                    }
                }
            }
            bvn = -bvn / TWO_PI;
        }
        if r > 0.0 {
            bvn += norm_cdf(-h.max(k));
        } else {
            bvn = -bvn;
            if k > h {
                bvn += norm_cdf(k) - norm_cdf(h);
            }
        }
    }
    bvn.clamp(0.0, 1.0)
}

/// Standard bivariate-normal CDF `P(X <= h, Y <= k)` with correlation `r`.
///
/// Accurate to roughly 1e-14 in the central region; the high-correlation
/// branch is accurate to about 1e-10.
pub fn bvn_cdf(h: f64, k: f64, r: f64) -> f64 {
    // P(X<=h, Y<=k) = Phi(h) + Phi(k) - 1 + P(X>=h, Y>=k).
    let l = bvnu(h, k, r);
    (norm_cdf(h) + norm_cdf(k) - 1.0 + l).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solow_distributions::norm_cdf;

    #[test]
    fn independence_factorises() {
        for h in [-1.5, 0.0, 0.8] {
            for k in [-0.7, 0.0, 1.2] {
                let got = bvn_cdf(h, k, 0.0);
                let want = norm_cdf(h) * norm_cdf(k);
                assert!((got - want).abs() < 1e-12, "h={h} k={k}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn perfect_positive_correlation_limit() {
        // As r -> 1, C(h,k) -> min(Phi(h), Phi(k)).
        for (h, k) in [(0.3, 0.7), (-0.5, 0.2), (1.0, -0.4)] {
            let got = bvn_cdf(h, k, 0.999_999);
            let want = norm_cdf(h).min(norm_cdf(k));
            assert!((got - want).abs() < 1e-4, "h={h} k={k}: {got} vs {want}");
        }
    }

    #[test]
    fn symmetry_in_arguments() {
        for r in [-0.6, 0.2, 0.85] {
            let a = bvn_cdf(0.4, -0.3, r);
            let b = bvn_cdf(-0.3, 0.4, r);
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn marginal_recovery() {
        // P(X<=h, Y<inf) = Phi(h).
        let big = 8.0;
        for h in [-1.0, 0.0, 1.3] {
            for r in [-0.5, 0.3, 0.9] {
                let got = bvn_cdf(h, big, r);
                assert!((got - norm_cdf(h)).abs() < 1e-9, "h={h} r={r}: {got}");
            }
        }
    }
}
