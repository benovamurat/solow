//! Scalar root finding and bounded minimization, faithful re-implementations of
//! the routines the reference relies on.
//!
//! * [`brentq`] mirrors the classic Brent root finder (Van Wijngaarden–Dekker–Brent)
//!   used by the reference's `optimize.brentq`.
//! * [`fminbound`] mirrors the bounded Brent scalar minimizer used by the
//!   reference's `optimize.fminbound`, including its golden-section / parabolic
//!   step selection so the converged point agrees to many digits.

/// Find a root of `f` bracketed by `[xa, xb]` (requires `f(xa) * f(xb) <= 0`).
///
/// This follows the same Van Wijngaarden–Dekker–Brent iteration as the
/// reference's C `brentq`, with the same default tolerances
/// (`xtol = 2e-12`, `rtol = 8.881784197001252e-16`, `maxiter = 100`).
pub fn brentq<F>(mut f: F, xa: f64, xb: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    brentq_tol(&mut f, xa, xb, 2e-12, 8.881_784_197_001_252e-16, 100)
}

/// [`brentq`] with explicit tolerances, matching the reference's C implementation.
pub fn brentq_tol<F>(f: &mut F, xa: f64, xb: f64, xtol: f64, rtol: f64, maxiter: usize) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut xpre = xa;
    let mut xcur = xb;
    let mut xblk = 0.0_f64;
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    let mut fblk = 0.0_f64;
    let mut spre = 0.0_f64;
    let mut scur = 0.0_f64;

    if fpre == 0.0 {
        return xpre;
    }
    if fcur == 0.0 {
        return xcur;
    }
    // No sign change: return the best guess (the reference would error; we
    // never call brentq without a valid bracket).
    if fpre.signum() == fcur.signum() {
        return xcur;
    }

    for _ in 0..maxiter {
        if fpre != 0.0 && fcur != 0.0 && (fpre.signum() != fcur.signum()) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = xcur - xpre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return xcur;
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate (secant)
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate (inverse quadratic)
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // accept step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }

        fcur = f(xcur);
    }
    xcur
}

/// Bounded scalar minimization of `f` over `[x1, x2]` via Brent's method.
///
/// Faithful port of the reference's `optimize.fminbound`
/// (`xatol = 1e-5`, `maxiter = 500`). Returns `(xopt, fval)`.
pub fn fminbound<F>(mut f: F, x1: f64, x2: f64) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    fminbound_opts(&mut f, x1, x2, 1e-5, 500)
}

/// [`fminbound`] with explicit absolute x-tolerance and function-call budget.
pub fn fminbound_opts<F>(f: &mut F, x1: f64, x2: f64, xatol: f64, maxfun: usize) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let mut nfc = fulc;
    let mut xf = fulc;
    let mut rat = 0.0_f64;
    let mut e = 0.0_f64;
    let mut x;
    let mut fx = f(xf);
    let mut num = 1usize;

    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;

            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign_with_zero(xm - xf);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }

        if golden {
            if xf >= xm {
                e = a - xf;
            } else {
                e = b - xf;
            }
            rat = golden_mean * e;
        }

        let si = sign_with_zero(rat);
        x = xf + si * rat.abs().max(tol1);
        let fu = f(x);
        num += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;

        if num >= maxfun {
            break;
        }
    }

    (xf, fx)
}

/// `np.sign(v) + (v == 0)`: returns +1 for v >= 0, -1 for v < 0.
fn sign_with_zero(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn brentq_finds_simple_root() {
        // root of x^2 - 2 in [0, 2] -> sqrt(2)
        let r = brentq(|x| x * x - 2.0, 0.0, 2.0);
        assert_abs_diff_eq!(r, 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn brentq_handles_endpoint_root() {
        let r = brentq(|x| x - 1.0, 1.0, 3.0);
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn fminbound_minimizes_parabola() {
        // (x-1)^2 over [-4, 4] -> 1.0
        let (xopt, fval) = fminbound(|x| (x - 1.0).powi(2), -4.0, 4.0);
        assert_abs_diff_eq!(xopt, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(fval, 0.0, epsilon = 1e-9);
    }
}
