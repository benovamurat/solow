//! Numeric formatting helpers for summary tables.
//!
//! These mirror the *display* conventions of a typical regression-results
//! summary: coefficients and standard errors are shown to roughly four
//! significant decimals, test statistics to three decimals, and p-values to
//! three decimals (collapsing to `0.000` when smaller than the displayed
//! precision). The exact rendering policy here is Solow's own; only the
//! underlying numbers are ever cross-checked against a reference.

/// Format a value with `prec` significant figures, similar to the `%g`
/// family but with a fixed number of *significant* digits and without an
/// exponent unless the magnitude makes a plain decimal unwieldy.
///
/// This is used for coefficients and standard errors (`prec == 4` typically),
/// giving output like `1.493`, `-1.906`, `0.07289`, `355.9`, `1.234e+05`.
pub fn format_g(value: f64, prec: usize) -> String {
    if !value.is_finite() {
        if value.is_nan() {
            return "nan".to_string();
        }
        return if value < 0.0 {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let prec = prec.max(1);
    let exp = value.abs().log10().floor() as i32;
    // Decide whether to use scientific notation, mirroring %g: switch to
    // scientific when the exponent is < -4 or >= precision.
    if exp < -4 || exp >= prec as i32 {
        let s = format!("{:.*e}", prec - 1, value);
        return tidy_exponent(&s);
    }
    // Fixed notation: number of digits after the decimal point so that we keep
    // `prec` significant digits.
    let decimals = (prec as i32 - 1 - exp).max(0) as usize;
    let s = format!("{value:.decimals$}");
    trim_trailing_zeros(&s)
}

/// Format a value with a fixed number of digits after the decimal point.
pub fn format_fixed(value: f64, decimals: usize) -> String {
    if !value.is_finite() {
        if value.is_nan() {
            return "nan".to_string();
        }
        return if value < 0.0 {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    format!("{value:.decimals$}")
}

/// Format a p-value to three decimals, displaying `0.000` for values that are
/// smaller than the displayed precision (rather than rounding artifacts).
pub fn format_pvalue(p: f64) -> String {
    if !p.is_finite() {
        if p.is_nan() {
            return "nan".to_string();
        }
        return if p < 0.0 {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    // Three decimals; tiny but non-zero values collapse to 0.000.
    format!("{p:.3}")
}

/// Trim trailing zeros (and a dangling decimal point) from a fixed-point
/// decimal string while preserving at least the integer part.
fn trim_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let trimmed = s.trim_end_matches('0');
    let trimmed = trimmed.trim_end_matches('.');
    trimmed.to_string()
}

/// Normalise an exponent produced by Rust's `{:e}` formatter (e.g. `1.234e5`)
/// into a conventional `1.234e+05` form with a sign and at least two digits.
fn tidy_exponent(s: &str) -> String {
    let Some(idx) = s.find('e') else {
        return s.to_string();
    };
    let (mantissa, exp_part) = s.split_at(idx);
    let exp_digits = &exp_part[1..]; // strip the 'e'
    let (sign, digits) = if let Some(rest) = exp_digits.strip_prefix('-') {
        ('-', rest)
    } else if let Some(rest) = exp_digits.strip_prefix('+') {
        ('+', rest)
    } else {
        ('+', exp_digits)
    };
    let digits = if digits.len() < 2 {
        format!("{digits:0>2}")
    } else {
        digits.to_string()
    };
    format!("{mantissa}e{sign}{digits}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_g_basic_significant_digits() {
        assert_eq!(format_g(1.49339521, 4), "1.493");
        assert_eq!(format_g(-1.90617444, 4), "-1.906");
        assert_eq!(format_g(0.0728888, 4), "0.07289");
        assert_eq!(format_g(355.9185911967316, 4), "355.9");
    }

    #[test]
    fn format_g_zero_and_scientific() {
        assert_eq!(format_g(0.0, 4), "0");
        // Very small -> scientific.
        assert_eq!(format_g(0.0000123456, 4), "1.235e-05");
        // Very large -> scientific.
        assert_eq!(format_g(123456.0, 4), "1.235e+05");
    }

    #[test]
    fn format_g_trims_trailing_zeros() {
        assert_eq!(format_g(2.0, 4), "2");
        assert_eq!(format_g(2.5, 4), "2.5");
        assert_eq!(format_g(2.50, 4), "2.5");
    }

    #[test]
    fn format_fixed_decimals() {
        assert_eq!(format_fixed(20.48867957, 3), "20.489");
        assert_eq!(format_fixed(-23.38232831, 3), "-23.382");
        assert_eq!(format_fixed(0.95059006, 3), "0.951");
    }

    #[test]
    fn format_pvalue_collapses_tiny() {
        assert_eq!(format_pvalue(8.69594441e-22), "0.000");
        assert_eq!(format_pvalue(0.0001), "0.000");
        assert_eq!(format_pvalue(0.0234), "0.023");
        assert_eq!(format_pvalue(0.5), "0.500");
    }

    #[test]
    fn non_finite_handled() {
        assert_eq!(format_g(f64::NAN, 4), "nan");
        assert_eq!(format_g(f64::INFINITY, 4), "inf");
        assert_eq!(format_fixed(f64::NEG_INFINITY, 3), "-inf");
        assert_eq!(format_pvalue(f64::NAN), "nan");
    }
}
