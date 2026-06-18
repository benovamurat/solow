//! Shared formatting helpers for the `.summary()` tables of the discrete models.
//!
//! These mirror the reference's printf semantics (`%#.Ng`) and the
//! `%a, %d %b %Y` / `%H:%M:%S` header stamps, so the rendered tables match the
//! reference field-for-field. The same small helpers back the OLS/GLM summaries
//! elsewhere in the workspace.

/// Format with `sig` significant figures keeping trailing zeros, matching the
/// reference's `%#.Ng`: scientific (with a signed, ≥2-digit exponent) when the
/// exponent is `< -4` or `>= sig`, fixed otherwise.
pub(crate) fn fmt_g(v: f64, sig: usize) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return format!("{:.*}", sig.saturating_sub(1), 0.0);
    }
    // Determine the decimal exponent *after* rounding to `sig` significant
    // figures: rounding can carry across a power-of-10 boundary (e.g.
    // 0.9999707 → 1.000, 9.9996 → 10.00), and C's `%g` picks the precision
    // from the rounded magnitude, not the raw one.
    let raw_exp = v.abs().log10().floor() as i32;
    let rounded = {
        let scale = 10f64.powi(sig as i32 - 1 - raw_exp);
        (v.abs() * scale).round() / scale
    };
    let exp = rounded.log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        // Rust prints e.g. "8.344e-6"; the reference uses a signed, ≥2-digit
        // exponent ("8.344e-06"), so normalize the exponent field.
        let raw = format!("{:.*e}", sig.saturating_sub(1), v);
        if let Some(epos) = raw.find('e') {
            let (mant, exp_part) = raw.split_at(epos);
            let digits = &exp_part[1..];
            let (sign, mag) = if let Some(rest) = digits.strip_prefix('-') {
                ("-", rest)
            } else {
                ("+", digits.strip_prefix('+').unwrap_or(digits))
            };
            format!("{mant}e{sign}{mag:0>2}")
        } else {
            raw
        }
    } else {
        let dec = (sig as i32 - 1 - exp).max(0) as usize;
        format!("{:.*}", dec, v)
    }
}

/// The reference's `forg(x, prec)` cell formatter for the parameter table:
/// fixed `%.<prec>f` in the "normal" magnitude band, switching to `%.<prec>g`
/// (general/scientific) when `|x| >= 1e4` or `|x| < 1e-4`. Only `prec ∈ {3, 4}`
/// occur in the tables (coef uses 4; std err / z / CI use 3).
pub(crate) fn forg(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        // The reference renders non-finite cells as bare "nan"/"inf".
        return format!("{x}");
    }
    let ax = x.abs();
    if !(1e-4..1e4).contains(&ax) {
        // %.<prec>g — `prec` significant figures, scientific where needed, but
        // (unlike `%#g`) without forced trailing zeros. (`x == 0` lands here too,
        // since `0 < 1e-4`, and renders as a bare "0" exactly like the reference.)
        fmt_g_trim(x, prec)
    } else {
        format!("{:.*}", prec, x)
    }
}

/// `%.<sig>g` without the `#` flag: `sig` significant figures, trailing zeros
/// trimmed, with the reference/C signed ≥2-digit exponent in scientific mode.
fn fmt_g_trim(v: f64, sig: usize) -> String {
    // Start from the keep-trailing-zeros form, then strip trailing zeros (and a
    // dangling decimal point) from the mantissa, matching plain `%g`.
    let full = fmt_g(v, sig);
    if let Some(epos) = full.find('e') {
        let (mant, exp) = full.split_at(epos);
        let mant = trim_mantissa(mant);
        format!("{mant}{exp}")
    } else if full.contains('.') {
        trim_mantissa(&full).to_string()
    } else {
        full
    }
}

fn trim_mantissa(m: &str) -> &str {
    if m.contains('.') {
        let t = m.trim_end_matches('0');
        t.trim_end_matches('.')
    } else {
        m
    }
}

/// Center `title` inside a field of width `w` (left bias on odd remainders,
/// matching the reference's title placement).
pub(crate) fn centered(title: &str, w: usize) -> String {
    let pad = (w.saturating_sub(title.len())) / 2;
    format!("{}{}", " ".repeat(pad), title)
}

/// Emit one header line: two `label: value` pairs in the reference's two-column
/// widths (left block 37, a 3-space gutter, right block 38; total 78).
///
/// Within each block the label is left-justified and the value right-justified
/// against the block's right edge, so an over-long value (e.g. the
/// `GeneralizedPoisson` model name) consumes the label's trailing padding rather
/// than overflowing the block — exactly the reference's `SimpleTable` behavior.
pub(crate) fn header_row(l1: &str, v1: &str, l2: &str, v2: &str) -> String {
    format!(
        "{}   {}",
        header_block(l1, v1, 37),
        header_block(l2, v2, 38)
    )
}

/// One `label … value` block of total width `w`: label flush left, value flush
/// right, the gap between them filled with spaces (at least one column).
fn header_block(label: &str, value: &str, w: usize) -> String {
    let used = label.len() + value.len();
    let gap = w.saturating_sub(used).max(1);
    format!("{}{}{}", label, " ".repeat(gap), value)
}

/// Emit the coefficient-table column header, reproducing the reference's data
/// column right edges (21 / 32 / 43 / 54 / 66 / 78).
pub(crate) fn coef_header() -> String {
    format!(
        "{:<11}{:>10}{:>11}{:>11}{:>11}{:>12}{:>12}",
        "", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
    )
}

/// Emit a per-equation coefficient-table column header carrying a left-side
/// equation label (e.g. `"y=1"`), as used by the multi-equation MNLogit table.
/// The label sits in the stub column and the data columns keep the reference's
/// right edges (21 / 32 / 43 / 54 / 66 / 78).
pub(crate) fn coef_header_labeled(eq_label: &str) -> String {
    format!(
        "{:>10}{:>11}{:>11}{:>11}{:>11}{:>12}{:>12}",
        eq_label, "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
    )
}

/// Emit one coefficient row in the reference's parameter-table layout.
///
/// Each numeric cell is formatted with the reference's `forg` rule (coef at
/// precision 4; std err, z, and the CI bounds at precision 3) and right-justified
/// into its column; `P>|z|` keeps the reference's fixed `%#6.3f`. The data-column
/// right edges land at 21 / 32 / 43 / 54 / 66 / 78.
#[allow(clippy::too_many_arguments)]
pub(crate) fn coef_row(
    name: &str,
    coef: f64,
    std_err: f64,
    z: f64,
    p: f64,
    lo: f64,
    hi: f64,
) -> String {
    format!(
        "{:<11}{:>10}{:>11}{:>11}{:>11}{:>12}{:>12}",
        name,
        forg(coef, 4),
        forg(std_err, 3),
        forg(z, 3),
        format!("{:.3}", p),
        forg(lo, 3),
        forg(hi, 3),
    )
}

/// Current UTC date/time as `("Thu, 18 Jun 2026", "03:21:29")`, matching the
/// reference header look. Uses only `std::time` via the civil-from-days algorithm.
pub(crate) fn utc_now_strings() -> (String, String) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (h, mi, sc) = (tod / 3600, (tod % 3600) / 60, tod % 60);

    // Howard Hinnant's civil_from_days (1970-01-01 == day 0).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let wdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    // 1970-01-01 was a Thursday (index 3).
    let wd = (days.rem_euclid(7) + 3).rem_euclid(7) as usize;
    let date = format!(
        "{}, {:02} {} {}",
        wdays[wd],
        d,
        months[(m as usize - 1).min(11)],
        year
    );
    let time = format!("{h:02}:{mi:02}:{sc:02}");
    (date, time)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_g_matches_reference_printf() {
        // Cases lifted from the reference's `%#.Ng` output.
        assert_eq!(fmt_g(1.0, 5), "1.0000");
        assert_eq!(fmt_g(4.14, 3), "4.14");
        assert_eq!(fmt_g(0.9991, 4), "0.9991");
        assert_eq!(fmt_g(0.002449, 4), "0.002449");
        assert_eq!(fmt_g(0.5450, 4), "0.5450");
        // Scientific with a signed two-digit exponent.
        assert_eq!(fmt_g(8.344e-6, 4), "8.344e-06");
        assert_eq!(fmt_g(1.5e8, 4), "1.500e+08");
        // Rounding that carries across a power-of-10 boundary must pick the
        // precision of the *rounded* magnitude (C `%g`), not the raw one.
        assert_eq!(fmt_g(0.99999707, 4), "1.000");
        assert_eq!(fmt_g(9.9996, 4), "10.00");
        assert_eq!(fmt_g(0.099996, 4), "0.1000");
    }
}
