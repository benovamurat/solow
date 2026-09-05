//! Cross-validation of the copula implementations against golden reference
//! values frozen in `tests/fixtures/copula.json` (generated from the
//! canonical Python reference's `distributions.copula.api`).

use serde_json::Value;
use solow_copula::{
    kendalls_tau, spearmans_rho, ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula,
    StudentTCopula,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/copula.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_copula.py)");
    serde_json::from_str(&s).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn vec1(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

/// Absolute error (the dumped quantities are O(1), so absolute tolerances
/// are appropriate and tighter than relative ones near zero).
fn close(got: f64, want: f64, tol: f64, label: &str) {
    let e = (got - want).abs();
    assert!(
        e <= tol,
        "{label}: abs-err {e:.3e} (got {got}, want {want})"
    );
}

#[derive(Clone, Copy)]
enum Arch {
    Clayton,
    Frank,
    Gumbel,
}

fn arch_cdf(kind: Arch, theta: f64, u: f64, v: f64) -> f64 {
    match kind {
        Arch::Clayton => ClaytonCopula::new(theta).cdf(u, v),
        Arch::Frank => FrankCopula::new(theta).cdf(u, v),
        Arch::Gumbel => GumbelCopula::new(theta).cdf(u, v),
    }
}

fn arch_pdf(kind: Arch, theta: f64, u: f64, v: f64) -> f64 {
    match kind {
        Arch::Clayton => ClaytonCopula::new(theta).pdf(u, v),
        Arch::Frank => FrankCopula::new(theta).pdf(u, v),
        Arch::Gumbel => GumbelCopula::new(theta).pdf(u, v),
    }
}

fn arch_tau(kind: Arch, theta: f64) -> f64 {
    match kind {
        Arch::Clayton => ClaytonCopula::new(theta).tau(),
        Arch::Frank => FrankCopula::new(theta).tau(),
        Arch::Gumbel => GumbelCopula::new(theta).tau(),
    }
}

#[test]
fn archimedean_matches_reference() {
    let fx = load();
    for fam in fx["archimedean"].as_array().unwrap() {
        let name = fam["name"].as_str().unwrap();
        let kind = match name {
            "Clayton" => Arch::Clayton,
            "Frank" => Arch::Frank,
            "Gumbel" => Arch::Gumbel,
            other => panic!("unknown family {other}"),
        };
        for case in fam["cases"].as_array().unwrap() {
            let theta = f(&case["theta"]);
            // Kendall's tau: closed form (Clayton/Gumbel) or Debye-integral
            // (Frank) -- verified tight at 1e-9.
            close(
                arch_tau(kind, theta),
                f(&case["tau"]),
                1e-9,
                &format!("{name}(theta={theta}).tau"),
            );
            for pt in case["points"].as_array().unwrap() {
                let u = f(&pt["u"]);
                let v = f(&pt["v"]);
                close(
                    arch_cdf(kind, theta, u, v),
                    f(&pt["cdf"]),
                    1e-8,
                    &format!("{name}(theta={theta}).cdf({u},{v})"),
                );
                close(
                    arch_pdf(kind, theta, u, v),
                    f(&pt["pdf"]),
                    1e-8,
                    &format!("{name}(theta={theta}).pdf({u},{v})"),
                );
            }
        }
    }
}

#[test]
fn gaussian_matches_reference() {
    let fx = load();
    let fam = &fx["gaussian"];
    for case in fam["cases"].as_array().unwrap() {
        let rho = f(&case["rho"]);
        let g = GaussianCopula::new(rho);
        // tau = (2/pi) arcsin(rho): analytic, verified at 1e-9.
        close(
            g.tau(),
            f(&case["tau"]),
            1e-9,
            &format!("Gaussian(rho={rho}).tau"),
        );
        // Spearman's rho = (6/pi) arcsin(rho/2): analytic, verified at 1e-9.
        close(
            g.spearmans_rho(),
            f(&case["spearman"]),
            1e-9,
            &format!("Gaussian(rho={rho}).spearman"),
        );
        for pt in case["points"].as_array().unwrap() {
            let u = f(&pt["u"]);
            let v = f(&pt["v"]);
            // Closed-form copula density via the normal-quantile transform.
            close(
                g.pdf(u, v),
                f(&pt["pdf"]),
                1e-8,
                &format!("Gaussian(rho={rho}).pdf({u},{v})"),
            );
            // CDF via the bivariate-normal CDF (Drezner-Wesolowsky/Genz).
            // The spec target is 1e-7; the achieved accuracy across the grid
            // is ~3e-16, so a far tighter bound holds.
            close(
                g.cdf(u, v),
                f(&pt["cdf"]),
                1e-10,
                &format!("Gaussian(rho={rho}).cdf({u},{v})"),
            );
        }
    }
}

#[test]
fn studentt_matches_reference() {
    let fx = load();
    let fam = &fx["studentt"];
    for case in fam["cases"].as_array().unwrap() {
        let rho = f(&case["rho"]);
        let df = f(&case["df"]);
        let t = StudentTCopula::new(rho, df);
        // Elliptical tau = (2/pi) arcsin(rho): analytic, verified at 1e-9.
        close(
            t.tau(),
            f(&case["tau"]),
            1e-9,
            &format!("StudentT(rho={rho},df={df}).tau"),
        );
        for pt in case["points"].as_array().unwrap() {
            let u = f(&pt["u"]);
            let v = f(&pt["v"]);
            // Density via the t-quantile transform; depends on the
            // t-quantile inverse `t_ppf`. Achieved accuracy is ~1e-14, so a
            // 1e-9 bound holds comfortably (spec lists this as optional/1e-6).
            close(
                t.pdf(u, v),
                f(&pt["pdf"]),
                1e-9,
                &format!("StudentT(rho={rho},df={df}).pdf({u},{v})"),
            );
        }
    }
}

#[test]
fn paired_rank_correlations_match_reference() {
    let fx = load();
    for s in fx["paired"].as_array().unwrap() {
        let x = vec1(&s["x"]);
        let y = vec1(&s["y"]);
        close(
            kendalls_tau(&x, &y),
            f(&s["kendalls_tau"]),
            1e-12,
            "kendalls_tau",
        );
        close(
            spearmans_rho(&x, &y),
            f(&s["spearmans_rho"]),
            1e-12,
            "spearmans_rho",
        );
    }
}
