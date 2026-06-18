//! Cross-validation of the ECDF / StepFunction and the extended distribution
//! library against golden values frozen in `tests/fixtures/distributions_ext.json`.
//!
//! ECDF / StepFunction goldens come from the modeling reference; the
//! distribution goldens come from `scipy.stats`. Regenerate with
//! `SOLOW_REFERENCE=<modeling-pkg> python3 tools/reference/gen_distributions_ext.py`.

use serde_json::Value;
use solow_distributions::{
    Beta, Binomial, Cauchy, Ecdf, Exponential, Gamma, Geometric, Laplace, LogNormal, Logistic,
    NegativeBinomial, Pareto, Poisson, Side, StepFunction, Uniform, WeibullMin,
};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/distributions_ext.json"
    );
    let s = fs::read_to_string(p)
        .expect("fixture present (run tools/reference/gen_distributions_ext.py)");
    serde_json::from_str(&s).unwrap()
}

fn f(v: &Value) -> f64 {
    // Non-finite values are dumped as string sentinels.
    if let Some(s) = v.as_str() {
        return match s {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            other => panic!("unexpected string number {other:?}"),
        };
    }
    v.as_f64().unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

/// Absolute-or-relative closeness, robust for both tiny densities and large
/// quantiles: `|got - want| <= tol * (1 + |want|)`.
fn close(got: f64, want: f64, tol: f64) -> bool {
    if got.is_infinite() && want.is_infinite() {
        return got.signum() == want.signum();
    }
    (got - want).abs() <= tol * (1.0 + want.abs())
}

fn check(label: &str, got: f64, want: f64, tol: f64) {
    assert!(
        close(got, want, tol),
        "{label}: |{got} - {want}| = {:.3e} exceeds tol {tol:.0e}",
        (got - want).abs()
    );
}

fn side_of(v: &Value) -> Side {
    match v.as_str().unwrap() {
        "right" => Side::Right,
        "left" => Side::Left,
        other => panic!("unknown side {other}"),
    }
}

/// Compare an `x`/`y` knot pair where the first abscissa is the `-inf` sentinel.
fn check_knots(label: &str, got_x: &[f64], got_y: &[f64], exp_x: &Value, exp_y: &Value) {
    let ex = exp_x.as_array().unwrap();
    let ey = vecf(exp_y);
    assert_eq!(got_x.len(), ex.len(), "{label}: x length");
    assert_eq!(got_y.len(), ey.len(), "{label}: y length");
    assert!(
        got_x[0].is_infinite() && got_x[0] < 0.0,
        "{label}: x[0] not -inf"
    );
    assert_eq!(ex[0].as_str(), Some("-inf"), "{label}: exp x[0] not -inf");
    for i in 1..ex.len() {
        check(&format!("{label}.x[{i}]"), got_x[i], f(&ex[i]), 1e-12);
    }
    for i in 0..ey.len() {
        check(&format!("{label}.y[{i}]"), got_y[i], ey[i], 1e-12);
    }
}

#[test]
fn ecdf_matches_reference() {
    let fx = load();
    for c in fx["ecdf"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let data = vecf(&c["data"]);
        let side = side_of(&c["side"]);
        let e = Ecdf::with_side(&data, side);
        check_knots(name, e.x(), e.y(), &c["x"], &c["y"]);
        let query = vecf(&c["query"]);
        let want = vecf(&c["eval"]);
        for (i, (&q, &w)) in query.iter().zip(want.iter()).enumerate() {
            // ECDF vs the modeling reference: exact to ~1e-12.
            check(&format!("{name}.eval[{i}]@{q}"), e.eval(q), w, 1e-12);
        }
    }
}

#[test]
fn stepfunction_matches_reference() {
    let fx = load();
    for c in fx["step"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let x_in = vecf(&c["x_in"]);
        let y_in = vecf(&c["y_in"]);
        let ival = f(&c["ival"]);
        let side = side_of(&c["side"]);
        let s = StepFunction::new(&x_in, &y_in, ival, false, side);
        check_knots(name, s.x(), s.y(), &c["x"], &c["y"]);
        let query = vecf(&c["query"]);
        let want = vecf(&c["eval"]);
        for (i, (&q, &w)) in query.iter().zip(want.iter()).enumerate() {
            check(&format!("{name}.eval[{i}]@{q}"), s.eval(q), w, 1e-12);
        }
    }
}

/// A uniform accessor over the continuous distributions so the test loop can
/// stay generic.
struct Cont {
    pdf: Box<dyn Fn(f64) -> f64>,
    logpdf: Box<dyn Fn(f64) -> f64>,
    cdf: Box<dyn Fn(f64) -> f64>,
    sf: Box<dyn Fn(f64) -> f64>,
    ppf: Box<dyn Fn(f64) -> f64>,
    mean: f64,
    var: f64,
}

fn build_cont(c: &Value) -> Cont {
    let kind = c["kind"].as_str().unwrap();
    macro_rules! wrap {
        ($d:expr) => {{
            let d = $d;
            Cont {
                pdf: Box::new(move |x| d.pdf(x)),
                logpdf: Box::new(move |x| d.logpdf(x)),
                cdf: Box::new(move |x| d.cdf(x)),
                sf: Box::new(move |x| d.sf(x)),
                ppf: Box::new(move |p| d.ppf(p)),
                mean: d.mean(),
                var: d.var(),
            }
        }};
    }
    match kind {
        "gamma" => wrap!(Gamma::new(f(&c["a"]), f(&c["scale"]))),
        "beta" => wrap!(Beta::new(f(&c["a"]), f(&c["b"]))),
        "expon" => wrap!(Exponential::new(f(&c["scale"]))),
        "lognorm" => wrap!(LogNormal::new(f(&c["s"]), f(&c["scale"]))),
        "uniform" => wrap!(Uniform::new(f(&c["loc"]), f(&c["scale"]))),
        "weibull_min" => wrap!(WeibullMin::new(f(&c["c"]))),
        "laplace" => wrap!(Laplace::new(f(&c["loc"]), f(&c["scale"]))),
        "logistic" => wrap!(Logistic::new(f(&c["loc"]), f(&c["scale"]))),
        "cauchy" => wrap!(Cauchy::new(f(&c["loc"]), f(&c["scale"]))),
        "pareto" => wrap!(Pareto::new(f(&c["b"]))),
        other => panic!("unknown continuous kind {other}"),
    }
}

#[test]
fn continuous_matches_scipy() {
    let fx = load();
    for c in fx["continuous"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let d = build_cont(c);
        let xs = vecf(&c["x"]);
        let pdf = vecf(&c["pdf"]);
        let logpdf = vecf(&c["logpdf"]);
        let cdf = vecf(&c["cdf"]);
        let sf = vecf(&c["sf"]);
        for i in 0..xs.len() {
            let x = xs[i];
            // pdf / cdf / sf: tight to 1e-10.
            check(&format!("{name}.pdf@{x}"), (d.pdf)(x), pdf[i], 1e-10);
            check(&format!("{name}.cdf@{x}"), (d.cdf)(x), cdf[i], 1e-10);
            check(&format!("{name}.sf@{x}"), (d.sf)(x), sf[i], 1e-10);
            // logpdf only where finite (scipy emits -inf outside support).
            if logpdf[i].is_finite() {
                check(
                    &format!("{name}.logpdf@{x}"),
                    (d.logpdf)(x),
                    logpdf[i],
                    1e-10,
                );
            }
        }
        let ps = vecf(&c["p"]);
        let ppf = vecf(&c["ppf"]);
        for i in 0..ps.len() {
            // ppf: tight to 1e-8.
            check(
                &format!("{name}.ppf@{}", ps[i]),
                (d.ppf)(ps[i]),
                ppf[i],
                1e-8,
            );
        }
        if let Some(m) = c["mean"].as_f64() {
            check(&format!("{name}.mean"), d.mean, m, 1e-10);
        }
        if let Some(v) = c["var"].as_f64() {
            check(&format!("{name}.var"), d.var, v, 1e-10);
        }
    }
}

struct Disc {
    pmf: Box<dyn Fn(u64) -> f64>,
    logpmf: Box<dyn Fn(u64) -> f64>,
    cdf: Box<dyn Fn(f64) -> f64>,
    sf: Box<dyn Fn(f64) -> f64>,
    mean: f64,
    var: f64,
}

fn build_disc(c: &Value) -> Disc {
    let kind = c["kind"].as_str().unwrap();
    macro_rules! wrap {
        ($d:expr) => {{
            let d = $d;
            Disc {
                pmf: Box::new(move |k| d.pmf(k)),
                logpmf: Box::new(move |k| d.logpmf(k)),
                cdf: Box::new(move |x| d.cdf(x)),
                sf: Box::new(move |x| d.sf(x)),
                mean: d.mean(),
                var: d.var(),
            }
        }};
    }
    match kind {
        "poisson" => wrap!(Poisson::new(f(&c["mu"]))),
        "binom" => wrap!(Binomial::new(c["n"].as_u64().unwrap(), f(&c["p"]))),
        "geom" => wrap!(Geometric::new(f(&c["p"]))),
        "nbinom" => wrap!(NegativeBinomial::new(f(&c["n"]), f(&c["p"]))),
        other => panic!("unknown discrete kind {other}"),
    }
}

#[test]
fn discrete_matches_scipy() {
    let fx = load();
    for c in fx["discrete"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let d = build_disc(c);
        let ks: Vec<u64> = c["k"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let pmf = vecf(&c["pmf"]);
        let logpmf = vecf(&c["logpmf"]);
        for i in 0..ks.len() {
            let k = ks[i];
            // pmf: tight to 1e-10.
            check(&format!("{name}.pmf@{k}"), (d.pmf)(k), pmf[i], 1e-10);
            if logpmf[i].is_finite() {
                check(
                    &format!("{name}.logpmf@{k}"),
                    (d.logpmf)(k),
                    logpmf[i],
                    1e-10,
                );
            }
        }
        let cg = vecf(&c["cgrid"]);
        let cdf = vecf(&c["cdf"]);
        let sf = vecf(&c["sf"]);
        for i in 0..cg.len() {
            let x = cg[i];
            check(&format!("{name}.cdf@{x}"), (d.cdf)(x), cdf[i], 1e-10);
            check(&format!("{name}.sf@{x}"), (d.sf)(x), sf[i], 1e-10);
        }
        check(&format!("{name}.mean"), d.mean, f(&c["mean"]), 1e-10);
        check(&format!("{name}.var"), d.var, f(&c["var"]), 1e-10);
    }
}
