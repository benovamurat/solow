//! Cross-validation of the rendered summary table against golden reference
//! values frozen in `tests/fixtures/summary.json`.
//!
//! The fixture is produced by fitting a small OLS in a modeling reference and
//! dumping its parameter table and header statistics (see
//! `tools/reference/gen_summary.py`). We feed those numbers into a
//! [`RegressionSummary`], render it, then parse the relevant cells back out of
//! the rendered text and compare them to the dumped values *to the displayed
//! precision*.
//!
//! The LAYOUT here is Solow's own -- only the NUMBERS are cross-checked. The
//! tests therefore locate values by reading the appropriate column out of the
//! data rows rather than asserting any particular spacing.

use serde_json::Value;
use solow_summary::{HeaderStats, RegressionSummary};
use std::fs;

fn load() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/summary.json"
    );
    let s = fs::read_to_string(p).expect("fixture present (run tools/reference/gen_summary.py)");
    serde_json::from_str(&s).unwrap()
}

fn vec1(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn conf_int(v: &Value) -> Vec<(f64, f64)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let r = row.as_array().unwrap();
            (r[0].as_f64().unwrap(), r[1].as_f64().unwrap())
        })
        .collect()
}

/// Relative error on the *displayed* digits. Both sides are themselves rounded
/// to the displayed precision before comparing, so this only catches a genuine
/// mismatch (wrong value in the wrong cell), not last-digit rounding.
fn close_displayed(got: f64, want: f64) -> bool {
    (got - want).abs() / (1.0 + want.abs()) <= 1e-3
}

/// Find the data rows of the coefficient block: every line that, when split on
/// whitespace, begins with one of the parameter names and has the expected
/// number of trailing numeric columns.
fn coef_row<'a>(rendered: &'a str, name: &str) -> Vec<&'a str> {
    for line in rendered.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        // A coefficient row is: name coef std_err stat pval ci_lo ci_hi
        if toks.len() == 7 && toks[0] == name {
            return toks;
        }
    }
    panic!("could not find coefficient row for {name:?} in:\n{rendered}");
}

fn build(case: &Value) -> RegressionSummary {
    let names: Vec<String> = case["exog_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();
    let params = vec1(&case["params"]);
    let bse = vec1(&case["bse"]);
    let tvalues = vec1(&case["tvalues"]);
    let pvalues = vec1(&case["pvalues"]);
    let ci = conf_int(&case["conf_int"]);

    let header = HeaderStats {
        model: Some(case["model"].as_str().unwrap().to_string()),
        nobs: Some(case["nobs"].as_f64().unwrap()),
        df_model: Some(case["df_model"].as_f64().unwrap()),
        df_resid: Some(case["df_resid"].as_f64().unwrap()),
        rsquared: Some(case["rsquared"].as_f64().unwrap()),
        rsquared_adj: Some(case["rsquared_adj"].as_f64().unwrap()),
        fvalue: Some(case["fvalue"].as_f64().unwrap()),
        f_pvalue: Some(case["f_pvalue"].as_f64().unwrap()),
        llf: Some(case["llf"].as_f64().unwrap()),
        aic: Some(case["aic"].as_f64().unwrap()),
        bic: Some(case["bic"].as_f64().unwrap()),
        ..HeaderStats::new()
    };

    RegressionSummary::new(&names, &params, &bse, &tvalues, &pvalues, &ci, header)
}

#[test]
fn coefficient_cells_match_reference() {
    let fx = load();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let summary = build(case);
        let rendered = summary.to_string();

        let exog: Vec<String> = case["exog_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        let params = vec1(&case["params"]);
        let bse = vec1(&case["bse"]);
        let tvalues = vec1(&case["tvalues"]);
        let pvalues = vec1(&case["pvalues"]);
        let ci = conf_int(&case["conf_int"]);

        for (i, pname) in exog.iter().enumerate() {
            // (c) every parameter name appears.
            assert!(
                rendered.contains(pname),
                "{name}: missing param name {pname}"
            );

            let toks = coef_row(&rendered, pname);
            // toks = [name, coef, std_err, stat, pval, ci_lo, ci_hi]
            let coef: f64 = toks[1].parse().unwrap();
            let std_err: f64 = toks[2].parse().unwrap();
            let stat: f64 = toks[3].parse().unwrap();
            let pval: f64 = toks[4].parse().unwrap();
            let ci_lo: f64 = toks[5].parse().unwrap();
            let ci_hi: f64 = toks[6].parse().unwrap();

            // (a) coef / std-err / t / p parse back and match to displayed
            //     precision.
            assert!(
                close_displayed(coef, params[i]),
                "{name}.{pname}: coef cell {coef} vs {}",
                params[i]
            );
            assert!(
                close_displayed(std_err, bse[i]),
                "{name}.{pname}: std err cell {std_err} vs {}",
                bse[i]
            );
            assert!(
                close_displayed(stat, tvalues[i]),
                "{name}.{pname}: t cell {stat} vs {}",
                tvalues[i]
            );
            // p-values for these cases are tiny -> displayed as 0.000; the
            // displayed value rounds the true value, so compare absolute
            // displayed rounding.
            let want_p_disp = format!("{:.3}", pvalues[i]).parse::<f64>().unwrap();
            assert!(
                (pval - want_p_disp).abs() <= 1e-9,
                "{name}.{pname}: p cell {pval} vs displayed {want_p_disp}"
            );
            // CI bounds to 3 decimals.
            assert!(
                close_displayed(ci_lo, ci[i].0),
                "{name}.{pname}: CI lo {ci_lo} vs {}",
                ci[i].0
            );
            assert!(
                close_displayed(ci_hi, ci[i].1),
                "{name}.{pname}: CI hi {ci_hi} vs {}",
                ci[i].1
            );
        }
    }
}

#[test]
fn header_stats_formatted_correctly() {
    let fx = load();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let summary = build(case);
        let rendered = summary.to_string();

        // (b) header contains R-squared / AIC / BIC formatted correctly.
        let rsq = case["rsquared"].as_f64().unwrap();
        let rsq_disp = format!("{rsq:.3}");
        assert!(
            rendered.contains("R-squared:"),
            "{name}: missing R-squared label"
        );
        assert!(
            rendered.contains(&rsq_disp),
            "{name}: R-squared value {rsq_disp} not in:\n{rendered}"
        );

        // AIC / BIC are rendered with ~5 sig figs via format_g; recover the
        // value by reading the cell after the label.
        for (label, key) in [("AIC:", "aic"), ("BIC:", "bic")] {
            let want = case[key].as_f64().unwrap();
            let line = rendered
                .lines()
                .find(|l| l.trim_start().starts_with(label))
                .unwrap_or_else(|| panic!("{name}: no {label} line"));
            let cell = line.split_whitespace().last().unwrap();
            let got: f64 = cell.parse().unwrap();
            assert!(
                close_displayed(got, want),
                "{name}: {label} cell {got} vs {want}"
            );
        }

        // model name present.
        let model = case["model"].as_str().unwrap();
        assert!(
            rendered.contains(model),
            "{name}: model name {model} missing"
        );
    }
}
