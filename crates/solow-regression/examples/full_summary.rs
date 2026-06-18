//! Print the full reference-style OLS summary on a fixed dataset.
//! Run: cargo run -p solow-regression --example full_summary

use ndarray::{Array1, Array2};
use solow_regression::LinearModel;

fn main() {
    let n = 50usize;
    let mut x = Array2::<f64>::zeros((n, 4));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let fi = i as f64;
        let x1 = fi * 0.2;
        let x2 = (fi * 0.5).sin();
        let x3 = (fi * 0.3).cos() * 5.0;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = x2;
        x[[i, 3]] = x3;
        y[i] = 5.0 + 0.5 * x1 + 0.5 * x2 - 0.3 * x3 + 0.3 * (fi * 1.7).sin();
    }
    let res = LinearModel::ols(y, x).unwrap().fit().unwrap();
    println!("{}", res.summary(Some(&["const", "x1", "x2", "x3"])));
}
