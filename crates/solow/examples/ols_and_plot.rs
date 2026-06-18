//! End-to-end demo: fit an OLS model and render a scatter + fitted-line figure.
//!
//! Run with:  cargo run -p solow --example ols_and_plot

use solow::core::tools::{add_constant, HasConstant};
use solow::ndarray::{Array1, Array2};
use solow::regression::LinearModel;
use solow::viz::{Color, Figure};

fn main() {
    // Deterministic synthetic data: y = 2 + 1.5 x + mild structured noise.
    let n = 40usize;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| 2.0 + 1.5 * x + 0.6 * ((i as f64 * 1.3).sin()))
        .collect();

    let x_col = Array2::from_shape_vec((n, 1), xs.clone()).unwrap();
    let design = add_constant(&x_col, true, HasConstant::Add).unwrap();
    let y = Array1::from_vec(ys.clone());

    let res = LinearModel::ols(y, design).unwrap().fit().unwrap();
    println!("{}", res.summary(Some(&["const", "x"])));

    // Fitted line across the x-range.
    let intercept = res.params[0];
    let slope = res.params[1];
    let line_x = [xs[0], xs[n - 1]];
    let line_y = [intercept + slope * xs[0], intercept + slope * xs[n - 1]];

    let mut fig = Figure::new(720, 480);
    {
        let ax = fig.axes();
        ax.set_title("OLS fit")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.scatter_styled(&xs, &ys, Color::BLUE, 3.0);
        ax.plot_styled(&line_x, &line_y, Color::RED, 2.0);
    }
    let out = std::env::temp_dir().join("solow_ols_fit.svg");
    fig.save_svg(&out).unwrap();
    println!("\nFigure written to {}", out.display());
}
