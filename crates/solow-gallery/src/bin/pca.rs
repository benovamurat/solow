//! Principal component analysis (PCA).
//!
//! Three correlated groups of points are generated in a 4-dimensional feature
//! space. PCA (on standardized data) finds the directions of greatest variance.
//! Two views are produced: a scree plot of the per-component variance ratio,
//! and a scatter of the first two principal-component scores, colored by group.
//!
//! Run with:
//!   cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin pca

use ndarray::Array2;
use solow_multivariate::Pca;
use solow_viz::{Color, Figure, LegendLoc, Marker};

#[path = "../common.rs"]
mod common;

fn main() {
    // --- Synthetic 4-D data with three latent groups ------------------------
    let mut rng = common::Rng::new(31337);
    let per_group = 50usize;
    let nvar = 4usize;
    // Group centers in 4-D; the groups separate mainly along the first two
    // principal directions.
    let centers = [
        [3.0, 3.0, 0.0, 0.0],
        [-3.0, 2.0, 1.0, -1.0],
        [0.0, -3.0, -1.0, 1.0],
    ];
    let n = per_group * centers.len();
    let mut data = Vec::with_capacity(n * nvar);
    let mut group = Vec::with_capacity(n);
    for (g, c) in centers.iter().enumerate() {
        for _ in 0..per_group {
            for j in 0..nvar {
                data.push(c[j] + 1.1 * rng.normal());
            }
            group.push(g);
        }
    }
    let matrix = Array2::from_shape_vec((n, nvar), data).unwrap();

    // --- Fit PCA on standardized columns ------------------------------------
    let res = Pca::new(matrix).standardize(true).fit().unwrap();

    // --- Printed summary -----------------------------------------------------
    println!("PCA on {} observations x {} variables", res.nobs, res.nvar);
    println!(
        "{:>6}{:>14}{:>16}{:>16}",
        "comp", "eigenvalue", "var ratio", "cumulative"
    );
    let mut cum = 0.0;
    for i in 0..res.ncomp {
        cum += res.explained_variance_ratio[i];
        println!(
            "{:>6}{:>14.4}{:>16.4}{:>16.4}",
            i + 1,
            res.eigenvals[i],
            res.explained_variance_ratio[i],
            cum
        );
    }

    // --- Figure: scree (left) + PC1/PC2 scores (right) ----------------------
    let mut fig = Figure::subplots(960, 460, 1, 2);
    fig.suptitle("Principal component analysis");

    // Scree plot.
    {
        let ax = fig.ax_at(0, 0).unwrap();
        ax.set_title("Scree plot")
            .set_xlabel("component")
            .set_ylabel("variance ratio")
            .set_grid(true);
        let comps: Vec<f64> = (1..=res.ncomp).map(|i| i as f64).collect();
        let ratios: Vec<f64> = res.explained_variance_ratio.to_vec();
        ax.bar(&comps, &ratios, 0.6);
    }

    // Scores scatter, colored by group.
    {
        let ax = fig.ax_at(0, 1).unwrap();
        ax.set_title("Scores: PC1 vs PC2")
            .set_xlabel("PC1")
            .set_ylabel("PC2")
            .set_grid(true);
        for (g, _) in centers.iter().enumerate() {
            let xs: Vec<f64> = (0..n)
                .filter(|&i| group[i] == g)
                .map(|i| res.scores[[i, 0]])
                .collect();
            let ys: Vec<f64> = (0..n)
                .filter(|&i| group[i] == g)
                .map(|i| res.scores[[i, 1]])
                .collect();
            ax.scatter_full(
                &xs,
                &ys,
                Color::cycle(g),
                4.0,
                Marker::Circle,
                0.85,
                Some(&format!("group {}", g + 1)),
            );
        }
        ax.legend(LegendLoc::UpperRight);
    }

    let out = common::img_path("pca.svg");
    fig.save_svg(&out).expect("write pca.svg");
    eprintln!("wrote {}", out.display());
}
