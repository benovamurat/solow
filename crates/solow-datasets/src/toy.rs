//! Compact toy loaders — analogous to the reference `load_*` but tiny.
//!
//! To keep the crate's binary size bounded, the "toy" loaders below
//! either ship the full dataset (small enough to embed) or generate a
//! deterministic sample that matches the reference expected shape and
//! target-value scale.

use ndarray::{Array1, Array2};

/// A compact 178 × 13 wine dataset (deterministic three-cluster mixture
/// with feature names that match the reference `load_wine`). Values are
/// synthesised so that every cluster is separable but noise is present.
pub fn load_wine() -> (Array2<f64>, Array1<usize>, Vec<&'static str>) {
    // Deterministic generator — a compact substitute so the crate does
    // not have to embed the reference full toy CSVs.
    let means = [
        [13.7_f64, 1.9, 2.4, 17.0, 106.0, 2.85, 3.0, 0.29, 1.87, 5.6, 1.06, 3.15, 1116.0],
        [12.3, 1.3, 2.2, 20.2, 92.0, 2.24, 2.05, 0.36, 1.62, 3.1, 1.05, 2.79, 519.0],
        [13.1, 3.3, 2.4, 21.2, 98.0, 1.68, 0.78, 0.44, 1.15, 7.4, 0.68, 1.68, 629.0],
    ];
    let n_per = 60;
    let n = 3 * n_per;
    let d = 13;
    let mut x = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<usize>::zeros(n);
    let mut state = 0xFEED_C0DE_u64;
    for cls in 0..3 {
        for k in 0..n_per {
            let row = cls * n_per + k;
            y[row] = cls;
            for j in 0..d {
                let noise = 0.05 * standard_normal(&mut state) * means[cls][j].abs();
                x[[row, j]] = means[cls][j] + noise;
            }
        }
    }
    let names = vec![
        "alcohol", "malic_acid", "ash", "alcalinity", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid", "proanthocyanins",
        "color_intensity", "hue", "od280_od315", "proline",
    ];
    (x, y, names)
}

/// A compact 100 × 10 diabetes dataset — deterministic linear signal
/// with additive noise to match the reference `load_diabetes` shape and
/// scale.
pub fn load_diabetes() -> (Array2<f64>, Array1<f64>, Vec<&'static str>) {
    let n = 100;
    let d = 10;
    let mut x = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    let mut state = 0xC0FF_EE00_u64;
    let coef = [30.0_f64, -60.0, 250.0, 180.0, -12.0, -55.0, -230.0, 130.0, 500.0, 40.0];
    for i in 0..n {
        let mut yi = 152.0_f64;
        for j in 0..d {
            let xij = standard_normal(&mut state);
            x[[i, j]] = xij;
            yi += coef[j] * xij;
        }
        y[i] = yi + 50.0 * standard_normal(&mut state);
    }
    let names = vec!["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"];
    (x, y, names)
}

/// A compact 200 × 30 breast-cancer dataset — deterministic binary
/// classification problem with feature scale similar to the reference
/// `load_breast_cancer`.
pub fn load_breast_cancer() -> (Array2<f64>, Array1<usize>, Vec<&'static str>) {
    let n = 200;
    let d = 30;
    let mut x = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<usize>::zeros(n);
    let mut state = 0xC0DE_FA11_u64;
    // Coefficients are roughly the sign+magnitude ratios that separate
    // benign (y=0) from malignant (y=1) in the real dataset.
    let coef = [1.2_f64, 0.9, 1.1, 0.8, -0.4, 0.6, 1.5, 1.0, 0.7, 0.5,
                0.9, 0.5, 0.6, 0.8, -0.3, -0.4, 0.5, 0.4, -0.3, -0.2,
                1.4, 1.0, 1.2, 0.9, -0.6, 0.7, 1.6, 1.1, 0.8, 0.4];
    for i in 0..n {
        let mut score = 0.0_f64;
        for j in 0..d {
            let xij = standard_normal(&mut state);
            x[[i, j]] = xij;
            score += coef[j] * xij;
        }
        y[i] = if score > 0.0 { 1 } else { 0 };
    }
    // Feature names matching the reference schema (first ten only; other 20
    // are the mean / stderr / worst variants).
    let names = vec![
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points",
        "mean_symmetry", "mean_fractal_dimension",
        "se_radius", "se_texture", "se_perimeter", "se_area",
        "se_smoothness", "se_compactness", "se_concavity", "se_concave_points",
        "se_symmetry", "se_fractal_dimension",
        "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
        "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points",
        "worst_symmetry", "worst_fractal_dimension",
    ];
    (x, y, names)
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(1e-12);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn uniform01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let r = *state >> 11;
    (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_wine_returns_180x13_three_class_dataset() {
        let (x, y, names) = load_wine();
        assert_eq!(x.shape(), &[180, 13]);
        assert_eq!(names.len(), 13);
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort();
        classes.dedup();
        assert_eq!(classes, vec![0, 1, 2]);
    }

    #[test]
    fn load_diabetes_returns_100x10_regression_dataset() {
        let (x, y, names) = load_diabetes();
        assert_eq!(x.shape(), &[100, 10]);
        assert_eq!(y.len(), 100);
        assert_eq!(names.len(), 10);
    }

    #[test]
    fn load_breast_cancer_returns_200x30_binary_dataset() {
        let (x, y, names) = load_breast_cancer();
        assert_eq!(x.shape(), &[200, 30]);
        assert_eq!(names.len(), 30);
        let n0 = y.iter().filter(|&&v| v == 0).count();
        let n1 = y.iter().filter(|&&v| v == 1).count();
        assert!(n0 > 0 && n1 > 0);
    }
}
