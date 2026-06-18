//! Criterion benchmarks for the core Solow estimators.
//!
//! These measure *speed*, not correctness: there is no reference fixture here
//! (correctness is covered by the per-crate fixture tests). All inputs are
//! generated deterministically by a small linear-congruential generator so that
//! the workloads are reproducible and never seeded from the wall clock.
//!
//! Workloads:
//!   - OLS fit (`n = 100` and `n = 1000`, `k = 5` predictors + intercept)
//!   - GLM Poisson IRLS fit (log link)
//!   - discrete Logit Newton fit
//!   - a `200 x 50` economy SVD
//!   - a `100 x 100` symmetric eigendecomposition
//!   - normal and Student-t `cdf` + `ppf` throughput

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use ndarray::{Array1, Array2};

use solow::core::tools::{add_constant, HasConstant};
use solow::discrete::Logit;
use solow::distributions::{norm_cdf, norm_ppf, t_cdf, t_ppf};
use solow::glm::{Family, Glm, Link};
use solow::linalg::{eigh, svd};
use solow::regression::LinearModel;

/// Deterministic linear-congruential generator (Numerical Recipes constants),
/// producing reproducible pseudo-random `f64` in `[0, 1)` with no clock seed.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    fn unif(&mut self) -> f64 {
        // Use the top 53 bits for a full-precision mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately standard-normal via a 12-uniform sum (Irwin-Hall).
    #[inline]
    fn normal(&mut self) -> f64 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unif();
        }
        s - 6.0
    }
}

/// Build a deterministic design matrix `X` (n x k, no intercept) of standard-normal
/// draws plus an intercept column, returning `(design, true_beta)`.
fn make_design(n: usize, k: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg::new(seed);
    let mut raw = Array2::<f64>::zeros((n, k));
    for v in raw.iter_mut() {
        *v = rng.normal();
    }
    let design = add_constant(&raw, true, HasConstant::Add).unwrap();
    // True coefficients including intercept (k + 1 of them), small magnitude.
    let mut beta = Array1::<f64>::zeros(k + 1);
    for b in beta.iter_mut() {
        *b = 0.5 * rng.normal();
    }
    beta[0] = 0.2; // intercept
    (design, beta)
}

/// Continuous Gaussian response: `y = X beta + noise`.
fn make_ols_data(n: usize, k: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let (x, beta) = make_design(n, k, seed);
    let mut rng = Lcg::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut y = x.dot(&beta);
    for v in y.iter_mut() {
        *v += 0.3 * rng.normal();
    }
    (y, x)
}

/// Poisson counts with a log link: `mu = exp(X beta)`, `y ~ Poisson(mu)`
/// approximated deterministically by rounding `mu` plus bounded jitter so the
/// IRLS workload is realistic but reproducible.
fn make_poisson_data(n: usize, k: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let (x, mut beta) = make_design(n, k, seed);
    // Keep the linear predictor small so counts stay modest.
    beta.mapv_inplace(|b| 0.25 * b);
    beta[0] = 0.4;
    let eta = x.dot(&beta);
    let mut rng = Lcg::new(seed ^ 0xD1B5_4A32_D192_ED03);
    let y = eta.mapv(|e| {
        let mu = e.exp();
        let jitter = (rng.unif() - 0.5) * mu.sqrt();
        (mu + jitter).round().max(0.0)
    });
    (y, x)
}

/// Binary response for a Logit fit: `p = logistic(X beta)`, thresholded
/// deterministically against a uniform draw.
fn make_logit_data(n: usize, k: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let (x, beta) = make_design(n, k, seed);
    let eta = x.dot(&beta);
    let mut rng = Lcg::new(seed ^ 0x2545_F491_4F6C_DD1D);
    let y = eta.mapv(|e| {
        let p = 1.0 / (1.0 + (-e).exp());
        if rng.unif() < p {
            1.0
        } else {
            0.0
        }
    });
    (y, x)
}

/// Deterministic dense matrix of standard-normal draws.
fn make_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = Lcg::new(seed);
    let mut a = Array2::<f64>::zeros((rows, cols));
    for v in a.iter_mut() {
        *v = rng.normal();
    }
    a
}

/// A well-conditioned symmetric positive-definite matrix `A = M Mᵀ + d·I`.
fn make_symmetric(n: usize, seed: u64) -> Array2<f64> {
    let m = make_matrix(n, n, seed);
    let mut a = m.dot(&m.t());
    for i in 0..n {
        a[[i, i]] += n as f64;
    }
    a
}

fn bench_ols(c: &mut Criterion) {
    let mut g = c.benchmark_group("ols_fit");
    for &n in &[100usize, 1000usize] {
        let (y, x) = make_ols_data(n, 5, 0x00A1_1CE5);
        g.bench_function(format!("n{n}_k5"), |b| {
            b.iter(|| {
                let model = LinearModel::ols(black_box(y.clone()), black_box(x.clone())).unwrap();
                let res = model.fit().unwrap();
                black_box(res.params.clone())
            })
        });
    }
    g.finish();
}

fn bench_glm_poisson(c: &mut Criterion) {
    let mut g = c.benchmark_group("glm_poisson_irls");
    let (y, x) = make_poisson_data(500, 5, 0x0000_B00B);
    g.bench_function("n500_k5", |b| {
        b.iter_batched(
            || (y.clone(), x.clone()),
            |(yy, xx)| {
                let model = Glm::with_link(yy, xx, Family::Poisson, Link::Log).unwrap();
                let res = model.fit().unwrap();
                black_box(res.params.clone())
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn bench_logit(c: &mut Criterion) {
    let mut g = c.benchmark_group("logit_newton");
    let (y, x) = make_logit_data(500, 5, 0x00C0_FFEE);
    g.bench_function("n500_k5", |b| {
        b.iter_batched(
            || (y.clone(), x.clone()),
            |(yy, xx)| {
                let model = Logit::new(yy, xx).unwrap();
                let res = model.fit().unwrap();
                black_box(res.params.clone())
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn bench_linalg(c: &mut Criterion) {
    let mut g = c.benchmark_group("linalg");

    let a = make_matrix(200, 50, 0xDEAD_BEEF);
    g.bench_function("svd_200x50", |b| {
        b.iter(|| {
            let (u, s, vt) = svd(black_box(&a)).unwrap();
            black_box((u, s, vt))
        })
    });

    let sym = make_symmetric(100, 0xFEED_FACE);
    g.bench_function("eigh_100x100", |b| {
        b.iter(|| {
            let (w, v) = eigh(black_box(&sym)).unwrap();
            black_box((w, v))
        })
    });

    g.finish();
}

fn bench_distributions(c: &mut Criterion) {
    let mut g = c.benchmark_group("dist_throughput");

    // A deterministic spread of evaluation points across the bulk of the
    // distribution; ppf is evaluated on probabilities in (0, 1).
    let n = 1024usize;
    let xs: Vec<f64> = (0..n)
        .map(|i| -4.0 + 8.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let ps: Vec<f64> = (0..n)
        .map(|i| 1e-4 + (1.0 - 2e-4) * (i as f64) / (n as f64 - 1.0))
        .collect();

    g.bench_function("norm_cdf", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &x in &xs {
                acc += norm_cdf(black_box(x));
            }
            black_box(acc)
        })
    });
    g.bench_function("norm_ppf", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &p in &ps {
                acc += norm_ppf(black_box(p));
            }
            black_box(acc)
        })
    });
    g.bench_function("t_cdf_df5", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &x in &xs {
                acc += t_cdf(black_box(x), 5.0);
            }
            black_box(acc)
        })
    });
    g.bench_function("t_ppf_df5", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &p in &ps {
                acc += t_ppf(black_box(p), 5.0);
            }
            black_box(acc)
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_ols,
    bench_glm_poisson,
    bench_logit,
    bench_linalg,
    bench_distributions
);
criterion_main!(benches);
