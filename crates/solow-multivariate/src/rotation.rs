//! Factor rotation by the gradient-projection algorithm (GPA).
//!
//! This is a port of the reference's `multivariate.factor_rotation`
//! ([`rotate_factors`]) gradient-projection rotation of Bernaards & Jennrich
//! (2005). Given an unrotated `p × k` loading matrix `A`, a rotation
//! transforms it to a "simpler" loading matrix `L` together with a rotation
//! matrix `T`.
//!
//! * For **orthogonal** rotations (`varimax`, `quartimax`) `T` is orthogonal
//!   and `L = A T`.
//! * For **oblique** rotations (`oblimin`, `quartimin`) `T` is a normal matrix
//!   with unit-norm columns and `L = A (T^{-1})^\top`.
//! * **promax** is an analytic oblique method: it first computes a varimax
//!   solution, builds a power target, solves a Procrustes problem and
//!   normalises; here `L = A T` with a non-orthogonal `T`.
//!
//! The orthogonal and oblique families minimise the *oblimin / orthomax*
//! objective
//!
//! ```text
//! phi(L) = (1/4) <L∘L, (I - gamma C)(L∘L) N>,
//! ```
//!
//! where `N = ones(k,k) - I`, `C = ones(p,p)/p`, `∘` is the elementwise
//! (Hadamard) product and `<X,Y> = tr(Xᵀ Y)`. For orthogonal rotations the
//! orthomax form `phi(L) = -(1/4)<L∘L,(I - gamma C)(L∘L)>` is used, which is
//! equivalent on the orthogonal manifold. With `gamma = 1` this is varimax and
//! with `gamma = 0` quartimax (orthogonal) or quartimin (oblique).

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::{inv, svd};

/// The rotation method requested by [`rotate_factors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationMethod {
    /// Orthogonal varimax (orthomax with `gamma = 1`).
    Varimax,
    /// Orthogonal quartimax (orthomax with `gamma = 0`).
    Quartimax,
    /// Oblique oblimin with the default parameter `gamma = 0`
    /// (equivalent to [`RotationMethod::Quartimin`]).
    Oblimin,
    /// Oblique quartimin (oblimin with `gamma = 0`).
    Quartimin,
    /// Analytic oblique promax with the default power `k = 4`.
    Promax,
}

impl RotationMethod {
    /// Parse the reference's string method names (case-insensitive).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "varimax" => Ok(RotationMethod::Varimax),
            "quartimax" => Ok(RotationMethod::Quartimax),
            "oblimin" => Ok(RotationMethod::Oblimin),
            "quartimin" => Ok(RotationMethod::Quartimin),
            "promax" => Ok(RotationMethod::Promax),
            other => Err(Error::Shape(format!("unknown rotation method: {other}"))),
        }
    }
}

/// The result of a factor rotation: the rotated loadings `L` and the rotation
/// matrix `T`.
#[derive(Clone, Debug)]
pub struct Rotation {
    /// Rotated loadings, `p × k`. For orthogonal rotations and promax
    /// `L = A T`; for oblique rotations `L = A (T^{-1})^\top`.
    pub loadings: Array2<f64>,
    /// Rotation matrix, `k × k`.
    pub rotation: Array2<f64>,
}

/// Rotate the `p × k` loading matrix `loadings` with the named `method`.
///
/// Supported names (case-insensitive): `"varimax"`, `"quartimax"` (orthogonal),
/// `"oblimin"`, `"quartimin"` (oblique) and `"promax"`. Mirrors the reference
/// defaults: gradient-projection with `max_tries = 501`, `tol = 1e-5`, oblimin
/// `gamma = 0`, and promax power `k = 2`.
pub fn rotate_factors(loadings: &Array2<f64>, method: &str) -> Result<Rotation> {
    rotate_with(loadings, RotationMethod::from_name(method)?)
}

/// Rotate using a parsed [`RotationMethod`].
pub fn rotate_with(loadings: &Array2<f64>, method: RotationMethod) -> Result<Rotation> {
    match method {
        RotationMethod::Promax => promax(loadings, 2.0),
        RotationMethod::Varimax => gpa(loadings, 1.0, true),
        RotationMethod::Quartimax => gpa(loadings, 0.0, true),
        // oblimin defaults to gamma = 0, which equals quartimin.
        RotationMethod::Oblimin | RotationMethod::Quartimin => gpa(loadings, 0.0, false),
    }
}

/// Rotate `A` to `L = A T` (orthogonal) or `L = A (T^{-1})^\top` (oblique).
fn rotate_a(a: &Array2<f64>, t: &Array2<f64>, orthogonal: bool) -> Result<Array2<f64>> {
    if orthogonal {
        Ok(a.dot(t))
    } else {
        // L = A (T^{-1})^T.
        let ti = inv(t)?;
        Ok(a.dot(&ti.t()))
    }
}

/// Oblimin / orthomax objective value and gradient `Gphi` with respect to `L`.
///
/// `orthogonal` selects the orthomax form (used on the orthogonal manifold);
/// otherwise the oblimin form is used. Returns `(phi, Gphi)`.
fn objective(l: &Array2<f64>, gamma: f64, orthogonal: bool) -> (f64, Array2<f64>) {
    let (p, k) = l.dim();
    // L2 = L ∘ L.
    let l2 = l.mapv(|x| x * x);

    if orthogonal {
        // orthomax: X = (I - gamma C) L2 ; phi = -<L2, X>/4 ; Gphi = -L ∘ X.
        let x = if gamma == 0.0 {
            l2.clone()
        } else {
            // (I - gamma C) L2 = L2 - (gamma/p) * ones(p,p) @ L2.
            // ones(p,p) @ L2 has every row equal to the column sums of L2.
            let mut col_sums = Array1::<f64>::zeros(k);
            for j in 0..k {
                col_sums[j] = l2.column(j).sum();
            }
            let mut x = l2.clone();
            for i in 0..p {
                for j in 0..k {
                    x[[i, j]] -= (gamma / p as f64) * col_sums[j];
                }
            }
            x
        };
        let phi = -elem_dot(&l2, &x) / 4.0;
        let gphi = -hadamard(l, &x);
        (phi, gphi)
    } else {
        // oblimin: X = (I - gamma C) L2 N ; phi = <L2, X>/4 ; Gphi = L ∘ X.
        // First (I - gamma C) L2:
        let m = if gamma == 0.0 {
            l2.clone()
        } else {
            let mut col_sums = Array1::<f64>::zeros(k);
            for j in 0..k {
                col_sums[j] = l2.column(j).sum();
            }
            let mut m = l2.clone();
            for i in 0..p {
                for j in 0..k {
                    m[[i, j]] -= (gamma / p as f64) * col_sums[j];
                }
            }
            m
        };
        // Then multiply on the right by N = ones(k,k) - I:
        // (M N)[i,c] = sum_{j != c} M[i,j] = rowsum(M)[i] - M[i,c].
        let mut x = Array2::<f64>::zeros((p, k));
        for i in 0..p {
            let row_sum: f64 = (0..k).map(|j| m[[i, j]]).sum();
            for c in 0..k {
                x[[i, c]] = row_sum - m[[i, c]];
            }
        }
        let phi = elem_dot(&l2, &x) / 4.0;
        let gphi = hadamard(l, &x);
        (phi, gphi)
    }
}

/// Elementwise (Hadamard) product `A ∘ B`.
fn hadamard(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut out = a.clone();
    out.zip_mut_with(b, |x, &y| *x *= y);
    out
}

/// Frobenius inner product `sum_{ij} A_{ij} B_{ij}`.
fn elem_dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Frobenius norm of a matrix.
fn fro_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// The gradient-projection algorithm (GPA) for the oblimin/orthomax family.
///
/// `gamma` is the oblimin parameter and `orthogonal` selects the rotation
/// manifold. Mirrors the reference `GPA` with `max_tries = 501`, `tol = 1e-5`
/// and the identity initial guess.
fn gpa(a: &Array2<f64>, gamma: f64, orthogonal: bool) -> Result<Rotation> {
    let max_tries = 501usize;
    let tol = 1e-5;
    let k = a.ncols();

    let mut t = Array2::<f64>::eye(k);
    let mut al = 1.0_f64;

    // Initialise f and G.
    let (mut f, mut g) = compute_fg(a, &t, gamma, orthogonal)?;

    for _ in 0..max_tries {
        // Project the gradient onto the tangent space.
        let gp = if orthogonal {
            // M = Tᵀ G ; S = (M + Mᵀ)/2 ; Gp = G - T S.
            let m = t.t().dot(&g);
            let s = (&m + &m.t()).mapv(|x| x / 2.0);
            &g - &t.dot(&s)
        } else {
            // Gp = G - T diag(sum(T ∘ G, axis=0)).
            let tg = hadamard(&t, &g);
            let mut gp = g.clone();
            for j in 0..k {
                let col_sum: f64 = tg.column(j).sum();
                for i in 0..k {
                    gp[[i, j]] -= t[[i, j]] * col_sum;
                }
            }
            gp
        };

        let s = fro_norm(&gp);
        if s < tol {
            break;
        }

        // Line search. The reference always adopts the last `Tt` evaluated in
        // the inner loop (whether or not it broke on sufficient decrease), so
        // we carry the latest candidate out of the loop.
        al *= 2.0;
        let mut tt = t.clone();
        let mut ft = f;
        let mut g_next = g.clone();
        for _ in 0..11 {
            let x = &t - &gp.mapv(|v| v * al);
            tt = if orthogonal {
                // Tt = U Vᵀ from the thin SVD of X.
                let (u, _s, vt) = svd(&x)?;
                u.dot(&vt)
            } else {
                // Tt = X diag(1/sqrt(colsumsq(X))).
                let mut tt = x.clone();
                for j in 0..k {
                    let nrm: f64 = (0..k).map(|i| x[[i, j]] * x[[i, j]]).sum::<f64>().sqrt();
                    let v = 1.0 / nrm;
                    for i in 0..k {
                        tt[[i, j]] = x[[i, j]] * v;
                    }
                }
                tt
            };

            let (fnew, gnew) = compute_fg(a, &tt, gamma, orthogonal)?;
            ft = fnew;
            g_next = gnew;
            if ft < f - 0.5 * s * s * al {
                break;
            }
            al /= 2.0;
        }

        t = tt;
        f = ft;
        g = g_next;
    }

    let loadings = rotate_a(a, &t, orthogonal)?;
    Ok(Rotation {
        loadings,
        rotation: t,
    })
}

/// Compute the objective value `f` and the manifold gradient `G` at `T`.
fn compute_fg(
    a: &Array2<f64>,
    t: &Array2<f64>,
    gamma: f64,
    orthogonal: bool,
) -> Result<(f64, Array2<f64>)> {
    if orthogonal {
        let l = a.dot(t);
        let (f, gq) = objective(&l, gamma, true);
        let g = a.t().dot(&gq);
        Ok((f, g))
    } else {
        let ti = inv(t)?;
        let l = a.dot(&ti.t());
        let (f, gq) = objective(&l, gamma, false);
        // G = -((Lᵀ Gq Ti)ᵀ).
        let inner = l.t().dot(&gq).dot(&ti);
        let g = inner.t().mapv(|x| -x);
        Ok((f, g))
    }
}

/// Analytic promax rotation with power `power` (the reference default is 2).
///
/// Steps mirror the reference `promax`:
/// 1. varimax-rotate `A` to `V`;
/// 2. build the target `H = |V|^power / V`;
/// 3. solve the Procrustes problem `S = (Aᵀ A)^{-1} Aᵀ H`;
/// 4. normalise: `d = sqrt(diag((Sᵀ S)^{-1}))`, `T = (S diag(d))^{-1}ᵀ`;
/// 5. return `(A T, T)`.
fn promax(a: &Array2<f64>, power: f64) -> Result<Rotation> {
    // 1. varimax target.
    let v = gpa(a, 1.0, true)?.loadings;

    // 2. H = |V|^power / V (elementwise). Where V == 0 the reference yields a
    // NaN/inf; in practice varimax loadings are non-zero. Guard against zeros.
    let h = v.mapv(|x| {
        if x == 0.0 {
            0.0
        } else {
            x.abs().powf(power) / x
        }
    });

    // 3. Procrustes: S = inv(Aᵀ A) Aᵀ H.
    let ata = a.t().dot(a);
    let ata_inv = inv(&ata)?;
    let s = ata_inv.dot(&a.t()).dot(&h);

    // 4. d = sqrt(diag(inv(Sᵀ S))) ; D = diag(d) ; T = inv(S D)ᵀ.
    let sts = s.t().dot(&s);
    let sts_inv = inv(&sts)?;
    let k = s.ncols();
    let mut d = Array1::<f64>::zeros(k);
    for j in 0..k {
        d[j] = sts_inv[[j, j]].sqrt();
    }
    // S D scales column j of S by d[j].
    let mut sd = s.clone();
    for j in 0..k {
        for i in 0..sd.nrows() {
            sd[[i, j]] *= d[j];
        }
    }
    let sd_inv = inv(&sd)?;
    let t = sd_inv.t().to_owned();

    // 5. L = A T.
    let loadings = a.dot(&t);
    Ok(Rotation {
        loadings,
        rotation: t,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample() -> Array2<f64> {
        array![[0.8, 0.1], [0.7, 0.2], [0.2, 0.9], [0.1, 0.8], [0.5, 0.5],]
    }

    #[test]
    fn varimax_is_orthogonal_and_consistent() {
        let a = sample();
        let r = rotate_factors(&a, "varimax").unwrap();
        // T orthogonal: TᵀT = I.
        let tt = r.rotation.t().dot(&r.rotation);
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((tt[[i, j]] - want).abs() < 1e-9, "TᵀT not identity");
            }
        }
        // L = A T.
        let l = a.dot(&r.rotation);
        for i in 0..a.nrows() {
            for j in 0..2 {
                assert!((l[[i, j]] - r.loadings[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn quartimax_runs() {
        let a = sample();
        let r = rotate_factors(&a, "quartimax").unwrap();
        let l = a.dot(&r.rotation);
        for i in 0..a.nrows() {
            for j in 0..2 {
                assert!((l[[i, j]] - r.loadings[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn quartimin_is_oblique_consistent() {
        let a = sample();
        let r = rotate_factors(&a, "quartimin").unwrap();
        // L = A inv(T)ᵀ.
        let l = rotate_a(&a, &r.rotation, false).unwrap();
        for i in 0..a.nrows() {
            for j in 0..2 {
                assert!((l[[i, j]] - r.loadings[[i, j]]).abs() < 1e-12);
            }
        }
        // Oblique T has unit-norm columns.
        for j in 0..2 {
            let nrm: f64 = (0..2)
                .map(|i| r.rotation[[i, j]].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((nrm - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn oblimin_defaults_to_quartimin() {
        let a = sample();
        let r1 = rotate_factors(&a, "oblimin").unwrap();
        let r2 = rotate_factors(&a, "quartimin").unwrap();
        for i in 0..a.nrows() {
            for j in 0..2 {
                assert!((r1.loadings[[i, j]] - r2.loadings[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn promax_consistent() {
        let a = sample();
        let r = rotate_factors(&a, "promax").unwrap();
        // promax returns L = A T.
        let l = a.dot(&r.rotation);
        for i in 0..a.nrows() {
            for j in 0..2 {
                assert!((l[[i, j]] - r.loadings[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn unknown_method_errs() {
        let a = sample();
        assert!(rotate_factors(&a, "nonsense").is_err());
    }
}
