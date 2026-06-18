# solow-linalg

Pure-Rust dense linear algebra for the Solow statistical stack. No system
LAPACK/BLAS is required: the decompositions are implemented here and validated
against an authoritative reference.

- [`cholesky`] — `A = L Lᵀ`
- [`lu_factor`] — `P A = L U` (powers [`solve`], [`inv`], [`det`])
- [`qr`] — economy Householder QR
- [`svd`] — economy one-sided Jacobi SVD
- [`eigh`] — symmetric eigendecomposition (cyclic Jacobi)
- [`pinv`] / [`lstsq`] / [`matrix_rank`] — SVD-based

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-linalg) · License: BSD-3-Clause
