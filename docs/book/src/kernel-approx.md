# Kernel approximation

`solow-kernel-approx` covers random Fourier features (RBF sampler), the Nyström method, and explicit feature maps for chi-squared, skewed chi-squared, and polynomial kernels.

These transformers let linear models emulate kernel methods with predictable memory and time budgets, on data sizes where a full Gram matrix would not fit.

## Transformers

| Transformer | What it does |
|---|---|
| `RBFSampler` | Rahimi-Recht (2007) random Fourier features approximating the RBF kernel. |
| `Nystroem` | Nyström approximation of an arbitrary kernel using a random subset of training rows. |
| `AdditiveChi2Sampler` | Vedaldi-Zisserman explicit feature map for the additive chi-squared kernel. |
| `SkewedChi2Sampler` | Explicit feature map for the skewed chi-squared kernel. |
| `PolynomialCountSketch` | Pham-Pagh tensor sketch for polynomial kernels. |

## Example

```rust
use solow_kernel_approx::RBFSampler;

let sampler = RBFSampler::new(500).gamma(0.1).seed(42).fit(&x)?;
let x_rff = sampler.transform(&x)?;
// x_rff now has 500 random Fourier features; feed to any linear model.
# Ok::<_, solow_core::Error>(())
```

## References

- Rahimi, A., & Recht, B. (2007). *Random features for large-scale kernel machines*. NIPS.
- Williams, C. K. I., & Seeger, M. (2001). *Using the Nyström method to speed up kernel machines*. NIPS.
- Vedaldi, A., & Zisserman, A. (2012). *Efficient additive kernels via explicit feature maps*. IEEE TPAMI.
