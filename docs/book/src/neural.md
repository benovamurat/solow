# Neural networks

`solow-neural` covers small feed-forward multi-layer perceptrons for classification and regression, plus a Bernoulli restricted Boltzmann machine. Every model is deterministic under a caller seed. Weight initialisation uses Glorot with an MMIX-LCG PRNG so a fixed seed reproduces the fit bit-for-bit across runs and platforms.

## Estimators

| Estimator | What it does |
|---|---|
| `MlpClassifier` | Multi-layer perceptron with softmax output and cross-entropy loss. Any number of hidden layers with configurable widths. |
| `MlpRegressor` | Multi-layer perceptron with linear output and mean squared error loss. |
| `BernoulliRbm` | Bernoulli restricted Boltzmann machine trained with contrastive divergence. |

## Activations

`Identity` (no non-linearity, for linear regression), `Logistic` (sigmoid), `Tanh`, `ReLU`.

## Optimizers

- `Sgd` with optional momentum and learning-rate schedule.
- `Adam` (Kingma-Ba 2014) with configurable β₁, β₂, ε.

## Example

```rust
use solow_neural::{MlpClassifier, Activation, Solver};

let mlp = MlpClassifier::builder()
    .hidden_layer_sizes(vec![64, 32])
    .activation(Activation::ReLU)
    .solver(Solver::Adam)
    .max_iter(200)
    .seed(42)
    .fit(&x, &y)?;

let proba = mlp.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization*. arXiv:1412.6980.
- Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS.
