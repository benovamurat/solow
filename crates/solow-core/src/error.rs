//! Error and result types shared across the Solow workspace.

use thiserror::Error;

/// The error type returned by fallible Solow operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
    /// A linear-algebra routine failed (non-convergence, ill-conditioning, …).
    #[error("linear algebra error: {0}")]
    Linalg(String),

    /// A matrix was singular or not positive-definite where invertibility was required.
    #[error("singular matrix: {0}")]
    Singular(String),

    /// An iterative procedure failed to converge.
    #[error("convergence failure: {0}")]
    Convergence(String),

    /// An input value was invalid (out of domain, wrong sign, …).
    #[error("invalid value: {0}")]
    Value(String),

    /// Array shapes were incompatible.
    #[error("shape mismatch: {0}")]
    Shape(String),

    /// The requested feature is recognized but not yet implemented.
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

/// Convenience alias for `Result<T, solow_core::Error>`.
pub type Result<T> = core::result::Result<T, Error>;
