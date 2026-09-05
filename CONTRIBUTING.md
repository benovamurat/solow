# Contributing to Solow

Thank you for your interest in Solow — a from-scratch, pure-Rust
re-implementation of the canonical statistics stack. This document describes how
to add a model, how the verification discipline works, and the one naming rule
that all contributions must follow.

## Ground rules

- **Pure Rust.** The numerical core (linear algebra, special functions,
  distributions) is implemented in Rust with no system LAPACK/BLAS dependency.
  Do not introduce a dependency that pulls in a native math library; the only
  foundational dependency is `ndarray`.
- **Stable toolchain.** Solow builds on stable Rust (see `rust-toolchain.toml`).
- **Everything is verified.** Every model is cross-checked against an
  authoritative reference implementation through committed golden fixtures
  (see below). A model without a fixture-backed test will not be merged.

## Before you push

Run the same checks CI runs:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

If you touched the user guide, build it too (see
[Building the book](#building-the-book)).

## The naming rule

Solow re-implements the behavior of an authoritative Python reference, but **the
name of that reference package must never appear anywhere in this
repository.** Concretely:

- Do not write the reference package's name (the one formed by joining the words
  "stats" and "models") in source, comments, docs, tests, fixtures, or commit
  messages. Refer to it only as "an authoritative reference" or "the reference."
- `numpy`, `scipy`, and `patsy` *may* be named directly — they are ordinary
  scientific-Python dependencies of the fixture generators, not the reference
  being shadowed.
- The reference modeling package is loaded **indirectly** in the fixture
  generators, by importing a module name supplied through an environment
  variable:

  ```python
  import importlib, os
  pkg = os.environ["SOLOW_REFERENCE"]
  model_mod = importlib.import_module(pkg + ".<submodule>")
  ```

  This keeps the package name out of the source tree entirely; it lives only in
  the environment at generation time.

A simple way to self-check before opening a pull request:

```sh
# Should print nothing.
git grep -n -i -e 'stats''models' .
```

(The taboo token is the two words joined with no space; it is split here only so
this very file does not contain it.)

## Adding a new model

A model lands in three coordinated pieces: the estimator, a fixture generator,
and a fixture-backed test.

### 1. Implement the estimator

- Put it in the crate that matches its category (e.g. a new regression
  estimator in `solow-regression`, a new count model in `solow-discrete`). If it
  is a genuinely new category, add a crate to the workspace `members` list in
  the root `Cargo.toml` and follow the existing crate layout.
- Follow the established shape: a model type built from `endog` / `exog` (you do
  **not** add an intercept automatically — the caller passes the constant
  column), a `fit()` returning a results struct, and the standard inference
  quantities as public fields (`params`, `bse`, `tvalues` / `zvalues`,
  `pvalues`, information criteria, etc.).
- Re-export the new public types from the crate's `lib.rs`, and add the crate to
  the umbrella `solow` re-exports if it is a new crate.
- Document the public API with rustdoc, including a runnable doctest.

### 2. Add a reference fixture generator

Fixtures live as committed JSON under `tests/fixtures/`, produced by a generator
script in `tools/reference/`. The Rust tests read the frozen JSON, so **no
Python is needed at test time** — only when (re)generating fixtures.

- Add or extend a `gen_<area>.py` script. It must:
  - load the reference package indirectly through `SOLOW_REFERENCE` (see the
    [naming rule](#the-naming-rule)), never by a literal import;
  - build inputs with `numpy` (use a fixed RNG seed for reproducibility);
  - fit the reference model and dump every quantity your Rust results expose
    into the JSON, as plain numbers / arrays.
- Regenerate the fixture:

  ```sh
  SOLOW_REFERENCE=<reference-import-name> python3 tools/reference/gen_<area>.py
  ```

- Commit the resulting `tests/fixtures/<area>.json`. Reviewers independently
  confirm the golden values are genuine before merge.

### 3. Add a fixture-backed test

- In the crate's `tests/` directory, read the committed JSON and assert that the
  Solow estimator reproduces each reference quantity.
- Use `approx` (`assert_abs_diff_eq!` / `assert_relative_eq!`) with an
  appropriate tolerance. Match the precision the rest of the suite holds to:
  most quantities to `1e-8`, maximum-likelihood / variational-Bayes estimates to
  `1e-6`, and the formula engine to `1e-12`. Justify any looser tolerance in a
  comment.

A change is complete only when `cargo test --workspace` exercises the new model
against its fixture and passes.

## Documentation

User-facing docs live in the mdBook under `docs/book/`. If your model is one a
user would reach for, add a short how-to (or extend an existing chapter) with a
real, compilable Rust snippet that matches the crate's actual API.

### Building the book

```sh
cargo install mdbook --locked
mdbook build docs/book
mdbook serve docs/book   # live preview at http://localhost:3000
```

CI builds the book on every pull request, so keep the snippets and links
healthy.

## Pull requests

- Keep changes focused; one model (or one coherent feature) per pull request.
- Make sure `fmt`, `clippy -D warnings`, `test --workspace`, and the docs build
  are all green.
- Update `CHANGELOG.md` under an "Unreleased" heading when you add or change
  user-visible behavior.

By contributing you agree that your work is licensed under the project's
BSD-3-Clause license.
