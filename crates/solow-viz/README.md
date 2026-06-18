# solow-viz

A dependency-light, general-purpose plotting backend that renders to SVG
(with optional PNG raster export behind the `png` feature). It aims to be a
small but genuinely useful, matplotlib-class engine: a [`Figure`] holding a
grid of [`Axes`], each carrying its own data, scales, limits, legend, and
styling.

## Quick start

```
use solow_viz::Figure;
let mut fig = Figure::new(640, 480);
fig.axes().set_title("demo").plot(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 4.0, 9.0]);
let svg = fig.to_svg();
assert!(svg.starts_with("<svg"));
```

## Subplots, legends, scales

```
use solow_viz::{Figure, Scale, LegendLoc, Color, Marker, LineStyle};
let mut fig = Figure::subplots(800, 600, 2, 2);
fig.suptitle("gallery");
let ax = fig.ax_at(0, 0).unwrap();
ax.set_yscale(Scale::Log);
ax.line(&[1.0, 2.0, 3.0], &[10.0, 100.0, 1000.0],
        Color::BLUE, 2.0, LineStyle::Dashed, Marker::Circle, 1.0, Some("series"));
ax.legend(LegendLoc::UpperLeft);
assert_eq!(fig.naxes(), 4);
```

The major capabilities:

- **Layout**: [`Figure::subplots`] / [`Figure::ax_at`], a figure-level
  [`Figure::suptitle`], per-axes labels/limits/scales.
- **Scales**: linear, [`Scale::Log`], and [`Scale::Symlog`] with log tick
  locators and formatters; [`Axes::set_xlim`]/[`Axes::set_ylim`];
  [`Axes::invert_xaxis`]/[`Axes::invert_yaxis`].
- **Plot types**: line, scatter, bar (vertical/horizontal/stacked/grouped),
  histogram, error bars, step, fill-between/area, boxplot, violin, and a 2-D
  heatmap with a colorbar.
- **Styling**: [`Marker`]s, [`LineStyle`]s, per-series alpha, the `tab10`
  color cycle plus sequential [`Colormap`]s, spines, tick direction,
  font-size controls, annotations, and reference lines / spans.
- **Legends**: per-series labels and an auto-placed [`Axes::legend`] box.

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-viz) · License: BSD-3-Clause
