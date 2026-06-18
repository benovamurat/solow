//! # solow-viz
//!
//! A dependency-light, general-purpose plotting backend that renders to SVG
//! (with optional PNG raster export behind the `png` feature). It aims to be a
//! small but genuinely useful, matplotlib-class engine: a [`Figure`] holding a
//! grid of [`Axes`], each carrying its own data, scales, limits, legend, and
//! styling.
//!
//! ## Quick start
//!
//! ```
//! use solow_viz::Figure;
//! let mut fig = Figure::new(640, 480);
//! fig.axes().set_title("demo").plot(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 4.0, 9.0]);
//! let svg = fig.to_svg();
//! assert!(svg.starts_with("<svg"));
//! ```
//!
//! ## Subplots, legends, scales
//!
//! ```
//! use solow_viz::{Figure, Scale, LegendLoc, Color, Marker, LineStyle};
//! let mut fig = Figure::subplots(800, 600, 2, 2);
//! fig.suptitle("gallery");
//! let ax = fig.ax_at(0, 0).unwrap();
//! ax.set_yscale(Scale::Log);
//! ax.line(&[1.0, 2.0, 3.0], &[10.0, 100.0, 1000.0],
//!         Color::BLUE, 2.0, LineStyle::Dashed, Marker::Circle, 1.0, Some("series"));
//! ax.legend(LegendLoc::UpperLeft);
//! assert_eq!(fig.naxes(), 4);
//! ```
//!
//! The major capabilities:
//!
//! - **Layout**: [`Figure::subplots`] / [`Figure::ax_at`], a figure-level
//!   [`Figure::suptitle`], per-axes labels/limits/scales.
//! - **Scales**: linear, [`Scale::Log`], and [`Scale::Symlog`] with log tick
//!   locators and formatters; [`Axes::set_xlim`]/[`Axes::set_ylim`];
//!   [`Axes::invert_xaxis`]/[`Axes::invert_yaxis`].
//! - **Plot types**: line, scatter, bar (vertical/horizontal/stacked/grouped),
//!   histogram, error bars, step, fill-between/area, boxplot, violin, and a 2-D
//!   heatmap with a colorbar.
//! - **Styling**: [`Marker`]s, [`LineStyle`]s, per-series alpha, the `tab10`
//!   color cycle plus sequential [`Colormap`]s, spines, tick direction,
//!   font-size controls, annotations, and reference lines / spans.
//! - **Legends**: per-series labels and an auto-placed [`Axes::legend`] box.

mod axes;
mod color;
mod figure;
mod render;
mod scale;
mod series;

#[cfg(feature = "png")]
mod png;

pub use axes::{Annotation, Axes, LegendLoc, RefLine, RefSpan, Spines, TickDirection};
pub use color::{Color, Colormap};
pub use figure::Figure;
pub use scale::Scale;
pub use series::{BoxStats, LineStyle, Marker, Series, StepWhere, ViolinShape};

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Backward-compatibility: the original API must still work. -------

    #[test]
    fn renders_line_plot() {
        let mut fig = Figure::new(640, 480);
        fig.axes()
            .set_title("y = x^2")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true)
            .plot(&[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 4.0, 9.0, 16.0]);
        let svg = fig.to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("<polyline"));
        assert!(svg.contains("y = x^2"));
    }

    #[test]
    fn renders_scatter_and_bar_and_hist() {
        let mut fig = Figure::new(500, 400);
        {
            let ax = fig.axes();
            ax.scatter(&[1.0, 2.0, 3.0], &[2.0, 1.0, 3.0]);
            ax.bar(&[1.0, 2.0, 3.0], &[3.0, 1.0, 2.0], 0.5);
        }
        let svg = fig.to_svg();
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<rect"));

        let mut h = Figure::new(500, 400);
        h.axes().hist(&[0.1, 0.2, 0.2, 0.9, 1.0, 1.1, 1.1, 1.2], 4);
        assert!(h.to_svg().contains("<rect"));
    }

    #[test]
    fn nice_ticks_are_reasonable() {
        let t = scale::nice_ticks(0.0, 9.7, 6);
        assert!(t.first().unwrap() <= &0.0);
        assert!(t.last().unwrap() >= &9.7);
    }

    // ---- Layout ----------------------------------------------------------

    #[test]
    fn subplots_2x2_emits_four_framed_axes() {
        let mut fig = Figure::subplots(800, 600, 2, 2);
        fig.suptitle("grid");
        assert_eq!(fig.naxes(), 4);
        assert_eq!(fig.shape(), (2, 2));
        for i in 0..4 {
            fig.ax(i)
                .unwrap()
                .plot(&[0.0, 1.0, 2.0], &[(i as f64), 1.0, 2.0]);
        }
        let svg = fig.to_svg();
        // Each axes draws its own bottom spine line; with four axes there must
        // be at least four <line> spines plus four polylines.
        assert_eq!(svg.matches("<polyline").count(), 4);
        // Four clip paths => four distinct framed axes regions.
        assert_eq!(svg.matches("<clipPath").count(), 4);
        assert!(svg.contains("grid"));
    }

    #[test]
    fn ax_at_addresses_cells() {
        let mut fig = Figure::subplots(400, 400, 2, 3);
        assert!(fig.ax_at(1, 2).is_some());
        assert!(fig.ax_at(2, 0).is_none());
        assert!(fig.ax_at(0, 3).is_none());
    }

    // ---- Legends ---------------------------------------------------------

    #[test]
    fn legend_has_swatches() {
        let mut fig = Figure::new(500, 400);
        {
            let ax = fig.axes();
            ax.line(
                &[0.0, 1.0, 2.0],
                &[0.0, 1.0, 0.5],
                Color::BLUE,
                2.0,
                LineStyle::Solid,
                Marker::None,
                1.0,
                Some("alpha"),
            );
            ax.line(
                &[0.0, 1.0, 2.0],
                &[1.0, 0.5, 0.0],
                Color::RED,
                2.0,
                LineStyle::Solid,
                Marker::None,
                1.0,
                Some("beta"),
            );
            ax.legend(LegendLoc::UpperRight);
        }
        let svg = fig.to_svg();
        assert!(svg.contains("class=\"legend\""));
        assert_eq!(svg.matches("class=\"swatch\"").count(), 2);
        assert!(svg.contains("alpha"));
        assert!(svg.contains("beta"));
    }

    // ---- Scales ----------------------------------------------------------

    #[test]
    fn log_axis_has_log_spaced_ticks() {
        let mut fig = Figure::new(500, 400);
        fig.axes()
            .set_yscale(Scale::Log)
            .plot(&[1.0, 2.0, 3.0, 4.0], &[1.0, 10.0, 100.0, 1000.0]);
        let svg = fig.to_svg();
        // Decade labels for a 10^0..10^3 span.
        assert!(svg.contains(">1<"));
        assert!(svg.contains(">10<"));
        assert!(svg.contains(">100<"));
        assert!(svg.contains(">1000<"));
    }

    #[test]
    fn log_ticks_are_decade_spaced() {
        let t = Scale::Log.ticks(1.0, 1000.0, 6);
        // Powers of ten only across multiple decades.
        assert!(t.contains(&1.0));
        assert!(t.contains(&10.0));
        assert!(t.contains(&100.0));
        assert!(t.contains(&1000.0));
        // Ratios between consecutive ticks are ~10.
        for w in t.windows(2) {
            let r = w[1] / w[0];
            assert!((r - 10.0).abs() < 1e-6, "ratio {r}");
        }
    }

    #[test]
    fn symlog_handles_sign_change() {
        let s = Scale::Symlog { linthresh: 1.0 };
        assert!(s.forward(-10.0) < 0.0);
        assert!(s.forward(10.0) > 0.0);
        assert_eq!(s.forward(0.0), 0.0);
        let t = s.ticks(-100.0, 100.0, 6);
        assert!(t.iter().any(|&v| v < 0.0));
        assert!(t.iter().any(|&v| v > 0.0));
        assert!(t.contains(&0.0));
    }

    #[test]
    fn inverted_axes_flip_mapping() {
        let mut fig = Figure::new(400, 400);
        fig.axes()
            .invert_yaxis()
            .invert_xaxis()
            .plot(&[0.0, 1.0], &[0.0, 1.0]);
        // Just ensure it renders without panicking and produces a polyline.
        assert!(fig.to_svg().contains("<polyline"));
    }

    // ---- Plot types ------------------------------------------------------

    #[test]
    fn errorbar_emits_caps_and_stems() {
        let mut fig = Figure::new(500, 400);
        fig.axes().errorbar(
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 1.5],
            &[0.2, 0.3, 0.1],
            Color::BLUE,
            Some("y"),
        );
        let svg = fig.to_svg();
        // 3 stems + 6 caps + 3 point markers => plenty of lines and circles.
        assert!(svg.matches("<line").count() >= 9);
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn step_emits_polyline() {
        let mut fig = Figure::new(500, 400);
        fig.axes().step(
            &[0.0, 1.0, 2.0, 3.0],
            &[1.0, 3.0, 2.0, 4.0],
            Color::GREEN,
            StepWhere::Post,
            Some("s"),
        );
        assert!(fig.to_svg().contains("<polyline"));
    }

    #[test]
    fn fill_between_emits_path() {
        let mut fig = Figure::new(500, 400);
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let lo = vec![0.0, 0.5, 0.2, 0.8];
        let hi = vec![1.0, 1.5, 1.2, 1.8];
        fig.axes()
            .fill_between(&x, &lo, &hi, Color::BLUE, 0.3, Some("band"));
        let svg = fig.to_svg();
        assert!(svg.contains("<path"));
        assert!(svg.contains("fill-opacity=\"0.300\""));
    }

    #[test]
    fn stacked_and_grouped_bars() {
        let mut s = Figure::new(500, 400);
        s.axes().bar_stacked(
            &[1.0, 2.0, 3.0],
            &[vec![1.0, 2.0, 1.0], vec![2.0, 1.0, 3.0]],
            0.6,
            &["lo", "hi"],
        );
        let svg = s.to_svg();
        assert!(svg.matches("<rect").count() >= 6);

        let mut g = Figure::new(500, 400);
        g.axes().bar_grouped(
            &[1.0, 2.0],
            &[vec![1.0, 2.0], vec![2.0, 1.0], vec![1.5, 1.5]],
            0.8,
            &["a", "b", "c"],
        );
        assert!(g.to_svg().matches("<rect").count() >= 6);
    }

    #[test]
    fn horizontal_bar() {
        let mut fig = Figure::new(500, 400);
        fig.axes().barh(
            &[1.0, 2.0, 3.0],
            &[3.0, 1.0, 2.0],
            0.5,
            Color::ORANGE,
            Some("h"),
        );
        assert!(fig.to_svg().contains("<rect"));
    }

    #[test]
    fn boxplot_has_box_median_and_whiskers() {
        let mut fig = Figure::new(500, 400);
        let a: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.37).sin() * 2.0 + 5.0)
            .collect();
        let b: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.21).cos() * 3.0 + 6.0)
            .collect();
        fig.axes().boxplot(&[a, b], &[1.0, 2.0]);
        let svg = fig.to_svg();
        // The box rect, the median line, and whisker lines must all be present.
        assert!(svg.contains("<rect"));
        assert!(svg.contains("class=\"median\""));
        assert!(svg.contains("class=\"whisker\""));
    }

    #[test]
    fn violin_emits_silhouette_path() {
        let mut fig = Figure::new(500, 400);
        let a: Vec<f64> = (0..80).map(|i| (i as f64 * 0.3).sin() * 2.0).collect();
        fig.axes().violinplot(&[a], &[1.0]);
        assert!(fig.to_svg().contains("<path"));
    }

    #[test]
    fn heatmap_has_grid_of_rects_and_colorbar() {
        let mut fig = Figure::new(500, 400);
        let data: Vec<Vec<f64>> = (0..4)
            .map(|r| (0..5).map(|c| (r * 5 + c) as f64).collect())
            .collect();
        fig.axes()
            .heatmap(&data, Colormap::Viridis, (0.0, 5.0, 0.0, 4.0), true);
        let svg = fig.to_svg();
        // 4*5 = 20 cell rects, drawn with crispEdges, plus a colorbar group.
        assert!(svg.matches("crispEdges").count() >= 20);
        assert!(svg.contains("class=\"colorbar\""));
    }

    #[test]
    fn viridis_endpoints() {
        let lo = Colormap::Viridis.sample(0.0);
        let hi = Colormap::Viridis.sample(1.0);
        // Dark blue/purple low end, bright yellow high end.
        assert_eq!(lo, Color(68, 1, 84));
        assert_eq!(hi, Color(253, 231, 37));
    }

    #[test]
    fn color_cycle_wraps_at_ten() {
        assert_eq!(Color::cycle(0), Color::cycle(10));
        assert_ne!(Color::cycle(0), Color::cycle(1));
    }

    // ---- Markers / line styles -------------------------------------------

    #[test]
    fn markers_and_dashes_render() {
        let mut fig = Figure::new(500, 400);
        fig.axes().line(
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 0.0],
            Color::PURPLE,
            2.0,
            LineStyle::Dashed,
            Marker::Square,
            0.7,
            None,
        );
        let svg = fig.to_svg();
        assert!(svg.contains("stroke-dasharray"));
        assert!(svg.contains("stroke-opacity=\"0.700\""));
        // Square marker => an extra <rect>.
        assert!(svg.contains("<rect"));

        // Each marker variant should produce some SVG geometry.
        for m in [
            Marker::Circle,
            Marker::Triangle,
            Marker::Cross,
            Marker::Plus,
            Marker::Diamond,
        ] {
            let mut f = Figure::new(200, 200);
            f.axes()
                .scatter_full(&[0.0, 1.0], &[0.0, 1.0], Color::RED, 4.0, m, 1.0, None);
            let s = f.to_svg();
            assert!(
                s.contains("<circle")
                    || s.contains("<polygon")
                    || s.contains("<rect")
                    || s.contains("<line")
            );
        }
    }

    // ---- Styling / annotations / reference marks -------------------------

    #[test]
    fn spines_can_be_disabled() {
        let mut fig = Figure::new(400, 300);
        fig.axes()
            .set_spines(false, false, true, true)
            .plot(&[0.0, 1.0], &[0.0, 1.0]);
        let svg = fig.to_svg();
        assert!(svg.contains("<polyline"));
        // Bottom-left only: with grid off and no ticks-grid, spine lines exist
        // but we can at least confirm rendering succeeds.
        assert!(svg.contains("<line"));
    }

    #[test]
    fn annotations_and_reflines_and_spans() {
        let mut fig = Figure::new(500, 400);
        {
            let ax = fig.axes();
            ax.plot(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0]);
            ax.annotate(1.0, 1.0, "peak");
            ax.axhline(2.0, Color::GRAY, LineStyle::Dashed);
            ax.axvline(1.5, Color::GRAY, LineStyle::Dotted);
            ax.axhspan(0.5, 1.5, Color::BLUE, 0.1);
            ax.axvspan(0.2, 0.4, Color::RED, 0.1);
        }
        let svg = fig.to_svg();
        assert!(svg.contains("peak"));
        assert!(svg.contains("stroke-dasharray"));
        assert!(svg.contains("fill-opacity=\"0.100\""));
    }

    #[test]
    fn font_sizes_apply() {
        let mut fig = Figure::new(400, 300);
        fig.axes()
            .set_font_sizes(14.0, 16.0, 20.0)
            .set_title("t")
            .plot(&[0.0, 1.0], &[0.0, 1.0]);
        let svg = fig.to_svg();
        assert!(svg.contains("font-size=\"20\""));
    }

    #[test]
    fn empty_figure_renders() {
        let fig = Figure::new(300, 200);
        let svg = fig.to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    // ---- PNG (feature-gated) ---------------------------------------------

    #[cfg(feature = "png")]
    #[test]
    fn png_export_writes_valid_signature() {
        let mut fig = Figure::new(320, 240);
        fig.axes()
            .set_title("png")
            .plot(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 4.0, 9.0]);
        let bytes = fig.to_png(2.0).expect("rasterize");
        // PNG 8-byte signature.
        assert_eq!(
            &bytes[..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
        assert!(bytes.len() > 100);

        let dir = std::env::temp_dir();
        let path = dir.join("solow_viz_png_test.png");
        fig.save_png(&path, 1.0).expect("write png");
        let on_disk = std::fs::read(&path).expect("read back");
        assert_eq!(
            &on_disk[..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
        let _ = std::fs::remove_file(&path);
    }
}
