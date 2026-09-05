//! A single set of axes: data series, scales, limits, styling, and annotations.

use crate::color::{Color, Colormap};
use crate::scale::Scale;
use crate::series::{BoxStats, LineStyle, Marker, Series, StepWhere, ViolinShape};

/// Where to place the auto legend box within the axes frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LegendLoc {
    #[default]
    UpperRight,
    UpperLeft,
    LowerRight,
    LowerLeft,
}

/// Direction tick marks point relative to the axis spine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TickDirection {
    /// Ticks point outward, away from the plot (default).
    #[default]
    Out,
    /// Ticks point inward, into the plot.
    In,
}

/// A text annotation anchored at a data coordinate.
#[derive(Clone, Debug)]
pub struct Annotation {
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub color: Color,
    pub font_size: f64,
}

/// A horizontal or vertical reference line spanning the axes.
#[derive(Clone, Debug)]
pub struct RefLine {
    /// `true` for a horizontal line at `value` (a y level); `false` vertical.
    pub horizontal: bool,
    pub value: f64,
    pub color: Color,
    pub width: f64,
    pub style: LineStyle,
}

/// A shaded band spanning the full opposite axis between `lo` and `hi`.
#[derive(Clone, Debug)]
pub struct RefSpan {
    pub horizontal: bool,
    pub lo: f64,
    pub hi: f64,
    pub color: Color,
    pub alpha: f64,
}

/// Which spines (frame edges) are drawn.
#[derive(Clone, Copy, Debug)]
pub struct Spines {
    pub top: bool,
    pub right: bool,
    pub bottom: bool,
    pub left: bool,
}

impl Default for Spines {
    fn default() -> Self {
        Spines {
            top: true,
            right: true,
            bottom: true,
            left: true,
        }
    }
}

/// A single set of axes onto which series are drawn.
#[derive(Clone, Debug)]
pub struct Axes {
    pub(crate) series: Vec<Series>,
    pub(crate) xlim: Option<(f64, f64)>,
    pub(crate) ylim: Option<(f64, f64)>,
    pub(crate) title: Option<String>,
    pub(crate) xlabel: Option<String>,
    pub(crate) ylabel: Option<String>,
    pub(crate) grid: bool,
    pub(crate) next_color: usize,
    pub(crate) xscale: Scale,
    pub(crate) yscale: Scale,
    pub(crate) xinvert: bool,
    pub(crate) yinvert: bool,
    pub(crate) legend: Option<LegendLoc>,
    pub(crate) spines: Spines,
    pub(crate) tick_dir: TickDirection,
    pub(crate) tick_font_size: f64,
    pub(crate) label_font_size: f64,
    pub(crate) title_font_size: f64,
    pub(crate) annotations: Vec<Annotation>,
    pub(crate) reflines: Vec<RefLine>,
    pub(crate) refspans: Vec<RefSpan>,
    /// Heatmap colorbar request (the source series index).
    pub(crate) colorbar: Option<usize>,
}

impl Default for Axes {
    fn default() -> Self {
        Axes {
            series: Vec::new(),
            xlim: None,
            ylim: None,
            title: None,
            xlabel: None,
            ylabel: None,
            grid: false,
            next_color: 0,
            xscale: Scale::Linear,
            yscale: Scale::Linear,
            xinvert: false,
            yinvert: false,
            legend: None,
            spines: Spines::default(),
            tick_dir: TickDirection::Out,
            tick_font_size: 11.0,
            label_font_size: 12.0,
            title_font_size: 15.0,
            annotations: Vec::new(),
            reflines: Vec::new(),
            refspans: Vec::new(),
            colorbar: None,
        }
    }
}

impl Axes {
    pub(crate) fn auto_color(&mut self) -> Color {
        let c = Color::cycle(self.next_color);
        self.next_color += 1;
        c
    }

    // ---- Line / scatter --------------------------------------------------

    /// Add a line series connecting `(x, y)`.
    pub fn plot(&mut self, x: &[f64], y: &[f64]) -> &mut Self {
        let color = self.auto_color();
        self.series.push(Series::Line {
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            width: 1.6,
            style: LineStyle::Solid,
            marker: Marker::None,
            alpha: 1.0,
            label: None,
        });
        self
    }

    /// Add a line series with an explicit color and width.
    pub fn plot_styled(&mut self, x: &[f64], y: &[f64], color: Color, width: f64) -> &mut Self {
        self.series.push(Series::Line {
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            width,
            style: LineStyle::Solid,
            marker: Marker::None,
            alpha: 1.0,
            label: None,
        });
        self
    }

    /// Add a fully-specified line series (color/width/style/marker/alpha/label).
    #[allow(clippy::too_many_arguments)]
    pub fn line(
        &mut self,
        x: &[f64],
        y: &[f64],
        color: Color,
        width: f64,
        style: LineStyle,
        marker: Marker,
        alpha: f64,
        label: Option<&str>,
    ) -> &mut Self {
        self.series.push(Series::Line {
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            width,
            style,
            marker,
            alpha,
            label: label.map(str::to_string),
        });
        self
    }

    /// Add a scatter series.
    pub fn scatter(&mut self, x: &[f64], y: &[f64]) -> &mut Self {
        let color = self.auto_color();
        self.series.push(Series::Scatter {
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            radius: 3.0,
            marker: Marker::Circle,
            alpha: 1.0,
            label: None,
        });
        self
    }

    /// Add a scatter series with an explicit color and marker radius.
    pub fn scatter_styled(&mut self, x: &[f64], y: &[f64], color: Color, radius: f64) -> &mut Self {
        self.series.push(Series::Scatter {
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            radius,
            marker: Marker::Circle,
            alpha: 1.0,
            label: None,
        });
        self
    }

    /// Add a scatter series with full styling.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_full(
        &mut self,
        x: &[f64],
        y: &[f64],
        color: Color,
        radius: f64,
        marker: Marker,
        alpha: f64,
        label: Option<&str>,
    ) -> &mut Self {
        self.series.push(Series::Scatter {
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            radius,
            marker,
            alpha,
            label: label.map(str::to_string),
        });
        self
    }

    // ---- Bars ------------------------------------------------------------

    /// Add a bar series with bar centers `x`, heights `y`, and bar `width`.
    pub fn bar(&mut self, x: &[f64], y: &[f64], width: f64) -> &mut Self {
        let color = self.auto_color();
        self.series.push(Series::Bar {
            x: x.to_vec(),
            y: y.to_vec(),
            width,
            color,
            base: vec![0.0; x.len()],
            horizontal: false,
            alpha: 0.85,
            label: None,
        });
        self
    }

    /// Add a labelled bar series with an explicit color.
    pub fn bar_styled(
        &mut self,
        x: &[f64],
        y: &[f64],
        width: f64,
        color: Color,
        label: Option<&str>,
    ) -> &mut Self {
        self.series.push(Series::Bar {
            x: x.to_vec(),
            y: y.to_vec(),
            width,
            color,
            base: vec![0.0; x.len()],
            horizontal: false,
            alpha: 0.85,
            label: label.map(str::to_string),
        });
        self
    }

    /// Add a horizontal bar series (positions `y`, lengths `x`).
    pub fn barh(
        &mut self,
        y: &[f64],
        x: &[f64],
        height: f64,
        color: Color,
        label: Option<&str>,
    ) -> &mut Self {
        self.series.push(Series::Bar {
            x: y.to_vec(),
            y: x.to_vec(),
            width: height,
            color,
            base: vec![0.0; y.len()],
            horizontal: true,
            alpha: 0.85,
            label: label.map(str::to_string),
        });
        self
    }

    /// Add a stacked-bar group: each row of `series_y` is stacked on the
    /// previous one at the shared `x` positions.
    pub fn bar_stacked(
        &mut self,
        x: &[f64],
        series_y: &[Vec<f64>],
        width: f64,
        labels: &[&str],
    ) -> &mut Self {
        let mut base = vec![0.0; x.len()];
        for (k, ys) in series_y.iter().enumerate() {
            let color = self.auto_color();
            let label = labels.get(k).map(|s| s.to_string());
            self.series.push(Series::Bar {
                x: x.to_vec(),
                y: ys.clone(),
                width,
                color,
                base: base.clone(),
                horizontal: false,
                alpha: 0.9,
                label,
            });
            for (b, v) in base.iter_mut().zip(ys.iter()) {
                *b += *v;
            }
        }
        self
    }

    /// Add a grouped-bar cluster: each row of `series_y` is offset within the
    /// slot of width `group_width` centered on each `x`.
    pub fn bar_grouped(
        &mut self,
        x: &[f64],
        series_y: &[Vec<f64>],
        group_width: f64,
        labels: &[&str],
    ) -> &mut Self {
        let k = series_y.len().max(1);
        let bw = group_width / k as f64;
        for (j, ys) in series_y.iter().enumerate() {
            let color = self.auto_color();
            let label = labels.get(j).map(|s| s.to_string());
            let offset = -group_width / 2.0 + bw * (j as f64 + 0.5);
            let centers: Vec<f64> = x.iter().map(|&xi| xi + offset).collect();
            self.series.push(Series::Bar {
                x: centers,
                y: ys.clone(),
                width: bw * 0.9,
                color,
                base: vec![0.0; x.len()],
                horizontal: false,
                alpha: 0.9,
                label,
            });
        }
        self
    }

    /// Bin `data` into `bins` equal-width bins and draw a histogram.
    pub fn hist(&mut self, data: &[f64], bins: usize) -> &mut Self {
        if data.is_empty() || bins == 0 {
            return self;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in data {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if lo == hi {
            hi = lo + 1.0;
        }
        let w = (hi - lo) / bins as f64;
        let mut counts = vec![0.0_f64; bins];
        for &v in data {
            let mut b = ((v - lo) / w).floor() as isize;
            if b < 0 {
                b = 0;
            }
            if b as usize >= bins {
                b = bins as isize - 1;
            }
            counts[b as usize] += 1.0;
        }
        let centers: Vec<f64> = (0..bins).map(|i| lo + (i as f64 + 0.5) * w).collect();
        let color = self.auto_color();
        self.series.push(Series::Bar {
            x: centers,
            y: counts,
            width: w * 0.98,
            color,
            base: vec![0.0; bins],
            horizontal: false,
            alpha: 0.85,
            label: None,
        });
        self
    }

    // ---- Statistical / specialized --------------------------------------

    /// Add vertical error bars at `(x, y)` with half-heights `yerr`.
    pub fn errorbar(
        &mut self,
        x: &[f64],
        y: &[f64],
        yerr: &[f64],
        color: Color,
        label: Option<&str>,
    ) -> &mut Self {
        self.series.push(Series::ErrorBar {
            x: x.to_vec(),
            y: y.to_vec(),
            yerr: yerr.to_vec(),
            color,
            capsize: 4.0,
            label: label.map(str::to_string),
        });
        self
    }

    /// Add a step (staircase) line.
    pub fn step(
        &mut self,
        x: &[f64],
        y: &[f64],
        color: Color,
        wher: StepWhere,
        label: Option<&str>,
    ) -> &mut Self {
        self.series.push(Series::Step {
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            width: 1.6,
            style: LineStyle::Solid,
            wher,
            label: label.map(str::to_string),
        });
        self
    }

    /// Shade the region between `y1` and `y2` over `x` (area/fill-between).
    pub fn fill_between(
        &mut self,
        x: &[f64],
        y1: &[f64],
        y2: &[f64],
        color: Color,
        alpha: f64,
        label: Option<&str>,
    ) -> &mut Self {
        self.series.push(Series::FillBetween {
            x: x.to_vec(),
            y1: y1.to_vec(),
            y2: y2.to_vec(),
            color,
            alpha,
            label: label.map(str::to_string),
        });
        self
    }

    /// Shade the area between `y` and zero (filled curve).
    pub fn area(&mut self, x: &[f64], y: &[f64], color: Color, alpha: f64) -> &mut Self {
        let zeros = vec![0.0; x.len()];
        self.fill_between(x, y, &zeros, color, alpha, None)
    }

    /// Draw a boxplot from raw samples; box `k` sits at position `positions[k]`.
    pub fn boxplot(&mut self, samples: &[Vec<f64>], positions: &[f64]) -> &mut Self {
        let boxes: Vec<BoxStats> = samples
            .iter()
            .enumerate()
            .filter_map(|(k, s)| {
                let pos = positions.get(k).copied().unwrap_or(k as f64 + 1.0);
                BoxStats::from_sample(pos, s)
            })
            .collect();
        let color = self.auto_color();
        self.series.push(Series::Box {
            boxes,
            color,
            width: 0.5,
            label: None,
        });
        self
    }

    /// Draw a violin plot from raw samples.
    pub fn violinplot(&mut self, samples: &[Vec<f64>], positions: &[f64]) -> &mut Self {
        let violins: Vec<ViolinShape> = samples
            .iter()
            .enumerate()
            .filter_map(|(k, s)| {
                let pos = positions.get(k).copied().unwrap_or(k as f64 + 1.0);
                ViolinShape::from_sample(pos, s, 64)
            })
            .collect();
        let color = self.auto_color();
        self.series.push(Series::Violin {
            violins,
            color,
            width: 0.7,
            alpha: 0.6,
            label: None,
        });
        self
    }

    /// Draw a 2-D heatmap of `data` (`data[row][col]`), optionally with a
    /// colorbar. `extent` is the data-coordinate box `(x0, x1, y0, y1)`.
    pub fn heatmap(
        &mut self,
        data: &[Vec<f64>],
        cmap: Colormap,
        extent: (f64, f64, f64, f64),
        colorbar: bool,
    ) -> &mut Self {
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for row in data {
            for &v in row {
                if v.is_finite() {
                    vmin = vmin.min(v);
                    vmax = vmax.max(v);
                }
            }
        }
        if !vmin.is_finite() {
            vmin = 0.0;
            vmax = 1.0;
        }
        if vmin == vmax {
            vmax = vmin + 1.0;
        }
        let idx = self.series.len();
        self.series.push(Series::Heatmap {
            data: data.to_vec(),
            cmap,
            vmin,
            vmax,
            extent,
            label: None,
        });
        if colorbar {
            self.colorbar = Some(idx);
        }
        self
    }

    // ---- Annotations / reference marks ----------------------------------

    /// Place text at a data coordinate.
    pub fn annotate(&mut self, x: f64, y: f64, text: &str) -> &mut Self {
        self.annotations.push(Annotation {
            x,
            y,
            text: text.to_string(),
            color: Color::BLACK,
            font_size: 11.0,
        });
        self
    }

    /// Place colored text at a data coordinate.
    pub fn annotate_styled(
        &mut self,
        x: f64,
        y: f64,
        text: &str,
        color: Color,
        font_size: f64,
    ) -> &mut Self {
        self.annotations.push(Annotation {
            x,
            y,
            text: text.to_string(),
            color,
            font_size,
        });
        self
    }

    /// Draw a horizontal reference line at y = `y`.
    pub fn axhline(&mut self, y: f64, color: Color, style: LineStyle) -> &mut Self {
        self.reflines.push(RefLine {
            horizontal: true,
            value: y,
            color,
            width: 1.2,
            style,
        });
        self
    }

    /// Draw a vertical reference line at x = `x`.
    pub fn axvline(&mut self, x: f64, color: Color, style: LineStyle) -> &mut Self {
        self.reflines.push(RefLine {
            horizontal: false,
            value: x,
            color,
            width: 1.2,
            style,
        });
        self
    }

    /// Shade a horizontal band between `y0` and `y1`.
    pub fn axhspan(&mut self, y0: f64, y1: f64, color: Color, alpha: f64) -> &mut Self {
        self.refspans.push(RefSpan {
            horizontal: true,
            lo: y0.min(y1),
            hi: y0.max(y1),
            color,
            alpha,
        });
        self
    }

    /// Shade a vertical band between `x0` and `x1`.
    pub fn axvspan(&mut self, x0: f64, x1: f64, color: Color, alpha: f64) -> &mut Self {
        self.refspans.push(RefSpan {
            horizontal: false,
            lo: x0.min(x1),
            hi: x0.max(x1),
            color,
            alpha,
        });
        self
    }

    // ---- Setters ---------------------------------------------------------

    pub fn set_xlim(&mut self, lo: f64, hi: f64) -> &mut Self {
        self.xlim = Some((lo, hi));
        self
    }
    pub fn set_ylim(&mut self, lo: f64, hi: f64) -> &mut Self {
        self.ylim = Some((lo, hi));
        self
    }
    pub fn set_title(&mut self, s: &str) -> &mut Self {
        self.title = Some(s.to_string());
        self
    }
    pub fn set_xlabel(&mut self, s: &str) -> &mut Self {
        self.xlabel = Some(s.to_string());
        self
    }
    pub fn set_ylabel(&mut self, s: &str) -> &mut Self {
        self.ylabel = Some(s.to_string());
        self
    }
    pub fn set_grid(&mut self, on: bool) -> &mut Self {
        self.grid = on;
        self
    }
    pub fn set_xscale(&mut self, scale: Scale) -> &mut Self {
        self.xscale = scale;
        self
    }
    pub fn set_yscale(&mut self, scale: Scale) -> &mut Self {
        self.yscale = scale;
        self
    }
    /// Invert the x axis (largest value on the left).
    pub fn invert_xaxis(&mut self) -> &mut Self {
        self.xinvert = true;
        self
    }
    /// Invert the y axis (largest value at the bottom).
    pub fn invert_yaxis(&mut self) -> &mut Self {
        self.yinvert = true;
        self
    }
    /// Enable an auto-placed legend at `loc`.
    pub fn legend(&mut self, loc: LegendLoc) -> &mut Self {
        self.legend = Some(loc);
        self
    }
    /// Turn individual spines on or off.
    pub fn set_spines(&mut self, top: bool, right: bool, bottom: bool, left: bool) -> &mut Self {
        self.spines = Spines {
            top,
            right,
            bottom,
            left,
        };
        self
    }
    pub fn set_tick_direction(&mut self, dir: TickDirection) -> &mut Self {
        self.tick_dir = dir;
        self
    }
    /// Set tick-label, axis-label, and title font sizes (px).
    pub fn set_font_sizes(&mut self, tick: f64, label: f64, title: f64) -> &mut Self {
        self.tick_font_size = tick;
        self.label_font_size = label;
        self.title_font_size = title;
        self
    }

    /// `true` if this axes holds no drawable series.
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// Auto-computed data bounds `(xmin, xmax, ymin, ymax)` across all series.
    pub(crate) fn data_bounds(&self) -> (f64, f64, f64, f64) {
        let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut ax = |x: f64| {
            if x.is_finite() {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
            }
        };
        let mut accx: Vec<f64> = Vec::new();
        let mut accy: Vec<f64> = Vec::new();
        macro_rules! pushx {
            ($v:expr) => {
                accx.push($v)
            };
        }
        macro_rules! pushy {
            ($v:expr) => {
                accy.push($v)
            };
        }
        for s in &self.series {
            match s {
                Series::Line { x, y, .. } | Series::Scatter { x, y, .. } => {
                    for &v in x {
                        pushx!(v);
                    }
                    for &v in y {
                        pushy!(v);
                    }
                }
                Series::Step { x, y, .. } => {
                    for &v in x {
                        pushx!(v);
                    }
                    for &v in y {
                        pushy!(v);
                    }
                }
                Series::Bar {
                    x,
                    y,
                    width,
                    base,
                    horizontal,
                    ..
                } => {
                    if *horizontal {
                        for (i, &xi) in x.iter().enumerate() {
                            pushy!(xi - width / 2.0);
                            pushy!(xi + width / 2.0);
                            let b = base.get(i).copied().unwrap_or(0.0);
                            pushx!(b);
                            pushx!(b + y[i]);
                        }
                        pushx!(0.0);
                    } else {
                        for (i, &xi) in x.iter().enumerate() {
                            pushx!(xi - width / 2.0);
                            pushx!(xi + width / 2.0);
                            let b = base.get(i).copied().unwrap_or(0.0);
                            pushy!(b);
                            pushy!(b + y[i]);
                        }
                        pushy!(0.0);
                    }
                }
                Series::ErrorBar { x, y, yerr, .. } => {
                    for &v in x {
                        pushx!(v);
                    }
                    for (i, &yi) in y.iter().enumerate() {
                        let e = yerr.get(i).copied().unwrap_or(0.0);
                        pushy!(yi - e);
                        pushy!(yi + e);
                    }
                }
                Series::FillBetween { x, y1, y2, .. } => {
                    for &v in x {
                        pushx!(v);
                    }
                    for &v in y1 {
                        pushy!(v);
                    }
                    for &v in y2 {
                        pushy!(v);
                    }
                }
                Series::Box { boxes, width, .. } => {
                    for b in boxes {
                        pushx!(b.pos - width / 2.0);
                        pushx!(b.pos + width / 2.0);
                        pushy!(b.whislo);
                        pushy!(b.whishi);
                        for &o in &b.outliers {
                            pushy!(o);
                        }
                    }
                }
                Series::Violin { violins, width, .. } => {
                    for v in violins {
                        pushx!(v.pos - width / 2.0);
                        pushx!(v.pos + width / 2.0);
                        pushy!(v.ymin);
                        pushy!(v.ymax);
                    }
                }
                Series::Heatmap { extent, .. } => {
                    pushx!(extent.0);
                    pushx!(extent.1);
                    pushy!(extent.2);
                    pushy!(extent.3);
                }
            }
        }
        for v in accx {
            ax(v);
        }
        for v in accy {
            if v.is_finite() {
                ymin = ymin.min(v);
                ymax = ymax.max(v);
            }
        }
        if !xmin.is_finite() {
            xmin = 0.0;
            xmax = 1.0;
            ymin = 0.0;
            ymax = 1.0;
        }
        if xmin == xmax {
            xmin -= 0.5;
            xmax += 0.5;
        }
        if ymin == ymax {
            ymin -= 0.5;
            ymax += 0.5;
        }
        (xmin, xmax, ymin, ymax)
    }
}
