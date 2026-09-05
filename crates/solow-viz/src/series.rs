//! Series types and per-series styling (markers, line styles).

use crate::color::{Color, Colormap};

/// Marker glyph drawn at scatter / line vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Marker {
    /// No marker.
    #[default]
    None,
    Circle,
    Square,
    Triangle,
    /// A diagonal cross (`x`).
    Cross,
    /// A plus sign (`+`).
    Plus,
    Diamond,
}

/// Stroke dash pattern for lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LineStyle {
    #[default]
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

impl LineStyle {
    /// The SVG `stroke-dasharray` value, or `None` for a solid line.
    pub fn dasharray(&self) -> Option<&'static str> {
        match self {
            LineStyle::Solid => None,
            LineStyle::Dashed => Some("6 4"),
            LineStyle::Dotted => Some("1.5 3"),
            LineStyle::DashDot => Some("6 3 1.5 3"),
        }
    }
}

/// How a step plot places its risers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StepWhere {
    /// Step changes at the left edge of each interval (default).
    #[default]
    Pre,
    /// Step changes at the right edge.
    Post,
    /// Step changes at the midpoint.
    Mid,
}

/// A drawable data series. Variants are additive; the original
/// `Line`/`Scatter`/`Bar` shapes are preserved for backward compatibility.
#[derive(Clone, Debug)]
pub enum Series {
    Line {
        x: Vec<f64>,
        y: Vec<f64>,
        color: Color,
        width: f64,
        style: LineStyle,
        marker: Marker,
        alpha: f64,
        label: Option<String>,
    },
    Scatter {
        x: Vec<f64>,
        y: Vec<f64>,
        color: Color,
        radius: f64,
        marker: Marker,
        alpha: f64,
        label: Option<String>,
    },
    Bar {
        x: Vec<f64>,
        y: Vec<f64>,
        width: f64,
        color: Color,
        /// Stacking baseline per bar (0 for un-stacked).
        base: Vec<f64>,
        horizontal: bool,
        alpha: f64,
        label: Option<String>,
    },
    /// Vertical error bars at `(x, y)` with half-height `err`.
    ErrorBar {
        x: Vec<f64>,
        y: Vec<f64>,
        yerr: Vec<f64>,
        color: Color,
        capsize: f64,
        label: Option<String>,
    },
    /// A step (staircase) line.
    Step {
        x: Vec<f64>,
        y: Vec<f64>,
        color: Color,
        width: f64,
        style: LineStyle,
        wher: StepWhere,
        label: Option<String>,
    },
    /// Shaded region between `y1` and `y2` over `x`.
    FillBetween {
        x: Vec<f64>,
        y1: Vec<f64>,
        y2: Vec<f64>,
        color: Color,
        alpha: f64,
        label: Option<String>,
    },
    /// One or more box-and-whisker boxes.
    Box {
        /// Per-box statistics.
        boxes: Vec<BoxStats>,
        color: Color,
        width: f64,
        label: Option<String>,
    },
    /// One or more violins (kernel-density silhouettes).
    Violin {
        violins: Vec<ViolinShape>,
        color: Color,
        width: f64,
        alpha: f64,
        label: Option<String>,
    },
    /// A 2-D heatmap / image: `data[row][col]` with a colormap.
    Heatmap {
        data: Vec<Vec<f64>>,
        cmap: Colormap,
        vmin: f64,
        vmax: f64,
        /// Data-coordinate extent `(x0, x1, y0, y1)`.
        extent: (f64, f64, f64, f64),
        label: Option<String>,
    },
}

/// Five-number summary plus outliers for one boxplot box.
#[derive(Clone, Debug)]
pub struct BoxStats {
    /// Center position along the category axis (data units).
    pub pos: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    /// Lower whisker end.
    pub whislo: f64,
    /// Upper whisker end.
    pub whishi: f64,
    pub outliers: Vec<f64>,
}

impl BoxStats {
    /// Compute box statistics from a raw sample at category position `pos`,
    /// using the 1.5*IQR whisker rule.
    pub fn from_sample(pos: f64, data: &[f64]) -> Option<BoxStats> {
        if data.is_empty() {
            return None;
        }
        let mut v = data.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let q1 = quantile(&v, 0.25);
        let median = quantile(&v, 0.50);
        let q3 = quantile(&v, 0.75);
        let iqr = q3 - q1;
        let lo_fence = q1 - 1.5 * iqr;
        let hi_fence = q3 + 1.5 * iqr;
        let whislo = v.iter().copied().find(|&x| x >= lo_fence).unwrap_or(v[0]);
        let whishi = v
            .iter()
            .rev()
            .copied()
            .find(|&x| x <= hi_fence)
            .unwrap_or(*v.last().unwrap());
        let outliers: Vec<f64> = v
            .iter()
            .copied()
            .filter(|&x| x < whislo || x > whishi)
            .collect();
        Some(BoxStats {
            pos,
            q1,
            median,
            q3,
            whislo,
            whishi,
            outliers,
        })
    }
}

/// A symmetric violin silhouette: density evaluated at sorted `y` grid points.
#[derive(Clone, Debug)]
pub struct ViolinShape {
    pub pos: f64,
    /// Grid of y values.
    pub ys: Vec<f64>,
    /// Density at each `ys` (max-normalized to 1).
    pub density: Vec<f64>,
    pub median: f64,
    pub ymin: f64,
    pub ymax: f64,
}

impl ViolinShape {
    /// Build a violin from a sample using a Gaussian KDE on a fixed grid.
    pub fn from_sample(pos: f64, data: &[f64], grid: usize) -> Option<ViolinShape> {
        if data.len() < 2 {
            return None;
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = var.sqrt().max(1e-9);
        // Silverman's rule-of-thumb bandwidth.
        let bw = 1.06 * std * n.powf(-0.2);
        let lo = data.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pad = bw * 2.0;
        let g = grid.max(2);
        let mut ys = Vec::with_capacity(g);
        let mut density = Vec::with_capacity(g);
        for i in 0..g {
            let y = (lo - pad) + (hi - lo + 2.0 * pad) * i as f64 / (g - 1) as f64;
            let d: f64 = data
                .iter()
                .map(|&xi| {
                    let u = (y - xi) / bw;
                    (-0.5 * u * u).exp()
                })
                .sum::<f64>()
                / (n * bw * (2.0 * std::f64::consts::PI).sqrt());
            ys.push(y);
            density.push(d);
        }
        let maxd = density.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
        for d in &mut density {
            *d /= maxd;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        Some(ViolinShape {
            pos,
            ys,
            density,
            median: quantile(&sorted, 0.5),
            ymin: lo,
            ymax: hi,
        })
    }
}

/// Linear-interpolated quantile of a pre-sorted slice.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let h = (sorted.len() as f64 - 1.0) * q.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

impl Series {
    /// The legend label, if this series carries one.
    pub fn label(&self) -> Option<&str> {
        match self {
            Series::Line { label, .. }
            | Series::Scatter { label, .. }
            | Series::Bar { label, .. }
            | Series::ErrorBar { label, .. }
            | Series::Step { label, .. }
            | Series::FillBetween { label, .. }
            | Series::Box { label, .. }
            | Series::Violin { label, .. }
            | Series::Heatmap { label, .. } => label.as_deref(),
        }
    }

    /// The representative color used in a legend swatch.
    pub fn legend_color(&self) -> Color {
        match self {
            Series::Line { color, .. }
            | Series::Scatter { color, .. }
            | Series::Bar { color, .. }
            | Series::ErrorBar { color, .. }
            | Series::Step { color, .. }
            | Series::FillBetween { color, .. }
            | Series::Box { color, .. }
            | Series::Violin { color, .. } => *color,
            Series::Heatmap { cmap, .. } => cmap.sample(0.7),
        }
    }
}
