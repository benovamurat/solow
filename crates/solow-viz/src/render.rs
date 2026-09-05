//! SVG rendering of a single [`Axes`] into a pixel rectangle.

use std::fmt::Write as _;

use crate::axes::{Axes, LegendLoc, TickDirection};
use crate::color::Color;
use crate::scale::Scale;
use crate::series::{Marker, Series, StepWhere};

/// A pixel rectangle the axes draws into.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Escape XML-special characters in text nodes / attributes.
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Resolved drawing geometry for one axes: the inner plot box and the
/// data->pixel transforms.
struct Frame {
    inner: Rect,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    xscale: Scale,
    yscale: Scale,
    xinvert: bool,
    yinvert: bool,
}

impl Frame {
    fn sx(&self, x: f64) -> f64 {
        let a = self.xscale.forward(x);
        let lo = self.xscale.forward(self.xmin);
        let hi = self.xscale.forward(self.xmax);
        let t = if (hi - lo).abs() < 1e-300 {
            0.0
        } else {
            (a - lo) / (hi - lo)
        };
        let t = if self.xinvert { 1.0 - t } else { t };
        self.inner.x + t * self.inner.w
    }
    fn sy(&self, y: f64) -> f64 {
        let a = self.yscale.forward(y);
        let lo = self.yscale.forward(self.ymin);
        let hi = self.yscale.forward(self.ymax);
        let t = if (hi - lo).abs() < 1e-300 {
            0.0
        } else {
            (a - lo) / (hi - lo)
        };
        let t = if self.yinvert { 1.0 - t } else { t };
        // Pixel y grows downward; flip so larger data is higher on screen.
        self.inner.y + (1.0 - t) * self.inner.h
    }
}

/// Render `ax` into `rect`, appending SVG into `s`.
pub fn render_axes(s: &mut String, ax: &Axes, rect: Rect) {
    // Inner plot box (leave margin for ticks/labels inside the cell).
    let has_ylabel = ax.ylabel.is_some();
    let has_xlabel = ax.xlabel.is_some();
    let has_title = ax.title.is_some();
    let needs_cbar = ax.colorbar.is_some();
    let ml = 50.0 + if has_ylabel { 16.0 } else { 0.0 };
    let mr = 14.0 + if needs_cbar { 56.0 } else { 0.0 };
    let mt = 14.0 + if has_title { 20.0 } else { 0.0 };
    let mb = 34.0 + if has_xlabel { 16.0 } else { 0.0 };
    let inner = Rect {
        x: rect.x + ml,
        y: rect.y + mt,
        w: (rect.w - ml - mr).max(1.0),
        h: (rect.h - mt - mb).max(1.0),
    };

    let (dxmin, dxmax, dymin, dymax) = ax.data_bounds();
    let (rxmin, rxmax) = ax
        .xlim
        .unwrap_or_else(|| pad_range(dxmin, dxmax, ax.xscale));
    let (rymin, rymax) = ax
        .ylim
        .unwrap_or_else(|| pad_range(dymin, dymax, ax.yscale));
    let (xmin, xmax) = ax.xscale.sanitize_range(rxmin, rxmax);
    let (ymin, ymax) = ax.yscale.sanitize_range(rymin, rymax);

    let f = Frame {
        inner,
        xmin,
        xmax,
        ymin,
        ymax,
        xscale: ax.xscale,
        yscale: ax.yscale,
        xinvert: ax.xinvert,
        yinvert: ax.yinvert,
    };

    let xticks = ax.xscale.ticks(xmin, xmax, 6);
    let yticks = ax.yscale.ticks(ymin, ymax, 6);

    // Clip path so series don't bleed outside the plot box.
    let clip_id = format!("clip{}_{}", rect.x as i64, rect.y as i64);
    let _ = write!(
        s,
        "<clipPath id=\"{clip_id}\"><rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\"/></clipPath>",
        inner.x, inner.y, inner.w, inner.h
    );

    // Reference spans (drawn first, behind everything).
    for span in &ax.refspans {
        if span.horizontal {
            let y0 = f.sy(span.lo.clamp(ymin, ymax));
            let y1 = f.sy(span.hi.clamp(ymin, ymax));
            let _ = write!(
                s,
                "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\" fill-opacity=\"{:.3}\"/>",
                inner.x, y1.min(y0), inner.w, (y0 - y1).abs(), span.color.hex(), span.alpha
            );
        } else {
            let x0 = f.sx(span.lo.clamp(xmin, xmax));
            let x1 = f.sx(span.hi.clamp(xmin, xmax));
            let _ = write!(
                s,
                "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\" fill-opacity=\"{:.3}\"/>",
                x0.min(x1), inner.y, (x1 - x0).abs(), inner.h, span.color.hex(), span.alpha
            );
        }
    }

    // Grid.
    if ax.grid {
        for &t in &xticks {
            if t < xmin || t > xmax {
                continue;
            }
            let px = f.sx(t);
            let _ = write!(
                s,
                "<line x1=\"{px:.2}\" y1=\"{:.2}\" x2=\"{px:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"1\"/>",
                inner.y, inner.y + inner.h, Color::LIGHT_GRAY.hex()
            );
        }
        for &t in &yticks {
            if t < ymin || t > ymax {
                continue;
            }
            let py = f.sy(t);
            let _ = write!(
                s,
                "<line x1=\"{:.2}\" y1=\"{py:.2}\" x2=\"{:.2}\" y2=\"{py:.2}\" stroke=\"{}\" stroke-width=\"1\"/>",
                inner.x, inner.x + inner.w, Color::LIGHT_GRAY.hex()
            );
        }
    }

    // Series (clipped).
    let _ = write!(s, "<g clip-path=\"url(#{clip_id})\">");
    for series in &ax.series {
        render_series(s, &f, series);
    }
    // Reference lines on top of series but inside clip.
    for rl in &ax.reflines {
        let dash = rl
            .style
            .dasharray()
            .map(|d| format!(" stroke-dasharray=\"{d}\""))
            .unwrap_or_default();
        if rl.horizontal {
            if rl.value < ymin || rl.value > ymax {
                continue;
            }
            let py = f.sy(rl.value);
            let _ = write!(
                s,
                "<line x1=\"{:.2}\" y1=\"{py:.2}\" x2=\"{:.2}\" y2=\"{py:.2}\" stroke=\"{}\" stroke-width=\"{}\"{}/>",
                inner.x, inner.x + inner.w, rl.color.hex(), rl.width, dash
            );
        } else {
            if rl.value < xmin || rl.value > xmax {
                continue;
            }
            let px = f.sx(rl.value);
            let _ = write!(
                s,
                "<line x1=\"{px:.2}\" y1=\"{:.2}\" x2=\"{px:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"{}\"{}/>",
                inner.y, inner.y + inner.h, rl.color.hex(), rl.width, dash
            );
        }
    }
    // Annotations (inside clip so they track the data box).
    for a in &ax.annotations {
        let px = f.sx(a.x);
        let py = f.sy(a.y);
        let _ = write!(
            s,
            "<text x=\"{px:.2}\" y=\"{py:.2}\" font-size=\"{}\" fill=\"{}\" text-anchor=\"middle\">{}</text>",
            a.font_size,
            a.color.hex(),
            escape(&a.text)
        );
    }
    let _ = write!(s, "</g>");

    // Spines.
    let spine = Color::BLACK.hex();
    let mut edge = |x1: f64, y1: f64, x2: f64, y2: f64| {
        let _ = write!(
            s,
            "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" stroke=\"{spine}\" stroke-width=\"1\"/>"
        );
    };
    let (ix, iy, iw, ih) = (inner.x, inner.y, inner.w, inner.h);
    if ax.spines.top {
        edge(ix, iy, ix + iw, iy);
    }
    if ax.spines.bottom {
        edge(ix, iy + ih, ix + iw, iy + ih);
    }
    if ax.spines.left {
        edge(ix, iy, ix, iy + ih);
    }
    if ax.spines.right {
        edge(ix + iw, iy, ix + iw, iy + ih);
    }

    // Ticks + labels.
    let tdir = if ax.tick_dir == TickDirection::In {
        -1.0
    } else {
        1.0
    };
    let tlen = 5.0;
    for &t in &xticks {
        if t < xmin - 1e-9 || t > xmax + 1e-9 {
            continue;
        }
        let px = f.sx(t);
        let y0 = inner.y + inner.h;
        let _ = write!(
            s,
            "<line x1=\"{px:.2}\" y1=\"{y0:.2}\" x2=\"{px:.2}\" y2=\"{:.2}\" stroke=\"{spine}\"/>",
            y0 + tlen * tdir
        );
        let _ = write!(
            s,
            "<text x=\"{px:.2}\" y=\"{:.2}\" font-size=\"{}\" text-anchor=\"middle\">{}</text>",
            y0 + 16.0,
            ax.tick_font_size,
            escape(&ax.xscale.format(t))
        );
    }
    for &t in &yticks {
        if t < ymin - 1e-9 || t > ymax + 1e-9 {
            continue;
        }
        let py = f.sy(t);
        let _ = write!(
            s,
            "<line x1=\"{:.2}\" y1=\"{py:.2}\" x2=\"{ix:.2}\" y2=\"{py:.2}\" stroke=\"{spine}\"/>",
            ix - tlen * tdir
        );
        let _ = write!(
            s,
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{}\" text-anchor=\"end\">{}</text>",
            ix - 8.0,
            py + 4.0,
            ax.tick_font_size,
            escape(&ax.yscale.format(t))
        );
    }

    // Titles / axis labels.
    if let Some(ref title) = ax.title {
        let _ = write!(
            s,
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{}\" font-weight=\"bold\" text-anchor=\"middle\">{}</text>",
            ix + iw / 2.0,
            iy - 8.0,
            ax.title_font_size,
            escape(title)
        );
    }
    if let Some(ref xl) = ax.xlabel {
        let _ = write!(
            s,
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{}\" text-anchor=\"middle\">{}</text>",
            ix + iw / 2.0,
            iy + ih + 30.0,
            ax.label_font_size,
            escape(xl)
        );
    }
    if let Some(ref yl) = ax.ylabel {
        let cx = rect.x + 16.0;
        let cy = iy + ih / 2.0;
        let _ = write!(
            s,
            "<text x=\"{cx:.2}\" y=\"{cy:.2}\" font-size=\"{}\" text-anchor=\"middle\" transform=\"rotate(-90 {cx:.2} {cy:.2})\">{}</text>",
            ax.label_font_size,
            escape(yl)
        );
    }

    // Colorbar.
    if let Some(idx) = ax.colorbar {
        if let Some(Series::Heatmap {
            cmap, vmin, vmax, ..
        }) = ax.series.get(idx)
        {
            render_colorbar(s, &f, *cmap, *vmin, *vmax, mr, ax.tick_font_size);
        }
    }

    // Legend.
    if let Some(loc) = ax.legend {
        render_legend(s, ax, &f, loc);
    }
}

/// Add symmetric padding (5%) to an auto-range.
fn pad_range(lo: f64, hi: f64, scale: Scale) -> (f64, f64) {
    if scale.is_log() {
        return (lo, hi);
    }
    let span = hi - lo;
    if span <= 0.0 {
        return (lo, hi);
    }
    (lo - span * 0.05, hi + span * 0.05)
}

fn marker_svg(s: &mut String, marker: Marker, cx: f64, cy: f64, r: f64, fill: &str, alpha: f64) {
    let op = format!(" fill-opacity=\"{alpha:.3}\"");
    match marker {
        Marker::None => {}
        Marker::Circle => {
            let _ = write!(
                s,
                "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{r:.2}\" fill=\"{fill}\"{op}/>"
            );
        }
        Marker::Square => {
            let _ = write!(
                s,
                "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{fill}\"{op}/>",
                cx - r,
                cy - r,
                2.0 * r,
                2.0 * r
            );
        }
        Marker::Triangle => {
            let _ = write!(
                s,
                "<polygon points=\"{:.2},{:.2} {:.2},{:.2} {:.2},{:.2}\" fill=\"{fill}\"{op}/>",
                cx,
                cy - r,
                cx - r,
                cy + r,
                cx + r,
                cy + r
            );
        }
        Marker::Diamond => {
            let _ = write!(
                s,
                "<polygon points=\"{:.2},{:.2} {:.2},{:.2} {:.2},{:.2} {:.2},{:.2}\" fill=\"{fill}\"{op}/>",
                cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy
            );
        }
        Marker::Cross => {
            let _ = write!(
                s,
                "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{fill}\" stroke-width=\"{:.2}\"/>\
                 <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{fill}\" stroke-width=\"{:.2}\"/>",
                cx - r, cy - r, cx + r, cy + r, r * 0.6,
                cx - r, cy + r, cx + r, cy - r, r * 0.6
            );
        }
        Marker::Plus => {
            let _ = write!(
                s,
                "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{fill}\" stroke-width=\"{:.2}\"/>\
                 <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{fill}\" stroke-width=\"{:.2}\"/>",
                cx - r, cy, cx + r, cy, r * 0.6,
                cx, cy - r, cx, cy + r, r * 0.6
            );
        }
    }
}

fn render_series(s: &mut String, f: &Frame, series: &Series) {
    match series {
        Series::Line {
            x,
            y,
            color,
            width,
            style,
            marker,
            alpha,
            ..
        } => {
            let mut pts = String::new();
            for i in 0..x.len().min(y.len()) {
                let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i]), f.sy(y[i]));
            }
            let dash = style
                .dasharray()
                .map(|d| format!(" stroke-dasharray=\"{d}\""))
                .unwrap_or_default();
            let _ = write!(
                s,
                "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"{width}\" stroke-opacity=\"{alpha:.3}\" points=\"{}\"{dash}/>",
                color.hex(),
                pts.trim_end()
            );
            if *marker != Marker::None {
                for i in 0..x.len().min(y.len()) {
                    marker_svg(
                        s,
                        *marker,
                        f.sx(x[i]),
                        f.sy(y[i]),
                        3.0,
                        &color.hex(),
                        *alpha,
                    );
                }
            }
        }
        Series::Scatter {
            x,
            y,
            color,
            radius,
            marker,
            alpha,
            ..
        } => {
            for i in 0..x.len().min(y.len()) {
                marker_svg(
                    s,
                    *marker,
                    f.sx(x[i]),
                    f.sy(y[i]),
                    *radius,
                    &color.hex(),
                    *alpha,
                );
            }
        }
        Series::Bar {
            x,
            y,
            width,
            color,
            base,
            horizontal,
            alpha,
            ..
        } => {
            for i in 0..x.len().min(y.len()) {
                let b = base.get(i).copied().unwrap_or(0.0);
                if *horizontal {
                    let lo = f.sy(x[i] - width / 2.0);
                    let hi = f.sy(x[i] + width / 2.0);
                    let x0 = f.sx(b);
                    let x1 = f.sx(b + y[i]);
                    let _ = write!(
                        s,
                        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\" fill-opacity=\"{alpha:.3}\"/>",
                        x0.min(x1),
                        lo.min(hi),
                        (x1 - x0).abs().max(0.5),
                        (hi - lo).abs().max(0.5),
                        color.hex()
                    );
                } else {
                    let left = f.sx(x[i] - width / 2.0);
                    let right = f.sx(x[i] + width / 2.0);
                    let y0 = f.sy(b);
                    let y1 = f.sy(b + y[i]);
                    let _ = write!(
                        s,
                        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\" fill-opacity=\"{alpha:.3}\"/>",
                        left.min(right),
                        y0.min(y1),
                        (right - left).abs().max(0.5),
                        (y0 - y1).abs().max(0.5),
                        color.hex()
                    );
                }
            }
        }
        Series::ErrorBar {
            x,
            y,
            yerr,
            color,
            capsize,
            ..
        } => {
            for i in 0..x.len().min(y.len()) {
                let e = yerr.get(i).copied().unwrap_or(0.0);
                let px = f.sx(x[i]);
                let yt = f.sy(y[i] + e);
                let yb = f.sy(y[i] - e);
                let _ = write!(
                    s,
                    "<line x1=\"{px:.2}\" y1=\"{yt:.2}\" x2=\"{px:.2}\" y2=\"{yb:.2}\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                    color.hex()
                );
                // Caps.
                let _ = write!(
                    s,
                    "<line x1=\"{:.2}\" y1=\"{yt:.2}\" x2=\"{:.2}\" y2=\"{yt:.2}\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                    px - capsize,
                    px + capsize,
                    color.hex()
                );
                let _ = write!(
                    s,
                    "<line x1=\"{:.2}\" y1=\"{yb:.2}\" x2=\"{:.2}\" y2=\"{yb:.2}\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                    px - capsize,
                    px + capsize,
                    color.hex()
                );
                // Point marker.
                marker_svg(s, Marker::Circle, px, f.sy(y[i]), 2.5, &color.hex(), 1.0);
            }
        }
        Series::Step {
            x,
            y,
            color,
            width,
            style,
            wher,
            ..
        } => {
            let mut pts = String::new();
            let n = x.len().min(y.len());
            for i in 0..n {
                match wher {
                    StepWhere::Pre => {
                        if i == 0 {
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i]), f.sy(y[i]));
                        } else {
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i]), f.sy(y[i - 1]));
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i]), f.sy(y[i]));
                        }
                    }
                    StepWhere::Post => {
                        let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i]), f.sy(y[i]));
                        if i + 1 < n {
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i + 1]), f.sy(y[i]));
                        }
                    }
                    StepWhere::Mid => {
                        if i == 0 {
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i]), f.sy(y[i]));
                        } else {
                            let mx = (x[i - 1] + x[i]) / 2.0;
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(mx), f.sy(y[i - 1]));
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(mx), f.sy(y[i]));
                            let _ = write!(pts, "{:.2},{:.2} ", f.sx(x[i]), f.sy(y[i]));
                        }
                    }
                }
            }
            let dash = style
                .dasharray()
                .map(|d| format!(" stroke-dasharray=\"{d}\""))
                .unwrap_or_default();
            let _ = write!(
                s,
                "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"{width}\" points=\"{}\"{dash}/>",
                color.hex(),
                pts.trim_end()
            );
        }
        Series::FillBetween {
            x,
            y1,
            y2,
            color,
            alpha,
            ..
        } => {
            let n = x.len().min(y1.len()).min(y2.len());
            if n < 2 {
                return;
            }
            let mut path = String::from("M");
            for i in 0..n {
                let _ = write!(path, " {:.2},{:.2}", f.sx(x[i]), f.sy(y1[i]));
            }
            for i in (0..n).rev() {
                let _ = write!(path, " {:.2},{:.2}", f.sx(x[i]), f.sy(y2[i]));
            }
            path.push_str(" Z");
            let _ = write!(
                s,
                "<path d=\"{path}\" fill=\"{}\" fill-opacity=\"{alpha:.3}\" stroke=\"none\"/>",
                color.hex()
            );
        }
        Series::Box {
            boxes,
            color,
            width,
            ..
        } => {
            for b in boxes {
                let cx = f.sx(b.pos);
                let left = f.sx(b.pos - width / 2.0);
                let right = f.sx(b.pos + width / 2.0);
                let yq1 = f.sy(b.q1);
                let yq3 = f.sy(b.q3);
                let ymed = f.sy(b.median);
                let ywlo = f.sy(b.whislo);
                let ywhi = f.sy(b.whishi);
                // Box rect.
                let _ = write!(
                    s,
                    "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\" fill-opacity=\"0.35\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                    left.min(right),
                    yq3.min(yq1),
                    (right - left).abs(),
                    (yq1 - yq3).abs(),
                    color.hex(),
                    color.hex()
                );
                // Median line.
                let _ = write!(
                    s,
                    "<line x1=\"{:.2}\" y1=\"{ymed:.2}\" x2=\"{:.2}\" y2=\"{ymed:.2}\" stroke=\"{}\" stroke-width=\"2\" class=\"median\"/>",
                    left, right, color.hex()
                );
                // Whiskers + caps.
                let capw = (right - left).abs() * 0.4;
                let _ = write!(
                    s,
                    "<line x1=\"{cx:.2}\" y1=\"{yq3:.2}\" x2=\"{cx:.2}\" y2=\"{ywhi:.2}\" stroke=\"{}\" stroke-width=\"1\" class=\"whisker\"/>",
                    color.hex()
                );
                let _ = write!(
                    s,
                    "<line x1=\"{cx:.2}\" y1=\"{yq1:.2}\" x2=\"{cx:.2}\" y2=\"{ywlo:.2}\" stroke=\"{}\" stroke-width=\"1\" class=\"whisker\"/>",
                    color.hex()
                );
                let _ = write!(
                    s,
                    "<line x1=\"{:.2}\" y1=\"{ywhi:.2}\" x2=\"{:.2}\" y2=\"{ywhi:.2}\" stroke=\"{}\" stroke-width=\"1\"/>",
                    cx - capw, cx + capw, color.hex()
                );
                let _ = write!(
                    s,
                    "<line x1=\"{:.2}\" y1=\"{ywlo:.2}\" x2=\"{:.2}\" y2=\"{ywlo:.2}\" stroke=\"{}\" stroke-width=\"1\"/>",
                    cx - capw, cx + capw, color.hex()
                );
                // Outliers.
                for &o in &b.outliers {
                    marker_svg(s, Marker::Circle, cx, f.sy(o), 2.0, &color.hex(), 0.8);
                }
            }
        }
        Series::Violin {
            violins,
            color,
            width,
            alpha,
            ..
        } => {
            for v in violins {
                let half = width / 2.0;
                let mut path = String::from("M");
                // Right side, top to bottom.
                for i in 0..v.ys.len() {
                    let px = f.sx(v.pos + v.density[i] * half);
                    let py = f.sy(v.ys[i]);
                    let _ = write!(path, " {px:.2},{py:.2}");
                }
                // Left side, bottom to top.
                for i in (0..v.ys.len()).rev() {
                    let px = f.sx(v.pos - v.density[i] * half);
                    let py = f.sy(v.ys[i]);
                    let _ = write!(path, " {px:.2},{py:.2}");
                }
                path.push_str(" Z");
                let _ = write!(
                    s,
                    "<path d=\"{path}\" fill=\"{}\" fill-opacity=\"{alpha:.3}\" stroke=\"{}\" stroke-width=\"1\"/>",
                    color.hex(),
                    color.hex()
                );
                // Median tick.
                let ym = f.sy(v.median);
                let _ = write!(
                    s,
                    "<line x1=\"{:.2}\" y1=\"{ym:.2}\" x2=\"{:.2}\" y2=\"{ym:.2}\" stroke=\"{}\" stroke-width=\"2\"/>",
                    f.sx(v.pos - half * 0.3),
                    f.sx(v.pos + half * 0.3),
                    Color::BLACK.hex()
                );
            }
        }
        Series::Heatmap {
            data,
            cmap,
            vmin,
            vmax,
            extent,
            ..
        } => {
            let nrows = data.len();
            if nrows == 0 {
                return;
            }
            let ncols = data[0].len();
            if ncols == 0 {
                return;
            }
            let (x0, x1, y0, y1) = *extent;
            for (r, row) in data.iter().enumerate() {
                for (c, &val) in row.iter().enumerate() {
                    let cx0 = x0 + (x1 - x0) * c as f64 / ncols as f64;
                    let cx1 = x0 + (x1 - x0) * (c as f64 + 1.0) / ncols as f64;
                    // Row 0 is drawn at the top of the extent.
                    let cy0 = y1 - (y1 - y0) * r as f64 / nrows as f64;
                    let cy1 = y1 - (y1 - y0) * (r as f64 + 1.0) / nrows as f64;
                    let px0 = f.sx(cx0);
                    let px1 = f.sx(cx1);
                    let py0 = f.sy(cy0);
                    let py1 = f.sy(cy1);
                    let t = (val - vmin) / (vmax - vmin);
                    let fill = cmap.sample(t).hex();
                    let _ = write!(
                        s,
                        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{fill}\" shape-rendering=\"crispEdges\"/>",
                        px0.min(px1),
                        py0.min(py1),
                        (px1 - px0).abs() + 0.5,
                        (py1 - py0).abs() + 0.5
                    );
                }
            }
        }
    }
}

fn render_colorbar(
    s: &mut String,
    f: &Frame,
    cmap: crate::color::Colormap,
    vmin: f64,
    vmax: f64,
    mr: f64,
    font: f64,
) {
    let bar_w = 14.0;
    let bar_x = f.inner.x + f.inner.w + (mr - 50.0).max(10.0);
    let bar_y = f.inner.y;
    let bar_h = f.inner.h;
    let steps = 40;
    let _ = write!(s, "<g class=\"colorbar\">");
    for i in 0..steps {
        let t0 = i as f64 / steps as f64;
        let t1 = (i as f64 + 1.0) / steps as f64;
        // Top of bar = vmax.
        let yy = bar_y + (1.0 - t1) * bar_h;
        let hh = (t1 - t0) * bar_h + 0.5;
        let fill = cmap.sample((t0 + t1) / 2.0).hex();
        let _ = write!(
            s,
            "<rect x=\"{bar_x:.2}\" y=\"{yy:.2}\" width=\"{bar_w:.2}\" height=\"{hh:.2}\" fill=\"{fill}\"/>"
        );
    }
    let _ = write!(
        s,
        "<rect x=\"{bar_x:.2}\" y=\"{bar_y:.2}\" width=\"{bar_w:.2}\" height=\"{bar_h:.2}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1\"/>",
        Color::BLACK.hex()
    );
    let ticks = crate::scale::nice_ticks(vmin, vmax, 5);
    for &t in &ticks {
        if t < vmin || t > vmax {
            continue;
        }
        let frac = (t - vmin) / (vmax - vmin);
        let yy = bar_y + (1.0 - frac) * bar_h;
        let _ = write!(
            s,
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{font}\" text-anchor=\"start\">{}</text>",
            bar_x + bar_w + 3.0,
            yy + 3.0,
            escape(&crate::scale::fmt_tick(t))
        );
    }
    let _ = write!(s, "</g>");
}

fn render_legend(s: &mut String, ax: &Axes, f: &Frame, loc: LegendLoc) {
    let entries: Vec<(Color, &str)> = ax
        .series
        .iter()
        .filter_map(|sr| sr.label().map(|l| (sr.legend_color(), l)))
        .collect();
    if entries.is_empty() {
        return;
    }
    let row_h = 16.0;
    let pad = 6.0;
    let swatch = 14.0;
    let char_w = 6.5;
    let max_label = entries
        .iter()
        .map(|(_, l)| l.chars().count())
        .max()
        .unwrap_or(0) as f64;
    let box_w = pad * 2.0 + swatch + 6.0 + max_label * char_w;
    let box_h = pad * 2.0 + entries.len() as f64 * row_h;
    let (bx, by) = match loc {
        LegendLoc::UpperRight => (f.inner.x + f.inner.w - box_w - 8.0, f.inner.y + 8.0),
        LegendLoc::UpperLeft => (f.inner.x + 8.0, f.inner.y + 8.0),
        LegendLoc::LowerRight => (
            f.inner.x + f.inner.w - box_w - 8.0,
            f.inner.y + f.inner.h - box_h - 8.0,
        ),
        LegendLoc::LowerLeft => (f.inner.x + 8.0, f.inner.y + f.inner.h - box_h - 8.0),
    };
    let _ = write!(s, "<g class=\"legend\">");
    let _ = write!(
        s,
        "<rect x=\"{bx:.2}\" y=\"{by:.2}\" width=\"{box_w:.2}\" height=\"{box_h:.2}\" fill=\"white\" fill-opacity=\"0.85\" stroke=\"{}\" stroke-width=\"1\"/>",
        Color::GRAY.hex()
    );
    for (i, (color, label)) in entries.iter().enumerate() {
        let ry = by + pad + i as f64 * row_h;
        let _ = write!(
            s,
            "<rect class=\"swatch\" x=\"{:.2}\" y=\"{:.2}\" width=\"{swatch:.2}\" height=\"10\" fill=\"{}\"/>",
            bx + pad,
            ry + 2.0,
            color.hex()
        );
        let _ = write!(
            s,
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" text-anchor=\"start\">{}</text>",
            bx + pad + swatch + 6.0,
            ry + 11.0,
            escape(label)
        );
    }
    let _ = write!(s, "</g>");
}
