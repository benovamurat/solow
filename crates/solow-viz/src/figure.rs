//! The [`Figure`]: a canvas holding one or more [`Axes`] in a grid.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

use crate::axes::Axes;
use crate::color::Color;
use crate::render::{escape, render_axes, Rect};

/// A figure: a pixel canvas holding a grid of [`Axes`].
///
/// A freshly-constructed figure has a single axes (the 1x1 grid), so the
/// original `fig.axes()` API keeps working. Use [`Figure::subplots`] for a
/// multi-panel grid.
#[derive(Clone, Debug)]
pub struct Figure {
    width: u32,
    height: u32,
    rows: usize,
    cols: usize,
    axes: Vec<Axes>,
    background: Option<Color>,
    suptitle: Option<String>,
    hspace: f64,
    wspace: f64,
}

impl Figure {
    /// Create a figure `width x height` pixels with a single axes.
    pub fn new(width: u32, height: u32) -> Self {
        Figure {
            width,
            height,
            rows: 1,
            cols: 1,
            axes: vec![Axes::default()],
            background: Some(Color::WHITE),
            suptitle: None,
            hspace: 0.0,
            wspace: 0.0,
        }
    }

    /// Create a `rows x cols` grid of axes on a `width x height` canvas.
    pub fn subplots(width: u32, height: u32, rows: usize, cols: usize) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        Figure {
            width,
            height,
            rows,
            cols,
            axes: vec![Axes::default(); rows * cols],
            background: Some(Color::WHITE),
            suptitle: None,
            hspace: 0.0,
            wspace: 0.0,
        }
    }

    /// The grid shape `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Total number of axes cells.
    pub fn naxes(&self) -> usize {
        self.axes.len()
    }

    /// Mutable access to the first (or only) axes — preserves the legacy API.
    pub fn axes(&mut self) -> &mut Axes {
        &mut self.axes[0]
    }

    /// Mutable access to the axes at flat index `i` (row-major), if present.
    pub fn ax(&mut self, i: usize) -> Option<&mut Axes> {
        self.axes.get_mut(i)
    }

    /// Mutable access to the axes at grid cell `(row, col)`.
    pub fn ax_at(&mut self, row: usize, col: usize) -> Option<&mut Axes> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.axes.get_mut(row * self.cols + col)
    }

    /// Set (or clear, with `None`) the figure background.
    pub fn set_background(&mut self, c: Option<Color>) -> &mut Self {
        self.background = c;
        self
    }

    /// Set a figure-level title centered above the grid.
    pub fn suptitle(&mut self, s: &str) -> &mut Self {
        self.suptitle = Some(s.to_string());
        self
    }

    /// Render the figure to an SVG document string.
    pub fn to_svg(&self) -> String {
        let w = self.width as f64;
        let h = self.height as f64;
        let top_pad = if self.suptitle.is_some() { 30.0 } else { 6.0 };
        let outer = 6.0;

        let mut s = String::new();
        let _ = write!(
            s,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" font-family=\"sans-serif\">",
            self.width, self.height, self.width, self.height
        );
        if let Some(bg) = self.background {
            let _ = write!(
                s,
                "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
                self.width,
                self.height,
                bg.hex()
            );
        }

        if let Some(ref t) = self.suptitle {
            let _ = write!(
                s,
                "<text x=\"{:.2}\" y=\"20\" font-size=\"18\" font-weight=\"bold\" text-anchor=\"middle\">{}</text>",
                w / 2.0,
                escape(t)
            );
        }

        let grid_x = outer;
        let grid_y = top_pad;
        let grid_w = w - 2.0 * outer;
        let grid_h = h - top_pad - outer;
        let cell_w = grid_w / self.cols as f64;
        let cell_h = grid_h / self.rows as f64;

        for (i, ax) in self.axes.iter().enumerate() {
            let r = i / self.cols;
            let c = i % self.cols;
            let rect = Rect {
                x: grid_x + c as f64 * cell_w + self.wspace / 2.0,
                y: grid_y + r as f64 * cell_h + self.hspace / 2.0,
                w: cell_w - self.wspace,
                h: cell_h - self.hspace,
            };
            render_axes(&mut s, ax, rect);
        }

        s.push_str("</svg>");
        s
    }

    /// Write the figure to `path` as an SVG file.
    pub fn save_svg<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        fs::write(path, self.to_svg())
    }

    /// Rasterize the SVG to a PNG byte buffer at the given `scale`.
    ///
    /// Available only with the `png` feature.
    #[cfg(feature = "png")]
    pub fn to_png(&self, scale: f32) -> io::Result<Vec<u8>> {
        crate::png::svg_to_png(&self.to_svg(), scale)
    }

    /// Rasterize the SVG and write a PNG to `path` at the given `scale`
    /// (e.g. `2.0` for high-DPI). Available only with the `png` feature.
    #[cfg(feature = "png")]
    pub fn save_png<P: AsRef<Path>>(&self, path: P, scale: f32) -> io::Result<()> {
        let bytes = self.to_png(scale)?;
        fs::write(path, bytes)
    }
}
