//! Optional PNG raster export (behind the `png` feature).
//!
//! Rasterizes the figure's SVG via `resvg` (which bundles `usvg` for SVG
//! parsing and `tiny-skia` for the CPU raster backend). No system fonts are
//! pulled in, keeping the dependency footprint minimal.

use std::io;

use resvg::tiny_skia;
use resvg::usvg;

/// Rasterize `svg` to PNG bytes at `scale` (1.0 = native pixel size).
pub fn svg_to_png(svg: &str, scale: f32) -> io::Result<Vec<u8>> {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_str(svg, &opt)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("svg parse: {e}")))?;
    let size = tree.size();
    let w = (size.width() * scale).ceil().max(1.0) as u32;
    let h = (size.height() * scale).ceil().max(1.0) as u32;
    let mut pixmap =
        tiny_skia::Pixmap::new(w, h).ok_or_else(|| io::Error::other("pixmap allocation failed"))?;
    let transform = tiny_skia::Transform::from_scale(scale, scale);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    pixmap
        .encode_png()
        .map_err(|e| io::Error::other(format!("png encode: {e}")))
}
