//! Colors, the categorical color cycle, and sequential colormaps.

/// An RGB color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(pub u8, pub u8, pub u8);

impl Color {
    pub const BLACK: Color = Color(0, 0, 0);
    pub const WHITE: Color = Color(255, 255, 255);
    pub const BLUE: Color = Color(31, 119, 180);
    pub const ORANGE: Color = Color(255, 127, 14);
    pub const GREEN: Color = Color(44, 160, 44);
    pub const RED: Color = Color(214, 39, 40);
    pub const PURPLE: Color = Color(148, 103, 189);
    pub const BROWN: Color = Color(140, 86, 75);
    pub const PINK: Color = Color(227, 119, 194);
    pub const GRAY: Color = Color(127, 127, 127);
    pub const OLIVE: Color = Color(188, 189, 34);
    pub const CYAN: Color = Color(23, 190, 207);
    pub const LIGHT_GRAY: Color = Color(221, 221, 221);

    /// The CSS `#rrggbb` form.
    pub fn hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }

    /// Construct from a 0..=1 RGB triple, clamping out-of-range channels.
    pub fn from_f64(r: f64, g: f64, b: f64) -> Color {
        let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        Color(q(r), q(g), q(b))
    }

    /// Linearly interpolate between two colors (`t` in `[0, 1]`).
    pub fn lerp(a: Color, b: Color, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        let mix = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
        Color(mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
    }

    /// The default categorical color-cycle entry for index `i` (matplotlib's
    /// `tab10`, wrapping after ten entries).
    pub fn cycle(i: usize) -> Color {
        const CYCLE: [Color; 10] = [
            Color::BLUE,
            Color::ORANGE,
            Color::GREEN,
            Color::RED,
            Color::PURPLE,
            Color::BROWN,
            Color::PINK,
            Color::GRAY,
            Color::OLIVE,
            Color::CYAN,
        ];
        CYCLE[i % CYCLE.len()]
    }
}

/// A sequential colormap mapping `t` in `[0, 1]` to a [`Color`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Colormap {
    /// The perceptually-uniform `viridis` map (dark blue -> green -> yellow).
    #[default]
    Viridis,
    /// `magma`: black -> purple -> orange -> pale yellow.
    Magma,
    /// `plasma`: dark blue -> magenta -> yellow.
    Plasma,
    /// A simple grayscale ramp (black -> white).
    Gray,
    /// A diverging blue -> white -> red map (useful for correlation matrices).
    Coolwarm,
}

impl Colormap {
    /// Sample the colormap at position `t` (clamped to `[0, 1]`).
    pub fn sample(&self, t: f64) -> Color {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Colormap::Gray => Color::lerp(Color::BLACK, Color::WHITE, t),
            Colormap::Coolwarm => {
                let cool = Color(59, 76, 192);
                let mid = Color(221, 221, 221);
                let warm = Color(180, 4, 38);
                if t < 0.5 {
                    Color::lerp(cool, mid, t * 2.0)
                } else {
                    Color::lerp(mid, warm, (t - 0.5) * 2.0)
                }
            }
            Colormap::Viridis => sample_table(&VIRIDIS, t),
            Colormap::Magma => sample_table(&MAGMA, t),
            Colormap::Plasma => sample_table(&PLASMA, t),
        }
    }
}

/// Linearly interpolate within a fixed RGB control-point table.
fn sample_table(table: &[(u8, u8, u8)], t: f64) -> Color {
    let n = table.len();
    if n == 0 {
        return Color::BLACK;
    }
    if n == 1 {
        let (r, g, b) = table[0];
        return Color(r, g, b);
    }
    let pos = t * (n - 1) as f64;
    let i = (pos.floor() as usize).min(n - 2);
    let frac = pos - i as f64;
    let (r0, g0, b0) = table[i];
    let (r1, g1, b1) = table[i + 1];
    Color::lerp(Color(r0, g0, b0), Color(r1, g1, b1), frac)
}

// 16-point control tables sampled from the matplotlib reference maps.
const VIRIDIS: [(u8, u8, u8); 16] = [
    (68, 1, 84),
    (72, 26, 108),
    (71, 47, 125),
    (65, 68, 135),
    (57, 86, 140),
    (49, 104, 142),
    (42, 120, 142),
    (35, 136, 142),
    (31, 152, 139),
    (34, 168, 132),
    (53, 183, 121),
    (84, 197, 104),
    (122, 209, 81),
    (165, 219, 54),
    (210, 226, 27),
    (253, 231, 37),
];

const MAGMA: [(u8, u8, u8); 16] = [
    (0, 0, 4),
    (12, 8, 38),
    (34, 12, 75),
    (60, 9, 102),
    (87, 16, 110),
    (112, 25, 110),
    (137, 34, 106),
    (163, 43, 97),
    (189, 55, 84),
    (212, 72, 66),
    (229, 97, 50),
    (242, 128, 38),
    (249, 161, 41),
    (252, 196, 65),
    (250, 228, 114),
    (252, 253, 191),
];

const PLASMA: [(u8, u8, u8); 16] = [
    (13, 8, 135),
    (49, 5, 151),
    (76, 2, 161),
    (102, 0, 167),
    (126, 3, 168),
    (148, 19, 161),
    (168, 34, 150),
    (186, 50, 137),
    (202, 67, 124),
    (216, 84, 111),
    (228, 102, 98),
    (238, 121, 85),
    (246, 142, 71),
    (251, 165, 56),
    (253, 191, 42),
    (240, 249, 33),
];
