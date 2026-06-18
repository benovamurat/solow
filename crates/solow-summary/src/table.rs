//! A general fixed-width text table used to render summary blocks.
//!
//! A [`SummaryTable`] is a titled, optionally-headered, row-based table whose
//! columns are sized to the widest cell and rendered right-aligned (with an
//! option to left-align individual columns, used for the leading label
//! column of a coefficient table). The visual layout is Solow's own.

use std::fmt;

/// Horizontal alignment of a column's cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

/// A titled, fixed-width text table.
#[derive(Clone, Debug, Default)]
pub struct SummaryTable {
    title: Option<String>,
    header: Option<Vec<String>>,
    rows: Vec<Vec<String>>,
    aligns: Vec<Align>,
    /// Minimum spaces between adjacent columns.
    gap: usize,
}

impl SummaryTable {
    /// Create an empty table with a two-space inter-column gap.
    pub fn new() -> Self {
        SummaryTable {
            title: None,
            header: None,
            rows: Vec::new(),
            aligns: Vec::new(),
            gap: 2,
        }
    }

    /// Set the (centered) title rendered above the table.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the inter-column gap (minimum spaces between columns).
    pub fn gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Provide a header row (column labels).
    pub fn header<I, S>(mut self, cols: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.header = Some(cols.into_iter().map(Into::into).collect());
        self
    }

    /// Set per-column alignment. Columns without an explicit alignment default
    /// to right-aligned.
    pub fn aligns(mut self, aligns: impl IntoIterator<Item = Align>) -> Self {
        self.aligns = aligns.into_iter().collect();
        self
    }

    /// Append a row of pre-formatted cell strings.
    pub fn push_row<I, S>(&mut self, cells: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.rows.push(cells.into_iter().map(Into::into).collect());
    }

    /// Number of columns, derived from the header and rows.
    fn n_cols(&self) -> usize {
        let mut n = self.header.as_ref().map_or(0, Vec::len);
        for r in &self.rows {
            n = n.max(r.len());
        }
        n
    }

    /// Alignment for a given column index (defaults to right).
    fn align_for(&self, col: usize) -> Align {
        self.aligns.get(col).copied().unwrap_or(Align::Right)
    }

    /// Compute the rendered width of each column.
    fn col_widths(&self) -> Vec<usize> {
        let n = self.n_cols();
        let mut widths = vec![0usize; n];
        if let Some(h) = &self.header {
            for (i, cell) in h.iter().enumerate() {
                widths[i] = widths[i].max(cell.chars().count());
            }
        }
        for r in &self.rows {
            for (i, cell) in r.iter().enumerate() {
                widths[i] = widths[i].max(cell.chars().count());
            }
        }
        widths
    }

    /// Render a single line of cells given column widths.
    fn render_line(&self, cells: &[String], widths: &[usize]) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(widths.len());
        for (i, w) in widths.iter().enumerate() {
            let cell = cells.get(i).map(String::as_str).unwrap_or("");
            let len = cell.chars().count();
            let pad = w.saturating_sub(len);
            let padded = match self.align_for(i) {
                Align::Left => format!("{cell}{}", " ".repeat(pad)),
                Align::Right => format!("{}{cell}", " ".repeat(pad)),
            };
            parts.push(padded);
        }
        // Join with the inter-column gap, then strip trailing whitespace.
        parts.join(&" ".repeat(self.gap)).trim_end().to_string()
    }

    /// Total rendered width of the table body (for centering the title and
    /// drawing rules).
    fn body_width(&self, widths: &[usize]) -> usize {
        if widths.is_empty() {
            return 0;
        }
        let sum: usize = widths.iter().sum();
        sum + self.gap * (widths.len() - 1)
    }
}

impl fmt::Display for SummaryTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths = self.col_widths();
        let width = self.body_width(&widths);
        let mut lines: Vec<String> = Vec::new();

        if let Some(t) = &self.title {
            let tlen = t.chars().count();
            if tlen < width {
                let pad = (width - tlen) / 2;
                lines.push(format!("{}{t}", " ".repeat(pad)));
            } else {
                lines.push(t.clone());
            }
        }

        if let Some(h) = &self.header {
            lines.push("=".repeat(width));
            lines.push(self.render_line(h, &widths));
            lines.push("-".repeat(width));
        }

        for r in &self.rows {
            lines.push(self.render_line(r, &widths));
        }

        if self.header.is_some() {
            lines.push("=".repeat(width));
        }

        write!(f, "{}", lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_aligned_columns_pad_correctly() {
        let mut t = SummaryTable::new().header(["a", "bbb"]);
        t.push_row(["1", "22"]);
        t.push_row(["333", "4"]);
        let s = t.to_string();
        let lines: Vec<&str> = s.lines().collect();
        // header line: columns widths are 3 ("333") and 3 ("bbb"), gap 2.
        // "  a" + "  " + "bbb"
        assert_eq!(lines[1], "  a  bbb");
        // row "1","22" -> "  1" + "  " + " 22"
        assert_eq!(lines[3], "  1   22");
        // row "333","4" -> "333" + "  " + "  4"
        assert_eq!(lines[4], "333    4");
    }

    #[test]
    fn left_align_first_column() {
        let mut t = SummaryTable::new()
            .header(["name", "v"])
            .aligns([Align::Left, Align::Right]);
        t.push_row(["const", "1.5"]);
        t.push_row(["x1", "10"]);
        let s = t.to_string();
        let lines: Vec<&str> = s.lines().collect();
        // first col width = 5 ("const"), second col width = 3 ("1.5")
        assert_eq!(lines[1], "name     v");
        assert_eq!(lines[3], "const  1.5");
        assert_eq!(lines[4], "x1      10");
    }

    #[test]
    fn rules_match_body_width() {
        let mut t = SummaryTable::new().header(["aa", "bb"]);
        t.push_row(["1", "2"]);
        let s = t.to_string();
        let lines: Vec<&str> = s.lines().collect();
        let width = lines[0].chars().count(); // "==..="
        assert!(lines[0].chars().all(|c| c == '='));
        assert!(lines[2].chars().all(|c| c == '-'));
        assert_eq!(lines[0].len(), lines[2].len());
        // body width = 2 + 2 (gap) + 2 = 6
        assert_eq!(width, 6);
    }

    #[test]
    fn title_is_centered() {
        let mut t = SummaryTable::new().title("Hi").header(["aaaa", "bbbb"]);
        t.push_row(["1", "2"]);
        let s = t.to_string();
        let first = s.lines().next().unwrap();
        // body width = 4 + 2 + 4 = 10, title len 2, pad = 4
        assert_eq!(first, "    Hi");
    }

    #[test]
    fn no_trailing_whitespace_on_rows() {
        let mut t = SummaryTable::new().header(["a", "b"]);
        t.push_row(["x", "y"]);
        let s = t.to_string();
        for line in s.lines() {
            assert_eq!(line, line.trim_end(), "line had trailing space: {line:?}");
        }
    }
}
