//! Render a rich gallery to inspect solow-viz output quality. This exercises
//! every plot type, scale, and styling feature the crate offers.
//!
//! Run: `cargo run -p solow-viz --example gallery`
//! With PNG raster output: `cargo run -p solow-viz --example gallery --features png`

use solow_viz::{
    Color, Colormap, Figure, LegendLoc, LineStyle, Marker, Scale, StepWhere, TickDirection,
};

fn main() {
    // ----------------------------------------------------------------------
    // 1) Classic multi-series line plot with a legend and markers.
    // ----------------------------------------------------------------------
    let xs: Vec<f64> = (0..120).map(|i| i as f64 * 0.05).collect();
    let s: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
    let c: Vec<f64> = xs.iter().map(|x| x.cos()).collect();
    let mut f1 = Figure::new(720, 440);
    {
        let ax = f1.axes();
        ax.set_title("sin and cos")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.line(
            &xs,
            &s,
            Color::BLUE,
            1.8,
            LineStyle::Solid,
            Marker::None,
            1.0,
            Some("sin"),
        );
        ax.line(
            &xs,
            &c,
            Color::RED,
            1.8,
            LineStyle::Dashed,
            Marker::None,
            1.0,
            Some("cos"),
        );
        ax.axhline(0.0, Color::GRAY, LineStyle::Dotted);
        ax.annotate(std::f64::consts::FRAC_PI_2, 1.0, "max");
        ax.legend(LegendLoc::UpperRight);
    }
    f1.save_svg("/tmp/viz_lines.svg").unwrap();

    // ----------------------------------------------------------------------
    // 2) Scatter + fitted line + error band (fill-between).
    // ----------------------------------------------------------------------
    let sx: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
    let sy: Vec<f64> = sx
        .iter()
        .enumerate()
        .map(|(i, &x)| 2.0 + 1.5 * x + 0.7 * ((i as f64 * 1.3).sin()))
        .collect();
    let fit_lo: Vec<f64> = sx.iter().map(|&x| 2.0 + 1.5 * x - 0.9).collect();
    let fit_hi: Vec<f64> = sx.iter().map(|&x| 2.0 + 1.5 * x + 0.9).collect();
    let mut f2 = Figure::new(720, 440);
    {
        let ax = f2.axes();
        ax.set_title("scatter + fit + band")
            .set_xlabel("x")
            .set_ylabel("y")
            .set_grid(true);
        ax.fill_between(&sx, &fit_lo, &fit_hi, Color::BLUE, 0.18, Some("95% band"));
        ax.scatter_full(
            &sx,
            &sy,
            Color::BLUE,
            3.0,
            Marker::Circle,
            0.9,
            Some("data"),
        );
        ax.plot_styled(
            &[sx[0], sx[39]],
            &[2.0 + 1.5 * sx[0], 2.0 + 1.5 * sx[39]],
            Color::RED,
            2.0,
        );
        ax.legend(LegendLoc::UpperLeft);
    }
    f2.save_svg("/tmp/viz_scatter.svg").unwrap();

    // ----------------------------------------------------------------------
    // 3) Histogram.
    // ----------------------------------------------------------------------
    let data: Vec<f64> = (0..400)
        .map(|i| {
            let u = (i as f64 + 0.5) / 400.0;
            (u - 0.5) * 6.0 + 0.8 * ((i as f64 * 0.7).sin())
        })
        .collect();
    let mut f3 = Figure::new(720, 440);
    {
        let ax = f3.axes();
        ax.set_title("histogram")
            .set_xlabel("value")
            .set_ylabel("count")
            .set_grid(true);
        ax.hist(&data, 24);
    }
    f3.save_svg("/tmp/viz_hist.svg").unwrap();

    // ----------------------------------------------------------------------
    // 4) A 2x3 subplot grid showcasing the new plot types.
    // ----------------------------------------------------------------------
    let mut grid = Figure::subplots(1100, 720, 2, 3);
    grid.suptitle("solow-viz gallery");

    // (0,0) error bars.
    {
        let ax = grid.ax_at(0, 0).unwrap();
        ax.set_title("error bars");
        let x: Vec<f64> = (1..=6).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.8).ln() + 1.0).collect();
        let e: Vec<f64> = x.iter().map(|&v| 0.1 + 0.03 * v).collect();
        ax.errorbar(&x, &y, &e, Color::PURPLE, Some("mean +/- se"));
        ax.legend(LegendLoc::LowerRight);
    }

    // (0,1) step + area.
    {
        let ax = grid.ax_at(0, 1).unwrap();
        ax.set_title("step + area");
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| ((v * 0.5).sin() * 3.0).round() + 4.0)
            .collect();
        ax.area(&x, &y, Color::GREEN, 0.2);
        ax.step(&x, &y, Color::GREEN, StepWhere::Mid, Some("step"));
        ax.legend(LegendLoc::UpperLeft);
    }

    // (0,2) stacked + grouped bars (two stacks shown as grouped here).
    {
        let ax = grid.ax_at(0, 2).unwrap();
        ax.set_title("grouped bars");
        let x: Vec<f64> = (1..=4).map(|i| i as f64).collect();
        ax.bar_grouped(
            &x,
            &[vec![3.0, 5.0, 2.0, 4.0], vec![2.0, 3.0, 4.0, 1.0]],
            0.8,
            &["2023", "2024"],
        );
        ax.legend(LegendLoc::UpperRight);
    }

    // (1,0) boxplot.
    {
        let ax = grid.ax_at(1, 0).unwrap();
        ax.set_title("boxplot");
        let groups: Vec<Vec<f64>> = (0..3)
            .map(|g| {
                (0..60)
                    .map(|i| {
                        ((i as f64 * (0.2 + g as f64 * 0.05)).sin()) * (2.0 + g as f64)
                            + g as f64 * 2.0
                    })
                    .collect()
            })
            .collect();
        ax.boxplot(&groups, &[1.0, 2.0, 3.0]);
    }

    // (1,1) violin.
    {
        let ax = grid.ax_at(1, 1).unwrap();
        ax.set_title("violin");
        let groups: Vec<Vec<f64>> = (0..3)
            .map(|g| {
                (0..120)
                    .map(|i| {
                        ((i as f64 * 0.27).sin() + (i as f64 * 0.13).cos()) * (1.5 + g as f64 * 0.5)
                    })
                    .collect()
            })
            .collect();
        ax.violinplot(&groups, &[1.0, 2.0, 3.0]);
    }

    // (1,2) heatmap with colorbar.
    {
        let ax = grid.ax_at(1, 2).unwrap();
        ax.set_title("heatmap");
        let nr = 12;
        let nc = 16;
        let field: Vec<Vec<f64>> = (0..nr)
            .map(|r| {
                (0..nc)
                    .map(|c| {
                        let x = c as f64 / nc as f64 * 6.0;
                        let y = r as f64 / nr as f64 * 6.0;
                        (x - 3.0).sin() * (y - 3.0).cos()
                    })
                    .collect()
            })
            .collect();
        ax.heatmap(&field, Colormap::Viridis, (0.0, 6.0, 0.0, 6.0), true);
    }

    grid.save_svg("/tmp/viz_grid.svg").unwrap();

    // ----------------------------------------------------------------------
    // 5) Log + symlog scales.
    // ----------------------------------------------------------------------
    let mut f5 = Figure::subplots(900, 380, 1, 2);
    f5.suptitle("scales");
    {
        let ax = f5.ax_at(0, 0).unwrap();
        ax.set_title("log y").set_grid(true).set_yscale(Scale::Log);
        let x: Vec<f64> = (1..=6).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 10f64.powf(v - 1.0)).collect();
        ax.line(
            &x,
            &y,
            Color::ORANGE,
            2.0,
            LineStyle::Solid,
            Marker::Circle,
            1.0,
            Some("10^(x-1)"),
        );
        ax.legend(LegendLoc::UpperLeft);
    }
    {
        let ax = f5.ax_at(0, 1).unwrap();
        ax.set_title("symlog y")
            .set_grid(true)
            .set_yscale(Scale::Symlog { linthresh: 1.0 });
        ax.set_tick_direction(TickDirection::In);
        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v - 10.0).powi(3) * 0.1).collect();
        ax.plot_styled(&x, &y, Color::CYAN, 2.0);
        ax.axhline(0.0, Color::GRAY, LineStyle::Dotted);
    }
    f5.save_svg("/tmp/viz_scales.svg").unwrap();

    // ----------------------------------------------------------------------
    // 6) Horizontal + stacked bars and a styled spine demo.
    // ----------------------------------------------------------------------
    let mut f6 = Figure::subplots(900, 400, 1, 2);
    {
        let ax = f6.ax_at(0, 0).unwrap();
        ax.set_title("horizontal bars").set_xlabel("score");
        ax.barh(
            &[1.0, 2.0, 3.0, 4.0],
            &[3.0, 7.0, 5.0, 9.0],
            0.6,
            Color::OLIVE,
            Some("team"),
        );
        ax.set_spines(false, false, true, true);
    }
    {
        let ax = f6.ax_at(0, 1).unwrap();
        ax.set_title("stacked bars");
        let x: Vec<f64> = (1..=4).map(|i| i as f64).collect();
        ax.bar_stacked(
            &x,
            &[
                vec![2.0, 3.0, 1.0, 4.0],
                vec![1.0, 2.0, 3.0, 1.0],
                vec![3.0, 1.0, 2.0, 2.0],
            ],
            0.6,
            &["A", "B", "C"],
        );
        ax.axvspan(2.5, 3.5, Color::LIGHT_GRAY, 0.4);
        ax.legend(LegendLoc::UpperLeft);
    }
    f6.save_svg("/tmp/viz_bars.svg").unwrap();

    println!(
        "wrote /tmp/viz_lines.svg /tmp/viz_scatter.svg /tmp/viz_hist.svg \
         /tmp/viz_grid.svg /tmp/viz_scales.svg /tmp/viz_bars.svg"
    );

    // Optional PNG raster export when built with `--features png`.
    #[cfg(feature = "png")]
    {
        grid.save_png("/tmp/viz_grid.png", 2.0).unwrap();
        println!("wrote /tmp/viz_grid.png (2x raster)");
    }
}
