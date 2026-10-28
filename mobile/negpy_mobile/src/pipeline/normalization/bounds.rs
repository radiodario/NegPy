//! The density bounds: `analyze_log_exposure_bounds_from_log` and the roll lock around it
//! (`resolve_bounds_detailed`, `mix_luma_color_bounds`, `luma_source_bounds`).

use negpy_core::bounds::{self, ColorPass};

use super::{Bounds, LUMA, Unmix, block_median_grid, metered, unmix_pixel};
use crate::config::ProcessConfig;
use crate::image::{Image, Roi};
use crate::pipeline::constants::EXPOSURE_CONSTANTS;

/// The middle of three values, `sorted(v)[1]`.
fn median3(v: [f64; 3]) -> f64 {
    let mut s = v;
    s.sort_by(f64::total_cmp);
    s[1]
}

/// `_sample_log_bounds`: per-channel (floors, ceils) at one clip level over the robust base
/// clip; a negative clip expands outward by that margin instead.
fn sample_log_bounds(grid: &Image, percentile_clip: f64, base: f64) -> ([f64; 3], [f64; 3]) {
    let (clip, margin) = if percentile_clip >= 0.0 {
        (0.00001f64.max(50.0f64.min(percentile_clip + base)), 0.0)
    } else {
        (base, -percentile_clip)
    };
    let per = bounds::channel_percentiles(&grid.data, &[clip, 100.0 - clip]);
    let mut floors: [f64; 3] = std::array::from_fn(|ch| per[ch][0]);
    let mut ceils: [f64; 3] = std::array::from_fn(|ch| per[ch][1]);
    if margin > 0.0 {
        for ch in 0..3 {
            if ceils[ch] >= floors[ch] {
                floors[ch] -= margin;
                ceils[ch] += margin;
            } else {
                floors[ch] += margin;
                ceils[ch] -= margin;
            }
        }
    }
    (floors, ceils)
}

/// `_same_pixel_color_floor_refs`: dense-end color refs from the luma band's lowest-chroma
/// pixels; None when that set is too small or too chromatic.
fn same_pixel_color_floor_refs(
    grid: &Image,
    luma_floors: [f64; 3],
    luma_ceils: [f64; 3],
    base_refs: [f64; 3],
    color_clip: f64,
) -> Option<[f64; 3]> {
    let c = &EXPOSURE_CONSTANTS;
    let pass = ColorPass {
        chroma_quantile: c.neutral_axis_chroma_quantile,
        chroma_cap: c.neutral_axis_chroma_cap,
        first_pass_cap: c.neutral_axis_first_pass_cap,
        min_pixels: c.neutral_axis_min_pixels,
        band_width: c.color_bounds_band_width,
    };
    bounds::same_pixel_color_floor_refs(
        &grid.data,
        luma_floors,
        luma_ceils,
        base_refs,
        color_clip,
        pass,
    )
}

/// `analyze_log_exposure_bounds_from_log`: the luma pass sets the window's span, the color pass
/// its per-channel refs; the window reaches past the refs by one shared fraction of each
/// channel's span.
pub fn analyze_log_exposure_bounds_from_log(
    img_log: &Image,
    roi: Option<Roi>,
    buffer: f64,
    percentile_clip: f64,
    color_clip: f64,
) -> Bounds {
    let grid = block_median_grid(metered(img_log, roi, buffer));
    let (mut floors, mut ceils) =
        sample_log_bounds(&grid, percentile_clip, EXPOSURE_CONSTANTS.base_luma_clip);
    let (mut c_floors, c_ceils) = sample_log_bounds(&grid, color_clip, 0.0);
    if color_clip >= 0.0
        && let Some(sp) = same_pixel_color_floor_refs(&grid, floors, ceils, c_ceils, color_clip)
    {
        c_floors = sp;
    }
    let spans: [f64; 3] = std::array::from_fn(|ch| c_floors[ch] - c_ceils[ch]);
    let total = spans[0] + spans[1] + spans[2];
    if spans.iter().all(|s| s * total > 0.0) {
        let reach_f =
            ((floors[0] - c_floors[0]) + (floors[1] - c_floors[1]) + (floors[2] - c_floors[2]))
                / total;
        let reach_c =
            ((ceils[0] - c_ceils[0]) + (ceils[1] - c_ceils[1]) + (ceils[2] - c_ceils[2])) / total;
        let shift = ((c_ceils[0] + c_ceils[1] + c_ceils[2]) - 3.0 * median3(c_ceils)) / total;
        floors = std::array::from_fn(|ch| c_floors[ch] + (reach_f + shift) * spans[ch]);
        ceils = std::array::from_fn(|ch| c_ceils[ch] + (reach_c + shift) * spans[ch]);
    } else {
        let mean_lf = (floors[0] + floors[1] + floors[2]) / 3.0;
        let mean_lc = (ceils[0] + ceils[1] + ceils[2]) / 3.0;
        let (mean_cf, mean_cc) = (median3(c_floors), median3(c_ceils));
        floors = std::array::from_fn(|ch| mean_lf + (c_floors[ch] - mean_cf));
        ceils = std::array::from_fn(|ch| mean_lc + (c_ceils[ch] - mean_cc));
    }
    Bounds::new(floors, ceils)
}

/// `mix_luma_color_bounds`: the color source's per-channel shape, shifted so its luma-weighted
/// centre and range match the luma source.
pub fn mix_luma_color_bounds(luma_src: &Bounds, color_src: &Bounds) -> Bounds {
    let centre = |b: &Bounds| {
        let t = |c: usize| LUMA[c] * (b.floors[c] + b.ceils[c]) / 2.0;
        t(0) + t(1) + t(2)
    };
    let range = |b: &Bounds| {
        let t = |c: usize| LUMA[c] * (b.ceils[c] - b.floors[c]);
        t(0) + t(1) + t(2)
    };
    let d_c = centre(luma_src) - centre(color_src);
    let d_r = range(luma_src) - range(color_src);
    let (df, dc) = (d_c - d_r / 2.0, d_c + d_r / 2.0);
    Bounds::new(
        color_src.floors.map(|f| f + df),
        color_src.ceils.map(|c| c + dc),
    )
}

/// `resolve_bounds_detailed`: (final, base). Luma and color each come from the roll baseline
/// or the per-frame base, then mix; `analyze` runs only when a base is needed. The desktop's
/// mix returns its input untouched when both sources are the same object, so the mix runs only
/// when exactly one axis is locked.
pub fn resolve_bounds_detailed(
    process: &ProcessConfig,
    analyze: impl FnOnce() -> Bounds,
) -> (Bounds, Bounds) {
    let roll_luma = process.use_luma_average && process.is_locked_initialized();
    let roll_color = process.use_color_average && process.is_locked_initialized();
    let locked = Bounds::from_rgb(process.locked_floors, process.locked_ceils);
    if roll_luma && roll_color {
        return (locked, locked);
    }
    let base = if process.is_local_initialized() {
        Bounds::from_rgb(process.local_floors, process.local_ceils)
    } else {
        analyze()
    };
    let final_bounds = if roll_luma == roll_color {
        base
    } else {
        mix_luma_color_bounds(
            if roll_luma { &locked } else { &base },
            if roll_color { &locked } else { &base },
        )
    };
    (final_bounds, base)
}

/// `luma_source_bounds`: what the exposure meters read, the roll baseline under luma-average.
pub fn luma_source_bounds(process: &ProcessConfig, base: &Bounds) -> Bounds {
    if process.use_luma_average && process.is_locked_initialized() {
        Bounds::from_rgb(process.locked_floors, process.locked_ceils)
    } else {
        *base
    }
}

/// `luminance_density_range`: the Rec. 709 weighting of the per-channel ranges.
pub fn luminance_density_range(bounds: &Bounds) -> f64 {
    let r = |c: usize| (bounds.ceils[c] - bounds.floors[c]).abs();
    LUMA[0] * r(0) + LUMA[1] * r(1) + LUMA[2] * r(2)
}

/// `np.dot` of the luma weights with a triplet. The three-term order of the BLAS dot is not
/// mirrored; this feeds the Analysis panel, not a render.
fn luma_dot(v: [f64; 3]) -> f64 {
    LUMA[0] * v[0] + LUMA[1] * v[1] + LUMA[2] * v[2]
}

/// `density_above_base`: (Dmin, Dmed, Dmax) above the clear base, luminance-weighted. None
/// without a base.
pub fn density_above_base(
    base_log: Option<[f64; 3]>,
    bounds: &Bounds,
    prefiltered: &Image,
    unmix: Option<&Unmix>,
) -> Option<[f64; 3]> {
    let base32 = base_log?.map(|v| v as f32);
    let base_rgb = match unmix {
        Some(m) => unmix_pixel(base32, &m.map(|row| row.map(|v| v as f32))),
        None => base32,
    };
    let base = luma_dot(base_rgb.map(|v| v as f64));
    let thin = luma_dot(bounds.ceils);
    let dense = luma_dot(bounds.floors);
    let median = negpy_core::analysis::luma_median(&prefiltered.data);
    Some([base - thin, base - median, base - dense])
}
