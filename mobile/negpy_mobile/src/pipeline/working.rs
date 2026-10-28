//! The working space's output transform: the Adobe RGB (1998) TRC, a pure 563/256 power
//! (`kernel/image/logic.py`).

use crate::image::Image;

pub const WORKING_GAMMA: f32 = 563.0 / 256.0;

/// Scene-linear to display-encoded code values in 0..1.
pub fn oetf_encode(img: &Image) -> Image {
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::color::oetf_encode(&img.data, &mut out, 1.0f32 / WORKING_GAMMA);
    Image::new(img.height, img.width, out)
}

/// Inverse of `oetf_encode`.
pub fn oetf_decode(img: &Image) -> Image {
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::color::oetf_decode(&img.data, &mut out, WORKING_GAMMA);
    Image::new(img.height, img.width, out)
}
