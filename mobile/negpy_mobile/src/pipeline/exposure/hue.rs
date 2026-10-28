//! `features/process/hue.py`: Hue Trim, a rotation of every hue about the neutral axis in
//! the working CIELAB, with the working Lab constants of `kernel/image/logic.py`.

use negpy_core::color::LabSpace;

use crate::image::Image;

pub const WORKING_TO_XYZ: [[f32; 3]; 3] = [
    [0.5767309, 0.185554, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
];
pub const XYZ_TO_WORKING: [[f32; 3]; 3] = [
    [2.041369, -0.5649464, -0.3446944],
    [-0.969266, 1.8760108, 0.041556],
    [0.0134474, -0.1183897, 1.0154096],
];
pub const WORKING_WHITE: [f32; 3] = [0.95047, 1.0, 1.08883];
pub const LAB_EPS: f32 = 0.008856;
pub const LAB_KAPPA: f32 = 7.787;

/// `WORKING_LAB` as the Lab kernels take it: the forward and the inverse space.
pub fn working_lab() -> (LabSpace, LabSpace) {
    (
        LabSpace::new(WORKING_TO_XYZ, WORKING_WHITE, LAB_EPS, LAB_KAPPA),
        LabSpace::new(XYZ_TO_WORKING, WORKING_WHITE, LAB_EPS, LAB_KAPPA),
    )
}

/// Rotates every hue by `degrees` in the working a*b* plane. Identity at 0.
pub fn apply_hue_trim(img: &Image, degrees: f64) -> Image {
    if degrees == 0.0 {
        return img.clone();
    }
    let rad = degrees * (std::f64::consts::PI / 180.0);
    let rotation = (libm::cos(rad) as f32, libm::sin(rad) as f32);
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::lab::hue_trim(&img.data, rotation, working_lab(), &mut out);
    Image::new(img.height, img.width, out)
}
