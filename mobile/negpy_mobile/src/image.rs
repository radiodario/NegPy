//! The pipeline's image: float32 scene-linear RGB, row-major, interleaved, like the desktop's
//! `ImageBuffer` (an `(h, w, 3)` float32 NumPy array).

/// (y1, y2, x1, x2) in pixels, the desktop's `ROI`.
pub type Roi = (usize, usize, usize, usize);

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub height: usize,
    pub width: usize,
    /// `height * width * 3` floats, pixel `(y, x)` at `(y * width + x) * 3`.
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(height: usize, width: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            height * width * 3,
            "image data does not match its shape"
        );
        Self {
            height,
            width,
            data,
        }
    }

    pub fn zeros(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            data: vec![0.0; height * width * 3],
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn pixel(&self, y: usize, x: usize) -> [f32; 3] {
        let i = (y * self.width + x) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// The rows `y1..y2` and columns `x1..x2`, copied.
    pub fn crop(&self, roi: Roi) -> Image {
        let (y1, y2, x1, x2) = roi;
        let (y1, y2) = (y1.min(self.height), y2.min(self.height));
        let (x1, x2) = (x1.min(self.width), x2.min(self.width));
        let (h, w) = (y2.saturating_sub(y1), x2.saturating_sub(x1));
        let mut out = Vec::with_capacity(h * w * 3);
        for y in y1..y2 {
            let row = &self.data[(y * self.width + x1) * 3..(y * self.width + x2) * 3];
            out.extend_from_slice(row);
        }
        Image::new(h, w, out)
    }

    /// Largest absolute difference against `other`, for the parity tests.
    pub fn max_abs_diff(&self, other: &Image) -> f32 {
        assert_eq!(self.shape(), other.shape(), "shapes differ");
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }
}

/// A single-channel float plane with the image's layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(height: usize, width: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            height * width,
            "plane data does not match its shape"
        );
        Self {
            height,
            width,
            data,
        }
    }
}
