//! NegPy's pipeline for the iOS app: the same stages as `negpy/services/rendering/engine.py`,
//! on the `negpy_core` kernels, without Python. Pixels match the desktop CPU engine; the
//! parity tests under `tests/` hold fixtures the desktop engine rendered.

pub mod config;
pub mod context;
pub mod image;
pub mod pipeline;

#[doc(hidden)]
pub mod fixture;
