//! The Swift-facing API (uniffi). A `Frame` is one decoded proxy; `render` runs the pipeline
//! on an edit given as the desktop's flat JSON and returns display-encoded RGBA.

use std::sync::Arc;

use crate::config::Edit;
use crate::image::Image;
use crate::pipeline::engine::{self, EngineError};

#[derive(Debug, uniffi::Error)]
#[uniffi(flat_error)]
pub enum NegPyError {
    Decode(String),
    Edit(String),
    Render(String),
}

impl std::fmt::Display for NegPyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NegPyError::Decode(m) | NegPyError::Edit(m) | NegPyError::Render(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for NegPyError {}

impl From<EngineError> for NegPyError {
    fn from(e: EngineError) -> Self {
        NegPyError::Render(e.to_string())
    }
}

#[derive(uniffi::Record)]
pub struct Rendered {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, display-encoded RGBA, alpha 255.
    pub rgba: Vec<u8>,
    /// The metrics the Analysis panel shows, as JSON.
    pub metrics_json: String,
}

#[derive(uniffi::Object)]
pub struct Frame {
    source: Image,
}

#[uniffi::export]
impl Frame {
    /// Decodes a proxy PNG from the sync folder.
    #[uniffi::constructor]
    pub fn from_png(png: Vec<u8>) -> Result<Arc<Self>, NegPyError> {
        let source =
            crate::proxy::decode_png(&png).map_err(|e| NegPyError::Decode(e.to_string()))?;
        Ok(Arc::new(Self { source }))
    }

    pub fn width(&self) -> u32 {
        self.source.width as u32
    }

    pub fn height(&self) -> u32 {
        self.source.height as u32
    }

    /// Renders the edit at a preview size whose long edge is at most `max_edge`.
    pub fn render(&self, edit_json: String, max_edge: u32) -> Result<Rendered, NegPyError> {
        let edit = Edit::from_json(&edit_json).map_err(|e| NegPyError::Edit(e.to_string()))?;
        let out = engine::render_preview(&self.source, &edit.config, max_edge as usize)?;
        let mut rgba = vec![0u8; out.image.width * out.image.height * 4];
        let mut q = vec![0u8; out.image.data.len()];
        negpy_core::convert::to_uint(&out.image.data, &mut q);
        for (dst, src) in rgba
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(q.as_chunks::<3>().0)
        {
            *dst = [src[0], src[1], src[2], 255];
        }
        Ok(Rendered {
            width: out.image.width as u32,
            height: out.image.height as u32,
            rgba,
            metrics_json: out.metrics_json,
        })
    }
}

/// The desktop's default edit as flat JSON, for a frame with no record yet.
#[uniffi::export]
pub fn default_edit_json() -> String {
    engine::default_edit_json()
}
