//! The proxy a desktop writes into the sync folder (`negpy/services/sync/store.py`): the
//! frame's baked, scene-linear source as a 16-bit PNG. This reads that PNG back into the
//! pipeline's image: non-interlaced, 8 or 16 bits, gray, RGB or RGBA, which is every PNG the
//! desktop's encoder writes.

use crate::image::Image;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DecodeError {}

fn err<T>(msg: &str) -> Result<T, DecodeError> {
    Err(DecodeError(msg.to_string()))
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = (
        (p - a as i16).abs(),
        (p - b as i16).abs(),
        (p - c as i16).abs(),
    );
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Undoes the per-row filters in place, returning the raw sample bytes.
fn unfilter(raw: &[u8], h: usize, row_bytes: usize, bpp: usize) -> Result<Vec<u8>, DecodeError> {
    if raw.len() < h * (row_bytes + 1) {
        return err("PNG image data is short");
    }
    let mut out = vec![0u8; h * row_bytes];
    for y in 0..h {
        let filter = raw[y * (row_bytes + 1)];
        let src = &raw[y * (row_bytes + 1) + 1..(y + 1) * (row_bytes + 1)];
        let (before, cur) = out.split_at_mut(y * row_bytes);
        let cur = &mut cur[..row_bytes];
        let prev: &[u8] = if y > 0 {
            &before[(y - 1) * row_bytes..]
        } else {
            &[]
        };
        for x in 0..row_bytes {
            let a = if x >= bpp { cur[x - bpp] } else { 0 };
            let b = if y > 0 { prev[x] } else { 0 };
            let c = if y > 0 && x >= bpp { prev[x - bpp] } else { 0 };
            cur[x] = match filter {
                0 => src[x],
                1 => src[x].wrapping_add(a),
                2 => src[x].wrapping_add(b),
                3 => src[x].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => src[x].wrapping_add(paeth(a, b, c)),
                _ => return err("PNG row filter is unknown"),
            };
        }
    }
    Ok(out)
}

/// Decodes the PNG into a scene-linear RGB image. Gray repeats into three channels; an alpha
/// channel is dropped.
pub fn decode_png(bytes: &[u8]) -> Result<Image, DecodeError> {
    if bytes.len() < 8 || bytes[..8] != [137, 80, 78, 71, 13, 10, 26, 10] {
        return err("not a PNG file");
    }
    let (mut w, mut h, mut depth, mut color) = (0usize, 0usize, 0u8, 0u8);
    let mut idat = Vec::new();
    let mut pos = 8;
    while pos + 8 <= bytes.len() {
        let len = be32(&bytes[pos..]) as usize;
        let tag = &bytes[pos + 4..pos + 8];
        let data = bytes
            .get(pos + 8..pos + 8 + len)
            .ok_or_else(|| DecodeError("PNG chunk is short".into()))?;
        match tag {
            b"IHDR" => {
                if data.len() < 13 {
                    return err("PNG header is short");
                }
                w = be32(data) as usize;
                h = be32(&data[4..]) as usize;
                depth = data[8];
                color = data[9];
                if data[12] != 0 {
                    return err("interlaced PNG is not supported");
                }
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        pos += 12 + len;
    }
    let channels = match color {
        0 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        _ => return err("PNG color type is not supported"),
    };
    if !(depth == 8 || depth == 16) || w == 0 || h == 0 {
        return err("PNG depth is not supported");
    }
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&idat)
        .map_err(|_| DecodeError("PNG data does not inflate".into()))?;
    let bpp = channels * depth as usize / 8;
    let samples = unfilter(&raw, h, w * bpp, bpp)?;
    let n = w * h;
    let mut rgb = vec![0f32; n * 3];
    if depth == 16 {
        let values: Vec<u16> = samples
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_be_bytes(*b))
            .collect();
        let mut lin = vec![0f32; values.len()];
        negpy_core::convert::uint16_to_float32(&values, &mut lin);
        spread(&lin, channels, &mut rgb);
    } else {
        let mut lin = vec![0f32; samples.len()];
        negpy_core::convert::uint8_to_float32(&samples, &mut lin);
        spread(&lin, channels, &mut rgb);
    }
    Ok(Image::new(h, w, rgb))
}

fn spread(src: &[f32], channels: usize, rgb: &mut [f32]) {
    for (i, px) in rgb.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let s = &src[i * channels..(i + 1) * channels];
        match channels {
            1 | 2 => *px = [s[0], s[0], s[0]],
            _ => *px = [s[0], s[1], s[2]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode16(img: &Image) -> Vec<u8> {
        let mut q = vec![0u16; img.data.len()];
        negpy_core::convert::to_uint(&img.data, &mut q);
        negpy_core::png::encode(img.height, img.width, 3, 16, 6, |y, out| {
            for v in &q[y * img.width * 3..(y + 1) * img.width * 3] {
                out.extend_from_slice(&v.to_be_bytes());
            }
        })
    }

    #[test]
    fn a_proxy_round_trips_to_16_bit_precision() {
        let (h, w) = (9, 13);
        let data: Vec<f32> = (0..h * w * 3).map(|i| (i as f32 * 0.37).fract()).collect();
        let img = Image::new(h, w, data);
        let back = decode_png(&encode16(&img)).unwrap();
        assert_eq!(back.shape(), (h, w));
        assert!(back.max_abs_diff(&img) <= 0.5 / 65535.0);
    }

    #[test]
    fn junk_is_refused() {
        assert!(decode_png(b"not a png").is_err());
    }
}
