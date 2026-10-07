//! BitNet-style ternary inference primitives for MSSA.
//!
//! This is an inference-only, post-training probe.  It follows b1.58's
//! absmean weight quantizer and symmetric per-token int8 activation scale, but
//! keeps the recurrent carry, normalization, memory values, and FP32 master
//! weights untouched.  The packed representation uses two bits per ternary
//! weight; a production kernel should decode several weights at a time rather
//! than extracting one code per scalar as this reference implementation does.

use crate::pssa::ParamMatrix;

const ACTIVATION_MAX: f32 = 127.0;

#[inline(always)]
fn decode_ternary(code: u8) -> i32 {
    debug_assert!(code <= 3);
    if code == 3 {
        0
    } else {
        code as i32 - 1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TernaryMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Four ternary values per byte: code 0 = -1, 1 = 0, 2 = +1.
    pub packed_weights: Vec<u8>,
    /// Absmean scale used to reconstruct each output row.
    pub row_scales: Vec<f32>,
    /// Matrix-wide scale retained for reporting the native b1.58 variant.
    pub scale: f32,
}

impl TernaryMatrix {
    pub fn from_param(matrix: &ParamMatrix) -> Result<Self, String> {
        Self::from_param_with_rowwise(matrix, false)
    }

    /// Post-training deployment calibration. BitNet's native training recipe
    /// uses a matrix-wide absmean scale; rowwise calibration is a safer bridge
    /// for an already-trained FP32 checkpoint because output rows have unequal
    /// magnitude distributions.
    pub fn from_param_rowwise(matrix: &ParamMatrix) -> Result<Self, String> {
        Self::from_param_with_rowwise(matrix, true)
    }

    fn from_param_with_rowwise(matrix: &ParamMatrix, rowwise: bool) -> Result<Self, String> {
        if matrix.rows == 0 || matrix.cols == 0 {
            return Err("BitNet matrix dimensions must be positive".into());
        }
        if matrix.data.iter().any(|x| !x.is_finite()) {
            return Err("BitNet quantization received a non-finite matrix".into());
        }
        let mean_abs = matrix.data.iter().map(|x| x.abs()).sum::<f32>()
            / matrix.data.len() as f32;
        let scale = if mean_abs > 0.0 { mean_abs } else { 1.0 };
        let mut packed_weights = vec![0u8; matrix.data.len().div_ceil(4)];
        let mut row_scales = vec![scale; matrix.rows];
        if rowwise {
            for row in 0..matrix.rows {
                let row_data = &matrix.data[row * matrix.cols..(row + 1) * matrix.cols];
                let row_mean = row_data.iter().map(|x| x.abs()).sum::<f32>() / matrix.cols as f32;
                row_scales[row] = if row_mean > 0.0 { row_mean } else { 1.0 };
            }
        }
        for (index, &weight) in matrix.data.iter().enumerate() {
            let row_scale = row_scales[index / matrix.cols];
            let code = ((weight / row_scale).round().clamp(-1.0, 1.0) as i8 + 1) as u8;
            packed_weights[index / 4] |= code << ((index % 4) * 2);
        }
        Ok(Self {
            rows: matrix.rows,
            cols: matrix.cols,
            packed_weights,
            row_scales,
            scale,
        })
    }

    #[cfg(test)]
    #[inline]
    fn weight_code(&self, index: usize) -> i8 {
        let code = (self.packed_weights[index / 4] >> ((index % 4) * 2)) & 0b11;
        decode_ternary(code) as i8
    }

    pub fn storage_bytes(&self) -> usize {
        self.packed_weights.len() + self.row_scales.len() * std::mem::size_of::<f32>()
    }

    /// Compute W_hat * x_hat and rescale it to the FP32 product domain.
    /// `activation_q` is caller-owned scratch, so the hot path allocates zero.
    pub fn matvec_int8(
        &self,
        input: &[f32],
        activation_q: &mut [i8],
        output: &mut [f32],
    ) -> Result<f32, String> {
        if input.len() != self.cols || activation_q.len() != self.cols || output.len() != self.rows {
            return Err("BitNet matvec dimensions do not match".into());
        }
        let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if !max_abs.is_finite() {
            return Err("BitNet activation is non-finite".into());
        }
        if max_abs == 0.0 {
            activation_q.fill(0);
            output.fill(0.0);
            return Ok(0.0);
        }
        let activation_scale = max_abs / ACTIVATION_MAX;
        for (q, &x) in activation_q.iter_mut().zip(input) {
            *q = (x / activation_scale).round().clamp(-ACTIVATION_MAX, ACTIVATION_MAX) as i8;
        }
        for row in 0..self.rows {
            let mut sum = 0i32;
            let offset = row * self.cols;
            let mut col = 0;
            // Keep the common aligned case branch-light: four ternary values
            // share one byte, and most dense model dimensions are multiples of
            // four. The fallback handles rows whose boundaries cross a byte.
            while col < self.cols {
                let index = offset + col;
                let lane = index % 4;
                let byte = self.packed_weights[index / 4];
                let available = (4 - lane).min(self.cols - col);
                if lane == 0 && available == 4 {
                    let code0 = decode_ternary(byte & 0b11);
                    let code1 = decode_ternary((byte >> 2) & 0b11);
                    let code2 = decode_ternary((byte >> 4) & 0b11);
                    let code3 = decode_ternary((byte >> 6) & 0b11);
                    sum += code0 * i32::from(activation_q[col]);
                    sum += code1 * i32::from(activation_q[col + 1]);
                    sum += code2 * i32::from(activation_q[col + 2]);
                    sum += code3 * i32::from(activation_q[col + 3]);
                } else {
                    for lane_offset in 0..available {
                        let code = decode_ternary((byte >> ((lane + lane_offset) * 2)) & 0b11);
                        sum += code * i32::from(activation_q[col + lane_offset]);
                    }
                }
                col += available;
            }
            output[row] = sum as f32 * self.row_scales[row] * activation_scale;
        }
        Ok(activation_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize) -> ParamMatrix {
        let mut matrix = ParamMatrix::zeros(rows, cols);
        for (i, x) in matrix.data.iter_mut().enumerate() {
            *x = ((i * 13 % 17) as f32 - 8.0) * 0.11;
        }
        matrix
    }

    #[test]
    fn weights_are_ternary_and_packed() {
        let matrix = matrix(3, 7);
        let quantized = TernaryMatrix::from_param(&matrix).unwrap();
        assert_eq!(quantized.packed_weights.len(), (3usize * 7).div_ceil(4));
        assert!(quantized.scale.is_finite() && quantized.scale > 0.0);
        for i in 0..21 {
            assert!((-1..=1).contains(&quantized.weight_code(i)));
        }
        assert!(quantized.storage_bytes() < matrix.data.len() * std::mem::size_of::<f32>());
    }

    #[test]
    fn int8_activation_path_is_finite_and_tracks_fp32_direction() {
        let matrix = matrix(5, 9);
        let quantized = TernaryMatrix::from_param(&matrix).unwrap();
        let input: Vec<f32> = (0..9).map(|i| i as f32 * 0.07 - 0.2).collect();
        let mut q = vec![0i8; 9];
        let mut actual = vec![0.0; 5];
        quantized.matvec_int8(&input, &mut q, &mut actual).unwrap();
        let mut expected = vec![0.0; 5];
        matrix.matvec(&input, &mut expected);
        assert!(actual.iter().all(|x| x.is_finite()));
        assert!(actual.iter().zip(&expected).any(|(a, e)| a.signum() == e.signum()));
    }

    #[test]
    fn zero_activation_is_zero_without_division() {
        let quantized = TernaryMatrix::from_param(&matrix(2, 3)).unwrap();
        let mut q = [7i8; 3];
        let mut out = [1.0; 2];
        assert_eq!(quantized.matvec_int8(&[0.0; 3], &mut q, &mut out), Ok(0.0));
        assert_eq!(q, [0, 0, 0]);
        assert_eq!(out, [0.0, 0.0]);
    }
}
