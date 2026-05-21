//! Pooling and normalization — the error-prone math, isolated and tested.
//!
//! Sentence-embedding models emit a per-token hidden state
//! `[seq_len, hidden_dim]`. To get a single vector you pool over tokens.
//! Two facts make this easy to get wrong, so they live here behind unit
//! tests rather than buried in the (un-CI-able) ONNX glue:
//!
//! 1. **Mean pooling must respect the attention mask.** Padding tokens
//!    have hidden states too; averaging over them corrupts the vector.
//!    The mask weights the sum and the divisor.
//! 2. **L2 normalization is what makes cosine == dot product.** BGE/E5
//!    cosine similarity assumes unit vectors; skipping normalization
//!    silently changes the similarity scale.
//!
//! These functions operate on plain `&[f32]` tensors so they're testable
//! with synthetic inputs, no model required.

use serde::{Deserialize, Serialize};

/// Token-pooling strategy.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum Pooling {
    /// Attention-mask-weighted mean over token hidden states. The right
    /// default for E5, GTE, and most sentence-transformers.
    Mean,
    /// Take the `[CLS]` token (position 0). Used by BGE and mxbai.
    Cls,
}

/// Pool a `[seq_len, hidden_dim]` row-major hidden-state tensor into a
/// single `hidden_dim` vector.
///
/// `attention_mask` is `seq_len` long: `1.0` for real tokens, `0.0` for
/// padding. For [`Pooling::Cls`] the mask is ignored (the CLS token is
/// always real).
///
/// Panics if `hidden_states.len() != seq_len * hidden_dim` or
/// `attention_mask.len() != seq_len` — these are programmer errors in
/// the calling glue, not runtime conditions.
pub fn pool(
    hidden_states: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    attention_mask: &[f32],
    pooling: Pooling,
) -> Vec<f32> {
    assert_eq!(
        hidden_states.len(),
        seq_len * hidden_dim,
        "hidden_states shape mismatch"
    );
    assert_eq!(attention_mask.len(), seq_len, "attention_mask shape mismatch");

    match pooling {
        Pooling::Cls => {
            // First token's row.
            hidden_states[0..hidden_dim].to_vec()
        }
        Pooling::Mean => {
            let mut acc = vec![0f32; hidden_dim];
            let mut mask_sum = 0f32;
            for t in 0..seq_len {
                let m = attention_mask[t];
                if m == 0.0 {
                    continue;
                }
                mask_sum += m;
                let off = t * hidden_dim;
                for d in 0..hidden_dim {
                    acc[d] += hidden_states[off + d] * m;
                }
            }
            if mask_sum > 0.0 {
                for d in 0..hidden_dim {
                    acc[d] /= mask_sum;
                }
            }
            acc
        }
    }
}

/// L2-normalize a vector in place. A zero vector is left unchanged
/// (rather than producing NaNs).
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cls_pooling_takes_first_token() {
        // 3 tokens, hidden_dim 2.
        let hidden = vec![
            1.0, 2.0, // token 0 (CLS)
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
        ];
        let mask = vec![1.0, 1.0, 1.0];
        let pooled = pool(&hidden, 3, 2, &mask, Pooling::Cls);
        assert_eq!(pooled, vec![1.0, 2.0]);
    }

    #[test]
    fn mean_pooling_ignores_padding() {
        // 3 tokens, hidden_dim 2; token 2 is padding (mask 0).
        let hidden = vec![
            2.0, 4.0, // token 0
            4.0, 8.0, // token 1
            100.0, 100.0, // token 2 — padding, must be ignored
        ];
        let mask = vec![1.0, 1.0, 0.0];
        let pooled = pool(&hidden, 3, 2, &mask, Pooling::Mean);
        // Mean over tokens 0,1 = (3.0, 6.0). Padding excluded.
        assert_eq!(pooled, vec![3.0, 6.0]);
    }

    #[test]
    fn mean_pooling_without_padding_is_plain_average() {
        let hidden = vec![1.0, 1.0, 3.0, 3.0];
        let mask = vec![1.0, 1.0];
        let pooled = pool(&hidden, 2, 2, &mask, Pooling::Mean);
        assert_eq!(pooled, vec![2.0, 2.0]);
    }

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let mut v = vec![3.0, 4.0]; // norm 5
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_leaves_zero_vector_unchanged() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn all_padding_does_not_divide_by_zero() {
        let hidden = vec![5.0, 5.0];
        let mask = vec![0.0];
        let pooled = pool(&hidden, 1, 2, &mask, Pooling::Mean);
        assert_eq!(pooled, vec![0.0, 0.0]);
    }
}
