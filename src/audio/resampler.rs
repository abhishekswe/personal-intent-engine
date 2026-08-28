use rubato::{FftFixedIn, Resampler};
use std::time::Duration;

const RESAMPLER_CHUNK_SIZE: usize = 1024;

/// Streaming resampler that converts mono audio to the target rate and emits
/// fixed-duration frames (e.g. 30 ms / 480 samples at 16 kHz) via a callback.
///
/// Frame-based resampler: accepts native-rate frames, outputs 16kHz for STT.
/// - `push()` buffers input and processes full chunks, emitting complete frames
/// - `finish()` flushes the buffered remainder (zero-padded) so no tail audio
///   is lost at the end of a recording
/// - `reset()` clears all state between recordings so FFT overlap buffers
///   cannot leak audio from one session into the next
///
/// Input must already be mono; multi-channel downmixing happens in the cpal
/// callback before samples reach this type.
pub struct AudioResampler {
    resampler: Option<FftFixedIn<f32>>,
    chunk_in: usize,
    in_buf: Vec<f32>,
    frame_samples: usize,
    pending: Vec<f32>,
}

impl AudioResampler {
    /// Create a resampler from `in_hz` to `out_hz` emitting frames of
    /// `frame_dur` duration. When rates match, input passes through and is
    /// only re-chunked into frames.
    #[must_use]
    pub fn new(in_hz: usize, out_hz: usize, frame_dur: Duration) -> Self {
        let frame_samples = ((out_hz as f64 * frame_dur.as_secs_f64()).round()) as usize;
        assert!(frame_samples > 0, "frame duration too short");

        let chunk_in = RESAMPLER_CHUNK_SIZE;

        let resampler = (in_hz != out_hz).then(|| {
            FftFixedIn::<f32>::new(in_hz, out_hz, chunk_in, 1, 1)
                .expect("Failed to create resampler")
        });

        Self {
            resampler,
            chunk_in,
            in_buf: Vec::with_capacity(chunk_in),
            frame_samples,
            pending: Vec::with_capacity(frame_samples),
        }
    }

    /// Feed mono samples; `emit` is called with each completed frame.
    pub fn push(&mut self, mut src: &[f32], emit: &mut impl FnMut(&[f32])) {
        if self.resampler.is_none() {
            self.emit_frames(src, emit);
            return;
        }

        while !src.is_empty() {
            let space = self.chunk_in - self.in_buf.len();
            let take = space.min(src.len());
            self.in_buf.extend_from_slice(&src[..take]);
            src = &src[take..];

            if self.in_buf.len() == self.chunk_in {
                if let Ok(out) = self
                    .resampler
                    .as_mut()
                    .expect("resampler checked above")
                    .process(&[&self.in_buf[..]], None)
                {
                    self.emit_frames(&out[0], emit);
                }
                self.in_buf.clear();
            }
        }
    }

    /// Flush buffered input and the partial pending frame (zero-padded) at the
    /// end of a recording. Without this, up to one chunk of tail audio — the
    /// speaker's last word — would be silently dropped.
    pub fn finish(&mut self, emit: &mut impl FnMut(&[f32])) {
        if self.resampler.is_some() && !self.in_buf.is_empty() {
            // Pad with zeros to reach chunk size
            self.in_buf.resize(self.chunk_in, 0.0);
            let result = self
                .resampler
                .as_mut()
                .expect("resampler checked above")
                .process(&[&self.in_buf[..]], None);
            if let Ok(out) = result {
                self.emit_frames(&out[0], emit);
            }
            // Drop the consumed input: a full in_buf would satisfy the
            // next push()'s chunk check immediately, re-processing this
            // padded tail into the following recording.
            self.in_buf.clear();
        }

        // Emit any remaining pending frame (padded with zeros)
        if !self.pending.is_empty() {
            self.pending.resize(self.frame_samples, 0.0);
            emit(&self.pending);
            self.pending.clear();
        }
    }

    /// Clear all internal buffers so the next `push()` starts from a clean state.
    ///
    /// Call this between recordings to prevent stale audio from the previous
    /// session leaking into the start of the next one via the FFT overlap buffers.
    pub fn reset(&mut self) {
        self.in_buf.clear();
        self.pending.clear();
        if let Some(ref mut resampler) = self.resampler {
            resampler.reset();
        }
    }

    fn emit_frames(&mut self, mut data: &[f32], emit: &mut impl FnMut(&[f32])) {
        while !data.is_empty() {
            let space = self.frame_samples - self.pending.len();
            let take = space.min(data.len());
            self.pending.extend_from_slice(&data[..take]);
            data = &data[take..];

            if self.pending.len() == self.frame_samples {
                emit(&self.pending);
                self.pending.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(sample_rate: usize, freq: f64, duration_secs: f64) -> Vec<f32> {
        let n = (sample_rate as f64 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin() as f32
            })
            .collect()
    }

    fn collect_output(resampler: &mut AudioResampler, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        resampler.push(input, &mut |frame| out.extend_from_slice(frame));
        out
    }

    #[test]
    fn passthrough_rechunks_into_frames() {
        // 16k -> 16k: no rubato, but output must still arrive in 480-sample frames
        let mut r = AudioResampler::new(16000, 16000, Duration::from_millis(30));
        let mut frames = Vec::new();
        r.push(&vec![0.1f32; 1000], &mut |frame| frames.push(frame.len()));
        assert_eq!(frames, vec![480, 480], "expected two complete frames");
    }

    #[test]
    fn downsample_preserves_duration() {
        // 1 second at 48kHz should come out as ~1 second at 16kHz
        let mut r = AudioResampler::new(48000, 16000, Duration::from_millis(30));
        let input = sine_wave(48000, 1000.0, 1.0);
        let mut total = collect_output(&mut r, &input).len();
        r.finish(&mut |frame| total += frame.len());
        // Allow one frame of slack for chunk boundaries and zero padding
        assert!(
            (total as i64 - 16000).unsigned_abs() as usize <= 480,
            "expected ~16000 output samples, got {total}"
        );
    }

    #[test]
    fn finish_flushes_buffered_tail() {
        // A recording that ends mid-chunk leaves samples in in_buf; finish()
        // must flush them — this is the "last word cut off" protection.
        // (Uses a realistic recording length: the FFT resampler has startup
        // latency, so a lone sub-chunk input can't test the tail flush.)
        let mut r = AudioResampler::new(48000, 16000, Duration::from_millis(30));
        let input = sine_wave(48000, 1000.0, 1.0); // 1s + 900-sample tail
        let after_push =
            collect_output(&mut r, &input).len() + collect_output(&mut r, &[0.5f32; 900]).len();

        let mut flushed = 0usize;
        r.finish(&mut |frame| flushed += frame.len());
        assert!(
            flushed > 0,
            "finish() must flush the buffered tail (had {after_push} samples from push)"
        );
    }

    #[test]
    fn reset_clears_in_buf_and_pending() {
        let mut r = AudioResampler::new(48000, 16000, Duration::from_millis(30));

        // Push less than one chunk (1024 samples) to leave data in in_buf
        let _ = collect_output(&mut r, &[0.5f32; 500]);

        r.reset();

        // Now push silence — should get only silence out, no remnants of 0.5
        let out = collect_output(&mut r, &vec![0.0f32; 4096]);
        let max_abs = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 0.01,
            "after reset, silence input should produce near-silence output, got max_abs={max_abs}"
        );
    }

    #[test]
    fn reset_clears_fft_overlap_buffers() {
        let mut r = AudioResampler::new(48000, 16000, Duration::from_millis(30));

        // Loud sine through the resampler (recording 1)
        let sine = sine_wave(48000, 1000.0, 0.5);
        let _ = collect_output(&mut r, &sine);
        r.finish(&mut |_| {});

        r.reset();

        // Silence (recording 2) — the sine tail must not leak through
        let out = collect_output(&mut r, &vec![0.0f32; 4096]);
        let max_abs = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 0.01,
            "FFT overlap should not leak after reset; got max_abs={max_abs}"
        );
    }

    #[test]
    fn finish_does_not_leak_tail_into_next_session() {
        let mut rs = AudioResampler::new(48000, 16000, Duration::from_millis(30));

        // Leave a partial chunk buffered, then end the session.
        rs.push(&[0.5f32; 100], &mut |_| {});
        rs.finish(&mut |_| {});

        // One fresh chunk yields ~341 output samples — below one frame, so
        // nothing should be emitted yet. If finish() left its padded tail in
        // in_buf, that tail is re-processed first, the output crosses the
        // 480-sample frame boundary, and a stale frame is emitted here.
        let mut emitted = 0usize;
        rs.push(&[0.25f32; RESAMPLER_CHUNK_SIZE], &mut |frame| {
            emitted += frame.len()
        });
        assert_eq!(
            emitted, 0,
            "stale resampler tail from finish() leaked into the next session"
        );
    }

    #[test]
    fn reset_passthrough_mode_clears_pending() {
        let mut r = AudioResampler::new(16000, 16000, Duration::from_millis(30));

        // Push partial frame (less than 480 samples) to leave data in pending
        let _ = collect_output(&mut r, &[1.0f32; 200]);

        r.reset();

        let out = collect_output(&mut r, &vec![0.0f32; 960]);
        if !out.is_empty() {
            let max_abs = out.iter().take(480).map(|s| s.abs()).fold(0.0f32, f32::max);
            assert!(
                max_abs < 0.001,
                "passthrough mode: pending buffer should be cleared after reset, got max_abs={max_abs}"
            );
        }
    }
}
