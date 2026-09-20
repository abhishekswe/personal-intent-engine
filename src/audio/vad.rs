use anyhow::Result;

/// Trait for Voice Activity Detection engines.
pub trait VoiceActivityDetector: Send + Sync {
    /// Push a frame and get back whether it's speech or noise.
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>>;

    /// Check if a frame is voice (without buffering)
    fn is_voice(&mut self, frame: &[f32]) -> Result<bool>;

    /// Reset internal state (called between recording sessions)
    fn reset(&mut self);

    /// Update hangover frames
    fn set_hangover_frames(&mut self, frames: usize);
}

/// Result of VAD classification
pub enum VadFrame<'a> {
    /// Frame contains speech (may include prefill buffer)
    Speech(&'a [f32]),
    /// Frame is noise/silence
    Noise,
}

/// How VAD should filter frames for a recording session
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadPolicy {
    /// Bypass VAD entirely
    Disabled,
    /// Standard offline profile
    Offline,
    /// Longer hangover for streaming models
    Streaming,
}

// VAD timing constants (in 30ms frames), tuned for 30ms frames at 16kHz.
pub const VAD_CONTEXT_FRAMES: usize = 15; // 450ms pre-speech context
pub const VAD_HANGOVER_FRAMES: usize = 15; // 450ms post-speech
pub const VAD_STREAM_HANGOVER_FRAMES: usize = 55; // 1.65s for streaming
pub const VAD_SPEECH_THRESHOLD_FRAMES: usize = 2; // 60ms onset detection

/// Smoothed VAD with onset detection, hangover tail, and prefill buffering.
///
/// Wraps a frame-level detector and:
/// - Buffers pre-speech frames for context
/// - Requires N consecutive voice frames before triggering speech
/// - Continues forwarding after speech ends (hangover tail)
/// - State machine: Silence -> Onset -> Speech -> Hangover -> Silence
pub struct VadPipeline {
    inner: Box<dyn VoiceActivityDetector>,
    prefill_frames: usize,
    hangover_frames: usize,
    onset_frames: usize,

    frame_buffer: std::collections::VecDeque<Vec<f32>>,
    hangover_counter: usize,
    onset_counter: usize,
    in_speech: bool,

    temp_out: Vec<f32>,
}

impl VadPipeline {
    /// Wrap an inner detector with onset, hangover, and prefill smoothing.
    #[must_use]
    pub fn new(
        inner: Box<dyn VoiceActivityDetector>,
        prefill_frames: usize,
        hangover_frames: usize,
        onset_frames: usize,
    ) -> Self {
        // The prefill ring holds at most `prefill_frames + 1` frames, and the
        // onset burst emits all of them at once — size both up front so steady
        // capture never reallocates.
        let ring_capacity = prefill_frames + 1;
        Self {
            inner,
            prefill_frames,
            hangover_frames,
            onset_frames,
            frame_buffer: std::collections::VecDeque::with_capacity(ring_capacity),
            hangover_counter: 0,
            onset_counter: 0,
            in_speech: false,
            temp_out: Vec::with_capacity(super::FRAME_SAMPLES * ring_capacity),
        }
    }
}

impl VoiceActivityDetector for VadPipeline {
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
        // Buffer every frame for pre-roll
        self.frame_buffer.push_back(frame.to_vec());
        while self.frame_buffer.len() > self.prefill_frames + 1 {
            self.frame_buffer.pop_front();
        }

        let is_voice = self.inner.is_voice(frame)?;

        match (self.in_speech, is_voice) {
            // Potential start of speech
            (false, true) => {
                self.onset_counter += 1;
                if self.onset_counter >= self.onset_frames {
                    self.in_speech = true;
                    self.hangover_counter = self.hangover_frames;
                    self.onset_counter = 0;

                    // Collect prefill + current
                    self.temp_out.clear();
                    for buf in &self.frame_buffer {
                        self.temp_out.extend(buf);
                    }
                    // Every buffered frame is now committed. Keeping it in the
                    // prefill ring would replay it if speech stops and resumes.
                    self.frame_buffer.clear();
                    Ok(VadFrame::Speech(&self.temp_out))
                } else {
                    Ok(VadFrame::Noise)
                }
            }
            // Ongoing speech
            (true, true) => {
                self.hangover_counter = self.hangover_frames;
                self.frame_buffer.clear();
                Ok(VadFrame::Speech(frame))
            }
            // End of speech (hangover)
            (true, false) => {
                if self.hangover_counter > 0 {
                    self.hangover_counter -= 1;
                    self.frame_buffer.clear();
                    Ok(VadFrame::Speech(frame))
                } else {
                    self.in_speech = false;
                    Ok(VadFrame::Noise)
                }
            }
            // Silence
            (false, false) => {
                self.onset_counter = 0;
                Ok(VadFrame::Noise)
            }
        }
    }

    fn is_voice(&mut self, frame: &[f32]) -> Result<bool> {
        self.inner.is_voice(frame)
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.frame_buffer.clear();
        self.hangover_counter = 0;
        self.onset_counter = 0;
        self.in_speech = false;
        self.temp_out.clear();
    }

    fn set_hangover_frames(&mut self, frames: usize) {
        self.hangover_frames = frames;
    }
}

/// Placeholder VAD that always returns speech (for testing without Silero)
pub struct PassthroughVad;

impl VoiceActivityDetector for PassthroughVad {
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
        Ok(VadFrame::Speech(frame))
    }

    fn is_voice(&mut self, _frame: &[f32]) -> Result<bool> {
        Ok(true)
    }

    fn reset(&mut self) {}

    fn set_hangover_frames(&mut self, _frames: usize) {}
}

/// Energy-based VAD (simple threshold, no ML model required)
pub struct EnergyVad {
    threshold: f32,
}

impl EnergyVad {
    /// Create an energy-threshold detector; frames with RMS above `threshold`
    /// are treated as speech.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
        if self.is_voice(frame)? {
            Ok(VadFrame::Speech(frame))
        } else {
            Ok(VadFrame::Noise)
        }
    }

    fn is_voice(&mut self, frame: &[f32]) -> Result<bool> {
        let energy: f32 = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
        Ok(energy.sqrt() > self.threshold)
    }

    fn reset(&mut self) {}

    fn set_hangover_frames(&mut self, _frames: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scripted inner VAD: returns a pre-programmed voice/silence sequence.
    struct ScriptedVad {
        script: Vec<bool>,
        pos: usize,
    }

    impl ScriptedVad {
        fn new(script: Vec<bool>) -> Self {
            Self { script, pos: 0 }
        }
    }

    impl VoiceActivityDetector for ScriptedVad {
        fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
            if self.is_voice(frame)? {
                Ok(VadFrame::Speech(frame))
            } else {
                Ok(VadFrame::Noise)
            }
        }

        fn is_voice(&mut self, _frame: &[f32]) -> Result<bool> {
            let v = self.script.get(self.pos).copied().unwrap_or(false);
            self.pos += 1;
            Ok(v)
        }

        fn reset(&mut self) {
            self.pos = 0;
        }

        fn set_hangover_frames(&mut self, _frames: usize) {}
    }

    const FRAME: [f32; 4] = [0.1, 0.2, 0.3, 0.4];

    fn smoothed(script: Vec<bool>, prefill: usize, hangover: usize, onset: usize) -> VadPipeline {
        VadPipeline::new(Box::new(ScriptedVad::new(script)), prefill, hangover, onset)
    }

    fn is_speech(vad: &mut VadPipeline) -> bool {
        matches!(vad.push_frame(&FRAME).unwrap(), VadFrame::Speech(_))
    }

    #[test]
    fn onset_requires_consecutive_voice_frames() {
        // onset=2: a single voiced frame must not trigger speech
        let mut vad = smoothed(vec![true, false, true, true], 0, 0, 2);
        assert!(!is_speech(&mut vad), "1st voiced frame: still onset");
        assert!(!is_speech(&mut vad), "silence breaks the onset run");
        assert!(!is_speech(&mut vad), "1st voiced frame again: still onset");
        assert!(is_speech(&mut vad), "2nd consecutive voiced frame triggers");
    }

    #[test]
    fn onset_trigger_emits_prefill_buffer() {
        // prefill=3, onset=1: the triggering frame must carry buffered pre-roll
        let mut vad = smoothed(vec![false, false, true], 3, 0, 1);
        let _ = vad.push_frame(&FRAME).unwrap();
        let _ = vad.push_frame(&FRAME).unwrap();
        match vad.push_frame(&FRAME).unwrap() {
            VadFrame::Speech(buf) => {
                // 2 buffered silence frames + the current frame
                assert_eq!(buf.len(), FRAME.len() * 3, "prefill context missing");
            }
            VadFrame::Noise => panic!("expected speech on onset trigger"),
        }
    }

    #[test]
    fn hangover_extends_speech_after_voice_ends() {
        // onset=1, hangover=2: after voice stops, 2 more frames stay Speech
        let mut vad = smoothed(vec![true, false, false, false], 0, 2, 1);
        assert!(is_speech(&mut vad), "voice triggers speech");
        assert!(is_speech(&mut vad), "hangover frame 1");
        assert!(is_speech(&mut vad), "hangover frame 2");
        assert!(!is_speech(&mut vad), "hangover exhausted -> noise");
    }

    #[test]
    fn speech_restart_does_not_reemit_prior_frames() {
        let mut vad = smoothed(vec![true, false, false, true], 3, 1, 1);
        let mut emitted = Vec::new();

        for id in 1..=4 {
            let frame = [id as f32; 4];
            if let VadFrame::Speech(samples) = vad.push_frame(&frame).unwrap() {
                emitted.extend(samples.iter().step_by(frame.len()).copied());
            }
        }

        assert_eq!(
            emitted,
            vec![1.0, 2.0, 3.0, 4.0],
            "every input frame must be emitted at most once across a speech boundary"
        );
    }

    #[test]
    fn continuous_speech_preserves_every_frame_including_repetition() {
        let frame_count = 100;
        let mut vad = smoothed(vec![true; frame_count], 3, 1, 1);
        let repeated_frame = [0.25; 4];
        let mut emitted_samples = 0;

        for _ in 0..frame_count {
            if let VadFrame::Speech(samples) = vad.push_frame(&repeated_frame).unwrap() {
                emitted_samples += samples.len();
            }
        }

        assert_eq!(emitted_samples, frame_count * repeated_frame.len());
    }

    #[test]
    fn reset_discards_uncommitted_prefill_from_previous_session() {
        let mut vad = smoothed(vec![false, true], 2, 0, 1);

        assert!(matches!(
            vad.push_frame(&[1.0; 4]).unwrap(),
            VadFrame::Noise
        ));
        vad.reset();

        assert!(matches!(
            vad.push_frame(&[2.0; 4]).unwrap(),
            VadFrame::Noise
        ));
        match vad.push_frame(&[3.0; 4]).unwrap() {
            VadFrame::Speech(samples) => {
                let frame_ids: Vec<f32> = samples.iter().step_by(4).copied().collect();
                assert_eq!(frame_ids, vec![2.0, 3.0]);
            }
            VadFrame::Noise => panic!("second-session speech should trigger"),
        }
    }

    #[test]
    fn silence_emits_no_audio() {
        let frame_count = 20;
        let mut vad = smoothed(vec![false; frame_count], 3, 1, 1);

        for _ in 0..frame_count {
            assert!(matches!(vad.push_frame(&FRAME).unwrap(), VadFrame::Noise));
        }
    }

    #[test]
    fn set_hangover_frames_changes_tail_length() {
        let mut vad = smoothed(vec![true, false, false], 0, 5, 1);
        vad.set_hangover_frames(1);
        assert!(is_speech(&mut vad), "voice triggers speech");
        assert!(is_speech(&mut vad), "hangover frame 1");
        assert!(!is_speech(&mut vad), "shortened hangover exhausted");
    }

    #[test]
    fn reset_clears_speech_state() {
        let mut vad = smoothed(vec![true, true], 0, 10, 1);
        assert!(is_speech(&mut vad), "voice triggers speech");
        vad.reset();
        // Inner ScriptedVad also resets to pos 0 (true), but in_speech and
        // counters must be back to the silence state: with onset=1 a voiced
        // frame re-triggers, which is correct — verify via internals instead.
        assert!(!vad.in_speech, "reset must clear in_speech");
        assert_eq!(vad.hangover_counter, 0, "reset must clear hangover");
        assert_eq!(vad.onset_counter, 0, "reset must clear onset counter");
        assert!(vad.frame_buffer.is_empty(), "reset must clear prefill");
        assert!(vad.temp_out.is_empty(), "reset must clear temp_out");
    }

    #[test]
    fn energy_vad_thresholds() {
        let mut vad = EnergyVad::new(0.05);
        let loud = [0.5f32; 480];
        let quiet = [0.001f32; 480];
        assert!(vad.is_voice(&loud).unwrap());
        assert!(!vad.is_voice(&quiet).unwrap());
    }
}
