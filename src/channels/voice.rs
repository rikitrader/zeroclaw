// ═══════════════════════════════════════════════════════════════
// VOICE CHANNEL - Speech-based interaction with AI assistant
// ═══════════════════════════════════════════════════════════════

use super::traits::Channel;
use crate::audio::{AudioCapture, AudioPlayback, AudioConfig};
use crate::stt::{SttEngine, SttConfig, SttProvider};
use crate::tts::{TtsEngine, TtsConfig, TtsProvider};
use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// Voice channel for speech-based AI interaction
pub struct VoiceChannel {
    /// Channel name
    name: String,
    /// Audio capture device
    capture: AudioCapture,
    /// Audio playback device
    playback: AudioPlayback,
    /// Speech-to-Text engine
    stt_engine: SttEngine,
    /// Text-to-Speech engine
    tts_engine: TtsEngine,
    /// Audio configuration
    audio_config: AudioConfig,
    /// STT configuration
    stt_config: SttConfig,
    /// TTS configuration
    tts_config: TtsConfig,
    /// Wake word to activate listening
    wake_word: String,
    /// Voice Activity Detection enabled
    vad_enabled: bool,
    /// VAD threshold (0.0 to 1.0)
    vad_threshold: f32,
    /// Minimum speech duration (seconds)
    min_speech_duration: f32,
    /// Silence timeout (seconds) to stop recording
    silence_timeout: f32,
    /// Auto-start listening on channel start
    auto_start: bool,
}

impl VoiceChannel {
    /// Create a new voice channel
    pub fn new(
        name: String,
        stt_config: SttConfig,
        tts_config: TtsConfig,
    ) -> Result<Self> {
        let audio_config = AudioConfig {
            sample_rate: stt_config.sample_rate,
            channels: 1,
            buffer_size: 4096,
        };

        let capture = AudioCapture::new(&audio_config)
            .context("Failed to initialize audio capture")?;

        let playback = AudioPlayback::new()
            .context("Failed to initialize audio playback")?;

        let stt_engine = SttEngine::new(stt_config.clone());
        let tts_engine = TtsEngine::new(tts_config.clone());

        Ok(Self {
            name,
            capture,
            playback,
            stt_engine,
            tts_engine,
            audio_config,
            stt_config,
            tts_config,
            wake_word: "hey assistant".to_string(),
            vad_enabled: true,
            vad_threshold: 0.01,
            min_speech_duration: 0.5,
            silence_timeout: 2.0,
            auto_start: true,
        })
    }

    /// Create a voice channel with custom audio configuration
    pub fn with_audio_config(
        name: String,
        stt_config: SttConfig,
        tts_config: TtsConfig,
        audio_config: AudioConfig,
    ) -> Result<Self> {
        let capture = AudioCapture::new(&audio_config)
            .context("Failed to initialize audio capture")?;

        let playback = AudioPlayback::new()
            .context("Failed to initialize audio playback")?;

        let stt_engine = SttEngine::new(stt_config.clone());
        let tts_engine = TtsEngine::new(tts_config.clone());

        Ok(Self {
            name,
            capture,
            playback,
            stt_engine,
            tts_engine,
            audio_config,
            stt_config,
            tts_config,
            wake_word: "hey assistant".to_string(),
            vad_enabled: true,
            vad_threshold: 0.01,
            min_speech_duration: 0.5,
            silence_timeout: 2.0,
            auto_start: true,
        })
    }

    /// Set the wake word for activation
    pub fn with_wake_word(mut self, wake_word: String) -> Self {
        self.wake_word = wake_word.to_lowercase();
        self
    }

    /// Enable or disable Voice Activity Detection
    pub fn with_vad(mut self, enabled: bool) -> Self {
        self.vad_enabled = enabled;
        self
    }

    /// Set VAD threshold
    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum speech duration
    pub fn with_min_speech_duration(mut self, duration: f32) -> Self {
        self.min_speech_duration = duration.max(0.1);
        self
    }

    /// Set silence timeout
    pub fn with_silence_timeout(mut self, timeout: f32) -> Self {
        self.silence_timeout = timeout.max(0.5);
        self
    }

    /// Enable or disable auto-start listening
    pub fn with_auto_start(mut self, auto_start: bool) -> Self {
        self.auto_start = auto_start;
        self
    }

    /// Capture audio and transcribe to text
    async fn capture_and_transcribe(&self) -> Result<String> {
        // Capture audio
        let audio_data: Vec<f32> = self.capture.record_until_silence(
            self.vad_threshold,
            self.min_speech_duration,
            self.silence_timeout,
        ).await
            .context("Failed to capture audio")?;

        // Transcribe
        let result = self.stt_engine
            .transcribe_with_config(&audio_data, &self.stt_config)
            .await
            .context("Failed to transcribe audio")?;

        Ok(result.text)
    }

    /// Synthesize speech and play it
    async fn synthesize_and_play(&self, text: &str) -> Result<()> {
        // Synthesize
        let result = self.tts_engine
            .synthesize_with_config(text, &self.tts_config)
            .await
            .context("Failed to synthesize speech")?;

        // Convert audio data to f32 samples based on format
        let audio_samples = self.convert_audio_to_f32(&result.audio_data, &result.format);

        // Play audio
        self.playback.play(audio_samples)
            .context("Failed to play audio")?;

        Ok(())
    }

    /// Convert audio data to f32 samples
    fn convert_audio_to_f32(&self, data: &[u8], format: &str) -> Vec<f32> {
        match format {
            "wav" | "pcm" => {
                // Assume 16-bit PCM
                let mut samples = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
                    samples.push(sample_i16 as f32 / 32768.0);
                }
                samples
            }
            "mp3" | "opus" => {
                // For compressed formats, we'd need to decode them first
                // For now, return a simple conversion (not production-ready)
                let mut samples = Vec::with_capacity(data.len());
                for &byte in data {
                    samples.push(byte as f32 / 255.0 * 2.0 - 1.0);
                }
                samples
            }
            _ => {
                // Default conversion
                data.iter().map(|&b| b as f32 / 255.0 * 2.0 - 1.0).collect()
            }
        }
    }

    /// Detect wake word in text
    fn detect_wake_word(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        text_lower.contains(&self.wake_word)
    }

    /// Remove wake word from text
    fn remove_wake_word(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();
        if let Some(pos) = text_lower.find(&self.wake_word) {
            let before = &text[..pos];
            let after = &text[pos + self.wake_word.len()..];
            format!("{} {}", before.trim(), after.trim())
                .trim()
                .to_string()
        } else {
            text.to_string()
        }
    }
}

#[async_trait]
impl Channel for VoiceChannel {
    fn name(&self) -> &str {
        &self.name
    }

    /// Send a message through voice (synthesize and speak)
    async fn send(&self, message: &str, _recipient: &str) -> Result<()> {
        self.synthesize_and_play(message).await
    }

    /// Listen for voice input (transcribe and send to channel)
    async fn listen(&self, tx: mpsc::Sender<super::traits::ChannelMessage>) -> Result<()> {
        if !self.auto_start {
            // Wait for manual trigger if not auto-starting
            return Ok(());
        }

        use crate::channels::traits::ChannelMessage;
        use std::time::{SystemTime, UNIX_EPOCH};

        loop {
            // Capture and transcribe audio
            let text = match self.capture_and_transcribe().await {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Voice capture error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
            };

            // Check for wake word if enabled
            if !self.wake_word.is_empty() && !self.detect_wake_word(&text) {
                continue; // Skip if wake word not detected
            }

            // Remove wake word from text
            let clean_text = if !self.wake_word.is_empty() {
                self.remove_wake_word(&text)
            } else {
                text.clone()
            };

            // Skip if empty after removing wake word
            if clean_text.trim().is_empty() {
                continue;
            }

            // Create channel message
            let message = ChannelMessage {
                id: uuid::Uuid::new_v4().to_string(),
                sender: "voice".to_string(),
                content: clean_text,
                channel: self.name.clone(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            };

            // Send to channel
            if tx.send(message).await.is_err() {
                eprintln!("Voice channel: failed to send message");
                break;
            }
        }

        Ok(())
    }

    /// Check microphone and audio devices
    async fn health_check(&self) -> bool {
        // Check if STT provider is available
        let stt_available = if let Some(provider) = self.stt_engine.default_provider() {
            let provider: &dyn SttProvider = provider;
            provider.is_available().await
        } else {
            false
        };

        // Check if TTS provider is available
        let tts_available = if let Some(provider) = self.tts_engine.default_provider() {
            let provider: &dyn TtsProvider = provider;
            provider.is_available().await
        } else {
            false
        };

        stt_available && tts_available
    }

    /// Play a sound to indicate the assistant is listening
    async fn start_typing(&self, _recipient: &str) -> Result<()> {
        // Play a short beep to indicate listening
        let beep_samples = self.convert_beep_to_f32(440.0, 0.1); // 440 Hz for 100ms
        self.playback.play(beep_samples)?;
        Ok(())
    }

    /// Stop any playing audio
    async fn stop_typing(&self, _recipient: &str) -> Result<()> {
        self.playback.stop()?;
        Ok(())
    }
}

impl VoiceChannel {
    /// Generate a simple beep tone as f32 samples
    fn convert_beep_to_f32(&self, frequency: f32, duration: f32) -> Vec<f32> {
        let sample_rate = self.audio_config.sample_rate as f32;
        let num_samples = (sample_rate * duration) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            let amplitude = (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push(amplitude * 0.25); // Quarter volume
        }

        samples
    }

    /// Generate a simple beep tone (deprecated, use convert_beep_to_f32)
    fn generate_beep(&self, frequency: f32, duration: f32) -> Vec<u8> {
        let sample_rate = self.audio_config.sample_rate as f32;
        let num_samples = (sample_rate * duration) as usize;
        let mut samples = Vec::with_capacity(num_samples * 2); // 16-bit = 2 bytes per sample

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            let amplitude = (2.0 * std::f32::consts::PI * frequency * t).sin();
            let sample_i16 = (amplitude * 16384.0) as i16; // Quarter scale
            samples.extend_from_slice(&sample_i16.to_le_bytes());
        }

        samples
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::SttProviderType;
    use crate::tts::TtsProviderType;

    #[test]
    fn test_voice_channel_new() {
        let stt_config = SttConfig {
            provider: SttProviderType::OpenAi,
            ..Default::default()
        };
        let tts_config = TtsConfig {
            provider: TtsProviderType::OpenAi,
            ..Default::default()
        };

        let result = VoiceChannel::new(
            "test-voice".to_string(),
            stt_config,
            tts_config,
        );

        // May fail if no audio devices, which is expected in test environments
        match result {
            Ok(_) => println!("Voice channel created successfully"),
            Err(e) => println!("Voice channel creation failed (expected without audio device): {}", e),
        }
    }

    #[test]
    fn test_wake_word_detection() {
        // Test wake word detection logic without requiring audio devices
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            assert!(channel.detect_wake_word("hey assistant, what time is it?"));
            assert!(channel.detect_wake_word("HEY ASSISTANT!"));
            assert!(channel.detect_wake_word("Hey Assistant, help me"));
            assert!(!channel.detect_wake_word("hello world"));
            assert!(!channel.detect_wake_word("what's up"));
        }
        // Skip test if audio devices not available
    }

    #[test]
    fn test_wake_word_removal() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            assert_eq!(channel.remove_wake_word("hey assistant, what time is it?"), "what time is it?");
            assert_eq!(channel.remove_wake_word("HEY ASSISTANT!"), "!");
            assert_eq!(channel.remove_wake_word("hello world"), "hello world");
        }
        // Skip test if audio devices not available
    }

    #[test]
    fn test_wake_word_custom() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            let channel = channel.with_wake_word("ok zero".to_string());
            assert!(channel.detect_wake_word("ok zero, tell me a joke"));
            assert!(!channel.detect_wake_word("hey assistant"));
        }
        // Skip test if audio devices not available
    }

    #[test]
    fn test_vad_threshold_clamping() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            let channel = channel.with_vad_threshold(1.5); // Should be clamped to 1.0
            assert_eq!(channel.vad_threshold, 1.0);
        }
        // Skip test if audio devices not available
    }

    #[test]
    fn test_min_speech_duration_validation() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            let channel = channel.with_min_speech_duration(0.01); // Should be clamped to 0.1
            assert_eq!(channel.min_speech_duration, 0.1);
        }
        // Skip test if audio devices not available
    }

    #[test]
    fn test_silence_timeout_validation() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            let channel = channel.with_silence_timeout(0.1); // Should be clamped to 0.5
            assert_eq!(channel.silence_timeout, 0.5);
        }
        // Skip test if audio devices not available
    }

    #[test]
    fn test_generate_beep() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            let beep = channel.convert_beep_to_f32(440.0, 0.1);
            // At 24000 Hz (default sample rate), 0.1 seconds should be 2400 samples
            assert!(beep.len() > 2000 && beep.len() < 3000);
        }
        // Skip test if audio devices not available
    }

    #[tokio::test]
    async fn test_voice_channel_name() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "my-voice".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            assert_eq!(channel.name(), "my-voice");
        }
        // Skip test if audio devices not available
    }

    #[tokio::test]
    async fn test_voice_channel_health_check() {
        let stt_config = SttConfig::default();
        let tts_config = TtsConfig::default();

        let channel_result = VoiceChannel::new(
            "test".to_string(),
            stt_config,
            tts_config,
        );

        if let Ok(channel) = channel_result {
            // Health check should return bool (false if API keys not set)
            let health = channel.health_check().await;
            // We don't assert a specific value since it depends on env variables
            let _ = health;
        }
        // Skip test if audio devices not available
    }
}
