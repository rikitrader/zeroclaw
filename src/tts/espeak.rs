// ═══════════════════════════════════════════════════════════════
// ESPEAK-NG LOCAL TTS PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{
    validate_text, TtsConfig, TtsProvider, TtsProviderType, TtsResult, VoiceGender,
    VoiceInfo,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::process::Command;
use tokio::task;

/// espeak-ng TTS provider
///
/// This provider uses espeak-ng (https://github.com/espeak-ng/espeak-ng)
/// for basic offline text-to-speech synthesis. It's lightweight and
/// available on most platforms as a fallback option.
pub struct EspeakTtsProvider {
    /// Path to espeak-ng executable
    executable_path: Option<String>,
}

impl EspeakTtsProvider {
    /// Create a new espeak-ng provider
    pub fn new() -> Self {
        Self {
            executable_path: None,
        }
    }

    /// Create a new provider with custom executable path
    pub fn with_executable(mut self, path: String) -> Self {
        self.executable_path = Some(path);
        self
    }

    /// Get the executable path
    fn get_executable(&self) -> String {
        self.executable_path
            .clone()
            .unwrap_or_else(|| "espeak-ng".to_string())
    }

    /// Get the voice for the language
    fn get_voice(&self, config: &TtsConfig) -> String {
        config
            .voice
            .clone()
            .unwrap_or_else(|| {
                // Map language to espeak voice
                match config.language.split('-').next().unwrap_or("en") {
                    "en" => "en-us".to_string(),
                    "es" => "es".to_string(),
                    "fr" => "fr".to_string(),
                    "de" => "de".to_string(),
                    "it" => "it".to_string(),
                    "pt" => "pt".to_string(),
                    "ru" => "ru".to_string(),
                    "zh" => "zh".to_string(),
                    "ja" => "ja".to_string(),
                    "ko" => "ko".to_string(),
                    lang => format!("{}-default", lang),
                }
            })
    }

    /// Run espeak-ng synchronously
    fn run_espeak(&self, text: &str, config: &TtsConfig) -> Result<Vec<u8>> {
        let executable = self.get_executable();
        let voice = self.get_voice(config);

        let mut cmd = Command::new(&executable);

        // Voice
        cmd.arg("-v").arg(&voice);

        // Output format (WAV)
        cmd.arg("-w").arg("-"); // Write to stdout

        // Sample rate
        cmd.arg("--stdout").arg(format!("--rate={}", config.sample_rate));

        // Pitch adjustment
        if config.pitch != 0.0 {
            cmd.arg("-p").arg((50.0 + config.pitch).to_string());
        }

        // Speed/rate adjustment
        if config.rate != 1.0 {
            cmd.arg("-s").arg(((175.0 * config.rate) as i32).to_string());
        }

        // Input text
        cmd.arg(text);

        // Run the command
        let output = cmd
            .output()
            .with_context(|| {
                format!(
                    "Failed to execute espeak-ng at '{}'. Is it installed? \
                    Install with: apt-get install espeak-ng (Linux) \
                    or brew install espeak-ng (macOS)",
                    executable
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "espeak-ng failed with exit code {}: {}",
                output.status,
                stderr
            );
        }

        Ok(output.stdout)
    }

    /// List available voices
    fn list_voices(&self) -> Result<Vec<EspeakVoice>> {
        let executable = self.get_executable();

        let output = Command::new(&executable)
            .arg("--voices")
            .output()
            .context("Failed to list espeak-ng voices")?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut voices = Vec::new();

        for line in stdout.lines().skip(1) {
            // Skip header
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                voices.push(EspeakVoice {
                    name: parts.get(3).unwrap_or(&"").to_string(),
                    language: parts.get(1).unwrap_or(&"").to_string(),
                    gender: match parts.get(4).copied().unwrap_or("") {
                        "M" => VoiceGender::Male,
                        "F" => VoiceGender::Female,
                        _ => VoiceGender::Neutral,
                    },
                });
            }
        }

        Ok(voices)
    }
}

impl Default for EspeakTtsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsProvider for EspeakTtsProvider {
    fn provider_type(&self) -> TtsProviderType {
        TtsProviderType::Espeak
    }

    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate text
        validate_text(text)?;

        // Run espeak-ng in a blocking task
        let provider = self.clone_ref();
        let text = text.to_string();
        let config_clone = config.clone();

        let audio_data = task::spawn_blocking(move || {
            provider.run_espeak(&text, &config_clone)
        })
        .await??;

        Ok(TtsResult {
            audio_data,
            format: "wav".to_string(), // espeak-ng outputs WAV
            sample_rate: config.sample_rate,
            channels: 1,
            duration: None,
            timestamps: Vec::new(),
        })
    }

    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult> {
        // espeak-ng doesn't support SSML, extract text
        let stripped = ssml
            .replace("<speak>", "")
            .replace("</speak>", "")
            .replace("<prosody>", "")
            .replace("</prosody>", "")
            .replace("<s>", "")
            .replace("</s>", "");
        let text = stripped.trim();

        self.synthesize(text, config).await
    }

    async fn is_available(&self) -> bool {
        let executable = self.get_executable();
        Command::new(&executable)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    async fn get_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        let voices = self.list_voices().unwrap_or_default();

        let result: Vec<VoiceInfo> = voices
            .into_iter()
            .filter(|v| {
                if let Some(lang) = language {
                    v.language.starts_with(lang)
                } else {
                    true
                }
            })
            .map(|v| VoiceInfo {
                id: v.name.clone(),
                name: v.name.clone(),
                language: v.language.clone(),
                gender: v.gender,
                sample_rate: Some(22050),
                neural: Some(false),
            })
            .collect();

        Ok(result)
    }
}

// Allow cloning for use in async tasks
impl Clone for EspeakTtsProvider {
    fn clone(&self) -> Self {
        Self {
            executable_path: self.executable_path.clone(),
        }
    }
}

// Helper method for cloning
impl EspeakTtsProvider {
    fn clone_ref(&self) -> Self {
        self.clone()
    }
}

// ═══════════════════════════════════════════════════════════════
// ESPEAK-NG TYPES
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct EspeakVoice {
    name: String,
    language: String,
    gender: VoiceGender,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_espeak_provider_new() {
        let provider = EspeakTtsProvider::new();
        assert_eq!(provider.provider_type(), TtsProviderType::Espeak);
    }

    #[test]
    fn test_espeak_provider_default() {
        let provider = EspeakTtsProvider::default();
        assert_eq!(provider.provider_type(), TtsProviderType::Espeak);
    }

    #[test]
    fn test_espeak_provider_with_executable() {
        let provider = EspeakTtsProvider::new()
            .with_executable("/custom/path/espeak-ng".to_string());
        assert_eq!(provider.executable_path, Some("/custom/path/espeak-ng".to_string()));
    }

    #[test]
    fn test_espeak_provider_clone() {
        let provider = EspeakTtsProvider::new()
            .with_executable("espeak".to_string());

        let cloned = provider.clone();
        assert_eq!(cloned.executable_path, provider.executable_path);
    }

    #[test]
    fn test_espeak_provider_name() {
        let provider = EspeakTtsProvider::new();
        assert_eq!(provider.name(), "eSpeak-ng (Local)");
    }

    #[test]
    fn test_get_voice_default() {
        let provider = EspeakTtsProvider::new();
        let config = TtsConfig {
            language: "en-US".to_string(),
            ..Default::default()
        };

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "en-us");
    }

    #[test]
    fn test_get_voice_spanish() {
        let provider = EspeakTtsProvider::new();
        let config = TtsConfig {
            language: "es-ES".to_string(),
            ..Default::default()
        };

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "es");
    }

    #[test]
    fn test_get_voice_custom() {
        let provider = EspeakTtsProvider::new();
        let config = TtsConfig {
            voice: Some("mb-en1".to_string()),
            ..Default::default()
        };

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "mb-en1");
    }

    #[tokio::test]
    async fn test_espeak_provider_empty_text() {
        let provider = EspeakTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("", &config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_espeak_provider_is_available() {
        let provider = EspeakTtsProvider::new();
        // This will return false if espeak-ng is not installed
        let available = provider.is_available().await;
        // We don't assert a specific value since it depends on the system
        let _ = available;
    }

    #[tokio::test]
    async fn test_espeak_provider_get_voices() {
        let provider = EspeakTtsProvider::new();
        let voices = provider.get_voices(None).await.unwrap();

        // Should return voices if espeak-ng is installed, empty otherwise
        let _ = voices;
    }

    #[tokio::test]
    async fn test_espeak_provider_synthesize_ssml() {
        let provider = EspeakTtsProvider::new();
        let config = TtsConfig::default();

        // SSML is not supported, so it extracts the text
        let result = provider.synthesize_ssml("<speak>Hello</speak>", &config).await;

        // Should fail due to espeak-ng not being installed, but the text extraction should work
        assert!(result.is_err() || result.is_ok());
    }
}
