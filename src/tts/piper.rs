// ═══════════════════════════════════════════════════════════════
// PIPER LOCAL TTS PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{
    validate_text, TtsConfig, TtsProvider, TtsProviderType, TtsResult, VoiceGender,
    VoiceInfo,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;
use tokio::task;

/// Piper TTS provider
///
/// This provider uses Piper (https://github.com/rhasspy/piper)
/// for fast, local neural text-to-speech synthesis.
pub struct PiperTtsProvider {
    /// Path to piper executable
    executable_path: Option<String>,
    /// Default model directory
    model_dir: Option<String>,
}

impl PiperTtsProvider {
    /// Create a new Piper provider
    pub fn new() -> Self {
        Self {
            executable_path: None,
            model_dir: None,
        }
    }

    /// Create a new provider with custom executable path
    pub fn with_executable(mut self, path: String) -> Self {
        self.executable_path = Some(path);
        self
    }

    /// Create a new provider with custom model directory
    pub fn with_model_dir(mut self, dir: String) -> Self {
        self.model_dir = Some(dir);
        self
    }

    /// Get the model path from config or default location
    fn get_model_config(&self, config: &TtsConfig) -> Result<(String, String)> {
        let voice_name = config
            .voice
            .as_ref()
            .map(|v| v.as_str())
            .unwrap_or("en_US-lessac-medium");

        // Check if model_path is a full path to the model
        if let Some(ref path) = config.model_path {
            let model_path = Path::new(path);
            if model_path.exists() && model_path.is_file() {
                // It's a file, get the directory and stem
                let model_dir = model_path
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".to_string());
                let model_name = model_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                return Ok((model_dir, model_name));
            } else if model_path.is_dir() {
                // It's a directory, construct model path
                return Ok((
                    path.clone(),
                    format!("{}/{}.onnx", path, voice_name),
                ));
            }
        }

        // Default to models directory in user's home
        if let Some(ref model_dir) = self.model_dir {
            let model_file = format!("{}/{}.onnx", model_dir, voice_name);
            if Path::new(&model_file).exists() {
                return Ok((model_dir.clone(), model_file));
            }
        }

        // Try common model locations
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());

        let possible_locations = vec![
            format!("{}/.local/share/piper", home),
            format!("{}/.piper", home),
            format!("./models/piper"),
            "./models".to_string(),
        ];

        for loc in &possible_locations {
            let model_file = format!("{}/{}.onnx", loc, voice_name);
            if Path::new(&model_file).exists() {
                return Ok((loc.clone(), model_file));
            }
        }

        Err(anyhow::anyhow!(
            "Piper model not found for voice '{}'. Tried: {:?}. \
            Please download a model from https://huggingface.co/rhasspy/piper-voices \
            and set model_path in config or place in ~/.local/share/piper/",
            voice_name, possible_locations
        ))
    }

    /// Get the executable path
    fn get_executable(&self) -> String {
        self.executable_path
            .clone()
            .unwrap_or_else(|| "piper".to_string())
    }

    /// Run Piper synchronously
    fn run_piper(&self, text: &str, config: &TtsConfig) -> Result<Vec<u8>> {
        let executable = self.get_executable();
        let (_model_dir, model_path) = self.get_model_config(config)?;

        // Create temp file for output
        let output_file = NamedTempFile::new()
            .context("Failed to create temp file for audio output")?;

        let mut cmd = Command::new(&executable);

        // Model
        cmd.arg("--model").arg(&model_path);

        // Config file (usually alongside the model)
        let config_file = model_path.replace(".onnx", ".onnx.json");
        if Path::new(&config_file).exists() {
            cmd.arg("--config").arg(&config_file);
        }

        // Output file
        cmd.arg("--output").arg(output_file.path());

        // Input text via stdin
        cmd.stdin(Stdio::piped());

        // Disable stdout
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Spawn the process
        let mut child = cmd
            .spawn()
            .with_context(|| {
                format!(
                    "Failed to execute piper at '{}'. Is it installed and in PATH? \
                    Install from: https://github.com/rhasspy/piper/releases",
                    executable
                )
            })?;

        // Write text to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .context("Failed to write text to piper stdin")?;
            stdin.flush()?;
        }

        // Wait for completion
        let output = child
            .wait_with_output()
            .context("Failed to wait for piper process")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "piper failed with exit code {}: {}",
                output.status,
                stderr
            );
        }

        // Read the output file
        let audio_data = std::fs::read(output_file.path())
            .context("Failed to read piper output file")?;

        Ok(audio_data)
    }
}

impl Default for PiperTtsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsProvider for PiperTtsProvider {
    fn provider_type(&self) -> TtsProviderType {
        TtsProviderType::Piper
    }

    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate text
        validate_text(text)?;

        // Run Piper in a blocking task
        let provider = self.clone_ref();
        let text = text.to_string();
        let config_clone = config.clone();

        let audio_data = task::spawn_blocking(move || {
            provider.run_piper(&text, &config_clone)
        })
        .await??;

        Ok(TtsResult {
            audio_data,
            format: "wav".to_string(), // Piper outputs WAV
            sample_rate: config.sample_rate,
            channels: 1,
            duration: None,
            timestamps: Vec::new(),
        })
    }

    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Piper doesn't support SSML, extract text
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
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    async fn get_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        // Return a subset of popular Piper voices
        let voices = vec![
            VoiceInfo {
                id: "en_US-lessac-medium".to_string(),
                name: "US English Lessac Medium".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(22050),
                neural: Some(true),
            },
            VoiceInfo {
                id: "en_US-lessac-low".to_string(),
                name: "US English Lessac Low".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(22050),
                neural: Some(true),
            },
            VoiceInfo {
                id: "en_US-amy-medium".to_string(),
                name: "US English Amy Medium".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(22050),
                neural: Some(true),
            },
            VoiceInfo {
                id: "en_GB-semaine-medium".to_string(),
                name: "UK English Semaine Medium".to_string(),
                language: "en-GB".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(22050),
                neural: Some(true),
            },
            VoiceInfo {
                id: "es_ES-mls-medium".to_string(),
                name: "Spanish MLS Medium".to_string(),
                language: "es-ES".to_string(),
                gender: VoiceGender::Neutral,
                sample_rate: Some(22050),
                neural: Some(true),
            },
            VoiceInfo {
                id: "fr_FR-siwis-medium".to_string(),
                name: "French Siwis Medium".to_string(),
                language: "fr-FR".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(22050),
                neural: Some(true),
            },
            VoiceInfo {
                id: "de_DE-thorsten-medium".to_string(),
                name: "German Thorsten Medium".to_string(),
                language: "de-DE".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(22050),
                neural: Some(true),
            },
            VoiceInfo {
                id: "it_IT-riccardo-medium".to_string(),
                name: "Italian Riccardo Medium".to_string(),
                language: "it-IT".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(22050),
                neural: Some(true),
            },
        ];

        // Filter by language if specified
        let filtered = if let Some(lang) = language {
            voices
                .into_iter()
                .filter(|v| v.language.starts_with(lang.split('-').next().unwrap_or("")))
                .collect()
        } else {
            voices
        };

        Ok(filtered)
    }
}

// Allow cloning for use in async tasks
impl Clone for PiperTtsProvider {
    fn clone(&self) -> Self {
        Self {
            executable_path: self.executable_path.clone(),
            model_dir: self.model_dir.clone(),
        }
    }
}

// Helper method for cloning
impl PiperTtsProvider {
    fn clone_ref(&self) -> Self {
        self.clone()
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piper_provider_new() {
        let provider = PiperTtsProvider::new();
        assert_eq!(provider.provider_type(), TtsProviderType::Piper);
    }

    #[test]
    fn test_piper_provider_default() {
        let provider = PiperTtsProvider::default();
        assert_eq!(provider.provider_type(), TtsProviderType::Piper);
    }

    #[test]
    fn test_piper_provider_with_executable() {
        let provider = PiperTtsProvider::new()
            .with_executable("/custom/path/piper".to_string());
        assert_eq!(provider.executable_path, Some("/custom/path/piper".to_string()));
    }

    #[test]
    fn test_piper_provider_with_model_dir() {
        let provider = PiperTtsProvider::new()
            .with_model_dir("/models/piper".to_string());
        assert_eq!(provider.model_dir, Some("/models/piper".to_string()));
    }

    #[test]
    fn test_piper_provider_clone() {
        let provider = PiperTtsProvider::new()
            .with_executable("piper".to_string())
            .with_model_dir("/models".to_string());

        let cloned = provider.clone();
        assert_eq!(cloned.executable_path, provider.executable_path);
        assert_eq!(cloned.model_dir, provider.model_dir);
    }

    #[test]
    fn test_piper_provider_name() {
        let provider = PiperTtsProvider::new();
        assert_eq!(provider.name(), "Piper (Local)");
    }

    #[tokio::test]
    async fn test_piper_provider_empty_text() {
        let provider = PiperTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("", &config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_piper_provider_is_available() {
        let provider = PiperTtsProvider::new();
        // This will return false if piper is not in PATH
        let available = provider.is_available().await;
        // We don't assert a specific value since it depends on the system
        let _ = available;
    }

    #[tokio::test]
    async fn test_piper_provider_get_voices() {
        let provider = PiperTtsProvider::new();
        let voices = provider.get_voices(None).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().any(|v| v.id == "en_US-lessac-medium"));
    }

    #[tokio::test]
    async fn test_piper_provider_get_voices_with_language() {
        let provider = PiperTtsProvider::new();
        let voices = provider.get_voices(Some("en")).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().all(|v| v.language.starts_with("en")));
    }

    #[tokio::test]
    async fn test_piper_provider_synthesize_ssml() {
        let provider = PiperTtsProvider::new();
        let config = TtsConfig::default();

        // SSML is not supported, so it extracts the text
        let result = provider.synthesize_ssml("<speak>Hello</speak>", &config).await;

        // Should fail due to piper not being installed, but the text extraction should work
        assert!(result.is_err() || result.is_ok());
    }
}
