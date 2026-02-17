// ═══════════════════════════════════════════════════════════════
// WHISPER.CPP LOCAL STT PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{audio_to_wav, SttConfig, SttProvider, SttProviderType, SttResult, SttWord};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::Path;
use std::process::Command;
use tokio::io::AsyncWriteExt;
use tokio::task;

/// Whisper.cpp local STT provider
///
/// This provider uses whisper.cpp (https://github.com/ggerganov/whisper.cpp)
/// for offline speech recognition. It requires whisper.cpp to be installed
/// and available in the PATH, or the path to be specified.
pub struct WhisperCppProvider {
    /// Path to whisper.cpp executable
    executable_path: Option<String>,
    /// Default model directory
    model_dir: Option<String>,
}

impl WhisperCppProvider {
    /// Create a new Whisper.cpp provider
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
    fn get_model_path(&self, config: &SttConfig) -> Result<String> {
        if let Some(ref path) = config.model_path {
            return Ok(path.clone());
        }

        // Default to models directory in user's home
        let model_name = config
            .model
            .as_deref()
            .unwrap_or("ggml-base.en.bin");

        if let Some(ref model_dir) = self.model_dir {
            Ok(format!("{}/{}", model_dir, model_name))
        } else {
            // Try common model locations
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".to_string());

            let possible_locations = vec![
                format!("{}/.local/share/whisper/{}", home, model_name),
                format!("{}/.whisper/{}", home, model_name),
                format!("./models/{}", model_name),
                model_name.to_string(),
            ];

            let possible_locations_for_error = possible_locations.clone();

            possible_locations
                .into_iter()
                .find(|loc| Path::new(loc).exists())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Whisper model not found. Tried: {:?}. \
                        Please download a model from https://huggingface.co/ggerganov/whisper.cpp \
                        and set model_path in config or place in ~/.local/share/whisper/",
                        possible_locations_for_error
                    )
                })
        }
    }

    /// Get the executable path
    fn get_executable(&self) -> String {
        self.executable_path
            .clone()
            .unwrap_or_else(|| "whisper-cli".to_string())
    }

    /// Get language code for whisper.cpp
    fn get_language_code(&self, config: &SttConfig) -> String {
        // whisper.cpp uses 2-letter language codes
        config
            .language
            .split('-')
            .next()
            .unwrap_or("en")
            .to_string()
    }

    /// Run whisper.cpp synchronously
    fn run_whisper_cpp(
        &self,
        audio_path: &Path,
        config: &SttConfig,
    ) -> Result<WhisperCppOutput> {
        let executable = self.get_executable();
        let model_path = self.get_model_path(config)?;
        let language = self.get_language_code(config);

        let mut cmd = Command::new(&executable);

        // Model
        cmd.arg("-m").arg(&model_path);

        // Audio file
        cmd.arg("-f").arg(audio_path);

        // Language
        cmd.arg("-l").arg(&language);

        // Output format (JSON)
        cmd.arg("-oj");

        // Threads
        if let Some(threads) = config.num_threads {
            cmd.arg("-t").arg(threads.to_string());
        }

        // Use GPU if requested
        if config.use_gpu {
            // Try different GPU backends based on platform
            #[cfg(target_os = "macos")]
            {
                cmd.arg("--gpu");
                cmd.arg("--device").arg("metal");
            }
        }

        // Quantization
        if let Some(ref quant) = config.quantization {
            cmd.arg("--model-type").arg(quant);
        }

        // Beam size
        if let Some(beam) = config.beam_size {
            cmd.arg("-bs").arg(beam.to_string());
        }

        // Punctuation
        if config.enable_punctuation {
            cmd.arg("-pd"); // Punctuation detection (if supported)
        }

        // Output to stdout for capture
        cmd.arg("-osrt"); // Output style: JSON with timestamps

        // Run the command
        let output = cmd
            .output()
            .context(format!("Failed to execute whisper.cpp at '{}'. Is it installed and in PATH?", executable))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "whisper.cpp failed with exit code {}: {}",
                output.status,
                stderr
            );
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: WhisperCppOutput = serde_json::from_str(&stdout)
            .context("Failed to parse whisper.cpp JSON output")?;

        Ok(result)
    }
}

impl Default for WhisperCppProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SttProvider for WhisperCppProvider {
    fn provider_type(&self) -> SttProviderType {
        SttProviderType::WhisperCpp
    }

    async fn transcribe(&self, audio_data: &[f32], config: &SttConfig) -> Result<SttResult> {
        // Validate audio data
        if audio_data.is_empty() {
            anyhow::bail!("Audio data is empty");
        }

        // Create temporary WAV file
        let wav_data = audio_to_wav(audio_data, config.sample_rate);

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!(
            "whisper_{}.wav",
            uuid::Uuid::new_v4()
        ));

        // Write audio data to temp file
        let mut file = tokio::fs::File::create(&temp_file).await?;
        file.write_all(&wav_data).await?;
        file.flush().await?;

        // Run whisper.cpp in a blocking task
        let temp_file_clone = temp_file.clone();
        let provider = self.clone_ref();
        let config_clone = config.clone();

        let output = task::spawn_blocking(move || {
            provider.run_whisper_cpp(&temp_file_clone, &config_clone)
        })
        .await??;

        // Clean up temp file
        let _ = tokio::fs::remove_file(&temp_file).await;

        // Convert output to SttResult
        let text = output
            .segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        let words = output
            .words
            .into_iter()
            .map(|w| SttWord {
                word: w.word,
                start_time: w.start,
                end_time: w.end,
                confidence: w.probability,
            })
            .collect();

        Ok(SttResult {
            text: text.trim().to_string(),
            confidence: output.complexity.unwrap_or(1.0), // Use complexity as proxy
            language: Some(config.language.clone()),
            alternatives: Vec::new(), // whisper.cpp doesn't provide alternatives
            words,
        })
    }

    async fn transcribe_file(
        &self,
        file_path: &Path,
        config: &SttConfig,
    ) -> Result<SttResult> {
        if !file_path.exists() {
            anyhow::bail!("Audio file does not exist: {}", file_path.display());
        }

        // Run whisper.cpp in a blocking task
        let provider = self.clone_ref();
        let file_path = file_path.to_path_buf();
        let config_clone = config.clone();

        let output = task::spawn_blocking(move || {
            provider.run_whisper_cpp(&file_path, &config_clone)
        })
        .await??;

        // Convert output to SttResult
        let text = output
            .segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        let words = output
            .words
            .into_iter()
            .map(|w| SttWord {
                word: w.word,
                start_time: w.start,
                end_time: w.end,
                confidence: w.probability,
            })
            .collect();

        Ok(SttResult {
            text: text.trim().to_string(),
            confidence: output.complexity.unwrap_or(1.0),
            language: Some(config.language.clone()),
            alternatives: Vec::new(),
            words,
        })
    }

    async fn is_available(&self) -> bool {
        let executable = self.get_executable();
        Command::new(&executable)
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

// Allow cloning for use in async tasks
impl Clone for WhisperCppProvider {
    fn clone(&self) -> Self {
        Self {
            executable_path: self.executable_path.clone(),
            model_dir: self.model_dir.clone(),
        }
    }
}

// Helper method for cloning
impl WhisperCppProvider {
    fn clone_ref(&self) -> Self {
        self.clone()
    }
}

// ═══════════════════════════════════════════════════════════════
// WHISPER.CPP OUTPUT TYPES
// ═══════════════════════════════════════════════════════════════

/// Whisper.cpp JSON output
#[derive(Debug, Clone, serde::Deserialize)]
struct WhisperCppOutput {
    /// Transcription segments
    #[serde(default)]
    segments: Vec<WhisperSegment>,
    /// Word-level timestamps
    #[serde(default)]
    words: Vec<WhisperWord>,
    /// Transcription complexity (used as confidence proxy)
    complexity: Option<f32>,
}

/// Whisper segment
#[derive(Debug, Clone, serde::Deserialize)]
struct WhisperSegment {
    /// Segment text
    text: String,
    /// Start time in seconds
    #[serde(default)]
    start: f32,
    /// End time in seconds
    #[serde(default)]
    end: f32,
}

/// Whisper word
#[derive(Debug, Clone, serde::Deserialize)]
struct WhisperWord {
    /// Word text
    word: String,
    /// Start time in seconds
    start: f32,
    /// End time in seconds
    end: f32,
    /// Probability/confidence
    probability: f32,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_provider_new() {
        let provider = WhisperCppProvider::new();
        assert_eq!(provider.provider_type(), SttProviderType::WhisperCpp);
    }

    #[test]
    fn test_whisper_provider_default() {
        let provider = WhisperCppProvider::default();
        assert_eq!(provider.provider_type(), SttProviderType::WhisperCpp);
    }

    #[test]
    fn test_whisper_provider_with_executable() {
        let provider = WhisperCppProvider::new()
            .with_executable("/custom/path/whisper-cli".to_string());
        assert_eq!(provider.executable_path, Some("/custom/path/whisper-cli".to_string()));
    }

    #[test]
    fn test_whisper_provider_with_model_dir() {
        let provider = WhisperCppProvider::new()
            .with_model_dir("/models/whisper".to_string());
        assert_eq!(provider.model_dir, Some("/models/whisper".to_string()));
    }

    #[test]
    fn test_whisper_provider_clone() {
        let provider = WhisperCppProvider::new()
            .with_executable("whisper".to_string())
            .with_model_dir("/models".to_string());

        let cloned = provider.clone();
        assert_eq!(cloned.executable_path, provider.executable_path);
        assert_eq!(cloned.model_dir, provider.model_dir);
    }

    #[test]
    fn test_get_language_code() {
        let provider = WhisperCppProvider::new();

        let config = SttConfig {
            language: "en-US".to_string(),
            ..Default::default()
        };
        assert_eq!(provider.get_language_code(&config), "en");

        let config = SttConfig {
            language: "es-ES".to_string(),
            ..Default::default()
        };
        assert_eq!(provider.get_language_code(&config), "es");

        let config = SttConfig {
            language: "zh-CN".to_string(),
            ..Default::default()
        };
        assert_eq!(provider.get_language_code(&config), "zh");
    }

    #[test]
    fn test_whisper_provider_name() {
        let provider = WhisperCppProvider::new();
        assert_eq!(provider.name(), "Whisper.cpp (Local)");
    }

    #[tokio::test]
    async fn test_whisper_provider_empty_audio() {
        let provider = WhisperCppProvider::new();
        let config = SttConfig::default();

        let result = provider.transcribe(&[], &config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_whisper_provider_is_available() {
        let provider = WhisperCppProvider::new();
        // This will return false if whisper-cli is not in PATH
        let available = provider.is_available().await;
        // We don't assert a specific value since it depends on the system
        let _ = available;
    }

    #[test]
    fn test_whisper_output_deserialize() {
        let json = r#"{
            "segments": [
                {"text": "Hello world", "start": 0.0, "end": 1.5}
            ],
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
                {"word": "world", "start": 0.6, "end": 1.5, "probability": 0.95}
            ],
            "complexity": 0.5
        }"#;

        let output: WhisperCppOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.segments.len(), 1);
        assert_eq!(output.segments[0].text, "Hello world");
        assert_eq!(output.words.len(), 2);
        assert_eq!(output.words[0].word, "Hello");
        assert_eq!(output.complexity, Some(0.5));
    }
}
