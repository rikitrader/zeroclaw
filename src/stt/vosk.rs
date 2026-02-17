// ═══════════════════════════════════════════════════════════════
// VOSK LOCAL STT PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{audio_to_wav, SttConfig, SttProvider, SttProviderType, SttResult, SttWord};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::Path;
use std::process::Command;
use tokio::io::AsyncWriteExt;
use tokio::task;

/// Vosk local STT provider
///
/// This provider uses Vosk (https://alphacephei.com/vosk/)
/// for offline speech recognition. It requires Vosk API or
/// vosk-transcriber to be installed.
pub struct VoskProvider {
    /// Path to vosk-transcriber executable
    executable_path: Option<String>,
    /// Default model directory
    model_dir: Option<String>,
}

impl VoskProvider {
    /// Create a new Vosk provider
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
            if Path::new(path).exists() {
                return Ok(path.clone());
            }
        }

        // Default to models directory in user's home
        let lang_code = config.language.split('-').next().unwrap_or("en");
        let model_name = format!("vosk-model-{}-small", lang_code);

        if let Some(ref model_dir) = self.model_dir {
            let path = format!("{}/{}", model_dir, model_name);
            if Path::new(&path).exists() {
                return Ok(path);
            }
        }

        // Try common model locations
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());

        let possible_locations = vec![
            format!("{}/.local/share/vosk/{}", home, model_name),
            format!("{}/.vosk/{}", home, model_name),
            format!("./models/{}", model_name),
        ];

        for loc in &possible_locations {
            if Path::new(loc).exists() {
                return Ok(loc.clone());
            }
        }

        Err(anyhow::anyhow!(
            "Vosk model not found. Tried: {:?}. \
            Please download a model from https://alphacephei.com/vosk/models \
            and set model_path in config or place in ~/.local/share/vosk/",
            possible_locations
        ))
    }

    /// Get the executable path
    fn get_executable(&self) -> String {
        self.executable_path
            .clone()
            .unwrap_or_else(|| "vosk-transcriber".to_string())
    }

    /// Run Vosk transcriber synchronously
    fn run_vosk(&self, audio_path: &Path, config: &SttConfig) -> Result<VoskOutput> {
        let executable = self.get_executable();
        let model_path = self.get_model_path(config)?;

        let mut cmd = Command::new(&executable);

        // Model
        cmd.arg("--model").arg(&model_path);

        // Audio file
        cmd.arg("--input").arg(audio_path);

        // Output format (JSON)
        cmd.arg("--output-format").arg("json");

        // Sample rate
        cmd.arg("--sample-rate").arg(config.sample_rate.to_string());

        // Language
        cmd.arg("--lang").arg(&config.language);

        // Run the command
        let output = cmd
            .output()
            .with_context(|| {
                format!(
                    "Failed to execute vosk-transcriber at '{}'. \
                    Is vosk-api installed? You can install it with: \
                    pip install vosk or use vosk-transcriber",
                    executable
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "vosk-transcriber failed with exit code {}: {}",
                output.status,
                stderr
            );
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: VoskOutput = serde_json::from_str(&stdout)
            .context("Failed to parse Vosk JSON output")?;

        Ok(result)
    }

    /// Run Vosk via Python API (alternative method)
    fn run_vosk_python(&self, audio_path: &Path, config: &SttConfig) -> Result<VoskOutput> {
        let model_path = self.get_model_path(config)?;

        // Python script to run Vosk
        let python_script = format!(
            r#"
import sys
import json
import wave
from vosk import Model, KaldiRecognizer

def transcribe(audio_path, model_path, sample_rate):
    try:
        model = Model(model_path)
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))
        sys.exit(1)

    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
        print(json.dumps({{"error": "Audio must be WAV format mono PCM, {} Hz".format(sample_rate)}}))
        sys.exit(1)

    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)

    result = {{"text": "", "words": []}}

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            partial = json.loads(rec.Result())
            result["text"] += " " + partial.get("text", "")
            if "result" in partial:
                result["words"].extend(partial["result"])

    # Final result
    partial = json.loads(rec.FinalResult())
    result["text"] = result["text"].strip() + " " + partial.get("text", "")
    if "result" in partial:
        result["words"].extend(partial["result"])

    result["text"] = result["text"].strip()
    print(json.dumps(result))

if __name__ == "__main__":
    transcribe("{audio_path}", "{model_path}", {sample_rate})
"#,
            audio_path = audio_path.display(),
            model_path = model_path.replace('\\', "\\\\"),
            sample_rate = config.sample_rate
        );

        // Run Python script
        let output = Command::new("python3")
            .arg("-c")
            .arg(&python_script)
            .output()
            .context("Failed to execute Python with vosk module. Is vosk installed? (pip install vosk)")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Vosk Python script failed: {}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        serde_json::from_str(&stdout).context("Failed to parse Vosk output")
    }
}

impl Default for VoskProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SttProvider for VoskProvider {
    fn provider_type(&self) -> SttProviderType {
        SttProviderType::Vosk
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
            "vosk_{}.wav",
            uuid::Uuid::new_v4()
        ));

        // Write audio data to temp file
        let mut file = tokio::fs::File::create(&temp_file).await?;
        file.write_all(&wav_data).await?;
        file.flush().await?;

        // Run Vosk in a blocking task
        let temp_file_clone = temp_file.clone();
        let provider = self.clone_ref();
        let config_clone = config.clone();

        let output = task::spawn_blocking(move || {
            // Try vosk-transcriber first, fall back to Python API
            provider
                .run_vosk(&temp_file_clone, &config_clone)
                .or_else(|_| provider.run_vosk_python(&temp_file_clone, &config_clone))
        })
        .await??;

        // Clean up temp file
        let _ = tokio::fs::remove_file(&temp_file).await;

        // Convert output to SttResult
        let words: Vec<SttWord> = output
            .words
            .into_iter()
            .map(|w| SttWord {
                word: w.word,
                start_time: w.start,
                end_time: w.end,
                confidence: w.conf,
            })
            .collect();

        Ok(SttResult {
            text: output.text,
            confidence: if words.is_empty() {
                0.0
            } else {
                words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
            },
            language: Some(config.language.clone()),
            alternatives: Vec::new(),
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

        // Run Vosk in a blocking task
        let provider = self.clone_ref();
        let file_path = file_path.to_path_buf();
        let config_clone = config.clone();

        let output = task::spawn_blocking(move || {
            provider
                .run_vosk(&file_path, &config_clone)
                .or_else(|_| provider.run_vosk_python(&file_path, &config_clone))
        })
        .await??;

        // Convert output to SttResult
        let words: Vec<SttWord> = output
            .words
            .into_iter()
            .map(|w| SttWord {
                word: w.word,
                start_time: w.start,
                end_time: w.end,
                confidence: w.conf,
            })
            .collect();

        Ok(SttResult {
            text: output.text,
            confidence: if words.is_empty() {
                0.0
            } else {
                words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
            },
            language: Some(config.language.clone()),
            alternatives: Vec::new(),
            words,
        })
    }

    async fn is_available(&self) -> bool {
        // Check if either vosk-transcriber or Python vosk module is available
        let transcriber_available = Command::new("vosk-transcriber")
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        if transcriber_available {
            return true;
        }

        // Check Python vosk module
        Command::new("python3")
            .arg("-c")
            .arg("import vosk")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

// Allow cloning for use in async tasks
impl Clone for VoskProvider {
    fn clone(&self) -> Self {
        Self {
            executable_path: self.executable_path.clone(),
            model_dir: self.model_dir.clone(),
        }
    }
}

// Helper method for cloning
impl VoskProvider {
    fn clone_ref(&self) -> Self {
        self.clone()
    }
}

// ═══════════════════════════════════════════════════════════════
// VOSK OUTPUT TYPES
// ═══════════════════════════════════════════════════════════════

/// Vosk JSON output
#[derive(Debug, Clone, serde::Deserialize)]
struct VoskOutput {
    /// Transcribed text
    text: String,
    /// Word-level timestamps
    #[serde(default)]
    words: Vec<VoskWord>,
}

/// Vosk word
#[derive(Debug, Clone, serde::Deserialize)]
struct VoskWord {
    /// Word text
    word: String,
    /// Start time in seconds
    start: f32,
    /// End time in seconds
    end: f32,
    /// Confidence score
    conf: f32,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vosk_provider_new() {
        let provider = VoskProvider::new();
        assert_eq!(provider.provider_type(), SttProviderType::Vosk);
    }

    #[test]
    fn test_vosk_provider_default() {
        let provider = VoskProvider::default();
        assert_eq!(provider.provider_type(), SttProviderType::Vosk);
    }

    #[test]
    fn test_vosk_provider_with_executable() {
        let provider = VoskProvider::new()
            .with_executable("/custom/path/vosk-transcriber".to_string());
        assert_eq!(provider.executable_path, Some("/custom/path/vosk-transcriber".to_string()));
    }

    #[test]
    fn test_vosk_provider_with_model_dir() {
        let provider = VoskProvider::new()
            .with_model_dir("/models/vosk".to_string());
        assert_eq!(provider.model_dir, Some("/models/vosk".to_string()));
    }

    #[test]
    fn test_vosk_provider_clone() {
        let provider = VoskProvider::new()
            .with_executable("vosk".to_string())
            .with_model_dir("/models".to_string());

        let cloned = provider.clone();
        assert_eq!(cloned.executable_path, provider.executable_path);
        assert_eq!(cloned.model_dir, provider.model_dir);
    }

    #[test]
    fn test_vosk_provider_name() {
        let provider = VoskProvider::new();
        assert_eq!(provider.name(), "Vosk (Local)");
    }

    #[tokio::test]
    async fn test_vosk_provider_empty_audio() {
        let provider = VoskProvider::new();
        let config = SttConfig::default();

        let result = provider.transcribe(&[], &config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_vosk_provider_is_available() {
        let provider = VoskProvider::new();
        // This will return false if vosk-transcriber or Python vosk module is not available
        let available = provider.is_available().await;
        // We don't assert a specific value since it depends on the system
        let _ = available;
    }

    #[test]
    fn test_vosk_output_deserialize() {
        let json = r#"{
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "conf": 0.98},
                {"word": "world", "start": 0.6, "end": 1.5, "conf": 0.95}
            ]
        }"#;

        let output: VoskOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.text, "hello world");
        assert_eq!(output.words.len(), 2);
        assert_eq!(output.words[0].word, "hello");
        assert_eq!(output.words[0].conf, 0.98);
    }

    #[test]
    fn test_vosk_output_empty_text() {
        let json = r#"{
            "text": "",
            "words": []
        }"#;

        let output: VoskOutput = serde_json::from_str(json).unwrap();
        assert!(output.text.is_empty());
        assert!(output.words.is_empty());
    }
}
