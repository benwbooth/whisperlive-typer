use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::warn;

use crate::commands::default_commands;

/// Top-level YAML config file structure.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ConfigFile {
    server: ServerSection,
    whisper: WhisperSection,
    audio: AudioSection,
    vad: VadSection,
    typer: TyperSection,
    commands: Option<CommandsSection>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ServerSection {
    host: Option<String>,
    port: Option<u16>,
    auto_start: Option<bool>,
    dir: Option<String>,
    start_timeout: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct WhisperSection {
    language: Option<String>,
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct AudioSection {
    device: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct VadSection {
    onset: Option<f64>,
    offset: Option<f64>,
    no_speech_thresh: Option<f64>,
    min_avg_logprob: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TyperSection {
    pending_debounce_ms: Option<u64>,
    ydotool_key_delay_ms: Option<u32>,
    ydotool_key_hold_ms: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct CommandsSection {
    keys: Option<HashMap<String, String>>,
    literals: Option<HashMap<String, String>>,
}

/// Runtime configuration with all defaults applied.
#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub language: String,
    pub model: String,
    pub device: String,
    pub vad_onset: f64,
    pub vad_offset: f64,
    pub no_speech_thresh: f64,
    pub min_avg_logprob: f64,
    pub pending_debounce_ms: u64,
    pub ydotool_key_delay_ms: u32,
    pub ydotool_key_hold_ms: u32,
    pub auto_start_server: bool,
    pub server_dir: String,
    pub server_start_timeout: u64,
    pub command_keys: HashMap<String, String>,
    pub command_literals: HashMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        let (default_keys, default_literals) = default_commands();
        Self {
            host: "localhost".into(),
            port: 9090,
            language: "en".into(),
            model: "small".into(),
            device: String::new(),
            vad_onset: 0.3,
            vad_offset: 0.2,
            no_speech_thresh: 0.45,
            min_avg_logprob: -0.8,
            pending_debounce_ms: 200,
            ydotool_key_delay_ms: 8,
            ydotool_key_hold_ms: 4,
            auto_start_server: true,
            server_dir: String::new(),
            server_start_timeout: 120,
            command_keys: default_keys,
            command_literals: default_literals,
        }
    }
}

impl Config {
    /// Platform-appropriate default config path.
    pub fn default_path() -> PathBuf {
        if cfg!(target_os = "macos") {
            dirs_path_macos()
        } else {
            dirs_path_linux()
        }
    }

    /// Load config from a YAML file, falling back to defaults.
    pub fn load(path: Option<&Path>) -> Self {
        let path = path
            .map(PathBuf::from)
            .unwrap_or_else(Self::default_path);

        let mut config = Self::default();

        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_yaml::from_str::<ConfigFile>(&contents) {
                    Ok(file) => {
                        // Server
                        if let Some(v) = file.server.host { config.host = v; }
                        if let Some(v) = file.server.port { config.port = v; }
                        if let Some(v) = file.server.auto_start { config.auto_start_server = v; }
                        if let Some(v) = file.server.dir { config.server_dir = v; }
                        if let Some(v) = file.server.start_timeout { config.server_start_timeout = v; }
                        // Whisper
                        if let Some(v) = file.whisper.language { config.language = v; }
                        if let Some(v) = file.whisper.model { config.model = v; }
                        // Audio
                        if let Some(v) = file.audio.device { config.device = v; }
                        // VAD
                        if let Some(v) = file.vad.onset { config.vad_onset = v; }
                        if let Some(v) = file.vad.offset { config.vad_offset = v; }
                        if let Some(v) = file.vad.no_speech_thresh { config.no_speech_thresh = v; }
                        if let Some(v) = file.vad.min_avg_logprob { config.min_avg_logprob = v; }
                        // Typer
                        if let Some(v) = file.typer.pending_debounce_ms { config.pending_debounce_ms = v; }
                        if let Some(v) = file.typer.ydotool_key_delay_ms { config.ydotool_key_delay_ms = v; }
                        if let Some(v) = file.typer.ydotool_key_hold_ms { config.ydotool_key_hold_ms = v; }
                        // Commands (merge on top of defaults)
                        if let Some(cmds) = file.commands {
                            if let Some(keys) = cmds.keys {
                                config.command_keys.extend(keys);
                            }
                            if let Some(literals) = cmds.literals {
                                config.command_literals.extend(literals);
                            }
                        }
                        tracing::info!("Loaded config from {}", path.display());
                    }
                    Err(e) => warn!("Failed to parse config: {e}"),
                },
                Err(e) => warn!("Failed to read config: {e}"),
            }
        }

        config
    }
}

fn dirs_path_linux() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home)
        .join(".config")
        .join("whisper-typer")
        .join("config.yaml")
}

fn dirs_path_macos() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home)
        .join("Library")
        .join("Application Support")
        .join("whisper-typer")
        .join("config.yaml")
}
