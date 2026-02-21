use serde::{Deserialize, Serialize};

/// Initial configuration sent to the WhisperLive server on connection.
#[derive(Debug, Serialize)]
pub struct ClientConfig {
    pub uid: u32,
    pub language: String,
    pub task: String,
    pub model: String,
    pub use_vad: bool,
    pub no_speech_thresh: f64,
    pub min_avg_logprob: f64,
    pub vad_parameters: VadParameters,
}

#[derive(Debug, Serialize)]
pub struct VadParameters {
    pub onset: f64,
    pub offset: f64,
    pub min_speech_duration_ms: u32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
}

/// A transcription message received from the server.
#[derive(Debug, Deserialize)]
pub struct ServerMessage {
    #[allow(dead_code)]
    pub uid: Option<serde_json::Value>,
    pub finalize: Option<String>,
    pub text: Option<String>,
    pub segments: Option<Vec<Segment>>,
}

/// A single transcription segment from the server.
#[derive(Debug, Deserialize)]
pub struct Segment {
    #[allow(dead_code)]
    pub start: Option<serde_json::Value>,
    #[allow(dead_code)]
    pub end: Option<serde_json::Value>,
    pub text: Option<String>,
    pub completed: Option<bool>,
    pub no_speech_prob: Option<f64>,
    pub avg_logprob: Option<f64>,
}

impl Segment {
    pub fn no_speech_prob_val(&self) -> f64 {
        self.no_speech_prob.unwrap_or(0.0)
    }

    pub fn avg_logprob_val(&self) -> f64 {
        self.avg_logprob.unwrap_or(0.0)
    }
}
