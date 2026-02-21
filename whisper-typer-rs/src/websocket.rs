use anyhow::{Context, Result};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::StreamExt;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tracing::info;

use crate::config::Config;
use crate::protocol::{ClientConfig, VadParameters};

pub type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;
pub type WsStream = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

/// Connect to the WhisperLive server and send initial configuration.
/// Returns the WebSocket split into sink (for sending) and stream (for receiving).
pub async fn connect(config: &Config) -> Result<(WsSink, WsStream)> {
    let uri = format!("ws://{}:{}", config.host, config.port);
    info!("Connecting to {uri}...");

    let (ws, _) = connect_async(&uri)
        .await
        .with_context(|| format!("Failed to connect to {uri}"))?;

    let (mut sink, stream) = ws.split();

    // Send initial configuration
    let client_config = ClientConfig {
        uid: std::process::id(),
        language: config.language.clone(),
        task: "transcribe".into(),
        model: config.model.clone(),
        use_vad: true,
        no_speech_thresh: config.no_speech_thresh,
        min_avg_logprob: config.min_avg_logprob,
        vad_parameters: VadParameters {
            onset: config.vad_onset,
            offset: config.vad_offset,
            min_speech_duration_ms: 0,
            min_silence_duration_ms: 300,
            speech_pad_ms: 100,
        },
    };

    let config_json = serde_json::to_string(&client_config)?;
    use futures_util::SinkExt;
    sink.send(Message::Text(config_json.into())).await?;
    info!("Connected and configured");

    Ok((sink, stream))
}
