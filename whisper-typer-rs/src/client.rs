use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::audio::AudioCapture;
use crate::commands::{CommandAction, CommandProcessor, CommandResult};
use crate::config::Config;
use crate::mute::SoftwareMuteDetector;
use crate::protocol::ServerMessage;
use crate::typer::StatefulTyper;
use crate::typer::platform::Notifier;
use crate::websocket;

/// Main client that ties audio → WebSocket → transcription → typing.
pub struct Client {
    config: Config,
    typer: Arc<StatefulTyper>,
    notifier: Box<dyn Notifier>,
    commands: CommandProcessor,
    device: Option<crate::audio::AudioDevice>,
}

impl Client {
    pub fn new(
        config: Config,
        typer: Arc<StatefulTyper>,
        notifier: Box<dyn Notifier>,
        device: Option<crate::audio::AudioDevice>,
    ) -> Self {
        let commands = CommandProcessor::new(
            config.command_keys.clone(),
            config.command_literals.clone(),
        );
        Self {
            config,
            typer,
            notifier,
            commands,
            device,
        }
    }

    pub async fn run(self) -> Result<()> {
        let cancel = CancellationToken::new();
        let notifier: Arc<dyn Notifier> = Arc::from(self.notifier);

        // Connect WebSocket
        let (ws_sink, ws_stream) = websocket::connect(&self.config).await?;

        // Start audio capture
        let mut audio = AudioCapture::start(self.device)?;
        let hw_muted = audio.hw_muted.clone();
        let sw_mute = Arc::new(SoftwareMuteDetector::new());

        // The WebSocket accepts connections before the per-client model has loaded.
        // Do not claim to be listening until WhisperLive sends SERVER_READY.
        notifier.show("Loading speech model...", true);

        // Channel for parsed server messages
        let (msg_tx, msg_rx) = mpsc::channel::<ServerMessage>(64);

        let cancel2 = cancel.clone();
        let cancel3 = cancel.clone();

        // Task: send audio chunks to server
        let send_handle = {
            let sw_mute = sw_mute.clone();
            let notifier_show = {
                // We can't move notifier into multiple tasks, so audio mute/unmute
                // notifications are handled in the handler task via a channel.
                let (mute_tx, mute_rx) = mpsc::channel::<bool>(4);
                let mute_tx2 = mute_tx.clone();
                let cancel_send = cancel2.clone();

                // Audio sender
                let send_task = tokio::spawn(async move {
                    let mut was_muted = false;
                    let mut ws_sink = ws_sink;
                    loop {
                        tokio::select! {
                            _ = cancel_send.cancelled() => break,
                            chunk = audio.rx.recv() => {
                                let Some(chunk) = chunk else { break };

                                // Check mute
                                let is_hw_muted = hw_muted.load(std::sync::atomic::Ordering::Relaxed);
                                let is_sw_muted = sw_mute.is_muted().await;
                                let is_muted = is_hw_muted || is_sw_muted;

                                if is_muted {
                                    if !was_muted {
                                        info!("Microphone muted, pausing audio stream");
                                        was_muted = true;
                                        let _ = mute_tx2.try_send(true);
                                        audio.drain();
                                    }
                                    continue;
                                } else if was_muted {
                                    info!("Microphone unmuted, resuming audio stream");
                                    was_muted = false;
                                    let _ = mute_tx2.try_send(false);
                                    audio.drain();
                                }

                                if let Err(e) = ws_sink.send(Message::Binary(chunk.into())).await {
                                    error!("Error sending audio: {e}");
                                    break;
                                }
                            }
                        }
                    }
                });

                (send_task, mute_rx)
            };
            notifier_show
        };
        let (send_task, mut mute_rx) = send_handle;

        // Task: receive WebSocket messages
        let recv_task = tokio::spawn({
            let cancel_recv = cancel3.clone();
            async move {
                let mut ws_stream = ws_stream;
                loop {
                    tokio::select! {
                        _ = cancel_recv.cancelled() => break,
                        msg = ws_stream.next() => {
                            match msg {
                                Some(Ok(Message::Text(text))) => {
                                    debug!("Received: {}", &text[..text.len().min(500)]);
                                    match serde_json::from_str::<ServerMessage>(&text) {
                                        Ok(server_msg) => {
                                            if msg_tx.send(server_msg).await.is_err() {
                                                break;
                                            }
                                        }
                                        Err(e) => warn!("Invalid JSON: {e}"),
                                    }
                                }
                                Some(Ok(Message::Close(_))) | None => break,
                                Some(Err(e)) => {
                                    error!("WebSocket error: {e}");
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        });

        // Handler: process transcription and status messages
        let handler_task = tokio::spawn({
            let typer = self.typer;
            let commands = self.commands;
            let config = self.config.clone();
            let cancel_handler = cancel.clone();
            let handler_notifier = notifier.clone();
            async move {
                handle_messages(
                    msg_rx,
                    &mut mute_rx,
                    &typer,
                    &commands,
                    &*handler_notifier,
                    &config,
                    cancel_handler,
                )
                .await;
            }
        });

        // Wait for any task to complete. Handle SIGTERM (systemctl stop)
        // and SIGINT (Ctrl+C) gracefully so we get a chance to dismiss the
        // notification on the way out.
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = send_task => info!("Send task ended"),
            _ = recv_task => info!("Receive task ended"),
            _ = handler_task => info!("Handler task ended"),
            _ = tokio::signal::ctrl_c() => {
                info!("SIGINT received, stopping...");
            }
            _ = sigterm.recv() => {
                info!("SIGTERM received, stopping...");
            }
        }

        cancel.cancel();
        notifier.close();
        info!("Client stopped");
        Ok(())
    }
}

async fn handle_messages(
    mut msg_rx: mpsc::Receiver<ServerMessage>,
    mute_rx: &mut mpsc::Receiver<bool>,
    typer: &StatefulTyper,
    commands: &CommandProcessor,
    notifier: &dyn Notifier,
    config: &Config,
    cancel: CancellationToken,
) {
    let pending_update_interval = Duration::from_millis(config.pending_debounce_ms);
    let no_speech_thresh = config.no_speech_thresh;
    let min_avg_logprob = config.min_avg_logprob;
    let command_dedupe_window = Duration::from_millis(350);
    let mut server_ready = false;
    let mut muted = false;

    // Apply the first provisional result immediately, then coalesce corrections
    // to a fixed rate. A trailing-edge debounce can postpone feedback forever
    // while the server is sending updates faster than the debounce interval.
    let mut queued_pending: Option<String> = None;
    let mut pending_deadline: Option<tokio::time::Instant> = None;
    let mut last_pending_update: Option<tokio::time::Instant> = None;

    loop {
        let sleep_fut = if let Some(deadline) = pending_deadline {
            tokio::time::sleep_until(deadline)
        } else {
            // Sleep forever (will never complete)
            tokio::time::sleep(Duration::from_secs(86400))
        };

        tokio::select! {
            _ = cancel.cancelled() => break,

            // Apply the newest correction at the fixed update interval.
            _ = sleep_fut, if pending_deadline.is_some() => {
                if let Some(pending) = queued_pending.take() {
                    pending_deadline = None;
                    tokio::task::block_in_place(|| {
                        typer.update_pending(&pending);
                    });
                    last_pending_update = Some(tokio::time::Instant::now());
                }
            }

            // Mute state change
            Some(is_muted) = mute_rx.recv() => {
                muted = is_muted;
                if muted {
                    notifier.show("\u{1f507} Muted", true);
                } else if server_ready {
                    notifier.show("\u{1f3a4} Listening...", true);
                } else {
                    notifier.show("Loading speech model...", true);
                }
            }

            // Server message
            Some(data) = msg_rx.recv() => {
                if data.is_ready() {
                    server_ready = true;
                    info!(backend = ?data.backend, "Speech model ready");
                    if muted {
                        notifier.show("\u{1f507} Muted", true);
                    } else {
                        notifier.show("\u{1f3a4} Listening...", true);
                    }
                }

                // Handle finalized text
                if let Some(finalized) = &data.finalize {
                    // Flush the newest provisional text before finalizing it.
                    if let Some(pending) = queued_pending.take() {
                        pending_deadline = None;
                        tokio::task::block_in_place(|| {
                            typer.update_pending(&pending);
                        });
                        last_pending_update = Some(tokio::time::Instant::now());
                    }

                    let mut finalized = finalized.clone();

                    // Confidence filter on latest completed segment
                    if let Some(seg) = get_latest_completed_segment(&data) {
                        if !segment_passes_filters(seg, no_speech_thresh, min_avg_logprob) {
                            info!("Dropping low-confidence finalized segment");
                            finalized.clear();
                        }
                    }

                    // Check for commands
                    let finalized_stripped = finalized.trim();
                    if !finalized_stripped.is_empty() {
                        let cmd_result = commands.process(finalized_stripped);
                        if matches!(cmd_result.action, CommandAction::Keys | CommandAction::Literal) {
                            if should_execute_command(typer, finalized_stripped, command_dedupe_window) {
                                tokio::task::block_in_place(|| {
                                    typer.clear_pending();
                                    execute_command(typer, &cmd_result);
                                });
                            }
                            // Skip normal finalization for commands
                            // Still process pending text below
                            if data.text.is_none() {
                                continue;
                            }
                            // Process pending text for command messages
                        } else {
                            // Normal finalization
                            tokio::task::block_in_place(|| {
                                typer.handle_finalize(&finalized);
                            });
                        }
                    } else {
                        // Empty finalization (cleared by filter)
                        tokio::task::block_in_place(|| {
                            typer.handle_finalize("");
                        });
                    }
                }

                // Handle pending text
                if let Some(pending) = &data.text {
                    let mut pending = pending.clone();

                    // Confidence filter
                    if !pending.is_empty() {
                        if let Some(seg) = get_latest_segment(&data) {
                            let seg_text = seg.text.as_deref().unwrap_or("");
                            if seg_text.trim() == pending.trim()
                                && !segment_passes_filters(seg, no_speech_thresh, min_avg_logprob)
                            {
                                info!(
                                    "Dropping low-confidence pending segment (no_speech_prob={:?}, avg_logprob={:?})",
                                    seg.no_speech_prob, seg.avg_logprob
                                );
                                pending.clear();
                            }
                        }
                    }

                    let now = tokio::time::Instant::now();
                    let apply_now = pending_update_interval.is_zero()
                        || last_pending_update
                            .map(|last| now.duration_since(last) >= pending_update_interval)
                            .unwrap_or(true);

                    if apply_now {
                        queued_pending = None;
                        pending_deadline = None;
                        tokio::task::block_in_place(|| {
                            typer.update_pending(&pending);
                        });
                        last_pending_update = Some(tokio::time::Instant::now());
                    } else {
                        queued_pending = Some(pending);
                        pending_deadline = last_pending_update
                            .map(|last| last + pending_update_interval);
                    }
                }
            }
        }
    }
}

fn get_latest_segment(data: &ServerMessage) -> Option<&crate::protocol::Segment> {
    data.segments.as_ref()?.last()
}

fn get_latest_completed_segment(data: &ServerMessage) -> Option<&crate::protocol::Segment> {
    data.segments
        .as_ref()?
        .iter()
        .rev()
        .find(|s| s.completed == Some(true))
}

fn segment_passes_filters(
    seg: &crate::protocol::Segment,
    no_speech_thresh: f64,
    min_avg_logprob: f64,
) -> bool {
    seg.no_speech_prob_val() <= no_speech_thresh && seg.avg_logprob_val() >= min_avg_logprob
}

fn should_execute_command(
    typer: &StatefulTyper,
    command_text: &str,
    dedupe_window: Duration,
) -> bool {
    let normalized = normalize_cmd_text(command_text);
    let mut state = typer.state.lock().unwrap();
    let now = monotonic_secs();

    if !state.last_command_text.is_empty() {
        let last_normalized = normalize_cmd_text(&state.last_command_text);
        if normalized == last_normalized && (now - state.last_command_at) < dedupe_window.as_secs_f64()
        {
            return false;
        }
    }

    state.last_command_text = command_text.to_string();
    state.last_command_at = now;
    true
}

fn normalize_cmd_text(text: &str) -> String {
    text.to_lowercase()
        .trim_end_matches(|c| ".,!?".contains(c))
        .to_string()
}

fn monotonic_secs() -> f64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn execute_command(typer: &StatefulTyper, cmd: &CommandResult) {
    info!("Executing command: {:?} -> {:?}", cmd.action, cmd.payload);
    match cmd.action {
        CommandAction::Keys => typer.send_keys(&cmd.payload),
        CommandAction::Literal => typer.type_text(&cmd.payload),
        CommandAction::None => {}
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    use super::handle_messages;
    use crate::commands::CommandProcessor;
    use crate::config::Config;
    use crate::protocol::ServerMessage;
    use crate::typer::StatefulTyper;
    use crate::typer::platform::{Notifier, Typer};

    #[derive(Clone)]
    struct RecordingTyper {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Typer for RecordingTyper {
        fn type_text(&self, text: &str) {
            self.events.lock().unwrap().push(format!("type:{text}"));
        }

        fn backspace(&self, count: usize) {
            self.events
                .lock()
                .unwrap()
                .push(format!("backspace:{count}"));
        }

        fn send_keys(&self, keys_str: &str) {
            self.events
                .lock()
                .unwrap()
                .push(format!("keys:{keys_str}"));
        }
    }

    #[derive(Clone)]
    struct RecordingNotifier {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Notifier for RecordingNotifier {
        fn show(&self, message: &str, _persistent: bool) {
            self.events.lock().unwrap().push(message.to_string());
        }

        fn close(&self) {}
    }

    fn message(text: Option<&str>, ready: bool) -> ServerMessage {
        ServerMessage {
            uid: None,
            message: ready.then(|| serde_json::Value::String("SERVER_READY".into())),
            backend: ready.then(|| "faster_whisper".into()),
            finalize: None,
            text: text.map(str::to_string),
            segments: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn applies_first_pending_immediately_and_coalesces_corrections() {
        let typing_events = Arc::new(Mutex::new(Vec::new()));
        let notification_events = Arc::new(Mutex::new(Vec::new()));
        let typer = StatefulTyper::new(Box::new(RecordingTyper {
            events: typing_events.clone(),
        }));
        let notifier = RecordingNotifier {
            events: notification_events.clone(),
        };
        let mut config = Config::default();
        config.pending_debounce_ms = 100;
        let commands = CommandProcessor::new(
            config.command_keys.clone(),
            config.command_literals.clone(),
        );
        let (message_tx, message_rx) = mpsc::channel(8);
        let (_mute_tx, mut mute_rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();

        let producer = async {
            message_tx.send(message(None, true)).await.unwrap();
            message_tx.send(message(Some("hello"), false)).await.unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;

            assert_eq!(&*typing_events.lock().unwrap(), &["type:hello"]);
            assert_eq!(
                &*notification_events.lock().unwrap(),
                &["\u{1f3a4} Listening..."]
            );

            message_tx.send(message(Some("help"), false)).await.unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert_eq!(&*typing_events.lock().unwrap(), &["type:hello"]);

            tokio::time::sleep(Duration::from_millis(100)).await;
            assert_eq!(
                &*typing_events.lock().unwrap(),
                &["type:hello", "backspace:2", "type:p"]
            );
            cancel.cancel();
        };

        tokio::join!(
            handle_messages(
                message_rx,
                &mut mute_rx,
                &typer,
                &commands,
                &notifier,
                &config,
                cancel.clone(),
            ),
            producer
        );
    }
}
