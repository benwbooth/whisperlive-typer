//! Keyboard injection through the xdg-desktop-portal RemoteDesktop API.
//!
//! ydotool writes to a uinput device, and the compositor reads it through
//! the kernel's per-device evdev buffer (~64 events). When KWin stalls for
//! 50-100 ms (common while Whisper is using the same GPU), fast typing
//! overflows that buffer and the kernel discards keys. The portal hands
//! events to the compositor over D-Bus instead, which queues rather than
//! drops, so we can type as fast as the compositor accepts events. It also
//! sidesteps keyd, which otherwise grabs ydotool's virtual keyboard.
//!
//! The first session shows a one-time permission dialog; after that a
//! restore token (saved in ~/.cache/whisper-typer) lets later sessions
//! start silently.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use ashpd::desktop::PersistMode;
use ashpd::desktop::remote_desktop::{
    DeviceType, KeyState, RemoteDesktop, SelectDevicesOptions,
};
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::linux::{key_codes, parse_combo};
use super::platform::Typer;

const KEY_BACKSPACE: i32 = 14;

enum Action {
    Text(String),
    Backspace(usize),
    Combo(Vec<u16>),
}

struct Job {
    action: Action,
    done: std_mpsc::SyncSender<()>,
}

pub struct PortalTyper {
    tx: mpsc::UnboundedSender<Job>,
    codes: HashMap<&'static str, u16>,
}

fn restore_token_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home)
        .join(".cache")
        .join("whisper-typer")
        .join("portal_restore_token")
}

/// X11 keysym for a character. Latin-1 keysyms equal their code point;
/// everything else uses the Unicode keysym range.
fn keysym_for(c: char) -> i32 {
    match c {
        '\n' | '\r' => 0xff0d, // Return
        '\t' => 0xff09,        // Tab
        c if (0x20..=0x7e).contains(&(c as u32)) || (0xa0..=0xff).contains(&(c as u32)) => {
            c as i32
        }
        c => (0x0100_0000 | c as u32) as i32,
    }
}

impl PortalTyper {
    /// Start a RemoteDesktop session. Blocks until the session is running
    /// (which may include the user answering the permission dialog).
    pub fn connect(key_delay_us: u64) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<Job>();
        let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<()>>(1);

        std::thread::Builder::new()
            .name("portal-typer".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.into()));
                        return;
                    }
                };
                rt.block_on(run_session(rx, ready_tx, key_delay_us));
            })
            .context("spawn portal thread")?;

        // Generous timeout: the first run waits for the user to click Allow.
        match ready_rx.recv_timeout(Duration::from_secs(120)) {
            Ok(Ok(())) => Ok(Self { tx, codes: key_codes() }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow!("timed out waiting for RemoteDesktop portal")),
        }
    }

    fn submit(&self, action: Action) {
        let (done_tx, done_rx) = std_mpsc::sync_channel(1);
        if self.tx.send(Job { action, done: done_tx }).is_err() {
            warn!("Portal session is gone; dropping input");
            return;
        }
        // Wait so callers keep the same ordering/backpressure semantics as
        // the ydotool typer.
        let _ = done_rx.recv();
    }
}

async fn run_session(
    mut rx: mpsc::UnboundedReceiver<Job>,
    ready_tx: std_mpsc::SyncSender<Result<()>>,
    key_delay_us: u64,
) {
    let setup = async {
        let proxy = RemoteDesktop::new().await?;
        let session = proxy.create_session(Default::default()).await?;
        let token_path = restore_token_path();
        let saved = std::fs::read_to_string(&token_path).ok();
        let saved = saved.as_deref().map(str::trim).filter(|t| !t.is_empty());
        proxy
            .select_devices(
                &session,
                SelectDevicesOptions::default()
                    .set_devices(DeviceType::KEYBOARD)
                    .set_persist_mode(PersistMode::ExplicitlyRevoked)
                    .set_restore_token(saved),
            )
            .await?;
        let started = std::time::Instant::now();
        let response = proxy.start(&session, None, Default::default()).await?.response()?;
        if !response.devices().contains(DeviceType::KEYBOARD) {
            return Err(anyhow!("keyboard access was not granted"));
        }
        if let Some(token) = response.restore_token() {
            let _ = std::fs::create_dir_all(token_path.parent().unwrap());
            if let Err(e) = std::fs::write(&token_path, token) {
                warn!("Could not save portal restore token: {e}");
            }
        }
        info!(
            "RemoteDesktop portal session started in {:.0?} (restored: {})",
            started.elapsed(),
            saved.is_some()
        );
        Ok::<_, anyhow::Error>((proxy, session))
    };

    let (proxy, session) = match setup.await {
        Ok(v) => {
            let _ = ready_tx.send(Ok(()));
            v
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };

    let delay = Duration::from_micros(key_delay_us);
    let pause = || async {
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
    };

    while let Some(job) = rx.recv().await {
        let result: ashpd::Result<()> = async {
            match &job.action {
                Action::Text(text) => {
                    for c in text.chars() {
                        let sym = keysym_for(c);
                        proxy
                            .notify_keyboard_keysym(&session, sym, KeyState::Pressed, Default::default())
                            .await?;
                        proxy
                            .notify_keyboard_keysym(&session, sym, KeyState::Released, Default::default())
                            .await?;
                        pause().await;
                    }
                }
                Action::Backspace(n) => {
                    for _ in 0..*n {
                        proxy
                            .notify_keyboard_keycode(&session, KEY_BACKSPACE, KeyState::Pressed, Default::default())
                            .await?;
                        proxy
                            .notify_keyboard_keycode(&session, KEY_BACKSPACE, KeyState::Released, Default::default())
                            .await?;
                        pause().await;
                    }
                }
                Action::Combo(codes) => {
                    for &code in codes {
                        proxy
                            .notify_keyboard_keycode(&session, code as i32, KeyState::Pressed, Default::default())
                            .await?;
                    }
                    for &code in codes.iter().rev() {
                        proxy
                            .notify_keyboard_keycode(&session, code as i32, KeyState::Released, Default::default())
                            .await?;
                    }
                    pause().await;
                }
            }
            Ok(())
        }
        .await;
        if let Err(e) = result {
            warn!("Portal input failed: {e}");
        }
        let _ = job.done.send(());
    }
}

impl Typer for PortalTyper {
    fn type_text(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        info!("Typing: {text:?}");
        self.submit(Action::Text(text.to_string()));
    }

    fn backspace(&self, count: usize) {
        if count > 0 {
            self.submit(Action::Backspace(count));
        }
    }

    fn send_keys(&self, keys_str: &str) {
        for combo in keys_str.split_whitespace() {
            let combo = combo.to_lowercase();
            let combo = combo.trim_end_matches(|c| ".,!?".contains(c));
            if let Some(codes) = parse_combo(&self.codes, combo) {
                info!("Sending keys: {combo}");
                self.submit(Action::Combo(codes));
            }
        }
    }
}
