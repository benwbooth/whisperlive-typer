use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use tracing::{info, warn};

use super::platform::{Notifier, Typer};

/// Linux keyboard scan codes for ydotool (input-event-codes.h).
fn key_codes() -> HashMap<&'static str, u16> {
    HashMap::from([
        // Special keys
        ("escape", 1), ("esc", 1),
        ("backspace", 14), ("back", 14),
        ("tab", 15),
        ("enter", 28), ("return", 28),
        ("space", 57),
        ("delete", 111), ("del", 111),
        // Modifiers
        ("ctrl", 29), ("control", 29), ("leftctrl", 29),
        ("shift", 42), ("leftshift", 42),
        ("alt", 56), ("leftalt", 56),
        ("super", 125), ("meta", 125), ("windows", 125), ("win", 125),
        ("rightctrl", 97), ("rightshift", 54), ("rightalt", 100),
        // Navigation
        ("home", 102), ("end", 107),
        ("pageup", 104), ("pgup", 104),
        ("pagedown", 109), ("pgdn", 109),
        ("up", 103), ("down", 108), ("left", 105), ("right", 106),
        ("insert", 110), ("ins", 110),
        // Function keys
        ("f1", 59), ("f2", 60), ("f3", 61), ("f4", 62), ("f5", 63), ("f6", 64),
        ("f7", 65), ("f8", 66), ("f9", 67), ("f10", 68), ("f11", 87), ("f12", 88),
        // Numbers (top row)
        ("1", 2), ("2", 3), ("3", 4), ("4", 5), ("5", 6),
        ("6", 7), ("7", 8), ("8", 9), ("9", 10), ("0", 11),
        // Letters
        ("a", 30), ("b", 48), ("c", 46), ("d", 32), ("e", 18), ("f", 33), ("g", 34),
        ("h", 35), ("i", 23), ("j", 36), ("k", 37), ("l", 38), ("m", 50), ("n", 49),
        ("o", 24), ("p", 25), ("q", 16), ("r", 19), ("s", 31), ("t", 20), ("u", 22),
        ("v", 47), ("w", 17), ("x", 45), ("y", 21), ("z", 44),
        // Punctuation
        ("minus", 12), ("equal", 13), ("equals", 13),
        ("leftbracket", 26), ("rightbracket", 27),
        ("semicolon", 39), ("apostrophe", 40), ("quote", 40),
        ("grave", 41), ("backslash", 43),
        ("comma", 51), ("period", 52), ("dot", 52), ("slash", 53),
    ])
}

/// Keyboard input using ydotool (Linux/Wayland).
pub struct YdotoolTyper {
    socket_path: String,
    dry_run: bool,
    key_delay: u32,
    key_hold: u32,
    codes: HashMap<&'static str, u16>,
}

impl YdotoolTyper {
    pub fn new(socket_path: Option<String>, dry_run: bool, key_delay: u32, key_hold: u32) -> Self {
        let socket_path = socket_path
            .or_else(|| std::env::var("YDOTOOL_SOCKET").ok())
            .unwrap_or_else(|| "/run/ydotoold/socket".into());
        Self {
            socket_path,
            dry_run,
            key_delay,
            key_hold,
            codes: key_codes(),
        }
    }

    fn run_ydotool(&self, args: &[&str]) {
        if self.dry_run {
            info!("[DRY RUN] Would execute: ydotool {}", args.join(" "));
            return;
        }
        let result = Command::new("ydotool")
            .args(args)
            .env("YDOTOOL_SOCKET", &self.socket_path)
            .output();
        match result {
            Ok(output) if !output.status.success() => {
                warn!(
                    "ydotool error: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(e) => warn!("ydotool not found: {e}"),
            _ => {}
        }
    }

    fn send_key_combo(&self, combo: &str) {
        let combo = combo.to_lowercase();
        let combo = combo.trim_end_matches(|c| ".,!?".contains(c));
        let parts: Vec<&str> = combo.split('+').collect();
        let mut codes = Vec::new();

        for part in &parts {
            let part = part.trim().trim_end_matches(|c: char| ".,!?".contains(c));
            if let Some(&code) = self.codes.get(part) {
                codes.push(code);
            } else {
                warn!("Unknown key: {part}");
                return;
            }
        }

        if codes.is_empty() {
            return;
        }

        info!("Sending keys: {combo}");

        let mut key_args = Vec::new();
        for &code in &codes {
            key_args.push(format!("{code}:1"));
        }
        for &code in codes.iter().rev() {
            key_args.push(format!("{code}:0"));
        }

        let delay_arg = format!("--key-delay={}", self.key_delay);
        let hold_arg = format!("--key-hold={}", self.key_hold);
        let mut args: Vec<&str> = vec!["key", &delay_arg, &hold_arg, "--clearmodifiers"];
        let key_arg_refs: Vec<&str> = key_args.iter().map(|s| s.as_str()).collect();
        args.extend_from_slice(&key_arg_refs);
        self.run_ydotool(&args);
    }
}

impl Typer for YdotoolTyper {
    fn type_text(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        info!("Typing: {text:?}");
        let delay_arg = format!("--key-delay={}", self.key_delay);
        let hold_arg = format!("--key-hold={}", self.key_hold);
        self.run_ydotool(&["type", &delay_arg, &hold_arg, "--clearmodifiers", "--", text]);
    }

    fn backspace(&self, count: usize) {
        if count == 0 {
            return;
        }
        let batch_size = 32;
        let mut remaining = count;
        let delay_arg = format!("--key-delay={}", self.key_delay);
        let hold_arg = format!("--key-hold={}", self.key_hold);
        while remaining > 0 {
            let batch = remaining.min(batch_size);
            let mut key_events = Vec::with_capacity(batch * 2);
            for _ in 0..batch {
                key_events.push("14:1".to_string());
                key_events.push("14:0".to_string());
            }
            let mut args: Vec<&str> = vec!["key", &delay_arg, &hold_arg, "--clearmodifiers"];
            let refs: Vec<&str> = key_events.iter().map(|s| s.as_str()).collect();
            args.extend_from_slice(&refs);
            self.run_ydotool(&args);
            remaining -= batch;
        }
    }

    fn send_keys(&self, keys_str: &str) {
        let sequences: Vec<&str> = keys_str.split_whitespace().collect();
        for combo in sequences {
            self.send_key_combo(combo);
        }
    }
}

/// Notification ID file path for cleanup.
fn notification_id_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home)
        .join(".cache")
        .join("whisper-typer")
        .join("notification_id")
}

/// Desktop notifications using notify-send (freedesktop).
pub struct LinuxNotifier {
    notification_id: Mutex<Option<u32>>,
}

impl LinuxNotifier {
    pub fn new() -> Self {
        Self {
            notification_id: Mutex::new(None),
        }
    }
}

impl Notifier for LinuxNotifier {
    fn show(&self, message: &str, persistent: bool) {
        let mut cmd_args = vec![
            "notify-send".to_string(),
            "--app-name=Whisper Typer".to_string(),
            "--print-id".to_string(),
        ];
        if persistent {
            cmd_args.push("--expire-time=0".to_string());
        }
        if let Some(id) = *self.notification_id.lock().unwrap() {
            cmd_args.push(format!("--replace-id={id}"));
        }
        cmd_args.push(message.to_string());

        let result = Command::new(&cmd_args[0])
            .args(&cmd_args[1..])
            .output();

        match result {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(id) = stdout.trim().parse::<u32>() {
                    *self.notification_id.lock().unwrap() = Some(id);
                    let id_path = notification_id_path();
                    let _ = std::fs::create_dir_all(id_path.parent().unwrap());
                    let _ = std::fs::write(&id_path, id.to_string());
                }
            }
            _ => {}
        }
    }

    fn close(&self) {
        let mut id_guard = self.notification_id.lock().unwrap();
        if let Some(id) = *id_guard {
            let _ = Command::new("gdbus")
                .args([
                    "call", "--session",
                    "--dest", "org.freedesktop.Notifications",
                    "--object-path", "/org/freedesktop/Notifications",
                    "--method", "org.freedesktop.Notifications.CloseNotification",
                    &id.to_string(),
                ])
                .output();
            *id_guard = None;
        }
        let _ = std::fs::remove_file(notification_id_path());
    }
}
