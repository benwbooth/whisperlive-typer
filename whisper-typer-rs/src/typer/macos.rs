#[cfg(target_os = "macos")]
mod imp {
    use std::process::Command;

    use tracing::{info, warn};

    use crate::typer::platform::{Notifier, Typer};

    /// macOS keyboard simulation using enigo.
    pub struct MacOSTyper {
        dry_run: bool,
        enigo: std::sync::Mutex<enigo::Enigo>,
    }

    impl MacOSTyper {
        pub fn new(dry_run: bool) -> Self {
            let enigo = enigo::Enigo::new(&enigo::Settings::default())
                .expect("Failed to create Enigo instance");
            Self {
                dry_run,
                enigo: std::sync::Mutex::new(enigo),
            }
        }

        fn send_key_combo(&self, combo: &str) {
            info!("Sending keys: {combo}");
            if self.dry_run {
                info!("[DRY RUN] Would send: {combo}");
                return;
            }
            use enigo::{Direction, Key, Keyboard};
            let combo = combo.to_lowercase();
            let parts: Vec<&str> = combo.split('+').collect();
            let mut keys = Vec::new();
            for part in &parts {
                let part = part.trim().trim_end_matches(|c: char| ".,!?".contains(c));
                if let Some(key) = macos_key(part) {
                    keys.push(key);
                } else {
                    warn!("Unknown key: {part}");
                    return;
                }
            }
            let mut enigo = self.enigo.lock().unwrap();
            for &key in &keys {
                let _ = enigo.key(key, Direction::Press);
            }
            for &key in keys.iter().rev() {
                let _ = enigo.key(key, Direction::Release);
            }
        }
    }

    impl Typer for MacOSTyper {
        fn type_text(&self, text: &str) {
            if text.is_empty() {
                return;
            }
            info!("Typing: {text:?}");
            if self.dry_run {
                info!("[DRY RUN] Would type: {text}");
                return;
            }
            use enigo::Keyboard;
            let mut enigo = self.enigo.lock().unwrap();
            let _ = enigo.text(text);
        }

        fn backspace(&self, count: usize) {
            if count == 0 {
                return;
            }
            if self.dry_run {
                info!("[DRY RUN] Would backspace {count} times");
                return;
            }
            use enigo::{Direction, Key, Keyboard};
            let mut enigo = self.enigo.lock().unwrap();
            for _ in 0..count {
                let _ = enigo.key(Key::Backspace, Direction::Click);
            }
        }

        fn send_keys(&self, keys_str: &str) {
            let sequences: Vec<&str> = keys_str.split_whitespace().collect();
            for combo in sequences {
                self.send_key_combo(combo);
            }
        }
    }

    fn macos_key(name: &str) -> Option<enigo::Key> {
        use enigo::Key;
        match name {
            "escape" | "esc" => Some(Key::Escape),
            "backspace" | "back" => Some(Key::Backspace),
            "tab" => Some(Key::Tab),
            "enter" | "return" => Some(Key::Return),
            "space" => Some(Key::Space),
            "delete" | "del" => Some(Key::Delete),
            "ctrl" | "control" => Some(Key::Control),
            "shift" => Some(Key::Shift),
            "alt" | "option" => Some(Key::Alt),
            "super" | "meta" | "command" | "cmd" => Some(Key::Meta),
            "home" => Some(Key::Home),
            "end" => Some(Key::End),
            "pageup" | "pgup" => Some(Key::PageUp),
            "pagedown" | "pgdn" => Some(Key::PageDown),
            "up" => Some(Key::UpArrow),
            "down" => Some(Key::DownArrow),
            "left" => Some(Key::LeftArrow),
            "right" => Some(Key::RightArrow),
            "f1" => Some(Key::F1),
            "f2" => Some(Key::F2),
            "f3" => Some(Key::F3),
            "f4" => Some(Key::F4),
            "f5" => Some(Key::F5),
            "f6" => Some(Key::F6),
            "f7" => Some(Key::F7),
            "f8" => Some(Key::F8),
            "f9" => Some(Key::F9),
            "f10" => Some(Key::F10),
            "f11" => Some(Key::F11),
            "f12" => Some(Key::F12),
            s if s.len() == 1 => {
                let c = s.chars().next().unwrap();
                Some(Key::Unicode(c))
            }
            _ => None,
        }
    }

    /// macOS notifications.
    pub struct MacOSNotifier;

    impl MacOSNotifier {
        pub fn new() -> Self {
            Self
        }
    }

    impl Notifier for MacOSNotifier {
        fn show(&self, message: &str, _persistent: bool) {
            let escaped = message.replace('"', "\\\"").replace('\'', "\\'");
            let script = format!("display notification \"{escaped}\" with title \"Whisper Typer\"");
            let _ = Command::new("osascript")
                .args(["-e", &script])
                .output();
        }

        fn close(&self) {}
    }
}

#[cfg(target_os = "macos")]
pub use imp::*;
