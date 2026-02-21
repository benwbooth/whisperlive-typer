use tracing::info;

use super::platform::{Notifier, Typer};

/// Logging-only Typer that doesn't actually type.
pub struct DryRunTyper;

impl Typer for DryRunTyper {
    fn type_text(&self, text: &str) {
        info!("[DRY RUN] type_text: {text:?}");
    }

    fn backspace(&self, count: usize) {
        info!("[DRY RUN] backspace: {count}");
    }

    fn send_keys(&self, keys_str: &str) {
        info!("[DRY RUN] send_keys: {keys_str:?}");
    }
}

/// Logging-only Notifier.
#[allow(dead_code)]
pub struct DryRunNotifier;

impl Notifier for DryRunNotifier {
    fn show(&self, message: &str, _persistent: bool) {
        info!("[DRY RUN] notification: {message}");
    }

    fn close(&self) {
        info!("[DRY RUN] notification closed");
    }
}
