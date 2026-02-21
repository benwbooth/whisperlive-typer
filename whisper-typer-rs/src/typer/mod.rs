pub mod platform;
pub mod linux;
pub mod macos;
pub mod dry_run;

use std::sync::Mutex;

use tracing::info;

use crate::grapheme::{grapheme_len, grapheme_vec};
use platform::Typer;

/// Tracks what has been typed to enable corrections.
#[derive(Debug, Default)]
pub struct TyperState {
    /// Characters typed that are permanent (never delete before this).
    pub finalized_length: usize,
    /// Current pending text (can change).
    pub pending_text: String,
    /// Last command text executed (to prevent re-execution).
    pub last_command_text: String,
    /// Monotonic timestamp when last command was executed.
    pub last_command_at: f64,
}

/// Wraps a platform Typer with state management for corrections.
pub struct StatefulTyper {
    typer: Box<dyn Typer>,
    pub state: Mutex<TyperState>,
}

impl StatefulTyper {
    pub fn new(typer: Box<dyn Typer>) -> Self {
        Self {
            typer,
            state: Mutex::new(TyperState::default()),
        }
    }

    pub fn type_text(&self, text: &str) {
        self.typer.type_text(text);
    }

    #[allow(dead_code)]
    pub fn backspace(&self, count: usize) {
        self.typer.backspace(count);
    }

    pub fn send_keys(&self, keys_str: &str) {
        self.typer.send_keys(keys_str);
    }

    /// Handle finalized text from server — type it and mark as permanent.
    pub fn handle_finalize(&self, text: &str) {
        let mut state = self.state.lock().unwrap();
        let pending = &state.pending_text;

        if text.is_empty() {
            // Nothing to finalize, just clear pending
            if !pending.is_empty() {
                self.typer.backspace(grapheme_len(pending));
                state.pending_text.clear();
            }
            return;
        }

        // Optimize: if finalized text starts with pending, just type the rest
        if !pending.is_empty() && text.starts_with(pending.as_str()) {
            let remainder = &text[pending.len()..];
            if !remainder.is_empty() {
                info!("Finalizing (appending): {remainder:?}");
                self.typer.type_text(remainder);
            } else {
                info!("Finalizing (already typed): {text:?}");
            }
            state.pending_text.clear();
            state.finalized_length += grapheme_len(text);
            self.typer.type_text(" ");
            state.finalized_length += 1;
            return;
        }

        // Optimize: if pending starts with finalized, backspace the extra
        if !pending.is_empty() && pending.starts_with(text) {
            let extra = grapheme_len(pending) - grapheme_len(text);
            if extra > 0 {
                info!("Finalizing (trimming {extra} graphemes): {text:?}");
                self.typer.backspace(extra);
            } else {
                info!("Finalizing (exact match): {text:?}");
            }
            state.pending_text.clear();
            state.finalized_length += grapheme_len(text);
            self.typer.type_text(" ");
            state.finalized_length += 1;
            return;
        }

        // General case: clear pending and type finalized
        if !pending.is_empty() {
            info!("Clearing pending before finalize: {pending:?}");
            self.typer.backspace(grapheme_len(pending));
            state.pending_text.clear();
        }

        info!("Finalizing: {text:?}");
        self.typer.type_text(text);
        state.finalized_length += grapheme_len(text);
        self.typer.type_text(" ");
        state.finalized_length += 1;
    }

    /// Update pending text using diff/backspace.
    pub fn update_pending(&self, new_text: &str) {
        let mut state = self.state.lock().unwrap();
        let old = &state.pending_text;

        if old == new_text {
            return;
        }

        info!("update_pending: {old:?} -> {new_text:?}");

        if old.is_empty() {
            if !new_text.is_empty() {
                self.typer.type_text(new_text);
                state.pending_text = new_text.to_string();
            }
            return;
        }

        if new_text.is_empty() {
            self.typer.backspace(grapheme_len(old));
            state.pending_text.clear();
            return;
        }

        // Find common prefix (grapheme-aware)
        let old_graphemes = grapheme_vec(old);
        let new_graphemes = grapheme_vec(new_text);
        let mut common_len = 0;
        let min_len = old_graphemes.len().min(new_graphemes.len());
        for i in 0..min_len {
            if old_graphemes[i] == new_graphemes[i] {
                common_len = i + 1;
            } else {
                break;
            }
        }

        let to_delete = old_graphemes.len() - common_len;
        let to_add: String = new_graphemes[common_len..].concat();

        info!(
            "Pending diff: common={common_len}, delete={to_delete}, add={}",
            grapheme_len(&to_add)
        );

        if to_delete > 0 {
            self.typer.backspace(to_delete);
        }

        if !to_add.is_empty() {
            self.typer.type_text(&to_add);
        }

        state.pending_text = new_text.to_string();
    }

    /// Clear pending text without finalizing.
    pub fn clear_pending(&self) {
        let mut state = self.state.lock().unwrap();
        if !state.pending_text.is_empty() {
            self.typer.backspace(grapheme_len(&state.pending_text));
            state.pending_text.clear();
        }
    }
}
