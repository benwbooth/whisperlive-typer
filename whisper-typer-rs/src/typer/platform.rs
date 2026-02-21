/// Trait for keyboard input simulation.
pub trait Typer: Send + Sync {
    /// Type text as if from keyboard.
    fn type_text(&self, text: &str);
    /// Send backspace key presses.
    fn backspace(&self, count: usize);
    /// Send key combination(s) like "ctrl+c", "enter enter".
    fn send_keys(&self, keys_str: &str);
}

/// Trait for desktop notifications.
pub trait Notifier: Send + Sync {
    /// Show a notification. If persistent, it stays until manually closed.
    fn show(&self, message: &str, persistent: bool);
    /// Close the current notification.
    #[allow(dead_code)]
    fn close(&self);
}
