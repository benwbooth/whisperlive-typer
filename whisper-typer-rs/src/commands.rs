use std::collections::HashMap;

/// Result of processing a voice command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub action: CommandAction,
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandAction {
    /// Send key combination(s) (e.g. "ctrl+z", "enter enter").
    Keys,
    /// Type literal text.
    Literal,
    /// Normal dictation — pass through as text.
    None,
}

/// Punctuation/symbol names to characters.
fn punctuation_chars() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("hyphen", "-"),
        ("underscore", "_"),
        ("plus sign", "+"),
        ("equals sign", "="),
        ("semicolon", ";"),
        ("colon", ":"),
        ("comma", ","),
        ("period", "."),
        ("forward slash", "/"),
        ("backslash", "\\"),
        ("pipe", "|"),
        ("apostrophe", "'"),
        ("single quote", "'"),
        ("double quote", "\""),
        ("quotation mark", "\""),
        ("backtick", "`"),
        ("tilde", "~"),
        ("at sign", "@"),
        ("hash sign", "#"),
        ("hashtag", "#"),
        ("dollar sign", "$"),
        ("percent sign", "%"),
        ("caret", "^"),
        ("ampersand", "&"),
        ("asterisk", "*"),
        ("left paren", "("),
        ("open paren", "("),
        ("right paren", ")"),
        ("close paren", ")"),
        ("left bracket", "["),
        ("open bracket", "["),
        ("right bracket", "]"),
        ("close bracket", "]"),
        ("left brace", "{"),
        ("open brace", "{"),
        ("right brace", "}"),
        ("close brace", "}"),
        ("less than", "<"),
        ("greater than", ">"),
        ("question mark", "?"),
        ("exclamation point", "!"),
    ])
}

/// Digit words -> digit characters.
fn digit_words() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
    ])
}

/// Action keys that send key codes, not characters.
fn action_keys() -> std::collections::HashSet<&'static str> {
    [
        "enter", "return", "tab", "escape", "esc", "backspace", "back",
        "delete", "del", "home", "end", "page up", "pageup", "page down", "pagedown",
        "up", "down", "left", "right", "insert", "ins",
        "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    ]
    .into_iter()
    .collect()
}

/// Valid key names (for command validation).
fn valid_keys() -> std::collections::HashSet<&'static str> {
    [
        "escape", "esc", "backspace", "back", "tab", "enter", "return", "space",
        "delete", "del",
        "ctrl", "control", "shift", "alt", "option", "super", "meta", "command",
        "cmd", "windows", "win",
        "home", "end", "pageup", "pgup", "pagedown", "pgdn",
        "up", "down", "left", "right", "insert", "ins",
        "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]
    .into_iter()
    .collect()
}

/// Modifier keys that indicate a key combo.
fn modifier_keys() -> std::collections::HashSet<&'static str> {
    ["ctrl", "control", "alt", "shift", "super", "meta", "windows", "win"]
        .into_iter()
        .collect()
}

/// Returns (default_key_commands, default_literal_commands).
pub fn default_commands() -> (HashMap<String, String>, HashMap<String, String>) {
    let keys = HashMap::from([
        ("scratch that".into(), "ctrl+z".into()),
        ("undo".into(), "ctrl+z".into()),
        ("undo that".into(), "ctrl+z".into()),
        ("redo".into(), "ctrl+shift+z".into()),
        ("redo that".into(), "ctrl+shift+z".into()),
        ("new line".into(), "enter".into()),
        ("newline".into(), "enter".into()),
        ("new paragraph".into(), "enter enter".into()),
        ("go back".into(), "backspace".into()),
        ("delete word".into(), "ctrl+backspace".into()),
        ("delete line".into(), "ctrl+shift+k".into()),
        ("select all".into(), "ctrl+a".into()),
        ("copy".into(), "ctrl+c".into()),
        ("copy that".into(), "ctrl+c".into()),
        ("paste".into(), "ctrl+v".into()),
        ("paste that".into(), "ctrl+v".into()),
        ("cut".into(), "ctrl+x".into()),
        ("cut that".into(), "ctrl+x".into()),
        ("save".into(), "ctrl+s".into()),
        ("save that".into(), "ctrl+s".into()),
    ]);
    let literals = HashMap::new();
    (keys, literals)
}

/// Processes voice commands from transcribed text.
pub struct CommandProcessor {
    command_keys: HashMap<String, String>,
    command_literals: HashMap<String, String>,
    punctuation: HashMap<&'static str, &'static str>,
    digits: HashMap<&'static str, &'static str>,
    action: std::collections::HashSet<&'static str>,
    valid: std::collections::HashSet<&'static str>,
    modifiers: std::collections::HashSet<&'static str>,
}

impl CommandProcessor {
    pub fn new(
        command_keys: HashMap<String, String>,
        command_literals: HashMap<String, String>,
    ) -> Self {
        Self {
            command_keys,
            command_literals,
            punctuation: punctuation_chars(),
            digits: digit_words(),
            action: action_keys(),
            valid: valid_keys(),
            modifiers: modifier_keys(),
        }
    }

    /// Process text to detect voice commands.
    ///
    /// Priority:
    /// 1. "say X" -> type X literally (escape hatch)
    /// 2. Configured key commands (e.g., "scratch that" -> ctrl+z)
    /// 3. Configured literals
    /// 4. Modifier combos (e.g., "control c" -> ctrl+c)
    /// 5. "press X" (e.g., "press enter")
    /// 6. Action keys (e.g., "enter", "backspace")
    /// 7. Punctuation names (e.g., "semicolon" -> ;)
    /// 8. Spelled-out letters/digits (e.g., "A-B-C" -> abc)
    /// 9. Single letters (e.g., "a" -> a)
    /// 10. Digit words (e.g., "one" -> 1)
    /// 11. Normal dictation
    pub fn process(&self, text: &str) -> CommandResult {
        let text_lower = text.trim().trim_end_matches(|c| ".,!?".contains(c)).to_lowercase();
        let text_clean = text.trim().trim_end_matches(|c| ".,!?".contains(c));

        // 1. "say X" escape — type literally
        if let Some(rest) = text_lower.strip_prefix("say ") {
            let _ = rest;
            // Preserve original case: skip "say " from the cleaned text
            let literal = &text_clean[4..];
            return CommandResult {
                action: CommandAction::Literal,
                payload: literal.to_string(),
            };
        }

        // 2. Configured key commands
        if let Some(keys) = self.command_keys.get(&text_lower) {
            return CommandResult {
                action: CommandAction::Keys,
                payload: keys.clone(),
            };
        }

        // 3. Configured literals
        if let Some(literal) = self.command_literals.get(&text_lower) {
            return CommandResult {
                action: CommandAction::Literal,
                payload: literal.clone(),
            };
        }

        // 4. Modifier combo (e.g., "control c", "alt f4")
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        if let Some(&first) = words.first() {
            if self.modifiers.contains(first) {
                return CommandResult {
                    action: CommandAction::Keys,
                    payload: self.normalize_keys(&text_lower),
                };
            }
        }

        // 5. "press X" prefix
        if let Some(key_name) = text_lower.strip_prefix("press ") {
            let key_name = key_name.trim();
            if self.action.contains(key_name) || self.valid.contains(key_name) {
                return CommandResult {
                    action: CommandAction::Keys,
                    payload: self.normalize_keys(key_name),
                };
            }
        }

        // 6. Action keys
        if self.action.contains(text_lower.as_str()) {
            return CommandResult {
                action: CommandAction::Keys,
                payload: self.normalize_keys(&text_lower),
            };
        }

        // 7. Punctuation names
        if let Some(&ch) = self.punctuation.get(text_lower.as_str()) {
            return CommandResult {
                action: CommandAction::Literal,
                payload: ch.to_string(),
            };
        }

        // 8. Spelled-out letters/digits (e.g., "A-B-C" or "A B C")
        if let Some(spelled) = self.parse_spelled_letters(&text_lower) {
            return CommandResult {
                action: CommandAction::None,
                payload: spelled,
            };
        }

        // 9. Single letters
        if text_lower.len() == 1 && text_lower.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
            return CommandResult {
                action: CommandAction::Literal,
                payload: text_lower,
            };
        }

        // 10. Digit words
        if let Some(&digit) = self.digits.get(text_lower.as_str()) {
            return CommandResult {
                action: CommandAction::Literal,
                payload: digit.to_string(),
            };
        }

        // 11. Normal dictation
        CommandResult {
            action: CommandAction::None,
            payload: text.to_string(),
        }
    }

    /// Normalize key names to standard format (e.g., "control c" -> "ctrl+c").
    fn normalize_keys(&self, keys: &str) -> String {
        let keys = keys.trim_end_matches(|c| ".,!?".contains(c));
        let parts: Vec<&str> = keys.split_whitespace().collect();
        let normalized: Vec<String> = parts
            .iter()
            .filter_map(|part| {
                let part = part.trim_end_matches(|c: char| ".,!?".contains(c)).to_lowercase();
                if part.is_empty() {
                    return None;
                }
                let p = match part.as_str() {
                    "control" => "ctrl".to_string(),
                    "windows" | "meta" | "win" => "super".to_string(),
                    _ => part,
                };
                Some(p)
            })
            .collect();
        normalized.join("+")
    }

    /// Parse spelled-out letters/digits like "A-B-C" or "A, B, C" into "abc".
    fn parse_spelled_letters(&self, text: &str) -> Option<String> {
        let parts: Vec<&str> = text
            .split(|c: char| c == '-' || c == ',' || c.is_whitespace())
            .map(|p| p.trim_matches(|c| ".,!?".contains(c)))
            .filter(|p| !p.is_empty())
            .collect();

        if parts.len() < 2 {
            return None;
        }

        let mut result = String::new();
        for part in &parts {
            let p = part.to_lowercase();
            if p.len() == 1 && p.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
                result.push_str(&p);
            } else if let Some(&digit) = self.digits.get(p.as_str()) {
                result.push_str(digit);
            } else if p.len() == 1 && p.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                result.push_str(&p);
            } else {
                return None;
            }
        }

        Some(result)
    }
}
