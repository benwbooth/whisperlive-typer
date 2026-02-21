use unicode_segmentation::UnicodeSegmentation;

/// Count grapheme clusters (user-perceived characters) in a string.
pub fn grapheme_len(s: &str) -> usize {
    s.graphemes(true).count()
}

/// Collect grapheme clusters into a Vec.
pub fn grapheme_vec(s: &str) -> Vec<&str> {
    s.graphemes(true).collect()
}
