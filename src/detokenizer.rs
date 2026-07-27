use crate::tokenizer::Tokenizer;

/// トークンを逐次 decode するストリーミングデトークナイザ。
/// マルチバイト文字がトークン境界で割れている間 (decode 結果が U+FFFD で終わる間) は
/// バッファし、確定した時点でまとめて返す。
pub struct StreamingDetokenizer<'a> {
    tokenizer: &'a dyn Tokenizer,
    pending: Vec<i32>,
}

impl<'a> StreamingDetokenizer<'a> {
    pub fn new(tokenizer: &'a dyn Tokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::new(),
        }
    }

    /// トークンを追加し、確定したテキスト断片があれば返す。
    pub fn push(&mut self, token: i32) -> Option<String> {
        self.pending.push(token);
        let decoded = self.tokenizer.decode(&self.pending, true).ok()?;
        if decoded.is_empty() || decoded.ends_with('\u{FFFD}') {
            return None;
        }
        self.pending.clear();
        Some(decoded)
    }

    /// 生成終了時に未確定分を吐き出す (不完全なマルチバイトは U+FFFD 込みで出す)。
    pub fn finish(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let decoded = self.tokenizer.decode(&self.pending, true).ok()?;
        self.pending.clear();
        if decoded.is_empty() { None } else { Some(decoded) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Message, TemplateOptions};

    /// 各トークン ID を UTF-8 の 1 バイトとして decode するモック。
    /// トークン境界でマルチバイト文字が割れるケースを再現できる。
    struct ByteTokenizer;

    impl Tokenizer for ByteTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Vec<i32>> {
            Ok(text.bytes().map(|b| b as i32).collect())
        }

        fn decode(&self, tokens: &[i32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn token_to_id(&self, _token: &str) -> Option<i32> {
            None
        }

        fn apply_chat_template(
            &self,
            _messages: &[Message],
            _opts: &TemplateOptions,
        ) -> anyhow::Result<String> {
            unimplemented!()
        }
    }

    #[test]
    fn test_ascii_streams_per_token() {
        let tokenizer = ByteTokenizer;
        let mut detok = StreamingDetokenizer::new(&tokenizer);
        assert_eq!(detok.push(b'H' as i32), Some("H".to_string()));
        assert_eq!(detok.push(b'i' as i32), Some("i".to_string()));
        assert_eq!(detok.finish(), None);
    }

    #[test]
    fn test_multibyte_buffered_until_complete() {
        let tokenizer = ByteTokenizer;
        let mut detok = StreamingDetokenizer::new(&tokenizer);
        // "日" = E6 97 A5: 途中のバイトでは何も出ず、最後のバイトで確定する
        let bytes = "日".as_bytes();
        assert_eq!(detok.push(bytes[0] as i32), None);
        assert_eq!(detok.push(bytes[1] as i32), None);
        assert_eq!(detok.push(bytes[2] as i32), Some("日".to_string()));
    }

    #[test]
    fn test_emoji_and_text_mixed() {
        let tokenizer = ByteTokenizer;
        let mut detok = StreamingDetokenizer::new(&tokenizer);
        let text = "a🎉日b";
        let mut chunks = String::new();
        for &b in text.as_bytes() {
            if let Some(chunk) = detok.push(b as i32) {
                chunks.push_str(&chunk);
            }
        }
        if let Some(rest) = detok.finish() {
            chunks.push_str(&rest);
        }
        assert_eq!(chunks, text);
    }

    #[test]
    fn test_finish_flushes_incomplete() {
        let tokenizer = ByteTokenizer;
        let mut detok = StreamingDetokenizer::new(&tokenizer);
        // 不完全なマルチバイト (E6 のみ) は finish で U+FFFD として出る
        assert_eq!(detok.push(0xE6), None);
        assert_eq!(detok.finish(), Some("\u{FFFD}".to_string()));
        assert_eq!(detok.finish(), None);
    }
}
