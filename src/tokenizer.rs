use anyhow::Result;

use crate::chat::{Message, TemplateOptions};

/// トークナイザーの抽象 (mlx-swift-lm の Tokenizer プロトコル相当)。
/// デフォルト実装は `tokenizer` feature の [`HfTokenizer`]。
/// `--no-default-features` の利用者は自前実装を [`crate::processor::Qwen35Processor::with_tokenizer`]
/// に注入できる。
pub trait Tokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>>;
    fn decode(&self, tokens: &[i32], skip_special_tokens: bool) -> Result<String>;
    fn token_to_id(&self, token: &str) -> Option<i32>;
    fn apply_chat_template(&self, messages: &[Message], opts: &TemplateOptions) -> Result<String>;
}

#[cfg(feature = "tokenizer")]
pub use hf::{HfTokenizer, render_chat_template};

#[cfg(feature = "tokenizer")]
mod hf {
    use std::path::Path;

    use anyhow::{Context, Result, anyhow, bail};

    use crate::chat::{Message, TemplateOptions};

    /// HF `tokenizers` + minijinja によるデフォルト実装。
    /// chat template は chat_template.jinja、なければ tokenizer_config.json の
    /// `chat_template` フィールドから読む。
    pub struct HfTokenizer {
        inner: tokenizers::Tokenizer,
        chat_template: Option<String>,
    }

    impl HfTokenizer {
        pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
            let dir = dir.as_ref();
            let tokenizer_path = dir.join("tokenizer.json");
            let inner = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Failed to load {}: {e}", tokenizer_path.display()))?;

            let jinja_path = dir.join("chat_template.jinja");
            let chat_template = if jinja_path.exists() {
                Some(
                    std::fs::read_to_string(&jinja_path)
                        .with_context(|| format!("Failed to read {}", jinja_path.display()))?,
                )
            } else {
                chat_template_from_tokenizer_config(dir)
            };

            Ok(Self {
                inner,
                chat_template,
            })
        }

        pub fn inner(&self) -> &tokenizers::Tokenizer {
            &self.inner
        }
    }

    fn chat_template_from_tokenizer_config(dir: &Path) -> Option<String> {
        let json = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
        let value: serde_json::Value = serde_json::from_str(&json).ok()?;
        value
            .get("chat_template")?
            .as_str()
            .map(|s| s.to_string())
    }

    impl super::Tokenizer for HfTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>> {
            let encoding = self
                .inner
                .encode(text, add_special_tokens)
                .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
            Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
        }

        fn decode(&self, tokens: &[i32], skip_special_tokens: bool) -> Result<String> {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            self.inner
                .decode(&ids, skip_special_tokens)
                .map_err(|e| anyhow!("Decoding failed: {e}"))
        }

        fn token_to_id(&self, token: &str) -> Option<i32> {
            self.inner.token_to_id(token).map(|id| id as i32)
        }

        fn apply_chat_template(
            &self,
            messages: &[Message],
            opts: &TemplateOptions,
        ) -> Result<String> {
            let Some(ref template) = self.chat_template else {
                bail!(
                    "No chat template found (chat_template.jinja or tokenizer_config.json \
                     `chat_template` field)"
                );
            };
            render_chat_template(template, messages, opts)
        }
    }

    /// HF 互換の chat template を minijinja でレンダリングする。
    pub fn render_chat_template(
        template: &str,
        messages: &[Message],
        opts: &TemplateOptions,
    ) -> Result<String> {
        let mut env = minijinja::Environment::new();
        // startswith / rstrip / split などの Python メソッド呼び出しを解決する
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_template("chat", template)
            .context("Failed to compile chat template")?;
        let tmpl = env.get_template("chat")?;

        // enable_thinking が None のときはテンプレート側で undefined にする必要がある
        // (`enable_thinking is defined` の分岐があるため context に含めない)
        let ctx = match opts.enable_thinking {
            Some(enable_thinking) => minijinja::context! {
                messages => messages,
                add_generation_prompt => opts.add_generation_prompt,
                enable_thinking => enable_thinking,
            },
            None => minijinja::context! {
                messages => messages,
                add_generation_prompt => opts.add_generation_prompt,
            },
        };

        tmpl.render(ctx).context("Failed to render chat template")
    }
}

// ゴールデンテスト: 期待値は Python jinja2 (transformers のリファレンス実装) で
// resource/chat_template.jinja をレンダリングした結果に固定している。
#[cfg(all(test, feature = "tokenizer"))]
mod tests {
    use super::render_chat_template;
    use crate::chat::{Message, TemplateOptions};

    const TEMPLATE: &str = include_str!("../resource/chat_template.jinja");

    fn opts_thinking(enable_thinking: Option<bool>) -> TemplateOptions {
        TemplateOptions {
            add_generation_prompt: true,
            enable_thinking,
        }
    }

    #[test]
    fn test_golden_text_only() {
        let messages = [
            Message::system("You are a helpful assistant."),
            Message::user("Hello!"),
        ];
        let rendered =
            render_chat_template(TEMPLATE, &messages, &TemplateOptions::default()).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn test_golden_image_and_text() {
        let messages = [
            Message::system("You are a helpful assistant."),
            Message::user_with_image("Describe this image."),
        ];
        let rendered =
            render_chat_template(TEMPLATE, &messages, &TemplateOptions::default()).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn test_golden_no_system() {
        let messages = [Message::user("Hi")];
        let rendered =
            render_chat_template(TEMPLATE, &messages, &TemplateOptions::default()).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn test_golden_thinking_variants() {
        let messages = [Message::user("Hi")];

        let rendered =
            render_chat_template(TEMPLATE, &messages, &opts_thinking(Some(true))).unwrap();
        assert_eq!(rendered, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");

        // enable_thinking=false と未定義 (None) はどちらも空 think ブロック
        for opts in [opts_thinking(Some(false)), opts_thinking(None)] {
            let rendered = render_chat_template(TEMPLATE, &messages, &opts).unwrap();
            assert_eq!(
                rendered,
                "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
        }
    }

    #[test]
    fn test_golden_multi_turn() {
        let messages = [
            Message::user("Hi"),
            Message::assistant("Hello! How can I help?"),
            Message::user("Tell me a joke."),
        ];
        let rendered =
            render_chat_template(TEMPLATE, &messages, &TemplateOptions::default()).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello! How can I help?<|im_end|>\n<|im_start|>user\nTell me a joke.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn test_golden_no_generation_prompt() {
        let messages = [Message::user("Hi")];
        let opts = TemplateOptions {
            add_generation_prompt: false,
            enable_thinking: None,
        };
        let rendered = render_chat_template(TEMPLATE, &messages, &opts).unwrap();
        assert_eq!(rendered, "<|im_start|>user\nHi<|im_end|>\n");
    }

    #[test]
    fn test_system_with_image_is_rejected() {
        let messages = [Message {
            role: crate::chat::Role::System,
            content: vec![crate::chat::ContentPart::Image],
        }];
        let result = render_chat_template(TEMPLATE, &messages, &TemplateOptions::default());
        assert!(result.is_err(), "system message with image should raise");
    }
}
