use mlx_rs as mlx;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// メッセージ内のコンテンツ断片。`Image` は画像スロットで、実画像は
/// `Qwen35Processor::prepare` の `images` 引数と出現順に対応付けられる。
#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
    Image,
}

// HF chat template が期待する形式に合わせる:
//   Text  → {"type": "text", "text": "..."}
//   Image → {"type": "image"}
impl Serialize for ContentPart {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ContentPart::Text(text) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
                map.end()
            }
            ContentPart::Image => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "image")?;
                map.end()
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    /// 画像スロット1つ + テキストの user メッセージ。
    pub fn user_with_image(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::Image, ContentPart::Text(text.into())],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    pub fn image_count(&self) -> usize {
        self.content
            .iter()
            .filter(|part| matches!(part, ContentPart::Image))
            .count()
    }
}

#[derive(Debug, Clone)]
pub struct TemplateOptions {
    pub add_generation_prompt: bool,
    /// None: テンプレート側で未定義扱い (Qwen3.5 では空の <think> ブロックが入る)。
    /// Some(true): <think>\n で生成開始 (thinking モード)。
    pub enable_thinking: Option<bool>,
}

impl Default for TemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: None,
        }
    }
}
