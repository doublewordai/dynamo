// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streamed text back into token ids, with the model's own tokenizer.
//!
//! Dynamo's frontend counts usage and migrates requests using the token ids a worker returns,
//! so the proxy returns real ids for the text it emits. Ids already emitted are never revised.
//!
//! A byte-level BPE tokenizer first splits text into pre-tokens with a regex and then encodes
//! each pre-token independently. Appending text therefore cannot change the tokens of an
//! earlier pre-token: once a later pre-token has started, the boundary before it is safe. On
//! each `push` the retokenizer runs the tokenizer's own normalization and pre-tokenizer over
//! the unemitted tail, encodes every pre-token but the last, and holds the last one (the
//! current partial word, CJK run, number or URL) until more text arrives or `finish` is
//! called. Surface forms without spaces (`中文字`, `https://…`) thus stream as soon as the
//! pre-tokenizer finds a boundary, instead of waiting for the whole stream.
//!
//! The pre-tokens are encoded individually rather than by encoding a truncated window: the
//! tokenizer's whitespace regex attaches trailing spaces differently at the end of the input,
//! so a window cut at a pre-token boundary can tokenize its last pre-token differently from
//! the full text.
//!
//! A tiktoken model (Kimi K3 ships `tiktoken.model`, no `tokenizer.json`) is loaded by Dynamo's
//! own loader, the one the frontend uses for the same card. That loader encodes and decodes but
//! does not expose its pre-tokenizer, so the retokenizer holds text back to the last position
//! that is a pre-token boundary of Kimi's regex whatever follows ([`kimi_boundary`]) and encodes
//! the text before it whole.

use std::path::Path;
use std::sync::Arc;

use tokenizers::tokenizer::pre_tokenizer::OffsetType;
use tokenizers::{Model, OffsetReferential, PreTokenizer, Tokenizer};

/// Longest tail held back waiting for a pre-token boundary. A boundary-free run (a long CJK run
/// without punctuation, base64, a hash) would otherwise be held, and re-normalized on every
/// push, until the stream ends. Past this length the tail is emitted as if the stream ended
/// there, so at most one pre-token per cap can differ from encoding the whole text at once.
pub const MAX_HELD_BYTES: usize = 4096;

/// Compact the already-tokenized prefix of the buffer once it grows past this many bytes.
/// Everything before the restart point is already represented in the ids returned to the
/// caller, so retaining it only grows memory for the whole stream. The threshold is a
/// multiple of [`MAX_HELD_BYTES`] so the held tail and the retained prefix stay bounded.
pub const COMPACT_AT_BYTES: usize = 2 * MAX_HELD_BYTES;

/// The tokenizer rejected provider text. Provider output is untrusted, so this is a per-stream
/// failure the engine reports as migratable, never a panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("retokenizing provider output failed: {0}")]
pub struct RetokenizeError(pub String);

/// Ids emitted by one `push` or `finish`, and exactly the text they encode. Text and ids are
/// held back together, so a consumer that streams `text` never shows text whose ids it has
/// not yet counted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Emitted {
    pub text: String,
    pub ids: Vec<u32>,
}

/// The model's tokenizer, loaded once per worker and shared by every stream.
#[derive(Clone)]
pub enum ModelTokenizer {
    /// A Hugging Face `tokenizer.json`.
    HuggingFace(Arc<Tokenizer>),
    /// A tiktoken model, as Dynamo's tiktoken loader builds it.
    TikToken(dynamo_tokenizers::Tokenizer),
}

impl ModelTokenizer {
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer
                .decode(ids, skip_special_tokens)
                .map_err(|err| anyhow::anyhow!("decode: {err}")),
            Self::TikToken(tokenizer) => Ok(tokenizer.decode(ids, skip_special_tokens)?.into()),
        }
    }
}

pub struct Retokenizer {
    /// Shared so a worker loads its tokenizer once and creates one retokenizer per stream.
    tokenizer: ModelTokenizer,
    /// Text received since the last compaction, so a safe restart can be re-encoded.
    text: String,
    /// Byte offset in `text` from which the next encode restarts; everything before it is
    /// already emitted and cannot change.
    restart_byte: usize,
}

impl Retokenizer {
    /// Load `tokenizer.json`.
    pub fn from_file(tokenizer_json: &Path) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_json).map_err(|err| {
            anyhow::anyhow!(
                "failed to load tokenizer {}: {err}",
                tokenizer_json.display()
            )
        })?;
        Ok(Self::new(tokenizer))
    }

    /// Wrap an already-built tokenizer, mainly for tests that do not want a file fixture.
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self::with_shared(Arc::new(tokenizer))
    }

    /// Start a stream with a tokenizer shared across streams.
    pub fn with_shared(tokenizer: Arc<Tokenizer>) -> Self {
        Self::with_model(ModelTokenizer::HuggingFace(tokenizer))
    }

    /// Start a stream with either kind of model tokenizer.
    pub fn with_model(tokenizer: ModelTokenizer) -> Self {
        Self {
            tokenizer,
            text: String::new(),
            restart_byte: 0,
        }
    }

    /// Append streamed text; return ids for text that is now stable.
    pub fn push(&mut self, text: &str) -> Result<Vec<u32>, RetokenizeError> {
        self.push_with_text(text).map(|emitted| emitted.ids)
    }

    /// Flush the held-back tail at stream end.
    pub fn finish(&mut self) -> Result<Vec<u32>, RetokenizeError> {
        self.finish_with_text().map(|emitted| emitted.ids)
    }

    /// Append streamed text; return the now-stable text together with its ids. The held-back
    /// tail's text is returned by a later call, with its ids.
    pub fn push_with_text(&mut self, text: &str) -> Result<Emitted, RetokenizeError> {
        self.text.push_str(text);
        self.emit(false)
    }

    /// Flush the held-back tail at stream end, with its text.
    pub fn finish_with_text(&mut self) -> Result<Emitted, RetokenizeError> {
        self.emit(true)
    }

    /// Encode the unemitted tail's complete pre-tokens. `flush` also emits the last, possibly
    /// incomplete one.
    fn emit(&mut self, flush: bool) -> Result<Emitted, RetokenizeError> {
        let tokenizer = match &self.tokenizer {
            ModelTokenizer::HuggingFace(tokenizer) => tokenizer.clone(),
            ModelTokenizer::TikToken(tokenizer) => {
                let tokenizer = tokenizer.clone();
                return self.emit_tiktoken(&tokenizer, flush);
            }
        };
        let tail = &self.text[self.restart_byte..];
        if tail.is_empty() {
            return Ok(Emitted::default());
        }

        // Replicate `Tokenizer::encode`'s normalization and pre-tokenization so the boundaries
        // are exactly the tokenizer's own; added tokens are extracted as `encode` would.
        let mut pretokenized = tokenizer
            .get_added_vocabulary()
            .extract_and_normalize(tokenizer.get_normalizer(), tail);
        if let Some(pretokenizer) = tokenizer.get_pre_tokenizer() {
            pretokenizer
                .pre_tokenize(&mut pretokenized)
                .map_err(|err| RetokenizeError(format!("pre-tokenize: {err}")))?;
        }
        let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte);

        // Emit every complete pre-token except the last. If the last pre-token begins with
        // whitespace, it was formed by the run's length (the whitespace regex leaves the final
        // space to the following word), so hold the whole run: re-tokenizing from the last
        // space alone would split the run differently. `emit_end` is that restart point.
        let last_start = splits
            .last()
            .map(|(_, offsets, _)| offsets.0)
            .unwrap_or(tail.len());
        let emit_end = if flush {
            tail.len()
        } else if tail[last_start..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
        {
            let mut start = last_start;
            while start > 0 {
                let previous = tail[..start].chars().next_back().expect("non-empty");
                if previous.is_whitespace() {
                    start -= previous.len_utf8();
                } else {
                    break;
                }
            }
            start
        } else {
            last_start
        };
        // Bound the held-back tail; see `MAX_HELD_BYTES`.
        let flush = flush || tail.len() - emit_end > MAX_HELD_BYTES;
        let emit_end = if flush { tail.len() } else { emit_end };

        let model = tokenizer.get_model();
        let mut emitted = Vec::new();
        // End of the last split actually emitted, relative to `tail`. Restarting there keeps
        // the restart on a real pre-token boundary: `emit_end` can fall inside a split (a
        // trailing whitespace run that begins mid-split, e.g. the `\n` of a `",\n"`
        // punctuation pre-token), and restarting at `emit_end` would drop the bytes between
        // the split start and `emit_end`.
        let mut emitted_end = 0usize;
        for (split, offsets, added) in &splits {
            if !flush && offsets.1 > emit_end {
                break;
            }
            // Added/special tokens already carry their id; only ordinary pre-tokens are run
            // through the model, exactly as `Tokenizer::do_tokenize` does.
            match added {
                Some(tokens) => emitted.extend(tokens.iter().map(|token| token.id)),
                None => {
                    // The model encodes each pre-token on its own, so complete pre-tokens are
                    // stable.
                    for token in model
                        .tokenize(split)
                        .map_err(|err| RetokenizeError(format!("tokenize: {err}")))?
                    {
                        emitted.push(token.id);
                    }
                }
            }
            emitted_end = offsets.1;
        }
        if emitted.is_empty() && !flush {
            return Ok(Emitted::default());
        }
        // Offsets are in the original text, so this is the provider text the ids were encoded
        // from. They decode back to it exactly when the normalizer preserves text, as it does
        // for the pinned families (none, or NFC on already-composed text; the real-tokenizer
        // tests assert the round trip). A lossy normalizer (lowercasing, say) would differ, as
        // the ids alone always did.
        let text = if flush {
            tail.to_string()
        } else {
            tail[..emitted_end].to_string()
        };

        self.restart_byte = if flush {
            self.text.len()
        } else {
            self.restart_byte + emitted_end
        };
        self.compact();
        Ok(Emitted { text, ids: emitted })
    }

    /// Encode the unemitted tail up to its last [`kimi_boundary`] (or all of it on `flush`, or
    /// once the held text passes [`MAX_HELD_BYTES`]). The encoded text starts and ends on
    /// pre-token boundaries outside any special token, so its ids are exactly the ids the whole
    /// stream's encoding has there.
    fn emit_tiktoken(
        &mut self,
        tokenizer: &dynamo_tokenizers::Tokenizer,
        flush: bool,
    ) -> Result<Emitted, RetokenizeError> {
        let tail = &self.text[self.restart_byte..];
        let mut end = 0;
        let mut previous = None;
        for (offset, c) in tail.char_indices() {
            if previous.is_some_and(|before| kimi_boundary(before, c)) {
                end = offset;
            }
            previous = Some(c);
        }
        if flush || tail.len() - end > MAX_HELD_BYTES {
            end = tail.len();
        }
        if end == 0 {
            return Ok(Emitted::default());
        }
        let text = tail[..end].to_string();
        let ids = tokenizer
            .encode(&text)
            .map_err(|err| RetokenizeError(format!("tokenize: {err}")))?
            .token_ids()
            .to_vec();
        self.restart_byte += end;
        self.compact();
        Ok(Emitted { text, ids })
    }

    /// Drop the already-tokenized prefix once it passes [`COMPACT_AT_BYTES`], adjusting the
    /// restart offset. Bytes before the restart point are already emitted and never re-read.
    fn compact(&mut self) {
        if self.restart_byte >= COMPACT_AT_BYTES {
            self.text.drain(..self.restart_byte);
            self.restart_byte = 0;
        }
    }
}

/// CJK sentence punctuation (all `\p{Po}`): not whitespace, letter, number or mark.
const CJK_PUNCTUATION: &[char] = &['，', '。', '！', '？', '；', '：', '、'];

/// Whether a pre-token of Kimi's tiktoken regex (the only pattern Dynamo's tiktoken loader
/// supports) ends between `before` and `after` whatever text follows. The regex is
///
/// ```text
/// [\p{Han}]+ | [^\r\n\p{L}\p{N}]?<non-Han letters>+('s|…)? (two orderings of case) | \p{N}{1,3}
///   | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
/// ```
///
/// No alternative continues a pre-token from a non-whitespace character into a following
/// whitespace character other than `\r`/`\n`, or from a newline into a following
/// non-whitespace character. A Han character is matched only by `[\p{Han}]+`, and no
/// alternative joins it to an ASCII character or CJK punctuation on either side. None of these
/// pairs can sit inside a special token (ASCII without whitespace), and the regex has no
/// lookbehind, so the text after the boundary encodes on its own exactly as it does in place.
pub fn kimi_boundary(before: char, after: char) -> bool {
    let han = |c: char| ('\u{4E00}'..='\u{9FFF}').contains(&c);
    let plain = |c: char| (c.is_ascii() && !c.is_whitespace()) || CJK_PUNCTUATION.contains(&c);
    (!before.is_whitespace() && after.is_whitespace() && !matches!(after, '\r' | '\n'))
        || (matches!(before, '\r' | '\n') && !after.is_whitespace())
        || (han(before) && plain(after))
        || (plain(before) && han(after))
}
