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

use std::path::Path;
use std::sync::Arc;

use tokenizers::tokenizer::pre_tokenizer::OffsetType;
use tokenizers::{Model, OffsetReferential, PreTokenizer, Tokenizer};

/// Longest tail held back waiting for a pre-token boundary. A boundary-free run (a long CJK run
/// without punctuation, base64, a hash) would otherwise be held, and re-normalized on every
/// push, until the stream ends. Past this length the tail is emitted as if the stream ended
/// there, so at most one pre-token per cap can differ from encoding the whole text at once.
pub const MAX_HELD_BYTES: usize = 4096;

pub struct Retokenizer {
    /// Shared so a worker loads `tokenizer.json` once and creates one retokenizer per stream.
    tokenizer: Arc<Tokenizer>,
    /// Full text received so far, so a safe restart can be re-encoded.
    text: String,
    /// Ids emitted so far, for callers that want to inspect them.
    ids: Vec<u32>,
    /// Byte offset in `text` from which the next encode restarts; everything before it is
    /// already represented in `ids` and cannot change.
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
        Self {
            tokenizer,
            text: String::new(),
            ids: Vec::new(),
            restart_byte: 0,
        }
    }

    /// All ids returned so far, in order.
    pub fn emitted_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Append streamed text; return ids for text that is now stable.
    pub fn push(&mut self, text: &str) -> Vec<u32> {
        self.text.push_str(text);
        self.emit(false)
    }

    /// Flush the held-back tail at stream end.
    pub fn finish(&mut self) -> Vec<u32> {
        self.emit(true)
    }

    /// Encode the unemitted tail's complete pre-tokens. `flush` also emits the last, possibly
    /// incomplete one.
    fn emit(&mut self, flush: bool) -> Vec<u32> {
        let tail = &self.text[self.restart_byte..];
        if tail.is_empty() {
            return Vec::new();
        }

        // Replicate `Tokenizer::encode`'s normalization and pre-tokenization so the boundaries
        // are exactly the tokenizer's own; added tokens are extracted as `encode` would.
        let mut pretokenized = self
            .tokenizer
            .get_added_vocabulary()
            .extract_and_normalize(self.tokenizer.get_normalizer(), tail);
        if let Some(pretokenizer) = self.tokenizer.get_pre_tokenizer() {
            pretokenizer
                .pre_tokenize(&mut pretokenized)
                .expect("pre-tokenizing valid UTF-8 cannot fail");
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

        let model = self.tokenizer.get_model();
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
                        .expect("tokenizing valid UTF-8 cannot fail")
                    {
                        emitted.push(token.id);
                    }
                }
            }
            emitted_end = offsets.1;
        }
        if emitted.is_empty() {
            return Vec::new();
        }

        self.restart_byte = if flush {
            self.text.len()
        } else {
            self.restart_byte + emitted_end
        };
        self.ids.extend_from_slice(&emitted);
        emitted
    }
}
