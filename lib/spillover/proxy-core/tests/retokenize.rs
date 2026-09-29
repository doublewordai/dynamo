// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retokenizer stability and round-trip against a tiny local tokenizer.
//!
//! The tokenizer is built in-process (no downloads) as a byte-level BPE with a handful of
//! merges, which is enough to exercise boundary shifts, whitespace runs and multi-byte UTF-8.

use dw_proxy_core::retokenize::Retokenizer;
use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeBuilder, Merges, Vocab};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

/// The GPT-2 byte-to-unicode mapping used by `ByteLevel`.
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bytes: Vec<u8> = (b'!'..=b'~').collect();
    bytes.extend(b'\xA1'..=b'\xAC');
    bytes.extend(b'\xAE'..=b'\xFF');
    let mut chars: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    let mut extra = 0u32;
    for byte in 0..=255u8 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(0x100 + extra);
            extra += 1;
        }
    }
    bytes
        .into_iter()
        .zip(chars.into_iter().map(|c| char::from_u32(c).unwrap()))
        .collect()
}

fn add_merge(vocab: &mut Vocab, merges: &mut Merges, next_id: &mut u32, left: &str, right: &str) {
    let merged = format!("{left}{right}");
    merges.push((left.to_string(), right.to_string()));
    vocab.entry(merged).or_insert_with(|| {
        let id = *next_id;
        *next_id += 1;
        id
    });
}

fn test_tokenizer() -> Tokenizer {
    test_tokenizer_with_prefix_space(false)
}

fn test_tokenizer_with_prefix_space(add_prefix_space: bool) -> Tokenizer {
    let map = bytes_to_unicode();
    let mut vocab: Vocab = Vocab::new();
    for (id, (_, ch)) in map.iter().enumerate() {
        vocab.insert(ch.to_string(), id as u32);
    }
    let ch = |byte: u8| map[byte as usize].1.to_string();
    let space = ch(b' ');

    let mut merges: Merges = Vec::new();
    let mut next_id = 256u32;
    // Hierarchical merges so that a partially-arrived word changes tokenization.
    add_merge(&mut vocab, &mut merges, &mut next_id, "h", "e");
    add_merge(&mut vocab, &mut merges, &mut next_id, "he", "l");
    add_merge(&mut vocab, &mut merges, &mut next_id, "hel", "l");
    add_merge(&mut vocab, &mut merges, &mut next_id, "hell", "o");
    add_merge(&mut vocab, &mut merges, &mut next_id, "w", "o");
    add_merge(&mut vocab, &mut merges, &mut next_id, "wo", "r");
    add_merge(&mut vocab, &mut merges, &mut next_id, "wor", "l");
    add_merge(&mut vocab, &mut merges, &mut next_id, "worl", "d");
    add_merge(&mut vocab, &mut merges, &mut next_id, &space, "hello");
    add_merge(&mut vocab, &mut merges, &mut next_id, &space, "world");

    let model = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("test BPE must build");
    let mut tokenizer = Tokenizer::new(model);
    let byte_level = ByteLevel::default().add_prefix_space(add_prefix_space);
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_post_processor(Some(
        ByteLevel::default().add_prefix_space(add_prefix_space),
    ));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    tokenizer
}

fn one_shot(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .expect("test tokenizer must encode")
        .get_ids()
        .to_vec()
}

/// Feed `text` in `chunks` and return the emitted ids plus the flushed tail.
fn stream(tokenizer: &Tokenizer, chunks: &[&str]) -> Vec<u32> {
    let mut retokenizer = Retokenizer::new(tokenizer.clone());
    let mut ids = Vec::new();
    for chunk in chunks {
        ids.extend(retokenizer.push(chunk));
    }
    ids.extend(retokenizer.finish());
    ids
}

fn assert_matches_one_shot(tokenizer: &Tokenizer, text: &str, chunks: &[&str]) {
    let got = stream(tokenizer, chunks);
    let want = one_shot(tokenizer, text);
    assert_eq!(got, want, "ids differ for {text:?} split as {chunks:?}");
    let decoded = tokenizer.decode(&got, false).unwrap();
    assert_eq!(decoded, text, "decoded text differs for {chunks:?}");
}

/// A small deterministic xorshift so tests need no extra dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

fn random_text(rng: &mut Rng, max_len: usize) -> String {
    const POOL: &[&str] = &[
        "a",
        "b",
        "c",
        "h",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
        " ",
        "  ",
        "   ",
        "\t",
        "\n",
        ".",
        ",",
        "!",
        "1",
        "中",
        "文",
        "日本語",
        "\u{3000}",
        "🙂",
        "🚀",
        "é",
        "ß",
    ];
    let len = rng.below(max_len + 1);
    let mut text = String::new();
    for _ in 0..len {
        text.push_str(POOL[rng.below(POOL.len())]);
    }
    text
}

/// Split at char boundaries only: `push` takes `&str`, so a split can never land inside a
/// UTF-8 sequence. Emoji and CJK still move between pushes at their boundaries.
fn random_chunks<'a>(rng: &mut Rng, text: &'a str) -> Vec<&'a str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    for (idx, _) in text.char_indices().skip(1) {
        if rng.below(3) == 0 {
            chunks.push(&text[start..idx]);
            start = idx;
        }
    }
    chunks.push(&text[start..]);
    chunks
}

#[test]
fn empty_stream_is_empty() {
    let tokenizer = test_tokenizer();
    let mut retokenizer = Retokenizer::new(tokenizer.clone());
    assert!(retokenizer.push("").is_empty());
    assert!(retokenizer.finish().is_empty());
    assert!(retokenizer.emitted_ids().is_empty());
}

#[test]
fn exact_single_push_matches_one_shot() {
    let tokenizer = test_tokenizer();
    assert_matches_one_shot(&tokenizer, "", &[""]);
    assert_matches_one_shot(&tokenizer, "hello world", &["hello world"]);
    assert_matches_one_shot(&tokenizer, "hello world ", &["hello world "]);
}

#[test]
fn every_split_of_sample_texts_matches_one_shot() {
    let tokenizer = test_tokenizer();
    for text in [
        "hello world",
        "hello",
        " hello ",
        "a b c d e",
        "a  b",
        "x\ty\nz",
        "🙂🚀",
        "中文字",
        "héllo wörld",
        "hello, world!",
    ] {
        let chars: Vec<char> = text.chars().collect();
        // Enumerate every way to cut at char boundaries (2^(n-1) splits).
        let splits = 1usize << chars.len().saturating_sub(1).min(12);
        for mask in 0..splits {
            let mut chunks: Vec<String> = Vec::new();
            let mut current = String::new();
            for (i, ch) in chars.iter().enumerate() {
                current.push(*ch);
                if i + 1 < chars.len() && (mask >> i) & 1 == 1 {
                    chunks.push(std::mem::take(&mut current));
                }
            }
            chunks.push(current);
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            assert_matches_one_shot(&tokenizer, text, &refs);
        }
    }
}

#[test]
fn random_texts_and_random_splits_match_one_shot() {
    let tokenizer = test_tokenizer();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..500 {
        let text = random_text(&mut rng, 120);
        let chunks = random_chunks(&mut rng, &text);
        let got = stream(&tokenizer, &chunks);
        let want = one_shot(&tokenizer, &text);
        assert_eq!(
            got, want,
            "case {case} differs for {text:?} chunks {chunks:?}"
        );
        assert_eq!(tokenizer.decode(&got, false).unwrap(), text);
    }
}

#[test]
fn very_long_push_matches_one_shot() {
    let tokenizer = test_tokenizer();
    let text = "hello world ".repeat(20_000);
    assert_matches_one_shot(&tokenizer, &text, &[&text]);
}

#[test]
fn long_stream_in_small_pieces_matches_one_shot() {
    let tokenizer = test_tokenizer();
    let text = "hello world, 中文字 🙂 ".repeat(2_000);
    // Cut at char boundaries, seven bytes-ish at a time.
    let mut safe: Vec<&str> = Vec::new();
    let mut start = 0;
    for (i, _) in text.char_indices() {
        if i - start >= 7 {
            safe.push(&text[start..i]);
            start = i;
        }
    }
    safe.push(&text[start..]);
    assert_matches_one_shot(&tokenizer, &text, &safe);
}

#[test]
fn emitted_ids_never_revise() {
    let tokenizer = test_tokenizer();
    let text = "hello world hello world, 中文字 🙂";
    let chunks = [
        "hel", "lo ", "wor", "ld ", "hel", "lo ", "wor", "ld, ", "中", "文", "字", " ", "🙂",
    ];
    let want = one_shot(&tokenizer, text);
    let mut retokenizer = Retokenizer::new(tokenizer.clone());
    let mut emitted = Vec::new();
    for chunk in chunks {
        emitted.extend(retokenizer.push(chunk));
        assert_eq!(
            emitted.as_slice(),
            &want[..emitted.len()],
            "already emitted ids were revised after {chunk:?}"
        );
        assert_eq!(retokenizer.emitted_ids(), emitted.as_slice());
    }
    emitted.extend(retokenizer.finish());
    assert_eq!(emitted, want);
}

#[test]
fn holds_back_until_a_safe_boundary_then_flushes() {
    let tokenizer = test_tokenizer();
    let mut retokenizer = Retokenizer::new(tokenizer.clone());

    // A single partial word has no safe boundary, so nothing is emitted yet.
    assert!(retokenizer.push("hel").is_empty());

    // Enough complete words produce stable ids before the stream ends.
    let ids = retokenizer.push("lo world hello world hello world");
    assert!(!ids.is_empty());
    let flush = retokenizer.finish();
    let all: Vec<u32> = ids.into_iter().chain(flush).collect();
    assert_eq!(
        all,
        one_shot(&tokenizer, "hello world hello world hello world")
    );
}

#[test]
fn prefix_space_tokenizer_matches_ids() {
    // Production tokenizer.json files often add a leading space; ids must still match the
    // one-shot encoding even though decoding then yields that leading space too.
    let tokenizer = test_tokenizer_with_prefix_space(true);
    let text = "hello world hello world";
    let got = stream(
        &tokenizer,
        &["hel", "lo ", "wor", "ld ", "hel", "lo ", "wor", "ld"],
    );
    assert_eq!(got, one_shot(&tokenizer, text));
    assert_eq!(tokenizer.decode(&got, false).unwrap(), format!(" {text}"));
}

#[test]
fn from_file_matches_in_memory_build() {
    let tokenizer = test_tokenizer();
    let dir = std::env::temp_dir().join(format!("dw-proxy-core-retok-file-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer.json");
    tokenizer.save(&path, false).unwrap();

    let mut from_file = Retokenizer::from_file(&path).unwrap();
    let mut in_memory = Retokenizer::new(tokenizer.clone());
    let chunks = ["hel", "lo ", "wor", "ld ", "🙂", "中", "文"];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for chunk in chunks {
        a.extend(from_file.push(chunk));
        b.extend(in_memory.push(chunk));
    }
    a.extend(from_file.finish());
    b.extend(in_memory.finish());
    assert_eq!(a, b);
    assert_eq!(a, one_shot(&tokenizer, "hello world 🙂中文"));
}

/// Production tokenizer families. The files are downloaded by `scripts/fetch-tokenizers.sh`
/// and gitignored; the tests below skip with a message when a family is missing.
const FAMILIES: &[(&str, &str)] = &[
    ("glm-5", "zai-org/GLM-5.3"),
    ("deepseek-v4", "deepseek-ai/DeepSeek-V4.1-Flash"),
    ("qwen3", "Qwen/Qwen3-8B"),
];

fn fixture_tokenizer(family: &str) -> Option<Tokenizer> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tokenizers")
        .join(family)
        .join("tokenizer.json");
    if !path.exists() {
        eprintln!(
            "skipping {family}: {} missing; run scripts/fetch-tokenizers.sh",
            path.display()
        );
        return None;
    }
    Some(Tokenizer::from_file(&path).expect("fixture tokenizer must load"))
}

/// Mixed English, Chinese, code, emoji, URLs and numbers, the shapes whose pre-tokens do not
/// end at a space.
const MIXED_POOL: &[&str] = &[
    "hello",
    " ",
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "\n",
    "\t",
    "  ",
    "中文",
    "测试",
    "你好",
    "日本語",
    "汉字",
    "今天",
    "天气",
    "fn",
    "main",
    "(",
    ")",
    "{",
    "}",
    "=>",
    "let",
    "x",
    "_",
    "self",
    "::",
    "pub",
    "struct",
    "Result",
    "?",
    ";",
    "\"",
    "//",
    "0x1f",
    "i32",
    "🙂",
    "🚀",
    "👍",
    "❤",
    "https://",
    "example.com",
    "/path",
    "?q=",
    "&x=1",
    "#frag",
    "123",
    "4567",
    "0.5",
    "9",
    "1,000",
    ".",
    ",",
    "!",
    ":",
    "-",
    "'s",
    "'t",
];

fn random_mixed_text(rng: &mut Rng, max_len: usize) -> String {
    let len = rng.below(max_len + 1);
    let mut text = String::new();
    for _ in 0..len {
        text.push_str(MIXED_POOL[rng.below(MIXED_POOL.len())]);
    }
    text
}

/// Real tokenizers must match a one-shot encode for arbitrary mixed text split into arbitrary
/// chunks, and the ids must decode back to the exact text.
#[test]
fn real_tokenizers_match_one_shot_on_random_mixed_text() {
    let mut loaded = 0;
    for (family, repo) in FAMILIES {
        let Some(tokenizer) = fixture_tokenizer(family) else {
            continue;
        };
        loaded += 1;
        let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ (repo.len() as u64));
        for case in 0..40 {
            let text = random_mixed_text(&mut rng, 150);
            let chunks = random_chunks(&mut rng, &text);
            let got = stream(&tokenizer, &chunks);
            let want = one_shot(&tokenizer, &text);
            assert_eq!(
                got, want,
                "{family} case {case}: ids differ for {text:?} split as {chunks:?}"
            );
            let decoded = tokenizer.decode(&got, false).unwrap();
            assert_eq!(decoded, text, "{family} case {case}: decoded text differs");
        }
    }
    if loaded == 0 {
        eprintln!("no tokenizer fixtures present; run scripts/fetch-tokenizers.sh");
    }
}

/// Text without spaces must still release ids while the stream is open. Chinese prose has
/// punctuation, so the pre-tokenizer finds boundaries long before the end.
#[test]
fn real_tokenizers_stream_chinese_before_finish() {
    let sentence = "在繁忙的城市里，人们每天匆匆忙忙地赶路。街道两旁的树木随风摇摆，\
        公园里传来孩子们的笑声。我们常常忘记停下来，看看天空的颜色，听听风的声音。\
        生活不只是工作和数字，还有诗歌和远方。";
    let mut text = String::new();
    while text.chars().count() < 2_000 {
        text.push_str(sentence);
    }

    let mut loaded = 0;
    for (family, _) in FAMILIES {
        let Some(tokenizer) = fixture_tokenizer(family) else {
            continue;
        };
        loaded += 1;
        let mut retokenizer = Retokenizer::new(tokenizer.clone());
        let chars: Vec<char> = text.chars().collect();
        let mut before_finish = 0usize;
        for chunk in chars.chunks(5) {
            let chunk: String = chunk.iter().collect();
            before_finish += retokenizer.push(&chunk).len();
        }
        let flushed = retokenizer.finish().len();
        let total = before_finish + flushed;
        let want = one_shot(&tokenizer, &text);
        assert_eq!(retokenizer.emitted_ids(), want.as_slice(), "{family} ids");
        assert_eq!(total, want.len(), "{family} total ids");
        eprintln!("{family}: {before_finish}/{total} ids emitted before finish");
        assert!(
            before_finish * 5 >= total * 4,
            "{family} emitted only {before_finish}/{total} ids before finish"
        );
    }
    if loaded == 0 {
        eprintln!("no tokenizer fixtures present; run scripts/fetch-tokenizers.sh");
    }
}

/// Whitespace runs are the pre-tokenizer's context-sensitive case: a run of two or more spaces
/// leaves its final space attached to the following word. Restarting from that last space alone
/// would re-split the run, so these texts are streamed one character at a time to pin down the
/// boundary handling against the real tokenizers.
#[test]
fn real_tokenizers_match_one_shot_across_whitespace_runs() {
    const TEXTS: &[&str] = &[
        "a  b",
        "a   b c",
        "hello  world",
        "中文  测试",
        "x 中文 y",
        "a\n\nb",
        "a \t b",
        "a  b  c  d",
        "1  234  five",
        "https://example.com  /path",
        "it's  a  test",
        "(a)  [b]  {c}",
        "  leading",
        "trailing  ",
        "a  🙂  b",
    ];

    let mut loaded = 0;
    for (family, _) in FAMILIES {
        let Some(tokenizer) = fixture_tokenizer(family) else {
            continue;
        };
        loaded += 1;
        for text in TEXTS {
            let chunks: Vec<String> = text.chars().map(String::from).collect();
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let got = stream(&tokenizer, &refs);
            let want = one_shot(&tokenizer, text);
            assert_eq!(got, want, "{family}: ids differ for {text:?}");
            assert_eq!(tokenizer.decode(&got, false).unwrap(), *text);
        }
    }
    if loaded == 0 {
        eprintln!("no tokenizer fixtures present; run scripts/fetch-tokenizers.sh");
    }
}
