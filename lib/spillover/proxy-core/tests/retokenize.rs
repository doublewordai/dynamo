// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retokenizer stability and round-trip against a tiny local tokenizer.
//!
//! The tokenizer is built in-process (no downloads) as a byte-level BPE with a handful of
//! merges, which is enough to exercise boundary shifts, whitespace runs and multi-byte UTF-8.

use dw_proxy_core::retokenize::{COMPACT_AT_BYTES, MAX_HELD_BYTES, Retokenizer};
use tokenizers::AddedToken;
use tokenizers::SplitDelimiterBehavior;
use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeBuilder, Merges, Vocab};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};

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

/// A tokenizer whose pre-tokenizer mirrors the production ` ?[^\s\p{L}\p{N}]+[\r\n]*`
/// punctuation alternative: a punctuation run keeps any trailing newline in the same split, so
/// a `",\n"` split starts with the comma and ends with the newline.
fn test_tokenizer_with_punctuation_newline() -> Tokenizer {
    let mut tokenizer = test_tokenizer();
    let split = Split::new(
        SplitPattern::Regex(r" ?[A-Za-z]+|[^\sA-Za-z]+[\r\n]*".to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .expect("test split regex must compile");
    let sequence = Sequence::new(vec![
        split.into(),
        ByteLevel::default()
            .add_prefix_space(false)
            .use_regex(false)
            .into(),
    ]);
    tokenizer.with_pre_tokenizer(Some(sequence));
    tokenizer
}

fn test_tokenizer_with_added_token(marker: &str) -> Tokenizer {
    let mut tokenizer = test_tokenizer();
    assert_eq!(tokenizer.add_tokens(&[AddedToken::from(marker, true)]), 1);
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
        ids.extend(retokenizer.push(chunk).unwrap());
    }
    ids.extend(retokenizer.finish().unwrap());
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
    assert!(retokenizer.push("").unwrap().is_empty());
    assert!(retokenizer.finish().unwrap().is_empty());
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
fn emitted_text_is_exactly_what_its_ids_encode() {
    // A consumer streams `text` and counts `ids`; if text ran ahead of its ids, the held-back
    // piece would reach the client twice (the proxy's repeated-last-word bug).
    let tokenizer = test_tokenizer();
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for case in 0..500 {
        let text = random_text(&mut rng, 120);
        let chunks = random_chunks(&mut rng, &text);
        let mut retokenizer = Retokenizer::new(tokenizer.clone());
        let mut emissions = Vec::new();
        for chunk in &chunks {
            emissions.push(retokenizer.push_with_text(chunk).unwrap());
        }
        emissions.push(retokenizer.finish_with_text().unwrap());

        let streamed: String = emissions.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(streamed, text, "case {case}: text lost or repeated");
        let ids: Vec<u32> = emissions.iter().flat_map(|e| e.ids.clone()).collect();
        assert_eq!(ids, one_shot(&tokenizer, &text), "case {case}: ids differ");
        for emitted in &emissions {
            assert_eq!(
                tokenizer.decode(&emitted.ids, false).unwrap(),
                emitted.text,
                "case {case}: text and ids disagree in {chunks:?}"
            );
        }
    }
}

#[test]
fn held_text_is_released_with_its_ids() {
    let tokenizer = test_tokenizer();
    let mut retokenizer = Retokenizer::new(tokenizer.clone());
    let first = retokenizer.push_with_text("hello world").unwrap();
    assert_eq!(first.text, "hello");
    let last = retokenizer.finish_with_text().unwrap();
    assert_eq!(last.text, " world");
    assert_eq!(tokenizer.decode(&last.ids, false).unwrap(), " world");
}

#[test]
fn very_long_push_matches_one_shot() {
    let tokenizer = test_tokenizer();
    let text = "hello world ".repeat(20_000);
    assert_matches_one_shot(&tokenizer, &text, &[&text]);
}

#[test]
fn boundary_free_run_is_not_held_past_the_cap() {
    let tokenizer = test_tokenizer();
    let text = "x".repeat(4 * MAX_HELD_BYTES);
    let mut retokenizer = Retokenizer::new(tokenizer.clone());
    let mut streamed = Vec::new();
    for chunk in text.as_bytes().chunks(64) {
        streamed.extend(
            retokenizer
                .push(std::str::from_utf8(chunk).unwrap())
                .unwrap(),
        );
    }
    // Ids arrive before the stream ends, and the held tail stays within the cap.
    assert!(
        !streamed.is_empty(),
        "a boundary-free run must not be held until finish"
    );
    let held = text.len() - tokenizer.decode(&streamed, false).unwrap().len();
    assert!(held <= MAX_HELD_BYTES + 64, "held {held} bytes");
    streamed.extend(retokenizer.finish().unwrap());
    assert_eq!(tokenizer.decode(&streamed, false).unwrap(), text);
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

/// A stream long enough to cross the compaction threshold in many places must still match a
/// one-shot encode: the prefix is dropped only after it has been fully emitted, so ids and
/// decoded text are unchanged.
#[test]
fn long_stream_across_compaction_boundary_matches_one_shot() {
    let tokenizer = test_tokenizer();
    // Several times the compaction threshold.
    let text = "hello world, 中文字 🙂 ".repeat(COMPACT_AT_BYTES);
    let mut chunks: Vec<&str> = Vec::new();
    let mut start = 0;
    for (i, _) in text.char_indices() {
        if i - start >= 5 {
            chunks.push(&text[start..i]);
            start = i;
        }
    }
    chunks.push(&text[start..]);
    assert_matches_one_shot(&tokenizer, &text, &chunks);
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
        emitted.extend(retokenizer.push(chunk).unwrap());
        assert_eq!(
            emitted.as_slice(),
            &want[..emitted.len()],
            "already emitted ids were revised after {chunk:?}"
        );
    }
    emitted.extend(retokenizer.finish().unwrap());
    assert_eq!(emitted, want);
}

#[test]
fn holds_back_until_a_safe_boundary_then_flushes() {
    let tokenizer = test_tokenizer();
    let mut retokenizer = Retokenizer::new(tokenizer.clone());

    // A single partial word has no safe boundary, so nothing is emitted yet.
    assert!(retokenizer.push("hel").unwrap().is_empty());

    // Enough complete words produce stable ids before the stream ends.
    let ids = retokenizer
        .push("lo world hello world hello world")
        .unwrap();
    assert!(!ids.is_empty());
    let flush = retokenizer.finish().unwrap();
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
    //
    // Only splits at spaces are exercised: `ByteLevel` with `add_prefix_space` prepends a
    // space to whatever it is handed, and a mid-stream tail that starts at a non-space
    // pre-token boundary would get a spurious one. That limitation is latent for the pinned
    // families (GLM-5.3, DeepSeek-V4.1, Qwen3-8B all set `add_prefix_space: false`).
    let tokenizer = test_tokenizer_with_prefix_space(true);
    let text = "hello world hello world";
    let got = stream(
        &tokenizer,
        &["hel", "lo ", "wor", "ld ", "hel", "lo ", "wor", "ld"],
    );
    assert_eq!(got, one_shot(&tokenizer, text));
    assert_eq!(tokenizer.decode(&got, false).unwrap(), format!(" {text}"));
}

/// A complete pre-token may precede a punctuation run that swallowed a trailing newline. The
/// restart must land on the split boundary, not inside the `",\n"` pre-token, or the comma's
/// bytes are skipped forever.
#[test]
fn punctuation_before_newline_is_not_dropped() {
    let tokenizer = test_tokenizer_with_punctuation_newline();
    let cases: &[(&str, &[&str])] = &[
        ("a,\n b", &["a,\n b"]),
        ("a,\n b", &["a", ",\n ", "b"]),
        ("hello,\n world", &["hello,\n world"]),
        ("hello,\n world", &["hello", ",\n ", "world"]),
        ("x.\n\n  y", &["x", ".\n\n  ", "y"]),
        ("x.\n\n  y", &["x.\n\n  y"]),
    ];
    for (text, chunks) in cases {
        assert_matches_one_shot(&tokenizer, text, chunks);
    }
}

/// Added/special tokens must keep their added id instead of being run through the BPE model as
/// literal bytes.
#[test]
fn added_tokens_keep_their_id() {
    let marker = "<tool_call>";
    let tokenizer = test_tokenizer_with_added_token(marker);
    let added_id = tokenizer.token_to_id(marker).unwrap();
    assert_eq!(one_shot(&tokenizer, marker), vec![added_id]);

    let text = format!("hello {marker} world");
    let refs = ["hello ", marker, " world"];
    assert_matches_one_shot(&tokenizer, &text, &refs);
    assert_matches_one_shot(&tokenizer, &text, &[&text]);
    assert!(one_shot(&tokenizer, &text).contains(&added_id));
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
        a.extend(from_file.push(chunk).unwrap());
        b.extend(in_memory.push(chunk).unwrap());
    }
    a.extend(from_file.finish().unwrap());
    b.extend(in_memory.finish().unwrap());
    assert_eq!(a, b);
    assert_eq!(a, one_shot(&tokenizer, "hello world 🙂中文"));
}

/// Production tokenizer families. The files are downloaded by `scripts/fetch-tokenizers.sh`
/// and gitignored. Tests skip with a message when a family is missing, unless
/// `DW_REQUIRE_TOKENIZERS=1` is set, in which case a missing fixture is a hard failure (CI sets
/// the variable after fetching, so the production-tokenizer coverage cannot silently vanish).
const FAMILIES: &[(&str, &str)] = &[
    ("glm-5", "zai-org/GLM-5.3"),
    ("deepseek-v4", "deepseek-ai/DeepSeek-V4.1-Flash"),
    ("qwen3", "Qwen/Qwen3-8B"),
];

fn require_tokenizers() -> bool {
    std::env::var("DW_REQUIRE_TOKENIZERS").as_deref() == Ok("1")
}

fn fixture_tokenizer(family: &str) -> Option<Tokenizer> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tokenizers")
        .join(family)
        .join("tokenizer.json");
    if !path.exists() {
        if require_tokenizers() {
            panic!(
                "DW_REQUIRE_TOKENIZERS=1 but the {family} fixture is missing: {}; \
                 run scripts/fetch-tokenizers.sh",
                path.display()
            );
        }
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
        let mut emitted = Vec::new();
        let mut before_finish = 0usize;
        for chunk in chars.chunks(5) {
            let chunk: String = chunk.iter().collect();
            let ids = retokenizer.push(&chunk).unwrap();
            before_finish += ids.len();
            emitted.extend(ids);
        }
        emitted.extend(retokenizer.finish().unwrap());
        let total = emitted.len();
        let want = one_shot(&tokenizer, &text);
        assert_eq!(emitted, want, "{family} ids");
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

/// Regression for r07-1: a punctuation pre-token that swallowed a trailing newline (`",\n"`)
/// must not be dropped when a complete pre-token precedes it in the same `emit`.
#[test]
fn real_tokenizers_keep_punctuation_before_newline() {
    let cases: &[(&str, &[&str])] = &[
        ("a,\n b", &["a,\n b"]),
        ("a,\n b", &["a", ",\n ", "b"]),
        ("hello,\n world", &["hello,\n world"]),
        ("hello,\n world", &["hello", ",\n ", "world"]),
        ("x.\n\n  y", &["x.\n\n  y"]),
        ("a,\n", &["a", ",\n"]),
    ];

    let mut loaded = 0;
    for (family, _) in FAMILIES {
        let Some(tokenizer) = fixture_tokenizer(family) else {
            continue;
        };
        loaded += 1;
        for (text, chunks) in cases {
            let got = stream(&tokenizer, chunks);
            let want = one_shot(&tokenizer, text);
            assert_eq!(got, want, "{family}: ids differ for {text:?} in {chunks:?}");
            assert_eq!(tokenizer.decode(&got, false).unwrap(), *text);
        }
    }
    if loaded == 0 {
        eprintln!("no tokenizer fixtures present; run scripts/fetch-tokenizers.sh");
    }
}

/// Regression for r07-2: markers the renderers emit (`<tool_call>`, `<sop>`, …) are added
/// vocabulary entries and must keep their added id instead of being BPE-tokenized.
#[test]
fn real_tokenizers_honor_added_tokens() {
    const MARKERS: &[&str] = &["<tool_call>", "</tool_call>", "<sop>", "[MASK]"];

    let mut loaded = 0;
    let mut checked = 0;
    for (family, _) in FAMILIES {
        let Some(tokenizer) = fixture_tokenizer(family) else {
            continue;
        };
        loaded += 1;
        for marker in MARKERS {
            if tokenizer.token_to_id(marker).is_none() {
                continue;
            }
            checked += 1;
            let text = format!("hello {marker} world");
            let got = stream(&tokenizer, &[&text]);
            let want = one_shot(&tokenizer, &text);
            assert_eq!(got, want, "{family}: ids differ for {text:?}");
            assert_eq!(tokenizer.decode(&got, false).unwrap(), text);
        }
    }
    if loaded == 0 {
        eprintln!("no tokenizer fixtures present; run scripts/fetch-tokenizers.sh");
    } else {
        assert!(checked > 0, "no fixture tokenizer declared an added marker");
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
