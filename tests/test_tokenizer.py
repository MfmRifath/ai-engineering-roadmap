"""BPE tokenizer tests — Raschka ch. 2.

The headline property is that **encode/decode round-trips exactly**, including
for text the tokenizer never saw during training. That is what "no
out-of-vocabulary" actually means, and it is what makes byte-level BPE work.
"""

from __future__ import annotations

import pytest

from aieng.tokenizer import BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick and the dog is lazy. "
    "a quick brown dog jumps over a lazy fox repeatedly, quickly, and quietly. "
) * 20


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer().train(CORPUS, vocab_size=400)


# --------------------------------------------------------------------------
# The properties that matter
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "a completely unseen sentence with novel words",
        "PUNCTUATION!? and 12345 numbers",
        "unicode: café, naïve, 日本語, emoji 🎉",
        "   leading and trailing whitespace   ",
        "a",
        "x" * 500,
    ],
)
def test_round_trip_is_exact(tokenizer, text):
    """Byte-level BPE cannot have unknown tokens, so decode(encode(t)) == t."""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_string(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_training_learns_merges(tokenizer):
    """Merges are learned, and training stops early when nothing repeats.

    A small, highly repetitive corpus exhausts its repeated pairs before
    reaching the requested vocabulary size — the loop breaks when the best
    remaining pair occurs only once, because merging a singleton encodes noise
    rather than structure. So the vocabulary is a *ceiling*, not a target.
    """
    assert len(tokenizer.merges) > 0
    assert 256 < tokenizer.vocab_size <= 400
    assert tokenizer.vocab_size == 256 + len(tokenizer.merges)


def test_merges_shorten_frequent_sequences(tokenizer):
    """The point of BPE: common strings become fewer tokens."""
    frequent = tokenizer.encode("the quick brown")
    raw_bytes = len(b"the quick brown")
    assert len(frequent) < raw_bytes


def test_unseen_text_costs_more_tokens_than_seen_text(tokenizer):
    """Domain mismatch is visible in the token count — Alammar ch. 2."""
    seen = len(tokenizer.encode("the quick brown fox jumps over the lazy dog"))
    unseen = len(tokenizer.encode("xyzzy plugh frobnicate zzyzx qwertyuiop"))
    assert unseen > seen


def test_leading_space_is_part_of_the_token(tokenizer):
    """`" the"` and `"the"` are different — why trailing whitespace changes output."""
    assert tokenizer.encode(" the") != tokenizer.encode("the")


def test_training_is_deterministic():
    """The same corpus must produce the same merges, or nothing is reproducible."""
    a = BPETokenizer().train(CORPUS, vocab_size=320)
    b = BPETokenizer().train(CORPUS, vocab_size=320)
    assert a.merges == b.merges


def test_base_vocabulary_is_all_bytes():
    tok = BPETokenizer()
    assert len(tok.vocab) == 256
    assert all(i in tok.vocab for i in range(256))


def test_rejects_vocab_size_below_byte_vocabulary():
    with pytest.raises(ValueError, match="256"):
        BPETokenizer().train(CORPUS, vocab_size=100)


# --------------------------------------------------------------------------
# Special tokens
# --------------------------------------------------------------------------


def test_special_tokens_are_never_split():
    tok = BPETokenizer().train(CORPUS, vocab_size=300)
    eot = tok.add_special_token("<|endoftext|>")

    ids = tok.encode("hello <|endoftext|> world")
    assert eot in ids
    assert ids.count(eot) == 1
    assert tok.decode(ids) == "hello <|endoftext|> world"


def test_longest_special_token_wins():
    tok = BPETokenizer().train(CORPUS, vocab_size=300)
    tok.add_special_token("<|end|>")
    long_id = tok.add_special_token("<|endoftext|>")
    assert long_id in tok.encode("a <|endoftext|> b")


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_save_and_load_round_trip(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.merges == tokenizer.merges
    text = "the quick brown fox and some unseen words"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_partial_decode_does_not_raise(tokenizer):
    """A truncated id sequence can end mid-UTF-8; streaming must not crash."""
    ids = tokenizer.encode("unicode: café, naïve, 日本語")
    for cut in range(1, len(ids)):
        tokenizer.decode(ids[:cut])  # must not raise
