"""Byte pair encoding, from scratch.

Reference: Raschka ch. 2 (and the bonus BPE material in his repo); Alammar ch. 2
for what tokenization explains about LLM behaviour.

The algorithm is genuinely short:

    vocabulary = every individual byte
    while len(vocabulary) < target:
        find the most frequent adjacent pair in the corpus
        merge it into a new token
        record the merge

Encoding then replays the recorded merges, in the order they were learned, over
the bytes of the input. Because the base vocabulary is all 256 bytes, **there is
no out-of-vocabulary case** — any input decomposes to something representable.

Working at the byte level rather than the character level is what makes this
true for arbitrary Unicode, and it is what GPT-2 onward actually do.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

# A word is a tuple of token ids; the corpus is a mapping of word -> frequency.
Word = tuple[int, ...]
Pair = tuple[int, int]


def _pair_counts(corpus: dict[Word, int]) -> Counter[Pair]:
    """Count adjacent token pairs across the corpus, weighted by word frequency."""
    counts: Counter[Pair] = Counter()
    for word, freq in corpus.items():
        for pair in pairwise(word):
            counts[pair] += freq
    return counts


def _merge_word(word: Word, pair: Pair, new_id: int) -> Word:
    """Replace every occurrence of ``pair`` in ``word`` with ``new_id``."""
    if len(word) < 2:
        return word
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


@dataclass
class BPETokenizer:
    """A byte-level BPE tokenizer you trained yourself.

    Attributes
    ----------
    merges:
        Learned merges in the order they were learned. ``merges[(a, b)] = new_id``.
        Order matters at encode time — later merges depend on earlier ones.
    vocab:
        ``token_id -> bytes``. Ids 0-255 are the raw bytes; everything above is
        a learned merge.
    special_tokens:
        ``string -> token_id`` for tokens that must never be split, e.g.
        ``<|endoftext|>``.
    """

    merges: dict[Pair, int] = field(default_factory=dict)
    vocab: dict[int, bytes] = field(default_factory=dict)
    special_tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocab:
            self.vocab = {i: bytes([i]) for i in range(256)}

    # -- training ---------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> BPETokenizer:
        """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``.

        ``vocab_size`` counts the 256 base byte tokens, so a target of 300 learns
        44 merges.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte vocabulary)")

        # Splitting on whitespace first keeps merges from spanning word
        # boundaries. The leading space is kept as part of the following word,
        # which is why " the" and "the" end up as different tokens.
        words = [w for w in _split_words(text) if w]
        corpus: dict[Word, int] = Counter(tuple(w.encode("utf-8")) for w in words)

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        next_id = 256

        while next_id < vocab_size:
            counts = _pair_counts(corpus)
            if not counts:
                break
            # max() with this key is deterministic: highest count, then lowest
            # pair, so training the same corpus twice gives identical merges.
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[best] < 2:
                break  # nothing repeats; further merges would be noise

            corpus = {_merge_word(w, best, next_id): f for w, f in corpus.items()}
            self.merges[best] = next_id
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if verbose:
                piece = self.vocab[next_id].decode("utf-8", errors="replace")
                print(f"  merge {next_id}: {best} -> {piece!r} (count {counts[best]})")
            next_id += 1

        return self

    def add_special_token(self, token: str) -> int:
        """Register a token that is matched literally and never split."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        token_id = max(self.vocab) + 1
        self.special_tokens[token] = token_id
        self.vocab[token_id] = token.encode("utf-8")
        return token_id

    # -- encoding / decoding ----------------------------------------------

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text to token ids."""
        if allowed_special and self.special_tokens:
            return self._encode_with_specials(text)
        return self._encode_ordinary(text)

    def _encode_with_specials(self, text: str) -> list[int]:
        # Longest first, so <|endoftext|> wins over any shorter overlapping token.
        for special in sorted(self.special_tokens, key=len, reverse=True):
            if special in text:
                head, _, tail = text.partition(special)
                return [
                    *self._encode_with_specials(head),
                    self.special_tokens[special],
                    *self._encode_with_specials(tail),
                ]
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for word in _split_words(text):
            if word:
                out.extend(self._encode_word(word))
        return out

    def _encode_word(self, word: str) -> list[int]:
        ids: Word = tuple(word.encode("utf-8"))
        # Replay merges in learned order. Applying the lowest-id applicable
        # merge each round reproduces training order without re-sorting.
        while len(ids) >= 2:
            candidates = {p: self.merges[p] for p in pairwise(ids) if p in self.merges}
            if not candidates:
                break
            best = min(candidates, key=candidates.get)  # type: ignore[arg-type]
            ids = _merge_word(ids, best, self.merges[best])
        return list(ids)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text.

        ``errors="replace"`` matters: an arbitrary slice of ids can end
        mid-UTF-8-sequence, and raising there would make partial decoding of a
        streaming response impossible.
        """
        parts = [self.vocab[i] for i in ids if i in self.vocab]
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        for (a, b), new_id in ((tuple(p), i) for p, i in payload["merges"]):
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        for token, token_id in payload["special_tokens"].items():
            tok.special_tokens[token] = token_id
            tok.vocab[token_id] = token.encode("utf-8")
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


def _split_words(text: str) -> list[str]:
    """Split text into words, keeping the leading space attached to each word.

    This is why `" the"` and `"the"` are distinct tokens in most BPE tokenizers,
    and why trailing whitespace in a prompt can change model output.
    """
    if not text:
        return []
    words: list[str] = []
    current = ""
    for ch in text:
        if ch.isspace():
            if current:
                words.append(current)
            current = ch
        else:
            current += ch
    if current:
        words.append(current)
    return words
