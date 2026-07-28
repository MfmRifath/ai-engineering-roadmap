"""Tokenization — Raschka ch. 2, Alammar ch. 2.

The whole point of writing BPE yourself is that afterwards, LLM tokenization
stops being mysterious: why models cannot count letters, why non-English text
costs more, why a leading space is part of the token.
"""

from aieng.tokenizer.bpe import BPETokenizer

__all__ = ["BPETokenizer"]
