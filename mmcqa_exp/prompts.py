"""Prompt templates. Each returns the user-turn text; the image is always passed separately."""

import hashlib

from . import settings as S
from .util import cap_words

# Identical to run.PROMPT_SUFFIX of the project root: pangea_p0 must reproduce the earlier Pangea run.
OLD_SUFFIX = "\nAnswer in the same language as the question. Give a concise answer based on the image."


def p0(question: str, track: str) -> str:
    return question.strip() + OLD_SUFFIX


def p_new(question: str, track: str) -> str:
    return (f"{question.strip()}\n"
            f"Answer in {S.VARIETY[track]}, in one or two complete sentences. "
            "Start by restating what the question asks about, name the specific item, place or practice, "
            "and add one supporting detail. No lists, no markdown.")


def p_cascade(question: str, track: str, hint: str) -> str:
    hint = cap_words(hint, S.FUSION_HINT_WORDS)
    return (f"{question.strip()}\n"
            f"Another model's first reading of this image: \"{hint}\"\n"
            "Use it only where it agrees with what you see in the image."
            + OLD_SUFFIX)


def fusion_order(item_id: str) -> int:
    """0 -> (A=first source, B=second source); 1 -> swapped. Deterministic per id."""
    return int(hashlib.sha256(str(item_id).encode()).hexdigest(), 16) % 2


def p_fusion(question: str, track: str, a: str, b: str) -> str:
    a, b = cap_words(a, S.FUSION_HINT_WORDS), cap_words(b, S.FUSION_HINT_WORDS)
    return (f"Question: {question.strip()}\n\n"
            f"Candidate answer A: {a}\n"
            f"Candidate answer B: {b}\n\n"
            "The candidates may be wrong. Look at the image, decide which facts are correct, and write the best "
            f"final answer in {S.VARIETY[track]}, in one or two complete sentences. Restate what the question asks "
            "about, name the specific item, place or practice, and add one supporting detail. "
            "Output only the answer. No lists, no markdown.")
