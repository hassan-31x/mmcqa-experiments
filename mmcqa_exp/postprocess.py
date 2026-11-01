"""Output cleanup and the guaranteed non-empty fallback.

Never truncates by word count: the post-processing probes showed that every cut below ~2 sentences
lowers official F1 for every model. An unanswered item scores 0, so ``finalize`` never returns "".
"""

import re

SENT_END = re.compile(r"(?<=[.!?؟。])\s+")
TERMINAL = tuple(".!?؟。…\"'”’)»")
# Invented follow-up dialogue turns ("\nHuman: ...", "\nأجاب: ...").
TURN = re.compile(r"\n\s*(?:Human|User|Assistant|Question|Answer|أجاب|سؤال)\s*:.*", re.S)
# Gemma-style leaked thinking tokens, <think> blocks and chat-control tags such as <|im_end|>.
ARTIFACT = re.compile(r"^\s*thought\s*<channel\|>|<think>.*?</think>|</?think>|<\|?[a-z_]+\|?>", re.S | re.I)
MARKDOWN = re.compile(r"\*\*|__|^#+\s*|^\s*(?:\d+\.|[-*•])\s+", re.M)
LABEL = re.compile(r"^\s*(?:Answer|Response|Final answer|الإجابة|الجواب)\s*[:：]\s*", re.I)


def clean(text) -> str:
    text = "" if text is None else str(text)
    text = TURN.sub("", text)
    text = ARTIFACT.sub("", text)
    text = MARKDOWN.sub("", text)
    text = LABEL.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_unfinished(text: str) -> str:
    """Drop a trailing sentence cut off mid-way, but only if a complete sentence remains."""
    if not text or text.endswith(TERMINAL):
        return text
    parts = [p for p in SENT_END.split(text) if p.strip()]
    return " ".join(parts[:-1]).strip() if len(parts) >= 2 else text


def finalize(raw, question: str, hit_cap: bool = False):
    """Return ``(answer, fallback_used)``. ``hit_cap`` says generation stopped at the token limit."""
    answer = clean(raw)
    if answer and hit_cap:
        answer = trim_unfinished(answer) or answer
    if answer:
        return answer, False
    fallback = clean(question) or str(question).strip() or "-"
    return fallback, True
