import re

from distill_lab.data.prepare import canonical_number

# -3, 72, 1,200, $5, 0.5, 1250.00. the lookbehind keeps "30-5" from reading as -5
_NUM = r"(?<![\w.])-?\$?\d+(?:,\d{3})*(?:\.\d+)?"
_HASH = re.compile(r"####\s*(" + _NUM + ")")
_BOXED = re.compile(r"\\boxed\{\s*(" + _NUM + r")")
_SAYS = re.compile(r"answer\s*(?:is\s*:?|:|=)\s*\**\s*(" + _NUM + ")", re.IGNORECASE)
_ANY = re.compile(_NUM)


def _clean(text: str) -> str:
    # qwen3 can think out loud first, only what comes after counts
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # base models answer, then start writing the next problem themselves
    cut = text.find("Question:")
    if cut > 0:
        text = text[:cut]
    return text


def _number(raw: str) -> str | None:
    try:
        return canonical_number(raw)
    except ValueError:
        return None


def extract_answer(text: str) -> str | None:
    """final answer as a canonical number string, or None.
    tries, in order: '#### n', '\\boxed{n}', 'the answer is n', then the last number in the text"""
    text = _clean(text)
    for pattern in (_HASH, _BOXED, _SAYS):
        m = pattern.search(text)
        if m:
            return _number(m.group(1))
    found = _ANY.findall(text)
    return _number(found[-1]) if found else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return abs(float(pred) - float(gold)) < 1e-6
