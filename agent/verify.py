"""Programmatic checks used by the Verifier step (no language model involved)."""
from __future__ import annotations

import re

# a minus sign directly after a digit is a range dash ("11.08-13.37"), not a sign
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?")


# Arabic-Indic and Persian digits, Arabic decimal and thousands separators -> ASCII
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")


def normalise_digits(text: str) -> str:
    return (text or "").translate(_DIGITS)


def numbers_in_text(text: str) -> list[tuple[float, int]]:
    """Return (value, decimals) for each number written in the text."""
    out = []
    for m in NUM_RE.finditer(normalise_digits(text)):
        s = m.group().replace(",", "")
        dec = len(s.split(".")[1]) if "." in s else 0
        try:
            out.append((float(s), dec))
        except ValueError:
            pass
    return out


def numbers_in_obj(obj, acc=None) -> set[float]:
    """All numeric leaves in a (nested) tool result, plus numbers inside strings."""
    acc = set() if acc is None else acc
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        acc.update(v for v, _ in numbers_in_text(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            numbers_in_obj(k, acc)
            numbers_in_obj(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numbers_in_obj(v, acc)
    return acc


def _matches(value: float, decimals: int, allowed: set[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for t in allowed:
        for cand in (t, t * 100):  # 0.373 may be written as 37.3 %
            if abs(abs(cand) - abs(value)) <= tol:
                return True
    return False


def dataset_facts() -> set[float]:
    """Basic facts about the dataset that any answer may mention."""
    from .tools import MEASUREMENTS, load_data
    df = load_data()
    counts = df["diagnosis"].value_counts()
    g = df["gender_simulated"].value_counts()
    facts = {len(df), df.shape[1], len(MEASUREMENTS), *counts.tolist(), *g.tolist()}
    facts |= {round(100 * v / len(df), 1) for v in [*counts.tolist(), *g.tolist()]}
    return {float(x) for x in facts}


def unsupported_numbers(text: str, tool_results: list, question: str = "") -> list[str]:
    """Numbers in `text` that do not appear (to the written precision) in any tool result."""
    allowed = set()
    for r in tool_results:
        numbers_in_obj(r, allowed)
    allowed.update(v for v, _ in numbers_in_text(question))
    allowed |= dataset_facts()
    bad = []
    for value, dec in numbers_in_text(text):
        if dec == 0 and 0 <= value <= 10:
            continue  # small counts like "top 3", "5-fold", "Q1"
        if not _matches(value, dec, allowed):
            bad.append(f"{value:g}")
    return sorted(set(bad), key=bad.index)


SIMULATED_COLUMNS = ("age_simulated", "gender_simulated")


def used_simulated(tool_calls: list[dict]) -> bool:
    return any(c in str(call.get("args", {})) for call in tool_calls for c in SIMULATED_COLUMNS) or \
        any(w in str(call.get("args", {})).lower() for call in tool_calls for w in ("'age'", "'gender'", "'sex'"))


def mentions_simulated(text: str) -> bool:
    return bool(re.search(r"simulat|synthetic|محاكا|مُحاكا|افتراضي|غير حقيقي|اصطناعي|مولد|مولّد",
                          text or "", re.I))


ADVICE_PATTERNS = [
    # only first-person requests about a real person's health; questions about
    # the data ("Should I use the median?", "treatment group") must pass
    r"\b(do|does|could|might|can|will) (i|my \w+) have (breast )?(cancer|a tumou?r|a malignant|a benign)",
    r"\bam i (sick|ill|at risk|going to die|likely to (have|get|die))",
    r"\bshould (i|she|my \w+) (be worried|worry|see a doctor|go to|get (a |an )?(biopsy|mammogram|ultrasound|test|tested|checked|screened|surgery|treatment|operation)|take|start|stop|have (surgery|chemo|a biopsy))",
    r"\bmy (own |\w+'?s? )?(biopsy|tumou?r|lump|mammogram|ultrasound|scan|diagnosis|oncologist|pathology report|breast)\b",
    r"\bi(?:'ve got| have got|'ve| have| found| feel| felt| noticed| got)\s+(?:a|an)\s+(?:[a-z-]+\s+){0,3}(?:lump|mass|tumou?r|swelling|cyst)",
    r"\bis (my|this) (lump|mass|tumou?r|cyst)\b",
    r"\b(?:is it|am i) (?:dangerous|serious|cancer|malignant|benign|going to)\b.*\b(?:my|i have)\b",
    r"\b(what|which) (treatment|medication|medicine|drug|dose|therapy) should\b",
    r"\b(diagnose|treat) me\b",
    # Arabic: first-person health questions ("I have a tumour", "do I have cancer", ...)
    r"(?:انا|أنا|اني|إني|إنني|انني)?\s*(?:عندي|لدي|لديّ|فيني|وجدت|لقيت|حسيت)\s+(?:\S+\s+){0,2}(?:ورم|كتل[ةه]|سرطان|تكتل|كيس)",
    r"هل\s+(?:انا|أنا)\s+(?:مصاب|مريض|معرض)",
    r"هل\s+(?:عندي|لدي|فيني)\s",
    r"(?:ورمي|كتلتي|خزعتي|تحاليلي|أشعتي|اشعتي|فحوصاتي|تشخيصي)",
    r"(?:وش|ايش|ما|ماذا)\s+(?:العلاج|الدواء)\s+(?:اللي|الذي)?\s*(?:اخذه|آخذه|استخدمه|يناسبني)",
    r"(?:شخصني|شخّصني|عالجني|انصحني)",
]

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def is_personal_medical_request(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(p, q) for p in ADVICE_PATTERNS)
