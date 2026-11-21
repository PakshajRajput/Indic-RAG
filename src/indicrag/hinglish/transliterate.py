"""Deterministic phonological transliterator from Hindi (Devanagari) to Roman-script Hinglish.

Preserves digits, punctuation, and question intent while generating natural phonetic
Roman-script representation that satisfies all HinglishValidator constraints.
"""

from __future__ import annotations

import re
import unicodedata

DEVANAGARI_DIGITS: dict[str, str] = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

VOWELS: dict[str, str] = {
    "\u0905": "a", "\u0906": "aa", "\u0907": "i", "\u0908": "ee",
    "\u0909": "u", "\u090a": "oo", "\u090f": "e", "\u0910": "ai",
    "\u0913": "o", "\u0914": "au", "\u090b": "ri",
}

MATRAS: dict[str, str] = {
    "\u093e": "aa", "\u093f": "i", "\u0940": "ee", "\u0941": "u",
    "\u0942": "oo", "\u0947": "e", "\u0948": "ai", "\u094b": "o",
    "\u094c": "au", "\u0943": "ri",
}

CONSONANTS: dict[str, str] = {
    "\u0915": "k", "\u0916": "kh", "\u0917": "g", "\u0918": "gh", "\u0919": "ng",
    "\u091a": "ch", "\u091b": "chh", "\u091c": "j", "\u091d": "jh", "\u091e": "ny",
    "\u091f": "t", "\u0920": "th", "\u0921": "d", "\u0922": "dh", "\u0923": "n",
    "\u0924": "t", "\u0925": "th", "\u0926": "d", "\u0927": "dh", "\u0928": "n",
    "\u092a": "p", "\u092b": "ph", "\u092c": "b", "\u092d": "bh", "\u092e": "m",
    "\u092f": "y", "\u0930": "r", "\u0932": "l", "\u0933": "l", "\u0935": "v",
    "\u0936": "sh", "\u0937": "sh", "\u0938": "s", "\u0939": "h",
    "\u0958": "q", "\u0959": "kh", "\u095a": "g", "\u095b": "z", "\u095c": "r",
    "\u095d": "rh", "\u095e": "f", "\u095f": "y",
}

COMMON_WORDS: dict[str, str] = {
    "क्या": "kya", "कौन": "kaun", "कहाँ": "kahan", "कब": "kab", "क्यों": "kyon",
    "कैसे": "kaise", "कितना": "kitna", "कितने": "kitne", "कितनी": "kitni",
    "किस": "kis", "किसके": "kiske", "किसका": "kiska", "किसकी": "kiski", "किसे": "kise",
    "है": "hai", "हैं": "hain", "था": "tha", "थी": "thi", "थे": "the",
    "में": "mein", "से": "se", "को": "ko", "का": "ka", "के": "ke", "की": "ki",
    "और": "aur", "ने": "ne", "पर": "par", "एक": "ek", "दो": "do", "तीन": "teen",
    "यह": "yeh", "वह": "woh", "नहीं": "nahi", "द्वारा": "dwara", "बारे": "baare",
    "रूप": "roop", "नाम": "naam", "वर्ष": "varsh", "समय": "samay", "स्थान": "sthan",
    "शहर": "shahar", "देश": "desh", "राज्य": "rajya", "नदी": "nadi", "लोगों": "logon",
    "दिन": "din", "साल": "saal", "महीने": "maheene", "पहले": "pehle", "बाद": "baad",
}


def transliterate_hindi_token(token: str) -> str:
    """Transliterate a single Hindi token to Roman-script Hinglish."""
    # Convert Devanagari digits first
    for dev_d, lat_d in DEVANAGARI_DIGITS.items():
        token = token.replace(dev_d, lat_d)

    # Separate punctuation prefixes and suffixes
    punc_pre_m = re.match(r"^[^\u0900-\u097Fa-zA-Z0-9]+", token)
    punc_suf_m = re.search(r"[^\u0900-\u097Fa-zA-Z0-9]+$", token)
    p_pre = punc_pre_m.group(0) if punc_pre_m else ""
    p_suf = punc_suf_m.group(0) if punc_suf_m else ""

    clean = token[len(p_pre) : len(token) - len(p_suf) if p_suf else len(token)]

    if clean in COMMON_WORDS:
        return f"{p_pre}{COMMON_WORDS[clean]}{p_suf}"

    # Character-by-character phonetic parsing
    res: list[str] = []
    i = 0
    n = len(clean)
    while i < n:
        c = clean[i]
        if c in DEVANAGARI_DIGITS:
            res.append(DEVANAGARI_DIGITS[c])
            i += 1
        elif c in VOWELS:
            res.append(VOWELS[c])
            i += 1
        elif c in CONSONANTS:
            base = CONSONANTS[c]
            if i + 1 < n:
                next_c = clean[i + 1]
                if next_c == "\u094d":  # Virama / halant
                    res.append(base)
                    i += 2
                elif next_c in MATRAS:
                    res.append(base + MATRAS[next_c])
                    i += 2
                elif next_c == "\u0902":  # Anusvara
                    res.append(base + "an")
                    i += 2
                elif next_c in CONSONANTS or next_c in VOWELS:
                    res.append(base + "a")
                    i += 1
                else:
                    res.append(base)
                    i += 1
            else:
                # Word-final consonant: schwa deletion in modern Hindi
                res.append(base)
                i += 1
        elif c in MATRAS:
            res.append(MATRAS[c])
            i += 1
        elif c == "\u0902" or c == "\u0901":  # Anusvara / Chandrabindu
            res.append("n")
            i += 1
        elif c == "\u094d":
            i += 1
        else:
            res.append(c)
            i += 1

    trans_str = "".join(res)
    return f"{p_pre}{trans_str}{p_suf}"


def hindi_to_hinglish(text: str) -> str:
    """Convert a Hindi sentence to Roman-script Hinglish."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", text)
    tokens = norm.split(" ")
    out_tokens = [transliterate_hindi_token(tok) for tok in tokens if tok]
    out = " ".join(out_tokens)
    # Strip any remaining Devanagari characters
    out = re.sub(r"[\u0900-\u097F]", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    if text.strip().endswith("?") and not out.endswith("?"):
        out += "?"
    return out
