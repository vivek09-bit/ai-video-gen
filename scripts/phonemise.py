"""Language-aware grapheme-to-phoneme, with per-word timing weights.

Why this module exists
----------------------
English uses ``misaki.en.G2P``, which returns tokens carrying per-word IPA
phonemes. That is what makes word timings accurate: a sentence's measured
duration is split across its words in proportion to how long each pronunciation
actually is.

Every other language (Hindi included) goes through ``misaki.espeak.EspeakG2P``,
which returns ``(phonemes, None)`` -- a flat phoneme string with **no tokens**.
The string is space-separated, so the words of the input can be zipped against
its groups to recover per-word phonemes. When that alignment does not line up
(espeak occasionally expands or drops tokens) the caller falls back to a
syllable estimate, which is cruder but never fails.
"""

from __future__ import annotations

import re

# Kokoro voice ids start with a language letter, e.g. `am_michael`, `hf_beta`.
VOICE_LANGUAGE = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}

# espeak language tags differ slightly from the voice letters above.
ESPEAK_LANGUAGE = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "es": "es",
    "fr": "fr",
    "hi": "hi",
    "it": "it",
    "ja": "ja",
    "pt-br": "pt",
    "cmn": "cmn",
}

STRESS = set("ˈˌ")
LENGTH_MARK = "ː"
# Vowels across the languages this project touches (IPA).
VOWELS = set("aeiouæɑɒɔəɜɛɪʊʌɐɨɵɘʉɞɶøœyAOIᵻᵊ")
# espeak marks non-speech output like this.
NON_SPEECH = re.compile(r"^[^\w\u0900-\u097F]+$")

# Devanagari ranges, for the syllable fallback.
DEVA_START, DEVA_END = 0x0900, 0x097F
DEVA_INDEPENDENT_VOWELS = set(range(0x0904, 0x0915))
DEVA_CONSONANTS = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960))
DEVA_MATRAS = set(range(0x093A, 0x094D)) | set(range(0x094E, 0x0950))
DEVA_VIRAMA = 0x094D
DEVA_ANUSVARA = {0x0902, 0x0901, 0x0903}


def resolve_language(voice: str, override: str | None = None) -> str:
    """Map a Kokoro voice id, alias, or multi-voice blend to an espeak language tag."""
    if override:
        ov = override.lower().strip()
        if ov in ("hi", "hindi", "hinglish"):
            return "hi"
        if ov in ("en", "english", "en-us"):
            return "en-us"
        if ov in ("en-gb", "british"):
            return "en-gb"
        if ov in ("ja", "japanese"):
            return "ja"
        return ESPEAK_LANGUAGE.get(ov, ov)
    # If voice is an alias or blend, extract primary resolved voice
    primary = voice.split(",")[0].split("+")[0].strip()
    try:
        from tts import resolve_voice
        resolved = resolve_voice(primary)
        primary = resolved.split(",")[0].strip()
    except Exception:
        pass
    letter = primary[:1].lower()
    return VOICE_LANGUAGE.get(letter, "en-us")


def _clean(word: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F']", "", word, flags=re.UNICODE)


def estimate_syllables(text: str) -> int:
    """Approximate syllable (akshara) count, used for word-timing weights.

    Devanagari: a consonant carries an inherent vowel unless a virama cancels
    it, so syllables ~= consonants + independent vowels - viramas.

    Latin script: vowel groups with a minimum of one syllable per word.
    """
    if not text:
        return 1

    if any(DEVA_START <= ord(ch) <= DEVA_END for ch in text):
        consonants = sum(1 for ch in text if ord(ch) in DEVA_CONSONANTS)
        independent = sum(1 for ch in text if ord(ch) in DEVA_INDEPENDENT_VOWELS)
        viramas = sum(1 for ch in text if ord(ch) == DEVA_VIRAMA)
        nasal = sum(1 for ch in text if ord(ch) in DEVA_ANUSVARA)
        return max(1, consonants + independent + nasal - viramas)

    groups = re.findall(r"[aeiouyAEIOUY]+", text)
    return max(1, len(groups))


def syllable_weights(words: list[str]) -> list[float]:
    """Fallback weighting when phoneme alignment is unavailable."""
    return [float(estimate_syllables(word)) for word in words]


def phoneme_weight(phonemes: str) -> float:
    """Relative duration of a pronunciation.

    Vowels carry most of a syllable's length; consonants are shorter; stress and
    length marks stretch the syllable they attach to.
    """
    if not phonemes:
        return 0.4

    weight = 0.0
    for char in phonemes:
        if char in STRESS or char == LENGTH_MARK:
            weight += 0.35
        elif char in VOWELS:
            weight += 1.0
        elif char.isspace():
            continue
        else:
            weight += 0.6
    return max(weight, 0.4)


class Phonemiser:
    """Caches one G2P engine per language."""

    def __init__(self) -> None:
        self._english = None
        self._espeak: dict[str, object] = {}

    def _en(self):
        if self._english is None:
            from misaki import en

            self._english = en.G2P()
        return self._english

    def _es(self, language: str):
        if language not in self._espeak:
            from misaki import espeak

            self._espeak[language] = espeak.EspeakG2P(language=ESPEAK_LANGUAGE.get(language, language))
        return self._espeak[language]

    def english_tokens(self, text: str) -> list | None:
        """Tokens carrying per-word phonemes, or None if not English.

        misaki emits punctuation as separate tokens (a lone ``.`` with the
        phoneme ``.``). Those are not spoken words, so they are dropped --
        otherwise the token count stops matching the input word count and the
        caller falls back to syllable estimates for no good reason.
        """
        try:
            _phonemes, tokens = self._en()(text)
        except Exception:
            return None

        words = []
        for token in tokens:
            value = getattr(token, "text", "") or ""
            if not _clean(value):
                continue
            words.append(token)
        return words or None

    def espeak_phonemes(self, text: str, language: str) -> str | None:
        try:
            phonemes, _tokens = self._es(language)(text)
        except Exception:
            return None
        return phonemes or None

    def word_phonemes(self, text: str, voice: str, language: str | None = None) -> list[str] | None:
        """Per-word phoneme strings for a sentence, or None if unavailable.

        English returns tokens directly. Every other language zips the input
        words against the space-separated phoneme groups espeak produces; if the
        counts disagree the alignment is abandoned rather than guessed at.
        """
        lang = resolve_language(voice, language)

        if lang.startswith("en"):
            tokens = self.english_tokens(text)
            if tokens:
                return [getattr(t, "phonemes", "") or "" for t in tokens]
            return None

        phoneme_string = self.espeak_phonemes(text, lang)
        if not phoneme_string:
            return None

        groups = [g for g in phoneme_string.split() if g]
        words = [w for w in text.split() if _clean(w)]
        if not words or not groups:
            return None

        if len(groups) == len(words):
            return groups

        # espeak sometimes emits nothing for pure punctuation, so retry against
        # only the words that carry letters.
        spoken = [w for w in words if _clean(w)]
        if len(groups) == len(spoken):
            aligned: list[str] = []
            index = 0
            for word in words:
                if _clean(word):
                    aligned.append(groups[index])
                    index += 1
                else:
                    aligned.append("")
            return aligned

        return None

    def weights(self, text: str, voice: str, language: str | None = None) -> list[float]:
        """Timing weights, one per input word. Always succeeds.

        Prefers phoneme-derived weights; falls back to syllable counts when the
        language cannot be phoneme-aligned.
        """
        words = [w for w in text.split() if w]
        if not words:
            return []

        phonemes = self.word_phonemes(text, voice, language)
        if phonemes and len(phonemes) == len(words):
            return [phoneme_weight(p) for p in phonemes]

        return syllable_weights(words)
