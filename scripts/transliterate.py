#!/usr/bin/env python3
"""Devanagari to Roman Hinglish transliteration and SRT subtitle generation.

Converts Devanagari Hindi text and word timings into natural, modern Roman Hinglish
(Latin alphabet) suitable for on-screen captions (Shorts / Reels) and standard .srt files.
"""

from __future__ import annotations

import re
from pathlib import Path

# Independent Devanagari vowels -> Roman
DEVA_VOWELS: dict[str, str] = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ee",
    "उ": "u",
    "ऊ": "oo",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "अं": "an",
    "अः": "ah",
    "ऑ": "o",
    "ॲ": "e",
    "ऍ": "e",
}

# Dependent vowel signs (Matras) -> Roman
DEVA_MATRAS: dict[str, str] = {
    "ा": "a",
    "ि": "i",
    "ी": "i",
    "ु": "u",
    "ू": "u",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ॉ": "o",
    "ॅ": "e",
    "ँ": "n",
    "ं": "n",
    "ः": "h",
}

# Consonants -> Roman
DEVA_CONSONANTS: dict[str, str] = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    # Nukta consonants
    "क़": "q",
    "ख़": "kh",
    "ग़": "g",
    "ज़": "z",
    "ड़": "r",
    "ढ़": "rh",
    "फ़": "f",
    "य़": "y",
    "ळ": "l",
}

# Special conjuncts
DEVA_CONJUNCTS: dict[str, str] = {
    "क्ष": "ksh",
    "त्र": "tra",
    "ज्ञ": "gya",
    "श्र": "shra",
    "द्य": "dya",
    "द्व": "dva",
}

# High-frequency dictionary of modern everyday loanwords and standard Hinglish terms
COMMON_HINGLISH_WORDS: dict[str, str] = {
    # Tech, Gadgets & Digital
    "मोबाइल": "mobile",
    "फ़ोन": "phone",
    "फोन": "phone",
    "स्कैम": "scam",
    "इंटरनेट": "internet",
    "कंप्यूटर": "computer",
    "कम्प्यूटर": "computer",
    "लैपटॉप": "laptop",
    "पासवर्ड": "password",
    "कैमरा": "camera",
    "वीडियो": "video",
    "स्क्रीन": "screen",
    "मैसेज": "message",
    "कॉल": "call",
    "ऑनलाइन": "online",
    "ऑफलाइन": "offline",
    "सिग्नल": "signal",
    "बैटरी": "battery",
    "साइबर": "cyber",
    "नेटवर्क": "network",
    "हैक": "hack",
    "हैकर": "hacker",
    "अकाउंट": "account",
    "बैंक": "bank",
    "ओटीपी": "OTP",
    "वेबसाइट": "website",
    "लिंक": "link",
    "ऐप": "app",
    "ऐप्लिकेशन": "application",
    "डेटा": "data",
    "क्लिक": "click",
    "डाउनलोड": "download",
    "अपलोड": "upload",
    "वायरल": "viral",
    "स्मार्टफोन": "smartphone",
    "अलर्ट": "alert",
    "फ्यूचर": "future",
    "फ्यूचरिस्टिक": "futuristic",
    "रोबोट": "robot",
    "एआई": "AI",
    "कोड": "code",
    "सिस्टम": "system",
    "टेक्नोलॉजी": "technology",
    "डिजिटल": "digital",
    "सिक्योरिटी": "security",
    "सुरक्षा": "suraksha",
    "खतरा": "khatra",
    "खतरनाक": "khatarnak",
    "डीपफेक": "deepfake",
    "स्कैमर": "scammer",
    "स्कैमर्स": "scammers",
    "यूजर": "user",
    "यूजर्स": "users",
    "यूज़र्स": "users",
    "एक्शन": "action",
    "डिफेंस": "defense",
    "सीक्रेट": "secret",
    "प्राइवेट": "private",
    "पर्सनल": "personal",
    "डिटेल्स": "details",
    "इन्फॉर्मेशन": "information",
    "नंबर": "number",
    "फोनपे": "PhonePe",
    "गूगलपे": "GooglePay",
    "पेटीएम": "Paytm",
    "यूपीआई": "UPI",
    "ट्रांजैक्शन": "transaction",
    "पेमेंट": "payment",
    "क्रेडिट": "credit",
    "डेबिट": "debit",
    "कार्ड": "card",
    "पिन": "PIN",
    "वेरिफिकेशन": "verification",
    "अवेयरनेस": "awareness",
    "सपोर्ट": "support",
    "कस्टमर": "customer",
    "केयर": "care",
    "सर्विस": "service",
    "कंपनी": "company",
    "रियल": "real",
    "फेक": "fake",
    "ट्रू": "true",
    "कॉलर": "caller",
    "ट्रूकॉलर": "Truecaller",
    "व्हाट्सएप": "WhatsApp",
    "वॉट्सऐप": "WhatsApp",
    "यूट्यूब": "YouTube",
    "इंस्टाग्राम": "Instagram",
    "फेसबुक": "Facebook",
    "सोशल": "social",
    "मीडिया": "media",
    "पोस्ट": "post",
    "लाइक": "like",
    "सब्सक्राइब": "subscribe",
    "चैनल": "channel",
    "क्रिएटर": "creator",
    "शॉर्ट्स": "shorts",
    "रील": "reel",
    "रील्स": "reels",
    "कंटेंट": "content",
    "पैसे": "paise",
    "रुपये": "rupaye",
    "रुपए": "rupaye",
    "चोरी": "chori",
    "पहचान": "pehchan",
    "रोकें": "rokein",
    "रोकना": "rokna",
    "फंस": "phans",
    "फंसे": "phanse",
    "बचाएं": "bachayein",
    "बचाव": "bachav",
    "ट्रिक": "trick",

    # Society, Professions & Places
    "डॉक्टर": "doctor",
    "हॉस्पिटल": "hospital",
    "पुलिस": "police",
    "ऑफिस": "office",
    "स्टेशन": "station",
    "ट्रेन": "train",
    "रेलगाड़ी": "railgadi",
    "कार": "car",
    "गाड़ी": "gadi",
    "रोड": "road",
    "ट्रैफिक": "traffic",
    "स्कूल": "school",
    "कॉलेज": "college",
    "मार्केट": "market",
    "दुकान": "dukaan",
    "होटल": "hotel",
    "घर": "ghar",
    "कमरा": "kamra",
    "दोस्त": "dost",
    "लोग": "log",
    "परिवार": "parivaar",
    "दुनिया": "duniya",

    # Common Hinglish verbs, auxiliaries & particles
    "है": "hai",
    "हैं": "hain",
    "था": "tha",
    "थी": "thi",
    "थे": "the",
    "हो": "ho",
    "होगा": "hoga",
    "होगी": "hogi",
    "होंगे": "honge",
    "होता": "hota",
    "होती": "hoti",
    "होते": "hote",
    "होने": "hone",
    "में": "mein",
    "से": "se",
    "को": "ko",
    "का": "ka",
    "की": "ki",
    "के": "ke",
    "और": "aur",
    "या": "ya",
    "पर": "par",
    "ने": "ne",
    "भी": "bhi",
    "तो": "to",
    "नहीं": "nahi",
    "न": "na",
    "मत": "mat",
    "कर": "kar",
    "करो": "karo",
    "करें": "karein",
    "करते": "karte",
    "करती": "karti",
    "करता": "karta",
    "किया": "kiya",
    "कीजिए": "kijiye",
    "रहा": "raha",
    "रही": "rahi",
    "रहे": "rahe",
    "रहो": "raho",
    "रहें": "rahein",
    "सावधान": "savdhan",
    "सतर्क": "satark",
    "सुरक्षा": "suraksha",
    "सुरक्षित": "surakshit",
    "जिंदगी": "zindagi",
    "ज़िंदगी": "zindagi",
    "जीवन": "jeevan",
    "सबसे": "sabse",
    "सब": "sab",
    "कभी": "kabhi",
    "अपना": "apna",
    "अपनी": "apni",
    "अपने": "apne",
    "किसी": "kisi",
    "किस": "kis",
    "साथ": "saath",
    "बात": "baat",
    "बातें": "baatein",
    "दौर": "daur",
    "अहम": "ahem",
    "हिस्सा": "hissa",
    "ध्यान": "dhyan",
    "याद": "yaad",
    "रखें": "rakhein",
    "रखो": "rakho",
    "रखना": "rakhna",
    "जानें": "jaanein",
    "जानना": "jaanna",
    "जानिए": "jaaniye",
    "जानते": "jaante",
    "समझें": "samjhein",
    "समझो": "samjho",
    "समझना": "samjhna",
    "बताएं": "batayein",
    "बताओ": "batao",
    "बताना": "batana",
    "शुरू": "shuru",
    "शुरुआत": "shuruat",
    "खत्म": "khatam",
    "पहला": "pehla",
    "पहली": "pehli",
    "पहले": "pehle",
    "दूसरा": "doosra",
    "दूसरी": "doosri",
    "दूसरे": "doosre",
    "तीसरा": "teesra",
    "तीसरी": "teesri",
    "तीसरे": "teesre",
    "समय": "samay",
    "वक्त": "waqt",
    "जरूरी": "zaroori",
    "ज़रूरी": "zaroori",
    "जरूरत": "zaroorat",
    "ज़रूरत": "zaroorat",
    "आसान": "aasan",
    "मुश्किल": "mushkil",
    "सवाल": "sawal",
    "जवाब": "jawab",
    "तरीका": "tareeka",
    "तरीके": "tareeke",
    "टिप्स": "tips",
    "ट्रिक्स": "tricks",
    "स्टेप्स": "steps",
    "कारण": "karan",
    "वजह": "wajah",
    "नुकसान": "nuksan",
    "फायदा": "fayda",
    "सामने": "saamne",
    "पीछे": "peeche",
    "ऊपर": "upar",
    "नीचे": "neeche",
    "अंदर": "andar",
    "बाहर": "bahar",
    "रात": "raat",
    "दिन": "din",
    "सुबह": "subah",
    "शाम": "shaam",
    "कहानी": "kahani",
    "हवेली": "haveli",
    "जंगल": "jungle",
    "अंधेरा": "andhera",
    "रोशनी": "roshni",
    "खौफ": "khauf",
    "डर": "dar",
    "सन्नाटा": "sannata",
    "रहस्य": "rahasya",
    "आवाज़": "aawaz",
    "आवाज": "aawaz",
    "दरवाज़ा": "darwaza",
    "दरवाजा": "darwaza",
    "कमरा": "kamra",
    "साया": "saaya",
    "परछाई": "parchhai",
    "मौत": "maut",
    "जिंदा": "zinda",
    "ज़िंदा": "zinda",
    "हमेशा": "hamesha",
    "सिर्फ": "sirf",
    "केवल": "kewal",
    "क्योंकि": "kyunki",
    "इसलिए": "isliye",
    "अगर": "agar",
    "मगर": "magar",
    "लेकिन": "lekin",
    "खुद": "khud",
    "पैसा": "paisa",
    "खाता": "khaata",
    "अंतिम": "antim",
    "प्रतीक्षा": "prateeksha",
    "अकेला": "akela",
    "योद्धा": "yoddha",
    "घातक": "ghatak",
    "प्रहार": "prahaar",
    "जीत": "jeet",
    "हार": "haar",
    "सच": "sach",
    "झूठ": "jhooth",
    "आज": "aaj",
    "कल": "kal",
    "अब": "ab",
    "जब": "jab",
    "तब": "tab",
    "यहाँ": "yahan",
    "वहाँ": "wahan",
    "कहाँ": "kahan",
    "आप": "aap",
    "हम": "hum",
    "तुम": "tum",
    "मेरा": "mera",
    "मेरी": "meri",
    "मेरे": "mere",
    "आपका": "aapka",
    "आपकी": "aapki",
    "आपके": "aapke",
    "हमारा": "hamara",
    "हमारी": "hamari",
    "हमारे": "hamare",
    "क्या": "kya",
    "क्यों": "kyun",
    "कैसे": "kaise",
    "कौन": "kaun",
    "कितना": "kitna",
    "बहुत": "bahut",
    "ज्यादा": "zyada",
    "कम": "kam",
    "अच्छा": "achha",
    "बुरा": "bura",
    "नया": "naya",
    "नई": "nayi",
    "नए": "naye",
    "पुराना": "purana",
    "पुराने": "purane",
    "बड़ा": "bada",
    "बड़ी": "badi",
    "बड़े": "bade",
    "छोटा": "chhota",
    "छोटी": "chhoti",
    "छोटे": "chhote",
    "कनेक्टेड": "connected",
    "शेयर": "share",
    "बचें": "bachein",
    "बचना": "bachna",
    "सकते": "sakte",
    "सकती": "sakti",
    "सकता": "sakta",
    "जानते": "jaante",
    "जानिए": "jaaniye",
    "समझें": "samjhein",
    "सुरक्षित": "surakshit",
    "विश्वास": "vishwas",
    "धोखा": "dhokha",
    "फ्रॉड": "fraud",
    "अंजाना": "anjaana",
    "अंजान": "anjaan",
}


def is_devanagari(text: str) -> bool:
    """Return True if string contains Devanagari script characters."""
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


def deva_word_to_hinglish(word: str) -> str:
    """Transliterate a single Devanagari or mixed word into clean Roman Hinglish."""
    if not word or not is_devanagari(word):
        return word

    # Extract leading/trailing punctuation (ASCII and Devanagari dandas)
    punct_pattern = r"^[.,!?:;'\"><#@$%^&*()\[\]{}\-_=+–—/\\|`~॥।\s]+"
    trail_pattern = r"[.,!?:;'\"><#@$%^&*()\[\]{}\-_=+–—/\\|`~॥।\s]+$"
    lead_match = re.match(punct_pattern, word)
    lead_punct = lead_match.group(0) if lead_match else ""
    trail_match = re.search(trail_pattern, word)
    trail_punct = trail_match.group(0) if trail_match else ""

    clean_w = word[len(lead_punct) : len(word) - len(trail_punct) if trail_punct else len(word)]
    if not clean_w:
        return word

    # Normalize danda punctuation to Latin period
    lead_punct = lead_punct.replace("।", ".").replace("॥", ".")
    trail_punct = trail_punct.replace("।", ".").replace("॥", ".")

    # Fast path: High-frequency loanword dictionary
    lower_w = clean_w.lower()
    if clean_w in COMMON_HINGLISH_WORDS:
        trans = COMMON_HINGLISH_WORDS[clean_w]
        return f"{lead_punct}{trans}{trail_punct}"
    if lower_w in COMMON_HINGLISH_WORDS:
        trans = COMMON_HINGLISH_WORDS[lower_w]
        return f"{lead_punct}{trans}{trail_punct}"

    chars = list(clean_w)
    out: list[str] = []
    n = len(chars)
    i = 0

    while i < n:
        c = chars[i]

        # Check special nasalized matra combinations (e.g. ें -> 'ein', ों -> 'on', ायें -> 'ayein')
        if c == "\u0947" and i + 1 < n and chars[i + 1] in ("\u0902", "\u0901"):  # ें -> 'ein'
            out.append("ein")
            i += 2
            continue
        if c == "\u094B" and i + 1 < n and chars[i + 1] in ("\u0902", "\u0901"):  # ों -> 'on'
            out.append("on")
            i += 2
            continue

        # Check special two-char conjuncts
        if i + 2 < n and chars[i + 1] == "\u094D":  # consonant + virama + consonant
            combo = c + chars[i + 1] + chars[i + 2]
            if combo in DEVA_CONJUNCTS:
                out.append(DEVA_CONJUNCTS[combo])
                i += 3
                continue

        # Check nukta combination (e.g. क़, ज़, फ़, ड़)
        if i + 1 < n and chars[i + 1] == "\u093C":
            combo = c + "\u093C"
            cons = DEVA_CONSONANTS.get(combo, DEVA_CONSONANTS.get(c, c))
            i += 2
        elif c in DEVA_CONSONANTS:
            cons = DEVA_CONSONANTS[c]
            i += 1
        elif c in DEVA_VOWELS:
            out.append(DEVA_VOWELS[c])
            i += 1
            continue
        elif c in DEVA_MATRAS:
            out.append(DEVA_MATRAS[c])
            i += 1
            continue
        elif c == "\u094D":  # standalone virama
            i += 1
            continue
        elif c in ("।", "॥"):
            out.append(".")
            i += 1
            continue
        else:
            out.append(c)
            i += 1
            continue

        # Determine if inherent vowel 'a' should follow consonant
        if i < n:
            next_c = chars[i]
            if next_c in DEVA_MATRAS:
                out.append(cons)
                out.append(DEVA_MATRAS[next_c])
                i += 1
            elif next_c == "\u094D":  # halant / virama cancels vowel
                out.append(cons)
                i += 1
            else:
                # Schwa deletion: if it is the final consonant of a multi-char word, drop 'a'
                if i == n and len(clean_w) > 1:
                    out.append(cons)
                else:
                    out.append(cons + "a")
        else:
            # Word-final consonant (Hindi schwa deletion)
            if len(clean_w) <= 1:
                out.append(cons + "a")
            else:
                out.append(cons)

    result = "".join(out)
    return f"{lead_punct}{result}{trail_punct}"


def deva_to_hinglish(text: str) -> str:
    """Transliterate an entire text string from Devanagari to clean Roman Hinglish."""
    if not text or not is_devanagari(text):
        return text

    words = text.split()
    converted = [deva_word_to_hinglish(w) for w in words]
    res = " ".join(converted)
    # Clean up double danda or repeated dots
    res = re.sub(r"\s*\|\s*", " ", res)
    res = re.sub(r"\.\.+", ".", res)
    return res


def format_srt_timestamp(seconds: float) -> str:
    """Convert floating point seconds to SRT timecode: HH:MM:SS,mmm"""
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    minute = total_min % 60
    hour = total_min // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def export_srt(
    timeline_scenes: list[dict],
    out_path: str | Path,
    words_per_cue: int = 4,
    transliterate_hinglish: bool = True,
) -> Path:
    """Generate a standard SubRip (.srt) subtitle file from timeline scenes.

    Groups word-level timings into synchronized cues.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cues: list[dict] = []
    cue_index = 1
    srt_lines: list[str] = []

    for scene in timeline_scenes:
        words = scene.get("words", [])
        if not words:
            continue
        scene_start = float(scene.get("start", 0.0))

        # Group words into subtitle cues
        chunk_size = max(1, words_per_cue)
        for i in range(0, len(words), chunk_size):
            chunk = words[i : i + chunk_size]
            if not chunk:
                continue

            # Word start/end are relative to scene start or scene audio
            cue_start = scene_start + float(chunk[0].get("start", 0.0))
            cue_end = scene_start + float(chunk[-1].get("end", cue_start + 1.0))

            raw_text = " ".join(w.get("text", "") for w in chunk)
            if transliterate_hinglish and is_devanagari(raw_text):
                text = deva_to_hinglish(raw_text)
            else:
                text = raw_text

            if not text.strip():
                continue

            start_tc = format_srt_timestamp(cue_start)
            end_tc = format_srt_timestamp(cue_end)

            srt_lines.append(f"{cue_index}")
            srt_lines.append(f"{start_tc} --> {end_tc}")
            srt_lines.append(text.strip())
            srt_lines.append("")
            cue_index += 1

    content = "\n".join(srt_lines)
    out_file.write_text(content, encoding="utf-8")
    return out_file
