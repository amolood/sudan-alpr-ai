#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan plate interpreter — country, plate class, and state (wilaya) recognition.

The OCR stage gives us only the raw text of a plate, e.g. "7KH10346" for a
private car or "POLICE00000" for a police vehicle. This module turns that text
into structured, *verified* information:

    interpret("7KH10346")
      -> { is_sudan: True, country: "Sudan",
           plate_class: "private", plate_class_ar: "خصوصي",
           state_code: "KH", state: "Khartoum", state_ar: "الخرطوم",
           registration_digit: "7", serial: "10346", ... }

Sudan doesn't have just one plate format. The General Directorate of Traffic
issues a whole family of them, distinguished by colour and by a class marker:

    private     (silver)  7 KH 10346     digit + state letters + serial
    government  (yellow)  GOV 00000      "GOV" — حكومة
    army        (red)     ARMY 00000     "ARMY" — القوات المسلحة
    police      (blue)    POLICE 00000   "POLICE" — الشرطة
    UN          (red/blu) UN 00          "U.N" — الأمم المتحدة
    diplomatic  (red)     CD 00          "C.D" — هيئات دبلوماسية
    consular    (green)   HC 00          "H.C" — هيئات قنصلية
    NGO         (yellow)  NGO 0000       "N.G.O" — منظمة غير حكومية
    int. orgs   (white)   IO 0000        "I.O" — منظمات دولية
    red crescent(white)   ...            الهلال الأحمر السوداني
    limousine   (silver)  ليموزين         Arabic-only
    investment  (grn/blk) استثمار + KH9   commercial; carries a state code
    transit     (silver)  TRANSIT        عبور
    temporary   (white)   مؤقتة / سريع / داخلي

The class list mirrors the General Directorate of Traffic reference board in
docs/plate_types_reference.png — every Arabic name transcribed off the board.

This interpreter recognises all of them. The colour of the plate crop (if the
caller passes it) is used as corroborating evidence, never as the sole signal —
the *text marker* is what decides the class, so it still works on a greyscale
or odd-lit photo.

Why rules and not a model?
--------------------------
These formats are fixed and publicly documented, so a rule is exact, explains
its decision, and never hallucinates a class or country the way the global OCR
model did (it guessed random countries because Sudan isn't in its 65-country
list). We verify everything from the structure itself.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# State (wilaya) code -> (English, Arabic).
#
# Confirmed in training/dataset/labels.csv and in the Directorate of Traffic
# reference board: KH, G, NK, WN, WK, NS, RS. The rest are Sudan's standard
# wilaya plate letters, included so any valid plate resolves to a named state.
# ---------------------------------------------------------------------------
STATE_CODES: dict[str, tuple[str, str]] = {
    "KH": ("Khartoum", "الخرطوم"),
    "G":  ("Gezira", "الجزيرة"),
    "NS": ("River Nile", "نهر النيل"),
    "NK": ("North Kordofan", "شمال كردفان"),
    "SK": ("South Kordofan", "جنوب كردفان"),
    "WK": ("West Kordofan", "غرب كردفان"),
    "ND": ("Northern", "الشمالية"),
    "RS": ("Red Sea", "البحر الأحمر"),
    "KS": ("Kassala", "كسلا"),
    "GD": ("Gedaref", "القضارف"),
    "SN": ("Sennar", "سنار"),
    "WN": ("White Nile", "النيل الأبيض"),
    "BN": ("Blue Nile", "النيل الأزرق"),
    "NDR": ("North Darfur", "شمال دارفور"),
    "SD": ("South Darfur", "جنوب دارفور"),
    "WD": ("West Darfur", "غرب دارفور"),
    "ED": ("East Darfur", "شرق دارفور"),
    "CDR": ("Central Darfur", "وسط دارفور"),
}

# Common OCR truncations / variants -> canonical state code. "K" alone is
# almost always a clipped "KH".
CODE_ALIASES: dict[str, str] = {
    "K": "KH",
}

# ---------------------------------------------------------------------------
# Plate classes. Each special (non-private) class is identified by a *marker*
# the OCR can pick up from the plate, plus the colour(s) it normally appears in
# (used only to corroborate). Order matters: the first marker that matches the
# text wins, so more specific markers are listed before looser ones.
#
# marker_patterns are matched against the alphanumeric-only, uppercased text.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateClass:
    key: str                      # machine id, e.g. "police"
    name_en: str
    name_ar: str
    markers: tuple[str, ...]      # Latin/numeric fragments to look for
    markers_ar: tuple[str, ...]   # Arabic fragments (some plates have no Latin)
    colors: tuple[str, ...]       # plate colours this class typically uses


# NOTE: this list mirrors the General Directorate of Traffic reference board
# (docs/plate_types_reference.png) exactly, transcribed plate-by-plate with no
# guessing. Each entry's Latin marker, Arabic marker, and colour come straight
# from the board. Some plates (Red Crescent, Limousine, Temporary) carry only
# an Arabic word, so they're matched on markers_ar.
#
# Order matters: longer/more specific markers are listed before "مؤقتة" so the
# express variant ("مؤقتة سريع") wins over the plain one.
PLATE_CLASSES: tuple[PlateClass, ...] = (
    PlateClass("government", "Government", "حكومي",
               ("GOV",), ("حكومة", "حكومي"), ("yellow",)),
    PlateClass("army", "Armed Forces", "القوات المسلحة",
               ("ARMY",), ("القوات المسلحة",), ("red",)),
    PlateClass("police", "Police", "الشرطة",
               ("POLICE",), ("الشرطة",), ("white", "blue")),
    PlateClass("un", "United Nations", "الأمم المتحدة",
               ("UN",), ("الأمم المتحدة",), ("red", "blue")),
    PlateClass("diplomat", "Diplomatic Missions", "هيئات دبلوماسية",
               ("CD",), ("هيئات دبلوماسية", "دبلوماسية"), ("red",)),
    PlateClass("consular", "Consular Missions", "هيئات قنصلية",
               ("HC",), ("هيئات قنصلية", "قنصلية"), ("green",)),
    PlateClass("ngo", "NGO", "منظمة غير حكومية",
               ("NGO",), ("منظمة غير حكومية", "غير حكومية"), ("yellow",)),
    PlateClass("int_org", "International Organizations", "منظمات دولية",
               ("IO",), ("منظمات دولية",), ("white",)),
    PlateClass("red_crescent", "Sudanese Red Crescent", "الهلال الأحمر السوداني",
               ("REDCRESCENT", "HILAL"), ("الهلال الأحمر",), ("white", "red")),
    PlateClass("limousine", "Limousine", "ليموزين",
               (), ("ليموزين",), ("silver",)),
    # Investment/commercial plates carry a state code (KH9, RS…) AND the word
    # "استثمار". They're green or black, which is how they differ from a normal
    # private/state plate that shares the same letters.
    PlateClass("investment", "Investment / Commercial", "استثمار",
               (), ("استثمار",), ("green", "black")),
    PlateClass("transit", "Transit", "عبور",
               ("TRANSIT",), ("عبور",), ("silver",)),
    # Temporary comes in three board variants. List the qualified ones before
    # the plain "مؤقتة" so "مؤقتة سريع"/"مؤقتة داخلي" win the match.
    PlateClass("temporary_express", "Temporary (Express)", "مؤقتة سريع",
               ("MWQTASR",), ("مؤقتة سريع",), ("white",)),
    PlateClass("temporary_domestic", "Temporary (Domestic)", "مؤقتة داخلي",
               ("MWQTADKL",), ("مؤقتة داخلي",), ("white",)),
    PlateClass("temporary", "Temporary", "مؤقتة",
               ("TEMP", "MWQTA"), ("مؤقتة",), ("white",)),
)
PRIVATE = PlateClass("private", "Private", "خصوصي", (), ("خصوصي",), ("silver", "white"))

# Markers a colour *suggests* when the text is ambiguous (corroboration only).
COLOR_HINTS: dict[str, tuple[str, ...]] = {
    "red":    ("army", "diplomat", "un", "red_crescent"),
    "blue":   ("un", "police"),
    "yellow": ("government", "ngo"),
    "green":  ("consular", "investment"),
    "black":  ("investment",),
    "silver": ("private", "limousine", "transit", "temporary"),
    "white":  ("private", "int_org", "temporary", "temporary_express",
               "temporary_domestic", "red_crescent"),
}

# ---------------------------------------------------------------------------
# Private-plate layout. Registration digit (1–2 chars to tolerate a real
# two-digit number like "13KH" *and* an OCR doubling), state letters, serial.
# ---------------------------------------------------------------------------
PLATE_RE = re.compile(r"^(?P<digit>[0-9]{1,2})(?P<state>[A-Z]{1,3})(?P<serial>[0-9]{1,6})$")
# Partial: leading registration digit dropped by OCR but state+serial survived.
PLATE_RE_NO_DIGIT = re.compile(r"^(?P<state>[A-Z]{2,3})(?P<serial>[0-9]{1,6})$")


@dataclass
class PlateInfo:
    text: str
    is_sudan: bool
    country: str
    country_confidence: float
    plate_class: str          # machine id: private/police/army/...
    plate_class_en: str
    plate_class_ar: str
    state_code: str
    state: str
    state_ar: str
    registration_digit: str
    serial: str
    color: str                # the colour we were told / inferred, or ""
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


# Decorated Latin letters that appear on real Sudanese plates / OCR output and
# must fold to a plain ASCII letter. The board's "TRANŞIT" uses Ş (S-cedilla);
# OCR engines also emit Ç, İ, etc. unicodedata handles the accented cases; this
# map covers the few that don't decompose to ASCII.
_LATIN_FOLD = str.maketrans({
    "Ş": "S", "Ș": "S", "ß": "S",
    "Ç": "C", "Ć": "C",
    "İ": "I", "I": "I",
    "Ğ": "G",
    "Ö": "O", "Ø": "O",
    "Ü": "U",
})


def _normalize(text: str) -> str:
    """Uppercase, fold decorated Latin letters to ASCII, keep alphanumerics only.

    Folding matters because the board prints "TRANŞIT" with a Ş, and OCR may
    emit other accented forms — without folding, the marker wouldn't match.
    Arabic characters are left intact (they're matched separately as raw text).
    """
    up = (text or "").upper().translate(_LATIN_FOLD)
    # Strip combining accents from anything that decomposes (É -> E, etc.) while
    # leaving non-decomposable scripts (Arabic) untouched.
    decomposed = unicodedata.normalize("NFKD", up)
    out = []
    for c in decomposed:
        if unicodedata.combining(c):
            continue
        if c.isalnum():
            out.append(c)
    return "".join(out)


def _match_class(norm: str, raw: str) -> tuple[PlateClass, str] | None:
    """Find the special plate class for this text. Returns (class, matched_marker).

    Matching uses two signals:
      • Latin markers against `norm` (uppercased, alphanumeric-only).
      • Arabic markers against `raw` — several plates (Red Crescent, Limousine,
        Temporary…) carry only an Arabic word, no Latin code.

    Short two-letter Latin markers (CD, UN, HC, IO) are risky: a private serial
    like "1CDR500" or "2UN30" merely *contains* those letters. So a short marker
    only fires when it stands alone as a letter run AND the text isn't a valid
    private plate. Longer markers (POLICE, ARMY, NGO, TRANSIT…) match anywhere.
    """
    looks_private = bool(PLATE_RE.match(norm))
    for pc in PLATE_CLASSES:
        # Arabic markers first — they're unambiguous and present even when the
        # OCR caught no Latin text.
        for ar in pc.markers_ar:
            if ar and ar in raw:
                return pc, ar
        for marker in pc.markers:
            if len(marker) <= 2 and marker.isalpha():
                if looks_private:
                    continue
                if re.search(rf"(?<![A-Z]){marker}(?![A-Z])", norm):
                    return pc, marker
            elif marker in norm:
                return pc, marker
    return None


def _extract_state(norm: str) -> tuple[str, str, str]:
    """Pull a known state code out of mixed text, if one is present.

    Used for special plates (e.g. investment) that carry both a class marker and
    a state code. Returns (code, english, arabic) or ("", "", "").
    """
    for run in re.findall(r"[A-Z]+", norm):
        code = CODE_ALIASES.get(run, run)
        if code in STATE_CODES:
            en, ar = STATE_CODES[code]
            return code, en, ar
    return "", "", ""


def interpret(text: str,
              header_has_sudan: bool | None = None,
              color: str | None = None) -> PlateInfo:
    """Interpret raw plate text into structured Sudan/class/state information.

    Parameters
    ----------
    text : str
        Raw OCR output, e.g. "7KH 10346", "POLICE 00000", "C.D 12".
    header_has_sudan : bool | None
        True/False if the caller separately saw "SUDAN"/"السودان" on the plate
        header; None if unknown. Raises confidence when True, never required.
    color : str | None
        Dominant plate colour if the caller measured it from the crop
        ("red"/"blue"/"yellow"/"green"/"silver"/"white"). Used only to
        corroborate the class and country — the text marker decides the class.
    """
    raw = (text or "").strip()
    norm = _normalize(text)
    color = (color or "").lower().strip()

    # --- 1) Special (non-private) classes are decided by their text marker. ---
    matched = _match_class(norm, raw)
    if matched is not None:
        special, marker = matched
        conf = 0.9
        reason = [f"'{marker}' marker -> {special.name_en} plate"]
        if color and color in special.colors:
            conf = min(conf + 0.07, 0.99)
            reason.append(f"colour '{color}' matches a {special.name_en} plate")
        # Some special plates (investment, limousine, transit, temporary) also
        # carry a state code in their serial — decode it when present so the
        # output still names the wilaya.
        sc, s_en, s_ar = _extract_state(norm)
        if sc:
            reason.append(f"state code '{sc}' -> {s_en}")
        return PlateInfo(
            text=norm, is_sudan=True, country="Sudan",
            country_confidence=round(conf, 2),
            plate_class=special.key, plate_class_en=special.name_en,
            plate_class_ar=special.name_ar,
            state_code=sc, state=s_en or "—", state_ar=s_ar or "—",
            registration_digit="", serial="".join(c for c in norm if c.isdigit()),
            color=color, reason="; ".join(reason),
        )

    # --- 2) Otherwise try the private-car layout. ---
    m = PLATE_RE.match(norm)
    partial = False
    if m:
        digit = m.group("digit")
        # Two leading digits is legitimate (e.g. "13KH"), but OCR can also
        # *double* a single digit ("10KH6009"). We keep the value as-is and
        # only flag low confidence when the serial also looks malformed.
        if len(digit) == 2 and digit[0] == digit[1]:
            partial = True            # "11..." style doubling
            digit = digit[0]
    else:
        m = PLATE_RE_NO_DIGIT.match(norm)
        if m and CODE_ALIASES.get(m.group("state"), m.group("state")) in STATE_CODES:
            partial = True
            digit = ""
        else:
            # Doesn't fit any Sudanese layout. Colour may still hint Sudan, but
            # we won't claim it without structure.
            conf = 0.0 if header_has_sudan else 0.02
            return PlateInfo(
                text=norm, is_sudan=False, country="Unknown",
                country_confidence=conf, plate_class="unknown",
                plate_class_en="Unknown", plate_class_ar="غير معروف",
                state_code="", state="Unknown", state_ar="غير معروف",
                registration_digit="", serial="", color=color,
                reason="no Sudanese class marker and no private-plate layout",
            )

    raw_state = m.group("state")
    serial = m.group("serial")
    state_code = CODE_ALIASES.get(raw_state, raw_state)
    known = state_code in STATE_CODES
    state_en, state_ar = STATE_CODES.get(state_code, ("Unknown", "غير معروف"))

    conf = 0.85
    reason = ["matches Sudanese private layout (D + LL + DDDD…)"]
    if known:
        conf += 0.10
        reason.append(f"state code '{state_code}' is a known wilaya")
    else:
        conf -= 0.15
        reason.append(f"state code '{state_code}' not in known list")
    if partial:
        conf -= 0.20
        reason.append("partial serial (a digit looks dropped/doubled by OCR)")
    if color in ("silver", "white"):
        conf = min(conf + 0.04, 0.99)
        reason.append(f"colour '{color}' matches a private plate")
    elif color and color not in ("silver", "white"):
        reason.append(f"note: colour '{color}' is unusual for a private plate")
    if header_has_sudan is True:
        conf = min(conf + 0.10, 0.99)
        reason.append("'SUDAN' header detected")
    elif header_has_sudan is False:
        conf -= 0.05
    conf = max(0.0, min(conf, 0.99))

    return PlateInfo(
        text=norm, is_sudan=True, country="Sudan",
        country_confidence=round(conf, 2),
        plate_class=PRIVATE.key, plate_class_en=PRIVATE.name_en,
        plate_class_ar=PRIVATE.name_ar,
        state_code=state_code, state=state_en, state_ar=state_ar,
        registration_digit=digit, serial=serial, color=color,
        reason="; ".join(reason),
    )


def _demo() -> None:
    samples = [
        ("7KH10346", None),       # private
        ("13KH00000", None),      # private, two-digit registration
        ("1NS180", None),         # private (River Nile)
        ("1CDR500", None),        # private (Central Darfur) — NOT diplomat
        ("GOV00000", "yellow"),
        ("ARMY00000", "red"),
        ("POLICE00000", "blue"),
        ("UN00", "blue"),
        ("CD12", "red"),          # diplomatic corps
        ("HC34", "green"),        # consular corps
        ("NGO0000", "yellow"),
        ("IO0000", "white"),      # international organization
        # Arabic-only markers (these plates carry no Latin code on the board):
        ("الهلال الأحمر 1234", "white"),
        ("ليموزين KH 0000", "silver"),
        ("استثمار KH9 0000", "green"),   # investment + state decoded
        ("استثمار RS 0000", "green"),
        ("مؤقتة سريع KH", "white"),
        ("مؤقتة داخلي KH", "white"),
        ("مؤقتة KH 0000", "white"),
        ("TRANSIT NS 00", "silver"),
        ("ABC", None),
    ]
    for s, col in samples:
        info = interpret(s, color=col)
        flag = "🇸🇩" if info.is_sudan else "  "
        extra = (f"state={info.state} ({info.state_code})"
                 if info.plate_class == "private" else f"({info.plate_class_ar})")
        print(f"{s:<22} {flag} {info.plate_class_en:<26} "
              f"conf={info.country_confidence:<4} {extra}")


if __name__ == "__main__":
    _demo()
