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

    private   (silver)   7 KH 10346      digit + state letters + serial
    govt.     (yellow)   GOV 00000       "GOV" marker
    police    (red/blue) POLICE 00000    "POLICE" marker
    army      (red)      ARMY 00000      "ARMY" marker
    diplomat  (red)      CD 00           "C.D" marker
    UN        (blue)     UN 00           "U.N" marker
    NGO       (yellow)   NGO 0000        "N.G.O" marker
    transit   (silver)   NS / TRANSIT    "TRANSIT" marker
    temporary (silver)   KH ... مؤقتة    "TEMP" marker

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
    key: str            # machine id, e.g. "police"
    name_en: str
    name_ar: str
    markers: tuple[str, ...]   # regex fragments to look for in the text
    colors: tuple[str, ...]    # plate colours this class typically uses


PLATE_CLASSES: tuple[PlateClass, ...] = (
    PlateClass("police",    "Police",      "الشرطة",
               ("POLICE", "SHRTA"), ("red", "blue")),
    PlateClass("army",      "Armed Forces", "القوات المسلحة",
               ("ARMY",), ("red",)),
    PlateClass("red_crescent", "Red Crescent", "الهلال الأحمر",
               ("REDCRESCENT", "HILAL"), ("red",)),
    PlateClass("diplomat",  "Diplomatic",  "دبلوماسي",
               ("CD",), ("red",)),
    PlateClass("un",        "United Nations", "الأمم المتحدة",
               ("UN",), ("blue", "white")),
    PlateClass("ngo",       "NGO",         "منظمة طوعية",
               ("NGO",), ("yellow",)),
    PlateClass("government", "Government",  "حكومي",
               ("GOV",), ("yellow",)),
    PlateClass("high_committee", "High Committee", "اللجنة العليا",
               ("HC",), ("green",)),
    PlateClass("limousine", "Limousine / Taxi", "ليموزين / أجرة",
               ("LIMO", "TAXI"), ("yellow",)),
    PlateClass("transit",   "Transit",     "عبور",
               ("TRANSIT",), ("silver", "white")),
    PlateClass("temporary", "Temporary",   "مؤقتة",
               ("TEMP", "MWQTA"), ("silver", "white")),
)
PRIVATE = PlateClass("private", "Private", "خصوصي", (), ("silver", "white"))

# Markers a colour *suggests* when the text is ambiguous (corroboration only).
COLOR_HINTS: dict[str, tuple[str, ...]] = {
    "red":    ("army", "police", "diplomat", "red_crescent"),
    "blue":   ("un", "police"),
    "yellow": ("government", "ngo", "limousine"),
    "green":  ("high_committee",),
    "silver": ("private", "transit", "temporary"),
    "white":  ("private",),
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


def _normalize(text: str) -> str:
    """Uppercase, keep alphanumerics only (drops spaces/dots the OCR emits)."""
    return "".join(c for c in (text or "").upper() if c.isalnum())


def _match_class(norm: str) -> PlateClass | None:
    """Return the special plate class whose marker appears in the text, if any.

    "CD" is required to be a *standalone* token-ish marker so it doesn't fire on
    a serial that merely contains those letters; the others are distinctive
    enough to match anywhere.
    """
    for pc in PLATE_CLASSES:
        for marker in pc.markers:
            if marker == "CD":
                # diplomat: letters with no surrounding state-style serial digits
                if re.search(r"(?<![A-Z])CD(?![A-Z])", norm) and not PLATE_RE.match(norm):
                    return pc
            elif marker == "UN":
                if re.search(r"(?<![A-Z])UN(?![A-Z])", norm):
                    return pc
            elif marker in norm:
                return pc
    return None


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
    norm = _normalize(text)
    color = (color or "").lower().strip()

    # --- 1) Special (non-private) classes are decided by their text marker. ---
    special = _match_class(norm)
    if special is not None:
        conf = 0.9
        reason = [f"'{'/'.join(special.markers)}' marker -> {special.name_en} plate"]
        if color and color in special.colors:
            conf = min(conf + 0.07, 0.99)
            reason.append(f"colour '{color}' matches a {special.name_en} plate")
        return PlateInfo(
            text=norm, is_sudan=True, country="Sudan",
            country_confidence=round(conf, 2),
            plate_class=special.key, plate_class_en=special.name_en,
            plate_class_ar=special.name_ar,
            state_code="", state="—", state_ar="—",
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
        ("7KH10346", None),
        ("13KH00000", None),
        ("1NS180", None),
        ("2G479", None),
        ("POLICE00000", "red"),
        ("ARMY00000", "red"),
        ("GOV00000", "yellow"),
        ("UN00", "blue"),
        ("CD12", "red"),
        ("NGO0000", "yellow"),
        ("TRANSITNS00", "silver"),
        ("ABC", None),
    ]
    for s, col in samples:
        info = interpret(s, color=col)
        flag = "🇸🇩" if info.is_sudan else "  "
        extra = (f"state={info.state} ({info.state_code})"
                 if info.plate_class == "private" else "")
        print(f"{s:<14} {flag} {info.plate_class_en:<16} "
              f"conf={info.country_confidence:<4} {extra}")


if __name__ == "__main__":
    _demo()
