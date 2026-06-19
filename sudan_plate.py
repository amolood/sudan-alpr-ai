#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan plate interpreter — country verification + state (wilaya) recognition.

The OCR stage gives us only the raw Latin serial of a plate, e.g. "7KH10346".
This module turns that string into structured, *verified* information:

    interpret("7KH10346")
      -> {
           "text": "7KH10346",
           "is_sudan": True,
           "country": "Sudan",
           "country_confidence": 0.97,
           "state_code": "KH",
           "state": "Khartoum",
           "state_ar": "الخرطوم",
           "registration_digit": "7",
           "serial": "10346",
           "reason": "matches Sudanese layout D + LL + DDDD…",
         }

Why a rule-based interpreter (and not a model)?
-----------------------------------------------
Sudanese civilian plates follow a fixed, publicly-known layout:

        ┌────────────────────────────────┐
        │      SUDAN        السودان       │   header (country, in words)
        │   N  <state>      <serial>      │   N = registration digit
        └────────────────────────────────┘

    Latin serial line  =  <digit><STATE LETTERS><serial digits>
    e.g.  7 KH 10346  ->  "7KH10346"

Because the format is deterministic, a model is unnecessary and worse: a rule
can *explain* its decision and never hallucinates a wrong country the way the
global OCR model did (it guessed random countries because Sudan isn't in its
65-country list). We verify "is this Sudanese?" from the structure itself.

The state-code map below is derived from (a) the codes actually present in our
labelled dataset and (b) Sudan's standard wilaya plate-letter scheme. Codes we
haven't positively confirmed are still accepted as Sudanese (the *layout* is
what proves the country) but reported with state="Unknown".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# State (wilaya) code -> name. Latin code : (English, Arabic).
#
# Codes confirmed present in training/dataset/labels.csv:
#   KH, G, NK, WN, WK, NS   (plus "K", treated as a truncated "KH" — see below)
# The rest are the standard Sudanese wilaya codes, included so any valid plate
# resolves to a named state, not just the ones our sample happened to contain.
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
    "ND2": ("North Darfur", "شمال دارفور"),
    "SD": ("South Darfur", "جنوب دارفور"),
    "WD": ("West Darfur", "غرب دارفور"),
    "ED": ("East Darfur", "شرق دارفور"),
    "CD": ("Central Darfur", "وسط دارفور"),
}

# Common OCR truncations / variants -> canonical code. "K" alone is almost
# always a clipped "KH" (the dataset has bare-"K" reads that are really KH).
CODE_ALIASES: dict[str, str] = {
    "K": "KH",
}

# A Sudanese Latin serial: one registration digit, 1–3 state letters, then the
# serial number (1–6 digits). e.g. 7KH10346, 1NS180, 2G479.
#   digit  = leading registration digit (1–2 chars to tolerate an OCR doubling
#            like "10KH6009"; we keep only the meaningful leading digit)
PLATE_RE = re.compile(r"^(?P<digit>[0-9]{1,2})(?P<state>[A-Z]{1,3})(?P<serial>[0-9]{1,6})$")

# A *partial* Sudanese serial where the OCR dropped the leading registration
# digit but still caught the state letters + serial. e.g. "KH5404", "KH19654".
# Recognised as Sudanese with lower confidence (the layout is right, a piece
# is missing).
PLATE_RE_NO_DIGIT = re.compile(r"^(?P<state>[A-Z]{2,3})(?P<serial>[0-9]{1,6})$")


@dataclass
class PlateInfo:
    text: str
    is_sudan: bool
    country: str
    country_confidence: float
    state_code: str
    state: str
    state_ar: str
    registration_digit: str
    serial: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    """Uppercase, keep alphanumerics only (drops spaces/dashes the OCR emits)."""
    return "".join(c for c in (text or "").upper() if c.isalnum())


def interpret(text: str, header_has_sudan: bool | None = None) -> PlateInfo:
    """Interpret a raw plate string into structured Sudan/state information.

    Parameters
    ----------
    text : str
        Raw OCR output, e.g. "7KH 10346" or "7kh10346".
    header_has_sudan : bool | None
        Optional extra evidence: True/False if the caller separately detected
        the word "SUDAN"/"السودان" on the plate's header line, None if unknown.
        When True it *raises* confidence; it is never required.

    The country decision is driven by the *structure* of the serial, which is
    specific to Sudanese civilian plates. Header evidence only adjusts the
    confidence score, so the function still works on a serial-only crop.
    """
    norm = _normalize(text)
    m = PLATE_RE.match(norm)
    partial = False

    if m:
        digit = m.group("digit")
        # An OCR doubling like "10KH6009" -> keep the last leading digit as the
        # registration digit (the real plate has a single one).
        if len(digit) > 1:
            partial = True
            digit = digit[-1]
    else:
        # Fallback: maybe the leading registration digit was dropped by the OCR
        # but the state letters + serial survived (e.g. "KH5404").
        m = PLATE_RE_NO_DIGIT.match(norm)
        if m and CODE_ALIASES.get(m.group("state"), m.group("state")) in STATE_CODES:
            partial = True
            digit = ""
        else:
            # Doesn't fit the Sudanese layout at all.
            conf = 0.0 if header_has_sudan else 0.02
            return PlateInfo(
                text=norm, is_sudan=False, country="Unknown",
                country_confidence=conf, state_code="", state="Unknown",
                state_ar="غير معروف", registration_digit="", serial="",
                reason="does not match Sudanese layout (D + LL + DDDD…)",
            )

    raw_state = m.group("state")
    serial = m.group("serial")

    state_code = CODE_ALIASES.get(raw_state, raw_state)
    known = state_code in STATE_CODES
    state_en, state_ar = STATE_CODES.get(state_code, ("Unknown", "غير معروف"))

    # Confidence: the layout match alone is strong evidence this is a Sudanese
    # plate. A recognised state code and/or a detected "SUDAN" header push it
    # higher; an unrecognised code lowers it slightly (could be a misread).
    conf = 0.85
    reason_bits = ["matches Sudanese layout (D + LL + DDDD…)"]
    if known:
        conf += 0.10
        reason_bits.append(f"state code '{state_code}' is a known wilaya")
    else:
        conf -= 0.15
        reason_bits.append(f"state code '{state_code}' not in known list")
    if partial:
        conf -= 0.20
        reason_bits.append("partial serial (a digit looks dropped/doubled by OCR)")
    if header_has_sudan is True:
        conf = min(conf + 0.10, 0.99)
        reason_bits.append("'SUDAN' header detected")
    elif header_has_sudan is False:
        conf -= 0.05
    conf = max(0.0, min(conf, 0.99))

    return PlateInfo(
        text=norm, is_sudan=True, country="Sudan",
        country_confidence=round(conf, 2),
        state_code=state_code, state=state_en, state_ar=state_ar,
        registration_digit=digit, serial=serial,
        reason="; ".join(reason_bits),
    )


def _demo() -> None:
    samples = ["7KH10346", "1NS180", "2G479", "1WN55", "1K91490", "ABC", "999"]
    for s in samples:
        info = interpret(s)
        flag = "🇸🇩" if info.is_sudan else "  "
        print(f"{s:<12} {flag} country={info.country:<8} "
              f"conf={info.country_confidence:<4} "
              f"state={info.state} ({info.state_code or '-'})")


if __name__ == "__main__":
    _demo()
