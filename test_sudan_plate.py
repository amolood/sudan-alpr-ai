#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Sudan plate interpreter (sudan_plate.py).

These lock in every behaviour we verified by hand while building the module:
country detection, plate-class recognition (Latin + Arabic markers), state
decoding, the tricky collision cases (1CDR500 must NOT be diplomat), Latin
letter folding (TRANŞIT), OCR-tolerance, and the honesty rules around
unconfirmed state codes.

Run:  ./venv/bin/python -m pytest -q
"""

import pytest

from sudan_plate import (
    interpret,
    STATE_CODES,
    UNMAPPED_ARABIC_LETTERS,
    PLATE_CLASSES,
)


# --------------------------------------------------------------------------
# Private plates: country + state
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,state_code,state_en", [
    ("7KH10346", "KH", "Khartoum"),
    ("1NS180", "NS", "River Nile"),
    ("2G479", "G", "Gezira"),
    ("1WN9212", "WN", "White Nile"),
    ("1RS740", "RS", "Red Sea"),
])
def test_private_plates_decode_state(text, state_code, state_en):
    info = interpret(text)
    assert info.is_sudan
    assert info.country == "Sudan"
    assert info.plate_class == "private"
    assert info.state_code == state_code
    assert info.state == state_en


def test_two_digit_registration_is_supported():
    # "13KH" — a legitimate two-digit registration number, not an error.
    info = interpret("13KH00000")
    assert info.is_sudan
    assert info.plate_class == "private"
    assert info.state_code == "KH"


def test_non_sudanese_text_is_rejected():
    for junk in ("ABC", "999", "HELLO", ""):
        info = interpret(junk)
        assert not info.is_sudan
        assert info.country == "Unknown"


# --------------------------------------------------------------------------
# Special classes via Latin markers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,cls", [
    ("GOV00000", "government"),
    ("ARMY00000", "army"),
    ("POLICE00000", "police"),
    ("UN00", "un"),
    ("CD12", "diplomat"),
    ("HC34", "consular"),
    ("NGO0000", "ngo"),
    ("IO0000", "int_org"),
    ("TRANSITNS00", "transit"),
])
def test_latin_class_markers(text, cls):
    info = interpret(text)
    assert info.is_sudan
    assert info.plate_class == cls


# --------------------------------------------------------------------------
# Special classes via Arabic markers (these plates carry no Latin code)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,cls", [
    ("الهلال الأحمر 1234", "red_crescent"),
    ("ليموزين KH 0000", "limousine"),
    ("استثمار KH9 0000", "investment"),
    ("مؤقتة سريع KH", "temporary_express"),
    ("مؤقتة داخلي KH", "temporary_domestic"),
    ("مؤقتة KH 0000", "temporary"),
])
def test_arabic_class_markers(text, cls):
    info = interpret(text)
    assert info.is_sudan
    assert info.plate_class == cls


def test_temporary_express_wins_over_plain_temporary():
    # Ordering guard: "مؤقتة سريع" must not be swallowed by "مؤقتة".
    assert interpret("مؤقتة سريع KH").plate_class == "temporary_express"
    assert interpret("مؤقتة داخلي KH").plate_class == "temporary_domestic"


def test_investment_plate_still_decodes_its_state():
    info = interpret("استثمار KH9 0000")
    assert info.plate_class == "investment"
    assert info.state_code == "KH"  # KH9 -> Khartoum


# --------------------------------------------------------------------------
# Collision guards: short markers must not fire inside a private serial
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["1CDR500", "2UN30", "1HC90", "3IO45"])
def test_short_markers_do_not_hijack_private_plates(text):
    info = interpret(text)
    # These look like private plates (digit + letters + digits); the embedded
    # CD/UN/HC/IO must NOT classify them as a special class.
    assert info.plate_class == "private"


# --------------------------------------------------------------------------
# Latin letter folding (the board prints "TRANŞIT" with an S-cedilla)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "TRANŞIT NS",
    "TRANSIT NS",
    "TRANSıT NS",
    "tranşit ns",
])
def test_decorated_latin_folds_to_transit(text):
    assert interpret(text).plate_class == "transit"


# --------------------------------------------------------------------------
# OCR tolerance on private plates
# --------------------------------------------------------------------------

def test_dropped_registration_digit_still_sudanese():
    # OCR lost the leading digit but kept state + serial.
    info = interpret("KH5404")
    assert info.is_sudan
    assert info.state_code == "KH"


def test_doubled_registration_digit_is_handled():
    info = interpret("10KH6009")
    assert info.is_sudan
    assert info.state_code == "KH"


# --------------------------------------------------------------------------
# Honesty rules around state codes
# --------------------------------------------------------------------------

def test_only_kh_is_confirmed():
    assert STATE_CODES["KH"][3] == "confirmed"
    for code, (_en, _ar, _letter, evidence) in STATE_CODES.items():
        if code != "KH":
            assert evidence == "observed"


def test_guessed_codes_were_removed():
    # Codes we couldn't support must not silently claim a state.
    for guessed in ("SK", "ND", "KS", "GD", "SN", "BN", "SD", "WD", "ED", "CDR"):
        assert guessed not in STATE_CODES


def test_unmapped_arabic_letters_kept_for_reference():
    # Their Arabic letters survive for documentation, but with no Latin code.
    assert "ج ك" in UNMAPPED_ARABIC_LETTERS  # South Kordofan
    assert len(UNMAPPED_ARABIC_LETTERS) >= 5


def test_observed_code_is_flagged_in_reason():
    info = interpret("2G479")  # G is observed, not confirmed
    assert "not officially confirmed" in info.reason


# --------------------------------------------------------------------------
# Confidence sanity
# --------------------------------------------------------------------------

def test_confidence_is_bounded():
    for text in ("7KH10346", "GOV00000", "ABC", "استثمار KH9"):
        c = interpret(text).country_confidence
        assert 0.0 <= c <= 0.99


def test_colour_corroboration_raises_confidence():
    without = interpret("ARMY00000").country_confidence
    with_red = interpret("ARMY00000", color="red").country_confidence
    assert with_red >= without


# --------------------------------------------------------------------------
# Structural sanity of the class table
# --------------------------------------------------------------------------

def test_every_class_has_at_least_one_marker():
    for pc in PLATE_CLASSES:
        assert pc.markers or pc.markers_ar, f"{pc.key} has no marker at all"


def test_class_keys_are_unique():
    keys = [pc.key for pc in PLATE_CLASSES]
    assert len(keys) == len(set(keys))
