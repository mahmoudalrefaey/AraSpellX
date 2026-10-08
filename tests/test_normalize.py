from araspellx.text.normalize import normalize

ALEF, LAM, KAF, YEH = chr(0x0627), chr(0x0644), chr(0x0643), chr(0x064A)
KEHEH, FARSI_YEH, TATWEEL, ZWNJ = chr(0x06A9), chr(0x06CC), chr(0x0640), chr(0x200C)
LAM_ALEF_LIGATURE = chr(0xFEFB)
FATHA = chr(0x064E)


def test_plain_arabic_is_unchanged():
    text = "ذهب الولد إلى المدرسة"
    result = normalize(text)
    assert result.text == text
    assert result.edits == []
    assert result.spans == [(i, i + 1) for i in range(len(text))]


def test_lookalikes_tatweel_and_invisible_characters():
    text = f"{KEHEH}تـ{TATWEEL}اب{ZWNJ} عل{FARSI_YEH}"
    result = normalize(text)
    assert result.text == f"{KAF}تاب عل{YEH}"
    replaced = [(e.start, e.end, e.original, e.replacement) for e in result.edits]
    assert replaced == [
        (0, 1, KEHEH, KAF),
        (2, 4, TATWEEL * 2, ""),
        (6, 7, ZWNJ, ""),
        (10, 11, FARSI_YEH, YEH),
    ]


def test_ligature_expands_and_maps_back_to_one_character():
    text = f"م{LAM_ALEF_LIGATURE}ذ"
    result = normalize(text)
    assert result.text == f"م{LAM}{ALEF}ذ"
    assert result.to_original(1, 2) == (1, 2)  # the lam inside the ligature
    assert result.to_original(1, 3) == (1, 2)  # the whole ligature
    assert result.to_original(3, 4) == (2, 3)


def test_insertion_points_map_between_original_characters():
    text = f"ا{TATWEEL}ب"
    result = normalize(text)
    assert result.text == "اب"
    assert result.to_original(1, 1) == (2, 2)  # before ب, after the removed tatweel
    assert result.to_original(2, 2) == (3, 3)  # end of text


def test_protected_spans_are_not_normalized():
    text = f"{KEHEH} ﴿{KEHEH}{TATWEEL}﴾"
    result = normalize(text, protected=[(2, 6)])
    assert result.text == f"{KAF} ﴿{KEHEH}{TATWEEL}﴾"


def test_diacritics_kept_unless_requested():
    text = f"ك{FATHA}تب"
    assert normalize(text).text == text
    assert normalize(text, remove_diacritics=True).text == "كتب"


def test_base_letter_plus_combining_hamza_is_composed():
    hamza_above = chr(0x0654)
    text = f"{ALEF}{hamza_above}ن {ALEF}{hamza_above}"  # how some sources store "أن أ"
    result = normalize(text)
    assert result.text == f"{chr(0x0623)}ن {chr(0x0623)}"
    assert [(e.start, e.end, e.replacement) for e in result.edits] == [(0, 2, chr(0x0623)), (4, 6, chr(0x0623))]
    assert result.to_original(0, 1) == (0, 2)  # the composed letter covers both original characters
    assert result.to_original(1, 2) == (2, 3)
    assert normalize(text, protected=[(0, 2)]).text.startswith(f"{ALEF}{hamza_above}")
