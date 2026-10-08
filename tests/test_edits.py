"""Mining spelling fixes from consecutive Wikipedia revisions."""
import io

from araspellx.testsets.edits import _pages, changed_lines, paragraph_pair, spelling_fixes

BEFORE = "== تاريخ ==\nذهب الطلاب الي الجامعه لحضور المحاضرات في الصباح الباكر.\nسطر آخر لم يتغير ابدا هنا."
AFTER = "== تاريخ ==\nذهب الطلاب إلى الجامعة لحضور المحاضرات في الصباح الباكر.\nسطر آخر لم يتغير ابدا هنا."


def test_changed_lines_finds_the_edited_line_only():
    assert changed_lines(BEFORE, AFTER) == [(BEFORE.split("\n")[1], AFTER.split("\n")[1])]
    assert changed_lines(BEFORE, BEFORE) == []


def test_spelling_fixes_accepts_small_word_changes_and_merges():
    assert spelling_fixes("ذهب الي الجامعه", "ذهب إلى الجامعة") == [("الي", "إلى"), ("الجامعه", "الجامعة")]
    assert spelling_fixes("في المعلومات المطلوبة", "في المعلوماتالمطلوبة") == [
        ("المعلومات المطلوبة", "المعلوماتالمطلوبة")]


def test_spelling_fixes_rejects_content_edits():
    assert spelling_fixes("ذهب الطلاب صباحا", "ذهب الطلاب مساء") is None  # a different word
    assert spelling_fixes("ذهب الطلاب", "ذهب الطلاب صباحا") is None      # a word added
    assert spelling_fixes("عام 1990", "عام 1991") is None                # not Arabic letters
    assert spelling_fixes("ذهب الطلاب", "ذهب الطلاب") is None             # nothing fixed
    assert spelling_fixes("لغة فرنسية", "لغة الفرنسية") is None           # the article added: grammar
    assert spelling_fixes("ثم يستخدم", "ثم ويستخدم") is None              # a conjunction added


def test_paragraph_pair_cleans_markup_on_both_sides():
    pair = paragraph_pair("ذهب [[طالب|الطلاب]] الي الجامعه لحضور المحاضرات في الصباح الباكر.",
                          "ذهب [[طالب|الطلاب]] إلى الجامعة لحضور المحاضرات في الصباح الباكر.")
    assert pair["noisy"].startswith("ذهب الطلاب الي") and pair["clean"].startswith("ذهب الطلاب إلى")
    assert pair["fixes"] == [("الي", "إلى"), ("الجامعه", "الجامعة")]


def test_pages_streams_revisions_of_articles_only():
    def revision(rid, text):
        return f"<revision><id>{rid}</id><contributor><id>9</id></contributor><text>{text}</text></revision>"
    xml = ("<mediawiki xmlns='http://www.mediawiki.org/xml/export-0.11/'><siteinfo/>"
           f"<page><title>T</title><ns>0</ns><id>7</id>{revision(1, 'a')}{revision(2, 'b')}</page>"
           f"<page><title>U</title><ns>1</ns><id>8</id>{revision(3, 'c')}</page>"
           f"<page><title>V</title><ns>0</ns><id>12</id>{revision(4, 'd')}</page></mediawiki>")
    seen = [(page_id, [(r["id"], r["text"]) for r in revisions]) for page_id, revisions in _pages(io.BytesIO(xml.encode()))]
    assert seen == [(7, [("1", "a"), ("2", "b")]), (12, [("4", "d")])]
