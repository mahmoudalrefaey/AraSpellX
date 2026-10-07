import random

from araspellx.noise.typed import add_typos, apply_habits, typed_noise


def habit(text, name):
    return apply_habits(text, random.Random(0), apply_prob=1.0, only=[name])


def test_habits_work_before_punctuation_and_after_prefixes():
    assert habit("ذهبت إلى المدرسة، وأكثر من ذلك للأدب.", "hamza_alef") == "ذهبت الى المدرسة، واكثر من ذلك للادب."
    assert habit("المدرسة، الجامعة.", "ta_marbuta") == "المدرسه، الجامعه."
    assert habit("على المستوى!", "alef_maqsura") == "علي المستوي!"
    assert habit("كفاءة قراءة أداء", "hamza_after_alef") == "كفائة قرائة أداء"
    assert habit("قاموا، يتمكنوا هو", "alef_fariqa") == "قامو، يتمكنو هو"
    assert habit("رأى سأل", "hamza_alef") == "رأى سأل"  # medial hamza untouched


def test_typos_never_touch_digits_latin_or_punctuation():
    rng = random.Random(1)
    text = "في عام 2026، أطلقت شركة OpenData نظاما جديدا (النسخة 3.1)."
    for _ in range(200):
        noisy = add_typos(text, rng, 0.3)
        for keep in ["2026", "OpenData", "3.1", "(", ")", "،"]:
            assert keep in noisy


def test_typed_noise_is_reproducible():
    text = "تعمل الشركة على تطوير نظام جديد لمعالجة البيانات."
    assert typed_noise(text, random.Random(5)) == typed_noise(text, random.Random(5))
