import pytest

from matplotlib.font_manager import FontProperties


keys = [
    "get_family",
    "get_style",
    "get_variant",
    "get_weight",
    "get_size",
]


def test_fontconfig_pattern():
    f1 = FontProperties()
    s = str(f1)
    f2 = FontProperties(s)
    for key in keys:
        assert getattr(f1, key)() == getattr(f2, key)(), "defaults " + key

    f1 = FontProperties(family="serif", size=20, style="italic")
    s = str(f1)
    f2 = FontProperties(s)
    for key in keys:
        assert getattr(f1, key)() == getattr(f2, key)(), "basic " + key

    f1 = FontProperties(
        family="sans-serif",
        size=24,
        weight="bold",
        style="oblique",
        variant="small-caps",
        stretch="expanded",
    )
    s = str(f1)
    f2 = FontProperties(s)
    for key in keys:
        assert getattr(f1, key)() == getattr(f2, key)(), "full " + key


def test_fontconfig_str():
    s = (
        "sans\\-serif:style=normal:variant=normal:weight=normal"
        ":stretch=normal:size=12.0"
    )
    font = FontProperties(s)
    right = FontProperties(size=12.0)
    for key in keys:
        assert getattr(font, key)() == getattr(right, key)(), "defaults " + key

    s = "serif-24:style=oblique:variant=small-caps:weight=bold:stretch=expanded"
    font = FontProperties(s)
    right = FontProperties(
        family="serif",
        size=24,
        weight="bold",
        style="oblique",
        variant="small-caps",
        stretch="expanded",
    )
    for key in keys:
        assert getattr(font, key)() == getattr(right, key)(), "full " + key


def test_fontconfig_unknown_constant():
    with pytest.raises(ValueError, match="ParseException"):
        FontProperties(":unknown")
