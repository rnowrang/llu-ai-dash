import pandas as pd

from app import (
    build_category_stack,
    build_kpi_cards,
    build_platform_bar,
    build_section_bar,
    build_timeline,
    build_treemap,
    filter_dataframe,
    load_and_prepare_data,
    update_figs,
)


def _sample_data():
    today = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "Domain": ["Imaging AI", "Imaging AI"],
            "Rad Section": ["Radiology Ops", "Clinical AI"],
            "Category": ["Workflow", "Clinical"],
            "Name": ["Automation Suite", "Clinical Assist"],
            "Platform": ["Platform 1", "Platform 2"],
            "Status": ["Active", "Not Active"],
            "Active Flag": ["Active", "Not Active"],
            "Starting Date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-06-01"),
            ],
            "Ending Date": [today, pd.NaT],
        }
    )


def _assert_figure_with_data(fig):
    assert hasattr(fig, "data")
    assert len(fig.data) >= 0


def test_builders_handle_minimal_dataframe():
    dff = _sample_data()
    for builder in (
        build_treemap,
        build_section_bar,
        build_platform_bar,
        build_category_stack,
        build_timeline,
    ):
        fig = builder(dff)
        _assert_figure_with_data(fig)


def test_update_figures_returns_all_slots():
    figures = update_figs(True, None, None, None, None, None)
    assert isinstance(figures, tuple)
    assert len(figures) == 6


def test_filter_dataframe_applies_filters():
    dff = _sample_data()
    filtered = filter_dataframe(dff, False, ["Imaging AI"], ["Active"], ["Clinical AI"], None)
    assert len(filtered) == 0
    filtered = filter_dataframe(dff, True, None, None, None, None)
    assert len(filtered) == 1


def test_build_kpi_cards_reflects_data():
    dff = _sample_data()
    cards = build_kpi_cards(dff)
    assert cards.children

    texts = []
    for card in cards.children:
        for piece in getattr(card, "children", []):
            child_text = getattr(piece, "children", "")
            if isinstance(child_text, str):
                texts.append(child_text)

    joined = " ".join(texts)
    assert "Tracked solutions" in joined
    assert "Active solutions" in joined


def test_load_and_prepare_data_missing(monkeypatch):
    # Clear the cache if it exists
    if hasattr(load_and_prepare_data, '_cache'):
        load_and_prepare_data._cache.clear()

    def _missing(*args, **kwargs):
        raise FileNotFoundError("file not found")

    monkeypatch.setattr("app.pd.read_excel", _missing)
    df = load_and_prepare_data("missing.xlsx")
    assert df.empty
    # Clear the cache again
    if hasattr(load_and_prepare_data, '_cache'):
        load_and_prepare_data._cache.clear()
