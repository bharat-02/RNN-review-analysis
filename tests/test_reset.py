"""Streamlit Reset-button test using scripted UI runs. Run: python tests/test_reset.py

1. Enter a review -> 2. Analyze -> prediction appears ->
3. Reset -> text area empty AND prediction/results gone.
Exit code 0 = PASS, 1 = FAIL.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest


def main():
    at = AppTest.from_file("streamlit_app.py", default_timeout=180).run()
    assert not at.exception, f"startup errors: {at.exception}"
    tab = at.tabs[0]
    assert tab.label == "Review Analysis", tab.label
    assert tab.text_area[0].value == "", "text area not empty on launch"

    # 1-2. enter a review and analyze
    tab.text_area[0].set_value(
        "This movie was fantastic! The acting was great."
    ).run()
    assert tab.text_area[0].value.startswith("This movie"), "typing failed"
    analyze = next(b for b in tab.button if b.label == "Analyze Review")
    analyze.click().run()
    tab = at.tabs[0]
    assert not at.exception, f"errors after Analyze: {at.exception}"
    shown = [s.value for s in tab.subheader] + [m.value for m in tab.markdown]
    assert any("Prediction" in s for s in shown), "prediction missing"
    import re

    scores = [
        float(m.group(1))
        for m in (re.search(r"Raw score: ([0-9.]+)", s) for s in shown)
        if m
    ]
    assert len(scores) == 1 and 0.0 <= scores[0] <= 1.0, f"bad score: {shown}"
    print("analyze shows prediction: PASS")

    # 3. reset and verify everything is cleared
    reset = next(b for b in tab.button if b.label == "Reset")
    reset.click().run()
    tab = at.tabs[0]
    assert not at.exception, f"errors after Reset: {at.exception}"
    assert tab.text_area[0].value == "", (
        f"text area not cleared: {tab.text_area[0].value!r}"
    )
    leftovers = [s.value for s in tab.subheader] + [
        m.value for m in tab.markdown if "Prediction" in m.value or "Confidence" in m.value
    ]
    assert not leftovers, f"results not cleared: {leftovers}"
    print("reset clears text + results: PASS")

    # 4. summary + evaluation tabs render real data
    for i, want in ((1, "Model Summary"), (2, "Model Evaluation")):
        assert at.tabs[i].label == want, at.tabs[i].label
    summary_vals = [m.value for m in at.tabs[1].metric]
    assert "SimpleRNN" in summary_vals, (
        f"summary missing model info: {summary_vals}"
    )
    eval_vals = [m.value for m in at.tabs[2].metric]
    assert "0.8282" in eval_vals and "0.8298" in eval_vals, (
        f"evaluation metrics missing: {eval_vals}"
    )
    print("summary + evaluation tabs show real data: PASS")
    print("Reset-button test: PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"Reset-button test: FAIL ({exc})")
        sys.exit(1)
