"""Tool outputs vs. the ground truth in Section 6 of the report (computed with pandas)."""
from agent.tools import run_tool as R


def test_T1_T3_T4_overview():
    r = R("dataset_overview", {})
    assert (r["rows"], r["columns"]) == (569, 14)
    assert r["total_missing"] == 0 and r["duplicate_rows"] == 0
    assert len(r["column_names"]) == 14


def test_T6_D3_radius():
    s = R("describe_variable", {"variable": "radius"})["overall"]
    assert round(s["min"], 2) == 6.98 and s["max"] == 28.11 and round(s["mean"], 2) == 14.13


def test_T7_D4_area():
    s = R("describe_variable", {"variable": "area"})["overall"]
    assert (s["median"], round(s["mean"], 1), s["min"], s["max"]) == (551.1, 654.9, 143.5, 2501.0)


def test_T9_texture_outliers():
    assert R("find_outliers", {"variable": "texture"})["n_outliers"] == 7


def test_T10_std():
    assert round(R("describe_variable", {"variable": "radius"})["overall"]["std"], 2) == 3.52
    assert round(R("describe_variable", {"variable": "area"})["overall"]["std"], 2) == 351.91


def test_D1_D2_counts():
    c = {x["diagnosis"]: x for x in R("count_cases", {"group_by": ["diagnosis"]})["counts"]}
    assert (c["Malignant"]["count"], c["Benign"]["count"]) == (212, 357)
    assert (c["Malignant"]["percent"], c["Benign"]["percent"]) == (37.3, 62.7)


def test_D5_D6_D7_compare():
    comp = {x["variable"]: x for x in
            R("compare_groups", {"variables": ["radius", "texture", "area"]})["comparison"]}
    assert round(comp["radius"]["Benign"]["mean"], 2) == 12.15
    assert round(comp["radius"]["Malignant"]["mean"], 2) == 17.46
    assert round(comp["texture"]["Benign"]["mean"], 2) == 17.91
    assert round(comp["texture"]["Malignant"]["mean"], 2) == 21.60
    assert comp["area"]["Benign"]["median"] == 458.4 and comp["area"]["Malignant"]["median"] == 932.0


def test_D8_correlation_ranking():
    top = R("correlation", {"with_diagnosis": True})["ranking"][:3]
    assert [(t["variable"], t["r"]) for t in top] == [
        ("concave_points", 0.78), ("perimeter", 0.74), ("radius", 0.73)]


def test_D9_prediction():
    r = R("predict_case", {"measurements": {"radius": 20, "concave_points": 0.15, "texture": 24}})
    assert r["more_similar_to"] == "Malignant" and r["cross_validated_accuracy"] > 0.9
    r = R("predict_case", {"measurements": {"radius": 11, "concave_points": 0.02}})
    assert r["more_similar_to"] == "Benign"
    assert "NOT a diagnosis" in r["notes"][0]


def test_D10_concave_points_medians():
    cp = R("compare_groups", {"variables": ["concave_points"]})["comparison"][0]
    assert round(cp["Malignant"]["median"], 3) == 0.086 and round(cp["Benign"]["median"], 3) == 0.023


def test_guardrail_simulated_note_added():
    r = R("describe_variable", {"variable": "age", "filters": [
        {"column": "diagnosis", "op": "==", "value": "Malignant"}]})
    assert "SIMULATED" in r["notes"][0]
    assert round(r["overall"]["mean"], 1) == 57.8


def test_guardrail_survival_not_in_dataset():
    r = R("describe_variable", {"variable": "five-year survival"})
    assert r["error_type"] == "not_in_dataset"


def test_chart_created(tmp_path, monkeypatch):
    import agent.tools as T
    monkeypatch.setattr(T, "CHART_DIR", tmp_path)
    r = R("make_chart", {"chart_type": "boxplot", "variable": "area"})
    assert r["chart_path"].endswith(".png")
