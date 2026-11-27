"""Data tools for the breast cancer agent.

Every number the agent reports must come from one of these functions. Each tool
works on final_dataset.csv (569 rows x 14 columns) and returns a JSON-friendly
dict. Guardrails that do not depend on the language model live here:

* unknown variables (survival, treatment, ...) raise DataNotAvailable with the
  list of columns that do exist, so the agent can say "not in the dataset";
* any tool call that touches age_simulated or gender_simulated adds a
  "simulated data" note to its output automatically.
"""
from __future__ import annotations

import os
import re
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATASET_PATH", ROOT / "data" / "final_dataset.csv"))
CHART_DIR = Path(os.getenv("CHART_DIR", ROOT / "outputs" / "charts"))

MEASUREMENTS = [
    "radius", "perimeter", "area", "texture", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
]
SIMULATED = ["age_simulated", "gender_simulated"]
NUMERIC = MEASUREMENTS + ["age_simulated"]
CATEGORICAL = ["diagnosis", "gender_simulated"]
ALL_COLUMNS = ["patient_id", "gender_simulated", "age_simulated", *MEASUREMENTS, "diagnosis"]

SIMULATED_NOTE = (
    "Age and gender in this dataset are SIMULATED (randomly generated, independent "
    "of the diagnosis). Results that use them have no medical meaning."
)

# Words users commonly type, mapped to the real column names.
ALIASES = {
    "age": "age_simulated", "gender": "gender_simulated", "sex": "gender_simulated",
    "concave points": "concave_points", "concavepoints": "concave_points",
    "fractal dimension": "fractal_dimension", "fractal": "fractal_dimension",
    "size": "radius", "diagnoses": "diagnosis", "class": "diagnosis",
    "label": "diagnosis", "id": "patient_id",
}

# Things people ask about that the dataset simply does not contain.
NOT_IN_DATASET = [
    "survival", "death", "mortality", "outcome", "treatment", "therapy",
    "chemotherapy", "surgery", "stage", "grade", "recurrence", "relapse",
    "ethnicity", "race", "bmi", "weight", "family history", "genetics", "brca",
    "smoking", "country", "hospital", "date", "year",
]


class DataNotAvailable(ValueError):
    """Raised when a question needs a variable the dataset does not have."""


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def resolve_column(name: str) -> str:
    """Map a user-facing name to a real column or raise DataNotAvailable."""
    if name is None:
        raise DataNotAvailable("No variable given.")
    key = str(name).strip().lower().replace("-", " ").replace("mean ", "")
    key = re.sub(r"\s+", " ", key)
    if key in ALL_COLUMNS:
        return key
    if key.replace(" ", "_") in ALL_COLUMNS:
        return key.replace(" ", "_")
    if key in ALIASES:
        return ALIASES[key]
    for word in NOT_IN_DATASET:
        if word in key:
            raise DataNotAvailable(
                f"'{name}' is not in the dataset. The dataset only contains these "
                f"columns: {', '.join(ALL_COLUMNS)}. It has no survival, treatment, "
                "outcome or clinical history information."
            )
    raise DataNotAvailable(
        f"Unknown variable '{name}'. Available columns: {', '.join(ALL_COLUMNS)}."
    )


def _numeric(name: str) -> str:
    col = resolve_column(name)
    if col not in NUMERIC:
        raise DataNotAvailable(f"'{col}' is not a numeric measurement. Numeric columns: {', '.join(NUMERIC)}.")
    return col


def _r(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return round(float(x), nd)


def _normalise_value(col: str, value):
    if col == "diagnosis":
        v = str(value).strip().lower()
        if v in {"m", "malignant", "cancer", "1"}:
            return "Malignant"
        if v in {"b", "benign", "0"}:
            return "Benign"
        raise DataNotAvailable("diagnosis can only be 'Malignant' or 'Benign'.")
    if col == "gender_simulated":
        v = str(value).strip().lower()
        if v in {"f", "female", "woman", "women"}:
            return "Female"
        if v in {"m", "male", "man", "men"}:
            return "Male"
        raise DataNotAvailable("gender_simulated can only be 'Female' or 'Male'.")
    return value


_OPS = {
    "==": lambda s, v: s == v, "=": lambda s, v: s == v, "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v, ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v, "<=": lambda s, v: s <= v,
    "in": lambda s, v: s.isin(v if isinstance(v, list) else [v]),
}


def apply_filters(df: pd.DataFrame, filters: list[dict] | None, used: set) -> tuple[pd.DataFrame, list[str]]:
    """filters: [{"column": "age_simulated", "op": ">", "value": 60}, ...]"""
    described = []
    for f in filters or []:
        col = resolve_column(f.get("column"))
        op = str(f.get("op", "==")).strip().lower()
        if op not in _OPS:
            raise ValueError(f"Unsupported operator '{op}'. Use one of {list(_OPS)}.")
        value = f.get("value")
        if op == "in":
            value = [_normalise_value(col, v) for v in (value if isinstance(value, list) else [value])]
            if col in NUMERIC:
                value = [float(v) for v in value]
        else:
            value = _normalise_value(col, value)
            if col in NUMERIC:
                value = float(value)
        df = df[_OPS[op](df[col], value)]
        used.add(col)
        described.append(f"{col} {op} {value}")
    return df, described


def _result(payload: dict, used: set) -> dict:
    notes = payload.setdefault("notes", [])
    if used & set(SIMULATED):
        notes.insert(0, SIMULATED_NOTE)
    payload["source"] = "final_dataset.csv"
    return payload


def _stats(s: pd.Series) -> dict:
    return {
        "count": int(s.count()), "mean": _r(s.mean()), "std": _r(s.std()),
        "min": _r(s.min()), "q1": _r(s.quantile(0.25)), "median": _r(s.median()),
        "q3": _r(s.quantile(0.75)), "max": _r(s.max()),
    }


# ---------------------------------------------------------------- tools ----

def dataset_overview() -> dict:
    """Shape, column names, data types, missing values and duplicates."""
    df = load_data()
    return _result({
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: ("integer" if pd.api.types.is_integer_dtype(t) else
                       "float" if pd.api.types.is_float_dtype(t) else "text")
                   for c, t in df.dtypes.items()},
        "missing_values_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_patient_id": int(df.drop(columns="patient_id").duplicated().sum()),
        "notes": ["age_simulated and gender_simulated are synthetic columns."],
    }, set())


def describe_variable(variable: str, group_by: str | None = None,
                      filters: list[dict] | None = None) -> dict:
    """Count, mean, std, min, quartiles, median and max of one numeric column,
    optionally per group (e.g. diagnosis) and after filters."""
    df = load_data()
    used: set = set()
    col = _numeric(variable)
    used.add(col)
    df, fdesc = apply_filters(df, filters, used)
    if df.empty:
        return _result({"variable": col, "filters": fdesc, "rows_matched": 0,
                        "notes": ["No rows match these filters."]}, used)
    out = {"variable": col, "filters": fdesc, "rows_matched": int(len(df))}
    if group_by:
        g = resolve_column(group_by)
        if g not in CATEGORICAL:
            raise DataNotAvailable(f"Can only group by {CATEGORICAL}.")
        used.add(g)
        out["group_by"] = g
        out["by_group"] = {str(k): _stats(v[col]) for k, v in df.groupby(g)}
    else:
        out["overall"] = _stats(df[col])
    return _result(out, used)


def count_cases(group_by: list[str] | str | None = None,
                filters: list[dict] | None = None) -> dict:
    """Number and percentage of records, optionally per group and after filters."""
    df = load_data()
    used: set = set()
    df, fdesc = apply_filters(df, filters, used)
    out = {"filters": fdesc, "total": int(len(df))}
    groups = [group_by] if isinstance(group_by, str) else (group_by or [])
    if groups:
        cols = [resolve_column(g) for g in groups]
        for c in cols:
            if c not in CATEGORICAL:
                raise DataNotAvailable(f"Can only count by {CATEGORICAL}; use filters for numeric ranges.")
        used.update(cols)
        counts = df.groupby(cols).size()
        out["group_by"] = cols
        out["counts"] = [
            {**dict(zip(cols, k if isinstance(k, tuple) else (k,))),
             "count": int(v), "percent": _r(100 * v / len(df), 1)}
            for k, v in counts.items()
        ]
    return _result(out, used)


def compare_groups(variables: list[str] | str, group_by: str = "diagnosis",
                   filters: list[dict] | None = None) -> dict:
    """Side-by-side mean, median, std and IQR of variables for each group, plus
    the difference between groups."""
    df = load_data()
    used: set = set()
    g = resolve_column(group_by)
    if g not in CATEGORICAL:
        raise DataNotAvailable(f"Can only compare groups of {CATEGORICAL}.")
    used.add(g)
    df, fdesc = apply_filters(df, filters, used)
    variables = [variables] if isinstance(variables, str) else variables
    rows = []
    for v in variables:
        col = _numeric(v)
        used.add(col)
        row = {"variable": col}
        for name, sub in df.groupby(g):
            s = sub[col]
            row[str(name)] = {"n": int(s.count()), "mean": _r(s.mean()), "median": _r(s.median()),
                              "std": _r(s.std()), "q1": _r(s.quantile(.25)), "q3": _r(s.quantile(.75))}
        names = sorted(k for k in row if k != "variable")
        if len(names) == 2:
            a, b = names
            row["difference_in_means"] = {f"{b}_minus_{a}": _r(row[b]["mean"] - row[a]["mean"])}
            iqr_overlap = not (row[a]["q3"] < row[b]["q1"] or row[b]["q3"] < row[a]["q1"])
            row["interquartile_ranges_overlap"] = bool(iqr_overlap)
        rows.append(row)
    return _result({"group_by": g, "filters": fdesc, "comparison": rows}, used)


def correlation(variables: list[str] | None = None, with_diagnosis: bool = False,
                method: str = "pearson") -> dict:
    """Correlation matrix of numeric columns, or (with_diagnosis=True) each
    variable's correlation with a malignant diagnosis (Malignant=1, Benign=0),
    ranked from strongest to weakest."""
    df = load_data()
    used: set = set()
    cols = [_numeric(v) for v in variables] if variables else list(MEASUREMENTS)
    used.update(cols)
    if method not in {"pearson", "spearman"}:
        method = "pearson"
    if with_diagnosis:
        y = (df["diagnosis"] == "Malignant").astype(int)
        r = df[cols].corrwith(y, method=method)
        ranking = [{"variable": c, "r": _r(v, 2)} for c, v in
                   sorted(r.items(), key=lambda kv: abs(kv[1]), reverse=True)]
        return _result({"method": method, "target": "diagnosis (Malignant=1, Benign=0)",
                        "ranking": ranking,
                        "notes": ["Correlation shows association, not causation."]}, used)
    m = df[cols].corr(method=method).round(2)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append({"a": a, "b": b, "r": float(m.loc[a, b])})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return _result({"method": method, "matrix": m.to_dict(),
                    "strongest_pairs": pairs[:5], "weakest_pairs": pairs[-3:]}, used)


def find_outliers(variable: str, group_by: str | None = None, k: float = 1.5) -> dict:
    """Outliers by the IQR rule: below Q1 - k*IQR or above Q3 + k*IQR."""
    df = load_data()
    used: set = set()
    col = _numeric(variable)
    used.add(col)

    def _iqr(sub: pd.DataFrame) -> dict:
        s = sub[col]
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (s < lo) | (s > hi)
        flagged = sub.loc[mask, ["patient_id", col, "diagnosis"]].sort_values(col)
        return {"q1": _r(q1), "q3": _r(q3), "iqr": _r(iqr), "lower_fence": _r(lo),
                "upper_fence": _r(hi), "n_outliers": int(mask.sum()),
                "n_low": int((s < lo).sum()), "n_high": int((s > hi).sum()),
                "outliers": flagged.head(30).to_dict(orient="records")}

    out = {"variable": col, "rule": f"IQR rule, k={k}"}
    if group_by:
        g = resolve_column(group_by)
        used.add(g)
        out["by_group"] = {str(n): _iqr(sub) for n, sub in df.groupby(g)}
    else:
        out.update(_iqr(df))
    out["notes"] = ["In this dataset outliers are kept: they are valid measurements of large tumours."]
    return _result(out, used)


def make_chart(chart_type: str, variable: str | None = None, y_variable: str | None = None,
               by_diagnosis: bool = True) -> dict:
    """Draw a chart and save it as PNG. chart_type: histogram, boxplot, scatter,
    bar (counts per diagnosis) or heatmap (correlation matrix)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_data()
    used: set = set()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"Benign": "#2a9d8f", "Malignant": "#e76f51"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ct = chart_type.lower().strip()
        summary: dict = {}
        if ct in {"histogram", "hist", "distribution"}:
            col = _numeric(variable); used.add(col)
            if by_diagnosis:
                for name, sub in df.groupby("diagnosis"):
                    ax.hist(sub[col], bins=30, alpha=0.6, label=name, color=colors[name])
                    summary[name] = {"median": _r(sub[col].median()), "mean": _r(sub[col].mean())}
                ax.legend()
            else:
                ax.hist(df[col], bins=30, color="#457b9d")
            summary["overall"] = _stats(df[col])
            ax.set(xlabel=col, ylabel="Number of cases", title=f"Distribution of {col}")
        elif ct in {"boxplot", "box"}:
            col = _numeric(variable); used.add(col)
            groups = ["Benign", "Malignant"]
            ax.boxplot([df.loc[df.diagnosis == g, col] for g in groups])
            ax.set_xticks([1, 2], groups)  # works on old and new matplotlib
            summary = {g: _stats(df.loc[df.diagnosis == g, col]) for g in groups}
            ax.set(ylabel=col, title=f"{col} by diagnosis")
        elif ct == "scatter":
            x, y = _numeric(variable), _numeric(y_variable or "concave_points")
            used.update({x, y})
            for name, sub in df.groupby("diagnosis"):
                ax.scatter(sub[x], sub[y], s=12, alpha=0.6, label=name, color=colors[name])
            ax.legend(); ax.set(xlabel=x, ylabel=y, title=f"{x} vs {y}")
            summary = {"r": _r(df[x].corr(df[y]), 2)}
        elif ct in {"bar", "count", "counts"}:
            c = df["diagnosis"].value_counts()
            ax.bar(c.index, c.values, color=[colors[i] for i in c.index])
            ax.set(ylabel="Number of cases", title="Cases by diagnosis")
            summary = {k: int(v) for k, v in c.items()}
        elif ct in {"heatmap", "correlation"}:
            m = df[MEASUREMENTS].corr()
            im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(m)), m.columns, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(len(m)), m.columns, fontsize=8)
            fig.colorbar(im, ax=ax); ax.set_title("Correlation matrix")
        else:
            raise ValueError("chart_type must be histogram, boxplot, scatter, bar or heatmap.")
        fig.tight_layout()
        path = CHART_DIR / f"{ct}_{variable or 'all'}_{uuid.uuid4().hex[:6]}.png"
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
    return _result({"chart_type": ct, "chart_path": str(path), "summary": summary}, used)


@lru_cache(maxsize=1)
def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = load_data()
    X, y = df[MEASUREMENTS], (df["diagnosis"] == "Malignant").astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
    pipe.fit(X, y)
    return pipe, float(acc)


def predict_case(measurements: dict) -> dict:
    """Educational estimate of whether a new case resembles malignant or benign
    cases, using logistic regression on the ten mean measurements. Missing
    measurements are filled with the dataset median (reported)."""
    df = load_data()
    pipe, acc = _model()
    row, filled = {}, []
    for k, v in (measurements or {}).items():
        col = _numeric(k)
        if col not in MEASUREMENTS:
            raise DataNotAvailable("Prediction uses only the ten measurements, not age or gender.")
        row[col] = float(v)
    if not row:
        raise ValueError("Give at least one measurement, e.g. {'radius': 18, 'concave_points': 0.1}.")
    # radius, perimeter and area are near-duplicates (r >= 0.99): derive missing
    # ones from a given one so the new case stays physically consistent.
    k_p = float((df.perimeter / df.radius).median())
    k_a = float((df.area / df.radius ** 2).median())
    radius = row.get("radius") or (row["perimeter"] / k_p if "perimeter" in row else
                                   (row["area"] / k_a) ** 0.5 if "area" in row else None)
    estimated = []
    if radius is not None:
        for c, val in (("radius", radius), ("perimeter", radius * k_p), ("area", k_a * radius ** 2)):
            if c not in row:
                row[c] = val; estimated.append(c)
    for c in MEASUREMENTS:
        if c not in row:
            row[c] = float(df[c].median()); filled.append(c)
    X = pd.DataFrame([row])[MEASUREMENTS]
    p = float(pipe.predict_proba(X)[0, 1])
    scaler, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    contrib = (scaler.transform(X)[0] * lr.coef_[0])
    top = sorted(zip(MEASUREMENTS, contrib), key=lambda t: abs(t[1]), reverse=True)[:3]
    return _result({
        "input_used": {k: _r(v) for k, v in row.items()},
        "estimated_from_size": estimated,
        "filled_with_median": filled,
        "probability_malignant": _r(p, 3),
        "more_similar_to": "Malignant" if p >= 0.5 else "Benign",
        "top_contributing_measurements": [
            {"variable": c, "pushes_towards": "Malignant" if w > 0 else "Benign"} for c, w in top],
        "model": "Logistic regression on 10 standardised mean measurements",
        "cross_validated_accuracy": _r(acc, 3),
        "notes": ["Educational estimate from 569 records only. This is NOT a diagnosis; "
                  "only a clinician with a proper examination can diagnose a breast mass."],
    }, set())


TOOL_FUNCTIONS = {
    f.__name__: f for f in [
        dataset_overview, describe_variable, count_cases, compare_groups,
        correlation, find_outliers, make_chart, predict_case,
    ]
}


def run_tool(name: str, args: dict) -> dict:
    """Run a tool by name; errors are returned (not raised) so the agent can
    read them and revise its plan."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool '{name}'.", "available_tools": list(TOOL_FUNCTIONS)}
    try:
        return TOOL_FUNCTIONS[name](**(args or {}))
    except DataNotAvailable as e:
        return {"error": str(e), "error_type": "not_in_dataset"}
    except Exception as e:  # noqa: BLE001 - shown to the agent so it can retry
        return {"error": f"{type(e).__name__}: {e}", "error_type": "tool_error"}
