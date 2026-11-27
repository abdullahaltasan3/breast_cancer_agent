"""LangChain tool wrappers (schemas the language model sees) for agent/tools.py."""
from __future__ import annotations

from typing import Literal, Optional, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import tools as T

Column = str  # validated inside the tools, so aliases like "age" also work


class Filter(BaseModel):
    column: Column = Field(description="Column to filter on, e.g. diagnosis, age_simulated, radius")
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in"] = "=="
    value: Union[float, str, list[str]] = Field(
        description="Value, e.g. 'Malignant' or 60; a list of strings only with op 'in'")


class NoArgs(BaseModel):
    pass


class DescribeArgs(BaseModel):
    variable: Column = Field(description="Numeric column, e.g. radius, area, age_simulated")
    group_by: Optional[Literal["diagnosis", "gender_simulated"]] = None
    filters: Optional[list[Filter]] = None


class CountArgs(BaseModel):
    group_by: Optional[list[Literal["diagnosis", "gender_simulated"]]] = None
    filters: Optional[list[Filter]] = None


class CompareArgs(BaseModel):
    variables: list[Column] = Field(description="One or more numeric columns to compare")
    group_by: Literal["diagnosis", "gender_simulated"] = "diagnosis"
    filters: Optional[list[Filter]] = None


class CorrArgs(BaseModel):
    variables: Optional[list[Column]] = Field(None, description="Defaults to all ten measurements")
    with_diagnosis: bool = Field(False, description="True ranks variables by correlation with a malignant diagnosis")
    method: Literal["pearson", "spearman"] = "pearson"


class OutlierArgs(BaseModel):
    variable: Column
    group_by: Optional[Literal["diagnosis"]] = None
    k: float = 1.5


class ChartArgs(BaseModel):
    chart_type: Literal["histogram", "boxplot", "scatter", "bar", "heatmap"]
    variable: Optional[Column] = None
    y_variable: Optional[Column] = Field(None, description="Second variable, scatter only")
    by_diagnosis: bool = True


class PredictArgs(BaseModel):
    """Give the measurements of the hypothetical new case; leave unknown ones empty."""
    radius: Optional[float] = None
    perimeter: Optional[float] = None
    area: Optional[float] = None
    texture: Optional[float] = None
    smoothness: Optional[float] = None
    compactness: Optional[float] = None
    concavity: Optional[float] = None
    concave_points: Optional[float] = None
    symmetry: Optional[float] = None
    fractal_dimension: Optional[float] = None


def _wrap(fn, schema):
    def _call(**kwargs):
        if fn.__name__ == "predict_case":  # flat fields -> {"measurements": {...}}
            kwargs = {"measurements": {k: v for k, v in kwargs.items() if v is not None}}
        # pydantic objects -> plain dicts for the pandas layer
        clean = {k: ([f.model_dump() if hasattr(f, "model_dump") else f for f in v]
                     if isinstance(v, list) else v) for k, v in kwargs.items()}
        return T.run_tool(fn.__name__, clean)
    return StructuredTool.from_function(
        func=_call, name=fn.__name__, description=(fn.__doc__ or "").strip(),
        args_schema=schema)


LC_TOOLS = [
    _wrap(T.dataset_overview, NoArgs),
    _wrap(T.describe_variable, DescribeArgs),
    _wrap(T.count_cases, CountArgs),
    _wrap(T.compare_groups, CompareArgs),
    _wrap(T.correlation, CorrArgs),
    _wrap(T.find_outliers, OutlierArgs),
    _wrap(T.make_chart, ChartArgs),
    _wrap(T.predict_case, PredictArgs),
]
LC_TOOLS_BY_NAME = {t.name: t for t in LC_TOOLS}
