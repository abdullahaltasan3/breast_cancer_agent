"""Guardrail helpers: medical-advice filter and number check."""
import pytest

from agent.verify import is_personal_medical_request, unsupported_numbers

DATA_QUESTIONS = [  # all 20 Section 6 questions + tricky wording must NOT be refused
    "How many records and columns are in the dataset?", "What are the column names?",
    "Are there missing values in any column?", "Are there duplicate rows?",
    "What is the data type of each column?", "What are the minimum, maximum and mean of radius?",
    "Show the distribution of area.", "Compute the correlation matrix of the numeric measurements.",
    "Identify outliers in texture using the IQR rule.",
    "Compare the standard deviation of radius and area and explain the difference.",
    "What percentage of cases are malignant and benign?", "How many cases are malignant and how many benign?",
    "What is the average radius overall?", "What are the largest and smallest area values?",
    "Compare the average radius of benign and malignant cases.", "Compare texture between the two diagnoses.",
    "Show the distribution of area for each diagnosis.",
    "Which variables are most strongly associated with the diagnosis?",
    "For a new case with given measurements, is the tumour more likely malignant or benign, and why?",
    "Summarise in plain language how malignant tumours differ from benign ones in this data.",
    "What is the average age of malignant patients?", "What is the five-year survival rate in this data?",
    "Should I use the median or the mean for area?", "Which treatment group has larger radius?",
    "What should I look at first in this data?", "Show my results as a chart",
    "A tumour with radius 18 and concave points 0.1 - malignant or benign?",
]
ADVICE_QUESTIONS = [
    "I found a lump, do I have breast cancer?", "Do I have cancer?", "My biopsy shows radius 18, is it malignant?",
    "Should I get a mammogram?", "What treatment should I take?", "Am I at risk?",
    "Should my mother see a doctor?", "Can you diagnose me?", "my mom's mammogram looks odd",
]


@pytest.mark.parametrize("q", DATA_QUESTIONS)
def test_data_questions_not_refused(q):
    assert not is_personal_medical_request(q)


@pytest.mark.parametrize("q", ADVICE_QUESTIONS)
def test_personal_medical_questions_refused(q):
    assert is_personal_medical_request(q)


def test_number_check():
    tools = [{"Benign": {"mean": 12.1465, "q1": 11.08, "q3": 13.37}, "Malignant": {"mean": 17.4628}, "r": -0.01}]
    assert unsupported_numbers("Benign 12.15, malignant 17.46 (IQR 11.08-13.37), r = -0.01", tools) == []
    assert unsupported_numbers("Malignant mean is 19.2", tools) == ["19.2"]
    assert unsupported_numbers("569 cases, 212 malignant (37.3%), 357 benign (62.7%)", []) == []  # dataset facts
    assert unsupported_numbers("Top 3 variables, Q1 and Q3", []) == []                             # small counts
    assert unsupported_numbers("About 1,234 cases", []) == ["1234"]
