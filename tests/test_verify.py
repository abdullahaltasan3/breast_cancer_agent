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


ARABIC_DATA_QUESTIONS = [
    "كم شخص عنده ورم خبيث؟", "كم عدد الاناث وكم عدد الذكور؟", "كم عدد الحالات اللي عندها ورم خبيث",
    "قارن متوسط حجم الورم بين الحميد والخبيث", "ما هي الأعمدة الموجودة في البيانات؟",
]
PERSONAL_QUESTIONS_MORE = [
    "I have a small tumor, is it dangerous?", "I've got a hard lump", "I found a small painless lump",
    "انا مثلا عندي ورم حجمه صغير", "هل عندي سرطان؟", "هل انا مصاب بالسرطان", "ورمي حجمه 2 سم",
    "لقيت كتلة صغيرة", "شخصني لو سمحت",
]


@pytest.mark.parametrize("q", ARABIC_DATA_QUESTIONS + ["How many patients have a malignant tumour?",
                                                        "Which tumours are larger, benign or malignant?"])
def test_data_questions_not_refused_arabic(q):
    assert not is_personal_medical_request(q)


@pytest.mark.parametrize("q", PERSONAL_QUESTIONS_MORE)
def test_more_personal_questions_refused(q):
    assert is_personal_medical_request(q)


def test_arabic_digits_are_checked():
    assert unsupported_numbers("المتوسط ١٧٫٤٦", [{"m": 17.4628}]) == []
    assert unsupported_numbers("المتوسط ١٩٫٢", [{"m": 17.4628}]) == ["19.2"]


def test_arabic_simulated_disclosure_recognised():
    from agent.verify import mentions_simulated
    assert mentions_simulated("متوسط العمر 57.82 (العمر بيانات محاكاة)")


@pytest.mark.parametrize("q", ["I am worried about the results of this analysis", "hello", "What does radius mean?"])
def test_general_messages_not_treated_as_personal(q):
    assert not is_personal_medical_request(q)


@pytest.mark.parametrize("q", ["i feel sick", "I am worried that I have cancer", "انا تعبان", "احس بألم"])
def test_feelings_about_health_get_supportive_path(q):
    assert is_personal_medical_request(q)


@pytest.mark.parametrize("q", ["why do i have it ?", "Why did I get breast cancer?", "why me", "is it my fault?",
                               "ليش عندي سرطان", "لماذا أصبت بالسرطان"])
def test_why_do_i_have_it_gets_support(q):
    assert is_personal_medical_request(q)


@pytest.mark.parametrize("q", ["how many people have it? and why they have it", "Why do malignant tumours have larger radius?",
                               "Why is concave_points the strongest variable?"])
def test_why_questions_about_data_stay_data(q):
    assert not is_personal_medical_request(q)


@pytest.mark.parametrize("q", ["i have breasts cancer", "I have brest cancer", "i have cancer", "I've got stage 2 breast cancer",
                               "I was diagnosed with breast cancer"])
def test_typos_and_variants_of_i_have_cancer(q):
    assert is_personal_medical_request(q)


@pytest.mark.parametrize("q", ["I have a question about cancer data", "how many ppl have it ?", "How many cancer cases are there?"])
def test_cancer_word_alone_is_not_personal(q):
    assert not is_personal_medical_request(q)
