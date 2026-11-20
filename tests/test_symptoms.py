from tscd_net.symptoms import (
    get_symptoms,
    ground_truth_symptom_vector,
    symptom_names,
)


def test_paper15_inventory_and_order() -> None:
    symptoms = get_symptoms("paper15")
    assert len(symptoms) == 15
    assert symptom_names(symptoms)[:3] == [
        "sadness",
        "anhedonia",
        "appetite_problem",
    ]
    assert symptom_names(symptoms)[-3:] == [
        "sympathetic_arousal",
        "panic",
        "anxious",
    ]


def test_anhedonia_combines_released_interest_and_pleasure_labels() -> None:
    symptoms = get_symptoms("paper15")
    vector = ground_truth_symptom_vector(
        {"symptoms": {"pleasure_loss": True}},
        symptoms,
    )
    assert vector[symptom_names(symptoms).index("anhedonia")] == 1
