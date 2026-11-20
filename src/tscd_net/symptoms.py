from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Symptom:
    key: str
    paper_name: str
    chinese_name: str
    description: str
    category: str
    dataset_keys: tuple[str, ...]


# The inventory and English prompts follow Table I of TSCD-Net. "Anhedonia"
# combines the released SWDD interest-loss and pleasure-loss annotations.
PAPER15_SYMPTOMS: tuple[Symptom, ...] = (
    Symptom(
        "sadness",
        "Sadness",
        "悲伤",
        "Feel sad, empty, hopeless most of the day.",
        "emotion",
        ("sadness",),
    ),
    Symptom(
        "anhedonia",
        "Anhedonia",
        "兴趣和愉悦感丧失",
        "Markedly diminished interest or pleasure in all, or almost all, activities.",
        "cognition",
        ("interest_loss", "pleasure_loss"),
    ),
    Symptom(
        "appetite_problem",
        "Appetite problem",
        "食欲问题",
        "Significant decrease or increase in appetite, including nausea and vomiting.",
        "somatization",
        ("appetite_problem",),
    ),
    Symptom(
        "weight_problem",
        "Weight problem",
        "体重问题",
        "Significant weight loss when not dieting, or a change of more than five percent in a month.",
        "behavioral",
        ("weight_problem",),
    ),
    Symptom(
        "insomnia",
        "Insomnia",
        "失眠",
        "Self-described insomnia or a large proportion of posts made between midnight and 6 am.",
        "somatization",
        ("insomnia",),
    ),
    Symptom(
        "hypersomnia",
        "Hypersomnia",
        "嗜睡",
        "Self-described hypersomnia, such as finding it hard to get up during the daytime.",
        "behavioral",
        ("hypersomnia",),
    ),
    Symptom(
        "agitation",
        "Agitation",
        "精神运动性激越",
        "Psychomotor agitation, including aggressive words or persistent feelings of restlessness.",
        "emotion",
        ("agitation",),
    ),
    Symptom(
        "retardation",
        "Retardation",
        "精神运动性迟滞",
        "Psychomotor retardation, as indicated by subjective feelings of being slowed down.",
        "behavioral",
        ("retardation",),
    ),
    Symptom(
        "energy_loss",
        "Energy loss",
        "精力丧失",
        "Fatigue or loss of energy most of the time.",
        "somatization",
        ("energy_loss",),
    ),
    Symptom(
        "self_blame",
        "Self-shame",
        "自我贬低与过度内疚",
        "Feelings of worthlessness or excessive and inappropriate guilt.",
        "cognition",
        ("self_blame",),
    ),
    Symptom(
        "concentration_problem",
        "Concentration problem",
        "注意力与决策问题",
        "Diminished ability to think or concentrate.",
        "cognition",
        ("concentration_problem",),
    ),
    Symptom(
        "suicidal_ideation",
        "Suicidal ideation",
        "自杀意念",
        "Recurrent thoughts of death, suicide attempts, or a specific suicide plan.",
        "cognition",
        ("suicidal_ideation",),
    ),
    Symptom(
        "sympathetic_arousal",
        "Sympathetic arousal",
        "交感神经兴奋",
        "Self-described palpitations, chest pain, or headaches without a clear reason.",
        "somatization",
        ("sympathetic_arousal",),
    ),
    Symptom(
        "panic",
        "Panic",
        "惊恐",
        "Self-described terror without a clear reason.",
        "emotion",
        ("panic",),
    ),
    Symptom(
        "anxious",
        "Anxious",
        "焦虑",
        "Feeling worried or tense all the time without a clear reason.",
        "emotion",
        ("anxious",),
    ),
)


SYMPTOM_SETS: dict[str, tuple[Symptom, ...]] = {
    "paper15": PAPER15_SYMPTOMS,
    "emotion": tuple(
        symptom
        for symptom in PAPER15_SYMPTOMS
        if symptom.category == "emotion"
    ),
    "cognition": tuple(
        symptom
        for symptom in PAPER15_SYMPTOMS
        if symptom.category == "cognition"
    ),
    "somatization": tuple(
        symptom
        for symptom in PAPER15_SYMPTOMS
        if symptom.category == "somatization"
    ),
}


def get_symptoms(name: str) -> tuple[Symptom, ...]:
    try:
        return SYMPTOM_SETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown symptom set: {name}") from exc


def symptom_names(symptoms: tuple[Symptom, ...]) -> list[str]:
    return [symptom.key for symptom in symptoms]


def ground_truth_symptom_vector(
    payload: dict,
    symptoms: tuple[Symptom, ...],
) -> list[int]:
    indicators = payload.get("symptoms")
    if not isinstance(indicators, dict):
        return [0] * len(symptoms)
    return [
        int(any(bool(indicators.get(key, False)) for key in symptom.dataset_keys))
        for symptom in symptoms
    ]
