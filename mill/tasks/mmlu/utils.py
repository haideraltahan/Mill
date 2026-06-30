"""MMLU prompt function."""
from mill.api.task import Doc

CHOICES = ["A", "B", "C", "D"]


def mmlu_prompt(row: dict, subject: str | None = None) -> Doc:
    question = row["question"].strip()
    options = row["choices"]
    answer_idx = int(row["answer"])
    subj = subject or row.get("subject", "")

    subject_desc = (
        f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
        if subj else ""
    )
    choices_text = "".join(f"\n{letter}. {opt}" for letter, opt in zip(CHOICES, options))
    query = f"{subject_desc}Question: {question}{choices_text}\nAnswer:"

    return Doc(
        query=query,
        choices=[f" {c}" for c in CHOICES],  # leading space for log-prob scoring
        target_index=answer_idx,
        metadata={"subject": subj},
    )
