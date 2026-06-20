"""Smoke tests for Mill core components (no GPU, no network)."""
import pytest


def test_output_type_enum():
    from mill.api.instance import OutputType
    assert OutputType.GENERATIVE == "generate_until"
    assert OutputType.LOGPROBS == "loglikelihood"
    assert OutputType.PERPLEXITY == "loglikelihood_rolling"


def test_doc_is_multimodal():
    from mill.api.task import Doc
    text_doc = Doc(query="hello")
    assert not text_doc.is_multimodal
    mm_doc = Doc(query="describe this", visuals=["image.png"])
    assert mm_doc.is_multimodal


def test_metric_registry():
    from mill.api.metrics import get_metric, list_metrics
    metrics = list_metrics()
    assert "exact_match" in metrics
    assert "acc" in metrics
    m = get_metric("exact_match")
    assert m.higher_is_better


def test_metric_aggregate_bootstrap():
    from mill.api.metrics import get_metric
    m = get_metric("acc")
    score, stderr = m.aggregate([1.0, 0.0, 1.0, 0.0, 1.0])
    assert abs(score - 0.6) < 1e-6
    assert stderr is not None and stderr >= 0


def test_task_config_fields():
    from mill.api.instance import OutputType
    from mill.api.task import Doc, MillTaskConfig
    config = MillTaskConfig(
        name="test_task",
        hf_repo="dummy/repo",
        output_type=OutputType.GENERATIVE,
    )
    assert config.name == "test_task"
    assert config.n_shots == 0


def test_chat_messages_from_text():
    from mill.api.protocol import ChatMessages
    msgs = ChatMessages.from_text("What is 2+2?")
    images, videos, audios = msgs.extract_media()
    assert images == [] and videos == [] and audios == []
    hf = msgs.to_hf_messages()
    assert hf[0]["role"] == "user"
    assert hf[0]["content"][0]["type"] == "text"


def test_chat_messages_from_text_and_images():
    from unittest.mock import MagicMock
    from mill.api.protocol import ChatMessages
    fake_image = MagicMock()
    msgs = ChatMessages.from_text_and_images("Describe this", [fake_image])
    images, videos, audios = msgs.extract_media()
    assert len(images) == 1
    assert images[0] is fake_image


def test_output_handler_is_completed_empty(tmp_path):
    from mill.output import OutputHandler
    handler = OutputHandler(output_dir=tmp_path)
    assert not handler.is_completed("llama3", "gsm8k", 0)


def test_output_handler_flush_and_aggregate(tmp_path):
    from mill.output import OutputHandler
    handler = OutputHandler(output_dir=tmp_path)
    handler.add_sample(model="m", task="t", n_shot=0, doc_id=0, split="test",
                       prediction="42", gold="42", acc=1.0)
    handler.add_sample(model="m", task="t", n_shot=0, doc_id=1, split="test",
                       prediction="5", gold="10", acc=0.0)
    handler.flush("m", "t", 0)
    agg = handler.aggregate("m", "t", 0, ["acc"])
    assert abs(agg["acc"] - 0.5) < 1e-6
    assert handler.is_completed("m", "t", 0)


def test_registry_model():
    from mill.api.registry import get_model_class, list_models
    import mill.models  # register models
    assert "hf" in list_models()
    assert "vllm" in list_models()


def test_gsm8k_prompt():
    pytest.importorskip("mill.tasks.gsm8k.utils", reason="gsm8k task not installed")
    from mill.tasks.gsm8k.utils import gsm8k_prompt, gsm8k_exact_match
    row = {
        "question": "There are 3 apples. If 1 is eaten, how many remain?",
        "answer": "2 apples remain.\n#### 2",
    }
    doc = gsm8k_prompt(row)
    assert "Question:" in doc.query
    assert doc.target_index == "2"
    assert gsm8k_exact_match(doc, "The answer is #### 2") == 1.0
    assert gsm8k_exact_match(doc, "#### 3") == 0.0


def test_mmlu_pro_prompt_and_metric():
    from mill.tasks.mmlu_pro.utils import (
        extract_answer_letter,
        mmlu_pro_acc,
        mmlu_pro_prompt,
    )
    row = {
        "question": "What is 2+2?",
        "options": ["3", "4", "5", "6"],
        "answer": "B",
        "answer_index": 1,
        "category": "math",
    }
    doc = mmlu_pro_prompt(row)
    assert doc.target_index == "B"
    assert doc.choices == ["A", "B", "C", "D"]
    assert "B. 4" in doc.query and doc.query.rstrip().endswith("Answer:")

    assert extract_answer_letter("Therefore, the answer is (D).") == "D"
    assert extract_answer_letter("blah\nAnswer: C") == "C"
    assert extract_answer_letter("no letters here") is None

    assert mmlu_pro_acc(doc, "reasoning ... Answer: B") == 1.0
    assert mmlu_pro_acc(doc, "Answer: C") == 0.0


def test_mmlu_pro_registered():
    import mill.tasks  # noqa: F401  triggers auto-discovery
    from mill.api.registry import list_benchmarks, list_tasks
    assert "mmlu_pro" in list_tasks()
    assert "mmlu_pro" in list_benchmarks()


def test_filter_pending_all_done(tmp_path):
    from mill.output import OutputHandler
    handler = OutputHandler(output_dir=tmp_path)
    handler.add_sample(model="m", task="gsm8k", n_shot=0, doc_id=0, split="test",
                       prediction="1", gold="1", acc=1.0)
    handler.flush("m", "gsm8k", 0)
    handler.aggregate("m", "gsm8k", 0, ["acc"])
    pending = handler.filter_pending("m", ["gsm8k", "mmlu"], n_shot=0)
    assert "gsm8k" not in pending
    assert "mmlu" in pending
