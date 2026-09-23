"""Workflow configuration accepts only current dev capabilities."""

import pytest
from wenyi_api.schemas import ExportRequest, ProjectCreate, ReviewRunRequest, StartTranslation
from wenyi_api.strategies import PRESET_TEMPLATES, strategy_to_config
from wenyi_core.config import Config


def base():
    return Config.from_dict({"llm": {"preset": "fake"}})


def test_defaults_match_dev_and_preserve_language_direction():
    assert {row["name"] for row in PRESET_TEMPLATES} == {"标准翻译", "快速出稿"}
    cfg = strategy_to_config({"template": "标准翻译"}, base(), source_lang="zh", target_lang="en")
    assert cfg.source_lang == "zh" and cfg.target_lang == "en"
    assert all(
        getattr(cfg.pipeline, key)
        for key in ("book_understanding", "polish", "review", "review_autofix")
    )
    quick = strategy_to_config({"template": "快速出稿"}, base())
    assert not any(
        getattr(quick.pipeline, key)
        for key in ("book_understanding", "polish", "review", "review_autofix")
    )
    custom = strategy_to_config({"steps": {"review": False, "polish": True}}, base())
    assert custom.pipeline.polish and not custom.pipeline.review_autofix


def test_afterword_is_opt_in_and_export_override_is_optional():
    assert not base().pipeline.translator_afterword
    assert not strategy_to_config({"template": "标准翻译"}, base()).pipeline.translator_afterword
    assert strategy_to_config(
        {"steps": {"translator_afterword": True}}, base()
    ).pipeline.translator_afterword
    assert ExportRequest().include_translator_afterword is None
    assert ExportRequest(include_translator_afterword=False).include_translator_afterword is False


@pytest.mark.parametrize("step", ["backtranslate", "consistency_qa", "chapter_review", "autofix"])
def test_retired_steps_fail_instead_of_silently_ignoring(step):
    with pytest.raises(ValueError):
        strategy_to_config({"steps": {step: True}}, base())


@pytest.mark.parametrize(
    "model,body",
    [
        (StartTranslation, {"do_qa": True}),
        (ReviewRunRequest, {"force": True}),
        (ExportRequest, {"format": "invalid"}),
        (ProjectCreate, {"name": "x", "target_lang": "auto"}),
    ],
)
def test_invalid_or_removed_request_fields_are_rejected(model, body):
    with pytest.raises(ValueError):
        model.model_validate(body)
