"""The fixed prompt suite, and loading custom ones."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_bench.errors import UsageError
from model_bench.models import PromptSpec
from model_bench.prompts import (
    BUILTIN_PROMPTS,
    SETS,
    builtin,
    deduplicated,
    load_prompts,
    prompt_set,
)


def test_there_are_three_built_in_prompts() -> None:
    assert [prompt.name for prompt in BUILTIN_PROMPTS] == ["tiny", "short", "long"]


def test_every_built_in_prompt_caps_its_output() -> None:
    # Without a cap, a chattier model would look slower for being chattier.
    assert all(prompt.max_tokens >= 1 for prompt in BUILTIN_PROMPTS)


def test_the_built_in_prompts_grow_in_size() -> None:
    caps = [prompt.max_tokens for prompt in BUILTIN_PROMPTS]
    assert caps == sorted(caps)
    assert len(set(caps)) == 3


def test_built_in_prompt_names_are_unique() -> None:
    names = [prompt.name for prompt in BUILTIN_PROMPTS]
    assert len(set(names)) == len(names)


def test_builtin_looks_a_prompt_up_by_name() -> None:
    assert builtin("tiny").max_tokens == 16


def test_builtin_lists_what_exists_for_an_unknown_name() -> None:
    with pytest.raises(UsageError, match="there is: tiny, short, long"):
        builtin("enormous")


def test_the_all_set_is_every_built_in_prompt() -> None:
    assert len(prompt_set("all")) == len(BUILTIN_PROMPTS)


def test_the_quick_set_omits_the_long_prompt() -> None:
    assert [prompt.name for prompt in prompt_set("quick")] == ["tiny", "short"]


def test_every_set_names_only_real_prompts() -> None:
    known = {prompt.name for prompt in BUILTIN_PROMPTS}
    for members in SETS.values():
        assert set(members) <= known


def test_an_unknown_set_lists_the_real_ones() -> None:
    with pytest.raises(UsageError, match="there is: all, long"):
        prompt_set("gigantic")


def test_load_prompts_reads_a_json_list(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps([{"name": "a", "text": "hi", "max_tokens": 4}]), encoding="utf-8"
    )
    prompts = load_prompts(path)
    assert prompts == (PromptSpec("a", "hi", 4),)


def test_load_prompts_accepts_an_object_with_a_prompts_key(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"prompts": [{"name": "a", "text": "hi"}]}), "utf-8")
    assert load_prompts(path)[0].name == "a"


def test_load_prompts_defaults_the_token_cap(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps([{"name": "a", "text": "hi"}]), encoding="utf-8")
    assert load_prompts(path)[0].max_tokens == 128


def test_load_prompts_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="cannot read prompt file"):
        load_prompts(tmp_path / "absent.json")


def test_load_prompts_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(UsageError, match="not valid JSON"):
        load_prompts(path)


def test_load_prompts_rejects_a_bare_string(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text('"hello"', encoding="utf-8")
    with pytest.raises(UsageError, match="must contain a list of prompts"):
        load_prompts(path)


def test_load_prompts_rejects_an_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(UsageError, match="contains no prompts"):
        load_prompts(path)


def test_load_prompts_rejects_a_non_object_entry(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(["hello"]), encoding="utf-8")
    with pytest.raises(UsageError, match="must be a JSON object"):
        load_prompts(path)


def test_load_prompts_names_every_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps([{}]), encoding="utf-8")
    with pytest.raises(UsageError, match="missing name, text"):
        load_prompts(path)


def test_load_prompts_rejects_a_non_integer_cap(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps([{"name": "a", "text": "hi", "max_tokens": "lots"}]), encoding="utf-8"
    )
    with pytest.raises(UsageError, match="max_tokens must be an integer"):
        load_prompts(path)


def test_load_prompts_rejects_a_boolean_cap(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps([{"name": "a", "text": "hi", "max_tokens": True}]), encoding="utf-8"
    )
    with pytest.raises(UsageError, match="max_tokens must be an integer"):
        load_prompts(path)


def test_load_prompts_rejects_duplicate_names(tmp_path: Path) -> None:
    # Two prompts under one name would merge two measurements into one row.
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps([{"name": "a", "text": "x"}, {"name": "a", "text": "y"}]),
        encoding="utf-8",
    )
    with pytest.raises(UsageError, match="duplicate prompt name 'a'"):
        load_prompts(path)


def test_deduplicated_passes_distinct_names_through() -> None:
    specs = [PromptSpec("a", "x"), PromptSpec("b", "y")]
    assert deduplicated(specs, "here") == specs


def test_deduplicated_reports_where_the_duplicate_came_from() -> None:
    specs = [PromptSpec("a", "x"), PromptSpec("a", "y")]
    with pytest.raises(UsageError, match="here: duplicate prompt name"):
        deduplicated(specs, "here")
