"""
tests/test_part1.py — Contract tests for the entire Part 1 foundation layer.

Covers:
  - synapse/state.py      (Pydantic output schemas and LangGraph state shapes)
  - synapse/config/loader.py  (YAML cascade: defaults → global → local)
  - synapse/lexer.py      (tokenisation, chaining, macro expansion, aliases)

Run with:
    pytest tests/test_part1.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# ── Imports from the package under test ────────────────────────────────────
from synapse.state import (
    AccessMode,
    ArchitectOutput,
    CoderOutput,
    DebuggerAnalysis,
    FilePatch,
    GlobalState,
    StackType,
    SubTask,
    SubTaskState,
    SummaryOutput,
    TaskStatus,
)
from synapse.config.loader import _deep_merge, _load_yaml, load_all_configs, load_config
from synapse.lexer import Lexer, LexedToken, TokenType


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — state.py
# ═══════════════════════════════════════════════════════════════════════════

class TestEnumerations:
    """
    Enums are the vocabulary the rest of the system uses.
    We test that the values are exactly what the agents.yaml, pipeline.yaml,
    and all graph nodes expect to see — a typo here would be a silent runtime bug.
    """

    def test_stack_type_values(self):
        assert StackType.FAST.value       == "fast"
        assert StackType.BALANCED.value   == "balanced"
        assert StackType.AUTONOMOUS.value == "autonomous"

    def test_access_mode_values(self):
        assert AccessMode.TRUST.value    == "trust"
        assert AccessMode.NO_TRUST.value == "no_trust"

    def test_task_status_lifecycle_order(self):
        # This is not about ordering by index, it is about confirming
        # all four lifecycle states exist with the exact string values
        # the SubTask model and graph conditional edges will use.
        assert TaskStatus.PENDING.value     == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.SUCCESS.value     == "success"
        assert TaskStatus.FAILED.value      == "failed"


class TestFilePatch:
    """
    FilePatch is the atomic unit of code change. Every write operation
    the Coder performs flows through this schema. If it is too permissive,
    the apply-patch logic later receives garbage and corrupts files silently.
    """

    def test_valid_patch_accepted(self):
        patch = FilePatch(
            file_path="src/auth/router.py",
            unified_diff="--- a/src/auth/router.py\n+++ b/src/auth/router.py\n@@ -1 +1 @@\n-old\n+new",
            description="Replace old with new in auth router",
        )
        assert patch.file_path == "src/auth/router.py"
        assert "--- a/" in patch.unified_diff

    def test_missing_required_field_raises(self):
        # In strict Pydantic v2, omitting a required field raises ValidationError.
        # This matters because the LLM must include all three fields or the
        # validation-feedback loop in agents/base.py will trigger.
        with pytest.raises(ValidationError) as exc_info:
            FilePatch(file_path="x.py", unified_diff="--- a/x.py")  # missing description
        # Confirm pydantic tells us *which* field is missing so the error message
        # forwarded to the LLM is actionable, not vague.
        assert "description" in str(exc_info.value)

    def test_strict_mode_rejects_type_coercion(self):
        # strict=True means Pydantic will NOT silently coerce 123 (int) to "123" (str).
        # This is intentional — we want loud failures so the retry prompt is precise.
        with pytest.raises(ValidationError):
            FilePatch(
                file_path=123,          # should be str, not int
                unified_diff="...",
                description="...",
            )


class TestSubTask:
    """
    SubTask is dispatched as an independent branch in the autonomous map-reduce.
    The defaults matter enormously: iteration_count must start at 0 and status
    at PENDING, otherwise the hard-abort logic at iteration > 5 misfires.
    """

    def test_defaults_are_correct(self):
        task = SubTask(
            task_id="task_1_database_layer",
            description="Build the SQLAlchemy models",
            relevant_files=["src/models.py"],
        )
        assert task.status          == TaskStatus.PENDING
        assert task.iteration_count == 0
        assert task.result          is None
        assert task.patch           is None
        assert task.failure_analysis is None

    def test_iteration_count_can_be_incremented(self):
        task = SubTask(
            task_id="task_1",
            description="Build models",
            relevant_files=["models.py"],
            iteration_count=3,
        )
        assert task.iteration_count == 3

    def test_status_transition_to_success(self):
        task = SubTask(
            task_id="task_1",
            description="Build models",
            relevant_files=["models.py"],
            status=TaskStatus.SUCCESS,
        )
        assert task.status == TaskStatus.SUCCESS

    def test_nested_patch_validates(self):
        patch = FilePatch(
            file_path="models.py",
            unified_diff="--- a/models.py\n+++ b/models.py\n@@ -1 +1 @@\n-x\n+y",
            description="Add User model",
        )
        task = SubTask(
            task_id="task_1",
            description="Build models",
            relevant_files=["models.py"],
            patch=patch,
            status=TaskStatus.SUCCESS,
        )
        assert task.patch.file_path == "models.py"


class TestArchitectOutput:
    """
    The Architect's output gates the entire execution of both the balanced
    and autonomous stacks. If it passes validation, the downstream graph
    knows it has a clean task list to work with.
    """

    def test_valid_output_accepted(self):
        output = ArchitectOutput(
            rationale="The task splits cleanly into two independent concerns.",
            tasks=[
                SubTask(
                    task_id="task_1_models",
                    description="Create database models",
                    relevant_files=["src/models.py"],
                ),
                SubTask(
                    task_id="task_2_routes",
                    description="Create API routes",
                    relevant_files=["src/routes.py"],
                ),
            ],
        )
        assert len(output.tasks) == 2
        assert output.tasks[0].task_id == "task_1_models"

    def test_empty_tasks_list_is_invalid(self):
        # An empty task list from the Architect means the planner produced
        # nothing actionable. We treat this as a validation error so the
        # retry loop forces the Architect to produce at least one task.
        with pytest.raises(ValidationError):
            ArchitectOutput(rationale="Nothing to do", tasks=[])

    def test_missing_rationale_raises(self):
        with pytest.raises(ValidationError):
            ArchitectOutput(
                tasks=[SubTask(task_id="t1", description="x", relevant_files=[])]
            )


class TestCoderOutput:
    """CoderOutput enforces that the LLM always returns structured patches."""

    def test_valid_coder_output(self):
        output = CoderOutput(
            patches=[
                FilePatch(
                    file_path="app.py",
                    unified_diff="--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x\n+y",
                    description="Replace x with y",
                )
            ],
            notes="Simple single-line replacement.",
        )
        assert len(output.patches) == 1

    def test_empty_patches_list_invalid(self):
        with pytest.raises(ValidationError):
            CoderOutput(patches=[], notes="Did nothing")


class TestDebuggerAnalysis:
    """
    The Debugger's verdict is a binary gate in the graph. PASS routes forward,
    FAIL routes back to the Coder. Any value other than these two exact strings
    must be rejected so the routing logic never receives an ambiguous value.
    """

    def test_pass_verdict_accepted(self):
        analysis = DebuggerAnalysis(
            verdict="PASS",
            root_cause="",
            fix_instruction="",
        )
        assert analysis.verdict == "PASS"

    def test_fail_verdict_accepted(self):
        analysis = DebuggerAnalysis(
            verdict="FAIL",
            root_cause="NullPointerException on line 42",
            fix_instruction="Add a None check before dereferencing user.profile",
        )
        assert analysis.verdict == "FAIL"

    def test_invalid_verdict_rejected(self):
        # This is the most important test in this class. If "MAYBE" slips
        # through, the conditional edge in the graph has no matching branch
        # and raises a cryptic KeyError at runtime instead of a clean error here.
        with pytest.raises(ValidationError):
            DebuggerAnalysis(
                verdict="MAYBE",
                root_cause="unclear",
                fix_instruction="unclear",
            )


class TestSummaryOutput:
    """SummaryOutput is the last thing the user sees. It must always be valid."""

    def test_valid_summary(self):
        summary = SummaryOutput(
            summary="- Added User model\n- Added /auth/login route",
            files_modified=["src/models.py", "src/routes.py"],
        )
        assert len(summary.files_modified) == 2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — config/loader.py
# ═══════════════════════════════════════════════════════════════════════════

class TestDeepMerge:
    """
    _deep_merge is the function that makes the config cascade work correctly.
    The critical behaviour to test is that nested dicts are merged recursively
    rather than replaced wholesale — without this, any project-local override
    of a single agent model would silently delete all other agents.
    """

    def test_top_level_override(self):
        base     = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result   = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_dict_merges_not_replaces(self):
        # The crucial test. If override only touches one nested key,
        # the other keys in the same nested dict must survive intact.
        base = {
            "agents": {
                "coder_fast":   {"model": "llama-4", "provider": "groq"},
                "summariser":   {"model": "mistral",  "provider": "nvidia_nim"},
            }
        }
        override = {
            "agents": {
                "coder_fast": {"model": "llama-5"}  # only changing the model
            }
        }
        result = _deep_merge(base, override)
        # The overridden field updated
        assert result["agents"]["coder_fast"]["model"]    == "llama-5"
        # The sibling field in the same nested dict survived
        assert result["agents"]["coder_fast"]["provider"] == "groq"
        # The completely untouched agent survived entirely
        assert result["agents"]["summariser"]["model"]    == "mistral"

    def test_does_not_mutate_base(self):
        # Immutability is essential — if the loader mutated the default config
        # on the first project load, every subsequent project would see a
        # corrupted default and the tests would become order-dependent.
        base     = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # base must be untouched

    def test_does_not_mutate_override(self):
        base     = {"a": 1}
        override = {"a": 2, "b": {"c": 3}}
        _deep_merge(base, override)
        assert override["b"]["c"] == 3  # override must be untouched

    def test_list_values_are_replaced_not_merged(self):
        # Lists in YAML configs (like the 'tools' array in agents.yaml)
        # should be fully replaced by the override, not concatenated.
        # Merging lists would lead to duplicate tool registrations.
        base     = {"tools": ["file_read", "file_write"]}
        override = {"tools": ["file_read"]}
        result   = _deep_merge(base, override)
        assert result["tools"] == ["file_read"]

    def test_empty_override_returns_base_copy(self):
        base   = {"a": 1, "b": {"c": 2}}
        result = _deep_merge(base, {})
        assert result == base
        assert result is not base  # must be a copy, not the same object


class TestLoadYaml:
    """
    _load_yaml must silently return an empty dict for missing files.
    This is what makes the cascade safe — a user with no global config
    directory causes no errors, they just get the defaults.
    """

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        result = _load_yaml(tmp_path / "does_not_exist.yaml")
        assert result == {}

    def test_loads_valid_yaml_correctly(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value\nnested:\n  a: 1\n")
        result = _load_yaml(config_file)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        result = _load_yaml(config_file)
        assert result == {}


class TestLoadConfig:
    """
    load_config is the function every part of the system calls to get its
    configuration. The cascade and the ValueError guard are the critical paths.
    """

    def test_raises_for_unknown_config_name(self):
        with pytest.raises(ValueError, match="Unknown config file"):
            load_config("not_a_real_config.yaml")

    def test_loads_defaults_without_overrides(self):
        # The default agents.yaml must at minimum contain the agents the
        # master plan specifies. If this fails, the defaults file is missing
        # or malformed.
        config = load_config("agents.yaml", project_root=None)
        assert "agents" in config
        assert "coder_fast"          in config["agents"]
        assert "summariser"          in config["agents"]
        assert "architect_balanced"  in config["agents"]
        assert "coder_autonomous"    in config["agents"]
        assert "debugger_autonomous" in config["agents"]

    def test_local_override_takes_precedence(self, tmp_path):
        # Create a fake project .synapse directory with a local override
        synapse_dir = tmp_path / ".synapse"
        synapse_dir.mkdir()

        # This local override only changes coder_fast's model.
        # Everything else must survive unchanged from the defaults.
        local_agents = {
            "agents": {
                "coder_fast": {
                    "model": "my-custom-model"
                }
            }
        }
        (synapse_dir / "agents.yaml").write_text(yaml.dump(local_agents))

        config = load_config("agents.yaml", project_root=str(tmp_path))

        # The override took effect
        assert config["agents"]["coder_fast"]["model"] == "my-custom-model"
        # The untouched field in the same agent survived (deep merge, not replace)
        assert "provider" in config["agents"]["coder_fast"]
        # An entirely unrelated agent survived
        assert "summariser" in config["agents"]

    def test_load_all_configs_returns_all_three(self):
        all_configs = load_all_configs(project_root=None)
        assert set(all_configs.keys()) == {"agents", "pipeline", "cli_config"}
        # Each value must be a non-empty dict from the defaults
        assert all_configs["agents"]
        assert all_configs["pipeline"]
        assert all_configs["cli_config"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — lexer.py
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cli_config():
    """
    A minimal cli_config dict that exercises all lexer features without
    depending on the actual YAML files on disk. This makes the lexer tests
    fully self-contained and immune to changes in the default YAML content.
    """
    return {
        "aliases": {
            "/s":  "/stack",
            "/a":  "/access",
            "/rs": "/resume",
        },
        "macros": {
            "@arch": "System Directive: Architect mode. Analyse the following:",
            "@test": "System Directive: Write comprehensive tests for:",
        },
    }


@pytest.fixture
def lexer(cli_config):
    return Lexer(cli_config)


class TestLexerBasicTokenisation:
    """
    The most fundamental behaviour: given a raw string, produce the right
    number of tokens with the right types. These tests form the baseline
    that every other lexer test builds on top of.
    """

    def test_empty_input_returns_empty_list(self, lexer):
        assert lexer.tokenize("") == []

    def test_whitespace_only_returns_empty_list(self, lexer):
        assert lexer.tokenize("   ") == []

    def test_plain_text_becomes_prompt_token(self, lexer):
        tokens = lexer.tokenize("Build a FastAPI authentication router")
        assert len(tokens) == 1
        assert tokens[0].token_type == TokenType.PROMPT
        assert tokens[0].content    == "Build a FastAPI authentication router"

    def test_command_without_argument(self, lexer):
        tokens = lexer.tokenize("/init-docker")
        assert len(tokens)            == 1
        assert tokens[0].token_type   == TokenType.COMMAND
        assert tokens[0].command      == "init-docker"
        assert tokens[0].argument     is None

    def test_command_with_argument(self, lexer):
        tokens = lexer.tokenize("/stack balanced")
        assert len(tokens)           == 1
        assert tokens[0].token_type  == TokenType.COMMAND
        assert tokens[0].command     == "stack"
        assert tokens[0].argument    == "balanced"

    def test_command_names_are_lowercased(self, lexer):
        # Commands should be case-insensitive so /Stack and /stack are equivalent.
        tokens = lexer.tokenize("/Stack Balanced")
        assert tokens[0].command  == "stack"
        assert tokens[0].argument == "Balanced"  # only the command name is lowered


class TestLexerChaining:
    """
    && chaining is what makes Synapse's shell feel like a real shell.
    The tokens must come out in left-to-right order and each must be
    correctly classified independently.
    """

    def test_two_commands_chained(self, lexer):
        tokens = lexer.tokenize("/stack balanced && /access trust")
        assert len(tokens) == 2
        assert tokens[0].command  == "stack"
        assert tokens[0].argument == "balanced"
        assert tokens[1].command  == "access"
        assert tokens[1].argument == "trust"

    def test_command_and_prompt_chained(self, lexer):
        tokens = lexer.tokenize("/stack fast && Write a docstring for main.py")
        assert len(tokens) == 2
        assert tokens[0].token_type == TokenType.COMMAND
        assert tokens[1].token_type == TokenType.PROMPT
        assert tokens[1].content    == "Write a docstring for main.py"

    def test_three_segments_chained(self, lexer):
        tokens = lexer.tokenize("/stack balanced && /init-docker && Write tests")
        assert len(tokens) == 3

    def test_trailing_ampersand_handled_gracefully(self, lexer):
        # A trailing && produces an empty segment after the split.
        # The lexer must skip it rather than creating a ghost token.
        tokens = lexer.tokenize("/stack fast &&")
        assert len(tokens) == 1
        assert tokens[0].command == "stack"

    def test_preserves_order(self, lexer):
        tokens = lexer.tokenize("/init-docker && /stack autonomous && /access trust")
        assert tokens[0].command == "init-docker"
        assert tokens[1].command == "stack"
        assert tokens[2].command == "access"


class TestLexerAliases:
    """
    Aliases are the first transformation applied. The rest of the lexer
    must see the canonical form, not the shorthand.
    """

    def test_simple_alias_expansion(self, lexer):
        tokens = lexer.tokenize("/s balanced")
        assert tokens[0].command  == "stack"
        assert tokens[0].argument == "balanced"

    def test_alias_in_chain(self, lexer):
        tokens = lexer.tokenize("/s fast && /a trust")
        assert tokens[0].command == "stack"
        assert tokens[1].command == "access"

    def test_unknown_command_passes_through(self, lexer):
        # An unrecognised /command that is not an alias should still parse
        # as a COMMAND token rather than raising — the main loop decides
        # what to do with unrecognised commands, not the lexer.
        tokens = lexer.tokenize("/resume")
        assert tokens[0].token_type == TokenType.COMMAND
        assert tokens[0].command    == "resume"


class TestLexerMacros:
    """
    Macro expansion is the lexer's most complex transformation. The @token
    is looked up in the macros table and its expansion is prepended to the
    user's remaining instruction with a blank line separator.
    """

    def test_macro_with_user_text_expands_correctly(self, lexer):
        tokens = lexer.tokenize("@arch Build a microservices gateway")
        assert len(tokens) == 1
        assert tokens[0].token_type == TokenType.MACRO
        # The macro prefix must appear in the content
        assert "Architect mode" in tokens[0].content
        # The user's text must appear after the prefix
        assert "Build a microservices gateway" in tokens[0].content
        # They must be separated by a blank line
        assert "\n\n" in tokens[0].content

    def test_macro_without_user_text(self, lexer):
        tokens = lexer.tokenize("@test")
        assert tokens[0].token_type == TokenType.MACRO
        assert "Write comprehensive tests for" in tokens[0].content

    def test_unknown_macro_passes_through_as_prompt(self, lexer):
        # An unrecognised @token must not raise. It passes through as a PROMPT
        # so the user's message still reaches the agent.
        tokens = lexer.tokenize("@unknownmacro do something")
        assert tokens[0].token_type == TokenType.MACRO
        # The raw segment becomes the content unchanged
        assert "@unknownmacro do something" in tokens[0].content

    def test_macro_in_chain(self, lexer):
        tokens = lexer.tokenize("/stack balanced && @arch Build the auth layer")
        assert len(tokens) == 2
        assert tokens[0].token_type == TokenType.COMMAND
        assert tokens[1].token_type == TokenType.MACRO
        assert "Build the auth layer" in tokens[1].content