from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from claude_tap import parse_args
from claude_tap.cli import CLIENT_CONFIGS, run_client
from claude_tap.cli_clients import _detect_codewiz_cc_target


class _DummyProc:
    def __init__(self) -> None:
        self.pid = 12345
        self.returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def test_codewiz_cc_registered_in_client_configs() -> None:
    cfg = CLIENT_CONFIGS["codewiz-cc"]

    assert cfg.cmd == "codewiz-cc"
    assert cfg.label == "CodeWiz CC"
    assert cfg.default_target == "https://codewizllmproxy.devops.xiaohongshu.com/llmratelimit/v3/claude"
    assert cfg.base_url_env == "ANTHROPIC_BASE_URL"
    assert cfg.base_url_suffix == ""
    assert cfg.default_proxy_mode == "reverse"
    assert cfg.inject_settings_env is True
    assert cfg.nesting_env_keys == ("CLAUDECODE", "CLAUDE_CODE_SSE_PORT")


def test_parse_args_codewiz_cc_defaults_to_reverse_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CODEWIZ_CC_ROOT", str(tmp_path))

    args = parse_args(["--tap-client", "codewiz-cc"])

    assert args.client == "codewiz-cc"
    assert args.target == "https://codewizllmproxy.devops.xiaohongshu.com/llmratelimit/v3/claude"
    assert args.proxy_mode == "reverse"


def test_detect_codewiz_cc_target_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com/anthropic")

    assert _detect_codewiz_cc_target() == "https://gateway.example.com/anthropic"


def test_detect_codewiz_cc_target_reads_cached_model_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CODEWIZ_CC_ROOT", str(tmp_path))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "codewiz-model.json").write_text(
        json.dumps(
            {
                "opus": "claude-4.8-opus-google",
                "sonnet": "claude-4.8-opus-google",
                "haiku": "claude-4.8-opus-google",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "codewiz-model-options.json").write_text(
        json.dumps(
            [
                {
                    "value": "claude-4.5-haiku-google",
                    "_profile": {
                        "apiUrl": "https://legacy.example.com/claude",
                        "opus": "claude-4.5-haiku-google",
                        "sonnet": "claude-4.5-haiku-google",
                        "haiku": "claude-4.5-haiku-google",
                    },
                },
                {
                    "value": "claude-4.8-opus-google[1m]",
                    "_profile": {
                        "apiUrl": "https://llmproxy.example.com/llmratelimit/v3/claude",
                        "opus": "claude-4.8-opus-google",
                        "sonnet": "claude-4.8-opus-google",
                        "haiku": "claude-4.8-opus-google",
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    assert _detect_codewiz_cc_target() == "https://llmproxy.example.com/llmratelimit/v3/claude"


def test_detect_codewiz_cc_target_uses_first_model_option_without_cached_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CODEWIZ_CC_ROOT", str(tmp_path))
    (tmp_path / "codewiz-model-options.json").write_text(
        json.dumps(
            [
                {
                    "value": "claude-4.8-opus-google[1m]",
                    "_profile": {
                        "apiUrl": "https://llmproxy.example.com/llmratelimit/v3/claude",
                        "opus": "claude-4.8-opus-google",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    assert _detect_codewiz_cc_target() == "https://llmproxy.example.com/llmratelimit/v3/claude"


def test_detect_codewiz_cc_target_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CODEWIZ_CC_ROOT", str(tmp_path))

    assert _detect_codewiz_cc_target() == "https://codewizllmproxy.devops.xiaohongshu.com/llmratelimit/v3/claude"


@pytest.mark.asyncio
async def test_run_client_codewiz_cc_reverse_sets_base_url_and_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _DummyProc()

    monkeypatch.setenv("CLAUDECODE", "nested")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "43100")
    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda _: "/tmp/codewiz-cc")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = await run_client(43123, ["-p", "Reply OK"], client="codewiz-cc", proxy_mode="reverse")

    assert code == 0
    env = captured["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:43123"
    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_SSE_PORT" not in env

    cmd = captured["cmd"]
    assert cmd[0] == "/tmp/codewiz-cc"
    assert cmd[1] == "--settings"
    settings = json.loads(cmd[2])
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:43123"
    assert cmd[3:] == ("-p", "Reply OK")


@pytest.mark.asyncio
async def test_run_client_codewiz_cc_reverse_does_not_duplicate_existing_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _DummyProc()

    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda _: "/tmp/codewiz-cc")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = await run_client(
        43123,
        ["--settings", '{"env":{}}', "-p", "Reply OK"],
        client="codewiz-cc",
        proxy_mode="reverse",
    )

    assert code == 0
    cmd = captured["cmd"]
    assert sum(1 for arg in cmd if arg == "--settings") == 1
