"""Regression tests for resolve_provider_client's ``aws_sdk`` (Bedrock) branch.

Bug: within resolve_provider_client()'s aws_sdk branch, auxiliary-task calls
configured against a *non-Anthropic* Bedrock model (Amazon Nova, Meta Llama,
Mistral, Cohere, etc.) were being routed to AnthropicAuxiliaryClient instead
of BedrockConverseAuxiliaryClient. Sending Anthropic-shaped kwargs (top-level
``max_tokens``) to those models produces an HTTP 400: "extraneous key
[max_tokens]". Only the aws_sdk/Bedrock branch was affected — the main
tool-calling loop (agent/transports/bedrock.py) always builds proper Converse
kwargs regardless of model family and was never broken.

This module only exercises the auxiliary-task client resolution path, i.e.
resolve_provider_client(provider="bedrock", model=...).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(key, raising=False)


def _patch_bedrock_common(*, is_anthropic_model: bool):
    """Patch the shared bedrock_adapter/anthropic_adapter surface used by
    the aws_sdk branch, keeping AWS credential/region checks deterministic.
    """
    return [
        patch("agent.bedrock_adapter.has_aws_credentials", return_value=True),
        patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1"),
        patch("agent.bedrock_adapter.is_anthropic_bedrock_model", return_value=is_anthropic_model),
    ]


def test_bedrock_non_anthropic_model_routes_to_converse_client():
    """Amazon Nova (non-Claude) Bedrock model must use BedrockConverseAuxiliaryClient,
    not AnthropicAuxiliaryClient — this is the exact bug that was fixed."""
    from agent.auxiliary_client import (
        resolve_provider_client,
        AnthropicAuxiliaryClient,
        BedrockConverseAuxiliaryClient,
    )

    patches = _patch_bedrock_common(is_anthropic_model=False)
    with patches[0], patches[1], patches[2], patch(
        "agent.auxiliary_client.BedrockConverseAuxiliaryClient"
    ) as mock_converse_cls:
        fake_converse_client = MagicMock(name="bedrock_converse_client")
        mock_converse_cls.return_value = fake_converse_client

        client, model = resolve_provider_client(
            "bedrock", model="amazon.nova-lite-v1:0",
        )

    assert client is fake_converse_client, (
        "Non-Anthropic Bedrock model must resolve to BedrockConverseAuxiliaryClient"
    )
    assert not isinstance(client, AnthropicAuxiliaryClient), (
        "Regression: non-Anthropic Bedrock models must never be routed to "
        "AnthropicAuxiliaryClient — this sends top-level max_tokens and "
        "triggers HTTP 400 'extraneous key [max_tokens]' on Nova/Llama/etc."
    )
    mock_converse_cls.assert_called_once_with("us-east-1", "amazon.nova-lite-v1:0")
    assert model == "amazon.nova-lite-v1:0"


def test_bedrock_anthropic_claude_model_routes_to_anthropic_client():
    """Claude-on-Bedrock must still use AnthropicAuxiliaryClient (prompt
    caching / thinking support) — confirms the fix didn't break the Claude path."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    patches = _patch_bedrock_common(is_anthropic_model=True)
    fake_real_client = MagicMock(name="anthropic_bedrock_sdk_client")
    with patches[0], patches[1], patches[2], patch(
        "agent.anthropic_adapter.build_anthropic_bedrock_client",
        return_value=fake_real_client,
    ):
        client, model = resolve_provider_client(
            "bedrock", model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )

    assert isinstance(client, AnthropicAuxiliaryClient), (
        f"Claude Bedrock model must resolve to AnthropicAuxiliaryClient, got {type(client).__name__}"
    )
    assert model == "anthropic.claude-haiku-4-5-20251001-v1:0"


def test_bedrock_no_credentials_returns_none():
    """Sanity: missing AWS credentials must short-circuit to (None, None),
    not silently fall through to Anthropic."""
    from agent.auxiliary_client import resolve_provider_client

    with patch("agent.bedrock_adapter.has_aws_credentials", return_value=False):
        client, model = resolve_provider_client("bedrock", model="amazon.nova-lite-v1:0")

    assert client is None
    assert model is None
