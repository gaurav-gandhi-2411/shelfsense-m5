"""CLI → Dagster integration tests.

Each test invokes a shelfsense CLI command through typer's CliRunner in
SHELFSENSE_TEST_MODE=1, asserts exit code 0 and key output substrings,
and checks that expected on-disk artifacts appeared.

Gated by RUN_CLI_INTEGRATION=1.  Requires:
- Raw CSVs at data/raw/m5-forecasting-accuracy/ (not rebuilt here)
- Feature parquets at data/processed/features_test/ (built by test_cli_features_build)
- MLflow reachable at http://localhost:5000 (soft requirement — assets skip on failure)

Runtime budget: < 5 min when model pkls are already cached from prior test runs.
Run:
    RUN_CLI_INTEGRATION=1 uv run --no-dev pytest tests/integration/test_cli_dagster.py -v
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from shelfsense.cli import app

runner = CliRunner()

_cli_test = pytest.mark.skipif(
    os.environ.get("RUN_CLI_INTEGRATION") != "1",
    reason=(
        "CLI Dagster integration tests disabled — set RUN_CLI_INTEGRATION=1. "
        "Requires raw CSVs + ~5 min (models cached)."
    ),
)

_ENV = {**os.environ, "SHELFSENSE_TEST_MODE": "1"}


# ── data ──────────────────────────────────────────────────────────────────────


@_cli_test
def test_cli_data_validate():
    result = runner.invoke(app, ["data", "validate"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "Validation passed" in result.output


# ── features ──────────────────────────────────────────────────────────────────


@_cli_test
def test_cli_features_build():
    result = runner.invoke(app, ["features", "build"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "Features written" in result.output
    assert os.path.isdir("data/processed/features_test")


# ── train ─────────────────────────────────────────────────────────────────────


@_cli_test
def test_cli_train_tvp_13():
    result = runner.invoke(app, ["train", "tweedie-mh", "--tvp", "1.3"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "model_tvp_13" in result.output
    assert os.path.isdir("data/models/test_tvp_1p3")


@_cli_test
def test_cli_train_tvp_17():
    result = runner.invoke(app, ["train", "tweedie-mh", "--tvp", "1.7"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "model_tvp_17" in result.output
    assert os.path.isdir("data/models/test_tvp_1p7")


@_cli_test
def test_cli_train_store_dept():
    result = runner.invoke(app, ["train", "store-dept"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "model_store_dept" in result.output
    assert os.path.isdir("data/models/test_store_dept")


@_cli_test
def test_cli_train_invalid_tvp():
    result = runner.invoke(app, ["train", "tweedie-mh", "--tvp", "2.0"], env=_ENV)
    assert result.exit_code == 1
    assert "1.3 or 1.7" in result.output


# ── stubs ─────────────────────────────────────────────────────────────────────


def test_cli_per_store_runs_dag(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("SHELFSENSE_TEST_MODE", "1")
    result_mock = MagicMock()
    result_mock.success = True

    with patch("dagster.materialize", return_value=result_mock):
        result = runner.invoke(app, ["train", "per-store"])
    assert result.exit_code == 0


def test_cli_per_dept_runs_dag(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("SHELFSENSE_TEST_MODE", "1")
    result_mock = MagicMock()
    result_mock.success = True

    with patch("dagster.materialize", return_value=result_mock):
        result = runner.invoke(app, ["train", "per-dept"])
    assert result.exit_code == 0


def test_cli_report_stub():
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 1
    assert "Stage 6" in result.output or "deferred" in result.output.lower()


# ── ensemble + submit (heavier, same gate) ────────────────────────────────────


@_cli_test
def test_cli_ensemble():
    result = runner.invoke(app, ["ensemble"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "Ensemble materialized" in result.output
    assert os.path.isdir("data/predictions/test_ensemble")


@_cli_test
def test_cli_submit_no_kaggle():
    result = runner.invoke(app, ["submit", "--no-kaggle"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "Submission written" in result.output
    assert os.path.isdir("submissions/test")
