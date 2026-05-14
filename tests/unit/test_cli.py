"""Smoke tests for the shelfsense CLI surface."""

import re

from typer.testing import CliRunner

import shelfsense
from shelfsense.cli import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_root_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_root_help_lists_top_level_commands():
    result = runner.invoke(app, ["--help"])
    clean = _strip_ansi(result.output)
    for name in (
        "data",
        "features",
        "train",
        "ensemble",
        "submit",
        "materialize",
        "report",
        "version",
    ):
        assert name in clean, f"'{name}' missing from top-level --help"


def test_version_prints_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert shelfsense.__version__ in result.output


def test_data_help_lists_subcommands():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    clean = _strip_ansi(result.output)
    assert "download" in clean
    assert "validate" in clean


def test_train_help_lists_subcommands():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    clean = _strip_ansi(result.output)
    for name in ("tweedie-mh", "store-dept", "per-store", "per-dept"):
        assert name in clean, f"'{name}' missing from train --help"


def test_features_help_lists_subcommands():
    result = runner.invoke(app, ["features", "--help"])
    assert result.exit_code == 0
    assert "build" in _strip_ansi(result.output)


def test_per_store_train_runs_dag(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("SHELFSENSE_TEST_MODE", "1")
    result_mock = MagicMock()
    result_mock.success = True

    with patch("dagster.materialize", return_value=result_mock):
        result = runner.invoke(app, ["train", "per-store"])
    assert result.exit_code == 0


def test_per_dept_train_runs_dag(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("SHELFSENSE_TEST_MODE", "1")
    result_mock = MagicMock()
    result_mock.success = True

    with patch("dagster.materialize", return_value=result_mock):
        result = runner.invoke(app, ["train", "per-dept"])
    assert result.exit_code == 0


def test_materialize_help_exits_zero():
    result = runner.invoke(app, ["materialize", "--help"])
    assert result.exit_code == 0
    assert "--asset" in _strip_ansi(result.output)


def test_materialize_all_assets_success(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("SHELFSENSE_TEST_MODE", "1")
    result_mock = MagicMock()
    result_mock.success = True

    with patch("dagster.materialize", return_value=result_mock):
        result = runner.invoke(app, ["materialize", "--asset", "*"])
    assert result.exit_code == 0
    assert "Materialization complete" in result.output


def test_materialize_unknown_asset_exits_one():
    result = runner.invoke(app, ["materialize", "--asset", "no_such_asset"])
    assert result.exit_code == 1
    assert "Unknown asset" in result.output


def test_report_stub_exits_nonzero():
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 1
    assert "Stage 6" in result.output or "deferred" in result.output.lower()


# ── CLI helper functions (pure, no Dagster) ────────────────────────────────────


def test_mlflow_uri_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from shelfsense.cli import _mlflow_uri

    assert "localhost" in _mlflow_uri() and "5000" in _mlflow_uri()


def test_mlflow_uri_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://remote:9999")
    from shelfsense.cli import _mlflow_uri

    assert _mlflow_uri() == "http://remote:9999"


def test_is_test_mode_false(monkeypatch):
    monkeypatch.delenv("SHELFSENSE_TEST_MODE", raising=False)
    from shelfsense.cli import _is_test_mode

    assert _is_test_mode() is False


def test_is_test_mode_true(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_TEST_MODE", "1")
    from shelfsense.cli import _is_test_mode

    assert _is_test_mode() is True


def test_raw_ops_structure():
    from shelfsense.cli import _raw_ops

    ops = _raw_ops("/tmp/raw")
    assert set(ops.keys()) == {"raw_sales", "raw_calendar", "raw_prices"}
    assert ops["raw_sales"]["config"]["raw_dir"] == "/tmp/raw"


def test_features_op_production_mode():
    from shelfsense.cli import _features_op

    op = _features_op("/tmp/feats", test_mode=False)
    cfg = op["features"]["config"]
    assert "test_mode" not in cfg
    assert cfg["last_day"] == 1941


def test_features_op_test_mode():
    from shelfsense.cli import _features_op

    op = _features_op("/tmp/feats", test_mode=True)
    cfg = op["features"]["config"]
    assert cfg["test_mode"] is True
    assert "test_n_series" in cfg


def test_full_ops_cfg_has_all_asset_keys():
    from shelfsense.cli import _full_ops_cfg

    ops = _full_ops_cfg(test_mode=False)
    for key in (
        "raw_sales",
        "raw_calendar",
        "raw_prices",
        "features",
        "model_tvp_13",
        "model_tvp_17",
        "predictions_tvp_13",
        "predictions_tvp_17",
        "ensemble",
    ):
        assert key in ops, f"'{key}' missing from _full_ops_cfg output"


def test_dag_run_returns_true_on_success():
    from unittest.mock import MagicMock, patch

    from shelfsense.cli import _dag_run

    result_mock = MagicMock()
    result_mock.success = True

    with patch("dagster.materialize", return_value=result_mock):
        result = _dag_run([], {})

    assert result is True


def test_data_validate_success(monkeypatch):
    """data validate with mocked Dagster succeeds and emits success message."""
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("SHELFSENSE_TEST_MODE", "1")
    result_mock = MagicMock()
    result_mock.success = True

    with patch("dagster.materialize", return_value=result_mock):
        result = runner.invoke(app, ["data", "validate"])
    assert result.exit_code == 0
    assert "Validation passed" in result.output


def test_data_download_success(tmp_path):
    """data download creates directory, runs kaggle, emits success message."""
    from unittest.mock import MagicMock, patch

    from shelfsense.cli import _EXPECTED_RAW_FILES

    # Create the files kaggle would have downloaded
    for fname in _EXPECTED_RAW_FILES:
        (tmp_path / fname).write_text("")

    proc_mock = MagicMock()
    proc_mock.returncode = 0

    with patch("shelfsense.cli.subprocess.run", return_value=proc_mock):
        result = runner.invoke(app, ["data", "download", "--raw-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert str(len(_EXPECTED_RAW_FILES)) in result.output


def test_data_download_kaggle_failure(tmp_path):
    """data download exits 1 when kaggle returns non-zero."""
    from unittest.mock import MagicMock, patch

    proc_mock = MagicMock()
    proc_mock.returncode = 1
    proc_mock.stderr = "auth error"
    proc_mock.stdout = ""

    with patch("shelfsense.cli.subprocess.run", return_value=proc_mock):
        result = runner.invoke(app, ["data", "download", "--raw-dir", str(tmp_path)])

    assert result.exit_code == 1
