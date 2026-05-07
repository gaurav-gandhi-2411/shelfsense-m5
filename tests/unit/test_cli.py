"""Smoke tests for the shelfsense CLI surface."""
import shelfsense
from typer.testing import CliRunner

from shelfsense.cli import app

runner = CliRunner()


def test_root_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_root_help_lists_top_level_commands():
    result = runner.invoke(app, ["--help"])
    for name in ("data", "features", "train", "ensemble", "submit", "report", "version"):
        assert name in result.output, f"'{name}' missing from top-level --help"


def test_version_prints_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert shelfsense.__version__ in result.output


def test_data_help_lists_subcommands():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "download" in result.output
    assert "validate" in result.output


def test_train_help_lists_subcommands():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    for name in ("tweedie-mh", "store-dept", "per-store", "per-dept"):
        assert name in result.output, f"'{name}' missing from train --help"


def test_features_help_lists_subcommands():
    result = runner.invoke(app, ["features", "--help"])
    assert result.exit_code == 0
    assert "build" in result.output
