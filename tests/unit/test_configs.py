"""Smoke tests: every Hydra config combination must load without errors."""
import os

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../shelfsense/config")
)


def test_default_config_loads():
    """Default config (tvp=1.3 + default features + m5 data + optuna ensemble)."""
    with initialize_config_dir(config_dir=CONFIG_DIR, job_name="test", version_base=None):
        cfg = compose(config_name="config")
    assert cfg.data.last_train_day == 1913
    assert cfg.data.feat_start_day == 1000
    assert cfg.model.tweedie_variance_power == pytest.approx(1.3)
    assert cfg.model.learning_rate == pytest.approx(0.025)
    assert cfg.ensemble.n_trials == 50


def test_override_model_tvp17():
    """Override model group to tvp=1.7."""
    with initialize_config_dir(config_dir=CONFIG_DIR, job_name="test", version_base=None):
        cfg = compose(config_name="config", overrides=["model=tweedie_mh_tvp17"])
    assert cfg.model.tweedie_variance_power == pytest.approx(1.7)
    assert cfg.model.objective == "tweedie"


def test_override_model_rmse_mh():
    """Override model group to RMSE objective."""
    with initialize_config_dir(config_dir=CONFIG_DIR, job_name="test", version_base=None):
        cfg = compose(config_name="config", overrides=["model=rmse_mh"])
    assert cfg.model.objective == "regression"
    assert cfg.model.metric == "rmse"
    assert not hasattr(cfg.model, "tweedie_variance_power")


def test_override_model_store_dept():
    """Override model group to store×dept sliced training."""
    with initialize_config_dir(config_dir=CONFIG_DIR, job_name="test", version_base=None):
        cfg = compose(config_name="config", overrides=["model=store_dept"])
    assert cfg.model.optuna_trials == 10
    assert len(cfg.model.stores) == 10
    assert len(cfg.model.departments) == 7


def test_override_features_ylags():
    """Override features group to annual-lag set."""
    with initialize_config_dir(config_dir=CONFIG_DIR, job_name="test", version_base=None):
        cfg = compose(config_name="config", overrides=["features=ylags"])
    assert 91 in cfg.features.lag_windows
    assert 364 in cfg.features.lag_windows
    assert len(cfg.features.lag_windows) == 7


def test_override_ensemble_equal_weight():
    """Override ensemble group to equal-weight blender."""
    with initialize_config_dir(config_dir=CONFIG_DIR, job_name="test", version_base=None):
        cfg = compose(config_name="config", overrides=["ensemble=equal_weight"])
    assert "EqualWeightBlender" in cfg.ensemble._target_
