"""Smoke tests for the Dagster asset graph structure.

Verifies asset count, key presence, dependency edges, asset check
registration, and config schema presence without materializing anything.
"""

from __future__ import annotations

from dagster import AssetKey, Definitions


def _get_assets_def(defs: Definitions, key: AssetKey):
    """Return the AssetsDefinition for *key*, or None if not found."""
    return next(
        (a for a in defs.assets if hasattr(a, "keys") and key in a.keys),
        None,
    )


# ── Basic structure ───────────────────────────────────────────────────────────


def test_defs_is_dagster_definitions():
    from shelfsense.orchestration.assets import defs

    assert isinstance(defs, Definitions)


def test_asset_count():
    # 3 raw loaders + 19 computed = 22 total
    from shelfsense.orchestration.assets import defs

    assert len(defs.assets) == 22


def test_required_asset_keys_present():
    from shelfsense.orchestration.assets import defs

    required = {
        "raw_sales",
        "raw_calendar",
        "raw_prices",
        "raw_validated",
        "features",
        "features_validated",
        "model_tvp_13",
        "model_tvp_17",
        "model_rmse_mh",
        "model_store_dept",
        "model_ylags",
        "model_per_store",
        "model_per_dept",
        "predictions_tvp_13",
        "predictions_tvp_17",
        "predictions_rmse_mh",
        "predictions_store_dept",
        "predictions_ylags",
        "predictions_per_store",
        "predictions_per_dept",
        "ensemble",
        "submission",
    }
    found: set[str] = set()
    for a in defs.assets:
        if hasattr(a, "keys"):
            found.update(k.path[-1] for k in a.keys)
        elif hasattr(a, "key"):
            found.add(a.key.path[-1])
    assert required == found


# ── Dependency edges ──────────────────────────────────────────────────────────


def test_submission_depends_on_ensemble():
    from shelfsense.orchestration.assets import defs

    sub_def = _get_assets_def(defs, AssetKey("submission"))
    assert sub_def is not None
    assert AssetKey("ensemble") in sub_def.dependency_keys


def test_ensemble_depends_on_all_predictions():
    from shelfsense.orchestration.assets import defs

    ens_def = _get_assets_def(defs, AssetKey("ensemble"))
    assert ens_def is not None
    expected = {
        AssetKey("predictions_tvp_13"),
        AssetKey("predictions_tvp_17"),
        AssetKey("predictions_rmse_mh"),
        AssetKey("predictions_store_dept"),
        AssetKey("predictions_ylags"),
        AssetKey("predictions_per_store"),
        AssetKey("predictions_per_dept"),
    }
    assert expected <= ens_def.dependency_keys


def test_model_assets_depend_on_features_validated():
    from shelfsense.orchestration.assets import defs

    for key in (
        "model_tvp_13",
        "model_tvp_17",
        "model_rmse_mh",
        "model_store_dept",
        "model_ylags",
        "model_per_store",
        "model_per_dept",
    ):
        m_def = _get_assets_def(defs, AssetKey(key))
        assert m_def is not None, f"{key} not in defs"
        assert AssetKey("features_validated") in m_def.dependency_keys, (
            f"{key} should depend on features_validated"
        )


def test_raw_validated_depends_on_all_source_assets():
    from shelfsense.orchestration.assets import defs

    rv_def = _get_assets_def(defs, AssetKey("raw_validated"))
    assert rv_def is not None
    for src in ("raw_sales", "raw_calendar", "raw_prices"):
        assert AssetKey(src) in rv_def.dependency_keys, f"raw_validated should depend on {src}"


def test_predictions_depend_on_model_and_features():
    # Each predictions asset must depend on its model AND features_validated.
    from shelfsense.orchestration.assets import defs

    pairs = [
        ("predictions_tvp_13", "model_tvp_13"),
        ("predictions_tvp_17", "model_tvp_17"),
        ("predictions_rmse_mh", "model_rmse_mh"),
        ("predictions_store_dept", "model_store_dept"),
        ("predictions_ylags", "model_ylags"),
        ("predictions_per_store", "model_per_store"),
        ("predictions_per_dept", "model_per_dept"),
    ]
    for pred_key, model_key in pairs:
        p_def = _get_assets_def(defs, AssetKey(pred_key))
        assert p_def is not None, f"{pred_key} not in defs"
        deps = p_def.dependency_keys
        assert AssetKey(model_key) in deps, f"{pred_key} should depend on {model_key}"
        assert AssetKey("features_validated") in deps, (
            f"{pred_key} should depend on features_validated"
        )


# -- Asset checks + config schema (added in commit 24) ------------------------


def test_asset_checks_count():
    from shelfsense.orchestration.assets import defs

    # 3 data + 2×7 model + 7 predictions + 1 ensemble + 1 submission = 26
    assert len(defs.asset_checks) == 26


def test_features_has_config_schema():
    from shelfsense.orchestration.assets import features

    # config_schema is on the underlying op; verify test_mode field is registered
    schema = features.node_def.config_schema
    assert schema is not None
    assert "test_mode" in schema.config_type.fields
