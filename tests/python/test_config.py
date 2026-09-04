from pathlib import Path

from crossdating_model.config import load_config


def test_frozen_v5_contract() -> None:
    config = load_config(Path("configs/v5.yaml"))
    assert config["feature_count"] == 207
    assert config["window_width"] == 13
    assert config["cofecha_js_version"] == "0.2.0"
    assert config["operations"]["whole"] == {"minimum": -100, "maximum": 100, "exclude": [0]}
    assert config["categories"]["D"].endswith("at least 30 years apart")
