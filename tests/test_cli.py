import pytest
from conftest import SCENARIOS

from agent_factory.cli import main


@pytest.fixture(autouse=True)
def local_llmp(monkeypatch):
    monkeypatch.setenv("LLMP_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLMP_API_KEY", "test")


def test_check_shows_settings(capsys):
    assert main(["check", str(SCENARIOS / "guitar_amp_purchase.yaml")]) == 0
    out = capsys.readouterr().out
    assert "Privacy:  local (forced by a sensitive profile)" in out
    assert "Option 2 for me: Save first, then buy a used amplifier" in out


def test_sensitive_scenario_refuses_cloud_mode(capsys):
    assert main(["run", str(SCENARIOS / "guitar_amp_purchase.yaml"), "--privacy", "cloud_ok"]) == 2
    assert "forces privacy mode 'local'" in capsys.readouterr().err


def test_cloud_model_is_refused_before_anything_runs(capsys, tmp_path):
    code = main(["run", str(SCENARIOS / "tank_pincer.yaml"), "--out", str(tmp_path),
                 "--gm-model", "openrouter/qwen/qwen3-235b-a22b-2507"])
    assert code == 2
    assert "is not local" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
