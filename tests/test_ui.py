import shutil
import threading

import pytest
from conftest import ROOT, SCENARIOS, FakeLLM
from fastapi.testclient import TestClient
from test_engine import BUY_NEW, amp_llm

from janus.ui.server import create_app


@pytest.fixture
def folder(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    for name in ("guitar_amp_purchase.yaml", "tank_pincer.yaml"):
        shutil.copy(SCENARIOS / name, scenarios / name)
    # the amp scenario's profile path is relative to the repo root, which is the tests' working directory
    assert (ROOT / "profiles" / "example_me.yaml").exists()
    return tmp_path


def make_client(folder, llm=None) -> TestClient:
    app = create_app(folder / "scenarios", folder / "runs", llm_factory=lambda privacy, run_id: llm or amp_llm(BUY_NEW))
    return TestClient(app)


def test_the_app_is_served_with_a_strict_content_policy(folder):
    client = make_client(folder)
    page = client.get("/")
    assert page.status_code == 200 and "JANUS" in page.text
    policy = page.headers["content-security-policy"]
    assert "default-src 'self'" in policy and "script-src 'self'" in policy and "style-src 'self'" in policy
    for asset in ("/static/app.js", "/static/app.css", "/static/fonts/playfair-display.ttf",
                  "/static/brand/msi-lockup-horizontal-dark.svg"):
        assert client.get(asset).status_code == 200, asset


def test_scenarios_are_listed_with_their_privacy(folder):
    listing = {s["name"]: s for s in make_client(folder).get("/api/scenarios").json()}
    amp = listing["guitar_amp_purchase"]
    assert amp["privacy"] == "local" and amp["sensitive"] and amp["against"] == "nature"
    assert len(amp["options"]) == 3 and amp["tree"] is None
    assert listing["tank_pincer"]["against"] == "actors" and listing["tank_pincer"]["llm_actors"] == ["blue", "red"]


def test_an_exploration_job_builds_the_tree_the_explorer_shows(folder):
    client = make_client(folder)
    assert client.get("/api/scenarios/guitar_amp_purchase/tree").status_code == 404
    job = client.post("/api/scenarios/guitar_amp_purchase/explore",
                      json={"options": [0, 1], "depth": 1, "branches": 2, "budget": 60}).json()
    client.app.state.jobs.wait(job["id"], timeout=30)
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "done"

    tree = client.get("/api/scenarios/guitar_amp_purchase/tree").json()
    assert tree["actors"] == ["me"] and len(tree["objectives"]["me"]) == 3
    assert [o["label"] for o in tree["options"]] == ["Buy a new amplifier now for 1500 euros",
                                                     "Buy a used amplifier now for 800 euros"]
    nodes = tree["options"][0]["nodes"]
    assert {n["kind"] for n in nodes} == {"explore", "end"}
    end = next(n for n in nodes if n["kind"] == "end")
    assert end["terminal"] == "horizon" and end["via"]["steps"] == 11 and "me.savings" in end["state"]
    listing = {s["name"]: s for s in client.get("/api/scenarios").json()}
    assert listing["guitar_amp_purchase"]["tree"]["nodes"] > 0 and listing["guitar_amp_purchase"]["tree"]["current"]

    sens = client.get("/api/scenarios/guitar_amp_purchase/sensitivity").json()
    assert sens["focal"] == "me" and sens["options"] == [o["label"] for o in tree["options"]]
    assert sens["baseline"]["best"] in sens["options"] and sens["rows"]
    assert {"change", "expected", "best", "margin", "changes_best"} <= set(sens["rows"][0])


def test_only_one_job_runs_at_a_time(folder):
    release = threading.Event()

    def slow(call):
        release.wait(10)
        return {"narrative": "Done.", "changes": BUY_NEW}

    llm = FakeLLM(Assessment=amp_llm(BUY_NEW).handlers["Assessment"], Resolution=slow)
    client = make_client(folder, llm)
    first = client.post("/api/scenarios/guitar_amp_purchase/explore", json={"options": [0], "depth": 1, "budget": 5})
    assert first.status_code == 202
    second = client.post("/api/scenarios/guitar_amp_purchase/run", json={"option": 1})
    assert second.status_code == 409
    release.set()
    client.app.state.jobs.wait(first.json()["id"], timeout=30)


def test_privacy_is_checked_before_a_job_starts(folder):
    text = (folder / "scenarios" / "tank_pincer.yaml").read_text(encoding="utf-8")
    (folder / "scenarios" / "tank_pincer.yaml").write_text(
        text.replace("game_master: ollama/gpt-oss:20b", "game_master: openrouter/qwen/qwen3-235b-a22b-2507"),
        encoding="utf-8")
    response = make_client(folder).post("/api/scenarios/tank_pincer/explore", json={"budget": 5})
    assert response.status_code == 400 and "not local" in response.json()["detail"]


def test_unknown_scenarios_are_404(folder):
    client = make_client(folder)
    assert client.get("/api/scenarios/nope/tree").status_code == 404
    assert client.get("/api/scenarios/guitar_amp_purchase/sensitivity").status_code == 404  # not explored
    assert client.post("/api/scenarios/nope/explore", json={}).status_code == 404
