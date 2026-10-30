"""Explorations and runs started from the UI: one at a time, in a background thread, with live progress.

A job goes through the same checks as the command line (privacy mode, allowed models, LLMP address)
before it starts, and those failures are reported straight back to the request instead of as a job.
"""

from __future__ import annotations

import random
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime

from janus.llm.client import LLM, LLMPClient, LLMPSettings
from janus.llm.privacy import PrivacyMode, effective_mode
from janus.sim.engine import Simulation, check_models, resolve_models
from janus.sim.report import write_run
from janus.sim.scenario import Scenario
from janus.sim.trace import Node
from janus.sim.tree import Explorer, ExploreSettings, TreeStore
from janus.ui.data import Catalog

LLMFactory = Callable[[PrivacyMode, str], LLM]


class Busy(RuntimeError):
    """Another job is still running."""


@dataclass
class Job:
    id: str
    scenario: str
    kind: str  # explore or run
    settings: dict
    seed: int
    budget: int
    status: str = "running"  # running, done or failed
    spent: int = 0
    nodes: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    error: str | None = None
    skipped: list[str] = field(default_factory=list)
    report: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def default_llm_factory(catalog: Catalog) -> LLMFactory:
    def make(privacy: PrivacyMode, run_id: str) -> LLM:
        return LLMPClient(LLMPSettings.from_env(), privacy, run_id, cache_dir=catalog.out / ".cache")
    return make


class JobRunner:
    def __init__(self, catalog: Catalog, llm_factory: LLMFactory):
        self.catalog = catalog
        self.llm_factory = llm_factory
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def running(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.status == "running"), None)

    def explore(self, name: str, settings: ExploreSettings, options: list[int] | None, seed: int | None) -> Job:
        scenario = self.catalog.get(name)
        seed = seed if seed is not None else random.SystemRandom().randrange(1_000_000)
        job, llm, privacy, models = self._prepare(scenario, "explore", settings.as_dict(), seed, settings.budget)

        def work() -> None:
            with TreeStore(self.catalog.tree_path(scenario)) as store:
                explorer = Explorer(scenario, llm, models, privacy=privacy, store=store, settings=settings,
                                    seed=seed, run_id=job.id, on_node=lambda node, origin: self._progress(job, llm))
                explorer.explore(options)
                job.skipped = explorer.failures
                job.spent = explorer.spent

        return self._start(job, work)

    def run(self, name: str, option: int, seed: int | None, steps: int | None) -> Job:
        scenario = self.catalog.get(name)
        seed = seed if seed is not None else random.SystemRandom().randrange(1_000_000)
        job, llm, privacy, models = self._prepare(scenario, "run", {"option": option, "steps": steps}, seed, 0)
        simulation = Simulation(scenario, llm, models, privacy=privacy, run_id=job.id, option=option)

        def work() -> None:
            result = simulation.run(seed=seed, steps=steps, on_step=lambda node: self._progress(job, llm))
            _, report = write_run(result, scenario, self.catalog.out / scenario.name / job.id)
            with TreeStore(self.catalog.tree_path(scenario)) as store:
                store.add_root(simulation.root_id, simulation.option_label)
                for node in result.nodes:
                    store.add(node, simulation.root_id, "run")
                store.record_run(job.id, "run", simulation.root_id, result.nodes[-1].id, seed)
            job.report = str(report)

        return self._start(job, work)

    def _prepare(self, scenario: Scenario, kind: str, settings: dict, seed: int, budget: int):
        privacy = effective_mode(None, scenario.privacy, scenario.sensitive)
        models = resolve_models(scenario)
        check_models(models, privacy)
        job = Job(id=f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{uuid.uuid4().hex[:6]}", scenario=scenario.name,
                  kind=kind, settings=settings, seed=seed, budget=budget)
        llm = self.llm_factory(privacy, job.id)  # checks the LLMP address against the privacy mode
        return job, llm, privacy, models

    def _start(self, job: Job, work: Callable[[], None]) -> Job:
        with self._lock:
            if self.running() is not None:
                raise Busy("another exploration or run is still going")
            self.jobs[job.id] = job

        def target() -> None:
            try:
                work()
                job.status = "done"
            except Exception as e:  # the job reports any failure instead of dying silently
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")

        thread = threading.Thread(target=target, name=job.id, daemon=True)
        self._threads[job.id] = thread
        thread.start()
        return job

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        """For tests and shutdown: block until a job's thread ends."""
        if (thread := self._threads.get(job_id)) is not None:
            thread.join(timeout)

    @staticmethod
    def _progress(job: Job, llm: LLM, node: Node | None = None) -> None:
        job.nodes += 1
        job.spent = getattr(llm, "requests", job.spent)
