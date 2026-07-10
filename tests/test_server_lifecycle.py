"""Lifecycle tests for the background simulation worker (Finding #7).

A single stop_event / result_queue / thread used to be shared across runs, so a
Stop-then-Start could re-arm a still-alive worker via ``stop_event.clear()`` and
let two workers interleave output on one queue. ``_WorkerHandle`` gives each run
a FRESH stop_event + result_queue and joins the previous worker first.
"""

import queue
import threading
import time


def test_new_run_installs_fresh_objects():
    from cenop.server.main import _WorkerHandle

    h = _WorkerHandle()
    old_event = h.stop_event
    old_queue = h.result_queue

    new_event, new_queue = h.new_run()

    assert new_event is not old_event
    assert new_queue is not old_queue
    assert new_event is h.stop_event
    assert new_queue is h.result_queue
    # A fresh event must NOT be pre-set. The old bug cleared a SHARED event,
    # which re-armed a still-alive old worker; a fresh event cannot do that.
    assert not new_event.is_set()


def test_stop_and_join_joins_cooperative_worker():
    from cenop.server.main import _WorkerHandle

    h = _WorkerHandle()

    def cooperative():
        # Re-checks stop_event only between "batches", like the real worker.
        while not h.stop_event.is_set():
            h.result_queue.put(("tick", threading.get_ident()))
            time.sleep(0.01)

    t = h.start(target=cooperative, args=())
    assert t.is_alive()
    time.sleep(0.05)  # let it produce a few items

    ok = h.stop_and_join(timeout=5.0)

    assert ok is True
    assert not t.is_alive()
    assert h.thread is None
    assert h.stop_event.is_set()


def test_new_run_isolates_stubborn_worker():
    from cenop.server.main import _WorkerHandle

    h = _WorkerHandle()
    hard_kill = threading.Event()

    def stubborn():
        # Ignores stop_event entirely; only hard_kill releases it.
        hard_kill.wait(timeout=10.0)

    old_thread = h.start(target=stubborn, args=())
    old_queue = h.result_queue

    # new_run signals + joins (join times out because the worker ignores
    # stop_event) but STILL swaps to fresh objects so the new run is isolated.
    new_event, new_queue = h.new_run(timeout=0.2)

    try:
        assert old_thread.is_alive()  # stubborn worker survived the join
        assert new_queue is not old_queue  # new run got a fresh queue anyway
        assert new_event is h.stop_event
        assert h.result_queue is new_queue
        assert not new_event.is_set()  # fresh event, not the (set) old one
    finally:
        hard_kill.set()
        old_thread.join(timeout=5.0)
    assert not old_thread.is_alive()


# ---------------------------------------------------------------------------
# Integration: drive the REAL production worker (run_simulation_loop) through a
# _WorkerHandle across a Stop-then-Start with the first worker still alive.
# ---------------------------------------------------------------------------


class _StubState:
    year = 1


class _StubSim:
    def __init__(self):
        self.state = _StubState()


class _StubRunner:
    """Minimal stand-in exposing only what run_simulation_loop touches."""

    def __init__(self, worker_id, complete_after=None, park=None, ready=None):
        self.worker_id = worker_id
        self.complete_after = complete_after
        self.park = park
        self.ready = ready
        self.is_complete = False
        self.should_update_map = False
        self.tick = 0
        self.calls = 0
        self.max_ticks = 1000
        self.progress_percent = 0.0
        self.total_births = 0
        self.total_deaths = 0
        self.sim = _StubSim()

    def set_ticks_per_update(self, n):
        pass

    def step_ticks(self):
        self.tick += 1
        self.calls += 1
        if self.complete_after is not None and self.calls >= self.complete_after:
            self.is_complete = True
        if self.park is not None and self.calls >= 2:
            if self.ready is not None:
                self.ready.set()
            self.park.wait(timeout=5.0)  # park so the worker is definitely alive
        return {
            "population": 100 * self.worker_id,
            "year": 1,
            "day": self.tick % 360,
            "worker_id": self.worker_id,
        }


def _loop_args(runner, q, ev):
    # throttle=1.0 -> current_speed>=0.99 -> no sleeps, fast loop.
    return (
        runner,
        q,
        ev,
        [1.0],
        threading.Lock(),  # throttle_value, throttle_lock
        [48],
        threading.Lock(),  # ticks_per_update_value, ticks_lock
        [False],
        [2],
        threading.Lock(),  # trace_enabled, trace_length, trace_lock
        [False],
        threading.Lock(),  # skip_viz_value, skip_viz_lock
    )


def _drain_worker_ids(q):
    ids = set()
    has_complete = False
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break
        if msg.get("type") == "update":
            ids.add(msg["entry"]["worker_id"])
        elif msg.get("type") == "complete":
            has_complete = True
    return ids, has_complete


def test_stop_then_start_no_interleaving_and_old_worker_joined():
    from cenop.server.main import _WorkerHandle, run_simulation_loop

    h = _WorkerHandle()

    # --- start run #1 (production start_simulation path) ---
    ev1, q1 = h.new_run()
    runner1 = _StubRunner(worker_id=1, park=threading.Event(), ready=threading.Event())
    t1 = h.start(target=run_simulation_loop, args=_loop_args(runner1, q1, ev1))

    # Wait until worker #1 is parked mid-run so it is DEFINITELY still alive.
    assert runner1.ready.wait(timeout=5.0)
    assert t1.is_alive()

    # --- stop (production stop_simulation) ---
    h.stop_event.set()
    runner1.park.set()  # let worker #1 finish its current batch and observe stop

    # --- start run #2 (production start_simulation again) ---
    ev2, q2 = h.new_run(timeout=5.0)  # joins worker #1, installs fresh objects
    assert not t1.is_alive()  # old worker was joined / finished
    assert q2 is not q1
    assert ev2 is not ev1
    assert not ev2.is_set()

    runner2 = _StubRunner(worker_id=2, complete_after=3)
    t2 = h.start(target=run_simulation_loop, args=_loop_args(runner2, q2, ev2))
    t2.join(timeout=5.0)
    assert not t2.is_alive()

    # The fresh queue must contain ONLY worker #2 output (single producer).
    ids2, complete2 = _drain_worker_ids(q2)
    assert ids2 == {2}
    assert complete2 is True
    # Worker #1's output stayed isolated on the OLD queue.
    ids1, _ = _drain_worker_ids(q1)
    assert ids1 == {1}


def test_server_closures_use_worker_handle_and_drop_shared_clear():
    """Pin the Finding #7 fix in the Shiny server closures.

    The closures live inside server(input, output, session) and can't be
    invoked without a full reactive session, so guard the fix at the source
    level: the shared-event clear() must be gone and the closures must
    delegate to the fresh-per-run _WorkerHandle.
    """
    import inspect

    import cenop.server.main as main_mod

    src = inspect.getsource(main_mod)

    # The buggy pattern (re-arming a SHARED event) must be gone entirely.
    assert "stop_event.clear()" not in src
    # The server must own a fresh-per-run handle and route lifecycle through it.
    assert "worker = _WorkerHandle()" in src
    assert "worker.new_run(" in src  # start_simulation + reset_simulation
    assert "worker.stop_event.set()" in src  # stop_simulation
    assert "worker.start(" in src  # start_simulation
    assert "result_queue = worker.result_queue" in src  # poll snapshots the queue
