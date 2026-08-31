"""Tests for the [perf] phase-timing instrumentation and COMFY_EXTRA_ARGS parsing.

Run with:  python -m unittest discover -s tests -v

These pin down the two properties that matter for instrumentation: the phase arithmetic is
right, and nothing in here can take a job down. A metric that raises is worse than no
metric at all, so the failure paths get as much attention as the happy one.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-perf-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-perf-output-"))
os.environ.setdefault("H3_OUTPUT_MODE", "base64")

if "runpod" not in sys.modules:
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules["runpod"] = runpod_stub

if "websocket" not in sys.modules:
    websocket_stub = types.ModuleType("websocket")
    websocket_stub.WebSocketTimeoutException = type("WebSocketTimeoutException", (Exception,), {})
    websocket_stub.create_connection = lambda *a, **k: None
    sys.modules["websocket"] = websocket_stub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402


def field(line: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}=(\S+)", line)
    assert match, f"field {name!r} missing from: {line}"
    return match.group(1)


class ExtraArgsTest(unittest.TestCase):
    def test_empty_and_whitespace_yield_no_args(self):
        for raw in ("", "   ", "\t"):
            with self.subTest(raw=raw):
                self.assertEqual(handler.parse_extra_args(raw), [])

    def test_highvram_reaches_the_command(self):
        self.assertEqual(handler.parse_extra_args("--highvram"), ["--highvram"])

    def test_multiple_flags(self):
        self.assertEqual(
            handler.parse_extra_args("--highvram --reserve-vram 2"),
            ["--highvram", "--reserve-vram", "2"],
        )

    def test_quoted_value_survives(self):
        """str.split() used to shred this into three broken tokens."""
        self.assertEqual(handler.parse_extra_args("--foo 'a b'"), ["--foo", "a b"])

    def test_unbalanced_quotes_fall_back_instead_of_raising(self):
        # A malformed tuning flag must never stop the worker from starting.
        self.assertEqual(handler.parse_extra_args("--foo 'unbalanced"), ["--foo", "'unbalanced"])


class JobTimerPhaseTest(unittest.TestCase):
    def make_timer(self) -> tuple[handler.JobTimer, float]:
        workflow = {
            "3": {"class_type": "KSampler"},
            "7": {"class_type": "MiniMaxH3VideoVAEDecode"},
        }
        timer = handler.JobTimer(workflow)
        now = time.monotonic()
        # Mirrors the shape of a real Blackwell job: 13s pre-sampling, 50s sampling
        # (5s of it the first step), 9s decode.
        timer.start = now - 71.4
        timer.marks = {
            "submitted": now - 70.0,
            "sampling_start": now - 60.0,
            "first_step_done": now - 55.0,
            "sampling_end": now - 10.0,
            "execution_end": now - 1.0,
        }
        timer.spans = {"comfy_wait": 0.0, "submit": 0.2, "output": 1.3}
        timer.steps_seen = 20
        timer.steps_total = 20
        return timer, now

    def test_phase_arithmetic(self):
        timer, _ = self.make_timer()
        line = timer.summary(job_index=1, status="ok")
        self.assertEqual(field(line, "pre_sampling"), "10.0s")
        self.assertEqual(field(line, "first_step"), "5.0s")
        self.assertEqual(field(line, "sampling"), "50.0s")
        self.assertEqual(field(line, "decode"), "9.0s")
        self.assertEqual(field(line, "output"), "1.3s")
        self.assertEqual(field(line, "steps"), "20/20")
        self.assertEqual(field(line, "per_step"), "2.50s")
        self.assertEqual(field(line, "status"), "ok")

    def test_total_is_wall_clock_since_job_start(self):
        timer, _ = self.make_timer()
        total = float(field(timer.summary(job_index=1, status="ok"), "total").rstrip("s"))
        self.assertAlmostEqual(total, 71.4, delta=1.0)

    def test_cold_process_flag_tracks_job_index(self):
        timer, _ = self.make_timer()
        self.assertEqual(field(timer.summary(job_index=1, status="ok"), "cold_process"), "true")
        self.assertEqual(field(timer.summary(job_index=2, status="ok"), "cold_process"), "false")

    def test_process_id_is_reported_so_worker_reuse_is_visible(self):
        timer, _ = self.make_timer()
        self.assertEqual(field(timer.summary(job_index=1, status="ok"), "proc"), handler.PROCESS_ID)

    def test_missing_phases_report_na_rather_than_raising(self):
        """A job that dies before sampling still has to produce a line."""
        timer = handler.JobTimer({})
        line = timer.summary(job_index=1, status="workflow_error")
        self.assertEqual(field(line, "pre_sampling"), "n/a")
        self.assertEqual(field(line, "sampling"), "n/a")
        self.assertEqual(field(line, "decode"), "n/a")
        self.assertEqual(field(line, "status"), "workflow_error")

    def test_line_is_prefixed_for_grepping(self):
        timer, _ = self.make_timer()
        self.assertTrue(timer.summary(job_index=1, status="ok").startswith("[perf] "))


class JobTimerEventTest(unittest.TestCase):
    def test_first_progress_frame_opens_sampling(self):
        timer = handler.JobTimer({})
        timer.on_progress(0, 20)
        self.assertIn("sampling_start", timer.marks)
        # value=0 must not be mistaken for a completed step.
        self.assertNotIn("first_step_done", timer.marks)
        timer.on_progress(1, 20)
        self.assertIn("first_step_done", timer.marks)

    def test_sampling_end_tracks_the_latest_frame(self):
        timer = handler.JobTimer({})
        timer.on_progress(1, 20)
        first_end = timer.marks["sampling_end"]
        timer.on_progress(20, 20)
        self.assertGreaterEqual(timer.marks["sampling_end"], first_end)
        self.assertEqual(timer.steps_seen, 20)
        self.assertEqual(timer.steps_total, 20)

    def test_sampling_start_is_not_reset_by_later_frames(self):
        timer = handler.JobTimer({})
        timer.on_progress(1, 20)
        start = timer.marks["sampling_start"]
        timer.on_progress(2, 20)
        self.assertEqual(timer.marks["sampling_start"], start)

    def test_nodes_are_labelled_by_class_type(self):
        timer = handler.JobTimer({"3": {"class_type": "KSampler"}})
        timer.on_node("3")
        timer.on_node("9")  # unknown id closes the KSampler span
        timer.on_execution_end()
        summary = timer.node_summary()
        self.assertIn("KSampler=", summary)
        self.assertIn("node9=", summary)
        self.assertTrue(summary.startswith("[perf] nodes "))

    def test_repeated_node_time_accumulates(self):
        timer = handler.JobTimer({"3": {"class_type": "KSampler"}})
        timer.on_node("3")
        timer.on_node("3")
        timer.on_execution_end()
        self.assertEqual(len(timer._node_totals), 1)

    def test_node_summary_is_none_when_no_events_arrived(self):
        self.assertIsNone(handler.JobTimer({}).node_summary())

    def test_non_integer_progress_values_are_ignored_safely(self):
        timer = handler.JobTimer({})
        timer.on_progress(None, None)
        timer.on_progress("x", "y")
        self.assertEqual(timer.steps_seen, 0)
        self.assertIsNone(timer.steps_total)


class EmitPerfTest(unittest.TestCase):
    def test_emit_never_raises_even_on_a_broken_timer(self):
        class Exploding(handler.JobTimer):
            def summary(self, **kwargs):
                raise RuntimeError("boom")

        # Must not propagate: the job already succeeded by the time this runs.
        handler.emit_perf(Exploding({}), job_index=1, status="ok")

    def test_emit_prints_both_lines(self):
        timer = handler.JobTimer({"3": {"class_type": "KSampler"}})
        timer.on_node("3")
        timer.on_execution_end()
        handler.emit_perf(timer, job_index=1, status="ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
