"""Policy test for the capacity gate.

Why this test exists: the thresholds and the rule that memory pressure
overrides free memory are the whole product decision. A later tweak that
silently stops gating would let a meeting recording swap the machine to death
again, which is exactly what happened on 2026-10-20.
"""

import unittest

from transcribe_diarize_app.resources import (
    PRESSURE_CRITICAL,
    PRESSURE_NORMAL,
    PRESSURE_WARNING,
    SUMMARY_HEADROOM_GB,
    TRANSCRIBE_HEADROOM_GB,
    Capacity,
    shortfall,
)


def capacity(pressure: int = PRESSURE_NORMAL, headroom_gb: float = 16.0) -> Capacity:
    return Capacity(
        pressure=pressure, headroom_gb=headroom_gb, swap_used_gb=0.0, load_1m=0.0, cpus=8
    )


class CapacityPolicyTests(unittest.TestCase):
    def test_roomy_machine_allows_every_stage(self) -> None:
        idle = capacity()
        self.assertIsNone(shortfall(idle, TRANSCRIBE_HEADROOM_GB))
        self.assertIsNone(shortfall(idle, SUMMARY_HEADROOM_GB))

    def test_pressure_wins_over_free_memory(self) -> None:
        pressured = capacity(pressure=PRESSURE_WARNING, headroom_gb=16.0)
        self.assertIn("pressure", str(shortfall(pressured, TRANSCRIBE_HEADROOM_GB)))
        critical = capacity(pressure=PRESSURE_CRITICAL, headroom_gb=16.0)
        self.assertIsNotNone(shortfall(critical, TRANSCRIBE_HEADROOM_GB))

    def test_short_headroom_blocks_the_stage_that_needs_it(self) -> None:
        tight = capacity(headroom_gb=4.0)
        self.assertIsNone(shortfall(tight, TRANSCRIBE_HEADROOM_GB))
        self.assertIsNotNone(shortfall(tight, SUMMARY_HEADROOM_GB))

    def test_thresholds_match_what_the_models_need(self) -> None:
        self.assertEqual(TRANSCRIBE_HEADROOM_GB, 2.5)
        self.assertEqual(SUMMARY_HEADROOM_GB, 6.0)


if __name__ == "__main__":
    unittest.main()
