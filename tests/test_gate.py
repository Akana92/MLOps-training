import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).parents[1] / 'scripts/pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class GateTests(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        self.assertTrue(pipeline.passes_gate(0.80))
        self.assertFalse(pipeline.passes_gate(0.7999))

    def test_missing_and_nonfinite_metrics_fail(self):
        for value in (None, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertFalse(pipeline.passes_gate(value))


if __name__ == '__main__':
    unittest.main()
