from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SupportedSurfaceTests(unittest.TestCase):
    def test_governed_router_is_fail_fast_and_generated_routes_are_not_mounted(self):
        source = (ROOT / "app.py").read_text()
        self.assertIn("app.include_router(governed_orders_router)", source)
        self.assertNotIn('"backend.routers.ai"', source)
        self.assertNotIn('_try_include("backend.routers.ai_extras")', source)
        self.assertNotIn('_try_include("backend.routers.customViews")', source)
        self.assertIn('status_code=410', source)

    def test_legacy_status_mutations_are_explicitly_rejected(self):
        source = (ROOT / "backend" / "routers" / "orders.py").read_text()
        self.assertGreaterEqual(source.count("Direct status changes are disabled") + source.count("Bulk status changes are disabled"), 2)
        self.assertIn("status_code=409", source)


if __name__ == "__main__":
    unittest.main()
