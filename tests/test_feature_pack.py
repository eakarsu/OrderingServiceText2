import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.feature_pack import (
    fallback_fulfillment_risk,
    fallback_menu_insights,
    fallback_order_summary,
    has_provider,
    readable_sections,
    records_to_csv,
    records_to_json,
)


class FeaturePackTests(unittest.TestCase):
    def test_export_preserves_false_zero_empty(self):
        rows = [{"name": "A", "count": 0, "active": False, "note": ""}]
        csv_text = records_to_csv(rows, ["name", "count", "active", "note"])
        self.assertIn("A,0,False,", csv_text)
        parsed = json.loads(records_to_json(rows))
        self.assertEqual(parsed["records"][0]["count"], 0)
        self.assertIs(parsed["records"][0]["active"], False)
        self.assertEqual(parsed["records"][0]["note"], "")

    def test_fallback_is_labelled_and_grounded(self):
        out = fallback_order_summary([{"status": "pending", "order_data": {"total_price": 25}}, {"status": "pending", "order_data": {"total_price": 0}}])
        self.assertIs(out["aiGenerated"], False)
        self.assertEqual(out["sections"]["totalOrders"], 2)
        self.assertEqual(out["sections"]["totalRevenue"], 25.0)
        self.assertEqual(fallback_menu_insights([{"category_id": 1, "is_available": False, "price": 0}])["sections"]["unavailableCount"], 1)
        self.assertEqual(fallback_fulfillment_risk([{"status": "pending"}])["sections"]["pendingCount"], 1)
        self.assertFalse(has_provider({}))

    def test_readable_preserves_falsy(self):
        text = readable_sections({"count": 0, "active": False, "note": ""})
        self.assertIn("Count: 0", text)
        self.assertIn("Active: false", text)
        self.assertIn("Note: (empty)", text)

    def test_feature_router_mounted_with_auth(self):
        source = (ROOT / "app.py").read_text()
        self.assertIn("feature_pack_router", source)
        router_src = (ROOT / "backend" / "routers" / "feature_pack.py").read_text()
        self.assertIn("get_current_user", router_src)
        self.assertIn('"/tasks"', router_src)


if __name__ == "__main__":
    unittest.main()
