import os
import shutil
import unittest
from datetime import datetime
from typing import Any, Dict, List

from core.data import ResultWriter, write_html, write_json
from core.utils import generate_html_report_content


class TestReporting(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/output"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_result_writer_formats(self):
        output_path = os.path.join(self.test_dir, "test_results.csv")
        writer = ResultWriter(
            output_path, formats=["csv", "json", "html"], auto_write=True
        )

        test_data = [
            {"test_case_id": "1", "name": "Test 1", "result": "OK"},
            {"test_case_id": "2", "name": "Test 2", "result": "NG", "error_message": "Failed"},
            {"test_case_id": "3", "name": "Test 3", "result": "SKIP"},
        ]

        for case in test_data:
            writer.add_result(case, case["result"], case.get("error_message"))

        # Check if files exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_results.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_results.html")))

    def test_html_generation(self):
        data = [
            {"test_case_id": "1", "timestamp": "2023-01-01 10:00:00", "result": "OK", "details": "All good"},
            {"test_case_id": "2", "timestamp": "2023-01-01 10:01:00", "result": "NG", "error_message": "Error"},
        ]
        html = generate_html_report_content(data)
        
        self.assertIn("Automation Report", html)
        self.assertIn("All good", html)
        self.assertIn("class='ok'", html)
        self.assertIn("class='ng'", html)

if __name__ == "__main__":
    unittest.main()
