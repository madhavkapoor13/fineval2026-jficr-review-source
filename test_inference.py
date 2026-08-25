from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference import (  # noqa: E402
    VALID_LABELS,
    extract_stage2_label,
    parse_original_query,
    read_input,
    stage1_ok,
    validate_predictions,
)


class InferenceTests(unittest.TestCase):
    def test_query_parser(self) -> None:
        query = (
            "Instructions\n\n"
            "Financial Question: 方針を維持しますか。\n"
            "Company Response: 現行方針を維持します。\n\n"
            "Directly output the chosen label."
        )
        question, response = parse_original_query(query)
        self.assertEqual(question, "方針を維持しますか。")
        self.assertEqual(response, "現行方針を維持します。")

    def test_label_parser_is_marker_scoped(self) -> None:
        self.assertEqual(
            extract_stage2_label("<LABEL>+1</LABEL>\n<RATIONALE>x</RATIONALE>"),
            ("+1", "strict_label_tag"),
        )
        self.assertEqual(
            extract_stage2_label("The boundary is +1 versus 0."),
            (None, "invalid"),
        )

    def test_stage1_outer_tags(self) -> None:
        self.assertTrue(stage1_ok("<STAGE1_ANALYSIS><GLOBAL>x</GLOBAL>"))
        self.assertFalse(stage1_ok("<GLOBAL>x</GLOBAL>"))

    def test_read_raw_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            pd.DataFrame(
                [
                    {
                        "id": 1,
                        "query": (
                            "Financial Question: 実施しますか。\n"
                            "Company Response: 実施する予定です。\n\n"
                            "Directly output the chosen label."
                        ),
                    }
                ]
            ).to_csv(path, index=False)
            rows = read_input(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["financial_question"], "実施しますか。")

    def test_prediction_validation(self) -> None:
        rows = [{"id": index, "prediction": label} for index, label in enumerate(sorted(VALID_LABELS))]
        validate_predictions(rows, 5)
        with self.assertRaises(ValueError):
            validate_predictions([{"id": 1, "prediction": "1"}], 1)

    def test_frozen_config(self) -> None:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["model"], "google/gemma-4-31b-it")
        self.assertEqual(config["workers"], 4)
        self.assertEqual(
            config["provider"]["only"], ["deepinfra/turbo", "friendli"]
        )


if __name__ == "__main__":
    unittest.main()
