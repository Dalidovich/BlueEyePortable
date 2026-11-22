from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from benchmark import export_benchmarks, parse_batch_sizes, run_benchmark
from database import CSV_COLUMNS, Database, ResultRow, export_results_csv
from evaluation import LABELS, compute_metrics, folder_label_for, import_folder_labels
from scanner import Scanner, discover_images, sha256_file
from settings import to_portable_path
from tests.fakes import FakeEngine, write_corrupt, write_image


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="blueeye_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.images = self.tmp / "images"
        self.images.mkdir()
        self.db = Database(self.tmp / "test.db")
        self.addCleanup(self.db.close)

    def make_scanner(self, engine: FakeEngine) -> Scanner:
        return Scanner(self.db, engine, "Fake GPU", "rev1")


class DatabaseTests(TempDirTestCase):
    def test_schema_created(self) -> None:
        tables = {row[0] for row in self.db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"runs", "results", "labels", "benchmarks"} <= tables)
        self.assertEqual(self.db.conn.execute("PRAGMA user_version").fetchone()[0], 1)
        self.assertEqual(self.db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
        self.assertEqual(self.db.conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_upsert_keeps_one_row_per_path_and_precision(self) -> None:
        base = dict(path="a.png", file_name="a.png", status="ok", precision="fp32")
        self.db.save_results([ResultRow(**base, label="safe")])
        self.db.save_results([ResultRow(**base, label="explicit")])
        self.db.save_results([ResultRow(path="a.png", file_name="a.png", status="ok", precision="bf16")])
        rows = self.db.query_results()
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.db.find_result("a.png", "fp32")["label"], "explicit")

    def test_mark_interrupted_runs(self) -> None:
        running = self.db.start_run("x", True, "cpu", None, "fp32", 4)
        done = self.db.start_run("y", True, "cpu", None, "fp32", 4)
        self.db.finish_run(done, "completed")
        self.assertEqual(self.db.mark_interrupted_runs(), 1)
        statuses = {run["id"]: run["status"] for run in self.db.list_runs()}
        self.assertEqual(statuses[running], "interrupted")
        self.assertEqual(statuses[done], "completed")

    def test_order_by_is_whitelisted(self) -> None:
        with self.assertRaises(ValueError):
            self.db.query_results(order_by="id; DROP TABLE results")

    def test_summary_counts(self) -> None:
        base = dict(file_name="f", precision="fp32")
        self.db.save_results([
            ResultRow(path="1", status="ok", label="safe", **base),
            ResultRow(path="2", status="ok", label="explicit", **base),
            ResultRow(path="3", status="error", error="bad", **base),
        ])
        self.assertEqual(self.db.summary("fp32"),
                         {"total": 3, "ok": 2, "safe": 1, "suggestive": 0, "explicit": 1, "errors": 1})

    def test_folder_labels_do_not_override_manual(self) -> None:
        self.db.set_label("h1", "safe", "a.png", source="manual")
        self.db.set_label("h1", "explicit", "b.png", source="folder")
        self.assertEqual(self.db.get_labels()["h1"]["label"], "safe")
        self.db.set_label("h2", "safe", "a.png", source="folder")
        self.db.set_label("h2", "explicit", "b.png", source="folder")
        self.assertEqual(self.db.get_labels()["h2"]["label"], "explicit")
        self.db.set_label("h2", "suggestive", "b.png", source="manual")
        self.assertEqual(self.db.get_labels()["h2"]["source"], "manual")
        self.db.remove_label("h2")
        self.assertNotIn("h2", self.db.get_labels())

    def test_csv_export_columns_and_encoding(self) -> None:
        self.db.save_results([ResultRow(path="a.png", file_name="a.png", status="ok", precision="fp32",
                                        label="safe", p_safe=0.9, p_suggestive=0.05, p_explicit=0.05,
                                        confidence=0.9, seconds=0.1, file_hash="abc")])
        self.db.save_results([ResultRow(path="b.png", file_name="b.png", status="error", precision="fp32",
                                        error="broken")])
        target = self.tmp / "out" / "results.csv"
        self.assertEqual(export_results_csv(self.db, target, precision="fp32"), 2)
        self.assertEqual(target.read_bytes()[:3], b"\xef\xbb\xbf")
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()), list(CSV_COLUMNS))
        by_name = {row["file_name"]: row for row in rows}
        self.assertEqual(by_name["a.png"]["predicted_class"], "safe")
        self.assertEqual(by_name["a.png"]["safe_probability"], "0.9")
        self.assertEqual(by_name["b.png"]["error"], "broken")
        self.assertFalse(target.with_name("results.csv.tmp").exists())
        self.assertEqual(export_results_csv(self.db, target, label="safe"), 1)


class ScannerTests(TempDirTestCase):
    def make_images(self, count: int) -> list[Path]:
        paths = []
        for seed in range(count):
            path = self.images / f"img_{seed:02d}.png"
            write_image(path, seed + 1)
            paths.append(path)
        return paths

    def test_discover_images_filters_and_sorts(self) -> None:
        write_image(self.images / "B.PNG", 1)
        write_image(self.images / "a.jpg", 2)
        write_image(self.images / "sub" / "c.png", 3)
        (self.images / "notes.txt").write_text("x")
        flat = [p.name for p in discover_images(self.images, recursive=False)]
        deep = [p.name for p in discover_images(self.images, recursive=True)]
        self.assertEqual(flat, ["a.jpg", "B.PNG"])
        self.assertEqual(deep, ["a.jpg", "B.PNG", "c.png"])
        with self.assertRaises(FileNotFoundError):
            discover_images(self.images / "missing", True)

    def test_scan_then_skip_unchanged(self) -> None:
        self.make_images(5)
        engine = FakeEngine()
        first = self.make_scanner(engine).run(self.images, True, 2)
        self.assertEqual((first.status, first.processed, first.skipped, first.errors), ("completed", 5, 0, 0))
        self.assertEqual(len(engine.classified_paths), 5)
        engine_again = FakeEngine()
        second = self.make_scanner(engine_again).run(self.images, True, 2)
        self.assertEqual((second.processed, second.skipped, second.reused), (0, 5, 0))
        self.assertEqual(engine_again.classified_paths, [])
        self.assertEqual(self.db.summary("fp32")["total"], 5)

    def test_touched_file_with_same_content_is_skipped_and_mtime_refreshed(self) -> None:
        paths = self.make_images(1)
        self.make_scanner(FakeEngine()).run(self.images, True, 2)
        stat = paths[0].stat()
        import os
        os.utime(paths[0], (stat.st_atime, stat.st_mtime + 100))
        engine = FakeEngine()
        summary = self.make_scanner(engine).run(self.images, True, 2)
        self.assertEqual((summary.skipped, summary.processed), (1, 0))
        self.assertEqual(self.db.find_result(to_portable_path(paths[0]), "fp32")["file_mtime"],
                         paths[0].stat().st_mtime)

    def test_reuse_by_hash_for_duplicate_content_at_other_path(self) -> None:
        self.make_images(2)
        self.make_scanner(FakeEngine()).run(self.images, True, 2)
        duplicate = self.images / "copy" / "dup.png"
        duplicate.parent.mkdir()
        shutil.copy2(self.images / "img_00.png", duplicate)
        engine = FakeEngine()
        summary = self.make_scanner(engine).run(self.images, True, 2)
        self.assertEqual((summary.reused, summary.processed, summary.skipped), (1, 0, 2))
        self.assertEqual(engine.classified_paths, [])
        row = self.db.find_result(to_portable_path(duplicate), "fp32")
        original = self.db.find_result(to_portable_path(self.images / "img_00.png"), "fp32")
        self.assertEqual(row["reused"], 1)
        self.assertEqual(row["seconds"], 0.0)
        self.assertEqual(row["label"], original["label"])
        self.assertEqual(row["p_safe"], original["p_safe"])

    def test_intra_queue_duplicates_are_inferred_once(self) -> None:
        write_image(self.images / "a.png", 7)
        shutil.copy2(self.images / "a.png", self.images / "b.png")
        shutil.copy2(self.images / "a.png", self.images / "c.png")
        write_image(self.images / "d.png", 8)
        engine = FakeEngine()
        summary = self.make_scanner(engine).run(self.images, True, 4)
        self.assertEqual(len(engine.classified_paths), 2)
        self.assertEqual((summary.processed, summary.reused, summary.errors), (2, 2, 0))
        self.assertEqual(self.db.summary("fp32")["ok"], 4)
        self.assertEqual(self.db.find_result(to_portable_path(self.images / "c.png"), "fp32")["reused"], 1)

    def test_corrupt_file_gets_error_row_and_is_retried(self) -> None:
        write_image(self.images / "good.png", 1)
        write_corrupt(self.images / "bad.png")
        engine = FakeEngine()
        first = self.make_scanner(engine).run(self.images, True, 2)
        self.assertEqual((first.processed, first.errors), (1, 1))
        bad = self.db.find_result(to_portable_path(self.images / "bad.png"), "fp32")
        self.assertEqual(bad["status"], "error")
        self.assertTrue(bad["error"])
        self.assertIsNotNone(bad["file_hash"])
        retry_engine = FakeEngine()
        second = self.make_scanner(retry_engine).run(self.images, True, 2)
        self.assertEqual((second.skipped, second.errors), (1, 1))
        self.assertEqual([p.name for p in retry_engine.classified_paths], ["bad.png"])
        write_image(self.images / "bad.png", 99)
        third = self.make_scanner(FakeEngine()).run(self.images, True, 2)
        self.assertEqual((third.processed, third.errors), (1, 0))
        self.assertEqual(self.db.find_result(to_portable_path(self.images / "bad.png"), "fp32")["status"], "ok")

    def test_force_reprocesses_everything(self) -> None:
        self.make_images(3)
        self.make_scanner(FakeEngine()).run(self.images, True, 2)
        engine = FakeEngine()
        summary = self.make_scanner(engine).run(self.images, True, 2, force=True)
        self.assertEqual((summary.processed, summary.skipped, summary.reused), (3, 0, 0))
        self.assertEqual(len(engine.classified_paths), 3)
        self.assertEqual(self.db.summary("fp32")["total"], 3)

    def test_stop_mid_scan_keeps_committed_batches(self) -> None:
        self.make_images(6)
        engine = FakeEngine()
        flag = {"stop": False}

        def on_progress(progress) -> None:
            if progress.phase == "inference" and progress.done >= 2:
                flag["stop"] = True

        summary = self.make_scanner(engine).run(self.images, True, 2, progress=on_progress,
                                                should_stop=lambda: flag["stop"])
        self.assertEqual(summary.status, "stopped")
        self.assertEqual(summary.processed, 2)
        self.assertEqual(self.db.summary("fp32")["total"], 2)
        run = self.db.get_run(summary.run_id)
        self.assertEqual(run["status"], "stopped")
        self.assertEqual(run["processed"], 2)
        resumed = self.make_scanner(FakeEngine()).run(self.images, True, 2)
        self.assertEqual((resumed.skipped, resumed.processed), (2, 4))

    def test_stop_during_checking_phase(self) -> None:
        self.make_images(4)
        summary = self.make_scanner(FakeEngine()).run(self.images, True, 2, should_stop=lambda: True)
        self.assertEqual(summary.status, "stopped")
        self.assertEqual(self.db.summary()["total"], 0)

    def test_failure_marks_run_failed_and_reraises(self) -> None:
        self.make_images(2)
        engine = FakeEngine()

        def broken(paths, batch_size, should_stop=None):
            raise RuntimeError("boom")
            yield

        engine.classify = broken
        with self.assertRaises(RuntimeError):
            self.make_scanner(engine).run(self.images, True, 2)
        run = self.db.list_runs()[0]
        self.assertEqual(run["status"], "failed")
        self.assertIn("boom", run["message"])

    def test_originals_are_untouched(self) -> None:
        paths = self.make_images(2)
        before = {p: (sha256_file(p), p.stat().st_mtime) for p in paths}
        self.make_scanner(FakeEngine()).run(self.images, True, 2)
        after = {p: (sha256_file(p), p.stat().st_mtime) for p in paths}
        self.assertEqual(before, after)

    def test_labeled_pairs_and_folder_labels(self) -> None:
        dataset = self.tmp / "dataset"
        write_image(dataset / "safe" / "a.png", 1)
        write_image(dataset / "Explicit" / "nested" / "b.png", 2)
        write_image(dataset / "other" / "c.png", 3)
        self.assertEqual(folder_label_for(dataset / "safe" / "a.png", dataset), "safe")
        self.assertEqual(folder_label_for(dataset / "Explicit" / "nested" / "b.png", dataset), "explicit")
        self.assertIsNone(folder_label_for(dataset / "other" / "c.png", dataset))
        self.assertEqual(import_folder_labels(self.db, dataset), 2)
        self.make_scanner(FakeEngine()).run(dataset, True, 2)
        pairs = self.db.labeled_pairs("fp32")
        self.assertEqual(sorted(p["true_label"] for p in pairs), ["explicit", "safe"])
        self.assertEqual(self.db.labeled_pairs("bf16"), [])


class EvaluationTests(unittest.TestCase):
    def test_compute_metrics_matches_hand_computed_matrix(self) -> None:
        pairs = ([("safe", "safe")] * 8 + [("safe", "suggestive")] * 2
                 + [("suggestive", "suggestive")] * 3 + [("suggestive", "explicit")] * 1
                 + [("suggestive", "safe")] * 1
                 + [("explicit", "explicit")] * 4 + [("explicit", "suggestive")] * 1)
        metrics = compute_metrics(pairs)
        self.assertEqual(metrics.confusion, [[8, 2, 0], [1, 3, 1], [0, 1, 4]])
        self.assertEqual((metrics.total, metrics.correct), (20, 15))
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        safe, suggestive, explicit = metrics.per_class
        self.assertAlmostEqual(safe.precision, 8 / 9)
        self.assertAlmostEqual(safe.recall, 8 / 10)
        self.assertAlmostEqual(suggestive.precision, 3 / 6)
        self.assertAlmostEqual(suggestive.recall, 3 / 5)
        self.assertAlmostEqual(explicit.precision, 4 / 5)
        self.assertAlmostEqual(explicit.recall, 4 / 5)
        self.assertAlmostEqual(suggestive.f1, 2 * 0.5 * 0.6 / 1.1)
        self.assertEqual((safe.support, suggestive.support, explicit.support), (10, 5, 5))
        self.assertAlmostEqual(suggestive.accuracy, 0.6)
        self.assertAlmostEqual(metrics.macro_recall, (0.8 + 0.6 + 0.8) / 3)
        expected_weighted = (safe.f1 * 10 + suggestive.f1 * 5 + explicit.f1 * 5) / 20
        self.assertAlmostEqual(metrics.weighted_f1, expected_weighted)

    def test_empty_input_gives_zeros(self) -> None:
        metrics = compute_metrics([])
        self.assertEqual((metrics.total, metrics.accuracy, metrics.macro_f1, metrics.weighted_f1), (0, 0.0, 0.0, 0.0))
        self.assertEqual(metrics.confusion, [[0] * 3] * 3)
        self.assertEqual([c.label for c in metrics.per_class], list(LABELS))

    def test_absent_class_does_not_divide_by_zero(self) -> None:
        metrics = compute_metrics([("safe", "safe")])
        self.assertEqual(metrics.per_class[2].precision, 0.0)
        self.assertEqual(metrics.per_class[2].accuracy, 0.0)


class BenchmarkTests(TempDirTestCase):
    def test_parse_batch_sizes(self) -> None:
        self.assertEqual(parse_batch_sizes("1, 2,4,8"), [1, 2, 4, 8])
        self.assertEqual(parse_batch_sizes("4,2,4,2"), [4, 2])
        for bad in ("", "1,,2", "0", "-3", "a", "1.5", "2,x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_batch_sizes(bad)

    def test_benchmark_does_not_write_results_and_exports(self) -> None:
        paths = []
        for seed in range(6):
            path = self.images / f"b{seed}.png"
            write_image(path, seed + 1)
            paths.append(path)
        engine = FakeEngine()
        engine.max_batch_size = 1
        results = run_benchmark(engine, paths, [1, 2], "Fake GPU", image_limit=4, db=self.db)
        self.assertEqual(len(results), 2)
        self.assertEqual({r.session_id for r in results}, {results[0].session_id})
        self.assertTrue(all(r.images == 4 for r in results))
        self.assertEqual(results[1].effective_batch_size, 2)
        self.assertEqual(self.db.summary()["total"], 0)
        self.assertEqual(self.db.query_results(), [])
        self.assertEqual(len(self.db.list_benchmarks(results[0].session_id)), 2)
        json_path = self.tmp / "bench.json"
        csv_path = self.tmp / "bench.csv"
        export_benchmarks(results, json_path, "json")
        export_benchmarks([r.__dict__ for r in results], csv_path, "csv")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("generated_at", payload)
        self.assertEqual(len(payload["results"]), 2)
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
