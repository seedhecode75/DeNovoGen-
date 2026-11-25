import collections
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

from denovogen.chem import (
    BOS,
    EOS,
    PAD,
    identity,
    inspect,
    molecule,
    query,
    score,
    tokenize,
)
from denovogen.common import read
from denovogen.data import Reference, Sequences, prepare, split_name
from denovogen.evaluate import export_run, summarize
from denovogen.model import (
    Generator,
    action_logits,
    decode,
    reinforce_loss,
    sample,
    trajectory,
)

ROOT = Path(__file__).resolve().parents[1]


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RDLogger.DisableLog("rdApp.error")
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.cfg = read(ROOT / "configs" / "smoke.json")
        cls.man = prepare(ROOT / "examples" / "smoke.smi", cls.folder, cls.cfg)
        cls.ref = Reference(cls.folder)
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.ref.close()
        cls.temp.cleanup()

    def test_tokenizer(self):
        for text in [
            "ClCBr",
            "[13CH3][C@@H](O)[NH3+]",
            "c1cc[nH]c1",
            "C%12CCCCC%12",
            "C/C=C\\C",
            "C.C",
            "C:C",
            "C12CCCCC1CCCCC2",
            "N->[Cu+2]",
            "C%(100)CCCCC%(100)",
        ]:
            self.assertEqual("".join(tokenize(text)), text)
        self.assertEqual(tokenize("C12"), ["C", "1", "2"])
        self.assertEqual(tokenize("C%12"), ["C", "%12"])
        self.assertRaises(ValueError, tokenize, "C!C")

    def test_chemistry_and_exact_references(self):
        key = self.ref.con.execute(
            "SELECT key FROM items WHERE part='train' LIMIT 1"
        ).fetchone()[0]
        other = Chem.MolToSmiles(molecule(key), doRandom=True)
        for s in [key, other]:
            r = inspect(s, self.ref, query("CN"), full=True)
            self.assertFalse(r["train_novel"])
            self.assertFalse(r["source_novel"])
        held = self.ref.con.execute(
            "SELECT key FROM items WHERE part='test' LIMIT 1"
        ).fetchone()[0]
        r = inspect(held, self.ref, query("CN"), full=True)
        self.assertTrue(r["train_novel"])
        self.assertFalse(r["source_novel"])
        self.assertEqual(identity(molecule("OCC")), identity(molecule("CCO")))
        self.assertTrue(molecule("NC").HasSubstructMatch(query("CN")))
        self.assertFalse(inspect("", self.ref, query("CN"))["valid"])

    def test_fixed_budget_views_and_splits(self):
        c = Sequences(self.folder, "train")
        r = Sequences(self.folder, "train", "randomized")
        self.assertTrue(np.array_equal(c.offsets, r.offsets))
        self.assertTrue(np.any(c.tokens != r.tokens))
        self.assertGreater(self.man["changed_training_views"], 0)
        self.assertTrue(self.man["complete_source_reference"])
        self.assertEqual(self.man["split_counts"]["train"], len(c))
        self.assertEqual(
            self.ref.con.execute(
                "SELECT count(*) FROM (SELECT key FROM items GROUP BY key HAVING count(*)>1)"
            ).fetchone()[0],
            0,
        )

    def test_subset_reference_and_preparation_guard(self):
        cfg = dict(self.cfg, max_rows=180, full_reference=False)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            m = prepare(ROOT / "examples" / "smoke.smi", folder, cfg)
            self.assertFalse(m["complete_source_reference"])
            ref = Reference(folder)
            self.assertIsNone(ref.lookup("CCCC", full=True)["source_novel"])
            ref.close()
            with self.assertRaises(ValueError):
                prepare(
                    ROOT / "examples" / "smoke.smi", folder, dict(cfg, data_seed=100)
                )

    def test_scaffold_generalization(self):
        scaffold_set = [
            "c1ccccc1",
            "c1ccncc1",
            "c1ccoc1",
            "c1ccsc1",
            "C1CC1",
            "C1CCC1",
            "C1CCCC1",
            "C1CCCCC1",
            "C1CCCCCC1",
            "C1CCCCCCC1",
            "C1COC1",
            "C1CCOC1",
            "C1CCOCC1",
            "C1CCNCC1",
            "C1CCNC1",
            "C1CNC1",
            "C1CN1",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "groups.smi"
            f.write_text(
                "\n".join(
                    prefix + r for prefix in ("C", "CC", "CCC") for r in scaffold_set
                )
            )
            seed = next(
                i
                for i in range(100)
                if len({split_name(s, i) for s in scaffold_set}) == 3
            )
            cfg = dict(self.cfg, split="scaffold", data_seed=seed)
            folder = Path(tmp) / "prepared"
            prepare(f, folder, cfg)
            ref = Reference(folder)
            self.assertEqual(
                ref.con.execute(
                    "SELECT count(*) FROM (SELECT grp FROM items GROUP BY grp HAVING count(DISTINCT part)>1)"
                ).fetchone()[0],
                0,
            )
            ref.close()

    def test_correct_reward_gradient(self):
        logits = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([logits], lr=0.2)
        before = logits.softmax(-1)[0].item()
        loss = reinforce_loss(logits.log_softmax(-1), torch.tensor([1.0, 0.0]))
        loss.backward()
        opt.step()
        self.assertGreater(logits.softmax(-1)[0].item(), before)

    def test_padding_eos_and_cached_logits(self):
        torch.manual_seed(17)
        model = Generator(9, self.cfg).eval()
        x = torch.tensor([[BOS, 4, 5, EOS]])
        out, _ = model(x)
        state = None
        for j in range(x.size(1)):
            one, state = model(x[:, j : j + 1], state)
            torch.testing.assert_close(one[:, 0], out[:, j])
        a, _ = trajectory(model, x)
        manual = (
            action_logits(out[:, :-1]).log_softmax(-1).gather(-1, x[:, 1:, None]).sum()
        )
        torch.testing.assert_close(a.sum(), manual)
        first = torch.autograd.grad(a.sum(), model.output.weight)[0]
        b, _ = trajectory(model, torch.nn.functional.pad(x, (0, 4), value=PAD))
        torch.testing.assert_close(a, b)
        second = torch.autograd.grad(b.sum(), model.output.weight)[0]
        torch.testing.assert_close(first, second)

    def test_sampling_support_and_rewards(self):
        vocab = self.man["vocab"]
        model = Generator(len(vocab), self.cfg)
        seq = sample(
            model, 16, 12, torch.device("cpu"), torch.Generator().manual_seed(13)
        )
        self.assertEqual(len(decode(seq, vocab)), 16)
        row = dict(
            valid=True, qed=0.6, train_novel=True, fragment=True, scaffold="ring"
        )
        scores = [
            score(row, a, collections.Counter(ring=20), self.cfg) for a in range(1, 7)
        ]
        np.testing.assert_allclose(scores, [0.25, 0.5, 0.4, 0.65, 0.9, 0.45])
        self.assertTrue(
            all(score(dict(valid=False), a, {}, self.cfg) == 0 for a in range(1, 7))
        )

    def test_metric_denominators_and_export_resume(self):
        valid = dict(inspect("CCO", self.ref, query("CN"), full=True), eos=True)
        same = dict(valid, smiles="OCC")
        invalid = dict(valid=False, smiles="(", eos=False)
        m = summarize([valid, same, invalid], 123, self.cfg)
        self.assertAlmostEqual(m["validity"], 2 / 3)
        self.assertEqual(m["raw_uniqueness"], 1.0)
        self.assertEqual(m["uniqueness"], 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            vocab = self.man["vocab"]
            model = Generator(len(vocab), self.cfg)
            args = (
                model,
                vocab,
                self.ref,
                self.cfg,
                torch.device("cpu"),
                dest,
                55,
                19,
                4,
                "test-model",
            )
            first = export_run(*args)
            (dest / "metrics.json").unlink()
            for name in [
                "chunk_000000016.json",
                "chunk_000000016.csv",
                "chunk_000000016.time",
            ]:
                (dest / name).unlink()
            resumed = export_run(*args)
            self.assertEqual(first["valid"], resumed["valid"])
            ids = [
                r["attempt_id"]
                for f in sorted(dest.glob("chunk_*.json"))
                for r in read(f)["rows"]
            ]
            self.assertEqual(ids, list(range(19)))
            with self.assertRaises(ValueError):
                export_run(*args[:-1], "different-model")


if __name__ == "__main__":
    unittest.main()
