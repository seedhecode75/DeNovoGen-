import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from rdkit import RDLogger

from denovogen.common import checkpoint, read, restore
from denovogen.data import Reference, prepare
from denovogen.pretrain import train
from denovogen.rl import choose, train_arm

ROOT = Path(__file__).resolve().parents[1]


class Restart(unittest.TestCase):
    def test_interrupted_training_reproduces_uninterrupted_weights(self):
        torch.set_num_threads(1)
        RDLogger.DisableLog("rdApp.error")
        cfg = dict(
            read(ROOT / "configs" / "smoke.json"),
            epochs=2,
            updates=2,
            max_rows=220,
            val_samples=4,
        )
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            prepare(ROOT / "examples" / "smoke.smi", folder / "data", cfg)

            def interrupt(path, state):
                checkpoint(path, state)
                if path.name == "last.pt":
                    raise InterruptedError(
                        "Simulated interruption after an atomic checkpoint"
                    )

            with patch("denovogen.pretrain.checkpoint", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    train(folder, "canonical", cfg, device)
            train(folder, "canonical", cfg, device)
            resumed = restore(folder / "pretrain/canonical/last.pt")["model"]
            (folder / "pretrain/canonical/last.pt").unlink()
            train(folder, "canonical", cfg, device)
            for key, v in restore(folder / "pretrain/canonical/last.pt")[
                "model"
            ].items():
                torch.testing.assert_close(v, resumed[key], atol=0, rtol=0)
            train(folder, "randomized", cfg, device)
            ref = Reference(folder / "data")
            try:
                choose(folder, cfg, device, ref)
                with patch("denovogen.rl.checkpoint", side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        train_arm(folder, cfg, device, ref, 6, 11)
                train_arm(folder, cfg, device, ref, 6, 11)
                dest = folder / "rl/R6/seed_11"
                resumed = restore(dest / "final.pt")["model"]
                (dest / "last.pt").unlink()
                (dest / "final.pt").unlink()
                train_arm(folder, cfg, device, ref, 6, 11)
                for key, v in restore(dest / "final.pt")["model"].items():
                    torch.testing.assert_close(v, resumed[key], atol=0, rtol=0)
            finally:
                ref.close()
