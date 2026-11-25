import math
import re

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    QED,
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD, BOS, EOS, UNK = range(4)
PATTERN = re.compile(
    r"\[[^\[\]\s]+\]|%\([0-9]+\)|%[0-9]{2}|Br|Cl|[BCNOPSFIbcnops]|[0-9]|->|<-|[()=#/\\:.~*\-]"
)
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def tokenize(s):
    out = []
    pos = 0
    while pos < len(s):
        m = PATTERN.match(s, pos)
        if m is None:
            raise ValueError(f"Unsupported token at position {pos} in {s!r}")
        out.append(m.group())
        pos = m.end()
    if not out or "".join(out) != s:
        raise ValueError(f"Empty or non-roundtripping SMILES {s!r}")
    return out


def category(token):
    if token in SPECIAL:
        return token
    if token[0].isdigit() or token.startswith("%"):
        return "ring"
    if token.startswith("["):
        return "bracket_atom"
    if token in ("(", ")"):
        return "branch"
    if token[0].isalpha():
        return "aromatic_atom" if token.islower() else "atom"
    return "bond_or_separator"


def molecule(s):
    mol = Chem.MolFromSmiles(s) if s else None
    return mol if mol is not None and mol.GetNumAtoms() else None


def identity(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def query(spec):
    q = Chem.MolFromSmarts(spec)
    if q is None or not q.GetNumAtoms():
        raise ValueError(f"Invalid SMARTS {spec!r}")
    return q


def inspect(s, reference, goal, *, extended=False, full=False):
    mol = molecule(s)
    row = dict(smiles=s, valid=mol is not None)
    if mol is None:
        return row
    key = identity(mol)
    found = reference.lookup(key, full=full)
    row.update(
        identity=key,
        scaffold=scaffold(mol),
        train_novel=not found["train"],
        source_novel=found["source_novel"],
        source_id=found["source_id"],
        fragment=bool(mol.HasSubstructMatch(goal)),
    )
    try:
        row["qed"] = float(QED.qed(mol))
    except (RuntimeError, ValueError, ArithmeticError):
        row["qed"] = None
    if extended:
        mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
        hbd, hba = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
        row.update(
            mw=mw,
            logp=logp,
            hbd=hbd,
            hba=hba,
            tpsa=Descriptors.TPSA(mol),
            rotatable=Lipinski.NumRotatableBonds(mol),
            fsp3=rdMolDescriptors.CalcFractionCSP3(mol),
            ro5_violations=sum((mw > 500, logp > 5, hbd > 5, hba > 10)),
        )
    return row


def diversity(smiles, seed, pair_budget=100000):
    fps = [FP.GetFingerprint(molecule(s)) for s in smiles]
    n = len(fps)
    pairs = n * (n - 1) // 2
    if not pairs:
        return None, 0, None
    if pairs <= pair_budget:
        total = sum(
            sum(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
            for i in range(1, n)
        )
        return 1 - total / pairs, pairs, 0.0
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, size=pair_budget)
    b = rng.integers(0, n - 1, size=pair_budget)
    b += b >= a
    scores = np.fromiter(
        (1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i, j in zip(a, b)),
        dtype=float,
        count=pair_budget,
    )
    return (
        float(scores.mean()),
        pair_budget,
        float(scores.std(ddof=1) / math.sqrt(pair_budget)),
    )


def score(row, arm, memory, cfg):
    if not row["valid"]:
        return 0.0
    q = row["qed"] if row["qed"] is not None else 0.0
    n, f = float(row["train_novel"]), float(row["fragment"])
    values = [1, 1 + n, 1 + q, 1 + n + q, 1 + n + q + f, 1 + n + q + f]
    value = values[arm - 1] / 4
    if arm == 6:
        crowd = min(1.0, memory[row["scaffold"]] / cfg["crowd_cap"])
        value *= 1 - cfg["crowd_lambda"] * crowd
    return value
