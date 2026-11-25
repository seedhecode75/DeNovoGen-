import csv
from functools import lru_cache

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import QED, Crippen, Descriptors, FilterCatalog
from rdkit.Contrib.SA_Score import sascorer

from .chem import FP, molecule
from .common import save_json
from .report import csv_file


def reference_sample(con, part, n, seed):
    size = con.execute("SELECT count(*) FROM items WHERE part=?", (part,)).fetchone()[0]
    use = set(
        np.random.default_rng(seed).choice(size, min(n, size), replace=False).tolist()
    )
    return [
        key
        for i, (key,) in enumerate(
            con.execute("SELECT key FROM items WHERE part=? ORDER BY key", (part,))
        )
        if i in use
    ]


def compare_reference(folder, cfg, ref):
    # Bound diagnostic cost and retain the actual reference panel for reproducibility.
    out = folder / "quality"
    out.mkdir(exist_ok=True)
    size = cfg["quality_sample"]
    train = reference_sample(ref.con, "train", size, 7321)
    test = reference_sample(ref.con, "test", size, 7322)
    save_json(out / "reference_panel.json", dict(train=train, test=test))
    neighbors = [FP.GetFingerprint(molecule(s)) for s in train]
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    @lru_cache(maxsize=8192)
    def props(key):
        mol = molecule(key)
        fp = FP.GetFingerprint(mol)
        return dict(
            qed=QED.qed(mol),
            mw=Descriptors.MolWt(mol),
            logp=Crippen.MolLogP(mol),
            sa=sascorer.calculateScore(mol),
            pains_alert=int(catalog.HasMatch(mol)),
            nearest_in_train_panel=max(
                DataStructs.BulkTanimotoSimilarity(fp, neighbors)
            ),
        )

    result = []

    def add(name, smiles):
        vals = [props(k) for k in smiles]
        if vals:
            for key in vals[0]:
                v = [x[key] for x in vals]
                result.append(
                    dict(
                        sample=name,
                        metric=key,
                        n=len(v),
                        mean=float(np.mean(v)),
                        p10=float(np.quantile(v, 0.1)),
                        median=float(np.median(v)),
                        p90=float(np.quantile(v, 0.9)),
                    )
                )

    add("heldout_test", test)
    for dest in sorted((folder / "final").glob("*/seed_*")):
        # Uniform reservoir over valid attempts, including repeats.
        rng = np.random.default_rng(7323)
        keys = []
        seen = 0
        for p in sorted(dest.glob("chunk_*.csv")):
            with p.open() as f:
                for r in csv.DictReader(f):
                    if r["valid"] != "True":
                        continue
                    seen += 1
                    if len(keys) < size:
                        keys.append(r["identity"])
                    else:
                        j = int(rng.integers(seen))
                        if j < size:
                            keys[j] = r["identity"]
        add(dest.parent.name + "/" + dest.name, keys)
    csv_file(out / "reference_comparison.csv", result)
