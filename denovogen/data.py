import collections
import hashlib
import json
import sqlite3

import numpy as np
import rdkit
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .chem import BOS, EOS, PAD, SPECIAL, UNK, identity, molecule, scaffold, tokenize
from .common import digest, read, save_json

SPLITS = ("train", "val", "test")


def rows(path):
    with open(path, "rb") as f:
        for line in f:
            parts = line.strip().split()
            if (
                not parts
                or parts[0].lower() in (b"smiles", b"canonical_smiles")
                or parts[0].startswith(b"#")
            ):
                continue
            yield parts


def inventory(path):
    files = sorted(path.glob("*.smi")) if path.is_dir() else [path]
    if not files or any(not f.is_file() for f in files):
        raise ValueError(f"No .smi files at {path}")
    out = []
    for f in files:
        h = hashlib.sha256()
        n = 0
        with f.open("rb") as stream:
            for line in stream:
                h.update(line)
                p = line.split(maxsplit=1)
                n += bool(
                    p
                    and p[0].lower() not in (b"smiles", b"canonical_smiles")
                    and not p[0].startswith(b"#")
                )
        out.append(dict(file=f.name, sha256=h.hexdigest(), rows=n))
    return files, out


def split_name(key, seed):
    value = hashlib.blake2b(f"{seed}/{key}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(value, "big") % 100
    return "train" if bucket < 80 else ("val" if bucket < 90 else "test")


def allocation(counts, limit):
    total = sum(counts)
    if not limit or limit >= total:
        return counts
    q = [n * limit // total for n in counts]
    for i in sorted(range(len(q)), key=lambda i: -(counts[i] * limit % total))[
        : limit - sum(q)
    ]:
        q[i] += 1
    return q


def connect(path):
    return sqlite3.connect(path)


class Reference:
    def __init__(self, folder):
        self.manifest = read(folder / "manifest.json")
        self.con = sqlite3.connect(
            (folder / "ids.sqlite").resolve().as_uri() + "?mode=ro", uri=True
        )
        if self.manifest["split_counts"]["train"] < 1:
            raise ValueError("Missing train reference")
        self.complete = self.manifest["complete_source_reference"]

    def lookup(self, key, full=False):
        hit = self.con.execute("SELECT part FROM items WHERE key=?", (key,)).fetchone()
        source = None
        if full and self.complete:
            source = self.con.execute(
                "SELECT origin FROM sources WHERE key=?", (key,)
            ).fetchone()
        return dict(
            train=bool(hit and hit[0] == "train"),
            source_novel=(source is None) if full and self.complete else None,
            source_id=source[0] if source else None,
        )

    def close(self):
        self.con.close()


def ingest(con, files, spec, cfg, folder):
    quotas = allocation([a["rows"] for a in spec], cfg["max_rows"])
    complete = cfg["full_reference"] or quotas == [a["rows"] for a in spec]
    for i, (path, meta, quota) in enumerate(zip(files, spec, quotas)):
        if con.execute("SELECT 1 FROM progress WHERE name=?", (path.name,)).fetchone():
            continue
        rng = np.random.default_rng(cfg["data_seed"] + i)
        chosen = (
            None
            if quota == meta["rows"]
            else set(rng.choice(meta["rows"], quota, replace=False).tolist())
        )
        counts = collections.Counter()
        rejects = []
        with con:
            for j, fields in enumerate(rows(path)):
                use = chosen is None or j in chosen
                if not use and not complete:
                    continue
                s = fields[0].decode()
                mol = molecule(s)
                if mol is None:
                    counts["parse_rejected"] += 1
                    if len(rejects) < 16:
                        rejects.append([j, s, "parse"])
                    continue
                key = identity(mol)
                origin = fields[1].decode() if len(fields) > 1 else f"{path.name}/{j}"
                con.execute("INSERT OR IGNORE INTO sources VALUES (?,?)", (key, origin))
                if not use:
                    continue
                counts["selected_valid_rows"] += 1
                try:
                    tokens = tokenize(key)
                except ValueError:
                    counts["token_rejected"] += 1
                    if len(rejects) < 16:
                        rejects.append([j, s, "token"])
                    continue
                if len(tokens) + 1 > cfg["max_steps"]:
                    counts["length_rejected"] += 1
                    continue
                group = key
                if cfg["split"] == "scaffold":
                    group = scaffold(mol) or "<ACYCLIC>"
                part = split_name(group, cfg["data_seed"])
                con.execute(
                    "INSERT OR IGNORE INTO items VALUES (?,?,?)", (key, part, group)
                )
            con.execute(
                "INSERT INTO progress VALUES (?,?,?)",
                (path.name, json.dumps(counts), json.dumps(rejects)),
            )
        print(f"Indexed shard {i + 1}/{len(files)}: {path.name}", flush=True)
    return complete


def encode_views(key, vocab, split, seed):
    canonical = tokenize(key)
    tokens = canonical
    if split == "train":
        random_seed = (
            int.from_bytes(
                hashlib.blake2b(f"{seed}/{key}".encode(), digest_size=4).digest(), "big"
            )
            % 2147483646
            + 1
        )
        for view in Chem.MolToRandomSmilesVect(
            molecule(key), 4, randomSeed=random_seed
        ):
            try:
                other = tokenize(view)
            except ValueError:
                continue
            if (
                view != key
                and len(other) == len(canonical)
                and all(t in vocab for t in other)
            ):
                tokens = other
                break

    def encode(items):
        return np.asarray([BOS, *(vocab.get(t, UNK) for t in items), EOS], dtype="<u2")

    return encode(canonical), encode(tokens), tokens != canonical


def prepare(source, folder, cfg):
    folder.mkdir(parents=True, exist_ok=True)
    files, spec = inventory(source)
    keys = ("max_rows", "full_reference", "max_steps", "data_seed", "split")
    request = dict(source=spec, rdkit=rdkit.__version__, **{k: cfg[k] for k in keys})
    stamp = digest(request)
    if (folder / "request.json").exists() and digest(
        read(folder / "request.json")
    ) != stamp:
        raise ValueError(
            "Data or preparation settings changed. Choose a fresh run directory."
        )
    if (folder / "manifest.json").exists():
        return read(folder / "manifest.json")
    save_json(folder / "request.json", request)
    con = connect(folder / "ids.sqlite")
    con.executescript("""
      CREATE TABLE IF NOT EXISTS sources(key TEXT PRIMARY KEY, origin TEXT) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS items(key TEXT PRIMARY KEY, part TEXT, grp TEXT) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS progress(name TEXT PRIMARY KEY, counts TEXT, rejects TEXT);
    """)
    complete = ingest(con, files, spec, cfg, folder)
    con.execute("CREATE INDEX IF NOT EXISTS split_items ON items(part,key)")
    counts = {
        p: con.execute("SELECT count(*) FROM items WHERE part=?", (p,)).fetchone()[0]
        for p in SPLITS
    }
    if min(counts.values()) == 0:
        raise ValueError(f"Empty split {counts}. Use more data or another split seed.")
    vocab = set()
    for (key,) in con.execute("SELECT key FROM items WHERE part='train'"):
        vocab.update(tokenize(key))
    names = SPECIAL + sorted(vocab)
    if len(names) > 65536:
        raise ValueError("Vocabulary exceeds uint16 token cache")
    idx = {k: i for i, k in enumerate(names)}
    augmented = 0
    for p in SPLITS:
        offsets = np.lib.format.open_memmap(
            folder / f"{p}.offsets.npy", mode="w+", dtype="<i8", shape=(counts[p] + 1,)
        )
        offsets[0] = 0
        with (folder / f"{p}.tokens").open("wb") as a:
            alt = (folder / "randomized.tokens").open("wb") if p == "train" else None
            try:
                for i, (key,) in enumerate(
                    con.execute("SELECT key FROM items WHERE part=? ORDER BY key", (p,))
                ):
                    seq, view, changed = encode_views(key, idx, p, cfg["data_seed"])
                    a.write(seq.tobytes())
                    if alt:
                        alt.write(view.tobytes())
                        augmented += changed
                    offsets[i + 1] = offsets[i] + len(seq)
            finally:
                if alt:
                    alt.close()
        offsets.flush()
        print(f"Cached {counts[p]:,} {p} identities", flush=True)
    totals = collections.Counter()
    examples = []
    for name, counts_json, rejected in con.execute(
        "SELECT * FROM progress ORDER BY name"
    ):
        totals.update(json.loads(counts_json))
        examples.extend([[name, *r] for r in json.loads(rejected)])
    manifest = dict(
        request_id=stamp,
        source_files=spec,
        split_counts=counts,
        vocab=names,
        source_raw_rows=sum(m["rows"] for m in spec),
        selected_rows=sum(allocation([m["rows"] for m in spec], cfg["max_rows"])),
        reference_identities=con.execute("SELECT count(*) FROM sources").fetchone()[0],
        complete_source_reference=complete,
        changed_training_views=augmented,
        counters=dict(totals),
        group_counts={
            p: con.execute(
                "SELECT count(DISTINCT grp) FROM items WHERE part=?", (p,)
            ).fetchone()[0]
            for p in SPLITS
        },
        acyclic_group_size=con.execute(
            "SELECT count(*) FROM items WHERE grp='<ACYCLIC>'"
        ).fetchone()[0],
    )
    con.close()
    save_json(folder / "rejections.json", examples)
    save_json(folder / "manifest.json", manifest)
    return manifest


class Sequences(Dataset):
    def __init__(self, folder, part, variant="canonical"):
        self.offsets = np.load(folder / f"{part}.offsets.npy", mmap_mode="r")
        name = "randomized" if part == "train" and variant == "randomized" else part
        self.tokens = np.memmap(folder / f"{name}.tokens", dtype="<u2", mode="r")

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return torch.from_numpy(
            self.tokens[self.offsets[i] : self.offsets[i + 1]].astype("int64")
        )


def batches(data, cfg, seed, shuffle):
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(seed)
    ix = rng.permutation(len(data)) if shuffle else np.arange(len(data))

    def groups():
        for start in range(0, len(ix), 1024):
            block = ix[start : start + 1024]
            lengths = data.offsets[block + 1] - data.offsets[block]
            order = block[np.argsort(lengths, kind="stable")]
            group = []
            for j in order:
                length = int(data.offsets[j + 1] - data.offsets[j])
                if group and (
                    len(group) >= cfg["batch_size"]
                    or (len(group) + 1) * length > cfg["batch_tokens"]
                ):
                    yield group
                    group = []
                group.append(int(j))
            if group:
                yield group

    return DataLoader(
        data, batch_sampler=groups(), collate_fn=collate, num_workers=cfg["workers"]
    )


def collate(items):
    return pad_sequence(items, batch_first=True, padding_value=PAD)
