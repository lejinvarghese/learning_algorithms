"""Benchmark one training step (forward+backward) on cpu vs mps.

Run:  .venv/bin/python bench_device.py
"""

import json
import random
import time

import torch

from train_real import SEQ_LEN, build_graph, load_data
from train_text import (TextPlaylistModel, build_lex_vocab, build_text_cache,
                        encode_batch)
import torch.nn.functional as F


def bench(dev_name, feats, edge_index, nbrs, cache, n2i, w2i, lexmap, chunk,
          hidden, steps=8):
    dev = torch.device(dev_name)
    feats, edge_index = feats.to(dev), edge_index.to(dev)
    nbrs = [t.to(dev) for t in nbrs]
    torch.manual_seed(0)
    model = TextPlaylistModel(feats.size(1), cache["name_emb"].size(1),
                              hidden, len(lexmap["words"])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    emb, mask, lex, seqs = (t.to(dev) for t in encode_batch(chunk, cache, n2i, w2i, lexmap))
    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        z = model.songs(feats, edge_index)
        tok, tmask, pooled = model.query(emb, mask, lex)
        logits, cov = model.dec(z, tok, tmask, pooled, nbrs, seqs)
        loss = F.cross_entropy(logits.flatten(0, 1), seqs.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        if i >= 2:                                  # skip warmup
            times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main():
    random.seed(7)
    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]
    random.shuffle(pairs)
    train = pairs[len(pairs) // 10:]
    edge_index, nbrs = build_graph(train, len(songs))
    cache = build_text_cache(sorted({p["name"] for p in pairs}))
    n2i = {n: i for i, n in enumerate(cache["names"])}
    w2i = {w: i for i, w in enumerate(cache["words"])}
    lex_vocab = build_lex_vocab(train)
    lexmap = {"words": {w: i + 1 for i, w in enumerate(lex_vocab)}, "phrases": {}}
    chunk = train[:64]

    for hidden in (64, 128):
        row = f"hidden={hidden:<4}"
        for dev in ("cpu", "mps"):
            try:
                t = bench(dev, feats, edge_index, nbrs, cache, n2i, w2i,
                          lexmap, chunk, hidden)
                row += f"  {dev}: {t*1000:7.1f} ms/step"
            except Exception as e:
                row += f"  {dev}: failed ({type(e).__name__})"
        print(row)


if __name__ == "__main__":
    main()
