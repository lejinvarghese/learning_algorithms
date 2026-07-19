"""Query-conditioned playlist generation over a song transition graph.

Pipeline:
  1. Simulate a song catalog (audio-ish features + genre) and listening
     sessions, whose consecutive plays form a sparse *directed* transition
     graph (the "watch history" components).
  2. Encode songs inductively with a direction-aware GraphSAGE (features +
     in/out neighborhoods -> song embeddings). New songs need only features.
  3. Encode a free-text query as a *set of intent tokens* (chill / punk /
     sunny / morning ...). Token-level embeddings are kept, not just a
     pooled vector, so the decoder can attend to different intent facets
     at different steps.
  4. Autoregressive decoder: a GRU walks the graph song-by-song. At every
     step it cross-attends over the query tokens, scores all songs against
     the (state ⊕ attended-intent) projection, and adds a learned bias for
     candidates that are graph out-neighbors of the current song.
  5. Train with teacher forcing on (query, playlist) pairs; generate with
     top-k sampling + no-repeat masking.

Run:  python playlist_gnn.py --query "chill punk rock vibes for a sunny morning"
"""

import argparse
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# ---------------------------------------------------------------- data ----

SCALARS = ["energy", "valence", "acousticness", "tempo", "brightness"]
GENRES = ["punk", "rock", "jazz", "electronic", "folk", "hiphop", "classical", "pop"]

GENRE_PROFILE = {
    "punk":       dict(energy=.85, valence=.55, acousticness=.15, tempo=.80, brightness=.60),
    "rock":       dict(energy=.70, valence=.55, acousticness=.30, tempo=.65, brightness=.55),
    "jazz":       dict(energy=.40, valence=.60, acousticness=.75, tempo=.45, brightness=.50),
    "electronic": dict(energy=.75, valence=.60, acousticness=.05, tempo=.75, brightness=.65),
    "folk":       dict(energy=.35, valence=.60, acousticness=.90, tempo=.40, brightness=.60),
    "hiphop":     dict(energy=.65, valence=.55, acousticness=.20, tempo=.60, brightness=.50),
    "classical":  dict(energy=.30, valence=.50, acousticness=.95, tempo=.40, brightness=.50),
    "pop":        dict(energy=.65, valence=.70, acousticness=.35, tempo=.60, brightness=.70),
}

# Mood tokens -> target values on scalar feature dims. In a real system this
# table is replaced by a text encoder; the architecture downstream is unchanged.
INTENTS = {
    "chill":    dict(energy=.20, tempo=.30),
    "mellow":   dict(energy=.25, tempo=.35),
    "hype":     dict(energy=.95, tempo=.85),
    "workout":  dict(energy=.95, tempo=.90),
    "dance":    dict(energy=.85, tempo=.85),
    "sunny":    dict(valence=.90, brightness=.80),
    "happy":    dict(valence=.90),
    "moody":    dict(valence=.15, brightness=.25),
    "sad":      dict(valence=.10),
    "morning":  dict(brightness=.80, acousticness=.55),
    "night":    dict(brightness=.20),
    "acoustic": dict(acousticness=.95),
    "focus":    dict(energy=.35, valence=.50),
}

VOCAB = ["<pad>"] + GENRES + list(INTENTS)
TOK2ID = {t: i for i, t in enumerate(VOCAB)}

ADJ_WORDS = ["Neon", "Velvet", "Rusty", "Golden", "Silent", "Electric", "Paper",
             "Midnight", "Sunlit", "Broken", "Lazy", "Wild", "Static", "Hollow"]
NOUN_WORDS = ["Riot", "Harbor", "Avenue", "Motel", "Garden", "Parade", "Mirror",
              "Skyline", "Circuit", "Meadow", "Anthem", "Arcade", "Postcard", "Tide"]


def clamp(v, lo=0.02, hi=0.98):
    return max(lo, min(hi, v))


def make_songs(n, rng):
    feats, meta = [], []
    for i in range(n):
        g = rng.randrange(len(GENRES))
        prof = GENRE_PROFILE[GENRES[g]]
        vals = {k: clamp(rng.gauss(prof[k], .13)) for k in SCALARS}
        onehot = [0.0] * len(GENRES)
        onehot[g] = 1.0
        feats.append([vals[k] for k in SCALARS] + onehot)
        name = f"{ADJ_WORDS[rng.randrange(len(ADJ_WORDS))]} {NOUN_WORDS[rng.randrange(len(NOUN_WORDS))]} #{i}"
        meta.append(dict(genre=g, name=name, **vals))
    return torch.tensor(feats), meta


def sample_query(rng):
    genres = rng.sample(GENRES, k=rng.choice([1, 1, 2]))
    moods = rng.sample(list(INTENTS), k=rng.randint(1, 3))
    tokens = genres + moods
    rng.shuffle(tokens)
    return tokens


def query_target(tokens):
    """Compose token intents into (preferred genres, target scalar dims)."""
    genres, tgt, cnt = set(), {}, {}
    for t in tokens:
        if t in GENRE_PROFILE:
            genres.add(t)
        else:
            for k, v in INTENTS[t].items():
                tgt[k] = tgt.get(k, 0.0) + v
                cnt[k] = cnt.get(k, 0) + 1
    return genres, {k: v / cnt[k] for k, v in tgt.items()}


def gen_playlist(feats, meta, genres, tgt, rng, length=10, temp=0.3):
    """Ground-truth generator: sample a vibe-coherent, smooth song sequence."""
    n = feats.size(0)
    aff = torch.zeros(n)
    for i, m in enumerate(meta):
        d = sum(abs(m[k] - v) for k, v in tgt.items()) / max(len(tgt), 1)
        gbonus = 1.0 if (not genres or GENRES[m["genre"]] in genres) else 0.0
        aff[i] = -3.0 * d + 2.0 * gbonus
    seq = [torch.multinomial(F.softmax(aff / temp, 0), 1).item()]
    for _ in range(length - 1):
        cur = feats[seq[-1], :5]
        smooth = -(feats[:, :5] - cur).abs().mean(1)          # transition smoothness
        logits = aff + 2.0 * smooth
        logits[torch.tensor(seq)] = -1e9                       # no repeats
        seq.append(torch.multinomial(F.softmax(logits / temp, 0), 1).item())
    return seq


def build_graph(feats, meta, rng, n_sessions=4000, min_count=2):
    """Accumulate transition edges from simulated (query-free) history sessions."""
    counts = Counter()
    for _ in range(n_sessions):
        genres, tgt = query_target(sample_query(rng))
        seq = gen_playlist(feats, meta, genres, tgt, rng)
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    edges = [e for e, c in counts.items() if c >= min_count]
    src, dst = zip(*edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    adj = torch.zeros(feats.size(0), feats.size(0))
    adj[edge_index[0], edge_index[1]] = 1.0
    return edge_index, adj


def make_pairs(feats, meta, rng, n_pairs):
    pairs = []
    for _ in range(n_pairs):
        toks = sample_query(rng)
        genres, tgt = query_target(toks)
        pairs.append((toks, gen_playlist(feats, meta, genres, tgt, rng)))
    return pairs


# --------------------------------------------------------------- model ----

class SongEncoder(nn.Module):
    """Direction-aware GraphSAGE: separate aggregation over in- and out-edges."""

    def __init__(self, in_dim, hid):
        super().__init__()
        self.fwd1, self.bwd1 = SAGEConv(in_dim, hid), SAGEConv(in_dim, hid)
        self.lin1 = nn.Linear(2 * hid, hid)
        self.fwd2, self.bwd2 = SAGEConv(hid, hid), SAGEConv(hid, hid)
        self.lin2 = nn.Linear(2 * hid, hid)

    def forward(self, x, edge_index):
        rev = edge_index.flip(0)
        h = F.relu(self.lin1(torch.cat([self.fwd1(x, edge_index), self.bwd1(x, rev)], -1)))
        h = self.lin2(torch.cat([self.fwd2(h, edge_index), self.bwd2(h, rev)], -1))
        return h


class QueryEncoder(nn.Module):
    """Embeds intent tokens; returns token-level embeddings + a pooled vector."""

    def __init__(self, vocab, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, hid, padding_idx=0)

    def forward(self, tokens):                       # tokens: [B, T] (0 = pad)
        emb = self.emb(tokens)
        mask = tokens != 0                           # [B, T] True = real token
        pooled = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return emb, mask, pooled


class PlaylistDecoder(nn.Module):
    """GRU walker with per-step cross-attention over query intent tokens."""

    def __init__(self, hid, heads=4):
        super().__init__()
        self.gru = nn.GRUCell(hid, hid)
        self.init = nn.Linear(hid, hid)
        self.attn = nn.MultiheadAttention(hid, heads, batch_first=True)
        self.out = nn.Linear(2 * hid, hid)           # asymmetric: state-side projection
        self.start = nn.Parameter(torch.zeros(hid))
        self.edge_bias = nn.Parameter(torch.tensor(1.0))

    def step(self, h, inp, tok_emb, tok_mask, z, adj, last):
        h = self.gru(inp, h)
        c, _ = self.attn(h.unsqueeze(1), tok_emb, tok_emb,
                         key_padding_mask=~tok_mask)
        p = self.out(torch.cat([h, c.squeeze(1)], -1))
        logits = p @ z.t()
        if last is not None:                         # prefer graph out-neighbors
            logits = logits + self.edge_bias * adj[last]
        return h, logits

    def forward(self, z, tok_emb, tok_mask, pooled, adj, seqs):
        B, L = seqs.shape
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(B, -1)
        last, all_logits = None, []
        for t in range(L):
            h, logits = self.step(h, inp, tok_emb, tok_mask, z, adj, last)
            all_logits.append(logits)
            last = seqs[:, t]                        # teacher forcing
            inp = z[last]
        return torch.stack(all_logits, 1)            # [B, L, N]

    @torch.no_grad()
    def generate(self, z, tok_emb, tok_mask, pooled, adj, length=10,
                 temp=0.7, topk=20):
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(1, -1)
        seq, last = [], None
        for _ in range(length):
            h, logits = self.step(h, inp, tok_emb, tok_mask, z, adj, last)
            logits = logits.squeeze(0)
            if seq:
                logits[torch.tensor(seq)] = -1e9     # no repeats
            kth = logits.topk(topk).values[-1]
            logits[logits < kth] = -1e9
            nxt = torch.multinomial(F.softmax(logits / temp, 0), 1).item()
            seq.append(nxt)
            last = torch.tensor([nxt])
            inp = z[last]
        return seq


class PlaylistModel(nn.Module):
    def __init__(self, in_dim, hid, vocab):
        super().__init__()
        self.songs = SongEncoder(in_dim, hid)
        self.query = QueryEncoder(vocab, hid)
        self.dec = PlaylistDecoder(hid)


# ------------------------------------------------------------ training ----

def pad_tokens(tok_lists, max_len=5):
    out = torch.zeros(len(tok_lists), max_len, dtype=torch.long)
    for i, toks in enumerate(tok_lists):
        ids = [TOK2ID[t] for t in toks][:max_len]
        out[i, :len(ids)] = torch.tensor(ids)
    return out


def run_epoch(model, opt, pairs, feats, edge_index, adj, batch=64):
    model.train()
    random.shuffle(pairs)
    total, nb = 0.0, 0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        tokens = pad_tokens([p[0] for p in chunk])
        seqs = torch.tensor([p[1] for p in chunk])
        z = model.songs(feats, edge_index)
        tok_emb, tok_mask, pooled = model.query(tokens)
        logits = model.dec(z, tok_emb, tok_mask, pooled, adj, seqs)
        loss = F.cross_entropy(logits.flatten(0, 1), seqs.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, nb = total + loss.item(), nb + 1
    return total / nb


@torch.no_grad()
def eval_hit_at_k(model, pairs, feats, edge_index, adj, k=10):
    model.eval()
    tokens = pad_tokens([p[0] for p in pairs])
    seqs = torch.tensor([p[1] for p in pairs])
    z = model.songs(feats, edge_index)
    tok_emb, tok_mask, pooled = model.query(tokens)
    logits = model.dec(z, tok_emb, tok_mask, pooled, adj, seqs)
    topk = logits.topk(k, dim=-1).indices                     # [B, L, k]
    return (topk == seqs.unsqueeze(-1)).any(-1).float().mean().item()


@torch.no_grad()
def eval_intent_match(model, queries, feats, meta, edge_index, adj, rng):
    """Generate playlists for held-out queries; measure fit to query target."""
    model.eval()
    z = model.songs(feats, edge_index)
    err_m, err_r, gm, gr = [], [], [], []
    for toks in queries:
        genres, tgt = query_target(toks)
        tokens = pad_tokens([toks])
        tok_emb, tok_mask, pooled = model.query(tokens)
        seq = model.dec.generate(z, tok_emb, tok_mask, pooled, adj)
        rand = rng.sample(range(len(meta)), len(seq))
        for sel, err, g in ((seq, err_m, gm), (rand, err_r, gr)):
            for s in sel:
                if tgt:
                    err.append(sum(abs(meta[s][k] - v) for k, v in tgt.items()) / len(tgt))
                if genres:
                    g.append(float(GENRES[meta[s]["genre"]] in genres))
    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    return avg(err_m), avg(err_r), avg(gm), avg(gr)


# ----------------------------------------------------------- inference ----

def tokenize_query(text):
    words = re.findall(r"[a-z]+", text.lower())
    toks = [w for w in words if w in TOK2ID and w != "<pad>"]
    dropped = [w for w in words if w not in TOK2ID]
    return toks, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="chill punk rock vibes for a sunny morning")
    ap.add_argument("--songs", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    feats, meta = make_songs(args.songs, rng)
    edge_index, adj = build_graph(feats, meta, rng)
    print(f"graph: {args.songs} songs, {edge_index.size(1)} directed edges "
          f"(density {edge_index.size(1) / args.songs**2:.3%})")

    pairs = make_pairs(feats, meta, rng, 3200)
    train_pairs, test_pairs = pairs[:3000], pairs[3000:]

    model = PlaylistModel(feats.size(1), 64, len(VOCAB))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        loss = run_epoch(model, opt, train_pairs, feats, edge_index, adj)
        hit = eval_hit_at_k(model, test_pairs, feats, edge_index, adj)
        print(f"epoch {ep + 1}: loss {loss:.3f}  held-out hit@10 {hit:.1%} "
              f"(random ≈ {10 / args.songs:.1%})")

    em, er, gm, gr = eval_intent_match(
        model, [p[0] for p in test_pairs[:60]], feats, meta, edge_index, adj, rng)
    print(f"\nintent fit on held-out queries (lower = better): "
          f"model {em:.3f} vs random {er:.3f}")
    print(f"genre match: model {gm:.1%} vs random {gr:.1%}")

    toks, dropped = tokenize_query(args.query)
    print(f"\nquery: {args.query!r}")
    print(f"intent tokens: {toks}   (ignored filler: {dropped})")
    z = model.songs(feats, edge_index)
    tok_emb, tok_mask, pooled = model.query(pad_tokens([toks]))
    seq = model.dec.generate(z, tok_emb, tok_mask, pooled, adj)

    print(f"\n{'#':>2}  {'song':<24} {'genre':<11} energy valence acoustic bright  edge")
    for i, s in enumerate(seq):
        m = meta[s]
        on_edge = "  ·" if i == 0 else ("  ✓" if adj[seq[i - 1], s] > 0 else "  ✗")
        print(f"{i + 1:>2}  {m['name']:<24} {GENRES[m['genre']]:<11} "
              f"{m['energy']:.2f}   {m['valence']:.2f}    {m['acousticness']:.2f}     "
              f"{m['brightness']:.2f} {on_edge}")


if __name__ == "__main__":
    main()
