"""Train the playlist model with frozen ModernBERT text embeddings as the
query encoder (Alibaba-NLP/gte-modernbert-base via sentence-transformers).

Query tokens = [whole-name sentence embedding] + per-word dense embeddings,
optionally + learned exact-match lexical token embeddings (--lexical), all
feeding the same cross-attention decoder.

--coverage W adds an attention-coverage loss: every real query token must
accumulate attention mass across the decode steps, penalizing "unmet
intents" (dominant-facet capture).

Run:  .venv/bin/python train_text.py --lexical [--coverage 1.0]
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from playlist_gnn import SongEncoder
from train_real import SEQ_LEN, SparseNeighborDecoder, build_graph, load_data

EMB_MODEL = "Alibaba-NLP/gte-modernbert-base"
MAX_WORDS = 7            # dense tokens = 1 sentence emb + up to 7 word embs
MAX_LEX = 8              # exact-match lexical/genre anchors per query
COV_TAU = 0.4            # min attention mass a real token should accumulate
ARTIST_DIM = 32
CACHE = "data/text_cache.pt"
TITLE_CACHE = "data/title_cache.pt"
GENRE_CACHE = "data/genre_cache.pt"

LEX_STOP = set("""the and for with you your our from into this that all not out
off are was were has have had can will its it's dont don't когда les des los
las der die und mit von songs song music playlist playlists list mix tracks
various artists artist album albums best top new old good great favourite
favourites favorite favorites stuff misc random liked starred radio play
spotify shazam soundrop iphone library""".split())


def words_of(name):
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9]+", name.lower()):
        if len(w) >= 2 and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)


def build_text_cache(names):
    if os.path.exists(CACHE):
        return torch.load(CACHE)
    enc = get_encoder()
    words = sorted({w for n in names for w in words_of(n)})
    print(f"encoding {len(names)} names + {len(words)} words with {EMB_MODEL} ...")
    name_emb = torch.tensor(enc.encode(names, batch_size=128, normalize_embeddings=True))
    word_emb = torch.tensor(enc.encode(words, batch_size=256, normalize_embeddings=True))
    cache = {"names": names, "name_emb": name_emb, "words": words, "word_emb": word_emb}
    torch.save(cache, CACHE)
    return cache


def content_features(songs):
    """Raw catalog columns unused by exps 0-6: key, mode, time signature,
    duration, explicit — the beat/rhythm/structure side of the catalog."""
    n = len(songs)
    key = torch.zeros(n, 12)
    k = torch.tensor(songs["key"].fillna(0).astype(int).clip(0, 11).values)
    key[torch.arange(n), k] = 1.0
    dur = torch.log1p(torch.tensor(songs["duration_ms"].fillna(2e5).values,
                                   dtype=torch.float)) / 14.0
    extra = torch.stack([
        torch.tensor(songs["mode"].fillna(0).values, dtype=torch.float),
        torch.tensor(songs["time_signature"].fillna(4).values,
                     dtype=torch.float).clip(0, 5) / 5.0,
        dur.clip(0, 1),
        torch.tensor(songs["explicit"].fillna(False).astype(float).values,
                     dtype=torch.float),
    ], dim=1)
    return torch.cat([key, extra], 1)


KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def song_profile(r):
    """Serialize a full song record into one text profile so every feature —
    title, artist, album, genre, and binned continuous audio features — shares
    ModernBERT's semantic space with the queries. Numbers become qualitative
    words (transformers handle 'high energy' far better than '0.83')."""
    def level(v, lo, hi, low_w, high_w):
        return low_w if v < lo else (high_w if v > hi else None)
    bits = [f"{r['track_name']} by {str(r['artists']).split(';')[0]}.",
            f"Album: {r['album_name']}.", f"Genre: {r['track_genre']}."]
    quals = [q for q in (
        level(r["energy"], .3, .7, "low energy", "high energy"),
        level(r["valence"], .3, .7, "sad mood", "happy mood"),
        level(r["danceability"], .3, .7, "not danceable", "very danceable"),
        level(r["acousticness"], .3, .7, None, "acoustic"),
        level(r["instrumentalness"], .5, .5, None, "instrumental"),
        level(r["speechiness"], .33, .33, None, "spoken vocals"),
        level(r["liveness"], .8, .8, None, "live recording"),
        level(r["tempo"], 90, 140, "slow tempo", "fast tempo"),
    ) if q]
    if quals:
        bits.append(", ".join(quals) + ".")
    try:
        bits.append(f"Key {KEYS[int(r['key'])]} {'major' if r['mode'] else 'minor'}.")
    except (ValueError, IndexError, TypeError):
        pass
    return " ".join(bits)


def build_title_cache(songs):
    """Frozen ModernBERT embeddings of full song text profiles."""
    if os.path.exists(TITLE_CACHE):
        return torch.load(TITLE_CACHE)
    enc = get_encoder()
    texts = [song_profile(songs.iloc[i]) for i in range(len(songs))]
    print(f"encoding {len(texts)} song profiles, e.g.: {texts[0][:120]!r}")
    emb = torch.tensor(enc.encode(texts, batch_size=128, normalize_embeddings=True))
    torch.save(emb, TITLE_CACHE)
    return emb


def build_lex_vocab(train_pairs, min_freq=40, cap=300):
    cnt = Counter(w for p in train_pairs for w in set(words_of(p["name"])))
    return [w for w, c in cnt.most_common(cap) if c >= min_freq]


def build_lex_v2(train_pairs, cache, songs, min_freq=10, cap=1500):
    """Expanded anchor vocab: frequent name words (stopword-pruned) plus all
    catalog genre labels as phrase anchors ('synth pop' -> genre synth-pop).
    Embeddings are initialized from ModernBERT vectors and fine-tuned through
    the shared dense projection — rare anchors start meaningful. Also
    returns song_genre_ids (song -> genre index) and anchor_genre (anchor
    id -> genre index) for the decoder's direct genre-bias channel."""
    cnt = Counter(w for p in train_pairs for w in set(words_of(p["name"])))
    words = [w for w, c in cnt.most_common() if c >= min_freq
             and w not in LEX_STOP][:cap]
    genres = sorted(set(songs["track_genre"].dropna()))
    g2idx = {g: i for i, g in enumerate(genres)}
    song_genre_ids = torch.tensor(
        [g2idx.get(g, 0) for g in songs["track_genre"].fillna(genres[0])])
    gnorm = [" ".join(re.findall(r"[a-z0-9]+", g.lower())) for g in genres]
    if os.path.exists(GENRE_CACHE):
        genre_emb = torch.load(GENRE_CACHE)
    else:
        genre_emb = torch.tensor(get_encoder().encode(
            [f"{g} music" for g in gnorm], normalize_embeddings=True))
        torch.save(genre_emb, GENRE_CACHE)
    wc = {w: i for i, w in enumerate(cache["words"])}
    dim = cache["word_emb"].size(1)
    init = torch.zeros(1 + len(words) + len(genres), dim)
    word2id, phrase2id, anchor_genre, vocab = {}, {}, {}, []
    for j, w in enumerate(words):
        init[1 + j] = cache["word_emb"][wc[w]]
        word2id[w] = 1 + j
        vocab.append(w)
    for k, g in enumerate(gnorm):
        aid = 1 + len(words) + k
        init[aid] = genre_emb[k]
        phrase2id[g] = aid
        anchor_genre[aid] = k
        vocab.append(f"genre:{genres[k]}")
    return vocab, word2id, phrase2id, init, song_genre_ids, anchor_genre, len(genres)


def find_anchors(name, lexmap):
    """Anchor ids for a query: genre-phrase matches first, then word matches."""
    if not lexmap:
        return []
    ws = words_of(name)
    text = " " + " ".join(ws) + " "
    ids = [pid for ph, pid in lexmap.get("phrases", {}).items()
           if f" {ph} " in text]
    ids += [lexmap["words"][w] for w in ws if w in lexmap["words"]]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:MAX_LEX]


def build_co_graph(pairs, n, window=4, min_count=2, topk=64):
    """Union graph: all adjacency edges (full bias weight 1.0) plus
    co-membership edges — songs within `window` positions in the same
    playlist, symmetric, weighted by GLOBALLY log-normalized co-count.
    Uses FULL playlists. Playlist order in this dump is alphabetical noise,
    so adjacency-only edges undersample the true co-listen relation; but
    adjacency pairs are exactly what the next-song metric asks for, so they
    keep maximal weight."""
    import math
    co, adj = Counter(), Counter()
    for p in pairs:
        ids = p["track_ids"]
        for a, b in zip(ids, ids[1:]):
            if a != b:
                adj[(a, b)] += 1
        for i, a in enumerate(ids):
            for b in ids[i + 1:i + 1 + window]:
                if a != b:
                    co[(a, b)] += 1
                    co[(b, a)] += 1
    cmax = math.log1p(max(co.values()))
    wts = {}
    for e, c in co.items():
        if c >= min_count:
            wts[e] = math.log1p(c) / cmax
    for e, c in adj.items():
        if c >= min_count:
            wts[e] = 1.0
    per = [[] for _ in range(n)]
    for (a, b), w in wts.items():
        per[a].append((w, b))
    src, dst, nbrs = [], [], []
    for a, lst in enumerate(per):
        lst.sort(reverse=True)
        lst = lst[:topk]
        idx = torch.tensor([b for _, b in lst], dtype=torch.long)
        w = torch.tensor([wt for wt, _ in lst], dtype=torch.float)
        nbrs.append((idx, w))
        src += [a] * len(lst)
        dst += [b for _, b in lst]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index, nbrs


class DeepSongEncoder(nn.Module):
    """N direction-aware SAGE layers with residuals + LayerNorm (prevents
    oversmoothing at depth), followed by a 2-layer MLP head."""

    def __init__(self, in_dim, hid, layers=3):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.inp = nn.Linear(in_dim, hid)
        self.fwd = nn.ModuleList(SAGEConv(hid, hid) for _ in range(layers))
        self.bwd = nn.ModuleList(SAGEConv(hid, hid) for _ in range(layers))
        self.mix = nn.ModuleList(nn.Linear(2 * hid, hid) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hid) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(),
                                  nn.Linear(hid, hid))

    def forward(self, x, edge_index):
        rev = edge_index.flip(0)
        h = F.relu(self.inp(x))
        for f, b, m, n in zip(self.fwd, self.bwd, self.mix, self.norm):
            u = m(torch.cat([f(h, edge_index), b(h, rev)], -1))
            h = n(h + F.relu(u))
        return self.head(h)


class TextQueryEncoder(nn.Module):
    """Dense (frozen, projected) tokens + learned lexical anchor tokens.

    v1 anchors: small random-init table at hidden size. v2 (lex_init given):
    full-width table initialized from ModernBERT vectors, fine-tuned, and
    passed through the SAME projection as the dense tokens."""

    def __init__(self, in_dim, hid, lex_vocab=0, lex_init=None):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.v2 = lex_init is not None
        if self.v2:
            self.lex = nn.Embedding.from_pretrained(lex_init.clone(),
                                                    freeze=False, padding_idx=0)
        else:
            self.lex = nn.Embedding(lex_vocab + 1, hid, padding_idx=0) if lex_vocab else None

    def forward(self, emb, mask, lex_ids=None):
        tok = self.proj(emb)
        if self.lex is not None and lex_ids is not None:
            ltok = self.lex(lex_ids)
            if self.v2:
                ltok = self.proj(ltok)
            tok = torch.cat([tok, ltok], 1)
            mask = torch.cat([mask, lex_ids != 0], 1)
        pooled = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return tok, mask, pooled


class HybridDecoder(SparseNeighborDecoder):
    """Adds attention-coverage tracking, learned coverage steering, optional
    MMR-diverse generation, and an optional direct genre-match bias.

    Steering (--steer): at each step the under-attended query tokens form a
    deficit-weighted context vector fed into the GRU input — the model is
    TRAINED to move toward unserved intents, instead of policing them only
    via the loss.

    Genre bias (--genrebias): a direct, learned additive bonus for songs
    matching an anchored query genre — the same pattern as edge_bias for
    graph neighbors. Rationale: a single genre anchor token competing inside
    cross-attention against a whole-sentence embedding + up to 15 other
    tokens can get attended (satisfying coverage) without ever winning
    enough to reorder the output; an explicit channel guarantees genre
    actually moves the ranking."""

    def __init__(self, hid, steer=False, genrebias=False):
        super().__init__(hid)
        self.steer_lin = nn.Linear(hid, hid) if steer else None
        self.genre_bias = nn.Parameter(torch.tensor(1.0)) if genrebias else None

    def step(self, h, inp, tok_emb, tok_mask, z, nbrs, last, cov=None,
             song_genre_ids=None, qgenre=None):
        if self.steer_lin is not None and cov is not None:
            deficit = F.relu(COV_TAU - cov) * tok_mask.float()
            w = deficit / deficit.sum(1, keepdim=True).clamp(min=1e-6)
            inp = inp + self.steer_lin((w.unsqueeze(1) @ tok_emb).squeeze(1))
        h = self.gru(inp, h)
        c, attn = self.attn(h.unsqueeze(1), tok_emb, tok_emb,
                            key_padding_mask=~tok_mask, need_weights=True)
        p = self.out(torch.cat([h, c.squeeze(1)], -1))
        logits = p @ z.t()
        if last is not None:
            bonus = torch.zeros_like(logits)
            for b, l in enumerate(last.tolist()):
                nb = nbrs[l]
                if isinstance(nb, tuple):          # weighted co-membership graph
                    idx, w = nb
                    if idx.numel():
                        bonus[b, idx] = w
                elif nb.numel():
                    bonus[b, nb] = 1.0
            logits = logits + self.edge_bias * bonus
        if self.genre_bias is not None and song_genre_ids is not None and qgenre is not None:
            hit = qgenre.gather(1, song_genre_ids.unsqueeze(0).expand(qgenre.size(0), -1))
            logits = logits + self.genre_bias * hit
        return h, logits, attn.squeeze(1)          # attn: [B, T]

    def forward(self, z, tok_emb, tok_mask, pooled, nbrs, seqs,
                song_genre_ids=None, qgenre=None):
        B, L = seqs.shape
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(B, -1)
        last, all_logits, cov = None, [], torch.zeros(B, tok_emb.size(1),
                                                      device=tok_emb.device)
        for t in range(L):
            h, logits, attn = self.step(h, inp, tok_emb, tok_mask, z, nbrs,
                                        last, cov, song_genre_ids, qgenre)
            all_logits.append(logits)
            cov = cov + attn
            last = seqs[:, t]
            inp = z[last]
        return torch.stack(all_logits, 1), cov

    @torch.no_grad()
    def generate(self, z, tok_emb, tok_mask, pooled, nbrs, length=SEQ_LEN,
                 temp=0.7, topk=20, mmr=0.0, artists=None, max_artist=0,
                 song_genre_ids=None, qgenre=None):
        zn = F.normalize(z, dim=-1)
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(1, -1)
        seq, last = [], None
        cov = torch.zeros(1, tok_emb.size(1), device=tok_emb.device)
        for _ in range(length):
            h, logits, attn = self.step(h, inp, tok_emb, tok_mask, z, nbrs,
                                        last, cov, song_genre_ids, qgenre)
            cov = cov + attn
            logits = logits.squeeze(0)
            if seq:
                picked = torch.tensor(seq, device=z.device)
                logits[picked] = -1e9
                if artists is not None and max_artist > 0:
                    # DJ rule: no artist more than max_artist times
                    cnt = Counter(artists[s].item() for s in seq)
                    over = [a for a, c in cnt.items() if c >= max_artist]
                    if over:
                        logits[torch.isin(artists, torch.tensor(over, device=z.device))] = -1e9
                if mmr > 0:                        # penalize similarity to picks
                    sim = (zn @ zn[picked].t()).max(dim=1).values
                    scale = logits.topk(topk).values.std().clamp(min=1e-3)
                    logits = logits - mmr * scale * sim
            if temp <= 0:                          # greedy: deterministic
                nxt = logits.argmax().item()
            else:
                kth = logits.topk(topk).values[-1]
                logits[logits < kth] = -1e9
                nxt = torch.multinomial(F.softmax(logits / temp, 0), 1).item()
            seq.append(nxt)
            last = torch.tensor([nxt], device=z.device)
            inp = z[last]
        return seq


class TextPlaylistModel(nn.Module):
    def __init__(self, feat_dim, emb_dim, hid, lex_vocab=0, n_artists=0,
                 layers=0, lex_init=None, steer=False, genrebias=False):
        super().__init__()
        self.artist_emb = nn.Embedding(n_artists, ARTIST_DIM) if n_artists else None
        in_dim = feat_dim + (ARTIST_DIM if n_artists else 0)
        self.songs = (DeepSongEncoder(in_dim, hid, layers) if layers
                      else SongEncoder(in_dim, hid))
        self.query = TextQueryEncoder(emb_dim, hid, lex_vocab, lex_init)
        self.dec = HybridDecoder(hid, steer, genrebias)

    def song_z(self, feats, artist_ids, edge_index):
        x = feats
        if self.artist_emb is not None:
            x = torch.cat([x, self.artist_emb(artist_ids)], -1)
        return self.songs(x, edge_index)


def query_embs_for(name, cache, n2i, w2i, lexmap):
    D = cache["name_emb"].size(1)
    emb = torch.zeros(1 + MAX_WORDS, D)
    mask = torch.zeros(1 + MAX_WORDS, dtype=torch.bool)
    emb[0] = cache["name_emb"][n2i[name]]
    mask[0] = True
    ws = words_of(name)
    for j, w in enumerate(ws[:MAX_WORDS]):
        emb[1 + j] = cache["word_emb"][w2i[w]]
        mask[1 + j] = True
    lex = torch.zeros(MAX_LEX, dtype=torch.long)
    hits = find_anchors(name, lexmap)
    if hits:
        lex[:len(hits)] = torch.tensor(hits)
    return emb, mask, lex


def encode_batch(chunk, cache, n2i, w2i, lexmap):
    embs, masks, lexs = zip(*(query_embs_for(p["name"], cache, n2i, w2i, lexmap)
                              for p in chunk))
    seqs = torch.tensor([p["track_ids"][:SEQ_LEN] for p in chunk])
    lexs = torch.stack(lexs)
    qgenre = None
    ag, ng = lexmap.get("anchor_genre"), lexmap.get("n_genres")
    if ag:
        qgenre = torch.zeros(len(chunk), ng)
        for b in range(len(chunk)):
            for aid in lexs[b].tolist():
                if aid in ag:
                    qgenre[b, ag[aid]] = 1.0
    return torch.stack(embs), torch.stack(masks), lexs, seqs, qgenre


def coverage_loss(cov, mask):
    """Penalty for real query tokens left under-attended after all steps."""
    unmet = F.relu(COV_TAU - cov) * mask
    return unmet.sum() / mask.sum().clamp(min=1)


def listwise_loss(logits, seqs):
    """ListNet-style position-discounted set loss (approx-NDCG family).

    At step t the target is a distribution over ALL remaining playlist songs,
    with DCG gains 1/log2(2+offset): the true next song gets the largest gain,
    later songs smaller ones. Optimizes presence of the whole upcoming set,
    graded by position, instead of a single hard next-item label."""
    B, L, N = logits.shape
    loss = 0.0
    for t in range(L):
        rest = seqs[:, t:]                                     # [B, R]
        gains = 1.0 / torch.log2(torch.arange(rest.size(1), device=logits.device)
                                 .float() + 2.0)
        y = torch.zeros(B, N, device=logits.device)
        y.scatter_(1, rest, gains.expand(B, -1))
        y = y / y.sum(1, keepdim=True)
        loss = loss - (y * F.log_softmax(logits[:, t], -1)).sum(1).mean()
    return loss / L


@torch.no_grad()
def ndcg_at_10(logits, seqs):
    """NDCG@10 per step: relevance of a predicted song = its DCG gain in the
    remaining true set; discounted by predicted rank; vs the ideal ordering."""
    B, L, _ = logits.shape
    top10 = logits.topk(10, -1).indices                        # [B, L, 10]
    rank_disc = 1.0 / torch.log2(torch.arange(10, device=logits.device).float() + 2.0)
    total = 0.0
    for t in range(L):
        rest = seqs[:, t:]                                     # [B, R]
        gains = 1.0 / torch.log2(torch.arange(rest.size(1), device=logits.device)
                                 .float() + 2.0)
        match = top10[:, t].unsqueeze(-1) == rest.unsqueeze(1)  # [B, 10, R]
        rel = (match * gains.view(1, 1, -1)).amax(-1)          # [B, 10]
        dcg = (rel * rank_disc).sum(-1)
        ideal = (gains[:10] * rank_disc[:gains[:10].size(0)]).sum()
        total = total + (dcg / ideal).mean().item()
    return total / L


def run_epoch(model, opt, pairs, feats, edge_index, nbrs, cache, n2i, w2i,
              lex2id, cov_w, artist_ids=None, listwise=0.0, log_prior=None,
              shuffle_p=0.0, song_genre_ids=None, batch=64):
    model.train()
    dev = feats.device
    random.shuffle(pairs)
    total, nb = 0.0, 0
    for i in range(0, len(pairs), batch):
        emb, mask, lex, seqs, qgenre = encode_batch(pairs[i:i + batch], cache, n2i, w2i, lex2id)
        emb, mask, lex, seqs = emb.to(dev), mask.to(dev), lex.to(dev), seqs.to(dev)
        if qgenre is not None:
            qgenre = qgenre.to(dev)
        if shuffle_p > 0:
            # RecSys'18 finding: random playlist subsets condition better than
            # sequential prefixes; playlist order here is alphabetical noise
            # anyway, so shuffling regularizes the adjacency-edge shortcut.
            for b in range(seqs.size(0)):
                if random.random() < shuffle_p:
                    seqs[b] = seqs[b][torch.randperm(seqs.size(1), device=dev)]
        z = model.song_z(feats, artist_ids, edge_index)
        tok, tmask, pooled = model.query(emb, mask, lex)
        logits, cov = model.dec(z, tok, tmask, pooled, nbrs, seqs,
                                song_genre_ids, qgenre)
        # logit adjustment (Menon et al. 2021): add the popularity log-prior
        # during training only, so the model learns popularity-residual scores
        # and rare songs get larger margins; inference uses raw logits.
        train_logits = logits + log_prior if log_prior is not None else logits
        ce = F.cross_entropy(train_logits.flatten(0, 1), seqs.flatten())
        loss = (1 - listwise) * ce + listwise * listwise_loss(logits, seqs) \
            if listwise > 0 else ce
        if cov_w > 0:
            loss = loss + cov_w * coverage_loss(cov, tmask.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, nb = total + loss.item(), nb + 1
    return total / nb


@torch.no_grad()
def evaluate(model, pairs, feats, edge_index, nbrs, cache, n2i, w2i, lex2id,
             artist_ids=None, song_genre_ids=None, k=10, batch=256):
    """Returns (hit@k, unmet-intent rate, ndcg@10 over the remaining set)."""
    model.eval()
    dev = feats.device
    z = model.song_z(feats, artist_ids, edge_index)
    hits = n = unmet = ntok = 0
    ndcg, nb = 0.0, 0
    for i in range(0, len(pairs), batch):
        emb, mask, lex, seqs, qgenre = encode_batch(pairs[i:i + batch], cache, n2i, w2i, lex2id)
        emb, mask, lex, seqs = emb.to(dev), mask.to(dev), lex.to(dev), seqs.to(dev)
        if qgenre is not None:
            qgenre = qgenre.to(dev)
        tok, tmask, pooled = model.query(emb, mask, lex)
        logits, cov = model.dec(z, tok, tmask, pooled, nbrs, seqs,
                                song_genre_ids, qgenre)
        topk = logits.topk(k, dim=-1).indices
        hits += (topk == seqs.unsqueeze(-1)).any(-1).sum().item()
        n += seqs.numel()
        unmet += ((cov < 0.2) & tmask).sum().item()
        ntok += tmask.sum().item()
        ndcg += ndcg_at_10(logits, seqs)
        nb += 1
    return hits / n, unmet / ntok, ndcg / nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lexical", action="store_true")
    ap.add_argument("--coverage", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cograph", action="store_true",
                    help="co-membership graph (window=4, weighted) instead of adjacency")
    ap.add_argument("--content", action="store_true",
                    help="full content node features: key/mode/duration/etc + artist emb + title text emb")
    ap.add_argument("--listwise", type=float, default=0.0,
                    help="blend weight for position-discounted listwise set loss")
    ap.add_argument("--layers", type=int, default=0,
                    help="deep residual GNN with this many SAGE layers (0 = legacy 2-layer)")
    ap.add_argument("--logitadj", type=float, default=0.0,
                    help="tau for logit-adjusted softmax (popularity prior at train time)")
    ap.add_argument("--lexv2", action="store_true",
                    help="expanded ModernBERT-initialized anchor vocab + genre-phrase anchors")
    ap.add_argument("--steer", action="store_true",
                    help="learned coverage steering in the decoder")
    ap.add_argument("--shuffle", type=float, default=0.0,
                    help="prob of shuffling a training playlist's order per sample")
    ap.add_argument("--genrebias", action="store_true",
                    help="direct learned genre-match bias in the decoder (requires --lexv2)")
    ap.add_argument("--out", default="data/model_text.pt")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]

    artist_ids, n_artists = None, 0
    if args.content:
        first_artist = songs["artists"].fillna("").str.split(";").str[0].str.lower()
        a2i = {a: i for i, a in enumerate(sorted(set(first_artist)))}
        n_artists = len(a2i)
        artist_ids = torch.tensor([a2i[a] for a in first_artist])
        feats = torch.cat([feats, content_features(songs), build_title_cache(songs)], 1)

    names = sorted({p["name"] for p in pairs})
    cache = build_text_cache(names)
    n2i = {n: i for i, n in enumerate(cache["names"])}
    w2i = {w: i for i, w in enumerate(cache["words"])}
    emb_dim = cache["name_emb"].size(1)

    random.shuffle(pairs)
    n_test = len(pairs) // 10
    test, train = pairs[:n_test], pairs[n_test:]
    if args.cograph:
        edge_index, nbrs = build_co_graph(train, len(songs))
        nbrs = [(i.to(dev), w.to(dev)) for i, w in nbrs]
    else:
        edge_index, nbrs = build_graph(train, len(songs))
        nbrs = [t.to(dev) for t in nbrs]
    feats, edge_index = feats.to(dev), edge_index.to(dev)

    lex_init, song_genre_ids, n_genres = None, None, 0
    if args.lexv2:
        (lex_vocab, word2id, phrase2id, lex_init, song_genre_ids,
         anchor_genre, n_genres) = build_lex_v2(train, cache, songs)
        song_genre_ids = song_genre_ids.to(dev)
        lexmap = {"words": word2id, "phrases": phrase2id,
                  "anchor_genre": anchor_genre, "n_genres": n_genres}
    else:
        lex_vocab = build_lex_vocab(train) if args.lexical else []
        lexmap = {"words": {w: i + 1 for i, w in enumerate(lex_vocab)},
                  "phrases": {}} if lex_vocab else {}
    print(f"{len(songs)} songs, {edge_index.size(1)} edges (train-only), "
          f"{len(train)} train / {len(test)} test, emb dim {emb_dim}, "
          f"lexical vocab {len(lex_vocab)} (v2={args.lexv2}), "
          f"steer={args.steer}, genrebias={args.genrebias}, "
          f"coverage weight {args.coverage}")

    if artist_ids is not None:
        artist_ids = artist_ids.to(dev)
    log_prior = None
    if args.logitadj > 0:
        from collections import Counter as _C
        pop = _C(s for p in train for s in p["track_ids"])
        counts = torch.tensor([pop.get(i, 0) + 1.0 for i in range(len(songs))])
        log_prior = (args.logitadj * torch.log(counts / counts.sum())).to(dev)
    model = TextPlaylistModel(feats.size(1), emb_dim, args.hidden,
                              len(lex_vocab), n_artists, args.layers,
                              lex_init, args.steer, args.genrebias).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        loss = run_epoch(model, opt, train, feats, edge_index, nbrs, cache,
                         n2i, w2i, lexmap, args.coverage, artist_ids,
                         args.listwise, log_prior, args.shuffle, song_genre_ids)
        hit, unmet, ndcg = evaluate(model, test, feats, edge_index, nbrs, cache,
                                    n2i, w2i, lexmap, artist_ids, song_genre_ids)
        print(f"epoch {ep+1}: loss {loss:.3f}  test hit@10 {hit:.2%}  "
              f"ndcg@10 {ndcg:.3f}  unmet-intent rate {unmet:.1%}  "
              f"(random {10/len(songs):.2%})")

    torch.save({"model": model.state_dict(), "emb_dim": emb_dim,
                "hidden": args.hidden, "lex_vocab": lex_vocab,
                "content": args.content, "n_artists": n_artists,
                "layers": args.layers, "steer": args.steer,
                "genrebias": args.genrebias, "n_genres": n_genres,
                "lexv2": args.lexv2,
                "lexmap": {"words": lexmap.get("words", {}),
                           "phrases": lexmap.get("phrases", {}),
                           "anchor_genre": lexmap.get("anchor_genre", {}),
                           "n_genres": n_genres}}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
