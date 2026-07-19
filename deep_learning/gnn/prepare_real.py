"""Join the Zenodo #nowplaying Spotify playlists with the HF 114k-track
audio-features dataset, and mine playlist names as query tokens.

Outputs (in data/):
  songs_real.parquet   catalog of matched songs + audio features + genre
  playlists_real.jsonl one line per kept playlist: {tokens, name, track_ids}

Caveat: track order inside playlists in this dump is mostly alphabetical, so
edges built from it represent co-listening (same-playlist membership), not
true play transitions.
"""

import csv
import json
import re
from collections import Counter, defaultdict

import pandas as pd

MIN_TRACKS = 8          # playlist must have this many matched tracks
MIN_SONG_FREQ = 3       # song must appear in this many kept playlists
MIN_TOKEN_FREQ = 40     # name token must appear in this many playlists
MAX_TOKENS = 300

STOP = set("""the a an of and to in for on with my me i you your it is at from de la
el los las und der die das mix playlist music songs song list lists tracks best top
new good great stuff misc random favorites favourites favs faves fav other others
vol volume part one two three no not this that all things by""".split())

print("loading track features ...")
t = pd.read_parquet("data/tracks.parquet")
t["first_artist"] = t["artists"].fillna("").str.split(";").str[0].str.lower().str.strip()
t["tname"] = t["track_name"].fillna("").str.lower().str.strip()
t = t.drop_duplicates(subset=["first_artist", "tname"]).reset_index(drop=True)
key2row = {(a, n): i for i, (a, n) in enumerate(zip(t["first_artist"], t["tname"]))}
print(f"  {len(t)} unique (artist, track) with features")

print("streaming playlists csv ...")
playlists = defaultdict(list)   # (user, playlistname) -> [track_row_idx, ...]
nrows = 0
with open("data/spotify_dataset.csv", errors="replace") as f:
    r = csv.reader(f, skipinitialspace=True)
    next(r)
    for row in r:
        if len(row) != 4:
            continue
        nrows += 1
        idx = key2row.get((row[1].lower().strip(), row[2].lower().strip()))
        if idx is not None:
            playlists[(row[0], row[3])].append(idx)
print(f"  {nrows} rows, {len(playlists)} raw playlists")

# dedupe tracks within playlist (keep first occurrence), enforce min length
kept = {}
for k, tracks in playlists.items():
    seen, uniq = set(), []
    for s in tracks:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    if len(uniq) >= MIN_TRACKS:
        kept[k] = uniq
print(f"  {len(kept)} playlists with >= {MIN_TRACKS} matched tracks")

# keep songs frequent enough across kept playlists, then re-filter playlists
freq = Counter(s for tracks in kept.values() for s in tracks)
good_songs = {s for s, c in freq.items() if c >= MIN_SONG_FREQ}
kept = {k: [s for s in v if s in good_songs] for k, v in kept.items()}
kept = {k: v for k, v in kept.items() if len(v) >= MIN_TRACKS}
print(f"  {len(good_songs)} songs (freq >= {MIN_SONG_FREQ}), {len(kept)} playlists after re-filter")

# mine query tokens from playlist names
def name_tokens(name):
    return [w for w in re.findall(r"[a-z]+", name.lower()) if len(w) > 2 and w not in STOP]

tok_freq = Counter(tok for (_, name) in kept for tok in set(name_tokens(name)))
vocab = [w for w, c in tok_freq.most_common(MAX_TOKENS) if c >= MIN_TOKEN_FREQ]
vocab_set = set(vocab)
print(f"  query vocab: {len(vocab)} tokens; top 30: {vocab[:30]}")

# remap song ids to a compact range and write outputs
songs = sorted(good_songs)
remap = {s: i for i, s in enumerate(songs)}
t.iloc[songs].reset_index(drop=True).to_parquet("data/songs_real.parquet")

n_out = 0
with open("data/playlists_real.jsonl", "w") as f:
    for (user, name), tracks in kept.items():
        toks = [w for w in name_tokens(name) if w in vocab_set]
        if not toks:
            continue
        f.write(json.dumps({"name": name, "tokens": toks,
                            "track_ids": [remap[s] for s in tracks]}) + "\n")
        n_out += 1
print(f"wrote {n_out} (query, playlist) pairs, {len(songs)} songs")
