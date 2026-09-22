import json, os, sqlite3
import numpy as np
import face_utils

ROLL = "101"   # <-- CHANGE THIS: open your faces/ folder, see the jpg filename, use that roll

# 1) Encode the image file that registration saved
path = os.path.join("faces", f"{ROLL}.jpg")
e1 = face_utils.get_face_encoding(path)
print("A) fresh encoding -> len:", len(e1), "| min/max:", round(float(e1.min()), 2), round(float(e1.max()), 2))
print("   first 5 values:", np.round(e1[:5], 3))

# 2) Auto-find the sqlite database file in this folder
dbfile = None
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".db"):
            dbfile = os.path.join(root, f)
            break
    if dbfile:
        break
print("B) database file:", dbfile)

# 3) Auto-find the embedding column and compare
con = sqlite3.connect(dbfile)
cur = con.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("C) tables:", tables)
for t in tables:
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})")]
    emb_col = next((c for c in cols if "embed" in c.lower()), None)
    if not emb_col:
        continue
    for row in cur.execute(f"SELECT * FROM {t}"):
        stored = np.array(json.loads(row[cols.index(emb_col)]))
        d = float(np.linalg.norm(e1 - stored))
        print(f"D) table={t} | stored len={len(stored)} | min/max={round(float(stored.min()),2)}/{round(float(stored.max()),2)}")
        print("   first 5 stored:", np.round(stored[:5], 3))
        print("   DISTANCE (stored vs fresh of SAME image) =", round(d, 3))