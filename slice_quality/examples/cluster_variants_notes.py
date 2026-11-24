"""The VOC gallery's notes for the three variants of the cluster gates: on every image, which certified cluster
groups it falls in and under which variant each group is certified. CPU only, reads only.

The variants are copies of the official run rescored by outputs/chain_relaxed_cluster.sh:
  ufficiale  the thesis gates (attribute shared at p <= 0.05, coherence >= 0.6)
  A          attribute at p <= 0.10, coherence >= 0.5
  B          no attribute gate, coherence as in the thesis
A group is its member set (the same set certified by two variants is one group, under one id). Its name is
naming.py's, taken from the first variant that certifies it, as the galleries show it. The output is the
IMAGE_NOTES file regen_gallery.py reads: the images of every group first (official groups, then those A adds,
then those only B certifies), a note on each listing its groups, a badge with the variants, a header.
Run (CPU, server):  python -u slice_quality/examples/cluster_variants_notes.py
Env: VARIANTS ("ufficiale=dir A=dir B=dir"), OUT
"""
import json
import os
import pickle
from collections import defaultdict

from slice_quality.naming import cluster_names

VARIANTS = [v.split("=", 1) for v in os.environ.get(
    "VARIANTS", "ufficiale=outputs/slice_quality_inst A=outputs/slice_quality_inst_relaxA "
                "B=outputs/slice_quality_inst_relaxB").split()]
OUT = os.environ.get("OUT", "outputs/slice_quality_inst_clustervariants/cluster_variants_notes.json")
RULES = {"ufficiale": "i vincoli della tesi: attributo comune con p &le; 0,05, coerenza &ge; 0,6",
         "A": "attributo comune con p &le; 0,10, coerenza &ge; 0,5",
         "B": "nessun vincolo sull'attributo comune, coerenza &ge; 0,6 come nella tesi"}
TYPE_IT = {"fn": "mancati", "fp": "inventati", "boundary": "bordo impreciso", "merge": "fusi",
           "split": "frammentati", "extent": "estensione sbagliata", "confusion": "classe sbagliata"}
NAMES = [v for v, _d in VARIANTS]

groups = {}                        # member set -> {"vars": [...], "sid", "name", ...}
per_var = {}
for v, d in VARIANTS:
    idx = json.load(open(f"{d}/slices_index.json", encoding="utf-8"))["slices"]
    cert = [s for s in idx if s.get("in_descriptions") and s["slice_id"].startswith("cluster")]
    names = cluster_names(pickle.load(open(f"{d}/result_tagged.pkl", "rb"))["result"].scored)
    per_var[v] = (len(cert), len({i for s in cert for i in s["images"]}))
    for s in cert:
        ms = frozenset(m for ids in s["images"].values() for m in ids)
        g = groups.setdefault(ms, {"vars": [], "sid": s["slice_id"], "name": names.get(s["slice_id"], s["slice_id"]),
                                   "type": s.get("dominant_error_type", ""), "size": s["size"],
                                   "images": {i: len(ids) for i, ids in s["images"].items()}})
        g["vars"].append(v)

by_img = defaultdict(list)
for g in groups.values():
    for img, n in g["images"].items():
        by_img[img].append((g, n))


def first_var(img):
    return min(NAMES.index(g["vars"][0]) for g, _n in by_img[img])


def label(vs):
    return ", ".join(vs) if len(vs) > 1 else f"solo {vs[0]}"


order = sorted(by_img, key=lambda i: (first_var(i), -len(by_img[i]), i))
notes, badges = {}, {}
for img in order:
    gs = sorted(by_img[img], key=lambda gn: (NAMES.index(gn[0]["vars"][0]), -gn[1], gn[0]["sid"]))
    lines = [f"{g['name']}  [{g['sid']}] — {n} {'oggetto' if n == 1 else 'oggetti'} di questa immagine, "
             f"{TYPE_IT.get(g['type'], g['type'])} — varianti: {label(g['vars'])}" for g, n in gs]
    notes[img] = ["gruppi cluster di questa immagine, e in quali varianti sono certificati", "\n".join(lines)]
    vs = [v for v in NAMES if any(v in g["vars"] for g, _n in gs)]
    badges[img] = [f"cluster: {' · '.join(vs)}",
                   "varianti in cui questa immagine cade in almeno un gruppo cluster certificato: " + ", ".join(vs)]

n_first = defaultdict(int)
for img in order:
    n_first[NAMES[first_var(img)]] += 1
kinds = defaultdict(int)
for g in groups.values():
    vs = g["vars"]
    kinds["in tutte e tre" if len(vs) == len(NAMES) else f"solo in {vs[0]}" if len(vs) == 1
          else "in " + " e ".join(vs)] += 1
header = ("<b>Varianti dei gruppi cluster.</b> " + " &middot; ".join(
    f"<b>{v}</b> ({RULES.get(v, '')}): {per_var[v][0]} gruppi su {per_var[v][1]} immagini" for v in NAMES)
    + f". In tutto {len(groups)} gruppi distinti ("
    + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda kv: (kv[0] != "in tutte e tre", kv[0].startswith("solo"), kv[0])))
    + f") su {len(order)} immagini. "
    + "Nella vista per immagine ci sono tutte: prima le " + ", poi le ".join(
        f"{n_first[v]} " + ("dei gruppi ufficiali" if k == 0 else f"che entrano con {v}")
        for k, v in enumerate(NAMES) if n_first[v])
    + ". Su ogni immagine una nota dice in quali gruppi cade, con quanti oggetti, e in quali varianti il gruppo "
    "&egrave; certificato; il badge arancione riassume le varianti. La vista per gruppo resta quella del run "
    "ufficiale: le schede dei gruppi nuovi, con miniature e valori dei vincoli, sono in relaxed_clusters.html.")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"order": order, "notes": notes, "badges": badges, "header": header},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"groups {len(groups)} | images {len(order)} | first by variant {dict(n_first)} | {dict(kinds)}")
print("CLUSTER_VARIANTS_NOTES_DONE", flush=True)
