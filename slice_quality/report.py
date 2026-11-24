"""Report: PipelineResult -> JSON + markdown (the final deliverable).

Turns the kept slices into a prioritised, human-readable list of validated
failure modes, each with its Slice-Quality Vector, diagnostic sentence, and
repair advice.

Usage
-----
    from slice_quality.report import to_json, to_markdown
    to_json(result, "outputs/slices.json")
    print(to_markdown(result))
"""

from __future__ import annotations

import json
import os
from typing import Any


def result_to_records(result) -> list[dict[str, Any]]:
    """One dict per kept slice, sorted by repair priority (desc)."""
    records = []
    for ks in result.kept:
        q = ks.quality
        records.append({
            "slice_id": q.slice_id,
            "source": ks.slice_.source,
            "hypothesis": ks.slice_.hypothesis,
            "size": q.size,
            "dominant_error_type": q.dominant_error_type.value,
            "axes": {
                "omega_squared": round(q.omega_squared, 4),
                "manifold_compactness": round(q.manifold_compactness, 4),
                "misprediction_coherence": round(q.misprediction_coherence, 4),
                "error_type_purity": round(q.error_type_purity, 4),
                "slice_purity": round(q.slice_purity, 4),
                "significant": bool(q.significant),
            },
            "passes_def1": bool(q.passes_def1),
            "explanation": ks.explanation.sentence,
            "over_represented": [
                {"tag": t, "slice_pct": round(sp, 3), "pop_pct": round(pp, 3)}
                for t, sp, pp in ks.explanation.over_represented
            ],
            "advice": {
                "text": ks.advice.text,
                "fix_family": ks.advice.fix_family,
                "target_data": ks.advice.target_data,
                "cost_hint": ks.advice.cost_hint,
                "priority": round(ks.advice.priority, 4),
            },
        })
    records.sort(key=lambda r: r["advice"]["priority"], reverse=True)
    return records


def to_dict(result) -> dict[str, Any]:
    return {
        "residual_error": round(result.residual_error, 4),
        "n_kept": len(result.kept),
        "slices": result_to_records(result),
    }


def to_json(result, path: str, indent: int = 2) -> str:
    payload = to_dict(result)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
    return path


def export_slice_index(result, json_path: str, txt_path: str | None = None,
                       used_ids: set | None = None) -> str:
    """The full slice -> images/instances index (EVERY scored slice).

    Unlike ``to_json`` (kept slices only, no members) and
    ``image_descriptions.json`` (the inverse map, validated slices only), this
    lists every slice both branches generated — all verdicts — with the images
    it lives in and the instance ids per image. One JSON (machine-readable) and
    an optional TXT (one block per slice, for eyeballing).

    ``used_ids``: slice_ids that actually feed the per-image descriptions
    (kept cluster slices + language slices passing the ADAPTED gate). The
    ``verdict`` column alone is misleading for language slices: Def.1's N_min
    (~35) exceeds their typical size, so they read DISCARD even when validated
    by the adapted gate -- ``in_descriptions`` tells the truth.
    """
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    # scored = every top-level candidate; kept adds the REFINE sub-slices that
    # were kept (they are in result.kept but not in result.scored).
    pairs = list(result.scored) + [(ks.slice_, ks.quality) for ks in result.kept]
    for s, q in pairs:
        if q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        by_img: dict[str, list[str]] = {}
        for m in s.members:
            by_img.setdefault(m.image_id, []).append(m.instance_id)
        records.append({
            "slice_id": q.slice_id,
            "source": s.source,
            "verdict": q.verdict,
            "in_descriptions": bool(used_ids and q.slice_id in used_ids),
            "hypothesis": s.hypothesis,
            "aliases": (getattr(s, "metadata", None) or {}).get("aliases", []),
            "corroborated_by": (getattr(s, "metadata", None) or {}).get(
                "corroborated_by", []),
            "dominant_error_type": q.dominant_error_type.value,
            "size": q.size,
            "n_images": len(by_img),
            "significant": bool(q.significant),
            "sig_test": q.extras.get("sig_test", ""),
            "omega_squared": round(q.omega_squared, 4),
            "images": {img: sorted(v) for img, v in sorted(by_img.items())},
        })
    order = {"KEEP": 0, "REFINE": 1, "DISCARD": 2}
    records.sort(key=lambda r: (not r["in_descriptions"],
                                order.get(r["verdict"], 3), -r["size"]))

    payload = {"n_slices": len(records), "slices": records}
    parent = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(parent, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if txt_path:
        lines = [f"{len(records)} slices (used-in-descriptions first, "
                 "then KEEP / REFINE / DISCARD; size desc)", ""]
        for r in records:
            hyp = f'  "{r["hypothesis"]}"' if r["hypothesis"] else ""
            used = " | IN-DESCRIPTIONS" if r["in_descriptions"] else ""
            lines.append(
                f'== {r["slice_id"]} | {r["source"]} | {r["verdict"]}{used} | '
                f'{r["dominant_error_type"]} | {r["size"]} inst / {r["n_images"]} imgs | '
                f'sig{"+" if r["significant"] else "-"} | w2={r["omega_squared"]}{hyp}')
            for a in r["aliases"]:
                lines.append(f'   also: "{a}"')
            for img, insts in r["images"].items():
                lines.append(f"   {img}  ({len(insts)}: {', '.join(insts)})")
            lines.append("")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return json_path


def to_markdown(result) -> str:
    recs = result_to_records(result)
    lines = [
        f"# Discovered error slices ({len(recs)} kept, "
        f"residual error {result.residual_error:.3f})",
        "",
        "| # | type | n | ω² | MC | mispred | ETP | purity | sig | prio |",
        "|---|------|---|----|----|---------|-----|--------|-----|------|",
    ]
    for i, r in enumerate(recs, 1):
        a = r["axes"]
        lines.append(
            f"| {i} | {r['dominant_error_type']} | {r['size']} | "
            f"{a['omega_squared']:.2f} | {a['manifold_compactness']:.2f} | "
            f"{a['misprediction_coherence']:.2f} | {a['error_type_purity']:.2f} | "
            f"{a['slice_purity']:.2f} | {'Y' if a['significant'] else 'N'} | "
            f"{r['advice']['priority']:.2f} |"
        )
    lines.append("")
    for i, r in enumerate(recs, 1):
        lines.append(f"### {i}. {r['dominant_error_type']} slice (`{r['slice_id']}`)")
        if r["hypothesis"]:
            lines.append(f"*hypothesis:* {r['hypothesis']}")
        lines.append(f"**diagnosis:** {r['explanation']}")
        lines.append(f"**advice:** {r['advice']['text']}")
        lines.append("")
    return "\n".join(lines)
