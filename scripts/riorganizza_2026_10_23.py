"""Reorganisation of the SLICELENS working folder (Daniele, 23 October 2026: "metti i run vecchi in una cartella
unica e quelli ufficiali in un'altra, crea una cartella con tutti i documenti della tesi e una con tutti gli html").

Only moves files; deletes nothing but the empty folder slice_quality_inst_clustervariants and the __pycache__ folders.
Every move is written to SLICELENS/RIORGANIZZAZIONE_2026-10-23.tsv (from, to), so it can be undone. A move whose
destination already exists is skipped and reported. The scripts find the runs through
docs/_revision_numbers.out_path(), which looks in RUN_UFFICIALI, RUN_VECCHI and the old outputs/ folder.
Run: python SlicelenceCode/scripts/riorganizza_2026_10_23.py [--dry]
"""
import os
import shutil
import sys

S = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # SLICELENS
C = os.path.join(S, "SlicelenceCode")
O = os.path.join(C, "outputs")
PO = os.path.join(S, "presentazione", "ott2026")
DRY = "--dry" in sys.argv
moves, skipped = [], []

UFF = ["slice_quality_inst", "slice_quality_sbd", "slice_quality_ea", "slice_quality_ea_auc", "slice_quality_slcmp",
       "slice_quality_slcmp_auc", "slope_permutation", "pascal_ea", "pascal_ea_samples",
       "slope_vs_auc_sim.json", "slope_vs_auc_sim.txt", "slope_rules_sim.json", "slope_rules_sim.txt",
       "slope_vs_auc_disagreement.json", "slope_vs_auc_disagreement.txt"]
VEC = ["slice_quality", "slice_quality_cov05", "slice_quality_cov05b", "slice_quality_inst_slope",
       "slice_quality_inst_auc44", "slice_quality_sbd_slope", "slice_quality_inst_v6", "slice_quality_inst_relabel",
       "slice_quality_inst_extra", "slice_quality_inst_extra_full", "slice_quality_inst_dd", "slice_quality_inst_dd_base",
       "slice_quality_ea_dd", "slice_quality_ea_dd_base", "slice_quality_sbd_dd", "slice_quality_inst_pending",
       "slice_quality_pending"]
VARIE = ["slice_quality_inst_extra_merge.log", "bicycle_seg_mock.json", "bicycle_slices.json",
         "bicycle_slices_mock.json", "diag_mock.json", "seg_check.json", "run_extra_hyp.log", "run_extra_reground.log"]
# the pages, out of the run folders (the run they come from is in the destination's name)
HTML = {
    "slice_quality_inst": "HTML/VOC_ufficiale", "slice_quality_ea": "HTML/Pascal-EA", "slice_quality_sbd": "HTML/SBD",
    "slice_quality_inst_auc44": "HTML/storico/VOC_21ottobre_44situazioni",
    "slice_quality": "HTML/storico/VOC_riferimento_luglio", "slice_quality_cov05b": "HTML/storico/VOC_cov05b_settembre",
}
DOCS_NOW = [(os.path.join(C, "docs", "thesis_document_EN.docx"), "DOCUMENTI_TESI"),
            (os.path.join(PO, "Tesi_panoramica_POLIMI_v6.pptx"), "DOCUMENTI_TESI"),
            (os.path.join(PO, "Tesi_breve_15min_POLIMI_v6.pptx"), "DOCUMENTI_TESI"),
            (os.path.join(PO, "tempi_versione_breve.md"), "DOCUMENTI_TESI")]
OLD_DOCS = ([os.path.join(PO, f) for f in sorted(os.listdir(PO)) if f.endswith(".pptx") and "_v6" not in f]
            + [os.path.join(C, "docs", f) for f in ("panoramica_pipeline_IT.docx", "pipeline_implementation.docx",
                                                     "pipeline_presentation.pptx", "pipeline_presentation_IT.pptx",
                                                     "speech_IT.docx")]
            + [os.path.join(S, "Reports", f) for f in ("SliceLens_Report.pdf", "Tipologie_Errore_SliceLens_POLIMI_NEW.pptx",
                                                       "match_per_immagine_80_llm.csv")])
NOTES = [os.path.join(C, "docs", f) for f in ("report_relatore_SBD_GESD.md", "slices_voc_sbd.md")]
OLD_BUILDERS = ["build_deck_ott2026.py", "build_deck_sbd_gesd.py", "build_deck_sbd_gesd_v1.py", "build_deck_tesi_v1.py",
                "build_deck_tesi_v2_backup.py"]


def mv(src, dst_dir, name=None):
    if not os.path.exists(src):
        return
    dst = os.path.join(S, dst_dir, name or os.path.basename(src))
    if os.path.exists(dst):
        skipped.append((src, dst))
        return
    moves.append((src, dst))
    if not DRY:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


# pages first, while the runs are still where they were
for run, dest in HTML.items():
    d = os.path.join(O, run)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith((".html", ".zip")):
                mv(os.path.join(d, f), dest)
mv(os.path.join(O, "relaxed_clusters.html"), "HTML/VOC_varianti_cluster")
mv(os.path.join(PO, "card18_page.html"), "HTML/altro")
# runs
for r in UFF:
    mv(os.path.join(O, r), "RUN_UFFICIALI")
for r in VEC:
    mv(os.path.join(O, r), "RUN_VECCHI")
for f in VARIE:
    mv(os.path.join(O, f), "RUN_VECCHI/varie")
# documents
for src, dest in DOCS_NOW:
    mv(src, dest)
for src in OLD_DOCS:
    mv(src, "DOCUMENTI_TESI/versioni_precedenti")
for src in NOTES:
    mv(src, "DOCUMENTI_TESI/note_e_report")
# the deck's folder: old builders and old renders apart
for f in OLD_BUILDERS:
    mv(os.path.join(PO, f), "presentazione/ott2026/_archivio/builder_vecchi")
for f in sorted(os.listdir(PO)):
    if f.startswith("render") and os.path.isdir(os.path.join(PO, f)):
        mv(os.path.join(PO, f), "presentazione/ott2026/_archivio/render")
# the June presentation material, together
P = os.path.join(S, "presentazione")
for f in sorted(os.listdir(P)):
    if f not in ("ott2026", "TemplatePOLIMI.pptx", "giugno_2026", "README.md"):
        mv(os.path.join(P, f), "presentazione/giugno_2026")

# empty or regenerable folders
rm = [os.path.join(O, "slice_quality_inst_clustervariants")] + [
    os.path.join(x, "__pycache__") for x in (O, os.path.join(C, "docs"), PO)]
for d in rm:
    if os.path.isdir(d) and (d.endswith("__pycache__") or not os.listdir(d)):
        moves.append((d, "(removed)"))
        if not DRY:
            shutil.rmtree(d)
if not DRY and os.path.isdir(os.path.join(S, "Reports")) and not os.listdir(os.path.join(S, "Reports")):
    os.rmdir(os.path.join(S, "Reports"))
    moves.append((os.path.join(S, "Reports"), "(removed, empty)"))

rel = lambda p: os.path.relpath(p, S) if os.path.isabs(p) else p
if not DRY:
    with open(os.path.join(S, "RIORGANIZZAZIONE_2026-10-23.tsv"), "a", encoding="utf-8") as f:
        for a, b in moves:
            f.write(f"{rel(a)}\t{rel(b)}\n")
print(("DRY RUN: " if DRY else "") + f"{len(moves)} moves")
for a, b in moves:
    print(f"  {rel(a)}  ->  {rel(b)}")
for a, b in skipped:
    print(f"  SKIPPED (destination exists): {rel(a)} -> {rel(b)}")
