"""The single English document of the thesis work, built on the inst run.

Assembled from make_document.py (whose sections keep their text and their
numbers-from-files loaders) plus the chapters added for the merge: a summary,
the inst results, the improvement study (SEGIMPROVER), conclusions and two
appendices. Section numbers in this source are those of make_document.py;
renumber() maps them to the final order before saving.
"""
import json
import os
import statistics as _stat
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

DOCS = os.path.dirname(os.path.abspath(__file__))
doc = Document()

# ---- base styles ----
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
for i, sz in [(1, 18), (2, 14), (3, 12)]:
    st = doc.styles[f"Heading {i}"]
    st.font.name = "Calibri"
    st.font.size = Pt(sz)
    st.font.color.rgb = RGBColor(0x1a, 0x1a, 0x1a)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, italic=False, bold=False):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.italic = italic
    r.bold = bold
    return par


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def code(text):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    par.paragraph_format.left_indent = Inches(0.25)
    return par


EM_DASH = "—"


def figure(name, caption, width=6.5):
    """Insert docs/<name> with an italic caption; skip silently if not built yet."""
    path = os.path.join(DOCS, name)
    if not os.path.exists(path):
        print("  (figure missing, skipped:", name + ")")
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)


def caption(text):
    """A centred italic caption under a table (tables carry their number in it)."""
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(text)
    r.italic = True
    r.font.size = Pt(9)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htext in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = ""
        run = c.paragraphs[0].add_run(htext)
        run.bold = True
        run.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)
    return t


# ============================ TITLE ============================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr = title.add_run("Error-Slice Discovery and Per-Image Failure Description\n"
                   "for Semantic Segmentation")
tr.bold = True
tr.font.size = Pt(20)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sub.add_run("The complete technical account — a two-branch pipeline integrating SliceLens, "
                 "Beyond IoU, MCSD, CB-SLICE and SEG4SEG, what it says about a segmenter, and "
                 "what that diagnosis can do for the segmenter. Daniele Spini, October 2026.")
sr.italic = True
sr.font.size = Pt(11)
doc.add_paragraph()

# ============================ 0. SUMMARY (new) ============================
# Every number here is read from the run files: the inst run (the reference of this
# document), the judges, the extra-hypothesis study and the SEGIMPROVER folder.
import sys as _sys
_sys.path.insert(0, DOCS)
from _revision_numbers import (load_inst_summary as _lis, load_runs as _lr,
                               load_cause as _lc, load_extra_hyp as _lx,
                               load_segimprover as _lsg)
IS, RRS, CZS, XHS, SG = _lis(), _lr(), _lc(), _lx(), _lsg()
from _revision_numbers import INST as _RN_INST, out_path as _out_path
try:
    from _revision_numbers import load_dd_hyp as _ldd
    DDH0 = _ldd()
except Exception:                       # the data-driven summary not copied yet
    DDH0 = None
try:
    from _revision_numbers import load_sbd as _lsbd, load_ea as _lea
    SB0, EA0 = _lsbd(), _lea()
except Exception:                       # the SBD / Pascal-EA summaries not copied yet
    SB0 = EA0 = None
try:
    from _revision_numbers import load_slope_verdicts as _lsv0, load_ea_naming as _len0, load_ea_repair as _ler0
    SV0, NM0, ER0 = _lsv0(), _len0(), _ler0()
except Exception:                       # the October 15-16 results not copied yet
    SV0 = NM0 = ER0 = None
try:
    from _revision_numbers import load_ea_dd as _ledd0
    EDD0 = _ledd0()
except Exception:                       # the val-EA data-driven copies not in outputs/slice_quality_ea_dd yet
    EDD0 = None
try:
    from _revision_numbers import load_sbd_dd as _lsdd0
    SDD0 = _lsdd0()
except Exception:                       # the SBD data-driven copies not in outputs/slice_quality_sbd_dd yet
    SDD0 = None
try:
    from _revision_numbers import load_slicelens_cmp as _lslc0, load_slice_image_iou as _lsii0
    _SLC0, _SII0 = _lslc0(), _lsii0()
    from _revision_numbers import load_official_validations as _lov0
    OV0 = _lov0()
except Exception as _e:                 # the validations of 21 October not copied yet
    OV0 = None
    print("  (validations on the official run unavailable:", _e, ")")
try:
    from _revision_numbers import load_gesd as _lgs0
    GS0 = _lgs0()
except Exception:                       # the GESD copies not in docs/gesd yet
    GS0 = None
from _revision_numbers import load_claims as _lcl_load
_lcl = _lcl_load()
_v3 = RRS["inst_v3"]

h("0. Summary: what this work establishes", 1)
p("A segmentation model is usually summarised by one number, its mean IoU. This work "
  "builds, measures and uses a pipeline that says instead, for every image on which the "
  "model fails, WHAT went wrong — each error typed and narrated in the register a person "
  "uses — and, where the failure is systematic, IN WHAT SITUATION it tends to recur. Two "
  "independent discovery branches find the systematic groups: a cluster branch on visual "
  "appearance and typed-error signature, and a language branch that grounds situational "
  "hypotheses with a vision-language model. Every group must pass a statistical "
  "certification before it is allowed to explain anything, and every claim the output makes "
  "is checked by an instrument that carries a negative control. The model under test is "
  "LR-ASPP MobileNetV3 on PASCAL VOC 2012 val.")
p("Which run this document reports. The pipeline was run three times with three "
  "definitions of what an error instance is, each in its own directory so that none "
  "overwrote another (§5.1, Appendix §11). The reference of this document is the inst "
  "run: containment claims beside the IoU claims, ground-truth instances taken from the VOC "
  "object masks so that an occluded object is one object, and the revised narrative. It is "
  "the typing a person agrees with, and every description-side result below is measured on "
  "it. The validation experiments of §6 — synthetic corruption, description precision, the "
  "tagging channel, split-half replication, the second segmenter — were run on the earlier "
  "reference typing (IoU claims, connected components) and are reported on it, labelled as "
  "such; they test instruments that do not depend on how a claim is made, except where §6 "
  "says otherwise. Since 21 October the language branch of the inst run, and of the SBD run, is "
  "certified by the rank test of §18.3 instead of SliceLens's slope rule; the judges and four "
  "of the validation experiments were run again on that state (§18.5). Since 23 October the inst "
  "run also holds the situations that the rank test certifies on the 120-object sample but that the slope "
  "rule had kept from the whole-class grounding, validated like the others before being admitted "
  "(§18.8). Since 25 October the rank test is run on a whole class only when the class has at least ten correct "
  "objects, and SliceLens's data-driven hypotheses are in the run (§18.9); the numbers below are those of this "
  "official run.")
p("How to read it. §1 states the goal, the proposal it started from and the research questions; "
  "§21 the background and the first studies (the literature, how VLMs work and fail, a first "
  "experiment) with the choices they justify; §2–§5 the methods integrated and the pipeline block by "
  "block, ending with the three runs. §6 gives the results and the validation experiments. "
  "§9 introduces the per-image narrative and the rubric judge, §10 the revision of Block 1 "
  "that the human comparison forced, §12 the plausibility of the blamed situation, §13 an "
  "attempt to widen the vocabulary of situations, §18 the same pipeline on a dataset six "
  "times larger (SBD), §19 a test with a planted cause (material edits of val), §20 the "
  "benchmark published with SliceLens, whose slices were defined by experts (GESD), and §14 what "
  "the diagnosis can do to improve the segmenter. §7 collects the design decisions with the evidence that decided "
  "them, §8 the limits and §15 the conclusions. Appendix §11 sets the three runs side by "
  "side on four images, Appendix §16 is the chronology of the work and Appendix §17 says how "
  "to reproduce every number.")
_ts = IS["slices"]
_cv = 100.0 * IS["cov_union"] / IS["n_err"]
_al = CZS.get("all", {}).get("pipeline", {})
_a2 = SG["arm2"]
table(["Quantity (inst run unless stated)", "Value", "§"],
      [["Typed error instances / correct references on val",
        "%d / %d" % (IS["n_err"], IS["n_ref"]), "§6"],
       ["Certified situations (language branch) / certified cluster slices",
        "%d / %d" % (_ts["language"], _ts["cluster"]), "§6, §13"],
       ["Images with a situational description",
        "%d of %d images with errors" % (IS["situational_images"], IS["images"]), "§6"],
       ["Error instances inside a certified group (both branches)",
        "%d of %d (%.1f%%)" % (IS["cov_union"], IS["n_err"], _cv), "§6"],
       ["Narrative of every typed error, rubric judge (0–10) vs mismatched control",
        "%.2f vs %.2f (79 annotated images)" % (_v3["narrative"]["score"], _v3["narrative"]["control"]),
        "§9, §10.8"],
       ["Same narrative, length-neutral facts score (0–10) vs control",
        (("%.2f vs %.2f, verdicts written by Claude (all %d annotated images: VOC %.2f, synthetic "
          "%.2f); the automatic judge Gemma gives %.2f vs %.2f"
          % (_lcl["manual"]["summary"]["tutte"]["genuine"], _lcl["manual"]["summary"]["tutte"]["control"],
             _lcl["manual"]["summary"]["tutte"]["n"], _lcl["manual"]["summary"]["voc"]["genuine"],
             _lcl["manual"]["summary"]["synth"]["genuine"], _lcl["summary"]["all_narrative"]["genuine"],
             _lcl["summary"]["all_narrative"]["control"]))
         if _lcl.get("manual") else
         ("%.2f vs %.2f (all %d annotated images: VOC %.2f, synthetic %.2f; the text written "
          "twice: unchanged in %d of %d)"
          % (_lcl["summary"]["all_narrative"]["genuine"], _lcl["summary"]["all_narrative"]["control"],
             _lcl["summary"]["all_narrative"]["n"], _lcl["summary"]["voc_narrative"]["genuine"],
             _lcl["summary"]["synth_narrative"]["genuine"],
             _lcl["duplicate"]["facts"]["same"], _lcl["duplicate"]["facts"]["n"])))
        if _lcl else "—", "§9.5"],
       ["Plausibility of the blamed situation (0–10) vs control",
        "%.2f vs %.2f (%d images)" % (_al.get("score", float("nan")), _al.get("control", float("nan")),
                                       _al.get("n", 0)), "§12.1"],
       ["Situations re-verified by the VLM on their own instances",
        ("precision %.3f against a same-class control of %.3f (reference typing: %.3f / %.3f)"
         % (OV0["prec"]["summary"]["language"]["precision"], OV0["prec"]["summary"]["language"]["control_rate_same_class"],
            OV0["before"]["prec"]["summary"]["language"]["precision"],
            OV0["before"]["prec"]["summary"]["language"]["control_rate_same_class"])) if OV0 else "—", "§6.2, §18.5"],
       ["Split-half replication of certified cluster slices",
        ("%d of %d (reference typing: %d of %d)" % (OV0["split"]["n_replicated"], OV0["split"]["n_discovered"],
                                                    OV0["before"]["split"]["n_replicated"],
                                                    OV0["before"]["split"]["n_discovered"])) if OV0 else "—", "§6, §18.5"],
       ["Second segmenter (DeepLabV3-R101): certified situations that re-certify",
        ("%d of %d (reference typing: %d of %d)" % (OV0["model2"]["recertified"], OV0["model2"]["official_language"],
                                                    OV0["before"]["model2"]["recertified"],
                                                    OV0["before"]["model2"]["official_language"])) if OV0 else "—",
        "§6.6.2, §18.5"],
       ["Eight extra generic hypotheses: certified after full re-verification",
        "%d of %d candidate pairs" % (XHS.get("full_certified", XHS["sample_certified"]), XHS["pairs"]),
        "§13"],
       ["Data-driven hypotheses (SliceLens's own generator), 16 classes: certified after full "
        "re-verification", ("%d of %d" % (DDH0["n_cert"], DDH0["n_hyp"])) if DDH0 else "—", "§13.8"],
       ["The same generator on val-EA, where the cause is planted: phrases with a material word "
        "(none about the edited object); victims added to certified slices",
        ("%d of %d; %+d (%d → %d of %d)" % (len(EDD0["material_phrases"]), EDD0["n_hyp"],
                                            EDD0["covered_dd"] - EDD0["covered_base"],
                                            EDD0["covered_base"], EDD0["covered_dd"], EDD0["victims"])) if EDD0 else "—",
        "§19.4"],
       ["The same pipeline on SBD (%s images): certified situations / cluster slices; errors "
        "inside a certified group" % (SB0["miou"]["SBD"]["n_images"] if SB0 else "—"),
        ("%d / %d; %.1f%% (VOC val %.1f%%)" % (SB0["run"]["slices"]["language"], SB0["run"]["slices"]["cluster"],
                                               100.0 * SB0["run"]["cov_union"] / SB0["run"]["n_err"],
                                               100.0 * IS["cov_union"] / IS["n_err"])) if SB0 else "—", "§18"],
       ["Planted cause (material edits of val): objects lost to the edit that fall inside a "
        "certified slice, after / before the edit",
        ("%d of %d / %d; the material is never named" % (len(EA0["planted"]["covered"]), len(EA0["planted"]["victims"]),
                                                         len(EA0["planted"]["covered_inst"]))) if EA0 else "—", "§19"],
       ["Certification with a rank test instead of SliceLens's slope rule (official since 21 "
        "October; on VOC val it also chooses what is grounded on the whole class since 23 October; since 25 "
        "October only in classes with at least ten correct objects): certified situations, VOC val / SBD",
        ("%d / %d (slope rule %d / %d; on VOC val %d when the rank test judged only the hypotheses the slope "
         "rule had sent to the whole class)"
         % (IS["slices"]["language"], SV0["rank"]["compare"]["SBD"]["new"],
            SV0["rank"]["compare"]["VOC val"]["base"], SV0["rank"]["compare"]["SBD"]["base"],
            SV0["rank"]["compare"]["VOC val"]["new"]))
        if SV0 else "—", "§18.3, §18.8, §18.9"],
       ["SliceLens alone against the pipeline, same data and VLM: errors inside a certified group, VOC val; "
        "facts score of the per-image description",
        ("%.1f%% vs %.1f%%; %.2f vs %.2f" % (100 * SV0["rank"]["compare"]["VOC val"]["summary_base"]["cov_language"]
                                              / SV0["rank"]["compare"]["VOC val"]["summary_base"]["n_err"],
                                              100 * IS["cov_union"] / IS["n_err"], _SLC0["facts"]["sl"],
                                              _SLC0["facts"]["ours"])) if (SV0 and _SLC0 and _SLC0.get("facts")) else "—",
        "§18.6"],
       ["False alarms with no real association, classes of 400 objects: SliceLens's slope rule / rank test",
        ("%.0f%% / %.0f%%" % (100 * _SLC0["sim"]["cell"][(0.0, 400)]["slope"], 100 * _SLC0["sim"]["cell"][(0.0, 400)]["auc"]))
        if (_SLC0 and _SLC0.get("sim")) else "—", "§18.6"],
       ["Images of the certified slices: mIoU in / out; their class worse where the situation holds "
        "(language slices)",
        ("%.1f / %.1f; %d of %d" % (100 * _SII0["refs"]["images in at least one certified slice"]["miou_pooled"],
                                    100 * _SII0["refs"]["images with an error, in no certified slice"]["miou_pooled"],
                                    _SII0["language"]["worse"], _SII0["language"]["n"])) if _SII0 else "—", "§18.7"],
       ["The planted cause named in an object's own description: victims / edited objects kept / "
        "unedited objects",
        ("%.0f%% / %.1f%% / %.1f%%" % (100.0 * NM0["all"]["victim"]["top1"] / NM0["all"]["victim"]["n"],
                                       100.0 * NM0["all"]["edited, not lost"]["top1"] / NM0["all"]["edited, not lost"]["n"],
                                       100.0 * NM0["sample"]["auc"]["unedited image"]["top1"] / NM0["sample"]["auc"]["unedited image"]["n"]))
        if NM0 else "—", "§19.3"],
       ["Repair of the planted cause: recall of held-out victims, repair − control (5 seeds); mIoU elsewhere",
        ("%+.2f (t = %.0f); %+.3f" % (ER0["paired"]["repair-control"]["recall.B_victims_edited"]["mean"],
                                      ER0["paired"]["repair-control"]["recall.B_victims_edited"]["t"],
                                      ER0["paired"]["repair-control"]["miou.unedited"]["mean"])) if ER0 else "—", "§14.8"],
       ["GESD, the benchmark published with SliceLens: Precision@10 of a sort by class and error kind "
        "(no hypothesis, no VLM) / of the published method",
        ("%.3f / %.3f" % (GS0["baselines"]["baselines"]["C"]["p_at_k"], GS0["ghesd_p10"])) if GS0 else "—", "§20.1"],
       ["GESD: Block 1's error type agrees with the error the expert slice's name states "
        "(hallucination / unclear boundary / missed / merged)",
        ("%.1f%% (%s)" % (100.0 * GS0["pointA"]["block1"] / GS0["pointA"]["n"],
                          " / ".join("%.0f%%" % (100.0 * GS0["pointA"]["groups"][k]["block1"] / GS0["pointA"]["groups"][k]["n"])
                                     for k in ("mis-segmented as (hallucination)", "boundary unclear", "missed", "merged"))))
        if GS0 else "—", "§20.2"],
       ["GESD: expert situations named by one of our class-specific hypotheses / whose naming phrase best "
        "isolates the experts' objects; expert-slice objects whose own description names the situation "
        "among its first three",
        ("%d / %d of %d; %.0f%%" % (sum(1 for v in GS0["vocab"].values() if v["hits"].get("gemma") or v["hits"].get("data-driven")),
                                    sum(1 for r in GS0["cc"]["level2b"].values() if r.get("named_rank") == 1), len(GS0["vocab"]),
                                    100.0 * sum(v["n"] * v["top3"] for v in GS0["full"]["level3"].values() if v.get("n"))
                                    / sum(v["n"] for v in GS0["full"]["level3"].values() if v.get("n"))))
        if GS0 else "—", "§20.4"],
       ["Diagnosis-driven fine-tuning, best arm: paired gain in mIoU over %d seeds"
        % len(_a2["total"]["d"]),
        "%+.4f, t = %.1f (threshold %.2f): not distinguishable from seed noise"
        % (_a2["total"]["mean"], _a2["total"]["t"], _a2["total"]["crit"]), "§14"]],
      widths=[3.2, 2.3, 0.9])
caption("Table 0.1. Headline numbers. Each is read from the run files when the document is "
        "built; the section column says where it is measured and discussed.")
p("What follows from them, in five sentences.")
bullet("Typing every error before anything else is what makes the rest possible: it lets "
       "severity be compared within an error type, it makes split and merge observable, and "
       "it shows a shift a single IoU hides — on a stronger model the error budget moves from "
       "perception to delineation (§4.1, §6.6).")
bullet("Statistical certification is necessary and not sufficient. Groups are tested against "
       "chance, not against fixed thresholds, and they replicate out of sample; but a group "
       "can pass a slope test and still be described by the wrong words, which is why the "
       "description itself is re-verified, judged against human annotations, and judged for "
       "the plausibility of its cause (§6.2, §9, §12, §13.4). And the slope test SliceLens uses "
       "depends on the size of the population: on a dataset six times larger it certifies fewer "
       "situations, and at VOC's size it certifies as many again; a rank test removes the "
       "dependence and certifies as many situations on both datasets (§18).")
bullet("The narrative of every typed error reaches %.2f/10 against human annotations, %.1f "
       "points above a mismatched control; the situations blamed are plausible (%.2f/10) and "
       "better named by the language branch than by visual clusters (§9, §12)."
       % (_v3["narrative"]["score"], _v3["narrative"]["score"] - _v3["narrative"]["control"],
          _al.get("score", float("nan"))))
bullet("More hypotheses do not certify more situations: generic hypotheses certify far "
       "more often than class-specific ones, and neither eight more of them aimed at the "
       "uncovered error types nor the hypotheses SliceLens derives from captions of the data "
       "add much. But the vocabulary limits what can be NAMED: of the situations experts defined "
       "on the benchmark published with SliceLens, only about half are among our hypotheses, and "
       "where one is, the VLM picks out the experts' objects (§20). Missed objects, wrong extent "
       "and wrong class have no situation "
       "a VLM can name and verify on a region (§13); and a cause planted on purpose is found — "
       "its victims grouped — and named object by object, though never as a certified group (§19).")
bullet("Diagnosis is repair only where it names a gap in the data. For situations the training "
       "data already holds, re-weighting the data or the loss with the diagnosis does not improve "
       "the segmenter, and manufacturing the situation in the training crops gives a positive but "
       "not significant gain at five seeds — three seeds had looked conclusive. For a cause the "
       "data lacks, adding it repairs most of the failure, and the images the diagnosis picks "
       "deliver most of the gain (§14).")

# ============================ 1. OVERVIEW ============================
h("1. Goal and overview", 1)
p("The goal is, for every image on which a segmentation model errs, to produce a "
  "textual description of the systematic situations that induce the failure — not "
  "merely a description of the error itself. The pipeline discovers error slices "
  "(coherent groups of failing instances) and turns them into per-image, natural-language "
  "explanations.")
p("The system runs two complementary branches over the same erroneous instances:")
bullet("Cluster branch (class-agnostic): groups error instances by visual appearance and "
       "typed-error signature. It answers what and how the model fails (e.g. “small "
       "objects of class ‘bicycle’ not detected”).")
bullet("Language branch (SliceLens): matches instances to situational hypotheses, verified "
       "by a VLM. It answers in what situation the model fails (e.g. “an object cut off at "
       "the image border”, “several overlapping objects of the same type”).")
figure("panels/2007_003088.png",
       "Figure 1. What the system produces, on one image. The bottle (purple in the "
       "ground truth) is small and clipped by the frame; the second person is behind "
       "the first. Both branches fire and say different things: "
       "\u201csmall objects of class \u2018bottle\u2019 not detected\u201d and "
       "\u201coccluded objects of class \u2018person\u2019 not detected\u201d from the "
       "cluster branch, \u201cbottle \u2014 \u2018an object cut off at the image "
       "border\u2019\u201d from the language branch.")
p("Each per-image description combines the two: the SliceLens hypothesis gives the "
  "situation and the cluster/typed-error branch gives the error type "
  "(“‘backlit person’ → not detected (missed)”).")


h("1.1  The proposal this work started from, and why it changed", 2)
p("The project grew out of the proposal Improve Segmentation Models using a Visual Language Model "
  "(N. Catalano). It proposed to replace a purely quantitative loss with a qualitative one: a VLM "
  "would write a Diagnostic Text Description of how and where a segmentation model's prediction "
  "departs from the ground truth, and that description, through a distance in the VLM's embedding "
  "space, would guide back-propagation and distil the VLM's knowledge into a small segmentation "
  "model. The proposal is stated here because the reason it changed is part of the argument of "
  "this thesis.")
p("The first experiment run under that framing (§21.6) asked several VLMs to write one free-text "
  "paragraph per image on how a prediction differs from the ground truth, scored by a judge model "
  "against a description written by a person. The quality of a paragraph depended more on how the "
  "images were handed to the model (one composite picture or three separate attachments) than on "
  "the error itself, and even the best model stayed below 5 out of 10. A description that unstable "
  "is not a usable training signal. The work therefore changed what a description rests on: every "
  "claim is attached to one object region, the kind of error is decided from the masks without a "
  "model, and the situation that accompanies it is verified by a VLM against a chance control. The "
  "distillation loss was set aside and the diagnosis became the deliverable — a narrower goal than "
  "the proposal's, but one that can be measured. Whether the diagnosis can still improve the "
  "segmenter is asked at the end, with ordinary training (§14).")

h("1.2  Research questions", 2)
p("Five questions were set in May 2026, before the pipeline existed, and are kept in their "
  "original wording. Each points to where this document answers it; §15 returns to them.")
table(["Question", "What it asks", "Answered in"],
      [["RQ1  Granularity", "How can error slicing, developed for image-level classification, be extended "
        "to the dense, multi-object output of semantic segmentation?", "§4.1, §6"],
       ["RQ2  Data scarcity", "How can errors be diagnosed when no dataset records the root cause of a "
        "failure? Can generative models fill the gap?", "§4.6, §6.1, §19, §20, §14"],
       ["RQ3  Semantic–geometric mapping", "How can a geometric taxonomy of errors (boundary, missed, "
        "wrong class) be joined to a natural-language explanation a person can use?", "§4.8, §9, §10"],
       ["RQ4  Beyond the closed world", "Can slices be discovered without a predefined vocabulary of tags "
        "or a bounded search space?", "§4.3, §4.4, §13"],
       ["RQ5  Validation", "How can the correctness of a qualitative explanation be measured when no "
        "gold standard exists?", "§6, §9, §12, §20"]],
      widths=[1.6, 3.6, 1.2])
caption("Table 1.1. The research questions of May 2026 and the sections that answer them.")

# ============================================================================
#  21. Background and first studies (March-May 2026): the literature review, how
#      segmentation models fail, how VLMs work and fail, the first experiment and
#      the first architecture -- from the documents in UNI/TESI/GENERALI/LAVORO
#      FINALE. The first experiment is recomputed from its own score files
#      (docs/first_experiment_stats.py -> docs/first_experiment/first_experiment.json).
# ============================================================================
_FXP = os.path.join(DOCS, "first_experiment", "first_experiment.json")
FX = json.load(open(_FXP)) if os.path.exists(_FXP) else None
h("21. Background and first studies", 1)
p("Before any of the pipeline was built, the work went through four steps that decided much of "
  "what came later: a study of how vision-language models are built and where they fail (March "
  "2026), a first architecture (April), a review of the literature on error slicing and on the "
  "failures of segmentation models, and a first experiment on whether a VLM can describe a "
  "segmentation error in its own words (May). This chapter reports them, and for each one the "
  "choices of the final pipeline it justifies (Table 21.3).")

h("21.1  Error slicing, and why segmentation makes it harder", 2)
p("A model can score well on average and still fail systematically on a subset of the data that "
  "shares a semantic or visual attribute: objects in low light, in an unusual pose, of a rare "
  "sub-category. Such a subset S of the test set D is an error slice, and the gap Δ = |Metric(D) − "
  "Metric(S)| is what an aggregate metric such as the mean IoU hides. In safety-critical uses "
  "(driving, medical imaging) a hidden gap of that kind is the failure that matters.")
p("Slice discovery was developed for image classification, where one image carries one label and "
  "is either right or wrong. A segmentation image holds several objects, each segmented well or "
  "badly in its own way: one is missed, another bleeds into the background, two are merged into "
  "one. The unit of analysis therefore has to move from the image to the object, and the error "
  "has to be described before it can be grouped — the granularity problem (RQ1) that Block 1 "
  "answers (§4.1).")

h("21.2  Slice discovery: from classification to multi-instance tasks", 2)
p("The review of May 2026 covered the methods that made failures nameable, in the order the "
  "field developed.")
bullet("Domino (Eyuboglu et al., 2022) formalised slice discovery with cross-modal embeddings: "
       "a mixture model in CLIP space finds regions where errors are dense, and the shared text "
       "space gives them a name (“dogs on snowy backgrounds”). It showed that language can "
       "describe a failure mode.")
bullet("GEORGE (Sohoni et al., 2020) addressed hidden stratification — a class handled well "
       "except for a rare sub-class — by clustering the model's own activations.")
bullet("LADDER (Ghosh et al., Findings of ACL 2025) has a large language model read the captions "
       "(or the radiology reports) of a vision classifier's errors and propose, in natural language, "
       "hypotheses on where the classifier fails, which it then uses to mitigate the errors: the "
       "premise this thesis shares, that an error is a semantic gap that can be described, not only "
       "a wrong label.")
bullet("HiBug2 (Chen et al., ICLR 2025) generates task-specific visual attributes and enumerates "
       "slices hierarchically, extending discovery to detection and pose estimation; its tags, "
       "however, come from a bounded vocabulary.")
bullet("SliceLens (Zhang et al., arXiv 2025; final version GH-ESD, ECCV 2026) is the closest work: an LLM proposes failure hypotheses "
       "and a VLM verifies them on each instance, so a slice is grounded in objects (“closed "
       "umbrellas missed due to non-typical pose”) rather than in whole images. It became the "
       "language branch of this pipeline (§2, §4.4). The benchmark proposed with its first "
       "version (FeSD) checks whether hypotheses match real drops in performance but does not "
       "score the language; the final version of the paper (GH-ESD) published the expert slices "
       "used in §20.")
p("Five works addressed segmentation directly, each from a different side.")
bullet("Beyond IoU (Bernhard et al., WACV 2024) looks at what lies outside the intersection: "
       "every wrong pixel is assigned to one mutually exclusive error type (boundary, extent, "
       "segment), so the IoU is decomposed into kinds of failure. It is precise about WHAT "
       "happened and silent about why. It became the core of Block 1 (§4.1).")
bullet("SEG4SEG (Weng et al., MIDL 2026) asks whether subgroup discovery methods built for "
       "classification find systematic failures of medical segmentation without annotations, and "
       "categorises their sources (shape, intensity, anatomical context). Its clustering and "
       "success criterion entered the cluster branch (§2).")
bullet("Manifold Compactness (Yu et al., AAAI 2026) measures the coherence of a slice in the latent "
       "space without metadata: a slice is interpretable only if its members are close and "
       "consistent. It became the compactness axis and the residual peeling (§2, §4.5, §4.7).")
bullet("Pascal-EA (Yin et al., CVPR 2024) edits real images with diffusion models while keeping "
       "their masks valid — the colour, material, pattern or style of an object or of the whole "
       "image — to stress-test segmenters in a controlled way. Its journal extension, Gen4Seg (Yin "
       "et al., arXiv 2026), adds geometry (size, position) and weather, on Pascal-EA and COCO-EA. "
       "The material edits of Pascal-EA are the planted cause of §19.")
bullet("SemiEarth (Wang et al., arXiv 2026) uses a VLM to purify the pseudo-labels of semi-supervised "
       "remote-sensing segmentation, especially at boundaries: an example of a VLM used to repair "
       "the data rather than only to find errors, the direction of §14.")

h("21.3  What is missing for segmentation, and what each gap decided", 2)
p("The same review listed what the literature did not yet offer for segmentation. Each gap "
  "became a constraint on the design (Table 21.1).")
table(["Gap found in May 2026", "Why it matters", "Answer in this pipeline"],
      [["Little work on pixel-level slicing", "Methods assume one error state per image; in "
        "segmentation the error lives in regions of single objects",
        "Errors typed per object instance, 7 cases (§4.1)"],
       ["No dataset records why a model fails", "Masks say what an object is, never why the model "
        "misses it; no metadata (weather, occlusion) to check a slice against",
        "Groups certified against chance rather than against a root-cause label (§4.6, §18.3); "
        "known causes built where possible: corruption (§6.1), planted edits (§19), expert slices (§20)"],
       ["Competing taxonomies", "Geometric (Beyond IoU) is precise but says no why; attribute-based "
        "(HiBug2, SliceLens) names situations but not the error; domain taxonomies do not transfer",
        "Each description joins a geometric part decided from the masks to a situational part "
        "verified by the VLM (§4.8, §9)"],
       ["Benchmarks score groups, not language", "FeSD checks drops in performance, Beyond IoU is "
        "quantitative, Pascal-EA depends on the generator's fidelity",
        "The language is scored on its own: precision (§6.2), human notes (§6.4, §9), plausibility (§12)"],
       ["No gold standard for an explanation", "An explanation is judged by people or by another "
        "model, and either can be wrong", "Every judge runs with a negative control (§6.5, §9.3)"],
       ["Search space and closed vocabulary", "Subgroup search finds only what its input attributes "
        "encode (SEG4SEG); a fixed tag list is assigned, not seen in the pixels (HiBug2)",
        "Two independent branches: an open embedding space with no vocabulary, and hypotheses "
        "generated freely but verified on each object's region (§4.3, §4.4, §13)"]],
      widths=[1.5, 2.5, 2.5])
caption("Table 21.1. The gaps found by the literature review of May 2026 and the design choice "
        "each one led to.")
p("One gap has begun to close since the review. NEMO (Hyeon-Woo et al., workshop poster at ICML 2026) "
  "proposes a benchmark of natural-language explanations of the errors of vision classifiers, scored "
  "by a language model acting as judge, and the final version of SliceLens (GH-ESD) judges whether the "
  "name of each slice it finds matches an expert slice with Gemini 2.5 Pro (its Sec. 5.2), apart from "
  "its Precision@10, which reads the objects only (§20.1). Both judge the language with a model; this "
  "thesis does the same, and adds a negative control to every judge (§6.5, §9.3).")

h("21.4  How segmentation models fail: what the literature reports", 2)
p("A second part of the review collected the failures reported for segmentation models, from "
  "segmentation networks to foundation models of the SAM family. They fall into a few recurring "
  "regimes:")
bullet("Thin, branching or low-contrast structures (vessels, camouflaged or textured targets) are "
       "over- or under-segmented, because local structure is confused with background texture; "
       "fine-tuning helps only in part.")
bullet("Size, parts and context: small objects and minor parts are missed (detail is under-used), "
       "and small regions of large objects are mislabelled (context is under-used).")
bullet("Dataset and objectness biases: methods that succeed on simple synthetic scenes fail on "
       "real images whose appearance and geometry they were not built for.")
bullet("Distribution shift and the open world: models with near-saturated benchmark scores fail "
       "on open-world images, often on the same images across models; zero-shot foundation models "
       "are inconsistent on medical and overhead imagery.")
bullet("Transformations and artefacts: rotation, extreme scale and realistic imaging artefacts "
       "cause large drops.")
bullet("Prompting and ambiguity, for promptable models: boxes work better than points, and "
       "ambiguous prompts give unstable masks.")
p("The tools proposed to expose these failures are maximum-discrepancy mining (comparing models to "
  "find the images most likely to break them), uncertainty maps and learned failure detectors that "
  "flag likely wrong pixels at test time, and human-in-the-loop correction or adapter modules, "
  "which repair some corner cases but not the structural limits. The review concluded that "
  "segmentation failures cluster in predictable regimes, linked to inductive biases, to the "
  "under-use of detail or context and to domain gaps — which is the premise of error slicing: "
  "the failures are not random, so they can be grouped and named.")
p("The overview was compiled with a literature-search assistant whose numbered references do "
  "not resolve to their sources; the primary references are still to be added. §13.7 sets its "
  "categories beside the hypotheses of the language branch and beside what the run certifies.",
  italic=True)

h("21.5  Vision-language models: how they are built and where they fail", 2)
p("The pipeline uses a VLM to verify situations and a language model to propose them and to judge "
  "texts. Their design explains both what they can be trusted with and how they should be asked. "
  "The notes of March 2026 covered the main families:")
bullet("The common design: a vision encoder (a CNN or a Vision Transformer) turns the image into "
       "feature vectors, a projection maps them into the token space of a language model, and the "
       "language model reads image tokens and text tokens in one sequence. VQA, captioning and "
       "document understanding are all answered this way.")
bullet("The Vision Transformer (Dosovitskiy et al., 2021) splits the image into 16×16 patches, "
       "projects each linearly, adds a class token and position embeddings and runs a "
       "transformer encoder; with fewer built-in assumptions than a CNN it needs more data, and "
       "attends to the whole image from the first layer.")
bullet("CLIP (Radford et al., 2021) trains an image encoder and a text encoder together, "
       "contrasting matching and non-matching pairs, so that images and captions share one "
       "embedding space; classes become text prompts and classification becomes zero-shot.")
bullet("Fusion of the two modalities happens early (one transformer on both), late (separate "
       "encoders aligned by a loss, as in CLIP) or through cross-attention, where image tokens "
       "and text tokens attend to each other.")
bullet("Flamingo (Alayrac et al., 2022) keeps a vision encoder and a language model frozen, "
       "compresses the visual features into a fixed number of tokens with a Perceiver Resampler, "
       "and inserts gated cross-attention layers whose gate starts at zero, so the language model "
       "begins unchanged and learns to use the image.")
bullet("BLIP-2 (Li et al., 2023) bridges a frozen image encoder and a frozen language model with a "
       "small Q-Former: a fixed set of learned queries extracts, by cross-attention, the visual "
       "information the language model needs.")
bullet("LLaVA (Liu et al., 2023) projects CLIP ViT-L/14 features into the word-embedding space of "
       "the language model and trains on visual instructions. Qwen2.5-VL, the VLM of this pipeline, "
       "follows the same encoder–projector–language-model design.")
p("The same notes recorded three weaknesses. An image needs many tokens, and every way of "
  "compressing them loses detail — exactly the small objects and thin parts where segmenters "
  "fail. The model can produce an answer that is plausible for the context but not supported by "
  "the pixels (hallucination). And it inherits the biases of its training data. Three choices of "
  "the pipeline follow from them. The VLM is never asked to compare two whole masks in free text; "
  "it is asked one closed question about one object, whose region is given by its box, which "
  "spends its tokens where the error is. Its answer is read as the probability of “yes” in the "
  "next-token distribution — a number that can be ranked and tested — rather than as prose "
  "(§4.4). And every VLM verdict is set against a control of the same kind (§6.2, §18.3), so that "
  "a bias that says “yes” to everything shows up as no lift.")

h("21.6  A first experiment: can a VLM describe a segmentation error in its own words?", 2)
if FX:
    _fm = FX["models"]
    _cells = [("JOIN-D1", "one picture, wording 1"), ("JOIN-D2", "one picture, wording 2"),
              ("SEP", "three images, wording 1")]
    p("The question of the original proposal, tested directly in May 2026 on %d VOC val images. "
      "For each image a person had written how the prediction of a MobileNet segmenter differs "
      "from the ground truth — the same annotations later used as the human reference of §6.4 and "
      "§9. Four VLMs (Gemma 3n E4B, Gemma 4, Kimi K2.5, Qwen 3.5 9B; Gemini 3.5 Flash was added on "
      "26 May) received the image, the ground-truth mask and the predicted mask and were asked for "
      "2–6 sentences on how the prediction differs: missing regions, hallucinated regions, "
      "boundary and shape quality, with the class named and no numbers. A judge model "
      "(MiniMax M2.7) scored each paragraph 0–10 against the human description with a rubric: "
      "completeness 0–3, correctness 0–3, specificity 0–2, conciseness 0–1, fluency 0–1." % FX["n_images"])
    p("The design crossed two factors. The images were sent either as one composite picture "
      "(original, ground truth and prediction side by side: JOIN) or as three separate "
      "attachments (SEP); the prompt came in two wordings with the same content (D1, D2). Each "
      "set of paragraphs was scored three times (JOIN) or twice (SEP) by the same judge, which "
      "measures the judge's own noise. The numbers below are recomputed from the score files "
      "(docs/first_experiment_stats.py).")
    table(["VLM"] + [lab for _c, lab in _cells],
          [[m] + ["%.2f  (%d missing)" % (_fm[m][c]["mean"], _fm[m][c]["na"]) for c, _l in _cells]
           for m in ["gemma-4", "Qwen3.5-9B", "gemma-3n-E4B-it", "Kimi-K2.5", "gemini-3.5-flash"]],
          widths=[1.6, 1.6, 1.6, 1.6])
    caption("Table 21.2. Mean judge score (0–10) of each VLM's paragraphs on the answers it gave, and "
            "how many of the %d images it gave no usable answer for. Gemini 3.5 Flash was added "
            "later and is not part of the comparisons below." % FX["n_images"])
    _f, _w, _j = FX["format"], FX["wording"], FX["judge_rerun"]
    p("Three sources of variation, in decreasing order. The input format dominates: the same "
      "question on the same images scores %.2f with one composite picture and %.2f with three "
      "separate images (four VLMs, %d images; Wilcoxon p = %.1e). It is not a uniform shift: the "
      "per-image correlation between the two formats is %.2f, against %.2f between the two "
      "prompt wordings — which are new paragraphs too — and %.2f between two scorings of the same "
      "paragraphs. Which images a model describes well depends on how it is shown them. The "
      "wording moves single images by %.2f points on average but not the mean (%.2f against "
      "%.2f, p = %.2f). The judge's own noise is the smallest effect: %.2f points of standard "
      "deviation when it scores the same paragraph again."
      % (_f["mean_a"], _f["mean_b"], _f["n"], _f["p_wilcoxon"], _f["pearson"], _w["pearson"],
         _j["pearson_mean"], _w["mean_abs_diff"], _w["mean_a"], _w["mean_b"], _w["p_wilcoxon"],
         _j["sd_mean"]))
    p("Among the models, Gemma 4 wrote the most accurate paragraphs when it answered (%.2f and %.2f "
      "with one picture) but gave no usable answer for %d of the %d images when the images came "
      "separately. Gemma 3n scored less (%.2f) and almost always answered. Qwen 3.5 was second with "
      "one picture (%.2f) and collapsed with three (%.2f, %d missing). Kimi was last. Even the "
      "best model stays below 5 out of 10."
      % (_fm["gemma-4"]["JOIN-D1"]["mean"], _fm["gemma-4"]["JOIN-D2"]["mean"], _fm["gemma-4"]["SEP"]["na"],
         FX["n_images"], _fm["gemma-3n-E4B-it"]["JOIN-D1"]["mean"], _fm["Qwen3.5-9B"]["JOIN-D1"]["mean"],
         _fm["Qwen3.5-9B"]["SEP"]["mean"], _fm["Qwen3.5-9B"]["SEP"]["na"]))
    _jj, _ll = FX["judges"], FX["length"]
    p("Two checks on the same paragraphs followed at the end of May and in June. A second judge "
      "(Gemini) agrees with the first only in part (Pearson %.2f over %d paragraphs) and is more "
      "generous (%.2f against %.2f). And both judges give higher scores to longer paragraphs "
      "(Spearman between word count and score %.2f and %.2f; median length %d words). That is an "
      "observational signal — a longer paragraph may also say more — and the causal test prepared "
      "for it (the same content rewritten at three lengths) was never scored."
      % (_jj["pearson"], _jj["n"], _jj["mean_gemini"], _jj["mean_minimax"], _ll["spearman_minimax"],
         _ll["spearman_gemini"], _ll["median_words"]))
    p("What it decided. A free-text description is not a stable function of the error, so it "
      "cannot be the training signal of the original proposal (§1.1). Three principles follow and "
      "run through the whole pipeline. Ground every model call in one object's region, never in a "
      "whole image. Decide the geometric half of a description — missed, hallucinated, merged, "
      "wrong class — from the masks, without a model (§4.1). And never trust a judge without a "
      "control: every judge of this document is run beside a mismatched control (§6.5, §9.3), and "
      "the length signal led to the length-neutral facts score (§9.5). The experiment also "
      "supports the choice of the language model: Gemma 4, the most accurate writer here, "
      "proposes the hypotheses and judges the texts, run locally (§5). The VLM that verifies "
      "situations, Qwen2.5-VL, was not chosen here: it is the model of the SliceLens protocol, "
      "asked a yes/no question about a box, a task this experiment did not test.")
else:
    p("(The recomputed numbers of the first experiment are missing: run docs/first_experiment_stats.py.)",
      italic=True)

h("21.7  The first architecture, and what became of it", 2)
figure("fig_first_architecture.png",
       "Figure 1b. The first architecture (April 2026): a language model proposes hypotheses from "
       "the class name, a VLM evaluates them on the image and on the overlap of ground truth and "
       "prediction, and the evaluations are clustered into slices.", width=4.5)
p("The first sketch already had the idea that survived: a language model proposes, a VLM "
  "verifies. Four things changed. The error is typed first, from the masks (§4.1), instead of "
  "being read by the VLM from the overlap of ground truth and prediction. Clustering no longer "
  "groups the VLM's evaluations: it became an independent branch on visual embeddings and typed-"
  "error features (§4.3), so that the two routes can be compared — and they turned out to cover "
  "largely different errors (§6). Every group must pass a statistical certification against "
  "chance (§4.6, §18.3). And the VLM is asked about one object, by its box (§4.4). The May plan "
  "also proposed a generative-first strategy — building the data with Gen4Seg's edits. It "
  "returned in another role: as a cause planted in known images, to check that the pipeline "
  "finds it (§19), and as a way to manufacture a diagnosed situation for training (§14.5).")
table(["First study", "What it showed", "Choice it justifies", "§"],
      [["Literature review (May)", "Slicing is image-level; no root-cause data; taxonomies do not meet",
        "Typed object instances; certification against chance; geometric + situational description",
        "§4.1, §4.6, §4.8"],
       ["Failures of segmentation models (May)", "Failures cluster in recurring regimes",
        "Errors can be grouped and named; the canonical situations; the vocabulary check", "§4.4, §13.7"],
       ["VLM foundations (March)", "Token bottleneck, hallucination, data bias",
        "One closed question per object region; P(yes) read from the logits; matched controls",
        "§4.4, §6.2"],
       ["First experiment (May)", "Free-text quality depends on the input format; judges reward length",
        "Diagnosis as the deliverable; geometry decided from masks; judges with controls; "
        "length-neutral score; Gemma 4 as language model", "§1.1, §4.1, §9.3, §9.5"],
       ["First architecture (April)", "LLM proposes, VLM verifies",
        "Kept as the language branch; clustering made an independent branch", "§4.3, §4.4"]],
      widths=[1.5, 1.8, 2.4, 0.9])
caption("Table 21.3. The first studies and the choices of the final pipeline they justify.")

# ============================ 2. PAPERS ============================
h("2. Integrated methods", 1)
p("The pipeline is assembled from five published methods; each contributes a specific block.")
table(
    ["Method", "Contribution to the pipeline"],
    [["SliceLens", "Language branch: LLM situational hypotheses + VLM grounding + slope-trend verification "
                   "(replaced by a rank test since 21 October, §18.5)."],
     ["Beyond IoU", "Typed error decomposition of every erroneous pixel (boundary / extent / segment)."],
     ["MCSD", "Manifold compactness metric (kNN graph) + residual peeling for coverage."],
     ["CB-SLICE", "Misprediction coherence and the keyword-concept explanation idea (ECSA)."],
     ["SEG4SEG", "γ-weighted GMM clustering, the ω² disparity measure, and the Def.1 success criterion."]],
    widths=[1.4, 5.0])
p("Instance-level typed errors (split / merge / confusion), the six-axis Slice-Quality "
  "Vector, the block-bootstrap significance test, the adapted validation gate for language "
  "slices, and the per-image description generator are contributions of this work.", italic=True)

h("2.1  Related work not integrated, and how it positions this pipeline", 2)
p("Two further lines of work were reviewed. Neither is integrated as a block, but both "
  "sharpen what this pipeline claims.")
p("Automatic discovery of interpretable systematic errors in semantic segmentation (Singh et "
  "al., Bosch Research, arXiv:2411.10845, not published at a venue) is the one work found "
  "that audits a SEMANTIC segmentation model. On driving images (BDD100K, ACDC) it takes the "
  "false positives of two classes, person and bicycle, as patches, checks with "
  "open-vocabulary detectors (GroundingDINO, OWL-ViT) that the class is not there, captions "
  "each patch with BLIP-2 and labels each patch systematic or not by comparing its caption "
  "and its CLIP embedding with those of its nearest neighbours, against a threshold set by "
  "eye; the labels are checked in a human study. The contrast is informative on three "
  "points. (a) Error definition: that work starts from a single failure notion, the false "
  "positives of two classes, whereas Block 1 here decomposes every erroneous pixel of every "
  "class into a typed taxonomy (FN, FP, boundary, extent, merge, split, confusion) and "
  "normalises severity WITHIN each type — without that normalisation a boundary error, "
  "whose IoU is high by construction, could never compete with a missed object. (b) "
  "Evidence: there each patch is judged on its own, with no grouping into slices and no "
  "test against chance; here every group must survive a statistical gate (chance-"
  "calibrated purity, block-bootstrap or permutation significance, BH-FDR) and a "
  "split-half replication, which is what turns 'a plausible cluster' into 'a slice the "
  "data supports'. (c) Naming: there the explanation is the caption of the single patch; "
  "here a situation is one phrase, verified by the VLM on every member's region and "
  "certified for the group, and the per-image description joins it to the error type. What "
  "the paper does support is the shared premise — a VLM's open vocabulary is a workable "
  "source of failure hypotheses for a closed-set segmentation model — and its results, on "
  "another dataset, are a useful external argument that this premise is not specific to VOC.")
p("SegQC (segmentation quality control at inference time) estimates per-image and "
  "per-pixel segmentation quality without ground truth, and evaluates the estimates "
  "against expert CORRECTIONS of the model output rather than against an independent "
  "re-annotation. Two things transfer. First, the evaluation protocol is the same logic "
  "used here for split-half validation: a quality claim must be checked on data the claim "
  "was not fitted on, and corrections are exactly such a source. Second, SegQC reports "
  "that error detection remains useful at low IoU thresholds — which is the empirical "
  "justification for the permissive matching thresholds in Block 1 (tau_match = 0.5 with "
  "tau_frag = 0.25 for fragments): a fragment that overlaps a ground-truth object only "
  "partially is still evidence about that object, and discarding it would silently "
  "convert split/merge situations into unexplained false negatives. SegQC is "
  "complementary rather than competing: it scores quality per image, this pipeline "
  "explains WHY the failures happen and groups them into situations.")

# ============================ 3. FIGURE ============================
h("3. Pipeline overview", 1)
fig = os.path.join(DOCS, "pipeline_figure.png")
if os.path.exists(fig):
    doc.add_picture(fig, width=Inches(6.5))
    cap = doc.paragraphs[-1]
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = c.add_run("Figure 2. The two-branch pipeline. Colours denote the source method; "
                   "orange blocks are contributions of this work.")
    cr.italic = True
    cr.font.size = Pt(9)
p("The unit of analysis is the instance (a connected component / object region), not the "
  "image: an image inherits the slices of its instances. All blocks after typed-error "
  "extraction operate per-instance; only hypothesis generation is dataset/class-level.")

# ============================ 4. BLOCKS ============================
h("4. The pipeline, block by block", 1)

h("4.1  Block 1 — Typed error extraction", 2)
figure("fig_taxonomy.png",
       "Figure 3. The seven typed-error cases. An IoU number says how MUCH is wrong; "
       "this block says WHAT is wrong, per object. Split and merge are the reason for a "
       "second, lower threshold: two fragments of one object can never both reach IoU "
       "0.5, so under a single threshold those two cases are invisible by construction "
       "\u2014 and indeed they counted zero until the second threshold was added.",
       width=6.6)

p("Input: predicted and ground-truth masks of the validation set. Pascal-VOC is semantic "
  "segmentation, so each per-class mask is split into connected components; each component "
  "is one instance. Predicted and GT instances are matched in two stages:")
bullet("Stage 1 — class-agnostic greedy matching by mask-IoU. The PRIMARY match needs "
       "IoU ≥ τ_match (0.5); split/merge CLAIMS use a lower bar τ_frag = τ_match/2, because "
       "two disjoint fragments can never both reach IoU 0.5 with the same GT (their area "
       "shares would sum past 1) — with a single threshold, split and merge never fire and "
       "fragments silently degrade to FPs. Among multiple claimers, the primary is the "
       "highest-IoU one; the others are fragments.")
bullet("Stage 2 — class check on the matched pairs.")
p("Every instance is then assigned exactly one of six cases:")
table(["Cardinality / class", "Case", "Signal"],
      [["1↔1, class ✓, IoU high", "correct (reference)", "—"],
       ["1↔1, class ✓, IoU imperfect", "boundary / extent", "pixel split (Beyond IoU)"],
       ["1↔1, class ✗", "confusion", "wrong class"],
       ["1↔0", "FP (hallucination)", "spurious detection"],
       ["0↔1", "FN (miss)", "embedding pooled on GT mask"],
       ["N↔1", "split (over-seg.)", "one object, many masks"],
       ["1↔M", "merge (under-seg.)", "many objects, one mask"]],
      widths=[2.3, 2.1, 2.0])
p("For matched-correct pairs the pixel error is split into boundary vs extent with the "
  "Beyond IoU distance criterion. Each instance carries: its IoU (scalar performance), the "
  "typed-error vector [E_bnd, E_ext, E_seg], the seg-subtype, and its class. This block is "
  "fully deterministic (masks only) — computed once and cached.")

p("The final form of Block 1, used by the reference run of this document. Two rules were "
  "added after the comparison with human annotations (§10) and are part of the block from "
  "here on. First, containment claims beside the IoU claims: a ground-truth region also "
  "claims the prediction that holds at least half of its area (claim_cov = 0.5), and a "
  "prediction of the same class lying at least half inside a ground-truth region claims it; "
  "an object swallowed by a merge is kept as an instance of its own, and a merge is measured "
  "against the union of what it merged (§10.2, §10.4). Second, a ground-truth instance is a "
  "VOC object (SegmentationObject), not a connected component: an object cut in two by an "
  "occluder is one instance, two touching objects of one class are two (§10.7). Predictions "
  "stay connected components, because a semantic model has nothing better to offer. The "
  "description above — connected components and IoU claims only — is the earlier reference "
  "typing, on which the validation experiments of §6 were run; Appendix §11 sets the three "
  "typings side by side.", italic=True)

h("4.2  Block 2 — Instance embeddings (DINOv2)", 2)
p("Each instance is embedded with a frozen DINOv2 (vits14, 384-dim) by mask-pooling the "
  "backbone feature map over the instance mask. False negatives have no predicted mask, so "
  "they are pooled over the GT mask — keeping every instance in the same embedding space. "
  "DINOv2 uses < 1 GB, so an 11 GB GPU is sufficient.")

h("4.3  Blocks 3–4 — Cluster branch", 2)
p("The per-instance feature concatenates three L2-normalised blocks and a performance channel:")
code("feature = [ DINOv2 emb | α · typed-error E | one-hot(subtype) | γ · (1 − IoU) ]")
p("A diagonal Gaussian Mixture Model (SEG4SEG’s γ-weighted spirit) partitions the "
  "instances; each component is a cluster slice. α balances appearance vs failure "
  "signature; γ weights performance. Membership is a hard argmax of the GMM posterior.")
p("Clustering is STRATIFIED by error subtype: one GMM per subtype (FN, FP, boundary, …) "
  "instead of a single global mixture. A global GMM lets the dominant FN/FP mass absorb "
  "boundary/extent instances into impure blobs; per subtype, every cluster slice is "
  "error-type pure by construction and the visual structure within a type (what KIND of "
  "objects get missed) emerges cleanly. The number of components per stratum is selected "
  "by BIC over a candidate grid rather than fixed a priori; strata too small to model "
  "(< 40 instances) are kept as one whole-stratum slice and judged by the SQV as-is.")

h("4.4  Language branch (SliceLens)", 2)
p("Three phases, reusing the model under test’s errors:")
bullet("Hypotheses: 7 canonical class-agnostic situations (“a small object”, “an object "
       "occluded by another object”, “an object cut off at the image border”, …) plus, per "
       "class, up to 15 written by the LLM (Gemma) from world knowledge (e.g. “beige chair "
       "against beige wall”); stored as query files. The canonical ones turned out to be "
       "almost the only ones that certify, and §13 adds eight more of that kind.")
bullet("Grounding (VLM, Qwen2.5-VL): for each (instance, hypothesis) it reads the yes/no "
       "logit to obtain the slice confidence P(yes | q, r).")
bullet("Verification (slope-trend): SliceLens flags a hypothesis as systematic if the error "
       "rate rises with confidence (is_systematic).")
p("Near-duplicate hypotheses are merged before scoring: the LLM often emits paraphrases "
  "(“a dark cow” / “a cow in shadow”) that ground onto largely the same instances. "
  "Slices whose member sets overlap with Jaccard ≥ 0.6 are grouped (connected "
  "components); the representative with the highest mean slice confidence is kept and "
  "the other hypotheses become its aliases (shown in the gallery). This removes "
  "redundant description rows and lightens the Benjamini-Hochberg multiple-testing "
  "burden. The criterion is purely member-based — no embedding or cluster-branch input "
  "— so the branches stay independent.")
p("Both branches share the SAME instance substrate: the SliceLens run is fed with the "
  "Block-1 connected-component instances (run_voc.py --substrate cc), which preserve the "
  "Block-1 instance_id. Mapping the grounded run back onto the typed instances is therefore "
  "an exact join by id — no thresholds, no mis-pairs — and a language slice’s members are "
  "the same DINOv2-embedded instances as the cluster branch. When the run records ids, the "
  "join uses the FULL scored list, not the top-10 display cut (which capped every language "
  "slice at 10 members and inflated the gate’s mean confidence). For legacy runs saved "
  "without ids (one instance per image-class pair), a (image, bbox-IoU ≥ 0.4) fallback on "
  "the top-10 is kept; it is coarser, because a class-level bbox spanning several objects "
  "matches none of the individual components well. Correctly-predicted instances (needed "
  "only for the slope-trend) are explicitly excluded from slice membership.")

h("4.5  Block 5 — The Slice-Quality Vector", 2)
p("Every candidate slice (both branches) is scored on six axes covering three properties: "
  "severity, coherence, and reliability.")
table(["Axis", "Measures", "Source"],
      [["ω²", "one-vs-rest performance disparity of the slice against the REST of the "
        "dataset (errors + reference), not \"errors vs correct\"", "SEG4SEG"],
       ["manifold compactness", "density in the DINOv2 kNN graph", "MCSD"],
       ["misprediction coherence", "same wrong output? (entropy of pred. labels)", "CB-SLICE"],
       ["error-type purity", "one dominant error type?", "new (Beyond IoU)"],
       ["slice purity", "one dominant Qwen attribute? (tagged members only)", "SEG4SEG"],
       ["significance", "adaptive test vs the OTHER errors + BH-FDR (below)", "new"]],
      widths=[1.9, 3.5, 1.0])
p("Significance adapts to what is actually testable for the slice. Testing errors against "
  "the correct reference is vacuous (any group of errors “fails more”), so the null is the "
  "other ERRORS: (A) when the slice’s severity carries information (boundary/extent or "
  "mixed slices), a block bootstrap — resampling images, since instances within an image "
  "are correlated — on the subtype-normalised severity lift vs the remaining errors; "
  "(B) when severity is degenerate (pure FN/FP/confusion slices, where every peer has "
  "1−IoU = 1 by construction), a permutation test of manifold compactness against random "
  "same-size subsets of the same subtype pool — is this slice tighter than chance? For "
  "language slices test B is non-circular (they were not formed from embeddings); for "
  "cluster slices it is post-selection (the GMM optimises compactness) — a declared "
  "limitation. Benjamini-Hochberg FDR is then applied across the candidate batch.")
p("The two output/error axes (misprediction coherence, error-type purity) are independent of "
  "the embedding space, so they are non-circular checks on the geometric axes.")

h("4.6  Block 6 — Validation", 2)
figure("fig_purity_null.png",
       "Figure 4. Why the purity gate tests against chance instead of a fixed number. "
       "The histogram is the agreement reached by random groups of same-subtype errors "
       "drawn from the same noisy tag vocabulary; the solid line is a certified slice. "
       "The dashed line is the absolute 0.80 threshold this pipeline used first and "
       "abandoned: it sits far above anything the channel can produce, so it was "
       "measuring the noise of the tagger rather than the coherence of the slice.")

p("Cluster slices use SEG4SEG’s Def.1, adapted: (a) severe = the slice’s mean "
  "subtype-normalised severity ≥ 0 (fails at least typically FOR ITS ERROR TYPE — a raw "
  "mean-performance check is biased against boundary/extent slices, and an ω² ≥ 0.14 gate "
  "is size-confounded in the one-vs-rest form: a 3%-of-dataset slice explains little total "
  "variance however badly it fails, so ω² is reported but not gated on; certification "
  "against the peers is the significance axis’s job); (b) big enough (N_min); "
  "(c) pure enough — a CHANCE-CALIBRATED attribute-purity test. Purity counts only "
  "INFORMATIVE Qwen tags (“normal”/“other”/“error” assert the absence of an attribute; "
  "counting them both rewards vacuous slices and dilutes genuine ones), and instead of "
  "SEG4SEG’s absolute p_theta = 0.80 — calibrated for clean ground-truth attributes, not "
  "a noisy 7-word VLM vocabulary — the slice must beat CHANCE: a permutation test asks "
  "whether its informative-tag agreement exceeds that of random same-size draws of "
  "same-subtype errors (p ≤ 0.05). The null draws carry the same VLM noise and the same "
  "vocabulary limits, so the comparison is fair by construction; and since attributes are "
  "NOT part of the clustering features, this is genuinely non-circular evidence for "
  "cluster slices. Slices with fewer than 5 informative tags are not certifiable — there "
  "is deliberately NO fallback (an earlier error-type-purity fallback was vacuously 1.0 "
  "under stratification and silently passed every under-tagged slice). "
  "Verdict: KEEP / REFINE / DISCARD.")
p("Language slices are validated with an adapted gate. The full Def.1 (with geometric "
  "compactness and error-type purity) is mismatched for language slices, which group by "
  "semantic hypothesis rather than geometry. Instead, a language slice is valid when it has "
  "high error and high confidence — mirroring SliceLens’s error-vs-confidence idea "
  "but using the cluster branch’s typed-error severity:")
bullet("high error, TYPE-AWARE: the raw severity (1 − IoU) is not comparable across error "
       "types — FN/FP sit at 1.0 by construction while boundary errors live around "
       "0.25–0.45, so a flat threshold always admits FN slices and can never admit a "
       "boundary slice, however systematic. The gate therefore normalises severity within "
       "each subtype (z-score of 1 − IoU against that subtype’s population mean/std) and "
       "requires the slice’s mean z ≥ −0.25: members must not fail clearly more mildly than is "
       "typical FOR THEIR ERROR TYPE. The cluster branch asks z ≥ 0; a phrase slice gets a "
       "margin because its mean z is a noisy average around 0 when misses dominate it, and an "
       "exact 0 would reject about half of those by chance;")
bullet("high confidence: mean slice confidence ≥ threshold (the VLM strongly agrees);")
bullet("plus severity (ω²), statistical significance, and a minimum size.")

h("4.7  Block 7 — Refine and residual peeling", 2)
p("A severe but incoherent slice is refined (split deterministically by error type or "
  "confusion pair, re-clustered, or — last resort — the LLM refines the hypothesis). "
  "After keeping a slice its instances are peeled and the loop repeats on the residual "
  "(MCSD), until the unexplained error mass falls below a threshold.")

h("4.8  Block 8 — Explanation and per-image description", 2)
p("A sample of each kept slice’s members is captioned by Qwen with a controlled attribute "
  "vocabulary (small / dark / occluded / truncated / cluttered), activating the slice-purity "
  "axis and the semantic half of the description. The per-image generator then aggregates, "
  "for every erroneous image, the systematic patterns its instances belong to:")
bullet("cluster pattern: “[attribute] objects of class ‘X’ [error type]”;")
bullet("language pattern: “‘situation’ (SliceLens) → [error type]” "
       "— SliceLens supplies the situation, the cluster/typed error supplies the error.")
p("Descriptions cite each instance’s actual class (not the slice-level dominant), so an "
  "image is never described with a class it does not contain. Isolated errors (instances in "
  "no certified slice) are not silent: every instance is hard-assigned to some GMM component "
  "each peeling round, so the description lists its NEAR-MISS candidate pattern "
  "(“~ 3 isolated objs near candidate pattern cluster_fn_31_r0 (REFINE, not certified)”) — "
  "covering 100% of isolated instances. Certified cluster slices are additionally "
  "cross-corroborated against the verified language slices (containment of the situation’s "
  "members inside the visual cluster): when the two independent branches discover the same "
  "failure mode, the cluster card carries the verified hypothesis as its semantic name. On "
  "the reference run the overlap is small (3 of the 14 certified cluster slices are corroborated) — the branches carve the error space along complementary axes (visual vs "
  "situational), which is itself evidence that neither branch is redundant. Each kept "
  "slice also receives repair advice (block 9: fix family, target data, priority), stored "
  "with its record; it was shown on the gallery cards until August 2026 and then removed "
  "from them, because it read as a recommendation the pipeline had not tested — §14 is "
  "where such a recommendation was finally tested. Output: image_descriptions.json / .txt, "
  "descriptions_index (description → "
  "all its images, per branch), slices_index (every slice → images/instances, verdicts and "
  "corroborations), a self-contained HTML gallery, and per-slice quality JSON.")

h("4.9  The values the pipeline uses, and where they come from", 2)
p("Every threshold and weight of the pipeline in one place, in the order they act, with the formula, what "
  "the value decides and its source. “Ours” marks a value or a test introduced in this work; the others are "
  "taken from the method named, sometimes with a test against chance in place of the method's fixed threshold.")
_nmin = int(0.005 * (IS["n_err"] + IS["n_ref"]))       # pipeline.py: dataset_size = errors + reference
table(["Value", "Formula", "What it decides", "Source"],
      [["τ_match = 0.5", "IoU = |P ∩ G| / |P ∪ G| ≥ τ_match", "a prediction and a ground-truth object match",
        "COCO, Beyond IoU"],
       ["claim 0.25; containment 0.5", "IoU ≥ τ_match/2, or at least half of one mask inside the other",
        "merges and splits become observable (with 0.5 they never fire)", "ours"],
       ["d = 1% of the diagonal", "a wrong pixel within 2d of the outline is boundary, else extent",
        "imprecise outline versus wrong extent", "Beyond IoU"],
       ["γ = 10", "feature = [DINOv2 ; typed error ; γ·(1 − IoU)]", "the weight of severity in the clustering",
        "SEG4SEG"],
       ["k by BIC", "BIC = p·ln n − 2·ln L_max, k ≤ n/15; strata under 40 errors kept whole",
        "how many groups per error type", "Schwarz (1978); bounds ours"],
       ["ω²", "(SS_between − MS_within) / (SS_total + MS_within), slice against the rest",
        "share of the IoU variance explained by the slice; reported, not a gate", "SEG4SEG"],
       ["severity z ≥ 0 (≥ −0.25 for a phrase)", "z = ((1 − IoU) − μ_type) / σ_type, mean over the members",
        "at least as severe as usual for its error type", "ours"],
       ["lift, B = 1000; MC", "mean z against the other errors, image-block bootstrap; for misses and "
        "hallucinations the share of the 10 nearest neighbours inside the slice, against random groups",
        "different from the other errors beyond chance", "ours; MC from MCSD"],
       ["coherence ≥ 0.6", "1 − H(p) / ln L over the wrong output and over the error type",
        "the members fail the same way", "CB-SLICE (Eq. 8); for the type, ours"],
       ["purity, p ≤ 0.05", "share of the most common attribute, against random same-type groups",
        "an attribute shared beyond chance", "SEG4SEG (Def. 1); permutation test ours"],
       ["N_min = 0.5%%", "|slice| ≥ 0.005 · objects, errors and correct ones (%d on VOC val)" % _nmin, "no tiny groups", "SEG4SEG (Def. 1)"],
       ["q ≤ 0.05", "Benjamini–Hochberg over all the groups and phrases tried",
        "few false discoveries among hundreds of tests", "Benjamini and Hochberg (1995)"],
       ["conf ≥ 0.5; mean ≥ 0.6", "Qwen's probability of “yes”; members = errors with conf ≥ 0.5",
        "who enters a language slice; the VLM sure enough on its members", "SliceLens; 0.6 ours"],
       ["AUC, q ≤ 0.05", "share of (wrong, correct) pairs in which the wrong object gets the higher “yes”; "
        "one-sided, image-block bootstrap SE (500 draws)", "the phrase separates the errors of its class",
        "Mann–Whitney; the test ours (§18.3)"]],
      widths=[1.25, 2.45, 1.75, 1.15])
caption("Table 4.1. The values of the pipeline.")

# ============================ 5. IMPLEMENTATION ============================
h("5. Data and implementation", 1)
table(["Component", "Choice"],
      [["Model under test", "LR-ASPP MobileNetV3 (torchvision), short-side 520 + centre-crop"],
       ["Dataset", "Pascal-VOC 2012 segmentation val (1449 images)"],
       ["Backbone (cluster)", "DINOv2 vits14 (frozen, 384-dim)"],
       ["VLM (language + tags)", "Qwen2.5-VL-7B (bf16 on 24 GB, or 4-bit on 11 GB)"],
       ["LLM (hypotheses)", "Gemma (local)"],
       ["Compute", "AIRLab servers (RTX 6000 24 GB / GTX 1080 Ti 11 GB), Docker"]],
      widths=[2.0, 4.4])
p("The pipeline is a modular Python package (slice_quality/) with one file per block. The "
  "expensive steps (DINOv2, Qwen) run on the GPU once; the scored result and embedded "
  "instances are pickled, so re-scoring, adding language classes, and regenerating the "
  "gallery run on CPU.")

h("5.1  The three runs, and which one this document uses", 2)
p("Block 1 was revised twice after the pipeline had first been evaluated (§10). Each revision "
  "was run into its own directory rather than over the previous one, so every number of an "
  "earlier state stays reproducible and any two states can be compared image by image with "
  "the same judge and the same annotations.")
table(["Run (directory)", "Block 1 typing", "Role in this document"],
      [["outputs/slice_quality", "IoU claims; a GT instance is a connected component",
        "the earlier reference: §4.1 as first written, every validation experiment of §6"],
       ["outputs/slice_quality_cov05b", "+ containment claims; merges measured against their union",
        "an intermediate state, compared in §10 and Appendix §11"],
       ["outputs/slice_quality_inst", "+ ground-truth instances = VOC objects; narrative v6; "
        "the extra generic hypotheses of §13",
        "THE REFERENCE of this document: §0, the results of §6, §9–§10.8, §12, §13"]],
      widths=[1.9, 2.4, 2.2])
caption("Table 5.1. The three runs. The typing of a run cannot be compared with another's "
        "instance by instance, which is why they live apart.")
p("The choice of inst is a choice between a description that is right about occlusion and "
  "a cluster branch that certifies more groups (§10.7, Appendix §11): the object typing is "
  "the true one and the annotators, the judge and the synthetic set all prefer the texts "
  "built on it, while its cluster branch, which needs many small homogeneous instances, "
  "certifies fewer slices. The cost is reported where it appears rather than avoided by "
  "reporting the other run.")

# ============================ 6. RESULTS ============================
h("6. Results and validation on VOC2012 val", 1)
_tx = IS["tax"]
_ct = _ts["cluster_types"]
p("The chapter opens with the reference run of this document (inst), then reports the "
  "earlier reference typing as it was published, its audit trail, and the validation "
  "experiments that were run on it.")
table(["Quantity, inst run", "Value"],
      [["Images / images with at least one typed error", "1449 / %d" % IS["images"]],
       ["Typed error instances",
        "%d — missed %d, hallucinated %d, merge %d, boundary %d, split %d, extent %d, "
        "wrong class %d" % (IS["n_err"], _tx.get("fn", 0), _tx.get("fp", 0), _tx.get("merge", 0),
                            _tx.get("boundary", 0), _tx.get("split", 0), _tx.get("extent", 0),
                            _tx.get("confusion", 0))],
       ["Correct references (IoU ≥ 0.85)", str(IS["n_ref"])],
       ["Certified situations (language branch)", str(_ts["language"])],
       ["Certified cluster slices",
        "%d (%s)" % (_ts["cluster"], ", ".join("%s %d" % (k, v) for k, v in sorted(_ct.items())))],
       ["Images with at least one certified pattern / with a situational description",
        "%d / %d" % (IS["with_pattern"], IS["situational_images"])],
       ["Error instances inside a certified group, union of both branches",
        "%d (%.1f%%): language %d, cluster %d, both %d"
        % (IS["cov_union"], _cv, IS["cov_language"], IS["cov_cluster"], IS["overlap"])]],
      widths=[3.2, 3.2])
caption("Table 6.1. The inst run on VOC2012 val. Coverage counts every instance of every "
        "slice that ships in the descriptions, from both branches (the definition fixed in "
        "§6.6.2).")
p("Against the earlier reference typing (Table 6.2 below) the population has changed shape "
  "more than size: the typing now distinguishes an object swallowed by a neighbour from one "
  "that was missed, so missed objects fall and merges rise, and an occluded object is one "
  "instance instead of several (§10.3, §10.7). The language branch certifies %d situations "
  "(%d on the reference typing), while the cluster branch certifies %d slices instead of %d: "
  "object instances are fewer, larger and more heterogeneous, and the GMM certified slices "
  "out of exactly the small, homogeneous, numerous instances the new typing merges. That is "
  "the main cost of the truer typing and §8 states it as a limitation. The validation "
  "experiments that follow (§6.1–§6.6) were run on the reference typing and are not repeated "
  "on inst: they measure the discovery and certification machinery, the tagging channel and "
  "the VLM's verification of a phrase, none of which depends on how Block 1 decides a claim."
  % (_ts["language"], RRS["ref"]["slices"]["language"], _ts["cluster"], RRS["ref"]["slices"]["cluster"]))
p("The reference typing, as published in July–August and as every validation experiment of "
  "this chapter was run on. Final numbers after the full audit AND full-coverage grounding: all 20 VOC classes on "
  "the shared BLOCK-1 substrate (exact instance_id join); for the 18 classes whose "
  "hypotheses were validated, the grounding was re-run on EVERY instance of the class "
  "(23,999 VLM evaluations, split across two GPUs) instead of the original 120-instance "
  "sample used for hypothesis discovery:")
table(["Quantity", "Value"],
      [["Images / errors / reference", "1449 / 6406 / 522"],
       ["Typed taxonomy (fragment-aware)", "FN 2651, FP 2628, boundary 799, extent 188, "
        "merge 98, confusion 38, split 4 — merge/split were invisible before the "
        "τ_frag fix (fragments counted as FPs, swallowed GTs as 137 extra FNs)"],
       ["Kept cluster slices (certified)", "14 (FN 13, FP 1; purities 0.43–0.69 on "
        "informative tags, every one at permutation p ≤ 0.035) — under the FINAL chance-calibrated "
        "purity gate with verification-mode tagging applied to the WHOLE tag pool "
        "(§6.3); 9 with captioning-mode tags; see the audit trail below"],
       ["Language slices (mapped / dedup / validated)", "367 / 351 / 35 — all "
        "slope-trend-verified on the FULL class population, not a sample; per class: "
        "cat 6, bus 5, dog 4, car 4, boat/cow/tvmonitor/sheep/bird 2, five classes at 1"],
       ["Images with situational descriptions", "826 of 1232 described images "
        "(pattern rows: FN 757, FP 206, boundary 33)"],
       ["Systematic instance coverage (union of both branches)", "36.7% of the 6406 "
        "errors fall in a certified/validated slice from EITHER branch (2354 "
        "instances; 356 of those are found by both — see §6.6.2 on how this "
        "figure itself was corrected)"],
       ["Split-half replication (cluster branch)", "94.1% (16/17) under the FINAL gates — "
        "fewer slices are discovered per half (the chance-calibrated purity gate loses "
        "tag coverage along with the halved data) but every one re-certifies out-of-"
        "sample; 94.3% (33/35) under the earlier gate configuration"]],
      widths=[3.0, 3.4])
caption("Table 6.2. The reference typing (IoU claims, connected components), final state "
        "after the July audit and full-coverage grounding.")
figure("fig_audit_trail.png",
       "Figure 5. The certified-slice count over the life of the project. Not one of "
       "these steps changed the clustering or the discovery code: each changed how the "
       "slices were TESTED, or the instrument the test reads from. Two of the five "
       "numbers were wrong for reasons that only a later measurement exposed, and both "
       "are kept here.", width=6.4)
p("Audit trail of the cluster-slice count — five numbers, each honest about what was "
  "known at the time. 22 slices: certified under a purity gate that silently FELL BACK to "
  "error-type purity for slices with fewer than 5 Qwen tags — vacuous under "
  "stratification (etp = 1.0 by construction), so under-tagged slices passed for free; "
  "the synthetic-corruption experiment (§6.1) exposed this. 6 slices: after a 3,000-tag "
  "coverage top-up on the current slice structure, with the absolute p_theta = 0.80 "
  "threshold now actually enforced — but that threshold was calibrated for clean "
  "ground-truth attributes, and under a noisy 7-word VLM vocabulary (56% of tags are the "
  "“normal” catch-all) it both rejected genuinely coherent slices (informative-tag "
  "agreement up to 0.71, diluted below 0.8 by the catch-all) and certified one vacuous "
  "slice that was 82% “normal”. 9 slices: “normal”/“other”/“error” excluded from purity "
  "(they assert the absence of an attribute), and the absolute threshold replaced by a "
  "permutation test against chance (§4.6), which self-calibrates to the tag noise; every "
  "certified slice now has an attribute agreement that provably beats random same-subtype "
  "grouping. 19 slices: the gate unchanged, the MEASUREMENT feeding it rebuilt in "
  "verification mode (§6.3) — but on only part of the tag pool, so the slice and its null "
  "were read with different instruments; §6.3 shows it was an artefact. 14 slices — the "
  "final number of the reference typing: verification mode applied to the WHOLE pool. Read "
  "as a whole, the first drop removed slices that were never tested, the second removed the "
  "wrong test, the third fixed the test, the fourth was a measurement artefact caught by its "
  "own caveat, and the fifth fixed the instrument the test reads from — without touching a "
  "line of the discovery or gating code.")
figure("fig_complementarity.png",
       "Figure 6. What each branch reaches on its own. Only 356 of the 2354 covered "
       "error instances are found by both; each branch contributes roughly a thousand "
       "that the other never sees. This is the empirical answer to the natural "
       "objection that two branches are one too many — and it is also why "
       "cross-branch corroboration is rare (3 of 14 certified cluster slices) rather "
       "than a sign that something is broken.", width=6.4)
p("The split-half protocol answers the post-selection-inference objection (slices are "
  "formed AND tested on the same data): discovery runs on half of the images, each kept "
  "slice is induced on the held-out half by nearest-centroid assignment (same subtype, "
  "radius = the 90th-percentile member distance), and must pass the certification gates "
  "out-of-sample. Under the final gates 16 of 17 replicate (94.1%); under the earlier gate "
  "configuration 33 of 35 did (94.3%), and its two failures were informative — a boundary "
  "slice, the type whose attribute tags were stale, and a slice whose held-out mass fell "
  "below N_min. The discovered patterns are properties of the model's failure modes, not "
  "artefacts of one data half." + ((" On the official run (§18.5), which certifies only %d cluster slices, "
  "the halves discover %d and %d replicate; the one that does not holds too few members on the "
  "held-out half to pass the size gate." % (_ts["cluster"], OV0["split"]["n_discovered"],
  OV0["split"]["n_replicated"])) if OV0 else ""))
p("Reading the numbers. (a) The cluster branch settles at 14 certified slices (9 before "
  "the tagging channel was rebuilt) — fewer than the first draft, but the only count in "
  "which every slice passed a fair, non-vacuous, chance-calibrated purity test on a "
  "homogeneous tag channel. (b) The language "
  "branch tells the growth story: "
  "extending grounding from 4 to 20 classes, then from a 120-instance SAMPLE to the FULL "
  "class population for validated hypotheses, raised validated slices from 8 (4-class "
  "baseline) to 35 and images with situational descriptions from 43 to 565 — a 13x "
  "increase earned entirely by grounding COVERAGE, with the slope-trend and confidence "
  "thresholds left unchanged. (c) Full-population re-grounding is a RE-VERIFICATION, not "
  "just more data: the slope-trend systematicity test and the confidence gate are "
  "recomputed on the true population, so a few hypotheses that looked systematic on a "
  "lucky 120-instance sample no longer do — a small dip relative to the "
  "sampled-coverage checkpoint (594 → 565 images) reflects this correction, not a "
  "regression. (d) Fine-grained error types now have dedicated coverage: 33 boundary "
  "pattern rows exist only because of the type-aware severity gate, impossible under the "
  "original flat threshold.")
_dsc = json.load(open(_out_path("slice_quality_inst",
                                   "image_descriptions.json"), encoding="utf-8"))
_both = sorted(i for i, r in _dsc.items()
               if {q["branch"] for q in r.get("patterns", [])} >= {"cluster", "language"})
if _both:
    p("An example of the per-image description, from the inst run: the first image (in id "
      "order) on which both branches fire; near-miss lines are omitted.")
    code(chr(10).join(l for l in _dsc[_both[0]]["text"].split(chr(10))
                      if not l.lstrip().startswith("~")))

# ============================ 6bis. CORRUPTION EXPERIMENT ============================
h("6.1 External validation: synthetic-corruption recovery", 2)
p("The results above are internally consistent (split-half, purity, significance), but do "
  "not prove the pipeline recovers a KNOWN failure pattern end-to-end. To test this without "
  "any external labelled benchmark, a controlled signal is injected directly into the real "
  "VOC error corpus: the model’s correct-ish prediction for a chosen, KNOWN subset of real "
  "GT objects is deliberately deleted, guaranteeing they become FN through the ordinary "
  "Block-1 matching logic (no hand-crafted instances) — then the pipeline is asked to find "
  "them, with no knowledge of which instances were touched. Two rules, one per branch, both "
  "grounded in patterns the pipeline already knows to look for:")
bullet("Rule A — “small chair”: the correct prediction is removed for the smallest-area "
       "quartile of GT chair instances (13 eligible objects). Recovery target: the CLUSTER "
       "branch should group them by appearance and certify a slice.")
bullet("Rule B — “car cut off at the image border”: the correct prediction is removed for "
       "GT car instances whose mask touches the image edge (60 objects). Recovery target: "
       "the LANGUAGE branch’s already-validated canonical hypothesis of the same name "
       "should re-identify them — checked against the ARCHIVED grounding confidence for "
       "these exact bounding boxes (Qwen’s judgment depends only on appearance, not on "
       "correctness, so the full-population re-grounding already scored them; zero new "
       "VLM calls needed).")
p("Robustness note: removing a prediction shifts the “#pI” index of later predictions in "
  "that image. Instance identity for embedding/tag reuse is joined on (image, bbox, error "
  "subtype) — geometry-based, so it is index-shift-proof; for archived grounding, the exact "
  "original instance_id is reconstructed from the removed prediction’s index instead of a "
  "bbox match (a CORRECT instance stores the PREDICTION’s bbox, not the GT’s — a bbox join "
  "would silently miss every case).")
table(["Rule", "Injected", "Outcome", "Read"],
      [["A — small chair (cluster)", "13",
        "4/13 (30.8%) certified; but 11/13 (85%) land in a cluster shared with another "
        "injected instance",
        "grouping works; certification is the bottleneck"],
       ["B — car cut off at border (language)", "60",
        "23/60 (38.3%) pass the confidence gate (conf. range 0.13–0.98, mean 0.50)",
        "measures agreement between the geometric injection rule and Qwen’s "
        "PERCEPTUAL judgment, not a pipeline failure"]],
      widths=[2.1, 0.9, 2.7, 1.6])
p("Rule A, diagnosed: the largest coherent host cluster (6 of 13 victims, landing "
  "TOGETHER) fails certification on exactly one axis — slice_purity = 0.571 against the "
  "0.80 threshold — while every other gate passes comfortably (error-type purity 1.000, "
  "significant, size 136 ≫ the 34-instance minimum, severity at the pass boundary). The "
  "other two DISCARDed hosts fail on the SAME axis (0.500 and 0.684). This is not a "
  "grouping failure: it is the STALE-TAG-COVERAGE limitation already flagged qualitatively "
  "elsewhere in this document, now confirmed with a controlled, single-cause diagnosis — "
  "the purity estimate rests on only 12–19 Qwen-tagged members per cluster (the injected "
  "victims themselves are untagged by design, to keep the experiment GPU-free).")
p("Epilogue: the re-tagging pass this diagnosis motivated (3,000 fresh Qwen tags, "
  "prioritising purity-blocked slices) did NOT simply confirm the old slices — it exposed "
  "that the under-tagged fallback had been silently passing them, and that the absolute "
  "purity threshold penalises a noisy VLM tag channel. Both fixes (informative-only "
  "purity, chance-calibrated permutation gate — §4.6) came out of this experiment: a "
  "controlled corruption run designed to measure recovery ended up repairing the "
  "validation itself. The final certified count is in §6.")
p("Re-run under the FINAL gates, with a dose-response extension (13 victims at the 25th "
  "area percentile; 26 at the 50th). Candidate-tier recovery is stable and strong at both "
  "doses: the injected chairs concentrate in a handful of clusters (8/13, and 8+7+9 of 26, "
  "sharing hosts) — the DISCOVERY machinery finds the pattern regardless of dose. "
  "Certified-tier recovery, however, is 0% at BOTH doses — and the dose-response is "
  "precisely what exposes why this is a DESIGN INTERACTION, not a pipeline property: the "
  "injected victims are deliberately untagged (the experiment was built GPU-free), and "
  "the final purity gate is tag-based — so even a perfectly pure victim-only cluster "
  "would be untestable and uncertifiable at ANY dose. Rule B is unaffected by the "
  "gate change (38.3%, archived grounding confidences are tag-independent).")
p("That hypothesis was then TESTED, and it turned out to be wrong — which is the more "
  "interesting outcome. Re-running the experiment with a Qwen pass over the victims' host "
  "clusters (CORRUPTION_TAG_HOSTS=1) revealed that those clusters were not short of tags "
  "at all: across the 4 host clusters only 14 members still lacked one (the 3000-tag "
  "re-tagging campaign had already restored coverage globally), so the certification gate "
  "WAS measurable and did reject them on the merits. The diagnosis is in what the tags "
  "say: of the 14 newly tagged members Qwen labelled 12 “normal” and 2 “small”. The "
  "injected property — a deliberately deleted SMALL CHAIR — is not what the tagging "
  "channel sees. Since “normal” is a catch-all excluded from purity, a cluster holding 12 "
  "of the 26 victims cannot demonstrate attribute coherence, and DISCARD is the correct "
  "verdict under the gate as defined.")
p("So the corruption experiment separates two abilities cleanly. DISCOVERY works: the "
  "victims concentrate (12 and 9 of 26 in two clusters) purely from appearance, without "
  "any knowledge of the injection. CERTIFICATION requires something more — that the "
  "shared situation be NAMEABLE in the attribute vocabulary — and “small chair” is not, "
  "because the VLM calls those regions normal. This is the same weakness that §6.2 "
  "measures from the opposite direction (attribute phrases lift only +0.125 over a "
  "matched control, with dark and blurry at zero). Two experiments built for different "
  "purposes converge on one component: the attribute tagging channel, not the slice "
  "discovery. That, and not the 0% headline, is the result to carry forward.")
p("Rule B, read carefully: a 38.3% pass rate is not a 38.3% failure of the language "
  "branch — it measures how often a coarse GEOMETRIC proxy (“the mask touches the image "
  "edge within 3 px”) agrees with Qwen’s PERCEPTUAL judgment of “cut off”. A car with a "
  "single noisy boundary pixel at x=0 is not visually “cut off” the way a car truncated "
  "to a quarter of its body is; the wide confidence spread (0.13–0.98) reflects exactly "
  "this severity gradient, which the geometric rule does not grade. A stricter injection "
  "rule (e.g. requiring a minimum fraction of the bbox outside the visible frame) would be "
  "expected to raise the recovery rate — left as a refinement.")
p("Manifests and full per-victim detail: corruption_manifest.json, corruption_results.json.",
  italic=True)

h("6.2 Description precision: is what we tell the user actually true?", 2)
p("Every check so far asks whether a GROUP is statistically unusual. None of them asks "
  "the question the deliverable actually makes: when the output says “a cow cut off at "
  "the image border”, is that true OF THAT OBJECT? This section answers it by "
  "re-verifying each shipped phrase with the VLM on the instance's own region "
  "(eval_description_precision.py).")
p("Protocol. For each distinct phrase, up to K=8 of the instances it was attached to are "
  "scored with P(yes | phrase, region) >= 0.5 (positives). Precision alone would be "
  "uninterpretable — a VLM says “yes” to a lot — so each positive is paired with two "
  "nulls: a control instance of the SAME error subtype AND the SAME class, drawn outside "
  "the slice (the strict null: it isolates the SITUATIONAL claim, because both regions "
  "already satisfy the class term), and a control of the same subtype but any class (the "
  "loose null). One-sided Fisher exact per phrase against the strict null, BH-FDR at 5%. "
  "The evaluation was re-run after every change to the tagging channel; the table "
  "reports the FINAL state (1797 evaluations, 104 phrases, 599 positives, whole tag "
  "pool in verification mode) with the captioning baseline on the last row.")
table(
    ["Description family", "Phrases", "Precision", "Strict null (same class)", "Lift"],
    [["Language (SliceLens situations)", "9", "0.901", "0.056", "+0.845"],
     ["Cluster with attribute (“a small bicycle”)", "79", "0.494", "0.246", "+0.248"],
     ["Cluster class-only (“a bottle”)", "16", "0.536", "0.557", "−0.021"],
     ["All", "104", "0.549", "0.274", "+0.275"],
     ["— attribute phrases, captioning channel (before §6.3)",
      "30", "0.454", "0.329", "+0.125"]],
    widths=[2.4, 0.7, 0.9, 1.4, 0.7])
figure("precision_figure.png",
       "Figure 7. Description precision against a matched control. Each shipped phrase "
       "is re-verified by the VLM on the instance it was attached to; the pale bar is "
       "the same question asked about an error of the SAME class and the SAME error "
       "type taken from outside the slice, so naming the class correctly earns nothing.",
       width=6.3)
p("The language row counts 9 phrases rather than 34 because the shipped wording is now "
  "the class-agnostic hypothesis with the class printed outside the quotes, so all "
  "classes sharing a canonical situation form ONE phrase group.", italic=True)
p("Reading. The language branch holds up strongly: its situations are endorsed on 90% of "
  "their own instances against 5.6% of same-class, same-subtype errors taken from "
  "elsewhere, and all 9 phrases clear BH-FDR individually. The class term cannot be doing "
  "that work, because the null already has it. The cluster branch is weaker but no longer "
  "marginal: +0.248 after the channel change against +0.125 before it, i.e. the "
  "attribute-based descriptions went from barely-better-than-chance to twice that. The "
  "class-only phrases, by contrast, are now measurably empty: precision 0.536 against a "
  "strict null of 0.557, a lift of −0.021. A description that says only “a bottle” carries "
  "no information beyond the class the ground truth already provides, and the honest "
  "conclusion is that it should be dropped from the user-facing text rather than reported "
  "as a pattern.")
p("The first run is what exposed the problem, and it is worth recording how it looked. "
  "Per phrase, the failure was concentrated and diagnosable: the attribute tags dark and "
  "blurry collapsed (a dark pottedplant 0.12, a dark person 0.00, a dark sheep 0.00, a "
  "blurry object 0.00) while truncated, small and the SliceLens situations held (a person "
  "that blends into its background 1.00, a car cut off at the image border 0.88, several "
  "overlapping bus objects 0.88). That is a defect of the TAGGING CHANNEL — a 7-word "
  "vocabulary elicited with a captioning prompt — not of the slices: the same instances "
  "group and replicate fine, they were simply being NAMED with a word the VLM would not "
  "confirm when asked directly. §6.3 acts on exactly this, and the final run measures "
  "the result: blurry moves from a single 1.00 outlier and a 0.00 to a family that tops "
  "the whole table (a blurry bottle, dog, horse and motorbike all at 1.00 against a null "
  "of 0.00), while dark stays broken (a dark sofa 0.00, a dark sheep 0.00 in either "
  "channel). Darkness is the one attribute this VLM cannot judge reliably in any mode and "
  "should be dropped from the vocabulary rather than re-tried. Two phrases fail for a "
  "different reason worth noting — an occluded bird 0.00, an occluded motorbike 0.00 — "
  "where the region is a small fragment and occlusion is a claim about what is NOT "
  "visible; a per-region query is the wrong instrument for that, and those slices would "
  "need image-level context to be judged fairly.")
p("Independence caveat, stated plainly. For language phrases the judge is the same model "
  "that selected the members, so the test is only partly independent: it is independent "
  "with respect to the WORDING (membership was grounded on the generic hypothesis, the "
  "shipped phrase names the class) but not with respect to the situation itself. For "
  "cluster phrases it is fully independent — those words come from a different prompt "
  "(region captioning against a fixed vocabulary), never from this verification query — "
  "and it is precisely the fully independent half that scored worst in the first run. "
  "The honest summary is therefore: the situational language of the pipeline is "
  "supported; the attribute language inherited from the tagging channel was weak, and "
  "re-eliciting it with a verification-style prompt (§6.3) doubled its lift without "
  "touching the slice-discovery code.")
p("A paired wording ablation settled a question this measurement itself raised. The "
  "shipped phrases had been made class-specific for readability (“a cow cut off at the "
  "image border” instead of “an object cut off at the image border”). Re-scoring the SAME "
  "instances with the SAME seed but the pre-rewrite wording gives 0.784 precision against "
  "a same-class control of 0.051 (lift +0.733, 27 of 32 phrases individually above "
  "chance), versus 0.598 / 0.216 for the class-specific form. The class name comes from "
  "the ground truth and is therefore never wrong; but folding it into the query turns a "
  "single question into a conjunction and shifts the model's attention from “is it cut "
  "off?” to “is it a cow?” — positives fall and controls rise, because the control shares "
  "the class term. The largest drops are exactly where the class is hard to read off a "
  "fragment (a boat cut off: 1.00 to 0.38; a cat that looks similar to a different "
  "category: 1.00 to 0.38), and there is no drop where the object is unambiguous "
  "(a person that blends into its background: 0.86 to 1.00).")
p("Two caveats keep this honest. The 0.784 figure is not an independent measurement: the "
  "generic query is the very query membership was grounded with, so it largely reproduces "
  "the selection criterion. And the comparison is only valid because the two runs share "
  "instances and seed. What it does establish is a rendering decision, now implemented: "
  "the class is printed OUTSIDE the quotes (cow — “an object cut off at the image "
  "border”), so the quoted text is exactly what the pipeline asserts and grounded, while "
  "the class prefix is ground-truth metadata that tells the reader which object of the "
  "image is meant.")
h("6.3 Acting on the measurement: verification-mode attribute tagging", 2)
p("Two experiments built for different purposes had converged on one component. §6.1 "
  "found that deliberately shrunk chairs come back labelled “normal” 12 times out of 14; "
  "§6.2 found that the attribute phrases those labels produce barely beat chance, with "
  "dark and blurry at zero. The common factor is not the slices but HOW the attribute is "
  "elicited: the model is asked to pick one word from a closed list, in free generation. "
  "The ablation in §6.2 had already shown the same model to be far more reliable when "
  "asked a grounded yes/no question. So the channel was rebuilt in that mode "
  "(tagging.tag_instances_verify): one question per attribute — “a small object, "
  "occupying only a small part of the image”, “an object partly hidden behind another "
  "object”, and so on — and the tag is argmax over P(yes | attribute, region), with "
  "“normal” only when no attribute reaches 0.5. Every score is stored in "
  "metadata[\"attr_scores\"], so thresholds can be re-tuned without paying for the GPU "
  "pass again. Cost: 6 forward passes per instance instead of one short generation; "
  "2000 instances took about 5.5 hours on a shared RTX6000.")
p("The instances were chosen to make the two channels comparable: slices are visited "
  "near-miss first, and within a slice the members that ALREADY carried a captioning tag "
  "are re-tagged first, with the old value kept in metadata[\"attribute_before\"]. That "
  "gives a paired sample of 652 instances measured twice, by two elicitation modes, on "
  "identical regions.")
p("The switch was carried out in two passes, and the intermediate state produced a "
  "methodological lesson worth more than the numbers, so both are reported.")
table(
    ["Paired sample (all 3651 re-elicited instances)", "Captioning", "Verification"],
    [["Informative tags (not normal/other/error)", "1568 (43%)", "2173 (60%)"],
     ["Share of the whole tag pool labelled “normal”", "56%", "40%"],
     ["Times “occluded” was used", "38", "797"],
     ["Times “cluttered” was used", "15", "500"]],
    widths=[3.4, 1.5, 1.5])
p("The flips say exactly what was happening. On the second pass alone (2999 instances): "
  "normal → occluded 244, normal → truncated 184, normal → cluttered 160, normal → small "
  "133. Instances the captioning channel dismissed as unremarkable had a nameable "
  "attribute all along; the model simply would not volunteer it when asked to choose a "
  "single word. The traffic also runs the other way where it should: dark → normal 118, "
  "i.e. the verification channel WITHDRAWS the label whose descriptions scored 0.00 in "
  "§6.2.")
table(
    ["Downstream effect", "Captioning", "Mixed (2000)", "Verification (all)"],
    [["Certified cluster slices", "9 (FN 7, FP 2)", "19 (FN 17, FP 2)", "14 (FN 13, FP 1)"],
     ["Split-half replication", "100% (7/7)", "100% (14/14)", "94.1% (16/17)"],
     ["Unexplained residual error", "0.86", "0.66", "0.755"],
     ["Images receiving a description", "565 / 1232", "902 / 1232", "826 / 1232 (67%)"],
     ["Cross-branch corroboration", "2", "4", "3"],
     ["Instance-level coverage (union of both branches)", "30.4%", "45.4%", "36.7%"]],
    widths=[2.4, 1.4, 1.4, 1.6])
p("Read the middle column with suspicion, because it is wrong — and the reason is a trap "
  "worth naming. The first pass deliberately re-elicited the NEAR-MISS slices first, so "
  "that the two channels could be compared on the same instances. The side effect is that "
  "in the intermediate state those slices carried rich verification tags while the rest of "
  "the pool still carried captioning tags, 56% of them the “normal” catch-all. The "
  "permutation null draws its comparison groups from that pool. So the slice was measured "
  "with one instrument and the null with another, and the better instrument sat "
  "systematically on the slice's side. The test was formally correct and substantively "
  "unfair: 19 slices was an artefact of a heterogeneous measurement channel, not a "
  "result. Completing the re-elicitation removes the asymmetry and leaves 14. The general "
  "rule: a chance-calibrated test is only as valid as the homogeneity of the channel that "
  "feeds both the statistic and its null — and a paired-sampling design, which is exactly "
  "what one wants for comparing two channels, is a good way to break that homogeneity "
  "without noticing.")
p("Against the honest baseline the change pays across every axis, including one this "
  "section originally reported as a trade-off and did not have to. Certified slices "
  "9 → 14, images receiving a description 565 → 804, residual 0.86 → 0.77, split-half "
  "replication 16 of 16, the attribute phrases' lift over matched chance doubles (§6.2) "
  "— and instance-level coverage, computed correctly as the union of both branches "
  "rather than the cluster branch alone (§6.6.2 below explains why that distinction "
  "matters and how it was caught), RISES from 30.4% to 35.7%, not the 30.4% → 20.7% "
  "fall an earlier draft of this section claimed.")
p("Absolute slice purity also FALLS (0.42–0.68, median 0.48, against 0.57–1.00 with "
  "captioning tags) while every certified slice passes the permutation test at p ≤ 0.035. "
  "This should not be misread as degradation: it is the chance-calibrated gate behaving as "
  "designed. When two or three labels do all the work, agreement is high by default and "
  "0.57 means little; when seven labels are genuinely in play, the agreement expected by "
  "chance drops and 0.48 becomes strong evidence. A fixed 0.80 threshold would have "
  "rejected every slice in the improved state — the argument of §4.6, now confirmed by an "
  "independent change of the measurement channel.")
p("Three consequences were applied. First, dark is no "
  "longer used to build a description: it fails in both elicitation modes, so a sentence "
  "resting on it would be a claim the VLM itself will not confirm. Second, a slice with no "
  "verified attribute is no longer described by its class alone; it is reported for what "
  "the certificate actually supports — “objects of class ‘bicycle’ grouped by appearance "
  "only (no verified attribute) not detected”. Dropping the row altogether would have "
  "hidden real structure (the group is certified and replicates); asserting “a bicycle” "
  "would have stated something the measurement contradicts. After the change: 309 distinct "
  "descriptions, 129 carrying a verified attribute, 44 declared as unnamed groups, zero "
  "built on dark.")
p("Third, dark was removed from the PURITY vocabulary as well, closing an incoherence "
  "that would otherwise have survived into the thesis: the tag was not trusted enough to "
  "SAY, yet was still counted as evidence to CERTIFY. Evidence one is unwilling to state "
  "is not evidence. The change was expected to cost slices — 237 instances carry the tag "
  "— and measurably did not: the certified set is unchanged at 14, the residual improves "
  "slightly (0.768 to 0.755), described images rise from 804 to 826 and true (both-branch) "
  "instance coverage from 35.7% to 36.7%. No certified slice had been held together by dark; "
  "it was noise "
  "floating inside groups that cohere on small, truncated, occluded or cluttered. "
  "Split-half replication moves from 100% (16/16) to 94.1% (16/17): the fold-level "
  "discovery finds one more slice than before and one of them fails to re-certify "
  "out-of-sample — a more demanding test passed at essentially the same rate, not a "
  "regression. Intermediate states are preserved as result_tagged_caption_backup.pkl, "
  "result_tagged_mixed_backup.pkl and result_tagged_verify_backup.pkl.")
p("The methodological point is worth stating on its own. The evaluation layer was built "
  "to grade the pipeline; it ended up repairing it, and then it caught the repair "
  "over-claiming. Neither the corruption experiment nor the precision evaluation was "
  "designed to test the tagging prompt, yet both located the same component; and the "
  "caveat written down when the state was still mixed is precisely what the final "
  "measurement confirmed. Reporting the intermediate number and its refutation costs "
  "nothing and is the only way the audit trail stays trustworthy.", italic=True)

h("6.4 Comparison with human error annotations", 2)
p("A set of 80 VOC images carries free-text, hand-written descriptions of what the "
  "segmentation got wrong (answer_gts.jsonl); 43 of them fall in the val split the "
  "pipeline runs on. Each was compared with the sentence the pipeline produces for the "
  "same image and graded HIGH (the main errors of the annotation are captured, with "
  "certified patterns), PARTIAL (the substance is there but only as an uncertified "
  "near-miss, or the type is expressed differently) or LOW (the described errors are not "
  "seen at all). Result: HIGH 17/43 (40%), PARTIAL 20/43 (47%), LOW 6/43 (14%). Full "
  "file: outputs/slice_quality/gt_comparison.md.")
p("Four things are worth extracting. (a) Whenever the annotation describes an object that "
  "was MISSED, HALLUCINATED or GIVEN THE WRONG CLASS, the pipeline nearly always has it, "
  "and with a certified pattern — hallucinated tvmonitors, a completely absent dining "
  "table, an unsegmented bicycle, a sheep missed and a dog confused with another "
  "category, a cow assigned to the wrong class. (b) The dominant cause of PARTIAL is a "
  "vocabulary gap on the error TYPE, not a miss: where a human writes “coarse "
  "boundaries”, the pipeline often says “not detected”, because at IoU ≥ 0.5 a badly "
  "segmented object loses its match and its components become false negatives. Same "
  "reality, different taxonomy. (c) Every LOW case is the same structural boundary: the "
  "annotation complains only about fine boundary quality, which is below the instance "
  "thresholds of Block 1 by design (IoU ≥ 0.85 counts as correct). (d) The gap is in "
  "CERTIFYING, not in DETECTING: in every PARTIAL case the objects the human described "
  "are present among the typed errors — in one image the pipeline even reaches the right "
  "error type (a confusion candidate for a sheep predicted as dog/cow) without "
  "certifying it.")
h("6.5 Scoring the comparison with an LLM judge, and calibrating the judge", 2)
p("The verdicts of §6.4 were assigned by reading, by the same agent that had been "
  "shaping the pipeline all day. That is the weakest link in the whole evaluation: not "
  "reproducible, single-rater, and not independent of the object being rated. So the "
  "comparison was re-scored mechanically (eval_gt_agreement.py) in three layers, and the "
  "instrument was validated before its output was believed.")
p("The design point that matters is the negative control. Alongside the 43 genuine pairs "
  "the judge receives 43 MISMATCHED pairs — the annotation of one image against the "
  "description of another — shuffled in and indistinguishable; the judge is not told that "
  "false pairs exist. Whatever it scores on that half is the noise floor. This is not a "
  "hypothetical precaution: the July version of §6.4 compared unrelated images for 57% of "
  "its pairs and still reported “70% partial agreement”, so a scale that agreeable on "
  "unrelated content had already been demonstrated. Judge: Gemma-4-12B local, "
  "deliberately NOT the Qwen that produced the groundings and the tags.")
table(
    ["Measure", "Genuine pairs", "Mismatched controls"],
    [["Object-class recall (rule-based, no model)", "0.484", "0.039"],
     ["Object-class precision (rule-based, no model)", "0.957", "0.062"],
     ["Error-nature overlap (rule-based, no model)", "0.240", "0.143"],
     ["Claim recall (judge)", "0.202", "0.000"],
     ["Pattern precision (judge)", "0.225", "0.000"],
     ["Judge answers “the two reports describe the same image”", "30.2%", "0.0%"]],
    widths=[3.2, 1.5, 1.7])
p("The instrument passes: on every axis the mismatched half sits at or near zero, with no "
  "parse failures over 86 pairs and 2.1 atomic claims extracted per annotation on "
  "average. The judge is not the flattering rater the earlier scale was — it is stricter "
  "than the human-written verdicts.")
p("Which raises the obvious question of who is right, and the answer is instructive. "
  "Cohen's kappa between the judge's bucketed recall and the hand-written HIGH/PARTIAL/"
  "LOW labels is 0.024 — no agreement at all. But the disagreement is not about ranking: "
  "mean judge recall is 0.42 on the images graded HIGH, 0.08 on PARTIAL and 0.00 on LOW, "
  "a perfectly monotone ordering; and of the 13 genuine pairs where the judge "
  "independently answers “same image”, 11 are exactly the HIGH ones (the other 2 are "
  "PARTIAL). The two scales agree "
  "on which cases are good and disagree on what the label means. The hand-written scale "
  "asked “was the MAIN error captured?”; the judge measures “what FRACTION of the "
  "annotation is covered?”. A human note contains 2.1 claims on average and the pipeline "
  "typically covers one of them, so the same evidence reads as 40% HIGH under the first "
  "question and 0.20 recall under the second. Both are true; the second is the one a "
  "reviewer should be given, and the first should be labelled for what it is — "
  "main-error capture, not coverage.")
p("The most useful result of this section is a number for something that had only been "
  "described in words. Agreement on the object CLASS is overwhelming (precision 0.957 "
  "genuine against 0.062 chance): when the pipeline names an object, it is an object the "
  "annotator also named. Agreement on the error TYPE is 0.24 against a chance floor of "
  "0.143 — barely distinguishable from unrelated images. The pipeline says WHICH object "
  "failed and gets it right; it says WHAT KIND of failure in a taxonomy that does not map "
  "onto how a person describes the same picture. That gap, quantified, is a stronger and "
  "more actionable statement than the qualitative “vocabulary gap” of §6.4, and it points "
  "at a concrete piece of future work: a translation layer between the typed taxonomy and "
  "the descriptive language humans actually use.")
p("A flaw in an earlier run's protocol, stated rather than hidden. The two passes were "
  "meant to measure the judge's self-consistency, but HFLLM decodes greedily, so "
  "re-asking an identical prompt returns an identical answer: the 1.00 agreement that "
  "earlier run reported measured nothing. The fix — reported here, not the earlier run "
  "— presents the two reports in the OPPOSITE order on the second pass, so a repeated "
  "verdict is no longer identical by construction; a real position/format-bias probe. "
  "Under the corrected protocol the judge's SAME/DIFFERENT verdict on the two passes "
  "agrees 97.7% of the time (Cohen's kappa 0.903) across all 86 pairs — genuinely high, "
  "and not a construction artefact this time: the judge is stable to a superficial "
  "reordering of its own input. This is the figure this thesis reports for judge "
  "self-consistency; the earlier 1.00 is not used anywhere else. Artefacts: "
  "gt_agreement.json / .txt.", italic=True)

p("A methodological caveat about this comparison itself. The annotation file indexes "
  "images by position, and the earlier version of this comparison assumed that position "
  "meant “the n-th image of trainval”. That holds only for the first 23 entries; from "
  "there the annotated images are scattered across the dataset, and the mapping is given "
  "by a separate file (voc_patch_matches.json). More than half of the previous "
  "comparison therefore put unrelated images side by side, and its figures (HIGH 16%) "
  "are void. The error was caught by checking the CLASSES named in each annotation "
  "against the classes actually present in the ground-truth mask — a cheap consistency "
  "check that any evaluation against an external reference should run before scoring "
  "anything.")

p("Per-phrase tables: eval_precision.json / .txt (current state), "
  "eval_precision_caption.json / .txt (captioning channel, before §6.3), "
  "eval_precision_generic.json / .txt (wording ablation).", italic=True)

h("6.6 Does any of this transfer? A second, much stronger segmenter", 2)
p("Everything above is true of one model. That is the sharpest limitation of the work "
  "as it stands: when the pipeline reports that objects cut off at the image border are "
  "systematically missed, a reader is entitled to ask whether that is a property of the "
  "task or of LR-ASPP MobileNetV3. The question is only answerable by running the same "
  "pipeline over a different model, so the first stage of that experiment was carried "
  "out: DeepLabV3-ResNet101 (torchvision weights trained on the COCO subset with the 21 "
  "VOC classes \u2014 no training, and the class indices already match) was run over the "
  "same 1449 validation images, and Block 1 was applied to both prediction sets.")
p("Two checks came first, because a silent misalignment here would invalidate every "
  "number downstream: the class indices were verified against the ground truth on a "
  "16-image probe (mean IoU 0.48 for MobileNet against 0.82 for DeepLab, the latter "
  "better on 15 of 16 images \u2014 consistent with published figures, so the label "
  "spaces line up), and the masks are written at original resolution so the existing "
  "loader applies exactly the same resize and centre-crop to both.")
table(
    ["Quantity", "LR-ASPP MobileNetV3", "DeepLabV3-ResNet101"],
    [["Typed errors", "6406", "4818  (\u221225%)"],
     ["Correct references (IoU \u2265 0.85)", "522", "883  (+69%)"],
     ["Images containing at least one error", "1232 of 1449", "1091 of 1449"],
     ["Errors per image", "4.42", "3.33"],
     ["Missed objects (FN)", "2651  (41.4%)", "2224  (46.2%)"],
     ["Invented objects (FP)", "2628  (41.0%)", "1418  (29.4%)"],
     ["Imprecise boundaries", "799  (12.5%)", "829  (17.2%)"],
     ["Wrong extent", "188  (2.9%)", "174  (3.6%)"],
     ["Objects fused (merge)", "98  (1.5%)", "124  (2.6%)"],
     ["Wrong class (confusion)", "38  (0.6%)", "41  (0.9%)"],
     ["Object broken (split)", "4  (0.1%)", "8  (0.2%)"]],
    widths=[2.6, 1.9, 2.0])
figure("fig_model_comparison.png",
       "Figure 8. The same pipeline over two segmenters. Left: counts per error type. "
       "Right: how the error budget is composed. The headline is not the 25% reduction "
       "but the change in shape.", width=6.6)
p("The result is more interesting than \u201cthe better model makes fewer errors\u201d. "
  "Invented objects nearly halve (\u221246%), which is what one would expect from a "
  "stronger backbone. But imprecise boundaries grow in ABSOLUTE number (+4%) and their "
  "share of the error budget rises from 12.5% to 17.2%, and fused objects grow by 27%. "
  "The failure mode moves from perception \u2014 is there an object here? \u2014 to "
  "delineation: where does it end, and is it one thing or two. A single IoU figure would "
  "have reported 0.48 \u2192 0.82 and concealed that shift entirely; it is visible only "
  "because every error carries a type. This is the clearest argument in the report for "
  "Block 1 existing at all.")
p("One consequence matters for the pipeline itself. The scarcity of correct references "
  "(522) is what forced \u03c9\u00b2 to be reported rather than gated (\u00a74.5); at "
  "883 references against 4818 errors the ratio is still unbalanced but far less "
  "extreme, so the second stage will re-test whether \u03c9\u00b2 can carry a gate on "
  "this model. The risk anticipated before the run \u2014 that a strong model would "
  "leave too few errors for any slice to clear N_min \u2014 did not materialise: 4818 "
  "errors is ample.")
h("6.6.1  Stage 2: the cluster branch on the second model", 3)
p("The same cluster branch was then run end to end on the DeepLab errors "
  "(run_second_model_cluster.py), writing to a separate pickle so the certified state "
  "of the first model stays intact. One efficiency change was made: instead of "
  "spending the tagging budget blindly, a provisional clustering is run FIRST on the "
  "untagged instances, and the budget is then spent on the members of the largest "
  "candidate slices (40 each, 2000 in total). With the first model half the budget had "
  "gone to slices that were never going to approach certification.")
table(
    ["Cluster branch, same code, two models", "MobileNetV3", "DeepLabV3-R101"],
    [["Typed errors it works from", "6406", "4818"],
     ["Candidate slices", "~600", "57"],
     ["Certified slices", "14", "11"],
     ["by error type", "FN 13, FP 1", "FN 9, FP 1, boundary 1"],
     ["Slice purity (min / median / max)", "0.43 / 0.48 / 0.69", "0.47 / 0.62 / 0.75"],
     ["Largest permutation p among certified", "0.035", "0.050"],
     ["Instance coverage", "21.8%", "20.0%"],
     ["Images with a certified cluster pattern", "536", "448"],
     ["\u03c9\u00b2 range over certified slices", "\u2014 (not gated)", "0.004 \u2013 0.016"]],
    widths=[3.0, 1.7, 1.8])
p("Three things come out of this, and the first is the one the whole experiment was "
  "for. A BOUNDARY slice is certified for the first time \u2014 “truncated objects of "
  "class \u2018person\u2019 with imprecise boundaries”, 11 images. Across every run on "
  "the first model, all certified slices were FN or FP; boundary groups existed as "
  "candidates but never cleared the gate. Stage 1 predicted exactly this from the "
  "taxonomy shift (boundary errors rising from 12.5% to 17.2% of the budget), and the "
  "prediction is borne out at the level of certified patterns, not just raw counts. The "
  "pipeline tracks the model it is auditing rather than reproducing a fixed set of "
  "findings.")
p("Second, the method transfers without retuning: 11 certified slices against 14, on a "
  "model with a quarter fewer errors, with no threshold, feature or gate changed. Slice "
  "purity is in fact HIGHER (median 0.62 against 0.48) \u2014 fewer, cleaner errors "
  "produce more coherent groups \u2014 while the permutation test still admits nothing "
  "above p = 0.05.")
p("Third, a question raised in \u00a74.5 is now answered, negatively. \u03c9\u00b2 was "
  "demoted from gate to reported axis because the first model left only 522 correct "
  "references, and the suspicion was that the measure was starved of contrast. With 883 "
  "references the \u03c9\u00b2 of certified slices is still 0.004\u20130.016, two "
  "orders of magnitude below SEG4SEG\u2019s 0.14 criterion. The problem is therefore "
  "NOT reference scarcity: a one-vs-rest effect size is intrinsically tiny when the "
  "slice is a small fraction of the population, no matter how good the base model. That "
  "closes the question rather than leaving it open as future work.")
p("One caveat on Stage 2 stands regardless of what Stage 3 adds below: the tagging "
  "budget covered 2000 of 4818 instances (41%) against roughly 78% for the first "
  "model, which is why a larger share of the patterns is reported as \u201cgrouped by "
  "appearance only\u201d \u2014 that is a budget artefact, not a property of the "
  "model.", italic=True)

h("6.6.2  A Second Self-Caught Measurement Bug, and Stage 3: the Language Branch on "
  "the Second Model", 3)
p("Stage 3 re-grounds the SAME 35 validated hypotheses \u2014 nothing is "
  "re-discovered \u2014 on the stronger model's own instances "
  "(examples/reground_validated.py, pointed at Stage 2's substrate instead of the "
  "first model's), each still required to clear SliceLens's own slope-trend test on "
  "the new data. Computing this stage's headline number, instance coverage, surfaced "
  "a second measurement bug that had been sitting unnoticed in every coverage figure "
  "quoted earlier in this document, including in Table 6.2 and in \u00a76.3 above \u2014 "
  "and it is corrected here rather than silently, for the same reason the earlier "
  "audit trail keeps its wrong numbers rather than erasing them.")
p("The bug: every coverage figure so far had been computed as the size of "
  "res.kept's member set. Cluster-branch slices reach res.kept by passing Block 6's "
  "Def.1/refine loop, but language-branch slices are validated by SliceLens's own "
  "slope-trend test and never enter that loop at all \u2014 exactly why the raw "
  "\u201ckept: language 0\u201d line that Stage 3's own script printed on this run "
  "looked alarming and was not: it counts only the cluster branch's kept slices). The "
  "consequence is that every coverage number computed via res.kept was silently "
  "CLUSTER-ONLY, undercounting by exactly the instances the language branch alone "
  "accounts for. Figure 6 (branch complementarity) had already computed the correct "
  "union for the final state \u2014 2354 covered instances, 356 of them found by "
  "both branches \u2014 but the headline coverage percentage printed alongside it, "
  "and in the tables of \u00a76.3, used the other method and was never cross-checked "
  "against the figure sitting next to it. Recomputing coverage as the union of both "
  "branches' certified members, for every state the pickled results were still "
  "available for, corrects Table 6.2 (21.8% \u2192 36.7%) and the \u00a76.3 trend, "
  "which reads very differently once fixed.")
table(
    ["State", "Cluster-only (the old, wrong figure)", "TRUE union of both branches"],
    [["Captioning baseline", "12.5%", "30.4% (unchanged \u2014 this one was "
      "computed correctly the first time)"],
     ["Mixed channel (artefact, \u00a76.3)", "30.7%", "45.4%"],
     ["Verification channel, before removing dark", "20.7%", "35.7%"],
     ["Final", "21.8%", "36.7%"]],
    widths=[2.3, 2.6, 2.6])
p("The corrected trend tells a cleaner story than the one it replaces. \u00a76.3 had "
  "reported instance coverage falling from 30.4% to 20.7% as the one metric moving "
  "\u201cthe wrong way\u201d under the channel rebuild, framed as an acceptable "
  "trade-off. It was not a trade-off: true coverage rose, from 30.4% to 35.7% before "
  "the dark fix and to 36.7% after, in step with every other axis. No result in this "
  "thesis rested on the wrong number \u2014 \u00a76.3's own certified-slice counts, "
  "residuals and image counts were always computed independently and were correct "
  "\u2014 but the coverage figure specifically was wrong in every place it was "
  "quoted, and is corrected throughout this document rather than only here.")
p("With that instrument now trustworthy, Stage 3's own combined result: 31 of the "
  "35 hypotheses re-certify on the stronger model's instances (four do not \u2014 "
  "occlusion on horse, thin/elongated parts on car, category confusion on cat, "
  "smallness on tvmonitor \u2014 plausibly situations a model that hallucinates "
  "roughly half as often, per Stage 1, is simply less prone to). Combined with "
  "Stage 2's 11 cluster slices, the stronger model's two-branch coverage is 34.7% "
  "(1674 of 4818 errors, 319 found by both branches), against the first model's "
  "36.7% \u2014 essentially the same systematic reach, on a quarter fewer errors "
  "to explain in the first place.")
table(
    ["Two branches, same code, two models", "MobileNetV3 (final)", "DeepLabV3-R101 (Stage 3)"],
    [["Certified cluster slices", "14 (FN 13, FP 1)", "11 (FN 9, FP 1, boundary 1)"],
     ["Re-certified language hypotheses", "35 / 35", "31 / 35"],
     ["Images with at least one certified pattern", "826 / 1232 (67%)", "698 / 1091 (64%)"],
     ["Instance coverage (true union)", "36.7%", "34.7%"],
     ["Cross-branch corroboration", "3", "3"]],
    widths=[3.0, 2.2, 2.4])
p("Artefacts: run_second_model_language.py, recompute_true_coverage.py, "
  "outputs/slice_quality/model2/*.", italic=True)

# ===========================================================================
# 9. From typed errors to human-style narratives, and a rubric judge
#    (September 2026). Numbers from docs/_revision_numbers.load_narrative_stage.
# ===========================================================================
import re as _re
import sys as _sys
_sys.path.insert(0, DOCS)
from _revision_numbers import load_narrative_stage as _load_stage
try:
    S1 = _load_stage()
except Exception as _ex:
    S1 = None
    print("  (narrative-stage numbers unavailable, section 9 skipped:", _ex, ")")

if S1:
    _nar, _pat, _hol = S1["narrative"], S1["pattern"], S1["holistic"]

    h("9. From typed errors to human-style narratives, and a rubric judge "
      "(September 2026)", 1)
    p("The comparison of §6.4–6.5 put the pipeline's per-image OUTPUT next to what a "
      "person writes, and the two were not the same kind of text. The per-image description "
      "(§4.8) speaks only about errors that fell inside a certified slice; the annotators "
      "describe every error they see, in a fixed register — what kind of error, on which "
      "object, where in the scene. Everything in this section follows from taking that "
      "register as the target: a second generated text that narrates EVERY typed error, the "
      "mask-level facts that text needed and Block 1 did not record, and a judge that scores "
      "agreement on a rubric instead of the claim-recall instrument of §6.5. The reference "
      "numbers of this section are those of outputs/slice_quality; §10 then revises Block 1 "
      "itself.")

    h("9.1  The narrative: one template sentence per (error type, class) group", 2)
    p("describe_errors.narrate turns the typed errors of an image into English by rule, with "
      "no language model: the text is a rendering of Block 1 and of the mask facts, so every "
      "sentence can be traced back to a number. Instances are grouped by (subtype, class), "
      "the groups are ordered as annotators order them (misses, hallucinations, confusions, "
      "merges, splits, extent, boundary), at most MAX_GROUPS = 6 groups are spelled out and "
      "the rest are counted (“A further three smaller errors on other objects are present”). "
      "Each sentence carries a size word (tiny / small / large, from the share of the frame "
      "the GROUND-TRUTH region takes — not the prediction, or a badly under-segmented object "
      "would read as tiny), a position from a 3×3 grid plus the frame edges (“on the "
      "bottom-left”, “on the right edge”, “scattered across the scene”), and the numbers that "
      "grade it: area IoU and, beside it, boundary IoU.")
    p("Examples of the register: “The two small ground truth BOAT regions on the right edge "
      "have been missed entirely.” — “Three tiny CHAIR patches have been hallucinated, "
      "scattered across the scene.” — “The ground truth DOG region in the center of the "
      "scene has been under-segmented, with whole parts of the region missing (IoU 0.47, "
      "boundary IoU 0.31).” — “One extra BICYCLE mask on the left edge splits a ground truth "
      "region that should be a single mask.”")

    h("9.2  Mask facts Block 1 does not record (mask_facts.py)", 2)
    p("Block 1 types an error and grades it with [E_bnd, E_ext, E_seg]; it does not say how "
      "the error is SHAPED, and that is what people write. mask_facts measures the rest, "
      "deterministically, from the masks, and stores it in the instance metadata:")
    bullet("Boundary IoU (Cheng et al., 2021), the IoU of the two masks restricted to a band "
           "of width 2d around their contours, d = 1% of the crop diagonal (≈14.7 px at "
           "520). It is reported as a CONTINUOUS attribute beside the area IoU, never used to "
           "re-classify. The reason is scale: area IoU is scale-dependent and hides a wrong "
           "outline on a large object — on val the gap IoU − boundary IoU grows from +0.22 on "
           "small objects to +0.60 on large ones (2011_002685: IoU 0.968, boundary IoU "
           "0.042). On small objects boundary IoU degenerates to area IoU, so it adds nothing "
           "there and everything on large regions.")
    bullet("Contour notes on CORRECT regions. A region at area IoU ≥ 0.85 produces no error "
           "and therefore, before this, no text; all six images the hand verdicts marked BASSA "
           "(“the pipeline saw nothing and should have”) contain exactly such a region. The "
           "note (“The large ground truth TRAIN region is correctly located, but its outline "
           "is loosely followed (boundary IoU 0.26)”) fires below a threshold anchored to the "
           "measured distribution of boundary IoU over the 522 reference regions: below the "
           "median (0.39) when the image has no typed error at all — the contour is then the "
           "only thing to say — and below the first quartile (0.30) when errors are present, "
           "so the note does not dilute them. The thresholds were calibrated on the 79 "
           "annotated images, and calibration is what settled their form: the label "
           "“annotation mentions the boundary” is non-discriminative (67% base rate, AUC "
           "0.512 — a majority baseline beats any threshold), whereas the BASSA verdict is "
           "discriminated (AUC 0.802), and the BASSA instances sit inside the bulk of the "
           "distribution, not in a tail: hence reporting thresholds by quantile rather than a "
           f"new error class. {S1['contour']['all']} images receive a contour note "
           f"({S1['contour']['val']} of val, the rest in train36), {S1['contour']['noerr']} of "
           "which have no typed error and produced no text before.")
    bullet("Thin details lost: a missed fragment counts as a DETAIL when a morphological "
           "opening of radius 2 erases it (fragments under 15 px are border noise) — that is "
           "what separates an antenna, a leg or a wing tip from a missing patch, which the "
           "extent term already covers. Reported when at least 8 fragments are lost (“Some "
           "fine details of the outline are lost: 12 thin fragments the prediction does not "
           "trace, 551 pixels in total”).")
    bullet("Sub-threshold ground truth: components under min_pixels = 200 are dropped by the "
           "adapter before Block 1 runs, so they were never instances and never missed. They "
           "are counted at a 15-px threshold and, until 26 September 2026, were also reported "
           "in the narrative (“One small ground truth region (44 pixels in total) is below "
           "the size the pipeline represents as an object, and has no counterpart in the "
           "prediction”). That sentence, and its counterpart for sub-threshold predicted "
           "patches, spoke about the instrument rather than about the image and have since been "
           "dropped from the text (§10.8); the measurement is unchanged and still travels "
           "in the metadata.")
    bullet("Internal holes of a predicted mask (enclosed background components ≥ 20 px), "
           "under- vs over-segmentation from missed and excess pixels (the SIGN the Beyond IoU "
           "vector does not carry), and — for every FN — the share of it a same-class "
           "prediction actually covers, which turns “missed entirely” into “absorbed into a "
           "neighbouring BOAT prediction: 66% covered, but with no outline of its own” where "
           "that is the truth. This last fact is the one that led to §10.")
    p("The systematic-pattern text of §4.8 gained the same facts as a detail line under each "
      "pattern row (boundary IoU beside area IoU, objects absorbed, thin fragments lost, holes, "
      "where in the frame), so the two texts on a card never contradict each other.")

    h("9.3  The judge: a rubric, a sum, and a negative control", 2)
    p("§6.5 scored the comparison with claim recall. For the narratives a rubric judge was "
      "built instead: Gemma-4-12B-it, local, 4-bit on a 24 GB card, greedy decoding (so no "
      "self-consistency figure can be quoted), receiving two texts and nothing else — it does "
      "not see the image, so it measures AGREEMENT with the human annotation, which is the "
      "reference; where the annotator is wrong, the pipeline pays. The rubric is completeness "
      "0–4, correctness 0–4, specificity 0–2, and the score reported everywhere is their SUM, "
      "so every point can be audited against the criteria beside it. The judge also returns "
      "a holistic score of its own; it does not coincide with the sum of its own criteria "
      f"(mean gap {_hol['mean_gap']:+.2f} over {_hol['n']} judgements, below the sum in "
      f"{_hol['below']}, above in {_hol['above']}, equal in {_hol['equal']}): a second, "
      "stricter judgement, kept in the report and not used as the score.")
    _js = open(os.path.join(os.path.dirname(DOCS), "slice_quality", "examples",
                            "judge_descriptions.py"), encoding="utf-8").read()
    code(_re.sub(r'\\\n', '', _js.split('JUDGE_SYSTEM = """')[1].split('"""')[0]).strip())
    p("Two design points. First, the prompt states that extra detail the reference does not "
      "mention — IoU values, hole counts, exact positions — is NOT an invention and counts "
      "against the candidate only if it CONTRADICTS the reference: an earlier version "
      "penalised true numbers as hallucinations. Second, the negative control of §6.5 is kept: "
      "each annotation is also paired, by a seeded derangement, with the text of ANOTHER "
      "image, and the control pairs are shuffled in and indistinguishable. Whatever the judge "
      "scores on them is the noise floor, and only the gap above it is evidence.")
    table(["Text", "Images", "Genuine pairs", "Mismatched controls", "Gap"],
          [["narrative", str(_nar["n"]), f"{_nar['genuine']:.2f}", f"{_nar['control']:.2f}",
            f"{_nar['genuine'] - _nar['control']:+.2f}"],
           ["systematic patterns", str(_pat["n"]), f"{_pat['genuine']:.2f}",
            f"{_pat['control']:.2f}", f"{_pat['genuine'] - _pat['control']:+.2f}"]],
          widths=[1.7, 0.9, 1.3, 1.6, 0.9])
    p(f"On the control pairs specificity still pays {_nar['crit_c']['specificity']:.2f}/2 — "
      "the narrative uses precise language whatever the image — while completeness and "
      f"correctness fall to {_nar['crit_c']['completeness']:.2f} and "
      f"{_nar['crit_c']['correctness']:.2f}: those two carry the signal. Split by image set, "
      f"the narrative scores {S1['split']['val']:.2f} on the 43 val images and "
      f"{S1['split']['train']:.2f} on the 36 train ones; against the hand verdicts of §6.4 it "
      f"scores {S1['by_verdict'].get('ALTA', (0, 0))[0]:.2f} on ALTA, "
      f"{S1['by_verdict'].get('PARZIALE', (0, 0))[0]:.2f} on PARZIALE and "
      f"{S1['by_verdict'].get('BASSA', (0, 0))[0]:.2f} on BASSA — the BASSA images score "
      "highest because the contour notes of §9.2 were written for exactly them.")
    p("Measured before/after on the same 66 images while the facts of §9.2 were being added "
      "(contour notes, thin details, “absorbed” for “missed”): the narrative went from 5.7 to "
      "6.7 (+1.0), with the largest gain on correctness (+0.53/4), 32 images up, 4 down, 30 "
      "unchanged; the control floor rose from 1.6 to 2.1 — the new sentences are more generic "
      "and match a little any annotation that talks about edges — but the gap above it still "
      "grew, from 4.0 to 4.4. One documented judge error: on 2011_002559 the judge penalised "
      "as hallucinated four PERSON patches that are in the prediction; the instrument "
      "measures agreement with the annotation, not truth.")
    p("An instrument lesson that cost a run: the first re-run with a changed rubric produced "
      "results identical to two decimals, because the string replacement on the prompt had "
      "silently failed and the resume cache had served the old verdicts. The prompt is now "
      "printed and checked before a launch, every rubric writes to its own output name, and "
      "(§10.6) the cache is keyed by the candidate text as well.")

    h("9.4  A synthetic set with a negative class", 2)
    _sy, _sc = S1["synth"], S1["synth_control"]
    p("The 80 VOC annotations all describe a defect, so they measure how much the pipeline "
      "SEES and never how well it stays silent when there is nothing to see. A set of 21 "
      "images with hand-altered masks fills that gap: 11 annotations describe catastrophic "
      "failures, 10 declare the prediction perfect. The set went through the same chain — "
      "Block 1, mask facts, narrative; the pattern branch by induction of the certified "
      "slices and the validated hypotheses, with no re-certification, since 21 images "
      "certify nothing — and the same judge. On the declared-correct half both texts score "
      f"{_sy['narrative']['good']:.2f}: the pipeline says “No segmentation error was found in "
      f"this image” and the judge agrees. Over the set the narrative scores "
      f"{_sy['narrative']['all']:.2f} (declared-wrong half {_sy['narrative']['bad']:.2f}) and "
      f"the pattern branch {_sy['pattern']['all']:.2f} ({_sy['pattern']['bad']:.2f}).")
    p("Two caveats are part of the result. The control pairing is gentler on the negative "
      "class: when a declared-perfect image is lent the text of ANOTHER perfect image, that "
      f"text is still true — it happens in {_sc['same']} of the {_sc['good_n']} controls and "
      f"lifts the floor from {_sc['floor_without']:.2f} to {_sc['floor_with']:.2f}; on the "
      f"wrong half the floor stays at {_sc['floor_bad']:.2f}. And a data check "
      "(docs/check_synth_dataset.py) found that the first 11 predictions of the set are the "
      "ground truths of OTHER VOC images pasted in — consecutive ids, IoU 1.000 in five cases "
      "— a pairing by list position rather than by content. The comparison page generates its "
      "warning from that check (self-IoU < 0.35) and will drop it when the folder is "
      "regenerated; on those 11 images the pipeline describes an artefact faithfully, not a "
      "realistic segmentation error.")

# ---- 9.5 a length-neutral score (judge_claims.py, October 2026) ----
from _revision_numbers import load_claims as _load_claims
CLM = _load_claims()
if CLM:
    _cs, _cd, _cc, _cv = CLM["summary"], CLM["duplicate"], CLM["correlation"], CLM["verdicts"]
    h("9.5  A length-neutral score: the annotation's facts, checked one by one (October 2026)", 2)
    p("The rubric of §9.3 tells the judge not to reward length, but a sentence in a prompt does "
      "not remove the bias: completeness and specificity are judged on the whole candidate, and "
      "a longer text offers more to match. A second score was therefore built so that length "
      "cannot buy points (slice_quality/examples/judge_claims.py, inst run, Gemma-4-12B in "
      "4-bit, the work split across two GPUs). It compares the human annotation with the "
      "pipeline's narrative only. The systematic-pattern text is not scored: it says which "
      "certified situation an image belongs to, not what went wrong in it, so checking the "
      "annotator's facts in it would measure a different thing.")
    bullet("Facts. Gemma splits each human annotation into its atomic facts — one object, one "
           "thing right or wrong about its mask, where — once per annotation, so the genuine, "
           "the control and the duplicated narrative of an image are all checked against the "
           "same list. Synonyms for one problem are one fact. %d annotations, %.1f facts on "
           "average (%d–%d)." % (CLM["n_annotations"], CLM["facts_mean"], CLM["facts_min"],
                                CLM["facts_max"]))
    bullet("Verdicts. For each fact the judge says whether the narrative states it, states it in "
           "part, contradicts it, or leaves it out; whatever else the narrative says is ignored "
           "unless it contradicts a fact.")
    bullet("Score, computed in code: 10 × max(0, stated + ½ partial − contradicted) / number of "
           "facts. The denominator comes from the human text; the price, declared, is that "
           "errors invented about objects the annotator did not mention are not penalised.")
    _ft, _v1 = CLM.get("fact_types", {}), CLM.get("v1", {})
    if _ft and _v1.get("all_genuine"):
        bullet("Glossary. The narrative is written in a fixed technical vocabulary, and a first "
               "version of the judge did not read it: of the 46 val facts it did not find stated, "
               "a reading by hand put about 27 in the narrative in other words (“spilling over the "
               "object, 30% of its area in excess” for “the background around the horse is "
               "labelled HORSE”; “heavily imprecise boundaries, many details lost” was even "
               "marked contradicted against “a coarser boundary, missing details”). The "
               "verification prompt now carries a glossary of the narrative's phrases and what "
               "each one states (spilling over = over-extended, background labelled as the "
               "class; thin fragments not traced = thin parts missed; patches hallucinated = "
               "false positives; class synonyms such as BICYCLE = bike), the same for genuine, "
               "control and duplicated pairings.")
        bullet("Fact types. Each fact is labelled once, with the facts: about the prediction "
               "(an object and what is wrong or right with its mask; %d facts), an overall "
               "judgement naming no specific problem (%d), or a remark on the ground truth "
               "itself (%d). The narrative reports errors object by object and takes the ground "
               "truth as the reference, so it cannot state the last two kinds; the score counts "
               "the facts about the prediction, and all facts when an annotation has none (“the "
               "prediction is perfect”). The score on every fact is kept and reported."
               % (_ft.get("prediction", 0), _ft.get("overall", 0), _ft.get("ground_truth", 0)))
    _lab = {"voc_narrative": "VOC (val + train36)", "synth_narrative": "synthetic",
            "all_narrative": "all annotated images"}
    table(["Annotated images", "narrative", "control", "gap", "images"],
          [[_lab.get(k, k.replace("_", " ")), "%.2f" % v["genuine"], "%.2f" % v["control"],
            "%+.2f" % (v["genuine"] - v["control"]), str(v["n"])] for k, v in _cs.items()],
          widths=[2.0, 1.0, 1.0, 1.0, 0.8])
    caption("Table 9.2. The facts score of the narrative on the 79 VOC and 21 synthetic annotated "
            "images, and on all 100 together, against a mismatched control (the narrative of "
            "another image, drawn within the same set).")
    if _v1.get("all_genuine"):
        _rows = []
        for _k, _lb in (("voc", "VOC (val + train36)"), ("synth", "synthetic"), ("all", "all annotated images")):
            _g, _c = _v1.get(_k + "_genuine"), _v1.get(_k + "_control")
            if _g and _c:
                _rows.append([_lb, "%.2f / %.2f" % (_g["v1"], _c["v1"]),
                              "%.2f / %.2f" % (_g["all_facts"], _c["all_facts"]),
                              "%.2f / %.2f" % (_g["now"], _c["now"]), str(_g["n"])])
        table(["Annotated images", "first version", "glossary, all facts",
               "glossary, facts on the prediction", "images"], _rows,
              widths=[1.9, 1.2, 1.3, 1.5, 0.7])
        caption("Table 9.2b. Genuine / control, step by step: the first version of the judge, the "
                "glossary alone (every fact counted), and the glossary with only the facts about "
                "the prediction counted (the score of Table 9.2). The control moves with the "
                "judge too: a gain that the control shared would be leniency, not reading.")
        _vv = CLM.get("split", {}).get("narrative_val", [float("nan")])[0]
        p("The glossary closes only part of the gap found by hand: on the 43 val images the "
          "score goes from 6.60 to %.2f, where reading the 46 unmatched facts suggested about "
          "8.3. The 12-billion-parameter judge in 4-bit still misses some paraphrases even with "
          "the glossary in front of it (on 2007_000332 “spilling over the object, 30%% of its "
          "area in excess” is still not read as “the background around the horse is labelled "
          "HORSE”), so the facts score is best taken as a lower bound on the agreement between "
          "narrative and annotation. Moving the type of the facts changes little (%d of %d "
          "facts are not about the prediction). On the synthetic set, whose annotations are "
          "mostly one overall verdict, the score moves little and slightly down (%.2f to %.2f)."
          % (_vv, _ft.get("overall", 0) + _ft.get("ground_truth", 0), sum(_ft.values()),
             _v1.get("synth_genuine", {}).get("v1", float("nan")),
             _v1.get("synth_genuine", {}).get("now", float("nan"))))
    _vn = _cv.get("voc_narrative", {})
    p("On VOC the narrative states %.0f%% of the annotators' facts, contradicts %.0f%% and leaves "
      "%.0f%% out. The control floor is lower than the rubric's (%.2f against %.2f), because a "
      "narrative about another image matches almost no fact of this one; the gap above it is of "
      "the same size."
      % (100 * _vn.get("stated", 0), 100 * _vn.get("contradicted", 0), 100 * _vn.get("absent", 0),
         _cs["voc_narrative"]["control"], RRS["inst_v3"]["narrative"]["control"]))
    if _cd.get("facts") and _cd.get("old"):
        table(["Check (narrative)", "facts score", "rubric of §9.3"],
              [["same text written twice: score unchanged",
                "%d / %d" % (_cd["facts"]["same"], _cd["facts"]["n"]),
                "%d / %d" % (_cd["old"]["same"], _cd["old"]["n"])],
               ["same text written twice: mean change (up / down)",
                "%+.2f (%d / %d)" % (_cd["facts"]["mean"], _cd["facts"]["up"], _cd["facts"]["down"]),
                "%+.2f (%d / %d)" % (_cd["old"]["mean"], _cd["old"]["up"], _cd["old"]["down"])],
               ["Spearman score vs length (VOC, genuine)",
                "%+.2f" % _cc["facts_genuine_narrative"][0], "%+.2f" % _cc["old_genuine_narrative"][0]]],
              widths=[3.2, 1.5, 1.5])
        caption("Table 9.3. Does length matter? Doubling a narrative without adding content is the "
                "clean test, on the 79 VOC and 21 synthetic annotated images.")
        p("Doubling a narrative leaves the facts score unchanged in %d of %d cases, and the "
          "changes go both ways (%d up, %d down, mean %+.2f): judge variability, not a length "
          "effect. The rubric changes in %d of %d and drifts upwards (%d up, %d down, mean "
          "%+.2f): the same content, written twice, earns more. The correlation with length is "
          "%+.2f for the facts score (p = %.2f). The comparison page shows the facts and the "
          "verdict on each under every narrative."
          % (_cd["facts"]["same"], _cd["facts"]["n"], _cd["facts"]["up"], _cd["facts"]["down"],
             _cd["facts"]["mean"], _cd["old"]["n"] - _cd["old"]["same"], _cd["old"]["n"],
             _cd["old"]["up"], _cd["old"]["down"], _cd["old"]["mean"],
             _cc["facts_genuine_narrative"][0], _cc["facts_genuine_narrative"][1]))

    _man = CLM.get("manual")
    if _man:
        _ms, _ag = _man["summary"], _man.get("agreement", {})
        _gg = _ag.get("genuine", {})
        _cf = _gg.get("confusion", {})
        p("Verdicts written by hand. Because the automatic judge still misread the narrative's "
          "wording, the same %d facts of the %d annotations (same fact list, same types, same "
          "formula) were also checked one by one by Claude — the assistant this pipeline was built "
          "with, so not an independent judge, and declared as such "
          "(slice_quality/examples/judge_claims_manual.py). To limit the bias, every genuine pairing "
          "was mixed with its control pairing (the narrative of another image, the same derangement "
          "as above), shuffled, and shown without the image name or the pairing; the rules were "
          "those of the automatic judge's prompt. The duplicate test is not repeated: a reader gives "
          "a text written twice the same verdicts by construction."
          % (sum(len(r["verdicts"]) for r in _man["rows"] if r["pairing"] == "genuine"),
             _ms["tutte"]["n"]))
        _rows = []
        for _k, _lb in (("voc_val", "VOC val"), ("voc_train", "VOC train36"), ("synth", "synthetic"),
                        ("tutte", "all annotated images")):
            _v = _ms.get(_k)
            if _v:
                _rows.append([_lb, "%.2f" % _v["genuine"], "%.2f" % _v["control"], "%.2f" % _v["gemma"],
                              "%.2f" % _v["gemma_control"], str(_v["n"])])
        table(["Annotated images", "by hand", "control", "Gemma", "Gemma control", "images"], _rows,
              widths=[1.9, 0.9, 0.9, 0.9, 1.1, 0.8])
        _nst = sum(1 for r in _man["rows"] if r["pairing"] == "genuine" and r.get("gemma_same_text") is False)
        caption("Table 9.2c. The facts score with verdicts written by hand, beside the automatic "
                "judge (Gemma-4-12B with the glossary), on the same facts and the same pairings."
                + (" The hand verdicts are on narrative v5; Gemma's, on %d of the images, are on the "
                   "previous text (see below)." % _nst if _nst else ""))
        _vv = _man.get("verdicts", {}).get("voc", {})
        p("By hand the VOC narrative states %.0f%% of the annotators' facts, states %.0f%% in part, "
          "contradicts %.0f%% and leaves %.0f%% out; the score is %.2f against a control of %.2f, "
          "where Gemma gives %.2f against %.2f. The control barely moves between the two judges, so "
          "the hand verdicts are not lenient across the board. Fact by fact the two judges agree on %.0f%% "
          "of the genuine facts (Cohen's kappa %.2f): of the %d facts stated by hand, Gemma "
          "marks %d absent and %d contradicted; of the %d stated in part by hand, it marks %d "
          "absent. The disagreement is almost all one way — a paraphrase the automatic judge does "
          "not recognise — which is what the first reading by hand had found. The control pairings say "
          "less about this than they seem to: a narrative about another image rarely talks about "
          "the same objects, so it cannot catch a reader who is lenient only on paraphrase; the "
          "check that remains is the verdicts themselves, which the comparison page shows fact by "
          "fact, with Gemma's beside every fact on which it differs."
          % (100 * _vv.get("stated", 0), 100 * _vv.get("partial", 0), 100 * _vv.get("contradicted", 0),
             100 * _vv.get("absent", 0), _ms["voc"]["genuine"], _ms["voc"]["control"],
             _ms["voc"]["gemma"], _ms["voc"]["gemma_control"], 100 * _gg.get("agreement", 0),
             _gg.get("kappa", 0), sum(v for k, v in _cf.items() if k.startswith("stated|")),
             _cf.get("stated|absent", 0), _cf.get("stated|contradicted", 0),
             sum(v for k, v in _cf.items() if k.startswith("partial|")), _cf.get("partial|absent", 0)))
        # the October revisions of the text (v5: which class took the place of which; v6: where
        # exactly, and the small things). Each paragraph reads its own pair of narrative files.
        _INST = _out_path("slice_quality_inst")
        _cm = os.path.join(_INST, "claims_manual")

        def _narr(tag):
            """{(set, image): text} of narrative version ``tag`` ('' = the current one)."""
            out = {}
            for _set, _f in (("voc", "error_narratives%s.json"),
                             ("synth", os.path.join("synth", "narratives_synth%s.json"))):
                _p = os.path.join(_INST, _f % tag)
                if not os.path.exists(_p):
                    return None
                for k, v in json.load(open(_p, encoding="utf-8")).items():
                    out[(_set, k)] = v["text"] if isinstance(v, dict) else v
            return out

        def _missed(tag):
            _p = os.path.join(_cm, "missed_facts_categories%s.json" % tag)
            return json.load(open(_p, encoding="utf-8"))["facts"] if os.path.exists(_p) else None

        _gen = [(r["set"], r["image"]) for r in _man["rows"] if r["pairing"] == "genuine"]
        _n4, _n6 = _narr("_v4"), _narr("")
        _n5 = _narr("_v5") or _n6
        _mf4, _mf6 = _missed("_v4"), _missed("")
        _mf5 = _missed("_v5") or _mf6
        # verdict moves per revision, from the full history of hand verdicts (a later revision's
        # line replaces an item's earlier one in the score, not in the history)
        _hist = {}
        for _l in open(os.path.join(_cm, "verdicts.jsonl"), encoding="utf-8"):
            if _l.strip():
                _o = json.loads(_l)
                _t = _o.get("note", "")[:2]
                if _t in ("v5", "v6"):
                    _hist[(_t, _o["item"])] = _o.get("note", "")
        _moves = lambda tag: sum(n.count("->") for (t, _i), n in _hist.items() if t == tag)
        _nst = sum(1 for r in _man["rows"] if r["pairing"] == "genuine" and r.get("gemma_same_text") is False)
        if _n4 and _mf4:
            _chg = sum(1 for k in _n5 if _n4.get(k) != _n5[k])
            _ann5 = sum(1 for k in _gen if _n4.get(k) != _n5.get(k))
            _a4, _a5 = (sum(f["category"] == "A" for f in x) for x in (_mf4, _mf5))
            p("Which class took the place of which (October 2026, narrative v5). Of the %d facts the "
              "hand verdicts found not fully stated, the largest group (%d) was of one kind: part of an "
              "object labelled with another class — “the DININGTABLE region is often misclassified as "
              "CHAIR”. The narrative told the two halves apart (“the CHAIR has been under-segmented, 46%% "
              "missing” and, elsewhere, “three small SOFA patches have been hallucinated”) and never said "
              "that the patches lay on the chair. mask_facts now measures, for every ground-truth object, "
              "the other class predicted on the largest part of it (relabelled_as, from 3%% of its area "
              "and 200 px, the thresholds the narrative already uses) and, for every hallucinated patch, "
              "the object it lies on (over_gt, half of the patch at least); the narrative says it once, on "
              "the patch or on the object: “Parts of the ground truth SOFA have been labelled CHAIR "
              "instead (three CHAIR patches on the right edge)”, “Part of the DININGTABLE (31%% of its "
              "area) has been labelled CHAIR instead”. %d of the %d narratives changed, %d of the %d "
              "annotated images among them. Their facts were read again by hand, with the same blind "
              "protocol: %d verdicts moved, all from partial or absent to stated and none the other way, "
              "and the facts of this kind still missing fell from %d to %d."
              % (len(_mf4), _a4, _chg, len(_n5), _ann5, _ms["tutte"]["n"], _moves("v5"), _a4, _a5))
        if _n6 is not _n5 and _narr("_v5"):
            _chg6 = sum(1 for k in _n6 if _n5.get(k) != _n6[k])
            _ann6 = sum(1 for k in _gen if _n5.get(k) != _n6.get(k))
            _w = lambda d: sum(len(t.split()) for t in d.values()) / len(d)
            _cat = lambda fs, c: sum(f["category"] == c for f in fs)
            p("Where exactly, and the small things (October 2026, narrative v6). The next facts the "
              "narrative missed were about place and size: the CORNER of a defect (“incomplete in the "
              "top-left corner”), WHERE an outline is ragged (“coarse boundaries, especially towards the "
              "bottom-right”), two classes on one object (“misclassified with classes DOG and COW”), "
              "small hallucinated patches (“a small PERSON region on the right”) and holes in a region "
              "that is otherwise correct. All five are again read from the masks: a corner is named "
              "when one cell of a 3×3 grid on the object's box holds half of the error outside the "
              "middle cell (the sides stay as before otherwise); the ragged part of an outline when one "
              "cell holds 40%% of the prediction–ground-truth disagreement along the edge; up to two "
              "other classes per object, also on objects swallowed by a merge; predicted patches between "
              "50 and 200 px, which Block 1 never makes into instances; and only the holes the object "
              "itself does not have — a prediction equal to a ground truth with a hole in it has made no "
              "error. Saying also the smaller of two opposite defects (“and a smaller gap of 7%%”) was "
              "tried and left out: it changed 900 sentences for about one fact of the annotations. "
              "%d of the %d narratives changed (%d of the annotated images), from %.0f to %.0f words on "
              "average; read again by hand, %d verdicts moved, again none the wrong way, and the facts "
              "not fully stated fell from %d to %d (part of an object with another class %d → %d, where "
              "exactly %d → %d, details below the size of an instance %d → %d). Two facts the narrative "
              "seemed to miss turned out to be the annotation's: on a synthetic image and on 2011_003151 "
              "the masks contain no DININGTABLE where the note puts one and no AEROPLANE at all, and the "
              "VOC palette paints DININGTABLE (192,128,0) almost like PERSON (192,128,128) and CHAIR "
              "(192,0,0) almost like AEROPLANE (128,0,0). The hand score above is the one on the current "
              "text. Gemma was not run again: on %d of the annotated images its verdicts refer to an "
              "earlier narrative, so the fact-by-fact agreement above is computed on the other pairings "
              "only, and the plausibility of the cause in §12 was likewise measured with an earlier text."
              % (_chg6, len(_n6), _ann6, _w(_n5), _w(_n6), _moves("v6"), len(_mf5), len(_mf6),
                 _cat(_mf5, "A"), _cat(_mf6, "A"), _cat(_mf5, "C"), _cat(_mf6, "C"),
                 _cat(_mf5, "D"), _cat(_mf6, "D"), _nst))

# ===========================================================================
# 10. Revision of Block 1 after the human comparison (September 2026)
# Every number below is read from the run files by docs/_revision_numbers.py.
# ===========================================================================
import sys as _sys
_sys.path.insert(0, DOCS)
from _revision_numbers import load as _load_rev
try:
    N = _load_rev()
except Exception as _ex:            # the section needs the revised run's files
    N = None
    print("  (revision numbers unavailable, section 10 skipped:", _ex, ")")

if N:
    J, C = N["judge"], N["criteria"]
    tb, ta = N["tax_before"], N["tax_after"]

    h("10. Revision of Block 1 after the human comparison (September 2026)", 1)
    p("§4–§6 describe the pipeline as it was first evaluated. Comparing its per-image "
      "narratives (§9) against the 79 hand-written annotations exposed a defect in "
      "Block 1 that no aggregate metric had shown, and fixing it changed the error "
      "population enough to deserve its own section. Everything here was run as a separate, "
      "reproducible configuration (config.TypedErrorConfig.claim_cov, switched on by "
      "SQ_CLAIM_COV=0.5, written to outputs/slice_quality_cov05b) so that the reference run "
      "the earlier sections report is untouched and the two can be compared instance by "
      "instance.")

    h("10.1  The symptom: “missed entirely” for objects that were not missed", 2)
    p("The most frequent sentence in the generated narratives was “The ground truth X region "
      "has been missed entirely”. Measured on the 80 annotated images against the masks, it "
      "was false more often than true: of the 223 FN instances, 52% were covered for at least "
      "half of their area by a prediction of their own class, another 8% by a prediction of "
      "another class, and only 27% were genuinely background. The object had not been missed; "
      "it had been swallowed into a neighbouring blob and had lost its outline — which is what "
      "the annotators write (“the two boats are merged”, “the person is labelled as "
      "motorbike”).")
    p("The cause is the claim test of the two-stage matcher (§4.1). Split and merge fire on "
      "IoU ≥ τ_frag = 0.25 between a prediction and a GT region. IoU is blind to asymmetric "
      "containment: a small GT swallowed at 70% by a blob that also covers a large GT has "
      "IoU ≈ 0.1 with that blob, claims nothing, and falls through to FN. Beyond IoU itself, "
      "the paper the block follows, calls a GT segment missed only when NO pixel of it is a "
      "true positive (evaluator.get_extent_segment_errors); the instance-level rule was "
      "stricter than the pixel-level rule it was meant to lift.")

    h("10.2  The rule: a containment claim beside the IoU claim", 2)
    p("For the regions the IoU rule leaves orphan, a second claim is added: a GT region also "
      "claims the prediction that holds at least claim_cov = 0.5 of its area, and a prediction "
      "of the same class that lies at least 0.5 inside a GT region claims that region. At 0.5 "
      "the containing region is unique — two disjoint masks cannot each hold more than half of "
      "the same region — so the containment claims are as well-posed as τ_match = 0.5 itself. "
      "Three consequences follow in decompose_image:")
    bullet("a prediction whose only claim is a GT it contains becomes that GT's 1↔1 pair "
           "(boundary / extent / confusion by the usual pixel split) instead of one FP plus "
           "one FN;")
    bullet("a GT whose best claimer lies mostly inside it is paired with that claimer, and any "
           "further claimer is a split fragment of it;")
    bullet("a prediction with two or more claims is a merge, as before — but every GT it "
           "swallows is now KEPT as an instance of its own (subtype MERGE, from_gt_mask, "
           "swallowed_by, covered_frac), pooled on the GT mask like a FN. A merge therefore "
           "yields 1 + M instances. This is declared double counting, and it replaces an "
           "undeclared one: under the IoU rule every over-extended mask already produced an "
           "FP and an FN.")
    p("Why keep the swallowed objects as instances at all: the first attempt did not, and the "
      "cluster branch lost its substrate. Thirteen of the fourteen certified cluster slices of "
      "the reference run were FN slices; folding 1 800 FN instances into their blobs left "
      "four certified slices, because the blob's embedding is dominated by the large object "
      "it mostly is, not by the small one it swallowed. A slice such as “small persons beside "
      "large persons get absorbed” can only be found on the swallowed objects' side. Keeping "
      "them restored ten certified slices, six of them MERGE slices made mostly of swallowed "
      "objects.")
    p("A bug found on the way: a merge on an IoU-matched prediction did not remove the "
      "swallowed GTs from the FN set (matched_gts.update was missing in that branch), so those "
      "objects were counted twice. With claim_cov=None the revised code reproduces the "
      "reference run instance for instance except for this.")

    h("10.3  Effect on the error population (VOC2012 val)", 2)
    table(["Type", "IoU rule (reference)", "Containment rule (revised)"],
          [["FN", str(tb["fn"]), f"{ta['fn']}  (−{100 - 100 * ta['fn'] // tb['fn']}%)"],
           ["FP", str(tb["fp"]), str(ta["fp"])],
           ["merge (blobs)", str(tb["merge"]), str(N["merge_blobs"])],
           ["merge (swallowed objects, GT side)", "—", str(N["merge_gt_side"])],
           ["split", str(tb["split"]), str(ta["split"])],
           ["extent / boundary", f"{tb['extent']} / {tb['boundary']}",
            f"{ta['extent']} / {ta['boundary']}"],
           ["confusion", str(tb["confusion"]), str(ta["confusion"])],
           ["correct (reference population)", str(N["ref_before"]), str(N["ref_after"])]],
          widths=[2.6, 1.9, 2.4])
    p(f"{N['pairs']} pairs were born by containment ({N['pairs_by_type'].get('extent', 0)} "
      f"extent, {N['pairs_by_type'].get('boundary', 0)} boundary, "
      f"{N['pairs_by_type'].get('confusion', 0)} confusion), each of them an FP and an FN "
      f"before. {N['merge_mixed']} of the {N['merge_blobs']} merge blobs mix classes: a table "
      "swallowed by a chair blob is a merge and a confusion at once, and the narrative now "
      "says so (“among them four DININGTABLE regions now labelled CHAIR”). Of the FN that "
      "remain on the 80 annotated images, two thirds are covered by less than 10% of any "
      "prediction: “missed entirely” is finally true when it is said.")
    p(f"Method note. {N['ref_moved']} of the {N['ref_before']} CORRECT references "
      f"({100 * N['ref_moved'] / N['ref_before']:.0f}%) become MERGE, because the blob, at "
      "IoU ≥ 0.85 on its own object, also holds a small GT (adjacent cows; a rider inside the "
      f"motorbike). The reference population of ω² and of the contour notes shrinks to "
      f"{N['ref_after']} regions. This is the existing convention for IoU claims applied "
      "consistently, not a new rule, but it is a population shift a reader of §6 should know.")

    h("10.4  Merges measured against the union they merged", 2)
    p("A merged mask has no single ground truth to be held to. mask_facts measured its "
      "contour against the most similar single GT, and the narrative said nothing about it "
      "unless the blob had an IoU match of its own (257 of the 559 blobs had none). "
      "2007_000121 is the clean case: two monitors side by side, 5 px apart, one predicted "
      "mask over both. Against either monitor the mask has IoU 0.44 — below τ_match, so it "
      "is a merge under both rules — and the sentence stopped at “merges two ground truth "
      "regions into a single mask”. The annotator wrote about under-segmented stands and "
      "less defined boundaries. Against the UNION of the two monitors the mask has IoU 0.80, "
      "E_bnd > E_ext, boundary IoU 0.08: exactly the annotator's reading.")
    p("Block 1 now records which GTs a merge swallowed (gt_ids); mask_facts measures every "
      "merge blob against their union (IoU, E_bnd/E_ext, boundary IoU, missed and excess "
      "pixels, thin details lost) and the sentence continues: “…; taken together, the merged "
      "regions are segmented with heavily imprecise boundaries (IoU 0.80, boundary IoU "
      "0.08)”. The adjective itself was corrected on the way: it was graded by area IoU "
      "alone, which is scale-dependent and on a large object stays high while the outline is "
      "wrong (“slightly coarser boundaries” at boundary IoU 0.08). It is now demoted one step "
      "below boundary IoU 0.30 — the worst quartile of the reference contours, the cut the "
      "contour notes already use — and two steps below 0.15. On val, 86 of the 318 pairs "
      "worded “slightly coarser” had a boundary IoU under 0.30.")

    h("10.5  What the judge says, paired on the same images", 2)
    p("Same judge, same rubric and same control design as §9.3 (Gemma-4-12B local, "
      "completeness 4 + correctness 4 + specificity 2, mismatched control pairs shuffled "
      "in). Scores are paired image by image between the reference run and the revised one; "
      "the 43 val images are the ones whose typing was re-derived, the 36 train images were "
      "re-typed and re-described with the same code.")
    _r = lambda d: f"{d['a']:.2f} → {d['b']:.2f}"
    _u = lambda d: f"{d['up']} / {d['dn']} / {d['eq']}"
    table(["Text", "Images", "Reference → revised", "up / down / same"],
          [["narrative", f"all {J['narrative']['all']['n']}", _r(J['narrative']['all']), _u(J['narrative']['all'])],
           ["narrative", f"val {J['narrative']['val']['n']}", _r(J['narrative']['val']), _u(J['narrative']['val'])],
           ["narrative", f"train {J['narrative']['train']['n']}", _r(J['narrative']['train']), _u(J['narrative']['train'])],
           ["pattern", f"all {J['pattern']['all']['n']}", _r(J['pattern']['all']), _u(J['pattern']['all'])],
           ["pattern", f"val {J['pattern']['val']['n']}", _r(J['pattern']['val']), _u(J['pattern']['val'])],
           ["pattern", f"train {J['pattern']['train']['n']}", _r(J['pattern']['train']), _u(J['pattern']['train'])],
           ["control floor, narrative / pattern", "all",
            f"{J['narrative']['control_a']:.2f} / {J['pattern']['control_a']:.2f} → "
            f"{J['narrative']['control_b']:.2f} / {J['pattern']['control_b']:.2f}", "—"]],
          widths=[1.9, 1.1, 2.0, 1.5])
    p(f"On the narrative the gain is carried by the two criteria that carry the signal: "
      f"completeness {C['completeness'][0]:.2f} → {C['completeness'][1]:.2f} and correctness "
      f"{C['correctness'][0]:.2f} → {C['correctness'][1]:.2f} out of 4, specificity unchanged "
      f"at {C['specificity'][1]:.2f}. The images that gain most are the containment cases "
      "themselves (2007_000129: 4 → 10, “four BICYCLE regions now labelled PERSON”). The "
      "pattern branch is flat overall: it lost some certified FN slices and gained MERGE "
      f"slices ({N['slices_before']['cluster']} → {N['slices_after']['cluster']} cluster "
      f"slices, {N['slices_before']['language']} → {N['slices_after']['language']} language "
      f"slices; images with at least one pattern {N['cov_before']['with_pattern']} → "
      f"{N['cov_after']['with_pattern']} of {N['cov_after']['images']}, share of errors "
      f"covered {N['cov_before']['share']:.2f} → {N['cov_after']['share']:.2f}). The "
      f"remaining loss has a precise cause: the {N['pairs']} containment pairs carry their "
      "instance on the prediction side, with the embedding pooled on the blob, where the old "
      "FN had its own on the GT mask.")
    p(f"Synthetic set (§9.4, 21 images, half declared perfect): narrative "
      f"{N['synth_before']['narrative']['all']:.2f} → {N['synth_after']['narrative']['all']:.2f}, "
      f"pattern {N['synth_before']['pattern']['all']:.2f} → {N['synth_after']['pattern']['all']:.2f}; "
      f"on the declared-correct half both branches stay at "
      f"{N['synth_after']['narrative']['good']:.2f}. The drop sits entirely in the declared-wrong "
      "half, whose predictions are other images' ground truths pasted in (a dataset defect "
      "documented on the comparison page): there the containment rule says “TRAIN labelled "
      "SHEEP” where the annotator wrote “missed”, and pixel for pixel the rule is right, but "
      "the annotation is the reference.")

    h("10.6  Two instrument fixes that surfaced during the re-runs", 2)
    bullet("Internal holes of a predicted mask were a fact computed by a side script for val "
           "only; re-generating train36 from scratch silently dropped the sentence “the "
           "predicted mask also has two holes” and cost up to 4 judge points on an image. "
           "Holes are now a mask fact in mask_facts.add_mask_facts (≥ 20 px, unchanged rule), "
           "so every runner — val, train36, synthetic — measures them.")
    bullet("The judge's resume cache was keyed by (image, kind, pairing) alone, so a "
           "re-generated text would have kept the old verdict. It now reuses a judgement only "
           "when the candidate text is identical; after a change only the changed texts are "
           "re-judged (14 minutes instead of two hours for 304 pairs).")
    p("Reproducibility: retype_pickle.py re-runs Block 1 on CPU and joins the new typing onto "
      "the cached DINOv2 embeddings and Qwen tags by instance id (the connected components do "
      "not change), so the whole revision needed no re-embedding; the GPU was used only for "
      "the judge, the train36/synthetic inductions and their Qwen groundings.")


# ---- 10.7 GT instances from VOC object masks (numbers from _revision_numbers.load_inst) ----
from _revision_numbers import load_inst as _load_inst
try:
    NI = _load_inst()
except Exception as _ex:
    NI = None
    print("  (instance-run numbers unavailable, 10.7 skipped:", _ex, ")")

if N and NI:
    _a, _j, _t = NI["amb"], NI["judge"], NI["tax"]
    h("10.7  Ground-truth instances from VOC object masks: one occluded object is one object", 2)
    p("A reader of the narratives noticed that they go wrong around occlusion: an object "
      "partly covered by another comes out as a merge, a miss, or two objects. The cause is "
      "the definition of instance in §4.1 — a connected component of the class mask, since "
      "VOC is semantic segmentation. Occlusion breaks that definition both ways: an object cut "
      "in two by something in front of it becomes two GT instances (and a prediction covering "
      "both parts becomes a merge; 2007_000121's two monitors were really two objects, but the "
      "same wording covers a horse behind a fence), while two objects of one class that touch "
      "become one. VOC ships the answer: SegmentationObject, one id per object, for every image "
      "of the segmentation subset.")
    p(f"Measured on val through the pipeline's substrate (520 crop, ≥ 200 px): "
      f"{_a['components']} GT components stand for {_a['objects']} objects; "
      f"{_a['cut']} objects are cut into ≥ 2 components ({_a['parts']} parts) in "
      f"{_a['img_cut']} images ({100 * _a['img_cut'] / _a['images']:.0f}% of val) — occlusion, "
      f"but also an object's own thin structure (spokes, chair legs); {_a['touching']} components "
      f"hold ≥ 2 touching objects ({_a['touching_objects']} objects, {_a['img_touch']} images). "
      f"Held against the object ids, {_a['false_merge']} of the {_a['false_merge'] + _a['real_merge'] + _a['unknown_merge']} "
      f"merge blobs of §10.3 were one occluded object, not several, and {_a['partial_miss']} of "
      f"{_a['partial_miss'] + _a['whole_miss']} “missed” GT instances were parts of an object "
      "another part of which had been matched.")
    p("The change is confined to the adapter, which is the point: Block 1 works on masks and "
      "does not care whether they are connected. With gt_instances=\"object\" "
      "(SQ_GT_INSTANCES=object) a GT instance is an object — class by majority of its pixels, "
      "n_parts recorded — while predictions stay connected components, because a semantic model "
      "has nothing better to offer; images without an object mask (the synthetic set) fall back "
      "to components. mask_facts then names the OCCLUDER of a cut object — the other-class object "
      "touching at least two of its parts; when none does, the object is cut by its own "
      "structure and the narrative keeps quiet — and the sentence gains a clause: “The ground "
      "truth DININGTABLE region in the center of the scene, partly hidden behind the CHAIR, has "
      "been segmented with heavily imprecise boundaries (IoU 0.46, boundary IoU 0.22)”. The "
      "dataset dependence is therefore explicit: any dataset with instance ids (COCO, "
      "Cityscapes, ADE20K) gets the exact behaviour through its adapter; a semantic-only "
      "dataset keeps the component approximation, declared as such.")
    p("Re-typing reused the cache where the masks did not change — every predicted component, "
      "and every GT object whose mask equals an old component — and embedded and tagged only "
      "the rest (664 objects: cut, touching, or never embedded). Population: FN "
      f"{ta['fn']} → {_t['fn']}, merge {N['tax_after']['merge']} → {_t['merge']} (blobs plus "
      f"swallowed objects), split {ta['split']} → {_t['split']} — a cut object matched by one "
      f"part now has its other predicted fragments as split rather than FP — boundary / extent "
      f"{ta['boundary']} / {ta['extent']} → {_t['boundary']} / {_t['extent']}, references "
      f"{N['ref_after']} → {NI['ref']}. {NI['cut_inst']} instances are cut objects; the occluder "
      f"was identified for {NI['occluded']} of them ("
      + ", ".join(f"{c.upper()} {k}" for c, k in NI['occluders']) +
      f"), and {NI['hidden']} of the 1 483 narratives carry a “partly hidden” clause "
      f"({NI['hidden79']} of the 79 annotated).")
    _r = lambda d: f"{d['a']:.2f} → {d['b']:.2f}"
    _u = lambda d: f"{d['up']} / {d['dn']} / {d['eq']}"
    table(["Text", "Images", "Reference → objects", "up / down / same", "Containment → objects"],
          [["narrative", f"all {_j['narrative']['vs_ref']['n']}", _r(_j['narrative']['vs_ref']), _u(_j['narrative']['vs_ref']), _r(_j['narrative']['vs_cov'])],
           ["narrative", f"val {_j['narrative']['val']['n']}", _r(_j['narrative']['val']), _u(_j['narrative']['val']), "—"],
           ["narrative", f"train {_j['narrative']['train']['n']}", _r(_j['narrative']['train']), _u(_j['narrative']['train']), "—"],
           ["pattern", f"all {_j['pattern']['vs_ref']['n']}", _r(_j['pattern']['vs_ref']), _u(_j['pattern']['vs_ref']), _r(_j['pattern']['vs_cov'])],
           ["control floor, narrative / pattern", "all",
            f"{_j['narrative']['control']:.2f} / {_j['pattern']['control']:.2f}", "—", "—"]],
          widths=[1.7, 1.0, 1.6, 1.3, 1.6])
    p(f"Correctness is where objects pay: {C['correctness'][0]:.2f} → {C['correctness'][1]:.2f} "
      f"→ {NI['criteria']['correctness']:.2f} out of 4 across the three runs, with completeness "
      f"flat ({NI['criteria']['completeness']:.2f}) and specificity flat "
      f"({NI['criteria']['specificity']:.2f}). The train images gain most "
      f"({_j['narrative']['train']['a']:.2f} → {_j['narrative']['train']['b']:.2f}), which is "
      "where the annotators' “the horse behind the fence” sentences were being contradicted.")
    p(f"The cost is on the cluster branch, again, and for the same reason as §10.2: "
      f"{N['slices_after']['cluster']} → {NI['slices']['cluster']} certified cluster slices "
      f"({NI['slices']['language']} language slices), images with a pattern "
      f"{N['cov_after']['with_pattern']} → {NI['cov']['with_pattern']}, share of errors covered "
      f"{N['cov_after']['share']:.2f} → {NI['cov']['share']:.2f} (both runs certified by SliceLens's slope "
      "rule, as when they were compared; the official run is in §18.5). Merging an object's parts "
      "into one instance halves the GT-side population (2 207 → "
      f"{NI['gt_side']}) and pools each embedding over a larger, more heterogeneous mask — "
      "exactly the small, homogeneous instances the GMM certified before. The judge does not "
      f"see it (pattern {_j['pattern']['vs_ref']['a']:.2f} → {_j['pattern']['vs_ref']['b']:.2f}), "
      "because the language rows carry that score; the discovery side does. Which of the two "
      "typings the thesis reports is a choice between a narrative that is right about "
      "occlusion and a cluster branch that finds more: the object typing is the true one, and "
      "the honest framing is that the cluster branch, as configured, needs many small "
      "instances, a limitation the object masks expose rather than create.")


# ---- 10.8 What the annotators name first (numbers from _revision_numbers.load_narr) ----
from _revision_numbers import load_narr as _load_narr
try:
    NN = _load_narr()
except Exception as _ex:
    NN = None
    print("  (narrative-revision numbers unavailable, 10.8 skipped:", _ex, ")")

if N and NN:
    _jn, _cn, _k = NN["judge"]["narrative"], NN["criteria"], NN["counts"]
    h("10.8  What the annotators name first: direction, side, and the regions that are fine", 2)
    p("Reading the 79 cards of the object run against the hand-written notes, sorted by the "
      f"judge's score, gave a clean picture of where the low scores were: {NN['low_before']} of "
      "the 79 narratives scored 6 or less, and five causes covered them. In order of weight: "
      "(1) a spill into the background told as “coarser boundaries” or “under-segmented” — "
      "Beyond IoU calls every wrong pixel within 2d of a true positive a boundary error, so a "
      "20-px spill along a whole side is “boundary” to the taxonomy and “over-extended on the "
      "right” to a person (11 images); (2) a correct region with no sentence at all, or a "
      "detail sentence with no subject (“Many fine details of the outline are lost”, of what?) "
      "(6); (3) same-class patches touching the object told as hallucinations, where the "
      "annotator sees the object's mask spilling (4); (4) a wrong-class patch under 200 px, "
      "dropped by the adapter and therefore unmentioned (4); (5) ordering and position: six "
      "tiny specks before the object the note is about, “scattered across the scene” for "
      "patches that are all in the right half, and — on the synthetic set — a blob covering "
      "half the image told after three small misses (3). The remaining low cards were "
      "annotation or judge noise (a note naming classes the prediction does not contain).")
    p("The common root of the first three is that the narrative said how wrong an outline was "
      "and the annotators say in which direction and where. Five changes, all in mask_facts and "
      "describe_errors, none in Block 1:")
    bullet("Direction and side on every pair, merge and correct region: from the missed and "
           "excess pixels, their share of the GT area and the sides of the GT box they lie on "
           "— “segmented with coarser boundaries, spilling over the object on the top and right "
           "(8% of its area in excess)”, “under-segmented, 82% of its area missing on the "
           "bottom”. Silent under 3% of the area, which is the ordinary raggedness the boundary "
           "IoU already grades.")
    bullet("A sentence for every correct region not already covered by a contour note, graded "
           "by boundary IoU and with the same clause — “segmented closely, spilling over the "
           "object on the top (5% of its area in excess) (IoU 0.93, boundary IoU 0.50)”. “No "
           "segmentation error was found” is kept only for regions at IoU ≥ 0.99, so the ten "
           "declared-perfect synthetic images keep it. The detail sentence names its object.")
    bullet("A same-class patch within 8 px of a GT region of that class is the object's mask "
           "spilling, not a hallucination: “The BUS mask also spills beyond the object in two "
           "small patches on the sides”.")
    bullet("Predicted patches under 200 px whose class the ground truth does not have there are "
           "counted in a closing line (“…there is also one tiny DININGTABLE patch on the DOG”), "
           "as the sub-threshold GT regions already were. Two traps, both hit and fixed: the "
           "patch must be looked at against the LOW-threshold ground truth, or a tiny patch "
           "sitting on an equally tiny GT region of its class reads as an error (the ten "
           "perfect synthetic predictions fell to 6.6/10 for exactly this), and slivers on "
           "VOC's void band are neither right nor wrong.")
    bullet("An opening verdict when the failure is global — “The prediction is largely wrong: "
           "76% of the annotated area has been labelled DOG, a class that is not in the "
           "image”, or “misses most of the annotated area: 82% of it is predicted as "
           "background” — above half of the annotated area; any group covering 15% of the "
           "frame is told first whatever its type; “on the right half of the scene” when all "
           "patches share a half; tiny hallucinated patches gathered in one closing line.")
    p(f"On val the narratives now carry {_k['spill']} spill clauses and {_k['missing']} missing-part "
      f"clauses over {_k['images']} images, {_k['correct']} sentences on correct regions, "
      f"{_k['spill_patch']} spilling patches, {_k['verdict']} opening verdicts and {_k['half']} "
      f"“half of the scene” locations; no image is left with “No segmentation error was found”.")
    _r = lambda d: f"{d['a']:.2f} → {d['b']:.2f}"
    _u = lambda d: f"{d['up']} / {d['dn']} / {d['eq']}"
    table(["Text", "Images", "Objects → revised narrative", "up / down / same", "Reference → revised"],
          [["narrative", f"all {_jn['vs_obj']['n']}", _r(_jn['vs_obj']), _u(_jn['vs_obj']), _r(_jn['vs_ref'])],
           ["narrative", f"val {_jn['val']['n']}", _r(_jn['val']), _u(_jn['val']), "—"],
           ["narrative", f"train {_jn['train']['n']}", _r(_jn['train']), _u(_jn['train']), "—"],
           ["control floor, narrative", "all", f"{_jn['control_a']:.2f} → {_jn['control']:.2f}", "—", "—"]],
          widths=[1.6, 1.0, 1.9, 1.3, 1.6])
    p(f"Completeness {_cn['completeness'][0]:.2f} → {_cn['completeness'][1]:.2f}, correctness "
      f"{_cn['correctness'][0]:.2f} → {_cn['correctness'][1]:.2f}, specificity "
      f"{_cn['specificity'][0]:.2f} → {_cn['specificity'][1]:.2f} out of 4 / 4 / 2; "
      f"{NN['ten_before']} → {NN['ten_after']} of the 79 narratives score 10, "
      f"{NN['low_before']} → {NN['low_after']} score 6 or less, and the control floor does not "
      "move — the gain is agreement, not verbosity. Against the reference run of §6 the "
      f"narrative goes from {_jn['vs_ref']['a']:.2f} to {_jn['vs_ref']['b']:.2f}, "
      f"{_jn['vs_ref']['up']} images up and {_jn['vs_ref']['dn']} down. On the synthetic set the "
      f"declared-wrong half rises from {NN['synth_before']['narrative']['bad']:.2f} to "
      f"{NN['synth_after']['narrative']['bad']:.2f} — the opening verdict is what those "
      f"annotations say first — and the declared-perfect half stays at "
      f"{NN['synth_after']['narrative']['good']:.2f}. Of the {NN['low_after']} cards still at 6 "
      "or less, most are disagreements the masks cannot settle: a note naming a class the "
      "prediction does not contain, or the opposite sign on the same object (the BUS the "
      "annotator calls slightly under-segmented, where the mask spills 5% on top). Six cards "
      "lost one or two points to a spill clause the judge did not find in the note — the "
      "extra detail the prompt says not to penalise and sometimes does.")
    from _revision_numbers import stale_cards as _stale
    p("Removed afterwards, on 26 September 2026: the two sub-threshold sentences — the one "
      "about ground-truth regions under the adapter's 200 px and the one about predicted "
      "patches — both said what the INSTRUMENT does (“below the size the pipeline "
      "represents as an object”), and a description of an image should not be about the "
      "tool that reads it. The facts are still measured in mask_facts and still counted in the "
      "narrative record; they are simply no longer said, and an image whose whole text was that "
      "one sentence now reads “No segmentation error was found in this image”. Every "
      "judge number in this section was measured with those sentences in place: on %d of the 79 "
      "cards the text has since changed, and both the comparison page and the examples of "
      "§11 declare that their score predates the change." % _stale())


# ============================================================================
#  12. Is the CAUSE plausible? (September 2026). Numbers from
#      docs/_revision_numbers.load_cause -- nothing here is typed in.
# ============================================================================
from _revision_numbers import load_cause as _load_cause
try:
    CZ = _load_cause()
except Exception as _ex:
    CZ = None
    print("  (cause-plausibility numbers unavailable, section 12 skipped:", _ex, ")")

if CZ and "voc" in CZ:
    _v = CZ["voc"]
    _a = _v["agree"]
    _sy = CZ.get("synth")

    h("12. Is the cause plausible? Scoring the other half of the deliverable "
      "(September 2026)", 1)
    p("Everything measured so far scores what the pipeline says WENT WRONG. But the two "
      "branches do not exist to describe an error: they exist to name the SITUATION a group "
      "of errors shares — “an object cut off at the image border” from the "
      "language branch, “small objects of class ‘person’ with imprecise "
      "boundaries” from the cluster branch — and that claim is the deliverable. "
      "This section scores it: given what goes wrong in an image, how plausible is the "
      "situation the system blames for it?")
    p("The question is put to the same local Gemma, on a rubric of two components, and it is "
      "put TWICE per image against two independent accounts of the same errors — the "
      "hand-written annotation and the pipeline's own narrative — with the judge never "
      "told which of the two it is reading. The agreement between those two scores is the "
      "result that matters: where they agree, the generated text can stand in for the human "
      "one when a failure mode is being judged, which is what the method needs on the "
      "1 483 images nobody annotated.")
    bullet("Consistency (0–5): do the classes, the error types and the circumstances named "
           "in the cause agree with the error description? Each contradiction costs a point.")
    bullet("Explanatory power (0–5): if the situation named were true of this image, would "
           "it account for the errors described — the main one, or only a marginal part? "
           "A cause so generic that it would fit any image loses here.")
    bullet("Plausibility is the two added, 0–10. The cause text carries the situation, the "
           "class and the error type it claims, and NOT the mask numbers (mean IoU, boundary "
           "IoU, fragments lost): those measure the very errors being described, and a judge "
           "given them would match measurements instead of weighing an explanation.")
    bullet("Chance calibration, as everywhere else: each pair has a control in which the same "
           "error description is given the cause of ANOTHER image, seeded, deranged and "
           "shuffled in. Only the gap above that floor is evidence.")
    _cv = CZ["coverage"]
    p("It can only be asked where a cause exists. Of the %d annotated images, %d have at least "
      "one certified pattern (%d val, %d train) and are the ones measured here; on %d others "
      "Block 1 types errors that fall in no certified slice, so nothing is blamed and there is "
      "nothing to judge; the remaining %d have no typed error at all. The comparison page says "
      "which of the three a card is in, rather than leaving the same silence for all of them. "
      "The synthetic set adds %d images, its branch induced as in train36."
      % (_cv["annotated"], _v["human"]["n"], _v["n_val"], _v["n_train"],
         _cv["no_pattern"], _cv["no_error"],
         (_sy or {}).get("human", {}).get("n", 0)))
    table(["Plausibility of the cause (0–10)", "errors told by the annotation",
           "errors told by the narrative"],
          [["genuine pairs", "%.2f" % _v["human"]["score"], "%.2f" % _v["pipeline"]["score"]],
           ["control floor (another image's cause)",
            "%.2f" % _v["human"]["control"], "%.2f" % _v["pipeline"]["control"]],
           ["gap above the floor",
            "%.2f" % (_v["human"]["score"] - _v["human"]["control"]),
            "%.2f" % (_v["pipeline"]["score"] - _v["pipeline"]["control"])],
           ["consistency / 5", "%.2f" % _v["human"]["cons"], "%.2f" % _v["pipeline"]["cons"]],
           ["explanatory power / 5", "%.2f" % _v["human"]["expl"], "%.2f" % _v["pipeline"]["expl"]],
           ["words per error description",
            "%.0f" % _v["human"]["words"], "%.0f" % _v["pipeline"]["words"]]],
          widths=[2.6, 2.0, 2.0])
    p("Five to six points above the noise floor: the situations the system certifies are not "
      "generic enough to fit any error, which is the first thing this measurement had to rule "
      "out. Then the branches separate, and this is the finding of the section.")
    _b = _v["by_branch"]
    table(["Cause proposed by", "told by the annotation", "told by the narrative", "images"],
          [["the language branch (a SliceLens hypothesis)",
            "%.2f" % _b["human"]["language"][0], "%.2f" % _b["pipeline"]["language"][0],
            str(_b["human"]["language"][1])],
           ["the cluster branch (a visual group)",
            "%.2f" % _b["human"]["cluster"][0], "%.2f" % _b["pipeline"]["cluster"][0],
            str(_b["human"]["cluster"][1])],
           ["both branches on the same image",
            "%.2f" % _b["human"]["both"][0], "%.2f" % _b["pipeline"]["both"][0],
            str(_b["human"]["both"][1])]],
          widths=[3.0, 1.6, 1.6, 0.8])
    p("A SliceLens hypothesis names a situation a reader can picture and check — an object "
      "cut off by the frame, one occluded by another, several overlapping objects of the same "
      "type — and it scores %.2f / %.2f. A cluster slice names a group and an error type, "
      "and when no attribute has been verified for it the description degenerates into "
      "“objects of class ‘person’ grouped by appearance only (no verified "
      "attribute) not detected”, which asserts almost nothing: %.2f / %.2f, and %s of the three "
      "lowest-scoring causes of the whole set are visual groups. The cluster branch is what "
      "FINDS the group; the language branch is what can SAY why. Figure 6 argued the two "
      "branches are complementary from their overlap — only 356 of the 2 354 covered "
      "instances are reached by both; this measures the complementarity from the other side, "
      "on the text a reader actually gets."
      % (_b["human"]["language"][0], _b["pipeline"]["language"][0],
         _b["human"]["cluster"][0], _b["pipeline"]["cluster"][0],
         ("none", "one", "two", "all")[sum(1 for r in sorted(
             (r for r in json.load(open(os.path.join(_RN_INST, "judge_cause.json"), encoding="utf-8"))["judgements"]
              if r.get("kind") == "pipeline" and r.get("pairing") == "genuine" and "score" in r),
             key=lambda r: r["score"])[:3] if "SliceLens situation" not in r["cause"])]))
    p("Agreement between the two accounts, image by image: mean difference %+.2f in favour of "
      "the narrative, mean absolute difference %.2f points, within one point on %d of the %d "
      "images and within two on %d, Pearson r = %.2f. %d causes are plausible (≥ 7) under "
      "both accounts and %d under neither. The direction is shared; the single image is not "
      "interchangeable."
      % (_a["diff"], _a["absdiff"], _a["within1"], _a["n"], _a["within2"], _a["r"],
         _a["both_high"], _a["both_low"]))
    p("Two caveats belong beside that +%.2f, because it is not a clean advantage. The narrative "
      "is %.0f words against the annotation's %.0f, so it offers the cause more to agree with "
      "— and more to contradict. More seriously, the narrative and the cause are both "
      "generated from Block 1's taxonomy and share its vocabulary (“merged”, "
      "“missed”, “imprecise boundaries”), so part of the higher consistency "
      "(%.2f against %.2f) is lexical alignment rather than better evidence. The honest reading "
      "is the agreement, not the ranking."
      % (_a["diff"], _v["pipeline"]["words"], _v["human"]["words"],
         _v["pipeline"]["cons"], _v["human"]["cons"]))
    _w = _v["worst"][0]
    p("The disagreements are informative rather than noisy. The largest, %s, scores %.1f with "
      "the annotation and %.1f with the narrative: the two accounts tell different parts of the "
      "image, and the same certified cause explains much more of one than of the other. The pair of "
      "scores measures, in effect, how much of an image a single certified situation accounts "
      "for, which is a property of the slice and not of either text."
      % (_w[1], _w[2], _w[3]))
    if _sy:
        _s = _sy["agree"]
        p("The synthetic set answers the question where the answer is known. Of its 21 images "
          "only %d fall inside an induced pattern, and NONE of them is one of the ten whose "
          "annotation declares the prediction perfect: where there is nothing to explain, "
          "nothing is blamed, which is exactly what the negative class exists to test. On the "
          "%d that are declared wrong the causes score %.2f with the annotation and %.2f with "
          "the narrative against a floor of %.2f / %.2f — clearly weaker than on VOC, and "
          "for a reason already documented in §9.4: eleven of those predictions are other "
          "images' ground truths pasted in, so no situation certified on val can honestly "
          "explain them. The agreement also loosens (r = %.2f, mean absolute difference %.2f), "
          "as it should on errors that are artefacts."
          % (_sy["human"]["n"], _sy["human"]["n"], _sy["human"]["score"],
             _sy["pipeline"]["score"], _sy["human"]["control"], _sy["pipeline"]["control"],
             _s["r"], _s["absdiff"]))
    if CZ.get("all"):
        _al = CZ["all"]["pipeline"]
        _ab, _ae = CZ["all"]["by_branch"]["pipeline"], CZ["all"]["by_etype"]
        _bd = CZ["all"]["bands"]
        h("12.1  The same question over the whole coverage of the pipeline", 2)
        p("The %d annotated images are where a human account exists, not where the pipeline "
          "works. Repeating the measurement on EVERY image that has a certified cause — "
          "%d of them, %d val and %d train36 — judged against the narrative alone, since "
          "outside the annotated set no human text exists (which is the reason the pipeline "
          "exists at all): the causes score %.2f against a control floor of %.2f, a gap of "
          "%.2f, with consistency %.2f/5 and explanatory power %.2f/5. The control was drawn "
          "on a seeded sample of 150 images rather than on all %d: the floor is a property of "
          "the judge and the corpus, not of the single image. %d of the %d images score 7 or "
          "more, %d sit at 5–6 and %d below 5."
          % (_v["human"]["n"], _al["n"], CZ["all"]["n_val"], CZ["all"]["n_train"],
             _al["score"], _al["control"], _al["score"] - _al["control"], _al["cons"],
             _al["expl"], _al["n"], _bd["high"], _al["n"], _bd["mid"], _bd["low"]))
        p("The wider measurement corrects the impression the 39 gave about the two branches. "
          "There the cluster causes scored %.2f against the language branch's %.2f, but on "
          "only %d images; over the full coverage the cluster branch reaches %.2f on %d images "
          "and the language branch %.2f on %d, and the images where BOTH branches fire score "
          "highest of all (%.2f on %d). The direction survives — a named situation still "
          "explains an error better than a visual group — but the gap is %.1f points, not "
          "%.1f, and the 39-image figure should be read as the small sample it is. "
          "That the two branches together score above either alone is the same complementarity "
          "of Figure 6, measured on the explanation instead of on the coverage."
          % (_b["human"]["cluster"][0], _b["human"]["language"][0], _b["human"]["cluster"][1],
             _ab["cluster"][0], _ab["cluster"][1], _ab["language"][0], _ab["language"][1],
             _ab["both"][0], _ab["both"][1], _ab["language"][0] - _ab["cluster"][0],
             _b["human"]["language"][0] - _b["human"]["cluster"][0]))
        table(["The cause claims an error of type", "plausibility", "images"],
              [[k, "%.2f" % m, str(n)] for k, (m, n) in
               sorted(_ae.items(), key=lambda kv: -kv[1][1])],
              widths=[3.0, 1.5, 1.2])
        p("Read down that table and the ranking is the pipeline's own honesty about itself: a "
          "cause that blames a spurious detection is the least convincing (%.2f), because "
          "“the model invented an object here” is the hardest thing for a situation "
          "to explain, while causes about merged regions, wrong extent or a wrong class score "
          "%.2f to %.2f — those are failures with a visible circumstance behind them. It is "
          "also worth saying that plausibility rises with the number of certified patterns on "
          "an image, from %.2f with one to %.2f with four or more: more of the image is "
          "accounted for, and explanatory power is what gains."
          % (_ae["spurious"][0], min(_ae[k][0] for k in ("merged", "wrong extent", "wrong class") if k in _ae),
             max(_ae[k][0] for k in ("merged", "wrong extent", "wrong class") if k in _ae),
             CZ["all"]["by_npatterns"][1][0],
             CZ["all"]["by_npatterns"][4][0]))

    from _revision_numbers import load_gallery as _load_gal
    try:
        _GL = _load_gal()
    except Exception as _ex:
        _GL = None
        print("  (gallery numbers unavailable:", _ex, ")")
    if _GL:
        p("The gallery carries the same two things, at the level where each belongs. A slice "
          "card shows the MEAN plausibility over its member images with the count it is "
          "averaged over %s the number says something about the slice, and the count says how "
          "much to trust it: %d of the %d cards have one. An image card shows its own score, "
          "and now also the two texts the pipeline writes about that image, labelled so they "
          "cannot be confused: the NARRATIVE of every typed error and the PATTERN description "
          "of the certified slices it belongs to. They answer different questions and were "
          "indistinguishable before. The cards carry three pictures each and stop at %d "
          "images; the %d narratives of the whole run are listed as plain text at the end of "
          "the by-image view, each with its plausibility where measured, which costs under a "
          "megabyte and makes the file answer “what does the pipeline say about THIS "
          "image” for every image, not only for the ones with pictures."
          % (EM_DASH, _GL["slice_cards_scored"], _GL["slice_cards"], _GL["image_cards"],
             _GL["narratives"]))
    p("Where it is reported: every card of the comparison page that has a certified cause "
      "carries the two scores under its pattern block, with both criteria and the judge's "
      "reason, and the two averages sit in the page header beside the judge's. Cards with "
      "typed errors but no certified slice say so explicitly. The measurement is "
      "slice_quality/examples/judge_cause_plausibility.py (SYNTH=1 for the synthetic set); it "
      "resumes from its own output and re-asks only what changed.", italic=True)

# ---------------------------------------------------------------------------
# 13. Widening the hypothesis vocabulary (October 2026). Every number comes from
#     docs/_revision_numbers.load_extra_hyp; figures from make_extra_hyp_figures.py.
# ---------------------------------------------------------------------------
from _revision_numbers import load_extra_hyp as _load_xh
try:
    XH = _load_xh()
except Exception as _ex:
    XH = None
    print("  (extra-hypothesis numbers unavailable, section 13 skipped:", _ex, ")")

if XH:
    _cn, _sp = XH["canon"], XH["specific"]
    _rate = lambda c, n: 100.0 * c / n
    _S = XH["sample"]["totals"]
    _F = XH["full"]["totals"] if "full" in XH else None
    _per = XH["sample"]["per_hypothesis"]
    _TY = {"fn": "missed", "extent": "extent", "confusion": "wrong class", "boundary": "boundary",
           "fp": "hallucinated", "merge": "merge", "split": "split"}
    h("13. Widening the hypothesis vocabulary of the language branch (October 2026)", 1)
    p("The language branch can only certify a situation it has been asked about. Until now its "
      "questions came from two sources: the 7 canonical class-agnostic hypotheses (“an object "
      "cut off at the image border”, “an object occluded by another object”, …) and up "
      "to 15 class-specific ones per class written by Gemma with the knowledge-driven prompt of "
      "the SliceLens paper. Their yield on the inst run differs by %s."
      % ("more than an order of magnitude"
         if _rate(_cn["certified"], _cn["evaluated"]) >= 10 * _rate(_sp["certified"], _sp["evaluated"])
         else "a factor of %.1f" % (_rate(_cn["certified"], _cn["evaluated"]) / _rate(_sp["certified"], _sp["evaluated"]))))
    table(["Hypotheses", "class × hypothesis slices evaluated", "certified", "rate"],
          [["7 canonical, class-agnostic", _cn["evaluated"], _cn["certified"],
            "%.1f%%" % _rate(_cn["certified"], _cn["evaluated"])],
           ["class-specific (Gemma)", _sp["evaluated"], _sp["certified"],
            "%.1f%%" % _rate(_sp["certified"], _sp["evaluated"])]],
          widths=[2.6, 1.9, 0.9, 0.8])
    _XH44 = _load_xh(_out_path("slice_quality_inst_auc44"))
    _cn44, _sp44 = _XH44["canon"], _XH44["specific"]
    p("%d distinct class-specific hypotheses certify, among them %s. With the slope rule choosing which "
      "hypotheses were grounded on the whole class (until 21 October) the class-specific ones certified "
      "%.1f%% of the time; since the rank test also makes that choice (§18.8) they certify %.1f%%, while the "
      "canonical ones went from %.1f%% to %.1f%%: the pre-selection had cost the specific phrases most. "
      "Many of the others describe the appearance of a sub-population (“matte black car”, “square "
      "monitor”, “glass top dining table”) rather than a situation, and §6.2 already measured that putting "
      "the class inside the query makes the VLM's answer less precise (0.598 against 0.784 for the generic "
      "wording)."
      % (len(XH["specific_certified"]), ", ".join("“%s”" % x for x in XH["specific_certified"][:6]),
         _rate(_sp44["certified"], _sp44["evaluated"]), _rate(_sp["certified"], _sp["evaluated"]),
         _rate(_cn44["certified"], _cn44["evaluated"]), _rate(_cn["certified"], _cn["evaluated"])))
    _it = XH["inst_types"]
    p("What the certified situations leave uncovered is just as clear. Each certified slice has "
      "a dominant error type, and across the %d of the inst run they are %s — %s, although "
      "Block 1 types %.0f%% of the val errors as one of the three. The question of this section is whether "
      "GENERIC hypotheses aimed at those gaps add situations the vocabulary was missing."
      % (sum(_it.values()), ", ".join("%s %d" % (_TY[k], v) for k, v in
                                      sorted(_it.items(), key=lambda kv: -kv[1])),
         "none is dominated by missed objects, by wrong extent or by a wrong class"
         if not sum(_it.get(k, 0) for k in ("fn", "extent", "confusion")) else
         "missed objects, wrong extent and a wrong class dominate only %d of them together"
         % sum(_it.get(k, 0) for k in ("fn", "extent", "confusion")),
         100.0 * sum(IS["tax"].get(k, 0) for k in ("fn", "extent", "confusion")) / IS["n_err"]))

    h("13.1  The eight hypotheses and how they were chosen", 2)
    _aim = {
        "a picture, poster or screen showing an object": ("hallucinated", "people or animals in paintings, posters, on monitor screens"),
        "a reflection of an object in a mirror, window or water": ("hallucinated", "boats reflected in water, cars in shop windows"),
        "a toy, statue or model of an object": ("hallucinated", "plastic horses, statues, toy cars"),
        "an object with holes or see-through parts": ("boundary / extent", "chairs, bicycles, motorbikes, plants: the mask fills the gaps"),
        "an object behind a fence, net, bars or branches": ("missed / split", "animals behind fences, cars behind railings"),
        "an object seen from an unusual viewpoint or in an unusual pose": ("missed", "lying cats, planes seen from above, animals from behind"),
        "a very large object seen up close that fills most of the image": ("missed / extent", "close-ups: the model mostly sees mid-scale objects"),
        "an object in an unusual setting for its kind": ("missed / wrong class", "context reliance: a boat on land, a cow indoors"),
    }
    table(["Hypothesis", "aimed at", "where one expects it in VOC"],
          [[q, _aim[q][0], _aim[q][1]] for q in XH["extra"]], widths=[2.6, 1.1, 2.6])
    p("Four rules, each from an earlier measurement, decided what was allowed in:")
    bullet("One visible situation per hypothesis, and no class name in it (§6.2): the class is "
           "printed outside the quotes, from the ground truth, as for the canonical seven.")
    bullet("Nothing photometric: “dark” and “blurry” collapsed to 0.00–0.12 precision in "
           "§6.2, because the VLM cannot verify them on a region.")
    bullet("Nothing that restates the error geometry. A semantic segmenter has no instances, so "
           "two touching objects of the same class always come out as one region and Block 1 "
           "types it a merge; a hypothesis like “objects touching each other” would certify by "
           "construction and say nothing about the model.")
    bullet("Class-agnostic, so the branch stays open-world: nothing is conditioned on what the "
           "cluster branch found.")
    p("The eight live in one tuple, slice_quality/describe.py::_GENERIC_EXTRA, from which the "
      "grounding script writes its query file, so the text grounded and the text rendered in a "
      "description cannot drift apart.", italic=True)

    h("13.2  Protocol: the same sample, the same code, two rescores", 2)
    _sm = XH["sample"]
    p("The eight were grounded with run_voc.py itself — same substrate cache, same seeded "
      "120-instance sample per class as the July discovery runs, same slope-trend test — with "
      "Qwen2.5-VL loaded once per worker (examples/ground_extra_hypotheses.py): %d classes × 8 "
      "hypotheses × 120 instances = %d VLM evaluations, on two GPUs. The sample turned out "
      "identical to July's in all classes but cow, whose July run predates the substrate cache."
      % (XH["n_classes"], XH["n_classes"] * 8 * 120))
    p("The inst pickle was then rescored twice, into separate directories, once with the July "
      "runs only and once with the extra runs added (LANG_DIRS). Only the hypotheses differ "
      "between the two, so code changes since September cannot pass for an effect; and the "
      "first rescore reproduces the inst run exactly (%d certified language slices in both). "
      "Every slice certified on the sample was then re-grounded on ALL instances of its class "
      "(reground_validated.py), as in July — here on one 11 GB card in 4-bit, as half of the "
      "July re-grounding was." % _S["certified_before"])

    h("13.3  Results", 2)
    figure("fig_extra_yield_en.png",
           "Figure 13a. Certification rate of the three sources of hypotheses (class × hypothesis "
           "slices certified over evaluated).", width=5.8)
    rows = []
    for q in XH["extra"]:
        v = _per[q]
        cert = ", ".join(c["class"] for c in v["certified"]) or "—"
        row = [q, "%d / %d" % (len(v["systematic_classes"]), XH["n_classes"]), cert]
        if _F:
            row.append(", ".join(XH["full_held"].get(q, [])) or "—")
        rows.append(row)
    table(["Hypothesis", "slope-systematic on the sample", "certified on the sample"]
          + (["still systematic on all instances"] if _F else []), rows,
          widths=[2.5, 1.1, 1.6] + ([1.4] if _F else []))
    figure("fig_extra_per_hyp_en.png", "Figure 13b. The eight hypotheses class by class.", width=5.8)
    _never = [q for q in XH["extra"] if not _per[q]["systematic_classes"]]
    _sysonly = [q for q in XH["extra"] if _per[q]["systematic_classes"] and not _per[q]["certified"]]
    p("On the sample %d of the %d class × hypothesis pairs are slope-systematic and %d are "
      "certified, %.1f%%: %s the class-specific yield, %.0f%% of the canonical one. "
      "%s%s"
      % (XH["sample_systematic"], XH["pairs"], XH["sample_certified"],
         _rate(XH["sample_certified"], XH["pairs"]),
         (lambda r: "%.0f times" % r if r >= 1.5 else "about" if r > 0.67 else "%.0f%% of" % (100 * r))(
             _rate(XH["sample_certified"], XH["pairs"]) / _rate(_sp["certified"], _sp["evaluated"])),
         100 * _rate(XH["sample_certified"], XH["pairs"]) / _rate(_cn["certified"], _cn["evaluated"]),
         ("%d hypotheses are never even systematic: %s. " % (len(_never), "; ".join("“%s”" % q for q in _never)))
         if _never else "",
         ("%s %s systematic in %s but %s not pass the other gates."
          % ("; ".join("“%s”" % q for q in _sysonly), "is" if len(_sysonly) == 1 else "are",
             ", ".join(c for q in _sysonly for c in _per[q]["systematic_classes"]),
             "does" if len(_sysonly) == 1 else "do")) if _sysonly else ""))
    p("On the sample the certified situations go from %d to %d and the images with a "
      "situational description from %d to %d, +%d: the new slices land mostly on images that "
      "already had one. No existing slice lost its certification: neither the heavier "
      "multiple-testing burden of %d more candidate slices nor the paraphrase deduplication "
      "cost the July vocabulary anything."
      % (_S["certified_before"], _S["certified_after"], _S["images_before"], _S["images_after"],
         _S["images_after"] - _S["images_before"], XH["pairs"]))
    _PF = XH["full"]["per_hypothesis"] if _F else _per
    if _F:
        _drop = {q: c for q, c in XH["full_dropped"].items() if c}
        _surv = [(q, c) for q in XH["extra"] for c in _PF[q]["certified"]]
        p("Re-grounded on every instance of their classes, %d of the %d remain certified%s. The "
          "survivors are %s. With them, the certified language slices go from %d to %d and the "
          "images that carry a situational description from %d to %d."
          % (XH["full_certified"], XH["sample_certified"],
             ("; the slope test no longer holds for " + "; ".join(
                 "“%s” on %s" % (q, ", ".join(c)) for q, c in _drop.items())) if _drop else "",
             "; ".join("%s — “%s” (%d instances on %d images, dominated by %s)"
                       % (c["class"], q, c["size"], c["n_images"], _TY[c["type"]]) for q, c in _surv),
             _F["certified_before"], _F["certified_after"], _F["images_before"], _F["images_after"]))
    figure("fig_extra_types_en.png", "Figure 13c. Dominant error type of the certified situations, "
           "before and after the extra hypotheses.", width=5.8)
    _ta = (_F or _S)["types_after"]
    _hall = XH["extra"][:3]          # picture, reflection, toy: the three written for hallucinations
    _hall_cert = [(q, c) for q in _hall for c in _PF[q]["certified"]]
    _fp_new = [(q, c) for q in XH["extra"] for c in _PF[q]["certified"] if c["type"] == "fp"]
    p("And the gap the eight were aimed at is still there: %s. Of the three hypotheses written "
      "for hallucinations (picture, reflection, toy) %s%s"
      % ("no certified situation is dominated by missed objects, wrong extent or a wrong class"
         if not any(_ta.get(k) for k in ("fn", "extent", "confusion")) else
         ", ".join("%s %d" % (_TY[k], _ta.get(k, 0)) for k in ("fn", "extent", "confusion")),
         ("%d certif%s: %s" % (len(_hall_cert), "ies" if len(_hall_cert) == 1 else "y",
                               ", ".join("%s as %s" % (c["class"], _TY[c["type"]]) for _q, c in _hall_cert)))
         if _hall_cert else "none certifies",
         ("; the new hallucination-dominated situations, %s, come from hypotheses written for "
          "other gaps." % ", ".join("%s (“%s”)" % (c["class"], q) for q, c in _fp_new))
         if _fp_new else ", and no new situation is dominated by hallucinations at all."))

    h("13.4  Looking at the instances", 2)
    p("A slope-trend test says that the instances the VLM rates as matching a hypothesis are "
      "more often errors; it does not say that the hypothesis describes them. Each of the new "
      "slices was therefore checked by eye on its highest-confidence error instances of the "
      "sample stage (crops in docs/extra_hyp/, verdicts in docs/extra_hyp/visual_check.json).")
    figure("fig_extra_examples_en.png", "Figure 13d. Highest-confidence error instances of the new "
           "slices, with the verdict of the visual check (green holds, amber partly, red artefact).",
           width=6.3)
    _vis = XH.get("visual", {}).get("slices", [])
    table(["Slice", "hypothesis", "what the images show", "verdict"],
          [[s["class"], s["hypothesis"], s["en"], s["verdict"]] for s in _vis],
          widths=[0.8, 1.9, 2.8, 0.8])
    _held = lambda s: s["class"] in XH["full_held"].get(s["hypothesis"], [])
    _hold = [s for s in _vis if s["verdict"] == "holds" and (not _F or _held(s))]
    _bus = next((s for s in _vis if s["verdict"] == "artefact"), None)
    _seen_dropped = [s for s in _vis if _F and s["verdict"] == "holds" and not _held(s)]
    _lucky = [s for s in _vis if _F and s["verdict"] != "holds" and _held(s)]
    if _bus:
        p("The %s slice is the instructive one. “%s” fits any photograph — every photo is a "
          "picture showing an object — and the VLM says yes when the object fills the frame; the "
          "errors the slice is dominated by come from frame-filling %s instances, not from images inside the image. "
          "The test on the sample passed it; the visual check did not%s."
          % (_bus["class"], _bus["hypothesis"].capitalize(), _bus["class"],
             (", and the full-population test then dropped it as well"
              if _F and not _held(_bus) else "")))
    if _F:
        p("The two checks answer different questions, and the results show it. %s%s Counting only "
          "what passes both, the extra vocabulary leaves %d situation%s: %s."
          % (("The visual check confirms the situation is in the images — %s — but on all "
              "instances of the class it is no longer associated with errors: on the sample the "
              "association was partly luck." % "; ".join(
                  "%s “%s”" % (s["class"], s["hypothesis"]) for s in _seen_dropped))
             if _seen_dropped else "",
             (" Conversely %s hold%s statistically while the images show something else or "
              "something vague (%s)." % (
                  " and ".join(s["class"] for s in _lucky), "s" if len(_lucky) == 1 else "",
                  "; ".join("%s: %s" % (s["class"], s["verdict"]) for s in _lucky)))
             if _lucky else "",
             len(_hold), "" if len(_hold) == 1 else "s",
             ", ".join("%s — “%s”" % (s["class"], s["hypothesis"]) for s in _hold) or "none"))

    h("13.5  What it means", 2)
    bullet("The language branch is not limited by the NUMBER of hypotheses. Generic hypotheses "
           "certify far more often than class-specific ones, and eight more of them, chosen to "
           "fill known gaps, add %d images to %d."
           % ((_F or _S)["images_after"] - (_F or _S)["images_before"], (_F or _S)["images_before"]))
    _gap = sum(XH["inst_types"].get(k, 0) for k in ("fn", "extent", "confusion"))
    bullet(("Missed objects, wrong extent and wrong class remain without a certified situation "
            "even when one is looked for on purpose." if not _gap else
            "Missed objects, wrong extent and wrong class dominate few certified situations "
            "(%d of %d), even when looked for on purpose." % (_gap, sum(XH["inst_types"].values())))
           + " The likeliest reading is that those errors "
           "are not explained by a situation a VLM can name and verify on a region — the same "
           "line §6.1's corruption experiment drew: discovery works, certification needs a "
           "nameable situation.")
    _nv = len(XH.get("visual", {}).get("slices", []))
    _both = [x for x in XH.get("visual", {}).get("slices", []) if x["verdict"] == "holds"
             and (not _F or x["class"] in XH["full_held"].get(x["hypothesis"], []))]
    bullet("Statistical certification needs independent checks beside it, and they do not "
           "replace each other: the bus artefact passed the slope test on the sample and failed "
           "a look at the instances%s%s. The description-precision evaluation (§6.2) and the "
           "cause-plausibility judge (§12) exist for the same reason." % (
               (", while %d slice%s that look%s right fail%s on the full population"
                % (len(_seen_dropped), "" if len(_seen_dropped) == 1 else "s",
                   "s" if len(_seen_dropped) == 1 else "", "s" if len(_seen_dropped) == 1 else ""))
               if _F and _seen_dropped else "",
               "; of the %d new slices only %d pass%s both" % (_nv, len(_both), "es" if len(_both) == 1 else "")
               if _F else ""))
    _MG = XH.get("merged", [])
    if _MG:
        bullet("Merged by the same rule as the July vocabulary, not re-worded. The %d situations "
               "that survived the full-population test were added to the inst run (October 2026), "
               "so the gallery and the comparison page show them; no manual veto was applied to "
               "them that the 38 older ones never had. Re-wording the failed hypotheses after "
               "seeing the results and testing them again on the same val images is the textbook "
               "way to manufacture false positives — a rewording needs images that played no part "
               "in choosing it.%s" % (len(_MG), (" Since the rank test also chooses what is grounded on the whole "
               "class (§18.8), the same eight phrases certify %d more situations, in %d classes."
               % (len(XH["merged_pend"]), len({c for c, _q in XH["merged_pend"]}))) if XH.get("merged_pend") else ""))
    else:
        bullet("Not merged into the inst run, and not re-worded. Integrating the surviving slices "
               "would mean re-running judges, plausibility and gallery for a handful of images; and "
               "re-wording the failed hypotheses after seeing the results and testing them again on "
               "the same val images is the textbook way to manufacture false positives — a rewording "
               "needs images that played no part in choosing it.")
    _MP = XH.get("merged_plaus")
    if _MG and _MP:
        h("13.6  The added situations, judged like all the others", 2)
        p("Once in the inst run, the new situations go through the same independent check as "
          "every other cause: the plausibility judge of §12, against the pipeline narrative of "
          "the same image. Over all judged images the cause scores %.2f (%d images); the images "
          "whose cause names one of the new hypotheses score as follows." % _MP["all"])
        table(["Hypothesis", "slice", "images judged", "mean plausibility (0–10)"],
              [[s["hypothesis"], s["slice_id"], _MP[s["slice_id"]][1],
                ("%.2f" % _MP[s["slice_id"]][0]) if _MP[s["slice_id"]][0] is not None else "—"]
               for s in _MG], widths=[2.8, 1.4, 0.9, 1.2])
        _vv = {s["slice"].replace("_extra_", "_"): s["verdict"]
               for s in XH.get("visual", {}).get("slices", [])}
        _above = [s for s in _MG if (_MP[s["slice_id"]][0] or 0) >= _MP["all"][0]]
        _doubt = [s for s in _MG if any(v != "holds" and k.split("_")[0] == s["slice_id"].split("_")[0]
                                        for k, v in _vv.items())]
        p("%d of the %d score at or above the overall mean; those the visual check had found doubtful "
          "score %s (%s). The two instruments do not contradict each other; they look at different "
          "things. The plausibility judge never sees the image: it reads the narrated errors and "
          "the blamed situation and asks whether the second would explain the first. It cannot "
          "notice that the situation is not in the picture — only the visual check, or the "
          "VLM's re-verification of the phrase on the region (§6.2), can. That is why the "
          "pipeline needs all three, and why the gallery and the comparison page show each of "
          "these images with its score."
          % (len(_above), len(_MG),
             " and ".join("%s (%.2f)" % (s["slice_id"].split("_")[0], _MP[s["slice_id"]][0]) for s in _doubt) or "none",
             " and ".join({k.split("_")[0]: v for k, v in _vv.items()}.get(s["slice_id"].split("_")[0], "?")
                          for s in _doubt) or "—"), italic=True)
    p("Reproduce: outputs/run_extra_hyp.sh (grounding on the sample, two rescores, "
      "compare_extra_hypotheses.py), outputs/run_extra_reground.sh (full-population "
      "re-grounding, rescore, comparison)%s; outputs in outputs/slice_quality_inst_extra{,_base,"
      "_full}/ and /data/outputs_extra, never in the July directories."
      % (" and outputs/chain_inst_extra.sh (merge into the inst run, train36 and synthetic "
         "induction, judges, gallery; the run before the merge is kept in "
         "outputs/slice_quality_inst_bak_20261006)" if _MG else ""), italic=True)

    # ---- 13.7 the vocabulary against the failure modes reported for segmentation models ----
    from _revision_numbers import INST as _INST_DIR, _cls as _rn_cls
    _six = json.load(open(os.path.join(_INST_DIR, "slices_index.json"), encoding="utf-8"))["slices"]
    _cert = [s for s in _six if s.get("in_descriptions")]
    _byh = {}
    for s in _cert:
        if s.get("source") == "language":
            _byh.setdefault(s["hypothesis"], []).append(_rn_cls(s["slice_id"]))
    _cdesc = {}                                    # certified cluster slice -> its phrases
    for _d in json.load(open(os.path.join(_INST_DIR, "image_descriptions.json"), encoding="utf-8")).values():
        for _p in _d.get("patterns", []):
            if _p["branch"] == "cluster":
                _cdesc.setdefault(_p["slice_id"], set()).add(_p["description"])
    _ncl = sum(1 for s in _cert if s.get("source") == "cluster")

    def _found(hyps):
        parts = []
        for hq in hyps:
            cl = sorted(set(_byh.get(hq, [])))
            parts.append("“%s”: %s" % (hq, ("%d (%s)" % (len(cl), ", ".join(cl))) if cl else "0"))
        return "; ".join(parts)

    _blur = sorted(hq for hq in _byh if "blur" in hq)
    _small_cl = sorted(k for k, v in _cdesc.items() if any(x.startswith("small ") for x in v))
    h("13.7  The vocabulary against the failure modes reported for segmentation models", 2)
    p("The hypotheses were chosen from the SliceLens paper (the seven canonical ones), from a "
      "language model asked about each class (the class-specific ones) and from the error types "
      "the first two left uncovered (the eight of §13.1). A different check is whether the list "
      "covers the failure categories that studies of segmentation models report. The table sets "
      "the categories of a literature overview of segmentation failures (foundation models of "
      "the SAM family and semantic segmentation networks) beside the hypotheses and attributes "
      "of this pipeline, and beside what the inst run certifies for each. Two categories of that "
      "overview are left out because they do not apply here: unsupervised object discovery "
      "trained on synthetic scenes, and imaging artefacts specific to medical scans.")
    table(["Failure category reported", "Hypotheses / attributes of this pipeline",
           "Certified situations in the inst run (classes)"],
          [["Small objects, minor parts", "“a small object”; the attribute small",
            _found(["a small object"]) + ("; cluster branch: %d of the %d certified slices hold "
                                          "objects tagged small (missed or with imprecise boundaries)"
                                          % (len(_small_cl), _ncl) if _small_cl else "")],
           ["Thin, branching structures",
            "“an object with thin or elongated parts”, “… holes or see-through parts”, “… behind a "
            "fence, net, bars or branches”; narratives report the thin fragments the mask misses",
            _found(["an object with thin or elongated parts", "an object with holes or see-through parts",
                    "an object behind a fence, net, bars or branches"])],
           ["Low contrast, camouflage", "“an object that blends into its background”; the attribute "
            "dark (dropped: the VLM confirmed it on 0.00 of the regions, §6.2)",
            _found(["an object that blends into its background"])],
           ["Geometric transformations: rotation, extreme scale",
            "“an object seen from an unusual viewpoint or in an unusual pose”, “a very large object "
            "seen up close that fills most of the image”; in-plane rotation of the image is not a "
            "hypothesis",
            _found(["an object seen from an unusual viewpoint or in an unusual pose",
                    "a very large object seen up close that fills most of the image"])],
           ["Distribution shift, open-world images",
            "“an object in an unusual setting for its kind”, “a toy, statue or model of an object”, "
            "“a picture, poster or screen showing an object”, “a reflection of an object …”",
            _found(["an object in an unusual setting for its kind", "a toy, statue or model of an object",
                    "a picture, poster or screen showing an object",
                    "a reflection of an object in a mirror, window or water"])],
           ["Small regions of a large object given the wrong label (context underused)",
            "not a situation but an error type: Block 1 records it (confusion, extent) and the "
            "narrative says it (“labelled X instead”, “patches hallucinated”)",
            "reported image by image, not as a certified situation"],
           ["Imaging artefacts", "the attribute blurry; class-specific blur hypotheses",
            (_found(_blur) if _blur else "0") + "; noise, compression and other artefacts are not "
            "hypotheses"],
           ["Ambiguous prompts (promptable models)",
            "does not apply to LR-ASPP, which has no prompt; the VLM that grounds the hypotheses "
            "is given a box, the prompt type that overview finds most reliable", "—"]],
          widths=[1.5, 2.6, 2.3])
    caption("Table 13.1. The failure categories reported for segmentation models, the hypotheses "
            "that cover them here, and what the inst run certifies on VOC2012 val.")
    _top = sorted(_byh.items(), key=lambda kv: -len(set(kv[1])))[:3]
    p("Small objects and thin structures are covered and certified. Low contrast and open-world "
      "situations are covered but rarely certified, and for low contrast a measured reason is "
      "known: the VLM does not verify photometric properties such as darkness on a region "
      "(§6.2). Viewpoint and close-up are covered and %s; image artefacts "
      "other than blur, and in-plane rotation, are not covered at all. In the other direction, "
      "the situations this pipeline certifies most often are not in the overview: %s. They "
      "produce merges and hallucinations, errors of an instance rather than of pixels, and they "
      "become visible only because Block 1 separates the instances — the overview reasons about "
      "pixels and domains."
      % ((lambda v: "never certified on VOC" if not v else "certified in %d class%s only (%s)"
          % (len(v), "" if len(v) == 1 else "es", ", ".join(v)))(sorted(
             {c for q in ("an object seen from an unusual viewpoint or in an unusual pose",
                          "a very large object seen up close that fills most of the image")
              for c in _byh.get(q, [])})),
         "; ".join("“%s” (%d classes)" % (hq, len(set(cl))) for hq, cl in _top)))

# ---- 13.8 the hypotheses SliceLens derives from the data (October 2026). Numbers from
#      _revision_numbers.load_dd_hyp (dd_summary.json, written on the server by
#      slice_quality/examples/dd_summary.py); figure from docs/make_dd_hyp_figures.py ----
from _revision_numbers import load_dd_hyp as _load_dd
try:
    DDH = _load_dd()
except Exception as _ex:
    DDH = None
    print("  (data-driven numbers unavailable, section 13.8 skipped:", _ex, ")")

if XH and DDH:
    _Y = DDH["yield_sample"]
    _T = DDH["totals"]
    _cls = DDH["classes"]
    _hy = [(c, h) for c in _cls for h in DDH["per_class"][c]["hypotheses"]]
    _cert_dd = [(c, h) for c, h in _hy if h["certified"]]
    _dropped = [(c, h) for c, h in _hy if h["sample_certified"] and not h["certified"]]
    _pl = DDH["pilot"]
    _att0 = DDH["per_class"][_cls[0]]["attributes"]
    _ns = DDH["per_class"][_cls[0]]["n_sampled"]
    h("13.8  The hypotheses SliceLens derives from the data (October 2026)", 2)
    p("The SliceLens paper has a second way of writing hypotheses beside the knowledge-driven "
      "prompt: it asks the data. A VLM captions instances attribute by attribute, a language "
      "model extracts the value of the attribute from each caption and groups the values, and "
      "the groups are refined into queries. The June pilot ran this on %s and added six "
      "hypotheses per class to their vocabulary; the other %d classes had only the "
      "knowledge-driven ones. In October it was run on those %d, with the same models as the rest "
      "of the pipeline (examples/gen_datadriven.py):"
      % (", ".join(_pl["classes"]), len(_cls), len(_cls)))
    bullet("Attributes. The paper seeds the captions with the attributes of the knowledge-driven "
           "prompt; Gemma returned fewer than two for %s of the %d classes, so the pilot's four "
           "were used: %s."
           % ("all" if DDH["seeds"].get("fallback") == len(_cls) else "%d" % DDH["seeds"].get("fallback", 0),
              len(_cls), ", ".join(a["attribute"] for a in _att0)))
    bullet("Captions. Qwen2.5-VL describes %d instances per class, drawn at random among all "
           "the instances of the class — errors and correct ones alike — once per attribute, "
           "on the instance's box." % _ns)
    bullet("Values and groups. Gemma extracts the value of the attribute from each caption — it "
           "finds one in %d of the %d captions, the others do not mention the attribute — and "
           "groups the values: “asphalt road” and “city street” become “Roadways”, “soft, diffused "
           "light” joins “Soft & Diffused”." % (DDH["n_values"], DDH["n_captions"]))
    bullet("Hypotheses. The six largest groups of the class, over all attributes, become one "
           "phrase each, written by Gemma as a situation of one object that a person can check on "
           "the image (“a bicycle on an asphalt road”). The pilot kept six per class as well; this "
           "step is the part it had left implicit.")
    p("The %d hypotheses went through exactly the protocol of §13.2: the seeded 120-instance "
      "sample of each class, two rescores of copies of the inst pickle that differ only in "
      "the data-driven runs (the first reproduces the inst run: %d certified language "
      "situations), then re-grounding on every instance of the class for whatever was "
      "certified on the sample. The inst run itself is not modified."
      % (DDH["n_hyp"], _T["certified_base"]))
    _rows = []
    for k, lab in (("canonical", "7 canonical, class-agnostic"), ("class-specific", "class-specific (Gemma)"),
                   ("extra", "8 extra generic (§13.1)"), ("data-driven", "data-driven (this section)")):
        sy, n = _Y[k]
        _rows.append([lab, n, sy, "%.0f%%" % (100.0 * sy / n) if n else "—"])
    table(["Hypotheses, on the same %d classes" % len(_cls), "class × hypothesis pairs on the sample",
           "slope-systematic", "rate"], _rows, widths=[2.6, 1.6, 1.1, 0.8])
    caption("Table 13.2. The four sources of hypotheses on the same 120-instance samples of the "
            "%d classes: how often SliceLens's slope-trend test calls a pair systematic." % len(_cls))
    _rk = sorted(_Y, key=lambda k: _Y[k][0] / max(_Y[k][1], 1))
    _below = [k for k in _rk if _Y[k][0] / max(_Y[k][1], 1) > _Y["data-driven"][0] / max(_Y["data-driven"][1], 1)]
    p("On the sample %d of the %d data-driven hypotheses are systematic, %d%% — %s. %d of them "
      "pass the other gates and are certified on the sample; re-grounded on every instance of "
      "their class, %d remain%s. With them the certified language situations go from %d to %d "
      "and the images that carry a situational description from %d to %d; %s."
      % (DDH["n_sample_sys"], DDH["n_hyp"], round(100.0 * DDH["n_sample_sys"] / DDH["n_hyp"]),
         "the lowest of the four sources" if _rk[0] == "data-driven" and len(_below) == 3 else
         "below %s" % ", ".join("the %s ones" % k for k in _below) if _below else "the highest of the four",
         DDH["n_sample_cert"], DDH["n_cert"],
         (" (%s no longer hold%s)" % (" and ".join("“%s”" % h["query"] for _c, h in _dropped),
                                     "s" if len(_dropped) == 1 else "")) if _dropped else "",
         _T["certified_base"], _T["certified_dd"], _T["images_base"], _T["images_dd"],
         "no existing situation loses its certification" if not _T["lost"] else
         "%d existing situation%s lose%s its certification" % (len(_T["lost"]), "" if len(_T["lost"]) == 1 else "s",
                                                              "s" if len(_T["lost"]) == 1 else "")))
    table(["Class", "hypothesis", "slope: sample / all instances", "instances", "images", "dominated by"],
          [[c, h["query"], "%.2f / %s" % (h["sample_slope"], "%.2f" % h["full_slope"] if h["full_slope"] is not None else "—"),
            h["certified"]["size"], h["certified"]["n_images"], _TY.get(h["certified"]["type"], h["certified"]["type"])]
           for c, h in _cert_dd], widths=[0.8, 2.5, 1.2, 0.8, 0.7, 0.9])
    caption("Table 13.3. The data-driven situations certified after re-grounding on all "
            "instances (threshold of the slope test 0.20).")
    _pc = _pl["certified"]
    p("The June pilot's hypotheses on the other four classes fare no better: of its %d, %s is "
      "certified in the inst run. Over the twenty classes the data-driven generator has written "
      "%d hypotheses and %d certified situations, %.1f%% — against %.1f%% for the class-specific "
      "ones (a count that includes the pilot's), %.1f%% for the eight extra generic ones after re-grounding and %.1f%% for the "
      "canonical seven. Those counts come from the generator's own pass, on copies of the run where the slope "
      "rule still chose what to ground; its hypotheses are not part of the official run, so the rank test has "
      "not made that choice for them (§18.8 did it for the July vocabulary and the eight generic ones)."
      % (_pl["n"], "none" if not _pc else "%d" % _pc, _pl["n"] + DDH["n_hyp"], _pc + DDH["n_cert"],
         100.0 * (_pc + DDH["n_cert"]) / (_pl["n"] + DDH["n_hyp"]),
         _rate(_sp["certified"], _sp["evaluated"]),
         _rate(XH.get("full_certified", XH["sample_certified"]), XH["pairs"]),
         _rate(_cn["certified"], _cn["evaluated"])))
    _ex = (DDH.get("visual") or {}).get("slices", [])
    if _ex:
        p("The certified situations were looked at as in §13.4, on the ten instances the VLM rates "
          "highest after re-grounding (crops in docs/dd_hyp/, verdicts in docs/dd_hyp/visual_check.json).")
        figure("fig_dd_examples_en.png", "Figure 13e. Highest-confidence instances of the "
               "data-driven situations, with the verdict of the visual check (green holds, amber "
               "real but differently named).", width=6.3)
        table(["Slice", "hypothesis", "what the images show", "verdict"],
              [[s["class"], s["hypothesis"], s["en"], s["verdict"]] for s in _ex],
              widths=[0.8, 1.9, 2.8, 0.8])
        _okv = [s for s in _ex if s["verdict"] == "holds"]
        _mis = [s for s in _ex if s["verdict"] != "holds"]
        p("Counting only what passes both checks, the data-driven generator adds %d situation%s "
          "to the vocabulary%s.%s"
          % (len(_okv), "" if len(_okv) == 1 else "s",
             (": " + ", ".join("%s — “%s”" % (s["class"], s["hypothesis"]) for s in _okv)) if _okv else "",
             "".join(" The %s slice is a real group under another name: %s." % (m["class"], m["reading"])
                     for m in _mis if m.get("reading"))))
    _fc, _fr = DDH["flat_common"], DDH["flat_rare"]
    _pick = lambda hs: [h for i, h in enumerate(hs) if h["query"].split()[1] not in
                        {x["query"].split()[1] for x in hs[:i]}][:3]
    _ty_dd = {h["certified"]["type"] for _c, h in _cert_dd}
    p("Why so little. On the sample %d of the %d hypotheses have a maximum positive slope of "
      "zero: the error rate never rises with the VLM's confidence. They fail in two opposite "
      "ways. %d describe the usual look of the class, and the VLM confirms them on at least half "
      "of the sample (%s); a situation most instances share cannot be more frequent among the "
      "errors. %d read like a detail of one captioned instance, too particular to recur: the "
      "VLM confirms them on at most five of the %d instances (%s). Both follow from how the "
      "groups are formed: the captions are taken on a random sample of the class, not on its "
      "errors, the groups are ranked by size, and a group of two or three values out of twenty "
      "is still turned into a phrase. The records show two further "
      "weaknesses. %d of the %d phrases are about light or contrast, the kind of photometric "
      "property §6.2 found the VLM unable to confirm on a region — lighting is one of the four "
      "seed attributes, since Gemma proposed none of its own. And in %d cases Gemma did not group "
      "the values at all, returning one group that holds every value (%s)."
      % (DDH["n_flat"], DDH["n_hyp"], len(_fc),
         "; ".join("“%s”, %d of %d" % (h["query"], h["sample_match"], h["sample_scored"]) for h in _pick(_fc)),
         len(_fr), _fr[0]["sample_scored"] if _fr else 120,
         "; ".join("“%s”, %d" % (h["query"], h["sample_match"]) for h in _pick(_fr)),
         DDH["light"], DDH["n_hyp"], len(DDH["collapsed"]),
         "; ".join("%s, %s" % (c, a.replace("the ", "", 1)) for c, a, _n, _v in DDH["collapsed"])))
    p("%s, so the gap of §13.3 — missed objects, wrong extent, wrong class — stays open. A "
      "variant that captions only the errors, or contrasts them with correct instances of the "
      "same class, would aim the groups at what differs; it has not been run."
      % (("Both certified situations are dominated by %s errors" % _TY.get(next(iter(_ty_dd)), "")
          if len(_ty_dd) == 1 and len(_cert_dd) == 2 else
          "The certified situations are dominated by %s" % ", ".join(sorted(_TY.get(t, t) for t in _ty_dd)))
         if _cert_dd else "No situation is certified"))
    p("Reproduce: outputs/chain_dd_gen.sh (generation, two GPUs, about 4 h including half an "
      "hour to load the models), outputs/chain_dd_voc.sh (sample grounding, the two rescores, compare_dd_hypotheses.py, "
      "full re-grounding, rescore and comparison), slice_quality/examples/dd_summary.py (this "
      "section's numbers); outputs in /data/outputs_dd (hypotheses and every intermediate step, "
      "dd_<class>.json), /data/outputs_dd_ground and outputs/slice_quality_inst_dd{,_base}/.",
      italic=True)

# ============================================================================
#  18. A second, larger dataset: SBD (October 2026). Numbers from
#      _revision_numbers.load_sbd (sbd_summary.json written on the server by
#      slice_quality/examples/doc_numbers_sbd_ea.py, slope_robustness.json, the two
#      slice indexes) -- nothing typed in.
# ============================================================================
from _revision_numbers import load_sbd as _load_sbd, load_ea as _load_ea
try:
    SB = _load_sbd()
except Exception as _ex:
    SB = None
    print("  (SBD numbers unavailable, chapter 18 skipped:", _ex, ")")
try:
    EAN = _load_ea()
except Exception as _ex:
    EAN = None
    print("  (Pascal-EA numbers unavailable, chapter 19 skipped:", _ex, ")")
from collections import Counter
_pct = lambda a, b: 100.0 * a / b if b else float("nan")
_TYN = {"fn": "missed", "extent": "extent", "confusion": "wrong class", "boundary": "boundary",
        "fp": "hallucinated", "merge": "merge", "split": "split"}

if SB:
    _R, _MI, _FN, _RB = SB["run"], SB["miou"], SB["funnel"], SB["robust"]
    _ns_img = _MI["SBD"]["n_images"]
    h("18. A second, larger dataset: SBD (October 2026)", 1)
    p("Everything above is measured on VOC2012 val, 1 449 images. The Semantic Boundaries "
      "Dataset (SBD) annotates 11 355 images of the same twenty classes with class and instance "
      "masks; %d of them are in neither the VOC 2012 segmentation train set nor its val set. The "
      "whole pipeline was run on those %d with the protocol of the inst run and nothing changed: "
      "the same segmenter and 520 centre crop, Block 1 with ground-truth objects from the SBD "
      "instance masks, DINOv2 and the per-type mixtures of the cluster branch with the "
      "verification-mode tags, the same hypotheses for the language branch (the July vocabulary "
      "and the eight of §13) on a seeded 120-instance sample per class, re-grounding on the "
      "whole class of whatever is systematic on the sample, and the narrative v6. The question is "
      "whether what the pipeline finds on VOC is a property of the method or of a small dataset."
      % (_ns_img, _ns_img))
    _tv, _tsb = IS["tax"], _R["tax"]
    _nv, _nsb = IS["n_err"], _R["n_err"]
    _tyrow = lambda k: ["%s, share of the errors" % _TYN[k].capitalize(),
                        "%.1f%%" % _pct(_tv.get(k, 0), _nv), "%.1f%%" % _pct(_tsb.get(k, 0), _nsb)]
    _ssb = _R["slices"]
    table(["Quantity", "VOC2012 val (inst)", "SBD"],
          [["Images", "1449", str(_ns_img)],
           ["mIoU of the segmenter (520 crop)", "%.1f" % (100 * _MI["VOC val"]["miou"]),
            "%.1f" % (100 * _MI["SBD"]["miou"])],
           ["Typed error instances (per image)", "%d (%.2f)" % (_nv, _nv / 1449.0),
            "%d (%.2f)" % (_nsb, _nsb / float(_ns_img))]]
          + [_tyrow(k) for k in ("fp", "merge", "boundary", "split", "extent", "fn", "confusion")]
          + [["Certified situations (language branch)", str(_ts["language"]), str(_ssb["language"])],
             ["Certified cluster slices", str(_ts["cluster"]), str(_ssb["cluster"])],
             ["Images with at least one certified pattern",
              "%d (%.1f%%)" % (IS["with_pattern"], _pct(IS["with_pattern"], 1449)),
              "%d (%.1f%%)" % (_R["with_pattern"], _pct(_R["with_pattern"], _ns_img))],
             ["Errors inside a certified group: union",
              "%.1f%%" % _pct(IS["cov_union"], _nv), "%.1f%%" % _pct(_R["cov_union"], _nsb)],
             ["   … of which in a language slice / in a cluster slice",
              "%.1f%% / %.1f%%" % (_pct(IS["cov_language"], _nv), _pct(IS["cov_cluster"], _nv)),
              "%.1f%% / %.1f%%" % (_pct(_R["cov_language"], _nsb), _pct(_R["cov_cluster"], _nsb))]],
          widths=[3.0, 1.7, 1.7])
    caption("Table 18.1. The same pipeline on VOC2012 val and on the %d SBD images outside VOC "
            "2012 segmentation. Coverage as in Table 6.1." % _ns_img)
    _pc = {c: 100 * (_MI["SBD"]["per_class"][c] - _MI["VOC val"]["per_class"][c])
           for c in _MI["SBD"]["per_class"] if c != "background"}
    _up = [c for c in _pc if _pc[c] > 0]
    _dn = sorted((c for c in _pc if _pc[c] < 0), key=lambda c: _pc[c])
    p("SBD is harder for this segmenter: %.1f points of mIoU less, and every class loses except %s "
      "(the largest losses: %s). %s SBD also has no void band: VOC leaves a strip of unlabelled "
      "pixels along every outline out of the evaluation, SBD counts them. The error population is "
      "otherwise much the same — %.2f errors per image against %.2f, the same order of types, "
      "merges and splits a little more frequent, boundary errors less."
      % (100 * (_MI["VOC val"]["miou"] - _MI["SBD"]["miou"]), " and ".join(_up) or "none",
         ", ".join("%s %.1f" % (c, _pc[c]) for c in _dn[:3]),
         ("%s gains %.1f points: SBD outlines a bicycle as a filled shape where VOC traces the "
          "frame and the spokes, a difference of annotation rather than of the model."
          % (_up[0].capitalize(), _pc[_up[0]]))
         if _up == ["bicycle"] else "",
         _nsb / float(_ns_img), _nv / 1449.0))

    h("18.1  What is certified on both", 2)
    _lv, _lsb = SB["lang_voc"], SB["lang_sbd"]
    _both = sorted(set(_lv) & set(_lsb))
    _only_sb = sorted(set(_lsb) - set(_lv))
    _fmt = lambda d: "%d err. / %d img., %s" % (d["size"], d["n_images"], _TYN.get(d["type"], d["type"]))
    table(["Class", "situation (SliceLens hypothesis)", "VOC2012 val", "SBD"],
          [[c, q, _fmt(_lv[(c, q)]), _fmt(_lsb[(c, q)])] for c, q in _both]
          + [[c, q, "—", _fmt(_lsb[(c, q)])] for c, q in _only_sb],
          widths=[0.9, 2.6, 1.5, 1.5])
    caption("Table 18.2. The situations the language branch certifies on SBD; the first %d are "
            "certified on VOC2012 val too. The %d certified on VOC only are in the inst run's "
            "slice index." % (len(_both), len(set(_lv) - set(_lsb))))
    _hq = Counter(q for _c, q in _lsb)
    _top_q = _hq.most_common(1)[0]
    p("%d of the %d SBD situations were certified on VOC as well, all of them %s; %d are new. "
      "“%s” is again the most frequent (%d classes), and on SBD it is dominated by hallucinated "
      "regions in %d of its %d classes. The situation certified on the largest group, %s — “%s”, "
      "holds %d errors on %d images."
      % (len(_both), len(_lsb),
         "about objects at the border, occluded or overlapping" if all(
             any(w in q for w in ("border", "occluded", "overlapping")) for _c, q in _both) else "listed first",
         len(_only_sb), _top_q[0], _top_q[1],
         sum(1 for (c, q), d in _lsb.items() if q == _top_q[0] and d["type"] == "fp"), _top_q[1],
         *max(((c, q, d["size"], d["n_images"]) for (c, q), d in _lsb.items()), key=lambda x: x[2])))
    _cs = SB["clus_sbd"]
    _anon = sum(1 for x in _cs if "appearance only" in x["phrase"])
    table(["Cluster slice", "dominated by", "errors / images", "phrase shown most often"],
          [[x["slice_id"], _TYN.get(x["type"], x["type"]), "%d / %d" % (x["size"], x["n_images"]), x["phrase"]]
           for x in _cs], widths=[1.4, 0.9, 0.9, 3.3])
    caption("Table 18.3. The %d cluster slices certified on SBD (VOC2012 val: %d)." % (len(_cs), len(SB["clus_voc"])))
    p("The cluster branch behaves as its limits in §8 lead one to expect: it needs many "
      "homogeneous instances, and SBD has six times more. It certifies %d slices against %d, much larger ones "
      "(%d to %d errors each), and covers %.1f%% of the errors against %.1f%% on VOC — so that "
      "the union coverage is about the same on both datasets (%.1f%% against %.1f%%) although "
      "the language branch certifies %d situations instead of %d. But %d of the %d cluster slices "
      "carry no verified attribute (“grouped by appearance only”): the errors are grouped, not "
      "explained."
      % (len(_cs), len(SB["clus_voc"]), min(x["size"] for x in _cs), max(x["size"] for x in _cs),
         _pct(_R["cov_cluster"], _nsb), _pct(IS["cov_cluster"], _nv),
         _pct(_R["cov_union"], _nsb), _pct(IS["cov_union"], _nv), _ssb["language"], _ts["language"],
         _anon, len(_cs)))

    h("18.2  Why six times more images give fewer certified situations", 2)
    _fv, _fs = _FN["VOC val"]["total"], _FN["SBD"]["total"]
    table(["Step of the language branch", "VOC2012 val", "SBD"],
          [["class × hypothesis pairs tried on the 120-instance sample", _fv[0], _fs[0]],
           ["slope-systematic on the sample", "%d (%.0f%%)" % (_fv[1], _pct(_fv[1], _fv[0])),
            "%d (%.0f%%)" % (_fs[1], _pct(_fs[1], _fs[0]))],
           ["re-grounded on every instance of the class", _fv[2], _fs[2]],
           ["still systematic on the whole class", "%d (%.0f%%)" % (_fv[3], _pct(_fv[3], _fv[2])),
            "%d (%.0f%%)" % (_fs[3], _pct(_fs[3], _fs[2]))]],
          widths=[3.4, 1.5, 1.5])
    caption("Table 18.4. The language branch step by step (VOC: the July runs and the eight extra "
            "hypotheses; a pair re-grounded is one certified on the sample).")
    _A, _B = _RB["A"], _RB["B"]
    _vw, _sw = _RB["VOC val"], _RB["SBD"]
    p("The sample stage is no stricter on SBD — %.0f%% of the pairs are systematic there, %.0f%% "
      "on VOC. The difference is made by the re-grounding: on VOC %.0f%% of what the sample "
      "validated is still systematic on the whole class, on SBD %.0f%%. Two explanations are "
      "possible, and both were tested on the groundings already on disk, with SliceLens's own "
      "test (slice_quality/examples/slope_robustness.py; for VOC on the %d hypotheses re-grounded "
      "in July, %d of them systematic, %.0f%%, median slope %.2f against %.2f on SBD)."
      % (_pct(_fs[1], _fs[0]), _pct(_fv[1], _fv[0]), _pct(_fv[3], _fv[2]), _pct(_fs[3], _fs[2]),
         _vw["n"], _vw["systematic"], _pct(_vw["systematic"], _vw["n"]), _vw["median_slope"],
         _sw["median_slope"]))
    bullet("Size. The %d re-grounded SBD hypotheses were tested again on random subsets of SBD "
           "with VOC's number of images, 1 449, a hundred times. On subsets of VOC's size %.1f of "
           "them are systematic on average (%.0f%%; 5–95%% of the subsets between %d and %d), "
           "against %d (%.0f%%) on the whole of SBD and %.0f%% on VOC's July hypotheses; the "
           "median slope comes back "
           "to %.2f. At VOC's size, SBD looks like VOC."
           % (_sw["n"], _A["systematic_mean"], _pct(_A["systematic_mean"], _sw["n"]), _A["p05"], _A["p95"],
              _sw["systematic"], _pct(_sw["systematic"], _sw["n"]), _pct(_vw["systematic"], _vw["n"]),
              _A["median_slope_mean"]))
    _bv, _bs = _B["VOC val | no boundary"], _B["SBD | no boundary"]
    _bv2, _bs2 = _B["VOC val | no boundary+extent"], _B["SBD | no boundary+extent"]
    bullet("Coarser outlines. If SBD's polygons made boundary errors noisy, the test should "
           "recover once they stop counting. With boundary errors treated as correct, SBD goes "
           "from %.0f%% to %.0f%% (%.0f%% without extent errors as well), VOC from %.0f%% to "
           "%.0f%% (%.0f%%): the annotations are not what flattens SBD's slopes."
           % (_pct(_sw["systematic"], _sw["n"]), _pct(_bs["systematic"], _bs["n"]), _pct(_bs2["systematic"], _bs2["n"]),
              _pct(_vw["systematic"], _vw["n"]), _pct(_bv["systematic"], _bv["n"]), _pct(_bv2["systematic"], _bv2["n"])))
    p("The slope-trend test of SliceLens is not neutral to the size of the population. Its "
      "statistic is the largest positive slope over windows of the confidence range; on a small "
      "population a window holds few instances and the largest of many noisy slopes is easily "
      "above 0.20. The certified situations of VOC are therefore not a property of the situations "
      "alone: some of them would probably not survive on more images of the same kind. SBD does "
      "not certify less because its errors are less systematic, but because the test is harder "
      "to pass on a whole SBD class than on the same class in VOC, six times smaller. A criterion that knows the size of the "
      "population — the null distribution of the maximum slope obtained by permuting the error "
      "labels at the class's own N, with the multiple-testing correction of the rest of the "
      "pipeline — would make the two datasets comparable; §18.3 builds it.")

    try:
        from _revision_numbers import load_slope_verdicts as _lsv
        SV = _lsv()
    except Exception as _ex:
        SV = None
        print("  (slope-verdict numbers unavailable, section 18.3 skipped:", _ex, ")")
    if SV:
        _PS, _KS = SV["perm"]["stages"], SV["rank"]["stages"]
        _PC, _KC = SV["perm"]["compare"], SV["rank"]["compare"]
        h("18.3  A certification test that knows the size of the population", 2)
        p("Two verdicts were computed on the groundings already on disk — no new call to the VLM — "
          "and set beside SliceLens's rule. Both ask the slope test's question, whether errors are "
          "more frequent where the VLM says the situation holds; both measure it against chance at "
          "the population's own size, with the Benjamini–Hochberg correction the rest of the "
          "pipeline uses (q ≤ 0.05 within each dataset and stage) "
          "(slice_quality/examples/slope_permutation.py):")
        bullet("Permutation null. SliceLens's own statistic, computed exactly as it computes it — the "
               "confidence bins and the windows depend on the confidences only, so the slope of every "
               "window is a linear function of the bins' error rates — compared with its distribution "
               "when the error labels are shuffled among the instances, %d times." % SV["perm"]["draws"])
        bullet("Rank test. The area under the ROC curve of the VLM's confidence for telling errors from "
               "correct instances (0.5: no association), one-sided, with a standard error from a "
               "bootstrap that resamples whole images (%d draws), because the objects of one image are "
               "not independent." % SV["rank"]["draws"])
        _row = lambda run, stage, lab: [lab, _PS[(run, stage)]["n"], _PS[(run, stage)]["old"],
                                        _PS[(run, stage)]["new"], _KS[(run, stage)]["new"]]
        table(["Groundings", "class × hypothesis pairs", "SliceLens: slope ≥ 0.20",
               "slope, permutation null", "AUC test"],
              [_row("VOC val", "sample", "VOC val, 120-instance sample"),
               _row("VOC val", "full", "VOC val, re-grounded on the whole class"),
               _row("SBD", "sample", "SBD, 120-instance sample"),
               _row("SBD", "full", "SBD, re-grounded on the whole class")],
              widths=[2.4, 1.0, 1.1, 1.1, 0.9])
        caption("Table 18.5. Hypotheses called systematic by the three verdicts, on the same groundings.")
        _sh = lambda S, run, stage, k: "%.1f%%" % (100 * S[(run, stage)][k])
        table(["Same groundings, error labels shuffled", "SliceLens: slope ≥ 0.20",
               "permutation null, p < 0.05", "AUC test, p < 0.05"],
              [[lab, _sh(_PS, run, stage, "null_old"), _sh(_PS, run, stage, "null_new"),
                _sh(_KS, run, stage, "null_new")]
               for run, stage, lab in (("VOC val", "sample", "VOC val, sample"), ("VOC val", "full", "VOC val, whole class"),
                                       ("SBD", "sample", "SBD, sample"), ("SBD", "full", "SBD, whole class"))],
              widths=[2.4, 1.3, 1.3, 1.3])
        caption("Table 18.6. Where nothing is there: the share of hypotheses each verdict calls "
                "systematic once every association has been destroyed. An honest test at 5% gives about 5%.")
        _vf, _sf = _PS[("VOC val", "full")], _PS[("SBD", "full")]
        p("SliceLens's rule calls systematic between %.0f%% and %.0f%% of the hypotheses whose "
          "association has been destroyed. Given a null at the right size, the same statistic becomes "
          "honest but finds almost nothing on the real data either: %d of the %d hypotheses "
          "re-grounded on VOC, %d of the %d on SBD. The rank test is honest on the whole classes "
          "(%.1f%% and %.1f%% on shuffled labels) and finds %d and %d. The associations are there; "
          "the statistic is the problem — the largest slope over windows of a few noisy bins is a "
          "high-variance statistic, inflated by a fixed threshold on small samples and powerless "
          "against its own null. (On the 120-instance samples the rank test is still somewhat "
          "generous: %.0f–%.0f%% on shuffled labels.)"
          % (100 * min(v["null_old"] for v in _PS.values()), 100 * max(v["null_old"] for v in _PS.values()),
             _vf["new"], _vf["n"], _sf["new"], _sf["n"], 100 * _KS[("VOC val", "full")]["null_new"],
             100 * _KS[("SBD", "full")]["null_new"], _KS[("VOC val", "full")]["new"], _KS[("SBD", "full")]["new"],
             100 * min(_KS[("VOC val", "sample")]["null_new"], _KS[("SBD", "sample")]["null_new"]),
             100 * max(_KS[("VOC val", "sample")]["null_new"], _KS[("SBD", "sample")]["null_new"])))
        _ksz = SV["rank"]["size_check"]
        p("The size check of §18.2, repeated with the rank test: on random SBD subsets of VOC's size "
          "it calls %.1f of the %d hypotheses systematic (%.0f%%), against %d (%.0f%%) on the whole "
          "of SBD — with more data it finds more, as a test should; SliceLens's rule went the other "
          "way (%.0f%% on the subsets, %.0f%% on the whole)."
          % (_ksz["new_mean"], _ksz["k"], 100 * _ksz["new_mean"] / _ksz["k"], _KS[("SBD", "full")]["new"],
             100 * _KS[("SBD", "full")]["new"] / _ksz["k"], 100 * SV["perm"]["size_check"]["old_mean"] / _ksz["k"],
             100 * _sf["old"] / _sf["n"]))
        _cv = lambda c: 100.0 * c["cov_union"] / c["n_err"]
        _rw = []
        for run, lab in (("VOC val", "VOC2012 val"), ("SBD", "SBD")):
            b, pp, kk = _KC[run]["summary_base"], _PC[run]["summary_new"], _KC[run]["summary_new"]
            _rw += [[lab + ", certified situations", _KC[run]["base"], _PC[run]["new"], _KC[run]["new"]],
                    [lab + ", images with a situational description", b["situational_images"],
                     pp["situational_images"], kk["situational_images"]],
                    [lab + ", errors inside a certified group", "%.1f%%" % _cv(b), "%.1f%%" % _cv(pp), "%.1f%%" % _cv(kk)]]
        table(["Run (rescored copies)", "SliceLens's rule (before 21 October)", "permutation null", "AUC test"], _rw,
              widths=[3.0, 1.2, 1.1, 1.1])
        caption("Table 18.7. The certified situations of the inst and SBD runs under each verdict; "
                "everything else in the pipeline unchanged.")
        _kv = _KC["VOC val"]
        _tv = any(c == "tvmonitor" and "unusual setting" in (h_ or "") for c, h_ in _kv["lost"])
        p("With the rank test SBD certifies as many situations as VOC (%d and %d) and the paradox of "
          "§18.2 disappears. On VOC it keeps %d of the %d situations of the thesis run, adds %d and "
          "drops %d%s. Since 21 October the thesis runs use the rank test, and their judges and "
          "validations were run again (§18.5). The latest version of the SliceLens paper (GH-ESD, v4) "
          "also mentions a rank-correlation test as an alternative to the slope."
          % (_KC["SBD"]["new"], _kv["new"], len(_kv["kept"]), _kv["base"], len(_kv["gained"]), len(_kv["lost"]),
             " — among them the tvmonitor “unusual setting”, which the visual check of §13.4 had found vague"
             if _tv else ""))
    try:
        from _revision_numbers import load_sbd_dd as _lsdd
        SDD = _lsdd()
    except Exception as _ex:
        SDD = None
        print("  (SBD data-driven numbers unavailable, section 18.4 skipped:", _ex, ")")
    if SDD:
        _sn = sorted(SDD["cmp"]["new"].values(), key=lambda v: -v["size"])
        h("18.4  The data-driven hypotheses on SBD", 2)
        p("The generator of §13.8 was also run on SBD, on the same %d classes and from SBD's own "
          "instances (outputs/chain_dd_sbd.sh): %d phrases, grounded on the 120-instance samples of "
          "the SBD run. On the sample %d are systematic for SliceLens's slope rule and %d for the rank "
          "test of §18.3. The %d that were certified on the sample or significant for the rank test "
          "were re-grounded on every instance of their class (chain_dd_sbd_full.sh): there the slope "
          "rule keeps %d — %s — and the rank test %d of the %d (median AUC %.2f)."
          % (SDD["n_classes"], SDD["n_hyp"], SDD["sample_slope"], SDD["sample_auc"], SDD["n_full"],
             SDD["full_slope"], "; ".join("“%s”" % q for _c, q in SDD["full_slope_queries"]),
             SDD["full_auc"], SDD["n_full"], SDD["full_median_auc"]))
        p("With the slope rule the certified situations of a copy of the SBD run go from %d to %d: %s. "
          "%s, because the images they describe already carried a situation; %s. As on VOC (§13.8), "
          "the data-driven phrases add little and describe light, background and occlusion; the rank "
          "test admits more of them, but the copy with that verdict was not rescored."
          % (SDD["cmp"]["base"], SDD["cmp"]["dd"],
             "; ".join("“%s” (%d errors, %s)" % (v["hypothesis"], v["size"], _TYN.get(v["type"], v["type"]))
                       for v in _sn),
             "No image gains a situational description" if not SDD["cmp"]["images_gained"]
             else "%d images gain a situational description" % SDD["cmp"]["images_gained"],
             "no situation is lost" if not SDD["cmp"]["lost"] else "%d situations are lost" % len(SDD["cmp"]["lost"])))
    if OV0:
        from _revision_numbers import load_inst_summary as _lis_s, load_cause as _lca_s
        _ROOT = os.path.dirname(DOCS)
        _IS0 = _lis_s(_out_path("slice_quality_inst_slope"))
        _CZ0 = _lca_s(_out_path("slice_quality_inst_slope"))
        _JC0 = json.load(open(_out_path("slice_quality_inst", "judge_claims_v2gemma.json"),
                              encoding="utf-8"))["summary"]["all_narrative"]
        _sb0f = _out_path("slice_quality_sbd_slope", "sbd_summary.json")
        _SB0 = json.load(open(_sb0f, encoding="utf-8"))["run"] if os.path.exists(_sb0f) else None
        _OV = OV0
        _pl, _pb = _OV["prec"]["summary"]["language"], _OV["before"]["prec"]["summary"]["language"]
        _ca, _cb = CZS["all"]["pipeline"], _CZ0["all"]["pipeline"]
        _cv0, _cv1 = 100.0 * _IS0["cov_union"] / _IS0["n_err"], 100.0 * IS["cov_union"] / IS["n_err"]
        _sfail = [s for f in _OV["split"]["folds"] for s in f["slices"] if not s["replicated"]]
        _c25, _c50, _cbf = _OV["corr"]["p25"]["rule_a"], _OV["corr"]["p50"]["rule_a"], _OV["before"]["corr"]["rule_a"]
        _m2 = _OV["model2"]
        _B44 = _out_path("slice_quality_inst_auc44")
        _IS44, _CZ44 = _lis_s(_B44), _lca_s(_B44)
        from _revision_numbers import load_official_validations as _lov44
        _OV44 = _lov44(_B44)
        _pl44, _m244 = _OV44["prec"]["summary"]["language"], _OV44["model2"]
        _ca44, _cv44 = _CZ44["all"]["pipeline"], 100.0 * _IS44["cov_union"] / _IS44["n_err"]
        # the official run of 23 October (111 situations), kept when the 10-correct rule was applied (§18.9)
        _B111 = _out_path("slice_quality_inst_auc111")
        _IS111, _CZ111 = _lis_s(_B111), _lca_s(_B111)
        _OV111 = _lov44(_B111)
        _pl111, _m2111 = _OV111["prec"]["summary"]["language"], _OV111["model2"]
        _ca111, _cv111 = _CZ111["all"]["pipeline"], 100.0 * _IS111["cov_union"] / _IS111["n_err"]
        from _revision_numbers import load_rule_min as _lrm
        _RM = _lrm()
        h("18.5  The official run under the rank test (21, 23 and 25 October)", 2)
        p("After these results the rank test became the certification of the language branch in the "
          "thesis runs. Inst and SBD were rescored with it on the saved groundings "
          "(outputs/chain_auc_official.sh; the slope-rule versions are kept as "
          "outputs/slice_quality_{inst,sbd}_slope), and the certified slices came out identical to "
          "the rescored copies of Table 18.7; the cluster branch does not change. One limit stays and "
          "is declared: the rank verdict judges the hypotheses that had been re-grounded on the whole "
          "class, and the re-grounding had followed the slope rule on the sample. %d hypotheses on "
          "VOC and %d on SBD are significant for the rank test on the sample but had never been "
          "re-grounded. On VOC they were grounded on 22 October, the situations they add were validated in "
          "the night of 22–23 October, and on 23 October they were made official (§18.8): the rank test now "
          "decides at both steps. On 24 October the rank test was found to fail in classes with almost no "
          "correct object, and a minimum of ten was set; on 25 October the official run was rescored under "
          "it, with SliceLens's data-driven hypotheses added by the same protocol (§18.9). That run is the "
          "last column of Table 18.8. On SBD the limit stays until its own %d hypotheses, being grounded "
          "since 23 October, are done."
          % (_OV["pending"]["VOC val"], _OV["pending"]["SBD"], _OV["pending"]["SBD"]))
        p("The judges and four validation experiments were then run again on the official run "
          "(outputs/chain_night_inst.sh, guarded GPU jobs):")
        _same = "unchanged (cluster branch)"
        _fmtp = lambda c: "%.2f vs %.2f (%d images)" % (c["score"], c["control"], c["n"])
        _fmtq = lambda q: "%.3f vs %.3f; %d of %d groups above chance" % (q["precision"], q["control_rate_same_class"],
                                                                          q["groups_above_chance"], q["groups"])
        _fmtm = lambda m: "%d of %d (%d significant for the rank test)" % (m["recertified"], m["official_language"],
                                                                           m["auc_sys"])
        _rows = [["Certified situations / cluster slices, VOC val", "%d / %d" % (_IS0["slices"]["language"], _IS0["slices"]["cluster"]),
                  "%d / %d" % (_IS44["slices"]["language"], _IS44["slices"]["cluster"]),
                  "%d / %d" % (_IS111["slices"]["language"], _IS111["slices"]["cluster"]),
                  "%d / %d" % (IS["slices"]["language"], IS["slices"]["cluster"])],
                 ["Errors inside a certified group, VOC val", "%.1f%%" % _cv0, "%.1f%%" % _cv44, "%.1f%%" % _cv111,
                  "%.1f%%" % _cv1],
                 ["Images with a situational description, VOC val", "%d" % _IS0["situational_images"],
                  "%d" % _IS44["situational_images"], "%d" % _IS111["situational_images"],
                  "%d" % IS["situational_images"]]]
        if _SB0 and SB0:
            _rows.append(["Certified situations; errors inside a certified group, SBD",
                          "%d; %.1f%%" % (_SB0["slices"]["language"], 100.0 * _SB0["cov_union"] / _SB0["n_err"]),
                          "%d; %.1f%%" % (SB0["run"]["slices"]["language"], 100.0 * SB0["run"]["cov_union"] / SB0["run"]["n_err"]),
                          "as on 21 October", "as on 21 October (every class has ≥ 10 correct objects)"])
        _rows += [["Facts score, Gemma (0–10) vs control", "%.2f vs %.2f (earlier narrative)" % (_JC0["genuine"], _JC0["control"]),
                   "%.2f vs %.2f" % (_lcl["summary"]["all_narrative"]["genuine"], _lcl["summary"]["all_narrative"]["control"]),
                   "unchanged (the narrative does not name situations)", "unchanged"],
                  ["Plausibility of the blamed situation (0–10) vs control", _fmtp(_cb), _fmtp(_ca44), _fmtp(_ca111),
                   _fmtp(_ca)],
                  ["Situations re-verified by the VLM: precision vs same-class control",
                   "%.3f vs %.3f (reference typing)" % (_pb["precision"], _pb["control_rate_same_class"]),
                   _fmtq(_pl44), _fmtq(_pl111), _fmtq(_pl)],
                  ["Split-half replication of the cluster slices",
                   "%d of %d (reference typing)" % (_OV["before"]["split"]["n_replicated"], _OV["before"]["split"]["n_discovered"]),
                   "%d of %d" % (_OV["split"]["n_replicated"], _OV["split"]["n_discovered"]), _same, _same],
                  ["Second segmenter: certified situations that re-certify",
                   "%d of %d (reference typing)" % (_OV["before"]["model2"]["recertified"], _OV["before"]["model2"]["official_language"]),
                   _fmtm(_m244), _fmtm(_m2111), _fmtm(_m2)],
                  ["Corruption, small chairs made into misses: inside a certified slice",
                   "%d of %d (reference typing, captioning tags)" % (_cbf["recovered"], _cbf["injected"]),
                   "%d of %d (low dose); %d of %d (high dose)" % (_c25["recovered"], _c25["matched_as_fn"],
                                                                 _c50["recovered"], _c50["matched_as_fn"]), _same, _same]]
        table(["Measure", "before", "rank test, 21 October", "official run, 23 October",
               "official run, 25 October"], _rows, widths=[1.75, 1.2, 1.2, 1.2, 1.25])
        caption("Table 18.8. The thesis run before the rank test; with the rank test judging the hypotheses the "
                "slope rule had sent to the whole class (21 October); the official run where the rank test also "
                "decides which hypotheses are grounded on the whole class (23 October, §18.8); and the official run "
                "now, where the rank test is run only in classes with at least ten correct objects and SliceLens's "
                "data-driven hypotheses are added (25 October, §18.9). Rows marked “reference typing” compare with "
                "the validations of §6, measured on the earlier typing; “unchanged” rows do not depend on the "
                "language branch.")
        _cars = sum(1 for c, _q in _m2["not_recertified"] if c == "car")
        _hv = _OV["corr"]["p50"]["host_verdicts"]
        p("Reading the table. More situations are certified and more of the errors are explained, and "
          "the descriptions keep their quality: the VLM confirms the shipped situations on %.0f%% of "
          "their members against %.0f%% on comparable objects outside them — lower than the %.0f%% of "
          "the reference typing, because the rank test admits situations the slope rule had missed, "
          "but far above chance — and the judges score the narrative and the blamed situation as "
          "before. The split-half replication now rests on very few groups: the official run "
          "certifies %d cluster slices, the halves discover %d, and the one that does not replicate "
          "holds %d members on the held-out half, below the size gate. On the second segmenter %d of "
          "the %d certified situations re-certify (%d stay significant for the rank test); %d of those "
          "that do not are about cars. In the "
          "corruption experiment, re-run on the inst typing with the verification channel, the small "
          "chairs made into misses fall in a certified slice in %d of %d cases at the low dose and %d "
          "of %d at the high one; at the high dose %d of them share one group that fails certification "
          "— the limit of §6.1: “small” is hard for the VLM to confirm, so the group cannot prove "
          "what its members have in common. The car rule of §6.1 gives the same reading as before: "
          "%d of %d cars cut off at the border pass the VLM's confidence gate."
          % (100 * _pl["precision"], 100 * _pl["control_rate_same_class"], 100 * _pb["precision"],
             IS["slices"]["cluster"], _OV["split"]["n_discovered"], _sfail[0]["n_test"] if _sfail else 0,
             _m2["recertified"], _m2["official_language"], _m2["auc_sys"], _cars,
             _c25["recovered"], _c25["matched_as_fn"], _c50["recovered"], _c50["matched_as_fn"],
             _hv.get("DISCARD", 0), _OV["corr"]["p25"]["rule_b"]["passed"], _OV["corr"]["p25"]["rule_b"]["injected"]))
        p("Reproduce: outputs/chain_auc_official.sh (CPU) and outputs/chain_night_inst.sh (GPU 1, 2 and 6, "
          "every container under outputs/gpu_guard.sh): split_half.py, eval_description_precision.py, "
          "judge_cause_plausibility.py, corruption_experiment.py (TAG_MODE=verify, CORRUPTION_TAG_HOSTS=1), "
          "run_second_model_cluster.py and reground_validated.py on the second segmenter's objects, "
          "dd_auc.py with RANK_COPY, run_second_model_language.py — all with SQ_OUT=outputs/slice_quality_inst; "
          "outputs/chain_claims_night.sh for the facts score. Results in outputs/slice_quality_inst "
          "(night/ for the logs, model2/ for the second segmenter).", italic=True)
        from _revision_numbers import load_slicelens_cmp as _lslc
        _SLC = _lslc()
        if _SLC.get("sim") and _SLC.get("facts") and _SLC.get("ea"):
            _cell, _F, _E, _G, _RB, _DS = (_SLC["sim"]["cell"], _SLC["facts"], _SLC["ea"], _SLC["gesd"],
                                          _SLC.get("rubric"), _SLC.get("disagree"))
            _EA, _FA, _RA = _SLC.get("ea_auc") or _E, _SLC.get("facts_auc"), _SLC.get("rubric_auc")
            _svc = SV0["rank"]["compare"] if SV0 else None
            h("18.6  SliceLens alone against the pipeline, and the slope rule against the rank test", 2)
            p("The rank test replaced SliceLens's slope rule because the slope rule failed two checks: it "
              "calls “systematic” up to a third of the hypotheses when the labels are shuffled, and it "
              "certifies fewer situations on a dataset six times larger. Neither check says which test finds "
              "the RIGHT situations, because on real labels the truth is unknown; nor do they compare the "
              "thesis with SliceLens as a whole. Both are done here, on the same data, with the same VLM and "
              "the same hypotheses, in three configurations: SliceLens as published (language branch only, "
              "slope rule, an error is right or wrong), SliceLens with the rank test, and the full pipeline "
              "(typed errors, cluster branch, rank test, per-image narrative). SliceLens released no code, so "
              "its configuration is this thesis's implementation of the paper (§4.4).")
            _RU = _SLC.get("rules")
            p("Effects of known strength. The real confidence vectors of Qwen (%d whole-class groundings of VOC "
              "and SBD, their scale, ties and clumps unchanged) were given synthetic error labels whose "
              "dependence on the confidence is known: P(error) = sigmoid(a + b·z), z the normal score of the "
              "confidence's rank, a set so that the class keeps its real error rate. b = 0 is the null, where "
              "every “systematic” is a false alarm; b > 0 is a real association. Objects were subsampled "
              "image by image to 120, 400, 1,500 and 5,000, the labels drawn %d times per phrase, and every "
              "decision rule applied to the same draws. SliceLens's rule comes in two versions. The first version "
              "of the paper, which this thesis implements (§4.4), bins the confidences, fits a line on the "
              "error-rate curve above each of ten thresholds and calls a hypothesis systematic when the largest "
              "slope exceeds 0.2. The final version (arXiv 2512.24592v4, Sec. 3.4) keeps the units whose "
              "confidence exceeds 0.5, sorts them, slides a window over them (windows under about 5 units "
              "discarded), regresses the error on the confidence in each window and compares the largest slope "
              "with 0.2; it adds a “multi-scale” Spearman test, retained when ρ ≥ 0.5 and p < 0.05. The paper "
              "does not give the window size, the stride or the scales, so they are chosen here and declared: "
              "windows of 10 and of 30 units with stride 1 for the slope; for Spearman, one test on all the "
              "units above 0.5, and a multi-scale version on windows of 20 and 50 units (stride half a window) "
              "that passes when any window does. The rank test is one-sided at 0.05, per cell a single test."
              % (_SLC["sim"]["pairs"], _RU["R"] if _RU else _SLC["sim"]["R"]))
            _sim_rows = []
            _rc = _RU["cell"] if _RU else {}
            for _b, _lab in ((0.0, "none (false alarms)"), (0.25, "weak"), (0.5, "medium"), (1.0, "strong")):
                for _n in (120, 400, 1500, 5000):
                    _c, _r = _cell.get((_b, _n)), _rc.get((_b, _n))
                    if _c and _r:
                        _sim_rows.append([_lab if _n == 120 else "", "%.2f" % _c["auc_mean"], "{:,}".format(_n)]
                                         + ["%.0f%%" % (100 * _r[k]) for k in ("v1", "v4-slope W10", "v4-slope W30",
                                                                              "v4-rho", "v4-rho-ms", "AUC")])
            table(["Planted association", "AUC it makes", "Objects", "slope, v1", "slope v4, W 10", "slope v4, W 30",
                   "Spearman v4", "Spearman v4, multi-scale", "rank test"], _sim_rows,
                  widths=[1.05, 0.6, 0.6, 0.6, 0.65, 0.65, 0.65, 0.75, 0.6])
            caption("Table 18.9. Share of “systematic” verdicts on effects of known strength, for every SliceLens "
                    "rule and the rank test (slice_quality/examples/slope_rules_sim.py, %d draws per phrase). v1: "
                    "the rule implemented in this thesis; v4: the final version of the paper, with the window sizes "
                    "it leaves open chosen as stated. The 5,000-object rows rest on the few SBD classes that large."
                    % (_RU["R"] if _RU else _SLC["sim"]["R"]))
            _iv, _isb = _SLC["sim"]["VOC"], _SLC["sim"]["SBD"]
            p("Reading the table. With no association the rank test stays near its nominal 5%% (%.0f%% on the "
              "smallest classes, where its bootstrap is least precise); the slope rule of the first version raises "
              "a false alarm in %.0f%% of the cases on classes the size of VOC's. With a real but weak association "
              "the rank test finds it more often as the data grow, as a test should — %.0f%% at 1,500 objects, "
              "%.0f%% at 5,000 — while the first slope rule finds it LESS often, %.0f%% and %.0f%%: its maximum "
              "slope over windows of objects flattens as the windows fill. That is the SBD paradox of §18.2 with "
              "the truth known. The final version's rules fail in other ways. Its slope over a sliding window of "
              "sorted units, read literally, calls almost everything systematic: %.0f–%.0f%% false alarms, because "
              "a 0/1 error regressed on confidences a few hundredths apart gives steep slopes by chance. Its "
              "Spearman test on all units never fires, not even on a strong association (at most %.1f%% of the draws): with "
              "a 0/1 error a rank correlation of 0.5 is out of reach. The multi-scale Spearman, read as any window "
              "passing, raises false alarms that grow with the data, up to %.0f%% at 5,000 objects. Other readings "
              "of the open parameters may behave better; none of them is a test calibrated against chance, which "
              "is what the rank test adds."
              % (100 * _cell[(0.0, 120)]["auc"], 100 * _cell[(0.0, 400)]["slope"], 100 * _cell[(0.25, 1500)]["auc"],
                 100 * _cell[(0.25, 5000)]["auc"], 100 * _cell[(0.25, 1500)]["slope"], 100 * _cell[(0.25, 5000)]["slope"],
                 min(100 * _rc[(0.0, n)][k] for n in (120, 400, 1500, 5000) for k in ("v4-slope W10", "v4-slope W30")),
                 max(100 * _rc[(0.0, n)][k] for n in (120, 400, 1500, 5000) for k in ("v4-slope W10", "v4-slope W30")),
                 max(100 * _rc[(b_, n)]["v4-rho"] for b_ in (0.25, 0.5, 1.0) for n in (120, 400, 1500, 5000)),
                 100 * _rc[(0.0, 5000)]["v4-rho-ms"]))
            _rr = _RU["real"] if _RU else {}
            p("The same defects show on the real labels. Of the %d phrases grounded on whole VOC classes, the first "
              "slope rule certifies %d, of which %d point the wrong way — the wrong objects get LESS of a “yes” than "
              "the correct ones (AUC < 0.5) — among them the tvmonitor “unusual setting” that the visual check of "
              "§13.4 had found vague; on SBD %d of %d. The final version's sliding slope certifies %d and %d of the "
              "VOC phrases (windows of 10 and 30), %d and %d of them the wrong way; its Spearman tests certify %d and "
              "%d. The rank test certifies %d, none the wrong way — by construction, since it is one-sided."
              % (sum(1 for r in _SLC["sim"]["real"] if r["ds"] == "VOC"), _iv["slope_sys"], len(_iv["inverted"]),
                 len(_isb["inverted"]), _isb["slope_sys"],
                 _rr["VOC"]["v4-slope W10"][0], _rr["VOC"]["v4-slope W30"][0], _rr["VOC"]["v4-slope W10"][1],
                 _rr["VOC"]["v4-slope W30"][1], _rr["VOC"]["v4-rho"][0], _rr["VOC"]["v4-rho-ms"][0],
                 _rr["VOC"]["AUC"][0]))
            if _DS:
                _dr = {r["group"]: r for r in _DS["rows"]}
                _a, _s, _bo = _dr["AUC only"], _dr["slope only"], _dr["both"]
                p("Where the two tests disagree. On VOC %d situations are certified by both tests, %d only by the "
                  "rank test and %d only by the slope rule. Two checks that use neither test were run on all of "
                  "them: the VLM asked again whether the phrase is true of 10 members, against same-class objects "
                  "outside (eval_description_precision.py, on both runs), and the same phrase on the second "
                  "segmenter's objects, where its members must be wrong more often than the rest (one-sided "
                  "Fisher test, BH over all the situations). Neither group wins: the VLM confirms the rank-only "
                  "situations on %.0f%% of their members and the slope-only ones on %.0f%%, but says “yes” to "
                  "%.0f%% of the objects outside the latter against %.0f%% outside the former; on the second "
                  "segmenter %d of %d rank-only and %d of %d slope-only situations replicate (%d of %d of those "
                  "certified by both). The advantage of the rank test is therefore its error rate and its "
                  "stability as the data grow, not better situations found on VOC."
                  % (_bo["n"], _a["n"], _s["n"], 100 * _a["precision"], 100 * _s["precision"], 100 * _s["control"],
                     100 * _a["control"], _a["replicated"], _a["n_m2"], _s["replicated"], _s["n_m2"],
                     _bo["replicated"], _bo["n_m2"]))
            _cmp = []
            if _svc:
                for _ds, _lab in (("VOC val", "VOC val"), ("SBD", "SBD")):
                    # VOC val: the official run, where the rank test also chose what to ground (§18.8)
                    _bb, _nn = _svc[_ds]["summary_base"], (IS if _ds == "VOC val" else _svc[_ds]["summary_new"])
                    _cmp.append(["Errors inside a certified group, %s" % _lab, "%.1f%%" % (100 * _bb["cov_language"] / _bb["n_err"]),
                                 "%.1f%%" % (100 * _nn["cov_language"] / _nn["n_err"]), "%.1f%%" % (100 * _nn["cov_union"] / _nn["n_err"])])
            _cmp += [["False alarms with no association (400 objects)", "%.0f%%" % (100 * _cell[(0.0, 400)]["slope"]),
                      "%.0f%%" % (100 * _cell[(0.0, 400)]["auc"]), "%.0f%%" % (100 * _cell[(0.0, 400)]["auc"])],
                     ["Weak association found (1,500 objects)", "%.0f%%" % (100 * _cell[(0.25, 1500)]["slope"]),
                      "%.0f%%" % (100 * _cell[(0.25, 1500)]["auc"]), "%.0f%%" % (100 * _cell[(0.25, 1500)]["auc"])],
                     ["Pascal-EA: victims inside a certified group", "%d of %d" % (_E["language_total"], _E["victims"]),
                      "%d of %d" % (_EA["language_total"], _EA["victims"]), "%d of %d" % (_EA["union"], _EA["victims"])],
                     ["Merges and splits recognised as such", "no", "no",
                      "yes: %.0f%% of the errors" % (100 * (IS["tax"]["merge"] + IS["tax"]["split"]) / IS["n_err"])],
                     ["Description vs human notes, facts score (control)", "%.2f (%.2f)" % (_F["sl"], _F["sl_ctrl"]),
                      "%.2f (%.2f)" % (_FA["sl"], _FA["sl_ctrl"]) if _FA else "not measured",
                      "%.2f (%.2f)" % (_F["ours"], _F["ours_ctrl"])]]
            if _RB:
                _cmp.append(["Description vs human notes, rubric (control)", "%.2f (%.2f)" % (_RB["sl"], _RB["sl_ctrl"]),
                             "%.2f (%.2f)" % (_RA["sl"], _RA["sl_ctrl"]) if _RA else "not measured",
                             "%.2f (%.2f)" % (_RB["ours"], _RB["ours_ctrl"])])
            _cmp.append(["GESD: expert situations whose naming phrase is flagged", "%d of %d" % (_G["slope"], _G["n"]),
                         "%d of %d" % (_G["auc"], _G["n"]), "%d of %d" % (_G["auc"], _G["n"])])
            _GXc = (GS0 or {}).get("cmp") if GS0 else None
            if _GXc:
                _gk = lambda k: "%.3f" % _GXc["cols"][k]["p10"] + (
                    " (%.3f)" % _GXc["cols_all"][k]["p10"] if _GXc.get("cols_all") else "")
                _cmp.append(["GESD: the benchmark's Precision@10 (correct objects kept)",
                             _gk("SliceLens, verdict only"), _gk("+ rank test, verdict only"), _gk("pipeline")])
            _ST = _SLC.get("stats") or {}
            _pf = lambda p: "p < 0.0001" if p < 1e-4 else "p %.2g" % p
            _cvt = lambda ds, k="pipeline vs SliceLens": (
                "%+.1f points [%+.1f, %+.1f]; McNemar %s" % (100 * _ST["coverage"][ds][k]["diff"],
                                                           100 * _ST["coverage"][ds][k]["ci"][0],
                                                           100 * _ST["coverage"][ds][k]["ci"][1],
                                                           _pf(_ST["coverage"][ds][k]["p_mcnemar"])))
            _sim = lambda key: ("%.0f–%.0f%% against %.0f–%.0f%%" % tuple(100 * v for v in (
                _ST["sim"][key]["slope_ci"] + _ST["sim"][key]["auc_ci"])))
            _jdt = lambda key: ("%+.2f [%+.2f, %+.2f]; Wilcoxon %s" % (_ST["judges"][key]["diff"], _ST["judges"][key]["ci"][0],
                                                                       _ST["judges"][key]["ci"][1],
                                                                       _pf(_ST["judges"][key]["p_wilcoxon"])))
            _tst = {}
            if _ST:
                _tst = {"Errors inside a certified group, VOC val": _cvt("VOC val"),
                        "Errors inside a certified group, SBD": _cvt("SBD"),
                        "False alarms with no association (400 objects)": _sim("0.0_400"),
                        "Weak association found (1,500 objects)": _sim("0.25_1500"),
                        "Pascal-EA: victims inside a certified group": "%+d victims; McNemar %s" % (
                            _ST["ea"]["found"]["pipeline"] - _ST["ea"]["found"]["SliceLens"],
                            _pf(_ST["ea"]["pipeline vs SliceLens"]["p_mcnemar"])),
                        "Description vs human notes, facts score (control)": _jdt("facts vs SliceLens"),
                        "Description vs human notes, rubric (control)": _jdt("rubric vs SliceLens")}
                if _GXc:
                    _tg = _GXc["tests"]["pipeline vs SliceLens, verdict only"]
                    _tk = (_GXc.get("tests_all") or {}).get("pipeline vs SliceLens, verdict only")
                    _tst["GESD: the benchmark's Precision@10 (correct objects kept)"] = (
                        "%+.3f [%+.3f, %+.3f]; Wilcoxon %s" % (_tg["diff"], _tg["ci"][0], _tg["ci"][1], _pf(_tg["p_wilcoxon"]))
                        + (" (kept: %+.3f, %s)" % (_tk["diff"], _pf(_tk["p_wilcoxon"])) if _tk else "")
                        + " — SliceLens higher, §20.5")
            _cmp = [r + [_tst.get(r[0], "—")] for r in _cmp]
            table(["Measure", "SliceLens", "+ rank test", "this thesis",
                   "this thesis − SliceLens: 95% interval; paired test"], _cmp, widths=[2.2, 0.9, 0.9, 1.05, 1.55])
            caption("Table 18.10. SliceLens as published, SliceLens with the rank test, and the pipeline, on the "
                    "same data, VLM and hypotheses. VOC val with the rank test: the official run, where it also "
                    "chose which hypotheses to ground on the whole class (§18.8) and is run only in classes with at "
                    "least ten correct objects (§18.9); its two description rows were measured on the run of 21 "
                    "October. SBD: the rank test on the hypotheses "
                    "the slope rule had sent to the whole class. Pascal-EA with the rank test: the val-EA groundings "
                    "rescored with it, under the same minimum (§19 reports the val-EA run, which keeps the slope "
                    "rule: %d victims found). "
                    "Last column: the difference between the thesis and SliceLens on the same errors, victims or images, "
                    "with its 95%% interval (image-block bootstrap; for the simulation, the two Wilson intervals) and a "
                    "paired test." % _E["union"])
            p("Reading the table. Every row but those of GESD favours the thesis, and each has its own cause. The rank "
              "test alone raises the share of explained errors (it certifies more, and more reliably); the "
              "cluster branch adds the rest, most visibly on SBD and on the planted cause of Pascal-EA, where "
              "%d of the %d victims the pipeline finds are in a cluster slice only. The rank test alone does not "
              "change the descriptions (%s against %s on the facts score): that gain is the narrative's, built on "
              "the typed errors. Merges and splits exist only "
              "because Block 1 types the errors. For the descriptions SliceLens was given the most generous text "
              "it could write: the image's wrong objects by class and, for those inside a certified slice, its "
              "phrase — what it knows, without the error type or any measure. Checked fact by fact against the "
              "%d annotated VOC images that have both texts, it states %.0f%% of the annotators' facts, the "
              "narrative %.0f%%; per image the narrative scores higher %d times, lower %d. The last row is where "
              "SliceLens does not lose: on GESD the slope rule flags the phrase naming an expert situation for "
              "%d of %d situations and the rank test for %d, but in %d of the slope rule's the phrase points the "
              "wrong way (AUC < 0.5), and the rank test is the one that catches “tiny wrist watches”.%s"
              % (_EA["cluster"], _EA["union"], "%.2f" % _FA["sl"] if _FA else "—", "%.2f" % _F["sl"], _F["n"],
                 100 * _F["sl_v"]["stated"], 100 * _F["ours_v"]["stated"],
                 _F["ours_wins"], _F["sl_wins"], _G["slope"], _G["n"], _G["auc"], _G["slope_inverted"],
                 (" On GESD's own measure SliceLens scores higher; §20.5 shows that the measure rewards the number "
                  "of certified slices, not what they say." if _GXc else "")))
            if _ST:
                _c, _e, _dg = _ST["coverage"], _ST["ea"], _ST.get("disagree") or {}
                _ci = lambda ds, k: "%+.1f points [%+.1f, %+.1f]" % (100 * _c[ds][k]["diff"], 100 * _c[ds][k]["ci"][0],
                                                                    100 * _c[ds][k]["ci"][1])
                _ro = next((k for k in (_dg.get("precision") or {}) if not k.startswith("slope") and k != "p_mannwhitney"), None)
                _so = next((k for k in (_dg.get("precision") or {}) if k.startswith("slope")), None)
                p("Are the differences larger than chance? Every comparison of Table 18.10 is paired — the same "
                  "errors, victims, images or simulated draws under both methods — and is tested as such "
                  "(slice_quality/examples/slicelens_cmp_stats.py). The thesis against SliceLens is significant on "
                  "every row but those of GESD, where the benchmark's measure favours SliceLens (§20.5). The tests "
                  "also say where each gain comes from. The rank test alone adds "
                  "%s of explained errors on VOC and %s on SBD; the cluster branch then adds %s on VOC and %s on SBD "
                  "(McNemar p < 0.0001 for all four). On Pascal-EA the rank test alone finds %d victims against %d, "
                  "a difference within chance (%d found only with the rank test, %d only with the slope rule; McNemar "
                  "p %.2f); the cluster branch adds %d more (%s). The descriptions gain %.2f points of facts score per "
                  "image (95%% interval %.2f to %.2f, higher on %d of %d images)%s."
                  % (_ci("VOC val", "+ rank test vs SliceLens"), _ci("SBD", "+ rank test vs SliceLens"),
                     _ci("VOC val", "pipeline vs + rank test"), _ci("SBD", "pipeline vs + rank test"),
                     _e["found"]["+ rank test"], _e["found"]["SliceLens"], _e["+ rank test vs SliceLens"]["only_b"],
                     _e["+ rank test vs SliceLens"]["only_a"], _e["+ rank test vs SliceLens"]["p_mcnemar"],
                     _e["found"]["pipeline"] - _e["found"]["+ rank test"], _pf(_e["pipeline vs + rank test"]["p_mcnemar"]),
                     _ST["judges"]["facts vs SliceLens"]["diff"], _ST["judges"]["facts vs SliceLens"]["ci"][0],
                     _ST["judges"]["facts vs SliceLens"]["ci"][1], _ST["judges"]["facts vs SliceLens"]["wins"],
                     _ST["judges"]["facts vs SliceLens"]["n"],
                     ("; where the two tests disagree on VOC, neither is better: precision %.2f against %.2f "
                      "(Mann–Whitney p %.2f), re-found on the second segmenter %d of %d against %d of %d (Fisher p %.2f)"
                      % (_dg["precision"][_ro], _dg["precision"][_so], _dg["precision"]["p_mannwhitney"],
                         _dg["replication"][_ro][0], _dg["replication"][_ro][1], _dg["replication"][_so][0],
                         _dg["replication"][_so][1], _dg["replication"]["p_fisher"])) if _ro and _so else ""))
            p("Reproduce: outputs/chain_slicelens_cmp.sh (GPU 0, 1, 2 under outputs/gpu_guard.sh; "
              "outputs/cmp_rg_gpu1.sh resumed GPU 1 after its guard stopped it): "
              "slicelens_style_texts.py, judge_claims.py and judge_descriptions.py with SQ_OUT="
              "outputs/slice_quality_slcmp, eval_description_precision.py on outputs/slice_quality_inst_slope, "
              "reground_validated.py on the second segmenter for the slope-only situations, "
              "slope_vs_auc_sim.py, ea_branch_coverage.py, disagreement_report.py; then outputs/chain_night_pending.sh "
              "(night of 21-22 October) for the “+ rank test” column (the official run's situations in SliceLens style, "
              "both judges; val-EA rescored with dd_auc.py RANK_COPY) and slope_rules_sim.py; "
              "outputs/chain_rule_ea_stats.sh (25 October, CPU) for val-EA under the ten-correct minimum "
              "(rank_min_correct.py, the run before kept as outputs/slice_quality_ea_auc_nomin) and the paired tests "
              "on the official run (the earlier ones kept as outputs/slicelens_cmp_stats_111.*).", italic=True)
        from _revision_numbers import load_slice_image_iou as _lsii
        _SI = _lsii()
        if _SI:
            _L, _C, _R = _SI["language"], _SI["cluster"], _SI["refs"]
            h("18.7  How badly the images of each certified slice are segmented", 2)
            p("The certification asks whether a group of errors is unusual; it does not say how much the "
              "segmentation suffers where the group's situation holds. That is measured here on the official "
              "VOC val run, with the masks and the computation of the dataset's mIoU (one confusion matrix per "
              "image, the predicted masks of the main segmenter). For every certified slice the images that "
              "hold its members are taken, and the IoU of the slice's class on them is set against the IoU of "
              "the same class on the other images that contain it. The class of a language slice is in its "
              "phrase; a cluster slice's is its members' majority.")
            table(["Images", "Number", "mIoU", "mean per-image mIoU"],
                  [[k, "{:,}".format(v["n_images"]), "%.1f" % (100 * v["miou_pooled"]), "%.1f" % (100 * v["miou_img"])]
                   for k, v in _R.items()], widths=[3.0, 0.9, 0.9, 1.5])
            caption("Table 18.11. The segmentation quality of the images in and out of the certified slices.")
            _bp = _SI["by_phrase"]
            _gen = [t for t in _bp if t[2] >= 2]
            p("The certified slices fall on the images the segmenter handles worst: %s images belong to at least "
              "one, and their mIoU is %.1f against %.1f on the images that have errors but no certified slice "
              "(%.1f on all images). Within each slice the effect is on its own class. For %d of the %d language "
              "slices the class is segmented worse on the slice's images than on its other images, by %.1f "
              "points on average (%.1f against %.1f; one-sided Wilcoxon p %s), and %s (Spearman ρ %.2f between "
              "the drop and the slice's images). For %d of the %d cluster slices the drop is %s, "
              "%.1f points (%.1f against %.1f). By phrase, the situations that cost most are %s; %s."
              % ("{:,}".format(_R["images in at least one certified slice"]["n_images"]),
                 100 * _R["images in at least one certified slice"]["miou_pooled"],
                 100 * _R["images with an error, in no certified slice"]["miou_pooled"],
                 100 * _R["all images"]["miou_pooled"], _L["worse"], _L["n"], 100 * _L["drop_mean"],
                 100 * _L["cls_in"], 100 * _L["cls_out"],
                 "< 0.0001" if _L["wilcoxon_p"] < 1e-4 else "%.4f" % _L["wilcoxon_p"],
                 ("the smaller the slice, the larger the drop" if _L["size_rho"] < 0 else
                  "the larger the slice, the larger the drop") if _L["size_p"] < 0.05 else
                 "the drop does not depend on the size of the slice", _L["size_rho"],
                 _C["worse"], _C["n"], "larger" if _C["drop_mean"] > _L["drop_mean"] else "smaller",
                 100 * _C["drop_mean"], 100 * _C["cls_in"], 100 * _C["cls_out"],
                 ", ".join("“%s” (%.0f points, %d slices)" % (k, 100 * v, n) for k, v, n in _gen[:3]),
                 (" and ".join("“%s”" % k for k, v, n in _gen if v <= 0.005) + " cost nothing")
                 if any(v <= 0.005 for k, v, n in _gen) else "every phrase held by two slices or more costs "
                 "something, the least “%s” (%.0f points)" % (_gen[-1][0], 100 * _gen[-1][1])))
            p("The exceptions are informative. In %d language slices and in %d cluster slice the class is not worse "
              "where the situation holds — among them %s. There the error is about the object, not its pixels: "
              "two potted plants merged into one mask are one wrong object, yet almost every pixel labelled "
              "potted plant is right. A caution on the measure: the mIoU of a slice's images taken alone "
              "misleads when the images are few, because it averages every class present and one class wrong "
              "in a corner of one image weighs as much as the slice's own class; the class IoU in and out of "
              "the slice is the comparison that holds."
              % (len(_L["not_worse"]), len(_C["not_worse"]),
                 ", ".join("“%s”" % r["name"].replace('"', "") for r in (_L["not_worse"][:2] + _C["not_worse"][:1]))))
            figure("fig_slice_iou_dots.png",
                   "Figure 18a. Every certified slice of VOC val: the IoU of its class on the slice's images (filled, "
                   "value beside it) and on the class's other images (hollow). Language slices in orange, cluster "
                   "slices as grey squares; sorted from the worst segmented.", width=6.3)
            p("Reproduce: slice_quality/examples/slice_image_iou.py (CPU; outputs/slice_quality_inst/"
              "slice_image_iou.{txt,json}); the figure by presentazione/ott2026/fig_slice_iou_dots.py.", italic=True)
        _pf = _out_path("slice_quality_pending", "dd_auc_full.json")
        _vf = _out_path("slice_quality_inst_auc111", "validation_report.json")      # the report of 23 October
        if os.path.exists(_pf) and os.path.exists(_vf):
            _pr = json.load(open(_pf, encoding="utf-8"))
            _pc = sorted({r["cls"] for r in _pr})
            _VR = json.load(open(_vf, encoding="utf-8"))
            from _revision_numbers import load_inst_summary as _lis_p
            _PS = _lis_p(_out_path("slice_quality_inst_pending"))
            _po, _pn, _pm = _VR["precision_official"], _VR["precision_new"], _VR.get("precision_mixed")
            _mo, _mn, _pv = _VR["model2_official"], _VR["model2_new"], _VR["plausibility"]
            _miss = [r for r in _mn["rows"] if not r["auc_sys"]]
            _er = _VR["model2_error_rate"]
            _pot = sum(1 for r in _miss if r["cls"] == "pottedplant")
            _n67 = len(_mn["rows"])
            h("18.8  The hypotheses the slope rule had kept from the whole class: grounded, validated, made "
              "official (22–23 October)", 2)
            p("The limit declared in §18.5 — %d hypotheses significant for the rank test on the 120-object sample "
              "of VOC val but never grounded on their whole class, because the re-grounding had followed the slope "
              "rule — was closed in three steps. On 22 October %d hypotheses of %d classes were grounded on every "
              "object of their class (outputs/chain_night_pending.sh, GPU 1, 2 and 6 under the guard; "
              "/data/outputs_pending); on the whole classes the rank test calls %d of them significant (BH within "
              "these %d; the slope rule %d). On a copy of the run rescored with them beside the official "
              "groundings (outputs/slice_quality_inst_pending; all the gates of the language branch applied, "
              "paraphrases merged) the certified situations went from %d to %d, the errors inside a certified "
              "group from %.1f%% to %.1f%%, the images with a situational description from %d to %d; the cluster "
              "branch did not change (%d slices). The slope rule's pre-selection was therefore hiding a large part "
              "of what the rank test certifies on VOC — the same direction as on SBD (§18.3)."
              % (OV0["pending"]["VOC val"], len(_pr), len(_pc), sum(1 for r in _pr if r["auc_sys"]), len(_pr),
                 sum(1 for r in _pr if r["slope_sys"]), _IS44["slices"]["language"], _PS["slices"]["language"],
                 _cv44, 100 * _PS["cov_union"] / _PS["n_err"], _IS44["situational_images"],
                 _PS["situational_images"], _PS["slices"]["cluster"]))
            p("Before being admitted, the %d added situations went through the checks of the official ones, on the "
              "copy, in the night of 22–23 October (outputs/chain_validate_pending.sh: Qwen on GPU 2 and 6, Gemma "
              "in 4-bit on GPU 4, every container under the guard; slice_quality/examples/pending_validation_report.py "
              "for the comparison). Table 18.12 sets them beside the %d of 21 October."
              % (_n67, _mo["n"]))
            table(["Check", "the %d of 21 October" % _mo["n"], "the %d added" % _n67],
                  [["Precision: the VLM asked again on 10 members of each phrase group, against objects of the "
                    "same class outside it",
                    "%.3f vs %.3f; %d of %d groups above chance" % (_po["precision"], _po["control"],
                                                                     _po["above_chance"], _po["groups"]),
                    "%.3f vs %.3f; %d of %d groups above chance" % (_pn["precision"], _pn["control"],
                                                                     _pn["above_chance"], _pn["groups"])],
                   ["Second segmenter: significant for the rank test (BH within each set); mean AUC",
                    "%d of %d; %.2f" % (_mo["auc_sys"], _mo["n"], _mo["auc_mean"]),
                    "%d of %d; %.2f" % (_mn["auc_sys"], _mn["n"], _mn["auc_mean"])],
                   ["Second segmenter: members (confidence ≥ 0.5) wrong more often than the rest (Fisher, BH); "
                    "median members",
                    "%d of %d; %.0f" % (_mo["fisher_sys"], _mo["n"], _mo["members_median"]),
                    "%d of %d; %.0f" % (_mn["fisher_sys"], _mn["n"], _mn["members_median"])],
                   ["Plausibility of the blamed situation (0–10) vs control",
                    "%.2f vs %.2f (%d images)" % (_ca44["score"], _ca44["control"], _ca44["n"]),
                    "%.2f on the %d images that gain an added situation (all images with a cause: %.2f vs %.2f, %d)"
                    % (_pv["genuine_new"], _pv["n_new"], _pv["genuine"], _pv["control"], _pv["n"])]],
                  widths=[2.9, 1.7, 2.0])
            caption("Table 18.12. The situations added by the hypotheses the slope rule had kept from the whole class, "
                    "against the official ones of 21 October, on the same checks. A phrase is one precision group "
                    "across classes, so the %d groups that hold both old and new situations are left out of the "
                    "precision row (%.3f vs %.3f)." % (_pm["groups"] if _pm else 0, _pm["precision"] if _pm else 0,
                                                       _pm["control"] if _pm else 0))
            p("They hold up as the official ones do. The VLM confirms their phrase on %.0f%% of the members against "
              "%.0f%% of comparable objects outside; on the second segmenter %d of %d are significant for the rank "
              "test, as %d of %d of the official ones; the judge finds the blamed situation as plausible on the "
              "images they describe (%.2f) as on all images. The check that reads worse is the neutral one, members "
              "against the rest at a fixed cut: %d of %d against %d of %d. It has little power here: the median "
              "added situation holds %.0f members on the second segmenter, which gets most of the objects of most "
              "classes wrong (%.0f%% of the potted plants and chairs, %.0f%% of the people), and %d of the %d "
              "still point the right way. Where they do not replicate the cause is mostly one class: %d of the %d "
              "that miss the rank test are potted plants, where almost nothing correct is left to compare with."
              % (100 * _pn["precision"], 100 * _pn["control"], _mn["auc_sys"], _mn["n"], _mo["auc_sys"], _mo["n"],
                 _pv["genuine_new"], _mn["fisher_sys"], _mn["n"], _mo["fisher_sys"], _mo["n"], _mn["members_median"],
                 100 * min(_er["pottedplant"], _er["chair"]), 100 * _er["person"],
                 sum(1 for r in _mn["rows"] if r["lift"] > 0), _mn["n"], _pot, len(_miss)))
            _m2 = _m2111
            p("On 23 October they were made official (outputs/chain_official_111.sh, CPU). The official run was "
              "rescored in place with the pending groundings, and it certifies exactly what the validated copy "
              "certified; the validations measured on the copy were moved into it; the second segmenter's "
              "groundings of the old and the added situations were joined and judged by the rank test as one set "
              "(%d of %d significant; %d re-certify through the whole pipeline on its objects); the analyses of "
              "§18.7 and the galleries were rebuilt. The run of 21 October is kept as "
              "outputs/slice_quality_inst_auc44. This official run, with %d certified situations, lasted two "
              "days: on 24 October the rank test itself was found to fail in classes with almost no correct "
              "object, the potted plants first, and §18.9 explains why %d of its situations were withdrawn on "
              "25 October. On SBD the same "
              "step is under way: its %d hypotheses significant on the sample are being grounded on the whole "
              "class."
              % (_m2["auc_sys"], _m2["n_regrounded"], _m2["recertified"], _IS111["slices"]["language"],
                 len(_RM["lost"]) if _RM else 0, OV0["pending"]["SBD"]))
            p("Reproduce: outputs/chain_night_pending.sh and pend_gpu6.sh (grounding), pending_rescore.sh (the "
              "copy), chain_validate_pending.sh and pending_validation_report.py (OFFICIAL=outputs/"
              "slice_quality_inst_auc44; the report in outputs/slice_quality_inst_auc111/validation_report.*), "
              "chain_official_111.sh (adoption).", italic=True)
        if _RM:
            _cl = {c["cls"]: c for c in _RM["classes"]}
            _bl, _fm, _dd, _vr = _RM["below"], _RM["fam"], _RM["dd"], _RM["val"]
            _nsc = [c for c in _RM["classes"] if c["shipped"] > c["shipped_conf"]]
            _gn = _RM["ge"]["not_conf"]
            _lost = Counter(c for c, _q in _RM["lost"])
            _nm = {"pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv monitor"}
            _cn = lambda c: _nm.get(c, c)
            _bt = _cl.get("boat", {})
            h("18.9  A rank test needs correct objects: a minimum of ten, and the data-driven hypotheses "
              "(24–25 October)", 2)
            p("The rank test of §18.3 compares the VLM's confidence on the wrong objects of a class with its "
              "confidence on the correct ones, and takes its p-value from a bootstrap over images. In three classes "
              "of VOC val the segmenter gets almost every object wrong: %s (Table 18.13). With one correct object "
              "the bootstrap has nothing to resample on that side; its spread collapses, and the p-value comes out "
              "near zero whatever the phrase says. An exact rank test shows it (one-sided Mann–Whitney, BH within "
              "each family as before; slice_quality/examples/rank_few_correct.py): it confirms %d of the %d "
              "potted-plant hypotheses the bootstrap had certified and %d of the %d chair ones. Of the %d situations "
              "the official run of 23 October shipped, %d are not confirmed by the exact test, %d of them potted "
              "plants or chairs. The fault came to light on 24 October, while SliceLens's data-driven hypotheses "
              "were being added to the run (four of the twelve the rank test keeps on their sample are potted "
              "plants)."
              % (", ".join("%d correct object%s among %d %ss" % (c["n_ok"], "" if c["n_ok"] == 1 else "s",
                                                                    c["n_ok"] + c["n_err"], _cn(c["cls"]))
                           for c in _bl),
                 _cl["pottedplant"]["confirmed"], _cl["pottedplant"]["cert"], _cl["chair"]["confirmed"],
                 _cl["chair"]["cert"], _RM["shipped"], _RM["shipped"] - _RM["shipped_conf"],
                 sum(c["shipped"] - c["shipped_conf"] for c in _nsc if c["cls"] in ("pottedplant", "chair"))))
            _tr = [[_cn(c["cls"]), "%d / %d" % (c["n_ok"], c["n_err"]), "%d" % c["cert"], "%d" % c["confirmed"],
                    "%d" % c["shipped"], "%d" % c["now"]] for c in _RM["classes"]]
            _tr.append(["all", "", "%d" % sum(c["cert"] for c in _RM["classes"]),
                        "%d" % sum(c["confirmed"] for c in _RM["classes"]), "%d" % _RM["shipped"],
                        "%d" % sum(c["now"] for c in _RM["classes"])])
            table(["Class", "correct / wrong objects", "certified by the bootstrap", "confirmed by the exact test",
                   "situations, 23 October", "situations, 25 October"], _tr, widths=[1.2, 1.15, 1.0, 1.0, 1.0, 1.0])
            caption("Table 18.13. VOC val, class by class, sorted by the number of correct objects: the hypotheses "
                    "the rank test certified on the whole class (the three families of the official run, before "
                    "the minimum), how many an exact rank test confirms, and the situations the official run "
                    "shipped before and after the minimum of ten correct objects (the last column includes the "
                    "data-driven situations). Bicycle has no certified hypothesis and is left out.")
            p("Since 24 October the rank test is run on a whole class only when the class has at least ten correct "
              "objects. The hypotheses of the other classes are “not testable”: they leave the BH family, and the "
              "p-values of the rest are kept as they were (slice_quality/examples/rank_min_correct.py re-verdicts "
              "the stored copies, after checking that a minimum of zero gives back the verdicts it reads; dd_auc.py "
              "applies the same minimum, MIN_CORRECT). The sample stage does not change: it only chooses what is "
              "grounded. Ten is a convention, not a derivation. In the classes that reach it the bootstrap and the "
              "exact test agree on %d of the %d certified hypotheses, and the %d where they differ are borderline "
              "(exact q from %.3f to %.3f). Below it the rule is cautious: the exact test would have kept all %d "
              "boat hypotheses (%d correct objects), and the rule withdraws them with the others."
              % (_RM["ge"]["confirmed"], _RM["ge"]["cert"], len(_gn), min(r["q_exact"] for r in _gn),
                 max(r["q_exact"] for r in _gn), _bt.get("confirmed", 0), _bt.get("n_ok", 0)))
            _ea = _RM.get("ea")
            p("On VOC val the minimum leaves out potted plant, chair and boat. The families of the official run "
              "lose %d of %d certified hypotheses (those grounded first, §18.5) and %d of %d (those of §18.8); "
              "none is gained, since no p-value changes. %d shipped situations are withdrawn: %s. On SBD and on "
              "GESD every class has at least ten correct objects, and nothing changes.%s"
              % (_fm["main"]["before"] - _fm["main"]["now"], _fm["main"]["before"],
                 _fm["pending"]["before"] - _fm["pending"]["now"], _fm["pending"]["before"], len(_RM["lost"]),
                 ", ".join("%d %s" % (k, _cn(c)) for c, k in _lost.most_common()),
                 (" On Pascal-EA (§18.6) %s: %d of its certified hypotheses are withdrawn (%d → %d over the "
                  "whole val-EA family), and the victims inside a certified group go from %d to %d of %d (in the "
                  "language branch from %d to %d)."
                  % (", ".join("%s keeps %d correct objects after the edits" % (_cn(c), k) for c, k in _ea["n_ok"].items()),
                     _ea["before"] - _ea["now"], _ea["before"], _ea["now"], _ea["cov_before"]["union"],
                     _ea["cov"]["union"], _ea["cov"]["victims"], _ea["cov_before"]["language_total"],
                     _ea["cov"]["language_total"])) if _ea else ""))
            _m2n = {(r["cls"], r["query"]): r for r in _vr["model2_new"]["rows"]} if _vr["model2_new"].get("rows") else {}
            _ddf = {(r["cls"], r["query"]): r for r in _dd["full"]}
            p("The same step brought SliceLens's own data-driven hypotheses into the official run (§13.8): %d "
              "phrases over %d classes, written by its generator from the images and until then grounded only on "
              "the 120-object sample. The protocol is the official run's: the rank test on the sample decides "
              "what goes to the whole class. It keeps %d (the slope rule %d); the %s ones are not grounded, since the "
              "minimum cannot test them on the whole class. The other %d were grounded on every object of their "
              "class, and all %d are significant there (median AUC %.2f; the slope rule calls %d). After the gates "
              "of the language branch %d become situations (Table 18.14); the seventh, “a train that is a small "
              "model or miniature replica”, keeps a single object and fails the size gate."
              % (_dd["n"], _dd["n_classes"], _dd["sample_auc"], _dd["sample_slope"],
                 " and ".join("%d %s" % (k, _cn(c).replace(" ", "-")) for c, k in _dd["excluded"].most_common()),
                 _dd["n_full"], _dd["full_auc"], _dd["full_median_auc"], _dd["full_slope"], len(_RM["gained"])))
            _tg = []
            for c, q in _RM["gained"]:
                r1, r2 = _ddf.get((c, q)), _m2n.get((c, q))
                _tg.append([_cn(c), "“%s”" % q, "%.2f" % r1["auc"] if r1 else "—", "%d" % r1["n_ok"] if r1 else "—",
                            ("%.2f%s" % (r2["auc"], " (significant)" if r2["auc_sys"] else "")) if r2 else "—"])
            table(["Class", "Data-driven phrase", "AUC, whole class", "correct objects", "AUC, second segmenter"], _tg,
                  widths=[0.9, 2.9, 0.8, 0.8, 1.1])
            caption("Table 18.14. The situations the data-driven hypotheses add to the official run: the rank test's "
                    "AUC on the whole class (all significant, BH within the seven grounded), the correct objects it "
                    "compares with, and the same phrase on the second segmenter's objects (BH within the six).")
            _pn, _po, _mn, _pv = _vr["precision_new"], _vr["precision_official"], _vr["model2_new"], _vr["plausibility"]
            p("Before being admitted, the %d went through the checks of §18.8, on a copy of the run "
              "(outputs/slice_quality_inst_withdd). The VLM confirms their phrase on %.0f%% of the members against "
              "%.0f%% of comparable objects outside (%d of %d groups above chance; %.0f%% against %.0f%% for the "
              "other situations); on the second segmenter %d of %d are significant for the rank test, and %s "
              "the members-against-the-rest test at a fixed cut, which has little power with a median of %.0f "
              "members; the judge finds the blamed situation plausible on the %d images that gain one (%.2f, against "
              "%.2f on all images with a cause and %.2f for the control)."
              % (len(_RM["gained"]), 100 * _pn["precision"], 100 * _pn["control"], _pn["above_chance"], _pn["groups"],
                 100 * _po["precision"], 100 * _po["control"], _mn["auc_sys"], _mn["n"],
                 "none passes" if not _mn["fisher_sys"] else "%d pass" % _mn["fisher_sys"],
                 _mn["members_median"], _pv["n_new"], _pv["genuine_new"], _pv["genuine"], _pv["control"]))
            p("On 25 October the official run was rescored in place with the families under the minimum and the "
              "data-driven groundings (outputs/chain_dd_official_v3.sh; the run of 23 October is kept as "
              "outputs/slice_quality_inst_auc111). It certifies exactly what the validated copy certified: %d "
              "situations and %d cluster slices, %.1f%% of the errors inside a certified group (%.1f%% on 23 "
              "October), %d images with a situational description (%d). The VLM confirms the shipped situations on "
              "%.0f%% of their members against %.0f%% outside; on the second segmenter %d of %d are significant for "
              "the rank test and %d re-certify through the whole pipeline; the blamed situation scores %.2f for "
              "plausibility against %.2f for the control (%d images). The analyses of §18.7, the comparison of "
              "§18.6 and the galleries were rebuilt on it; Table 18.8 has it in its last column."
              % (IS["slices"]["language"], IS["slices"]["cluster"], _cv1, _cv111, IS["situational_images"],
                 _IS111["situational_images"], 100 * _pl["precision"], 100 * _pl["control_rate_same_class"],
                 OV0["model2"]["auc_sys"], OV0["model2"]["n_regrounded"], OV0["model2"]["recertified"],
                 _ca["score"], _ca["control"], _ca["n"]))
            p("What the minimum costs. The language branch is now silent where the segmenter is worst: no "
              "situation can be certified for %s, which hold %s wrong objects of VOC val. Their errors are reached "
              "only by the cluster branch. A test for these classes would need correct objects from elsewhere — "
              "the training split, or the same class in another dataset — or a reference other than the correct "
              "objects, for instance the class's errors of another kind. Neither is done here."
              % (", ".join("%ss" % _cn(c["cls"]) for c in _bl[:-1]) + " or %ss" % _cn(_bl[-1]["cls"]),
                 " + ".join("{:,}".format(c["n_err"]) for c in _bl)
                 + " = {:,}".format(sum(c["n_err"] for c in _bl))))
            p("Reproduce: slice_quality/examples/rank_few_correct.py and rank_min_correct.py (CPU; reports in "
              "outputs/rank_few_correct.* and outputs/slice_quality_inst_withdd/rank_min_{main,pending}.txt); "
              "outputs/chain_dd_official_v2.sh (the data-driven groundings on GPU 6, /data/outputs_dd_voc_full; "
              "stopped by the guard when another user took the GPU) and chain_dd_official_v3.sh (GPU 4 under the "
              "guard: the second segmenter, the plausibility judge, the adoption; the precision check and the "
              "groundings are reused from v2); outputs/chain_rule_ea_stats.sh (CPU: val-EA under the minimum, the "
              "paired tests of §18.6). The local copies of the reports are in "
              "outputs/slice_quality_inst/rule_min_correct/.", italic=True)
    p("Reproduce: slice_quality/examples/build_sbd.py (layout, predictions, substrate, tags), "
      "outputs/chain_sbd.sh with chain_sbd_worker.sh on two GPUs (language branch: "
      "examples/ground_ea_hypotheses.py with DATASET=SBD, SAMPLING=srs; the person class split "
      "into 24 blocks by examples/ground_split_class.py), slice_quality/examples/slope_robustness.py "
      "doc_numbers_sbd_ea.py and slope_permutation.py (PHASE=test, rank, compare; "
      "outputs/chain_slope_perm.sh, chain_slope_rank.sh) on CPU; outputs in outputs/slice_quality_sbd/, "
      "outputs/slope_permutation/ and the rescored copies outputs/slice_quality_{inst,sbd}_{perm,rank}/ (gallery "
      "gallery_sbd_full.html) and /data/outputs_sbd.", italic=True)

# ============================================================================
#  19. A planted cause: material edits of VOC val (Pascal-EA, October 2026).
#      Numbers from _revision_numbers.load_ea -- nothing typed in.
# ============================================================================
if EAN:
    _S3 = EAN["seg"]["v3"]
    _PL = EAN["planted"]
    _M = EAN["miou"]["miou"]
    _nv = len(_PL["victims"])
    _ncov, _ninst = len(_PL["covered"]), len(_PL["covered_inst"])
    _sc = EAN["survivors_cov"]
    h("19. A planted cause: material edits of VOC val (October 2026)", 1)
    p("Every validation so far asks whether a group is real or whether its description is true; "
      "none knows what actually caused the errors. A planted cause does. Pascal-EA (CVPR 2024) "
      "edits the appearance of one object in a VOC image with a diffusion model and keeps its "
      "mask, so the ground truth still applies. Nico Catalano (AIRLab) generated the material "
      "edits of VOC val with the authors' code (“made of metal / paper / plastic / wood”); %d of them, one per image, "
      "replace the originals in a copy of val called val-EA. Nothing else changes: the other "
      "images, the segmenter, the protocol. The edits were not filtered by hand, as the "
      "benchmark's authors do, so some are crude." % EAN["n_edited"])

    h("19.1  What the edit does to the segmenter", 2)
    p("On %d pairs of original and edited image the recall of the edited object falls from %.3f "
      "to %.3f on average, but the median change is %.3f: most edits do nothing. %.0f%% of the "
      "objects lose at least 0.10 of their recall and %.0f%% (%d) at least 0.30, while outside the "
      "object the accuracy does not move (%.3f → %.3f): the edit acts on the object, not on the "
      "rest of the image. How much the edit changes the pixels barely predicts it (Spearman %.2f). "
      "Over the edited images the mIoU goes from %.1f to %.1f, over the whole of val from %.1f to "
      "%.1f."
      % (_S3["n"], _S3["rec_orig"], _S3["rec_edit"], _S3["median_delta"], 100 * _S3["drop_010"],
         100 * _S3["drop_030"], _S3["n_drop_030"], _S3["acc_out_orig"], _S3["acc_out_edit"],
         _S3["spearman_diffin_delta"][0],
         100 * _M["VOC val, edited images"]["miou"], 100 * _M["val-EA, edited images"]["miou"],
         100 * _M["VOC val"]["miou"], 100 * _M["val-EA"]["miou"]))
    _bm = {r["key"]: r for r in _S3["by_material"]}
    _bc = sorted((r for r in _S3["by_class"] if r["n"] >= 20), key=lambda r: r["delta"])
    table(["Edited object", "pairs", "recall: original → edited", "lose ≥ 0.30"],
          [["made of " + k, _bm[k]["n"], "%.3f → %.3f" % (_bm[k]["rec_orig"], _bm[k]["rec_edit"]),
            "%.0f%%" % (100 * _bm[k]["drop_030"])] for k in ("paper", "metal", "wood", "plastic") if k in _bm]
          + [[r["key"], r["n"], "%.3f → %.3f" % (r["rec_orig"], r["rec_edit"]), "%.0f%%" % (100 * r["drop_030"])]
             for r in _bc], widths=[1.8, 0.8, 2.0, 1.2])
    caption("Table 19.1. The segmenter on the original and on the edited image, by material and by "
            "class of the edited object (classes with at least 20 pairs).")
    _lost_bg = dict(map(tuple, _S3["lost_becomes"])).get("background") if _S3.get("lost_becomes") else None
    p("Paper is by far the most damaging material, and animals are hit, vehicles are not: a cow, "
      "a dog or a horse re-rendered in an unnatural material loses its recall in about four cases "
      "out of ten, a bus, a train or a motorbike almost never. When the object is lost it becomes "
      "background%s, not another class. This is the planted situation, with known members: "
      "“an animal re-rendered in paper or another unnatural material”."
      % ((" (%d of the %d)" % (_lost_bg, _S3["n_drop_030"])) if _lost_bg else ""))
    figure("fig_ea_examples_en.jpg", "Figure 19a. Four of the edits that make the segmenter lose the "
           "object entirely (the worst cases, one per class; most edits change nothing).", width=6.0)

    h("19.2  Does the pipeline find it, and does it name it?", 2)
    p("The pipeline was run on val-EA in the inst configuration, with the same hypotheses and "
      "the same sampling as on val. The %d victims are the edited objects whose recall falls by at "
      "least 0.30; on the original image %d of them were segmented correctly and the others had "
      "these errors: %s. On val-EA they are %s."
      % (_nv, EAN["victim_types_orig"].get("correct", 0),
         ", ".join("%s %d" % (_TYN.get(k, k), v) for k, v in EAN["victim_types_orig"].items() if k != "correct"),
         ", ".join("%s %d" % (_TYN.get(k, k), v) for k, v in EAN["victim_types_ea"].items())))
    _bcl = sorted(EAN["by_class"].items(), key=lambda kv: -kv[1][1])
    _veh = ("car", "train", "motorbike", "aeroplane", "bicycle", "bus", "boat")
    _vc = [sum(EAN["by_class"].get(c, [0, 0])[i] for c in _veh) for i in (0, 1)]
    bullet("Discovery works. %d of the %d victims (%.0f%%) fall inside a certified slice of val-EA; "
           "on the original images the same objects were inside one in %d cases. For comparison, "
           "of the edited objects the edit did not hurt but that are errors anyway, %d of %d "
           "(%.0f%%) are in a certified slice. Coverage follows the classes the edit hurts — "
           "%s — and the vehicles are hardly covered (%d of %d)."
           % (_ncov, _nv, _pct(_ncov, _nv), _ninst, _sc[0], _sc[2], _pct(_sc[0], _sc[2]),
              ", ".join("%s %d/%d" % (c, a, n) for c, (a, n) in _bcl if c not in _veh and n >= 10),
              _vc[0], _vc[1]))
    _top = _PL["slices_with_victims"][0]
    bullet("The cluster branch isolates the planted population almost pure: %s holds %d errors, "
           "%d of them from edited images, %d of them victims. It was found with no hypothesis "
           "about materials at all."
           % (_top["slice_id"], _top["members"], _top["from_edited"], _top["victims"]))
    _mat_words = ("paper", "metal", "plastic", "wood", "toy", "statue", "model", "material")
    _named = [(c, q) for c, q in EAN["lang_ea"] if any(w in q.lower() for w in _mat_words)]
    _lang_v = [s for s in _PL["slices_with_victims"] if s.get("hypothesis")]
    bullet("Naming fails. The cluster slice is phrased “%s”: the tag the VLM verified on its "
           "members is “occluded”, not the material. The language branch files the victims under "
           "situations it already had — %s — and %s certified situation of val-EA mentions a "
           "material or a toy, statue or model, although “a toy, statue or model of an object” is "
           "one of the hypotheses."
           % (_top["description"],
              "; ".join("%s “%s” %d" % (s["slice_id"].split("_")[0], s["hypothesis"], s["victims"])
                        for s in sorted(_lang_v, key=lambda s: -s["victims"])[:3]),
              "no" if not _named else "only %d" % len(_named)))
    _JM = EAN.get("judge_means")
    if _JM:
        _g, _c, _o = _JM["genuine"], _JM["control"], _JM["original"]
        p("A judge confirms the reading. Gemma was shown, for each victim, the patterns of its "
          "image and asked how well they cover the failure (0–5) and whether they name its cause "
          "(0–5), against two controls: the patterns of another victim, and those of the same "
          "image before the edit. Coverage is %.2f against %.2f and %.2f (%.2f on the %d victims "
          "that have at least one pattern); the cause scores %.2f, against %.2f and %.2f. The "
          "patterns find the edited object and its error; they never say what happened to it. "
          "(The judge's single 1–10 score ignored its own anchors — full coverage with no cause "
          "got 1 — so the two components are reported instead.)"
          % (_g["coverage"], _c["coverage"], _o["coverage"], _g["coverage_wp"], _g["n_with_patterns"],
             _g["cause"], _c["cause"], _o["cause"]))
    _lostl = _PL["lost_language"]
    p("The rest of the vocabulary moves too, since almost half of val's images are different: the "
      "certified situations go from %d to %d (%d lost, %d of them “cut off at the image border”; "
      "%d new) and the cluster slices from %d to %d."
      % (EAN["cert_inst"], EAN["cert_ea"], len(_lostl), sum(1 for _c, q in _lostl if "border" in q),
         len(_PL["new_language"]), EAN["cert_inst_cl"], EAN["cert_ea_cl"]))

    try:
        from _revision_numbers import load_ea_naming as _lean
        NM = _lean()
    except Exception as _ex:
        NM = None
        print("  (per-object naming numbers unavailable, section 19.3 skipped:", _ex, ")")
    if NM:
        _NA, _NS = NM["all"], NM["sample"]
        h("19.3  Naming the cause object by object", 2)
        p("Every grounded object has, besides the group test, a description of its own: the "
          "hypotheses the VLM confirms on it (P(yes) ≥ %.1f), highest first. Raw, that description is "
          "long and mostly about appearance, so it is filtered by a class-level rule: only the "
          "hypotheses associated with the errors of that class on the ORIGINAL VOC val sample, under "
          "the rank test of §18.3, are kept — a filter learnt without a single edited image. The "
          "description names the planted cause when it contains one of the words fixed before the "
          "results were looked at (%s). To measure it on every victim, the hypotheses the filter "
          "admits were grounded on all of them and on every edited subject the segmenter keeps "
          "(examples/ground_ea_naming.py)."
          % (NM["tau"], ", ".join(NM["cause_words"])))
        _pc = lambda d, k: "%d (%.1f%%)" % (d[k], 100.0 * d[k] / d["n"])
        _un = _NS["auc"]["unedited image"]
        table(["Objects", "n", "cause in the first phrase", "in the first three", "phrases per object"],
              [["victims: edited, lost by the segmenter", _NA["victim"]["n"], _pc(_NA["victim"], "top1"),
                _pc(_NA["victim"], "top3"), "—"],
               ["edited subjects the segmenter keeps", _NA["edited, not lost"]["n"],
                _pc(_NA["edited, not lost"], "top1"), _pc(_NA["edited, not lost"], "top3"), "—"],
               ["objects of unedited images (120-instance samples)", _un["n"], _pc(_un, "top1"),
                _pc(_un, "top3"), "%.1f" % _un["phrases"]]],
              widths=[2.6, 0.6, 1.3, 1.2, 0.9])
        caption("Table 19.2. Does an object's own description name the planted cause?")
        _sv = {r: _NS[r]["victim"] for r in ("all", "auc", "slope")}
        p("The VLM names the cause in the first phrase for %.0f%% of the victims, %.1f%% of the edited "
          "subjects the segmenter still recognises and %.1f%% of the objects of unedited images. The "
          "description does not say that an object was edited — the edited objects the model keeps "
          "rarely get it — but that it has become artificial enough for the model to fail. On the %d "
          "victims inside the samples, where every hypothesis was grounded, the filter is what makes "
          "the description usable: without it the cause is first for %.0f%% of them, among %.1f phrases "
          "per object; with it, %.0f%% among %.1f. Filtered by SliceLens's slope verdict instead, the "
          "cause is named for %d%%: the slope test never admits these hypotheses, because they concern "
          "few objects and its confidence bins with fewer than ten instances are discarded."
          % (100.0 * _NA["victim"]["top1"] / _NA["victim"]["n"],
             100.0 * _NA["edited, not lost"]["top1"] / _NA["edited, not lost"]["n"], 100.0 * _un["top1"] / _un["n"],
             _sv["auc"]["n"], 100.0 * _sv["all"]["top1"] / _sv["all"]["n"], _sv["all"]["phrases"],
             100.0 * _sv["auc"]["top1"] / _sv["auc"]["n"], _sv["auc"]["phrases"],
             round(100.0 * _sv["slope"]["top1"] / _sv["slope"]["n"])))
        _bm = sorted(_NA["by_material_v3"].items(), key=lambda kv: -kv[1][1])
        _bc = sorted(_NA["by_voc_class"].items(), key=lambda kv: -kv[1][1])
        _big = {k: (a, n) for k, (a, n) in _bc if n >= 5}
        _drop = [k for k in NM["toy_not_admitted"] if k in _big and _big[k][0] == 0]
        _kept0 = [k for k, (a, n) in _big.items() if a == 0 and k not in NM["toy_not_admitted"]]
        _plr = lambda ks: ", ".join(k + "s" for k in ks[:-1]) + (" and " if len(ks) > 1 else "") + (ks[-1] + "s" if ks else "")
        p("By material: %s. By class (at least five victims): %s. Where the cause is never named the "
          "records say why: %s%s; and where the edit barely changes the look, the segmenter still "
          "loses the object but neither the VLM nor a person sees anything artificial (Figure 19b). The two "
          "unedited model trains of Figure 19b are “false alarms” only with respect to the planted "
          "cause: the description is true of them. And “a toy, statue or model” is the closest the "
          "vocabulary gets to “made of paper”: the material itself is never named."
          % (", ".join("%s %d/%d" % (k, a, n) for k, (a, n) in _bm),
             ", ".join("%s %d/%d" % (k, a, n) for k, (a, n) in _big.items()),
             ("for %s the filter drops the toy hypothesis, which on unedited VOC is not associated with "
              "errors" % _plr(_drop)) if _drop else "",
             ("%sfor %s it is admitted but the VLM does not confirm it on the edited objects"
              % ("; " if _drop else "", _plr(_kept0))) if _kept0 else ""))
        figure("fig_ea_naming_en.jpg", "Figure 19b. The description of single objects on val-EA: the "
               "original, the edited image with the lost object boxed, the phrases the VLM confirms "
               "under the class filter (green: they name the cause). Four named, two not named, two "
               "“false alarms” on an unedited image.", width=6.2)

    try:
        from _revision_numbers import load_ea_dd as _leadd
        EDD = _leadd()
    except Exception as _ex:
        EDD = None
        print("  (val-EA data-driven numbers unavailable, section 19.4 skipped:", _ex, ")")
    if EDD:
        _pcd = EDD["sum"]["per_class"]
        _new = sorted(EDD["cmp"]["new"].values(), key=lambda v: -v["size"])
        _cap = EDD["captions"]
        _an = [c for c in ("dog", "cow", "cat") if c in _pcd]
        _rq = lambda c, q: EDD["auc_by_query"].get((c, q), {})
        _mod = [(c, q) for c, q in EDD["material_phrases"] if any(w in q.lower() for w in ("model", "miniature", "toy", "replica"))]
        h("19.4  Can the data-driven generator name it?", 2)
        p("§19.3 left one question open. SliceLens's second way of writing hypotheses asks the data, "
          "and on val-EA the data contain the cause. The generator of §13.8 was run on val-EA with "
          "the same protocol, on all %d classes — the four of the June pilot too, whose hypotheses "
          "came from the original images: Qwen captions %d instances per class, drawn at random among "
          "the instances of the class, once per attribute; Gemma extracts the value of the attribute "
          "from each caption, groups the values and turns the six largest groups into phrases. The "
          "%d phrases were grounded on the same 120-instance samples as the val-EA run, and two "
          "rescores of copies of the val-EA pickle differ only in them; val-EA itself is not modified."
          % (EDD["n_classes"], next(iter(_pcd.values()))["n_sampled"], EDD["n_hyp"]))
        bullet("Attributes. As on VOC, Gemma proposed no attribute of its own for %s of the %d classes, "
               "so the captions were asked about the pilot's four: %s. None of them asks about material."
               % ("any" if EDD["seeds_fallback"] == EDD["n_classes"] else "%d" % EDD["seeds_fallback"],
                  EDD["n_classes"], ", ".join(EDD["attributes"])))
        bullet("Groups. %d phrases are systematic for the slope rule on the sample and pass the other "
               "gates: the certified language situations of the copy go from %d to %d, and %d more "
               "images receive a situational description%s. They are %s. The rank test of §18.3 finds "
               "%d of the %d phrases significant (median AUC %.2f)."
               % (EDD["n_slope_sys"], EDD["cmp"]["base"], EDD["cmp"]["dd"], EDD["cmp"]["images_gained"],
                  "; no existing situation is lost" if not EDD["cmp"]["lost"] else "",
                  "; ".join("“%s” (%d errors, %s)" % (v["hypothesis"], v["size"], _TYN.get(v["type"], v["type"]))
                            for v in _new),
                  EDD["n_auc_sys"], EDD["n_hyp"], EDD["median_auc"]))
        bullet("The planted cause. The victims inside a certified slice are %d of %d with the data-driven "
               "phrases and %d without them; the copy without them reproduces the evaluation of §19.2 "
               "exactly. None of the new situations concerns the edited objects."
               % (EDD["covered_dd"], EDD["victims"], EDD["covered_base"]))
        bullet("The material. %d of the %d phrases contain a material or replica word, and none is about "
               "the edited object: %s.%s"
               % (len(EDD["material_phrases"]), EDD["n_hyp"],
                  "; ".join("%s “%s”" % (c, q) for c, q in EDD["material_phrases"]),
                  (" The one closest to a replica, %s, is not significant (AUC %.2f, q %.2f)."
                   % ("; ".join("“%s”" % q for _c, q in _mod), _rq(*_mod[0]).get("auc", float("nan")),
                      _rq(*_mod[0]).get("q", float("nan")))) if _mod else ""))
        p("The captions behind the phrases do see it. Of the captions of instances that lie in edited "
          "images %d of %d (%.0f%%) contain a material or replica word, against %d of %d (%.0f%%) in "
          "untouched images — little difference over all classes, since wooden floors and metallic "
          "frames are common, but a clear one on the animals the edit hurts: %s. The split is coarse, "
          "since an instance of an edited image is not always the edited object. So the information "
          "reaches the captions and is lost after them: the captions answer questions about "
          "background, size, light and occlusion, Gemma extracts only the value of the attribute "
          "asked, and a material mentioned in passing never reaches a group. A generic seed attribute "
          "such as “the material or surface of the object”, the same for every class, would be the "
          "fair test of whether the generator can name the cause; it has not been run."
          % (_cap["edited"][1], _cap["edited"][0], 100.0 * _cap["edited"][1] / max(_cap["edited"][0], 1),
             _cap["untouched"][1], _cap["untouched"][0], 100.0 * _cap["untouched"][1] / max(_cap["untouched"][0], 1),
             "; ".join("%ss %d of %d against %d of %d (%s)"
                       % (c, _pcd[c]["captions"]["edited"][1], _pcd[c]["captions"]["edited"][0],
                          _pcd[c]["captions"]["untouched"][1], _pcd[c]["captions"]["untouched"][0],
                          ", ".join(_pcd[c]["caption_words"]["edited"])) for c in _an)))
        p("Reproduce: outputs/chain_dd_ea.sh \"1 2\" and outputs/chain_dd_ea_join.sh 6 (generation and "
          "sample grounding, GPU, guarded by outputs/gpu_guard.sh; then the two rescores, "
          "compare_dd_hypotheses.py, dd_auc.py and ea_planted_eval.py on CPU), "
          "slice_quality/examples/dd_ea_summary.py (material words); outputs in /data/outputs_dd_ea, "
          "/data/outputs_dd_ea_ground and outputs/slice_quality_ea_dd{,_base}/.", italic=True)

    h("19.5  What it means", 2)
    bullet("Discovery works, as in the corruption experiment of §6.1: the failure population is "
           "found, grouped and separated from the unharmed one. Naming works object by object "
           "(§19.3) and fails as a group: no certified situation names the cause, because the group "
           "step uses SliceLens's slope test, blind to situations that concern few objects (§18.3).")
    bullet("The cluster branch earns its place here. Without it the planted population would be "
           "scattered over generic situations; with it there is one group that a person can "
           "open and recognise at a glance (Figure 19a, and the val-EA gallery).")
    bullet(("SliceLens's own generator of hypotheses from the data, run on val-EA (§19.4), adds %d "
            "situations and %s victim to the certified slices, and never names the material: its "
            "captions mention it on the animals the edit hurts, but the attributes it asks about "
            "never do, so the material is lost before the phrases are written. A generator that "
            "captions the failing objects, or asks about their material, is the candidate left."
            % (EDD["cmp"]["dd"] - EDD["cmp"]["base"],
               "no" if EDD["covered_dd"] == EDD["covered_base"] else "%+d" % (EDD["covered_dd"] - EDD["covered_base"])))
           if EDD else
           "A generator of hypotheses from the failing data is the obvious candidate to name the "
           "material itself, but §13.8 found SliceLens's own generator weak when it captions the "
           "class at random.")
    bullet("The diagnosis is actionable here: adding the edited images the diagnosis picks to the "
           "training data repairs most of the failure on new edited images (§14.8).")
    p("Reproduce: slice_quality/examples/pascal_ea_compare.py, pascal_ea_report.py and "
      "pascal_ea_examples.py (LANG_FIG=en; outputs/pascal_ea/), build_voc_ea.py, retype_ea.py, "
      "examples/ground_ea_hypotheses.py with outputs/chain_ea.sh, ea_planted_eval.py, "
      "judge_ea_patterns.py; per_instance_naming_ea.py with examples/ground_ea_naming.py "
      "(outputs/chain_ea_naming.sh, guarded by outputs/gpu_guard.sh); outputs in outputs/slice_quality_ea/ (gallery "
      "gallery_val_ea_full.html, with a random control section of unchanged edits) and "
      "/data/outputs_ea.", italic=True)

# ============================================================================
#  20. The benchmark published with SliceLens (GESD, October 2026): Block 1 and the
#      whole pipeline against the experts' slices. Numbers from
#      _revision_numbers.load_gesd -- nothing typed in except the published GH-ESD figure.
# ============================================================================
try:
    from _revision_numbers import load_gesd as _load_gesd
    GSD = _load_gesd()
except Exception as _e:                 # the GESD copies not in docs/gesd yet
    print("  GESD chapter skipped:", _e)
    GSD = None
if GSD:
    _GA, _GB = GSD["pointA"], GSD["baselines"]["baselines"]
    _GF, _GC = GSD["full"], GSD["cc"]
    _NS = len(GSD["vocab"])
    _gr = _GA["groups"]

    def _l3(stage):
        """object-level naming over every expert-slice object grounded (weighted by objects)."""
        rows = [v for v in GSD[stage]["level3"].values() if v.get("n")]
        n = sum(v["n"] for v in rows)
        return (n, 100.0 * sum(v["n"] * v["top1"] for v in rows) / n, 100.0 * sum(v["n"] * v["top3"] for v in rows) / n)

    _l3n, _l3t1, _l3t3 = _l3("full")
    _p10 = lambda stage: sum(r["certified"]["p"] for r in GSD[stage]["level2"].values()) / len(GSD[stage]["level2"])
    _named1 = lambda stage: sum(1 for r in GSD[stage]["level2b"].values() if r.get("named_rank") == 1)
    _spec = [s for s, v in GSD["vocab"].items() if v["hits"].get("gemma") or v["hits"].get("data-driven")]
    _dd_only = [s for s, v in GSD["vocab"].items() if v["hits"].get("data-driven") and not v["hits"].get("gemma")]

    h("20. The benchmark published with SliceLens: our pipeline against expert slices (October 2026)", 1)
    p("With the final version of their paper (renamed GH-ESD) the authors of SliceLens published "
      "the benchmark they evaluate on, GESD. Its instance-segmentation part takes the predictions "
      "of a Mask R-CNN on COCO val2017 for %d classes and groups the errors into %d slices defined "
      "by experts. Each slice carries a name that says both what went wrong and why: “Sliced carrot "
      "under-segmented due to shape change”, “Clustered bananas merged, boundaries unclear”, “Watch "
      "missed when segmenting clock”. %d of its objects can be typed, on %d images: %d belong to an "
      "expert slice and %d are correct. The benchmark scores a method only on WHICH objects its "
      "slices contain (Precision@k); the names are never scored."
      % (GSD["n_classes"], _NS, _GA["n_typed"], _GA["images"], _GA["n"], _GA["na_n"]))
    p("Three questions follow. What can the benchmark's own measure tell (§20.1)? Does Block 1's "
      "error type agree with the error the expert's name states (§20.2)? And the question the "
      "benchmark does not ask: do the pipeline's descriptions name the experts' situations "
      "(§20.3, §20.4)?")

    h("20.1  What the benchmark's own measure can tell", 2)
    _bl = [("A", "One slice per class, every object (correct ones too)"),
           ("B", "One slice per class, errors only"),
           ("C", "Class × error kind (missed / hallucinated / both masks)"),
           ("D", "Class × error kind, most severe first")]
    table(["Grouping, without hypotheses or VLM", "Precision@10", "expert slices at 1.0"],
          [[lab, "%.3f" % _GB[k]["p_at_k"], "%d of %d" % (_GB[k]["at_1"], _NS)] for k, lab in _bl if k in _GB]
          + [["GH-ESD, the published method", "%.3f" % GSD["ghesd_p10"], "—"]],
          widths=[3.6, 1.2, 1.6])
    caption("Table 20.1. Precision@10 on GESD's segmentation slices (for each expert slice, the "
            "best predicted slice; hits among its first 10 objects divided by min(10, slice size); "
            "mean over the %d slices). The expert names are used only to score." % _NS)
    p("Sorting the errors by class and by which masks exist — no hypothesis, no VLM — scores "
      "%.3f, the published method %.3f. On this part of the benchmark the measure cannot tell a "
      "slice-discovery method from a sort by class and error kind: every error of the file "
      "belongs to an expert slice, the correct objects carry a prediction copied from their "
      "ground truth, and only a few slices need more than class and kind (sliced against "
      "clustered fruit, a bed with a pet against a bed with a person, an open suitcase). What "
      "the benchmark can still be used for, with measurements of our own, is the error type and "
      "the name." % (_GB["C"]["p_at_k"], GSD["ghesd_p10"]))

    h("20.2  Block 1's error type against the experts' words", 2)
    p("Block 1, in the configuration of this thesis (containment claims at 0.5), typed every "
      "prediction of the file with every COCO object of the image as ground-truth context; where "
      "an expert's mask matches a COCO object (IoU ≥ 0.5) the expert's mask replaces it. GESD "
      "does not store the predicted class, so a prediction gets its object's class. The type each "
      "expert name implies was fixed before the results, from its words (Table 20.2). Beside "
      "Block 1, a naive typing that looks only at which masks exist: a prediction alone is a "
      "hallucination, a ground truth alone a miss, both a boundary or extent error.")
    _order = ["mis-segmented as (hallucination)", "boundary unclear", "missed", "over-segmented / expanded",
              "under-segmented", "merged"]
    table(["Word in the expert name", "slices", "objects", "expected type", "Block 1 (range over slices)", "naive"],
          [[g, _gr[g]["slices"], _gr[g]["n"], " / ".join(_gr[g]["expected"]),
            "%.0f%% (%.0f–%.0f%%)" % (100.0 * _gr[g]["block1"] / _gr[g]["n"], 100 * _gr[g]["range"][0],
                                      100 * _gr[g]["range"][1]),
            "%.0f%%" % (100.0 * _gr[g]["naive"] / _gr[g]["n"])] for g in _order if g in _gr]
          + [["all", _NS, _GA["n"], "", "%.1f%%" % (100.0 * _GA["block1"] / _GA["n"]),
              "%.1f%%" % (100.0 * _GA["naive"] / _GA["n"])]],
          widths=[1.7, 0.6, 0.7, 1.4, 1.4, 0.7])
    caption("Table 20.2. How often Block 1's type is the type the expert name states. The naive "
            "typing counts a hit when either of its two types (boundary, extent) is expected.")
    _hal, _bnd, _mis, _mrg = (_gr[g] for g in ("mis-segmented as (hallucination)", "boundary unclear", "missed", "merged"))
    bullet("Agreement is high where the word describes the masks: hallucinations %.0f–%.0f%%, the "
           "tie's unclear boundary %.0f%%. Overall the naive typing scores almost as well (%.1f%% "
           "against %.1f%%), because the file's layout already tells hallucinations from misses; but "
           "only Block 1 recognises merges (%d of %d merged objects, the naive typing none)."
           % (100 * _hal["range"][0], 100 * _hal["range"][1], 100.0 * _bnd["block1"] / _bnd["n"],
              100.0 * _GA["naive"] / _GA["n"], 100.0 * _GA["block1"] / _GA["n"], _mrg["block1"], _mrg["n"]))
    _lp = _GA["laptop_partial"]
    bullet("Most disagreements are vocabulary, and each was measured. For the experts "
           "“under-segmented” includes the pieces never predicted at all (%d of the %d sliced-fruit "
           "objects have no prediction: misses, for Block 1). “Merged” includes a mask that spills "
           "only partly onto a neighbour — in %d of the %d laptop-keyboard objects Block 1 does not "
           "call a merge, the mask covers between 10%% and 50%% of the laptop (median %.0f%%) — or onto "
           "objects COCO does not annotate, like the toys beside a teddy bear; Block 1 calls a merge "
           "only a mask that covers at least half of two annotated objects."
           % (_GA["under_no_pred"], _GA["under_n"], _lp[0], _lp[1], 100 * (_lp[2] or 0)))
    bullet("One gap is in the data. %d objects the experts call missed have a predicted mask, %d of "
           "them overlapping the expert's mask with IoU ≥ 0.5; they were most likely predicted with "
           "another class, which GESD does not record. Block 1, which has to assume the object's "
           "class, calls them boundary errors (missed slices: %.0f–%.0f%% agreement)."
           % (_GA["missed_with_pred"], _GA["missed_with_pred_iou50"], 100 * _mis["range"][0], 100 * _mis["range"][1]))
    bullet("Two findings point the other way: %d of the %d objects of “Paper mis-segmented as "
           "book” are books in COCO's own annotation, which Block 1 therefore calls correct; and of "
           "the %d objects the experts call correct, Block 1 calls %d correct (%.1f%%; %d are merges "
           "with another COCO object)."
           % (_GA["paper_books"], _GA["paper_n"], _GA["na_n"], _GA["na_correct"],
              100.0 * _GA["na_correct"] / _GA["na_n"], _GA["na_merge"]))
    p("Read with the experts' vocabulary after the fact — a missing piece as under-segmentation, "
      "any spill as a merge, a boundary error as an expanded boundary — Block 1 agrees on %.1f%% "
      "of the objects. That reading was not fixed in advance and is reported only to size the "
      "vocabulary effect." % (100.0 * _GA["posthoc"] / _GA["n"]))

    h("20.3  The whole pipeline on GESD", 2)
    _nh = GSD["n_hyp"]
    p("The pipeline was then run on GESD with the protocol of this thesis, on the same typed "
      "objects (Mask R-CNN predicts instances, so one set of objects serves both branches). "
      "Every class gets up to 36 hypotheses: the 7 canonical ones; up to 15 that Gemma generates "
      "from the class name alone, as for VOC (for %d classes Gemma's answer was malformed JSON, "
      "repaired with one one-character syntax edit each, no phrase touched); 6 data-driven ones, "
      "from SliceLens's own generator (Qwen captions 20 objects of the class, Gemma condenses the "
      "captions); and the 8 extra generic ones of §13 — %d canonical, %d Gemma, %d data-driven and "
      "%d extra hypotheses over the %d classes. The expert names enter no step of the "
      "run. Each class was grounded on a 120-object sample; the certified hypotheses and those "
      "significant under the rank test of §18.3 were then grounded on the whole class, and the run "
      "was rescored on them. Separately, and only to measure, the class-specific phrases that name "
      "an expert situation and were not already among them were also grounded on their whole "
      "class; they never enter a slice."
      % (len(GSD["repairs"]), _nh["canonical"], _nh["gemma"], _nh["data-driven"], _nh["extra"], GSD["n_classes"]))
    p("On the sample SliceLens's slope rule calls %d of the %d hypotheses systematic and the rank "
      "test %d (both %d); on whole classes, of the %d re-grounded, %d and %d. The final run "
      "certifies %d situations and %d cluster slices. Whether a phrase of ours names an expert "
      "situation is decided by rules written before any hypothesis was generated: the phrase "
      "belongs to the slice's class and contains one of a few keywords chosen for that slice "
      "(“slice”, “cut”, “diced”… for sliced fruit; “folded”, “closed” for the umbrella; “watch”, "
      "“wrist” for the clock). Matches through the generic hypotheses are counted apart."
      % (_GC["slope"], _GC["n_hyp"], _GC["auc"], _GC["both"], _GF["n_hyp"], _GF["slope"], _GF["auc"],
         GSD["certified_full"].get("language", 0), GSD["certified_full"].get("cluster", 0)))

    h("20.4  Do our descriptions name the experts' situations?", 2)
    _src_tag = lambda v: ("Gemma" if v["hits"].get("gemma") else "data-driven" if v["hits"].get("data-driven")
                          else "generic only" if v["hits"] else "—")
    _rows = []
    for s, v in sorted(GSD["vocab"].items(), key=lambda kv: -(GSD["full"]["level3"].get(kv[0], {}).get("n") or 0)):
        b = GSD["full"]["level2b"].get(s)
        l3 = GSD["full"]["level3"].get(s, {})
        best = ("“%s” %.2f%s" % (b["top"][0], b["top"][1], " ✓" if b.get("named_rank") == 1 else "")) if b else "—"
        named = ("%.0f%% / %.0f%%" % (100 * l3["top3"], 100 * l3["na_top3"])) if l3.get("n") and l3.get("na_n") else "—"
        _rows.append([s, _src_tag(v), best, named])
    table(["Expert slice", "its situation in our hypotheses", "phrase that best isolates its objects (AUC; ✓ = names it)",
           "named among an object's first 3: slice / correct"], _rows, widths=[2.0, 1.0, 2.4, 1.2])
    caption("Table 20.3. The %d expert slices, whole classes. Second column: the source of a "
            "class-specific phrase that names the situation, by the rules fixed in advance. Third: "
            "among the phrases grounded on the whole class, the one whose confidence best separates "
            "the slice's objects from the rest of the class. Fourth: how often an object's own "
            "description (§19.3: the hypotheses that pass the rank test for its class, ranked by the "
            "VLM's confidence) names the situation among its first three, on the slice's objects and "
            "on the correct objects of the same class." % _NS)
    bullet("Half of the situations are not in the vocabulary. %d of the %d have a class-specific "
           "phrase among our hypotheses (%d through Gemma; the data-driven generator adds %d that the "
           "others lack: %s). A few more are matched only by a generic hypothesis, genuinely for the "
           "clustered bananas (“several overlapping objects of the same type”), the tie (“an object "
           "that blends into its background”) and the occluded airplane, and by accident of the "
           "substring rule for the sliced orange (“cut” in “cut off”) and the bed with a pet (“cat” "
           "in “category”). A vase without flowers, a bed with a person, an Apple logo, an open book "
           "and an open suitcase have no phrase at all."
           % (len(_spec), _NS, sum(1 for s in _spec if GSD["vocab"][s]["hits"].get("gemma")), len(_dd_only),
              "; ".join(_dd_only) or "none"))
    _b2 = {s: r for s, r in GSD["full"]["level2b"].items() if r.get("named_rank") == 1}
    bullet("Where the phrase exists, the VLM finds the experts' objects. The phrase that best "
           "separates an expert slice's objects from the rest of its class is the one that names "
           "the situation for %d of the %d slices when all hypotheses compete on the sample, and "
           "for %d among the phrases grounded on whole classes: %s."
           % (_named1("cc"), _NS, _named1("full"),
              "; ".join("“%s” %.2f (%s)" % (r["top"][0], r["top"][1], " ".join(s.split()[:2]).lower())
                     for s, r in sorted(_b2.items(), key=lambda kv: -kv[1]["top"][1]))))
    _good = sorted(((s, v) for s, v in GSD["full"]["level3"].items()
                    if v.get("n") and v.get("na_n") and v["top3"] >= 0.3 and v["top3"] > 2 * (v["na_top3"] or 0)),
                   key=lambda kv: -kv[1]["top3"])
    bullet("Object by object the result is weaker: an expert-slice object's own description names "
           "the situation first for %.1f%% of the %d objects grounded, among its first three for "
           "%.1f%%. It works where the naming phrase passes the class-level filter — %s — and fails "
           "where the filter drops it. A description keeps only the hypotheses that, over the whole "
           "class, tell its errors from its correct objects; “sliced banana pieces” or “fruit "
           "basket” isolate one expert slice almost perfectly but do not separate the class's errors "
           "from its correct objects (a fruit basket also holds many well-segmented apples)."
           % (_l3t1, _l3n, _l3t3,
              "; ".join("“%s” %.0f%% (correct objects %.0f%%)" % (s, 100 * v["top3"], 100 * v["na_top3"])
                        for s, v in _good)))
    bullet("Measured as the benchmark measures, the best of our certified slices reaches "
           "Precision@10 %.3f on the sample and %.3f on whole classes, against %.3f for the "
           "published method and %.3f for the sort by class and error kind. The winners are almost "
           "always a cluster slice or a generic phrase: the right name is found as a candidate "
           "phrase, not as a certified slice."
           % (_p10("cc"), _p10("full"), GSD["ghesd_p10"], _GB["C"]["p_at_k"]))

    _GX = GSD.get("cmp")
    if _GX:
        _gc, _gt, _gw = _GX["cols"], _GX["tests"], _GX["why"]
        _COLN = [("SliceLens, verdict only", "SliceLens (slope rule at both steps)"),
                 ("+ rank test, verdict only", "+ rank test (at both steps)"),
                 ("SliceLens, thesis gates", "SliceLens + the gates of the language branch"),
                 ("+ rank test, thesis gates", "+ rank test + the gates"),
                 ("pipeline", "the pipeline (rank test, gates, cluster branch)")]
        h("20.5  SliceLens against the pipeline on GESD, with the benchmark's measure (23 October)", 2)
        p("The comparison of §18.6 on the benchmark published with SliceLens: the same objects, VLM and "
          "hypotheses, and only the verdict changes. SliceLens sends to the whole class the hypotheses its slope "
          "rule calls systematic on the 120-object sample (%d) and certifies those still systematic there (%d); "
          "the rank test does the same with its own verdict (%d and %d). %d of SliceLens's hypotheses had been "
          "grounded on the sample only (the run of §20.3 followed the protocol of this thesis), and were grounded "
          "on their whole class for this comparison (outputs/chain_gesd_cmp.sh, GPU 2 and 5 under the guard). "
          "Each verdict is read twice: alone, a slice being the class's errors the VLM says yes to, ranked by "
          "its confidence; and after the gates of the language branch of this thesis, on "
          "a copy of the run rescored with that verdict (slice_quality/examples/gesd_cmp.py). The paper ranks "
          "correct and erroneous units alike and does not say whether the slice it outputs drops the correct "
          "ones, so the measure is also taken with them kept (last column of Table 20.4)."
          % (_GX["sample"]["slope"], _GX["whole"]["slope"], _GX["sample"]["auc"], _GX["whole"]["auc"],
             len({(v["cls"], v["q"]) for v in json.load(open(os.path.join(DOCS, "gesd", "pointB_cc.json"),
                                                              encoding="utf-8"))["verdicts"] if v["slope_sys"]}
                 - {(v["cls"], v["q"]) for v in json.load(open(os.path.join(DOCS, "gesd", "pointB_full.json"),
                                                              encoding="utf-8"))["verdicts"]})))
        _GBc = GSD["baselines"]["baselines"]
        _ga = _GX.get("cols_all") or {}
        table(["Slices from", "certified", "classes with a certified phrase", "Precision@10", "expert slices at 1.0",
               "expert situations named", "Precision@10, correct objects kept"],
              [[lab, str(_gc[k]["slices"]), str(_gw[k]["classes_with_language_slice"]) if k in _gw else "—",
                "%.3f" % _gc[k]["p10"], "%d of %d" % (_gc[k]["at_1"], _GX["n_expert"]), str(_gc[k]["named"]),
                "%.3f" % _ga[k]["p10"] if k in _ga else "—"]
               for k, lab in _COLN]
              + [["One slice per class, errors only (no hypothesis, no VLM)", str(GSD["n_classes"]), "—",
                  "%.3f" % _GBc["B"]["p_at_k"], "%d of %d" % (_GBc["B"]["at_1"], _GX["n_expert"]), "—", "—"],
                 ["Class × error kind (no hypothesis, no VLM)", "—", "—", "%.3f" % _GBc["C"]["p_at_k"],
                  "%d of %d" % (_GBc["C"]["at_1"], _GX["n_expert"]), "—", "—"],
                 ["GH-ESD, as published", "—", "—", "%.3f" % GSD["ghesd_p10"], "—", "—", "—"]],
              widths=[2.0, 0.65, 0.8, 0.75, 0.75, 0.75, 0.85])
        caption("Table 20.4. SliceLens and the pipeline on GESD's segmentation slices, with the benchmark's "
                "Precision@10 (the best slice of a method for each of the %d expert slices). Expert situations "
                "named: a certified phrase of the slice's class contains one of the keywords fixed before the run, "
                "generic phrases excluded. Last column: the same measure when each phrase slice keeps every object "
                "of its class the VLM says yes to, correct ones included, so that a correct object among the first "
                "ten is a miss; cluster slices are groups of errors and do not change." % _GX["n_expert"])
        _t1, _t2 = _gt["+ rank test, verdict only vs SliceLens, verdict only"], _gt["pipeline vs SliceLens, verdict only"]
        _ob = _GX["objects"]
        p("On the benchmark's measure SliceLens scores higher, and the difference is beyond chance: the rank test "
          "alone scores %+.3f against it (95%% interval %+.3f to %+.3f, worse on %d expert slices, better on %d; "
          "Wilcoxon p %.2g), the pipeline %+.3f (%+.3f to %+.3f; p %.2g). Where the meaning of the phrases is "
          "tested the two verdicts tie: certified phrases name %d expert situations under the slope rule and %d "
          "under the rank test (McNemar p %.2g); object by object, the experts' situation is among the first "
          "three phrases of %.1f%% of the experts' objects with SliceLens and %.1f%% with the rank test, and is "
          "given wrongly to %.1f%% and %.1f%% of the correct objects."
          % (_t1["diff"], _t1["ci"][0], _t1["ci"][1], _t1["worse"], _t1["better"], _t1["p_wilcoxon"],
             _t2["diff"], _t2["ci"][0], _t2["ci"][1], _t2["p_wilcoxon"],
             _gc["SliceLens, verdict only"]["named"], _gc["+ rank test, verdict only"]["named"],
             _t1["p_mcnemar_named"], 100 * _ob["SliceLens"]["rate"], 100 * _ob["+ rank test"]["rate"],
             100 * _ob["SliceLens"]["false_rate"], 100 * _ob["+ rank test"]["false_rate"]))
        _cut = _gw["slicelens_cut_to_rank_count"]
        p("What the measure rewards explains it. It takes, for every expert slice, the best of a method's slices "
          "and charges nothing for slices that match none, and every error of the file belongs to an expert "
          "slice: a class with any certified slice of errors scores, whatever its phrase says. SliceLens certifies "
          "a phrase in %d of the %d classes, the rank test in %d, the pipeline in %d; %d expert slices fall in a "
          "class where the rank test certifies nothing. Cut at random to the rank test's %d slices, SliceLens "
          "scores %.3f (%.3f to %.3f over 500 draws), as the rank test does. Ordering a slice's objects by the "
          "phrase barely matters: with the objects of every SliceLens slice shuffled the measure is %.3f instead "
          "of %.3f. A rule that certifies more wins it — the slope rule, whose false alarms §18.6 measured — and "
          "so does a sort by class and error kind without any phrase (%.3f). On GESD neither verdict is shown to "
          "be better; what the benchmark shows is that its measure cannot tell."
          % (_gw["SliceLens, verdict only"]["classes_with_language_slice"], GSD["n_classes"],
             _gw["+ rank test, verdict only"]["classes_with_language_slice"],
             _gw["pipeline"]["classes_with_language_slice"],
             _gw["+ rank test, verdict only"]["expert_slices_without_language_slice_in_class"],
             _cut["n"], _cut["p10_mean"], _cut["p10_ci"][0], _cut["p10_ci"][1],
             _gw["SliceLens, verdict only, members shuffled"]["p10_mean"], _gc["SliceLens, verdict only"]["p10"],
             _GBc["C"]["p_at_k"]))
        if _ga:
            _ta = _GX["tests_all"]
            _a1, _a2, _a3 = (_ta["+ rank test, verdict only vs SliceLens, verdict only"],
                             _ta["pipeline vs SliceLens, verdict only"],
                             _ta["pipeline, errors only vs SliceLens, verdict only"])
            _ac = _ga["slicelens_cut_to_rank_count"]
            p("Keeping the correct objects. The paper evaluates correct and erroneous units alike and ranks those "
              "the VLM matches by its score; it does not say whether the slice it outputs drops the correct ones. "
              "Read that way, a correct object among a slice's first ten is a miss, and SliceLens scores %.3f "
              "instead of %.3f: %.0f%% of the first ten objects of its slices are correct, against %.0f%% for the "
              "rank test, whose verdict is precisely that a phrase ranks errors above correct objects. The gap "
              "narrows but keeps its sign: the rank test alone %+.3f (%+.3f to %+.3f; Wilcoxon p %.2g), the "
              "pipeline %+.3f (%+.3f to %+.3f; p %.2g). The pipeline's own slices hold errors only; compared as "
              "it outputs them with SliceLens read with the correct objects, the difference, %+.3f (%+.3f to "
              "%+.3f; p %.2g), is within chance. Cut at random to the rank test's %d slices, SliceLens scores %.3f "
              "(%.3f to %.3f over 500 draws) against the rank test's %.3f: at the same number of slices the two "
              "verdicts do not differ, and what remains of SliceLens's lead is again the number of its slices. The "
              "published GH-ESD figure, %.3f, is closer to the reading without correct objects (%.3f), which this "
              "chapter keeps as the main one: it is the reading most favourable to SliceLens."
              % (_ga["SliceLens, verdict only"]["p10"], _gc["SliceLens, verdict only"]["p10"],
                 100 * _ga["SliceLens, verdict only"]["correct_in_top10"],
                 100 * _ga["+ rank test, verdict only"]["correct_in_top10"],
                 _a1["diff"], _a1["ci"][0], _a1["ci"][1], _a1["p_wilcoxon"],
                 _a2["diff"], _a2["ci"][0], _a2["ci"][1], _a2["p_wilcoxon"],
                 _a3["diff"], _a3["ci"][0], _a3["ci"][1], _a3["p_wilcoxon"],
                 _ac["n"], _ac["p10_mean"], _ac["p10_ci"][0], _ac["p10_ci"][1],
                 _ga["+ rank test, verdict only"]["p10"], GSD["ghesd_p10"], _gc["SliceLens, verdict only"]["p10"]))
        _mi = (_GX.get("missing") or {}).get("classes") or {}
        if _mi:
            _grp = lambda r: [c for c, v in _mi.items() if v["reason"] == r]
            _cn = lambda cs: ", ".join(c.replace("_", " ") for c in cs[:-1]) + (" and " if len(cs) > 1 else "") + \
                cs[-1].replace("_", " ")
            _ne = lambda cs: sum(len(_mi[c]["expert_slices"]) for c in cs)
            _ms, _mg = _grp("sample"), _grp("gates")
            _zs = [g["sev_z"] for c in _mg for g in _mi[c]["gated"]]
            _pv = lambda cs, k: (lambda xs: sum(xs) / len(xs))([v for c in cs for v in _mi[c][k].values()])
            _nz = sum(v > 0 for c in _mi for v in _mi[c]["p10_pipeline"].values())
            _mq = sorted(_mi[c]["best_sample"]["q_bh"] for c in _ms)
            p("Where the pipeline certifies no phrase. In %d of the %d classes the pipeline certifies no phrase, and "
              "their %d expert slices can score only through the cluster branch. There are two reasons. In %s (%d "
              "expert slices) no hypothesis passes the rank test on the 120-object sample: the best of the %d tried "
              "per class reaches q %.2f to %.2f after the correction for the number tried, against 0.05, so nothing "
              "goes on to the whole class, while the slope rule sends %s of them on. In %s (%d expert slices) the "
              "rank test certifies phrases on the whole class and the gates of the language branch remove every one, "
              "all for the same reason: their errors are milder than the typical error of their kind (mean severity "
              "z from %+.2f to %+.2f, where the gate asks %+.2f, §4.6), while their confidence and size pass. For "
              "bed and teddy bear the experts describe merges, a bed mask that also covers the person and toys "
              "merged with the bear: a mask that spills onto a neighbour still covers its own object, so by IoU the "
              "error looks slight. The gate was set on VOC to drop groups of slight boundary errors; here it drops "
              "situations the experts named: with the rank test alone, without the gates, those %d expert slices "
              "score %.2f on average, in the pipeline %.2f, through its clusters only. The cluster branch gives %d of the %d expert slices a "
              "score above zero."
              % (len(_mi), GSD["n_classes"], _ne(list(_mi)), _cn(_ms), _ne(_ms), _mi[_ms[0]]["n_sample"],
                 _mq[0], _mq[-1], _cn([str(_mi[c]["slope_sys_sample"]) for c in _ms]), _cn(_mg), _ne(_mg),
                 min(_zs), max(_zs), _GX["missing"]["gate"]["lang_sev_z_min"], _ne(_mg),
                 _pv(_mg, "p10_rank_verdict"), _pv(_mg, "p10_pipeline"), _nz, _ne(list(_mi))))
        p("Reproduce: outputs/chain_gesd_cmp.sh (GPU 2 and 5, guarded): gesd_cmp.py MODE=prep, "
          "examples/ground_ea_hypotheses.py with FULL_HYPS=/data/full_slopeonly.json, gesd_cmp.py MODE=slope, "
          "rescore_with_language.py on outputs/slice_quality_gesd_cmp_{auc,slope}, gesd_cmp.py MODE=report "
          "(docs/gesd/gesd_cmp.{txt,json}; the variant with the correct objects and the classes without a phrase "
          "added on 26 October, CPU, the rest of the report unchanged).", italic=True)
    h("20.6  What it means", 2)
    p("GESD is the first check of the pipeline against people who did not build it. Its two "
      "halves hold up where they can be checked: where an expert's word describes the masks, "
      "Block 1 says the same, and where our vocabulary holds the expert's situation, the VLM "
      "picks out the expert's objects. The limits are on either side of the VLM. Before it, half "
      "of the experts' situations are not in the vocabulary, and the data-driven generator, which "
      "SliceLens proposes to fill exactly that gap, mostly describes light and background. After "
      "it, both the certification and the per-object filter judge a phrase on the whole class, "
      "and so discard situations that concern only part of its errors. The benchmark's own "
      "measure sees none of this: a sort by class and error kind matches the published method, "
      "the names are never scored, and a rule that certifies more slices scores higher, by less when the correct "
      "objects the VLM matches stay in the slices (§20.5). The comparison "
      "on names, with rules fixed in advance, is "
      "what this chapter adds; its numbers are lower bounds, because the rules are strict — "
      "“black silk tie on black shirt” does not count as colour blending.")

# ============================================================================
#  14. Using the diagnosis to improve the segmenter (SEGIMPROVER, Sept-Oct 2026).
#      Numbers from _revision_numbers.load_segimprover, figures from
#      SEGIMPROVER/make_figures.py (LANG_FIG=en) -- nothing typed in.
# ============================================================================
_SGF = lambda n: os.path.join(os.environ.get("SEGIMPROVER", r"<home>\Desktop\SEGIMPROVER"),
                              "figs", n)
_f4 = lambda x: "%.4f" % x
_sg = lambda x: "%+.4f" % x
_pm = lambda m_s: "%.4f ± %.4f" % m_s
_dg, _a1, _aA, _a2 = SG["diag"], SG["arm1"], SG["armA"], SG["arm2"]
_TYEN = {"boundary": "boundary", "extent": "extent", "fn": "missed (FN)", "fp": "hallucinated (FP)",
         "confusion": "wrong class", "merge": "merge", "split": "split"}

h("14. Using the diagnosis to improve the segmenter (September–October 2026)", 1)
p("Everything so far observes the model. The pipeline produces two kinds of knowledge that "
  "ordinary training does not have: in WHICH situations the model fails (the certified "
  "situations of the language branch) and WHAT kind of error it makes, pixel by pixel "
  "(Block 1). The question of this chapter is falsifiable: if the model is trained with "
  "that knowledge, does it improve — and does it improve where the diagnosis pointed? The "
  "second half matters as much as the first, because a generic gain could come from any "
  "change to training and would say nothing about the diagnosis.")
figure(_SGF("fig_flow_en.png"), "Figure 14a. The diagnosis of the starting checkpoint feeds "
       "three arms; each is compared with an identical control and measured with the same four "
       "readings.", width=6.3)

h("14.1  Three ways to bring the diagnosis into training, and the protocol", 2)
table(["Arm", "Information used", "Where it enters", "What changes against the control"],
      [["1 — resample", "a certified situation (geometric proxy on train)", "sampler",
        "images of the group are drawn 4× more often"],
       ["A — weight the loss", "typed-error maps per pixel", "loss",
        "each wrong pixel weighs 1 + α(type)"],
       ["2 — manufacture the situation", "a certified situation (object cut off at the border)",
        "crop", "half of the crops cut an object with the crop edge"]],
      widths=[1.4, 2.0, 1.0, 2.1])
caption("Table 14.1. The three arms.")
p("Model and data: the same LR-ASPP MobileNetV3 with its published weights, fine-tuned on "
  "VOC 2012 train (1 464 images) and measured on val (1 449), which is never read during "
  "training. Every arm has a control arc and a repair arc that start from the same "
  "checkpoint with the same seed, the same recipe (18 epochs, batch 8, 480 crops after a "
  "random 0.5–2× rescale, horizontal flip, AdamW lr 1e-4 with cosine decay) and the same "
  "number of steps; the only difference is the intervention. The diagnosis is computed once, "
  "on the starting checkpoint, and then frozen: a model chasing its own updated errors also "
  "chases its own noise, and a fixed reference is what lets a gain be attributed to it.")
p("Differences are read PAIRED by seed (control and repair of a seed see the same random "
  "sequence) and reported with the t statistic of the paired differences; a difference "
  "counts as distinguishable from noise when |t| exceeds the two-sided 95%% Student "
  "threshold with n − 1 degrees of freedom (4.30 with 3 seeds, 2.78 with 5). Four readings: "
  "global mIoU on val (the constraint: a repair must not cost elsewhere), mIoU on the val "
  "images that contain the situation (the effect sought), mIoU on the others (if it rises as "
  "much, the gain is not localised), and a re-diagnosis — Block 1 re-run on the trained "
  "model's masks, an instrument independent of the repair. Seeds per arm: %d for arm 1 "
  "(a pilot), %d for arm A, %d for arm 2." % (1, len(_aA["diff"]["d"]), len(_a2["total"]["d"])))

h("14.2  The diagnosis on train", 2)
p("Block 1 was run on the %d train images with the starting checkpoint's predictions, at "
  "original resolution (so the maps align with the training crops), with the inst recipe. It "
  "yields %d typed error instances and, for every image, a map in which each pixel carries "
  "the code of its error type; %d images (%d%%) have no typed error, the median is %g "
  "instances per image." % (_dg["n_images"], _dg["n_inst"], _dg["no_err"],
                            round(100 * _dg["no_err"] / _dg["n_images"]), _dg["median"]))
figure(_SGF("fig_diag_maps_en.png"), "Figure 14b. Three error maps on train, chosen by rule "
       "(the dominant type of the map, among images with at least three errors): two people "
       "fused into one region (merge); a silhouette that is right but spills onto the bed "
       "(boundary and extent); an aeroplane mostly missed, with a hallucinated part elsewhere.")
figure(_SGF("fig_error_distribution_en.png"), "Figure 14c. The same errors counted two ways: "
       "by instances, hallucinations, merges and boundary errors dominate; by pixels, merges "
       "and extent.", width=6.0)
_px = _dg["px"]
p("The two rankings disagree, and that is a design constraint for arm A: a merge covers on "
  "average %s pixels when it occurs, a boundary error %s, but merges occur in %d images and "
  "boundary errors in %d. A weight proportional to area would give merges %.0f%% of the "
  "signal, so the weights must be normalised per type."
  % ("{:,.0f}".format(_px["merge"] / _dg["by_type"]["merge"]),
     "{:,.0f}".format(_px["boundary"] / _dg["by_type"]["boundary"]),
     _dg["img_with"]["merge"], _dg["img_with"]["boundary"], 100 * _px["merge"] / sum(_px.values())))

h("14.3  Arm 1 — resample the images of a situation", 2)
_gr = SG["groups"]
_SIT = {"border": "an object cut off at the image border",
        "occlusion": "an object occluded by another object",
        "same_class_overlap": "several overlapping objects of the same type",
        "thin": "an object with thin or elongated parts"}
_PROXY = {"border": "an object's box touches the border, or the 520 evaluation window cuts it",
          "occlusion": "an object in ≥ 2 parts touched by an object of another class",
          "same_class_overlap": "≥ 2 objects of one class with overlapping boxes",
          "thin": "a radius-2 morphological opening erases ≥ 15% of an object"}
_idx_inst = json.load(open(_out_path("slice_quality_inst",
                                        "slices_index.json"), encoding="utf-8"))["slices"]
_rows = []
for _s, _hyp in _SIT.items():
    _sl = set()
    for _x in _idx_inst:
        if _x.get("in_descriptions") and _x.get("hypothesis") == _hyp:
            _sl |= set(_x.get("images") or [])
    _g = _gr[_s]
    _rows.append([_hyp, _PROXY[_s], "%d (%d%%)" % (_g["train"], round(100 * _g["train"] / _g["n_train"])),
                  "%d/%d (%d%%)" % (len(_sl & _g["val"]), len(_sl), round(100 * len(_sl & _g["val"]) / max(1, len(_sl))))])
p("The simplest idea: the model sees more often the images in which the pipeline says it "
  "fails. On train the VLM anchors do not exist, and producing them would cost a GPU and add "
  "the grounding noise, so the group is defined by a geometric proxy computed from the masks, "
  "and its agreement with the certified slice is measured on val, where both exist.")
table(["Situation", "Geometric proxy", "Train images in the group",
       "Val slice recovered by the proxy"], _rows, widths=[1.7, 2.3, 1.2, 1.3])
caption("Table 14.2. The four proxies; the last column is the recall of the certified slice.")
if _a1:
    p("The border situation is excluded at once — its proxy takes %d%% of train, and weighting "
      "almost everything four times is weighting nothing — and occlusion and thin parts agree "
      "too little with their slices. That leaves several overlapping objects of the same type: "
      "selective, the best agreement among the selective proxies and the highest plausibility "
      "of §12. On one seed, the group gains %s while the rest loses %s and the global mIoU "
      "%s (%s → %s): the gap between group and rest widens, but it is a trade, not a cure — the "
      "classic signature of resampling, which moves capacity instead of adding it."
      % (round(100 * _gr["border"]["train"] / _gr["border"]["n_train"]),
         _sg(_a1["group"]["repair"] - _a1["group"]["control"]),
         _sg(_a1["rest"]["repair"] - _a1["rest"]["control"]),
         _sg(_a1["all"]["repair"] - _a1["all"]["control"]),
         _f4(_a1["all"]["control"]), _f4(_a1["all"]["repair"])))
if SG.get("rediag_arm1"):
    p("The re-diagnosis goes the wrong way: typed error instances on val %d → %d, hallucinations "
      "%d → %d." % (SG["rediag_arm1"]["total"] + SG["rediag_arm1"]["fp"]))

h("14.4  Arm A — weight the loss on the pixels the diagnosis marks", 2)
p("Here the finest information enters: every pixel the diagnosis marks as wrong weighs "
  "1 + α(type) in the cross-entropy, with the α chosen so that EVERY TYPE receives the same "
  "share of the added signal — so that merges, almost half of the marked pixels, do not "
  "drown the boundaries — capped at 5, which only the rare split reaches. The loss is divided "
  "by the sum of the weights so the two arcs share a learning rate.")
table(["Error type", "share of marked pixels", "share of the added signal", "α"],
      [[_TYEN[n], "%.1f%%" % (100 * sh), "%.1f%%" % (100 * tg), "%.2f" % al]
       for n, sh, tg, al in _aA["weights"]], widths=[1.8, 1.7, 1.8, 0.8])
caption("Table 14.3. The weights of arm A, computed from the diagnosis on train.")
table(["Seed", "control", "repair", "difference"],
      [[s, _f4(c), _f4(r), _sg(r - c)] for s, c, r in _aA["per_seed"]]
      + [["mean ± sd", _pm(_aA["control"]), _pm(_aA["repair"]),
          "%s (t = %.1f)" % (_sg(_aA["diff"]["mean"]), _aA["diff"]["t"])]],
      widths=[1.0, 1.7, 1.7, 1.9])
caption("Table 14.4. Arm A, global mIoU per seed.")
p("The difference (%s global, %s on the group) is smaller than the seed-to-seed dispersion "
  "of either arc (±%.4f and ±%.4f), and its t (%.1f) %s the threshold of %.2f: %s This arm "
  "also gave the scale of the noise below which no difference in this chapter should be "
  "taken seriously."
  % (_sg(_aA["diff"]["mean"]), _sg(_aA["diff_group"]["mean"]), _aA["control"][1], _aA["repair"][1],
     _aA["diff"]["t"], "stays below" if not _aA["diff"]["signif"] else "exceeds", _aA["diff"]["crit"],
     "no effect can be shown, and if there is one it is negative." if not _aA["diff"]["signif"]
     else "a degradation."))
if SG.get("rediag_armA"):
    p("The re-diagnosis worsens here too: typed error instances %d → %d, hallucinations %d → %d. "
      "Re-weighting — images or pixels — does not repair this model."
      % (SG["rediag_armA"]["total"] + SG["rediag_armA"]["fp"]))

h("14.5  Arm 2 — manufacture the situation", 2)
p("Arm 2 changes the question: instead of insisting on the examples the model gets wrong, "
  "it shows it MORE EXAMPLES of the diagnosed situation. The situation is an object cut off "
  "at the image border, and it can be manufactured for free: in repair, half of the crops "
  "pick an object and put one side of the window on a random point inside its box; the "
  "other half stay random as in the control. This changes two things at once — crops "
  "concentrate on objects AND cut them — so a third arc, objcentre, was added: the same "
  "share of dedicated crops on a random object, but framing it WHOLE. repair − objcentre is "
  "then the effect of the cut alone, the geometry the pipeline diagnosed.")
figure(_SGF("fig_crops_en.png"), "Figure 14d. The three crop policies on the same train image. "
       "Dashed: object boxes; solid white: the chosen object.")
table(["Seed", "control", "objcentre", "repair", "framing", "cut", "total"],
      [[s, _f4(c), _f4(o), _f4(r), _sg(o - c), _sg(r - o), _sg(r - c)] for s, c, o, r in _a2["per_seed"]]
      + [["mean", _pm(_a2["control"]), _pm(_a2["objcentre"]), _pm(_a2["repair"]),
          _sg(_a2["frame"]["mean"]), _sg(_a2["cut"]["mean"]), _sg(_a2["total"]["mean"])],
         ["paired t", "", "", "", "%.1f" % _a2["frame"]["t"], "%.1f" % _a2["cut"]["t"],
          "%.1f" % _a2["total"]["t"]]],
      widths=[0.6, 1.1, 1.1, 1.1, 0.85, 0.85, 0.85])
caption("Table 14.5. Arm 2, global mIoU per seed (%d seeds). The last three columns are paired "
        "differences: objcentre − control, repair − objcentre, repair − control." % len(_a2["per_seed"]))
figure(_SGF("fig_arms_en.png"), "Figure 14e. The three arms seed by seed. The controls of arms "
       "A and 2 (same recipe, different scripts) differ by %.4f: the baseline is stable."
       % SG["ctrl_gap"])
figure(_SGF("fig_decomposition_en.png"), "Figure 14f. The three paired differences of arm 2, "
       "seed by seed (numbers are seeds); tick and bar are the mean and its 95% interval — an "
       "interval containing zero is a difference not distinguishable from noise.", width=5.6)
_f3 = _a2["first3"]
if _a2["later"] and _f3["total"]["signif"] and not _a2["total"]["signif"]:
    p("With the first %d seeds arm 2 looked like it worked: %s on the total, every difference "
      "positive, t = %.1f, and the decomposition gave %s to framing and %s to the cut "
      "(t = %.1f). Seeds %s overturn it: repair ends %s from the control."
      % (len(_f3["seeds"]), _sg(_f3["total"]["mean"]), _f3["total"]["t"], _sg(_f3["frame"]["mean"]),
         _sg(_f3["cut"]["mean"]), _f3["cut"]["t"], " and ".join(map(str, _a2["later"])),
         " and ".join(_sg(r - c) for s, c, o, r in _a2["per_seed"] if s in _a2["later"])))
p("Over %d seeds the mean gain is %s ± %.4f (t = %.1f); framing is worth %s (t = %.1f), the "
  "cut %s (t = %.1f). With the threshold of %.2f, %s If the true gain were the observed mean, "
  "about %s seeds would be needed for t to cross the threshold; and repair varies more than "
  "the control (standard deviation %.4f against %.4f), which alone is enough to produce "
  "brilliant or disappointing results by chance on few seeds."
  % (len(_a2["total"]["d"]), _sg(_a2["total"]["mean"]), _a2["total"]["sd"], _a2["total"]["t"],
     _sg(_a2["frame"]["mean"]), _a2["frame"]["t"], _sg(_a2["cut"]["mean"]), _a2["cut"]["t"],
     _a2["total"]["crit"],
     "none of the three differences is distinguishable from noise, and the cut — the only part "
     "attributable to the diagnosis — is the weakest." if not any(_a2[k]["signif"] for k in ("frame", "cut", "total"))
     else "some of the differences are distinguishable from noise.",
     _a2["needed"] or "more than 30", _a2["repair"][1], _a2["control"][1]))
p("Where there is a gain it is not localised on the group: repair − control is %s on the "
  "val images with objects at the border and %s on the others, and the cut alone %s and %s. "
  "The border group covers %d of the %d val images, so the others are only %d and their mIoU "
  "is noisier — which is why the decisive test is the re-diagnosis, which counts errors "
  "object by object."
  % (_sg(_a2["total_group"]["mean"]), _sg(_a2["total_rest"]["mean"]), _sg(_a2["cut_group"]["mean"]),
     _sg(_a2["cut_rest"]["mean"]), _a2["n_group"], _a2["n_val"], _a2["n_val"] - _a2["n_group"]))
figure(_SGF("fig_per_class_en.png"), "Figure 14g. The same differences per class, paired means "
       "over the seeds. Class by class the differences are noisier still: the figure shows where "
       "the model moves, not demonstrated effects.", width=5.6)

h("14.6  The re-diagnosis: has the diagnosed defect decreased?", 2)
_rd = SG.get("rediag")
if _rd:
    _s = _rd["summary"]
    p("Block 1 is re-run on the val masks of the trained models; error instances are split "
      "between those whose box touches the image border (within %d px) and the interior "
      "ones, and the arcs are compared seed by seed. The criterion was fixed before looking: "
      "if the diagnosed cut repairs something of its own, repair against objcentre must "
      "reduce the errors at the border more than the interior ones." % _rd["edge_px"])
    table(["Arc", "seeds", "errors at the border", "interior errors", "FN at the border", "FN interior"],
          [[a, _s[a]["n_seeds"]] + ["%.0f ± %.0f" % (_stat.mean(_s[a][k]), _stat.stdev(_s[a][k]) if len(_s[a][k]) > 1 else 0)
                                    for k in ("edge", "inner", "fn_edge", "fn_inner")]
           for a in ("control", "objcentre", "repair") if a in _s],
          widths=[1.1, 0.6, 1.4, 1.4, 1.2, 1.2])
    caption("Table 14.6. Typed error instances on val per arc (mean ± sd over the seeds).")
    figure(_SGF("fig_rediag_en.png"), "Figure 14h. Relative paired change of the typed errors, per "
           "seed. The deciding panel is the last: repair against objcentre.")
    p("%s Against objcentre, repair changes the errors at the border by %+.1f%% and the interior "
      "ones by %+.1f%% (border below interior in %d of %d seeds). Every difference between arcs "
      "is of the order of 1%%, against a seed-to-seed dispersion of the same arc up to ±%.1f%% "
      "on the border errors: the independent instrument sees no effect of the cut, consistently "
      "with the mIoU."
      % ("The criterion is met." if _rd["met"] else "The criterion is not met.",
         100 * _rd["edge"], 100 * _rd["inner"], _rd["per_seed_met"], _rd["n"], 100 * _rd["noise"]))

h("14.7  Examples on val", 2)
_vs = SG["picks"]["val_stats"]
figure(_SGF("fig_val_examples_en.png"), "Figure 14i. Control and repair (seed 0) on the same val "
       "images: the three largest per-image gains among images with an object within 10 px of "
       "the border, and the largest loss. Seed 0 is one of the seeds where repair wins: the "
       "examples show what changes between the two models, not the average effect.", width=6.4)
p("The largest gains are large objects cut by the border that the control does not recognise "
  "(a train taken for background) or confuses (a sofa instead of the background around a "
  "cat); the worst case shows the reverse — the person is segmented almost identically, but "
  "a small patch of another class adds a class with zero IoU and halves the image's mIoU. "
  "Even on seed 0, of %d images with objects at the border repair improves the per-image mIoU "
  "on %d and worsens it on %d (mean change %s): the dataset mIoU sums pixels and rewards "
  "recovered large objects, while the per-image mIoU is dominated by small spurious classes."
  % (_vs["n_border_images"], _vs["better"], _vs["worse"], _sg(_vs["mean_gain"])))

try:
    from _revision_numbers import load_ea_repair as _lear
    ER = _lear()
except Exception as _ex:
    ER = None
    print("  (Pascal-EA repair numbers unavailable, section 14.8 skipped:", _ex, ")")
if ER:
    _EM, _EP = ER["means"], ER["paired"]
    h("14.8  A planted cause: when the diagnosis names a gap in the data", 2)
    p("The arms above aim at situations that are frequent in VOC train already — objects at the "
      "border, objects overlapping. The planted cause of §19 is the opposite case: animals "
      "re-rendered in paper, metal, plastic or wood are absent from the training data, and the "
      "diagnosis names them object by object (§19.3). The experiment asks whether the same recipe "
      "repairs a failure when the diagnosis names a real gap in the data, and whether the images "
      "the diagnosis picks are the ones worth adding.")
    p("The Pascal-EA pairs are split into two halves stratified by class and by victim: %d pairs "
      "with %d victims (A) and %d with %d (B). Three arcs fine-tune the starting checkpoint on VOC "
      "train plus the %d images of half A, with the recipe of the arms above (18 epochs, %d seeds "
      "each, the same number of images and steps); they differ only in which of those images carry "
      "the edit — control none, diagnosed only the %d victims of half A (the objects the pipeline "
      "finds the starting model losing), repair all of them. Half B is only tested. Criteria fixed "
      "before the runs: primary, the subject recall of half B's victims on their edited images, arc "
      "minus control paired by seed, positive with t > 2.776; constraint, the mIoU of the val "
      "images never edited must not fall by more than 0.005 — the seed noise of §14.5 — nor "
      "significantly (SEGIMPROVER/ea_repair_report.py)."
      % (ER["n_A"], ER["victims_A"], ER["n_B"], ER["victims_B"], ER["n_A"], ER["seeds"]["control"],
         ER["victims_A"]))
    _ek = [("recall.B_victims_edited", "PRIMARY: recall of half B's victims, edited images"),
           ("miou.unedited", "CONSTRAINT: mIoU, val images never edited"),
           ("recall.B_victims_original", "recall of the same victims, original images"),
           ("recall.B_others_edited", "recall of the other half-B subjects, edited"),
           ("miou.B_edited", "mIoU, half-B edited images")]
    table(["Metric", "start", "control", "diagnosed", "repair"],
          [[lab, "%.3f" % ER["start"][k]] + ["%.3f" % _EM[a][k] for a in ("control", "diagnosed", "repair")]
           for k, lab in _ek], widths=[2.9, 0.7, 0.8, 0.9, 0.8])
    caption("Table 14.8. The three arcs, means over %d seeds, beside the starting checkpoint." % ER["seeds"]["control"])
    _d = lambda pair, k: _EP[pair][k]
    _fmt = lambda pair, k: "%+.4f (t = %.1f)" % (_d(pair, k)["mean"], _d(pair, k)["t"])
    table(["Paired by seed", "repair − control", "diagnosed − control", "repair − diagnosed"],
          [[lab] + [_fmt(pr, k) for pr in ("repair-control", "diagnosed-control", "repair-diagnosed")]
           for k, lab in _ek], widths=[2.5, 1.3, 1.3, 1.3])
    caption("Table 14.9. Differences paired by seed; the threshold for 5 seeds is |t| > 2.776.")
    _rc, _dc = _EP["repair-control"], _EP["diagnosed-control"]
    _share = _dc["recall.B_victims_edited"]["mean"] / _rc["recall.B_victims_edited"]["mean"]
    p("The primary criterion is met by both arcs, by a wide margin: the recall of the held-out "
      "victims rises by %.2f (repair, t = %.0f) and %.2f (diagnosed, t = %.0f) over the control, "
      "with the same sign on every seed. The constraint is not: the val images never edited lose "
      "%.3f of mIoU with the full repair and %.3f with the diagnosed images (t = %.1f and %.1f), "
      "beyond the seed noise — the repair costs something elsewhere. The victims' original images do not suffer (%+.3f). The images the diagnosis "
      "picks, %.0f%% of half A, deliver %.0f%% of the full repair's gain. And the repair is partial: "
      "the victims reach %.2f on their edited images against %.2f on their originals; the control "
      "alone, which only shows the scenes unedited, already lifts them from %.2f to %.2f."
      % (_rc["recall.B_victims_edited"]["mean"], _rc["recall.B_victims_edited"]["t"],
         _dc["recall.B_victims_edited"]["mean"], _dc["recall.B_victims_edited"]["t"],
         -_rc["miou.unedited"]["mean"], -_dc["miou.unedited"]["mean"], _rc["miou.unedited"]["t"],
         _dc["miou.unedited"]["t"], _rc["recall.B_victims_original"]["mean"],
         100.0 * ER["victims_A"] / ER["n_A"], 100 * _share, _EM["repair"]["recall.B_victims_edited"],
         _EM["repair"]["recall.B_victims_original"], ER["start"]["recall.B_victims_edited"],
         _EM["control"]["recall.B_victims_edited"]))
    p("Halves A and B come from the same edit generator and the same classes: the repair is shown on "
      "new images with the same kind of edit, not on other materials or on real toys and statues.",
      italic=True)
h("14.9  What it means", 2)
bullet("Using the diagnosis to WEIGHT is not enough. Resampling a situation moves the gain "
       "from one part of the dataset to another; weighting pixels by error type moves nothing. "
       "As a weight, the diagnostic signal is most likely redundant with what the cross-entropy "
       "already sees: wrong pixels already have the highest loss.")
bullet("Manufacturing the situation did not give a demonstrable improvement. The mean is "
       "positive (%s), but with %d seeds it is not distinguishable from noise, and the part "
       "attributable to the diagnosis (%s) even less; the re-diagnosis does not see the "
       "diagnosed defect decrease."
       % (_sg(_a2["total"]["mean"]), len(_a2["total"]["d"]), _sg(_a2["cut"]["mean"])))
if _a2["later"] and _f3["total"]["signif"]:
    bullet("Three seeds were not enough. The first three gave t = %.1f on the total and %.1f on "
           "the cut; the seeds added before concluding cancelled the effect. The protocol — paired "
           "control, the objcentre arc, a criterion fixed in advance, seeds added before drawing "
           "the conclusion — is what prevented a false success, and it is the part of this "
           "chapter that carries over to any future attempt." % (_f3["total"]["t"], _f3["cut"]["t"]))
if ER:
    bullet("When the diagnosis names a gap in the data, it repairs. Adding the planted situation "
           "lifts the held-out victims' recall by %.2f with t = %.0f, and the images the diagnosis "
           "picks deliver most of it (§14.8) — the contrast with arms 1, A and 2, aimed at situations "
           "the training data already contains, is the finding of this chapter. It is not free: "
           "%.3f of mIoU elsewhere." % (ER["paired"]["repair-control"]["recall.B_victims_edited"]["mean"],
                                        ER["paired"]["repair-control"]["recall.B_victims_edited"]["t"],
                                        -ER["paired"]["repair-control"]["miou.unedited"]["mean"]))
bullet("What the diagnosis keeps. It describes the errors verifiably — the result of this "
       "document — but turning that description into a better model is a different problem: "
       "solved here only where the diagnosis names a gap in the data (§14.8), open for situations "
       "the data already contains. Still untested: a stronger intervention (more cut crops, longer "
       "training), many more seeds to measure effects of a few thousandths, and a third arm "
       "in which the model is made to predict the situation instead of having its data changed.")
h("14.10  How standard is this procedure?", 2)
p("The ingredients are standard; their combination and two choices are not. The table "
  "separates the two, so the chapter can be read against common practice.")
_lo, _hi = min(_a2["control"][0], _aA["control"][0]), max(_a2["repair"][0], _aA["control"][0])
table(["Component", "Standard?", "Note"],
      [["Fine-tuning recipe (random 0.5–2× rescale, crop, flip, AdamW with cosine decay)",
        "yes — the common recipe for semantic segmentation on VOC",
        "but small-scale: 1 464 images and 18 epochs, where the usual VOC setting trains on the "
        "~10 500 images of SBD for tens of thousands of iterations"],
       ["mIoU on VOC val", "yes",
        "the GEOMETRY is not: the evaluation uses the pipeline's short side 520 + centre crop, so "
        "the %.2f–%.2f measured here are not comparable with published figures" % (_lo, _hi)],
       ["Arm 1, resampling a group", "yes — the family of group or error up-weighting "
        "(Just Train Twice, Liu et al. 2021; Group DRO, Sagawa et al. 2020)",
        "defining the group by a geometric proxy instead of labels is ours"],
       ["Arm A, per-pixel loss weighting", "yes — the family of focal (Lin et al. 2017) and "
        "class-balanced (Cui et al. 2019) losses",
        "weights derived from error types, with an equal share per type, are ours"],
       ["Arm 2, augmentation targeted at a situation", "the idea is known — using discovered "
        "failures to guide data and augmentation (e.g. Jain et al. 2023, failure directions in "
        "latent space)", "the specific rule, a crop anchored on an object's border, is ours"],
       ["Paired control arc, same seed and same number of steps", "good practice, not universal",
        "many works report a single run"],
       ["The objcentre arc that decomposes the effect", "not standard",
        "an ablation designed for this question; the methodologically strongest point"],
       ["Paired t with a Student threshold; criterion fixed in advance",
        "standard statistics; fixing the criterion in advance is rare in computer vision", ""],
       ["Re-diagnosis with Block 1", "no", "an instrument of this work; the original contribution "
        "of the protocol"]],
      widths=[2.1, 2.1, 2.3])
caption("Table 14.7. What in the improvement protocol follows common practice and what was "
        "built for this work.")
p("Limits to declare when the procedure is presented as such:")
bullet("%d seeds are few for effects of a few thousandths: at the observed effect about %s "
       "would be needed." % (len(_a2["total"]["d"]), _a2["needed"] or "30"))
bullet("The t-test on %d paired differences assumes they are roughly normal."
       % len(_a2["total"]["d"]))
bullet("Several arms are compared without a correction for multiple comparisons.")
bullet("Arm 1 has a single seed.")
bullet("The evaluation geometry is not the standard one.")
p("In one sentence: the fine-tuning recipe and the metric follow common practice for VOC; the "
  "interventions are instances of known families (resampling, weighted loss, targeted "
  "augmentation); the evaluation protocol — paired arcs, a decomposition control, a criterion "
  "fixed in advance, and the re-diagnosis — was built for this work.")

p("Reproduce: SEGIMPROVER/ — build_groups.py, diagnose_train.py, finetune_weighted.py, "
  "finetune_errorweighted.py, finetune_border.py (POLICY=control | objcentre | repair), "
  "evaluate_arm.py, rediag_compare.py; the Italian report report_segimprover.docx is built "
  "from the same files.", italic=True)

# ============================ 7. DECISIONS ============================
from _revision_numbers import load_rule_min as _lrm_d
_RMd = _lrm_d()
h("7. Key design decisions", 1)
p("Every row below was a fork in the road. The middle column is the option that was "
  "tried or was the obvious default; the right column is what made it lose. Where the "
  "evidence is a measurement made in this work, the section is cited \u2014 several of "
  "these decisions were taken only AFTER the naive choice had been implemented and "
  "shown to fail, which is why the report keeps those episodes.")
table(
    ["Decision", "Alternative rejected", "Evidence that decided it"],
    [["The diagnosis is the deliverable",
      "Distilling a VLM's free-text description into the segmenter through an embedding-space loss "
      "(the original proposal)",
      "The judged quality of a free-text description changed more with the input format than with "
      "the error (per-image correlation %s between formats) and stayed below 5/10: not a usable "
      "training signal (§21.6)." % ("%.2f" % FX["format"]["pearson"] if FX else "low")],
     ["One closed question per object region, its box given",
      "One free-text paragraph comparing the whole prediction with the ground truth",
      "Same first experiment; and the token bottleneck and hallucination of VLMs (§21.5)."],
     ["Gemma 4 as the language model (hypotheses, judges)", "Gemma 3n, Kimi K2.5, Qwen 3.5 9B",
      "Most accurate paragraphs in the first experiment when it answers (§21.6); run locally."],
     ["Type every error (7 cases) before anything else",
      "A single IoU score per object",
      "An IoU says how much is wrong, never what. Severity is only comparable WITHIN a "
      "type: a boundary error has high IoU by construction and could never compete with "
      "a missed object under one scale (\u00a74.1, Figure 3)."],
     ["Two matching thresholds (0.5 to claim, 0.25 to touch)",
      "One threshold at 0.5",
      "Split and merge are geometrically impossible to observe under a single threshold "
      "\u2014 two fragments of one object cannot both exceed 0.5. They counted zero "
      "until the second threshold was added; the reference typing has 98 merges and 4 "
      "splits, the object typing of the inst run %d and %d (\u00a74.1, \u00a76)."
      % (IS["tax"].get("merge", 0), IS["tax"].get("split", 0))],
     ["One instance extraction shared by both branches",
      "Each branch extracts its own instances, merged by bbox-IoU",
      "The heuristic join silently mismatched objects; the shared substrate makes the "
      "merge an exact instance_id join, which is also what makes cross-branch "
      "corroboration meaningful (\u00a74.4)."],
     ["The two branches never see each other before validation",
      "Condition hypothesis generation on the discovered clusters",
      "Conditioning would make the manifold-compactness test circular and would hide "
      "situations the visual feature space does not separate. Measured consequence: "
      "corroboration is RARE (3 of 14), i.e. the branches genuinely cut the error space "
      "along different axes (\u00a76)."],
     ["One GMM per error type, k chosen by BIC",
      "A single clustering over all errors",
      "Without stratification the severity comparison is dominated by type: FN slices "
      "pass for free and boundary/extent slices are excluded a priori (\u00a74.3)."],
     ["Purity tested against chance (permutation, p \u2264 0.05)",
      "The absolute threshold p_theta = 0.80 from SEG4SEG",
      "0.80 was calibrated for clean metadata. On a noisy 7-word VLM vocabulary it both "
      "rejected coherent slices (informative agreement 0.71) and certified a vacuous one "
      "that was 82% \u201cnormal\u201d. The permutation null carries the same noise as "
      "the statistic (\u00a74.6, Figure 4)."],
     ["\u201cnormal/other/error\u201d excluded from purity",
      "Count every tag the VLM returns",
      "The catch-all was 56% of all tags: counting it both rewarded empty slices and "
      "diluted genuine ones (\u00a74.6)."],
     ["\u201cdark\u201d excluded from purity as well",
      "Keep it as evidence while not printing it",
      "Phrases built on it are not confirmed by the VLM in either elicitation mode "
      "(0.00 at or below the matched control). Evidence one is unwilling to state is not "
      "evidence. Cost measured: none \u2014 the certified set stayed at 14 "
      "(\u00a76.2, \u00a76.3)."],
     ["Attributes elicited by grounded yes/no verification",
      "One free choice from a closed vocabulary (captioning)",
      "The same model answers \u201cnormal\u201d for 56% of instances when asked to "
      "choose a word, and withdraws that answer when asked a direct question: informative "
      "tags 38% \u2192 64% on identical regions, certified slices 9 \u2192 14 "
      "(\u00a76.3)."],
     ["\u03c9\u00b2 reported but NOT used as a gate",
      "Gate on \u03c9\u00b2 \u2265 0.14 as in SEG4SEG",
      "One-vs-rest \u03c9\u00b2 is size-confounded for small slices; the gate zeroed "
      "every slice, including ones with verified SliceLens patterns (\u00a74.5)."],
     ["Language slices validated by SliceLens\u2019s own slope trend",
      "The instance-level tests built for cluster slices",
      "Those tests use the wrong null for a semantic grouping and rejected 58 of 62 "
      "slices; the slope trend trains on error AND correct groundings, which the "
      "instance tests never see (\u00a74.4). Superseded on 21 October by the rank test below, "
      "which uses the same groundings (\u00a718.5)."],
     ["The class is printed OUTSIDE the quoted claim",
      "Fold the class into the phrase (\u201ca cow cut off\u2026\u201d)",
      "Same instances, same seed: the conjunctive phrasing drops verified precision from "
      "0.78 to 0.60, because it shifts the model from \u201cis it cut off?\u201d to "
      "\u201cis it a cow?\u201d. The class comes from ground truth and is not a claim "
      "needing confirmation (\u00a76.2)."],
     ["Slices with no verified attribute are declared as such",
      "Describe them by class alone, or drop the row",
      "\u201cA bottle\u201d scores 0.536 against a same-class control of 0.557 "
      "(lift \u22120.021): it states nothing. Dropping the row would instead hide "
      "structure that is certified and replicates (\u00a76.2)."],
     ["Every validation carries a negative control",
      "Report agreement on genuine pairs only",
      "The July human-annotation comparison paired UNRELATED images for 57% of its cases "
      "and still read \u201c70% partial agreement\u201d. A scale that agreeable on "
      "scrambled content measures politeness (\u00a76.5)."],
     ["A containment claim beside the IoU claim (claim_cov = 0.5)",
      "IoU claims only",
      "IoU is blind to asymmetric containment: on the annotated images 52% of the objects "
      "the text called \u201cmissed entirely\u201d were half-covered by a prediction of their own "
      "class. The swallowed objects are kept as instances, or the cluster branch loses its "
      "substrate (\u00a710.1\u2013\u00a710.3)."],
     ["A ground-truth instance is a VOC object",
      "A connected component of the class mask",
      "Occlusion breaks the component definition both ways (one object in pieces, two objects "
      "as one); the narrative, the judge and the annotators agree with the object typing. "
      "Cost declared: fewer certified cluster slices (\u00a710.7)."],
     ["Narrate EVERY typed error, in the annotators' register",
      "Describe only the errors inside certified slices",
      "People describe every error they see, by direction and side; a text limited to "
      "systematic patterns cannot be compared with them, and the low judge scores were "
      "missing statements, not wrong ones (\u00a79, \u00a710.8)."],
     ["Judgements carry a fingerprint of their reference",
      "Cache verdicts by image and kind",
      "A corrected annotation or a regenerated text would otherwise keep an old verdict; "
      "the cache is keyed by the candidate text and the reference's hash, so only what "
      "changed is re-asked (\u00a710.6, \u00a712)."],
     ["Generic, class-agnostic hypotheses",
      "More LLM-generated, class-specific hypotheses",
      "Certification yield %.0f%% against %.0f%% on the same run; the class inside the query "
      "lowers the precision of the VLM's answer (\u00a76.2, \u00a713)%s."
      % (100 * XHS["canon"]["certified"] / XHS["canon"]["evaluated"],
         100 * XHS["specific"]["certified"] / XHS["specific"]["evaluated"],
         ("; the hypotheses SliceLens derives from the data certify %.0f%% (\u00a713.8)"
          % (100.0 * DDH0["n_cert"] / DDH0["n_hyp"])) if DDH0 else "")],
     ["A certified situation is also LOOKED AT",
      "Trust the slope test and the full-population re-verification",
      "A phrase that fits any photograph passed the slope test on the sample; two slices that "
      "look right failed on the full population. The checks answer different questions (\u00a713.4)."],
     ["The same protocol on SBD: sample, then re-grounding of what is systematic there",
      "Every hypothesis on every instance of the new dataset",
      "An estimated 1.3 million VLM calls, about two weeks on two GPUs; and only the same "
      "protocol makes the two datasets comparable (\u00a718)."],
     ["A planted cause with known members (material edits of val)",
      "Only natural failures, whose cause nobody knows",
      "The one test where the right answer is known: %s (\u00a719)."
      % (("%d%% of the victims found, the material never named"
          % round(100.0 * len(EAN["planted"]["covered"]) / len(EAN["planted"]["victims"]))) if EAN else "—")],
     ["Compare with the published benchmark on error types and on names, with rules fixed in advance",
      "Report the benchmark's Precision@k only",
      "%s (\u00a720)." % (("a sort by class and error kind, without hypotheses or VLM, scores %.3f against "
                             "the published %.3f: on this part the measure cannot separate methods, and it "
                             "never scores a name" % (GS0["baselines"]["baselines"]["C"]["p_at_k"], GS0["ghesd_p10"]))
                            if GS0 else "—")],
     ["A rank test (AUC with an image-block bootstrap) to certify a language situation",
      "SliceLens's fixed threshold on the largest slope",
      "%s (\u00a718.3)." % (("the slope rule calls up to %.0f%% of hypotheses with shuffled labels "
                             "systematic; the rank test is honest on whole classes and certifies %d situations "
                             "on VOC, %d on SBD" % (100 * max(v["null_old"] for v in SV0["perm"]["stages"].values()),
                                                    SV0["rank"]["compare"]["VOC val"]["new"],
                                                    SV0["rank"]["compare"]["SBD"]["new"])) if SV0 else "—")],
     ["Run the rank test on a whole class only when it has at least ten correct objects",
      "Trust the bootstrap's p-value whatever the number of correct objects",
      "%s (§18.9)." % (("with %d correct potted plant the bootstrap certified %d hypotheses and an exact "
                             "rank test %d; from ten correct objects up the two agree on %d of %d"
                             % (_RMd["classes"][0]["n_ok"], _RMd["classes"][0]["cert"], _RMd["classes"][0]["confirmed"],
                                _RMd["ge"]["confirmed"], _RMd["ge"]["cert"])) if _RMd else "—")],
     ["Describe each object with the hypotheses its class's errors are associated with",
      "Name a cause only through a certified group",
      "the planted cause is named for %s of the victims and almost never elsewhere (\u00a719.3)."
      % (("%.0f%%" % (100.0 * NM0["all"]["victim"]["top1"] / NM0["all"]["victim"]["n"])) if NM0 else "—")],
     ["Repair arms against a paired control, a decomposition arc and a criterion fixed in advance",
      "Compare a repaired model with the starting checkpoint",
      "Training longer improves a model by itself; the objcentre arc separates the diagnosed "
      "geometry from generic object-centred cropping, and the re-diagnosis criterion was "
      "written before the numbers were seen (\u00a714)."],
     ["Add seeds before concluding",
      "Stop when three seeds agree",
      "%d seeds gave t = %.1f for arm 2; %d gave t = %.1f (\u00a714.5)."
      % (len(SG["arm2"]["first3"]["seeds"]), SG["arm2"]["first3"]["total"]["t"],
         len(SG["arm2"]["total"]["d"]), SG["arm2"]["total"]["t"])]],
    widths=[1.9, 1.8, 2.9])
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
_r = cap.add_run("Table 7.1. Design decisions, the alternative each one displaced, and "
                 "the evidence that decided it.")
_r.italic = True
_r.font.size = Pt(9)
p("Two further choices are practical rather than methodological: heavy steps are cached "
  "to a pickle, so descriptions, gates and gallery can be iterated on CPU alone; and "
  "per-image descriptions are built from each image\u2019s OWN instances rather than the "
  "slice-level dominant class, which was what once produced sentences naming classes "
  "absent from the picture.")

# ============================ 8. LIMITATIONS ============================
from _revision_numbers import load_rule_min as _lrm_l
_RMl = _lrm_l()
h("8. Limitations and future work", 1)
p("Grouped by what they limit: the method, the evaluation, the scope.")
bullet("Post-selection inference on the cluster branch: slices are formed by optimising "
       "coherence on the same data used to test them, so their significance is optimistic — "
       "including the compactness permutation test, which the GMM optimises by construction "
       "(it is honest only for language slices). The split-half replication of §6 answers it "
       "empirically on the reference typing (94.1%); it was not repeated on inst.")
bullet("Benjamini–Hochberg FDR is applied per scoring batch (per peeling round, and again on "
       "refine sub-slices); multiplicity across rounds is not jointly corrected.")
bullet("The cluster branch needs many small, homogeneous instances. Under the object typing "
       "of the reference run it certifies %d slices against %d on the component typing (§6, "
       "§10.7): the truer instances are fewer and larger. The language branch carries most of "
       "the situational coverage on inst." % (_ts["cluster"], RRS["ref"]["slices"]["cluster"]))
bullet("Substrate blind spots: the 520 centre-crop removes objects at the image borders and "
       "min_pixels = 200 removes very small objects; both branches inherit these holes. "
       "Predictions remain connected components, because a semantic model has no instances: two "
       "touching objects of one class are always one predicted region, so part of the merge "
       "population is structural. Ground-truth objects require a dataset with instance ids; a "
       "semantic-only dataset falls back to components.")
bullet("Severity is uninformative within pure FN/FP/confusion strata (1 − IoU = 1 by "
       "construction): those slices are validated on coherence, size and the permutation test.")
bullet("VLM confidences are uncalibrated yes/no probabilities with the model's own yes-bias; "
       "the 0.5 / 0.6 thresholds are conventions, not calibrated operating points. The "
       "per-evaluation cost (about 1.5 s) is the scaling bottleneck of the language branch.")
bullet("The language branch can only certify situations a VLM can name and verify on a region. "
       "Missed objects, wrong extent and wrong class have no certified situation even when one "
       "is looked for on purpose (§13); a planted cause — animals re-rendered in paper or "
       "metal — is named object by object only as “a toy, statue or model”, never as its material, "
       "and never as a certified group (§19); attributes like “dark” cannot be verified at "
       "all (§6.2). "
       "Hypothesis deduplication keeps the highest-confidence paraphrase, which can prefer a "
       "vaguer wording.")
bullet("SliceLens's slope-trend test is not neutral to the size of the population: its "
       "statistic, the largest slope over windows of the confidence range, is inflated by noise "
       "on small populations, so VOC's certified situations are partly a property of a small "
       "dataset and SBD's fewer (§18). §18.3 replaces it with a rank test, the certification of "
       "the thesis runs since 21 October (§18.5). Three limits remain: the rank test is itself "
       "somewhat generous on the 120-instance samples; on SBD it judges only the hypotheses that had "
       "been re-grounded on the whole class after passing SliceLens's rule on the sample%s; and it needs "
       "correct objects to compare with. Since 25 October it is not run in classes with fewer than ten "
       "(%s), so the language branch is silent exactly where the segmenter is worst and their errors are "
       "reached only by the cluster branch (§18.9)."
       % ((" (%d significant on the sample were not re-grounded; on VOC the %d were grounded, validated and "
           "made official on 23 October, §18.8)" % (OV0["pending"]["SBD"], OV0["pending"]["VOC val"]))
          if OV0 else "",
          ("%s on VOC val%s" % (", ".join("%s %d" % ({"pottedplant": "potted plant"}.get(c["cls"], c["cls"]), c["n_ok"])
                                         for c in _RMl["below"]),
                               ", sheep %d on Pascal-EA" % _RMl["ea"]["n_ok"]["sheep"]
                               if _RMl.get("ea") and "sheep" in _RMl["ea"]["n_ok"] else ""))
          if _RMl else "—"))
bullet("The comparison with the experts of GESD (§20) has its own limits. GESD stores no "
       "predicted class, so a prediction is given its object's class and a wrong-class detection "
       "looks like a boundary error. The rules that decide whether a phrase names an expert "
       "situation were written by the author, who knew the names, though before any hypothesis "
       "was generated; they are strict (a paraphrase without a keyword does not count) and two "
       "generic matches are accidents of the substring rule. The per-object filter and the "
       "certification judge a phrase on the whole class, which discards situations that concern "
       "only part of its errors.")
_GMl = ((GS0 or {}).get("cmp") or {}).get("missing") if GS0 else None
if _GMl:
    _gl = [c for c, v in _GMl["classes"].items() if v["reason"] == "gates"]
    bullet("The gates of the language branch were set on VOC. The severity gate keeps a phrase only if its errors "
           "are not clearly milder than the typical error of their kind (mean z ≥ %+.2f); on GESD it removes every "
           "phrase the rank test certifies in %d classes (%s), and with them %d expert slices (§20.5). In two of "
           "those classes the experts describe merges, which look slight by IoU because the mask still covers its "
           "own object. A severity measured on the wrongly labelled pixels rather than on the IoU of the whole "
           "object would not penalise them; it is not tested here."
           % (_GMl["gate"]["lang_sev_z_min"], len(_gl), ", ".join(c.replace("_", " ") for c in _gl),
              sum(len(_GMl["classes"][c]["expert_slices"]) for c in _gl)))
bullet("The validation experiments of §6 (corruption, precision, tagging channel, split-half, "
       "second model) were run on the earlier reference typing and are not repeated on inst; "
       "the second segmenter was analysed only on that typing.")
bullet("The judges are one local LLM (Gemma-4-12B), deterministic but run in 4-bit, which is "
       "not bit-reproducible across machines; they measure agreement with a human annotation, "
       "not truth, on 79 annotated images; 11 of the 21 synthetic predictions are other images' "
       "ground truths pasted in (§9.4).")
bullet("The repair of the planted cause (§14.8) is shown on held-out images from the same edit "
       "generator, not on other materials or on real toys, and it costs a little mIoU elsewhere.")
bullet("The improvement study (§14) is one model, one dataset and effects of a few "
       "thousandths of mIoU; five seeds do not separate them from noise (about %s would be "
       "needed at the observed effect), and the arm that makes the model predict the situation "
       "was not tried." % (SG["arm2"]["needed"] or "30"))

# ============================ 15. CONCLUSIONS (new) ============================
h("15. Conclusions", 1)
p("The pipeline does what it was built to do: for every image on which the segmenter fails it "
  "says what went wrong, object by object, in a register a person agrees with (%.2f/10 against "
  "human annotations, %.1f points above a mismatched control), and where the failure is "
  "systematic it names a situation that is plausible as an explanation (%.2f/10) and that the "
  "VLM confirms on the object's own region. Five findings carry beyond this model."
  % (_v3["narrative"]["score"], _v3["narrative"]["score"] - _v3["narrative"]["control"],
     _al.get("score", float("nan"))))
bullet("Typing errors first is worth it. It makes severity comparable within a type, makes "
       "split and merge observable, and reveals how a failure profile changes between models "
       "where a single IoU hides it.")
bullet("Certify against chance, not against thresholds, and give every instrument a negative "
       "control. Every fixed threshold this work started with measured the noise of its channel "
       "instead of the quality of the slice; every judge without a control would have reported "
       "agreement on unrelated images. The one test this work borrowed unchanged, SliceLens's "
       "slope threshold, turned out to depend on the size of the dataset; a rank test, measured "
       "against chance at the right size, removes the dependence, and since 21 October it is the "
       "certification of the thesis runs (§18). A test calibrated against chance still needs something to "
       "compare with: in classes with almost no correct object it is not run at all (§18.9).")
if _SLC0 and _SLC0.get("facts") and SV0:
    _vb = SV0["rank"]["compare"]["VOC val"]["summary_base"]
    bullet("Against SliceLens alone, on the same data, with the same VLM and hypotheses, the pipeline explains "
           "%.1f%% of the errors instead of %.1f%%, finds %d of the %d victims of the planted cause instead of %d, "
           "and its description of an image states %.0f%% of an annotator's facts instead of %.0f%%. SliceLens's "
           "decision rule, in either version of its paper, is not calibrated against chance; the rank test is "
           "(§18.6). On GESD, the benchmark published with SliceLens, its own measure scores SliceLens higher, but "
           "it rewards the number of certified slices rather than what they say, and on naming the experts' "
           "situations the two tie (§20.5)."
           % (100 * IS["cov_union"] / IS["n_err"], 100 * _vb["cov_language"] / _vb["n_err"],
              (_SLC0.get("ea_auc") or _SLC0["ea"])["union"], _SLC0["ea"]["victims"], _SLC0["ea"]["language_total"],
              100 * _SLC0["facts"]["ours_v"]["stated"], 100 * _SLC0["facts"]["sl_v"]["stated"]))
bullet("Situations explain better than clusters, and the two branches are complementary: the "
       "cluster branch finds groups the language branch never sees, the language branch says "
       "why. But the vocabulary of nameable situations is small and generic, and some error "
       "types have no situation a VLM can verify. A planted cause shows both halves at once: "
       "the cluster branch isolates its victims almost pure, no certified situation names it, and "
       "the description of each object does (§19.3). Against the slices experts defined on the "
       "benchmark published with SliceLens the same holds: Block 1 agrees with the experts where "
       "their words describe the masks, the VLM finds their objects when our vocabulary holds "
       "their situation, and half of their situations are not in it (§20).")
bullet("Description and discovery pull in opposite directions. The typing that describes "
       "errors truthfully (objects, containment) leaves the cluster branch fewer groups to "
       "certify; this document reports the truer typing and states the cost.")
bullet("A verifiable diagnosis is a repair only where it names a gap in the data: adding the "
       "planted situation repairs most of it, and the images the diagnosis picks are the useful "
       "ones (§14.8). Where it names situations the data already holds, re-weighting with it does not help; "
       "manufacturing the diagnosed situation may help a little, but not demonstrably at five "
       "seeds — and a careful protocol is what stopped three lucky seeds from being reported as "
       "a success.")
h("15.1  The research questions, revisited", 2)
bullet("RQ1, granularity. Error slicing reaches segmentation through the object: Block 1 assigns every "
       "erroneous pixel to one of seven cases per object instance (§4.1), and the groups are formed and "
       "certified over those objects.")
bullet("RQ2, data scarcity. No dataset of root causes is needed: groups are certified against chance "
       "(§4.6, §18.3). Where a known cause could be built, the pipeline was checked against it — a "
       "controlled corruption (§6.1), diffusion edits planted in val (§19) and slices named by experts "
       "(§20). Generative editing helped as a test and as a repair, not as the data source the May plan "
       "proposed.")
bullet("RQ3, semantic–geometric mapping. Each description joins a geometric part decided from the "
       "masks to a situational part verified by the VLM (§4.8); the narrative and its revisions are "
       "scored against human notes (§9, §10).")
bullet("RQ4, beyond the closed world. The cluster branch has no vocabulary at all; the language branch "
       "generates hypotheses freely and verifies them on each region (§4.3, §4.4). Its limit is the "
       "vocabulary: what is not proposed cannot be named (§13, §20).")
bullet("RQ5, validation. Every instrument was given a control, and two of the checks found real "
       "defects in the pipeline that were then fixed (§6); an exact test set beside the bootstrap found one in "
       "the certification itself, in the classes with almost no correct object (§18.9). The answer is not a "
       "single score but a set of independent checks that can each fail.")
p("Every number in this document is read from the run files when it is built, the earlier "
  "states of the pipeline are kept in their own directories, and the episodes in which a "
  "measurement turned out to be wrong are reported with their correction rather than removed.",
  italic=True)

# ============================================================================
#  11. The three typings and the narrative revision, side by side
#      (September 2026). Numbers from docs/_revision_numbers.load_runs and the
#      quoted texts from load_examples -- nothing in this section is typed in.
# ============================================================================
from _revision_numbers import load_runs as _load_runs, load_examples as _load_examples

EX_IDS = ["2010_003383", "2007_000129", "2008_006784", "2008_003703"]
try:
    RR, EX = _load_runs(), _load_examples(EX_IDS)
except Exception as _ex:
    RR = EX = None
    print("  (per-run numbers unavailable, section 11 skipped:", _ex, ")")

if RR and EX:
    _COLS = ["ref", "cov05b", "inst", "inst_v3"]

    def _quote(label, rec, key):
        """One run's narrative for an example image, with its judge score."""
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Inches(0.25)
        r = par.add_run("%s (%.0f/10)  " % (label, rec["runs"][key]["score"]))
        r.bold = True
        r.font.size = Pt(9)
        r = par.add_run(rec["runs"][key]["text"])
        r.italic = True
        r.font.size = Pt(9)
        return par

    def _example(img, title, comment):
        rec = EX[img]
        h(title, 2)
        figure("panels/%s.png" % img,
               "Figure. %s — image, ground truth, prediction (LR-ASPP MobileNetV3), "
               "cropped the way the pipeline crops them." % img)
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Inches(0.25)
        r = par.add_run("hand-written annotation  ")
        r.bold = True
        r.font.size = Pt(9)
        r = par.add_run(rec["human"])
        r.italic = True
        r.font.size = Pt(9)
        _quote("reference", rec, "ref")
        _quote("containment", rec, "cov05b")
        _quote("objects", rec, "inst")
        _quote("objects + narrative v3", rec, "inst_v3")
        p(comment)

    h("11. The three typings and the narrative revision, side by side "
      "(September 2026)", 1)
    p("§10 introduced three changes one at a time, and each was run into its own "
      "directory instead of overwriting the previous one: the reference numbers of §6 stay "
      "reproducible, and any two runs can be compared image by image with the same judge, the "
      "same annotations and the same control design. This section is the summary a reader needs "
      "before choosing which run the thesis reports: what each run is, what it did to the error "
      "population, what it did to the text, and — on four images — what actually "
      "changes on the page. One caveat on the quotes below: each text is read from the run's "
      "narrative file as it stands now, while the score beside it is the judge's on the text "
      "as it was sent to him. Where the two differ — the sub-threshold sentences removed "
      "on 26 September 2026, §10.8 — the score predates the change, and the "
      "comparison page marks those cards.")
    table(["Run (directory)", "What changes in it", "How it is produced", "§"],
          [["outputs/slice_quality",
            "The reference. A claim is an IoU test (≥ τ_match·frag_frac); a GT "
            "instance is a connected component of the class mask.",
            "the original chain", "§4.1, §6"],
           ["outputs/slice_quality_cov05b",
            "Containment claims (SQ_CLAIM_COV=0.5): a GT also claims the prediction holding half "
            "of it, and a prediction mostly inside a same-class GT claims it; swallowed GTs are "
            "kept as GT-side MERGE instances (1 + M); a merge is measured against the union of "
            "what it merged.",
            "retype_pickle.py, CPU, no re-embedding", "§10.1–§10.6"],
           ["outputs/slice_quality_inst",
            "Containment claims AND a GT instance is a VOC OBJECT (SQ_GT_INSTANCES=object): an "
            "object cut by an occluder is one instance with a non-connected mask, two touching "
            "objects of a class are two. Predictions stay connected components.",
            "retype_instances.py, GPU only for the 664 new instances", "§10.7"],
           ["the same directory, narrative v3",
            "Nothing in Block 1: the TEXT gains direction and side, a sentence on every correct "
            "region, spills, sub-threshold predicted patches and an opening verdict.",
            "narrate_errors.py, then the judge again", "§10.8"],
           ["the same directory, narrative v5",
            "Nothing in Block 1: the TEXT says which class took the place of which, on the object "
            "(relabelled_as) and on the patch lying on it (over_gt).",
            "narrate_errors.py (CPU); hand verdicts read again", "§9.5"],
           ["the same directory, narrative v6",
            "Nothing in Block 1: the TEXT names the corner of a defect, where an outline is ragged, "
            "two classes per object, small patches and the holes of correct regions.",
            "narrate_errors.py (CPU); hand verdicts read again", "§9.5"]],
          widths=[1.5, 3.1, 1.5, 0.5])
    _t = {k: RR[k]["tax"] for k in ("ref", "cov05b", "inst")}
    table(["Typed errors on val", "reference", "containment", "objects"],
          [[n] + [str(_t[k].get(n, 0)) for k in ("ref", "cov05b", "inst")]
           for n in ("fn", "fp", "boundary", "extent", "confusion", "split", "merge")]
          + [["total error instances"]
             + [str(RR[k]["n_err"]) for k in ("ref", "cov05b", "inst")],
             ["CORRECT references (ω², contour notes)"]
             + [str(RR[k]["n_ref"]) for k in ("ref", "cov05b", "inst")]],
          widths=[2.4, 1.4, 1.4, 1.4])
    p("The populations are not comparable instance by instance, which is why the runs live "
      "apart. Under the IoU rule %d regions are “missed”; the containment claim turns "
      "most of them into merges the text can describe (%d merge instances, blobs plus the objects "
      "they swallowed) and into 1↔1 pairs that were an FP and an FN before, which is also "
      "where %d splits come from. Object instances then halve the misses that remain (%d → "
      "%d), because an object matched by one of its parts is no longer missed by the others, and "
      "cut the merges (%d → %d), because a blob covering two parts of one object merges "
      "nothing." % (_t["ref"]["fn"], _t["cov05b"]["merge"], _t["cov05b"]["split"],
                    _t["cov05b"]["fn"], _t["inst"]["fn"], _t["cov05b"]["merge"],
                    _t["inst"]["merge"]))
    table(["Result (same 79 annotated images)", "reference", "containment", "objects",
           "objects + v3"],
          [["narrative, judge score"] + ["%.2f" % RR[k]["narrative"]["score"] for k in _COLS],
           ["    completeness / 4"] + ["%.2f" % RR[k]["criteria"]["completeness"] for k in _COLS],
           ["    correctness / 4"] + ["%.2f" % RR[k]["criteria"]["correctness"] for k in _COLS],
           ["    specificity / 2"] + ["%.2f" % RR[k]["criteria"]["specificity"] for k in _COLS],
           ["    scoring 10 / scoring ≤ 6"]
           + ["%d / %d" % (RR[k]["ten"], RR[k]["low"]) for k in _COLS],
           ["    control floor (mismatched pairs)"]
           + ["%.2f" % RR[k]["narrative"]["control"] for k in _COLS],
           ["pattern text, judge score"] + ["%.2f" % RR[k]["pattern"]["score"] for k in _COLS],
           ["certified cluster slices"] + [str(RR[k]["slices"]["cluster"]) for k in _COLS],
           ["language slices"] + [str(RR[k]["slices"]["language"]) for k in _COLS],
           ["share of errors covered by a pattern"] + ["%.2f" % RR[k]["cov"]["share"] for k in _COLS],
           ["synthetic set, declared-wrong half"]
           + ["%.2f" % RR[k]["synth_scores"]["narrative"]["bad"] for k in _COLS],
           ["synthetic set, declared-perfect half"]
           + ["%.2f" % RR[k]["synth_scores"]["narrative"]["good"] for k in _COLS]],
          widths=[2.5, 1.0, 1.1, 0.9, 1.0])
    p("Read down the first row and the trade-off of the whole revision is visible: the narrative "
      "rises %.2f → %.2f → %.2f → %.2f while the certified cluster slices fall "
      "%d → %d → %d and the share of errors a pattern covers falls %.2f → %.2f (the language branch "
      "certified by SliceLens's slope rule in every column, as when the runs were compared). "
      "The two are one fact seen from both ends: every change MERGES instances that used to be "
      "separate — a swallowed GT and the blob holding it, the parts of one occluded object "
      "— and the GMM certified slices out of exactly those small, homogeneous, numerous "
      "instances. The description side gets truer, the discovery side has fewer groups to "
      "certify. The control floor barely moves across the four columns, so none of the gain is "
      "verbosity."
      % (RR["ref"]["narrative"]["score"], RR["cov05b"]["narrative"]["score"],
         RR["inst"]["narrative"]["score"], RR["inst_v3"]["narrative"]["score"],
         RR["ref"]["slices"]["cluster"], RR["cov05b"]["slices"]["cluster"],
         RR["inst"]["slices"]["cluster"], RR["ref"]["cov"]["share"], RR["inst"]["cov"]["share"]))
    _c, _c3 = RR["ref"]["text_counts"], RR["inst_v3"]["text_counts"]
    p("The same shift inside the 1 483 narratives: “missed entirely” appears in %d of "
      "them under the IoU rule and in %d under object instances; “absorbed into a "
      "neighbouring prediction”, the fact that led to §10, %d → %d (it became a "
      "merge with a name); merges told against their union %d → %d; “partly hidden "
      "behind” 0 → %d; spill clauses %d → %d; opening verdicts 0 → %d; and "
      "“No segmentation error was found”, which was the whole text on %d images, "
      "%d → %d."
      % (_c["missed"], _c3["missed"], _c["absorbed"], _c3["absorbed"], _c["union"], _c3["union"],
         _c3["hidden"], _c["spill"], _c3["spill"], _c3["verdict"], _c["silent"], _c["silent"],
         _c3["silent"]))

    _example("2010_003383", "11.1  A cow in pieces: what the containment claim buys",
             "The annotation is about one cow with coarse boundaries, especially the legs. Under "
             "the IoU rule the pipeline reports a merge of two regions plus six “absorbed” "
             "ground-truth regions — every statement true, but it is counting the animal's "
             "disconnected parts. The containment claim binds them to the blob and measures the "
             "blob against their union, so the boundary sentence the annotator wrote finally "
             "exists. Object instances then say it with the right arithmetic: three objects, not "
             "eight fragments.")
    _example("2007_000129", "11.2  Bicycles behind riders: what object instances buy",
             "“Six BICYCLE regions missed entirely” is what the component rule sees: the "
             "frames are cut into pieces by the riders' legs, and each piece is an instance of its "
             "own. The containment claim fixes the arithmetic (they are swallowed by the PERSON "
             "masks, and the text says “four BICYCLE regions now labelled PERSON”), but "
             "only the object masks say what the annotator says — one bicycle, partly hidden "
             "behind the person, under-segmented — and the narrative revision adds on which "
             "side and by how much.")
    _example("2008_006784", "11.3  A sofa read as background: what the opening verdict buys",
             "All four texts describe the same failure and all four are true; only the last one "
             "says first what the annotator says first, that the object is essentially gone. The "
             "verdict is computed from the masks (81% of the annotated area predicted as "
             "background, above the VERDICT_MIN = 50% cut) and put in front. Nothing in the "
             "taxonomy changed here — the gain is ordering and one sentence.")
    _example("2008_003703", "11.4  A well-segmented aeroplane: the silence that cost points",
             "No typed error, so for three runs the text was “No segmentation error was found "
             "in this image”, while the annotator wrote that the plane is segmented quite "
             "well with some boundaries not perfectly defined — a description of a CORRECT "
             "region, which the pipeline had no sentence for above the contour-note threshold. "
             "This is the case that makes the point of §10.8: the low scores were not wrong "
             "statements, they were missing ones.")
    p("Which run the thesis reports is a choice, and the honest version of it is this. The object "
      "typing is the true one — an occluded object is one object, and the narratives that "
      "the judge, the annotators and the synthetic set all prefer are built on it — while "
      "the reference run is the one whose cluster branch certifies the most patterns, and every "
      "number of §6 belongs to it. The two branches are measured separately throughout this "
      "report precisely so that the choice need not be hidden: the description side can be "
      "reported from the object run, the discovery side from the run it was measured on, and the "
      "cluster branch's dependence on many small instances stated as the limitation it is. This "
      "report has made that choice: the inst run (objects + narrative v3) is its reference from "
      "§0 on, and every measurement taken on another run says so where it is reported.")


# ============================ 16. APPENDIX: CHRONOLOGY (new) ============================
from _revision_numbers import load_inst_summary as _lis_c, load_rule_min as _lrm_c
_IS111c = _lis_c(_out_path("slice_quality_inst_auc111"))      # the official run of 23 October
_RMc = _lrm_c()
h("16. Chronology of the work", 1)
p("Every step, in the order it was taken, with what it changed and where this document "
  "reports it. Dates are those of the run files and logs.")
table(["When (2026)", "Step", "What it changed", "§"],
      [["March", "How VLMs are built and fail (ViT, CLIP, Flamingo, BLIP-2, LLaVA)",
        "questions asked per region, P(yes) from the logits", "§21.5"],
       ["April", "First architecture: LLM hypotheses, VLM evaluator, clustering of the evaluations",
        "the language branch; clustering later made independent", "§21.7"],
       ["May", "Literature review: slice discovery, segmentation failures, research questions",
        "the gaps that shaped the design; RQ1–RQ5", "§1.2, §21.2–§21.4"],
       ["May – June", "First experiment: free-text descriptions by four VLMs, judged (JOIN / SEP); "
        "second judge; length analysis", "the pivot from distillation to a grounded diagnosis", "§1.1, §21.6"],
       ["June – early July", "The two-branch pipeline built from five methods",
        "Blocks 1–9, VOC run on 12 classes", "§3, §4"],
       ["3–4 July", "Shared instance substrate; type-aware language gate; per-type GMM with BIC; "
        "paraphrase deduplication", "exact id join between branches; boundary slices possible", "§4.3, §4.4"],
       ["8 July", "Full audit", "fragment threshold for split/merge; SQV population against the other "
        "errors; adaptive significance; ω² reported, not gated; language slices validated by "
        "their own slope test", "§4.5, §4.6, §6"],
       ["8–11 July", "All 20 classes grounded; validated hypotheses re-grounded on every instance "
        "(23 999 VLM evaluations); split-half replication", "35 validated situations, 94% replication",
        "§4.4, §6"],
       ["18 July", "Synthetic-corruption experiment; purity tested against chance",
        "the fallback that passed under-tagged slices removed", "§4.6, §6.1"],
       ["25 July", "Description precision with a matched control; wording ablation",
        "class printed outside the quoted claim", "§6.2"],
       ["27–30 July", "Attribute tagging rebuilt in verification mode; “dark” dropped",
        "certified cluster slices 9 → 14; the mixed-channel artefact caught", "§6.3"],
       ["30 July – August", "Second segmenter (DeepLabV3-R101), stages 1–3; true union coverage",
        "the coverage figure corrected everywhere", "§6.6"],
       ["28 August", "LLM judge of the human comparison, with an order-swapped self-consistency check",
        "κ = 0.903 on the corrected protocol", "§6.4, §6.5"],
       ["3 September", "The 36 annotated train images; deterministic mask facts",
        "79 annotated images in the comparison", "§9.2"],
       ["September", "Narrative of every typed error; rubric judge with mismatched controls; "
        "synthetic set with a negative class", "a second text, scored against the annotations", "§9"],
       ["15–20 September", "Containment claims (cov05b), then ground-truth objects (inst), then the "
        "narrative revision (v3)", "the typing a person agrees with; fewer cluster slices", "§10, §11"],
       ["26 September", "Self-referential sentences removed from the narratives; plausibility of the "
        "cause judged on 39, then on every image with a cause", "the other half of the deliverable measured",
        "§10.8, §12"],
       ["29 September", "Six annotations corrected; judgements fingerprinted and re-asked",
        "every score on the pages belongs to the text shown", "§10.6, §12"],
       ["30 September – 2 October", "Improvement study: resampling, error-weighted loss, manufactured "
        "situation with the objcentre control; five seeds; re-diagnosis",
        "no arm beyond seed noise; three seeds had misled", "§14"],
       ["4–6 October", "Eight extra generic hypotheses: sample, full re-verification, visual check; the "
        "three survivors merged into the inst run; this document", "41 certified situations", "§13"],
       ["7–10 October", "Pascal-EA material edits of val: the segmenter on the pairs, the pipeline on "
        "val-EA, the planted-cause evaluation, a judge, the gallery with a random control section",
        "found, not named", "§19"],
       ["9–14 October", "The pipeline on SBD (9 118 images) with the thesis protocol; why fewer "
        "certified situations: two checks on the saved groundings", "the slope test depends on N", "§18"],
       ["15–16 October", "Size-aware certification (permutation null, rank test) on the saved "
        "groundings; the planted cause named object by object; its repair by fine-tuning (3 arcs, 5 seeds)",
        "the slope rule is the bottleneck; named and repaired when visible", "§18.3, §19.3, §14.8"],
       ["14–15 October", "SliceLens's data-driven hypotheses on the 16 classes without them: generation, "
        "sample, full re-verification, visual check (on copies of the inst run)",
        ("%d of %d certified; the gap stays open" % (DDH0["n_cert"], DDH0["n_hyp"])) if DDH0 else "—",
        "§13.8"],
       ["15 October", "GESD, the benchmark published with SliceLens: its annotations read, sorts "
        "without hypotheses or VLM scored with its own measure", "a sort by class and error kind "
        "matches the published method", "§20.1"],
       ["17 October", "Block 1 against the experts' words, every COCO object as context",
        "agrees where the words describe the masks; the rest is vocabulary", "§20.2"],
       ["17–19 October", "The whole pipeline on GESD: 36 hypotheses per class, sample, re-grounding; "
        "naming measured at three levels with rules fixed in advance",
        "the VLM finds the experts' objects when the phrase exists; half the phrases are missing", "§20.3–§20.4"],
       ["19–20 October", "SliceLens's data-driven hypotheses on SBD, 16 classes: generation, sample, "
        "re-grounding on the whole classes (on copies of the SBD run)",
        ("%+d situations under the slope rule; %d of %d under the rank test"
         % (SDD0["cmp"]["dd"] - SDD0["cmp"]["base"], SDD0["full_auc"], SDD0["n_full"])) if SDD0 else "—", "§18.4"],
       ["20 October", "SliceLens's data-driven hypotheses on val-EA, 20 classes: generation and sample "
        "(on copies of the val-EA run)",
        ("%+d situations, %+d victims; the material never named"
         % (EDD0["cmp"]["dd"] - EDD0["cmp"]["base"], EDD0["covered_dd"] - EDD0["covered_base"])) if EDD0 else "—",
        "§19.4"],
       ["21 October", "The rank test made the certification of the thesis runs; judges and validations "
        "run again on it", "the official numbers of §18.5", "§18.5"],
       ["21–22 October", "SliceLens alone against the pipeline; every SliceLens rule against the rank test on "
        "planted effects; the slope-only and rank-only situations checked on the second segmenter",
        "§18.6 and its two tables", "§18.6"],
       ["22 October", "The segmentation quality of the images of every certified slice; the sample-only "
        "hypotheses grounded on whole classes", "§18.7, §18.8", "§18.7"],
       ["22–23 October", "The situations they add validated like the official ones: precision, second "
        "segmenter, plausibility of the blamed situation", "§18.8", "§18.8"],
       ["23 October", "Those situations made official: the rank test decides at both steps on VOC val",
        "%d certified situations, %.1f%% of the errors inside a certified group"
        % (_IS111c["slices"]["language"], 100.0 * _IS111c["cov_union"] / _IS111c["n_err"]), "§18.5, §18.8"],
       ["24 October", "The rank test found unreliable in classes with almost no correct object (an exact rank "
        "test against the bootstrap); a minimum of ten correct objects per class set for the whole-class test",
        ("%d of the %d shipped situations not confirmed by the exact test" % (_RMc["shipped"] - _RMc["shipped_conf"],
                                                                           _RMc["shipped"])) if _RMc else "—",
        "§18.9"],
       ["24–25 October", "The official run rescored under the minimum, with SliceLens's data-driven hypotheses "
        "added and validated; val-EA and the paired tests of the comparison redone",
        "%d certified situations, %.1f%% of the errors inside a certified group"
        % (IS["slices"]["language"], 100.0 * IS["cov_union"] / IS["n_err"]), "§18.5, §18.9"]],
      widths=[1.1, 2.4, 2.2, 0.8])
caption("Table 16.1. The chronology of the work.")

# ============================ 17. APPENDIX: REPRODUCTION (new) ============================
h("17. Reproducing the numbers", 1)
p("Everything runs from the repository (slice_quality/ and examples/) inside the Docker image "
  "segciani:dev; GPU steps need a 24 GB card in bf16 or an 11 GB card in 4-bit. The "
  "scored results are pickled, so re-scoring, descriptions, pages and documents run on CPU.")
table(["What", "How", "Output"],
      [["The first experiment (May 2026)", "docs/first_experiment_stats.py on the score files of "
        "UNI/TESI/MYCODES/PASCAL_BestVLMModel (describe.py, evaluate.py of pascal_vlm-separate)",
        "docs/first_experiment/first_experiment.{json,txt}"],
       ["The reference run (inst)",
        "SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object; outputs/chain_inst.sh, then chain_inst_narr*.sh "
        "and outputs/chain_inst_extra.sh (extra hypotheses)", "outputs/slice_quality_inst/"],
       ["The earlier runs", "the original chain; retype_pickle.py (cov05b)",
        "outputs/slice_quality/, outputs/slice_quality_cov05b/"],
       ["Language branch", "examples/run_voc.py --substrate cc; examples/reground_validated.py; "
        "examples/ground_extra_hypotheses.py; examples/gen_datadriven.py (outputs/chain_dd_gen.sh, "
        "chain_dd_voc.sh)", "/data/outputs, /data/outputs_extra, /data/outputs_dd{,_ground}"],
       ["SBD", "slice_quality/examples/build_sbd.py; outputs/chain_sbd.sh, chain_sbd_worker.sh; "
        "slope_robustness.py; doc_numbers_sbd_ea.py (CPU)", "outputs/slice_quality_sbd/, /data/outputs_sbd"],
       ["Planted cause (Pascal-EA)", "pascal_ea_compare.py, pascal_ea_report.py, pascal_ea_examples.py; "
        "build_voc_ea.py, retype_ea.py, outputs/chain_ea.sh; ea_planted_eval.py; judge_ea_patterns.py",
        "outputs/pascal_ea/, outputs/slice_quality_ea/, /data/outputs_ea"],
       ["GESD (the SliceLens benchmark)", "outputs/gesd_baselines.py; slice_quality/examples/gesd_block1.py; "
        "slice_quality/examples/build_gesd.py, outputs/chain_gesd.sh, chain_gesd_worker.sh, chain_gesd_full.sh "
        "(GPU, guarded), gesd_full_lists.py, gesd_repair_knowledge.py; slice_quality/examples/gesd_pointB.py (CPU)",
        "outputs/gesd/, outputs/slice_quality_gesd/, /data (= pascal_vlm/gesd); copies in docs/gesd/"],
       ["Certification verdicts", "slice_quality/examples/slope_permutation.py (PHASE=test, rank, compare); "
        "outputs/chain_slope_perm.sh, chain_slope_rank.sh (CPU)", "outputs/slope_permutation/, rescored copies *_perm, *_rank"],
       ["Per-object naming (Pascal-EA)", "examples/ground_ea_naming.py (GPU, outputs/chain_ea_naming.sh with "
        "outputs/gpu_guard.sh); slice_quality/examples/per_instance_naming_ea.py (CPU)",
        "/data/outputs_ea_naming, outputs/slice_quality_ea/per_instance_naming*"],
       ["Data-driven hypotheses on SBD", "outputs/chain_dd_sbd.sh, chain_dd_sbd_join.sh, chain_dd_sbd_full.sh "
        "(GPU, guarded); slice_quality/examples/dd_auc.py (CPU)",
        "/data/outputs_dd_sbd{,_ground}, outputs/slice_quality_sbd_dd{,_base}/"],
       ["Data-driven hypotheses on val-EA", "outputs/chain_dd_ea.sh, chain_dd_ea_join.sh (GPU, guarded); "
        "slice_quality/examples/dd_ea_summary.py (CPU)",
        "/data/outputs_dd_ea{,_ground}, outputs/slice_quality_ea_dd{,_base}/"],
       ["SliceLens against the pipeline", "outputs/chain_slicelens_cmp.sh, cmp_rg_gpu1.sh, chain_night_pending.sh "
        "(GPU, guarded): slicelens_style_texts.py, judge_claims.py, judge_descriptions.py, ea_branch_coverage.py, "
        "disagreement_report.py; slope_vs_auc_sim.py, slope_rules_sim.py (CPU)",
        "outputs/slice_quality_slcmp{,_auc}/, outputs/slice_quality_ea_auc/, outputs/slope_*_sim.*"],
       ["Images of the certified slices", "slice_quality/examples/slice_image_iou.py (CPU)",
        "outputs/slice_quality_inst/slice_image_iou.*"],
       ["Sample-only hypotheses on whole classes", "outputs/chain_night_pending.sh, pend_gpu6.sh (GPU, guarded); "
        "dd_auc.py, pending_rescore.sh (CPU)", "/data/outputs_pending{,_rank}, outputs/slice_quality_pending/, "
        "outputs/slice_quality_inst_pending/"],
       ["Their validation and adoption", "outputs/chain_validate_pending.sh (GPU 2, 4, 6, guarded): "
        "eval_description_precision.py, reground_validated.py on the second segmenter, judge_cause_plausibility.py; "
        "pending_validation_report.py; outputs/chain_official_111.sh (CPU): rescore in place, model-2 groundings "
        "merged, dd_auc.py, run_second_model_language.py, slice_image_iou.py, galleries",
        "outputs/slice_quality_inst_auc111/validation_report.*, /data/outputs_model2_{pending,inst111,inst111_rank}, "
        "outputs/slice_quality_inst_auc44/ (the run before)"],
       ["The minimum of ten correct objects; the data-driven hypotheses in the official run",
        "slice_quality/examples/rank_few_correct.py, rank_min_correct.py (CPU); outputs/chain_dd_official_v2.sh "
        "(GPU 6) and chain_dd_official_v3.sh (GPU 4), guarded; outputs/chain_rule_ea_stats.sh (CPU)",
        "/data/outputs_{rank,extra_rank,pending_rank,ea_rank}_m10, /data/outputs_dd_voc_{full,rank}, "
        "/data/outputs_model2_{dd,inst_dd,inst_dd_rank}, outputs/slice_quality_inst_withdd/ (the validated copy), "
        "outputs/slice_quality_inst_auc111/ and slice_quality_ea_auc_nomin/ (the runs before)"],
       ["Repair of the planted cause", "SEGIMPROVER/: ea_split.py, finetune_ea.py, evaluate_ea.py, "
        "ea_repair_report.py, run_ea_repair.sh (GPU, guarded)", "SEGIMPROVER/runs/ea/"],
       ["Judges", "slice_quality/examples/judge_descriptions.py, judge_synth.py, "
        "judge_cause_plausibility.py (ALL=1, SYNTH=1)", "judge_*.json in the run directory"],
       ["Pages", "docs/make_compare_html.py (SHOW_SYNTH_INTRO=0, SYNTH_JUDGE=judge_synth.json); "
        "slice_quality/examples/regen_gallery.py; docs/make_share_gallery.py",
        "confronto_umano_pipeline_inst.html, gallery_full_tagged.html, gallery_share.html"],
       ["Improvement study", "SEGIMPROVER/: build_groups.py, diagnose_train.py, finetune_*.py, "
        "evaluate_arm.py, rediag_compare.py; make_figures.py (LANG_FIG=en)", "SEGIMPROVER/runs, diag, figs"],
       ["This document", "docs/make_extra_hyp_figures.py, then docs/make_thesis_en.py; every number "
        "through docs/_revision_numbers.py", "docs/thesis_document_EN.docx"]],
      widths=[1.4, 3.6, 1.6])
caption("Table 17.1. Where each number comes from.")

# ============================================================================
#  Renumbering. The source keeps the section numbers of make_document.py (plus 0,
#  14-17 for the new chapters); this maps every heading, every §-reference, every
#  "Figure 13x"/"Figure 14x" and every "Table s.n" to the final order, and lists
#  any reference that points nowhere, so a broken cross-reference fails loudly.
# ============================================================================
import re as _re2

SECMAP = {"0": "1", "1": "2", "21": "3", "2": "4", "3": "5", "4": "6", "5": "7", "6": "8",
          "9": "9", "10": "10", "12": "11", "13": "12", "18": "13", "19": "14", "20": "15", "14": "16",
          "7": "17", "8": "18", "15": "19", "11": "A", "16": "B", "17": "C"}
FIGMAP = {"13": "9", "19": "10", "14": "11", "18": "12"}
DANGLING = set()


def _num(n):
    head, *rest = n.split(".")
    if head not in SECMAP:
        DANGLING.add(n)
        return n
    return ".".join([SECMAP[head]] + rest)


def _ref(m):
    pre, a, sep, b = m.group(1), m.group(2), m.group(3), m.group(5)
    na = _num(a)
    out = ("Appendix " + na) if na[0].isalpha() else ("§" + na)
    if sep:
        nb = _num(b)
        out += sep + (("" if not m.group(4) else ("§" if nb[0].isdigit() else "")) + nb)
    return out


_REF = _re2.compile(r"(Appendix\s+)?§(\d+(?:\.\d+)*)(?:(\s*[–-]\s*)(§?)(\d+(?:\.\d+)*))?")
_FIG = _re2.compile(r"Figure (\d+)([a-z])")
_TAB = _re2.compile(r"Table (\d+)\.(\d+)")
_HEAD = _re2.compile(r"^(\d+(?:\.\d+)*)(\.?\s+)")


def _fix(text):
    text = _REF.sub(_ref, text)
    text = _FIG.sub(lambda m: "Figure %s%s" % (FIGMAP.get(m.group(1), m.group(1)), m.group(2)), text)
    text = _TAB.sub(lambda m: "Table %s.%s" % (_num(m.group(1)), m.group(2)), text)
    return text


def renumber(document):
    def paras():
        yield from document.paragraphs
        for t in document.tables:
            for row in t.rows:
                for cell in row.cells:
                    yield from cell.paragraphs
    for par in paras():
        if par.style.name.startswith("Heading") and par.runs:
            m = _HEAD.match(par.runs[0].text)
            if m:
                new = _num(m.group(1))
                lead = ("Appendix %s." % new) if (new[0].isalpha() and "." not in new) else new
                par.runs[0].text = lead + (" " if lead.endswith(".") else m.group(2)) + par.runs[0].text[m.end():]
        for r in par.runs:
            if "§" in r.text or "Figure 1" in r.text or "Table " in r.text:
                r.text = _fix(r.text)


renumber(doc)
if DANGLING:
    print("  DANGLING section references:", sorted(DANGLING))
out = os.environ.get("THESIS_OUT") or os.path.join(os.path.dirname(os.path.dirname(DOCS)), "DOCUMENTI_TESI",
                                                  "thesis_document_EN.docx")   # SLICELENS/DOCUMENTI_TESI; THESIS_OUT when Word holds it
os.makedirs(os.path.dirname(out), exist_ok=True)
doc.save(out)
print("saved", out)
