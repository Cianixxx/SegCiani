"""Generate the technical implementation document (English) as a .docx."""
import os
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
sr = sub.add_run("Implementation report — a two-branch pipeline integrating "
                 "SliceLens, Beyond IoU, MCSD, CB-SLICE and SEG4SEG")
sr.italic = True
sr.font.size = Pt(11)
doc.add_paragraph()

# ============================ 1. OVERVIEW ============================
h("1. Goal and overview", 1)
p("The goal is, for every image on which a segmentation model errs, to produce a "
  "textual description of the systematic situations that induce the failure — not "
  "merely a description of the error itself. The pipeline discovers error slices "
  "(coherent groups of failing instances) and turns them into per-image, natural-language "
  "explanations.")
p("How to read this report. Sections 1-3 state the goal and the shape of the "
  "pipeline. Section 4 walks through the blocks in the order the data flows, and every "
  "design decision that is not obvious is followed by the evidence that forced it — "
  "several of them were adopted only after a measurement showed the naive choice was "
  "wrong, and those episodes are kept rather than tidied away. Section 5 covers data and "
  "implementation, section 6 the results and the five validation experiments, section 7 "
  "collects the decisions with their rejected alternatives, and section 8 the limits. A "
  "reader in a hurry can get the argument from Figure 3 (what an error IS here), "
  "Figure 4 (why slices are certified against chance) and Table 7.1 (what was decided "
  "and on what evidence).")
p("The system runs two complementary branches over the same erroneous instances:")
bullet("Cluster branch (class-agnostic): groups error instances by visual appearance and "
       "typed-error signature. It answers what and how the model fails (e.g. “small "
       "objects of class ‘bicycle’ not detected”).")
bullet("Language branch (SliceLens): matches instances to LLM-generated situational "
       "hypotheses, verified by a VLM. It answers in what situation the model fails "
       "(e.g. “beige chair against beige wall”, “aeroplane in high-contrast "
       "sunlight”).")
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

# ============================ 2. PAPERS ============================
h("2. Integrated methods", 1)
p("The pipeline is assembled from five published methods; each contributes a specific block.")
table(
    ["Method", "Contribution to the pipeline"],
    [["SliceLens", "Language branch: LLM situational hypotheses + VLM grounding + slope-trend verification."],
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
p("Systematic-error discovery with a zero-shot VLM (Bosch Research, arXiv:2411.10845) "
  "mines systematic failures of a perception model by captioning failure cases with a "
  "generic VLM and clustering the captions in language space. The contrast is "
  "informative on three points. (a) Error definition: that work starts from a single "
  "failure notion (essentially false positives / detections that do not match), whereas "
  "Block 1 here decomposes every erroneous pixel into a typed taxonomy (FN, FP, boundary, "
  "extent, merge, split, confusion) and normalises severity WITHIN each type — without "
  "that normalisation a boundary error, whose IoU is high by construction, could never "
  "compete with a missed object. (b) Evidence: the caption-clustering pipeline reports "
  "groups and inspects them qualitatively; here every group must survive a statistical "
  "gate (chance-calibrated purity, block-bootstrap or permutation significance, BH-FDR) "
  "and a split-half replication, which is what turns 'a plausible cluster' into 'a slice "
  "the data supports'. (c) Grounding: their captions describe the whole image, so the "
  "attribution of a failure to an object is implicit; the unit here is the instance and "
  "the VLM is queried on its region, so a description is attached to the object it is "
  "about. What the paper does support is the shared premise — a VLM's open vocabulary is "
  "a workable source of failure hypotheses for a closed-set perception model — and its "
  "results are a useful external argument that this premise is not specific to VOC.")
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
bullet("Hypothesis generation (LLM, Gemma): per class, situational failure hypotheses from "
       "world knowledge + data-driven attribute mining (e.g. “beige chair against beige "
       "wall”). Stored as query files.")
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
       "requires the slice’s mean z ≥ 0: members must fail worse than is typical FOR THEIR "
       "ERROR TYPE;")
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
  "the current 12-class run the overlap is minimal (1 of 22 clusters corroborated, on weak "
  "evidence) — the branches carve the error space along complementary axes (visual vs "
  "situational), which is itself evidence that neither branch is redundant. Each kept "
  "slice’s repair advice (block 9: fix family, target data, priority) is shown on its "
  "gallery card. Output: image_descriptions.json / .txt, descriptions_index (description → "
  "all its images, per branch), slices_index (every slice → images/instances, verdicts and "
  "corroborations), a self-contained HTML gallery, and per-slice quality JSON.")

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

# ============================ 6. RESULTS ============================
h("6. Results on VOC2012 val", 1)
p("Final numbers after the full audit AND full-coverage grounding: all 20 VOC classes on "
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
figure("fig_audit_trail.png",
       "Figure 5. The certified-slice count over the life of the project. Not one of "
       "these steps changed the clustering or the discovery code: each changed how the "
       "slices were TESTED, or the instrument the test reads from. Two of the five "
       "numbers were wrong for reasons that only a later measurement exposed, and both "
       "are kept here.", width=6.4)
p("Audit trail of the cluster-slice count — three numbers, each honest about what was "
  "known at the time. 22 slices: certified under a purity gate that silently FELL BACK to "
  "error-type purity for slices with fewer than 5 Qwen tags — vacuous under "
  "stratification (etp = 1.0 by construction), so under-tagged slices passed for free; "
  "the synthetic-corruption experiment (§6.1) exposed this. 6 slices: after a 3,000-tag "
  "coverage top-up on the current slice structure, with the absolute p_theta = 0.80 "
  "threshold now actually enforced — but that threshold was calibrated for clean "
  "ground-truth attributes, and under a noisy 7-word VLM vocabulary (56% of tags are the "
  "“normal” catch-all) it both rejected genuinely coherent slices (informative-tag "
  "agreement up to 0.71, diluted below 0.8 by the catch-all) and certified one vacuous "
  "slice that was 82% “normal”. 9 slices — the final number: “normal”/“other”/“error” "
  "excluded from purity (they assert the absence of an attribute), and the absolute "
  "threshold replaced by a permutation test against chance (§4.6), which self-calibrates "
  "to the tag noise. Every certified slice now has an attribute-agreement level that "
  "provably beats random same-subtype grouping. 19 slices — the current number: the gate "
  "is unchanged, but the MEASUREMENT feeding it was rebuilt in verification mode (§6.3). "
  "The progression 22 → 6 → 9 → 19 is worth reading as a whole: the first drop removed "
  "slices that were never tested, the second removed the wrong test, the third fixed the "
  "test, and the fourth fixed the instrument the test reads from. Only the last one "
  "increased the count, and it did so without touching a single line of the discovery "
  "or gating code. (A fifth number, 19, appeared while only part of the tag pool had "
  "been re-elicited; §6.3 explains why it was an artefact of a mixed measurement "
  "channel and why 14 is the honest figure.)")
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
  "out-of-sample. 33 of 35 slices replicate — the discovered patterns are properties of "
  "the model's failure modes, not artefacts of one data half. The two failures are "
  "informative: a boundary slice (the type whose attribute tags are stale) and a slice "
  "whose held-out mass fell below N_min.")
p("Reading the numbers. (a) The cluster branch settles at 9 certified slices — smaller "
  "than any earlier draft, but the only count in which every slice passed a fair, "
  "non-vacuous, chance-calibrated purity test on fresh tag coverage. (b) The language "
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
p("Example per-image descriptions:")
code("Image 2007_004392: 7 wrong objects, 4 systematic, 3 isolated.\n"
     "  - objects of class 'person' not detected (missed)        [cluster]\n"
     "  - \"beige chair against beige wall\" (SliceLens) -> confusion   [language]\n"
     "  - \"sofa\" (SliceLens) -> confusion")

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
  "quoted earlier in this document, including in Table 1 and in \u00a76.3 above \u2014 "
  "and it is corrected here rather than silently, for the same reason the earlier "
  "audit trail keeps its wrong numbers rather than erasing them.")
p("The bug: every coverage figure so far had been computed as the size of "
  "res.kept's member set. Cluster-branch slices reach res.kept by passing Block 6's "
  "Def.1/refine loop, but language-branch slices are validated by SliceLens's own "
  "slope-trend test and never enter that loop at all \u2014 exactly why the raw "
  "\u201ckept: language 0\u201d line that Stage 3's own script printed on this run "
  "looked alarming and was not (\u00a76.6.3 has that specific reading). The "
  "consequence is that every coverage number computed via res.kept was silently "
  "CLUSTER-ONLY, undercounting by exactly the instances the language branch alone "
  "accounts for. Figure 7 (branch complementarity) had already computed the correct "
  "union for the final state \u2014 2354 covered instances, 356 of them found by "
  "both branches \u2014 but the headline coverage percentage printed alongside it, "
  "and in the tables of \u00a76.3, used the other method and was never cross-checked "
  "against the figure sitting next to it. Recomputing coverage as the union of both "
  "branches' certified members, for every state the pickled results were still "
  "available for, corrects Table 1 (21.8% \u2192 36.7%) and the \u00a76.3 trend, "
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

# ============================ 7. DECISIONS ============================
h("7. Key design decisions", 1)
p("Every row below was a fork in the road. The middle column is the option that was "
  "tried or was the obvious default; the right column is what made it lose. Where the "
  "evidence is a measurement made in this work, the section is cited \u2014 several of "
  "these decisions were taken only AFTER the naive choice had been implemented and "
  "shown to fail, which is why the report keeps those episodes.")
table(
    ["Decision", "Alternative rejected", "Evidence that decided it"],
    [["Type every error (7 cases) before anything else",
      "A single IoU score per object",
      "An IoU says how much is wrong, never what. Severity is only comparable WITHIN a "
      "type: a boundary error has high IoU by construction and could never compete with "
      "a missed object under one scale (\u00a74.1, Figure 3)."],
     ["Two matching thresholds (0.5 to claim, 0.25 to touch)",
      "One threshold at 0.5",
      "Split and merge are geometrically impossible to observe under a single threshold "
      "\u2014 two fragments of one object cannot both exceed 0.5. They counted zero "
      "until the second threshold was added; the real taxonomy has 98 merges and 4 "
      "splits (\u00a74.1)."],
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
      "instance tests never see (\u00a74.4)."],
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
      "scrambled content measures politeness (\u00a76.5)."]],
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
h("8. Limitations and future work", 1)
bullet("Post-selection inference on the cluster branch (slices formed and tested on the "
       "same data) is empirically addressed by the split-half replication above (94.3%); "
       "the language branch is exempt by construction (hypotheses are generated "
       "class-level and verified by the slope-trend on held-out-like groundings).")
bullet("Post-selection inference: cluster slices are FORMED by optimising coherence on the "
       "same data used to test them, so their significance is optimistic — this includes "
       "the compactness permutation test, which the GMM optimises by construction (it is "
       "honest only for language slices). The clean remedy is split-half validation: "
       "discover on one half, certify on the other.")
bullet("FDR is applied per scoring batch (per peeling round, and again on refine "
       "sub-slices); multiplicity across rounds is not jointly corrected.")
bullet("Substrate blind spots: the 520 centre-crop removes objects at image borders, and "
       "min_pixels=200 removes very small objects — both branches inherit these holes "
       "(notably, extremely small objects are a classic error slice this pipeline cannot see).")
bullet("“Instances” are connected components: adjacent same-class objects merge in the "
       "prediction (no instance boundaries), so split/merge statistics are approximations "
       "of that choice.")
bullet("Severity is uninformative WITHIN pure FN/FP/confusion strata (1−IoU = 1 by "
       "construction): for those slices validation rests on coherence, size and the "
       "permutation test, not on graded severity.")
bullet("VLM confidences are uncalibrated raw yes/no probabilities (Qwen has its own yes "
       "bias); the 0.5 / 0.7 thresholds are conventions, not calibrated operating points.")
bullet("Superseded by full-population re-grounding (§4.4, §6.6.2): all 20 classes are "
       "now covered on both models, and validated hypotheses are re-grounded on every "
       "instance of their class rather than a 120-instance sample — the per-evaluation "
       "cost (~1-1.5 s) is the remaining scaling bottleneck, not sample size.")
bullet("Hypothesis dedup keeps the highest-mean-confidence paraphrase, which can prefer a "
       "vaguer wording over a more specific one (vague queries score high on everything); "
       "choosing the representative post-scoring would fix this.")
bullet("Language descriptions are class-level hypotheses matched to instances, not per-image "
       "captions; a per-instance VLM captioner could complement them for full coverage.")



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
    p("Sections 4–6 describe the pipeline as it was evaluated. Comparing its per-image "
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
      f"{N['cov_after']['share']:.2f} → {NI['cov']['share']:.2f}. Merging an object's parts "
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
            "the original chain", "4.1, 6"],
           ["outputs/slice_quality_cov05b",
            "Containment claims (SQ_CLAIM_COV=0.5): a GT also claims the prediction holding half "
            "of it, and a prediction mostly inside a same-class GT claims it; swallowed GTs are "
            "kept as GT-side MERGE instances (1 + M); a merge is measured against the union of "
            "what it merged.",
            "retype_pickle.py, CPU, no re-embedding", "10.1–10.6"],
           ["outputs/slice_quality_inst",
            "Containment claims AND a GT instance is a VOC OBJECT (SQ_GT_INSTANCES=object): an "
            "object cut by an occluder is one instance with a non-connected mask, two touching "
            "objects of a class are two. Predictions stay connected components.",
            "retype_instances.py, GPU only for the 664 new instances", "10.7"],
           ["the same directory, narrative v3",
            "Nothing in Block 1: the TEXT gains direction and side, a sentence on every correct "
            "region, spills, sub-threshold predicted patches and an opening verdict.",
            "narrate_errors.py, then the judge again", "10.8"]],
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
      "%d → %d → %d and the share of errors a pattern covers falls %.2f → %.2f. "
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
      "cluster branch's dependence on many small instances stated as the limitation it is.")


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
      "attribute) not detected”, which asserts almost nothing: %.2f / %.2f, and the three "
      "lowest-scoring causes of the whole set are all of this kind. The cluster branch is what "
      "FINDS the group; the language branch is what can SAY why. Figure 6 argued the two "
      "branches are complementary from their overlap — only 356 of the 2 354 covered "
      "instances are reached by both; this measures the complementarity from the other side, "
      "on the text a reader actually gets."
      % (_b["human"]["language"][0], _b["pipeline"]["language"][0],
         _b["human"]["cluster"][0], _b["pipeline"]["cluster"][0]))
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
      "the annotation and %.1f with the narrative: the annotator wrote only about the missed "
      "chair, which the cause explains exactly, while the narrative also reports the merged "
      "monitors, the over-segmented person and a hallucinated plant — so the same cause "
      "now explains a fraction of what is told, and explanatory power collapses. The pair of "
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
          "explains an error better than a visual group — but the gap is a point, not "
          "two and a half, and the 39-image figure should be read as the small sample it is. "
          "That the two branches together score above either alone is the same complementarity "
          "of Figure 6, measured on the explanation instead of on the coverage."
          % (_b["human"]["cluster"][0], _b["human"]["language"][0], _b["human"]["cluster"][1],
             _ab["cluster"][0], _ab["cluster"][1], _ab["language"][0], _ab["language"][1],
             _ab["both"][0], _ab["both"][1]))
        table(["The cause claims an error of type", "plausibility", "images"],
              [[k, "%.2f" % m, str(n)] for k, (m, n) in
               sorted(_ae.items(), key=lambda kv: -kv[1][1])],
              widths=[3.0, 1.5, 1.2])
        p("Read down that table and the ranking is the pipeline's own honesty about itself: a "
          "cause that blames a spurious detection is the least convincing (%.2f), because "
          "“the model invented an object here” is the hardest thing for a situation "
          "to explain, while causes about merged regions, wrong extent or a wrong class score "
          "above 8.7 — those are failures with a visible circumstance behind them. It is "
          "also worth saying that plausibility rises with the number of certified patterns on "
          "an image, from %.2f with one to %.2f with four or more: more of the image is "
          "accounted for, and explanatory power is what gains."
          % (_ae["spurious"][0], CZ["all"]["by_npatterns"][1][0],
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
      "the SliceLens paper. Their yield on the inst run differs by more than an order of "
      "magnitude.")
    table(["Hypotheses", "class × hypothesis slices evaluated", "certified", "rate"],
          [["7 canonical, class-agnostic", _cn["evaluated"], _cn["certified"],
            "%.1f%%" % _rate(_cn["certified"], _cn["evaluated"])],
           ["class-specific (Gemma)", _sp["evaluated"], _sp["certified"],
            "%.1f%%" % _rate(_sp["certified"], _sp["evaluated"])]],
          widths=[2.6, 1.9, 0.9, 0.8])
    p("The two class-specific hypotheses that certify are %s. Most of the others describe the "
      "appearance of a sub-population (“matte black car”, “square monitor”, “glass top "
      "dining table”) rather than a situation, and §6.2 already measured that putting the "
      "class inside the query makes the VLM's answer less precise (0.598 against 0.784 for the "
      "generic wording). More hypotheses of that kind would cost GPU hours for a 1%% return."
      % " and ".join("“%s”" % x for x in XH["specific_certified"]))
    _it = XH["inst_types"]
    p("What the certified situations leave uncovered is just as clear. Each certified slice has "
      "a dominant error type, and across the %d of the inst run they are %s — none is "
      "dominated by missed objects, by wrong extent or by a wrong class, although Block 1 types "
      "all three on val. The question of this section is whether GENERIC hypotheses "
      "aimed at those gaps add situations the vocabulary was missing."
      % (sum(_it.values()), ", ".join("%s %d" % (_TY[k], v) for k, v in
                                      sorted(_it.items(), key=lambda kv: -kv[1]))))

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
      "certified, %.1f%%: %.0f times the class-specific yield, %.0f%% of the canonical one. "
      "%s%s"
      % (XH["sample_systematic"], XH["pairs"], XH["sample_certified"],
         _rate(XH["sample_certified"], XH["pairs"]),
         _rate(XH["sample_certified"], XH["pairs"]) / _rate(_sp["certified"], _sp["evaluated"]),
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
    bullet("Missed objects, wrong extent and wrong class remain without a certified situation "
           "even when one is looked for on purpose. The likeliest reading is that those errors "
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
               "in choosing it." % len(_MG))
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
              [[s["hypothesis"], s["slice_id"], _MP[s["hypothesis"]][1],
                ("%.2f" % _MP[s["hypothesis"]][0]) if _MP[s["hypothesis"]][0] is not None else "—"]
               for s in _MG], widths=[2.8, 1.4, 0.9, 1.2])
        p("A plausibility well below the overall mean is the judge seeing what the visual check "
          "saw; one at or above it means the narrative of those images is compatible with the "
          "situation even where the look at the top instances was doubtful. The page shows each "
          "of these images with its score, so the reader can check either way.", italic=True)
    p("Reproduce: outputs/run_extra_hyp.sh (grounding on the sample, two rescores, "
      "compare_extra_hypotheses.py), outputs/run_extra_reground.sh (full-population "
      "re-grounding, rescore, comparison)%s; outputs in outputs/slice_quality_inst_extra{,_base,"
      "_full}/ and /data/outputs_extra, never in the July directories."
      % (" and outputs/chain_inst_extra.sh (merge into the inst run, train36 and synthetic "
         "induction, judges, gallery; the run before the merge is kept in "
         "outputs/slice_quality_inst_bak_20261006)" if _MG else ""), italic=True)

out = os.path.join(DOCS, "pipeline_implementation.docx")
doc.save(out)
print("saved", out)
