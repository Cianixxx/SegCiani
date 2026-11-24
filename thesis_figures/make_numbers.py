"""Numbers of the thesis, read from the official runs through the loaders of the technical document
(SlicelenceCode/docs/_revision_numbers.py) and written as LaTeX macros to generated/numbers.tex.
No number of the thesis is typed by hand: re-run this script after any run changes, then rebuild.

    python make_numbers.py
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "..", "SlicelenceCode", "docs")
sys.path.insert(0, DOCS)
import _revision_numbers as R  # noqa: E402

macros = {}


def put(name, value, fmt="{}"):
    macros[name] = fmt.format(value)


def pct(x, digits=1):
    return ("{:.%df}" % digits).format(100 * x) + "\\%"


# ---- VOC val, the official run (rank test, at least ten correct objects, data-driven hypotheses)
s = R.load_inst_summary()
put("VOCerrors", "{:,}".format(s["n_err"]))
put("VOCsituations", s["slices"]["language"])
put("VOCclusters", s["slices"]["cluster"])
put("VOCcoverage", pct(s["cov_union"] / s["n_err"]))
put("VOCsitImages", s["situational_images"])
put("VOCcorrect", s["n_ref"])
put("VOCobjects", "{:,}".format(s["n_err"] + s["n_ref"]))
put("VOCnmin", int(0.005 * (s["n_err"] + s["n_ref"])))       # N_min = 0.5% of the objects (pipeline.py)

put("VOCimages", "{:,}".format(s["images"] if False else 1449))   # VOC 2012 segmentation val (dataset fact)
put("VOCimgErr", "{:,}".format(s["images"]))                       # images with at least one typed error
put("VOCwithPattern", s["with_pattern"])
put("VOCcovLang", "{:,}".format(s["cov_language"]))
put("VOCcovClus", s["cov_cluster"])
put("VOCcovBoth", s["overlap"])
put("VOCcovUnion", "{:,}".format(s["cov_union"]))
TAXN = {"fn": "Missed", "fp": "Halluc", "merge": "Merge", "boundary": "Boundary", "split": "Split", "extent": "Extent",
        "confusion": "Confusion"}
for k, v in TAXN.items():
    put("VOCtax" + v, "{:,}".format(s["tax"].get(k, 0)))
put("VOCtaxExtConfPct", pct((s["tax"].get("extent", 0) + s["tax"].get("confusion", 0)) / sum(s["tax"].values()), 0))

# ---- datasets: SBD, Pascal-EA
sb = R.load_sbd()
put("VOCmiou", "{:.1f}".format(100 * sb["miou"]["VOC val"]["miou"]))
put("SBDmiou", "{:.1f}".format(100 * sb["miou"]["SBD"]["miou"]))
put("SBDimages", "{:,}".format(sb["miou"]["SBD"]["n_images"]))
put("SBDerrors", "{:,}".format(sb["run"]["n_err"]))
put("SBDcorrect", "{:,}".format(sb["run"]["n_ref"]))
put("SBDsituations", sb["run"]["slices"]["language"])
put("SBDclusters", sb["run"]["slices"]["cluster"])
put("SBDsitImages", "{:,}".format(sb["run"]["situational_images"]))
ea0 = R.load_ea()
put("EAedited", ea0["n_edited"])

# ---- validations of the official run
ov = R.load_official_validations()
pl = ov["prec"]["summary"]["language"]
put("PrecLang", "{:.2f}".format(pl["precision"]))
put("PrecLangNull", "{:.2f}".format(pl["control_rate_same_class"]))
put("PrecLangGroups", pl["groups"])
put("PrecLangAbove", pl["groups_above_chance"])
put("PrecEvals", "{:,}".format(ov["prec"]["summary"]["n_evaluations"]))
put("PrecK", ov["prec"]["summary"]["K"])
pc = ov["prec"]["summary"]["cluster_situational"]
put("PrecClus", "{:.2f}".format(pc["precision"]))
put("PrecClusNull", "{:.2f}".format(pc["control_rate_same_class"]))
m2 = ov["model2"]
put("MtwoAuc", m2["auc_sys"])
put("MtwoRecert", m2["recertified"])
put("MtwoLang", m2["official_language"])
put("MtwoClusters", m2["cluster_certified"])
put("MtwoErrors", "{:,}".format(m2["n_err"]))
put("MtwoCorrect", m2["n_ref"])
put("SplitDisc", ov["split"]["n_discovered"])
put("SplitRepl", ov["split"]["n_replicated"])
put("SplitDiscOld", ov["before"]["split"]["n_discovered"])
put("SplitReplOld", ov["before"]["split"]["n_replicated"])
for dose in ("p25", "p50"):
    ra, rb = ov["corr"][dose]["rule_a"], ov["corr"][dose]["rule_b"]
    tag = "Q" if dose == "p25" else "H"
    put("CorrA" + tag + "Inj", ra["injected"]); put("CorrA" + tag + "Rec", ra["recovered"])
    put("CorrB" + tag + "Inj", rb["injected"]); put("CorrB" + tag + "Pass", rb["passed"])
put("CorrBGate", ov["corr"]["p25"]["rule_b"]["gate"])

# ---- descriptions against human notes
cl = R.load_claims()
sa = cl["summary"]["all_narrative"]; sv = cl["summary"]["voc_narrative"]
put("FactsAll", "{:.2f}".format(sa["genuine"])); put("FactsAllCtrl", "{:.2f}".format(sa["control"])); put("FactsN", sa["n"])
put("FactsVOC", "{:.2f}".format(sv["genuine"])); put("FactsVOCCtrl", "{:.2f}".format(sv["control"])); put("FactsVOCn", sv["n"])
put("FactsTypedPred", cl["fact_types"]["prediction"]); put("FactsTypedAll", sum(cl["fact_types"].values()))   # facts the score counts
sy = cl["summary"]["synth_narrative"]
put("FactsSyn", "{:.2f}".format(sy["genuine"])); put("FactsSynCtrl", "{:.2f}".format(sy["control"])); put("FactsSynN", sy["n"])
du = cl["duplicate"]["facts"]; put("DupSame", du["same"]); put("DupUp", du["up"]); put("DupDown", du["down"])
duo = cl["duplicate"]["old"]; put("DupRubSame", duo["same"]); put("DupRubUp", duo["up"]); put("DupRubDown", duo["down"])
mm = cl["manual"]["summary"]
put("FactsHand", "{:.2f}".format(mm["tutte"]["genuine"] if isinstance(mm["tutte"], dict) else mm["tutte"]))
put("FactsHandCtrl", "{:.2f}".format(mm["tutte"]["control"] if isinstance(mm["tutte"], dict) else 0))

# ---- the slope rule against chance (shuffled error labels), on the same groundings
SV = R.load_slope_verdicts()
PS = SV["perm"]["stages"]
nulls = [PS[k]["null_old"] for k in PS]
put("SlopeNullMin", "{:.0f}\\%".format(100 * min(nulls)))
put("SlopeNullMax", "{:.0f}\\%".format(100 * max(nulls)))
put("RankBootDraws", SV["rank"]["draws"])

# ---- the minimum of ten correct objects
rm = R.load_rule_min()
low = {c["cls"]: c for c in rm["below"]}
for cls, tag in (("pottedplant", "Plant"), ("chair", "Chair"), ("boat", "Boat")):
    put("Min" + tag + "Ok", low[cls]["n_ok"])
    put("Min" + tag + "All", low[cls]["n_ok"] + low[cls]["n_err"])
    put("Min" + tag + "Cert", low[cls]["cert"])
    put("Min" + tag + "Conf", low[cls]["confirmed"])
put("MinAgreeConf", rm["ge"]["confirmed"])
put("MinAgreeCert", rm["ge"]["cert"])
put("MinAgreeDiff", rm["ge"]["cert"] - rm["ge"]["confirmed"])

# ---- SliceLens against the pipeline on the same data (paired tests)
c = R.load_slicelens_cmp()
cov = c["stats"]["coverage"]
for ds, tag in (("VOC val", "VOC"), ("SBD", "SBD")):
    put(tag + "SliceLensCov", pct(cov[ds]["rate"]["SliceLens"]))
    put(tag + "PipelineCov", pct(cov[ds]["rate"]["pipeline"]))
    put(tag + "GainPoints", "{:+.1f}".format(100 * cov[ds]["pipeline vs SliceLens"]["diff"]))
ea = c.get("ea_auc") or c["ea"]
put("EAvictims", ea["victims"])
put("EAfoundPipeline", ea["union"])
put("EAfoundSliceLens", c["ea"]["language_total"])

# ---- GESD, the benchmark published with SliceLens
g = json.load(open(os.path.join(DOCS, "gesd", "gesd_cmp.json"), encoding="utf-8"))
put("GESDexpertSlices", g["n_expert"])
put("GESDpTenSliceLens", "{:.3f}".format(g["cols"]["SliceLens, verdict only"]["p10"]))
put("GESDpTenPipeline", "{:.3f}".format(g["cols"]["pipeline"]["p10"]))
if g.get("cols_all"):
    put("GESDpTenSliceLensKept", "{:.3f}".format(g["cols_all"]["SliceLens, verdict only"]["p10"]))
    put("GESDpTenPipelineKept", "{:.3f}".format(g["cols_all"]["pipeline"]["p10"]))

GESD = R.load_gesd()
put("GESDpTenBaseline", "{:.3f}".format(GESD["baselines"]["baselines"]["C"]["p_at_k"]))
put("GESDpTenPublished", "{:.3f}".format(GESD["ghesd_p10"]))
gt = g["tests"]["pipeline vs SliceLens, verdict only"]
put("GESDdiff", "{:+.2f}".format(gt["diff"])); put("GESDp", "{:.4f}".format(gt["p_wilcoxon"]))
gta = g["tests_all"]["pipeline, errors only vs SliceLens, verdict only"]
put("GESDdiffOwn", "{:+.2f}".format(gta["diff"])); put("GESDpOwn", "{:.2f}".format(gta["p_wilcoxon"]))
gtk = g["tests_all"]["pipeline vs SliceLens, verdict only"]          # both keep the correct objects: symmetric
put("GESDdiffKept", "{:+.2f}".format(gtk["diff"])); put("GESDpKept", "{:.3f}".format(gtk["p_wilcoxon"]))
put("GESDcutPTen", "{:.2f}".format(g["why"]["slicelens_cut_to_rank_count"]["p10_mean"]))
put("GESDcutN", g["why"]["slicelens_cut_to_rank_count"]["n"])
put("GESDshuffled", "{:.3f}".format(g["why"]["SliceLens, verdict only, members shuffled"]["p10_mean"]))
for col, tag in (("SliceLens", "SL"), ("+ rank test", "Rank")):
    o = g["objects"][col]
    put("GESDname" + tag, "{:.1f}\\%".format(100 * o["rate"])); put("GESDfalse" + tag, "{:.1f}\\%".format(100 * o["false_rate"]))
put("GESDnamedSL", g["cols"]["SliceLens, verdict only"]["named"]); put("GESDnamedRank", g["cols"]["+ rank test, verdict only"]["named"])
put("GESDnamedPipe", g["cols"]["pipeline"]["named"])
mi = g["missing"]["classes"]
put("GESDnoPhraseClasses", len(mi))
put("GESDnoPhraseSlices", sum(len(v["expert_slices"]) for v in mi.values()))
put("GESDsampleSlices", sum(len(v["expert_slices"]) for v in mi.values() if v["reason"] == "sample"))
put("GESDgateSlices", sum(len(v["expert_slices"]) for v in mi.values() if v["reason"] == "gates"))

pa = GESD["pointA"]
put("GESDblockOne", "{:.1f}\\%".format(100 * pa["block1"] / pa["n"])); put("GESDnaive", "{:.1f}\\%".format(100 * pa["naive"] / pa["n"]))
l2b = GESD["full"]["level2b"]
put("GESDinVocab", sum(1 for v in l2b.values() if v.get("named_rank") is not None))
put("GESDbestSep", sum(1 for v in l2b.values() if v.get("named_rank") == 1))
put("GESDlevelN", len(l2b))
put("GESDsliceUnits", "{:,}".format(pa["n"]))
# Block 1 against the naive typing on the expert slices, with its uncertainty: image bootstrap of the difference of
# agreement (all units of an image together) and the mean over slices weighted equally (block1_vs_experts.json)
_bv = json.load(open(os.path.join(DOCS, "gesd", "block1_vs_experts.json"), encoding="utf-8"))
_bu = [x for x in _bv["rows"] if x["slice"] != "NA"]
_ex = {s: set(v["expected"]) for s, v in _bv["per_slice"].items()}
_img = {}
for x in _bu:
    _a = _img.setdefault(x["image"], [0, 0, 0])
    _a[0] += 1; _a[1] += x["type"] in _ex[x["slice"]]; _a[2] += bool(set(x["naive"].split("/")) & _ex[x["slice"]])
_A = __import__("numpy").array(list(_img.values()), float)
_rng = __import__("numpy").random.default_rng(20261102)
_bd = []
for _ in range(20000):
    _s2 = _A[_rng.integers(0, len(_A), len(_A))].sum(0)
    _bd.append(100 * (_s2[1] - _s2[2]) / _s2[0])
_lo, _hi = __import__("numpy").percentile(_bd, [2.5, 97.5])
put("GESDbOneDiff", "{:+.1f}".format(100 * (_A[:, 1].sum() - _A[:, 2].sum()) / _A[:, 0].sum()))
put("GESDbOneLo", "{:+.1f}".format(_lo)); put("GESDbOneHi", "{:+.1f}".format(_hi)); put("GESDbOneImgs", len(_A))
_sl = {}
for x in _bu:
    _a = _sl.setdefault(x["slice"], [0, 0, 0])
    _a[0] += 1; _a[1] += x["type"] in _ex[x["slice"]]; _a[2] += bool(set(x["naive"].split("/")) & _ex[x["slice"]])
put("GESDbOneSlice", pct(sum(v[1] / v[0] for v in _sl.values()) / len(_sl)))
put("GESDnaiveSlice", pct(sum(v[2] / v[0] for v in _sl.values()) / len(_sl)))
# where the Precision@10 gap comes from, slice by slice (gesd_cmp.json), and the slice that earns each score
# (gesd_best_slices.py)
_ps, _pp = g["cols"]["SliceLens, verdict only"]["per_slice"], g["cols"]["pipeline"]["per_slice"]
_both, _none = [k for k in _ps if _pp[k] > 0], [k for k in _ps if _pp[k] == 0]
_mean = lambda d, ks: sum(d[k] for k in ks) / len(ks)
put("GESDbothN", len(_both)); put("GESDnoneN", len(_none))
put("GESDbothSL", "{:.2f}".format(_mean(_ps, _both))); put("GESDbothPipe", "{:.2f}".format(_mean(_pp, _both)))
put("GESDnoneSL", "{:.2f}".format(_mean(_ps, _none)))
put("GESDgapAll", "{:.2f}".format(_mean(_ps, _ps) - _mean(_pp, _pp)))
put("GESDgapNone", "{:.2f}".format(sum(_ps[k] - _pp[k] for k in _none) / len(_ps)))
_zt = [k for v in mi.values() if v["reason"] == "sample" for k in v["expert_slices"] if _pp.get(k) == 0]
_zg = [k for v in mi.values() if v["reason"] == "gates" for k in v["expert_slices"] if _pp.get(k) == 0]
put("GESDnoneTest", len(_zt)); put("GESDnoneGate", len(_zg)); put("GESDnoneVocab", len(_none) - len(_zt) - len(_zg))
put("GESDgateGain", "{:.2f}".format(sum(v["p10_rank_verdict"][k] for v in mi.values() for k in v["expert_slices"]
                                         if k in _zg) / len(_ps)))
put("GESDclusRescue",sum(len(v["expert_slices"]) for v in mi.values()) - len(_zt) - len(_zg))
put("GESDgateWithout", "{:.2f}".format(_mean({k: v["p10_rank_verdict"][k] for v in mi.values() for k in v["expert_slices"]},
                                             _zg)))
GB_ = json.load(open(os.path.join(DOCS, "gesd", "gesd_best_slices.json"), encoding="utf-8"))
put("GESDwinNamedSL", g["cols"]["SliceLens, verdict only"]["best_named"])
put("GESDwinNamedPipe", g["cols"]["pipeline"]["best_named"])
put("GESDsingleClasses", sum(len(v["expert_slices"]) == 1 for v in GB_["per_class"].values()))
put("GESDwinPaperSL", "{:.1f}".format(GB_["expert"]["Paper mis-segmented as book"]["SliceLens, verdict only"]["p10"]))
put("GESDwinWatchSize", GB_["expert"]["Watch missed when segmenting clock"]["SliceLens, verdict only"]["size"])
# errors inside a certified group on GESD, as for VOC val and SBD (gesd_coverage.py), and the phrases behind it
# (gesd_inflation.py)
gcv = json.load(open(os.path.join(DOCS, "gesd", "gesd_coverage.json"), encoding="utf-8"))
put("GESDcovSL", pct(gcv["rate"]["SliceLens"])); put("GESDcovRank", pct(gcv["rate"]["+ rank test"]))
put("GESDcovPipe", pct(gcv["rate"]["pipeline"]))
put("GESDcovErrors", "{:,}".format(gcv["n_err"])); put("GESDcovImages", gcv["n_img"])
_cr = gcv["+ rank test vs SliceLens"]
put("GESDcovOnlySL", _cr["only_a"]); put("GESDcovOnlyRank", _cr["only_b"])
put("GESDcovRankDiff", "{:+.1f} [{:+.1f}, {:+.1f}]".format(100 * _cr["diff"], 100 * _cr["ci"][0], 100 * _cr["ci"][1]))
_cbc = gcv["by_class"]
put("GESDvaseSL", pct(_cbc["vase"]["SliceLens"])); put("GESDvaseRank", pct(_cbc["vase"]["+ rank test"]))
put("GESDvasePhrSL", _cbc["vase"]["phrases_slope"]); put("GESDvasePhrRank", _cbc["vase"]["phrases_auc"])
_slc = json.load(open(os.path.join(DOCS, "gesd", "deck_examples", "slice_counts.json"), encoding="utf-8"))
_bk = GB_["per_class"]["book"]["expert_slices"]
put("GESDbookPaper", pct(max(_slc[s] for s in _bk) / sum(_slc[s] for s in _bk), 0))
INFL = json.load(open(os.path.join(DOCS, "gesd", "gesd_inflation.json"), encoding="utf-8"))
_unc = [r for r in INFL if not r["rank_certified"]]
put("GESDinfN", len(INFL)); put("GESDinfPhrases", len({r["q"] for r in INFL}))
put("GESDslSlices", g["cols"]["SliceLens, verdict only"]["slices"]); put("GESDinfConfirmed", len(INFL) - len(_unc)); put("GESDinfUnconfirmed", len(_unc))
put("GESDinfClose", sum(r["yes_err"] - r["yes_ok"] < 0.15 for r in _unc))
put("GESDinfAgainst", sum(r["yes_ok"] > r["yes_err"] for r in _unc))
put("GESDinfUncAUC", "{:.2f}".format(sorted(r["auc"] for r in _unc)[len(_unc) // 2]))
put("GESDinfConfAUC", "{:.2f}".format(sorted(r["auc"] for r in INFL if r["rank_certified"])[(len(INFL) - len(_unc)) // 2]))
_iv = {(r["cls"], r["q"]): r for r in INFL}
for tag, key in (("Vase", ("vase", "a small object")), ("Book", ("book", "book from side view")),
                 ("Umb", ("umbrella", "an object that blends into its background"))):
    put("GESDinf" + tag + "Err", pct(_iv[key]["yes_err"], 0)); put("GESDinf" + tag + "Ok", pct(_iv[key]["yes_ok"], 0))
    put("GESDinf" + tag + "AUC", "{:.2f}".format(_iv[key]["auc"]))

# ---- SBD against VOC: situations certified on both
put("SBDoverlap", len(set(sb["lang_voc"]) & set(sb["lang_sbd"])))

# ---- SBD, official since 28 October: the hypotheses the slope rule had kept from the whole class, read on every object,
# the checks of the new run, and the run before them (kept as slice_quality_sbd_rank43)
SBDD = R.out_path("slice_quality_sbd")
pend = json.load(open(os.path.join(SBDD, "dd_auc_full.json"), encoding="utf-8"))
put("SBDpendN", len(pend)); put("SBDpendSig", sum(1 for r in pend if r["auc_sys"]))
put("SBDpendSlope", sum(1 for r in pend if r["slope_sys"]))
put("SBDpendCert", sum(1 for x in json.load(open(os.path.join(SBDD, "slices_index.json"), encoding="utf-8"))["slices"]
                       if x.get("in_descriptions") and "_pend_" in x["slice_id"]))
sp_ = json.load(open(os.path.join(SBDD, "eval_precision.json"), encoding="utf-8"))["summary"]
for k, tag in (("language", "Lang"), ("cluster_situational", "Clus")):
    put("SBDprec" + tag, "{:.2f}".format(sp_[k]["precision"])); put("SBDprec" + tag + "Null", "{:.2f}".format(sp_[k]["control_rate_same_class"]))
    put("SBDprec" + tag + "Above", sp_[k]["groups_above_chance"]); put("SBDprec" + tag + "Groups", sp_[k]["groups"])
put("SBDprecEvals", "{:,}".format(sp_["n_evaluations"]))
jr_ = [r for r in json.load(open(os.path.join(SBDD, "judge_cause_val.json"), encoding="utf-8"))["judgements"] if "score" in r]
g_ = {r["image"]: r for r in jr_ if r["pairing"] == "genuine"}
c_ = [r["score"] for r in jr_ if r["pairing"] == "control"]
put("SBDplaus", "{:.2f}".format(sum(r["score"] for r in g_.values()) / len(g_))); put("SBDplausN", "{:,}".format(len(g_)))
put("SBDplausCtrl", "{:.2f}".format(sum(c_) / len(c_))); put("SBDplausCtrlN", len(c_))
put("SBDplausHigh", "{:.0f}\\%".format(100 * sum(1 for r in g_.values() if r["score"] >= 7) / len(g_)))
bm_ = R._branch_means(g_)
for b, tag in (("language", "Lang"), ("cluster", "Clus"), ("both", "Both")):
    put("SBDplaus" + tag, "{:.2f}".format(bm_[b][0])); put("SBDplaus" + tag + "N", "{:,}".format(bm_[b][1]))
old_ = json.load(open(R.out_path("slice_quality_sbd_rank43", "sbd_summary.json"), encoding="utf-8"))["run"]
put("SBDoldSituations", old_["slices"]["language"]); put("SBDoldClusters", old_["slices"]["cluster"])
put("SBDoldCoverage", pct(old_["cov_union"] / old_["n_err"])); put("SBDoldSitImages", "{:,}".format(old_["situational_images"]))

# ---- corruption: objects that Block 1 types as missed after the deletion
put("CorrAQMatched", ov["corr"]["p25"]["rule_a"]["matched_as_fn"]); put("CorrAHMatched", ov["corr"]["p50"]["rule_a"]["matched_as_fn"])

# ---- Pascal-EA: who finds the victims, and naming object by object
ea_stats = c["stats"]["ea"]
put("EAfoundRank", ea_stats["found"]["+ rank test"])
put("EAlangOnly", ea["language"]); put("EAclusOnly", ea["cluster"]); put("EAboth", ea["both"])
put("EApMcNemar", "{:.4f}".format(ea_stats["pipeline vs SliceLens"]["p_mcnemar"]))
put("EApRank", "{:.2f}".format(ea_stats["+ rank test vs SliceLens"]["p_mcnemar"]))
nm = R.load_ea_naming()["all"]
put("NameVictimTop", nm["victim"]["top1"]); put("NameVictimN", nm["victim"]["n"])
put("NameVictimPct", "{:.0f}\\%".format(100 * nm["victim"]["top1"] / nm["victim"]["n"]))
put("NameKeptTop", nm["edited, not lost"]["top1"]); put("NameKeptN", nm["edited, not lost"]["n"])
put("NameKeptPct", "{:.1f}\\%".format(100 * nm["edited, not lost"]["top1"] / nm["edited, not lost"]["n"]))
dog = R.load_ea()["planted"]["slices_with_victims"][0]
put("EAdogMembers", dog["members"]); put("EAdogEdited", dog["from_edited"]); put("EAdogVictims", dog["victims"])
put("EAsituationsBefore", ea0["cert_inst"]); put("EAsituationsAfter", ea0["cert_ea"])

# ---- which objects the VLM read, and how the situations meet the final objects (rank_population_link.py):
# the groundings are on the first typing. Until 2 November the official run joined them to the final errors by id
# (the audit of that run, kept with it in slice_quality_inst_idjoin); since 3 November by geometry (rescore_geo.log),
# and the verdict with the final labels is the audit of the geometric run
LK0 = json.load(open(R.out_path("slice_quality_inst_idjoin", "rank_population_link.json"), encoding="utf-8"))
lki, lkm = LK0["ids"], LK0["members"]
put("LinkIdsAll", "{:,}".format(sum(lki.values()))); put("LinkIdsSame", "{:,}".format(lki["same id, same object"]))
put("LinkIdsOther", lki["same id, another object"]); put("LinkIdsAbsent", "{:,}".format(lki["absent from the final typing"]))
put("LinkMemb", "{:,}".format(lkm["official"])); put("LinkMis", lkm["mispaired"]); put("LinkMisCls", lkm["cross_class"])
put("LinkMisPct", pct(lkm["mispaired"] / lkm["official"]))
_gj = re.search(r"geometric join \(.*?\): (\d+) of (\d+) substrate objects",
                open(R.out_path("slice_quality_inst", "rescore_geo.log"), encoding="utf-8").read())
put("GeoLinked", "{:,}".format(int(_gj.group(1)))); put("GeoSubstrate", "{:,}".format(int(_gj.group(2))))
put("MinBikeAll", int(_gj.group(2)) - sum(lki.values()))     # bicycle: in the first typing, never read on the whole class
LK = json.load(open(R.out_path("slice_quality_inst", "rank_population_link.json"), encoding="utf-8"))
_lv = LK["verdict"]
_lsh = [v for v in _lv if v["shipped"]]
put("LinkHypDd", sum(v["official"] and v["family"] == "dd" for v in _lv))
assert sum(v["official"] for v in _lv) - sum(v["official"] and v["family"] == "dd" for v in _lv) == rm["ge"]["cert"]
put("LinkHypCert", sum(v["official"] for v in _lv)); put("LinkHypKept", sum(v["official"] and bool(v.get("certified_final")) for v in _lv))
put("LinkSitKept", sum(bool(v.get("certified_final")) for v in _lsh)); put("LinkSitLost", sum(not v.get("certified_final") for v in _lsh))
_lq = [v["q_final"] for v in _lsh if not v.get("certified_final") and v.get("q_final") is not None]
put("LinkLostQmin", "{:.2f}".format(min(_lq))); put("LinkLostQmax", "{:.2f}".format(max(_lq)))
# coverage counting only the situations that stay significant on the final labels (the official run is not changed)
import pickle
sys.path.insert(0, os.path.join(DOCS, ".."))
_res = pickle.load(open(R.out_path("slice_quality_inst", "result_tagged.pkl"), "rb"))
_mem = {s_.slice_id: {m_.instance_id for m_ in s_.members} for s_, _q in _res["result"].scored}
_ids = [e_["slice_id"] for e_ in json.load(open(R.out_path("slice_quality_inst", "slices_index.json"), encoding="utf-8"))["slices"]
        if e_.get("in_descriptions")]
_lost = {(v_["cls"], v_["q"]) for v_ in _lsh if not v_.get("certified_final")}
_gone = {x_["slice_id"] for x_ in LK["situations"] if (x_["cls"], x_["q"]) in _lost}
assert len(_gone) == sum(not v_.get("certified_final") for v_ in _lsh)
_kept = set().union(*[_mem[k_] for k_ in _ids if k_ not in _gone])
put("LinkCovKeptN", "{:,}".format(len(_kept))); put("LinkCovKept", pct(len(_kept) / len(_res["errors"])))

# ---- what the certified slices cost in segmentation quality
si = R.load_slice_image_iou()
put("IoUinSlices", "{:.1f}".format(100 * si["refs"]["images in at least one certified slice"]["miou_pooled"]))
put("IoUnoSlice", "{:.1f}".format(100 * si["refs"]["images with an error, in no certified slice"]["miou_pooled"]))
put("IoUworse", si["language"]["worse"]); put("IoUlangN", si["language"]["n"])
put("IoUdrop", "{:.1f}".format(100 * si["language"]["drop_mean"]))
put("IoUclsIn", "{:.1f}".format(100 * si["language"]["cls_in"])); put("IoUclsOut", "{:.1f}".format(100 * si["language"]["cls_out"]))
_rr = [r for r in si["rows"] if r["cls_iou_in"] is not None]          # the rows of fig:slice-iou, and its dashed line
put("IoUslices", len(_rr))
put("IoUbelowMean", sum(r["cls_iou_in"] < si["refs"]["all images"]["miou_pooled"] for r in _rr))

# ---- plausibility of the blamed situation, official run
cz = R.load_cause(val_only=True)["all"]         # VOC val only: the pool also holds 18 annotated images of train
put("PlausScore", "{:.2f}".format(cz["pipeline"]["score"])); put("PlausCtrl", "{:.2f}".format(cz["pipeline"]["control"]))
put("PlausN", cz["pipeline"]["n"])
for b, tag in (("language", "Lang"), ("cluster", "Clus"), ("both", "Both")):
    put("Plaus" + tag, "{:.2f}".format(cz["by_branch"]["pipeline"][b][0])); put("Plaus" + tag + "N", cz["by_branch"]["pipeline"][b][1])
put("PlausSpurious", "{:.2f}".format(cz["by_etype"]["spurious"][0]))
put("PlausMerge", "{:.2f}".format(cz["by_etype"]["merged"][0]))
assert cz["by_etype"]["merged"][0] == max(v[0] for v in cz["by_etype"].values())      # "the most convincing" in the text

# ---- tables written straight from the runs
GEN = os.path.join(HERE, "generated")
os.makedirs(GEN, exist_ok=True)


def tex_table(name, rows):
    with open(os.path.join(GEN, name), "w", encoding="utf-8") as f:
        f.write("% written by make_numbers.py -- do not edit\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")


p0 = lambda x: "{:.0f}\\%".format(100 * x)
p1 = lambda x: "{:.1f}\\%".format(100 * x)
# verdicts on the same groundings, and with shuffled labels
KS = SV["rank"]["stages"]
tex_table("tab_null.tex", [[lab, PS[k]["n"], PS[k]["old"], KS[k]["new"], p1(PS[k]["null_old"]), p1(KS[k]["null_new"])]
                           for k, lab in ((("VOC val", "sample"), "VOC val, sample"), (("VOC val", "full"), "VOC val, whole class"),
                                          (("SBD", "sample"), "SBD, sample"), (("SBD", "full"), "SBD, whole class"))])
# effects of known strength
cell, rcell = c["sim"]["cell"], c["rules"]["cell"]
rows = []
for b, lab in ((0.0, "none"), (0.25, "weak"), (0.5, "medium"), (1.0, "strong")):
    for n in (120, 400, 1500, 5000):
        if (b, n) in cell and (b, n) in rcell:
            r = rcell[(b, n)]
            rows.append([lab if n == 120 else "", "{:.2f}".format(cell[(b, n)]["auc_mean"]), "{:,}".format(n), p0(r["v1"]),
                         p0(r["v4-slope W10"]), p0(r["v4-rho-ms"]), p0(r["AUC"])])
tex_table("tab_sim.tex", rows)
put("SimDraws", c["rules"]["R"]); put("SimPairs", c["sim"]["pairs"])
put("SimFAslope", p0(rcell[(0.0, 400)]["v1"])); put("SimFArank", p0(rcell[(0.0, 400)]["AUC"]))
put("SimWeakSlope", p0(rcell[(0.25, 1500)]["v1"])); put("SimWeakRank", p0(rcell[(0.25, 1500)]["AUC"]))
put("SimWeakSlopeBig", p0(rcell[(0.25, 5000)]["v1"])); put("SimWeakRankBig", p0(rcell[(0.25, 5000)]["AUC"]))
# the same simulation at two shares of wrong objects, GESD's and that of the VOC/SBD groundings (slope_baserate_sim.py),
# and the share on the three datasets with the hypotheses each verdict sends on from the 120-object sample
BRS = R.out_path("slope_baserate_sim.json")
if os.path.exists(BRS):
    br = json.load(open(BRS, encoding="utf-8"))
    rg, rv = br["rates"]["gesd"], br["rates"]["vocsbd"]
    bc = {(r["b"], r["n"], round(r["rate"], 4)): r for r in br["table"]}
    cellbr = lambda b, n, rate: bc[(b, n, round(rate, 4))]
    put("BRshareGESD", pct(rg)); put("BRshareVS", pct(rv))
    put("BRslopeRatio", "{:.0f}".format(rg * (1 - rg) / (rv * (1 - rv))))     # the slope of one effect scales as p(1 - p)
    for tag, b, n in (("FA", 0.0, 400), ("FABig", 0.0, 1500), ("Weak", 0.25, 1500), ("WeakBig", 0.25, 5000)):
        put("BR" + tag + "SlopeG", p0(cellbr(b, n, rg)["slope"])); put("BR" + tag + "SlopeV", p0(cellbr(b, n, rv)["slope"]))
        put("BR" + tag + "RankG", p0(cellbr(b, n, rg)["auc"])); put("BR" + tag + "RankV", p0(cellbr(b, n, rv)["auc"]))
    tex_table("tab_baserate.tex", [[lab if n == 120 else "", "{:,}".format(n),
                                    p0(cellbr(b, n, rg)["slope"]), p0(cellbr(b, n, rv)["slope"]),
                                    p0(cellbr(b, n, rg)["auc"]), p0(cellbr(b, n, rv)["auc"])]
                                   for b, lab in ((0.0, "none"), (0.25, "weak"), (0.5, "medium"))
                                   for n in (120, 400, 1500, 5000) if (b, n, round(rg, 4)) in bc])
    put("GESDshareWrong", pct(br["gesd_wrong"] / (br["gesd_wrong"] + br["gesd_correct"])))
    put("GESDrefN", "{:,}".format(br["gesd_correct"]))           # Block 1's correct objects on GESD (substrate.pkl)
put("VOCshareWrong", pct(s["n_err"] / (s["n_err"] + s["n_ref"])))
put("SBDshareWrong", pct(sb["run"]["n_err"] / (sb["run"]["n_err"] + sb["run"]["n_ref"])) if "n_ref" in sb["run"] else "---")
for tag, ds in (("VOC", "VOC val"), ("SBD", "SBD")):
    put(tag + "sampleN", PS[(ds, "sample")]["n"]); put(tag + "sampleSlope", PS[(ds, "sample")]["old"])
    put(tag + "sampleRank", KS[(ds, "sample")]["new"])
_cc = json.load(open(os.path.join(DOCS, "gesd", "pointB_cc.json"), encoding="utf-8"))["verdicts"]
put("GESDsampleN", len(_cc)); put("GESDsampleSlope", sum(bool(v["slope_sys"]) for v in _cc))
put("GESDsampleRank", sum(bool(v["auc_sys"]) for v in _cc))
put("GESDsampleRatio", "{:.0f}".format(sum(bool(v["slope_sys"]) for v in _cc) / sum(bool(v["auc_sys"]) for v in _cc)))
# SliceLens, + rank test, pipeline
st = c["stats"]
ci = lambda d: "{:+.1f} [{:+.1f}, {:+.1f}]".format(100 * d["diff"], 100 * d["ci"][0], 100 * d["ci"][1])
fa, ru = c["facts"], c["rubric"]
fa2, ru2 = c.get("facts_auc") or {}, c.get("rubric_auc") or {}
jf, jr = st["judges"]["facts vs SliceLens"], st["judges"]["rubric vs SliceLens"]
rows = [["Errors in a significant slice, VOC val"] + [p1(cov["VOC val"]["rate"][k]) for k in ("SliceLens", "+ rank test", "pipeline")]
        + [ci(cov["VOC val"]["pipeline vs SliceLens"]) + " points"],
        ["Errors in a significant slice, SBD"] + [p1(cov["SBD"]["rate"][k]) for k in ("SliceLens", "+ rank test", "pipeline")]
        + [ci(cov["SBD"]["pipeline vs SliceLens"]) + " points"],
        ["Errors in a significant slice, GESD"] + [p1(gcv["rate"][k]) for k in ("SliceLens", "+ rank test", "pipeline")]
        + [ci(gcv["pipeline vs SliceLens"]) + " points"],
        ["False alarms, no association (400 objects)", p0(st["sim"]["0.0_400"]["slope"]), p0(st["sim"]["0.0_400"]["auc"]),
         p0(st["sim"]["0.0_400"]["auc"]), "---"],
        ["Weak association found (1,500 objects)", p0(st["sim"]["0.25_1500"]["slope"]), p0(st["sim"]["0.25_1500"]["auc"]),
         p0(st["sim"]["0.25_1500"]["auc"]), "---"],
        ["Pascal-EA victims in a significant slice"] + ["{} of {}".format(ea_stats["found"][k], ea_stats["victims"])
                                                       for k in ("SliceLens", "+ rank test", "pipeline")]
        + ["{:+d} victims".format(ea_stats["found"]["pipeline"] - ea_stats["found"]["SliceLens"])],
        ["Facts score against human notes", "{:.2f}".format(fa["sl"]), "{:.2f}".format(fa2.get("sl", float("nan"))) if fa2 else "---",
         "{:.2f}".format(fa["ours"]), "{:+.2f} [{:+.2f}, {:+.2f}]".format(jf["diff"], *st["judges"]["facts vs SliceLens"].get("ci", [float("nan")] * 2))],
        ["Rubric against human notes", "{:.2f}".format(ru["sl"]), "{:.2f}".format(ru2.get("sl", float("nan"))) if ru2 else "---",
         "{:.2f}".format(ru["ours"]), "{:+.2f} [{:+.2f}, {:+.2f}]".format(jr["diff"], *st["judges"]["rubric vs SliceLens"].get("ci", [float("nan")] * 2))],
        ["GESD Precision@10", "{:.3f}".format(g["cols"]["SliceLens, verdict only"]["p10"]),
         "{:.3f}".format(g["cols"]["+ rank test, verdict only"]["p10"]), "{:.3f}".format(g["cols"]["pipeline"]["p10"]),
         "{:+.2f} [{:+.2f}, {:+.2f}]".format(gt["diff"], *gt["ci"])]]
tex_table("tab_cmp.tex", rows)
put("FactsCmpN", jf["n"]); put("FactsCmpWins", jf["wins"]); put("FactsCmpLosses", jf["losses"])
put("FactsCmpOurs", "{:.2f}".format(fa["ours"])); put("FactsCmpSL", "{:.2f}".format(fa["sl"]))
put("RankGainVOC", ci(cov["VOC val"]["+ rank test vs SliceLens"]))
put("ClusGainVOC", ci(cov["VOC val"]["pipeline vs + rank test"]))
put("RankGainSBD", ci(cov["SBD"]["+ rank test vs SliceLens"]))
put("ClusGainSBD", ci(cov["SBD"]["pipeline vs + rank test"]))
# VOC against SBD
sv_ = R.load_inst_summary()
tex_table("tab_sbd.tex", [
    ["Images", "{:,}".format(1449), "{:,}".format(sb["miou"]["SBD"]["n_images"])],
    ["Mean IoU of the segmenter", "{:.1f}".format(100 * sb["miou"]["VOC val"]["miou"]), "{:.1f}".format(100 * sb["miou"]["SBD"]["miou"])],
    ["Wrong objects (per image)", "{:,} ({:.2f})".format(sv_["n_err"], sv_["n_err"] / 1449),
     "{:,} ({:.2f})".format(sb["run"]["n_err"], sb["run"]["n_err"] / sb["miou"]["SBD"]["n_images"])],
    ["Significant situations / clusters", "{} / {}".format(sv_["slices"]["language"], sv_["slices"]["cluster"]),
     "{} / {}".format(sb["run"]["slices"]["language"], sb["run"]["slices"]["cluster"])],
    ["Errors in a significant slice", p1(sv_["cov_union"] / sv_["n_err"]), p1(sb["run"]["cov_union"] / sb["run"]["n_err"])],
    ["\\quad in a situation / in a cluster", "{} / {}".format(p1(sv_["cov_language"] / sv_["n_err"]), p1(sv_["cov_cluster"] / sv_["n_err"])),
     "{} / {}".format(p1(sb["run"]["cov_language"] / sb["run"]["n_err"]), p1(sb["run"]["cov_cluster"] / sb["run"]["n_err"]))],
    ["Images with a situational description", "{:,}".format(sv_["situational_images"]), "{:,}".format(sb["run"]["situational_images"])]])
# GESD
gc, gw, ga = g["cols"], g["why"], g.get("cols_all", {})
tex_table("tab_gesd.tex", [[lab, gc[k]["slices"], gw[k]["classes_with_language_slice"] if k in gw else "---", "{:.3f}".format(gc[k]["p10"]),
                            "{:.3f}".format(ga[k]["p10"]) if k in ga else "---", gc[k]["named"]]
                           for k, lab in (("SliceLens, verdict only", "SliceLens (slope rule)"),
                                          ("+ rank test, verdict only", "SliceLens + rank test"),
                                          ("pipeline", "SegCiani"))]
          + [["Class $\\times$ error type, no VLM", "---", "---", "{:.3f}".format(GESD["baselines"]["baselines"]["C"]["p_at_k"]), "---", "---"],
             ["GH-ESD as published", "---", "---", "{:.3f}".format(GESD["ghesd_p10"]), "---", "---"]])
# the slice that earns the score of five expert slices (gesd_best_slices.py)
TYPE_GROUP = {"fp": "cluster of hallucinated objects", "fn": "cluster of missed objects", "merge": "cluster of merges"}


def _earner(b):
    if b["p10"] == 0:
        return "none"
    if b["branch"] == "cluster":
        return TYPE_GROUP[b["name"].split("_")[1]]
    return "``{}''".format(b["name"]) + (" ({} objects)".format(b["size"]) if b["size"] < 10 else "")


tex_table("tab_gesd_win.tex", [[s, _earner(GB_["expert"][s]["SliceLens, verdict only"]),
                                "{:.1f}".format(GB_["expert"][s]["SliceLens, verdict only"]["p10"]),
                                _earner(GB_["expert"][s]["pipeline"]), "{:.1f}".format(GB_["expert"][s]["pipeline"]["p10"])]
                               for s in ("Vase without flowers missed due to context prior", "Paper mis-segmented as book",
                                         "Bed over-segmented together with person",
                                         "Airplane boundary expanded under occlusion",
                                         "Watch missed when segmenting clock")])
# the phrases SliceLens certifies on GESD: the six largest groups the rank test does not confirm, those that hold more
# for the correct objects, and the three confirmed phrases with the highest AUC (gesd_inflation.py)
_big = sorted(_unc, key=lambda r: -r["yes_err"])[:6]
_neg = sorted((r for r in _unc if r["yes_ok"] > r["yes_err"]), key=lambda r: r["yes_err"] - r["yes_ok"])
_good = sorted((r for r in INFL if r["rank_certified"]), key=lambda r: -r["auc"])[:3]
CLS_EN = {"teddy_bear": "teddy bear"}
tex_table("tab_gesd_infl.tex", [[CLS_EN.get(r["cls"], r["cls"]), "``{}''".format(r["q"]), p0(r["yes_err"]), p0(r["yes_ok"]),
                                 "{:.2f}".format(r["auc"]), "yes" if r["rank_certified"] else "no"]
                                for r in _big + _neg + _good])
put("GESDinfBigN", len(_big)); put("GESDinfNegN", len(_neg)); put("GESDinfGoodN", len(_good))
# the 21 expert slices one by one: the types Block 1 gives their objects and the phrase that names their situation
# (the table of the advisor deck, "GESD: dove siamo uguali agli esperti"); bands of agreement on the type: more than
# half of the objects, from 30% to half, less
TYPE_EN = {"fn": "missed", "fp": "hallucinated", "merge": "merge", "boundary": "boundary", "extent": "extent",
           "split": "split", "confusion": "wrong class"}
_xs, _l2 = GESD["pointA"]["per_slice"], GESD["full"]["level2b"]
_band = lambda p: 0 if p["agree"] >= 0.5 * p["n"] else (1 if p["agree"] >= 0.3 * p["n"] else 2)
_rank = lambda s: (_l2.get(s) or {}).get("named_rank")            # no entry: no phrase read on the whole class


def _said(p):
    tops = sorted(((t, c) for t, c in p["types"].items() if t != "correct"), key=lambda kv: -kv[1])[:2]
    return ", ".join("{} {}".format(TYPE_EN[t], p0(c / p["n"])) for t, c in tops if c / p["n"] >= 0.15)


_xrows, _prev = [], 0
for _s in sorted(_xs, key=lambda k: (_band(_xs[k]), {1: 0, None: 2}.get(_rank(k), 1), -_xs[k]["n"])):
    _p, _rk = _xs[_s], _rank(_s)
    _xrows.append([("\\midrule " if _band(_p) != _prev else "") + _s, _p["n"], _said(_p), p0(_p["agree"] / _p["n"]),
                   "``{}''".format(_l2[_s]["named_best"][0].replace("_", " ")) if _rk else "none", _rk or "---"])
    _prev = _band(_p)
tex_table("tab_gesd_types.tex", _xrows)
for _b, _tag in ((0, "Ok"), (1, "Mid"), (2, "No")):
    put("GESDtype" + _tag, sum(_band(v) == _b for v in _xs.values()))
put("GESDnameElse", sum(1 for k in _xs if _rank(k) not in (None, 1)))
put("GESDnameNone", sum(1 for k in _xs if _rank(k) is None))
_bk = _xs["Open book missed due to shape variation"]
put("GESDbookN", _bk["n"]); put("GESDbookMissed", _bk["types"]["fn"]); put("GESDbookBoundary", _bk["types"]["boundary"])

# ---- the preliminary experiment: VLMs describing an error in free text (May 2026)
FX = json.load(open(os.path.join(DOCS, "first_experiment", "first_experiment.json"), encoding="utf-8"))
fxf, fxw, fxj = FX["format"], FX["wording"], FX["judge_rerun"]
put("FXimages", FX["n_images"]); put("FXjoin", "{:.2f}".format(fxf["mean_a"])); put("FXsep", "{:.2f}".format(fxf["mean_b"]))
put("FXformatR", "{:.2f}".format(fxf["pearson"])); put("FXwordR", "{:.2f}".format(fxw["pearson"]))
put("FXjudgeR", "{:.2f}".format(fxj["pearson_mean"])); put("FXjudgeSD", "{:.2f}".format(fxj["sd_mean"]))
put("FXwordA", "{:.2f}".format(fxw["mean_a"])); put("FXwordB", "{:.2f}".format(fxw["mean_b"]))
bm = max(FX["models"], key=lambda m: FX["models"][m]["JOIN-D1"]["mean"])
put("FXbest", "{:.2f}".format(FX["models"][bm]["JOIN-D1"]["mean"]))
put("FXlenRhoA", "{:.2f}".format(FX["length"]["spearman_minimax"])); put("FXlenRhoB", "{:.2f}".format(FX["length"]["spearman_gemini"]))

# ---- using the diagnosis in training (SEGIMPROVER)
SG = R.load_segimprover()
f4 = lambda x: "{:+.4f}".format(x)
a1 = SG["arm1"]
put("ArmOneGroup", f4(a1["group"]["repair"] - a1["group"]["control"])); put("ArmOneRest", f4(a1["rest"]["repair"] - a1["rest"]["control"]))
put("ArmOneAll", f4(a1["all"]["repair"] - a1["all"]["control"]))
aA = SG["armA"]
put("ArmAdiff", f4(aA["diff"]["mean"])); put("ArmAt", "{:.1f}".format(aA["diff"]["t"])); put("ArmAcrit", "{:.2f}".format(aA["diff"]["crit"]))
put("ArmAsd", "{:.4f}".format(aA["control"][1]))
a2 = SG["arm2"]
for k, tag in (("total", "Total"), ("frame", "Frame"), ("cut", "Cut")):
    put("ArmTwo" + tag, f4(a2[k]["mean"])); put("ArmTwo" + tag + "T", "{:.1f}".format(a2[k]["t"]))
put("ArmTwoCrit", "{:.2f}".format(a2["total"]["crit"])); put("ArmTwoNeeded", a2["needed"])
put("ArmTwoFirstT", "{:.1f}".format(a2["first3"]["total"]["t"] if isinstance(a2["first3"]["total"], dict) else a2["first3"]["total"]))
put("ArmTwoSdCtrl", "{:.4f}".format(a2["control"][1])); put("ArmTwoSdRep", "{:.4f}".format(a2["repair"][1]))
rd = SG["rediag"]
put("RediagEdge", "{:.1f}\\%".format(100 * rd["edge"])); put("RediagInner", "{:.1f}\\%".format(100 * rd["inner"]))
put("RediagNoise", "{:.1f}\\%".format(100 * rd["noise"]))
ER = R.load_ea_repair()
pr = ER["paired"]
put("RepRecall", "{:+.2f}".format(pr["repair-control"]["recall.B_victims_edited"]["mean"]))
put("RepRecallT", "{:.0f}".format(pr["repair-control"]["recall.B_victims_edited"]["t"]))
put("DiagRecall", "{:+.2f}".format(pr["diagnosed-control"]["recall.B_victims_edited"]["mean"]))
put("DiagRecallT", "{:.0f}".format(pr["diagnosed-control"]["recall.B_victims_edited"]["t"]))
put("RepShare", "{:.0f}\\%".format(100 * pr["diagnosed-control"]["recall.B_victims_edited"]["mean"]
                                    / pr["repair-control"]["recall.B_victims_edited"]["mean"]))
put("RepCost", "{:.3f}".format(pr["repair-control"]["miou.unedited"]["mean"]))
put("RepCostT", "{:.1f}".format(pr["repair-control"]["miou.unedited"]["t"]))
put("DiagCost", "{:.3f}".format(pr["diagnosed-control"]["miou.unedited"]["mean"]))
put("RepVictimsA", ER["victims_A"]); put("RepVictimsB", ER["victims_B"]); put("RepHalf", ER["n_A"])
put("RepPairs", ER["n_A"] + ER["n_B"])                 # pairs with a recall on both sides (ea_split.py)
put("RepDiagShareImgs", "{:.0f}\\%".format(100 * ER["victims_A"] / ER["n_A"]))
mn = ER["means"]
print("EA repair means keys:", list(mn)[:6], "start:", ER["start"] if not isinstance(ER["start"], dict) else list(ER["start"])[:6])
rows = []
for met, lab in (("recall.B_victims_edited", "Recall of the held-out victims, edited images"),
                 ("miou.unedited", "mIoU, val images never edited"),
                 ("recall.B_victims_original", "Recall of the same victims, original images"),
                 ("miou.B_edited", "mIoU, held-out edited images")):
    rows.append([lab, "{:.3f}".format(ER["start"][met])] + ["{:.3f}".format(mn[arc][met]) for arc in ("control", "diagnosed", "repair")])
tex_table("tab_earepair.tex", rows)

# the certified groups of the official VOC run: situations by phrase, and the clusters
from collections import Counter, defaultdict
idx = json.load(open(R.out_path("slice_quality_inst", "slices_index.json"), encoding="utf-8"))["slices"]
names = json.load(open(R.out_path("slice_quality_inst", "cluster_names.json"), encoding="utf-8"))
lang = [x for x in idx if x["source"] == "language" and x["in_descriptions"]]
by_phrase = defaultdict(lambda: {"cls": set(), "err": 0, "types": Counter()})
for x in lang:
    b = by_phrase[x["hypothesis"]]
    b["cls"].add(x["slice_id"].split("_")[0])
    b["err"] += x["size"]; b["types"][x["dominant_error_type"]] += 1
TYPE_EN = {"fp": "hallucinated", "fn": "missed", "merge": "merge", "boundary": "boundary", "split": "split",
           "extent": "extent", "confusion": "wrong class"}
top = sorted(by_phrase.items(), key=lambda kv: (-len(kv[1]["cls"]), -kv[1]["err"]))[:10]
tex_table("tab_phrases.tex", [["``%s''" % q, len(b["cls"]), b["err"], TYPE_EN[b["types"].most_common(1)[0][0]]] for q, b in top])
tt = Counter(x["dominant_error_type"] for x in lang)
for k, v in TYPE_EN.items():
    put("SitType" + v.title().replace(" ", ""), tt.get(k, 0))
put("SitPhrases", len(by_phrase))
clus = [x for x in idx if x["source"] == "cluster" and x["in_descriptions"]]
clean = lambda n: n.split(" #")[0].split(" · ")[0]
tex_table("tab_clusters.tex", [[clean(names.get(x["slice_id"], x["slice_id"])), TYPE_EN[x["dominant_error_type"]], x["size"], x["n_images"]]
                               for x in sorted(clus, key=lambda x: -x["size"])])

# the minimum of ten correct objects, class by class
tex_table("tab_min.tex", [[c["cls"], "{} / {}".format(c["n_ok"], c["n_err"]), c["cert"], c["confirmed"], c["now"]]
                          for c in rm["classes"] if c["cert"] > 0])
# the validations of the official run, in one table
tex_table("tab_valid.tex", [
    ["Situation confirmed by the VLM on the members (same-class control)",
     "{:.2f} ({:.2f})".format(pl["precision"], pl["control_rate_same_class"]),
     "{} of {} phrase slices above chance".format(pl["groups_above_chance"], pl["groups"])],
    ["Cluster name confirmed by the VLM on the members (same-class control)",
     "{:.2f} ({:.2f})".format(pc["precision"], pc["control_rate_same_class"]),
     "{} of {} names above chance".format(pc["groups_above_chance"], pc["groups"])],
    ["Situations significant on the second segmenter", "{} of {}".format(m2["auc_sys"], m2["official_language"]),
     "{} pass the whole SegCiani pipeline again".format(m2["recertified"])],
    ["Clusters replicated on held-out images", "{} of {}".format(ov["split"]["n_replicated"], ov["split"]["n_discovered"]),
     "{} of {} on the earlier typing".format(ov["before"]["split"]["n_replicated"], ov["before"]["split"]["n_discovered"])],
    ["Deleted small chairs in a significant cluster", "{} of {}".format(ov["corr"]["p25"]["rule_a"]["recovered"],
                                                                    ov["corr"]["p25"]["rule_a"]["matched_as_fn"]),
     "{} of {} at the higher dose".format(ov["corr"]["p50"]["rule_a"]["recovered"], ov["corr"]["p50"]["rule_a"]["matched_as_fn"])],
    ["Facts score against human notes (control)", "{:.2f} ({:.2f})".format(sa["genuine"], sa["control"]),
     "{} images, {} of them synthetic; {} ({}) on the {} of VOC".format(
         sa["n"], sa["n"] - int(macros["FactsVOCn"].replace("\\xspace", "")), macros["FactsVOC"].replace("\\xspace", ""),
         macros["FactsVOCCtrl"].replace("\\xspace", ""), macros["FactsVOCn"].replace("\\xspace", ""))],
    ["Plausibility of the blamed situation (control)", "{:.2f} ({:.2f})".format(cz["pipeline"]["score"], cz["pipeline"]["control"]),
     "{} images with a significant slice".format(cz["pipeline"]["n"])]])
# situations certified on both VOC and SBD, the largest on SBD first
both = sorted(set(sb["lang_voc"]) & set(sb["lang_sbd"]), key=lambda k: -sb["lang_sbd"][k]["size"])
tex_table("tab_sbdboth.tex", [[k[0], "``%s''" % k[1], "{:,}".format(sb["lang_voc"][k]["size"]), "{:,}".format(sb["lang_sbd"][k]["size"])]
                              for k in both[:10]])
# arm 2 seed by seed
tex_table("tab_arm2.tex", [[s_, "{:.4f}".format(c_), "{:.4f}".format(o_), "{:.4f}".format(r_), "{:+.4f}".format(o_ - c_),
                            "{:+.4f}".format(r_ - o_), "{:+.4f}".format(r_ - c_)] for s_, c_, o_, r_ in a2["per_seed"]]
          + [["mean", "{:.4f}".format(a2["control"][0]), "{:.4f}".format(a2["objcentre"][0]), "{:.4f}".format(a2["repair"][0]),
              "{:+.4f}".format(a2["frame"]["mean"]), "{:+.4f}".format(a2["cut"]["mean"]), "{:+.4f}".format(a2["total"]["mean"])],
             ["paired $t$", "", "", "", "{:.1f}".format(a2["frame"]["t"]), "{:.1f}".format(a2["cut"]["t"]), "{:.1f}".format(a2["total"]["t"])]])

# Appendix A: the three typings of Block 1 on the same predictions
RN = R.load_runs()
rows = []
for k in ("fn", "fp", "merge", "split", "boundary", "extent", "confusion"):
    rows.append([TYPE_EN[k]] + ["{:,}".format(RN[r]["tax"].get(k, 0)) for r in ("ref", "cov05b", "inst")])
rows.append(["\\midrule all errors"] + ["{:,}".format(RN[r]["n_err"]) for r in ("ref", "cov05b", "inst")])
rows.append(["correct objects"] + ["{:,}".format(RN[r]["n_ref"]) for r in ("ref", "cov05b", "inst")])
tex_table("tab_typings.tex", rows)
# Appendix B: every certified situation of the official VOC run
rows = []
for x in sorted(lang, key=lambda x: (x["slice_id"].split("_")[0], -x["size"])):
    cls_ = x["slice_id"].split("_")[0]
    rows.append([cls_, "``%s''" % x["hypothesis"].replace("&", "\\&"), TYPE_EN[x["dominant_error_type"]], x["size"], x["n_images"]])
tex_table("tab_allsit.tex", rows)
# situations whose members come from few images (the review of 2 Nov: a situation of one scene is weak recurrence)
put("SitOneImage", sum(x["n_images"] == 1 for x in lang)); put("SitThreeImages", sum(x["n_images"] >= 3 for x in lang))
_one = max((x for x in lang if x["n_images"] == 1), key=lambda x: x["size"], default=None)
if _one:
    put("SitOneImageEx", "``%s'' (%s, %d errors)" % (_one["hypothesis"], _one["slice_id"].split("_")[0], _one["size"]))

# ---- the second segmenter: how the error types change (official typing, night/model2_cluster.log)
import re
m2log = open(R.out_path("slice_quality_inst", "night", "model2_cluster.log"), encoding="utf-8").read()
mm2 = re.search(r"(\d+) errors, (\d+) reference \| (\{[^}]*\})", m2log)
tax2 = json.loads(mm2.group(3).replace("'", '"'))
tax1 = s["tax"]
n1, n2 = sum(tax1.values()), sum(tax2.values())
rows = []
for k in ("fp", "fn", "confusion", "merge", "split", "boundary", "extent"):
    rows.append([TYPE_EN[k], "{:,} ({})".format(tax1[k], p1(tax1[k] / n1)), "{:,} ({})".format(tax2[k], p1(tax2[k] / n2)),
                 "{:+.0f}\\%".format(100 * (tax2[k] / tax1[k] - 1)).replace("-", "$-$")])
rows.append(["\\midrule all errors", "{:,}".format(n1), "{:,}".format(n2),
             "{:+.0f}\\%".format(100 * (n2 / n1 - 1)).replace("-", "$-$")])
tex_table("tab_m2types.tex", rows)
put("MtwoHallucDrop", "{:.0f}\\%".format(100 * (1 - tax2["fp"] / tax1["fp"])))
put("MtwoMissedDrop", "{:.0f}\\%".format(100 * (1 - tax2["fn"] / tax1["fn"])))
put("MtwoBndA", "{:,}".format(tax1["boundary"])); put("MtwoBndB", "{:,}".format(tax2["boundary"]))
put("MtwoMergeA", "{:,}".format(tax1["merge"])); put("MtwoMergeB", "{:,}".format(tax2["merge"]))
put("MtwoShapeA", "{:.0f}\\%".format(100 * (tax1["boundary"] + tax1["merge"]) / n1))
put("MtwoShapeB", "{:.0f}\\%".format(100 * (tax2["boundary"] + tax2["merge"]) / n2))
put("MtwoPerceptA", "{:.0f}\\%".format(100 * (tax1["fp"] + tax1["fn"]) / n1))
put("MtwoPerceptB", "{:.0f}\\%".format(100 * (tax2["fp"] + tax2["fn"]) / n2))

# ---- which families of hypotheses certify: pairs read on the sample, situations of the official run
sys.path.insert(0, os.path.join(DOCS, ".."))
from slice_quality.describe import _CANONICAL, _GENERIC_EXTRA  # noqa: E402
DDH = R.load_dd_hyp()
dd_pairs = {(c, h["query"]) for c in DDH["classes"] for h in DDH["per_class"][c]["hypotheses"]}
dd_pairs |= {(c, h) for c, v in json.load(open(R.DD_PILOT, encoding="utf-8"))["hypotheses"].items() for h in v}


def family(c, h):
    if h in _CANONICAL:
        return "canonical"
    if h in _GENERIC_EXTRA:
        return "extra"
    return "dd" if (c, h) in dd_pairs else "specific"


rk = json.load(open(R.out_path("slope_permutation", "results_rank.json"), encoding="utf-8"))["results"]["VOC val"]
read = {(r["cls"], r["query"]) for r in rk if r["stage"] == "sample"} | dd_pairs
cert_pairs = {(x["slice_id"].split("_")[0], x["hypothesis"]) for x in lang}
fam_rows = []
for f, lab in (("canonical", "Canonical, class-agnostic (SliceLens)"), ("extra", "Class-agnostic, added in this work"),
               ("specific", "Class-specific, from the class name (Gemma~4)"), ("dd", "Data-driven, from captions (SliceLens)")):
    nr = sum(1 for k in read if family(*k) == f)
    nc = sum(1 for k in cert_pairs if family(*k) == f)
    ncls = len({c for c, h in read if family(c, h) == f})
    fam_rows.append([lab, ncls, nr, nc, p1(nc / nr)])
    put("Fam" + f.title() + "Read", nr); put("Fam" + f.title() + "Cert", nc); put("Fam" + f.title() + "Rate", "{:.0f}\\%".format(100 * nc / nr))
tex_table("tab_families.tex", fam_rows)
put("DDnHyp", DDH["n_hyp"]); put("DDnFlat", DDH["n_flat"]); put("DDnCommon", len(DDH["flat_common"])); put("DDnRare", len(DDH["flat_rare"]))
fc, fr_ = DDH["flat_common"][0], DDH["flat_rare"][0]
put("DDcommonEx", "``%s''" % fc["query"]); put("DDcommonMatch", fc["sample_match"]); put("DDcommonOf", fc["sample_scored"])
put("DDrareEx", "``%s''" % fr_["query"]); put("DDrareMatch", fr_["sample_match"])
# the failure regimes of the literature (Section sec:rel-failures) against the vocabulary and the official run
ncls_of = lambda h: len({c for c, q in cert_pairs if q == h})
cond = re.compile(r"\b(blur\w*|fog|rain|snow|night|shadows?|sunlight|light\w*|glare|overexposed|dark)\b", re.I)
cond_cert = sorted({q for c, q in cert_pairs if family(c, q) in ("specific", "dd") and cond.search(q)})
# the regimes of Chapter 2 with their sources, then the two the literature does not name
REG = [("Small objects~\\cite{zhao2017pspnet}", ["a small object"]),
       ("Similar categories, context~\\cite{zhao2017pspnet}", ["an object that looks similar to a different object category",
                                                              "an object in an unusual setting for its kind"]),
       ("Low contrast, camouflage~\\cite{zhao2017pspnet,ji2023samconcealed}", ["an object that blends into its background"]),
       ("Open-world images~\\cite{yan2021mad}", ["a toy, statue or model of an object",
                                                 "a picture, poster or screen showing an object",
                                                 "a reflection of an object in a mirror, window or water"]),
       ("Geometry: viewpoint, scale~\\cite{yin2026gen4seg}", ["an object seen from an unusual viewpoint or in an unusual pose",
                                                             "a very large object seen up close that fills most of the image"])]
OURS = [("Thin and see-through structures", ["an object with thin or elongated parts", "an object with holes or see-through parts",
                                             "an object behind a fence, net, bars or branches"]),
        ("Occlusion and truncation", ["an object occluded by another object", "an object cut off at the image border",
                                      "several overlapping objects of the same type"])]
rows = []
for reg, hs in REG:
    for i, h in enumerate(hs):
        rows.append([reg if i == 0 else "", "``%s''" % h, ncls_of(h)])
rows.append(["Corruptions, weather, night~\\cite{kamann2020robustness,sakaridis2021acdc}",
             "class-specific situations only (blur, light, weather, night)", "%d situations, one class each" % len(cond_cert)])
rows.append(["\\midrule \\multicolumn{3}{l}{\\emph{Not among the regimes of the literature}}"])
for reg, hs in OURS:
    for i, h in enumerate(hs):
        rows.append([reg if i == 0 else "", "``%s''" % h, ncls_of(h)])
tex_table("tab_regimes.tex", rows)
put("RegCondCert", len(cond_cert))

# ---- precision of the cluster names, the rubric judge, and the two accounts of the errors
put("PrecClusGroups", pc["groups"]); put("PrecClusAbove", pc["groups_above_chance"])
NR = R.load_narr()
put("RubricScore", "{:.2f}".format(NR["judge"]["narrative"]["vs_ref"]["b"]))
put("RubricCtrl", "{:.2f}".format(NR["judge"]["narrative"]["control"]))
put("RubricN", NR["judge"]["narrative"]["vs_ref"]["n"])
put("RubricTen", NR["ten_after"]); put("RubricLow", NR["low_after"])
cv = R.load_cause()["voc"]
put("TwoN", cv["agree"]["n"]); put("TwoR", "{:.2f}".format(cv["agree"]["r"])); put("TwoWithinOne", cv["agree"]["within1"])
put("TwoNote", "{:.2f}".format(cv["human"]["score"])); put("TwoNoteCtrl", "{:.2f}".format(cv["human"]["control"]))
put("TwoNarr", "{:.2f}".format(cv["pipeline"]["score"])); put("TwoNarrCtrl", "{:.2f}".format(cv["pipeline"]["control"]))
put("TwoNoteWords", "{:.0f}".format(cv["human"]["words"])); put("TwoNarrWords", "{:.0f}".format(cv["pipeline"]["words"]))
put("TwoBothHigh", cv["agree"]["both_high"])

# ---- the preliminary experiment, model by model
NICE = {"gemma-4": "Gemma 4", "Qwen3.5-9B": "Qwen 3.5 9B", "gemma-3n-E4B-it": "Gemma 3n E4B", "Kimi-K2.5": "Kimi K2.5",
        "gemini-3.5-flash": "Gemini 3.5 Flash$^{*}$"}
cellf = lambda d: "{:.2f} ({})".format(d["mean"], d["na"])
tex_table("tab_firstexp.tex", [[NICE.get(m, m), cellf(FX["models"][m]["JOIN-D1"]), cellf(FX["models"][m]["JOIN-D2"]),
                                cellf(FX["models"][m]["SEP"])]
                               for m in sorted(FX["models"], key=lambda m: -FX["models"][m]["JOIN-D1"]["mean"])])
put("FXformatP", "10^{%d}" % int(math.ceil(math.log10(fxf["p_wilcoxon"]))))
put("FXjudgesR", "{:.2f}".format(FX["judges"]["pearson"])); put("FXjudgesN", FX["judges"]["n"])
put("FXgemmaSepNA", FX["models"]["gemma-4"]["SEP"]["na"])

# ---- Appendix A: what each typing did to the text and to the groups
rows = []
cols = ("ref", "cov05b", "inst", "inst_v3")
rows.append(["Description, rubric score (control)"] + ["{:.2f} ({:.2f})".format(RN[r]["narrative"]["score"], RN[r]["narrative"]["control"])
                                                      for r in cols])
rows.append(["Descriptions scoring 10 / at most 6"] + ["{} / {}".format(RN[r]["ten"], RN[r]["low"]) for r in cols])
rows.append(["``missed entirely''"] + ["{:,}".format(RN[r]["text_counts"]["missed"]) for r in cols])
rows.append(["``absorbed into a neighbouring prediction''"] + ["{:,}".format(RN[r]["text_counts"]["absorbed"]) for r in cols])
rows.append(["merge measured against its union"] + ["{:,}".format(RN[r]["text_counts"]["union"]) for r in cols])
rows.append(["``partly hidden behind''"] + ["{:,}".format(RN[r]["text_counts"]["hidden"]) for r in cols])
rows.append(["\\midrule significant clusters"] + [str(RN[r]["slices"]["cluster"]) for r in cols[:3]] + ["---"])
rows.append(["significant situations (slope rule)"] + [str(RN[r]["slices"]["language"]) for r in cols[:3]] + ["---"])
tex_table("tab_typings_effect.tex", rows)
put("AbsorbedRef", RN["ref"]["text_counts"]["absorbed"]); put("MissedRef", RN["ref"]["text_counts"]["missed"])
put("NarrImages", "{:,}".format(RN["ref"]["text_counts"]["images"]))
_n36 = len(json.load(open(R.out_path("slice_quality_inst", "train36", "descriptions_train36_full.json"), encoding="utf-8")))
put("NarrTrain", _n36); put("NarrVal", "{:,}".format(RN["ref"]["text_counts"]["images"] - _n36))

# ---- looser cluster gates: copies of the official run rescored with relaxed gates (outputs/chain_relaxed_cluster.sh,
#      rescored with the geometric join on 3 November 2026 by chain_geo_join.sh; relaxed_report.py)
RX = json.load(open(R.out_path("relaxed_cluster_report.json"), encoding="utf-8"))
_rx_rows = []
for _tag, _key, _lab in (("Off", "official", "Official gates"),
                         ("A", "A", "A: attribute at $p \\leq 0.10$, coherence $\\geq 0.5$"),
                         ("B", "B", "B: no attribute test")):
    _r = RX[_key]
    _worse = sum(1 for _x in _r["rows"] if _x["drop"] > 0)
    _mdrop = sum(_x["drop"] for _x in _r["rows"]) / len(_r["rows"])
    put("Relax" + _tag + "Clus", _r["certified"])
    put("Relax" + _tag + "Cov", "{:.1f}\\%".format(_r["cov_union"]))
    put("Relax" + _tag + "CovClus", "{:.1f}\\%".format(_r["cov_cluster"]))
    put("Relax" + _tag + "Repl", _r["split_half"]["n_replicated"]); put("Relax" + _tag + "Disc", _r["split_half"]["n_discovered"])
    put("Relax" + _tag + "Worse", _worse); put("Relax" + _tag + "Drop", "{:.1f}".format(_mdrop))
    put("Relax" + _tag + "NoAttr", sum(1 for _x in _r["rows"] if "no verified attribute" in _x["description"]))
    _rx_rows.append([_lab, _r["certified"], "{:.1f}\\%".format(_r["cov_union"]), "{:.1f}\\%".format(_r["cov_cluster"]),
                     "{} of {}".format(_r["split_half"]["n_replicated"], _r["split_half"]["n_discovered"]),
                     "{} of {} ({:+.1f})".format(_worse, len(_r["rows"]), _mdrop)])
tex_table("tab_relaxed.tex", _rx_rows)
_cow = next(_x for _x in RX["A"]["rows"] if _x["slice_id"] == "cluster_merge_0_r2")
put("RelaxCowN", _cow["size"]); put("RelaxCowImg", _cow["n_images"])

# ---- Appendix B: failure directions in the embedding space (slice_quality/examples/failure_probe.py, 28 October)
FP_ = R.out_path("slice_quality_inst", "failure_probe.json")
if os.path.exists(FP_):
    fp = json.load(open(FP_, encoding="utf-8"))
    P_ = fp["populations"]
    put("ProbeClasses", len(fp["classes"])); put("ProbePerms", fp["perms"]); put("ProbeMinOk", fp["min_ok"])
    put("ProbeTrueMaskN", "{:,}".format(fp["n_true_mask"]))
    for k, tag in (("all", "All"), ("real", "Real"), ("truemask", "True")):
        sm = P_[k]["summary"]
        put("Probe" + tag + "AUC", "{:.2f}".format(sm["mean_auc"])); put("Probe" + tag + "Size", "{:.2f}".format(sm["mean_auc_size"]))
        put("Probe" + tag + "Both", "{:.2f}".format(sm["mean_auc_both"])); put("Probe" + tag + "N", sm["n_classes"])
        put("Probe" + tag + "Above", sm["above_chance"]); put("Probe" + tag + "BeatSize", sm["above_size"])
        if "spearman_cov" in sm:
            put("Probe" + tag + "Rho", "{:+.2f}".format(sm["spearman_cov"][0])); put("Probe" + tag + "RhoP", "{:.2f}".format(sm["spearman_cov"][1]))
    nulls = [c["null_p95"] for k in P_ for c in P_[k]["classes"]]
    put("ProbeNullMin", "{:.2f}".format(min(nulls))); put("ProbeNullMax", "{:.2f}".format(max(nulls)))
    top_ = lambda c: next(iter(c["attr_lift"]), "-")
    put("ProbeSmallTop", sum(1 for c in P_["all"]["classes"] if top_(c) == "small"))
    rows = []
    for k, lab in (("all", "every wrong object"), ("real", "without hallucinations"),
                   ("truemask", "pooled on the true mask (missed, swallowed)")):
        sm = P_[k]["summary"]
        rows.append([lab, sm["n_classes"], "{:.2f}".format(sm["mean_auc"]), "{:.2f}".format(sm["mean_auc_size"]),
                     "{:.2f}".format(sm["mean_auc_both"]), "{} of {}".format(sm["above_chance"], sm["n_classes"]),
                     "{} of {}".format(sm["above_size"], sm["n_classes"])])
    tex_table("tab_probe.tex", rows)
    tm = {c["cls"]: c for c in P_["truemask"]["classes"]}
    rows = []
    for c in sorted(P_["real"]["classes"], key=lambda c: -c["auc"]):
        t_ = tm.get(c["cls"])
        rows.append([c["cls"], "{} / {}".format(c["n_wrong"], c["n_ok"]), "{:.2f} [{:.2f}, {:.2f}]".format(c["auc"], *c["ci"]),
                     "{:.2f}".format(c["auc_size"]), "{:.2f}".format(t_["auc"]) if t_ else "---",
                     "{:.2f}".format(t_["auc_size"]) if t_ else "---", "{:.0f}\\%".format(100 * c["coverage"]), top_(c)])
    tex_table("tab_probe_cls.tex", rows)

# ---- values measured by the figure scripts (figs/fig_data.py, figs/fig_images.py) on the same runs
FV = os.path.join(HERE, "figs", "cache", "fig_data_values.json")
if os.path.exists(FV):
    fv = json.load(open(FV, encoding="utf-8"))
    car = fv["car"]
    put("CarAUC", "{:.2f}".format(car["auc"])); put("CarErr", car["n_err"]); put("CarCor", car["n_cor"])
    put("CarMedErr", "{:.2f}".format(car["median_err"])); put("CarMedCor", "{:.2f}".format(car["median_cor"]))
    put("CarSlope", "{:.2f}".format(car["slope_recomputed"])); put("CarWindows", car["windows"])
    for k, v in fv["bic"].items():
        put("BICk" + TAXN[k], v["k"])
    cp = fv["compactness"]
    for sid, tag in (("cluster_merge_12_r0", "Plant"), ("cluster_fn_4_r0", "Small"), ("cluster_fn_1_r0", "Chairs"),
                     ("cluster_boundary_11_r0", "People"), ("person_0", "Person")):
        put("Comp" + tag, "{:.2f}".format(cp[sid]["obs"])); put("Comp" + tag + "Null", "{:.2f}".format(cp[sid]["null_mean"]))
        put("Comp" + tag + "P", "{:.3f}".format(cp[sid]["p"]))
    ls = fv["compactness_lang_summary"]
    put("CompLangAbove", ls["above_chance"]); put("CompLangN", ls["n"]); put("CompLangRatio", "{:.1f}".format(ls["median_ratio"]))
    put("FactsGeEight", fv["facts"]["genuine_ge8"]); put("FactsCtrlLeOne", fv["facts"]["control_le1"])
    if "p_wilcoxon" in fv["facts"]:
        pw = fv["facts"]["p_wilcoxon"]
        put("FactsPairP", "10^{%d}" % int(math.ceil(math.log10(pw))) if pw < 1e-4 else "{:.4f}".format(pw))
        put("FactsPairWins", fv["facts"]["wins"]); put("FactsPairN", fv["facts"]["paired_n"])
    tex_table("tab_compact.tex", [[clean(names.get(s, s)), TYPE_EN[next(x for x in clus if x["slice_id"] == s)["dominant_error_type"]],
                                   cp[s]["n"], "{:.2f}".format(cp[s]["obs"]), "{:.2f}".format(cp[s]["null_mean"]),
                                   "{:.3f}".format(cp[s]["p"])]
                                  for s in sorted((x["slice_id"] for x in clus), key=lambda s: -cp[s]["obs"])])
    _cr = [cp[x["slice_id"]]["obs"] / cp[x["slice_id"]]["null_mean"] for x in clus]   # compactness over its null
    put("CompRatioMin", "{:.1f}".format(min(_cr))); put("CompRatioMax", "{:.0f}".format(max(_cr)))
    tex_table("tab_bic.tex", [[TYPE_EN[k], "{:,}".format(v["n"]), "kept whole" if v.get("whole") else v["k"]]
                              for k, v in sorted(fv["bic"].items(), key=lambda kv: -kv[1]["n"])])
C18 = os.path.join(HERE, "figs", "cache", "card18.json")
if os.path.exists(C18):
    c18 = json.load(open(C18, encoding="utf-8"))
    w = {"stated": 1.0, "partial": 0.5, "absent": 0.0, "contradicted": -1.0,          # the page writes the automatic
         "riportato": 1.0, "in parte": 0.5, "assente": 0.0, "contraddetto": -1.0}     # judge's verdicts in Italian
    hand = sum(w[v] for v, g, t in c18["facts"]) / len(c18["facts"])
    auto = sum(w[g] for v, g, t in c18["facts"]) / len(c18["facts"])
    put("HorseFactsHand", "{:.1f}".format(10 * max(0, hand))); put("HorseFactsAuto", "{:.1f}".format(10 * max(0, auto)))
    put("HorseNfacts", len(c18["facts"]))
    # plausibility of the blamed situation, read from the judge's own file (the page's card can be older than it)
    _jc = {(r["kind"], r["pairing"]): r["score"] for r in
           json.load(open(R.out_path("slice_quality_inst", "judge_cause.json"), encoding="utf-8"))["judgements"]
           if r["image"] == "2007_000332" and "score" in r}
    put("HorsePlausNarr", "{:.1f}".format(_jc[("pipeline", "genuine")])); put("HorsePlausNote", "{:.1f}".format(_jc[("human", "genuine")]))
    put("HorsePlausCtrl", "{:.1f}".format((_jc[("pipeline", "control")] + _jc[("human", "control")]) / 2))

# Italian twins of the numbers used in the Sommario and the Estratto (3 Nov): decimal comma, thousands point
IT_MACROS = ["EAfoundPipeline", "EAfoundSliceLens", "EAvictims", "FactsVOC", "FactsVOCCtrl", "FactsVOCn", "MtwoAuc",
             "MtwoLang", "PlausCtrl", "PlausScore", "RepShare", "RepVictimsA", "SBDPipelineCov", "VOCSliceLensCov",
             "VOCclusters", "VOCcoverage", "VOCerrors", "VOCimages", "VOCimgErr", "VOCmiou", "VOCsituations"]
for _k in IT_MACROS:
    _v = re.sub(r"(?<=\d),(?=\d{3})", "\x00", str(macros[_k]))
    macros[_k + "IT"] = re.sub(r"(?<=\d)\.(?=\d)", ",", _v).replace("\x00", ".")

with open(os.path.join(HERE, "generated", "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% written by make_numbers.py -- do not edit\n")
    for k, v in sorted(macros.items()):
        f.write("\\newcommand{\\%s}{%s\\xspace}\n" % (k, v))
print("\n".join("%-22s %s" % kv for kv in sorted(macros.items())))
