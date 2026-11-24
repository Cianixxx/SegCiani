#!/bin/bash
# Le 8 ipotesi generiche in piu' (slice_quality.describe._GENERIC_EXTRA).
# Lanciatore lato host: la GPU serve solo al grounding, il resto gira in un
# container CPU, cosi' la scheda si libera appena il grounding finisce.
#
#   GPU=6 bash outputs/run_extra_hyp.sh          # worker + (alla fine) confronto
#   GPU=N bash outputs/run_extra_hyp.sh worker   # solo un secondo worker: si divide
#                                                # le classi rimaste con il primo
# Uscite: /data/outputs_extra/voc_paper_<cls>_cc.json (mai /data/outputs),
#         outputs/slice_quality_inst_extra{,_base}/, rapporto in
#         outputs/slice_quality_inst_extra/extra_hypotheses_report.{json,txt}
set -u
GPU="${GPU:?specificare GPU=N}"
MODE="${1:-all}"
cd ~/storage/SLICELENS
EXTRA=$HOME/storage/pascal_vlm/outputs_extra
mkdir -p "$EXTRA"
step() { echo "$(date +%H:%M) == $1 =="; }
RUN="docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace \
  -v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -w /workspace segciani:dev"

step "grounding delle 8 ipotesi su GPU $GPU (Qwen caricato una volta)"
docker run --rm --gpus device="$GPU" --name extrahyp_gpu$GPU -u $(id -u):$(id -g) -e HOME=/tmp \
  -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e OUT_DIR=/data/outputs_extra \
  -v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -w /workspace segciani:dev \
  bash -c "pip install --user -q scipy >/dev/null 2>&1; python -u examples/ground_extra_hypotheses.py" \
  || { echo "FAIL grounding"; exit 1; }
echo "--- GPU $GPU:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "$((GPU + 1))p"
[ "$MODE" = "worker" ] && { echo "$(date +%H:%M) EXTRA_WORKER_EXIT"; exit 0; }

# il confronto parte una volta sola, quando ci sono tutte e 20 le classi
until [ "$(ls $EXTRA/voc_paper_*_cc.json 2>/dev/null | wc -l)" -ge 20 ]; do sleep 300; done
mkdir "$EXTRA/.compare_lock" 2>/dev/null || { echo "confronto gia' lanciato da un altro worker"; exit 0; }

step "rescore inst SENZA le nuove ipotesi (CPU): il riferimento per il confronto"
B=outputs/slice_quality_inst_extra_base; mkdir -p $B
cp outputs/slice_quality_inst/result_tagged.pkl $B/
$RUN bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object SQ_OUT=$B LANG_DIRS=/data/outputs \
  python -u slice_quality/examples/rescore_with_language.py" > $B/rescore.log 2>&1 \
  || { echo "FAIL rescore base"; tail -5 $B/rescore.log; exit 1; }

step "rescore inst CON le nuove ipotesi (CPU)"
E=outputs/slice_quality_inst_extra; mkdir -p $E
cp outputs/slice_quality_inst/result_tagged.pkl $E/
$RUN bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object SQ_OUT=$E LANG_DIRS=/data/outputs:/data/outputs_extra \
  python -u slice_quality/examples/rescore_with_language.py" > $E/rescore.log 2>&1 \
  || { echo "FAIL rescore extra"; tail -5 $E/rescore.log; exit 1; }

step "confronto"
$RUN bash -c "python -u slice_quality/examples/compare_extra_hypotheses.py" || { echo "FAIL compare"; exit 1; }
echo "$(date +%H:%M) EXTRA_HYP_DONE"
