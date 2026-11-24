#!/bin/bash
# Narrativa v4 sul run inst: tolte le due frasi che parlavano dello STRUMENTO
# ("below the size the pipeline represents as an object"), sia per le regioni GT
# sotto soglia sia per le toppe predette sotto soglia. Solo testo, nessuna GPU:
# il giudice NON viene rilanciato (deciso con Daniele, si rivaluta dopo).
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object SQ_OUT=outputs/slice_quality_inst
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

# le narrative attuali restano come erano, sotto il nome del passo che le ha prodotte
cp $D/error_narratives.json $D/error_narratives_v3.json
cp $D/error_narratives.txt  $D/error_narratives_v3.txt
cp $D/synth/narratives_synth.json $D/synth/narratives_synth_v3.json

step "1/2 narrate_errors val + train36 (CPU)"
PKL=$D/result_tagged.pkl TRAIN36=$D/train36 OUT=$D/error_narratives \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "2/2 narrate_synth (CPU)"
OUT=$D/synth python -u slice_quality/examples/narrate_synth.py || { echo "FAIL narrate_synth"; exit 1; }

echo "$(date +%H:%M) residuo 'pipeline represents' nelle narrative:"
grep -c "pipeline represents" $D/error_narratives.json || true
grep -c "pipeline represents" $D/synth/narratives_synth.json || true
echo "$(date +%H:%M) CHAIN_INST_NARR4_DONE"
