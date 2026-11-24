# slice_quality — pipeline di error-slice discovery per instance segmentation

Pipeline modulare che estende **SliceLens** con i contributi di **Beyond IoU**,
**MCSD**, **CB-SLICE** e **SEG4SEG**. Ogni blocco è un file separato, con una
docstring che dichiara: *cosa fa · da quale paper · input → output · dove modificare*.

## Mappa dei blocchi (ordine della pipeline)

| # | File | Blocco | Da quale paper |
|---|------|--------|----------------|
| 1 | `typed_error.py` | matching 2-stadi + decomposizione errore tipato (6 casi); con `claim_cov` anche il claim per contenimento e le GT inglobate come istanze lato GT (1 + M) | Beyond IoU (+ nuovo per instance-seg) |
| 2 | `embeddings.py` | mask-pooled embedding per istanza | MCSD / Domino (interfaccia) |
| 3 | `features.py` | feature vector `emb ⊕ E ⊕ subtype` (norm + α) | nuovo |
| 4 | `clustering.py` | GMM γ-pesato (ramo cluster) | SEG4SEG |
| 5 | `metrics/` | i 6 assi dello Slice-Quality Vector | misto (vedi sotto) |
| 6 | `slice_quality_vector.py` | assembla i 6 assi + criterio Def.1 | SEG4SEG + nuovo |
| 7 | `refine.py` | cascata di refine (split det. / re-cluster / LLM) | nuovo |
| 8 | `explanation.py` | spiegazione (deterministica + over-rep. attributi) | CB-SLICE (ECSA-like) |
| 9 | `recommendation.py` | routing riparazione per tipo d'errore | nuovo |
| 0 | `pipeline.py` | orchestratore end-to-end (wiring) | — |

### Moduli intorno ai blocchi (stessa cartella)
| File | Ruolo |
|------|-------|
| `adapters/` | dataset → `samples` (`voc_seg_samples.py`: VOC, istanze GT = componenti connesse o, con `SQ_GT_INSTANCES=object`, oggetti di `SegmentationObject`; `coco_samples.py`; `slicelens_run.py`; `to_slicelens.py` ponte verso le `Instance` di SliceLens) |
| `language_branch.py` | ponte SliceLens grounding → slice del ramo linguistico |
| `tagging.py` | tag di attributo con Qwen (verify mode) → attiva purezza e spiegazione semantica |
| `corroborate.py` | corroborazione fra rami: slice cluster certificate nominate da ipotesi verificate |
| `validate_split.py` | validazione split-half del ramo cluster (guardia post-selezione) |
| `describe.py` | descrizione per immagine dei PATTERN sistematici (solo errori in slice certificate) |
| `mask_facts.py` | fatti di maschera che il Blocco 1 non registra: boundary IoU, verso e lati dell'errore (`missed/excess_sides`), dettagli sottili persi, buchi, GT e toppe sotto soglia, merge contro l'UNIONE delle GT fuse, occlusore di un oggetto tagliato (`occluded_by`), verdetto d'immagine |
| `describe_errors.py` | NARRATIVA per immagine di OGNI errore tipato, nel registro degli annotatori (frase per gruppo tipo×classe, regioni corrette, sconfinamenti, verdetto in apertura) — il testo confrontato con le annotazioni umane dal giudice |
| `gallery.py`, `report.py` | galleria HTML delle slice; report JSON + markdown |
| `data_types.py`, `extractors.py` | strutture dati (`TypedInstance`, SQV); backbone DINOv2 |

### Variabili d'ambiente che cambiano il Blocco 1
| Env | Effetto | Run |
|-----|---------|-----|
| (nessuna) | claim per IoU, istanze GT = componenti | `outputs/slice_quality/` (riferimento) |
| `SQ_CLAIM_COV=0.5` | claim per contenimento: una GT rivendica la predizione che ne contiene ≥ metà; le GT inglobate restano istanze lato GT (merge = 1 + M) | `outputs/slice_quality_cov05b/` |
| `SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object` | come sopra, e un'istanza GT è un oggetto VOC (un oggetto occluso è una sola istanza) | `outputs/slice_quality_inst/` |

### I 6 assi (`metrics/`)
| File | Asse | Paper |
|------|------|-------|
| `omega_squared.py` | ω² disparità di performance | SEG4SEG |
| `manifold_compactness.py` | compattezza sulla varietà (grafo kNN) | MCSD |
| `misprediction_coherence.py` | coerenza dell'output sbagliato | CB-SLICE (Eq. 8) |
| `error_type_purity.py` | purezza del tipo d'errore | nuovo (da Beyond IoU) |
| `slice_purity.py` | purezza rispetto a un attributo | SEG4SEG (Def. 1) |
| `significance.py` | bootstrap a blocchi + correzione FDR | nuovo |

## Flusso dati

```
val set (immagini + maschere GT)  +  predizioni del modello
        │
   [1] typed_error  ──► lista di TypedInstance (E tipato, IoU, subtype, embedding)
        │
        ├──► ramo SliceLens (hypothesis + grounding)  ──┐
        │                                               ├─► slice candidate
        └──► [3][4] features + clustering  ─────────────┘
        │
   [6] slice_quality_vector  ──► per ogni slice: 6 assi + verdetto Def.1
        │
   ┌────┴─────────────────────────┐
KEEP                            REFINE ──► [7] refine ──► (ri-scoring)
   │
   ├─► [8] explanation  ──► scheda diagnostica
   └─► [9] recommendation ──► consiglio di riparazione
```

## Dove vivono le dipendenze esterne
* SliceLens   : `../slicelens` (data types, backend LLM/VLM, grounding)
* Beyond IoU  : `../../CB-SLICE/beyond-iou-main/beyond_iou` (decomposizione boundary/extent)
* MCSD        : `../../CB-SLICE/MCSD-main` (riferimento per manifold compactness)
* CB-SLICE    : `../../CB-SLICE/CB-SLICE-main` (riferimento per ECSA/keyword-concepts)
* SEG4SEG     : codice non disponibile → riimplementato qui (`clustering.py`, Def.1)

## Come si esegue
Vedi `pipeline.py` (funzione `run_pipeline`) e lo script di esempio
`scripts/run_slice_quality.py`. I parametri stanno tutti in `config.py`.
