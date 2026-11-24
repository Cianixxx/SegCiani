"""BLOCK 5 -- the six axes of the Slice-Quality Vector.

Each axis is a standalone module so it can be modified / swapped in isolation:

    omega_squared            severity        (SEG4SEG)
    manifold_compactness     geometric coh.  (MCSD)
    misprediction_coherence  output coh.     (CB-SLICE Eq. 8)
    error_type_purity        error-type coh. (new, from Beyond IoU)
    slice_purity             attribute coh.  (SEG4SEG Def. 1)
    significance             reliability     (new: block bootstrap + FDR)
"""
