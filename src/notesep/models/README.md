# Model code

This folder implements the released NoteSep inference checkpoint. It is not a
catalog of experimental model variants.

| File | Purpose |
| --- | --- |
| `occurrence_dual_hpss_separator.py` | Builds the checkpoint architecture and checks its saved contract. |
| `occurrence_dual_hpss_symmetric_alignment.py` | Connects the harmonic and percussive branches. |
| `occurrence_joint_tfc_tdf_separator.py` | Predicts each note from one HPSS branch. |
| `dna_residual_tfc_tdf_unet.py` | Shared TFC-TDF network used by both branches. |
| `dna_residual_tfc_framewise_conditioned.py` | Score conditioning layers. |
| `pitch_relative_harmonic_gate.py` | Applies the pitch-based harmonic input gate. |
| `hpss_target.py` | Creates the harmonic and percussive input signals. |
| `acoustic_multiplicative_temporal_ownership.py` | Applies the released ASO checkpoint. |
| `temporal_hybrid_set_ownership.py` | ASO architecture and configuration. |
| `adaptive_sparse_ownership.py` | Sparse ownership operation used by ASO. |
| `LICENSE-KUIELAB-SDX23.txt` | License for adapted model components. |
| `LICENSE-KUIELAB-SDX23-provenance.json` | Source and modification record for those components. |
