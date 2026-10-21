# Model architectures

`inference.py` reconstructs the released checkpoint from its stored contract. The other architecture modules support the model contracts and checkpoint families handled by that loader.

| File | Purpose |
| --- | --- |
| `__init__.py` | Declares the architecture package. |
| `acoustic_multiplicative_temporal_ownership.py` | Acoustic temporal ownership model. |
| `adaptive_sparse_ownership.py` | Adaptive sparse ownership baseline. |
| `dna_residual_tfc_framewise_conditioned.py` | Framewise-conditioned residual layers. |
| `dna_residual_tfc_tdf_unet.py` | TFC-TDF U-Net backbone. |
| `hpss_target.py` | Harmonic/percussive target features. |
| `occurrence_dual_hpss_band_split_roformer.py` | Band-split RoFormer separator contract. |
| `occurrence_dual_hpss_cqt_comb_film.py` | CQT-conditioned separator contract. |
| `occurrence_dual_hpss_frequencywise_joint_output.py` | Frequencywise joint-output separator. |
| `occurrence_dual_hpss_joint_complex_output.py` | Joint complex-output separator. |
| `occurrence_dual_hpss_learned_loop.py` | Learned-loop separator contract. |
| `occurrence_dual_hpss_merged_output.py` | Merged-output separator contract. |
| `occurrence_dual_hpss_separator.py` | Released dual HPSS separator and contract dispatcher. |
| `occurrence_dual_hpss_symmetric_alignment.py` | Symmetric branch alignment. |
| `occurrence_dual_hpss_tf_gridnet.py` | TF-GridNet separator contract. |
| `occurrence_joint_tfc_tdf_separator.py` | Joint TFC-TDF branch architecture. |
| `pitch_relative_harmonic_gate.py` | Pitch-relative harmonic gate. |
| `temporal_hybrid_set_ownership.py` | Temporal hybrid ownership architecture. |
| `LICENSE-KUIELAB-SDX23.txt` | License for the adapted SDX23 components. |
| `LICENSE-KUIELAB-SDX23-provenance.json` | Source and change record for those components. |
