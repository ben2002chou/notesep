# Separator model and training modules

| File | Purpose |
| --- | --- |
| `__init__.py` | Declares the separator model package. |
| `batching.py` | Assembles examples into model batches. |
| `corpus.py` | Validates the frozen corpus and split indexes. |
| `dna_harmonic_partial_loss.py` | Defines the harmonic partial loss. |
| `dna_residual_band_expert.py` | Defines the frequency-band residual expert models. |
| `dna_residual_corpus.py` | Reads residual training-corpus records. |
| `dna_residual_corrector.py` | Defines the residual correction model. |
| `dna_residual_losses.py` | Computes reconstruction, phase, boundary, and tail losses. |
| `dna_residual_store.py` | Stores and retrieves residual training data. |
| `dna_residual_tfc_framewise_conditioned.py` | Provides framewise-conditioned TFC layers. |
| `dna_residual_tfc_tdf_unet.py` | Defines the TFC-TDF U-Net backbone. |
| `harmonic_target.py` | Builds harmonic training targets. |
| `hpss_target.py` | Builds harmonic/percussive training targets. |
| `literal_residual_canary_store.py` | Reads the residual canary examples. |
| `occurrence_audio_augmentation.py` | Applies training audio augmentation. |
| `occurrence_dual_harmonic_residual_separator.py` | Defines the dual harmonic residual separator. |
| `occurrence_dual_hpss_separator.py` | Defines the selected dual HPSS separator. |
| `occurrence_dual_hpss_symmetric_alignment.py` | Aligns the separator's harmonic and percussive branches. |
| `occurrence_full_score_data.py` | Adapts the corpus reader to full-score examples. |
| `occurrence_joint_tfc_tdf_separator.py` | Defines the joint TFC-TDF separator. |
| `occurrence_subtraction_diagnostics.py` | Measures subtraction behavior during evaluation. |
| `packed_blob_shards.py` | Reads audio records from packed blob shards. |
| `pitch_relative_harmonic_gate.py` | Applies the pitch-relative harmonic gate. |
| `samplers.py` | Produces deterministic, resume-safe training batches. |
| `streaming_dataset.py` | Loads score-addressed examples lazily from the corpus. |
| `training_contract.py` | Defines the selected model and loss settings. |
