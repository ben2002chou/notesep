"""Sample native instrument libraries for NoteGrab's 250-update adaptation."""

from pathlib import Path

import numpy as np

from model_exp.samplers import ResumeSafeAudioBudgetBatchSampler


NATIVE_SOURCES = {
    "bengt_headroom_piano", "dsmolken_double_bass", "fluidr3_gm",
    "generaluser_gs", "ixox_flute", "jlearman_steel_drum",
    "mtg_solo_sax", "osiris_piano",
}


class NativeLibrarySampler:
    """Balance libraries, then draw duration-matched rows within each library."""

    index_path: Path

    def __init__(self, window_samples, **kwargs):
        windows = np.asarray(window_samples, dtype=np.int64)
        with np.load(self.index_path, allow_pickle=False) as arrays:
            names = [str(name) for name in arrays["source_names"].tolist()]
            source_ids = np.asarray(arrays["source_id"], dtype=np.int16)
        if len(windows) != len(source_ids):
            raise ValueError("Render-method index and training row selection differ")
        sources = np.asarray([names[index] for index in source_ids], dtype="U64")
        self.eligible = np.flatnonzero(
            np.isin(sources, sorted(NATIVE_SOURCES))
        ).astype(np.int32)
        if not len(self.eligible):
            raise ValueError("Native instrument libraries are absent")
        selected = sources[self.eligible]
        families = {
            name: np.flatnonzero(selected == name).astype(np.int32)
            for name in sorted(set(selected.tolist()))
        }
        kwargs.update(family_temperature_alpha=0.0, family_ordinals=families)
        self.window_samples = windows
        self.delegate = ResumeSafeAudioBudgetBatchSampler(
            windows[self.eligible], **kwargs
        )

    def __len__(self):
        return len(self.delegate)

    def __iter__(self):
        for batch in self.delegate:
            yield [
                (int(self.eligible[int(local)]), int(samples))
                for local, samples in batch
            ]


def install(trainer, index_path):
    """Keep the inherited trainer and change only the synthetic row sampler."""
    NativeLibrarySampler.index_path = Path(index_path)
    previous_contract = trainer.sampler_contract

    def sampler_contract(args):
        return previous_contract(args) | {
            "strategy": "scns_v3_render_mechanism_parameterized_v2",
            "allocation_unit": "model_rows_within_batch",
            "true_synth_fraction": 0.0,
            "native_sampler_fraction": 1.0,
            "adapted_score_bound_fraction": 0.0,
            "native_sampler_sources": sorted(NATIVE_SOURCES),
            "excluded_legacy_adapted_sources": ["nsynth_instruments", "tiny_sol"],
            "within_native_sampler": "uniform_source_library_then_rows",
            "within_adapted_score_bound": "uniform_source_library_then_least_used_physical_source",
            "new_source_bonus": False,
        }

    trainer.ResumeSafeAudioBudgetBatchSampler = NativeLibrarySampler
    trainer.sampler_contract = sampler_contract
