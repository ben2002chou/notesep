"""NoteGrab inference geometry from its saved training sampler contract.

The mixture is cropped BEFORE HPSS and the separator forward pass. Predictions
are then placed at that crop's original timestamp, not stretched or re-centred.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingWindow:
    lead_samples: int
    release_samples: int
    minimum_window_samples: int
    window_quantum: int

    @classmethod
    def from_checkpoint(cls, checkpoint):
        sampler = checkpoint["sampler_contract"]
        sampler = sampler.get("synthetic_sampler", sampler)
        return cls(**{name: int(sampler[name]) for name in cls.__dataclass_fields__})

    def bounds(self, onset, symbolic_offset, acoustic_offset, total):
        """Match V13StreamingResidualDataset._variable_window (no batch promotion).

        Acoustic release is the nonnegative distance from symbolic to acoustic
        endpoint. At inference this endpoint has the caller's provenance; it is
        not newly inferred here. Quantization slack is centred just as in training.
        """
        if not 0 <= onset <= symbolic_offset <= total:
            raise ValueError("Note bounds must use the full recording timeline")
        if not symbolic_offset <= acoustic_offset <= total:
            raise ValueError("Invalid acoustic endpoint")
        release = max(self.release_samples, acoustic_offset - symbolic_offset)
        natural = max(self.minimum_window_samples,
                      symbolic_offset - onset + self.lead_samples + release)
        desired = min(total, math.ceil(natural / self.window_quantum) * self.window_quantum)
        desired = max(symbolic_offset - onset, desired)
        minimum = max(0, symbolic_offset + release - desired)
        maximum = min(max(0, onset - self.lead_samples), total - desired)
        if minimum > maximum:
            minimum = max(0, min(onset, total - desired))
            maximum = min(max(minimum, symbolic_offset - desired), total - desired)
        start = max(0, min((minimum + maximum) // 2, total - desired))
        if not start <= onset <= symbolic_offset <= start + desired:
            raise ValueError("Training crop truncated the note")
        return start, start + desired


def place_prediction(prediction, crop_start, display_start, display_samples):
    """Place only predicted samples; preserve carry-ins and the display boundary."""
    import numpy as np
    prediction = np.asarray(prediction)
    if prediction.ndim != 1:
        raise ValueError("Expected a mono prediction")
    result = np.zeros(display_samples, dtype=prediction.dtype)
    left = max(crop_start, display_start)
    right = min(crop_start + len(prediction), display_start + display_samples)
    if right > left:
        result[left-display_start:right-display_start] = prediction[left-crop_start:right-crop_start]
    return result
