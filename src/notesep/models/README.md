# Models

Inference follows two steps: NoteGrab estimates each requested note, and ASO
refines the estimates together. Both training recipes import the shared layers
from this package.

## NoteGrab

| File | Role |
| --- | --- |
| `unet.py` | Time-frequency convolutional U-Net backbone and STFT. |
| `conditioning.py` | Add pitch and note-boundary conditioning to the backbone. |
| `separator.py` | Predict a note waveform from the conditioned mixture features. |
| `hpss_separator.py` | Combine harmonic and percussive separation branches. |
| `alignment.py` | Align the two branches and build the released separator from its saved settings. |
| `hpss.py` | Split audio into harmonic and percussive components. |
| `harmonic_gate.py` | Apply the pitch-dependent harmonic gate. |

Waveform batches have shape `[B, S]`: batch and samples. Pitch and boundary
vectors have shape `[B]`; boundaries use samples relative to each audio crop.
The model configuration saved in the checkpoint supplies its layer dimensions.
`inference.py` loads that configuration and requires an exact weight match.

## ASO

| File | Role |
| --- | --- |
| `ownership.py` | Extract mixture/proposal features and learn note ownership over time. |
| `aso.py` | Refine the raw ownership ratios using acoustic note boundaries. |
| `simplex.py` | Sparsemax projection used by the ownership layers. |

ASO takes a complex mixture spectrogram `[B, F, T]` and note estimates
`[B, N, F, T]`. It returns one complex spectrogram per note. Pitch and boundary
inputs have shape `[B, N]`; boundaries are sample positions within the passage.

`__init__.py` marks the package. `LICENSE-KUIELAB-SDX23.txt` contains the notice
for the adapted U-Net implementation.
