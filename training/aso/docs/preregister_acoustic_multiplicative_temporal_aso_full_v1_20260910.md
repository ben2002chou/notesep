# Acoustic multiplicative Temporal ASO full promotion v1

Frozen on 2026-09-10 after the development screen and before full training.

The original screen decision is preserved: job `11720038` missed its frozen
limit of four parent-relative losses worse than 3 dB, producing six among
1,118 events.  It is not relabeled as having passed that gate.  Four of the six
had parent SI-SDR below -20 dB; only one began above 0 dB.  Conversely, the
candidate improved paired median/mean SI-SDR by +0.567/+0.783 dB, reduced
median Off-E from 0.1417 to 0.1337, reduced median post-acoustic-endpoint energy
from 2.168e-4 to 1.417e-4, and correct acoustic coordinates beat shuffled ones
by +0.0745 dB median.  This is sufficient positive mechanism evidence to run
the already specified complete schedule, but not to claim safety or final
superiority.

The full run changes no model, data, seed, objective, optimizer, or
parameterization from the screen.  It trains from scratch for the fixed 29,058
updates on SCNS training data only.  SCNS-Eval-v2 remains unopened until the
complete 19,905-event development evaluation is frozen.

Promotion to SCNS-Eval-v2 requires all of the following on complete held-out
development:

- finite complete coverage and a nonzero correct-versus-shuffled acoustic
  coordinate effect;
- median and mean SI-SDR above the magnitude-partition parent;
- median SI-SDR at least 2.818 dB, the matched full symbolic-coordinate result;
- median Off-E no worse than 0.1392;
- fewer than 4.31% losses worse than 3 dB versus the magnitude-partition parent;
- median post-acoustic-endpoint energy below its matched parent.

Failure freezes the result and activates the preregistered raw-total carrier
successor; it does not authorize threshold or architecture sweeps on test data.
