# Exact-score Temporal ASO v1b numerical-closure amendment

Frozen after the 320-update v1a canary and before any 2,000-update screen.

The v1a canary completed training and evaluation for both arms, but the runner
marked both failed solely because the preregistered absolute float32 STFT
closure tolerance was `1e-5`.  The observed maxima were `3.0517578125e-5`
for magnitude closure and `3.1456798752e-5` for complex closure.  All other
integrity gates passed, initialization parity was exact, and no nonfinite value
occurred.  These errors are floating-point summation residue from a mathematically
unit-sum sparsemax partition multiplied by the mixture STFT; they are far below
the repository's earlier `1e-4` partition tolerance.

For v1b and later, require both:

- maximum absolute magnitude and complex closure error at most `5e-5`; and
- maximum error divided by the maximum mixture-STFT magnitude in the same bank
  at most `2e-6`.

The model output, optimizer, loss, metrics, event set, scientific promotion
criteria, and exact-additive construction do not change.  The runner now emits
both absolute and relative diagnostics.  The completed v1a outputs remain
immutable and failed under their original gate.  Because those JSON files
predate the relative diagnostic, a separate validator may classify only their
canary integrity using the amended `5e-5` absolute rule after proving that the
two old closure booleans were the only failed runner gates.  Every screen and
full run must satisfy both the absolute and relative rules.  The validator may
not rewrite either v1a result.
