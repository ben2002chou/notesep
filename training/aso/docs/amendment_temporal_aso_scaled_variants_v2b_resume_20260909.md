# Amendment: node-independent resume for scaled Temporal ASO v2

The three v2 full-training jobs completed all 29,058 optimizer updates and
atomically wrote their final checkpoints, but Slurm's pre-time-limit signal
interrupted the built-in held-development evaluation.  The first resumed
evaluation failed closed because `CONTRACT.json` recorded the immutable
preregistration's resolved node-local extraction path.  That path necessarily
changes across allocations even though the preregistration SHA-256 and every
scientific field are identical.

The resume check is amended to ignore only the top-level `preregistration`
path, `closure_amendment` path, and `runner_sha256` when comparing the newly
reconstructed contract with the recorded contract. Both node-local paths remain
bound by their unchanged SHA-256 fields. The original runner hash must equal
`3692b57f68881dc54d8361be047d61a27989b78e53ff4c645619e99d03d4dfa6`;
the replacement runner and this amendment are recorded separately in the final
result. The recorded contract is retained verbatim. Its
`preregistration_sha256`, all data and model-implementation hashes, model
contract, seed, schedule, initialization, and training exposure must still
match exactly.
No checkpoint, optimizer state, model output, evaluator, metric, or selection
rule is changed.  Replacement jobs resume at step 29,058 and perform only the
already-preregistered 19,905-event held-development evaluation.
