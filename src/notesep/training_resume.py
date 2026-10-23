"""Check that an ASO resume uses the same data and training settings."""

LOCATION_BINDINGS = {
    "fresh_index": "fresh_index_sha256",
    "recorded_authority": "recorded_authority_sha256",
    "proposal_root": "proposal_validation_sha256",
    "warm_start": "warm_start_sha256",
}


def validate_resume_contract(current, recorded):
    left = dict(current)
    right = dict(recorded)
    for name, binding in LOCATION_BINDINGS.items():
        for value in (left, right):
            if value.get(name) is not None:
                checksum = value.get(binding)
                if (
                    not isinstance(checksum, str)
                    or len(checksum) != 64
                    or any(c not in "0123456789abcdef" for c in checksum)
                ):
                    raise ValueError("Cannot relocate unbound resume input: " + name)
            value.pop(name, None)
    if left != right:
        raise ValueError("Resume scientific contract mismatch")
    return recorded
