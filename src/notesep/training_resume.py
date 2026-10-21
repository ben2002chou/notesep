"""Validate ASO resumes against scientific identities rather than file locations."""
LOCATION_BINDINGS={
    'fresh_index':'fresh_index_sha256',
    'recorded_authority':'recorded_authority_sha256',
    'proposal_root':'proposal_validation_sha256',
    'warm_start':'warm_start_sha256',
    'preregistration':'preregistration_sha256',
    'closure_amendment':'closure_amendment_sha256',
}


def validate_resume_contract(current,recorded,*,active_runner_sha256,allowed_parent_runners):
    if current.get('runner_sha256')!=active_runner_sha256:raise ValueError('Active resume runner hash mismatch')
    allowed=set(allowed_parent_runners)|{active_runner_sha256}
    if recorded.get('runner_sha256') not in allowed:raise ValueError('Resume parent runner hash mismatch')
    left=dict(current);right=dict(recorded)
    for name,binding in LOCATION_BINDINGS.items():
        for value in (left,right):
            if value.get(name) is not None:
                checksum=value.get(binding)
                if not isinstance(checksum,str) or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
                    raise ValueError('Cannot relocate unbound resume input: '+name)
            value.pop(name,None)
    for name in ('runner_sha256','historical_runner_sha256'):
        left.pop(name,None);right.pop(name,None)
    if left!=right:raise ValueError('Resume scientific contract mismatch')
    # Keep historical provenance verbatim in the restored checkpoint lineage.
    return recorded
