"""Optional, pinned MuScriptor draft generation; no correction policy is applied."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import threading
from unittest.mock import patch
from .assets import sha256

MODEL_REPO = 'MuScriptor/muscriptor-large'
MODEL_REVISION = '8809fdfbed2affa7ade94a7059e746e3880720e7'
MODEL_SHA256 = 'ac4eb6ea87dfc26b6ca6b954c6b967ab87ad4c7d08e078b25214f13ed051f397'
MODEL_BYTES = 5465642136
CONFIG_SHA256 = '16bedd02b18770e43740419b0d5777f231047e96e8987f498e8a1123c39c9852'
_LOAD_LOCK = threading.Lock()


def load_muscriptor(device='auto', model_file=None):
    """Load verified local assets or fetch the exact gated model revision."""
    try:
        version = importlib.metadata.version('muscriptor')
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError('Install notesep[muscriptor-local] for local transcription') from error
    if version != '0.3.0':
        raise ValueError('Use muscriptor==0.3.0 from the muscriptor-local extra')
    from muscriptor import TranscriptionModel
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device not in ('cuda', 'mps', 'cpu'):
        raise ValueError('Transcription device must be auto, cuda, mps or cpu')
    model_file = model_file or os.environ.get('NOTESEP_MUSCRIPTOR_MODEL_FILE')
    if model_file:
        weight = Path(model_file)
        config = weight.with_name('config.json')
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        try:
            weight = Path(hf_hub_download(MODEL_REPO, 'model.safetensors', revision=MODEL_REVISION))
            config = Path(hf_hub_download(MODEL_REPO, 'config.json', revision=MODEL_REVISION))
        except GatedRepoError as error:
            raise ValueError('Accept the MuScriptor model license on Hugging Face and authenticate with hf auth login, or use a public transcription provider') from error
    if weight.stat().st_size != MODEL_BYTES or sha256(weight) != MODEL_SHA256 or sha256(config) != CONFIG_SHA256:
        raise ValueError('Pinned MuScriptor checkpoint/config identity mismatch')
    # Safetensors' direct CUDA materialization bypasses ZeroGPU's supported
    # staging path. Read checkpoint tensors on CPU, then let the unchanged SDK
    # load_state_dict/to(device) place them through PyTorch's patched transport.
    import muscriptor.transcription_model as sdk_model
    original_load = sdk_model.load_file
    def cpu_load(filename, device='cpu'):
        return original_load(filename, device='cpu')
    with _LOAD_LOCK, patch.object(sdk_model, 'load_file', cpu_load):
        model = TranscriptionModel.load_model(str(weight), device=device)
    import muscriptor
    sdk_root = Path(muscriptor.__file__).resolve().parent
    sdk_files = {str(path.relative_to(sdk_root)): sha256(path) for path in sorted(sdk_root.rglob('*.py'))}
    sdk_code_hash = hashlib.sha256(json.dumps(sdk_files, sort_keys=True).encode()).hexdigest()
    return model, {'provider': 'MuScriptor local', 'model_repo': MODEL_REPO,
                   'model_revision': MODEL_REVISION, 'model_sha256': MODEL_SHA256,
                   'config_sha256': CONFIG_SHA256, 'sdk': 'muscriptor==0.3.0', 'device': device,
                   'sdk_runtime_code_sha256': sdk_code_hash,
                   'checkpoint_load_transport': 'CPU safetensors then unchanged SDK state transfer',
                   'adapter_sha256': sha256(Path(__file__))}


def local_midi(model, audio_path, instrument=''):
    """Deterministic audio-only draft; preserve physical timing without tempo quantization."""
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
    if instrument and instrument not in MT3_FULL_PLUS_GROUP_NAMES:
        raise ValueError('Unknown MuScriptor instrument hint')
    data = model.transcribe_to_midi(audio=str(audio_path), use_sampling=False,
        temperature=1.0, cfg_coef=1.0, instruments=[instrument] if instrument else None,
        batch_size=1, no_eos_is_ok=True, beam_size=1, prelude_forcing=True,
        detect_tempo=False)
    return data
