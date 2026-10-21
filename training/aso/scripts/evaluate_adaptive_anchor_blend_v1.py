#!/usr/bin/env python3
"""Frozen target-blind exclusive-anchor blend; SLURM development test only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

NFFT, HOP, SR = 2048, 512, 16000
PINS = {
    'bank': '943db15436dc281a36bc895391ae8bcd79165d3aa9c37de9a5b4f21d9ba4d373',
    'bank_validation': '1dedeca5258493c4dd76ed2f6be3b220c4bd67c4988c8f0b95e144c899d30a72',
    'controls': '14e99fa73025e7fd67322254e39f38a03c4a5a90ed75d89e8a2e15aad9d8657d',
    'controls_validation': '3b9818984df5760b772170a5aecd4f4782890adbf3d37f0641c29db1a3e767ac',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def stft(y):
    return torch.stft(y, NFFT, HOP, window=torch.hann_window(NFFT, device=y.device),
                      center=True, pad_mode='constant', return_complex=True)


def istft(y, length):
    return torch.istft(y, NFFT, HOP, window=torch.hann_window(NFFT, device=y.device),
                       center=True, length=length)


def partition(amplitude, mixture_stft, length):
    den = amplitude.sum(0, keepdim=True)
    weights = torch.where(den > 1e-20, amplitude / den.clamp_min(1e-20),
                          torch.zeros_like(amplitude))
    return istft(weights * mixture_stft[None], length).float()


def anchor_policy(raw_stft, pitches, onsets, midi_offsets, shifted=False):
    """No reference/target argument or access; returns amplitudes and exact mask."""
    amplitude = raw_stft.abs()
    power = amplitude.square()
    proposed = amplitude.clone()
    modified = torch.zeros_like(amplitude, dtype=torch.bool)
    freq = torch.arange(NFFT // 2 + 1, device=raw_stft.device) * SR / NFFT
    centers = torch.arange(amplitude.shape[-1], device=raw_stft.device) * HOP
    f0s = [440 * 2 ** ((int(p) - 69) / 12) for p in pitches]
    all_bands, anchor_bands, band_geometry = [], [], []
    for f0 in f0s:
        bands = []
        for h in range(1, int((SR / 2) // f0) + 1):
            hz = h * f0
            width = max(2 * SR / NFFT, hz * (2 ** (35 / 1200) - 1))
            bands.append((h, hz, (freq - hz).abs() <= width, width))
        all_bands.append(torch.stack([b[2] for b in bands]).any(0)
                         if bands else torch.zeros_like(freq, dtype=torch.bool))
        anchor_bands.append([(h, mask) for h, hz, mask, width in bands if h <= 16 and hz < 7500])
        band_geometry.append({h: (hz, width) for h, hz, mask, width in bands})
    boundaries = sorted(set([int(v) for v in onsets] + [int(v) for v in midi_offsets]))
    segments = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        active = [i for i in range(len(pitches)) if int(onsets[i]) <= start < int(midi_offsets[i])]
        frames = torch.where((centers >= start + 1024) & (centers < stop - 1024))[0]
        record = dict(start=start, stop=stop, active=active, frames=frames.tolist())
        if len(active) < 2 or len(frames) < 8:
            record['status'] = 'insufficient_polyphony_or_interior'
            segments.append(record)
            continue
        candidate_support = {i: torch.stack([b for _, b in anchor_bands[i]]).any(0)
                             if anchor_bands[i] else torch.zeros_like(freq, dtype=torch.bool)
                             for i in active}
        participation = torch.stack([candidate_support[i] for i in active])
        contested = participation.sum(0) >= 2
        anchors, envelopes = [], []
        for i in active:
            choices = []
            for h, band in anchor_bands[i]:
                hz, width = band_geometry[i][h]
                overlap = any(abs(hz - other_hz) <= width + other_width
                    for j in active if j != i for other_hz, other_width in band_geometry[j].values())
                if not overlap and bool(band.any()):
                    env = power[i][band][:, frames].sum(0)
                    choices.append((float(env.mean()), h, env, band))
            if not choices:
                break
            strength, harmonic, envelope, exclusive = max(choices, key=lambda v: (v[0], -v[1]))
            if strength <= 0:
                break
            envelope = envelope / strength
            if shifted:
                envelope = torch.roll(envelope, len(frames) // 2)
            anchors.append(dict(note=i, harmonic=harmonic, mean_power=strength,
                                exclusive_bins=torch.where(exclusive)[0].tolist()))
            envelopes.append(envelope)
        if len(anchors) != len(active):
            record['status'] = 'abstain_missing_anchor'
            segments.append(record)
            continue
        # Blend is zero at first/last eligible frame center, unity >=20 ms inward.
        distance = torch.minimum(centers[frames] - centers[frames[0]],
                                 centers[frames[-1]] - centers[frames]).float()
        blend = .5 - .5 * torch.cos(torch.pi * (distance / 320).clamp(0, 1))
        for i, envelope in zip(active, envelopes):
            bins = torch.where(contested & candidate_support[i])[0]
            if not len(bins):
                continue
            old_power = power[i][bins[:, None], frames[None, :]]
            predicted_power = old_power.mean(1, keepdim=True) * envelope[None]
            candidate = (old_power * predicted_power).clamp_min(0).pow(.25)
            old_amp = amplitude[i][bins[:, None], frames[None, :]]
            proposed[i][bins[:, None], frames[None, :]] = old_amp + blend[None] * (candidate - old_amp)
            modified[i][bins[:, None], frames[None, :]] = blend[None] > 0
        record.update(status='eligible', anchors=anchors, contested_bins=int(contested.sum()))
        segments.append(record)
    if not torch.equal(proposed[~modified], amplitude[~modified]):
        raise ValueError('amplitude changed outside support')
    return proposed, modified, segments


def metrics(target, estimate):
    t, e = target.double(), estimate.double()
    energy = t.square().sum(-1)
    if bool((energy <= 1e-12).any()):
        raise ValueError('silent target')
    gain = (e * t).sum(-1) / energy
    off = (e - gain[:, None] * t).square().sum(-1) / energy
    return dict(si_sdr_db=10 * torch.log10(gain.square().clamp_min(1e-30) / off.clamp_min(1e-30)),
                off_e=off, gain=gain,
                scale_invariant_distortion=off / gain.square().clamp_min(1e-30))


def unit_fixtures(device='cpu'):
    """Run inside compute allocation; no scientific target or outcome selection."""
    raw = torch.ones((2, NFFT // 2 + 1, 48), dtype=torch.complex64, device=device)
    for shifted in (False, True):
        unchanged, mask, segments = anchor_policy(raw, [60, 72], [0, 0], [24000, 24000], shifted)
        if bool(mask.any()) or not torch.equal(unchanged, raw.abs()):
            raise ValueError('octave abstention fixture')
        if not any(s['status'] == 'abstain_missing_anchor' for s in segments):
            raise ValueError('octave fixture did not reach anchor check')
        unchanged, mask, _ = anchor_policy(raw, [60, 67], [0, 0], [100, 100], shifted)
        if bool(mask.any()) or not torch.equal(unchanged, raw.abs()):
            raise ValueError('no eligible region identity fixture')
    return dict(octave_abstention=True, no_eligible_amplitude_identity=True,
                normal_and_shifted=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    for key in ('bank', 'controls', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('SLURM compute required')
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    fixtures = unit_fixtures(device)
    paths = {'bank': args.bank / 'VALIDATION.json', 'bank_validation': args.bank / 'INDEPENDENT_VALIDATION.json',
             'controls': args.controls / 'result.json', 'controls_validation': args.controls / 'INDEPENDENT_VALIDATION.json'}
    for key, path in paths.items():
        if sha(path) != PINS[key]:
            raise ValueError('authority mismatch: ' + key)
    bank = json.loads(paths['bank'].read_text())
    controls = json.loads(paths['controls'].read_text())
    cv = json.loads(paths['controls_validation'].read_text())
    if len(bank['records']) != 48 or len(controls['events']) != 128:
        raise ValueError('coverage')
    args.output.mkdir(parents=True, exist_ok=False)
    events, banks, archives, coverage, runtime = [], [], [], [], []
    max_raw_parity = max_mag_parity = 0.
    for number, rec in enumerate(bank['records']):
        if controls['banks'][number]['scene_id'] != rec['scene_id']:
            raise ValueError('control scene ordering')
        source = Path(rec['path'])
        control_path = args.controls / f'scene_{number:04d}.npz'
        if sha(source) != rec['sha256'] or sha(control_path) != cv['output_sha256'][control_path.name]:
            raise ValueError('archive hash')
        with np.load(source, allow_pickle=False) as z:
            x, targets, raw = [torch.from_numpy(z[k].copy()).to(device) for k in ('mixture', 'targets', 'estimates')]
            pitches, onsets, offsets = [z[k].copy() for k in ('pitches', 'onsets', 'midi_offsets')]
        with np.load(control_path, allow_pickle=False) as z:
            old = {k: torch.from_numpy(z[k].copy()).to(device) for k in ('raw', 'magnitude', 'lifecycle')}
        X, R = stft(x), stft(raw)
        outputs = dict(raw=raw, magnitude=partition(R.abs(), X, len(x)), lifecycle=old['lifecycle'])
        rp = float((raw - old['raw']).abs().max())
        mp = float((outputs['magnitude'] - old['magnitude']).abs().max())
        max_raw_parity, max_mag_parity = max(max_raw_parity, rp), max(max_mag_parity, mp)
        if max(rp, mp) > 1e-6:
            raise ValueError(f"control waveform parity scene={rec['scene_id']} index={number} raw={rp:.17g} magnitude={mp:.17g}")
        for label, shifted in (('candidate', False), ('shifted', True)):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.monotonic()
            amp, mask, segments = anchor_policy(R, pitches, onsets, offsets, shifted)
            outputs[label] = partition(amp, X, len(x))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            runtime.append(dict(scene_id=rec['scene_id'], method=label, seconds=time.monotonic() - start))
            coverage.append(dict(scene_id=rec['scene_id'], method=label, segments=segments,
                modified_note_bins=int(mask.sum()), total_note_bins=mask.numel(),
                modified_notes=int(mask.flatten(1).any(1).sum())))
        vals = {k: metrics(targets, e) for k, e in outputs.items()}
        for e in outputs.values():
            if e.shape != targets.shape or not bool(torch.isfinite(e).all()):
                raise ValueError('waveform invalid')
        for i in range(len(targets)):
            events.append(dict(scene_id=rec['scene_id'], note=i, instrument=rec['instrument'],
                metrics={k: {m: float(v[i]) for m, v in value.items()} for k, value in vals.items()}))
        banks.append(dict(scene_id=rec['scene_id'], metrics={k: float((e.double().sum(0) - x.double()).square().sum() / x.double().square().sum()) for k, e in outputs.items()}))
        dest = args.output / f'scene_{number:04d}.npz'
        np.savez(dest, **{k: v.cpu().numpy() for k, v in outputs.items()})
        archives.append(dict(path=str(dest), sha256=sha(dest), scene_id=rec['scene_id'],
                             input_path=str(source), input_sha256=rec['sha256'],
                             control_path=str(control_path), control_sha256=sha(control_path)))
        print(rec['scene_id'], flush=True)
    if len(events) != 128:
        raise ValueError('event coverage')
    summary = {}
    for method in outputs:
        summary[method] = {key: dict(median=float(np.median([e['metrics'][method][key] for e in events])),
                                    mean=float(np.mean([e['metrics'][method][key] for e in events]))) for key in vals[method]}
        summary[method]['bank_error'] = dict(median=float(np.median([b['metrics'][method] for b in banks])), mean=float(np.mean([b['metrics'][method] for b in banks])))
        summary[method]['absolute_gain_error'] = dict(median=float(np.median([abs(e['metrics'][method]['gain'] - 1) for e in events])), mean=float(np.mean([abs(e['metrics'][method]['gain'] - 1) for e in events])))
    comparisons = {}
    for method in ('candidate', 'shifted'):
        comparisons[method] = {}
        for baseline in ('raw', 'magnitude', 'lifecycle'):
            comparisons[method][baseline] = dict(
                regressions_gt_0p1db=sum(e['metrics'][method]['si_sdr_db'] < e['metrics'][baseline]['si_sdr_db'] - .1 for e in events),
                regressions_gt_3db=sum(e['metrics'][method]['si_sdr_db'] < e['metrics'][baseline]['si_sdr_db'] - 3 for e in events),
                reduction_ratios={key: {agg: summary[baseline][key][agg] / max(summary[method][key][agg], 1e-30)
                                       for agg in ('median', 'mean')}
                                  for key in ('off_e', 'scale_invariant_distortion', 'bank_error')})
            comp = comparisons[method][baseline]
            comp['mixture_precision_floor'] = summary[baseline]['bank_error']['median'] <= 1e-10
            comp['strict_gates'] = dict(
                off_e_2x=comp['reduction_ratios']['off_e']['median'] >= 2,
                si_distortion_2x=comp['reduction_ratios']['scale_invariant_distortion']['median'] >= 2,
                no_note_loss_gt_0p1db=comp['regressions_gt_0p1db'] == 0,
                gain_error_not_worse=summary[method]['absolute_gain_error']['median'] <= summary[baseline]['absolute_gain_error']['median'],
                bank_error_10x_arithmetic=comp['reduction_ratios']['bank_error']['median'] >= 10,
                bank_error_10x_interpretable=not comp['mixture_precision_floor'] and comp['reduction_ratios']['bank_error']['median'] >= 10)
    magnitude_raw_losses = sum(e['metrics']['magnitude']['si_sdr_db'] < e['metrics']['raw']['si_sdr_db'] - .1 for e in events)
    directional = dict(si_sdr_0p3db=summary['candidate']['si_sdr_db']['median'] >= summary['magnitude']['si_sdr_db']['median'] + .3,
        off_e_10percent=summary['candidate']['off_e']['median'] <= .9 * summary['magnitude']['off_e']['median'],
        no_more_raw_regressions=comparisons['candidate']['raw']['regressions_gt_0p1db'] <= magnitude_raw_losses,
        beats_shifted=summary['candidate']['si_sdr_db']['median'] > summary['shifted']['si_sdr_db']['median'])
    result = dict(status='completed_pending_independent_validation', role='development diagnostic; not score-held-out',
        summary=summary, comparisons=comparisons, events=events, banks=banks, archives=archives,
        coverage=coverage, runtime_seconds=runtime, directional_gates=directional, directional_pass=all(directional.values()),
        final_goal_proven=False, final_goal_limitation='Development-only sources/scores; magnitude consistency is at numerical floor.',
        authority={k: dict(path=str(paths[k]), sha256=v) for k, v in PINS.items()},
        raw_parity_max_abs=max_raw_parity, magnitude_parity_max_abs=max_mag_parity,
        unit_fixtures=fixtures,
        script_sha256=sha(Path(__file__)), job_id=os.environ['SLURM_JOB_ID'], device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        runtime_cuda_synchronized=device.type == 'cuda',
        runtime_scope='policy and partition ISTFT; excludes shared input STFT, inference and IO')
    with (args.output / 'result.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
