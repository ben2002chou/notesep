import argparse
import json
from pathlib import Path

from .api import NoteBank, NoteSep
from .assets import download, verify, PROFILES


def main(argv=None):
    parser = argparse.ArgumentParser(description="NoteSep music editing tools")
    sub = parser.add_subparsers(dest='command', required=True)
    for command, help_text in [('download', 'Download the paper checkpoint pair'),
                               ('verify', 'Verify checkpoint SHA-256 values')]:
        item = sub.add_parser(command, help=help_text)
        item.add_argument('--profile', choices=list(PROFILES))
    training_data = sub.add_parser('download-data', help='Download and verify training data')
    training_data.add_argument('--output', type=Path, required=True)
    training_data.add_argument('--repo-id', help='Hugging Face dataset repository')
    training_data.add_argument('--revision', help='Immutable dataset commit')
    training_data.add_argument('--manifest', type=Path, help='Verify a local dataset')
    separate = sub.add_parser('separate', help='Separate an aligned MIDI/JSON score')
    separate.add_argument('audio', type=Path)
    separate.add_argument('--score', type=Path, required=True)
    separate.add_argument('--output', type=Path, required=True)
    separate.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    separate.add_argument('--notegrab-only', action='store_true')
    separate.add_argument('--profile', choices=list(PROFILES))
    separate.add_argument('--window-start', type=float, help='Select a window starting at this recording time')
    separate.add_argument('--window-seconds', type=float, help='Selected window duration, 1.1–12 seconds')
    separate.add_argument('--score-timeline', choices=['recording', 'window'], default='recording')
    baseline = sub.add_parser('baseline', help='Run the paper float64 score-informed NMF function')
    baseline.add_argument('audio', type=Path)
    baseline.add_argument('--score', type=Path, required=True)
    baseline.add_argument('--output', type=Path, required=True)
    baseline.add_argument('--iterations', type=int, default=1000)
    studio = sub.add_parser("studio", help="Open the local piano-roll editor")
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=7860)
    studio.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    studio.add_argument('--profile', choices=list(PROFILES))
    render = sub.add_parser('render', help='Render edits from a saved note project')
    render.add_argument('project', type=Path)
    render.add_argument('--edits', type=Path, help='Override the project\'s saved edit controls')
    render.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ('download', 'verify'):
            print(json.dumps((download if args.command == 'download' else verify)(profile=args.profile), indent=2))
        elif args.command == 'download-data':
            from .training_data import prepare
            count, path_map = prepare(args.output, repo_id=args.repo_id,
                                      revision=args.revision, manifest_path=args.manifest)
            print(f'Verified {count} dataset files in {args.output.expanduser().resolve()}')
            if path_map is not None:
                print(f'Training path map: {path_map}')
        elif args.command == 'separate':
            bank = NoteSep(args.device, profile=args.profile).separate(args.audio, args.score,
                apply_aso=not args.notegrab_only, window_start=args.window_start,
                window_seconds=args.window_seconds, score_timeline=args.score_timeline)
            print(bank.export(args.output))
        elif args.command == 'baseline':
            from .baselines import separate_baseline
            bank = separate_baseline(args.audio, args.score, max_iter=args.iterations)
            print(bank.export(args.output))
        elif args.command == 'studio':
            import os
            os.environ['NOTESEP_DEVICE'] = NoteSep(args.device).device
            if args.profile:
                os.environ['NOTESEP_PROFILE'] = args.profile
            from .studio import launch
            launch(args.host, args.port)
        elif args.command == 'render':
            import soundfile as sf
            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            bank = NoteBank.load(args.project)
            edits=json.loads(args.edits.read_text()) if args.edits else bank.saved_edits
            sf.write(args.output, bank.render(edits),
                     bank.sample_rate, subtype='FLOAT')
            print(args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"notesep: {error}\n")


if __name__ == '__main__':
    main()
