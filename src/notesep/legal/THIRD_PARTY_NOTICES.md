# Third-party notices

- KUIELab SDX23 TFC-TDF architecture donor: Copyright (c) 2023 IELab@ Korea
  University, MIT. Complete notice and pinned retrieval identity are stored in
  `src/model_exp/LICENSE-KUIELAB-SDX23.txt` and the adjacent provenance JSON.
  Existing module documentation describes the NoteSep adaptations.
- SoundTouchJS audio transform library: retain
  `src/notesep/studio_static/vendor/soundtouch-core-LICENSE.txt`.
- Gradio JavaScript client: retain
  `src/notesep/studio_static/vendor/gradio-client-LICENSE.txt` and its pinned
  provenance JSON.
- score-mss loader retained for research evaluation: its complete original
  AGPL license is `research/third_party/score-mss/LICENSE`.
- peaqb-fast perceptual metric: retain
  `research/third_party/peaqb-fast/COPYING` and its source notices.
- JUCE 8.0.9, commit `f72bad64d29715216226685810c5196bd0d79d77`:
  native builds use its AGPL license option. The source preparation copies the
  pinned checkout's license files into `native/dependency-notices/JUCE/`.
  JUCE's main `LICENSE.md` enumerates its embedded dependencies, including VST3,
  AudioUnitSDK, FLAC, Ogg Vorbis, PNG, zlib, CHOC, HarfBuzz and SheenBidi.
  Preserve their original notices; native source releases include the exact
  pinned dependency source separately, rather than relicensing it.

Python dependencies are fetched by the package installer and retain their own
distribution licenses. Version locks identify the verified environments; they
are not a blanket license for those dependencies. Dataset/library/score credits
and checkpoint terms are separate from this source-code notice.
