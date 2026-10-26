# Third-party notices

The app's original code retains Apache License 2.0 in `LICENSE`. This does not relicense the separately attributed components below. The combined distribution includes noncommercial restoration backends.

## E2FGVI-HQ

E2FGVI-HQ is by Zhen Li, Cheng-Ze Lu, Jianhua Qin, Chun-Le Guo, and Ming-Ming Cheng, MCG-NKU. Copyright (c) 2022 MCG-NKU.

- Source: [MCG-NKU/E2FGVI](https://github.com/MCG-NKU/E2FGVI/tree/709cbe319edc21b8a365a28e14cba595a93d62cf).
- Pinned revision: `709cbe319edc21b8a365a28e14cba595a93d62cf`.
- License: **Creative Commons Attribution-NonCommercial 4.0 International**. Full text: `watermark_cleaner/backends/e2fgvi/LICENSE`.
- Modifications: package-relative imports; a Torchvision CPU deformable-convolution adapter; equivalent small Torch helpers in place of MMCV/MMEngine; removal of training-only download paths and redundant SPyNet downloads. The full E2FGVI-HQ checkpoint already contains SPyNet weights.

The E2FGVI-HQ checkpoint comes from a SHA-256-verified copy in [VIPLab/Track-Anything](https://huggingface.co/spaces/VIPLab/Track-Anything/tree/bb879e543c98747057ec76529bcdb3232c22f7fd/checkpoints). The pinned mirror revision is `bb879e543c98747057ec76529bcdb3232c22f7fd`. Its official-checkpoint provenance is recorded in `watermark_cleaner/backends/e2fgvi/UPSTREAM.md`.

## ProPainter

ProPainter is by Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, and Chen Change Loy, S-Lab, Nanyang Technological University. Copyright 2023 S-Lab.

- Source: [sczhou/ProPainter](https://github.com/sczhou/ProPainter/tree/e870e79321c31b733e2031af5aa2fb1fe3ac7eec).
- Pinned revision: `e870e79321c31b733e2031af5aa2fb1fe3ac7eec`.
- License: **S-Lab License 1.0**. It permits noncommercial redistribution and use subject to its conditions. The license gives upstream contacts for commercial permission. Full text: `watermark_cleaner/backends/propainter/LICENSE`.
- Modifications: package-relative imports; removal of unused training and optional CUDA correlation paths; verified local checkpoint loading; current explicit meshgrid/autocast APIs; a working Torchvision implementation of the base deformable layer. The pretrained inference architecture is retained.

ProPainter, recurrent flow completion, and RAFT checkpoints are downloaded from [ProPainter release v0.1.0](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0). Source and model use remain subject to upstream terms.

## RAFT

The ProPainter optical-flow modules include code from [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT), copyright (c) 2020 princeton-vl. These files come from the pinned ProPainter tree above. Their BSD 3-Clause license is retained at `watermark_cleaner/backends/propainter/raft/LICENSE`.

## Download integrity

These checkpoint hashes are fixed in `watermark_cleaner/model_store.py`. Files are accepted only after their size and SHA-256 match. Downloaded footage is not sent to these hosts.

| File | SHA-256 |
|---|---|
| E2FGVI-HQ-CVPR22.pth | `afff989d41205598a79ce24630b9c83af4b0a06f45b137979a25937d94c121a5` |
| raft-things.pth | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` |
| recurrent_flow_completion.pth | `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283` |
| ProPainter.pth | `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591` |

## Retained project attribution

The repository originated from SoraWatermarkCleaner, distributed under Apache License 2.0. Its license remains in `LICENSE`.

Earlier versions adapted DeMark-World and related timing/validation utilities. DeMark-World's MIT notice, copyright (c) 2025 LLinkedlist, is retained in `THIRD_PARTY_LICENSES/DeMark-World-MIT.txt`. The DeMark engine, LaMa restoration code/models, and Sora detector packages are no longer bundled or invoked by version 2. Historical attribution does not represent an active dependency.

## uv and Python dependencies

The macOS app bundles the official `uv` executable to provision a private Python runtime. It is redistributed under uv's MIT option; full text is in `THIRD_PARTY_LICENSES/uv-MIT.txt`.

The separately installed Python dependencies retain their upstream distribution licenses and notices in the private runtime. The fully resolved versions and distribution hashes are in `requirements-macos.lock`.
