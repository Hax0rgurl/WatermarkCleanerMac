# Third-Party Notices

## DeMark-World

The universal watermark detection and removal engine under `demark_world/` is
derived from [DeMark-World](https://github.com/linkedlist771/DeMark-World).

Copyright (c) 2025 LLinkedlist.

DeMark-World's repository currently contains an MIT `LICENSE` file even though
its README describes a different license. This distribution follows the actual
license file and includes its complete text at
`THIRD_PARTY_LICENSES/DeMark-World-MIT.txt`.

Modified files carry module-level notices where practical. Changes include
macOS-safe writable paths, deferred optional GPU imports, a small LaMa runtime,
clearer download errors, OpenCV-based metadata loading, and the
MAC WATERMARK CLEANER desktop/bootstrap experience.

## SoraWatermarkCleaner

The original project is distributed under Apache License 2.0. Its complete
license remains in `LICENSE`.

## IOPaint / LaMa

DeMark-World and SoraWatermarkCleaner include code derived from IOPaint by
Sanster and use the separately downloaded LaMa model. Existing upstream
notices and license files inside the vendored source are retained.

## uv

The macOS app package may bundle the `uv` executable to provision an isolated
Python 3.12 runtime. It is redistributed here under uv's MIT option; the
corresponding license text is included at
`THIRD_PARTY_LICENSES/uv-MIT.txt`.
