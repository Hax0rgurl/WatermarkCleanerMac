# Version 2 acceptance checks

- Only E2FGVI-HQ and ProPainter are selectable, E2FGVI-HQ defaults.
- No DeMark/LaMa/Sora restoration package is installed or called. Retain license notices required for reused utilities.
- First launch installs an isolated runtime and downloads both verified model sets with visible progress, retries safely, and reuses valid models on later launches.
- Select video, choose an engine, remove. Automatic named-brand detection and glyph masks stay local; visual timed corrections remain optional.
- Restoration runs native local CPU video models, bounded chunks, consistent shots and moving watermark locations.
- Preserve source, all decoded frames, exact variable timestamps, dimensions and original audio. Decode and verify output before reporting completion.
- Unit, integration, real pretrained tests for both engines, clean-runtime setup, app bundle self-test and strict signature validation pass.
- Push actual code, license notices, setup instructions and release artifacts to the existing user repository. Inspect remote CI result.
- Install the verified newest app, then clean obsolete app copies and generated caches without deleting sources, authored edits, or model caches needed by the app.
