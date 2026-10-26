# Desktop workflow

The native window selects local video files and passes identity-pinned paths to the cleaner. The main screen offers E2FGVI-HQ and ProPainter, with E2FGVI-HQ selected initially. First-launch setup prepares both model sets. If native setup fails, diagnostics show the error; reopening retries while keeping verified downloads. The source interface also offers a model-setup retry button. Removal remains unavailable until preparation succeeds.

Choose videos and press **REMOVE WATERMARKS**. Common platform watermarks are found automatically. The optional watermark-name field narrows a scan to a specific brand. There is no paid service or footage upload in this workflow.

**Review or adjust watermark times** opens the source video, scrubber, automatic scan button, and optional correction controls. **Set start here** and **Set stop here** use the actual decoded frame timestamps, including variable-frame-rate videos. The numerical controls are for optional calibration.

Corrections are saved against the selected source identity and watermark-name scope. Changing the source or requested name invalidates the old selection. Clearing every manually selected range requires a new selection or scan before processing, instead of silently restoring a removed range.

The pipeline verifies a complete output before the interface lists it as completed. Original files are retained. Handled job failures are quarantined in the app's Failed Jobs directory for diagnosis. An interrupted job is not offered as a completed download.

The model repositories include noncommercial license conditions. Their complete notices are bundled with the app and published source.
