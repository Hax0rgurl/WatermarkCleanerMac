# Batch Processing Benchmark Results

## Ablation Study

| Detector | Batch | Cleaner | TorchCompile | Bf16 | Time (s) | Speedup |
|:--------:|:-----:|:-------:|:------------:|:----:|:--------:|:-------:|
| YOLO     | ×     | LAMA    | ×            | ×    | 44.33    | -       |
| YOLO     | ×     | E2FGVI  | ×            | ×    | 142.42   | 1.00×   |
| YOLO     | ×     | E2FGVI  | ✓            | ×    | 117.19   | 1.22×   |
| YOLO     | 4     | E2FGVI  | ✓            | ×    | 82.63    | 1.72×   |
| YOLO     | 4     | E2FGVI  | ✓            | ✓    | 58.60    | 2.43×   |

> **Note**: 
> - Speedup is calculated relative to the E2FGVI baseline (142.42s).
> - LAMA is a different cleaner and not directly comparable.
> - When enabling both bf16 and torch.compile, the first inference may be very slow (~90s) due to compilation overhead. Subsequent inferences will be significantly faster (~58s) as the compiled artifacts are cached.
