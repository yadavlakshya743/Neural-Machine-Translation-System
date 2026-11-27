# Train + evaluate the small baseline Transformer for both directions on a local GPU.
# Safe to re-run: training resumes from models/<run>/last.pt and finished runs are skipped.
#
#   powershell -ExecutionPolicy Bypass -File scripts\train_baseline_local.ps1
#
# Do not run the FastAPI backend on the GPU at the same time (it can cause out-of-memory).

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:OPENBLAS_NUM_THREADS = "1"   # OpenBLAS reserves a big buffer per thread; saves RAM
$py = ".\.venv\Scripts\python.exe"
$config = "ml/configs/baseline_config.yaml"
$log = "models\baseline_local.log"
New-Item -ItemType Directory -Force models | Out-Null

foreach ($dir in @("en-hi", "hi-en")) {
    $run = "models\transformer_small_$dir"
    "==== $(Get-Date -Format s) training $dir ====" | Out-File -Append -Encoding utf8 $log
    & $py scripts/train.py --config $config --direction $dir --resume auto 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    if (Test-Path "$run\best.pt") {
        "==== $(Get-Date -Format s) evaluating $dir ====" | Out-File -Append -Encoding utf8 $log
        & $py scripts/evaluate.py --checkpoint "$run\best.pt" --strategy greedy 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
        & $py scripts/evaluate.py --checkpoint "$run\best.pt" --strategy beam_search --beam_size 4 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    }
}
"==== $(Get-Date -Format s) ALL DONE ====" | Out-File -Append -Encoding utf8 $log
