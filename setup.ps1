# One-time setup for the local, commercial-safe voice + animation stack.
# Run with:  powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Why a virtual environment exists here at all:
# this machine only has Python 3.14, but Kokoro's dependency chain has no 3.14
# wheels. `kokoro` pins numpy==1.26.4 and its `misaki[en]` extra imports spaCy,
# and neither can be built without a C toolchain. Rather than patch around that
# (which would mean hand-rolling a phonemizer), this script provisions a
# dedicated Python 3.12 environment with uv. uv downloads its own interpreter
# and needs no administrator rights.
#
# Remotion (Node) and Rhubarb (a standalone binary) are independent of all this.

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
# Console encoding: the G2P produces IPA (e.g. ˈ, ʃ, ɹ) which the default
# cp1252 console cannot encode.
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$venv = Join-Path $PSScriptRoot '.venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'

Write-Host "project root: $PSScriptRoot"

function Get-UvPath {
    $scriptsDir = (& py -c "import sysconfig; print(sysconfig.get_path('scripts'))").Trim()
    return (Join-Path $scriptsDir 'uv.exe')
}

Write-Host ''
Write-Host '=== [1/4] Installing uv (Python toolchain manager) ==='
$uv = Get-UvPath
if (Test-Path $uv) {
    Write-Host "  uv already present"
} else {
    py -m pip install --disable-pip-version-check --quiet uv
    Write-Host "  pip exit code: $LASTEXITCODE"
    $uv = Get-UvPath
}
if (-not (Test-Path $uv)) {
    Write-Host "  ERROR: uv not found at $uv"
    exit 1
}
& $uv --version

Write-Host ''
Write-Host '=== [2/4] Provisioning Python 3.12 in .venv ==='
if (Test-Path $venvPy) {
    Write-Host '  .venv already exists, reusing it'
} else {
    & $uv venv $venv --python 3.12 --seed
    Write-Host "  uv venv exit code: $LASTEXITCODE"
}
if (-not (Test-Path $venvPy)) {
    Write-Host "  ERROR: no interpreter at $venvPy"
    exit 1
}
& $venvPy --version

Write-Host ''
Write-Host '=== [3/4] Installing Kokoro TTS (Apache-2.0) ==='
# Use the venv's own pip rather than `uv pip`: uv does not reach PyPI from this
# environment (its requests time out), while pip does.

& $venvPy -m pip install --disable-pip-version-check --quiet torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    Write-Host '  CPU index unavailable, retrying against the default index'
    & $venvPy -m pip install --disable-pip-version-check --quiet torch
}
Write-Host "  torch exit code: $LASTEXITCODE"

# Stage the install instead of letting the resolver pick for kokoro. Left to
# itself it selects transformers 4.12.2, whose tokenizers==0.10.3 pin has no
# Python 3.12 wheel and cannot be built without a Rust toolchain. Installing a
# current transformers/tokenizers first, then adding kokoro without dependency
# resolution, keeps every package on a prebuilt wheel.
& $venvPy -m pip install --disable-pip-version-check --quiet numpy soundfile loguru huggingface-hub scipy attrs
Write-Host "  runtime deps exit code: $LASTEXITCODE"
& $venvPy -m pip install --disable-pip-version-check --quiet --only-binary ":all:" "transformers>=4.46" "tokenizers>=0.20"
Write-Host "  transformers exit code: $LASTEXITCODE"
& $venvPy -m pip install --disable-pip-version-check --quiet --only-binary ":all:" misaki spacy espeakng_loader num2words phonemizer
Write-Host "  misaki + spacy exit code: $LASTEXITCODE"
& $venvPy -m pip install --disable-pip-version-check --quiet --no-deps kokoro
Write-Host "  kokoro exit code: $LASTEXITCODE"

Write-Host ''
Write-Host '=== [4/4] Rhubarb Lip Sync (MIT) ==='
New-Item -ItemType Directory -Force -Path 'tools' | Out-Null
$zip = Join-Path $env:TEMP 'rhubarb-1.14.0.zip'
if (-not (Test-Path $zip)) {
    Invoke-WebRequest -UseBasicParsing -OutFile $zip `
        -Uri 'https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/Rhubarb-Lip-Sync-1.14.0-Windows.zip'
}
Expand-Archive -Path $zip -DestinationPath 'tools\rhubarb' -Force
$rhubarb = (Get-ChildItem -Recurse 'tools\rhubarb' -Filter 'rhubarb.exe' | Select-Object -First 1).FullName
Write-Host "  rhubarb: $rhubarb"

Write-Host ''
Write-Host '=== Verification ==='
& $venvPy -c "import sys, torch, numpy, soundfile; print('  python    :', sys.version.split()[0]); print('  torch     :', torch.__version__); print('  numpy     :', numpy.__version__); print('  soundfile :', soundfile.__version__)"
& $venvPy -c "from misaki import en; g = en.G2P(); print('  misaki G2P:', g('The ocean floor is not a flat plain.')[0][:52])"
& $venvPy -c "import kokoro; print('  kokoro    : import OK')"
& $rhubarb --version 2>&1 | ForEach-Object { Write-Host "  $($_.ToString().Trim())" }

Write-Host ''
Write-Host '=== SETUP COMPLETE ==='
