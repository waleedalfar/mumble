# One-shot setup for the voice dictation app: clones and builds whisper.cpp
# (CUDA build if you have a working NVIDIA CUDA toolchain, CPU-only build
# otherwise), downloads the required models, creates the Python venv, and
# writes a starter config.yaml matched to your hardware.
#
# Safe to re-run: every step is skipped if its output already exists, so this
# also works as a "check my setup" / "resume a partial setup" script.
#
# Usage (from the project root, in PowerShell):
#   .\setup.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "    $msg" }
function Write-Warn($msg) { Write-Host "    ! $msg" -ForegroundColor Yellow }

# ---- 1. Prerequisites --------------------------------------------------

Write-Step "Checking prerequisites"

function Test-Command($name) {
    try { Get-Command $name -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

if (-not (Test-Command "git")) {
    Write-Error "git not found. Install it from https://git-scm.com/downloads and re-run this script."
}
if (-not (Test-Command "python")) {
    Write-Error "python not found. Install Python 3.10+ from https://www.python.org/downloads/ and re-run this script."
}
Write-Info "git and python found."

# cmake: prefer one already on PATH, else the copy bundled with Visual Studio
$cmake = $null
if (Test-Command "cmake") {
    $cmake = "cmake"
} else {
    $vsCmake = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "cmake.exe" `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($vsCmake) { $cmake = $vsCmake.FullName }
}
if (-not $cmake) {
    Write-Error ("cmake not found, and no Visual Studio installation with the C++ workload was found either.`n" +
        "Install 'Visual Studio Build Tools' (or Community) with the 'Desktop development with C++' workload " +
        "from https://visualstudio.microsoft.com/downloads/ and re-run this script.")
}
Write-Info "cmake: $cmake"

# ---- 2. Detect GPU capability, decide build flavor ---------------------

Write-Step "Detecting GPU"

$cudaArch = $null
if (Test-Command "nvcc") {
    if (Test-Command "nvidia-smi") {
        try {
            $computeCap = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
            if ($computeCap -match '^\d+\.\d+$') {
                $cudaArch = $computeCap -replace '\.', ''
                Write-Info "NVIDIA GPU found (compute capability $computeCap) and nvcc is available -- building with CUDA."
            }
        } catch {}
    }
    if (-not $cudaArch) {
        Write-Warn "nvcc found but couldn't read a compute capability from nvidia-smi -- building CPU-only instead."
    }
} else {
    Write-Info "No CUDA toolkit (nvcc) found -- building CPU-only. This is completely fine; see README for the portable-hardware profile."
}

# ---- 3. Clone + build whisper.cpp --------------------------------------

Write-Step "whisper.cpp"

if (-not (Test-Path "whisper.cpp")) {
    Write-Info "Cloning whisper.cpp..."
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp whisper.cpp
} else {
    Write-Info "whisper.cpp already present, skipping clone."
}

$serverExe = "whisper.cpp\build\bin\Release\whisper-server.exe"
$quantizeExe = "whisper.cpp\build\bin\Release\whisper-quantize.exe"
if ((Test-Path $serverExe) -and (Test-Path $quantizeExe)) {
    Write-Info "whisper-server.exe and whisper-quantize.exe already built, skipping build."
} else {
    Push-Location whisper.cpp
    try {
        $cmakeArgs = @("-B", "build")
        if ($cudaArch) { $cmakeArgs += @("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=$cudaArch") }
        Write-Info "Configuring: $cmake $($cmakeArgs -join ' ')"
        & $cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

        Write-Info "Building (this can take several minutes)..."
        & $cmake --build build --config Release -j 8
        if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
    } finally {
        Pop-Location
    }
}

# ---- 4. Download models -------------------------------------------------

Write-Step "Models"

New-Item -ItemType Directory -Force -Path "models" | Out-Null

$vadModel = "models\silero_vad.onnx"
if (-not (Test-Path $vadModel)) {
    Write-Info "Downloading silero_vad.onnx..."
    Invoke-WebRequest -Uri "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" -OutFile $vadModel
} else {
    Write-Info "silero_vad.onnx already present."
}

# Model choice follows the build flavor: small.en if we built with CUDA
# (fast enough unquantized on a GPU), base.en if CPU-only (small.en would be
# noticeably slow to transcribe on CPU).
if ($cudaArch) {
    $whisperModelFile = "ggml-small.en.bin"
} else {
    $whisperModelFile = "ggml-base.en.bin"
}
$whisperModelPath = "models\$whisperModelFile"
if (-not (Test-Path $whisperModelPath)) {
    Write-Info "Downloading $whisperModelFile..."
    Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$whisperModelFile" -OutFile $whisperModelPath
} else {
    Write-Info "$whisperModelFile already present."
}

# ---- 5. Python environment ----------------------------------------------

Write-Step "Python environment"

if (-not (Test-Path "venv")) {
    Write-Info "Creating venv..."
    python -m venv venv
} else {
    Write-Info "venv already present."
}
Write-Info "Installing requirements..."
& ".\venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& ".\venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt

# ---- 6. Starter config ---------------------------------------------------

Write-Step "Config"

if (Test-Path "config.yaml") {
    Write-Info "config.yaml already exists, leaving it alone."
} else {
    $profile = if ($cudaArch) { "profiles\config.gpu.yaml.example" } else { "profiles\config.portable.yaml.example" }
    Copy-Item $profile "config.yaml"
    # match the model this script actually downloaded, in case it differs
    # from the profile template's default
    (Get-Content "config.yaml") -replace 'whisper_model: models/.*\.bin', "whisper_model: models/$whisperModelFile" |
        Set-Content "config.yaml"
    Write-Info "Wrote config.yaml from $profile."
}

# ---- Done -----------------------------------------------------------------

Write-Step "Setup complete"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  cd app"
Write-Host "  ..\venv\Scripts\python.exe main.py"
Write-Host ""
Write-Host "On first run it will list your microphones if 'mic_device' isn't set --"
Write-Host "edit config.yaml at the project root and re-run. See README.md for details."
