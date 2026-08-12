<#
    One-command setup for the local demo.

        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

    Creates a working virtualenv, installs dependencies, indexes the corpus and
    starts the demo. Safe to re-run — each step checks whether it is already done.

    Every python call uses the venv interpreter by absolute path rather than
    relying on activation. Activation only edits PATH, so if the venv is missing
    its own pip.exe, `pip install` silently falls through to the system pip and
    installs into a completely different Python. Naming the interpreter removes
    that whole class of problem.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Step($n, $text) { Write-Host "`n[$n] $text" -ForegroundColor Cyan }
function Ok($text)       { Write-Host "    $text" -ForegroundColor Green }
function Warn($text)     { Write-Host "    $text" -ForegroundColor Yellow }
function Fail($text)     { Write-Host "`n  $text`n" -ForegroundColor Red; exit 1 }

Write-Host "`nEnterprise RAG — local demo setup" -ForegroundColor White
Write-Host "$ProjectRoot" -ForegroundColor DarkGray

# ── 1. Find a supported Python ────────────────────────────────────────────────
# 3.14 is excluded deliberately: most of these packages have no cp314 wheels yet,
# so pip falls back to compiling from source and dies on pyarrow without CMake
# and Visual Studio.
Step 1 "Locating Python 3.12 or 3.13"

$Interpreter = $null
foreach ($candidate in @("3.12", "3.13")) {
    try {
        $found = & py "-$candidate" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) {
            $Interpreter = $found.Trim()
            Ok "Python $candidate — $Interpreter"
            break
        }
    } catch { }
}

if (-not $Interpreter) {
    Fail @"
No Python 3.12 or 3.13 found.

  Python 3.14 will not work — packages like pyarrow have no wheels for it yet
  and pip tries to compile them from source.

  Install 3.12 from python.org/downloads/release/python-3128/ and re-run this.
"@
}

# ── 2. Virtualenv ─────────────────────────────────────────────────────────────
Step 2 "Preparing the virtualenv"

$NeedsVenv = $true
if (Test-Path $VenvPython) {
    # A venv without its own pip is worse than no venv at all — it looks active
    # while installing everything into the system Python.
    & $VenvPython -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $NeedsVenv = $false
        Ok "Existing venv has a working pip"
    } else {
        Warn "Existing venv has no pip — repairing with ensurepip"
        & $VenvPython -m ensurepip --upgrade 2>$null | Out-Null
        & $VenvPython -m pip --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $NeedsVenv = $false
            Ok "Repaired"
        } else {
            Warn "Repair failed — recreating from scratch"
            Remove-Item -Recurse -Force ".venv"
        }
    }
}

if ($NeedsVenv) {
    & $Interpreter -m venv .venv
    if (-not (Test-Path $VenvPython)) { Fail "Failed to create the virtualenv." }
    Ok "Created"
}

$Report = & $VenvPython -m pip --version
if ($Report -notmatch [regex]::Escape($ProjectRoot)) {
    Fail "pip is not the venv's own: $Report"
}
Ok $Report

& $VenvPython -m pip install --upgrade pip --quiet
Ok "pip upgraded"

# ── 3. CPU-only PyTorch ───────────────────────────────────────────────────────
# sentence-transformers depends on torch, and pip's default wheel on Windows is
# the CUDA build (~2.5 GB). The CPU wheel is ~200 MB. Nothing here uses a GPU:
# the reranker is ONNX on CPU and the embedding model is tiny.
Step 3 "Installing CPU-only PyTorch (~200 MB, a few minutes)"

& $VenvPython -c "import torch" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Ok "Already installed"
} else {
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed. Scroll up for the reason." }
    Ok "Installed"
}

# ── 4. Everything else ────────────────────────────────────────────────────────
Step 4 "Installing project dependencies"

& $VenvPython -m pip install -r requirements-demo.txt
if ($LASTEXITCODE -ne 0) {
    Fail @"
Dependency install failed.

  If the error mentions building a wheel and CMake, a package had no prebuilt
  wheel for this Python version. Check the interpreter above is 3.12 or 3.13.
"@
}
Ok "Installed"

# ── 5. Configuration ──────────────────────────────────────────────────────────
Step 5 "Checking configuration"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Fail @"
Created .env from the template. Open it and set:

    GROQ_API_KEY=your_key_here      (free: console.groq.com/keys)
    LOCAL_MODE=true

  Then run this script again.
"@
}

$EnvText = Get-Content ".env" -Raw
if ($EnvText -notmatch '(?m)^\s*GROQ_API_KEY\s*=\s*\S') {
    Fail "GROQ_API_KEY is empty in .env. Get a free key at console.groq.com/keys"
}
Ok ".env present with a Groq key"

& $VenvPython scripts\preflight.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n  Preflight reported blockers — see above." -ForegroundColor Yellow
    Write-Host "  If it is only the empty index, that is expected on a first run.`n" -ForegroundColor DarkGray
}

# ── 6. Index the corpus ───────────────────────────────────────────────────────
Step 6 "Indexing the demo corpus"

# Ask the store how many vectors it holds rather than checking whether the
# directory exists. An abandoned run leaves .qdrant_local containing only a .lock
# and an empty meta.json — present on disk, no collection inside. Testing for the
# directory would report "already indexed" and hand back a demo where every
# answer is "the documentation does not cover this", with nothing anywhere saying
# why.
$VectorCount = 0
try {
    $probe = & $VenvPython -c @"
import logfire
logfire.configure(send_to_logfire=False, console=False)
from app.services.retrieval.qdrant_service import collection_stats
s = collection_stats()
print(s.get('vectors', 0) if s.get('exists') else 0)
"@ 2>$null
    if ($LASTEXITCODE -eq 0) { $VectorCount = [int]($probe | Select-Object -Last 1) }
} catch { $VectorCount = 0 }

if ($VectorCount -gt 0) {
    Ok "Index already populated — $VectorCount vectors (delete .qdrant_local to rebuild)"
} else {
    if (Test-Path ".qdrant_local") {
        Warn "Found .qdrant_local but it holds no vectors — reindexing"
        Remove-Item -Recurse -Force ".qdrant_local"
    }
    Warn "First run downloads the embedding model (~420 MB). Once only."
    & $VenvPython -m app.ingestion.processor DATA\true_data true --wipe
    if ($LASTEXITCODE -ne 0) { Fail "Ingestion failed. See the error above." }
    Ok "Indexed"
}

# ── 7. Go ─────────────────────────────────────────────────────────────────────
Step 7 "Starting the demo"
Write-Host "    API on :8000, UI on :8501. Ctrl+C stops both.`n" -ForegroundColor DarkGray

& $VenvPython scripts\demo.py
