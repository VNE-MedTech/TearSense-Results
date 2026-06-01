# PowerShell port of run.sh
# Usage:  .\run.ps1   (run from this directory, or anywhere — it anchors to its own folder)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Source library that sits next to this repo
$Source = Join-Path $PSScriptRoot '..\OutputLibraryTearSense'
if (-not (Test-Path $Source)) { throw "Source not found: $Source" }

# 1. Clean previous run artifacts (no error if they don't exist)
#    Equivalent of: rm -rf outputs/ external_assessor/ __pycache__/
foreach ($dir in 'outputs', 'external_assessor', '__pycache__') {
    if (Test-Path $dir) { Remove-Item -Path $dir -Recurse -Force }
}

# 2. Copy contents of the output library into here.
#    Equivalent of: cp -r ../OutputLibraryTearSense/* ./
#    bash's `*` skips dotfiles, so we exclude .git / .gitignore too — important
#    because this dir is a git submodule and overwriting its .git pointer would
#    break it. (On Windows those names aren't "hidden", so we filter by name.)
Get-ChildItem -Path $Source |
    Where-Object { $_.Name -notlike '.*' } |
    ForEach-Object { Copy-Item -Path $_.FullName -Destination $PSScriptRoot -Recurse -Force }

# 3. Activate the conda environment.
#    Equivalent of: conda activate tear_assessor
#    Loads conda's PowerShell hook first so `conda activate` works inside a script.
if (Get-Command conda -ErrorAction SilentlyContinue) {
    (& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
    conda activate interfundamental
} else {
    Write-Warning "conda not found on PATH; skipping 'conda activate tear_assessor'."
}

# 4. Run
python run.py
