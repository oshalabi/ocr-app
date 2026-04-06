<#
.SYNOPSIS
Build (and optionally push) the OCR sidecar Docker image.

.DESCRIPTION
Usage:
  .\scripts\build.ps1                        # build with tag ocr-sidecar:latest
  .\scripts\build.ps1 1.2.3                  # build with tag ocr-sidecar:1.2.3 + ocr-sidecar:latest
  .\scripts\build.ps1 1.2.3 ghcr.io/you/ocr  # build + push to a registry

Environment variables:
  IMAGE   — base image name          (default: ocr-sidecar)
  PUSH    — set to "1" to push       (default: 0)
#>

param(
    [Parameter(Position=0)]
    [string]$Version = "latest",
    
    [Parameter(Position=1)]
    [string]$Registry = ""
)

function Load-Env($File) {
    if (Test-Path $File) {
        Get-Content $File | Where-Object { $_ -match '^\s*([^#=]+)\s*=\s*(.*)' } | ForEach-Object {
            $val = $Matches[2].Trim()
            if ($val -match '^"(.*)"$') { $val = $Matches[1] }
            Set-Item "Env:\$($Matches[1].Trim())" $val
        }
    }
}
Load-Env ".env"
Load-Env ".env.local"

$ImageName = if ($env:IMAGE) { $env:IMAGE } else { "ocr-sidecar" }
$Push = if ($env:PUSH) { $env:PUSH } else { "0" }

if ($Registry) {
    $FullImage = "$Registry/$ImageName"
} else {
    $FullImage = $ImageName
}

$Tags = @("$FullImage`:$Version")
if ($Version -ne "latest") {
    $Tags += "$FullImage`:latest"
}

$TagArgs = @()
foreach ($tag in $Tags) {
    $TagArgs += "-t"
    $TagArgs += $tag
}

Write-Host "==> Building $($Tags -join ' ')"
docker build @TagArgs .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit $LASTEXITCODE
}

if ($Push -eq "1" -or $Registry) {
    foreach ($tag in $Tags) {
        Write-Host "==> Pushing $tag"
        docker push $tag
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker push failed for tag $tag"
            exit $LASTEXITCODE
        }
    }
}

Write-Host "==> Done"
