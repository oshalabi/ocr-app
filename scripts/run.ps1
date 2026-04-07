<#
.SYNOPSIS
Run the OCR sidecar Docker image locally.

.DESCRIPTION
Usage:
  .\scripts\run.ps1 [additional docker run args...]

Environment variables:
  IMAGE         — image name to run (default: ocr-sidecar:latest)
  PORT          — port to expose (default: 8080)
  TEMPLATE_DIR  — directory for templates (default: .\templates)
#>

[CmdletBinding(PositionalBinding=$false)]
param(
    [string]$Image = "ocr-sidecar:latest",
    [int]$Port = 5423,
    [string]$TemplateDir = "$PWD\templates",
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$AdditionalDockerArgs
)

function Import-EnvFile($File) {
    if (Test-Path $File) {
        Get-Content $File | Where-Object { $_ -match '^\s*([^#=]+)\s*=\s*(.*)' } | ForEach-Object {
            $val = $Matches[2].Trim()
            if ($val -match '^"(.*)"$') { $val = $Matches[1] }
            Set-Item "Env:\$($Matches[1].Trim())" $val
        }
    }
}
Import-EnvFile ".env"
Import-EnvFile ".env.local"

if (-not $PSBoundParameters.ContainsKey('Image') -and $env:IMAGE) { $Image = $env:IMAGE }
if (-not $PSBoundParameters.ContainsKey('Port') -and $env:PORT) { $Port = $env:PORT }
if (-not $PSBoundParameters.ContainsKey('TemplateDir') -and $env:TEMPLATE_DIR) { $TemplateDir = $env:TEMPLATE_DIR }

if (-not (Test-Path $TemplateDir)) {
    New-Item -ItemType Directory -Force -Path $TemplateDir | Out-Null
}

Write-Host "==> Building image ($Image)..."
& "$PSScriptRoot\build.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build the image"
    exit $LASTEXITCODE
}


$DockerArgs = @(
    "--rm",
    "-it",
    "-p", "$($Port):8080",
    "-v", "$($TemplateDir):/app/templates"
)

if (Test-Path ".env") {
    $DockerArgs += "--env-file"
    $DockerArgs += ".env"
}
if (Test-Path ".env.local") {
    $DockerArgs += "--env-file"
    $DockerArgs += ".env.local"
}

if ($AdditionalDockerArgs) {
    $DockerArgs += $AdditionalDockerArgs
}

Write-Host "==> Running $Image on port $Port..."
docker run @DockerArgs $Image
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed"
    exit $LASTEXITCODE
}
