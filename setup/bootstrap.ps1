$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$PythonVersion = '3.12.10'
$PythonDir = Join-Path $Root 'runtime\python'
$PythonExe = Join-Path $PythonDir 'python.exe'
$DownloadDir = Join-Path $Root 'runtime\downloads'
$TorchIndex = 'https://download.pytorch.org/whl/cu128'
$TorchRequirements = Join-Path $PSScriptRoot 'requirements-torch.txt'
$Requirements = Join-Path $PSScriptRoot 'requirements.txt'

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Python([string[]]$Arguments) {
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Install-EmbeddedPython {
    if (Test-Path $PythonExe) {
        Write-Step "Portable Python already present: $PythonExe"
        return
    }
    Write-Step "Downloading embeddable Python $PythonVersion"
    New-Item -ItemType Directory -Force $DownloadDir | Out-Null
    $zip = Join-Path $DownloadDir "python-$PythonVersion-embed-amd64.zip"
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    New-Item -ItemType Directory -Force $PythonDir | Out-Null
    Expand-Archive -Path $zip -DestinationPath $PythonDir -Force
    Remove-Item $zip -Force
}

function Set-PythonPathFile {
    $pth = Get-ChildItem -Path $PythonDir -Filter 'python3*._pth' | Select-Object -First 1
    if ($null -eq $pth) {
        throw "No ._pth file found in $PythonDir"
    }
    $zipName = ($pth.BaseName) + '.zip'
    $lines = @(
        $zipName,
        '.',
        'Lib\site-packages',
        '..\..\app',
        '..\..\app\official',
        'import site'
    )
    [IO.File]::WriteAllLines($pth.FullName, $lines, (New-Object Text.ASCIIEncoding))
    Write-Step "Configured $($pth.Name)"
}

function Install-Pip {
    if (Test-Path (Join-Path $PythonDir 'Lib\site-packages\pip')) {
        Write-Step 'pip already installed'
        return
    }
    Write-Step 'Installing pip'
    New-Item -ItemType Directory -Force $DownloadDir | Out-Null
    $getPip = Join-Path $DownloadDir 'get-pip.py'
    Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip -UseBasicParsing
    Invoke-Python @($getPip, '--no-warn-script-location')
    Remove-Item $getPip -Force
}

function Install-Dependencies {
    Write-Step 'Installing PyTorch (CUDA 12.8 build, bundles the CUDA runtime)'
    Invoke-Python @('-m', 'pip', 'install', '--no-cache-dir', '--no-warn-script-location', '--index-url', $TorchIndex, '-r', $TorchRequirements)
    Write-Step 'Installing application dependencies'
    Invoke-Python @('-m', 'pip', 'install', '--no-cache-dir', '--no-warn-script-location', '-r', $Requirements)
}

function Test-Runtime {
    Write-Step 'Verifying runtime'
    Invoke-Python @((Join-Path $PSScriptRoot 'verify_runtime.py'))
}

Install-EmbeddedPython
Set-PythonPathFile
Install-Pip
Install-Dependencies
Test-Runtime
if (Test-Path $DownloadDir) {
    Remove-Item $DownloadDir -Recurse -Force
}
Write-Step 'Runtime ready'
