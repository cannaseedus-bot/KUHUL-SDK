# Build K'UHUL Shaders
# ====================
# Compiles HLSL shaders to CSO (Compute Shader Objects)
#
# Prerequisites:
#   - Windows SDK 10+ (includes DXC)
#   - Or install DirectX Shader Compiler separately
#
# Usage:
#   .\build_shaders.ps1
#

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "K'UHUL Shader Build" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check for DXC
$dxcPath = $null
$possiblePaths = @(
    "$PSScriptRoot\..\gpu\bin\x86\dxc.exe",   # bundled local DXC (preferred)
    "C:\Program Files (x86)\Windows Kits\10\Bin\*\x64\dxc.exe",
    "C:\Program Files\Windows Kits\10\Bin\*\x64\dxc.exe",
    ".\dxc.exe"
)

foreach ($path in $possiblePaths) {
    $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        $dxcPath = $found.FullName
        break
    }
}

if (-not $dxcPath) {
    Write-Host "ERROR: DXC (DirectX Shader Compiler) not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Windows SDK 10+ or download from:" -ForegroundColor Yellow
    Write-Host "  https://github.com/microsoft/DirectXShaderCompiler" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "✓ Found DXC: $dxcPath" -ForegroundColor Green
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Create output directory
$outputDir = Join-Path $scriptDir "compiled"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
    Write-Host "✓ Created output directory: $outputDir" -ForegroundColor Green
}

# Shader compilation function
function Compile-Shader {
    param(
        [string]$InputFile,
        [string]$EntryPoint,
        [string]$OutputFile,
        [string]$Target = "cs_6_0"
    )
    
    Write-Host "Compiling: $InputFile" -ForegroundColor Cyan
    Write-Host "  Entry:  $EntryPoint" -ForegroundColor Gray
    Write-Host "  Target: $Target" -ForegroundColor Gray
    Write-Host "  Output: $OutputFile" -ForegroundColor Gray
    
    $arguments = @(
        "-T", $Target
        "-E", $EntryPoint
        "-O3"
        $InputFile
        "-Fo", $OutputFile
        "-Zi"  # Debug info
    )
    
    $process = Start-Process -FilePath $dxcPath `
        -ArgumentList $arguments `
        -Wait `
        -PassThru `
        -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        Write-Host "  ✓ Success" -ForegroundColor Green
        $size = (Get-Item $OutputFile).Length
        Write-Host "  Size:   $size bytes" -ForegroundColor Gray
        Write-Host ""
        return $true
    } else {
        Write-Host "  ✗ Failed (exit code: $($process.ExitCode))" -ForegroundColor Red
        Write-Host ""
        return $false
    }
}

# Compile shaders
$success = $true

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Compiling Shaders" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Shader 1: Glyph Compute (GRAM mode)
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "glyph_compute.hlsl") `
    -EntryPoint "CS_GlyphExec" `
    -OutputFile (Join-Path $outputDir "glyph_compute.cso") `
    -Target "cs_6_0")

# Shader 2: Attention Pass 1
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "attention_compute.hlsl") `
    -EntryPoint "CS_AttnPass1" `
    -OutputFile (Join-Path $outputDir "attention_pass1.cso") `
    -Target "cs_6_0")

# Shader 2: Attention Pass 2
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "attention_compute.hlsl") `
    -EntryPoint "CS_AttnPass2" `
    -OutputFile (Join-Path $outputDir "attention_pass2.cso") `
    -Target "cs_6_0")

# Shader 3: Orchestrator cs_6_0 (requires SM 6.0 wave intrinsics — discrete GPU)
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "orchestrate.hlsl") `
    -EntryPoint "main" `
    -OutputFile (Join-Path $outputDir "orchestrate.cso") `
    -Target "cs_6_0")

# Shader 4: Orchestrator cs_5_1 fallback (iGPU safe — no wave intrinsics)
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "orchestrate_51.hlsl") `
    -EntryPoint "main" `
    -OutputFile (Join-Path $outputDir "orchestrate_51.cso") `
    -Target "cs_5_1")

# Shader 5: COMPUTE_FOLD — token matmul + MoE routing
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "kuhul_fold_compute.hlsl") `
    -EntryPoint "CS_ComputeFold" `
    -OutputFile (Join-Path $outputDir "kuhul_fold_compute.cso") `
    -Target "cs_5_0")

# Shader 6: STORAGE_FOLD — snapshot/delta/seal + SHA-256
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "kuhul_fold_storage.hlsl") `
    -EntryPoint "CS_StorageFold" `
    -OutputFile (Join-Path $outputDir "kuhul_fold_storage.cso") `
    -Target "cs_5_0")

# Shader 7: META_FOLD — step hash + chain hash + Merkle
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "kuhul_fold_meta.hlsl") `
    -EntryPoint "CS_MetaFold" `
    -OutputFile (Join-Path $outputDir "kuhul_fold_meta.cso") `
    -Target "cs_5_0")

# Shader 8: XVM compute interpreter (fiber-based bytecode)
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "xvm_compute.hlsl") `
    -EntryPoint "CSMain" `
    -OutputFile (Join-Path $outputDir "xvm_compute.cso") `
    -Target "cs_5_0")

# Shader 9: XVM fused QKV attention (iGPU cs_5_0 path)
$success = $success -and (Compile-Shader `
    -InputFile (Join-Path $scriptDir "xvm_fused_qkv_attention.hlsl") `
    -EntryPoint "CSMain" `
    -OutputFile (Join-Path $outputDir "xvm_fused_qkv_attention.cso") `
    -Target "cs_5_0")

# Summary
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Build Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if ($success) {
    Write-Host "✓ All shaders compiled successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output files:" -ForegroundColor Cyan
    
    Get-ChildItem -Path $outputDir -Filter "*.cso" | ForEach-Object {
        Write-Host "  $($_.Name) ($($_.Length) bytes)" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "Usage (Python):" -ForegroundColor Cyan
    Write-Host "  from xcfe_runtime import XCFERuntime" -ForegroundColor Gray
    Write-Host "  runtime = XCFERuntime(use_gpu=True)" -ForegroundColor Gray
    Write-Host "  runtime.load_shader('compiled/glyph_compute.cso')" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "✗ Some shaders failed to compile." -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the errors above and fix HLSL syntax." -ForegroundColor Yellow
    exit 1
}
