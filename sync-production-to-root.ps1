# Manual sync script: Copy production release to root
# Usage: .\sync-production-to-root.ps1
# This is also run automatically before every `git push` via pre-push hook

param(
    [switch]$Commit = $false,
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Continue"

$PROD_DIR = "releases\KUHUL-SDK"
$ROOT_DIR = "."

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Sync Production Release → Root                            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] No files will be modified" -ForegroundColor Yellow
}

# Check if production directory exists
if (-not (Test-Path $PROD_DIR)) {
    Write-Host "ERROR: Production directory not found: $PROD_DIR" -ForegroundColor Red
    exit 1
}

Write-Host "Production source: $PROD_DIR" -ForegroundColor White
Write-Host "Target directory:  $ROOT_DIR" -ForegroundColor White
Write-Host ""

# Directories to sync
$SYNC_DIRS = @(
    "skills",
    "tools",
    "docs",
    "system",
    "config",
    "micronauts",
    "training",
    "llm",
    "pi-kuhul",
    "registry"
)

# Files to sync
$SYNC_FILES = @(
    "DEPLOYMENT.md",
    "INDEX.md",
    "MANIFEST.json",
    "PRODUCTION-MANIFEST-v1.1.md",
    "PRODUCTION-RELEASE-v1.1-COMPLETE.md",
    "RELEASE_NOTES.md",
    "RELEASE_NOTES_v1.1.md",
    "UNIFIED-ARCHITECTURE-v1.1.md",
    "VERSION.txt"
)

$syncCount = 0

Write-Host "Syncing directories:" -ForegroundColor Yellow
foreach ($dir in $SYNC_DIRS) {
    $PROD_PATH = Join-Path $PROD_DIR $dir
    $ROOT_PATH = Join-Path $ROOT_DIR $dir

    if (Test-Path $PROD_PATH) {
        Write-Host "  ✓ $dir" -ForegroundColor Green

        if (-not $DryRun) {
            if (Test-Path $ROOT_PATH) {
                Remove-Item -Path $ROOT_PATH -Recurse -Force -ErrorAction SilentlyContinue
            }

            # For llm/training: exclude raw .jsonl files (kept as .7z archives)
            if ($dir -eq "llm") {
                Copy-Item -Path $PROD_PATH -Destination $ROOT_PATH -Recurse -Force
                Get-ChildItem -Path "$ROOT_PATH/training" -Filter "*.jsonl" -ErrorAction SilentlyContinue | Remove-Item -Force
            } else {
                Copy-Item -Path $PROD_PATH -Destination $ROOT_PATH -Recurse -Force
            }
        }
        $syncCount++
    } else {
        Write-Host "  ✗ $dir (not found in production)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Syncing files:" -ForegroundColor Yellow
foreach ($file in $SYNC_FILES) {
    $PROD_FILE = Join-Path $PROD_DIR $file
    $ROOT_FILE = Join-Path $ROOT_DIR $file

    if (Test-Path $PROD_FILE) {
        Write-Host "  ✓ $file" -ForegroundColor Green

        if (-not $DryRun) {
            Copy-Item -Path $PROD_FILE -Destination $ROOT_FILE -Force
        }
        $syncCount++
    } else {
        Write-Host "  ✗ $file (not found in production)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Synced $syncCount items" -ForegroundColor Cyan

if (-not $DryRun) {
    # Check git status
    Write-Host ""
    Write-Host "Git status:" -ForegroundColor Yellow
    $changes = git status --porcelain | Measure-Object -Line
    $changeCount = $changes.Lines

    if ($changeCount -gt 0) {
        Write-Host "  Found $changeCount changed file(s)" -ForegroundColor Yellow

        if ($Commit) {
            Write-Host "  Staging all changes..." -ForegroundColor White
            git add -A

            Write-Host "  Creating commit..." -ForegroundColor White
            git commit -m "chore(sync): Sync production release to root

- Production directory: releases/kuhul-v1.0.0-production/
- Synced: all skills, tools, docs, micronauts, configs, manifests
- Triggered by: sync-production-to-root.ps1"

            Write-Host "  Commit created successfully" -ForegroundColor Green
        } else {
            Write-Host "  Run with -Commit flag to auto-commit changes" -ForegroundColor Gray
        }
    } else {
        Write-Host "  No changes (production and root are in sync)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Sync complete!" -ForegroundColor Green
