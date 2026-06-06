# launch-coordinator-and-services.ps1
# Called by KuhulCLI.bat — starts coordinator, orchestrator, and all 34 micronauts via bot launcher.
# This script blocks (runs coordinator in foreground) so the bat window stays alive.

$ErrorActionPreference = "Continue"
$Root        = Split-Path $PSScriptRoot -Parent
$Commands    = Join-Path $Root "commands"
$Logs        = Join-Path $Root "logs"
$Python      = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = "python" }

if (-not (Test-Path $Logs)) { New-Item -ItemType Directory -Path $Logs -Force | Out-Null }

Write-Host "[services] Root:     $Root"
Write-Host "[services] Commands: $Commands"
Write-Host "[services] Python:   $Python"

# ── Start Bot Launcher (all 34 micronauts) ─────────────────────────────────────
Write-Host "[services] Starting Bot Launcher (all 34 micronauts)..."
$botLog = Join-Path $Logs "bot-launcher.log"
$bot    = Start-Process -FilePath $Python `
            -ArgumentList "$Commands\bot-launcher.py", "--coordinator-port", "25100" `
            -WorkingDirectory $Commands `
            -RedirectStandardOutput $botLog `
            -RedirectStandardError  "$botLog.err" `
            -PassThru -WindowStyle Hidden
Write-Host "[services] bot-launcher.py started (pid $($bot.Id))"

# Brief pause to let all 34 bots boot before coordinator polls them
Start-Sleep -Seconds 5

# ── Start orchestrator on port 3300 ───────────────────────────────────────────
Write-Host "[services] Starting orchestrator on port 3300..."
$orchLog = Join-Path $Logs "orchestrator.log"
$orch    = Start-Process -FilePath $Python `
             -ArgumentList "$PSScriptRoot\orchestrator.py", "--port", "3300" `
             -WorkingDirectory $PSScriptRoot `
             -RedirectStandardOutput $orchLog `
             -RedirectStandardError  "$orchLog.err" `
             -PassThru -WindowStyle Hidden
Write-Host "[services] orchestrator launched (pid $($orch.Id))"

# ── Start Session Memory Service on port 25101 ───────────────────────────────────
Write-Host "[services] Starting Session Memory Service on port 25101..."
$smLog = Join-Path $Logs "session-memory.log"
$sm    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/session-memory-service.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $smLog `
           -RedirectStandardError  "$smLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] session-memory-service launched (pid $($sm.Id))"

# ── Start Plan Service on port 25102 ───────────────────────────────────────────
Write-Host "[services] Starting Plan Service (PM-1) on port 25102..."
$pmLog = Join-Path $Logs "plan-service.log"
$env:PORT = "25102"
$pm    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/pm1-plan-service.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $pmLog `
           -RedirectStandardError  "$pmLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] pm1-plan-service launched (pid $($pm.Id))"

# ── Start Responder Service on port 25103 ───────────────────────────────────────
Write-Host "[services] Starting Responder Service on port 25103..."
$rpLog = Join-Path $Logs "responder.log"
$env:PORT = "25103"
$rp    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/responder-service.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $rpLog `
           -RedirectStandardError  "$rpLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] responder-service launched (pid $($rp.Id))"

# ── Start Executor Service on port 25104 ────────────────────────────────────────
Write-Host "[services] Starting Executor Service on port 25104..."
$exLog = Join-Path $Logs "executor.log"
$env:PORT = "25104"
$ex    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/executor-service.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $exLog `
           -RedirectStandardError  "$exLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] executor-service launched (pid $($ex.Id))"

# ── Start Manager Service on port 25105 ─────────────────────────────────────────
Write-Host "[services] Starting Manager Service on port 25105..."
$mgLog = Join-Path $Logs "manager.log"
$env:PORT = "25105"
$mg    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/manager-service.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $mgLog `
           -RedirectStandardError  "$mgLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] manager-service launched (pid $($mg.Id))"

# ── Start Verb Router Service on port 25106 ─────────────────────────────────────
Write-Host "[services] Starting Verb Router Service on port 25106..."
$vrLog = Join-Path $Logs "verb-router.log"
$env:PORT = "25106"
$vr    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/verb-router.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $vrLog `
           -RedirectStandardError  "$vrLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] verb-router launched (pid $($vr.Id))"

# ── Start web-research stub on port 25108 ─────────────────────────────────────
Write-Host "[services] Starting web-research stub on port 25108..."
$wrLog = Join-Path $Logs "web-research.log"
$wr    = Start-Process -FilePath $Python `
           -ArgumentList (Join-Path $PSScriptRoot "web-research-stub.py") `
           -WorkingDirectory $PSScriptRoot `
           -RedirectStandardOutput $wrLog `
           -RedirectStandardError  "$wrLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] web-research stub launched (pid $($wr.Id))"

# ── Start skills router on port 25107 (simplified v2) ────────────────────────────
Write-Host "[services] Starting Skills Router v2 on port 25107..."
$env:PORT = "25107"
$skLog = Join-Path $Logs "skills-router-v2.log"
$sk    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "skills-router-v2.mjs") `
           -WorkingDirectory $PSScriptRoot `
           -RedirectStandardOutput $skLog `
           -RedirectStandardError  "$skLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] skills-router-v2 launched (pid $($sk.Id))"

# ── Start Model Router Service on port 25112 ─────────────────────────────────────
# Routes requests to local GGUF (5000) and DDS (5001) backends
Write-Host "[services] Starting Model Router Service on port 25112..."
$mrLog = Join-Path $Logs "model-router.log"
$env:PORT = "25112"
$env:GGUF_BACKEND = "http://127.0.0.1:5000"
$env:DDS_BACKEND = "http://127.0.0.1:5001"
$mr    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/model-router-service.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $mrLog `
           -RedirectStandardError  "$mrLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] model-router-service launched (pid $($mr.Id))"

# ── Start Queue Service on port 25109 (Phase 2.1) ────────────────────────────────
# Priority-based request queuing with backpressure detection
Write-Host "[services] Starting Queue Service on port 25109 (Phase 2.1)..."
$qLog = Join-Path $Logs "queue-service.log"
$env:PORT = "25109"
$qs   = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/queue-service.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $qLog `
           -RedirectStandardError  "$qLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] queue-service launched (pid $($qs.Id))"

# ── Start Circuit Breaker Service on port 25110 (Phase 2.1) ──────────────────────
# Fault tolerance with automatic recovery
Write-Host "[services] Starting Circuit Breaker Service on port 25110 (Phase 2.1)..."
$cbLog = Join-Path $Logs "circuit-breaker.log"
$env:PORT = "25110"
$cb   = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/circuit-breaker.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $cbLog `
           -RedirectStandardError  "$cbLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] circuit-breaker launched (pid $($cb.Id))"

# ── Start Concurrency Manager Service on port 25111 (Phase 2.1) ──────────────────
# AIMD-based adaptive concurrency scaling
Write-Host "[services] Starting Concurrency Manager on port 25111 (Phase 2.1)..."
$cmLog = Join-Path $Logs "concurrency-manager.log"
$env:PORT = "25111"
$cm   = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/concurrency-manager.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $cmLog `
           -RedirectStandardError  "$cmLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] concurrency-manager launched (pid $($cm.Id))"

# ── Start Semantic Field Engine Service on port 25115 (Phase 3) ────────────────
# Core FIELD mutation engine
Write-Host "[services] Starting Semantic Field Engine on port 25115 (Phase 3)..."
$sfeLog = Join-Path $Logs "semantic-field-engine.log"
$env:PORT = "25115"
$sfe   = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/semantic-field-engine.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $sfeLog `
           -RedirectStandardError  "$sfeLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] semantic-field-engine launched (pid $($sfe.Id))"

# ── Start Deterministic Router Service on port 25116 (Phase 3) ─────────────────
# SHA-256 routing to 26 Supernaut actions
Write-Host "[services] Starting Deterministic Router on port 25116 (Phase 3)..."
$drLog = Join-Path $Logs "deterministic-router.log"
$env:PORT = "25116"
$dr   = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/deterministic-router.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $drLog `
           -RedirectStandardError  "$drLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] deterministic-router launched (pid $($dr.Id))"

# ── Start Event Bridge Service on port 25117 (Phase 3) ────────────────────────
# Pub/sub for cross-service events
Write-Host "[services] Starting Event Bridge on port 25117 (Phase 3)..."
$ebLog = Join-Path $Logs "event-bridge.log"
$env:PORT = "25117"
$eb   = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/event-bridge.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $ebLog `
           -RedirectStandardError  "$ebLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] event-bridge launched (pid $($eb.Id))"

# ── Start Replay Engine Service on port 25118 (Phase 3) ──────────────────────
# Checkpoint and replay for deterministic recovery
Write-Host "[services] Starting Replay Engine on port 25118 (Phase 3)..."
$reLog = Join-Path $Logs "replay-engine.log"
$env:PORT = "25118"
$re   = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "node-services/replay-engine.mjs") `
           -WorkingDirectory (Join-Path $PSScriptRoot "node-services") `
           -RedirectStandardOutput $reLog `
           -RedirectStandardError  "$reLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] replay-engine launched (pid $($re.Id))"

# ── Start Data Harvester on port 25120 (Internet Learning) ───────────────────
# Autonomous URL loader — fetches GitHub, StackOverflow, arXiv, HuggingFace, etc.
# Rate-limited (1 req/domain/sec), writes JSONL batches to data/harvested/
Write-Host "[services] Starting Data Harvester on port 25120..."
$dhLog = Join-Path $Logs "data-harvester.log"
$dh    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "data-harvester.mjs") `
           -WorkingDirectory $PSScriptRoot `
           -RedirectStandardOutput $dhLog `
           -RedirectStandardError  "$dhLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] data-harvester launched (pid $($dh.Id))"

# ── Start Learning Engine on port 25121 (Internet Learning) ──────────────────
# Continuous training pipeline — spawns internet_harvester.py, hot-swaps model
Write-Host "[services] Starting Learning Engine on port 25121..."
$leLog = Join-Path $Logs "learning-engine.log"
$le    = Start-Process -FilePath "node" `
           -ArgumentList (Join-Path $PSScriptRoot "learning-engine.mjs") `
           -WorkingDirectory $PSScriptRoot `
           -RedirectStandardOutput $leLog `
           -RedirectStandardError  "$leLog.err" `
           -PassThru -WindowStyle Hidden
Write-Host "[services] learning-engine launched (pid $($le.Id))"

# ── Run coordinator in foreground on port 25100 ───────────────────────────────
# (bat polls this port — must be last so the process doesn't exit)
Write-Host "[services] Starting coordinator on port 25100 (foreground)..."
& $Python "$PSScriptRoot\coordinator.py" --port 25100
