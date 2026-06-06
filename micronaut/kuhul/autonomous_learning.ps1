#Requires -Version 5.1
# autonomous_learning.ps1 — ASX Prime OS K'UHUL Autonomous Learning Engine
#
# K'UHUL opcode notation for the internet-fed AI learning loop.
# Binds to: ⟁COMPUTE_FOLD⟁ | Micronaut: MM-CODER
#
# Load order:
#   . .\autonomous_learning.ps1
#   Start-AutonomousLearning -HarvesterPort 25109 -LearnerPort 25110

# ---------------------------------------------------------------------------
# ⟁Pop⟁ harvest_knowledge
# K'ayab' over all sources → Sek fetch_next_url → Wo store → Xul
# ---------------------------------------------------------------------------
function Invoke-HarvestKnowledge {
    param(
        [string]$HarvesterUrl = 'http://127.0.0.1:25109',
        [switch]$Wait
    )

    Write-Host "[KUHUL] ⟁Pop⟁ harvest_knowledge — Sek fetch → Ch'en raw_data"

    try {
        $resp = Invoke-RestMethod -Uri "$HarvesterUrl/harvest" -Method POST -ContentType 'application/json'
        Write-Host "[KUHUL] Wo data_sources → Ch'en queued=true cycle=$($resp.cycle)"
        if ($Wait) { Start-Sleep -Seconds 30 }
        return $resp
    } catch {
        Write-Warning "[KUHUL] Xul harvest_knowledge: $_"
        return $null
    }
}

# ---------------------------------------------------------------------------
# ⟁Pop⟁ train_on_new_data
# Yax queue → Sek decompress_batch → Sek update_ai_model → Sek hot_swap_active → Xul
# ---------------------------------------------------------------------------
function Invoke-TrainOnData {
    param(
        [Parameter(Mandatory)]
        [string]$BatchDir,

        [string]$ModelOut = 'E:\models\GPT2\med-GPT',
        [string]$LearnerUrl = 'http://127.0.0.1:25110'
    )

    Write-Host "[KUHUL] ⟁Pop⟁ train_on_new_data — Yax queue → Sek update_ai_model"

    $body = @{ batch_dir = $BatchDir; model_out = $ModelOut } | ConvertTo-Json
    try {
        $resp = Invoke-RestMethod -Uri "$LearnerUrl/train" -Method POST -Body $body -ContentType 'application/json'
        Write-Host "[KUHUL] Sek hot_swap_active → Ch'en new_model cycle=$($resp.cycle)"
        return $resp
    } catch {
        Write-Warning "[KUHUL] Xul train_on_new_data: $_"
        return $null
    }
}

# ---------------------------------------------------------------------------
# ⟁Pop⟁ continuous_learning
# K'ayab' learning_cycle → check_internet → if online: harvest → sleep 300 → Kumk'u → Xul
# ---------------------------------------------------------------------------
function Start-AutonomousLearning {
    param(
        [string]$HarvesterUrl    = 'http://127.0.0.1:25109',
        [string]$LearnerUrl      = 'http://127.0.0.1:25110',
        [string]$HarvestedDir    = (Join-Path $PSScriptRoot '..\..\data\harvested'),
        [int]$CycleSeconds       = 300
    )

    Write-Host "[KUHUL] ⟁Pop⟁ continuous_learning — K'ayab' learning_lifetime"

    # ⟁K'ayab'⟁ learning_lifetime
    while ($true) {

        # Sek check_internet_connection → Ch'en online
        $online = Test-Connection -ComputerName '8.8.8.8' -Count 1 -Quiet -ErrorAction SilentlyContinue

        if ($online) {
            Write-Host "[KUHUL] Yax online=true → Sek if → then start_data_harvesting"

            # Sek harvest_from_priority_sources → Ch'en new_data
            Invoke-HarvestKnowledge -HarvesterUrl $HarvesterUrl

            # Sek train on whatever landed in the batch dir
            $batches = Get-ChildItem -Path $HarvestedDir -Filter '*.jsonl' -ErrorAction SilentlyContinue
            if ($batches) {
                Invoke-TrainOnData -BatchDir $HarvestedDir -LearnerUrl $LearnerUrl
            }
        } else {
            Write-Host "[KUHUL] Yax online=false → offline_mode: internal_reasoning only"
        }

        # Sek sleep 300 — wait before next K'ayab' iteration
        Write-Host "[KUHUL] Sek sleep $CycleSeconds — next cycle in ${CycleSeconds}s"
        Start-Sleep -Seconds $CycleSeconds

    }
    # ⟁Kumk'u⟁ learning_lifetime  (unreachable — loop is infinite by design)
    # ⟁Xul
}

# ---------------------------------------------------------------------------
# ⟁Pop⟁ find_knowledge_gaps  — inspect harvester + learner status
# ---------------------------------------------------------------------------
function Get-LearningStatus {
    param(
        [string]$HarvesterUrl = 'http://127.0.0.1:25109',
        [string]$LearnerUrl   = 'http://127.0.0.1:25110'
    )

    Write-Host "[KUHUL] ⟁Pop⟁ find_knowledge_gaps — Sek analyze_coverage"

    $h = $l = $null
    try { $h = Invoke-RestMethod -Uri "$HarvesterUrl/status" -Method GET } catch { $h = @{ error = "$_" } }
    try { $l = Invoke-RestMethod -Uri "$LearnerUrl/status"   -Method GET } catch { $l = @{ error = "$_" } }

    [pscustomobject]@{
        Harvester    = $h
        LearningEngine = $l
        Fold         = '⟁COMPUTE_FOLD⟁'
        Timestamp    = (Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')
    }
}
