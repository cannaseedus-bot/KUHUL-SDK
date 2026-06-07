#Requires -Version 5.1
# micro_agents.ps1 — MicroAgent class, factory, and swarm coordinator
# Pipeline stage: N-GRAMS → MICRO_AGENTS → ATOMIC_BLOCKS
# Each MicroAgent: input contract, capability tags, Process(), emitted AtomicBlock

using namespace System.Collections.Concurrent

# Import K'UHUL runtime module if available
$RuntimePath = Join-Path $PSScriptRoot "kuhul-runtime.psm1"
if (Test-Path $RuntimePath) {
    Import-Module $RuntimePath -Force
}

# ---------------------------------------------------------------------------
# AtomicBlock — the universal output atom
# @control  = capability verbs the system can invoke on this block
# @variable = named state the block carries
# @view     = rendering / projection hints
# @links    = typed edges to other blocks by ID or tag
# ---------------------------------------------------------------------------
class AtomicBlock {
    [string]   $Id
    [string]   $Type
    [string[]] $Control
    [hashtable]$Variable
    [hashtable]$View
    [hashtable]$Links
    [datetime] $Timestamp

    AtomicBlock(
        [string]$type,
        [string[]]$control,
        [hashtable]$variable,
        [hashtable]$view,
        [hashtable]$links
    ) {
        $this.Id        = [guid]::NewGuid().ToString()
        $this.Type      = $type
        $this.Control   = $control
        $this.Variable  = $variable
        $this.View      = $view
        $this.Links     = $links
        $this.Timestamp = [datetime]::UtcNow
    }

    [hashtable] ToHashtable() {
        return @{
            id        = $this.Id
            type      = $this.Type
            control   = $this.Control
            variable  = $this.Variable
            view      = $this.View
            links     = $this.Links
            timestamp = $this.Timestamp.ToString('o')
        }
    }
}

# ---------------------------------------------------------------------------
# MicroAgent — minimal atomic processor
# ---------------------------------------------------------------------------
class MicroAgent {
    [string]      $Name
    [string[]]    $Control      # capability tags: @sense, @infer, @execute …
    [scriptblock] $Process      # receives: [pscustomobject[]] $ngrams → any
    [int]         $Activations  # runtime counter

    MicroAgent([string]$name, [string[]]$control, [scriptblock]$process) {
        $this.Name        = $name
        $this.Control     = $control
        $this.Process     = $process
        $this.Activations = 0
    }

    [object] Run([object]$input) {
        $this.Activations++
        return & $this.Process $input
    }
}

# ---------------------------------------------------------------------------
# Invoke-MicroAgents  (pipeline-friendly)
# Accepts a stream of KuhulNGram objects, buffers them, then fans-out to
# all registered agents.  Emits one result per agent per call.
# ---------------------------------------------------------------------------
function Invoke-MicroAgents {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, ValueFromPipeline)]
        [pscustomobject]$NGram,

        [Parameter(Mandatory)]
        [MicroAgent[]]$Agents
    )

    begin   { $buffer = [System.Collections.Generic.List[pscustomobject]]::new() }
    process { $buffer.Add($NGram) }
    end {
        $ngrams = $buffer.ToArray()
        foreach ($agent in $Agents) {
            [pscustomobject]@{
                Agent  = $agent.Name
                Tags   = $agent.Control
                Result = $agent.Run($ngrams)
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Convert-ToAtomicBlock  (pipeline-friendly)
# Receives agent result objects (from Invoke-MicroAgents) and synthesises
# a single AtomicBlock.  Also accepts direct -InputData bypass.
# ---------------------------------------------------------------------------
function Convert-ToAtomicBlock {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline)]
        [pscustomobject]$AgentResult,

        # Optional: pass raw text to synthesize without agent pipeline
        [string]$InputData = '',

        # Block type tag
        [string]$Type = 'INPUT_ATOMIC_BLOCK'
    )

    begin { $results = [System.Collections.Generic.List[pscustomobject]]::new() }
    process { if ($AgentResult) { $results.Add($AgentResult) } }
    end {
        # Extract entropy from entropy_agent result if available
        $entropy = 0.5  # default
        if ($results) {
            $entropyResult = $results | Where-Object { $_.Agent -eq 'entropy_agent' } | Select-Object -First 1
            if ($entropyResult -and $entropyResult.Result.Entropy) {
                $entropy = [Math]::Min($entropyResult.Result.Entropy / 8.0, 1.0)  # normalize to [0,1]
            }
        }

        [AtomicBlock]::new(
            $Type,
            @('@perception', '@representation', '@reasoning'),
            @{
                raw           = $InputData
                agent_results = $results.ToArray()
                entropy       = $entropy
                innovation    = 0.5
                stability     = 0.8
            },
            @{
                execution = 'powershell_pipeline'
                rendering = 'terminal_or_ui'
            },
            @{
                control_flows = @('DATA->NGRAMS', 'NGRAMS->AGENTS', 'AGENTS->BLOCK')
                state_flows   = @('raw', 'ngrams', 'agent_results')
            }
        )
    }
}

# ---------------------------------------------------------------------------
# Built-in canonical agents — mirrors DeepSeek micro-agent spec
# ---------------------------------------------------------------------------

# Perception: surface unigrams (sensory input layer — ⟁Pop)
# Extended: optionally calls gram_builder.py to extract PUA codepoints from text
$script:PerceptionAgent = [MicroAgent]::new(
    'perception_agent',
    @('@sense', '@detect', '@notice'),
    {
        param($ngrams)
        # Try to enrich with gram_builder if available
        if ($ngrams -and $ngrams[0].Key) {
            $textInput = $ngrams[0].Key
            $gramBuilderPath = Join-Path $PSScriptRoot ".." ".." "releases" "Kuhul-PY" "gram_builder.py"
            if (Test-Path $gramBuilderPath) {
                try {
                    $glyphResult = & python "$gramBuilderPath" tokenize_glyphs "$textInput" 2>$null
                    if ($glyphResult) {
                        # Merge glyph tokens into ngram output
                        $ngrams | Where-Object { $_.Size -eq 1 } | Select-Object -First 10
                        return
                    }
                } catch {
                    # Fallback to standard perception if gram_builder fails
                }
            }
        }
        # Default: unigrams only
        $ngrams | Where-Object { $_.Size -eq 1 } | Select-Object -First 10
    }
)

# Reasoning: n-gram size distribution (pattern analysis — ⟁Etz'nab')
$script:ReasoningAgent = [MicroAgent]::new(
    'reasoning_agent',
    @('@infer', '@connect', '@deduce'),
    {
        param($ngrams)
        $ngrams |
            Group-Object Size |
            ForEach-Object {
                [pscustomobject]@{ Size = [int]$_.Name; Count = $_.Count }
            }
    }
)

# Question detector: interrogative intent (⟁Xul — decision gate)
$script:QuestionAgent = [MicroAgent]::new(
    'question_agent',
    @('@detect', '@classify', '@intent'),
    {
        param($ngrams)
        $hasQ  = ($ngrams | Where-Object { $_.Key -match '\?' }).Count -gt 0
        $wh    = ($ngrams | Where-Object { $_.Key -match '^(what|who|where|when|why|how)$' -and $_.Size -eq 1 }).Count
        [pscustomobject]@{ IsQuestion = $hasQ; WHCount = $wh }
    }
)

# Entropy scorer: innovation signal (⟁K'ayab' — excitation/LTP)
$script:EntropyAgent = [MicroAgent]::new(
    'entropy_agent',
    @('@score', '@measure', '@calibrate'),
    {
        param($ngrams)
        $freq  = @{}
        $total = 0
        $ngrams | Where-Object { $_.Size -eq 1 } | ForEach-Object {
            $freq[$_.Key] = ($freq[$_.Key] ?? 0) + 1
            $total++
        }
        $H = 0.0
        if ($total -gt 0) {
            foreach ($c in $freq.Values) {
                $p  = $c / $total
                $H -= $p * [Math]::Log($p, 2)
            }
        }
        [pscustomobject]@{ Entropy = [Math]::Round($H, 4); Tokens = $total; Types = $freq.Count }
    }
)

# Default swarm: all four canonical agents
# Use global scope so the variable is visible when files are dot-sourced
$global:KuhulPerceptionAgent = $script:PerceptionAgent
$global:KuhulReasoningAgent  = $script:ReasoningAgent
$global:KuhulQuestionAgent   = $script:QuestionAgent
$global:KuhulEntropyAgent    = $script:EntropyAgent
$global:KuhulDefaultSwarm    = @(
    $script:PerceptionAgent,
    $script:ReasoningAgent,
    $script:QuestionAgent,
    $script:EntropyAgent
)

if ($MyInvocation.MyCommand.ScriptBlock.Module) {
    Export-ModuleMember -Function 'Invoke-MicroAgents', 'Convert-ToAtomicBlock' `
        -Variable 'KuhulPerceptionAgent', 'KuhulReasoningAgent', 'KuhulQuestionAgent',
                  'KuhulEntropyAgent', 'KuhulDefaultSwarm'
}
