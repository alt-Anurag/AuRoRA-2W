param(
    [string]$Manifest = 'data/phase2/prepared/research_baseline_v2/manifest.jsonl',
    [ValidateSet('baseline','idfa')][string]$Model = 'baseline',
    [string]$Run,
    [string]$Initialize,
    [string]$Resume
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"
Set-Location $AuroraRoot
if (-not (Test-Path -LiteralPath $Manifest)) { throw "Prepared manifest missing: $Manifest. Follow docs/DATA.md and docs/DEVELOPMENT.md." }
if ($Initialize -and $Resume) { throw 'Choose initialization OR resumption.' }
if ($Resume -and -not $Run) { throw 'Pass the existing run directory with -Run when resuming.' }
if (-not $Run) { $Run = "output/${Model}_$(Get-Date -Format yyyyMMdd_HHmmss)" }
$taskArgs = @('-m','aurora2w.cli','train','--config',"configs/rtx4050_$Model.json",'--manifest',$Manifest,'--device','cuda','--output',$Run)
if ($Initialize) { $taskArgs += @('--initialize',$Initialize) }
if ($Resume) { $taskArgs += @('--resume',$Resume) }
& python @taskArgs
if ($LASTEXITCODE) { throw 'Training failed; inspect the error above and the run output.' }
Write-Output "Run completed: $Run. Evaluate the held-out partitions before interpreting quality."
