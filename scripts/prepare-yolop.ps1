param([ValidateSet(320,640)][int]$Size=320,[switch]$PrepareAssets)
$ErrorActionPreference='Stop'
. "$PSScriptRoot\env.ps1"
Set-Location $AuroraRoot
$taskRepo=Join-Path $AuroraRoot 'third_party\YOLOP'
if(-not (Test-Path $taskRepo)) {
    New-Item -ItemType Directory -Path (Split-Path $taskRepo -Parent) -Force | Out-Null
    & git clone --no-checkout https://github.com/hustvl/YOLOP.git $taskRepo
    if($LASTEXITCODE) {throw 'Unable to fetch the official YOLOP repository'}
    & git -C $taskRepo checkout --detach 8d8f68df318c71f01d6f813c024df646c7d1978f
    if($LASTEXITCODE) {throw 'Unable to select verified YOLOP revision'}
}
$taskArguments=@('-m','aurora2w.yolop_baseline','--size',"$Size")
if($PrepareAssets) {$taskArguments+='--prepare-assets'}
& python @taskArguments
if($LASTEXITCODE) {throw 'YOLOP export or inference equivalence failed'}
Write-Output 'Pretrained inference preparation complete. No training was run.'
