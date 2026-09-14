param([switch]$Install, [switch]$SkipTests)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"
Set-Location (Join-Path $AuroraRoot 'android')
$taskArguments = @('--no-daemon','--console=plain','--max-workers=4','assembleDebug')
if (-not $SkipTests) { $taskArguments += @('testDebugUnitTest','lintDebug') }
& gradle @taskArguments
if ($LASTEXITCODE) { throw 'Android build or verification failed' }
$taskApk = Join-Path $AuroraRoot 'android\app\build\outputs\apk\debug\app-debug.apk'
New-Item -ItemType Directory -Path (Join-Path $AuroraRoot 'artifacts') -Force | Out-Null
Copy-Item -LiteralPath $taskApk -Destination (Join-Path $AuroraRoot 'artifacts\aurora-2w-a-debug.apk') -Force
if ($Install) {
    & adb install -r $taskApk
    if ($LASTEXITCODE) { throw 'Connect one USB debugging Android device and authorize this computer.' }
}
Write-Output (Join-Path $AuroraRoot 'artifacts\aurora-2w-a-debug.apk')
