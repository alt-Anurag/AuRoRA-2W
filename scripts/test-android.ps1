param()
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"
Set-Location $AuroraRoot
$taskDeviceOutput = & adb devices
if ($LASTEXITCODE) { throw 'Unable to list Android devices.' }
$taskDevices = @($taskDeviceOutput | Where-Object { $_ -match '^\S+\s+device$' })
if ($taskDevices.Count -ne 1) { throw 'Connect exactly one authorized Android device or start one emulator before running Android instrumentation.' }
& python -m aurora2w.android_fixture --ensure
if ($LASTEXITCODE) { throw 'Android parity fixture generation failed.' }
Set-Location (Join-Path $AuroraRoot 'android')
& gradle --no-daemon --console=plain --max-workers=4 connectedDebugAndroidTest
if ($LASTEXITCODE) { throw 'Android instrumented parity test failed.' }
Write-Output 'Android CPU parity passed. Reinstall the production APK with scripts/build-android.ps1 -Install if Gradle removed it after testing.'
