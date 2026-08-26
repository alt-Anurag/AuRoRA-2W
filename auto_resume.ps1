while ($true) {
    Write-Output "Starting training..."
    cmd.exe /c "python -u scripts/train_aurora2w.py >> training_resume.log 2>&1"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Training completed successfully!"
        break
    }
    
    Write-Output "Process crashed with exit code $LASTEXITCODE. Restarting in 10 seconds..."
    Start-Sleep -Seconds 10
}
