npx remotion render Main out/horror.mp4 2>&1 | Out-File -Encoding utf8 out\_render.log
$code = $LASTEXITCODE
Write-Host "render-exit=$code"
Write-Host ""
Write-Host "=== tail ==="
Get-Content out\_render.log -Tail 6 | ForEach-Object { $_.Trim() }
if ($code -ne 0) {
    Write-Host ""
    Write-Host "=== error ==="
    Select-String -Path out\_render.log -Pattern "An error occurred|Error  |Could not|Cannot|is not" |
        Select-Object -First 8 | ForEach-Object { $_.Line.Trim() }
}
