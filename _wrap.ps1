Write-Host "=== typecheck (unverified since the Captions rewrite) ==="
npx tsc --noEmit
Write-Host "tsc-exit=$LASTEXITCODE"

Write-Host ""
Write-Host "=== word-timing test ==="
npm run test --silent 2>&1 | Select-Object -Last 3

Write-Host ""
Write-Host "=== cleanup ==="
Remove-Item -Force out\cap-*.png, out\act-*.png -ErrorAction SilentlyContinue
Remove-Item -Force _diag.py, _test.ps1 -ErrorAction SilentlyContinue
Remove-Item -Force scripts\__pycache__ -Recurse -ErrorAction SilentlyContinue
Write-Host "  removed temp files"

Write-Host ""
Write-Host "=== deliverables ==="
Get-ChildItem out -File | Select-Object Name, @{ n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
