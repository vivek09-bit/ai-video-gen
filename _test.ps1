Remove-Item -Force out\act-*.png -ErrorAction SilentlyContinue

Write-Host "=== rendering stills at each cast scene's midpoint ==="
foreach ($f in 425, 1309) {
    npx remotion still Main "out/act-$f.png" --frame=$f 2>&1 | Out-Null
    if (Test-Path "out/act-$f.png") { Write-Host "  rendered frame $f" } else { Write-Host "  FAILED frame $f" }
}

Write-Host ""
& .venv\Scripts\python.exe scripts\_verify_acting.py

