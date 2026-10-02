Write-Host "=== rendering with optimised fog (no CSS blur) ==="
$elapsed = Measure-Command {
    npx remotion render Main out/horror.mp4 2>&1 | Out-File -Encoding utf8 out\_render2.log
}
$code = $LASTEXITCODE
Write-Host ("render-exit={0}" -f $code)
Write-Host ("elapsed={0:N1} s" -f $elapsed.TotalSeconds)

$frames = 1588
Write-Host ("throughput={0:N2} frames/sec" -f ($frames / $elapsed.TotalSeconds))
Write-Host ""
Write-Host "=== projected long-form cost at this rate ==="
foreach ($minutes in 10, 30, 60) {
    $f = $minutes * 60 * 30
    $hours = ($f / ($frames / $elapsed.TotalSeconds)) / 3600
    Write-Host ("  {0,2}-minute video ({1,6} frames)  ->  {2:N1} hours" -f $minutes, $f, $hours)
}

Write-Host ""
Write-Host "=== output ==="
Get-ChildItem out -Filter *.mp4 | Select-Object Name, @{ n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
