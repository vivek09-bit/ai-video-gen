$result = "out\_report.txt"
$lines = @()

$lines += "=== render log tail ==="
if (Test-Path out\_render2.log) {
    $lines += (Get-Content out\_render2.log -Tail 4 | ForEach-Object { $_.Trim() })
} else {
    $lines += "no log yet"
}

$lines += ""
$lines += "=== outputs ==="
Get-ChildItem out -Filter *.mp4 | ForEach-Object {
    $lines += ("{0}  {1:N2} MB  modified {2}" -f $_.Name, ($_.Length / 1MB), $_.LastWriteTime.ToString("HH:mm:ss"))
}

$lines += ""
$lines += "=== stills present ==="
$lines += ((Get-ChildItem out -Filter 'sh-*.png' | Measure-Object).Count.ToString() + " stills")

$lines | Set-Content -Encoding utf8 $result
Write-Host "wrote $result"
