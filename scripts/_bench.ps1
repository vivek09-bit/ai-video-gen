$out = "out\_bench.txt"
$lines = @()

$frames = 300

$elapsed = Measure-Command {
    npx remotion render Main out\_bench.mp4 --frames="0-$($frames - 1)" 2>&1 | Out-Null
}

$rate = $frames / $elapsed.TotalSeconds
$lines += "frames rendered : $frames"
$lines += ("elapsed         : {0:N1} s" -f $elapsed.TotalSeconds)
$lines += ("throughput      : {0:N2} frames/sec" -f $rate)
$lines += ""
$lines += "=== projected long-form render time ==="
foreach ($minutes in 10, 30, 60) {
    $f = $minutes * 60 * 30
    $hours = ($f / $rate) / 3600
    $lines += ("  {0,2} min video = {1,6} frames -> {2,5:N1} hours" -f $minutes, $f, $hours)
}

$lines += ""
$lines += "=== audio streams in the full render ==="
$bytes = [System.IO.File]::ReadAllBytes("out\horror.mp4")
$text = [System.Text.Encoding]::ASCII.GetString($bytes)
$lines += ("  avc1 video track : " + $text.Contains("avc1"))
$lines += ("  mp4a audio track : " + $text.Contains("mp4a"))

Remove-Item -Force out\_bench.mp4 -ErrorAction SilentlyContinue
$lines | Set-Content -Encoding utf8 $out
Write-Host "wrote $out"
