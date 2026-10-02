# Render verification stills at points where each distinct shot should be on screen.
# Scene boundaries at 30 fps (scene start, then shot weights within it):
#   gaon      frames    0-403   forest (0-269), void (269-403)
#   haveli    frames  403-788   house (403-557), corridor (557-711), candle (711-788)
#   aankhein  frames  789-1150  figure (789-1006), eyes (1006-1078), void (1078-1150)
#   ant       frames 1150-1588  rain (1150-1350), grave (1350-1588)
$targets = @(
    @{f = 60;   shot = 'forest' },
    @{f = 320;  shot = 'void-1' },
    @{f = 450;  shot = 'house' },
    @{f = 620;  shot = 'corridor' },
    @{f = 745;  shot = 'candle' },
    @{f = 900;  shot = 'figure' },
    @{f = 1040; shot = 'eyes' },
    @{f = 1110; shot = 'void-2' },
    @{f = 1250; shot = 'rain' },
    @{f = 1450; shot = 'grave' }
)

foreach ($t in $targets) {
    npx remotion still Main ("out/sh-" + $t.shot + ".png") --frame=$($t.f) 2>&1 | Out-Null
    if (Test-Path ("out/sh-" + $t.shot + ".png")) {
        Write-Host ("rendered {0,-10} frame {1}" -f $t.shot, $t.f)
    } else {
        Write-Host ("FAILED   {0,-10} frame {1}" -f $t.shot, $t.f)
    }
}
