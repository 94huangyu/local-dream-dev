# 2026-09-25 项目收尾：把散落在项目文件夹之外、需要保留的文件移进 D:\LocalDreamZImage，并归档顶层零碎旧文件。
# 只移动、不删除（同盘移动 = 改名，瞬间完成）。清单写到 archive\PATH_MOVES_20260925.tsv。
$ErrorActionPreference = 'Stop'
$root = 'D:\LocalDreamZImage'
$log = "$root\archive\PATH_MOVES_20260925.tsv"

$moves = New-Object System.Collections.Generic.List[object]
# 1) 高通 SDK
$moves.Add(@('D:\qairt\2.48.0.260626', "$root\sdk\qairt\2.48.0.260626"))
# 2) 发布物
foreach ($n in 'LocalDreamZImage_armv8a_2.8.1-zit2.apk', 'README.md', 'LICENSE', 'NOTICE', 'repack.py', 'hf') {
    $moves.Add(@("D:\ZImage_Work\publish\$n", "$root\release\$n"))
}
# 3) D:\ZImage_Work 剩下的全部（publish 之外）
Get-ChildItem -LiteralPath 'D:\ZImage_Work' -Force | Where-Object { $_.Name -ne 'publish' } | ForEach-Object {
    $moves.Add(@($_.FullName, "$root\archive\ZImage_Work\$($_.Name)"))
}
# 4) 项目根目录的零碎旧文件
$moves.Add(@("$root\local-dream-ZIT0813", "$root\archive\local-dream-ZIT0813"))
Get-ChildItem -LiteralPath $root -Force -Filter 'MAINLINE.md.bak_*' | ForEach-Object {
    $moves.Add(@($_.FullName, "$root\archive\MAINLINE_bak\$($_.Name)"))
}

# 预检：源都在、目标都不在
foreach ($m in $moves) {
    if (-not (Test-Path -LiteralPath $m[0])) { throw "源不存在: $($m[0])" }
    if (Test-Path -LiteralPath $m[1]) { throw "目标已存在: $($m[1])" }
}

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("from`tto")
foreach ($m in $moves) {
    $parent = Split-Path -Parent $m[1]
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force $parent | Out-Null }
    Move-Item -LiteralPath $m[0] -Destination $m[1]
    if ((Test-Path -LiteralPath $m[0]) -or -not (Test-Path -LiteralPath $m[1])) { throw "移动未生效: $($m[0])" }
    $lines.Add("$($m[0])`t$($m[1])")
    Write-Host "moved  $($m[0])  ->  $($m[1])"
}
[IO.File]::WriteAllLines($log, $lines, (New-Object Text.UTF8Encoding($true)))

# 5) 签名文件：复制备份（原件留原处，Gradle 编译要用）
$ks = "$env:USERPROFILE\.android\debug.keystore"
New-Item -ItemType Directory -Force "$root\keys" | Out-Null
Copy-Item -LiteralPath $ks "$root\keys\debug.keystore"
if ((Get-FileHash $ks).Hash -ne (Get-FileHash "$root\keys\debug.keystore").Hash) { throw "keystore 备份与原件不一致" }
Write-Host "copied $ks -> $root\keys\debug.keystore (sha256 一致)"

# 6) 清掉已空的目录（只删空目录）
foreach ($d in 'D:\ZImage_Work\publish', 'D:\ZImage_Work', 'D:\qairt', "$root\output") {
    if ((Test-Path -LiteralPath $d) -and -not (Get-ChildItem -LiteralPath $d -Force)) {
        Remove-Item -LiteralPath $d; Write-Host "removed empty dir $d"
    }
}
Write-Host "`n清单: $log  （$($moves.Count) 项）"
