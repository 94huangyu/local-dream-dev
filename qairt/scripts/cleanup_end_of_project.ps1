# 项目结束清理（2026-09-25，用户已批准原则：留签名、文档、QAIRT SDK；其余大文件删，ONNX 也删）
#
# 用法（PowerShell）：
#   预览（默认，不删任何东西）:  powershell -ExecutionPolicy Bypass -File D:\LocalDreamZImage\scripts\cleanup_end_of_project.ps1
#   真正执行:                    powershell -ExecutionPolicy Bypass -File D:\LocalDreamZImage\scripts\cleanup_end_of_project.ps1 -Execute
#
# 做法（DISK_INVENTORY / 指南 §21.4 的纪律）：白名单删除 —— 只删下面明确列出的路径，其余一概不碰；
# 删前把清单与大小写进 logs\cleanup_20260925\deleted_manifest.txt；删后核对保留清单全部还在。
param([switch]$Execute)

$ErrorActionPreference = 'Stop'

# ---- 要删的（白名单）：路径 + 理由 ----
$targets = @(
    @('D:\ZImage_Work\p0_experiments',                        '各轮实验 DLC/context；结论已入文档'),
    @('D:\ZImage_Work\ZImage_QNN_Evidence\onnx',              '源 ONNX（用户确认不再开发，可删）'),
    @('D:\ZImage_Work\ZImage_QNN_Evidence\calibration',       '校准数据'),
    @('D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline',      '早期 DLC 流水线'),
    @('D:\ZImage_Work\TIER2_L80',                             '文本编码器 DLC/context'),
    @('D:\ZImage_Work\package',                               '9-20 交付包；模型文件与 HF 发布包逐字节相同'),
    @('D:\ZImage_Work\publish\ZImageTurbo_A16W8_SM8750_qnn2.48.zip',    '发布包；HF 上已有并核对 sha256'),
    @('D:\ZImage_Work\publish\hf\ZImageTurbo_A16W8_SM8750_qnn2.48.zip', '上面那个 zip 的硬链接'),
    @('D:\ZImage_Work\device_only_backup_20260917',           '9-17 设备侧备份'),
    @('D:\ZImage_Work\venv-official',                         '官方对拍用 Python 环境'),
    @('D:\Z-Image-Turbo',                                     '原始权重；可从 HF 重新下载'),
    @('D:\ZIMAGE',                                            '早期部署暂存'),
    @('D:\LocalDreamZImage\local-dream\app\.cxx',             'NDK 编译缓存；重编会自动生成'),
    @('D:\LocalDreamZImage\local-dream\app\build',            'Gradle 编译输出；重编会自动生成')
)

# ---- 必须保留的（删后逐个核对还在）----
$keep = @(
    "$env:USERPROFILE\.android\debug.keystore",
    'D:\qairt\2.48.0.260626',
    'D:\LocalDreamZImage\docs\QNN_CONVERSION_GUIDE.md',
    'D:\LocalDreamZImage\docs\DELIVERY_FROZEN.md',
    'D:\LocalDreamZImage\MAINLINE.md',
    'D:\LocalDreamZImage\CLAUDE.md',
    'D:\LocalDreamZImage\scripts',
    'D:\LocalDreamZImage\local-dream\.git',
    'D:\LocalDreamZImage\local-dream\qairt',
    'D:\LocalDreamZImage\host-protoc',
    'D:\LocalDreamZImage\logs\apk_backup_20260925',
    'D:\ZImage_Work\publish\LocalDreamZImage_armv8a_2.8.1-zit2.apk',
    'D:\ZImage_Work\publish\README.md',
    'D:\ZImage_Work\ZImage_QNN_Evidence\model_revision.txt'
)

function Get-Size([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return -1 }
    $i = Get-Item -LiteralPath $p -Force
    if (-not $i.PSIsContainer) { return $i.Length }
    $s = (Get-ChildItem -LiteralPath $p -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($s) { return $s } else { return 0 }
}

# 删前自检：保留清单必须全在（否则说明环境与清点时不同，停下）
$missing = $keep | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($missing) { Write-Host "停止：以下应保留的东西现在就不在，环境与清点时不一致：" -ForegroundColor Red; $missing; exit 1 }

# 保险：白名单里任何一项都不许是保留项或其上级目录
foreach ($t in $targets) {
    foreach ($k in $keep) {
        if ($k.StartsWith($t[0] + '\') -or $k -eq $t[0]) { Write-Host "停止：删除项 $($t[0]) 包含保留项 $k" -ForegroundColor Red; exit 1 }
    }
}

$rows = foreach ($t in $targets) { [pscustomobject]@{ GB = [math]::Round((Get-Size $t[0]) / 1GB, 2); Path = $t[0]; Why = $t[1] } }
$rows | Format-Table -AutoSize | Out-String -Width 220 | Write-Host
$total = ($rows | Where-Object { $_.GB -gt 0 } | Measure-Object GB -Sum).Sum
Write-Host ("合计约 {0:N1} GB（硬链接的 zip 只算一次则约少 12 GB）" -f $total)
Write-Host "另：WSL 发行版 Ubuntu2404（D:\WSL，约 8 GB）用 wsl --unregister 删除（见末尾）。"

if (-not $Execute) {
    Write-Host "`n这是预览，什么都没删。确认无误后加 -Execute 重新运行。" -ForegroundColor Yellow
    exit 0
}

$logDir = 'D:\LocalDreamZImage\logs\cleanup_20260925'
New-Item -ItemType Directory -Force $logDir | Out-Null
$rows | Format-Table -AutoSize | Out-String -Width 220 | Set-Content "$logDir\deleted_manifest.txt" -Encoding utf8

foreach ($r in $rows) {
    if ($r.GB -lt 0) { Write-Host "跳过（已不存在）: $($r.Path)"; continue }
    Write-Host "删除: $($r.Path)"
    Remove-Item -LiteralPath $r.Path -Recurse -Force
}

Write-Host "`nWSL：注销 Ubuntu2404 并删除其目录"
wsl.exe --unregister Ubuntu2404
if (Test-Path -LiteralPath 'D:\WSL') { Remove-Item -LiteralPath 'D:\WSL' -Recurse -Force }

# 删后核对
$missing = $keep | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($missing) { Write-Host "警告：以下保留项不见了：" -ForegroundColor Red; $missing } else { Write-Host "`n保留项全部仍在 ✅" -ForegroundColor Green }
Write-Host ("D 盘可用：{0:N0} GB" -f ((Get-PSDrive D).Free / 1GB))
