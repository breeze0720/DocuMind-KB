# 从 ModelScope 下载 BAAI/bge-small-zh-v1.5 的 ONNX 版本（fastembed 需要的文件）
# 用法（在 Agent Project1 test 目录下，任选一种）：
#   1) .\download_model.ps1
#   2) powershell -ExecutionPolicy Bypass -File .\download_model.ps1
#      （如果提示"未对文件进行数字签名"，用第 2 种）
# 下载完成后 models\bge-small-zh-v1.5 约 91MB，之后启动完全不需要联网。
$ErrorActionPreference = "Stop"

$dest = Join-Path $PSScriptRoot "models\bge-small-zh-v1.5"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$base = "https://modelscope.cn/api/v1/models/Qdrant/bge-small-zh-v1.5/repo?Revision=master&FilePath="
$files = @(
    "config.json",
    "ort_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model_optimized.onnx"
)

foreach ($f in $files) {
    $out = Join-Path $dest $f
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 0)) {
        Write-Host ("跳过（已存在） {0}" -f $f)
        continue
    }
    Write-Host ("下载 {0} ..." -f $f)
    curl.exe -L --retry 3 --retry-delay 2 --max-time 900 -o $out ($base + $f)
    if ($LASTEXITCODE -ne 0) { throw "下载失败: $f" }
    Write-Host ("  -> {0:N0} 字节" -f (Get-Item $out).Length)
}

Write-Host ""
Write-Host "完成，目录：$dest"
Get-ChildItem $dest | ForEach-Object { "{0,-28} {1,12:N0}" -f $_.Name, $_.Length }