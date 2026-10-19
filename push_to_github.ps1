param(
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$Repo = "navigation-report",
    [string]$User = "orange09-26"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$headers = @{
    Authorization = "token $Token"
    "User-Agent"  = "codex-desktop"
    Accept        = "application/vnd.github+json"
}

$body = @{
    name        = $Repo
    description = "Navigation experiment reports"
    private     = $false
} | ConvertTo-Json

try {
    $null = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post `
        -Headers $headers -ContentType "application/json; charset=utf-8" -Body $body
    Write-Host "repository created: $User/$Repo"
}
catch {
    Write-Host "create skipped or failed (may already exist): $($_.Exception.Message)"
}

if (git remote 2>$null | Select-String -Quiet '^origin$') { git remote remove origin }
git remote add origin "https://$User`:$Token@github.com/$User/$Repo.git"
git -c http.sslBackend=openssl -c http.sslCAInfo="D:\anaconda\lib\site-packages\certifi\cacert.pem" `
    push -u origin master:main
git remote set-url origin "https://github.com/$User/$Repo.git"
Write-Host "done: https://github.com/$User/$Repo"
