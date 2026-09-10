# PowerShell script to mount SSHFS-Win using .env configuration
Param(
    [string]$EnvFile = ".env",
    [switch]$Unmount,
    [switch]$Check
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EnvPath = Join-Path $ScriptDir $EnvFile

# Default settings
$SshHost = "192.168.1.180"
$SshUser = "root"
$SshPort = "22"
$SshPassword = ""
$RemotePath = "/"
$MountDrive = "Y:"
$VolName = "jellyfin docker lxc"
$SshfsExe = "C:\Program Files\SSHFS-Win\bin\sshfs.exe"
$Debug = $true
$LogLevel = "debug1"
$MaxReadahead = "1GB"

# Parse .env if present
if (Test-Path $EnvPath) {
    Get-Content $EnvPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            if ($line.StartsWith("export ")) { $line = $line.Substring(7).Trim() }
            $parts = $line.Split("=", 2)
            $k = $parts[0].Trim()
            $v = $parts[1].Trim().Trim('"').Trim("'")
            switch ($k) {
                "SSH_HOST" { $SshHost = $v }
                "SSH_USER" { $SshUser = $v }
                "SSH_PORT" { $SshPort = $v }
                "SSH_PASSWORD" { $SshPassword = $v }
                "SSH_REMOTE_PATH" { $RemotePath = $v }
                "SSHFS_MOUNT_DRIVE" { $MountDrive = $v }
                "SSHFS_VOLNAME" { $VolName = $v }
                "SSHFS_EXE" { $SshfsExe = $v }
                "SSHFS_DEBUG" { $Debug = ($v -eq "true" -or $v -eq "1") }
                "SSHFS_LOGLEVEL" { $LogLevel = $v }
                "SSHFS_MAX_READAHEAD" { $MaxReadahead = $v }
            }
        }
    }
}

if (-not $MountDrive.EndsWith(":")) {
    $MountDrive = "$MountDrive`:"
}

if ($Check) {
    if (Test-Path "$MountDrive\") {
        Write-Host "Drive $MountDrive is ACCESSIBLE." -ForegroundColor Green
        Exit 0
    } else {
        Write-Host "Drive $MountDrive is NOT ACCESSIBLE / UNMOUNTED." -ForegroundColor Red
        Exit 1
    }
}

if ($Unmount) {
    Write-Host "Unmounting $MountDrive..." -ForegroundColor Yellow
    net use $MountDrive /delete /y
    Exit $LASTEXITCODE
}

if (Test-Path "$MountDrive\") {
    Write-Host "Drive $MountDrive is already mounted and accessible." -ForegroundColor Green
    Exit 0
}

if (-not (Test-Path $SshfsExe)) {
    Write-Host "Error: sshfs.exe not found at $SshfsExe" -ForegroundColor Red
    Exit 1
}

$target = "$SshUser@$SshHost`:$RemotePath"
$sshfsArgs = @(
    $target,
    $MountDrive,
    "-p$SshPort",
    "-ovolname=$VolName"
)

if ($Debug) {
    $sshfsArgs += "-odebug"
    if ($LogLevel) {
        $sshfsArgs += "-ologlevel=$LogLevel"
    }
}

$sshfsArgs += @(
    "-oStrictHostKeyChecking=no",
    "-oUserKnownHostsFile=/dev/null",
    "-oidmap=user",
    "-ouid=-1",
    "-ogid=-1",
    "-oumask=000",
    "-ocreate_umask=000",
    "-omax_readahead=$MaxReadahead",
    "-oallow_other",
    "-olarge_read",
    "-okernel_cache",
    "-ofollow_symlinks",
    "-oPreferredAuthentications=password",
    "-opassword_stdin"
)

Write-Host "Mounting $target to $MountDrive..." -ForegroundColor Cyan

$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = $SshfsExe
$pinfo.Arguments = ($sshfsArgs -join " ")
$pinfo.RedirectStandardInput = $true
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true

$p = [System.Diagnostics.Process]::Start($pinfo)
if ($SshPassword) {
    $p.StandardInput.WriteLine($SshPassword)
    $p.StandardInput.Flush()
}

Start-Sleep -Seconds 2

if (Test-Path "$MountDrive\") {
    Write-Host "Successfully mounted $MountDrive to $target" -ForegroundColor Green
    Exit 0
} else {
    Write-Host "Mount issued. Checking status..." -ForegroundColor Yellow
    for ($i = 0; $i -lt 6; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path "$MountDrive\") {
            Write-Host "Successfully mounted $MountDrive to $target" -ForegroundColor Green
            Exit 0
        }
    }
    Write-Host "Drive $MountDrive not accessible yet. Please check credentials." -ForegroundColor Red
    Exit 1
}
