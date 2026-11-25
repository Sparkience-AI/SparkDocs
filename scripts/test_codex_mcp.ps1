$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
$originalEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $uv = (Get-Command uv -CommandType Application).Source
    $fixture = New-Item -ItemType Directory -Force 'benchmark-results/codex-mcp-sample'
    & $uv sync --locked --extra mcp
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the SparkDocs MCP dependencies.' }

    & $uv run --locked --extra mcp python -c "from pathlib import Path; from sparkdocs.sample import sample_pdf; Path('benchmark-results/codex-mcp-sample/maple.pdf').write_bytes(sample_pdf())"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the fictional Maple handbook.' }

    $python = (Resolve-Path '.venv/Scripts/python.exe').Path
    $allowedRoot = $fixture.FullName
    $cache = if ($env:FASTEMBED_CACHE_PATH) {
        $env:FASTEMBED_CACHE_PATH
    } else {
        Join-Path ([IO.Path]::GetTempPath()) 'fastembed_cache'
    }
    $cache = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($cache)
    if ($python.Contains("'") -or $allowedRoot.Contains("'") -or $cache.Contains("'")) {
        throw 'The smoke test requires a checkout path without apostrophes.'
    }
    Write-Host 'Preparing the local retrieval models. First use may download model weights.'
    $warmup = [Diagnostics.Stopwatch]::StartNew()
    & $python -c "import os,sys; from pathlib import Path; os.environ['FASTEMBED_CACHE_PATH']=sys.argv[1]; from sparkdocs.mcp_workspace import DocumentWorkspace; w=DocumentWorkspace(Path(sys.argv[2])); assert w.ingest('maple.pdf').loaded; result=w.search('How long do I have to file a battery claim for a device used outdoors?'); assert 'PDF page 3' in str(result) and '7 calendar days' in str(result)" $cache $allowedRoot
    if ($LASTEXITCODE -ne 0) { throw 'Local retrieval warm-up failed before Codex was started.' }
    Write-Host ('Local warm-up completed in {0:F1}s.' -f $warmup.Elapsed.TotalSeconds)
    # TOML literal strings survive PowerShell native argument forwarding intact.
    $expected = @('ingest_document', 'search_document', 'clear_document', 'document_status')
    $toolPolicy = ($expected | ForEach-Object { "$_={approval_mode='approve'}" }) -join ','
    $enabledTools = ($expected | ForEach-Object { "'$_'" }) -join ','
    $config = "mcp_servers.sparkdocs={command='$python',args=['-m','sparkdocs.mcp_server','--allowed-root','$allowedRoot','--warm-models'],env={FASTEMBED_CACHE_PATH='$cache',HF_HUB_OFFLINE='1'},startup_timeout_sec=60,tool_timeout_sec=60,enabled_tools=[$enabledTools],tools={$toolPolicy}}"
    $prompt = @'
Use only SparkDocs MCP tools. Ingest maple.pdf, then search for the deadline for filing a battery
claim for a device used outdoors. Cite the PDF page in your final answer. Then clear the document
and call document_status to confirm it is unloaded. Report the deadline, citation, and final status.
'@
    Write-Host 'Running the Codex MCP smoke test with the warmed model cache.'
    $eventLog = Join-Path $fixture.FullName 'codex-events.jsonl'
    $toolStarts = @{}
    $items = & codex exec --json --ephemeral --skip-git-repo-check --sandbox read-only -c $config $prompt |
        Tee-Object -FilePath $eventLog | ForEach-Object {
            if ($_ -match '^\s*\{') {
                $event = $_ | ConvertFrom-Json
                if ($event.item.type -eq 'mcp_tool_call' -and $event.item.server -eq 'sparkdocs') {
                    if ($event.type -eq 'item.started') {
                        $toolStarts[$event.item.id] = [Diagnostics.Stopwatch]::StartNew()
                        Write-Host "Starting $($event.item.tool)..."
                    } elseif ($event.type -eq 'item.completed' -and $toolStarts.ContainsKey($event.item.id)) {
                        $elapsed = $toolStarts[$event.item.id].Elapsed.TotalSeconds
                        Write-Host ('{0}: {1} in {2:F1}s' -f $event.item.tool, $event.item.status, $elapsed)
                    }
                }
                $event
            }
        }
    if ($LASTEXITCODE -ne 0) {
        throw "Codex CLI exited $LASTEXITCODE. Events: $eventLog"
    }
    $items | Where-Object {
        $_.type -eq 'item.completed' -and $_.item.type -eq 'agent_message'
    } | ForEach-Object { $_.item.text }
    $toolEvents = @($items | Where-Object {
        $_.type -eq 'item.completed' -and $_.item.type -eq 'mcp_tool_call'
    } | ForEach-Object { $_.item })
    $calls = @($toolEvents | Where-Object {
        $_.server -eq 'sparkdocs' -and $_.status -ne 'failed' -and
        $_.result -and -not $_.error -and -not $_.result.isError
    })
    if (($calls.tool -join ',') -ne ($expected -join ',')) {
        $toolEvents | Select-Object server, tool, status, error | Format-List | Out-Host
        $items | Where-Object { $_.type -in 'error', 'turn.failed' } |
            ConvertTo-Json -Depth 10 | Out-Host
        throw "Expected four successful SparkDocs calls in order. Observed: $($calls.tool -join ', '). Events: $eventLog"
    }
    $evidence = $calls[1].result | ConvertTo-Json -Depth 30 -Compress
    $status = $calls[3].result | ConvertTo-Json -Depth 30 -Compress
    if ($evidence -notmatch 'PDF page 3' -or $evidence -notmatch '7 calendar days' -or
        ($status -replace '\\', '') -notmatch '"loaded"\s*:\s*false') {
        throw 'MCP returned unexpected evidence or the document was still loaded after clearing.'
    }
    Write-Host 'PASS: Codex called all four tools, found the seven-day deadline on page 3, and cleared the document.'
} finally {
    [Console]::OutputEncoding = $originalEncoding
    Pop-Location
}
