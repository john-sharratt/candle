$models = @(
    "qwen25-1b-q2",
    "qwen25-1b-q3",
    "qwen25-7b-q5",
    "qwen25-7b-f16",
    "qwen30-8b-q5",
    "qwen30-8b-f16",
    "qwen25-14b-q5",
    "qwen25-14b-q6"
)

$prompt = "The quick brown fox jumps over the lazy dog. This is a test of a much longer prompt to see how fast we can process input tokens. We want to measure the throughput of the model when given a large amount of text to process at once. This will help us understand if there are any bottlenecks in the input processing path. Let me add more text here to make this prompt even longer and more representative of real-world usage where users might paste in large documents or have long conversations. The model should be able to handle this efficiently with the GPU acceleration we have enabled. More text here to reach a good token count for testing purposes. Adding even more content to ensure we have a substantial number of tokens to process."

$results = @()

foreach ($model in $models) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Testing: $model" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    $output = & .\target\release\examples\quantized-qwen2-instruct.exe --which $model --prompt $prompt --sample-len 50 --temperature 0 2>&1 | Out-String
    
    Write-Host $output
    
    # Parse the output for performance metrics
    if ($output -match "(\d+)\s+prompt tokens processed:\s+([\d.]+)\s+token/s") {
        $promptTokens = $matches[1]
        $promptSpeed = $matches[2]
    }
    
    if ($output -match "(\d+)\s+tokens generated:\s+([\d.]+)\s+token/s") {
        $generatedTokens = $matches[1]
        $generationSpeed = $matches[2]
    }
    
    if ($output -match "loaded \d+ tensors \(([^)]+)\)") {
        $modelSize = $matches[1]
    }
    
    $results += [PSCustomObject]@{
        Model = $model
        Size = $modelSize
        PromptSpeed = $promptSpeed
        GenerationSpeed = $generationSpeed
    }
}

Write-Host "`n`n========================================" -ForegroundColor Green
Write-Host "BENCHMARK RESULTS" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

$results | Format-Table -AutoSize

Write-Host "`nMarkdown Table:" -ForegroundColor Cyan
Write-Host "| Model | Size | Prompt Speed (t/s) | Generation Speed (t/s) |"
Write-Host "|-------|------|-------------------|-----------------------|"
foreach ($r in $results) {
    Write-Host "| $($r.Model) | $($r.Size) | $($r.PromptSpeed) | $($r.GenerationSpeed) |"
}
