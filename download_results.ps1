$env:PYTHONUTF8=1

$files = @(
    "results/baselines.json",
    "results/bit_importance.json",
    "results/bit_importance.csv",
    "results/progressive_1bit_sweep.json",
    "results/layer_sensitivity.csv",
    "results/tensor_sensitivity.csv",
    "results/error_propagation.csv",
    "results/correlation_matrix.csv",
    "results/representation_statistics.csv"
)

foreach ($f in $files) {
    Write-Host "Downloading $f ..."
    modal volume get upr-data-vol $f $f
}
Write-Host "Done!"
