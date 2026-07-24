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
    "results/representation_statistics.csv",
    "results/upr_evaluation_report.json",
    "results/variable_precision_summary.json",
    "results/memory.csv"
)

foreach ($f in $files) {
    Write-Host "Downloading $f ..."
    modal volume get upr-data-vol $f $f --force
}
Write-Host "Done!"
