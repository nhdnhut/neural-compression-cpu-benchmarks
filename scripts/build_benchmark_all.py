import sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from profiling.reporter import save_report_csv

def main():
    results_dir = ROOT / "benchmarks" / "results"
    
    # 1. Baseline
    baseline_path = results_dir / "baseline.csv"
    df_base = pd.read_csv(baseline_path)
    print(f"Loaded {len(df_base)} baseline rows.")
    
    # 2. Quantization
    quant_path = results_dir / "quantization_sweep.csv"
    df_quant = pd.read_csv(quant_path)
    # only keep dynamic_int8 rows or unique rows
    df_quant = df_quant[df_quant["quantization"] == "dynamic_int8"]
    print(f"Loaded {len(df_quant)} quantization rows.")
    
    # 3. Low-rank
    lr_path = results_dir / "low_rank_sweep.csv"
    df_lr = pd.read_csv(lr_path)
    # skip rank_ratio == 1.0 because that's identical to baseline
    df_lr = df_lr[df_lr["rank_ratio"] < 1.0]
    print(f"Loaded {len(df_lr)} low-rank rows.")
    
    # 4. Pruning
    pruning_path = results_dir / "pruning_sweep.csv"
    df_prune = pd.read_csv(pruning_path)
    # If seed 42 is missing from pruning_sweep.csv, pull it from existing benchmark_all.csv
    if 42 not in df_prune["seed"].values:
        old_bench = results_dir / "benchmark_all.csv"
        if old_bench.exists():
            df_old = pd.read_csv(old_bench)
            old_p42 = df_old[(df_old["method"] == "pruning") & (df_old["seed"] == 42)]
            df_prune = pd.concat([old_p42, df_prune], ignore_index=True)
    # skip pruning_ratio == 0.0 because that's baseline
    df_prune = df_prune[df_prune["pruning_ratio"] > 0.0]
    print(f"Loaded {len(df_prune)} pruning rows.")
    
    # Merge all rows
    all_dfs = [df_base, df_prune, df_quant, df_lr]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    master_csv = results_dir / "benchmark_all.csv"
    combined_df.to_csv(master_csv, index=False)
    print(f"Saved {len(combined_df)} rows to {master_csv}")
    
    # Aggregate summary
    numeric_cols = [
        "test_accuracy", "test_macro_f1", "latency_mean_ms",
        "model_size_param_mb", "MACs_M", "global_sparsity", "params_trainable"
    ]
    numeric_cols = [c for c in numeric_cols if c in combined_df.columns]
    
    summary = (
        combined_df.groupby(["method", "pruning_ratio", "quantization", "rank_ratio"])[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = results_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    # Generate tables and macros
    print("\n--- Running generate_tables.py ---")
    res_gen = subprocess.run([sys.executable, "report/generate_tables.py", "--csv", str(master_csv)], cwd=str(ROOT), capture_output=True, text=True)
    print(res_gen.stdout)
    if res_gen.stderr:
        print(res_gen.stderr)
        
    # Generate Pareto plots
    print("\n--- Updating Pareto Figures ---")
    try:
        import shutil
        from visualization.pareto import main as gen_pareto
        gen_pareto([])
        fig_src = ROOT / "benchmarks" / "figures"
        fig_dst = ROOT / "report" / "figures"
        fig_dst.mkdir(parents=True, exist_ok=True)
        for img in fig_src.glob("*.png"):
            shutil.copy2(img, fig_dst / img.name)
        print("Pareto dashboard and frontier figures generated and copied to report/figures.")
    except Exception as e:
        print(f"Warning generating Pareto figures: {e}")
        
    # Rebuild PDF
    print("\n--- Compiling main.tex to main.pdf ---")
    report_dir = ROOT / "report"
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=str(report_dir), capture_output=True)
    subprocess.run(["bibtex", "main"], cwd=str(report_dir), capture_output=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=str(report_dir), capture_output=True)
    res_pdf = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=str(report_dir), capture_output=True, text=True)
    if (report_dir / "main.pdf").exists():
        size_kb = (report_dir / "main.pdf").stat().st_size / 1024
        print(f"SUCCESS: report/main.pdf generated! Size: {size_kb:.1f} KB")
    else:
        print("ERROR: main.pdf not found!")

if __name__ == "__main__":
    main()
