import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.loader import load_instance
from src.ga_static import run_ga_solver as run_static_ga
from src.ga_standalone import run_ga_standalone
from src.ga_adaptive import run_ga_adaptive

# Setup paths
RESULT_DIR = "outputs/results"
PLOTS_DIR = "outputs/plots"
FITNESS_DIR = "outputs/fitness_logs"
BENCHMARK_CSV = f"outputs/benchmark_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

for d in ["hybrid", "standalone", "adaptive"]:
    os.makedirs(f"{RESULT_DIR}/{d}", exist_ok=True)
    os.makedirs(f"{PLOTS_DIR}/{d}", exist_ok=True)
    os.makedirs(f"{FITNESS_DIR}/{d}", exist_ok=True)

def plot_fitness_curve(fitness_history, title, save_path, adapt_events=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(fitness_history)), fitness_history, marker='o', linestyle='-', color='blue')

    if adapt_events:
        for gen, method in adapt_events:
            if gen < len(fitness_history):
                plt.axvline(x=gen, color='red', linestyle='--', alpha=0.6)
                plt.text(gen, fitness_history[gen], f"{method}", rotation=90, color='red', fontsize=8, ha='right')

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    instance = load_instance('data/C103.csv')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_rows = []

    for run in range(10):
        print(f"\n--- Run {run+1}/10 ---")

        # --- Static Hybrid ---
        best_chrom_h, best_dist_h, fitness_h = run_static_ga(instance)
        print("Static Hybrid GA Best:", best_dist_h)
        with open(f"{FITNESS_DIR}/hybrid/fitness_{timestamp}_run{run+1}.csv", "w") as f:
            f.write("Generation,BestDistance\n")
            for i, val in enumerate(fitness_h):
                if i % 10 == 0 or i == len(fitness_h)-1:
                    f.write(f"{i},{val:.2f}\n")
        plot_fitness_curve(fitness_h, f"Hybrid GA Convergence Run {run+1}", f"{PLOTS_DIR}/hybrid/plot_{timestamp}_run{run+1}.png")

        # --- Standalone GA ---
        best_chrom_s, best_dist_s, fitness_s = run_ga_standalone(instance)
        print("Standalone GA Best:", best_dist_s)
        with open(f"{FITNESS_DIR}/standalone/fitness_{timestamp}_run{run+1}.csv", "w") as f:
            f.write("Generation,BestDistance\n")
            for i, val in enumerate(fitness_s):
                if i % 10 == 0 or i == len(fitness_s)-1:
                    f.write(f"{i},{val:.2f}\n")
        plot_fitness_curve(fitness_s, f"Standalone GA Convergence Run {run+1}", f"{PLOTS_DIR}/standalone/plot_{timestamp}_run{run+1}.png")

        # --- Adaptive GA ---
        best_chrom_a, best_dist_a, fitness_a, adapt_events = run_ga_adaptive(instance)
        print("Adaptive GA Best:", best_dist_a)
        with open(f"{FITNESS_DIR}/adaptive/fitness_{timestamp}_run{run+1}.csv", "w") as f:
            f.write("Generation,BestDistance\n")
            for gen, val in fitness_a:
                if gen % 10 == 0 or gen == fitness_a[-1][0]:
                    f.write(f"{gen},{val:.2f}\n")
        plot_fitness_curve([val for _, val in fitness_a], f"Adaptive GA Convergence Run {run+1}", f"{PLOTS_DIR}/adaptive/plot_{timestamp}_run{run+1}.png", adapt_events)

        adapt_str = "; ".join([f"Gen {gen}: {method}" for gen, method in adapt_events]) if adapt_events else "None"
        benchmark_rows.append({
            "Run": run + 1,
            "StaticHybrid": round(best_dist_h, 2),
            "StandaloneGA": round(best_dist_s, 2),
            "AdaptiveGA": round(best_dist_a, 2),
            "Adaptations": adapt_str
        })

    df = pd.DataFrame(benchmark_rows)
    df.to_csv(BENCHMARK_CSV, index=False)
    print(f"\n✅ Benchmark results saved to {BENCHMARK_CSV}")
