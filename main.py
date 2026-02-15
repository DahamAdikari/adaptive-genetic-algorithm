import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.loader import load_instance
from src.ga_static import run_ga_solver as run_static_ga
from src.ga_standalone import run_ga_standalone
from src.ga_adaptive import run_ga_adaptive

# Directories
RESULT_DIR = "outputs/results"
PLOTS_DIR = "outputs/plots"
FITNESS_DIR = "outputs/fitness_logs"
BENCHMARK_CSV = f"outputs/benchmark_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

os.makedirs(RESULT_DIR + "/hybrid", exist_ok=True)
os.makedirs(RESULT_DIR + "/standalone", exist_ok=True)
os.makedirs(RESULT_DIR + "/adaptive", exist_ok=True)
os.makedirs(PLOTS_DIR + "/hybrid", exist_ok=True)
os.makedirs(PLOTS_DIR + "/standalone", exist_ok=True)
os.makedirs(PLOTS_DIR + "/adaptive", exist_ok=True)
os.makedirs(FITNESS_DIR + "/hybrid", exist_ok=True)
os.makedirs(FITNESS_DIR + "/standalone", exist_ok=True)
os.makedirs(FITNESS_DIR + "/adaptive", exist_ok=True)

def plot_fitness_curve(fitness_history, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(fitness_history)), fitness_history, marker='o', linestyle='-', color='blue')
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

        # --- Static Hybrid GA ---
        best_chrom_h, best_dist_h, fitness_h = run_static_ga(instance)
        print("Static Hybrid GA Best:", best_dist_h)

        fitness_file_h = f"{FITNESS_DIR}/hybrid/fitness_{timestamp}_run{run+1}.csv"
        with open(fitness_file_h, "w") as f:
            f.write("Generation,BestDistance\n")
            for i, val in enumerate(fitness_h):
                if i % 10 == 0 or i == len(fitness_h)-1:
                    f.write(f"{i},{val:.2f}\n")

        plot_file_h = f"{PLOTS_DIR}/hybrid/plot_{timestamp}_run{run+1}.png"
        plot_fitness_curve(fitness_h, f"Hybrid GA Convergence Run {run+1}", plot_file_h)

        # --- Standalone GA ---
        best_chrom_s, best_dist_s, fitness_s = run_ga_standalone(instance)
        print("Standalone GA Best:", best_dist_s)

        fitness_file_s = f"{FITNESS_DIR}/standalone/fitness_{timestamp}_run{run+1}.csv"
        with open(fitness_file_s, "w") as f:
            f.write("Generation,BestDistance\n")
            for i, val in enumerate(fitness_s):
                if i % 10 == 0 or i == len(fitness_s)-1:
                    f.write(f"{i},{val:.2f}\n")

        plot_file_s = f"{PLOTS_DIR}/standalone/plot_{timestamp}_run{run+1}.png"
        plot_fitness_curve(fitness_s, f"Standalone GA Convergence Run {run+1}", plot_file_s)

        # --- Adaptive GA ---
        best_chrom_a, best_dist_a, fitness_a, adapt_events = run_ga_adaptive(instance)
        print("Adaptive GA Best:", best_dist_a)

        fitness_file_a = f"{FITNESS_DIR}/adaptive/fitness_{timestamp}_run{run+1}.csv"
        with open(fitness_file_a, "w") as f:
            f.write("Generation,BestDistance\n")
            for i, (gen, val) in enumerate(fitness_a):
                if gen % 10 == 0 or gen == fitness_a[-1][0]:
                    f.write(f"{gen},{val:.2f}\n")

        plot_file_a = f"{PLOTS_DIR}/adaptive/plot_{timestamp}_run{run+1}.png"
        plot_fitness_curve([val for _, val in fitness_a], f"Adaptive GA Convergence Run {run+1}", plot_file_a)

        adapt_str = "; ".join([f"Gen {gen}: {method}" for gen, method in adapt_events]) if adapt_events else "None"

        # --- Log benchmark ---
        benchmark_rows.append({
            "Run": run + 1,
            "StaticHybrid": round(best_dist_h, 2),
            "StandaloneGA": round(best_dist_s, 2),
            "AdaptiveGA": round(best_dist_a, 2),
            "Adaptations": adapt_str
        })

    # Save benchmark
    df = pd.DataFrame(benchmark_rows)
    df.to_csv(BENCHMARK_CSV, index=False)
    print(f"\nBenchmark results saved to {BENCHMARK_CSV}")