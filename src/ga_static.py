import random
from src.greedy import static_greedy_repair
from src.utils import (
    initialize_population,
    selection,
    crossover_ox,
    mutation_swap,
    evaluate,
    select_elites
)

MAX_GENERATIONS = 100
POP_SIZE = 50

def run_ga_solver(instance):
    population = initialize_population(instance, POP_SIZE, use_static_greedy=True)
    evaluated_population = []
    fitness_history = []

    for chrom in population:
        routes = static_greedy_repair(chrom, instance)
        fitness = evaluate(routes, depot=instance.depot)
        evaluated_population.append((chrom, fitness))

    for gen in range(MAX_GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE):
            parent1, parent2 = selection(evaluated_population)
            child = crossover_ox(parent1, parent2)
            child = mutation_swap(child)
            routes = static_greedy_repair(child, instance)
            fitness = evaluate(routes, depot=instance.depot)
            new_population.append((child, fitness))

        evaluated_population = select_elites(evaluated_population + new_population, POP_SIZE)

        # Track best fitness per generation
        best_fitness = min(evaluated_population, key=lambda x: x[1])[1]
        fitness_history.append(best_fitness)

        if gen % 10 == 0 or gen == MAX_GENERATIONS - 1:
            print(f"Generation {gen}: Best Distance = {best_fitness:.2f}")

    best_solution = min(evaluated_population, key=lambda x: x[1])
    return best_solution[0], best_solution[1], fitness_history

