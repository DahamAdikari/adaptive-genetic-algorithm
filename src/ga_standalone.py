import random
import math
from src.utils import compute_distance

POP_SIZE = 50
MAX_GENERATIONS = 100
MUTATION_RATE = 0.2


def initialize_population(customers):
    population = []
    for _ in range(POP_SIZE):
        chrom = random.sample(customers, len(customers))
        population.append(chrom)
    return population


def evaluate(chromosome, depot):
    total_dist = compute_distance(depot, chromosome[0])
    for i in range(len(chromosome) - 1):
        total_dist += compute_distance(chromosome[i], chromosome[i + 1])
    total_dist += compute_distance(chromosome[-1], depot)
    return total_dist


def tournament_selection(population, depot):
    a, b = random.sample(population, 2)
    return a if evaluate(a, depot) < evaluate(b, depot) else b


def crossover_ox(p1, p2):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = p1[start:end]
    fill = [c for c in p2 if c not in child]
    ptr = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[ptr]
            ptr += 1
    return child


def mutate_swap(chrom):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def run_ga_standalone(instance):
    customers = instance.customers
    depot = instance.depot
    population = initialize_population(customers)
    best_fitness_per_gen = []

    for gen in range(MAX_GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE):
            p1 = tournament_selection(population, depot)
            p2 = tournament_selection(population, depot)
            child = crossover_ox(p1, p2)
            child = mutate_swap(child)
            new_population.append(child)

        population = new_population
        best = min(population, key=lambda chrom: evaluate(chrom, depot))
        best_fitness = evaluate(best, depot)
        best_fitness_per_gen.append(best_fitness)

        if gen % 10 == 0 or gen == MAX_GENERATIONS - 1:
            print(f"Generation {gen}: Best Distance = {best_fitness:.2f}")

    best = min(population, key=lambda chrom: evaluate(chrom, depot))
    return best, evaluate(best, depot), best_fitness_per_gen
