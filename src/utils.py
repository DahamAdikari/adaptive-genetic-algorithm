import random
import math
from src.models import Customer, Instance

def compute_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def initialize_population(instance, pop_size, use_static_greedy=False):
    population = []
    customer_list = instance.customers.copy()
    for _ in range(pop_size):
        if use_static_greedy and random.random() < 0.3:
            sorted_customers = sorted(customer_list, key=lambda c: c.ready_time)
        else:
            sorted_customers = random.sample(customer_list, len(customer_list))
        population.append(sorted_customers)
    return population

def selection(population):
    p1 = random.choice(population)[0]
    p2 = random.choice(population)[0]
    return p1, p2

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

def mutation_swap(child):
    a, b = random.sample(range(len(child)), 2)
    child[a], child[b] = child[b], child[a]
    return child

def evaluate(routes, depot=None):
    if depot is None:
        print("Warning: Depot not passed to evaluate().")
        return float('inf')

    total_distance = 0.0
    for route in routes:
        if not route:
            continue
        prev = depot
        for customer in route:
            total_distance += compute_distance(prev, customer)
            prev = customer
        total_distance += compute_distance(prev, depot)  # Return to depot
    return total_distance

def select_elites(population, k):
    return sorted(population, key=lambda x: x[1])[:k]
