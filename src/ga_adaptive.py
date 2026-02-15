import random
from .utils import compute_distance, evaluate

# === Multi-Greedy Sorting Strategies ===
def sort_by_ready_time(customers):
    return sorted(customers, key=lambda c: c.ready_time)

def sort_by_demand(customers):
    return sorted(customers, key=lambda c: c.demand)

def sort_by_distance_to_depot(customers, depot):
    return sorted(customers, key=lambda c: compute_distance(c, depot))

def split_routes_by_capacity(chromosome, capacity):
    routes = []
    current_route = []
    current_load = 0

    for cust in chromosome:
        if current_load + cust.demand <= capacity:
            current_route.append(cust)
            current_load += cust.demand
        else:
            routes.append(current_route)
            current_route = [cust]
            current_load = cust.demand

    if current_route:
        routes.append(current_route)

    return routes

# === Adaptation Function Using Multi-Greedy ===
def apply_best_greedy_injection(instance, depot):
    strategies = [
        ("ready_time", lambda: sort_by_ready_time(instance.customers)),
        ("demand", lambda: sort_by_demand(instance.customers)),
        ("distance_to_depot", lambda: sort_by_distance_to_depot(instance.customers, depot)),
    ]

    best_order = None
    best_distance = float('inf')
    best_label = ""

    for label, strategy in strategies:
        ordered = strategy()
        routes = split_routes_by_capacity(ordered, instance.vehicle_capacity)
        dist = evaluate(routes, depot)
        if dist < best_distance:
            best_distance = dist
            best_order = ordered
            best_label = label

    return best_order, best_label, best_distance

# === Hybrid + Adaptive GA Loop ===
def run_ga_adaptive(instance, generations=100, pop_size=50):
    depot = instance.depot
    population = []
    fitness_log = []
    adapt_events = []

    for _ in range(int(pop_size * 0.1)):
        population.append(sort_by_ready_time(instance.customers))

    for _ in range(pop_size - len(population)):
        shuffled = random.sample(instance.customers, len(instance.customers))
        population.append(shuffled)

    def evaluate_fitness(chrom):
        routes = split_routes_by_capacity(chrom, instance.vehicle_capacity)
        return evaluate(routes, depot)

    best_chrom = None
    best_distance = float('inf')
    stagnation_counter = 0

    for gen in range(generations):
        scored = [(chrom, evaluate_fitness(chrom)) for chrom in population]
        scored.sort(key=lambda x: x[1])

        elites = [chrom for chrom, _ in scored[:5]]
        elite_scores = [evaluate_fitness(elite) for elite in elites]

        print(f"Generation {gen}: Best Distance = {scored[0][1]:.2f}")
        print(f"Generation {gen}: Elite fitnesses = {[f'{score:.2f}' for score in elite_scores]}")

        if scored[0][1] < best_distance:
            best_chrom = scored[0][0]
            best_distance = scored[0][1]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        fitness_log.append((gen, best_distance))

        if stagnation_counter >= 5:
            greedy_injected, strategy, dist = apply_best_greedy_injection(instance, depot)
            worst_elite_score = elite_scores[-1]
            if dist < worst_elite_score:
                elites[-1] = greedy_injected
                print(f"Generation {gen}: [ADAPT] Injected '{strategy}' greedy → {dist:.2f} replaces elite ({worst_elite_score:.2f})")
                adapt_events.append((gen, strategy))
                elite_scores[-1] = dist
            else:
                print(f"Generation {gen}: [ADAPT] Injected '{strategy}' greedy → {dist:.2f} discarded (worst elite = {worst_elite_score:.2f})")
            stagnation_counter = 0
            print(f"Generation {gen}: Elite fitnesses after adaptation = {[f'{score:.2f}' for score in elite_scores]}")

        next_gen = elites.copy()

        while len(next_gen) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            cut = random.randint(1, len(p1) - 2)
            child = p1[:cut] + [c for c in p2 if c not in p1[:cut]]

            if random.random() < 0.2:
                a, b = random.sample(range(len(child)), 2)
                child[a], child[b] = child[b], child[a]

            if random.random() < 0.1:
                child = sort_by_ready_time(child)

            next_gen.append(child)

        population = next_gen

    return best_chrom, best_distance, fitness_log, adapt_events
