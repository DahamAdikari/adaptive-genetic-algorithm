from src.utils import compute_distance

def static_greedy_repair(chromosome, instance):
    routes = []
    current_route = []
    current_load = 0
    current_time = instance.depot.ready_time
    current_location = instance.depot

    for customer in chromosome:
        dist = compute_distance(current_location, customer)
        arrival_time = current_time + dist

        if (current_load + customer.demand <= instance.vehicle_capacity and
            arrival_time <= customer.due_time):

            current_route.append(customer)
            current_load += customer.demand
            current_time = max(arrival_time, customer.ready_time) + customer.service_time
            current_location = customer
        else:
            routes.append(current_route)
            current_route = [customer]
            current_load = customer.demand
            current_time = max(instance.depot.ready_time + compute_distance(instance.depot, customer), customer.ready_time) + customer.service_time
            current_location = customer

    if current_route:
        routes.append(current_route)
    return routes
