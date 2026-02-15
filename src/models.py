class Customer:
    def __init__(self, id, x, y, demand, ready_time, due_time, service_time):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time

class Instance:
    def __init__(self, customers, depot, vehicle_capacity):
        self.customers = customers
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity
