import pandas as pd
from src.models import Customer, Instance

def load_instance(filepath):
    df = pd.read_csv(filepath)
    customers = []
    for _, row in df.iterrows():
        cust = Customer(
            id=int(row["CUST NO."]),
            x=row["XCOORD."],
            y=row["YCOORD."],
            demand=row["DEMAND"],
            ready_time=row["READY TIME"],
            due_time=row["DUE DATE"],
            service_time=row["SERVICE TIME"]
        )
        customers.append(cust)

    depot = customers[0]
    return Instance(customers[1:], depot, vehicle_capacity=200)
