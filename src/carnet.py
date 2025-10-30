"""
bayesian network for the car starting problem.
"""

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
])

# Defining the parameters using CPT


cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

def main():
    print("=" * 60)
    print("CAR NETWORK QUERIES")
    print("=" * 60)
    
    # p(battery | moves=no)
    print("\n1. P(Battery = Doesn't work | Moves = no):")
    q1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print(q1)
    
    # p(starts | radio broken)
    print("\n2. P(Starts = no | Radio = Doesn't turn on):")
    q2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print(q2)
    
    # independence test: does p(radio | battery) change with gas?
    print("\n3. P(Radio | Battery = Works):")
    q3a = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    print(q3a)
    
    print("\n   P(Radio | Battery = Works, Gas = Full):")
    q3b = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print(q3b)
    
    # explaining away: how does gas observation affect ignition probability?
    print("\n4. P(Ignition | Moves = no):")
    q4a = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    print(q4a)
    
    print("\n   P(Ignition | Moves = no, Gas = Empty):")
    q4b = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print(q4b)
    
    # p(starts | radio works, has gas)
    print("\n5. P(Starts | Radio = turns on, Gas = Full):")
    q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print(q5)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

