"""
extended car network with keypresent node.
"""

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# car model with keypresent node added
car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
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

cpd_keypresent = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.70], [0.30]],
    state_names={"KeyPresent":['yes', 'no']},
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

# starts cpd with keypresent as additional parent (8 combinations)
cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts":['yes','no'], 
        "Ignition":["Works", "Doesn't work"], 
        "Gas":['Full',"Empty"],
        "KeyPresent":['yes', 'no']
    },
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)
car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)

car_infer = VariableElimination(car_model)

def main():
    print("=" * 60)
    print("EXTENDED CAR NETWORK WITH KEYPRESENT")
    print("=" * 60)
    
    # p(keypresent | moves=no)
    print("\n1. P(KeyPresent = no | Moves = no):")
    q1 = car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"})
    print(q1)
    
    # show impact of keypresent on starting
    print("\n2. P(Starts = yes | KeyPresent = yes, Gas = Full, Ignition = Works):")
    q2 = car_infer.query(variables=["Starts"], evidence={"KeyPresent": "yes", "Gas": "Full", "Ignition": "Works"})
    print(q2)
    
    print("\n3. P(Starts = yes | KeyPresent = no, Gas = Full, Ignition = Works):")
    q3 = car_infer.query(variables=["Starts"], evidence={"KeyPresent": "no", "Gas": "Full", "Ignition": "Works"})
    print(q3)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

