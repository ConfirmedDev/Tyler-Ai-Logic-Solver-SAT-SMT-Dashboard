import requests

# Base URL of your running API (adjust if needed)
BASE_URL = "http://localhost:5000"

# Example batch inputs for different NP problems
batch_inputs = {
    "3sat": [
        "A !B C\n!A B !C\n!A !B C",
        "X !Y Z\n!X Y !Z\n!X !Y !Z",
    ],
    "clique": [
        {
            "edges": [["A", "B"], ["B", "C"], ["A", "C"]],
            "k": 3
        },
        {
            "edges": [["A", "B"], ["B", "C"], ["C", "D"]],
            "k": 2
        }
    ],
    "vertexcover": [
        {
            "edges": [["A", "B"], ["B", "C"], ["A", "C"]],
            "k": 2
        },
        {
            "edges": [["A", "B"], ["B", "C"], ["C", "D"]],
            "k": 3
        }
    ],
    "subsetsum": [
        {"numbers": "1,3,9,12", "target": 15},
        {"numbers": "2,4,6,8", "target": 7},
    ]
}

def test_3sat():
    print("Testing 3-SAT Instances:")
    for i, clauses in enumerate(batch_inputs["3sat"]):
        data = {"clauses": clauses}
        response = requests.post(f"{BASE_URL}/3sat", json=data)
        result = response.json()
        print(f"Instance {i} result: {result}")

def test_clique():
    print("\nTesting Clique Instances:")
    for i, instance in enumerate(batch_inputs["clique"]):
        data = {
            "edges": instance["edges"],
            "k": instance["k"]
        }
        response = requests.post(f"{BASE_URL}/clique", json=data)
        result = response.json()
        print(f"Instance {i} result: {result}")

def test_vertexcover():
    print("\nTesting Vertex Cover Instances:")
    for i, instance in enumerate(batch_inputs["vertexcover"]):
        data = {
            "edges": instance["edges"],
            "k": instance["k"]
        }
        response = requests.post(f"{BASE_URL}/vertexcover", json=data)
        result = response.json()
        print(f"Instance {i} result: {result}")

def test_subsetsum():
    print("\nTesting Subset Sum Instances:")
    for i, instance in enumerate(batch_inputs["subsetsum"]):
        data = {
            "numbers": instance["numbers"],
            "target": instance["target"]
        }
        response = requests.post(f"{BASE_URL}/subsetsum", json=data)
        result = response.json()
        print(f"Instance {i} result: {result}")

def run_all_tests():
    test_3sat()
    test_clique()
    test_vertexcover()
    test_subsetsum()

if __name__ == "__main__":
    run_all_tests()

