import requests
import os

# Auto switch between local and Render depending on environment
BASE_URL = os.environ.get("https://tyler-ai-logic-solver-sat-smt-dashboard.onrender.com", "http://127.0.0.1:5000")

print("Using backend:", BASE_URL)

def test_collatz_clause_reducer():
    print("Testing /collatz_clause_reducer...")

    sample_clauses = [
        "x1 !x2 x3",
        "!x1 x2 !x3",
        "x2 !x4 x5",
        "x1 !x3"  # 2-literal clause should be preserved
    ]

    payload = {"clauses": sample_clauses}
    response = requests.post(f"{BASE_URL}/collatz_clause_reducer", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("\nReducer Logs:")
        print("\n".join(data["logs"]))
        print("\nReduced Clauses:")
        print("\n".join(data["reduced_clauses"]))
        return data["reduced_clauses"]
    else:
        print("Error:", response.text)
        return []

def test_collatz_2sat_reducer(reduced_clauses):
    print("\nTesting /collatz_2sat_reducer...")

    payload = {"clauses": reduced_clauses}
    response = requests.post(f"{BASE_URL}/collatz_2sat_reducer", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("\n2-SAT Logs:")
        print("\n".join(data["logs"]))
        print("\nResult:", data["result"])
    else:
        print("Error:", response.text)

def test_run_tyler_benchmark(file_path):
    print("\nTesting /run_tyler_benchmark...")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    payload = {"file_path": file_path}
    response = requests.post(f"{BASE_URL}/run_tyler_benchmark", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"\nFile: {data.get('file')}")
        print(f"Original clause count: {data.get('original_clause_count')}")
        print(f"Reduced clause count: {data.get('reduced_clause_count')}")
        print("\nReducer Logs:")
        print("\n".join(data.get('reducer_logs', [])))
        print(f"\nReducer time: {data.get('reducer_time')} seconds")
        print(f"\nSolver result: {data.get('solver_result')}")
        print("\n2-SAT Logs:")
        print("\n".join(data.get('twosat_logs', [])))
        print(f"\n2-SAT solver time: {data.get('twosat_time')} seconds")
    else:
        print("Error:", response.text)

def main():
    # Test 1: Collatz Clause Reducer
    reduced = test_collatz_clause_reducer()

    # Test 2: Collatz 2-SAT Reducer using output of Test 1
    if reduced:
        test_collatz_2sat_reducer(reduced)

    # Test 3: Run benchmark on DIMACS file (adjust file path here)
    dimacs_file_path = r"C:\aim-50-1_6-no-4.cnf"  # Change to your actual file path
    test_run_tyler_benchmark(dimacs_file_path)

# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)



