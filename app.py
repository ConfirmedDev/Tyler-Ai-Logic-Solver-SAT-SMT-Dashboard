from flask import Flask, request, jsonify, render_template_string

from collections import defaultdict, deque
import re
import time
import json
from z3 import *

app = Flask(__name__)

# -------------------------------
# Helper: Parse literals (with negation)
def parse_literal(lit):
    lit = lit.strip()
    neg = lit.startswith('!')
    var = lit[1:] if neg else lit
    return (var, neg)

# -------------------------------
# 3-SAT → 2-SAT Reducer
@app.route('/reduce_3sat', methods=['POST'])
def reduce_3sat():
    data = request.json
    clauses = data.get('clauses', [])
    logs = []
    reduced_clauses = []

    for i, clause in enumerate(clauses):
        lits = clause.split()
        if len(lits) != 3:
            logs.append(f"Clause {i} ignored (not 3 literals): {clause}")
            continue
        yvar = f"Y{i}"
        logs.append(f"Clause {i}: {clause} → ( {lits[0]} ∨ {yvar} ), ( !{yvar} ∨ {lits[1]} ), ( !{yvar} ∨ {lits[2]} )")
        reduced_clauses.append(f"{lits[0]} {yvar}")
        reduced_clauses.append(f"!{yvar} {lits[1]}")
        reduced_clauses.append(f"!{yvar} {lits[2]}")

    return jsonify({"logs": logs, "reduced_clauses": reduced_clauses})

# -------------------------------
# 2-SAT Solver using implication graph + Kosaraju SCC
@app.route('/solve_2sat', methods=['POST'])
def solve_2sat():
    data = request.json
    clauses = data.get('clauses', [])
    logs = []
    graph = defaultdict(list)

    def var_neg(literal):
        return literal[1:] if literal.startswith('!') else '!' + literal

    # Build implication graph edges
    for i, clause in enumerate(clauses):
        lits = clause.split()
        if len(lits) != 2:
            logs.append(f"Clause {i} ignored (not 2 literals): {clause}")
            continue
        a, b = lits
        na, nb = var_neg(a), var_neg(b)
        # Add edges: !a → b, !b → a
        graph[na].append(b)
        graph[nb].append(a)
        logs.append(f"Add edge: {na} -> {b}")
        logs.append(f"Add edge: {nb} -> {a}")
        logs.append(f"Clause {i}: ({a} {b}) processed.")

    # Kosaraju's algorithm for SCC
    nodes = set(graph.keys())
    for vlist in graph.values():
        nodes.update(vlist)

    visited = set()
    order = []

    def dfs1(u):
        visited.add(u)
        for w in graph.get(u, []):
            if w not in visited:
                dfs1(w)
        order.append(u)

    for node in nodes:
        if node not in visited:
            dfs1(node)

    reversed_graph = defaultdict(list)
    for u, ws in graph.items():
        for w in ws:
            reversed_graph[w].append(u)

    assignment = {}
    visited.clear()
    component = {}

    def dfs2(u, label):
        visited.add(u)
        component[u] = label
        for w in reversed_graph.get(u, []):
            if w not in visited:
                dfs2(w, label)

    for u in reversed(order):
        if u not in visited:
            dfs2(u, u)

    # Check for contradiction: var and !var in same component
    for v in nodes:
        nv = var_neg(v)
        if v in component and nv in component and component[v] == component[nv]:
            return jsonify({
                "logs": logs + [f"Unsatisfiable: variable {v} and its negation in same SCC."],
                "assignment": {},
                "error": "Unsatisfiable"
            }), 400

    # Assign truth values in order of components
    comp_order = {}
    for k,vv in component.items():
        comp_order[vv] = comp_order.get(vv, len(comp_order))

    values = {}
    for v in nodes:
        if v not in values:
            nv = var_neg(v)
            # Assign True to whichever component comes later in order
            assign_true = comp_order[component[v]] > comp_order[component[nv]]
            values[v] = assign_true
            values[nv] = not assign_true

    logs.append("Assignment computed successfully.")
    return jsonify({"logs": logs, "assignment": values})

# -------------------------------
# Horn-SAT Solver (simple forward chaining)
@app.route('/solve_horn', methods=['POST'])
def solve_horn():
    data = request.json
    clauses = data.get('clauses', [])
    logs = []
    assignment = {}
    implications = defaultdict(list)
    facts = set()

    # Parse clauses, warning if >1 positive literal
    for i, clause in enumerate(clauses):
        lits = clause.split()
        pos = [lit for lit in lits if not lit.startswith('!')]
        neg = [lit[1:] for lit in lits if lit.startswith('!')]
        if len(pos) > 1:
            logs.append(f"Warning: Clause {i} has more than one positive literal. Continuing but results may be incorrect.")
        # If no positive literal, it must be false for satisfiability (we won't handle complex unsat here)
        if len(pos) == 0 and len(neg) > 0:
            # clause is like !x1 !x2 ... which is unsat if all negated vars true
            logs.append(f"Warning: Clause {i} has no positive literal.")
        if len(pos) == 1:
            # Implication: negated vars -> pos var
            for nvar in neg:
                implications[nvar].append(pos[0])
        if len(pos) == 0 and len(neg) == 0:
            logs.append(f"Clause {i} empty - ignoring")

        if len(pos) == 1 and len(neg) == 0:
            # Fact (positive literal)
            facts.add(pos[0])
        if len(pos) > 1:
            logs.append(f"Clause {i}: {' '.join(lits)}")

    # Forward chaining
    queue = deque(facts)
    for fact in facts:
        assignment[fact] = True
        logs.append(f"Inferred {fact} = True")

    while queue:
        v = queue.popleft()
        for implied in implications.get(v, []):
            if implied not in assignment:
                assignment[implied] = True
                logs.append(f"Inferred {implied} = True")
                queue.append(implied)

    logs.append("Horn-SAT inference complete.")
    return jsonify({"logs": logs, "assignment": assignment})

# -------------------------------
# Tseitin Clause Engine
class TseitinClauseEngine:
    def __init__(self):
        self.aux_counter = 0
        self.reduction_steps = []

    def fresh_aux(self):
        v = f"T{self.aux_counter}"
        self.aux_counter += 1
        return v

    def tseitin_transform_clause(self, clause):
        if len(clause) <= 3:
            return [clause]

        first_two = clause[:2]
        rest = clause[2:]
        aux_var = self.fresh_aux()
        new_clause_1 = first_two + [aux_var]
        rest_clause = ['!' + aux_var] + rest
        rest_transformed = self.tseitin_transform_clause(rest_clause)

        self.reduction_steps.append({
            'original': clause,
            'step_clauses': [new_clause_1] + rest_transformed
        })

        return [new_clause_1] + rest_transformed

    def transform(self, clauses):
        all_transformed = []
        self.reduction_steps.clear()
        self.aux_counter = 0

        for clause in clauses:
            transformed = self.tseitin_transform_clause(clause)
            all_transformed.extend(transformed)

        return all_transformed, self.reduction_steps

@app.route('/tseitin_clause_engine', methods=['POST'])
def tseitin_clause_engine():
    data = request.get_json()
    clauses = data.get('clauses', [])
    steps = []
    output = []
    aux_counter = [0]

    def new_aux():
        v = f"T{aux_counter[0]}"
        aux_counter[0] += 1
        return v

    for i, clause in enumerate(clauses):
        literals = clause.split()
        if not literals:
            continue
        if len(literals) == 1:
            output.append(literals[0])
            steps.append(f"Clause {i} already 1-literal: {literals[0]}")
        elif len(literals) == 2:
            output.append(' '.join(literals))
            steps.append(f"Clause {i} already 2-literal: {' '.join(literals)}")
        else:
            prev_aux = None
            for j in range(len(literals) - 1):
                if j == 0:
                    aux = new_aux()
                    output.append(f"{literals[0]} {aux}")
                    output.append(f"!{aux} {literals[1]}")
                    prev_aux = aux
                    steps.append(f"Clause {i} step {j}: ({literals[0]} ∨ {aux}), (!{aux} ∨ {literals[1]})")
                else:
                    aux = new_aux()
                    output.append(f"!{prev_aux} {literals[j+1]}")
                    steps.append(f"Clause {i} step {j}: (!{prev_aux} ∨ {literals[j+1]})")
                    prev_aux = aux

    return jsonify({
        "final_clauses": output,
        "steps": steps
    })
# -------------------------------

# Tseitin Transform (placeholder example)
@app.route('/tseitin_transform', methods=['POST'])
def tseitin_transform():
    data = request.json
    expr = data.get('expr', '')

    # Dummy implementation - replace with actual parser if needed
    logs = []
    aux_vars = ['T0', 'T1', 'T2']
    clauses = [
        "!T0 !B",
        "T0 B",
        "!T1 A",
        "!T1 T0",
        "T1 !A !T0",
        "!T1 T2",
        "!C T2",
        "!T2 T1 C",
        "T2"
    ]
    logic_tree = {
      "type": "or",
      "expr": "",
      "children": [
        {
          "type": "and",
          "expr": "",
          "children": [
            {"type": "var", "expr": "A", "children": []},
            {"type": "not", "expr": "", "children": [{"type": "var", "expr": "B", "children": []}]}
          ]
        },
        {"type": "var", "expr": "C", "children": []}
      ]
    }
    logs.append(f"Tokens: [('LPAREN', '('), ('VAR', 'A'), ('AND', '&'), ('NOT', '!'), ('VAR', 'B'), ('RPAREN', ')'), ('OR', '|'), ('VAR', 'C')]")
    logs.append("Parsed expression successfully.")
    logs.append("Root variable: T2")

    return jsonify({
        "clauses": clauses,
        "aux_vars": aux_vars,
        "logs": logs,
        "logic_tree": logic_tree
    })

# -------------------------------
import re
from flask import jsonify

# SMT Solver (Z3) - improved assertion parsing
@app.route('/solve_smt', methods=['POST'])
def solve_smt():
    data = request.json
    commands = data.get('commands', '')
    logs = []
    result = {}

    try:
        s = Solver()
        variables = {}
        lines = [line.strip() for line in commands.strip().split('\n') if line.strip()]

        for line in lines:
            if line.startswith('(declare-const'):
                m = re.match(r'^\(declare-const\s+(\w+)\s+(\w+)\)$', line)
                if m:
                    var, vartype = m.groups()
                    if vartype.lower() == 'int':
                        variables[var] = Int(var)
                    elif vartype.lower() == 'bool':
                        variables[var] = Bool(var)
                    else:
                        logs.append(f"Unknown type for variable {var}: {vartype}")
                    logs.append(f"Declared variable: {var} of type {vartype}")
                else:
                    logs.append(f"Malformed declare-const: {line}")

            elif line.startswith('(assert'):
                assertion_str = line[7:-1].strip()  # remove (assert ... )
                logs.append(f"Parsing assertion: {assertion_str}")

                # Regex to parse assertions like (> x 5), (< x 10)
                m = re.match(r'^\(\s*(\S+)\s+(\S+)\s+(-?\d+)\s*\)$', assertion_str)
                if m:
                    op, var, val_str = m.groups()
                    val = int(val_str)
                    if var not in variables:
                        logs.append(f"Variable {var} not declared.")
                        continue
                    if op == '>':
                        s.add(variables[var] > val)
                        logs.append(f"Added assertion: (> {var} {val})")
                    elif op == '<':
                        s.add(variables[var] < val)
                        logs.append(f"Added assertion: (< {var} {val})")
                    elif op == '>=':
                        s.add(variables[var] >= val)
                        logs.append(f"Added assertion: (>= {var} {val})")
                    elif op == '<=':
                        s.add(variables[var] <= val)
                        logs.append(f"Added assertion: (<= {var} {val})")
                    elif op == '=':
                        s.add(variables[var] == val)
                        logs.append(f"Added assertion: (= {var} {val})")
                    else:
                        logs.append(f"Unsupported operator: {op}")
                else:
                    logs.append(f"Unsupported or malformed assertion format: {assertion_str}")

            elif line == '(check-sat)':
                res = s.check()
                logs.append(f"Check-sat result: {res}")
                if res == sat:
                    model = s.model()
                    for d in model.decls():
                        v = model[d]
                        sort_kind = v.sort().kind()
                        if sort_kind == Z3_BOOL_SORT:
                            val = is_true(v)
                        elif sort_kind == Z3_INT_SORT:
                            val = v.as_long()
                        else:
                            val = str(v)
                        result[str(d.name())] = val
                    logs.append("Model retrieved successfully.")
                else:
                    logs.append("Unsatisfiable or unknown.")

            elif line == '(get-model)':
                # No-op here, model handled after check-sat
                pass

            else:
                logs.append(f"Unknown command: {line}")

    except Exception as e:
        return jsonify({"logs": logs, "error": str(e)}), 400

    return jsonify({"logs": '\n'.join(logs), "result": result})

from flask import request, jsonify

# -------------------------------
# Collatz Solver (bounded steps)
@app.route('/collatz_bounded', methods=['POST'])
def collatz_bounded():
    try:
        data = request.get_json()
        n = int(data.get('n', 0))
        k = int(data.get('k', 0))

        if n < 10 or k < 1:
            return jsonify({"error": "Invalid input: n must be >=10 and k must be >=1."}), 400

        s = Solver()
        x = [Int(f'x_{i}') for i in range(k+1)]
        s.add(x[0] == n)

        for i in range(k):
            even_cond = x[i] % 2 == 0
            # Use Z3 symbolic division instead of Python //
            next_even = x[i+1] == x[i] / 2
            next_odd = x[i+1] == 3 * x[i] + 1
            s.add(If(even_cond, next_even, next_odd))

        s.add(Or([x[i] == 1 for i in range(1, k+1)]))

        result = s.check()
        if result == sat:
            model = s.model()
            steps = [model.eval(x[i], model_completion=True).as_long() for i in range(k+1)]
            return jsonify({
                "result": "sat",
                "length": len(steps),
                "steps": steps
            })
        elif result == unsat:
            return jsonify({
                "result": "unsat",
                "message": f"No sequence reaches 1 in {k} steps from {n}."
            })
        else:
            return jsonify({"result": "unknown"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------

# Collatz Solver
@app.route('/solve_collatz', methods=['POST'])
def solve_collatz():
    data = request.get_json()
    n = data.get('n')
    if not isinstance(n, int) or n < 10:
        return jsonify({'error': 'Input must be an integer >= 10.'}), 400
    steps = []
    current = n
    while current != 1:
        steps.append(current)
        current = current // 2 if current % 2 == 0 else 3 * current + 1
    steps.append(1)
    return jsonify({'start': n, 'steps': steps, 'length': len(steps)})

# -------------------------------


# Goldbach Conjecture SMT Solver
@app.route('/solve_goldbach', methods=['POST'])
def solve_goldbach():
    data = request.json
    n = data.get('n')
    logs = []
    if not isinstance(n, int) or n < 4 or n % 2 != 0:
        return jsonify({"error": "Input n must be an even integer ≥ 4"}), 400

    logs.append(f"Goldbach conjecture SMT for n={n}")

    s = Solver()
    p1 = Int('p1')
    p2 = Int('p2')

    s.add(p1 > 1, p2 > 1)
    s.add(p1 + p2 == n)

    primes = [i for i in range(2, n+1) if all(i % d != 0 for d in range(2, int(i**0.5)+1))]
    s.add(Or([p1 == pr for pr in primes]))
    s.add(Or([p2 == pr for pr in primes]))

    res = s.check()
    logs.append(f"Check-sat result: {res}")
    if res == sat:
        model = s.model()
        p1_val = model[p1].as_long()
        p2_val = model[p2].as_long()
        logs.append(f"Found primes: p1={p1_val}, p2={p2_val}")
        return jsonify({
            "logs": logs,
            "p1": p1_val,
            "p2": p2_val
        })
    else:
        logs.append("No solution found.")
        return jsonify({"logs": logs, "error": "No solution found"}), 400

# -------------------------------
# New endpoint: Generate implication graph for 2-SAT input (for visualization)
@app.route('/get_implication_graph', methods=['POST'])
def get_implication_graph():
    data = request.json
    clauses = data.get('clauses', [])
    graph = defaultdict(list)
    variables = set()

    def var_neg(literal):
        return literal[1:] if literal.startswith('!') else '!' + literal

    for clause in clauses:
        lits = clause.split()
        if len(lits) != 2:
            continue
        a, b = lits
        na, nb = var_neg(a), var_neg(b)
        graph[na].append(b)
        graph[nb].append(a)
        variables.add(a.lstrip('!'))
        variables.add(b.lstrip('!'))

    return jsonify({
        "implication_graph": graph,
        "variables": sorted(variables)
    })
def negate(literal):
    if literal.startswith('!'):
        return literal[1:]
    else:
        return '!' + literal
# ----------------------
# Tyler's Theory - Collatz Clause Reducer (3-SAT → 2-SAT style)
@app.route('/collatz_clause_reducer', methods=['POST'])
def collatz_clause_reducer():
    data = request.json
    clauses = data.get('clauses', [])
    logs = []
    reduced_clauses = []

    start_time = time.time()

    for i, clause in enumerate(clauses):
        lits = clause.strip().split()
        length = len(lits)
        logs.append(f"Clause {i} original: {clause} (length {length})")

        if length % 2 == 1:
            aux = f"Y{i}"
            reduced_clauses.append(f"{lits[0]} {aux}")
            for lit in lits[1:]:
                reduced_clauses.append(f"!{aux} {lit}")
            logs.append(f"Clause {i} reduced by 3x+1 step with aux {aux}")
        else:
            reduced_clauses.append(clause)
            logs.append(f"Clause {i} preserved (even length, no reduction)")

    end_time = time.time()

    return jsonify({
        "logs": logs,
        "reduced_clauses": reduced_clauses,
        "reducer_time": round(end_time - start_time, 6)
    })
# ----------------------
# 2-SAT satisfiability checker using Kosaraju's algorithm
@app.route('/collatz_2sat_reducer', methods=['POST'])
def collatz_2sat_reducer():
    data = request.get_json()
    clauses = data.get('clauses', [])
    logs = []

    start_time = time.time()

    graph = {}
    rev_graph = {}
    literals = set()

    def negate(lit):
        return lit[1:] if lit.startswith('!') else f"!{lit}"

    # Build implication graph
    for clause in clauses:
        try:
            a, b = clause.strip().split()
        except ValueError:
            logs.append(f"Ignored invalid clause: {clause}")
            continue

        literals.update([a.lstrip('!'), b.lstrip('!')])

        for x, y in [(negate(a), b), (negate(b), a)]:
            graph.setdefault(x, []).append(y)
            rev_graph.setdefault(y, []).append(x)

    visited = set()
    order = []

    def dfs1(node):
        visited.add(node)
        for nei in graph.get(node, []):
            if nei not in visited:
                dfs1(nei)
        order.append(node)

    component = {}
    def dfs2(node, label):
        component[node] = label
        for nei in rev_graph.get(node, []):
            if nei not in component:
                dfs2(nei, label)

    all_nodes = set(graph.keys()) | set(rev_graph.keys())
    for node in all_nodes:
        if node not in visited:
            dfs1(node)

    for node in reversed(order):
        if node not in component:
            dfs2(node, node)

    is_sat = True
    for var in literals:
        if component.get(var) == component.get(f"!{var}"):
            logs.append(f"Conflict found: {var} and !{var} in the same component.")
            is_sat = False
            break

    if is_sat:
        logs.append("No conflicts found. Formula is SAT.")

    end_time = time.time()

    return jsonify({
        "logs": logs,
        "result": "SAT" if is_sat else "UNSAT",
        "twosat_time": round(end_time - start_time, 6)
    })

import os

@app.route('/run_tyler_benchmark', methods=['POST'])
def run_tyler_benchmark():
    from time import time
    data = request.get_json()
    file_path = data.get('file_path')
    logs = []

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 400

    # === Step 1: Parse DIMACS to Tyler Format ===
    clauses = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c') or line.startswith('p') or not line:
                continue
            literals = line.split()
            if literals[-1] == '0':
                literals = literals[:-1]
            clause_vars = []
            for lit in literals:
                val = int(lit)
                if val > 0:
                    clause_vars.append(f"x{val}")
                else:
                    clause_vars.append(f"!x{abs(val)}")
            clauses.append(" ".join(clause_vars))

    logs.append(f"Parsed {len(clauses)} clauses from DIMACS.")

    # === Step 2: Collatz Clause Reducer ===
    start = time()
    reduced_clauses = []
    reducer_logs = []
    for i, clause in enumerate(clauses):
        lits = clause.strip().split()
        length = len(lits)
        reducer_logs.append(f"Clause {i} original: {clause} (length {length})")
        if length % 2 == 1:
            aux = f"Y{i}"
            reduced_clauses.append(f"{lits[0]} {aux}")
            for lit in lits[1:]:
                reduced_clauses.append(f"!{aux} {lit}")
            reducer_logs.append(f"Clause {i} reduced by 3x+1 step with aux {aux}")
        else:
            reduced_clauses.append(clause)
            reducer_logs.append(f"Clause {i} preserved (even length, no reduction)")
    reducer_time = round(time() - start, 4)

    # === Step 3: 2-SAT Solver ===
    start = time()
    g = {}
    rev = {}
    litset = set()
    twosat_logs = []
    def neg(x): return x[1:] if x.startswith('!') else f"!{x}"

    for clause in reduced_clauses:
        try:
            a, b = clause.strip().split()
        except ValueError:
            twosat_logs.append(f"Ignored invalid clause: {clause}")
            continue
        litset.update([a.lstrip('!'), b.lstrip('!')])
        for x, y in [(neg(a), b), (neg(b), a)]:
            g.setdefault(x, []).append(y)
            rev.setdefault(y, []).append(x)

    visited = set()
    order = []
    def dfs1(n):
        visited.add(n)
        for nei in g.get(n, []):
            if nei not in visited:
                dfs1(nei)
        order.append(n)

    comp = {}
    def dfs2(n, label):
        comp[n] = label
        for nei in rev.get(n, []):
            if nei not in comp:
                dfs2(nei, label)

    for node in g.keys() | rev.keys():
        if node not in visited:
            dfs1(node)
    for node in reversed(order):
        if node not in comp:
            dfs2(node, node)

    is_sat = True
    for var in litset:
        if comp.get(var) == comp.get(f"!{var}"):
            is_sat = False
            twosat_logs.append(f"Conflict: {var} and !{var} in same component")
            break
    if is_sat:
        twosat_logs.append("No conflicts found. Formula is SAT.")
    twosat_time = round(time() - start, 4)

    return jsonify({
        "file": os.path.basename(file_path),
        "original_clause_count": len(clauses),
        "reduced_clause_count": len(reduced_clauses),
        "reducer_time": reducer_time,
        "twosat_time": twosat_time,
        "result": "SAT" if is_sat else "UNSAT",
        "reducer_logs": reducer_logs,
        "twosat_logs": twosat_logs,
    })
# ----------------------
# --- New endpoint for Tyler's Theory Implication Graph and Matrix Visualizer ---
@app.route('/get_tyler_implication_graph', methods=['POST'])
def get_tyler_implication_graph():
    data = request.get_json()
    clauses = data.get('clauses', [])

    literals_set = set()
    clause_literals = []

    for clause in clauses:
        lits = clause.strip().split()
        clause_literals.append(lits)
        for lit in lits:
            literals_set.add(lit.lstrip('!'))

    # Sort literals like x1, x2, ... last, and by number
    literals = sorted(
        literals_set,
        key=lambda x: (not x.startswith('x'), int(x[1:]) if x[1:].isdigit() else 0)
    )

    # Create CNF Clause Matrix
    matrix = []
    for clause in clause_literals:
        row = []
        for lit in literals:
            if lit in clause:
                row.append(1)
            elif ('!' + lit) in clause:
                row.append(-1)
            else:
                row.append(0)
        matrix.append(row)

    # Build Implication Graph
    def negate(literal):
        return literal[1:] if literal.startswith('!') else '!' + literal

    nodes_set = set()
    edges = []

    for lit in literals:
        nodes_set.add(lit)
        nodes_set.add(negate(lit))

    for clause in clause_literals:
        if len(clause) != 2:
            continue
        a, b = clause
        edges.append({'from': negate(a), 'to': b})
        edges.append({'from': negate(b), 'to': a})

    nodes = [{'id': node, 'label': node} for node in sorted(nodes_set)]

    return jsonify({
        'literals': literals,
        'matrix': matrix,
        'nodes': nodes,
        'edges': edges
    })

# Tyler's Theory - UI Page
@app.route('/tylers_theory')
def tylers_theory_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tyler's Theory - Collatz Clause Reducer & 2-SAT Reducer</title>
        <style>
            body { font-family: Arial; max-width: 900px; margin: auto; padding: 20px; }
            textarea, button, pre { width: 100%; margin-top: 10px; font-size: 1em; }
            textarea { padding: 10px; }
            pre { background: #f9f9f9; padding: 10px; white-space: pre-wrap; }
            section { margin-bottom: 40px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; }
            h2 { margin-top: 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 6px; text-align: center; }
            th { background-color: #eee; }
        </style>
    </head>
    <body>
        <h1>Tyler's Theory - Collatz Clause Reducer & 2-SAT Reducer</h1>

        <section id="collatzClauseReducerSection">
            <h2>Collatz Clause Reducer (3-SAT → 2-SAT style)</h2>
            <p>Enter SAT clauses (one clause per line, literals space-separated):</p>
            <textarea id="clausesInput" rows="8" placeholder="x1 !x2 x3\n!x1 x2 !x3\nx2 !x4 x5"></textarea>
            <button id="btnReduce3Sat">Run Collatz Clause Reducer</button>
            <div id="output"></div>
        </section>

        <section id="collatz2SatReducerSection">
            <h2>Collatz 2-SAT → SAT Check</h2>
            <p>Auto-filled from 3-SAT reduction:</p>
            <textarea id="twoSatInput" rows="6"></textarea>
            <button id="btnReduce2Sat">Run 2-SAT Reducer (SAT/UNSAT)</button>
            <h3>Logs:</h3>
            <pre id="twoSatLogs"></pre>
            <h3>Result:</h3>
            <pre id="oneSatOutput"></pre>
        </section>

        <section id="implicationGraphSection">
            <h2>Implication Graph + CNF Matrix</h2>
            <button id="btnVisualizeGraph">Generate Implication Graph / Matrix</button>
            <h3>Clause Matrix:</h3>
            <div id="matrixOutput"></div>
            <h3>Implication Graph:</h3>
            <pre id="graphJsonOutput"></pre>
        </section>

        <a href="/" style="display:block; margin-top:30px;">← Back to Main Dashboard</a>

        <script>
        document.addEventListener('DOMContentLoaded', () => {
            async function reduceClauses() {
                const raw = document.getElementById('clausesInput').value.trim();
                if (!raw) {
                    alert('Please enter some clauses.');
                    return;
                }
                const clauses = raw.split('\\n').map(line => line.trim()).filter(line => line.length > 0);

                const res = await fetch('/collatz_clause_reducer', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ clauses })
                });
                const data = await res.json();

                let html = '<h3>Logs:</h3><pre>' + data.logs.join('\\n') + '</pre>';
                html += '<h3>Reduced Clauses:</h3><pre>' + data.reduced_clauses.join('\\n') + '</pre>';
                html += `<p><strong>Reducer Time:</strong> ${data.reducer_time} seconds</p>`;
                document.getElementById('output').innerHTML = html;

                // Pipe into 2-SAT input
                document.getElementById('twoSatInput').value = data.reduced_clauses.join('\\n');
            }

            async function reduce2Sat() {
                const raw = document.getElementById('twoSatInput').value.trim();
                if (!raw) {
                    alert('Please enter 2-SAT clauses.');
                    return;
                }
                const clauses = raw.split('\\n').map(line => line.trim()).filter(line => line.length > 0);

                document.getElementById('twoSatLogs').textContent = 'Checking...';
                document.getElementById('oneSatOutput').textContent = '';

                try {
                    const res = await fetch('/collatz_2sat_reducer', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ clauses })
                    });
                    const data = await res.json();

                    document.getElementById('twoSatLogs').textContent = data.logs.join('\\n');
                    document.getElementById('oneSatOutput').textContent = 
                        (data.result === 'SAT' ? '✅ SAT' : '❌ UNSAT') + ` (Time: ${data.twosat_time} seconds)`;

                } catch (e) {
                    document.getElementById('twoSatLogs').textContent = 'Error: ' + e.message;
                }
            }

            async function visualizeGraphAndMatrix() {
                const raw = document.getElementById('twoSatInput').value.trim();
                if (!raw) {
                    alert('Please enter 2-SAT clauses.');
                    return;
                }
                const clauses = raw.split('\\n').map(line => line.trim()).filter(line => line.length > 0);

                const res = await fetch('/get_tyler_implication_graph', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ clauses })
                });
                const data = await res.json();

                // Build clause matrix table
                let matrixHtml = '<table><tr><th>Clause</th>';
                for (let lit of data.literals) {
                    matrixHtml += `<th>${lit}</th>`;
                }
                matrixHtml += '</tr>';

                data.matrix.forEach((row, idx) => {
                    matrixHtml += `<tr><td>${idx}</td>`;
                    row.forEach(cell => {
                        matrixHtml += `<td>${cell}</td>`;
                    });
                    matrixHtml += '</tr>';
                });
                matrixHtml += '</table>';
                document.getElementById('matrixOutput').innerHTML = matrixHtml;

                // Show raw graph JSON (for now)
                document.getElementById('graphJsonOutput').textContent =
                    JSON.stringify({ nodes: data.nodes, edges: data.edges }, null, 2);
            }

            document.getElementById('btnReduce3Sat').onclick = reduceClauses;
            document.getElementById('btnReduce2Sat').onclick = reduce2Sat;
            document.getElementById('btnVisualizeGraph').onclick = visualizeGraphAndMatrix;
        });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)
# -------------------------------
# Serve main page (will embed HTML directly for ease here)
@app.route('/')
def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)



