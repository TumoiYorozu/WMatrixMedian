from pysat.solvers import *
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc

Solver = MinisatGH


def find_new_counterexample_given_program(program, num_inputs, known_sorted_pairs, desired_sorted_pairs):
    # program is a list of ordered pairs representing the connections
    # returns an example input where the program doesn't work

    # v_ij: The value of wire j just before instruction i
    def v(i, j):
        return i * num_inputs + j + 1

    cnf = CNF()

    num_instructions = len(program)
    
    # A pool of auxiliary variables
    vpool = IDPool(start_from = v(len(program) + 1, 0))    

    # Encode the action of the sorting network
    for (i, (a, b)) in enumerate(program):
        # v(i + 1, a) == v(i, a) && v(i, b)

        # Let's convert to CNF 
        # X == Y && Z
        # (X => (Y && Z)) && ((Y && Z) => X)
        # (X => Y) && (X => Z) && ((Y && Z) => X)
        # (!X || Y) && (!X || Z) && (!Y || !Z || X)        

        X = v(i + 1, a)
        Y = v(i, a)
        Z = v(i, b)
        cnf.append([-X, Y])
        cnf.append([-X, Z])
        cnf.append([-Y, -Z, X])        

        # v(i + 1, b) == v(i, a) || v(i, b)
        
        # W == Y || Z
        # (W => (Y || Z)) && ((Y || Z) => W)
        # (W => (Y || Z)) && (Y => W) && (Z => W)
        W = v(i + 1, b)
        cnf.append([-W, Y, Z])
        cnf.append([-Y, W])
        cnf.append([-Z, W])

        # The wires that aren't a or b are unchanged
        for j in range(num_inputs):
            if a == j or b == j: continue

            X = v(i, j)
            Y = v(i+1, j)
            cnf.append([-X, Y])
            cnf.append([-Y, X])            
        
    # Assert the negation of the properties we want on the output (we're finding a counterexample)
    output_correct = CNF()
    for (i, j) in desired_sorted_pairs:
        output_correct.append([-v(num_instructions, i), v(num_instructions, j)])
    cnf.extend(output_correct.negate(vpool.id('negation_clauses')))
        
    # Anything extra we know about the input
    for (i, j) in known_sorted_pairs:
        cnf.append([-v(0, i), v(0, j)])
                    
    #print("  CNF query with", len(cnf.clauses), "clauses")
    l = Solver(bootstrap_with = cnf.clauses)
    l.solve()            
    model = l.get_model()    

    #print(model)
    
    if model is None: return None
    
    counterexample = []
    for i in range(num_inputs):
        if v(0, i) in model:
            counterexample.append(1)
        else:
            counterexample.append(0)
    
    return counterexample


# Adapted from McGuire's 5x5 shader at https://casual-effects.com/research/McGuire2008Median/median5.pix

program = []
def t2(a, b):
    program.append((a, b))

def t24(a, b, c, d, e, f, g, h):
    t2(a, b)
    t2(c, d)
    t2(e, f)
    t2(g, h)

def t25(a, b, c, d, e, f, g, h, i, j):
    t24(a, b, c, d, e, f, g, h)
    t2(i, j)

    
t25(0, 1,                   3, 4,           2, 4,           2, 3,           6, 7)
t25(5, 7,                   5, 6,           9, 7,           1, 7,           1, 4)
t25(12, 13,         11, 13,         11, 12,         15, 16,         14, 16)
t25(14, 15,         18, 19,         17, 19,         17, 18,         21, 22)
t25(20, 22,         20, 21,         23, 24,         2, 5,           3, 6)
t25(0, 6,                   0, 3,           4, 7,           1, 7,           1, 4)
t25(11, 14,         8, 14,          8, 11,          12, 15,         9, 15)
t25(9, 12,          13, 16,         10, 16,         10, 13,         20, 23)
t25(17, 23,         17, 20,         21, 24,         18, 24,         18, 21)
t25(19, 22,         8, 17,          9, 18,          0, 18,          0, 9)
t25(10, 19,         1, 19,          1, 10,          11, 20,         2, 20)
t25(2, 11,          12, 21,         3, 21,          3, 12,          13, 22)
t25(4, 22,          4, 13,          14, 23,         5, 23,          5, 14)
t25(15, 24,         6, 24,          6, 15,          7, 16,          7, 19)
t25(3, 11,          5, 17,          11, 17,         9, 17,          4, 10)
t25(6, 12,          7, 14,          4, 6,           4, 7,           12, 14)
t25(10, 14,         6, 7,           10, 12,         6, 10,          6, 17)
t25(12, 17,         7, 17,          7, 10,          12, 18,         7, 12)
t24(10, 18,         12, 20,         10, 20,         10, 12)

known = []
desired = [(i, 12) for i in range(12)] + [(12, i) for i in range(13, 25)]

counterexample = find_new_counterexample_given_program(program, 25, known, desired)

print("Counterexample:")
print(counterexample)

# Double-check this indeed doesn't work

state = counterexample[:]
for (p0, p1) in program:
    (a, b) = (state[p0], state[p1])
    tmp = a
    a = min(a, b)
    b = max(tmp, b)
    (state[p0], state[p1]) = (a, b)

print("Output:")
print(state)

counterexample.sort()

try:
    assert(state[12] == counterexample[12])
except:
    print("McGuire has a bug")
