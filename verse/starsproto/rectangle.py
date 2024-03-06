#a place for old rectangle code so we could still call it as needed.
from verse.analysis import Verifier
import copy, itertools, functools, pprint


def handle_continuous_resets:
    comb_list = Verifier._get_combinations(symbols, cont_var_dict)

    lb = float("inf")
    ub = -float("inf")

    for comb in comb_list:
        val_dict = {}
        tmp = copy.deepcopy(expr)
        for symbol_idx, symbol in enumerate(symbols):
            tmp = tmp.replace(symbol, str(comb[symbol_idx]))
        #apply reset on cont vars
        res = eval(tmp, {}, val_dict)
        lb = min(lb, res)
        ub = max(ub, res)

    rect[0][lhs_idx] = lb
    rect[1][lhs_idx] = ub

def add_constraints(self, cur_solver, state_vec):
    for symbol in symbols:
        start, end = continuous_variable_dict[symbols[symbol]]
        cur_solver.add(self.varDict[symbol] >= start, self.varDict[symbol] <= end)

def continuous_reset(reset_variable, expr, agent, ego_type,cont_var_dict, rect):
    lhs = reset_variable
    rhs = expr
    found = False
    for lhs_idx, cts_variable in enumerate(
        agent.decision_logic.state_defs[ego_type].cont
    ):
        if cts_variable == lhs:
            found = True
            break
    if not found:
        raise ValueError(f"Reset continuous variable {cts_variable} not found")
    # substituting low variables

    symbols = []
    for var in cont_var_dict:
        if var in expr:
            symbols.append(var)

    # TODO: Implement this function
    # The input to this function is a list of used symbols and the cont_var_dict
    # The ouput of this function is a list of tuple of values for each variable in the symbols list
    # The function will explor all possible combinations of low bound and upper bound for the variables in the symbols list
    comb_list = Verifier._get_combinations(symbols, cont_var_dict)

    lb = float("inf")
    ub = -float("inf")

    for comb in comb_list:
        val_dict = {}
        tmp = copy.deepcopy(expr)
        for symbol_idx, symbol in enumerate(symbols):
            tmp = tmp.replace(symbol, str(comb[symbol_idx]))
        res = eval(tmp, {}, val_dict)
        lb = min(lb, res)
        ub = max(ub, res)

    rect[0][lhs_idx] = lb
    rect[1][lhs_idx] = ub