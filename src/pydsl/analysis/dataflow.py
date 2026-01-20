import ast
from ast import AST, FunctionDef, NodeVisitor
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Generic, TypeVar
from copy import copy

N = TypeVar("N")
V = TypeVar("V")


def union(sets: set[set[V]]) -> set[V]:
    return reduce(lambda s1, s2: s1.union(s2), sets[1:], sets[0])


def intersection(sets: set[set[V]]) -> set[V]:
    return reduce(lambda s1, s2: s1.intersection(s2), sets[1:], sets[0])


class DataFlow(Generic[N]):
    """
    A directed graph indicating the flow of a program.

    N is the type held by each node.
    """

    pred: dict[N, set[N]] = {}
    succ: dict[N, set[N]] = {}

    def add_flow(self, nfrom, nto) -> None:
        """
        Write the fact that `nfrom` flows to `nto`.

        `pred` and `succ` will both be updated to reflect this flow.
        """

        # Initialize fields for nodes that's never seen before
        if nfrom not in self.pred:
            self.pred[nfrom] = set()
        if nto not in self.pred:
            self.pred[nto] = set()
        if nfrom not in self.succ:
            self.succ[nfrom] = set()
        if nto not in self.succ:
            self.succ[nto] = set()

        self.pred[nto].add(nfrom)
        self.succ[nfrom].add(nto)

    def preds(self, node: N) -> set[N]:
        return self.pred[node]

    def succs(self, node: N) -> set[N]:
        return self.succ[node]

    def loose(self) -> set[N]:
        return self.loose

    def nodes(self) -> set[N]:
        return set(self.pred.keys())


class GenKill(Generic[N, V]):
    """
    A mapping from nodes to gens and kills.

    N is the type held by each node.
    V is the type being generated or killed by each node.

    If a node is not present in the set, it is assumed that its gens and kills
    are empty sets.
    """

    gen: dict[N, set[V]]
    kill: dict[N, set[V]]

    def add_gen_kill(self, node: N, gens: set[V], kills: set[V]) -> None:
        self.gen.update({node: gens})
        self.kill.update({node: kills})

    def gens(self, node: N) -> set[V]:
        if node not in self.gen:
            return set()
        return self.gen[node]

    def kills(self, node: N) -> set[V]:
        if node not in self.kill:
            return set()
        return self.kill[node]


class DataFlowAnalysis(Generic[N, V], ABC):
    dataflow: DataFlow[N]
    genkill: GenKill[N, V]

    @abstractmethod
    def confop(self, sets: set[set[V]]) -> set[V]:
        """
        The confluence operator of the data flow analysis. Must be the
        least upper bound (i.e. join) of a set lattice.
        """
        ...

    def gens(self, node: N) -> set[V]:
        return self.genkill.gens(node)

    def kills(self, node: N) -> set[V]:
        return self.genkill.kills(node)

    def preds(self, node: N) -> set[N]:
        return self.dataflow.preds(node)

    def succs(self, node: N) -> set[N]:
        return self.dataflow.succs(node)

    def transfer(self, node: N) -> set[V]:
        """
        Compute the transfer of node N given the current partial result.

        i.e. the node's Out
        """
        return self.confop({
            self.gens(node),
            self.join(node) - self.kills(node),
        })

    def join(self, node: N) -> set[V]:
        """
        Compute the join of node N given the current partial result.

        i.e. the node's In
        """
        source = self.preds(node)
        return self.confop({self.outs[node] for node in source})

    def _analyze(self) -> dict[N, set[V]]:
        outs: dict[N, set[V]] = {}
        nodes = self.dataflow.nodes()

        self.outs = {n: set() for n in nodes}
        changed = copy(nodes)

        # Based on Kildall's method, this while loop will eventually terminate
        # as the system reaches a fixpoint, assuming self.confop is a lattice
        # join
        while len(changed) != 0:
            n = changed.pop()
            oldout = outs[n]
            newout = self.transfer(n)

            self.outs[n] = newout
            n_fixpoint = oldout == newout

            if not n_fixpoint:
                changed.update(self.succs(n))

        return outs


class ToDataFlow(NodeVisitor):
    dataflow: DataFlow

    loose_end: set[N]
    """
    Keeps track of all nodes whose successor is not yet known.

    This is useful while building up a
    DataFlow. When traversing a Python tree, you would frequently come across
    "last" node of a sequence of instructions with no clear answer as to what
    comes after, given only information localized within the scope.

    For example, when traversing a `for` statement, if you come across a
    `break`, the `for` loop would not have access to the node that succeeds
    `break`. Instead, the `for` loop should add `break` to `loose_break` which
    will be tied up later by some parent of the `for` statement.
    """
    loose_break: set[N]
    loose_continue: set[N]

    def __init__(self):
        self.dataflow = DataFlow()
        self.loose_end = set()
        self.loose_break = set()
        self.loose_continue = set()

    def _sequential_flow(self, node_sequence: list[AST]):
        """
        Given a sequence of nodes, connect them one-by-one with flow edge.

        The flow will throttle if it encounters any control flow statements
        such as `continue`, `break`, or `return`.

        The first node of the sequence will not have a pred.
        The last node of the sequence will not have a succ.
        """
        for i in range(len(node_sequence)):
            if i < len(node_sequence) - 1:
                nfrom, nto = node_sequence[i : i + 2]
            else:
                # nfrom is the last node in the sequence
                nfrom, nto = node_sequence[i], None

            match nfrom:
                case ast.Return():
                    break
                case ast.Continue():
                    self.loose_continue.add(nfrom)
                    break
                case ast.Break():
                    self.loose_break.add(nfrom)
                    break
                case ast.If() if len(nfrom.orelse) != 0:
                    self.visit(nfrom)
                    if nto is not None:
                        for l in self.loose_end:
                            self.dataflow.add_flow(l, nto)
                        self.loose_end.clear()
                case _:
                    self.visit(nfrom)

                    if nto is None:
                        self.loose_end.add(nfrom)
                    else:
                        self.dataflow.add_flow(nfrom, nto)

                        # pop all nodes in self.loose and add flow to nto
                        for l in self.loose_end:
                            self.dataflow.add_flow(l, nto)
                        self.loose_end.clear()

    def _looping_flow(self, start: AST, node_sequence: list[AST]):
        """
        Given a start and a sequence of nodes, connect every loose_end and
        loose_continue back to the start.

        Both loose_continues and loose_ends are popped.

        Note that this does not perform sequential flow.
        """

        for n in self.loose_continue.union(self.loose_end):
            self.dataflow.add_flow(n, start)

        self.loose_continue.clear()
        self.loose_end.clear()

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        # Note that we're making a shallow copy of FunctionDef object for the
        # inner scope to flow from since the original node is reserved for the
        # outer scope and should never allow outer flows to leak into the
        # inner scope.
        # i.e. you would never run a function while you are defining it.
        node = copy(node)

        if len(node.body) != 0:
            self.dataflow.add_flow(node, node.body[0])

        self._sequential_flow(node.body)

        # No loose end within a function should make its way out of a function
        # scope
        self.loose_end.clear()

        # It's worth noting that Python will prevent loose continues
        # and loose breaks from showing up outside of a loop during parsing
        # stage.
        # If this ever happens, it's always this visitor's fault.
        if len(self.loose_continue) != 0:
            raise AssertionError("loose continue left in function scope")

        if len(self.loose_break) != 0:
            raise AssertionError("loose break left in function scope")

    def visit_If(self, node: ast.If) -> None:
        if len(node.body) != 0:
            self.dataflow.add_flow(node, node.body[0])
        self._sequential_flow(node.body)

        if len(node.orelse) != 0:
            self.dataflow.add_flow(node, node.orelse[0])
        self._sequential_flow(node.orelse)

    def visit_For(self, node: ast.For) -> Any:
        if len(node.body) != 0:
            self.dataflow.add_flow(node, node.body[0])
        self._sequential_flow(node.body)
        self._looping_flow(node, node.body)

        # The elusive for-else construct
        if len(node.orelse) != 0:
            self.dataflow.add_flow(node, node.orelse[0])
        self._sequential_flow(node.orelse)

        # Turn all loose breaks into loose ends
        self.loose_end.update(self.loose_break)
        self.loose_break.clear()

    def visit_While(self, node: ast.For) -> Any:
        if len(node.body) != 0:
            self.dataflow.add_flow(node, node.body[0])
        self._sequential_flow(node.body)
        self._looping_flow(node, node.body)

        # The elusive while-else construct
        if len(node.orelse) != 0:
            self.dataflow.add_flow(node, node.orelse[0])
        self._sequential_flow(node.orelse)

        # Turn all loose breaks into loose ends
        self.loose_end.update(self.loose_break)
        self.loose_break.clear()

    @staticmethod
    def analyze(node: AST) -> DataFlow[AST]:
        tdf = ToDataFlow()
        tdf.visit(node)
        return tdf.dataflow


class ToVariableGenKill(NodeVisitor):
    genkill: GenKill[AST, AST]

    def __init__(self):
        self.genkill = GenKill()

    def visit(self, node: AST) -> Any:
        super().visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        pass

    @staticmethod
    def analyze(node: AST) -> GenKill[AST, AST]:
        tvgk = ToVariableGenKill()
        tvgk.visit(node)
        return tvgk.genkill


class LiveVariableAnalysis(DataFlowAnalysis[AST, ast.Name]):
    confop = union

    def gens(self, node: AST) -> set[str]:
        return

    def analyze_func(self, func: FunctionDef):
        # TODO
        pass
