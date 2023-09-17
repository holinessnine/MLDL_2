import random
from collections import Counter, deque
from functools import lru_cache
from itertools import product
from typing import AbstractSet, Dict, Tuple, Sequence, Collection, List, TypeVar, Hashable, Generic, Iterable, FrozenSet

# for test
import numpy
from scipy import stats

import networkx as nx
import pandas as pd
from numpy.random import randint, choice, rand

CPTType = Dict[Tuple[int, ...], Sequence[float]]
T = TypeVar('T')
H = TypeVar('H', bound=Hashable)


def shuffled(xs: Iterable[T]) -> List[T]:
    # a randomly shuffled list
    xs = list(xs)  # copy
    random.shuffle(xs)  
    return xs


class visited_queue(Generic[H]):
    # queue that ignores already added before
    def __init__(self, xs: Collection[H]):
        self.visited = set(xs)
        self.queue = deque(self.visited)

    def extend(self, xs: Collection[H]):
        # add each element of xs if not already added
        xs = set(xs) - self.visited
        self.visited |= xs
        for x in xs:
            self.queue.append(x)

    def pop(self) -> H:
        # pop an element
        return self.queue.pop()

    def __bool__(self):
        return bool(self.queue)


class BayesianNetwork:
    def __init__(self, G: nx.DiGraph, CPTs: Dict[int, CPTType]):
        self.G = G  # graph object
        self.CPTs = CPTs  # conditional probability tables

        self.Vs = frozenset(self.G.nodes)
        self.n_variables = len(self.Vs)  # TODO number of variables in int
        self.variable_order = list(nx.topological_sort(self.G))  # TODO compute the order of variables

    def sample(self, n_instances=10000): #-> pd.DataFrame:
        # TODO perform prior sampling.
        data = []
        for _ in range(n_instances):
            sample = {}
            for variable in self.variable_order:
                parents = list(self.G.predecessors(variable))
                if len(parents) == 0:
                    # Variable has no parents
                    cpt = self.CPTs[variable]
                    sample[variable] = choice(cpt['values'], p=cpt['probs'])
                else:
                    # Variable has parents
                    parent_values = tuple(sample[parent] for parent in parents)
                    cpt = self.CPTs[variable][parent_values]
                    sample[variable] = choice(cpt['values'], p=cpt['probs'])
            data.append(sample)
        
        # returns a data frame with columns 0, 1, 2, ..., self.n_variables
        return pd.DataFrame(data)

    def is_d_separated(self,
                       Xs: AbstractSet[int],
                       Ys: AbstractSet[int],
                       Zs: AbstractSet[int] = frozenset()) -> bool:
        # test xs _||_ ys | zs
        Xs, Ys = Xs - Zs, Ys - Zs
        if Xs & Ys:
            return False
        return all(self.__is_d_separated(x, y, Zs) for x, y in product(Xs, Ys))

    @lru_cache
    def Pa(self, u):
        # Parents
        return frozenset(self.G.predecessors(u))

    @lru_cache
    def Ch(self, u):
        # Children
        return frozenset(self.G.successors(u))

    @lru_cache
    def An(self, x) -> FrozenSet:
        # TODO Ancestors (including x itself)
        if not self.G.predecessors(x):
            # x has no parents, so it is its own ancestor
            return frozenset([x])
        else:
            ancestors = frozenset([x])
            for parent in self.G.predecessors(x):
                ancestors |= self.An(parent)
            return ancestors

    @lru_cache
    def De(self, x) -> FrozenSet:
        # TODO Descendants (including x itself)
        return ...

    def __is_d_separated(self,
                         X: int,
                         Y: int,
                         Zs: AbstractSet[int] = frozenset()) -> bool:
        # TODO test X _||_ Y | Zs based on self.G. You may utilize visited_queue
        assert X not in Zs
        assert Y not in Zs
        assert X != Y

        # Initialize visited nodes and active trail
        visited = set()
        active_trail = [(X, None)]
        
        while active_trail:
            node, parent = active_trail.pop()
            
            if node == Y:
                # We reached Y, so X and Y are d-connected given Zs
                return True
            
            if node in visited:
                # We already visited this node, so skip it
                continue
            
            visited.add(node)
            
            # Check the state of the node
            if node in Zs:
                # The node is in Zs, so it blocks the trail
                continue
                
            if parent is None or parent in Zs or (parent, node) in self.G.edges:
                # The node is a root node or a non-collider node with unblocked parents
                for child in self.G.successors(node):
                    active_trail.append((child, node))
                    
            else:
                # The node is a collider node with unblocked parents
                for child in self.G.successors(node):
                    if child != parent:
                        active_trail.append((child, node))

        # TODO
        return False


def check_conditional_independence(data, X, Y, Zs: AbstractSet[int] = frozenset(), alpha=0.05) -> bool: # : pd.DataFrame
    # TODO test conditional independence by performing chi-square test for each condition value
    
        # Compute the contingency table for X, Y, and Zs
    table = pd.crosstab(index=data[X], columns=[data[Y]] + [data[Z] for Z in Zs], margins=False)

        # Compute the expected frequencies under the null hypothesis of independence
    expected = numpy.outer(table.sum(axis=1), table.sum(axis=0)) / data.shape[0]

        # Compute the chi-square test statistic
    chi2 = numpy.sum((table - expected)**2 / expected)

        # Compute the degrees of freedom
    df = numpy.prod([table[c].shape[0] - 1 for c in table.columns])

        # Compute the p-value using the chi-square distribution
    p_value = 1 - stats.chi2.cdf(chi2, df)
    
    # TODO Use Bonferroni correction for multiple hypothesis testing.

        # Apply Bonferroni correction for multiple hypothesis testing
    alpha_corrected = alpha / len(Zs)

        # Test the null hypothesis of independence at the desired significance level
    return p_value > alpha_corrected


def random_DAG(num_vars: int, num_edges: int) -> nx.DiGraph:
    assert num_vars > 0 and num_edges >= 0
    # TODO randomly generate an acyclic graph with the specfied number of variables and number of edges.
    nodes = [i for i in range(num_vars)]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    
    order = shuffled(range(num_vars))
    
    # TODO the resulting DAG should follow the (topological) order above.
    result = nx.DiGraph([(order.index(u), order.index(v)) for (u, v) in graph.edges()])
    return result


def random_CPTs(G: nx.DiGraph) -> Dict[int, CPTType]:
    # randomly generate a CPT for every variable
    cpts = dict()
    for v_i in G.nodes:
        cpt_i = dict()
        parents = tuple(sorted(G.predecessors(v_i)))
        for pa_i in product([0, 1], repeat=len(parents)):  # for all parents values
            pr = rand() * 0.3 + 0.1  # 0.1~0.4
            if randint(2):  # 0 or 1
                pr = 1 - pr
            cpt_i[pa_i] = [pr, 1 - pr]
        cpts[v_i] = cpt_i
    return cpts


def random_BN(num_vars: int, num_edges: int) -> BayesianNetwork:
    # randomly generate a Bayesian network
    graph = random_DAG(num_vars, num_edges)
    CPTs = random_CPTs(graph)
    return BayesianNetwork(graph, CPTs)


def main():
    results = []
    alpha = 0.05

    for _ in range(25):
        # randomly generate a Bayesian network
        bn = random_BN(N := randint(3, 8),
                       randint(N - 1, N * (N - 1) // 2))

        # random sample
        data = bn.sample(10000)

        # # randomly check conditional independence from sample and d-separation, 0 <= ... <=N-2
        for _ in range(20):  # may repeat the same test... whatever :)
            Zs = set(choice(N, randint(0, min(3, N - 1)), replace=False))
            X, Y, *_ = shuffled(bn.Vs - Zs)

            truth = bn.is_d_separated({X}, {Y}, Zs)
            estim = check_conditional_independence(data, X, Y, Zs, alpha)

            results.append((truth, estim))

    counter = Counter(results)
    TP = counter.get((False, False), 0)
    TN = counter.get((True, True), 0)
    FP = counter.get((True, False), 0)

    print(f'true positive rate = {TP / (TP + FP):.3f} & false discovery rate = {FP / (TN + FP):.3f}, which should be close to {alpha=}')


if __name__ == '__main__':
    main()
