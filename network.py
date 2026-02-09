import numpy as np
import networkx as nx
from tqdm import tqdm

from .vessel import Vessel


class Heart:
    def __init__(self, inlet_file):
        self.input_data = np.loadtxt(inlet_file)
        self.cardiac_period = self.input_data[-1, 0]


class Network:
    def __init__(self, config, blood, heart, Ccfl, jump, tokeep, verbose=True):
        self.blood = blood
        self.heart = heart
        self.Ccfl = Ccfl

        self.graph = nx.DiGraph()
        self.vessels = {}
        self.edges = []

        if verbose:
            progress = tqdm(total=len(config), desc="Building network:")

        for vessel_config in config:
            vessel = Vessel(vessel_config, blood, jump, tokeep)
            self.graph.add_edge(vessel.sn, vessel.tn)
            self.vessels[(vessel.sn, vessel.tn)] = vessel
            self.edges.append((vessel.sn, vessel.tn))

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

        #self.edges = list(self.graph.edges())
        self._check()

    #def __iter__(self):
        #return self.vessels.items()    

    def _check(self):
        if self._in_degree_zero() !=1:
            raise ValueError("No input vessel")
        if self._out_degree_zero() == 0:
            raise ValueError("No output vessel(s)")
        if any(u == v for u, v in self.graph.edges()):
            raise ValueError("Self loop detected, i.e. sn == tn")
        if any(max(self.graph.in_degree(n), self.graph.out_degree(n)) > 2 for n in self.graph.nodes()):
            raise ValueError("Vertex belonging to more than 3 vessels")

    def _in_degree_zero(self):
        return sum(1 for _, deg in self.graph.in_degree() if deg == 0)

    def _out_degree_zero(self):
        return sum(1 for _, deg in self.graph.out_degree() if deg == 0)

