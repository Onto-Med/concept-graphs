"""
Created on May 6, 2021

@author: Navid Dianati
@author: (adapted code; 08.06.2022) Franz Matthies

Implements the Marginal Likelihood Filter which computes a significance score for each
edge of an integer-weighted graph based on a maximum likelihood null model derived
from the configuration model
"""

import copy
import logging
from numbers import Number

import networkx as nx
import numpy as np
from scipy.stats import binomtest

logger = logging.getLogger()
logger.setLevel("DEBUG")

# clip log of binomtest p-values to this value.
MAX_NEG_LOG = np.log(np.finfo(np.float64).max)

# ToDo: vectorize computation of significance


class MLF:
    """Marginal Likelihood Filter for weighted NetworkX graphs.

    The concept-graphs pipeline uses NetworkX internally, so this implementation
    intentionally accepts only simple ``networkx.Graph``/``networkx.DiGraph``
    inputs. Every edge must have a numeric ``weight`` attribute.
    """

    def __init__(self, directed=True):
        self.directed = directed

    def fit_transform(
        self, graph: nx.Graph, weight_as_percentile=True, return_copy=False
    ):
        """
        Receive a representation of the graph and return a similar
        representation, only with a significance score calculated
        for each edge.

        ``graph`` must be a simple NetworkX graph with numeric edge weights.
        The returned graph contains an additional ``significance`` edge
        attribute. By default the input graph is modified in place; pass
        ``return_copy=True`` to leave the input untouched.
        """
        self._check_graph(graph)

        if return_copy:
            h = copy.deepcopy(graph)
            g = self._compute_significance(
                graph=h,
                weight_is_percentile=weight_as_percentile,
                directed=h.is_directed(),
            )
        else:
            g = self._compute_significance(
                graph=graph,
                weight_is_percentile=weight_as_percentile,
                directed=graph.is_directed(),
            )
        return g

    @staticmethod
    def _check_graph(graph: nx.Graph) -> None:
        """Validate the supported MLF input graph shape."""
        if not isinstance(graph, nx.Graph):
            raise TypeError("MLF only supports networkx.Graph/networkx.DiGraph inputs.")
        if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError("MLF only supports simple graphs, not multigraphs.")
        if any(source == target for source, target in graph.edges):
            raise ValueError("MLF only supports simple graphs without self-loops.")
        for source, target, data in graph.edges(data=True):
            if "weight" not in data:
                raise ValueError(
                    f"Edge ({source!r}, {target!r}) is missing required 'weight'."
                )
            if not isinstance(data["weight"], Number):
                raise TypeError(
                    f"Edge ({source!r}, {target!r}) has non-numeric weight "
                    f"{data['weight']!r}."
                )

    # def _cast(self, graph, dtype):
    #     '''
    #      Convert the igraph.Graph instance back to the
    #      original format of the input.
    #      '''
    #     if dtype == ig.Graph:
    #         return graph
    #     elif dtype == pd.DataFrame:
    #         # graph was constructed from a dataframe and
    #         # therefore each vertex has a "name" attribute
    #         edgelist = [(graph.vs[e.source]['name'], graph.vs[e.target]['name'], e['weight'], e['significance']) for e in graph.es]
    #         df = pd.DataFrame(edgelist, columns=['source', "target", "weight", "significance"])
    #         return df
    #     elif dtype == list:
    #         # graph was constructed from a list of tuples
    #         # and therefore each vertex has a "name" attribute
    #         edgelist = [(graph.vs[e.source]['name'], graph.vs[e.target]['name'], e['weight'], e['significance']) for e in graph.es]
    #         return edgelist
    #     else:
    #         raise(TypeError('Can only recast the graph into one of: igraph.Graph, a DataFrame or a list of 3-tuples'))

    def _compute_significance(
        self, graph: nx.Graph, weight_is_percentile: bool, directed: bool
    ):
        mult = 1
        if weight_is_percentile:
            mult = 100
        ks = graph.degree(weight="weight")
        total_degree = sum(dict(ks).values()) * mult

        for i0, i1, d in graph.edges(data=True):
            p = None
            try:
                if directed:
                    p = _pvalue_directed(
                        w_uv=d["weight"] * mult,
                        ku_out=ks[i0] * mult,
                        kv_in=ks[i1] * mult,
                        q=total_degree / 2.0,
                    )
                else:
                    p = _pvalue_undirected(
                        w=d["weight"] * mult,
                        ku=ks[i0] * mult,
                        kv=ks[i1] * mult,
                        q=total_degree / 2.0,
                    )
                # Due to instabilities in binomtest at p-values
                # near "tiny", we clip all significance values
                # to MAX_NEG_LOG
                d["significance"] = min(
                    MAX_NEG_LOG, MAX_NEG_LOG if p <= 0 else -np.log(p)
                )
            except (ValueError, TypeError, ArithmeticError) as error:
                logger.warning("Could not compute edge significance: %s", error)
                logger.debug(
                    "Significance inputs weight=%s ks[i0]=%s ks[i1]=%s total_degree=%s p=%s",
                    d["weight"],
                    ks[i0],
                    ks[i1],
                    total_degree,
                    p,
                )
                d["significance"] = None

        try:
            max_sig = max(
                [s for s in graph.edges(data=True) if s is not None],
                key=lambda edge: edge[2].get("significance", 0.0),
                default=(
                    0,
                    0,
                    {"significance": 0.0},
                ),
            )[2]["significance"]
        except TypeError as te:  # ToDo: there were case where TypeErrors were thrown (comparison btw. 'NoneType') but I thought I made sure no 'NoneType' was allowed in the list...
            logging.error(f"{te}\n-->\t{graph.edges(data=True)}")
            max_sig = 0.0
        for _, _, d in graph.edges(data=True):
            if d["significance"] is None:
                d["significance"] = max_sig

        return graph

    # def _compute_significance_directed(self, G: nx.Graph, weight_is_percentile: bool):
    #     """
    #     Compute the edge significance for the edges of the
    #     given graph C{G} in place. C{G.es['weight']} is expected
    #     to have been set already.
    #
    #     @param G: C{igraph.Graph} instance. C{G} is assumed to be directed.
    #     """
    #     mult = 1
    #     if weight_is_percentile:
    #         mult = 100
    #     ks = G.degree(weight='weight') * mult
    #     total_degree = sum(dict(ks).values())
    #
    #     for i0, i1, d in G.edges(data=True):
    #         p = None
    #         try:
    #             p = _pvalue_directed(
    #                 w_uv=d['weight'] * mult, ku_out=ks[i0] * mult,
    #                 kv_in=ks[i1] * mult, q=total_degree / 2.0,
    #             )
    #             # Due to instabilities in binomtest at p-values
    #             # near "tiny", we clip all significance values
    #             # to MAX_NEG_LOG
    #             d['significance'] = min(
    #                 MAX_NEG_LOG, MAX_NEG_LOG if p <= 0 else -np.log(p)
    #             )
    #         except ValueError as error:
    #             logger.warning("warning: ValueError {}".format(str(error)))
    #             logger.debug("ValueError weight: {} ks[i0]:{} ks[i1]:{} total_degree:{} p:{}"
    #                          .format(d['weight'], ks[i0], ks[i1], total_degree, p))
    #             d['significance'] = None
    #         except Exception as error:
    #             logger.warning("warning: Exception {}".format(str(error)))
    #             d['significance'] = None
    #             # print "error computing significance", p
    #
    #     max_sig = max([s for s in G.edges(data=True) if s is not None],
    #                   key=lambda edge: edge[2].get("significance", .0),
    #                   default=(0, 0, {'significance': 0.0},))[2]["significance"]
    #     for _, _, d in G.edges(data=True):
    #         if d['significance'] is None:
    #             d['significance'] = max_sig
    #
    #     return G
    #
    # def _compute_significance_undirected(self, G: nx.Graph, weight_is_percentile: bool):
    #     """
    #     Compute the edge significance for the edges of the
    #     given graph C{G} in place. C{G.es['weight']} is expected
    #     to have been set already.
    #     @param G: C{igraph.Graph} instance. C{G} is assumed to be undirected.
    #     """
    #     mult = 1
    #     if weight_is_percentile:
    #         mult = 100
    #     ks = G.degree(weight='weight')
    #     total_degree = sum(dict(ks).values()) * mult
    #
    #     for i0, i1, d in G.edges(data=True):
    #         p = None
    #         try:
    #             p = _pvalue_undirected(
    #                 w=d['weight'] * mult, ku=ks[i0] * mult,
    #                 kv=ks[i1] * mult, q=total_degree / 2.0,
    #             )
    #
    #             # Due to instabilities in binomtest at p-values
    #             # near "tiny", we clip all significance values
    #             # to MAX_NEG_LOG
    #             d['significance'] = min(
    #                 MAX_NEG_LOG, MAX_NEG_LOG if p <= 0 else -np.log(p)
    #             )
    #         except ValueError as error:
    #             logger.warning("warning: ValueError {}".format(str(error)))
    #             logger.debug("ValueError weight: {} ks[i0]:{} ks[i1]:{} total_degree:{} p:{}"
    #                          .format(d['weight'], ks[i0], ks[i1], total_degree, p))
    #             d['significance'] = None
    #         except Exception as error:
    #             logger.warning("warning: Exception {}".format(str(error)))
    #             d['significance'] = None
    #
    #     max_sig = max([s for s in G.edges(data=True) if s is not None],
    #                   key=lambda edge: edge[2].get("significance", .0),
    #                   default=(0, 0, {'significance': 0.0},))[2]["significance"]
    #     for _, _, d in G.edges(data=True):
    #         if d['significance'] is None:
    #             d['significance'] = max_sig
    #     return G

    # def _convert_to_graph(self, graph, directed=False):
    #     '''
    #      If graph is not an instance of igraph.Graph, then convert
    #      it to one.
    #      '''
    #     if isinstance(graph, ig.Graph):
    #         if not graph.is_simple():
    #             raise ValueError('Expected a simple graph: no loops and no multiple edges')
    #         if (graph.is_directed() and not directed):
    #             raise ValueError('Provided graph is directed but the "directed" parameter is False')
    #         if (not graph.is_directed() and directed):
    #             raise ValueError('Provided graph is undirected but the "directed" parameter is True')
    #         return graph
    #
    #     if isinstance(graph, list):
    #         g = ig.Graph.TupleList(graph, directed=directed, weights=True)
    #         if not g.is_simple():
    #             raise ValueError('Expected a simple graph: no loops and no multiple edges')
    #         return g
    #     elif isinstance(graph, pd.DataFrame):
    #         g = ig.Graph.TupleList(
    #             graph[['source', 'target', 'weight']].itertuples(index=False),
    #             directed=directed,
    #             weights=True)
    #         if not g.is_simple():
    #             raise ValueError('Expected a simple graph: no loops and no multiple edges')
    #         return g
    #     else:
    #         raise TypeError(
    #             'graph must be an instance of one of the following: igraph.Graph, list, pandas.DataFrame.'
    #             )

    # def _check_types(self, graph):
    #     if type(graph) == pd.DataFrame:
    #         for column in ["source", "target", "weight"]:
    #             if column not in graph.columns:
    #                 raise ValueError('{} must be in the DataFrame\'s columns'.format(column))
    #         if graph["weight"].min() < 0:
    #             raise ValueError("All weights must be non-negative")
    #
    #     elif type(graph) == list:
    #         if set([type(x) for x in graph]) != {tuple}:
    #             raise TypeError("Each element of the edgelist must be a tuple")
    #
    #         if set([len(x) for x in graph]) != {3}:
    #             raise ValueError("Each element of the edgelist must have length 3")
    #
    #         if set([type(x[2]) for x in graph]) != {int}:
    #             raise TypeError("The 3rd element of each edge tuple must be an int")
    #
    #         if min([x[2] for x in graph]) < 0:
    #             raise ValueError("All weights must be non-negative")
    #
    #     elif type(graph) == ig.Graph:
    #         if "weight" not in graph.es.attributes():
    #             raise ValueError("The EdgeSequence must have a 'weight' attribute")
    #         types = set([type(e['weight']) for e in graph.es])
    #         if types not in  [{int}, {int, np.int64}, {np.int64}]:
    #             raise TypeError('The weight attribute of the edges must be an int')
    #         if min([e['weight'] for e in graph.es]) < 0:
    #             raise ValueError('All weights must be non-negative')
    #
    #     else:
    #         raise(ValueError('The graph must be given as a, igraph.Graph, a DataFrame or a list of 3-tuples'))


def _pvalue_undirected(w, ku, kv, q):
    """
    Compute the pvalue for the undirected edge null model.
    Use a standard binomial test from the statsmodels package.

    @param w: weight of the undirected edge.
    @param ku: total incident weight (strength) of the first vertex.
    @param kv: total incident weight (strength) of the second vertex.
    @keyparamword q: total incident weight of all vertices divided by two. Similar to the total number of edges in the graph.
    """
    if not all(v is not None for v in [w, ku, kv, q]):
        raise ValueError

    p = ku * kv * 1.0 / q / q / 2.0
    return binomtest(k=w, n=q, p=p, alternative="greater")


def _pvalue_directed(w_uv, ku_out, kv_in, q):
    """
    Compute the pvalue for the directed edge null model.
    Use a standard binomial test from the statsmodels package

    @param w_uv: Weight of the directe edge.
    @param ku_out: Total outgoing weight of the source vertex.
    @param kv_in: Total incoming weight of the destination vertex.
    @param q: Total sum of all edge weights in the graph.
    """
    if not all(v is not None for v in [w_uv, ku_out, kv_in, q]):
        raise ValueError

    p = 1.0 * ku_out * kv_in / q / q / 1.0
    return binomtest(k=w_uv, n=q, p=p, alternative="greater")
