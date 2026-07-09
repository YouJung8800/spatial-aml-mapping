import pytest
import networkx as nx
import numpy as np

def test_graph_construction():
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=100)
    assert G.number_of_edges() == 1
    assert G[1][2]["weight"] == 100

def test_pagerank_sums_to_one():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    pr = nx.pagerank(G)
    assert abs(sum(pr.values()) - 1.0) < 1e-6

def test_visibility_index_range():
    raw_scores = {1: 0.1, 2: 0.5, 3: 0.9}
    vmin, vmax = min(raw_scores.values()), max(raw_scores.values())
    normalized = {k: (v - vmin) / (vmax - vmin) for k, v in raw_scores.items()}
    assert all(0 <= v <= 1 for v in normalized.values())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
