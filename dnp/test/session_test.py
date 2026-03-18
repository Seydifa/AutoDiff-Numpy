"""
session_test.py
===============
Tests for dnp/core/session.py:
  - SessionGraph node/edge management
  - Grad-tracking enable/disable
  - no_grad() context manager
  - reset() lifecycle
  - Error handling (missing name, invalid edge ids)
  - Module-level singleton `session`
"""

# Third-party
import pytest
import networkx as nx

# Local imports
from dnp.core.session import SessionGraph, session


class MockNode:
    def __init__(self, name):
        self.name = name


# ===========================================================================
# Node management
# ===========================================================================


class TestAddNode:
    def test_returns_correct_id(self):
        sg = SessionGraph()
        node = MockNode("x")
        nid = sg.add_node(node)
        assert nid == "x_0"

    def test_node_exists_in_graph(self):
        sg = SessionGraph()
        node = MockNode("x")
        nid = sg.add_node(node)
        assert sg.G.has_node(nid)

    def test_node_stores_object(self):
        sg = SessionGraph()
        node = MockNode("x")
        nid = sg.add_node(node)
        assert sg.G.nodes[nid]["obj"] is node

    def test_counter_increments(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        id2 = sg.add_node(MockNode("b"))
        assert id1 == "a_0"
        assert id2 == "b_1"

    def test_missing_name_raises(self):
        sg = SessionGraph()

        class BadNode:
            pass

        with pytest.raises(AttributeError, match="doit posséder un attribut 'name'"):
            sg.add_node(BadNode())


# ===========================================================================
# Edge management
# ===========================================================================


class TestAddEdge:
    def test_adds_valid_edge(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        id2 = sg.add_node(MockNode("b"))
        sg.add_edge(id1, id2)
        assert sg.G.has_edge(id1, id2)

    def test_missing_parent_raises(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        with pytest.raises(KeyError, match="Le nœud parent 'ghost' n'existe pas"):
            sg.add_edge("ghost", id1)

    def test_missing_child_raises(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        with pytest.raises(KeyError, match="Le nœud enfant 'ghost' n'existe pas"):
            sg.add_edge(id1, "ghost")

    def test_none_parent_is_noop(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        sg.add_edge(None, id1)  # should not raise
        assert sg.G.number_of_edges() == 0

    def test_none_child_is_noop(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        sg.add_edge(id1, None)  # should not raise
        assert sg.G.number_of_edges() == 0

    def test_both_none_is_noop(self):
        sg = SessionGraph()
        sg.add_edge(None, None)
        assert sg.G.number_of_edges() == 0


# ===========================================================================
# Gradient tracking control
# ===========================================================================


class TestGradEnabled:
    def test_enabled_by_default(self):
        sg = SessionGraph()
        assert sg._grad_enabled is True

    def test_disable_skips_node(self):
        sg = SessionGraph()
        sg.set_grad_enabled(False)
        nid = sg.add_node(MockNode("x"))
        assert nid is None
        assert sg.G.number_of_nodes() == 0

    def test_re_enable_adds_node(self):
        sg = SessionGraph()
        sg.set_grad_enabled(False)
        sg.set_grad_enabled(True)
        nid = sg.add_node(MockNode("x"))
        assert nid is not None

    def test_disable_skips_edge(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        id2 = sg.add_node(MockNode("b"))
        sg.set_grad_enabled(False)
        sg.add_edge(id1, id2)
        assert sg.G.number_of_edges() == 0


# ===========================================================================
# no_grad() context manager
# ===========================================================================


class TestNoGrad:
    def test_disables_inside_block(self):
        sg = SessionGraph()
        with sg.no_grad():
            assert sg._grad_enabled is False
            nid = sg.add_node(MockNode("x"))
            assert nid is None

    def test_restores_after_block(self):
        sg = SessionGraph()
        with sg.no_grad():
            pass
        assert sg._grad_enabled is True

    def test_restores_after_exception(self):
        sg = SessionGraph()
        try:
            with sg.no_grad():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert sg._grad_enabled is True

    def test_nested_no_grad_restores_outer_state(self):
        sg = SessionGraph()
        sg.set_grad_enabled(False)  # outer state = disabled
        with sg.no_grad():
            assert sg._grad_enabled is False
        assert sg._grad_enabled is False  # restored to pre-context value


# ===========================================================================
# reset()
# ===========================================================================


class TestReset:
    def test_clears_nodes(self):
        sg = SessionGraph()
        sg.add_node(MockNode("a"))
        sg.add_node(MockNode("b"))
        sg.reset()
        assert sg.G.number_of_nodes() == 0

    def test_resets_counter(self):
        sg = SessionGraph()
        sg.add_node(MockNode("a"))
        sg.reset()
        assert sg.compteur == 0

    def test_clears_edges(self):
        sg = SessionGraph()
        id1 = sg.add_node(MockNode("a"))
        id2 = sg.add_node(MockNode("b"))
        sg.add_edge(id1, id2)
        sg.reset()
        assert sg.G.number_of_edges() == 0

    def test_can_add_nodes_after_reset(self):
        sg = SessionGraph()
        sg.add_node(MockNode("a"))
        sg.reset()
        nid = sg.add_node(MockNode("b"))
        assert nid == "b_0"


# ===========================================================================
# Module singleton
# ===========================================================================


class TestSessionSingleton:
    def test_singleton_is_session_graph(self):
        assert isinstance(session, SessionGraph)

    def test_importing_multiple_times_gives_same_object(self):
        from dnp.core.session import session as s1
        from dnp.core import session as s2

        assert s1 is s2
