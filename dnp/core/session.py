"""
dnp/core/session.py
===================
`SessionGraph` — the central, dynamic Directed Acyclic Graph (DAG) that
records every computation for automatic differentiation.

A module-level singleton `session` is provided so that `Tensor` objects
can reference the same graph without importing a circular dependency.
"""

# Standard library
from contextlib import contextmanager

# Third-party libraries
import networkx as nx

# Optional visualization support
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class SessionGraph:
    """Central manager of the dynamic computation graph (DAG)."""

    def __init__(self):
        self.G = nx.DiGraph()
        self.compteur = 0
        self._grad_enabled = True

    # ------------------------------------------------------------------
    # Gradient-tracking control
    # ------------------------------------------------------------------

    @contextmanager
    def no_grad(self):
        """Context manager: temporarily disable gradient tracking."""
        prev_state = self._grad_enabled
        self._grad_enabled = False
        try:
            yield
        finally:
            self._grad_enabled = prev_state

    def set_grad_enabled(self, mode: bool):
        """Enable or disable operation tracking (analogous to torch.no_grad())."""
        self._grad_enabled = mode

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, noeud):
        """Register a node in the graph (no-op when tracking is disabled)."""
        if not self._grad_enabled:
            return None

        if not hasattr(noeud, "name"):
            raise AttributeError(
                f"L'objet {type(noeud)} passé au graphe doit posséder un attribut 'name'."
            )

        id_noeud = f"{noeud.name}_{self.compteur}"
        self.compteur += 1
        self.G.add_node(id_noeud, obj=noeud)
        return id_noeud

    def add_edge(self, parent_id, enfant_id):
        """Record a dependency (parent → child) between two nodes."""
        if not self._grad_enabled or parent_id is None or enfant_id is None:
            return

        if not self.G.has_node(parent_id):
            raise KeyError(f"Le nœud parent '{parent_id}' n'existe pas dans le graphe.")
        if not self.G.has_node(enfant_id):
            raise KeyError(f"Le nœud enfant '{enfant_id}' n'existe pas dans le graphe.")

        self.G.add_edge(parent_id, enfant_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """
        Clear the entire graph from memory.
        Must be called after every weight update in the training loop.
        """
        self.G.clear()
        self.compteur = 0

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def show_graph(self, title="Graphe de Calcul (Passe Avant)"):
        """Render the current computation graph with matplotlib."""
        if plt is None:
            print("⚠️ Matplotlib n'est pas installé. Visualisation impossible.")
            return

        if self.G.number_of_nodes() == 0:
            print("⚠️ Le graphe est actuellement vide. Rien à afficher.")
            return

        plt.figure(figsize=(10, 6))
        labels = {n: self.G.nodes[n]["obj"].name for n in self.G.nodes}
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            labels=labels,
            node_size=2000,
            node_color="#A0CBE2",
            font_size=10,
            font_weight="bold",
            arrowsize=20,
            edge_color="gray",
        )
        plt.title(title)
        plt.axis("off")
        plt.show()


# ---------------------------------------------------------------------------
# Module-level singleton  — shared by every Tensor in the process
# ---------------------------------------------------------------------------
session = SessionGraph()
