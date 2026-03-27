"""
dnp/core/session.py
===================
`SessionGraph` — the central, dynamic Directed Acyclic Graph (DAG) that
records every computation for automatic differentiation.

A module-level singleton `session` is provided so that `Tensor` objects
can reference the same graph without importing a circular dependency.

Design note
-----------
NetworkX is an *optional* dependency used only for visualisation.  The hot
training path never touches an nx.DiGraph — only two plain dicts are updated
(O(1) per node/edge), which eliminates the networkx overhead from every
forward pass.  When ``show_graph()`` or ``session.G`` is accessed, an
nx.DiGraph is built on-demand from those dicts in O(n).

v3 additions
------------
* ``graph_fn`` decorator — wraps any callable so that raw numpy/cupy arrays
  and Python scalars are automatically promoted to Tensor before the function
  body executes.  This makes every op graph-compatible with zero boilerplate.
* ``session.graph()`` context manager — resets the graph on entry and on exit,
  giving a clean per-forward-pass scope for memory management.
"""

# Standard library
import functools
from contextlib import contextmanager

# Optional visualization support
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ---------------------------------------------------------------------------
# graph_fn decorator
# ---------------------------------------------------------------------------


def graph_fn(func):
    """Decorator that auto-converts non-Tensor inputs to Tensor.

    Any positional or keyword argument that is a numpy/cupy ndarray or a plain
    Python scalar (int, float) is transparently wrapped in a :class:`Tensor`
    before the decorated function runs.  Arguments that are already a
    :class:`Tensor` pass through unmodified.

    Usage
    -----
    >>> @graph_fn
    ... def my_op(x, y):
    ...     return x + y
    ...
    >>> my_op(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    # Both arrays are now Tensors inside my_op

    Notes
    -----
    The import of :class:`Tensor` is deferred to avoid circular-import issues
    (``tensor.py`` imports ``session``, so we cannot import it at module level).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from .tensor import Tensor  # deferred — avoids circular import

        def _to_tensor(v):
            if isinstance(v, Tensor):
                return v
            # Only promote ndarray-like objects (numpy / cupy arrays).
            # Plain Python scalars (int, float, bool) and structural params
            # (axis, shape, keepdims, p, training, …) are intentionally left
            # as-is so ops like expand_dims(x, 0) and dropout(x, p=0.5) work.
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return Tensor(v, name=f"{func.__name__}_in")
            return v  # leave int, float, bool, tuple, str, etc. unchanged

        new_args = tuple(_to_tensor(a) for a in args)
        new_kwargs = {k: _to_tensor(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    wrapper._graph_fn = True  # sentinel so callers can detect decoration
    return wrapper


class SessionGraph:
    """Central manager of the dynamic computation graph (DAG)."""

    def __init__(self):
        # Lightweight hot-path storage (no nx in the loop)
        self._nodes: dict = {}  # node_id  -> display name
        self._edges: list = []  # [(parent_id, child_id), ...]
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
    # Graph construction  (hot path — plain dict/list, no nx)
    # ------------------------------------------------------------------

    def add_node(self, noeud):
        """Register a node in the graph (no-op when tracking is disabled)."""
        if not self._grad_enabled:
            return None

        if not hasattr(noeud, "name"):
            raise AttributeError(
                f"L'objet {type(noeud)} passé au graphe doit posséder un attribut 'name'."
            )

        node_id = f"{noeud.name}_{self.compteur}"
        self.compteur += 1
        self._nodes[node_id] = noeud.name
        return node_id

    def add_edge(self, parent_id, enfant_id):
        """Record a dependency (parent → child) between two nodes."""
        if not self._grad_enabled or parent_id is None or enfant_id is None:
            return

        if parent_id not in self._nodes:
            raise KeyError(f"Le nœud parent '{parent_id}' n'existe pas dans le graphe.")
        if enfant_id not in self._nodes:
            raise KeyError(f"Le nœud enfant '{enfant_id}' n'existe pas dans le graphe.")

        self._edges.append((parent_id, enfant_id))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """
        Clear the entire graph from memory.
        Must be called after every weight update in the training loop.
        """
        self._nodes.clear()
        self._edges.clear()
        self.compteur = 0

    @contextmanager
    def graph(self):
        """Context manager for a clean per-forward-pass computation graph.

        Resets the session graph on entry so stale nodes from previous
        iterations are removed, preventing the unbounded memory growth that
        occurs when the global session accumulates nodes across training steps.

        After the ``with`` block exits (whether normally or via exception)
        the graph is reset again so parameter tensors re-register on the
        next forward pass.

        Example
        -------
        >>> for x_batch, y_batch in dataloader:
        ...     with session.graph():
        ...         pred = model(x_batch)
        ...         loss = criterion(pred, y_batch)
        ...         loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
        """
        self.reset()
        try:
            yield self
        finally:
            pass  # graph stays alive so backward() can traverse it after the block

    # ------------------------------------------------------------------
    # Backward-compatible nx.DiGraph property (built on-demand)
    # ------------------------------------------------------------------

    @property
    def G(self):
        """Return an nx.DiGraph built on-demand from the lightweight store.

        This property exists purely for backward compatibility with code that
        inspects ``session.G`` (tests, notebooks).  It is *not* called during
        the forward or backward pass.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "NetworkX is required for session.G / show_graph().  "
                "Install it with: pip install networkx"
            ) from exc

        G = nx.DiGraph()
        G.add_nodes_from((nid, {"obj_name": name}) for nid, name in self._nodes.items())
        G.add_edges_from(self._edges)
        return G

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def show_graph(
        self,
        title="Graphe de Calcul (Passe Avant)",
        save=False,
        filename="computation_graph.png",
        figsize=None,
    ):
        """Render the current computation graph with a clean hierarchical layout.

        Nodes are color-coded by role:
        - Green  : leaf inputs / parameters (no incoming edges)
        - Blue   : intermediate operation nodes
        - Orange : output / loss node (no outgoing edges)
        - Gray   : isolated nodes (e.g. freshly recreated parameters)

        Layout uses Sugiyama-style topological layers (top → bottom) when the
        graph is a DAG, falling back to a tuned spring layout for cyclic graphs.
        Figure size, node size and font size all scale automatically with the
        number of nodes so both tiny and large graphs look readable.

        Parameters
        ----------
        title : str
            Title shown above the figure.
        save : bool
            Whether to save the figure to *filename* instead of displaying it.
        filename : str
            Output path when *save=True*.
        figsize : tuple[float, float] | None
            Override the automatic figure size (width, height) in inches.
        """
        if plt is None:
            print("⚠️ Matplotlib n'est pas installé. Visualisation impossible.")
            return

        if not self._nodes:
            print("⚠️ Le graphe est actuellement vide. Rien à afficher.")
            return

        try:
            import networkx as nx
        except ImportError:
            print(
                "⚠️ NetworkX n'est pas installé. Impossible de générer la visualisation.\n"
                "   Installez-le avec: pip install networkx"
            )
            return

        # ---- Build directed graph ----------------------------------------
        G = nx.DiGraph()
        for nid, name in self._nodes.items():
            G.add_node(nid, _label=name)
        G.add_edges_from(self._edges)

        n_nodes = G.number_of_nodes()

        # ---- Classify nodes by role --------------------------------------
        connected = {u for u, v in G.edges()} | {v for u, v in G.edges()}
        isolated = [nd for nd in G.nodes() if nd not in connected]

        # ---- Hierarchical layout (Sugiyama / layered DAG) ----------------
        pos = None
        try:
            H = G.subgraph(connected)
            n_layers = 0
            if len(H):
                generations = list(nx.topological_generations(H))
                n_layers = len(generations)
                for layer_idx, layer_nodes in enumerate(generations):
                    for nd in layer_nodes:
                        G.nodes[nd]["_layer"] = layer_idx
            # Isolated nodes sit in their own layer past all connected ones
            for nd in isolated:
                G.nodes[nd]["_layer"] = n_layers
            pos = nx.multipartite_layout(
                G, subset_key="_layer", align="horizontal", scale=2.0
            )
        except Exception:
            pass  # cycle or old NetworkX — fall back below

        if pos is None:
            k_val = 2.5 / max(n_nodes**0.5, 1)
            pos = nx.spring_layout(G, seed=42, k=k_val, iterations=120)

        # ---- Adaptive figure / node / font sizing ------------------------
        if figsize is None:
            fw = max(10, min(26, 8 + n_nodes * 0.9))
            fh = max(6, min(20, 4 + n_nodes * 0.6))
            figsize = (fw, fh)

        node_sz = max(700, min(2400, 20000 // max(n_nodes, 1)))
        font_sz = max(7, min(11, 110 // max(n_nodes, 1)))
        arrow_sz = max(12, min(22, node_sz // 100))

        # ---- Color by node role ------------------------------------------
        # green  → leaf / parameter (no incoming edges)
        # blue   → intermediate operation (has both in and out edges)
        # orange → output / loss (no outgoing edges, has incoming)
        # gray   → isolated (no edges at all)
        node_colors = []
        for nd in G.nodes():
            if nd in isolated:
                node_colors.append("#C8C8C8")
            elif G.in_degree(nd) == 0:
                node_colors.append("#7EC8A4")  # teal-green
            elif G.out_degree(nd) == 0:
                node_colors.append("#F4A261")  # amber
            else:
                node_colors.append("#8CC4DB")  # sky-blue

        labels = {nd: G.nodes[nd]["_label"] for nd in G.nodes()}

        # ---- Draw --------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sz,
            node_color=node_colors,
            alpha=0.93,
            linewidths=1.2,
            edgecolors="#444444",
        )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            labels=labels,
            font_size=font_sz,
            font_weight="bold",
            font_color="#1A1A1A",
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrowsize=arrow_sz,
            edge_color="#555555",
            arrows=True,
            node_size=node_sz,
            connectionstyle="arc3,rad=0.04",
            min_source_margin=10,
            min_target_margin=10,
        )

        # ---- Legend ------------------------------------------------------
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor="#7EC8A4", edgecolor="#444", label="Input / Parameter"),
            Patch(facecolor="#8CC4DB", edgecolor="#444", label="Operation"),
            Patch(facecolor="#F4A261", edgecolor="#444", label="Output / Loss"),
        ]
        if isolated:
            legend_handles.append(
                Patch(facecolor="#C8C8C8", edgecolor="#444", label="Isolated node")
            )
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=8,
            framealpha=0.85,
            edgecolor="#AAAAAA",
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        ax.axis("off")
        plt.tight_layout(pad=1.5)

        if save:
            plt.savefig(filename, bbox_inches="tight", dpi=150)
            print(f"✅ Graphe sauvegardé sous : {filename}")
        else:
            plt.show()
        plt.close()

    # Alias for french/typo
    show_graphe = show_graph


# ---------------------------------------------------------------------------
# Module-level singleton  — shared by every Tensor in the process
# ---------------------------------------------------------------------------
session = SessionGraph()
