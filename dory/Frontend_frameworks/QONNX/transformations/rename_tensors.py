from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation


class RenameTensorsSequentially(BaseTrasformation):
    """
    Renames tensors to sequential integers following execution order,
    while guaranteeing that whenever the graph branches, one branch is
    fully completed (including any nested branches) before the sibling
    branch starts. Plain in-list or BFS/queue-based topological orders
    do NOT guarantee this -- they can interleave nodes from parallel
    branches, which breaks the L3 runtime's stack-based residual
    bookkeeping (residual_number / layers_pointers[]).

    Among sibling branches, the SHORTEST one is always scheduled (and
    fully drained) first -- this naturally prioritizes residual/skip
    branches, which are typically shorter than the main path they
    reconverge with, without needing to special-case any particular
    op_type (e.g. Add). "Shortest" is measured as the longest path
    (in edges) from a node to a graph sink, computed bottom-up before
    the forward walk. Two sibling branches share the exact same tail
    after they reconverge at a merge node, so that shared tail is a
    constant offset added equally to both branches' depth -- it cancels
    out, meaning comparing depth-to-sink is equivalent to comparing the
    branches' own lengths directly.
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)


    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        nodes = list(graph.node)
        n = len(nodes)

        producer_of = {}  
        for idx, node in enumerate(nodes):
            for out in node.output:
                producer_of[out] = idx

        predecessors = {idx: set() for idx in range(n)}
        successors = {idx: set() for idx in range(n)}
        for idx, node in enumerate(nodes):
            for i in range(min(2, len(node.input))):  # HACK to save bias tag
                in_name = node.input[i]
                if in_name in producer_of:
                    p = producer_of[in_name]
                    predecessors[idx].add(p)
                    successors[p].add(idx)

        # Step 1: compute depth[i] = longest path (in edges) from
        # node i to a sink. Process nodes only once all of
        # their successors are finalized (reverse topological order).
        remaining_out = {idx: len(successors[idx]) for idx in range(n)}
        depth = {}
        frontier = [idx for idx in range(n) if remaining_out[idx] == 0]
        for idx in frontier:
            depth[idx] = 0
        while frontier:
            idx = frontier.pop()
            for p in predecessors[idx]:
                remaining_out[p] -= 1
                if remaining_out[p] == 0:
                    depth[p] = 1 + max((depth[s] for s in successors[p]), default=0)
                    frontier.append(p)

        # --- Step 2: top-down -- DFS/stack-based topological sort, but
        # whenever several nodes become ready together (a fork), always
        # dive into the one with the smallest depth (shortest remaining
        # branch) first. Pushing in ascending-depth order (then popping
        # LIFO) means the smallest-depth node is pushed last and popped
        # first, driving the shortest branch to completion before the
        # stack ever offers the sibling branch's continuation. ---
        in_degree = {idx: len(predecessors[idx]) for idx in range(n)}
        stack = sorted(
            [idx for idx in range(n) if in_degree[idx] == 0],
            key=lambda x: (depth[x], x),
            reverse=True,
        )

        order = []
        visited = set()
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            order.append(idx)

            newly_ready = []
            for succ in sorted(successors[idx]):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    newly_ready.append(succ)
            # sort by depth descending so the smallest depth (shortest
            # branch) ends up pushed last, i.e. on top of the stack,
            # i.e. popped first
            newly_ready.sort(key=lambda x: (depth[x], x), reverse=True)
            for succ in newly_ready:
                stack.append(succ)

        assert len(order) == n, (
            "Could not produce a full topological order -- graph has a "
            "cycle or a disconnected node."
        )

        # Reorder the physical node list to match 
        ordered_nodes = [nodes[idx] for idx in order]
        del graph.node[:]
        graph.node.extend(ordered_nodes)

        rename_map = {}
        counter = 0
        for node in graph.node:
            for i in range(min(2, len(node.input))):  # HACK to save bias tag
                in_name = node.input[i]
                if in_name not in rename_map:
                    rename_map[in_name] = str(counter)
                    counter += 1
                node.input[i] = rename_map[in_name]
            for i in range(len(node.output)):
                out_name = node.output[i]
                if out_name not in rename_map:
                    rename_map[out_name] = str(counter)
                    counter += 1
                node.output[i] = rename_map[out_name]

        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.name in rename_map:
                vi.name = rename_map[vi.name]
        for init in graph.initializer:
            if init.name in rename_map:
                init.name = rename_map[init.name]

        return (model, False)