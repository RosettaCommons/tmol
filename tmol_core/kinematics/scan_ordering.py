"""Shared atom-tree scan planning and validated native scan metadata."""

from dataclasses import dataclass
import numpy

from . import _native_int32, _validate_parent_tree


def _get_children(parents):
    nelts = parents.shape[0]

    # Pass 1, count number of children for each parent.
    n_immediate_children = numpy.full(nelts, 0, dtype=numpy.int32)
    for i in range(nelts):
        p = parents[i]
        assert p <= i, "Invalid kinematic tree ordering, parent index >= child index."
        if p == i:  # root
            continue
        n_immediate_children[p] += 1

    # Pass 2, mark the span of each parent in the child node list.
    child_list = numpy.full(nelts, -1, dtype=numpy.int32)
    child_list_span = numpy.empty((nelts, 2), dtype=numpy.int32)

    child_list_span[0, 0] = 0
    child_list_span[0, 1] = n_immediate_children[0]
    for i in range(1, nelts):
        child_list_span[i, 0] = child_list_span[i - 1, 1]
        child_list_span[i, 1] = child_list_span[i, 0] + n_immediate_children[i]

    # Pass 3, fill the child list for each parent.
    # As we do this, sum total # of descendents at each node, used to
    #   prioritize scan ordering in DFS
    n_descendents = numpy.ones(nelts, dtype=numpy.int32)
    for i in range(nelts - 1, 0, -1):
        p = parents[i]
        if p == i:  # root
            continue
        n_descendents[p] += n_descendents[i]
        child_list[child_list_span[p, 0] + n_immediate_children[p] - 1] = i
        n_immediate_children[p] -= 1

    return n_descendents, child_list_span, child_list


# CPU scan operation
# inputs:
#    parents - the "parents" array from the kinematic tree
#              assumes proper ordering: child index > parent index
#    roots - the initial root nodes.  The code will efficiently support
#            a graph with many disconnected components
# returns:
#    nodes - a 1D array with the node indices of each scan, concatenated
#    scanStarts - a 1D index array in 'nodes' where each individual scan begins
#    genStarts - an N x 2 array giving indices where each gen begins
#         genStarts[:,0] indexes nodes
#         genStarts[:,1] indexes scanStarts
def _get_scans(parents, roots):
    """Partitioning of a tree into linear subpaths.

    The tree is stored as a list of pointers to parent indices, with an
    arbitrary set of roots.  Each root implies a connected subtree.
    Each index i has a parent index (p_i) "higher" in the tree (p_i < i).
    The tree structure is fully defined by parent pointers, an index array
    (parent) of len(tree_size) where parent[i] == i for all root nodes.

    The source tree implies a per-node child list of length >= 0, defined by
    all nodes for which the given node is a parent. Nodes with no children are
    leaves.

    Scan paths cuts the tree into linear paths, where each non-leaf node (i)
    has a *single* child (c_i) "lower" in the tree (i < c_i). The path
    structure is fully defined by child pointers, an index array (subpath_child)
    of subpath_child[i] == c_i (non-leaves) or subpath_child[i] = -1 (leaves).

    The path partitioning implies "subpath roots", the set of nodes which are
    *not* the subpath child of their parent (subpath_child[parent[i]] != i).

    Each path in the tree is labeled with a depth: a path with depth
    i may depend on the values computed for atoms with depths 0..i-1.
    All of the paths of the same depth can be processed in a single
    kernel execution with segmented scan.
    """

    nelts = parents.shape[0]
    n_descendents, child_list_span, child_list = _get_children(parents)

    # scan storage - allocate upper bound for 4-connected graph
    # scan indices are emitted as a 1D array of nodes ('scans') with:
    #   scanStarts: indices in 'scans' where individual scans start
    #   genStarts: indices in 'scanStarts' where generations start
    nodes = numpy.full(4 * nelts, -1, dtype=numpy.int32)
    scanStarts = numpy.full(nelts, -1, dtype=numpy.int32)
    genStarts = numpy.full((nelts, 2), -1, dtype=numpy.int32)

    # curr idx in each array
    genidx, scanidx, nodeidx = 0, 0, 0

    # store the active pool we are expanding
    activeFront = numpy.full(4 * nelts, -1, dtype=numpy.int32)
    nActiveFront = roots.shape[0]
    activeFront[:nActiveFront] = roots

    # DFS traversal through forest-of-rooted-trees drawing generational scan
    # paths. Each pass through the DFS search follows a set of scan paths
    # beginning from the current set of roots, following nodes to the full
    # depth of the scan path. At each node the scan is extended through the
    # child with the most descendents and the node is added to the generation
    # n+1 roots, where it will root 0-or-more scans in the n+1 generation
    # passing through its additional children.
    #
    # In typical kinematic trees this will minimize the total number of scan
    # generations by tracing a long backbone path with many short side paths.
    marked = numpy.zeros(nelts, dtype=numpy.int32)
    marked[0] = 1
    while not numpy.all(marked):
        genStarts[genidx, :] = [nodeidx, scanidx]
        for i in range(nActiveFront):
            currRoot = activeFront[i]

            # Active front nodes are members of an existing scan passing through
            # to one child (or are gen 0) and may root 1-or-more scans through
            # any additional children.
            for j in range(child_list_span[currRoot, 0], child_list_span[currRoot, 1]):
                expandedNode = child_list[j]

                if marked[expandedNode] != 0:
                    # Child node has already been 'scanned' in a prev
                    # generation and is not the first child of another scan.
                    continue

                # add the node as the start of a new scan
                # -> numbering is w.r.t. generation (hence subtracting gen start)
                scanStarts[scanidx] = nodeidx - genStarts[genidx, 0]
                scanidx += 1

                # Set root as start of the scan.
                nodes[nodeidx] = currRoot
                nodeidx += 1

                while True:
                    # Extend scan path into selected child and mark it.
                    marked[expandedNode] = 1
                    nodes[nodeidx] = expandedNode
                    nodeidx += 1

                    if (
                        child_list_span[expandedNode, 0]
                        == child_list_span[expandedNode, 1]
                    ):
                        # At a leaf node, scan path terminates.
                        break
                    else:
                        # Extend path through the child with
                        #  greatest number of descendants.
                        nextExtension = child_list[child_list_span[expandedNode, 0]]
                        for k in range(
                            child_list_span[expandedNode, 0] + 1,
                            child_list_span[expandedNode, 1],
                        ):
                            candidate = child_list[k]
                            if n_descendents[candidate] > n_descendents[nextExtension]:
                                nextExtension = candidate

                        expandedNode = nextExtension

        # Mark any node scanned in this generation as a potential root.
        # Old roots and nodes with 1 child will be filtered during expansion.
        if genStarts[genidx, 1] < scanidx:
            lastgenScan0 = genStarts[genidx, 0]
            activeFront.fill(-1)
            nActiveFront = nodeidx - lastgenScan0
            activeFront[:nActiveFront] = nodes[lastgenScan0:nodeidx]

        # next generation
        genidx += 1

    # Pad genStarts by 1 to make downstream code cleaner
    genStarts[genidx, :] = [nodeidx, scanidx]
    genidx += 1

    return nodes[:nodeidx], scanStarts[:scanidx], genStarts[:genidx, :]


@dataclass(frozen=True)
class ScanPlan:
    nodes: numpy.ndarray
    scans: numpy.ndarray
    gens: numpy.ndarray


@dataclass(frozen=True)
class ScanOrdering:
    forward: ScanPlan
    backward: ScanPlan


def _reverse_plan(nodes, scans, gens):
    """Reverse generations and paths using the established derivative ordering."""
    nodes_r = numpy.ascontiguousarray(numpy.flipud(nodes))
    gens_r = numpy.zeros_like(gens)
    gens_r[1:, 0] = nodes.shape[0] - numpy.flipud(gens[:-1, 0])
    gens_r[1:, 1] = scans.shape[0] - numpy.flipud(gens[:-1, 1])
    scans_r = numpy.zeros_like(scans)
    for i in range(gens.shape[0] - 1):
        start, stop = gens_r[i, 1], gens_r[i + 1, 1]
        if start == stop:
            continue
        count = gens_r[i + 1, 0] - gens_r[i, 0]
        offsets = count - numpy.ascontiguousarray(
            numpy.flipud(scans[(scans.shape[0] - stop) : (scans.shape[0] - start)])
        )
        scans_r[start] = 0
        scans_r[start + 1 : stop] = offsets[:-1]
    return nodes_r, scans_r, gens_r


def _snapshot_plan(nodes, scans, gens):
    return ScanPlan(
        _native_int32(nodes, "nodes"),
        _native_int32(scans, "scans"),
        _native_int32(gens, "gens", ndim=2),
    )


def _freeze_generated_plan(arrays):
    # The planner creates int32 arrays from validated parents; only external
    # plans require the full numeric/shape/topology checks below.
    snapshots = tuple(numpy.array(value, copy=True) for value in arrays)
    for value in snapshots:
        value.flags.writeable = False
    return ScanPlan(*snapshots)


def _scan_ordering_for_ordered_parent(parent):
    """Internal owned arrays for adapters with a validated ordered forest."""
    if len(parent) <= 1:
        arrays = (
            numpy.empty(0, numpy.int32),
            numpy.empty(0, numpy.int32),
            numpy.zeros((1, 2), numpy.int32),
        )
    else:
        arrays = _get_scans(parent, numpy.array([0], dtype=numpy.int32))
    return ScanOrdering(ScanPlan(*arrays), ScanPlan(*_reverse_plan(*arrays)))


def validate_scan_plan(parent, nodes, scans, gens, *, reverse=False):
    """Snapshot a native plan after checking shape, topology and dependencies.

    Generations delimit node and scan ranges; scan offsets are relative to
    their generation's node range. Every tree edge must appear exactly once.
    Paths in one generation may share an already-built starting node, but
    may not depend on writes from another path in that same generation.
    Reverse plans follow the identical invariant after reversing their paths
    and generations. Empty forests and root-only forests need no scan edges.
    """
    parent = _native_int32(parent, "parent")
    _validate_parent_tree(parent)
    plan = _snapshot_plan(nodes, scans, gens)
    nodes, scans, gens = plan.nodes, plan.scans, plan.gens
    if gens.shape[1] != 2 or not len(gens):
        raise ValueError("gens must contain start/end rows of two offsets")
    if numpy.any(gens[0] != 0) or numpy.any(gens[-1] != [len(nodes), len(scans)]):
        raise ValueError("gens endpoints must span all nodes and scans")
    if numpy.any(numpy.diff(gens, axis=0) < 0):
        raise ValueError("gens offsets must be monotonic")
    if numpy.any(nodes < 0) or numpy.any(nodes >= len(parent)):
        raise ValueError("scan node index is outside the forest")
    for (nb, sb), (ne, se) in zip(gens[:-1], gens[1:]):
        starts = scans[sb:se]
        if nb == ne and sb == se:
            continue
        if not len(starts) or starts[0] != 0:
            raise ValueError(
                "each nonempty generation must begin with scan offset zero"
            )
        if numpy.any(numpy.diff(starts) <= 0) or numpy.any(starts >= ne - nb):
            raise ValueError("scan offsets must define nonempty disjoint paths")
    if reverse:
        nodes, scans, gens = _reverse_plan(nodes, scans, gens)
    visited = numpy.zeros(len(parent), dtype=bool)
    if len(parent):
        visited[0] = True
    ready = visited.copy()
    for (nb, sb), (ne, se) in zip(gens[:-1], gens[1:]):
        for index in range(sb, se):
            begin = nb + scans[index]
            end = nb + scans[index + 1] if index + 1 < se else ne
            path = nodes[begin:end]
            if not ready[path[0]]:
                raise ValueError("scan path depends on an unfinished generation")
            if numpy.any(parent[path[1:]] != path[:-1]):
                raise ValueError("scan path does not follow the forest parent edges")
            if numpy.any(visited[path[1:]]):
                raise ValueError("a forest edge is visited more than once")
            visited[path[1:]] = True
        ready[:] = visited
    if not numpy.all(visited):
        raise ValueError("scan plan does not cover every forest edge")
    return plan


def prepare_scan_ordering(parent):
    """Create native forward/reverse plans with the shared mature tree planner.

    Already topologically ordered forests preserve their existing exact scan
    order. Other valid storage orders use a stable breadth-first remapping
    around that same planner; returned node indices address the original
    forest. All arrays are owned, read-only int32 NumPy snapshots.
    """
    parent = _native_int32(parent, "parent")
    is_ordered = _validate_parent_tree(parent)
    n = len(parent)
    order = None
    ordered_parent = parent
    if not is_ordered:
        children = [[] for _ in range(n)]
        for node in range(1, n):
            children[parent[node]].append(node)
        queue = [0]
        for node in queue:
            queue.extend(children[node])
        order = numpy.asarray(queue, dtype=numpy.int32)
        inverse = numpy.empty(n, dtype=numpy.int32)
        inverse[order] = numpy.arange(n, dtype=numpy.int32)
        ordered_parent = inverse[parent[order]]
    plans = _scan_ordering_for_ordered_parent(ordered_parent)

    def snapshot(plan):
        nodes = plan.nodes if order is None else order[plan.nodes]
        return _freeze_generated_plan((nodes, plan.scans, plan.gens))

    return ScanOrdering(snapshot(plans.forward), snapshot(plans.backward))


def scan_ordering_from_forward(parent, nodes, scans, gens):
    """Validate an existing forward schedule and derive its reverse schedule.

    The original path and generation order is preserved. This accepts plans
    assembled from residue-local conformer forests without rebuilding a tree
    traversal with a different ordering.
    """
    forward = validate_scan_plan(parent, nodes, scans, gens)
    backward = _freeze_generated_plan(
        _reverse_plan(forward.nodes, forward.scans, forward.gens)
    )
    return ScanOrdering(forward, backward)
