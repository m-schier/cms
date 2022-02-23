import numpy as np


def constraint_list_from_constraints(constraints):
    constraints = np.array(constraints, copy=False)
    # Accept both quadratic constraints matrices as well as Nx2 array containing all symetric links
    if len(constraints.shape) == 2 and constraints.shape[1] == 2 and np.issubdtype(constraints.dtype, np.integer):
        pass
    else:
        raise ValueError("Constraints not understood, shape={}, dtype={}".format(constraints.shape, constraints.dtype))

    # Ensure symmetric
    cl = np.concatenate([constraints, constraints[:, ::-1]])
    return np.unique(cl, axis=0)


# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def preprocess_constraints(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    assert np.all(np.logical_and(ml >= 0, ml < n))
    assert np.all(np.logical_and(cl >= 0, cl < n))

    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # Run DFS from each node to get all the graph's components
    # and add an edge for each pair of nodes in the component (create a complete graph)
    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    visited = [False] * n
    neighborhoods = []
    for i in range(n):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in cl:
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise ValueError('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods


def transitive_closure_constraints(cl_constraints, ml_constraints, n):
    ml_graph, cl_graph, _ = preprocess_constraints(ml_constraints, cl_constraints, n)

    result = []

    for k, vs in cl_graph.items():
        for v in vs:
            result.append((k, v))

    return np.array(result, dtype=np.int32)
