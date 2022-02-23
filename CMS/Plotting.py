def plot_clustering(points, assignments, centers=None, cl=None, ml=None, ax=None, labels=None,
                    alpha=1., legend=True, center_size=1., center_marker='o', point_size=1., point_marker='x',
                    palette=None, constraints_on_centers=True, pfx=True, equal_axis_scale=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    if pfx:
        import matplotlib.patheffects as path_effects
        path_effects = [path_effects.withStroke(linewidth=2, foreground='black')]
    else:
        path_effects = None

    if assignments is None:
        assignments = np.zeros(len(points), dtype=int)

    assert len(points) == len(assignments)

    if len(points.shape) != 2 or points.shape[-1] != 2:
        raise ValueError("Invalid points shape: {}".format(points.shape))

    def plot_wrap(fn, pts, *args, **kwargs):
        # Ensure always a 2D array even if just one point
        if len(pts.shape) == 1:
            pts = np.array([pts])

        return fn(pts[:, 0], pts[:, 1], *args, **kwargs)

    if centers is not None:
        n_centers = len(centers)
        assert np.max(assignments) < n_centers
    else:
        n_centers = np.max(assignments) + 1

    if palette is None:
        colors = plt.get_cmap('tab10').colors
        palette = [colors[i % 10] for i in range(n_centers)]

    unique = np.unique(assignments)

    if ax is None:
        ax = plt.axes()

    for u in unique:
        label = labels[u] if labels is not None else None

        p_color = palette[u]
        p_points = points[assignments == u]
        plot_wrap(ax.scatter, p_points, marker=point_marker, s=point_size * rcParams['lines.markersize'] ** 2,
                  color=p_color, label=label, alpha=alpha, path_effects=path_effects)

        if centers is not None:
            plot_wrap(ax.scatter, centers[assignments == u, :], marker=center_marker, color=palette[u],
                      s=center_size * rcParams['lines.markersize'] ** 2, alpha=alpha, path_effects=path_effects)

    cl_anchors = centers if constraints_on_centers else points

    if ml is not None:
        for i, j in ml:
            plot_wrap(ax.plot, np.stack([cl_anchors[i, :], cl_anchors[j, :]], axis=0), '--', color='tab:blue',
                      zorder=0.5, alpha=alpha, path_effects=path_effects)
    if cl is not None:
        for i, j in cl:
            plot_wrap(ax.plot, np.stack([cl_anchors[i, :], cl_anchors[j, :]], axis=0), '--', color='tab:red',
                      zorder=0.5, alpha=alpha, path_effects=path_effects)

    if labels is not None and legend:
        ax.legend()

    if equal_axis_scale:
        # Ensure X and Y axis have equal scale in visualization
        plt.gca().set_aspect('equal', adjustable='box')
    return ax
