import numpy as np


def constraint_list_by_fixed_per_cluster_choice(assignments, n=1, return_must_link=False, selected_idxs=None):
    if n == 0 and selected_idxs is None:
        if return_must_link:
            return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)
        else:
            return np.empty((0, 2), dtype=int)

    uniqs = np.unique(assignments)

    if selected_idxs is None:
        selected_idxs = np.array([np.random.choice(np.where(assignments == u)[0], size=n, replace=False) for u in uniqs]).flatten()
    else:
        selected_idxs = np.array(selected_idxs)
        # Set n to first count of selected indices, note that this only supports selecting equal amount per class
        n = np.unique(assignments[selected_idxs], return_counts=True)[1][0]

    # Check selected okay
    assert len(selected_idxs.shape) == 1
    check_counts = {u: 0 for u in uniqs}
    for i in selected_idxs:
        label = assignments[i]
        check_counts[label] = check_counts[label] + 1

    assert all([c == n for c in check_counts.values()])

    result = []
    must_link_result = []

    expected_count = len(uniqs) * n * (len(uniqs) - 1) * n / 2

    for i in selected_idxs:
        for j in selected_idxs:
            if i >= j:
                continue
            if assignments[i] == assignments[j]:
                if n == 1:
                    raise ValueError
                # Can only occur for multiple
                continue
            result.append((i, j))

    result = np.array(result, dtype=int)

    # Ensure that the expected number of connections was generated
    assert len(result) == expected_count

    # Ensure that all constraints generated were unique
    assert len(result) == len(np.unique(result, axis=0))

    # Ensure that all constraints are actually between different classes
    assert np.all(assignments[result[:, 0]] != assignments[result[:, 1]])

    expected_ml_count = len(uniqs) * n * (n - 1) / 2

    # Add must links
    for i in selected_idxs:
        for j in selected_idxs:
            if i < j and assignments[i] == assignments[j]:
                must_link_result.append((i, j))

    assert expected_ml_count == len(must_link_result)

    if expected_ml_count == 0:
        must_link_result = np.empty((0, 2), dtype=int)
    else:
        must_link_result = np.array(must_link_result, dtype=int)

    if return_must_link:
        return result, must_link_result
    else:
        return result


def generate_constraints_fixed_count(y, n):
    cls = constraint_list_by_fixed_per_cluster_choice(y, n=1).tolist()

    if len(cls) > n:
        raise ValueError("Minimum constraints for this data set is {}, requested was {}".format(len(cls), n))

    mls = []

    while len(cls) + len(mls) < n:
        while True:
            i, j = np.random.randint(0, len(y), size=2)
            if i != j:
                break

        if y[i] == y[j]:
            mls.append([i, j])
        else:
            cls.append([i, j])

    # Make np array and ensure valid 2D shapes even if empty
    cls = np.array(cls, dtype=int).reshape((-1, 2))
    mls = np.array(mls, dtype=int).reshape((-1, 2))
    return cls, mls


def generate_constraint_variations(y):
    minimum_constraint_count = len(constraint_list_by_fixed_per_cluster_choice(y, n=1))
    result = []

    for count_f in np.linspace(minimum_constraint_count, len(y) * 2, num=20):
        result.append(generate_constraints_fixed_count(y, int(count_f)))

    return result
