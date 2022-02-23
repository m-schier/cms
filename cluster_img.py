def cms_img(x, cl, ml, use_cuda=True):
    from CMS import CMS, AutoLinearPolicy
    from CMS.Constraints import transitive_closure_constraints

    iterations = 80

    cl = transitive_closure_constraints(cl, ml, len(x))

    pol = AutoLinearPolicy(x, iterations)
    cms = CMS(pol, max_iterations=iterations, blurring=False, kernel=.2, use_cuda=use_cuda, label_merge_k=.99)

    return cms.fit_predict(x, cl)


def main():
    import sys
    import numpy as np
    from Util.Sampling import generate_constraints_fixed_count
    from Util.CsvWriter import CsvWriter
    from Util.Datasets import load_embeddings
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from argparse import ArgumentParser

    data_choices = {
        'gtsrb': lambda: load_embeddings(which='gtsrb', sample='default'),
        'mnist': lambda: load_embeddings(which='mnist', sample='default'),
        'fashion-mnist': lambda: load_embeddings(which='fashion-mnist', sample='default'),
    }

    parser = ArgumentParser()
    parser.add_argument('--data', choices=data_choices.keys(), required=True)
    parser.add_argument('--repeats', metavar='N', type=int, default=100)
    parser.add_argument('--constraint-factor', metavar='F', type=float, default=1.)
    parser.add_argument('--nocuda', action="store_false", dest='use_cuda')

    args = parser.parse_args()
    use_cuda = args.use_cuda

    data_func = data_choices[args.data]

    print("Starting cluster_img.py, args = {}".format(args), file=sys.stderr, flush=True)

    file_name = 'cms-img-{}.csv'.format(args.data)

    with CsvWriter(file_name) as writer:
        for run in range(args.repeats):
            print('Run {}/{}'.format(run+1, args.repeats))

            data = data_func().normalized_linear()
            x, y = data.train
            n_c = int(len(y) * args.constraint_factor)
            cl, ml = generate_constraints_fixed_count(y, n_c)

            y_pred = cms_img(np.copy(x), np.copy(cl), np.copy(ml), use_cuda=use_cuda)

            nmi = normalized_mutual_info_score(y, y_pred)
            ari = adjusted_rand_score(y, y_pred)

            writer.write_row(algo='cms', data=data.name, ari=ari, nmi=nmi, n_c=n_c)


if __name__ == '__main__':
    main()
