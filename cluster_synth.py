import numpy as np


def cms_synth(x, cl, ml, use_cuda=True):
    from CMS import CMS, AutoLinearPolicy, constraint_list_from_constraints, transitive_closure_constraints

    cl = transitive_closure_constraints(cl, ml, len(x))
    cl = constraint_list_from_constraints(cl)
    iterations = 100
    pol = AutoLinearPolicy(x, iterations)
    cms = CMS(pol, max_iterations=iterations, blurring=False, kernel=.2, use_cuda=use_cuda, label_merge_b=.0, label_merge_k=.995)
    return cms.fit_predict(x, cl)


def get_datasets():
    from Util.Datasets import load_text_data, load_moons, load_s4
    import os

    res_path = os.path.join(os.path.dirname(__file__), 'data')

    return {
        'aggregation': lambda: load_text_data(os.path.join(res_path, 'Aggregation.txt'), 'aggregation'),
        'moons': lambda: load_moons(500),
        'jain': lambda: load_text_data(os.path.join(res_path, 'jain.txt'), 'jain'),
        's4': load_s4,
    }


def main():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from Util.Sampling import generate_constraints_fixed_count
    from Util.CsvWriter import CsvWriter
    from argparse import ArgumentParser

    datas = get_datasets()

    parser = ArgumentParser()
    parser.add_argument('--data', choices=datas.keys(), default='moons')
    parser.add_argument('--repeats', metavar='N', type=int, default=10)
    parser.add_argument('--constraint-factor', metavar='F', type=float, default=1.)
    parser.add_argument('--nocuda', action="store_false", dest='use_cuda')

    args = parser.parse_args()

    file_name = 'cms-synth-{}.csv'.format(args.data)
    use_cuda = args.use_cuda

    with CsvWriter(file_name) as writer:
        for run in range(args.repeats):
            print('Run {}/{}'.format(run + 1, args.repeats))
            
            try:
                x, y = datas[args.data]().normalized_linear().train
            except Exception as ex:
                raise RuntimeError("Failed to load data set {}, ensure that you have correctly downloaded the synthetic data sets by running 'download_synth.sh'".format(args.data)) from ex
            
            n_c = int(len(y) * args.constraint_factor)
            cl, ml = generate_constraints_fixed_count(y, n_c)

            y_pred = cms_synth(np.copy(x), np.copy(cl), np.copy(ml), use_cuda=use_cuda)

            ari = adjusted_rand_score(y, y_pred)
            nmi = normalized_mutual_info_score(y, y_pred)
            writer.write_row(algo='cms', data=args.data, ari=ari, nmi=nmi, n_c=n_c)


if __name__ == '__main__':
    main()
