import argparse
import os
import numpy as np
from utils import load_ndarray, save_ndarray, TimedBlock
import sklearn.metrics

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--feature', default='', help='path to feature matrix')
parser.add_argument('--mi', '--mutual-information', default='', help='output path for mutual information if specified')
parser.add_argument('-o', '--output', default='-',
                    help='top reranked features greedily selected wrt mutual information [default: \'-\' (stdout)]')
parser.add_argument('-k', '--top-k', type=int, default=-1, help='top k features [default: -1 (all)]')
parser.add_argument('-d', '--delimiter', default=',', help='only used when output to stdout [default: \',\']')
parser.add_argument('-q', '--num-bins', type=int, default=20, help='quantization levels [default: 20]')
parser.add_argument('-n', '--num-samples', type=int, default=-1, help='number of samples to use [default: -1 (all)]')
parser.add_argument('-z', '--zero-based', action='store_true', help='output with zero-based indexing if enabled')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
parser.add_argument('-p', '--pre-process', default='none', choices=('none', 'abs', 'square'),
                    help='feature pre-processing [default: \'none\']')
parser.add_argument('-t', '--score-type', default='mutual_info_score',
                    choices=(' mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score'),
                    help='mutual information type [default: \'standard\']')
parser.add_argument('-s', '--start-with', default='smallest', choices=('random', 'smallest'),
                    help='starting strategy [default: \'smallest\']')
parser.add_argument('--use-percentile', action='store_true', help='discretize using percentile if specified')
args = parser.parse_args()


def main():
    global args
    if args.verbose:
        print(args)
    if args.feature != '':
        feat = load_ndarray(args.feature)
        num_samples, num_feats = feat.shape
        if args.num_samples != -1 and num_samples > args.num_samples:
            feat = feat[np.random.choice(np.arange(num_samples), args.num_samples, replace=False)]
        if args.pre_process == 'abs':
            feat = np.abs(feat)
        elif args.pre_process == 'square':
            feat = np.power(feat, 2)

        # discretization
        with TimedBlock('* Discretizing features into {} bins'.format(args.num_bins), verbose=args.verbose):
            if args.use_percentile:
                feat = np.argsort(np.argsort(feat, axis=0), axis=0)
            for i in range(num_feats):
                feat[:, i] = np.digitize(feat[:, i], bins=np.linspace(feat[:, i].min(), feat[:, i].max(), args.num_bins+1))
            feat = np.maximum(np.minimum(feat, args.num_bins), 1)

        # compute mutual information for feature pairs
        with TimedBlock('* Computing mutual information', verbose=args.verbose):
            mi = np.zeros((num_feats, num_feats), dtype=np.float32)
            mi_fn = sklearn.metrics.__dict__[args.score_type]
            n_calls = int(0.5 * (num_feats ** 2 + num_feats))
            cnt = 0
            if args.verbose:
                print('')
            for i in range(num_feats):
                for j in range(i, num_feats):
                    mi[i, j] = mi_fn(feat[:, i], feat[:, j])
                    cnt += 1
                    if cnt % 1000 == 0 and args.verbose:
                        print('.', end='', flush=True)
                    if cnt % 80000 == 0 and args.verbose:
                        print(' {}/{}'.format(cnt, n_calls), flush=True)
            mi += np.triu(mi, 1).T  # fill lower triangle part to make it symmetrical
            if args.mi != '':
                try:
                    os.makedirs(os.path.split(args.mi)[0])
                except OSError:
                    pass
                save_ndarray(args.mi, mi, var_name=args.score_type)
    else:
        if args.mi == '':
            raise ValueError('either feature or pre-computed mutual-information should be provided')
        mi = load_ndarray(args.mi)

    # find top k features
    num_feats = mi.shape[0]
    top_k = num_feats if (args.top_k > num_feats or args.top_k == -1) else args.top_k
    with TimedBlock('* Choosing top {} features'.format(top_k), verbose=args.verbose):
        chosen_mask = np.zeros(num_feats, dtype=np.bool)
        if args.start_with == 'smallest':
            idx_to_choose = np.argmin(mi) % num_feats
        elif args.start_with == 'random':
            idx_to_choose = np.random.randint(num_feats)
        else:
            raise ValueError('Unknown value for start_with: {}'.format(args.start_with))
        chosen_mask[idx_to_choose] = True
        ranking = [idx_to_choose]
        for i in range(1, top_k):
            chosen = chosen_mask.nonzero()[0]
            unchosen = (1 - chosen_mask).nonzero()[0]
            idx_in_unchosen = np.argmin(mi[chosen[:, np.newaxis], unchosen].max(axis=0))
            idx_to_choose = (np.cumsum(1 - chosen_mask) - idx_in_unchosen > 0).nonzero()[0][0]
            chosen_mask[idx_to_choose] = True
            ranking.append(idx_to_choose)
        ranking = np.array(ranking, dtype=np.int32)
        if not args.zero_based:
            ranking += 1
        if args.output == '-':
            print(args.delimiter.join([str(v) for v in ranking]))
        else:
            try:
                os.makedirs(os.path.split(args.output)[0])
            except OSError:
                pass
            save_ndarray(args.output, np.array(ranking), var_name='feature')


if __name__ == '__main__':
    main()
