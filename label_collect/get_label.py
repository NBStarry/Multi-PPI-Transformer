import argparse
import sys
import os
import pandas as pd

sys.path.append('/data/data/hanzebei/MPIT/')

import utils

def arg_parse():
    parser = argparse.ArgumentParser(description='convert KEGG info to list')
    parser.add_argument('-c', '--cell-line', dest='cell_line')
    parser.add_argument('-t', "--test", dest='test', action="store_true")
    return parser.parse_args()


def get_label_dir(cell_line):
    return os.path.join('./label_collect', cell_line)

def get_data_dir(cel_line):
    if 'MCF7' in cel_line:
        return 'data/Breast_Cancer_Matrix'
    if 'K562' in cel_line:
        return 'data/Leukemia_Matrix'
    if 'A549' in cel_line:
        return 'data/Lung_Cancer_Matrix'
    return f'data/{cel_line}_Matrix'

def main():
    args = arg_parse()

    all_nodes = utils.get_gene_list()
    all_nodes['label'] = 0
    label_dir = get_label_dir(args.cell_line)
    pos_dir = os.path.join(label_dir, 'positive/Positive.txt') if not args.test else os.path.join(label_dir, 'positive/Positive_test.txt')
    pos = pd.read_csv(pos_dir, header=None)
    neg = pd.read_csv(os.path.join(label_dir, 'negative/NegativeM.txt'), header=None)
    positive = all_nodes['gene_name'].isin(pos[0]).astype(int)
    negative = all_nodes['gene_name'].isin(neg[0]).map({True: -1, False: 0})
    all_nodes['label'] = positive + negative
    print(f"Positives: {(all_nodes['label'] == 1).sum()} Negatives: {(all_nodes['label'] == -1).sum()}")
    if args.test:
        all_nodes.to_csv(os.path.join(get_data_dir(args.cell_line), f'{args.cell_line}-test-Label.txt'), index=False, sep='\t')
    else:
        all_nodes.to_csv(os.path.join(get_data_dir(args.cell_line), f'{args.cell_line}-Label.txt'), index=False, sep='\t')


if __name__ == '__main__':
    main()