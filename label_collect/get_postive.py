import argparse
import os
import pandas as pd

def arg_parse():
    parser = argparse.ArgumentParser(description='convert KEGG info to list')
    parser.add_argument('-d', '--dir', dest='dir')
    return parser.parse_args()


def main():
    args = arg_parse()

    ncg = pd.read_csv(os.path.join(args.dir, 'NCG.txt'), header=None)[0].tolist()
    cgc = pd.read_csv(os.path.join(args.dir, 'CGC.csv'), header=None)[0].tolist()
    disgenet = pd.read_csv(os.path.join(args.dir, 'DisGeNET.csv'), header=None)[0].tolist()
    driverdb = pd.read_csv(os.path.join(args.dir, 'DriverDBv4.csv'), header=None)[0].tolist()
    # digsee = pd.read_csv(os.path.join(args.dir, 'DigSEE.txt'), sep='\t')
    # digsee = digsee[digsee['EVIDENCE SENTENCE SCORE'] >= 0.95]
    # genes = digsee['RELATED GENES'].tolist()
    gene_list = []
    # for gene in genes:
    #     gene = gene.strip().split(' ')
    #     gene_list.extend(gene)

    gene_list.extend(ncg)
    gene_list.extend(cgc)
    gene_list.extend(disgenet)
    gene_list.extend(driverdb)
    gene_list = list(set(gene_list))

    with open(os.path.join(args.dir, 'Positive.txt'), 'w') as f:
        for gene in gene_list:
            f.write(f'{gene}\n')

if __name__ == '__main__':
    main()
