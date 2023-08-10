import numpy as np
import pandas as pd
import os
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='convert KEGG info to list')
    parser.add_argument('-d', '--dir', dest='dir')
    return parser.parse_args()


def deleteGene(genelist, file):
    noneg_genes = pd.read_csv(file, header=None)
    noneg_genes = noneg_genes[0].tolist()
    neg_genes = []
    for gene in genelist:
        if gene in noneg_genes:
            continue
        else:
            neg_genes.append(gene)
    neg_genes.sort()
    return neg_genes


def main():
    args = arg_parse()
    script_path = os.path.abspath(__file__)
    script_path = os.path.dirname(script_path)

    data = pd.read_csv(os.path.join(script_path, 'gencode.v34.codingGene.txt'), sep='\t', header=None)
    coding_genes = data[4]
    coding_genes = coding_genes.tolist()
    s1 = deleteGene(coding_genes, os.path.join(args.dir, '1_Positive.txt'))
    s2 = deleteGene(s1, os.path.join(args.dir, '2_KEGG.txt'))
    s3 = deleteGene(s2, os.path.join(args.dir, '3_mim.txt'))
    s4 = deleteGene(s3, os.path.join(args.dir, '4_GESA.txt'))
    f = open(os.path.join(args.dir, "NegativeM.txt"), 'w')
    for gene in s4:
        f.write(gene)
        f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
