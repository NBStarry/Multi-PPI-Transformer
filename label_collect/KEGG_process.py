import os
import pandas as pd
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='convert KEGG info to list')
    parser.add_argument('-d', '--dir', dest='dir')
    return parser.parse_args()

def main():
    args = arg_parse()
    
    fr = open(os.path.join(args.dir, "KEGG.txt"))
    fw = open(os.path.join(args.dir, "2_KEGG.txt"),'w')
    KEGGlist = []
    line = fr.readline()
    while line:
        line = line.split("               ")[1].split(";")[0].split(",")[0].split("_")
        i = 0
        for gene in line:
            if i != 0:
                line[i] = line[0][:(len(line[0])-1)]+line[i]
                #print(line[i])
            KEGGlist.append(line[i])
            i += 1
        line = fr.readline()
    fr.close()
    KEGGlist = list(set(KEGGlist))
    KEGGlist.sort()
    for gene in KEGGlist:
        fw.write(gene)
        fw.write('\n')
    fw.close()

if __name__ == '__main__':
    main()