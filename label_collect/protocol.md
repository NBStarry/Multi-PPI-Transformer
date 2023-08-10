## Positive
1. [NCG](http://network-cancer-genes.org/), download certain type tumor canonical cancer drivers.
2. [COSMIC CGC](https://cancer.sanger.ac.uk/census), search for lung adenocarcinoma and download the results.
3. [DigSee](http://gcancer.org/digsee), search for lung adenocarcinoma, select 'Gene Expression' and 'DNA Methylation' and download results with a confidence above 0.95.
4. If DigSee is offline, use [DisGeNET](https://www.disgenet.org/search) and [DriverDB](http://driverdb.tms.cmu.edu.tw/) or other cancer gene database instead.

## Negative
1. Use gencode.v34.codingGene.txt as a whole.
2. Exclude positive items.
3. [KEGG](https://www.genome.jp/kegg/) non-small cell lung cancer, KEGG ORTHOLOGY.
4. Remove all [OMIM](https://www.omim.org/static/omim/data/mim2gene.txt).
5. [MSigdb](https://www.gsea-msigdb.org/gsea/msigdb/human/search.jsp), search for 'lung AND adenocarcinoma'.