# Multiple-PPI-Transformer
Multiple PPI Transformer (MPIT) is based on Graph Transformer Networks (GTNs) to discern specific cancer driver genes. MPIT adeptly amalgamates data from diverse PPI and multi-omics sources through the synergy of embedding alignment and fusion.

![image](https://github.com/NBStarry/Multi-PPI-Transformer/blob/main/img/Framework.png)

## Conda Environment
We recommend using conda to configure the code runtime environment:
```
conda create -n mpit python=3.8.12
conda install pytorch==1.12.1 -c pytorch
pip install torch_geometric==2.3.0 transformers wandb
```
> Pytorch versions that allow torch_geometric >= 2.3.0 are okay.

## Installation
We recommend getting MPIT using Git from our Github repository through the following command:

```
git clone https://github.com/NBStarry/Multiple-PPI-Transformer.git
```

To verify a successful installation, just run:
```
unzip data/Lung_Cancer_Matrix/A549.zip
python main.py -g 0 -t -d # You need to select an idle GPU using "-g".
```
### System and Computing resources

| Item       | Details          |
| ---------- | ---------------- |
| System     | Ubuntu 20.04.1 LTS |
| RAM Memory | 256G               |
| GPU Memory | NVIDIA GeForce RTX, 24G     |
| Train Time | ~ 5h             |
```note
The above table reports our computing details during MPIT development and IS NOT our computing requirement.

If your computer does not satisfy the above, you may try to lower down the memory used during model training by reduce the sampling parameters, the batch size or so on. 
```

## Questions and Code Issues
If you are having problems with our work, please use the [Github issue page](https://github.com/NBStarry/Multi-PPI-Transformer/issues).
