# SD-GT
Arxiv: ["Taming Subnet-Drift in D2D-Enabled Fog Learning: A Hierarchical Gradient Tracking Approach"](https://arxiv.org/abs/2312.04728) (INFOCOM 2024)
## Requirements
- Python 3.8.10
- PyTorch 2.0.0+cu117
- Numpy 1.23.2

## CIFAR-100 setting1

```bash
>python3 mainNN2.py  --gpu_id 1 \
                    --batch 512 \
                    --dataset CIFAR100 \
                    --lr 3e-3 \
                    --K_val 3 \
                    --loss log --random --model TOMCNN \
                    --c_rounds 35000 --setting1 \
                    --name setting1

```


## CIFAR-100 setting2

```bash
>python3 mainNN2.py  --gpu_id 1 \
                    --batch 512 \
                    --dataset CIFAR100 \
                    --lr 3e-3 \
                    --K_val 3 \
                    --loss log --random --model TOMCNN \
                    --c_rounds 35000 --setting2 \
                    --name setting2

```

The results should all be dumped into a folder called 'results', oranized by the date that the experiments finish running. When the code it finish running, just send me the folder (or even just the .plt files in it) and I can draw the plots.


## Citation
If you find the repository or the paper useful, please cite the following paper
```bash
@article{chen2023taming,
  title={Taming Subnet-Drift in D2D-Enabled Fog Learning: A Hierarchical Gradient Tracking Approach},
  author={Chen, Evan and Wang, Shiqiang and Brinton, Christopher G},
  journal={arXiv preprint arXiv:2312.04728},
  year={2023}
}
```
