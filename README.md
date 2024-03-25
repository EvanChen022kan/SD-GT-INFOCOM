# SD-GT

## Requirements
- Python 3.8.10
- PyTorch 2.0.0+cu117
- Numpy 1.23.2

I need to run the following experiments:

## CIFAR-100 setting1
(Here I call gpu 1 since this is the gpu id on Chris's server, I'm not sure how IBM's computing resource works. If there's anything that I need to change feel free to tell me to change it.)
```bash
>python3 mainNN2.py  --gpu_id 1 \
                    --batch 512 \
                    --dataset CIFAR100 \
                    --lr 3e-3 \
                    --loss log --random --model TOMCNN \
                    --c_rounds 25000 --setting1 \
                    --name setting1

```
However I'm not sure if this one is going to look stable or not when the number of agents increases, so it's best if you can also run the version with a smaller step size.
```bash
>python3 mainNN2.py  --gpu_id 1 \
                    --batch 512 \
                    --dataset CIFAR100 \
                    --lr 1e-3 \
                    --loss log --random --model TOMCNN \
                    --c_rounds 25000 --setting1 \
                    --name setting1_smallerstepsize

```

## CIFAR-100 setting2

```bash
>python3 mainNN2.py  --gpu_id 1 \
                    --batch 512 \
                    --dataset CIFAR100 \
                    --lr 3e-3 \
                    --loss log --random --model TOMCNN \
                    --c_rounds 25000 --setting2 \
                    --name setting1

```
Still, the smaller stepsize version
```bash
>python3 mainNN2.py  --gpu_id 1 \
                    --batch 512 \
                    --dataset CIFAR100 \
                    --lr 1e-3 \
                    --loss log --random --model TOMCNN \
                    --c_rounds 25000 --setting2 \
                    --name setting2_smallerstepsize

```

The results should all be dumped into a folder called 'results', oranized by the date that the experiments finish running. When the code it finish running, just send me the folder (or even just the .plt files in it) and I can draw the plots.
