# Team Name: Buffalo^3     
- Vincent Luczkow           
- Jonathan Maloney-Lebensold
- Yi Feng Yan               

# Usage
General usage:

    python comparisons.py run_denmo $dset $dilations $erosions $epochs
    python comparisions.py run_baseline $model $dset $width $epochs
    
Run `scripts/training2csv.py` to convert Tensorflow logs to result CSVs.

## Getting tensorboard working
* `pip install tensorboard`
* `tensorboard --logdir training-logs/`


## Examples:
```
python comparisons.py run_denmo mnist --epochs=400 --dilations=5 --erosions=5
python comparisons.py run_denmo fashion_mnist --epochs=300 --dilations=400 --erosions=400
python comparisons.py run_denmo cifar10 --epochs=150 --dilations=100 --erosions=100
```

```
python comparisons.py run_denmo cifar10 --epochs=150 --dilations=100 --erosions=100
python comparisons.py run_baseline tanh cifar10 --epochs=150 --h-layers=200
python comparisons.py run_baseline relu cifar10 --epochs=150 --h-layers=200
python comparisons.py run_baseline maxout cifar10 --epochs=150 --h-layers=200
```

# Usage with Docker

### Start tensorboard:
```
docker-compose up
```

### Open bash
```
docker-compose run app bash
```