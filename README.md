# semantic-segmentation

Implementation of semantic segmentation models. The model currently contains:

### Models
- FCN8s

### Datasets
- camvid
- PascalVOC2012


## Development

You can use the Dockerfile to develop:
```
docker build -t semseg .
```

To run a `jupyter notebook`:
```
nvidia-docker run -it --rm -v /<path_to_your_repo>/semantic-segmentation:/project -v /mnt/datasets/public/issam/VOCdevkit:/datasets/pascal  -v /mnt/datasets/public/segmentation/camvid:/datasets/camvid -p 8888:8888 semseg
```

To enter the container in interactive mode:
```
nvidia-docker run -it --rm -v /<path_to_your_repo>/semantic-segmentation:/project -v /mnt/datasets/public/issam/VOCdevkit:/datasets/pascal  -v /mnt/datasets/public/segmentation/camvid:/datasets/camvid semseg bash
```

From inside the container, you can run `train.py` script to train your model:
```
python3.6 train.py -d pascal -o sdg -lr 0.00001 -bs 8 --n-epochs 50
```

* Note that you can change `nvidia-docker` for `docker` if you don't have access to GPUs. The model has only been tested on GPU.
