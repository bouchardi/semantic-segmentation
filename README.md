# eai-segmentation-models


```
docker build -t semseg .
nvidia-docker run -it --rm -v /mnt/home/isabelle/code/semantic-segmentation:/project -v /mnt/datasets/public/issam/VOCdevkit:/datasets  -p 8888:8888 semseg
```
