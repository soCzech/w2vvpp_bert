# W2VV++ BERT

### Docker image
```docker build -t w2vvpp_bert .```

```
docker run -it
           --rm
           --gpus 1
           -v /path/to/git/dir:/workspace
           -v /path/to/iacc.3:/iacc.3
           w2vvpp_bert bash
```

Note: [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) is required to utilize GPU.

### Run evaluation on TRECVid AVS tasks
- Download `vision_ds` and `text_ds` data from [github.com/li-xirong/w2vvpp](https://github.com/li-xirong/w2vvpp/#data).
- Download word2vec pretraind embeddings (`w2v_weights`) from [lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz](http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz).
```
python evaluate.py
    --model_weights checkpoint.pth.tar
    --output_file path/to/output_file.txt
    --vision_ds /iacc.3/FeatureData/mean_resnext101_resnet152
    --text_ds /iacc.3/TextData/tv?.avs.txt
    --w2v_weights vec500flickr30m
    --bow_vocab=data/bow_nsw_5.txt
``` 
- Run evaluation script from [github.com/li-xirong/w2vvpp/tree/master/tv-avs-eval](https://github.com/li-xirong/w2vvpp/tree/master/tv-avs-eval) on the output file.
