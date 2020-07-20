# W2VV++ BERT

This repository extends [W2VV++ Fully Deep Learning for Ad-hoc Video Search](https://dl.acm.org/doi/abs/10.1145/3343031.3350906) by adding BERT model that is fine-tuned. The fine-tuning is crucial to achieve better results than the original paper on some tasks.

Note that this repository uses in some way the following works, cite them if you are using this work: 

- Li, X., Xu, C., Yang, G., Chen, Z., & Dong, J.
  (2019, October).
  [W2VV++ Fully Deep Learning for Ad-hoc Video Search](https://dl.acm.org/doi/abs/10.1145/3343031.3350906).
  In *Proceedings of the 27th ACM International Conference on Multimedia* (pp. 1786-1794).
- Mettes, P., Koelma, D. C., & Snoek, C. G. (2020).
  [Shuffled ImageNet Banks for Video Event Detection and Search](https://dl.acm.org/doi/abs/10.1145/3377875).
  ACM *Transactions on Multimedia Computing, Communications, and Applications* (TOMM), 16(2), 1-21.
- The ResNet152 model from [Apache MXNet](https://mxnet.incubator.apache.org/) framework


**To use this model on your own data you need to:**
- Download any of the trained model weights ([link](https://drive.google.com/open?id=1qLGyyyyU5kao0SxT6Eu6BWma3OmRS-yK)).
- Extract video/image features as shown bellow (note the original paper does averaging over multiple image regions and muiltiple video frames).
- Encode the video/image features and text query using the model (see `evaluate.py` how to do the encoding).
- Compute cosine similarity between those encoded videos/images and encoded text queries.

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
- [Download trained model weights.](https://drive.google.com/open?id=1qLGyyyyU5kao0SxT6Eu6BWma3OmRS-yK)

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


### Train model from scratch
- Download _tgif-msrvtt10k_ and _tv2016train_ datasets from [github.com/li-xirong/w2vvpp](https://github.com/li-xirong/w2vvpp/#data).
- Download word2vec pretraind embeddings from [lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz](http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz).
- Edit `train.py` (update path to the datasets and word2vec model, possibly update other parameters).
- Run `train.py`.


### Extract visual features for your own data
- Download model weights:
    ```
    wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params
    wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json
    wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-1-0040.params
    wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-symbol.json
    ```

- Instantiate the networks in Python
    ```python
    import numpy as np
    import mxnet as mx
    from collections import namedtuple
    
    def get_network_fc(network_path, network_epoch, normalize_inputs):
        batch_def = namedtuple('Batch', ['data'])
        sym, arg_params, aux_params = mx.model.load_checkpoint(network_path, network_epoch)
    
        network = mx.mod.Module(symbol=sym.get_internals()['flatten0_output'],
                                label_names=None,
                                context=mx.gpu())
        network.bind(for_training=False,
                     data_shapes=[("data", (1, 3, 224, 224))])
        network.set_params(arg_params, aux_params)
    
        def fc(image):
            image = image.astype(np.float32)
            if normalize_inputs:  # true for resnext101
                image = image - np.array([[[123.68, 116.779, 103.939]]], dtype=np.float32)
            image = np.transpose(image, [2, 0, 1])[np.newaxis]
            inputs = batch_def([mx.nd.array(image)])
    
            network.forward(inputs)
            return network.get_outputs()[0].asnumpy()
    
        return fc

    resnext = get_network_fc("/path/to/resnext-101-1", 40, true)
    resnet = get_network_fc("/path/to/resnet-152", 0, false)
    ```

- Get the features
    ```python
    image = np.zeros([224, 224, 3], dtype=np.uint8)
    features_for_image = np.concatenate([resnext(image), resnet(image)], 1)
    ```
