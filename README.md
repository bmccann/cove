# Context Vectors (CoVe)

This repo provides the best MT-LSTM from the paper [Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://arxiv.org/abs/1708.00107).
For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors).

example.py uses [torchtext](https://github.com/pytorch/text/tree/master/torchtext) to load the [Stanford Natural Language Inference Corpus](https://nlp.stanford.edu/projects/snli/) and [GloVe](https://nlp.stanford.edu/projects/glove/).

It uses a [PyTorch](http://pytorch.org/) implementation of the MTLSTM class in mtlstm.py to load a pretrained encoder, 
which takes in sequences of vectors pretrained with GloVe and outputs CoVe.

## Running with Docker

Install [Docker](https://www.docker.com/get-docker).
Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if you would like to use with with a GPU.

### Run the CoVe server

If you are looking to get the raw CoVe, GloVe, and Character n-gram  embeddings,
you can run the CoVe server. Once the server is running, passing in a sequence 


```bash
nvidia-docker pull bmccann/cove-server                        # pull the docker image
nvidia-docker run -d --name cove-server -p 8888:8000 \        # start the CoVe server on port 8888
    -e "GPU=-1"                                               # specify GPU or -1 for CPU
    -v `pwd`/.embeddings:/cove/.embeddings                    # mount point for embeddings cache 
    -v `pwd`/.torch:/cove/cove/.torch \                       # mount point for MT-LSTM cache
   bmccann/cove-server
nvidia-docker logs -f cove-server                             # wait until server is ready
python cove.py -output_file hello_world.npy hello world       # get concatenation of CoVe, GloVe, Char
```

### Run the PyTorch example

```bash
nvidia-docker pull bmccann/cove          # pull the docker image
nvidia-docker docker run -it cove        # start a docker container
python /cove/test/example.py
```

## Running without Docker

Install [PyTorch](http://pytorch.org/).

```bash 
git clone https://github.com/salesforce/cove.git # use ssh: git@github.com:salesforce/cove.git
cd cove
pip install -r requirements.txt
python setup.py develop
python test/example.py
```


## References

If using this code, please cite:

   B. McCann, J. Bradbury, C. Xiong, R. Socher, [*Learned in Translation: Contextualized Word Vectors*](https://arxiv.org/abs/1708.00107)

```
@article{McCann2017LearnedIT,
  title={Learned in Translation: Contextualized Word Vectors},
  author={Bryan McCann and James Bradbury and Caiming Xiong and Richard Socher},
  journal={arXiv preprint arXiv:1708.00107},
  year={2017}
}
```

Contact: [bmccann@salesforce.com](mailto:bmccann@salesforce.com)
