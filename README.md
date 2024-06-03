[![Documentation Status](https://readthedocs.org/projects/reptrix/badge/?version=latest)](https://reptrix.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/PACKAGE?label=pypi%20package)](https://pypi.org/project/reptrix/)

<h3 align="center">
<p>Representation quality metrics for pretrained deep models! ⭐
</h3>
<br>


# Reptrix

<p align="center">
    <br>
    <img src="https://github.com/arnab39/reptrix/assets/6922389/06e6bffb-adac-460b-81b4-b1078859b563" alt="Reptrix" width="200"/>
    <br>
<p>

## About

Reptrix, short for Representation Metrics, is a [PyTorch](https://pytorch.org) library designed to simplify the evaluation of representation quality metrics in pretrained deep neural networks. Reptrix offers a suite of recently proposed metrics, predominanty in the vision self-supervised learning literature, that are essential for researchers and engineers focusing on design, deployment, evaluation and interpretability of deep neural networks in computer vision settings.

Key Features:
- *Comprehensive Metric Suite*: Includes a variety of metrics to assess various aspects of representation quality, that are indicative of capacity, robustness and downstream task performance.
- *PyTorch Integration*: Seamlessly integrates with existing PyTorch models and workflows, allowing for straightforward monitoring of learned representations with minimal setup.
- *Open Source*: Open for contributions and enhancements from the community, including any new metrics that are proposed.

Reptrix is the perfect tool for machine learning practitioners looking to quantitatively analyze learned representations and enhance the interpretability of their deep learning models, especially models trained in a self-supervised learning framework.
To learn more about why these metrics are essential in modern DL workflows, check out our [blogpost on Assessing Representation Quality in SSL](https://mila.quebec/en/article/a-req/)

### List of metrics currently supported

- **$\alpha$-ReQ**: This metric computes the eigenvalues of the covariance matrix of the representations and fits a power-law distribution to them. The exponent of the power-law distribution is called the $\alpha$ exponent, which measures the heavy-tailedness of the distribution. A lower alpha exponent indicates that the representations are more discriminative.
- **RankMe**: This metric computes the rank of the covariance matrix of the representations. A higher rank indicates representations of higher capacity.
- **LiDAR**: This metric computes the rank of the linear discriminant analysis (LDA) matrix. A higher rank indicates representations with higher degree of seperability among object manifolds.

**TODO**: Fill out the numbers

ResNet50

| Metric    | Time to compute (s) | Memory requirement (GB) |
|:---------:|:-------------------:|:-----------------------:|
| $\alpha$-ReQ |      2.400       |                         |
| RankMe    |         2.364       |                         |
| LiDAR     |         7.929       |                         |


ViT

| Metric    | Time to compute (s) | Memory requirement (GB) |
|:---------:|:-------------------:|:-----------------------:|
| $\alpha$-ReQ |      0.137          |                         |
| RankMe    |         0.091       |                         |
| LiDAR     |         0.162       |                         |


## Using Reptrix in your own workflow

1. Load your favourite pretrained network.

```
encoder = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
# Remove the final fully connected layer so that the model outputs the 2048 feature vector
encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
encoder.eval()
```
2. Extract features from the pretrained network.

```
def get_features(encoder_network, dataloader, transform=None, num_augmentations=10):
    # Loop over the dataset and collect the representations
    all_features = []

    # Loop over the dataset and collect the representations
    for i, data in enumerate(tqdm(dataloader, 0)):
        inputs, _ = data
        if transform:
            inputs = torch.cat([transform(inputs) for _ in range(num_augmentations)], dim=0)
        with torch.no_grad():
            features = encoder_network(inputs)
        if transform:
            # put the augmentations in an additonal dimension
            features = features.reshape(-1, num_augmentations, features.shape[1])
        all_features.append(features)


    # Concatenate all the features
    all_features = torch.cat(all_features, dim=0)
    return all_features

all_representations = get_features(encoder, loader)
num_augmentations = 10
all_representations_lidar = get_features(encoder, loader,
                                transform=transform_augs,
                                num_augmentations=num_augmentations)
num_samples = all_representations_lidar.shape[0]
```
3. Compute the representation metrics

```
from reptrix import alpha, rankme, lidar
metric_alpha = alpha.get_alpha(all_representations)
metric_rankme = rankme.get_rankme(all_representations)
metric_lidar = lidar.get_lidar(all_representations_lidar, num_samples,
                            num_augmentations,
                            del_sigma_augs=0.00001)
```


## Installation
**TODO: Update and test this!**

### Using pypi
You can install the latest version of reptrix using:

```pip install reptrix```

### Manual installation

You can clone this repository and manually install it with:

```pip install git+https://github.com/arnab39/reptrix```

### Setup Conda environment for examples

You can incorporate reptrix in your existing conda environment or create a new environment with the necessary packages:

```
conda env create -f conda_env.yaml
conda activate reptrix
pip install -e .
```



## Example code for Reptrix

We provide a [tutorial iPython notebook](tutorial.ipynb) that shows how you can incorporate metrics from our Reptrix library to your own code.

## Related papers and Citations

This library currently supports metrics proposed in three different papers:
1. $\alpha$[-ReQ : Assessing Representation Quality in Self-Supervised Learning by measuring eigenspectrum decay (NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/70596d70542c51c8d9b4e423f4bf2736-Abstract-Conference.html)
2. [RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank (ICML 2023)](https://proceedings.mlr.press/v202/garrido23a)
3. [LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures (ICLR 2024)](https://openreview.net/forum?id=f3g5XpL9Kb)



## Contact

For questions related to this code, please raise an issue and you can mail us:
[Arna Ghosh](mailto:ghosharn@mila.quebec), [Arnab K Mondal](arnab.mondal@mila.quebec), [Kumar K Agrawal](kagrawal@berkeley.edu)

## Contributing

You can check out the [contributor's guide](CONTRIBUTING.md).

This project uses `pre-commit`, you can install it before making any changes:

    pip install pre-commit
    cd reptrix
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate
