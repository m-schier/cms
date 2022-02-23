# Constrained Mean Shift Clustering

This is the official implementation of [*Constrained Mean Shift Clustering*](http://www.tnt.uni-hannover.de/papers/data/1553/CMS.pdf).

![CMS Animation on Moons data set](ReadmeMoons.gif)

Constrained Mean Shift (CMS) is a novel approach for mean shift clustering under sparse supervision 
using cannot-link constraints. The constraints provide a guidance in constrained clustering 
indicating that the respective pair should not be assigned to the same cluster. 
Our method introduces a density-based integration of the constraints to generate individual 
distributions of the sampling points per cluster. We also alleviate the (in general very sensitive) 
mean shift bandwidth parameter by proposing an adaptive bandwidth adjustment which is especially 
useful for clustering imbalanced data sets.

## Brief Usage

Given some data points and some binary cannot link constraints:

```python
from sklearn.datasets import make_moons

# Generate moons data set
x, y = make_moons(shuffle=False)
# Create one cannot-link constraint from center of one moon to another
cl = [[25, 75]]
```

CMS can be invoked similar to sklearn cluster methods:

```python
from CMS import CMS, AutoLinearPolicy

# Create bandwidth policy as used in our experiments
pol = AutoLinearPolicy(x, 100)
# Use nonblurring mean shift (do not move sampling points)
cms = CMS(pol, max_iterations=100, blurring=False)
cms.fit(x, cl)
```

The `cms` object now contains the following members:

Member | Description
--- | ---
`labels_` | Final cluster labels
`modes_` | Final position of the cluster centers/modes
`bandwidth_history_` | Bandwidths used per iteration
`mode_history_` | Cluster centers/modes per iteration
`kernel_history_` | Kernel weights per iteration
`block_history_` | Attraction reduction per iteration

To visualize the results, we provide a convenient Matplotlib routine:
```python
from CMS.Plotting import plot_clustering
import matplotlib.pyplot as plt

plot_clustering(x, cms.labels_, cms.modes_, cl=cl)

plt.show()
```

You can run [example_moons.py](example_moons.py) to try it yourself. You may also try adjusting the parameters of CMS:

Parameter | Description
--- | ---
``h`` | Set the bandwidth either to a scalar float value, or a callable ``f(int) -> float`` returning the bandwidth for the given iteration
``max_iterations`` | Maximum number of iterations
``blurring`` | If ``True`` use blurring mean shift, i.e. the sampling points are updated with the cluster centers after each iteration, thus blurring them in the process. If ``False`` use nonblurring mean shift, where sampling points remain stationary.
``kernel`` | If ``'ball'``, use a ball kernel, otherwise expects a float in range [0, 1) to use as truncation of a truncated Gaussian kernel. Thus setting ``kernel=0.`` uses a regular Gaussian kernel.
``c_scale`` | The constraint scaling parameter that determines the spatial influence of constraints. For lower values, constraints have less reducing influence on far attractions.
``label_merge_k`` | This implementation of CMS uses connected components to determine the final cluster labels from the final cluster centers. Specifies the minimum closeness in terms of kernel value to merge two cluster centers into one cluster.
``label_merge_b`` | Specifies the lowest weight reduction through constraints below which two cluster centers are never merged. Set to ``0.`` to disable.
``use_cuda`` | If ``True``, use the CUDA Toolkit to accelerate some calculations. You must have the CUDA Toolkit installed. Please consult the official CUDA documentation on how to install CUDA for your specific system.


## Installation

### Pip

To use Constrained Mean Shift (CMS) as a library, we provide easy installation through pip. Simply run
```
python -m pip install git+https://github.com/m-schier/cms
```
### Manual installation
Alternatively, you may clone this repository and install from the local folder
```
git clone git@github.com:m-schier/cms.git
cd cms
python -m pip install .
```

### Local

To run the experiments, it is not required to install CMS through pip. In this case you can create a Conda environment with the required dependencies by running the following commands. This will install most dependencies with the exact version used during out experiments.

```shell
conda create --name cms python=3.8
conda activate cms
pip install -r requirements.txt
```
## Experiments

### Synthetic Data

First, you must download the used synthetic data sets by running `./download_synth.sh`. To evaluate performance of CMS on the synthetic data sets, run `python cluster_synth.py --data <DATA>`, where `<DATA>` is one of `moons`, `jain`, `s4`, or `aggregation`, e.g., 

```shell
python cluster_synth.py --data aggregation
```


### Image Data

To evaluate performance on the pretrained image embeddings used in our work, run `python cluster_img.py --data <DATA>`, where `<DATA>` is one of the image data sets `mnist`, `fashion-mnist`, or `gtsrb`, e.g., 
```shell
python cluster_img.py --data gtsrb
```

To train a stacked denoising auto encoder and save its image embeddings, run `python pretrain.py --data <DATA>`.
In order to train on GTSRB, you must first download the GTSRB training data set by running `./download_gtsrb.sh`.


## Citation

If you found this library helpful in your research, please consider citing:

```bibtex
@inproceedings{schier2022constrained,
  title={Constrained Mean Shift Clustering},
  author={Schier, Maximilian and Reinders, Christoph and Rosenhahn, Bodo},
  booktitle={Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
  year={2022},
  organization={SIAM}
}
```
