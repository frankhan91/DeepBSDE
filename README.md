# [Deep BSDE Solver](https://doi.org/10.1073/pnas.1718942115) in TensorFlow

## Quick Installation

For a quick installation, you can create a conda environment for Python using the following command:

```bash
conda env create -f environment.yml
```

## Training

```
python main.py --config_path=configs/hjb_lq_d100.json
```

Command-line flags:

* `config_path`: Config path corresponding to the partial differential equation (PDE) to solve.
There are seven PDEs implemented so far. See [Problems](#problems) section below.
* `exp_name`: Name of numerical experiment, prefix of logging and output.
* `log_dir`: Directory to write logging and output array.


## Problems

`equation.py` and `config.py` now support the following problems:

Three examples in ref [1]:
* `HJBLQ`: Hamilton-Jacobi-Bellman (HJB) equation.
* `AllenCahn`: Allen-Cahn equation with a cubic nonlinearity.
* `PricingDefaultRisk`: Nonlinear Black-Scholes equation with default risk in consideration.


Four examples in ref [2]:
* `PricingDiffRate`: Nonlinear Black-Scholes equation for the pricing of European financial derivatives
with different interest rates for borrowing and lending.
* `BurgersType`: Multidimensional Burgers-type PDEs with explicit solution.
* `QuadraticGradient`: An example PDE with quadratically growing derivatives and an explicit solution.
* `ReactionDiffusion`: Time-dependent reaction-diffusion-type example PDE with oscillating explicit solutions.


New problems can be added very easily. Inherit the class `equation`
in `equation.py` and define the new problem. Note that the generator function
and terminal function should be TensorFlow operations while the sample function
can be python operation. A proper config is needed as well.


## Dependencies

The code is compatible with TensorFlow >=2.16 and Keras 3.

For those using older versions

* a version of the Deep BSDE solver that is compatible with TensorFlow 2.13 can be found in commit [`5af8c25`](https://github.com/frankhan91/DeepBSDE/commit/5af8c25).

* *a version of the Deep BSDE solver that is compatible with TensorFlow 1.12 and Python 2 can be found in commit [`9d4e332`](https://github.com/frankhan91/DeepBSDE/commit/9d4e3329613d7a2a1f389a0d4cc652e1c4606b86).


## Related Implementations

* [TorchBSDE-Package](https://github.com/Steve-Shao/TorchBSDE-Package): A nice PyTorch implementation of the Deep BSDE solver with some new feastures.


## Reference
[1] Han, J., Jentzen, A., and E, W. Overcoming the curse of dimensionality: Solving high-dimensional partial differential equations using deep learning,
<em>Proceedings of the National Academy of Sciences</em>, 115(34), 8505-8510 (2018). [[journal]](https://doi.org/10.1073/pnas.1718942115) [[arXiv]](https://arxiv.org/abs/1707.02568) <br />
[2] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations,
<em>Communications in Mathematics and Statistics</em>, 5, 349–380 (2017).
[[journal]](https://doi.org/10.1007/s40304-017-0117-6) [[arXiv]](https://arxiv.org/abs/1706.04702)


## Citation
```bibtex
@article{HanArnulfE2018solving,
  title={Solving high-dimensional partial differential equations using deep learning},
  author={Han, Jiequn and Jentzen, Arnulf and E, Weinan},
  journal={Proceedings of the National Academy of Sciences},
  volume={115},
  number={34},
  pages={8505--8510},
  year={2018},
  publisher={National Acad Sciences},
  url={https://doi.org/10.1073/pnas.1718942115}
}

@article{EHanArnulf2017deep,
  author={E, Weinan and Han, Jiequn and Jentzen, Arnulf},
  title={Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations},
  journal={Communications in mathematics and statistics},
  volume={5},
  number={4},
  pages={349--380},
  year={2017},
  publisher={Springer},
  url={https://doi.org/10.1007/s40304-017-0117-6}
}
```
