# [Deep BSDE Solver](https://arxiv.org/abs/1707.02568) in TensorFlow


## Training

```
python main.py --problem=SquareGradient
```

Command-line flags:

* `problem_name`: Name of partial differential equation (PDE) to solve. 
There are seven PDEs implemented so far. See [Problems](#problems) section below.
* `num_run`: Number of experiments to repeatedly run for the same problem.
* `log_dir`: Directory to write event logs and output array.


## Problems

`equation.py` and `config.py` now support the following problems:

* `AllenCahn`: Allen-Cahn equation with a cubic nonlinearity.
* `HJB`: Hamilton-Jacobi-Bellman (HJB) equation.
* `PricingOption`: Nonlinear Black-Scholes equation for the pricing of European financial derivatives
with different interest rates for borrowing and lending.
* `PricingDefaultRisk`: Nonlinear Black-Scholes equation with default risk in consideration.
* `BurgesType`: Multidimensional Burgers-type PDEs with explicit solution.
* `QuadraticGradients`: An example PDE with quadratically growing derivatives and an explicit solution.
* `ReactionDiffusion`: Time-dependent reaction-diffusion-type example PDE with oscillating explicit solutions.


New problems can be added very easily. Inherit the class `equation`
in `equation.py` and define the new problem. Note that the generator function 
and terminal function should be TensorFlow operation while the sample function
can be python operation. Also remember to a give proper config in `config.py`.


## Dependencies

* [TensorFlow >=1.2](https://www.tensorflow.org/)

## Reference
[1] Han, J., Jentzen, A., and E, W. Overcoming the curse of dimensionality: Solving high-dimensional partial differential equations using deep learning. 
[arXiv:1707.02568](https://arxiv.org/abs/1707.02568) (2017) <br />
[2] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations. 
[arXiv:1706.04702 ](https://arxiv.org/abs/1706.04702) (2017)


