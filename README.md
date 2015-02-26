<!--
  ** File Name:	README.md
  ** Author:	Aditya Ramesh
  ** Date:	02/09/2015
  ** Contact:	_@adityaramesh.com
-->

# Overview

A collection of utilities for deep learning using Torch.

## sopt

A small library for stochastic optimization that implements the following
algorithms:

- SGU (+ CM, NAG)
- AdaDelta (+ NAG)
- RMSProp (+ NAG)

The implementation of SGU is more generic than the implementation of SGD found
in the Torch `optim` package. This one allows the learning rate and momentum
annealing schedules to be arbitrary lambdas. Several [predefined
schedules](sopt/schedule.lua) are available.

## Usage

If `torch_utilities` is not already in a path accessible to the Torch
interpreter, then you can add its location to the package path by doing the
following:

	package.path = package.path .. ';<path to torch_utilities>/?.lua'

In the code above, `<path to torch_utilities>` should be replaced with the path
to the `torch_utilities` directory.

### sopt

To import `sopt`, use `require 'sopt'`. The code below demonstrates how to set
up an optimization algorithm. Assume that the following variables are defined:

- `opt_method`: The selected optimization method.
- `init_mom`: The initial momentum.
- `init_lr`: The initial learning rate.
- `lr_decay`: The decay constant for the learning rate. A value of zero means no
decay is desired.
- `ramp_up_momentum`: Determines whether the momentum should be kept constant or
ramped up to a given value over several hundred iterations.

The code below initializes the desired optimization algorithm.

	local lr, mom

	if lr_decay ~= 0 then
		lr = sopt.gentle_decay(init_lr, lr_decay)
	else
		lr = sopt.constant(init_lr)
	end

	if ramp_up_momentum then
		mom = sopt.sutskever_blend(init_mom)
	else
		mom = sopt.constant(init_mom)
	end

	local state = {
		learning_rate = lr,
		momentum = mom,
		momentum_type = 'NAG'
	}

	local method
	if opt_method == 'SGU' then
		method = sopt.sgu
	elseif opt_method == 'ADADELTA' then
		method = sopt.adadelta
	elseif opt_method == 'RMSPROP' then
		method = sopt.rmsprop
	end

To invoke the optimization method, use `method(f, params, state)`, where `f` is
the function being minimized and `params` is the current iterate.

### Selecting the Hyperparameters

The best value for the learning rate depends on the nature of the problem and
the chosen batch size. The smaller the batch size, the smaller the chosen
learning rate should be. Initially, the batch size should be chosen to be quite
large (e.g. one). Over time, the batch size should be increased and the learning
rate decreased. With a batch size of one, a typical value for the learning rate
is `1e-3` or `1e-4`. With a batch size of about 100, a typical value for the
learning rate is `0.1` or `0.01`. If learning rate decay is used, the decay
constant should be relatively small (e.g. `1e-7`). For best results, perform a
grid search over a small number of training instances (e.g. 200). It is a
well-known and convenient property of SGU that convergence behavior on a small
subset of the training sample will be reflected (albeit on a larger time scale)
when optimization is performed using the full training sample. For more
information, see "Stochastic Gradient Descent Tricks", by Leon Bottou.

The momentum value should be set to at least `0.9`. When performing a grid
search, the values `0.9, 0.99, 0.995` and `0.999` are usually tried. See "On the
Importance of Initialization and Momentum", by Sutskever et al., for more
details. Ramping up the momentum using the `sutskever_blend` schedule may
accelerate convergence during the first few epochs.

Both RMSProp and AdaDelta can be fine-tuned in the case the need to do so
arises. Both of these algorithm accept another parameter called `decay`,
sometimes referred to as the "memory size" in the literature. (I think that this
is a shitty name.) The default value is `0.95`; perhaps changing it to something
slightly larger or smaller will help. Whatever you do, *do not* change the
`epsilon` parameter!
