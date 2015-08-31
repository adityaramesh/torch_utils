<!--
  ** File Name: TODO.md
  ** Author:    Aditya Ramesh
  ** Date:      08/29/2015
  ** Contact:   _@adityaramesh.com
-->

# TODO

## Immediate Items

- Test adadelta.

## For Later

- Make a Python utility for iPython notebook that can plot data from the data
files produced by `model_utils.lua` on the fly. It should report the following:
  - Provide API access to all of the information below from the master server.
  - High-level summary of all models and statuses (paused, running).
  - History of training and validation accuracy for each model.
  - Statistics logged by the optimization algorithms.
  - Real-time plots.
  - Ideally the slave servers would send the data to a master server, which
  generates a dynamically-updated webpage. That way, we can run jobs on many
  client servers and visualize everything simultaneously.

- Implement adam (+ NAG), adamax (+ other norms, NAG).
- RMSProp with the expansion/contraction factor modification as described by
Climin. If this is successful, perhaps try modifying AdaDelta in the same way.
- AdamDelta (+ NAG), "AdamaxDelta" (+ other norms, NAG).
- AdaSecant, along with various modifications inspired by other approaches.
  - In particular, Durk Kingma and Tom Schaul were working on integrating Adam
  with AdaSecant. But they seem to have lost interest on that now. If this turns
  out to work well, then contact them to let them know about it.
- Update documentation.

## Ideas

- Combine adadelta and adam (perhaps call it adamdelta), i.e. incorporate the
bias corrections and try using the bias-corrected first moment estimate instead
of the exact gradient.
- Support for L1, L2 weight decay? It might be useful to try L1 weight decay for
the fully-connected layers.
- Dropout applied to entire convolutional filters.
- Batch normalization?
