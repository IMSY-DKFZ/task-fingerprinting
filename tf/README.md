# MML Task Fingerprinting plugin

This plugin provides code for the Task Fingerprinting paper. It can be used to explore task distances and evaluations
of them with respect to knowledge transfer experiments.

# Install

Follow the instructions given in `notebooks/transfer_exps/recent_exps/README.md` to
set up a virtual environment and install `mml`. With the installation of `mml-tf` (`pip install .`)
the plugin is ready to use.

# Usage

The notebooks in `notebooks` demonstrate the capabilities of the plugin. As it is intended as an interactive toolkit,
do not forget to call

```python
import mml.interactive

mml.interactive.init()
```

as a first step in any notebook you want to use `mml-tf` within.