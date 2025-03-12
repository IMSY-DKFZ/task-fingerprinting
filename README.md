# Beyond Knowledge Silos: Task Fingerprinting for Democratization of Medical Imaging AI

This repository accompanies our (submitted) paper and provides code to reproduce the experiments. 
It provides intermediate results that allow to explore the experiments or test other task distances
to compare outcomes. 

## Structure

 - `cache` contains pre-computed intermediate results (task distances and transfer outcomes)
 - `data` can be filled with extracted task features to compute fingerprints
 - `figures` contains the generated figures we used in the paper
 - `notebooks` contains `jupyter` notebooks that reproduce all experiments, it separates
   - the transfer experiments (`notebooks/transfer_exps`) that are computation heavy
   - the extraction of results and task distance precomputations (`notebooks/0_fill_cache.ipynb`)
   - all downstream evaluations that can be run independently (notebooks `1` to `12`) that produce the tables, numbers and figure of the paper as well as additional insights
 - `tf` contains an `mml` plugin and is the shared code basis for the `notebooks`

## MML

The software and experiments are based on the `mml-core` package. See [here](https://github.com/IMSY-DKFZ/mml) fore details.

## License

The code and data is licensed under the MIT license, see [LICENSE.txt](LICENSE.txt).

Copyright German Cancer Research Center (DKFZ) and contributors. 

Please cite our paper alongside.

```
@article{godau2024beyond,
  title={Beyond Knowledge Silos: Task Fingerprinting for Democratization of Medical Imaging AI},
  author={Godau, Patrick and Srivastava, Akriti and Adler, Tim and Maier-Hein, Lena},
  journal={arXiv preprint arXiv:2412.08763},
  year={2024}
}
```

## Contact

Main author: Patrick Godau, Deutsches Krebsforschungszentrum (DKFZ) Heidelberg

Division of Intelligent Medical Systems

Contact: patrick.godau@dkfz-heidelberg.de