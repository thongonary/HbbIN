Interaction networks for the identification of boosted Higgs to bb decays
======================================================================================

This is the Tensorflow 2.0 implementation of the interaction network model in
E. Moreno et al., Interaction networks for the identification of boosted Higgs to bb decays, [arXiv:1909.12285](https://arxiv.org/abs/1909.12285) [hep-ex]

For the original PyTorch implementation as well as plots making functionality, please refer to https://github.com/eric-moreno/IN

Requirements
======================================================================================
```
python 3.6
h5py 2.9.0
numpy 1.16.4
tensorflow-gpu 2.0.0
```

Optional:
```
setGPU 0.0.7
gpustat 0.6.0
```

Training
======================================================================================

Change the `test_path` and `train_path` in [training.py](training.py) to reflect the directories of the test and training datasets (in converted h5 format). 

Determine the parameters needed for the IN. For example: 

  - Output directory = IN_training
  - Vertex-vertex branch = 0 (turned off)
  - De = 20 
  - Do = 24
  - Hidden = 60

Would be run by doing:

```bash
python3 training.py IN_training 0 --De 20 --Do 24 --hidden 60 
```
