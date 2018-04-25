
# VIME-NNG

This code is based on [VIME repo](https://github.com/openai/vime) but with more flexible dynamics model - [noisy K-FAC](https://arxiv.org/pdf/1712.02390.pdf). To reproduce the results, you should first have [rllab](https://github.com/rllab/rllab) and Mujoco v1.31 configured. Then, run the following commands in the root folder of `rllab`:

```
git submodule add -f git@github.com:openai/vime.git sandbox/vime
touch sandbox/__init__.py
```

## Example
```
python sandbox/vime/experiments/run_trpo_bbb.py
python sandbox/vime/experiments/run_trpo_nng.py
```


## Citation
To cite this work, please use

```
@article{zhang2017noisy,
  title={Noisy Natural Gradient as Variational Inference},
  author={Zhang, Guodong and Sun, Shengyang and Duvenaud, David and Grosse, Roger},
  journal={arXiv preprint arXiv:1712.02390},
  year={2017}
}
```
