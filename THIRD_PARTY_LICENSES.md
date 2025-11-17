# Third-Party Licenses

This document lists all third-party dependencies used in ResNet-BK and their respective licenses.

## Direct Dependencies

### PyTorch (BSD-3-Clause License)
- **Package**: torch
- **Version**: >=2.0.0
- **License**: BSD-3-Clause
- **Homepage**: https://pytorch.org/
- **License Text**: https://github.com/pytorch/pytorch/blob/master/LICENSE

### NumPy (BSD-3-Clause License)
- **Package**: numpy
- **Version**: >=1.24.0
- **License**: BSD-3-Clause
- **Homepage**: https://numpy.org/
- **License Text**: https://github.com/numpy/numpy/blob/main/LICENSE.txt

### SciPy (BSD-3-Clause License)
- **Package**: scipy
- **Version**: >=1.10.0
- **License**: BSD-3-Clause
- **Homepage**: https://scipy.org/
- **License Text**: https://github.com/scipy/scipy/blob/main/LICENSE.txt

### Matplotlib (PSF License)
- **Package**: matplotlib
- **Version**: >=3.7.0
- **License**: PSF (Python Software Foundation)
- **Homepage**: https://matplotlib.org/
- **License Text**: https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE

### Seaborn (BSD-3-Clause License)
- **Package**: seaborn
- **Version**: >=0.12.0
- **License**: BSD-3-Clause
- **Homepage**: https://seaborn.pydata.org/
- **License Text**: https://github.com/mwaskom/seaborn/blob/master/LICENSE.md

### Pandas (BSD-3-Clause License)
- **Package**: pandas
- **Version**: >=2.0.0
- **License**: BSD-3-Clause
- **Homepage**: https://pandas.pydata.org/
- **License Text**: https://github.com/pandas-dev/pandas/blob/main/LICENSE

### tqdm (MIT License + MPL-2.0)
- **Package**: tqdm
- **Version**: >=4.65.0
- **License**: MIT + MPL-2.0
- **Homepage**: https://tqdm.github.io/
- **License Text**: https://github.com/tqdm/tqdm/blob/master/LICENCE

### PyYAML (MIT License)
- **Package**: pyyaml
- **Version**: >=6.0
- **License**: MIT
- **Homepage**: https://pyyaml.org/
- **License Text**: https://github.com/yaml/pyyaml/blob/master/LICENSE

### TensorBoard (Apache-2.0 License)
- **Package**: tensorboard
- **Version**: >=2.13.0
- **License**: Apache-2.0
- **Homepage**: https://www.tensorflow.org/tensorboard
- **License Text**: https://github.com/tensorflow/tensorboard/blob/master/LICENSE

### Transformers (Apache-2.0 License)
- **Package**: transformers
- **Version**: >=4.30.0
- **License**: Apache-2.0
- **Homepage**: https://huggingface.co/transformers/
- **License Text**: https://github.com/huggingface/transformers/blob/main/LICENSE

### Datasets (Apache-2.0 License)
- **Package**: datasets
- **Version**: >=2.12.0
- **License**: Apache-2.0
- **Homepage**: https://huggingface.co/docs/datasets/
- **License Text**: https://github.com/huggingface/datasets/blob/main/LICENSE

### Tokenizers (Apache-2.0 License)
- **Package**: tokenizers
- **Version**: >=0.13.0
- **License**: Apache-2.0
- **Homepage**: https://github.com/huggingface/tokenizers
- **License Text**: https://github.com/huggingface/tokenizers/blob/main/LICENSE

### Accelerate (Apache-2.0 License)
- **Package**: accelerate
- **Version**: >=0.20.0
- **License**: Apache-2.0
- **Homepage**: https://huggingface.co/docs/accelerate/
- **License Text**: https://github.com/huggingface/accelerate/blob/main/LICENSE

### einops (MIT License)
- **Package**: einops
- **Version**: >=0.6.0
- **License**: MIT
- **Homepage**: https://github.com/arogozhnikov/einops
- **License Text**: https://github.com/arogozhnikov/einops/blob/master/LICENSE

## Development Dependencies

### pytest (MIT License)
- **Package**: pytest
- **Version**: >=7.3.0
- **License**: MIT
- **Homepage**: https://pytest.org/
- **License Text**: https://github.com/pytest-dev/pytest/blob/main/LICENSE

### pytest-cov (MIT License)
- **Package**: pytest-cov
- **Version**: >=4.1.0
- **License**: MIT
- **Homepage**: https://pytest-cov.readthedocs.io/
- **License Text**: https://github.com/pytest-dev/pytest-cov/blob/master/LICENSE

### Black (MIT License)
- **Package**: black
- **Version**: >=23.3.0
- **License**: MIT
- **Homepage**: https://black.readthedocs.io/
- **License Text**: https://github.com/psf/black/blob/main/LICENSE

### Flake8 (MIT License)
- **Package**: flake8
- **Version**: >=6.0.0
- **License**: MIT
- **Homepage**: https://flake8.pycqa.org/
- **License Text**: https://github.com/PyCQA/flake8/blob/main/LICENSE

### MyPy (MIT License)
- **Package**: mypy
- **Version**: >=1.3.0
- **License**: MIT
- **Homepage**: https://mypy-lang.org/
- **License Text**: https://github.com/python/mypy/blob/master/LICENSE

### isort (MIT License)
- **Package**: isort
- **Version**: >=5.12.0
- **License**: MIT
- **Homepage**: https://pycqa.github.io/isort/
- **License Text**: https://github.com/PyCQA/isort/blob/main/LICENSE

## Optional Dependencies

### Weights & Biases (MIT License)
- **Package**: wandb
- **Version**: >=0.15.0
- **License**: MIT
- **Homepage**: https://wandb.ai/
- **License Text**: https://github.com/wandb/wandb/blob/main/LICENSE

### Hugging Face Hub (Apache-2.0 License)
- **Package**: huggingface_hub
- **Version**: >=0.16.0
- **License**: Apache-2.0
- **Homepage**: https://huggingface.co/docs/huggingface_hub/
- **License Text**: https://github.com/huggingface/huggingface_hub/blob/main/LICENSE

### Jupyter (BSD-3-Clause License)
- **Package**: jupyter
- **Version**: >=1.0.0
- **License**: BSD-3-Clause
- **Homepage**: https://jupyter.org/
- **License Text**: https://github.com/jupyter/jupyter/blob/master/LICENSE

### JupyterLab (BSD-3-Clause License)
- **Package**: jupyterlab
- **Version**: >=4.0.0
- **License**: BSD-3-Clause
- **Homepage**: https://jupyterlab.readthedocs.io/
- **License Text**: https://github.com/jupyterlab/jupyterlab/blob/master/LICENSE

## License Summary

| License Type | Count | Packages |
|--------------|-------|----------|
| MIT | 10 | tqdm, PyYAML, einops, pytest, pytest-cov, black, flake8, mypy, isort, wandb |
| BSD-3-Clause | 7 | PyTorch, NumPy, SciPy, Seaborn, Pandas, Jupyter, JupyterLab |
| Apache-2.0 | 6 | TensorBoard, Transformers, Datasets, Tokenizers, Accelerate, HF Hub |
| PSF | 1 | Matplotlib |
| MPL-2.0 | 1 | tqdm (dual license) |

## License Compatibility

All dependencies are compatible with the MIT License used by ResNet-BK:
- ✅ MIT License: Fully compatible
- ✅ BSD-3-Clause: Fully compatible
- ✅ Apache-2.0: Fully compatible
- ✅ PSF License: Fully compatible
- ✅ MPL-2.0: Compatible for library use

## Full License Texts

For complete license texts of all dependencies, please refer to their respective repositories linked above.

## Updating This Document

To update this document with the latest dependencies:

```bash
# Generate dependency list
pip list --format=freeze > requirements-frozen.txt

# Check licenses
pip-licenses --format=markdown --output-file=licenses.md
```

## Questions?

If you have questions about licensing or need clarification on any dependency, please:
1. Check the dependency's repository for license information
2. Open an issue in our repository
3. Contact us at arat252539@gmail.com

---

**Last Updated**: 2025-01-15

**Note**: This document is maintained to the best of our ability. If you notice any inaccuracies, please let us know by opening an issue.
