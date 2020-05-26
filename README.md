# PtyGer
A tool for **pty**chographic **GPU**(multipl**e**)-based **r**econstruction. 

## Installation from source
```bash
export CUDAHOME=path-to-cuda
python setup.py clean
python setup.py install
```

## Dependency 
cupy - for GPU acceleration of linear algebra operations in iterative schemes. See (https://cupy.chainer.org/). For installation use
```bash
conda install -c anaconda cupy
```

## Tests
Test PtyGer with siemens-star synthetic dataset:
```bash
cd tests/
python test.py 4 0 512 1 32 siemens4g60 -s siemens 256 60
```
`4`: number of GPUs. <br />
`0`: GPU id offset (e.g., the first GPU's id is 0). <br />
`512`: used for real-experiment data only. Meaningless in synthetic test run. <br />
`1`: stride length of the scans. <br />
`32`: number of iteration. <br />
`siemens4g60`: prefix of the performance data files. <br />
`-s`: accepting synthetic dataset (use -r for real-experiment dataset). <br />
`siemens`: name of the dataset. <br />
`256`: probe size (in this case the probe size is 256^2). <br />
`60`: number of scan position (in this case the total number of scan position is 60^2=3600). <br />

In the first running, the synthetic diffraction patterns will be generated and stored as tests/data/siemens/256data60.npy. Any repeated runnings then can directly load this data file. <br />
The reconstructed images will be stored in tests/rec_siemens.
