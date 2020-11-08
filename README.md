# Human Walking Radar Simulator
[![DOI](https://zenodo.org/badge/310109633.svg)](https://zenodo.org/badge/latestdoi/310109633)

The following code is based on MATLAB scripts by V. Chen for a walking human. The package provides a number of ways radar returns can be simulated.

## Direct Calls
The following call will generate the positional traces for the walking motion for a model moving `forward` at a relative velocity of `1.8`, a sampling rate of `100sps` and for `4` seconds. The radar `(x, y, z)` location will be `(0, 10, 0)`
```python
seg,segl  = generate_segments(forward_motion = True,
                                height = 1.8,
                                rv = 2.5,
                                fs = 100,
                                duration = 4,
                                radarloc = (0, 10, 0)
                               )
```
The `seg` dictionary contains all necessary kinematics data. With that, we can proceed to simulating the corresponding radar returns. The following function can be used to simulate a radar return at a carrier wavelength `lambda_` of `0.001` meters, rangeresolution of `0.01` meters, radar location of `(0, 10, 0)` and a `configuration` `basic_conf`:
```python
mat = simulate_radar(seg,
                     segl,
                     lambda_ = 0.001,
                     rangeres = 0.01,
                     radarloc = (0, 10, 0),
                     config = basic_conf
                    )
```

## Dataset Generation
For the same configuration file, a whole dataset of `512` samples can be easily created by calling
```python
n_samples = 512
ddir = 'my_static_dataset/'

generate(basic_conf, n_samples, ddir = ddir)
```

### Configuration Example
A configuration object can be defined using `OmegaConf`. This allows for 
```python
complete_human = {
    'Head' : 1,
    'Torso' : 1,
    'Left Shoulder': 1,
    'Right Shoulder': 1,
    'Left Upper Arm': 1,
    'Right Upper Arm': 1,
    'Left Lower Arm': 1,
    'Right Lower Arm': 1,
    'Left Hip': 1,
    'Right Hip': 1,
    'Left Upper Leg': 1,
    'Right Upper Leg': 1,
    'Left Lower Leg': 1,
    'Right Lower Leg': 1,
    'Left Foot': 1,
    'Right Foot': 1
}

basic_conf = OmegaConf.create({
                                  "fs" : 100,
                                  "simulator" : {
                                      "forward_motion": False,
                                      "duration" : 11,
                                      "height": (1.2, 1.8),
                                      "rv": (0.2, 1.0),
                                      "radarloc": (0,10,0),
                                      "lambda_": scipy.constants.c/24e9,
                                      "rangeres": 0.01,
                                      "body_parts" : complete_human
                                  }
                                 })
```

### Use
This code is free to use under MIT License.

Please cite this resource when used.
```latex
@software{human,
  author       = {Mikolaj Czerkawski},
  title        = {{Human Walking Radar Simulator}},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.1.1},
  doi          = {10.5281/zenodo.4245158},
  url          = {https://doi.org/10.5281/zenodo.4245158}
}
```
