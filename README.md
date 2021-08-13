# naswot
An unofficial replication of NAS Without Training. The official codebase lives [here](https://github.com/BayesWatch/nas-without-training). I am just replicating the results of the paper as a way to verify the correctness of some [tooling that I am building](https://github.com/jack-willturner/gymnastics). 

## Usage 

Provide a configuration file (an example is given in `experiment_configs/cifar_nasbench_sweep.yaml`) with details of which benchmarks to test on. 

Then running 

```
python nas.py --proxy NASWOT
python nas.py --proxy Fisher
```

should produce a results tables. 