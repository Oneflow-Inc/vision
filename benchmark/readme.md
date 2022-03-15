### Benchmark
you can add different models into different files.


#### First Step: Add a function
for example, if you want to register googlenet model to benchmark for pytest, you can create a file
 in the benchmak/ folder:
```python
from benchmark.net import *
from flowvision.models.googlenet import *


def run_googlenet(): return run_net(googlenet, [16, 3, 224, 224])
```
the benchmark.net offers a basic interface(run\_net()) for you to forward and backword the specific model.

you can also realize the function with custom internal realization.

#### Second Step: Register a function
you can register your function into benchmark/test\_benchmark.py.
for exmaple:
```python
def test_googlenet(benchmark):
    benchmark(run_googlenet)
```

After that, you can run pytest-benchmark in cli:
```sh
python3 -m pytest benchmark/
```

more features can be found in pytest-benchmark offical docs.
