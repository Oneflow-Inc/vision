from resnest import *

def test_resnest(benchmark):
    benchmark(run_resnest50)
    benchmark(run_resnest101)
    benchmark(run_resnest200)
    benchmark(run_resnest269)
