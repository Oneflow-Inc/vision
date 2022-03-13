from resnest import *

def test_resnest50(benchmark):
    benchmark(run_resnest50)
def test_resnest101(benchmark):
    benchmark(run_resnest101)
def test_resnest200(benchmark):
    benchmark(run_resnest200)
def test_resnest269(benchmark):
    benchmark(run_resnest269)
