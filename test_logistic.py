from logistic import logistic_function, iterate_f
import pytest, math
#from numpy.testing import assert_allclose
from math import isclose
#@pytest.mark.parametrize("r", [2.2,3.4,1.7])

#@pytest.mark.parametrize` to test the function for the following cases:

 # x=0.1, r=2.2 => f(x, r)=0.198
 # x=0.2, r=3.4 => f(x, r)=0.544
 # x=0.75, r=1.7 => f(x, r)=0.31875

@pytest.mark.parametrize("x,r,expected", [(0.1,2.2,0.198),(0.2,3.4, 0.544),(0.75,1.7, 0.31875)])
def test_logistic_function(x,r, expected):
    output = logistic_function(x, r)
    assert math.isclose(output,expected)
