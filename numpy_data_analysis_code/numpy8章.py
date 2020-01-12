#测试
import numpy as np 
print("小数点后六位",np.testing.assert_almost_equal(0.123456789,0.123456780,decimal=8))

eps = np.finfo(float).eps
print ("EPS", eps)
print ("1", np.testing.assert_array_max_ulp(1.0, 1.0 + eps))
print ("2", np.testing.assert_array_max_ulp(1.0, 1 + 2 * eps, maxulp=2))


print ("Equal?", np.testing.assert_equal((1, 2), (1, 3)))

print ("Pass", np.testing.assert_array_less([0, 0.123456789, np.nan], [1, 0.23456780, np.nan]))
print ("Fail", np.testing.assert_array_less([0, 0.123456789, np.nan], [0, 0.123456780, np.nan]))

print ("Pass", np.testing.assert_allclose([0, 0.123456789, np.nan], [0, 0.123456780, np.nan], rtol=1e-7, atol=0))
print ("Fail", np.testing.assert_array_equal([0, 0.123456789, np.nan], [0, 0.123456789, np.nan]))

