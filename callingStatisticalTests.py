from statisticalTests import *

describe_data(SP5002)
res = regress_bp(SP5002)
breusch_pagan_test(res)
whites_test(res)
robust_se_OLS(SP5002)
plot(SP5002)