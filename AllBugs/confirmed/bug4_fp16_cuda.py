# https://discuss.tvm.apache.org/t/error-identifier-hfabs-is-undefined/10567 

import tvm
from tvm import relay

mod = tvm.IRModule()
var_0 = relay.var('var_0', shape=(), dtype='float16')
var_1 = relay.round(var_0)
F = relay.Function([var_0], var_1)
mod['main'] = F

with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, "cuda")