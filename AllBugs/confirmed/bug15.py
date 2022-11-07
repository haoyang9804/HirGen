# https://discuss.tvm.apache.org/t/operator-right-shift-obtains-different-results-in-different-devices/11939

import tvm
from tvm import relay
from tvm.ir.transform import Sequential
from tvm.contrib import graph_runtime
import numpy as np

var_0 = relay.var("var_0", dtype = "uint64", shape = (1, 1, 10, 1))
var_1 = relay.var("var_1", dtype = "uint64", shape = (1, 1))
var_2 = relay.right_shift(var_0, var_1) # shape=(1, 1, 10, 1)
tuple = relay.Tuple([var_2])
F = relay.Function([var_0,var_1], tuple)
mod = tvm.IRModule()
mod['main'] = F
mod = relay.transform.InferType()(mod)

#~~~~~~~ build on llvm ~~~~~~~
graph, lib, params = relay.build(mod, target='llvm')
module = graph_runtime.create(graph, lib,tvm.device('llvm',0))

input_0= np.array([[[[963],[970],[903],[170],[621],[635],[226],[419],[699],[480]]]], dtype='uint64')
input_1= np.array([[966]], dtype='uint64')

module.set_input('var_0', input_0)
module.set_input('var_1', input_1)
module.set_input(**params)
module.run()
res0 = module.get_output(0).asnumpy()  # result on cpu

#~~~~~~~ build on cuda ~~~~~~~
graph, lib, params = relay.build(mod, target='cuda')
module = graph_runtime.create(graph, lib,tvm.device('cuda',0))

module.set_input('var_0', input_0)
module.set_input('var_1', input_1)
module.set_input(**params)
module.run()
res1 = module.get_output(0).asnumpy()   # result on gpu

print(res0)
print(res1)
