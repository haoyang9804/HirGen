# https://discuss.tvm.apache.org/t/non-single-or-double-precision-floating-point-in-mod/12300

import tvm
from tvm import relay
from tvm.ir.transform import Sequential
from tvm.contrib import graph_runtime

mod = tvm.IRModule()
var_22 = relay.var("var_22", dtype = "bool", shape = ())#candidate|22|()|var|bool
var_23 = relay.var("var_23", dtype = "bool", shape = ())#candidate|23|()|var|bool
bop_25 = relay.mod(var_23.astype('float16'), var_22.astype('float16')) # shape=()
output = relay.Tuple([bop_25,])
F = relay.Function([var_22,var_23,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========irmod built by Relay==========')
print(mod.astext(show_meta_data=False))
print('===================================')
graph, lib, params = relay.build(mod, target='llvm')  # run well
graph, lib, params = relay.build(mod, target='cuda')  # crash
