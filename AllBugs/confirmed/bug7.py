

import tvm
from tvm import relay

mod = tvm.IRModule()
var_0 = relay.var("var_0", dtype = "float32", shape = (7, 7))
# output = relay.Tuple([var_0,])
output = var_0
func_7 = relay.Function([var_0,], output)
mod['func_7'] = func_7
mod = relay.transform.InferType()(mod)
func_7_call = mod.get_global_var('func_7')
var_22 = relay.var("var_22", dtype = "float32", shape = (7, 7))
tmp = func_7_call(var_22, )

output = tmp
F = relay.Function([var_22], output)

mod['main'] = F
mod = relay.transform.InferType()(mod)
print(mod.astext(show_meta_data=False))
graph, lib, params = relay.build(mod, target='llvm')