# https://discuss.tvm.apache.org/t/failure-occurs-when-using-relay-floor-mod-and-the-divisor-is-of-type-uint64/11994
import tvm
from tvm import relay

const_6 = relay.const(126, dtype = "uint64")
var_7 = relay.var("var_7", dtype = "uint64", shape = ())
var_12 = relay.divide(const_6, var_7)
var_16 = relay.var("var_16", dtype = "uint64", shape = ())
var_17 = relay.floor_mod(var_16, var_12)
tuple = relay.Tuple([var_17,])
F = relay.Function([var_16,var_7], tuple)
mod = tvm.IRModule()
mod['main'] = F
mod = relay.transform.InferType()(mod)
print(mod.astext(show_meta_data=False))
graph, lib, params = relay.build(mod, target='cuda')
