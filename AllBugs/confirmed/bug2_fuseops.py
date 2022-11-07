# https://discuss.tvm.apache.org/t/crash-after-adding-optimization-fuseops/12092

# https://discuss.tvm.apache.org/t/problem-with-fuseops-and-embedded-constants-in-tir/12165
import tvm
from tvm import relay
from tvm.ir.transform import Sequential

var_0 = relay.var("var_0", dtype = "uint64", shape = ()) # shape=()
var_1 = relay.zeros_like(var_0) # shape=()
var_2 = relay.var("var_2", dtype = "uint64", shape = (9, 4, 10)) # shape=(9, 4, 10)
var_3 = relay.multiply(var_1, var_2) # shape=(9, 4, 10)
tuple = relay.Tuple([var_3,])
F = relay.Function([var_0,var_2,], tuple)

mod = tvm.IRModule()
mod['main'] = F
mod = relay.transform.InferType()(mod)
print(mod.astext(show_meta_data=False))
graph, lib, params = relay.build(mod, target='llvm')

seq = Sequential([
        relay.transform.FuseOps(3),
])
mod = seq(mod)
graph, lib, params = relay.build(mod, target='llvm')
