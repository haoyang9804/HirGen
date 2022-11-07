import tvm
from tvm import relay
from tvm.ir.transform import Sequential

var_0 = relay.var("var_0", dtype = "int64", shape = ()) # shape=()
var_1 = relay.zeros_like(var_0) # shape=()
var_2 = relay.var("var_2", dtype = "int64", shape = (9, 4, 10)) # shape=(9, 4, 10)
var_3 = relay.multiply(var_1, var_2) # shape=(9, 4, 10)
tuple = relay.Tuple([var_3,])
F = relay.Function([var_0,var_2,], tuple)

mod = tvm.IRModule()
mod['main'] = F
mod = relay.transform.InferType()(mod)
print(mod.astext(show_meta_data=False))
graph, lib, params = relay.build(mod, target='llvm')

seq = Sequential([
        # relay.transform.FuseOps(3),
        # relay.transform.AnnotateSpans(),
        relay.transform.FirstOrderGradient(),
        relay.transform.FuseOps(3),
])
mod = seq(mod)
graph, lib, params = relay.build(mod, target='llvm')
