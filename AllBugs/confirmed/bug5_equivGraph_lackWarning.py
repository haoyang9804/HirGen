# discuss: https://discuss.tvm.apache.org/t/check-failed-value-0u-1-vs-0-at-relay-build/12223
# fix: https://discuss.tvm.apache.org/t/check-failed-value-0u-1-vs-0-at-relay-build/12223
import tvm
from tvm import relay
var_1 = relay.const(1, dtype="uint64")
var_2 = relay.ones_like(var_1)
y = relay.negative(var_2)
F = relay.Function([], y)

mod = tvm.IRModule()
mod['main'] = F
graph, lib, params = relay.build(mod, target='llvm')  # no crash


var_3 = relay.var("var_3", dtype = "uint64", shape = ())
var_4 = relay.ones_like(var_3)
y = relay.negative(var_4)
F = relay.Function([var_3,], y)

mod = tvm.IRModule()
mod['main'] = F
graph, lib, params = relay.build(mod, target='llvm')  # crash as expected!