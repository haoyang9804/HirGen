# https://discuss.tvm.apache.org/t/crash-when-opt-level-0/12131
# fix: https://github.com/apache/tvm/pull/10347
import tvm
from tvm import relay

var_0 = relay.var("var_0", shape=(), dtype="float64")
var_1 = relay.var("var_1", shape=(), dtype="float64")
var_2 = relay.add(var_0, var_1)
tuple = relay.Tuple([var_2])
func = relay.Function([var_0, var_1], tuple)
mod = tvm.IRModule.from_expr(func)

exe = relay.backend.vm.compile(mod, "llvm")

with tvm.transform.PassContext(opt_level=0):
    exe = relay.backend.vm.compile(mod, "llvm")
