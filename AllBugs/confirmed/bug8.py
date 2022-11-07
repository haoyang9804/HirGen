import tvm
from tvm import relay
mod = tvm.IRModule()
c = relay.const(1, dtype="float32")
f = relay.Function([], c)
mod['f'] = f
mod = relay.transform.InferType()(mod)
fc = mod.get_global_var('f')
F = relay.Function([], fc())
mod['main'] = F
mod = relay.transform.InferType()(mod)
print(mod)
relay.optimize(mod, target="llvm")
