## https://github.com/apache/tvm/issues/10398
import tvm
from tvm import relay
mod = tvm.IRModule()

x1 = relay.var("x1", dtype='float64', shape=())
F1 = relay.Function([x1], x1)
mod['F1'] = F1

x2 = relay.var("x2", dtype='float64', shape=())
f = mod.get_global_var('F1')
y2 = f(x2)
F2 = relay.Function([x2], y2)
mod['main'] = F2

mod = relay.transform.InferType()(mod)
print(mod)