# fix : https://github.com/apache/tvm/pull/10502#event-6197333598
import tvm
from tvm import relay
mod = tvm.IRModule()
x = relay.var("x", shape=(1,), dtype="float32")
x2 = relay.add(x, x)
f = relay.Function([x], x2)
body = relay.Function([], f)
mod['main'] = body
mod = relay.transform.InferType()(mod)
print(mod.astext(show_meta_data=False))
graph, lib, params = relay.build(mod, target='llvm')