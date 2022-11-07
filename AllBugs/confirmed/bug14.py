# https://discuss.tvm.apache.org/t/cannot-allocate-memory-symbolic-tensor-shape/12514
import tvm
from tvm import relay
mod = tvm.IRModule()
x = relay.var("x", dtype = "uint8", shape = (1,2))
y = relay.var("y", dtype = "uint8", shape = (1,2))
z = relay.left_shift(x.astype('uint8'), relay.reshape(y.astype('uint8'), relay.shape_of(x)))
F = relay.Function([x,y,], z)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print(mod)
graph, lib, params = relay.build(mod, target='cuda')