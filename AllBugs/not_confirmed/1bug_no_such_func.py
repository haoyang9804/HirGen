# https://discuss.tvm.apache.org/t/check-failed-pf-nullptr-is-false-no-such-function-in-module/12345
import tvm
from tvm import relay
from tvm.contrib import graph_runtime

mod = tvm.IRModule()
var_0 = relay.var("var_0", dtype = "uint64", shape = ())#candidate|0|()|var|uint64
var_1 = relay.var("var_1", dtype = "uint64", shape = (10, 1, 4))#candidate|1|(10, 1, 4)|var|uint64
uop_2 = relay.add(var_0.astype('uint64'), var_1.astype('uint64')) # shape=(10, 1, 4)
output = uop_2
func_3 = relay.Function([var_0,var_1,], output)
mod['func_3'] = func_3
mod = relay.transform.InferType()(mod)
var_4 = relay.var("var_4", dtype = "uint8", shape = (2,))#candidate|4|(2,)|var|uint8
var_5 = relay.var("var_5", dtype = "uint8", shape = (2,))#candidate|5|(2,)|var|uint8
uop_6 = relay.add(var_4.astype('uint8'), relay.reshape(var_5.astype('uint8'), relay.shape_of(var_4))) # shape=(2,)
func_3_call = mod.get_global_var('func_3')
const_8 = relay.const(-3, dtype = "uint64")#candidate|8|()|const|uint64
const_9 = relay.const([9,-1,-2,4,-6,4,-1,9,-9,3,-2,6,-3,-4,-1,-10,8,5,-4,1,-7,2,-8,-1,-5,-6,9,-3,-2,-1,-9,-5,4,6,-1,8,6,8,-10,-2], dtype = "uint64")#candidate|9|(40,)|const|uint64
call_7 = func_3_call(relay.reshape(const_8.astype('uint64'), []), relay.reshape(const_9.astype('uint64'), [10, 1, 4]), )
var_10 = relay.var("var_10", dtype = "uint64", shape = (10, 4, 4))#candidate|10|(10, 4, 4)|var|uint64
uop_11 = relay.add(call_7.astype('float16'), var_10.astype('float16')) # shape=(10, 4, 4)
var_12 = relay.var("var_12", dtype = "int16", shape = (8, 6, 4))#candidate|12|(8, 6, 4)|var|int16
var_13 = relay.var("var_13", dtype = "int16", shape = (8, 6, 4))#candidate|13|(8, 6, 4)|var|int16
uop_14 = relay.add(var_12.astype('int16'), relay.reshape(var_13.astype('int16'), relay.shape_of(var_12))) # shape=(8, 6, 4)
func_3_call = mod.get_global_var('func_3')
call_15 = func_3_call(relay.reshape(const_8.astype('uint64'), []), relay.reshape(const_9.astype('uint64'), [10, 1, 4]), )
var_16 = relay.var("var_16", dtype = "float64", shape = ())#candidate|16|()|var|float64
var_17 = relay.var("var_17", dtype = "float64", shape = ())#candidate|17|()|var|float64
uop_18 = relay.add(var_16.astype('float64'), var_17.astype('float64')) # shape=()
output = relay.Tuple([uop_6,const_8,const_9,uop_11,uop_14,call_15,uop_18,])
func_19 = relay.Function([var_4,var_5,var_10,var_12,var_13,var_16,var_17,], output)
mod['func_19'] = func_19
mod = relay.transform.InferType()(mod)
var_20 = relay.var("var_20", dtype = "float32", shape = (4, 3, 3))#candidate|20|(4, 3, 3)|var|float32
var_21 = relay.var("var_21", dtype = "float32", shape = (4, 3, 3))#candidate|21|(4, 3, 3)|var|float32
uop_22 = relay.add(var_20.astype('float32'), relay.reshape(var_21.astype('float32'), relay.shape_of(var_20))) # shape=(4, 3, 3)
output = relay.Tuple([uop_22,])
func_23 = relay.Function([var_20,var_21,], output)
mod['func_23'] = func_23
mod = relay.transform.InferType()(mod)
var_24 = relay.var("var_24", dtype = "int64", shape = (2, 7))#candidate|24|(2, 7)|var|int64
var_25 = relay.var("var_25", dtype = "int64", shape = (2, 7))#candidate|25|(2, 7)|var|int64
uop_26 = relay.add(var_24.astype('int64'), relay.reshape(var_25.astype('int64'), relay.shape_of(var_24))) # shape=(2, 7)
func_3_call = mod.get_global_var('func_3')
var_28 = relay.var("var_28", dtype = "uint64", shape = ())#candidate|28|()|var|uint64
const_29 = relay.const([5,3,-3,-8,-4,4,-6,-2,5,6,-1,5,-6,3,3,8,-3,5,4,1,-4,-2,7,-7,-8,-10,-2,9,1,7,-5,9,-6,6,-10,7,-1,4,5,-2], dtype = "uint64")#candidate|29|(40,)|const|uint64
call_27 = func_3_call(relay.reshape(var_28.astype('uint64'), []), relay.reshape(const_29.astype('uint64'), [10, 1, 4]), )
func_19_call = mod.get_global_var('func_19')
const_31 = relay.const([-8,-3], dtype = "uint8")
const_32 = relay.const([[6,2,7,8,-5,5,10,-3,-4,-7,3,-5,8,-5,-4,10,4,-3,-3,9,3,-9,5,-5,5,1,10,8,-4,4,2,9,5,9,4,-6,-1,7,-9,4],[-6,3,-1,-6,5,-2,-4,-5,2,7,-1,2,-7,9,-3,1,7,7,-4,2,5,-6,-6,7,2,-5,-6,8,-8,-1,-9,-10,6,-8,-2,-10,6,-3,-5,-2],[6,-4,4,8,-7,3,-1,6,-2,-8,5,-9,-1,3,5,-5,-3,-5,-10,-9,4,-5,-4,-2,5,1,7,-1,2,-8,-1,10,-1,3,10,10,-3,-8,-8,-5],[10,3,6,10,9,-5,-1,6,-3,-5,8,-6,-10,2,-5,-1,3,-4,9,6,-2,-8,-7,-6,1,3,7,-4,4,-7,3,3,1,4,3,-7,1,7,10,-10]], dtype = "uint64")#candidate|32|(4, 40)|const|uint64
const_33 = relay.const([2,-5,-2,-2,-7,6,-10,7,-8,4,-5,-1,-6,-8,-1,2,-8,10,5,1,-9,-1,-6,3,-9,9,3,-6,10,5,-8,5,-3,8,-8,-9,-7,-6,-1,9,7,5,8,10,5,9,2,1,-8,-5,-7,-5,-5,9,1,7,6,9,3,-1,-5,-4,-2,-4,4,-9,4,-7,-5,6,-2,4,-2,1,-4,-4,3,-6,2,7,9,-7,2,-1,-8,-4,9,10,3,-5,-4,-4,6,3,-1,-4,-8,5,-4,-8,-3,-5,-10,-2,-8,10,7,-8,5,8,4,10,2,-4,10,-1,6,-7,-2,-9,-7,2,-4,-6,-4,9,-6,-9,-2,7,7,-7,5,-9,-5,7,9,8,10,4,-8,-3,10,5,7,4,-1,6,6,-4,6,-5,-9,-10,-4,-4,-7,-9,10,-9,-7,10,-7,-5,-6,6,-8,-8,-1,-9,-5,9,1,9,1,6,-4,-7,7,-3,-9,-6,9,8,-4,-7,10,-1,4,-10,-9,-3], dtype = "int16")#candidate|33|(192,)|const|int16
call_30 = relay.TupleGetItem(func_19_call(relay.reshape(const_31.astype('uint8'), [2,]), relay.reshape(const_31.astype('uint8'), [2,]), relay.reshape(const_32.astype('uint64'), [10, 4, 4]), relay.reshape(const_33.astype('int16'), [8, 6, 4]), relay.reshape(const_33.astype('int16'), [8, 6, 4]), relay.reshape(var_28.astype('float64'), []), relay.reshape(var_28.astype('float64'), []), ), 2)
func_3_call = mod.get_global_var('func_3')
call_34 = func_3_call(relay.reshape(var_28.astype('uint64'), []), relay.reshape(call_30.astype('uint64'), [10, 1, 4]), )
uop_35 = relay.add(call_34.astype('uint16'), relay.reshape(call_27.astype('uint16'), relay.shape_of(call_34))) # shape=(10, 1, 4)
output = relay.Tuple([uop_26,var_28,const_29,call_30,const_31,const_32,const_33,uop_35,])
func_36 = relay.Function([var_24,var_25,var_28,], output)
mod['func_36'] = func_36
mod = relay.transform.InferType()(mod)
const_37 = relay.const([-10,-9,3,3,3,-1], dtype = "uint8")#candidate|37|(6,)|const|uint8
var_38 = relay.var("var_38", dtype = "uint8", shape = (6,))#candidate|38|(6,)|var|uint8
uop_39 = relay.add(const_37.astype('uint8'), relay.reshape(var_38.astype('uint8'), relay.shape_of(const_37))) # shape=(6,)
func_23_call = mod.get_global_var('func_23')
var_41 = relay.var("var_41", dtype = "float32", shape = (9, 4))#candidate|41|(9, 4)|var|float32
call_40 = relay.TupleGetItem(func_23_call(relay.reshape(var_41.astype('float32'), [4, 3, 3]), relay.reshape(var_41.astype('float32'), [4, 3, 3]), ), 0)
var_42 = relay.var("var_42", dtype = "uint32", shape = (2,))#candidate|42|(2,)|var|uint32
var_43 = relay.var("var_43", dtype = "uint32", shape = (2,))#candidate|43|(2,)|var|uint32
uop_44 = relay.add(var_42.astype('uint32'), relay.reshape(var_43.astype('uint32'), relay.shape_of(var_42))) # shape=(2,)
output = relay.Tuple([uop_39,call_40,])
F = relay.Function([var_38,var_41,var_42,var_43,], output)
mod['main'] = F
graph, lib, params = relay.build(mod, target='llvm')
module = graph_runtime.create(graph, lib, tvm.device('llvm',0))   # crash.
