import tvm
from tvm import relay
from tvm.ir.transform import Sequential
from tvm.contrib import graph_runtime
import numpy as np
def vmobj_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return [o]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f, dtype))
        return result
    else:
        return o


mod = tvm.IRModule()
mutated_mod = tvm.IRModule()
const_7 = relay.const([1.303931,-3.024799,9.655493,7.552588,-7.793525,-9.506576,9.294083,-8.326786,5.516349,-3.975415], dtype = "float64")#candidate|7|(10,)|const|float64
uop_8 = relay.log(const_7.astype('float64')) # shape=(10,)
bop_12 = relay.minimum(uop_8.astype('uint32'), relay.reshape(const_7.astype('uint32'), relay.shape_of(uop_8))) # shape=(10,)
bop_21 = relay.logical_xor(uop_8.astype('int8'), relay.reshape(const_7.astype('int8'), relay.shape_of(uop_8))) # shape=(10,)
output = relay.Tuple([bop_12,bop_21,])
output2 = relay.Tuple([bop_12,bop_21,])
func_24 = relay.Function([], output)
mod['func_24'] = func_24
mod = relay.transform.InferType()(mod)
mutated_mod['func_24'] = func_24
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mutated_mod.get_global_var('func_24')
call_25 = func_24_call()
output = call_25
func_26 = relay.Function([], output)
mutated_mod['func_26'] = func_26
mutated_mod = relay.transform.InferType()(mutated_mod)
var_75 = relay.var("var_75", dtype = "int64", shape = ())#candidate|75|()|var|int64
var_76 = relay.var("var_76", dtype = "int64", shape = (10,))#candidate|76|(10,)|var|int64
bop_77 = relay.greater_equal(var_75.astype('bool'), var_76.astype('bool')) # shape=(10,)
output = relay.Tuple([bop_77,])
output2 = relay.Tuple([bop_77,])
func_80 = relay.Function([var_75,var_76,], output)
mod['func_80'] = func_80
mod = relay.transform.InferType()(mod)
mutated_mod['func_80'] = func_80
mutated_mod = relay.transform.InferType()(mutated_mod)
func_80_call = mutated_mod.get_global_var('func_80')
var_82 = relay.var("var_82", dtype = "int64", shape = ())#candidate|82|()|var|int64
var_83 = relay.var("var_83", dtype = "int64", shape = (10,))#candidate|83|(10,)|var|int64
call_81 = func_80_call(var_82,var_83,)
output = call_81
func_84 = relay.Function([var_82,var_83,], output)
mutated_mod['func_84'] = func_84
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_132 = relay.TupleGetItem(func_24_call(), 1)
call_133 = relay.TupleGetItem(func_26_call(), 1)
var_150 = relay.var("var_150", dtype = "int8", shape = (10,))#candidate|150|(10,)|var|int8
bop_151 = relay.left_shift(call_132.astype('int8'), relay.reshape(var_150.astype('int8'), relay.shape_of(call_132))) # shape=(10,)
bop_154 = relay.left_shift(call_133.astype('int8'), relay.reshape(var_150.astype('int8'), relay.shape_of(call_133))) # shape=(10,)
bop_160 = relay.floor_mod(bop_151.astype('float64'), relay.reshape(var_150.astype('float64'), relay.shape_of(bop_151))) # shape=(10,)
bop_163 = relay.floor_mod(bop_154.astype('float64'), relay.reshape(var_150.astype('float64'), relay.shape_of(bop_154))) # shape=(10,)
var_169 = relay.var("var_169", dtype = "int8", shape = (10,))#candidate|169|(10,)|var|int8
bop_170 = relay.power(bop_151.astype('float64'), relay.reshape(var_169.astype('float64'), relay.shape_of(bop_151))) # shape=(10,)
bop_173 = relay.power(bop_154.astype('float64'), relay.reshape(var_169.astype('float64'), relay.shape_of(bop_154))) # shape=(10,)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
var_176 = relay.var("var_176", dtype = "int64", shape = ())#candidate|176|()|var|int64
call_175 = relay.TupleGetItem(func_80_call(relay.reshape(var_176.astype('int64'), []), relay.reshape(bop_151.astype('int64'), [10,]), ), 0)
call_177 = relay.TupleGetItem(func_84_call(relay.reshape(var_176.astype('int64'), []), relay.reshape(bop_151.astype('int64'), [10,]), ), 0)
var_182 = relay.var("var_182", dtype = "int8", shape = (10,))#candidate|182|(10,)|var|int8
bop_183 = relay.bitwise_or(bop_151.astype('int16'), relay.reshape(var_182.astype('int16'), relay.shape_of(bop_151))) # shape=(10,)
bop_186 = relay.bitwise_or(bop_154.astype('int16'), relay.reshape(var_182.astype('int16'), relay.shape_of(bop_154))) # shape=(10,)
uop_191 = relay.log10(var_150.astype('float32')) # shape=(10,)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
call_201 = relay.TupleGetItem(func_80_call(relay.reshape(var_176.astype('int64'), []), relay.reshape(bop_170.astype('int64'), [10,]), ), 0)
call_202 = relay.TupleGetItem(func_84_call(relay.reshape(var_176.astype('int64'), []), relay.reshape(bop_170.astype('int64'), [10,]), ), 0)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_204 = relay.TupleGetItem(func_24_call(), 1)
call_205 = relay.TupleGetItem(func_26_call(), 1)
output = relay.Tuple([bop_160,bop_170,call_175,var_176,bop_183,uop_191,call_201,call_204,])
output2 = relay.Tuple([bop_163,bop_173,call_177,var_176,bop_186,uop_191,call_202,call_205,])
func_209 = relay.Function([var_150,var_169,var_176,var_182,], output)
mod['func_209'] = func_209
mod = relay.transform.InferType()(mod)
mutated_mod['func_209'] = func_209
mutated_mod = relay.transform.InferType()(mutated_mod)
func_209_call = mutated_mod.get_global_var('func_209')
var_211 = relay.var("var_211", dtype = "int8", shape = (10,))#candidate|211|(10,)|var|int8
var_212 = relay.var("var_212", dtype = "int8", shape = (10,))#candidate|212|(10,)|var|int8
var_213 = relay.var("var_213", dtype = "int64", shape = ())#candidate|213|()|var|int64
var_214 = relay.var("var_214", dtype = "int8", shape = (10,))#candidate|214|(10,)|var|int8
call_210 = func_209_call(var_211,var_212,var_213,var_214,)
output = call_210
func_215 = relay.Function([var_211,var_212,var_213,var_214,], output)
mutated_mod['func_215'] = func_215
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_232 = relay.TupleGetItem(func_24_call(), 0)
call_233 = relay.TupleGetItem(func_26_call(), 0)
func_209_call = mod.get_global_var('func_209')
func_215_call = mutated_mod.get_global_var('func_215')
var_260 = relay.var("var_260", dtype = "int64", shape = ())#candidate|260|()|var|int64
call_259 = relay.TupleGetItem(func_209_call(relay.reshape(call_232.astype('int8'), [10,]), relay.reshape(call_232.astype('int8'), [10,]), relay.reshape(var_260.astype('int64'), []), relay.reshape(call_232.astype('int8'), [10,]), ), 7)
call_261 = relay.TupleGetItem(func_215_call(relay.reshape(call_232.astype('int8'), [10,]), relay.reshape(call_232.astype('int8'), [10,]), relay.reshape(var_260.astype('int64'), []), relay.reshape(call_232.astype('int8'), [10,]), ), 7)
output = relay.Tuple([call_232,call_259,var_260,])
output2 = relay.Tuple([call_233,call_261,var_260,])
func_263 = relay.Function([var_260,], output)
mod['func_263'] = func_263
mod = relay.transform.InferType()(mod)
var_264 = relay.var("var_264", dtype = "int64", shape = ())#candidate|264|()|var|int64
output = func_263(var_264)
func_265 = relay.Function([var_264], output)
mutated_mod['func_265'] = func_265
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_267 = relay.TupleGetItem(func_24_call(), 1)
call_268 = relay.TupleGetItem(func_26_call(), 1)
var_272 = relay.var("var_272", dtype = "int8", shape = (10,))#candidate|272|(10,)|var|int8
bop_273 = relay.mod(call_267.astype('float64'), relay.reshape(var_272.astype('float64'), relay.shape_of(call_267))) # shape=(10,)
bop_276 = relay.mod(call_268.astype('float64'), relay.reshape(var_272.astype('float64'), relay.shape_of(call_268))) # shape=(10,)
func_209_call = mod.get_global_var('func_209')
func_215_call = mutated_mod.get_global_var('func_215')
const_294 = relay.const(2, dtype = "int64")#candidate|294|()|const|int64
call_293 = relay.TupleGetItem(func_209_call(relay.reshape(call_267.astype('int8'), [10,]), relay.reshape(bop_273.astype('int8'), [10,]), relay.reshape(const_294.astype('int64'), []), relay.reshape(var_272.astype('int8'), [10,]), ), 0)
call_295 = relay.TupleGetItem(func_215_call(relay.reshape(call_267.astype('int8'), [10,]), relay.reshape(bop_273.astype('int8'), [10,]), relay.reshape(const_294.astype('int64'), []), relay.reshape(var_272.astype('int8'), [10,]), ), 0)
func_263_call = mod.get_global_var('func_263')
func_265_call = mutated_mod.get_global_var('func_265')
call_297 = relay.TupleGetItem(func_263_call(relay.reshape(const_294.astype('int64'), [])), 1)
call_298 = relay.TupleGetItem(func_265_call(relay.reshape(const_294.astype('int64'), [])), 1)
func_263_call = mod.get_global_var('func_263')
func_265_call = mutated_mod.get_global_var('func_265')
call_299 = relay.TupleGetItem(func_263_call(relay.reshape(const_294.astype('int64'), [])), 2)
call_300 = relay.TupleGetItem(func_265_call(relay.reshape(const_294.astype('int64'), [])), 2)
bop_301 = relay.bitwise_xor(var_272.astype('int32'), relay.reshape(call_293.astype('int32'), relay.shape_of(var_272))) # shape=(10,)
bop_304 = relay.bitwise_xor(var_272.astype('int32'), relay.reshape(call_295.astype('int32'), relay.shape_of(var_272))) # shape=(10,)
bop_306 = relay.maximum(call_267.astype('float32'), relay.reshape(call_297.astype('float32'), relay.shape_of(call_267))) # shape=(10,)
bop_309 = relay.maximum(call_268.astype('float32'), relay.reshape(call_298.astype('float32'), relay.shape_of(call_268))) # shape=(10,)
bop_315 = relay.logical_and(bop_306.astype('bool'), relay.reshape(bop_301.astype('bool'), relay.shape_of(bop_306))) # shape=(10,)
bop_318 = relay.logical_and(bop_309.astype('bool'), relay.reshape(bop_304.astype('bool'), relay.shape_of(bop_309))) # shape=(10,)
var_323 = relay.var("var_323", dtype = "float64", shape = (10,))#candidate|323|(10,)|var|float64
bop_324 = relay.bitwise_and(bop_273.astype('uint16'), relay.reshape(var_323.astype('uint16'), relay.shape_of(bop_273))) # shape=(10,)
bop_327 = relay.bitwise_and(bop_276.astype('uint16'), relay.reshape(var_323.astype('uint16'), relay.shape_of(bop_276))) # shape=(10,)
bop_329 = relay.floor_divide(call_299.astype('float32'), call_297.astype('float32')) # shape=(10,)
bop_332 = relay.floor_divide(call_300.astype('float32'), call_298.astype('float32')) # shape=(10,)
output = relay.Tuple([const_294,bop_315,bop_324,bop_329,])
output2 = relay.Tuple([const_294,bop_318,bop_327,bop_332,])
func_338 = relay.Function([var_272,var_323,], output)
mod['func_338'] = func_338
mod = relay.transform.InferType()(mod)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
func_338_call = mutated_mod.get_global_var('func_338')
var_340 = relay.var("var_340", dtype = "int8", shape = (10,))#candidate|340|(10,)|var|int8
var_341 = relay.var("var_341", dtype = "float64", shape = (10,))#candidate|341|(10,)|var|float64
call_339 = func_338_call(var_340,var_341,)
output = call_339
func_342 = relay.Function([var_340,var_341,], output)
mutated_mod['func_342'] = func_342
mutated_mod = relay.transform.InferType()(mutated_mod)
var_396 = relay.var("var_396", dtype = "float32", shape = ())#candidate|396|()|var|float32
var_397 = relay.var("var_397", dtype = "float32", shape = (4,))#candidate|397|(4,)|var|float32
bop_398 = relay.power(var_396.astype('float32'), var_397.astype('float32')) # shape=(4,)
uop_410 = relay.sin(bop_398.astype('float64')) # shape=(4,)
output = relay.Tuple([uop_410,])
output2 = relay.Tuple([uop_410,])
func_417 = relay.Function([var_396,var_397,], output)
mod['func_417'] = func_417
mod = relay.transform.InferType()(mod)
var_418 = relay.var("var_418", dtype = "float32", shape = ())#candidate|418|()|var|float32
var_419 = relay.var("var_419", dtype = "float32", shape = (4,))#candidate|419|(4,)|var|float32
output = func_417(var_418,var_419,)
func_420 = relay.Function([var_418,var_419,], output)
mutated_mod['func_420'] = func_420
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_426 = relay.TupleGetItem(func_24_call(), 0)
call_427 = relay.TupleGetItem(func_26_call(), 0)
func_417_call = mod.get_global_var('func_417')
func_420_call = mutated_mod.get_global_var('func_420')
var_460 = relay.var("var_460", dtype = "float32", shape = ())#candidate|460|()|var|float32
var_461 = relay.var("var_461", dtype = "float32", shape = (4,))#candidate|461|(4,)|var|float32
call_459 = relay.TupleGetItem(func_417_call(relay.reshape(var_460.astype('float32'), []), relay.reshape(var_461.astype('float32'), [4,]), ), 0)
call_462 = relay.TupleGetItem(func_420_call(relay.reshape(var_460.astype('float32'), []), relay.reshape(var_461.astype('float32'), [4,]), ), 0)
func_338_call = mod.get_global_var('func_338')
func_342_call = mutated_mod.get_global_var('func_342')
call_465 = relay.TupleGetItem(func_338_call(relay.reshape(call_426.astype('int8'), [10,]), relay.reshape(call_426.astype('float64'), [10,]), ), 3)
call_466 = relay.TupleGetItem(func_342_call(relay.reshape(call_426.astype('int8'), [10,]), relay.reshape(call_426.astype('float64'), [10,]), ), 3)
output = relay.Tuple([call_426,call_459,var_460,var_461,call_465,])
output2 = relay.Tuple([call_427,call_462,var_460,var_461,call_466,])
func_470 = relay.Function([var_460,var_461,], output)
mod['func_470'] = func_470
mod = relay.transform.InferType()(mod)
mutated_mod['func_470'] = func_470
mutated_mod = relay.transform.InferType()(mutated_mod)
func_470_call = mutated_mod.get_global_var('func_470')
var_472 = relay.var("var_472", dtype = "float32", shape = ())#candidate|472|()|var|float32
var_473 = relay.var("var_473", dtype = "float32", shape = (4,))#candidate|473|(4,)|var|float32
call_471 = func_470_call(var_472,var_473,)
output = call_471
func_474 = relay.Function([var_472,var_473,], output)
mutated_mod['func_474'] = func_474
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_549 = relay.TupleGetItem(func_24_call(), 1)
call_550 = relay.TupleGetItem(func_26_call(), 1)
uop_565 = relay.sin(call_549.astype('float64')) # shape=(10,)
uop_567 = relay.sin(call_550.astype('float64')) # shape=(10,)
uop_572 = relay.asin(uop_565.astype('float32')) # shape=(10,)
uop_574 = relay.asin(uop_567.astype('float32')) # shape=(10,)
uop_576 = relay.cos(uop_565.astype('float64')) # shape=(10,)
uop_578 = relay.cos(uop_567.astype('float64')) # shape=(10,)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
const_580 = relay.const(-1, dtype = "int64")#candidate|580|()|const|int64
call_579 = relay.TupleGetItem(func_80_call(relay.reshape(const_580.astype('int64'), []), relay.reshape(call_549.astype('int64'), [10,]), ), 0)
call_581 = relay.TupleGetItem(func_84_call(relay.reshape(const_580.astype('int64'), []), relay.reshape(call_549.astype('int64'), [10,]), ), 0)
uop_582 = relay.acos(uop_565.astype('float32')) # shape=(10,)
uop_584 = relay.acos(uop_567.astype('float32')) # shape=(10,)
output = relay.Tuple([uop_572,uop_576,call_579,const_580,uop_582,])
output2 = relay.Tuple([uop_574,uop_578,call_581,const_580,uop_584,])
func_586 = relay.Function([], output)
mod['func_586'] = func_586
mod = relay.transform.InferType()(mod)
output = func_586()
func_587 = relay.Function([], output)
mutated_mod['func_587'] = func_587
mutated_mod = relay.transform.InferType()(mutated_mod)
var_607 = relay.var("var_607", dtype = "float32", shape = (6, 15, 8))#candidate|607|(6, 15, 8)|var|float32
uop_608 = relay.sqrt(var_607.astype('float32')) # shape=(6, 15, 8)
output = relay.Tuple([uop_608,])
output2 = relay.Tuple([uop_608,])
func_614 = relay.Function([var_607,], output)
mod['func_614'] = func_614
mod = relay.transform.InferType()(mod)
var_615 = relay.var("var_615", dtype = "float32", shape = (6, 15, 8))#candidate|615|(6, 15, 8)|var|float32
output = func_614(var_615)
func_616 = relay.Function([var_615], output)
mutated_mod['func_616'] = func_616
mutated_mod = relay.transform.InferType()(mutated_mod)
const_626 = relay.const([-3.130315,8.653008,5.449479,2.072049,9.915280,0.735706,0.267815,-7.409595,3.980469], dtype = "float32")#candidate|626|(9,)|const|float32
var_627 = relay.var("var_627", dtype = "float32", shape = (9,))#candidate|627|(9,)|var|float32
bop_628 = relay.subtract(const_626.astype('float32'), relay.reshape(var_627.astype('float32'), relay.shape_of(const_626))) # shape=(9,)
uop_634 = relay.tan(var_627.astype('float32')) # shape=(9,)
output = relay.Tuple([bop_628,uop_634,])
output2 = relay.Tuple([bop_628,uop_634,])
func_640 = relay.Function([var_627,], output)
mod['func_640'] = func_640
mod = relay.transform.InferType()(mod)
var_641 = relay.var("var_641", dtype = "float32", shape = (9,))#candidate|641|(9,)|var|float32
output = func_640(var_641)
func_642 = relay.Function([var_641], output)
mutated_mod['func_642'] = func_642
mutated_mod = relay.transform.InferType()(mutated_mod)
var_644 = relay.var("var_644", dtype = "uint64", shape = (7, 3))#candidate|644|(7, 3)|var|uint64
var_645 = relay.var("var_645", dtype = "uint64", shape = (7, 3))#candidate|645|(7, 3)|var|uint64
bop_646 = relay.logical_xor(var_644.astype('uint64'), relay.reshape(var_645.astype('uint64'), relay.shape_of(var_644))) # shape=(7, 3)
uop_651 = relay.log(var_644.astype('float64')) # shape=(7, 3)
uop_661 = relay.acosh(var_644.astype('float64')) # shape=(7, 3)
bop_669 = relay.less(bop_646.astype('bool'), relay.reshape(var_644.astype('bool'), relay.shape_of(bop_646))) # shape=(7, 3)
var_678 = relay.var("var_678", dtype = "float64", shape = (7, 3))#candidate|678|(7, 3)|var|float64
bop_679 = relay.bitwise_or(uop_651.astype('int64'), relay.reshape(var_678.astype('int64'), relay.shape_of(uop_651))) # shape=(7, 3)
bop_685 = relay.floor_mod(var_644.astype('float32'), relay.reshape(uop_661.astype('float32'), relay.shape_of(var_644))) # shape=(7, 3)
output = relay.Tuple([bop_669,bop_679,bop_685,])
output2 = relay.Tuple([bop_669,bop_679,bop_685,])
func_690 = relay.Function([var_644,var_645,var_678,], output)
mod['func_690'] = func_690
mod = relay.transform.InferType()(mod)
var_691 = relay.var("var_691", dtype = "uint64", shape = (7, 3))#candidate|691|(7, 3)|var|uint64
var_692 = relay.var("var_692", dtype = "uint64", shape = (7, 3))#candidate|692|(7, 3)|var|uint64
var_693 = relay.var("var_693", dtype = "float64", shape = (7, 3))#candidate|693|(7, 3)|var|float64
output = func_690(var_691,var_692,var_693,)
func_694 = relay.Function([var_691,var_692,var_693,], output)
mutated_mod['func_694'] = func_694
mutated_mod = relay.transform.InferType()(mutated_mod)
const_699 = relay.const([[[-10,4,8,-4,-2,-4,7,-5,-6,6,-2],[3,4,-7,8,-3,-2,2,6,8,-7,8],[-8,5,-7,9,1,5,10,2,-2,8,1],[-4,8,-4,6,8,-6,7,6,-5,10,2],[-4,7,9,-1,-5,-9,9,-5,5,7,10],[5,2,1,-1,-9,7,-2,9,-9,8,-7],[-1,-10,-10,8,-2,4,-4,-5,9,-10,2],[-3,5,7,-9,-10,7,-7,-10,-6,9,-8],[6,9,-9,-2,-1,2,-8,-1,3,9,5],[6,-4,-6,4,-7,-9,-6,5,9,-4,4]],[[-10,-10,8,3,-9,7,-3,6,8,8,10],[-6,-1,-3,-6,6,1,8,-9,-1,2,-10],[10,4,9,-5,-7,4,-8,2,1,9,-6],[-1,2,1,-7,-10,-7,7,10,5,-7,-5],[7,6,-5,-1,-5,-8,-2,10,-2,-10,-9],[-4,-1,4,9,2,7,3,-4,-7,-7,-6],[1,-10,-8,-1,7,4,8,7,-9,3,-6],[10,-3,1,-9,8,-2,-5,9,7,-5,4],[-9,1,10,6,2,-6,-2,-6,-3,-4,-3],[7,-9,7,-3,7,6,2,2,1,-5,-1]],[[-5,1,-1,5,6,-7,-4,8,9,-1,-1],[-8,2,-7,8,3,4,1,9,-1,-8,9],[-7,9,-2,9,-5,6,-10,-4,2,-2,-3],[4,-1,-8,-7,10,4,8,-2,-9,7,6],[-8,-4,10,-9,-10,-7,-9,-9,2,10,9],[-3,-4,9,8,9,-6,8,-10,10,-9,-1],[2,7,-7,1,3,-8,6,10,-5,-10,3],[-6,3,1,3,-2,-1,-10,-6,-6,-6,-9],[4,-6,10,1,-4,-4,-4,-10,-8,3,-3],[-8,5,7,-10,6,8,4,-8,9,3,8]],[[10,-4,10,8,2,10,1,1,-6,9,8],[1,-10,5,-6,9,-1,-6,10,-2,-1,-2],[-2,6,-1,-3,8,-5,6,2,-5,2,3],[4,-4,-8,1,-9,-4,-4,-10,3,10,-8],[10,-10,8,8,10,7,-2,-10,6,7,-4],[4,-7,4,5,-4,2,7,-6,3,8,-1],[-4,3,-10,-9,-3,-10,8,-8,10,7,5],[-3,-8,6,-6,-6,8,10,2,-4,-10,5],[-2,2,-4,3,-10,2,-1,-6,7,-7,5],[-3,-7,-2,-3,7,-10,5,-8,3,5,10]],[[6,-2,9,-6,-7,1,-7,-2,-2,6,-10],[7,-1,9,-8,9,4,4,1,8,-2,3],[-4,10,6,9,8,5,6,5,-1,-5,8],[-8,1,8,-7,-3,-5,-6,-2,6,-10,-1],[-5,10,3,-2,-7,9,-8,-1,7,5,-10],[5,-7,4,-2,-4,1,2,9,2,-4,-4],[-7,7,-1,-10,-5,-5,-3,-10,1,-3,-5],[-4,-1,-10,9,-9,-1,4,9,4,-6,6],[7,-6,7,-3,-6,-1,-5,-4,-5,3,-1],[9,-9,-6,4,-4,-8,-1,2,1,1,7]],[[-3,10,-5,-3,8,-8,4,1,8,9,8],[2,4,-9,3,10,-6,3,3,-10,9,9],[-4,1,8,7,-2,7,2,8,10,1,-10],[9,-2,6,1,-5,8,6,-8,3,3,5],[-5,-3,2,-3,10,-4,5,6,-7,-10,8],[4,-10,-5,8,10,-2,-8,-7,-7,-5,9],[-2,3,-5,6,-8,9,-7,2,-6,-9,-3],[5,8,-5,4,10,-1,5,5,-1,-4,-2],[7,-8,-4,5,6,7,3,-8,8,-3,2],[-4,1,-1,-6,-8,6,-9,-5,-8,-4,6]],[[7,-2,-3,-3,-5,3,4,3,-6,4,8],[-2,8,5,-5,7,6,-6,-6,-3,9,9],[9,-3,6,5,-3,-9,7,2,7,-5,-1],[-10,-1,10,-9,3,-2,3,6,-3,6,-1],[-6,-4,-4,-5,-9,2,10,-10,-10,8,-9],[-1,8,-7,6,7,3,-6,-2,2,8,1],[-5,-8,6,-6,-5,-2,-10,7,-9,7,-6],[10,4,9,-5,1,-3,-4,4,-6,6,-4],[-6,-4,3,-8,-4,7,8,3,8,3,2],[-2,7,7,-5,2,10,-6,-7,4,-1,-3]],[[-10,-4,8,-2,5,9,4,-7,-5,-10,-5],[1,-1,5,-7,4,-3,-6,6,-3,3,-4],[8,-4,2,-1,2,-1,7,6,-4,-3,-9],[-10,-8,7,-7,-8,4,-1,8,3,1,3],[-10,4,-9,3,-6,4,-5,5,4,8,-5],[-6,10,3,5,-1,-6,1,2,5,6,-1],[-2,5,-10,-6,4,10,-3,-5,-8,7,3],[-1,-1,-2,6,10,-2,6,9,-8,5,-9],[10,9,2,4,-4,-8,-3,-4,8,9,-10],[-3,-6,-4,-1,-1,6,-1,-8,4,1,-8]],[[4,3,7,-3,4,10,9,3,8,10,-6],[2,5,-7,-10,1,9,8,-4,7,-5,10],[-10,3,-8,1,6,-7,1,-7,1,5,-10],[7,3,3,-10,1,5,7,-9,-7,-6,-5],[-2,-8,-2,2,7,7,-7,8,-3,-9,-5],[4,-7,-8,-9,9,-7,4,-1,1,-8,-8],[9,2,4,-2,8,-4,1,9,5,1,5],[9,-2,4,9,6,-9,-1,2,4,1,6],[-10,-4,-2,-6,10,3,-8,-3,10,-6,-8],[3,-2,1,-3,-7,-3,-9,-6,4,-10,8]]], dtype = "int64")#candidate|699|(9, 10, 11)|const|int64
const_700 = relay.const([[[-10,-6,3,7,7,4,-9,-9,-7,-10,-10],[5,-9,-5,9,-2,3,6,-5,9,3,7],[4,-9,-6,10,-5,10,-5,2,10,-3,-3],[1,3,2,-3,8,-3,-8,5,-5,-5,6],[-1,-6,-2,4,-7,5,-4,8,4,-7,-2],[-5,5,9,6,-1,-2,6,5,1,-4,6],[3,-7,9,-5,-9,9,-10,-3,2,1,-3],[9,-8,2,9,2,-1,9,3,10,-4,-5],[5,-3,1,4,3,-4,6,-5,-3,6,-7],[3,-1,-9,1,-9,1,-7,6,-8,5,-1]],[[7,7,-4,5,2,-1,-2,5,-4,10,-8],[10,8,-6,8,-7,5,5,9,-3,8,1],[6,-3,-2,-8,10,6,-3,-3,2,-6,-2],[-8,-10,1,7,-10,-9,5,-6,6,8,3],[6,4,4,3,-10,-10,-4,-10,4,-5,-3],[-6,3,6,-4,5,-7,-6,-9,-8,3,-9],[8,9,-1,4,-7,4,2,1,1,6,5],[8,2,-10,5,-4,-3,6,9,10,2,-6],[-4,-10,3,2,9,-1,-9,8,4,-6,4],[1,-1,9,-4,8,2,-4,8,-4,-7,10]],[[5,9,-1,-4,-8,8,5,3,-2,7,10],[10,7,-7,-6,2,6,-1,-8,-3,2,-6],[-1,-4,3,-9,-3,3,5,-8,-6,4,-4],[10,-4,10,6,-2,-6,-10,-8,-4,-9,4],[-4,7,-3,-5,-6,2,-9,5,-8,-10,3],[-5,-2,3,5,-7,10,-10,-3,-3,-1,-10],[-5,4,1,-10,-8,7,-6,-1,-10,2,2],[-4,-9,5,-9,-8,1,-10,10,7,6,7],[-8,3,-7,-7,-6,-6,-9,-3,-8,-5,-5],[-1,4,2,-4,2,-7,10,-7,-5,-10,-8]],[[-3,-7,3,8,-6,4,7,-10,5,4,6],[-8,-1,-7,-3,1,-7,-7,-2,-10,-3,10],[10,-2,-3,-7,-10,-3,-7,5,-7,-5,7],[-5,-6,9,7,1,9,-10,-2,1,-1,5],[7,7,-6,-6,-6,-4,-4,-6,-4,8,-2],[8,-7,-6,-5,6,10,-4,8,9,-1,-8],[10,7,1,-7,-6,-7,-6,-3,3,-3,-3],[-7,-8,-4,-3,-5,-4,-5,8,8,-9,-8],[7,5,1,10,4,-4,6,-1,-10,-10,-9],[-9,-8,-2,1,1,-1,7,4,-3,-5,-3]],[[5,-9,-10,-9,-5,5,-7,-2,7,8,2],[8,-5,6,-8,-2,-3,-7,-2,-10,-7,2],[-2,-6,-10,4,2,-7,5,-6,2,-7,8],[8,2,-9,-4,5,-7,-4,-7,-6,2,-3],[-9,-1,6,-8,6,6,-5,-9,10,-5,-6],[4,1,2,5,2,6,-9,-3,-7,-2,-7],[10,5,1,-8,1,10,-2,-5,8,5,3],[6,-8,5,8,8,-7,9,-3,4,-9,-5],[-7,-9,8,-10,-7,-6,-1,-6,10,-8,-7],[6,-4,-3,5,3,8,8,8,-6,-6,2]],[[3,4,-4,-2,-9,-2,-9,5,4,-7,-4],[8,-4,-8,1,6,10,7,4,8,10,-6],[4,2,-3,-10,-5,-10,-7,7,-8,1,-5],[-5,-10,2,-2,10,4,-6,2,-6,6,-3],[-2,-2,7,10,2,3,-8,-5,-7,-8,8],[10,10,4,-9,3,-1,-4,8,2,-1,3],[2,-2,2,2,4,7,-5,1,2,-5,-9],[-3,3,7,2,5,-8,5,-7,-10,-8,-4],[-2,6,9,8,8,-4,-1,-7,3,-9,7],[2,2,4,1,-10,2,-8,7,10,-1,8]],[[-7,5,7,2,-9,-7,-8,-10,8,3,-5],[-3,-9,8,9,-2,-1,8,-2,-3,7,-3],[-10,4,-4,-7,8,-1,-3,6,7,8,-4],[7,7,10,-9,-8,7,-8,-9,-10,-7,3],[-2,-1,5,-6,-6,-7,-2,-8,2,-9,-10],[-10,1,5,2,7,-4,-2,8,10,-10,8],[5,-10,-3,-6,5,-1,7,-6,1,3,-4],[6,3,-9,-2,3,-4,-4,4,-2,-5,7],[7,1,1,-5,8,-8,-7,4,-3,-9,-1],[-5,-3,6,-10,-4,-8,-6,10,5,-5,1]],[[-1,3,1,2,5,7,-4,9,-5,2,-7],[-4,2,-7,4,8,2,-9,3,6,-9,-9],[-7,-9,-3,-8,9,-9,3,5,3,-3,7],[7,5,10,-8,9,1,-8,8,5,9,-8],[10,-4,-9,8,-10,5,-10,-9,8,-2,5],[1,5,3,-2,8,6,-2,8,-10,7,-4],[-4,3,-1,1,4,-1,10,5,-9,-1,-5],[-10,5,3,-9,8,10,-5,8,-8,8,3],[5,9,1,9,-4,-10,2,4,-2,5,6],[2,-8,8,2,-5,7,1,5,-3,2,2]],[[9,-5,8,3,-2,10,5,-7,9,-4,-5],[4,8,10,3,1,-6,2,-6,8,10,-7],[5,-2,5,1,-4,9,4,-1,1,-8,-8],[-1,-3,-2,8,-10,3,-5,9,-8,1,-6],[-9,2,8,-7,8,3,-10,6,6,10,7],[10,1,10,7,7,-2,-6,-8,-9,4,10],[8,-1,-2,-6,6,7,-3,-9,5,-6,4],[-1,6,9,-3,3,6,6,3,-9,9,6],[4,-4,-5,10,-2,-4,-1,8,3,-3,-5],[9,4,-7,-5,-7,-2,2,1,7,-10,-4]]], dtype = "int64")#candidate|700|(9, 10, 11)|const|int64
bop_701 = relay.bitwise_and(const_699.astype('int64'), relay.reshape(const_700.astype('int64'), relay.shape_of(const_699))) # shape=(9, 10, 11)
bop_707 = relay.equal(bop_701.astype('bool'), relay.reshape(const_700.astype('bool'), relay.shape_of(bop_701))) # shape=(9, 10, 11)
var_713 = relay.var("var_713", dtype = "int64", shape = (9, 10, 11))#candidate|713|(9, 10, 11)|var|int64
bop_714 = relay.power(bop_701.astype('float32'), relay.reshape(var_713.astype('float32'), relay.shape_of(bop_701))) # shape=(9, 10, 11)
output = relay.Tuple([bop_707,bop_714,])
output2 = relay.Tuple([bop_707,bop_714,])
func_724 = relay.Function([var_713,], output)
mod['func_724'] = func_724
mod = relay.transform.InferType()(mod)
mutated_mod['func_724'] = func_724
mutated_mod = relay.transform.InferType()(mutated_mod)
var_725 = relay.var("var_725", dtype = "int64", shape = (9, 10, 11))#candidate|725|(9, 10, 11)|var|int64
func_724_call = mutated_mod.get_global_var('func_724')
call_726 = func_724_call(var_725)
output = call_726
func_727 = relay.Function([var_725], output)
mutated_mod['func_727'] = func_727
mutated_mod = relay.transform.InferType()(mutated_mod)
func_586_call = mod.get_global_var('func_586')
func_587_call = mutated_mod.get_global_var('func_587')
call_744 = relay.TupleGetItem(func_586_call(), 4)
call_745 = relay.TupleGetItem(func_587_call(), 4)
output = call_744
output2 = call_745
func_751 = relay.Function([], output)
mod['func_751'] = func_751
mod = relay.transform.InferType()(mod)
output = func_751()
func_752 = relay.Function([], output)
mutated_mod['func_752'] = func_752
mutated_mod = relay.transform.InferType()(mutated_mod)
const_758 = relay.const([[4.039303,-4.563712,1.498938,5.642619,6.042844,-7.724842,-1.810108,0.846451,-7.354602,-3.496856,2.264963,-5.277393,-3.147460,-4.933109,-8.740352],[0.080805,-5.111881,9.835778,5.720509,1.571323,6.021686,-6.057663,-1.542919,-5.651694,9.408375,-6.566456,-9.516301,-2.757056,9.818893,4.981675],[-3.537776,7.199010,-1.901265,-5.830255,8.544355,2.726406,-5.868048,-6.859271,9.857870,1.259353,-4.577531,2.963330,0.999563,8.053424,0.953423]], dtype = "float64")#candidate|758|(3, 15)|const|float64
uop_759 = relay.cosh(const_758.astype('float64')) # shape=(3, 15)
const_761 = relay.const([[-0.985130,4.697398,-8.935854,6.138930,-1.150978,-0.326900,-7.407030,-1.236629,-6.188390,7.746584,-6.532192,8.017895,4.868312,-6.679178,0.458775],[-1.615177,-5.066451,-9.434637,-1.590128,9.278669,-9.713338,-1.331188,2.369261,-4.648184,-3.235649,5.219561,3.211791,-1.583725,3.388185,-5.676355],[-5.674354,-8.202006,-0.174595,-2.281853,-6.296240,-8.461760,8.884367,8.934241,2.787385,-2.290853,-9.614729,1.435945,-4.239947,-6.272666,-1.700484]], dtype = "float64")#candidate|761|(3, 15)|const|float64
bop_762 = relay.floor_divide(uop_759.astype('float32'), relay.reshape(const_761.astype('float32'), relay.shape_of(uop_759))) # shape=(3, 15)
output = bop_762
output2 = bop_762
func_765 = relay.Function([], output)
mod['func_765'] = func_765
mod = relay.transform.InferType()(mod)
output = func_765()
func_766 = relay.Function([], output)
mutated_mod['func_766'] = func_766
mutated_mod = relay.transform.InferType()(mutated_mod)
var_772 = relay.var("var_772", dtype = "float64", shape = (14,))#candidate|772|(14,)|var|float64
uop_773 = relay.log10(var_772.astype('float64')) # shape=(14,)
uop_775 = relay.asin(uop_773.astype('float32')) # shape=(14,)
uop_778 = relay.atanh(uop_773.astype('float32')) # shape=(14,)
const_781 = relay.const([2.017998,-6.038783,-4.603730,2.936579,7.778392,-6.200148,1.256039,-0.361269,6.765911,-1.646293,-0.180186,-8.910280,4.857733,8.817465], dtype = "float32")#candidate|781|(14,)|const|float32
bop_782 = relay.logical_and(uop_775.astype('bool'), relay.reshape(const_781.astype('bool'), relay.shape_of(uop_775))) # shape=(14,)
bop_796 = relay.mod(bop_782.astype('float32'), relay.reshape(uop_773.astype('float32'), relay.shape_of(bop_782))) # shape=(14,)
bop_799 = relay.multiply(uop_773.astype('int8'), relay.reshape(const_781.astype('int8'), relay.shape_of(uop_773))) # shape=(14,)
bop_802 = relay.maximum(const_781.astype('int32'), relay.reshape(uop_775.astype('int32'), relay.shape_of(const_781))) # shape=(14,)
func_640_call = mod.get_global_var('func_640')
func_642_call = mutated_mod.get_global_var('func_642')
var_806 = relay.var("var_806", dtype = "float32", shape = (1, 9))#candidate|806|(1, 9)|var|float32
call_805 = relay.TupleGetItem(func_640_call(relay.reshape(var_806.astype('float32'), [9,])), 0)
call_807 = relay.TupleGetItem(func_642_call(relay.reshape(var_806.astype('float32'), [9,])), 0)
output = relay.Tuple([uop_778,bop_796,bop_799,bop_802,call_805,var_806,])
output2 = relay.Tuple([uop_778,bop_796,bop_799,bop_802,call_807,var_806,])
func_820 = relay.Function([var_772,var_806,], output)
mod['func_820'] = func_820
mod = relay.transform.InferType()(mod)
mutated_mod['func_820'] = func_820
mutated_mod = relay.transform.InferType()(mutated_mod)
func_820_call = mutated_mod.get_global_var('func_820')
var_822 = relay.var("var_822", dtype = "float64", shape = (14,))#candidate|822|(14,)|var|float64
var_823 = relay.var("var_823", dtype = "float32", shape = (1, 9))#candidate|823|(1, 9)|var|float32
call_821 = func_820_call(var_822,var_823,)
output = call_821
func_824 = relay.Function([var_822,var_823,], output)
mutated_mod['func_824'] = func_824
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_844 = relay.TupleGetItem(func_24_call(), 0)
call_845 = relay.TupleGetItem(func_26_call(), 0)
output = relay.Tuple([call_844,])
output2 = relay.Tuple([call_845,])
func_871 = relay.Function([], output)
mod['func_871'] = func_871
mod = relay.transform.InferType()(mod)
output = func_871()
func_872 = relay.Function([], output)
mutated_mod['func_872'] = func_872
mutated_mod = relay.transform.InferType()(mutated_mod)
var_898 = relay.var("var_898", dtype = "float32", shape = (1, 14))#candidate|898|(1, 14)|var|float32
uop_899 = relay.tan(var_898.astype('float32')) # shape=(1, 14)
uop_904 = relay.rsqrt(uop_899.astype('float64')) # shape=(1, 14)
uop_907 = relay.asinh(uop_904.astype('float64')) # shape=(1, 14)
uop_909 = relay.exp(uop_907.astype('float32')) # shape=(1, 14)
uop_912 = relay.atan(uop_907.astype('float64')) # shape=(1, 14)
var_914 = relay.var("var_914", dtype = "float64", shape = (5, 14))#candidate|914|(5, 14)|var|float64
bop_915 = relay.greater_equal(uop_907.astype('bool'), var_914.astype('bool')) # shape=(5, 14)
uop_918 = relay.sqrt(uop_912.astype('float64')) # shape=(1, 14)
output = relay.Tuple([uop_909,bop_915,uop_918,])
output2 = relay.Tuple([uop_909,bop_915,uop_918,])
func_925 = relay.Function([var_898,var_914,], output)
mod['func_925'] = func_925
mod = relay.transform.InferType()(mod)
var_926 = relay.var("var_926", dtype = "float32", shape = (1, 14))#candidate|926|(1, 14)|var|float32
var_927 = relay.var("var_927", dtype = "float64", shape = (5, 14))#candidate|927|(5, 14)|var|float64
output = func_925(var_926,var_927,)
func_928 = relay.Function([var_926,var_927,], output)
mutated_mod['func_928'] = func_928
mutated_mod = relay.transform.InferType()(mutated_mod)
const_930 = relay.const(1, dtype = "int32")#candidate|930|()|const|int32
var_931 = relay.var("var_931", dtype = "int32", shape = (6,))#candidate|931|(6,)|var|int32
bop_932 = relay.maximum(const_930.astype('int32'), var_931.astype('int32')) # shape=(6,)
output = relay.Tuple([bop_932,])
output2 = relay.Tuple([bop_932,])
func_937 = relay.Function([var_931,], output)
mod['func_937'] = func_937
mod = relay.transform.InferType()(mod)
mutated_mod['func_937'] = func_937
mutated_mod = relay.transform.InferType()(mutated_mod)
var_938 = relay.var("var_938", dtype = "int32", shape = (6,))#candidate|938|(6,)|var|int32
func_937_call = mutated_mod.get_global_var('func_937')
call_939 = func_937_call(var_938)
output = call_939
func_940 = relay.Function([var_938], output)
mutated_mod['func_940'] = func_940
mutated_mod = relay.transform.InferType()(mutated_mod)
var_946 = relay.var("var_946", dtype = "float32", shape = (14, 2))#candidate|946|(14, 2)|var|float32
const_947 = relay.const([[5.835862,5.646365],[-8.531099,-0.792820],[-2.030115,-6.358767],[-9.875157,4.599303],[-6.663033,-1.925644],[-3.776434,-1.602565],[2.101599,9.163803],[-9.976473,-0.759314],[2.495058,4.541915],[2.504316,8.740824],[-9.498697,-2.613215],[-3.270472,8.384093],[8.921297,2.413839],[-8.474671,-5.207032]], dtype = "float32")#candidate|947|(14, 2)|const|float32
bop_948 = relay.equal(var_946.astype('bool'), relay.reshape(const_947.astype('bool'), relay.shape_of(var_946))) # shape=(14, 2)
uop_951 = relay.log10(bop_948.astype('float32')) # shape=(14, 2)
output = relay.Tuple([uop_951,])
output2 = relay.Tuple([uop_951,])
func_954 = relay.Function([var_946,], output)
mod['func_954'] = func_954
mod = relay.transform.InferType()(mod)
var_955 = relay.var("var_955", dtype = "float32", shape = (14, 2))#candidate|955|(14, 2)|var|float32
output = func_954(var_955)
func_956 = relay.Function([var_955], output)
mutated_mod['func_956'] = func_956
mutated_mod = relay.transform.InferType()(mutated_mod)
const_972 = relay.const([9.306426,-4.987407,-2.882416,8.411012], dtype = "float64")#candidate|972|(4,)|const|float64
var_973 = relay.var("var_973", dtype = "float64", shape = (4,))#candidate|973|(4,)|var|float64
bop_974 = relay.floor_mod(const_972.astype('float64'), relay.reshape(var_973.astype('float64'), relay.shape_of(const_972))) # shape=(4,)
func_470_call = mod.get_global_var('func_470')
func_474_call = mutated_mod.get_global_var('func_474')
var_982 = relay.var("var_982", dtype = "float32", shape = ())#candidate|982|()|var|float32
call_981 = relay.TupleGetItem(func_470_call(relay.reshape(var_982.astype('float32'), []), relay.reshape(bop_974.astype('float32'), [4,]), ), 4)
call_983 = relay.TupleGetItem(func_474_call(relay.reshape(var_982.astype('float32'), []), relay.reshape(bop_974.astype('float32'), [4,]), ), 4)
func_470_call = mod.get_global_var('func_470')
func_474_call = mutated_mod.get_global_var('func_474')
call_986 = relay.TupleGetItem(func_470_call(relay.reshape(var_982.astype('float32'), []), relay.reshape(const_972.astype('float32'), [4,]), ), 2)
call_987 = relay.TupleGetItem(func_474_call(relay.reshape(var_982.astype('float32'), []), relay.reshape(const_972.astype('float32'), [4,]), ), 2)
bop_989 = relay.maximum(bop_974.astype('uint32'), relay.reshape(const_972.astype('uint32'), relay.shape_of(bop_974))) # shape=(4,)
uop_995 = relay.atanh(bop_974.astype('float64')) # shape=(4,)
output = relay.Tuple([call_981,var_982,call_986,bop_989,uop_995,])
output2 = relay.Tuple([call_983,var_982,call_987,bop_989,uop_995,])
func_999 = relay.Function([var_973,var_982,], output)
mod['func_999'] = func_999
mod = relay.transform.InferType()(mod)
mutated_mod['func_999'] = func_999
mutated_mod = relay.transform.InferType()(mutated_mod)
func_999_call = mutated_mod.get_global_var('func_999')
var_1001 = relay.var("var_1001", dtype = "float64", shape = (4,))#candidate|1001|(4,)|var|float64
var_1002 = relay.var("var_1002", dtype = "float32", shape = ())#candidate|1002|()|var|float32
call_1000 = func_999_call(var_1001,var_1002,)
output = call_1000
func_1003 = relay.Function([var_1001,var_1002,], output)
mutated_mod['func_1003'] = func_1003
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1044 = relay.var("var_1044", dtype = "float32", shape = (15, 5))#candidate|1044|(15, 5)|var|float32
uop_1045 = relay.sinh(var_1044.astype('float32')) # shape=(15, 5)
uop_1048 = relay.acosh(uop_1045.astype('float64')) # shape=(15, 5)
func_765_call = mod.get_global_var('func_765')
func_766_call = mutated_mod.get_global_var('func_766')
call_1051 = func_765_call()
call_1052 = func_765_call()
var_1053 = relay.var("var_1053", dtype = "float64", shape = (15, 5))#candidate|1053|(15, 5)|var|float64
bop_1054 = relay.subtract(uop_1048.astype('int64'), relay.reshape(var_1053.astype('int64'), relay.shape_of(uop_1048))) # shape=(15, 5)
func_338_call = mod.get_global_var('func_338')
func_342_call = mutated_mod.get_global_var('func_342')
var_1058 = relay.var("var_1058", dtype = "int8", shape = (10,))#candidate|1058|(10,)|var|int8
call_1057 = relay.TupleGetItem(func_338_call(relay.reshape(var_1058.astype('int8'), [10,]), relay.reshape(var_1058.astype('float64'), [10,]), ), 2)
call_1059 = relay.TupleGetItem(func_342_call(relay.reshape(var_1058.astype('int8'), [10,]), relay.reshape(var_1058.astype('float64'), [10,]), ), 2)
uop_1062 = relay.cosh(uop_1048.astype('float32')) # shape=(15, 5)
func_871_call = mod.get_global_var('func_871')
func_872_call = mutated_mod.get_global_var('func_872')
call_1064 = relay.TupleGetItem(func_871_call(), 0)
call_1065 = relay.TupleGetItem(func_872_call(), 0)
uop_1067 = relay.log(uop_1062.astype('float32')) # shape=(15, 5)
bop_1069 = relay.multiply(uop_1067.astype('uint8'), relay.reshape(bop_1054.astype('uint8'), relay.shape_of(uop_1067))) # shape=(15, 5)
bop_1074 = relay.left_shift(uop_1062.astype('int16'), relay.reshape(bop_1054.astype('int16'), relay.shape_of(uop_1062))) # shape=(15, 5)
uop_1078 = relay.atanh(bop_1069.astype('float64')) # shape=(15, 5)
bop_1080 = relay.mod(uop_1067.astype('float32'), relay.reshape(uop_1078.astype('float32'), relay.shape_of(uop_1067))) # shape=(15, 5)
uop_1086 = relay.exp(uop_1078.astype('float32')) # shape=(15, 5)
uop_1088 = relay.sigmoid(uop_1086.astype('float32')) # shape=(15, 5)
bop_1090 = relay.bitwise_xor(uop_1088.astype('int16'), relay.reshape(var_1044.astype('int16'), relay.shape_of(uop_1088))) # shape=(15, 5)
output = relay.Tuple([call_1051,call_1057,var_1058,call_1064,bop_1074,bop_1080,bop_1090,])
output2 = relay.Tuple([call_1052,call_1059,var_1058,call_1065,bop_1074,bop_1080,bop_1090,])
func_1094 = relay.Function([var_1044,var_1053,var_1058,], output)
mod['func_1094'] = func_1094
mod = relay.transform.InferType()(mod)
var_1095 = relay.var("var_1095", dtype = "float32", shape = (15, 5))#candidate|1095|(15, 5)|var|float32
var_1096 = relay.var("var_1096", dtype = "float64", shape = (15, 5))#candidate|1096|(15, 5)|var|float64
var_1097 = relay.var("var_1097", dtype = "int8", shape = (10,))#candidate|1097|(10,)|var|int8
output = func_1094(var_1095,var_1096,var_1097,)
func_1098 = relay.Function([var_1095,var_1096,var_1097,], output)
mutated_mod['func_1098'] = func_1098
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1120 = relay.var("var_1120", dtype = "float32", shape = (2, 8))#candidate|1120|(2, 8)|var|float32
uop_1121 = relay.erf(var_1120.astype('float32')) # shape=(2, 8)
var_1123 = relay.var("var_1123", dtype = "float32", shape = (2, 8))#candidate|1123|(2, 8)|var|float32
bop_1124 = relay.mod(uop_1121.astype('float64'), relay.reshape(var_1123.astype('float64'), relay.shape_of(uop_1121))) # shape=(2, 8)
func_765_call = mod.get_global_var('func_765')
func_766_call = mutated_mod.get_global_var('func_766')
call_1127 = func_765_call()
call_1128 = func_765_call()
func_586_call = mod.get_global_var('func_586')
func_587_call = mutated_mod.get_global_var('func_587')
call_1133 = relay.TupleGetItem(func_586_call(), 0)
call_1134 = relay.TupleGetItem(func_587_call(), 0)
output = relay.Tuple([bop_1124,call_1127,call_1133,])
output2 = relay.Tuple([bop_1124,call_1128,call_1134,])
func_1135 = relay.Function([var_1120,var_1123,], output)
mod['func_1135'] = func_1135
mod = relay.transform.InferType()(mod)
var_1136 = relay.var("var_1136", dtype = "float32", shape = (2, 8))#candidate|1136|(2, 8)|var|float32
var_1137 = relay.var("var_1137", dtype = "float32", shape = (2, 8))#candidate|1137|(2, 8)|var|float32
output = func_1135(var_1136,var_1137,)
func_1138 = relay.Function([var_1136,var_1137,], output)
mutated_mod['func_1138'] = func_1138
mutated_mod = relay.transform.InferType()(mutated_mod)
func_586_call = mod.get_global_var('func_586')
func_587_call = mutated_mod.get_global_var('func_587')
call_1152 = relay.TupleGetItem(func_586_call(), 3)
call_1153 = relay.TupleGetItem(func_587_call(), 3)
output = relay.Tuple([call_1152,])
output2 = relay.Tuple([call_1153,])
func_1155 = relay.Function([], output)
mod['func_1155'] = func_1155
mod = relay.transform.InferType()(mod)
output = func_1155()
func_1156 = relay.Function([], output)
mutated_mod['func_1156'] = func_1156
mutated_mod = relay.transform.InferType()(mutated_mod)
func_765_call = mod.get_global_var('func_765')
func_766_call = mutated_mod.get_global_var('func_766')
call_1174 = func_765_call()
call_1175 = func_765_call()
func_263_call = mod.get_global_var('func_263')
func_265_call = mutated_mod.get_global_var('func_265')
const_1180 = relay.const(7, dtype = "int64")#candidate|1180|()|const|int64
call_1179 = relay.TupleGetItem(func_263_call(relay.reshape(const_1180.astype('int64'), [])), 1)
call_1181 = relay.TupleGetItem(func_265_call(relay.reshape(const_1180.astype('int64'), [])), 1)
output = relay.Tuple([call_1174,call_1179,const_1180,])
output2 = relay.Tuple([call_1175,call_1181,const_1180,])
func_1183 = relay.Function([], output)
mod['func_1183'] = func_1183
mod = relay.transform.InferType()(mod)
mutated_mod['func_1183'] = func_1183
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1183_call = mutated_mod.get_global_var('func_1183')
call_1184 = func_1183_call()
output = call_1184
func_1185 = relay.Function([], output)
mutated_mod['func_1185'] = func_1185
mutated_mod = relay.transform.InferType()(mutated_mod)
func_871_call = mod.get_global_var('func_871')
func_872_call = mutated_mod.get_global_var('func_872')
call_1189 = relay.TupleGetItem(func_871_call(), 0)
call_1190 = relay.TupleGetItem(func_872_call(), 0)
uop_1207 = relay.asinh(call_1189.astype('float64')) # shape=(10,)
uop_1209 = relay.asinh(call_1190.astype('float64')) # shape=(10,)
bop_1211 = relay.less_equal(call_1189.astype('bool'), relay.reshape(uop_1207.astype('bool'), relay.shape_of(call_1189))) # shape=(10,)
bop_1214 = relay.less_equal(call_1190.astype('bool'), relay.reshape(uop_1209.astype('bool'), relay.shape_of(call_1190))) # shape=(10,)
func_1135_call = mod.get_global_var('func_1135')
func_1138_call = mutated_mod.get_global_var('func_1138')
const_1216 = relay.const([-2.316317,-6.279897,0.518071,-3.667129,-6.941981,6.203616,2.822572,6.803649,-2.478594,7.233443,8.352138,-5.293894,-5.721412,-1.320440,-5.483285,-6.567137], dtype = "float32")#candidate|1216|(16,)|const|float32
call_1215 = relay.TupleGetItem(func_1135_call(relay.reshape(const_1216.astype('float32'), [2, 8]), relay.reshape(const_1216.astype('float32'), [2, 8]), ), 1)
call_1217 = relay.TupleGetItem(func_1138_call(relay.reshape(const_1216.astype('float32'), [2, 8]), relay.reshape(const_1216.astype('float32'), [2, 8]), ), 1)
output = relay.Tuple([bop_1211,call_1215,const_1216,])
output2 = relay.Tuple([bop_1214,call_1217,const_1216,])
func_1220 = relay.Function([], output)
mod['func_1220'] = func_1220
mod = relay.transform.InferType()(mod)
mutated_mod['func_1220'] = func_1220
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1220_call = mutated_mod.get_global_var('func_1220')
call_1221 = func_1220_call()
output = call_1221
func_1222 = relay.Function([], output)
mutated_mod['func_1222'] = func_1222
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_1223 = relay.TupleGetItem(func_24_call(), 1)
call_1224 = relay.TupleGetItem(func_26_call(), 1)
output = call_1223
output2 = call_1224
func_1237 = relay.Function([], output)
mod['func_1237'] = func_1237
mod = relay.transform.InferType()(mod)
output = func_1237()
func_1238 = relay.Function([], output)
mutated_mod['func_1238'] = func_1238
mutated_mod = relay.transform.InferType()(mutated_mod)
func_871_call = mod.get_global_var('func_871')
func_872_call = mutated_mod.get_global_var('func_872')
call_1249 = relay.TupleGetItem(func_871_call(), 0)
call_1250 = relay.TupleGetItem(func_872_call(), 0)
func_925_call = mod.get_global_var('func_925')
func_928_call = mutated_mod.get_global_var('func_928')
var_1254 = relay.var("var_1254", dtype = "float32", shape = (14,))#candidate|1254|(14,)|var|float32
var_1255 = relay.var("var_1255", dtype = "float64", shape = (70,))#candidate|1255|(70,)|var|float64
call_1253 = relay.TupleGetItem(func_925_call(relay.reshape(var_1254.astype('float32'), [1, 14]), relay.reshape(var_1255.astype('float64'), [5, 14]), ), 2)
call_1256 = relay.TupleGetItem(func_928_call(relay.reshape(var_1254.astype('float32'), [1, 14]), relay.reshape(var_1255.astype('float64'), [5, 14]), ), 2)
uop_1268 = relay.log2(var_1255.astype('float32')) # shape=(70,)
bop_1270 = relay.bitwise_xor(uop_1268.astype('int8'), relay.reshape(var_1255.astype('int8'), relay.shape_of(uop_1268))) # shape=(70,)
bop_1276 = relay.greater(uop_1268.astype('bool'), relay.reshape(bop_1270.astype('bool'), relay.shape_of(uop_1268))) # shape=(70,)
func_417_call = mod.get_global_var('func_417')
func_420_call = mutated_mod.get_global_var('func_420')
const_1280 = relay.const(5.531971, dtype = "float32")#candidate|1280|()|const|float32
const_1281 = relay.const([-9.188705,2.156513,7.407743,4.419983], dtype = "float32")#candidate|1281|(4,)|const|float32
call_1279 = relay.TupleGetItem(func_417_call(relay.reshape(const_1280.astype('float32'), []), relay.reshape(const_1281.astype('float32'), [4,]), ), 0)
call_1282 = relay.TupleGetItem(func_420_call(relay.reshape(const_1280.astype('float32'), []), relay.reshape(const_1281.astype('float32'), [4,]), ), 0)
bop_1284 = relay.logical_and(uop_1268.astype('bool'), relay.reshape(bop_1270.astype('bool'), relay.shape_of(uop_1268))) # shape=(70,)
uop_1288 = relay.asinh(bop_1276.astype('float32')) # shape=(70,)
output = relay.Tuple([call_1249,call_1253,var_1254,call_1279,const_1280,const_1281,bop_1284,uop_1288,])
output2 = relay.Tuple([call_1250,call_1256,var_1254,call_1282,const_1280,const_1281,bop_1284,uop_1288,])
func_1290 = relay.Function([var_1254,var_1255,], output)
mod['func_1290'] = func_1290
mod = relay.transform.InferType()(mod)
var_1291 = relay.var("var_1291", dtype = "float32", shape = (14,))#candidate|1291|(14,)|var|float32
var_1292 = relay.var("var_1292", dtype = "float64", shape = (70,))#candidate|1292|(70,)|var|float64
output = func_1290(var_1291,var_1292,)
func_1293 = relay.Function([var_1291,var_1292,], output)
mutated_mod['func_1293'] = func_1293
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1295 = relay.var("var_1295", dtype = "float32", shape = (4, 10))#candidate|1295|(4, 10)|var|float32
uop_1296 = relay.sigmoid(var_1295.astype('float32')) # shape=(4, 10)
uop_1301 = relay.log(uop_1296.astype('float32')) # shape=(4, 10)
output = relay.Tuple([uop_1301,])
output2 = relay.Tuple([uop_1301,])
func_1303 = relay.Function([var_1295,], output)
mod['func_1303'] = func_1303
mod = relay.transform.InferType()(mod)
mutated_mod['func_1303'] = func_1303
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1304 = relay.var("var_1304", dtype = "float32", shape = (4, 10))#candidate|1304|(4, 10)|var|float32
func_1303_call = mutated_mod.get_global_var('func_1303')
call_1305 = func_1303_call(var_1304)
output = call_1305
func_1306 = relay.Function([var_1304], output)
mutated_mod['func_1306'] = func_1306
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1319 = relay.var("var_1319", dtype = "int32", shape = (9, 12))#candidate|1319|(9, 12)|var|int32
var_1320 = relay.var("var_1320", dtype = "int32", shape = (9, 12))#candidate|1320|(9, 12)|var|int32
bop_1321 = relay.less(var_1319.astype('bool'), relay.reshape(var_1320.astype('bool'), relay.shape_of(var_1319))) # shape=(9, 12)
uop_1324 = relay.erf(bop_1321.astype('float32')) # shape=(9, 12)
uop_1330 = relay.tan(var_1320.astype('float32')) # shape=(9, 12)
bop_1332 = relay.floor_divide(uop_1330.astype('float64'), relay.reshape(uop_1324.astype('float64'), relay.shape_of(uop_1330))) # shape=(9, 12)
func_999_call = mod.get_global_var('func_999')
func_1003_call = mutated_mod.get_global_var('func_1003')
var_1339 = relay.var("var_1339", dtype = "float64", shape = (4,))#candidate|1339|(4,)|var|float64
var_1340 = relay.var("var_1340", dtype = "float32", shape = ())#candidate|1340|()|var|float32
call_1338 = relay.TupleGetItem(func_999_call(relay.reshape(var_1339.astype('float64'), [4,]), relay.reshape(var_1340.astype('float32'), []), ), 0)
call_1341 = relay.TupleGetItem(func_1003_call(relay.reshape(var_1339.astype('float64'), [4,]), relay.reshape(var_1340.astype('float32'), []), ), 0)
uop_1346 = relay.log(uop_1324.astype('float64')) # shape=(9, 12)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_1352 = relay.TupleGetItem(func_24_call(), 0)
call_1353 = relay.TupleGetItem(func_26_call(), 0)
output = relay.Tuple([bop_1332,call_1338,var_1339,var_1340,uop_1346,call_1352,])
output2 = relay.Tuple([bop_1332,call_1341,var_1339,var_1340,uop_1346,call_1353,])
func_1355 = relay.Function([var_1319,var_1320,var_1339,var_1340,], output)
mod['func_1355'] = func_1355
mod = relay.transform.InferType()(mod)
mutated_mod['func_1355'] = func_1355
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1355_call = mutated_mod.get_global_var('func_1355')
var_1357 = relay.var("var_1357", dtype = "int32", shape = (9, 12))#candidate|1357|(9, 12)|var|int32
var_1358 = relay.var("var_1358", dtype = "int32", shape = (9, 12))#candidate|1358|(9, 12)|var|int32
var_1359 = relay.var("var_1359", dtype = "float64", shape = (4,))#candidate|1359|(4,)|var|float64
var_1360 = relay.var("var_1360", dtype = "float32", shape = ())#candidate|1360|()|var|float32
call_1356 = func_1355_call(var_1357,var_1358,var_1359,var_1360,)
output = call_1356
func_1361 = relay.Function([var_1357,var_1358,var_1359,var_1360,], output)
mutated_mod['func_1361'] = func_1361
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1404 = relay.var("var_1404", dtype = "float32", shape = (9, 6))#candidate|1404|(9, 6)|var|float32
uop_1405 = relay.acosh(var_1404.astype('float32')) # shape=(9, 6)
bop_1409 = relay.bitwise_or(uop_1405.astype('int8'), relay.reshape(var_1404.astype('int8'), relay.shape_of(uop_1405))) # shape=(9, 6)
bop_1416 = relay.maximum(uop_1405.astype('uint32'), relay.reshape(var_1404.astype('uint32'), relay.shape_of(uop_1405))) # shape=(9, 6)
output = relay.Tuple([bop_1409,bop_1416,])
output2 = relay.Tuple([bop_1409,bop_1416,])
func_1424 = relay.Function([var_1404,], output)
mod['func_1424'] = func_1424
mod = relay.transform.InferType()(mod)
mutated_mod['func_1424'] = func_1424
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1425 = relay.var("var_1425", dtype = "float32", shape = (9, 6))#candidate|1425|(9, 6)|var|float32
func_1424_call = mutated_mod.get_global_var('func_1424')
call_1426 = func_1424_call(var_1425)
output = call_1426
func_1427 = relay.Function([var_1425], output)
mutated_mod['func_1427'] = func_1427
mutated_mod = relay.transform.InferType()(mutated_mod)
func_751_call = mod.get_global_var('func_751')
func_752_call = mutated_mod.get_global_var('func_752')
call_1452 = func_751_call()
call_1453 = func_751_call()
uop_1472 = relay.tan(call_1452.astype('float64')) # shape=(10,)
uop_1474 = relay.tan(call_1453.astype('float64')) # shape=(10,)
func_417_call = mod.get_global_var('func_417')
func_420_call = mutated_mod.get_global_var('func_420')
const_1477 = relay.const(-9.808214, dtype = "float32")#candidate|1477|()|const|float32
const_1478 = relay.const([2.222296,1.159303,-3.460219,-9.997158], dtype = "float32")#candidate|1478|(4,)|const|float32
call_1476 = relay.TupleGetItem(func_417_call(relay.reshape(const_1477.astype('float32'), []), relay.reshape(const_1478.astype('float32'), [4,]), ), 0)
call_1479 = relay.TupleGetItem(func_420_call(relay.reshape(const_1477.astype('float32'), []), relay.reshape(const_1478.astype('float32'), [4,]), ), 0)
output = relay.Tuple([uop_1472,call_1476,const_1477,const_1478,])
output2 = relay.Tuple([uop_1474,call_1479,const_1477,const_1478,])
func_1484 = relay.Function([], output)
mod['func_1484'] = func_1484
mod = relay.transform.InferType()(mod)
output = func_1484()
func_1485 = relay.Function([], output)
mutated_mod['func_1485'] = func_1485
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1486 = relay.var("var_1486", dtype = "uint32", shape = ())#candidate|1486|()|var|uint32
var_1487 = relay.var("var_1487", dtype = "uint32", shape = (10, 10))#candidate|1487|(10, 10)|var|uint32
bop_1488 = relay.left_shift(var_1486.astype('uint32'), var_1487.astype('uint32')) # shape=(10, 10)
output = relay.Tuple([bop_1488,])
output2 = relay.Tuple([bop_1488,])
func_1491 = relay.Function([var_1486,var_1487,], output)
mod['func_1491'] = func_1491
mod = relay.transform.InferType()(mod)
mutated_mod['func_1491'] = func_1491
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1491_call = mutated_mod.get_global_var('func_1491')
var_1493 = relay.var("var_1493", dtype = "uint32", shape = ())#candidate|1493|()|var|uint32
var_1494 = relay.var("var_1494", dtype = "uint32", shape = (10, 10))#candidate|1494|(10, 10)|var|uint32
call_1492 = func_1491_call(var_1493,var_1494,)
output = call_1492
func_1495 = relay.Function([var_1493,var_1494,], output)
mutated_mod['func_1495'] = func_1495
mutated_mod = relay.transform.InferType()(mutated_mod)
func_586_call = mod.get_global_var('func_586')
func_587_call = mutated_mod.get_global_var('func_587')
call_1497 = relay.TupleGetItem(func_586_call(), 4)
call_1498 = relay.TupleGetItem(func_587_call(), 4)
func_1290_call = mod.get_global_var('func_1290')
func_1293_call = mutated_mod.get_global_var('func_1293')
var_1518 = relay.var("var_1518", dtype = "float32", shape = (14,))#candidate|1518|(14,)|var|float32
const_1519 = relay.const([7.060216,-8.264501,6.333515,5.088577,-7.251307,-2.748004,2.957616,9.685760,9.060873,-4.804825,6.600388,7.892563,-5.235767,2.252519,1.263996,0.908528,9.869919,7.346729,6.101217,7.428238,-6.185668,6.844284,-2.269543,6.335707,8.119942,-4.939780,-8.296211,-0.493871,-4.116088,0.626557,-4.153458,4.281966,-5.226361,-7.042638,1.822829,2.493002,1.735073,6.967914,-5.053384,-1.487109,6.731151,-8.173782,-5.758385,-8.401678,-3.236933,7.919538,-1.155472,-9.022896,1.568062,2.769876,-6.093534,-4.563694,3.638035,7.464285,6.922220,7.465152,0.121879,-8.536821,5.753972,-9.791186,-9.551210,-4.143530,-8.912271,0.927085,2.719379,0.874787,-7.209871,-5.661448,4.616395,8.886608], dtype = "float64")#candidate|1519|(70,)|const|float64
call_1517 = relay.TupleGetItem(func_1290_call(relay.reshape(var_1518.astype('float32'), [14,]), relay.reshape(const_1519.astype('float64'), [70,]), ), 7)
call_1520 = relay.TupleGetItem(func_1293_call(relay.reshape(var_1518.astype('float32'), [14,]), relay.reshape(const_1519.astype('float64'), [70,]), ), 7)
func_765_call = mod.get_global_var('func_765')
func_766_call = mutated_mod.get_global_var('func_766')
call_1528 = func_765_call()
call_1529 = func_765_call()
func_1220_call = mod.get_global_var('func_1220')
func_1222_call = mutated_mod.get_global_var('func_1222')
call_1534 = relay.TupleGetItem(func_1220_call(), 2)
call_1535 = relay.TupleGetItem(func_1222_call(), 2)
func_1155_call = mod.get_global_var('func_1155')
func_1156_call = mutated_mod.get_global_var('func_1156')
call_1550 = relay.TupleGetItem(func_1155_call(), 0)
call_1551 = relay.TupleGetItem(func_1156_call(), 0)
func_24_call = mod.get_global_var('func_24')
func_26_call = mutated_mod.get_global_var('func_26')
call_1564 = relay.TupleGetItem(func_24_call(), 1)
call_1565 = relay.TupleGetItem(func_26_call(), 1)
bop_1566 = relay.bitwise_or(call_1517.astype('uint32'), call_1550.astype('uint32')) # shape=(70,)
bop_1569 = relay.bitwise_or(call_1520.astype('uint32'), call_1551.astype('uint32')) # shape=(70,)
var_1571 = relay.var("var_1571", dtype = "uint32", shape = (70,))#candidate|1571|(70,)|var|uint32
bop_1572 = relay.add(bop_1566.astype('uint64'), relay.reshape(var_1571.astype('uint64'), relay.shape_of(bop_1566))) # shape=(70,)
bop_1575 = relay.add(bop_1569.astype('uint64'), relay.reshape(var_1571.astype('uint64'), relay.shape_of(bop_1569))) # shape=(70,)
bop_1580 = relay.multiply(bop_1566.astype('float64'), call_1550.astype('float64')) # shape=(70,)
bop_1583 = relay.multiply(bop_1569.astype('float64'), call_1551.astype('float64')) # shape=(70,)
func_209_call = mod.get_global_var('func_209')
func_215_call = mutated_mod.get_global_var('func_215')
call_1590 = relay.TupleGetItem(func_209_call(relay.reshape(call_1564.astype('int8'), [10,]), relay.reshape(call_1497.astype('int8'), [10,]), relay.reshape(call_1550.astype('int64'), []), relay.reshape(call_1497.astype('int8'), [10,]), ), 1)
call_1591 = relay.TupleGetItem(func_215_call(relay.reshape(call_1564.astype('int8'), [10,]), relay.reshape(call_1497.astype('int8'), [10,]), relay.reshape(call_1550.astype('int64'), []), relay.reshape(call_1497.astype('int8'), [10,]), ), 1)
bop_1594 = relay.not_equal(bop_1566.astype('bool'), relay.reshape(bop_1580.astype('bool'), relay.shape_of(bop_1566))) # shape=(70,)
bop_1597 = relay.not_equal(bop_1569.astype('bool'), relay.reshape(bop_1583.astype('bool'), relay.shape_of(bop_1569))) # shape=(70,)
bop_1602 = relay.power(const_1519.astype('float64'), relay.reshape(call_1517.astype('float64'), relay.shape_of(const_1519))) # shape=(70,)
bop_1605 = relay.power(const_1519.astype('float64'), relay.reshape(call_1520.astype('float64'), relay.shape_of(const_1519))) # shape=(70,)
uop_1607 = relay.sinh(bop_1602.astype('float64')) # shape=(70,)
uop_1609 = relay.sinh(bop_1605.astype('float64')) # shape=(70,)
output = relay.Tuple([call_1497,var_1518,call_1528,call_1534,call_1564,bop_1572,call_1590,bop_1594,uop_1607,])
output2 = relay.Tuple([call_1498,var_1518,call_1529,call_1535,call_1565,bop_1575,call_1591,bop_1597,uop_1609,])
func_1610 = relay.Function([var_1518,var_1571,], output)
mod['func_1610'] = func_1610
mod = relay.transform.InferType()(mod)
var_1611 = relay.var("var_1611", dtype = "float32", shape = (14,))#candidate|1611|(14,)|var|float32
var_1612 = relay.var("var_1612", dtype = "uint32", shape = (70,))#candidate|1612|(70,)|var|uint32
output = func_1610(var_1611,var_1612,)
func_1613 = relay.Function([var_1611,var_1612,], output)
mutated_mod['func_1613'] = func_1613
mutated_mod = relay.transform.InferType()(mutated_mod)
func_751_call = mod.get_global_var('func_751')
func_752_call = mutated_mod.get_global_var('func_752')
call_1667 = func_751_call()
call_1668 = func_751_call()
func_470_call = mod.get_global_var('func_470')
func_474_call = mutated_mod.get_global_var('func_474')
const_1670 = relay.const(-3.455936, dtype = "float32")#candidate|1670|()|const|float32
var_1671 = relay.var("var_1671", dtype = "float32", shape = (2, 2))#candidate|1671|(2, 2)|var|float32
call_1669 = relay.TupleGetItem(func_470_call(relay.reshape(const_1670.astype('float32'), []), relay.reshape(var_1671.astype('float32'), [4,]), ), 0)
call_1672 = relay.TupleGetItem(func_474_call(relay.reshape(const_1670.astype('float32'), []), relay.reshape(var_1671.astype('float32'), [4,]), ), 0)
output = relay.Tuple([call_1667,call_1669,const_1670,var_1671,])
output2 = relay.Tuple([call_1668,call_1672,const_1670,var_1671,])
func_1678 = relay.Function([var_1671,], output)
mod['func_1678'] = func_1678
mod = relay.transform.InferType()(mod)
var_1679 = relay.var("var_1679", dtype = "float32", shape = (2, 2))#candidate|1679|(2, 2)|var|float32
output = func_1678(var_1679)
func_1680 = relay.Function([var_1679], output)
mutated_mod['func_1680'] = func_1680
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1220_call = mod.get_global_var('func_1220')
func_1222_call = mutated_mod.get_global_var('func_1222')
call_1701 = relay.TupleGetItem(func_1220_call(), 2)
call_1702 = relay.TupleGetItem(func_1222_call(), 2)
var_1726 = relay.var("var_1726", dtype = "float32", shape = (16,))#candidate|1726|(16,)|var|float32
bop_1727 = relay.logical_or(call_1701.astype('bool'), relay.reshape(var_1726.astype('bool'), relay.shape_of(call_1701))) # shape=(16,)
bop_1730 = relay.logical_or(call_1702.astype('bool'), relay.reshape(var_1726.astype('bool'), relay.shape_of(call_1702))) # shape=(16,)
output = bop_1727
output2 = bop_1730
func_1736 = relay.Function([var_1726,], output)
mod['func_1736'] = func_1736
mod = relay.transform.InferType()(mod)
mutated_mod['func_1736'] = func_1736
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1737 = relay.var("var_1737", dtype = "float32", shape = (16,))#candidate|1737|(16,)|var|float32
func_1736_call = mutated_mod.get_global_var('func_1736')
call_1738 = func_1736_call(var_1737)
output = call_1738
func_1739 = relay.Function([var_1737], output)
mutated_mod['func_1739'] = func_1739
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1746 = relay.var("var_1746", dtype = "float32", shape = (13, 16, 3))#candidate|1746|(13, 16, 3)|var|float32
uop_1747 = relay.acosh(var_1746.astype('float32')) # shape=(13, 16, 3)
var_1752 = relay.var("var_1752", dtype = "float32", shape = (13, 16, 3))#candidate|1752|(13, 16, 3)|var|float32
bop_1753 = relay.multiply(uop_1747.astype('uint8'), relay.reshape(var_1752.astype('uint8'), relay.shape_of(uop_1747))) # shape=(13, 16, 3)
uop_1756 = relay.atanh(uop_1747.astype('float64')) # shape=(13, 16, 3)
bop_1763 = relay.add(uop_1756.astype('int64'), relay.reshape(uop_1747.astype('int64'), relay.shape_of(uop_1756))) # shape=(13, 16, 3)
bop_1767 = relay.divide(uop_1747.astype('float32'), relay.reshape(bop_1763.astype('float32'), relay.shape_of(uop_1747))) # shape=(13, 16, 3)
output = relay.Tuple([bop_1753,bop_1767,])
output2 = relay.Tuple([bop_1753,bop_1767,])
F = relay.Function([var_1746,var_1752,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1746,var_1752,], output2)
mutated_mod['main'] = F
mutated_mod = relay.transform.InferType()(mutated_mod)
print('==========mutated_mod==========')
print(mutated_mod.astext(show_meta_data=False))
print('===================================')
graph, lib, params = relay.build(mod, target='llvm')
module1 = graph_runtime.create(graph, lib, tvm.device('llvm',0))
intrp2 = relay.build_module.create_executor('graph', mod, tvm.device('llvm',0),'llvm')
intrp3 = relay.build_module.create_executor('debug', mod, tvm.device('llvm',0),'llvm')
intrp4 = relay.build_module.create_executor('vm', mod, tvm.device('llvm',0),'llvm')
graph, lib, params = relay.build(mod, target='cuda')
module5 = graph_runtime.create(graph, lib, tvm.device('cuda',0))
intrp6 = relay.build_module.create_executor('graph', mod, tvm.device('cuda',0),'cuda')
intrp7 = relay.build_module.create_executor('debug', mod, tvm.device('cuda',0),'cuda')
intrp8 = relay.build_module.create_executor('vm', mod, tvm.device('cuda',0),'cuda')
seq = Sequential([
	relay.transform.AlterOpLayout(),
	relay.transform.AnnotateSpans(),
	relay.transform.BatchingOps(),
	relay.transform.CanonicalizeCast(),
	relay.transform.CanonicalizeOps(),
	relay.transform.DeadCodeElimination(),
	relay.transform.DynamicToStatic(),
	relay.transform.FastMath(),
	relay.transform.FirstOrderGradient(),
	relay.transform.EliminateCommonSubexpr(),
	relay.transform.MergeCompilerRegions(),
	relay.transform.Inline(),
	relay.transform.LambdaLift(),
	relay.transform.LazyGradientInit(),
	relay.transform.PartialEvaluate(),
	relay.transform.Legalize(),
	relay.transform.FoldConstant(),
])
mod = seq(mod)
print(mod.astext(show_meta_data=False))
graph, lib, params = relay.build(mod, target='llvm')
module9 = graph_runtime.create(graph, lib, tvm.device('llvm',0))
intrp10 = relay.build_module.create_executor('graph', mod, tvm.device('llvm',0),'llvm')
intrp11 = relay.build_module.create_executor('debug', mod, tvm.device('llvm',0),'llvm')
intrp12 = relay.build_module.create_executor('vm', mod, tvm.device('llvm',0),'llvm')
graph, lib, params = relay.build(mod, target='cuda')
module13 = graph_runtime.create(graph, lib, tvm.device('cuda',0))
intrp14 = relay.build_module.create_executor('graph', mod, tvm.device('cuda',0),'cuda')
intrp15 = relay.build_module.create_executor('debug', mod, tvm.device('cuda',0),'cuda')
intrp16 = relay.build_module.create_executor('vm', mod, tvm.device('cuda',0),'cuda')
graph, lib, params = relay.build(mutated_mod, target='llvm')
module17 = graph_runtime.create(graph, lib, tvm.device('llvm',0))
intrp18 = relay.build_module.create_executor('graph', mutated_mod, tvm.device('llvm',0),'llvm')
intrp19 = relay.build_module.create_executor('debug', mutated_mod, tvm.device('llvm',0),'llvm')
intrp20 = relay.build_module.create_executor('vm', mutated_mod, tvm.device('llvm',0),'llvm')
graph, lib, params = relay.build(mutated_mod, target='cuda')
module21 = graph_runtime.create(graph, lib, tvm.device('cuda',0))
intrp22 = relay.build_module.create_executor('graph', mutated_mod, tvm.device('cuda',0),'cuda')
intrp23 = relay.build_module.create_executor('debug', mutated_mod, tvm.device('cuda',0),'cuda')
intrp24 = relay.build_module.create_executor('vm', mutated_mod, tvm.device('cuda',0),'cuda')
input_1746= np.array([[[-4.568173,9.787517,4.280677],[0.278898,9.997326,-2.352002],[7.562460,-8.881631,4.859447],[0.946810,-0.801875,-0.342023],[-7.377378,7.089572,5.516675],[9.653293,-1.126882,-2.110992],[9.772920,3.322614,-9.703640],[6.621680,-1.291348,-0.265435],[1.492513,1.999513,7.154364],[3.482178,2.515190,1.621657],[2.611015,-8.278575,1.873794],[9.665849,-2.876094,3.232621],[1.118920,-1.718290,9.252208],[8.104744,-6.565891,-7.115699],[-8.634775,9.469044,2.039182],[-3.268802,-5.004680,-4.519008]],[[-3.262583,-0.230244,-2.120986],[3.453280,-6.895884,8.928835],[-5.840098,-7.695053,-2.766687],[-3.187643,4.973582,-8.912499],[-8.661929,8.758198,7.756162],[-4.938406,2.135119,-4.484264],[3.745749,6.440198,8.889382],[4.864969,-2.558363,-3.628748],[-8.908400,2.681976,1.073967],[-6.755456,3.936383,-6.366986],[5.385391,-1.082763,7.389720],[-3.037792,-0.074707,-4.931597],[7.768453,8.562213,-3.460290],[-7.542524,4.790573,-4.878883],[8.222456,-8.991591,-4.983682],[9.367333,-3.639152,-8.435143]],[[-5.164789,-6.194675,2.159238],[-6.381835,-1.906866,-8.267374],[6.912870,5.535773,-2.999409],[7.923412,7.396218,-3.106371],[-8.694365,9.018942,6.698951],[-0.984931,1.760348,-1.764874],[-3.293566,2.917828,6.028931],[7.602988,-1.744179,9.990312],[-8.158821,-6.726952,-0.514767],[4.240914,-5.517045,9.433315],[-0.343923,6.445231,-0.344497],[-3.632072,-1.827909,5.576169],[4.103448,-9.549684,-2.767829],[5.420712,5.651598,-2.038174],[-9.573683,-7.828943,-1.436476],[3.478923,-9.936261,-0.959953]],[[-2.662897,7.231683,-2.909339],[-8.174543,8.587926,3.921831],[-7.313703,-0.533620,-7.900776],[-7.330721,-9.013593,0.601125],[-2.416863,7.379605,2.550604],[-8.803661,6.334606,-3.063921],[-9.143353,6.277791,-6.985363],[7.958817,9.943683,-1.655697],[-4.063721,3.919897,5.679873],[-4.758938,6.908060,-4.046901],[2.180206,-6.495860,-1.303528],[8.328591,-3.989742,-9.900555],[1.233709,9.382983,9.141120],[9.350892,-5.237135,-5.446908],[8.440414,-8.217708,4.608056],[-7.686331,2.159519,7.430425]],[[1.079291,-1.027132,3.589185],[-0.085723,2.743244,-7.785687],[9.377924,-4.235096,-1.517316],[-8.181096,6.567808,-9.748237],[0.911068,-8.131386,1.877939],[8.506425,-8.305179,-6.314345],[-3.854920,-2.684182,4.983852],[0.455048,-4.927988,-4.614801],[-4.270795,0.935198,8.590239],[-9.426671,-7.376581,5.274429],[-9.686167,-0.163180,6.964904],[5.769277,-0.265740,7.125208],[-6.398894,-4.882421,-0.206012],[0.165136,4.650057,1.863708],[-3.255530,-3.654302,-3.269094],[-7.424167,-1.194348,9.727948]],[[-5.297919,8.325645,-4.764473],[-0.535340,7.434015,-5.545061],[7.448911,7.678841,-2.257534],[-2.077668,-3.089014,2.765728],[-2.375382,-7.303996,-7.253497],[-9.209952,-4.915379,0.086019],[-5.438955,8.387958,5.094980],[-8.352851,6.352022,6.553068],[-0.387069,-1.238436,-2.595313],[1.064123,2.011546,1.646642],[7.848896,-2.641612,-6.915020],[-5.213201,0.449345,-9.572843],[6.123044,-8.202130,-2.859041],[-7.737045,2.963249,-4.429580],[-9.390742,-0.889600,8.023671],[-0.925298,-7.861609,-3.267139]],[[-5.872585,0.960877,-0.439049],[-6.845249,-4.265184,-9.299438],[7.014437,6.221768,9.496337],[-0.050028,9.928384,7.745871],[6.955288,6.125005,5.959102],[-7.044907,6.601182,3.917891],[2.715161,8.273813,-7.542189],[4.976246,0.841598,-8.744980],[-1.776583,-5.703587,-1.426377],[-7.705968,1.470737,3.429872],[9.569605,9.469365,1.693260],[-6.328756,-3.605554,9.042147],[6.333230,-0.071092,2.749004],[7.891847,-7.578608,-6.160028],[9.385835,-3.493252,-0.116538],[2.701789,9.489834,-9.650916]],[[-8.915375,6.405528,5.239467],[7.567609,3.642201,0.267725],[0.817966,-2.151730,-4.225327],[-5.722411,4.468310,3.069206],[-7.490515,9.909114,5.900523],[-1.074080,-6.085951,-9.312137],[-4.135657,-7.249237,-7.018711],[5.732404,-4.162558,4.483757],[-0.555185,0.798036,-9.891145],[2.391807,0.023208,-5.682919],[1.262889,-3.349769,-1.693167],[8.956399,1.201376,9.460766],[9.345107,7.582993,-4.354143],[-9.325342,-7.934168,2.783524],[-3.785738,1.398862,2.977290],[-5.005379,-1.409189,9.002671]],[[-9.413416,-5.471293,-7.509491],[0.207085,-4.019055,-2.040683],[2.034327,1.525284,-5.715710],[-0.997918,4.093405,-3.496136],[0.008174,7.342392,-8.310380],[-6.481021,3.478055,-4.521105],[-9.514375,2.488546,-8.991342],[-5.240247,0.104650,8.422997],[-5.468094,-2.228999,8.459010],[-3.709268,-6.980608,-0.213366],[-3.534036,9.819898,0.645738],[-6.992862,7.633924,-7.462760],[4.994524,5.750304,-6.512277],[-4.031940,-4.904732,3.532710],[8.000216,-9.382323,4.078076],[-4.779568,7.226440,-2.638402]],[[2.924882,6.307343,-1.120318],[-0.185086,-5.478005,-7.560157],[-1.201852,8.675237,-4.714496],[-3.271227,-6.653074,5.084453],[-9.718058,-3.758033,2.309061],[-2.096715,5.745304,5.781173],[-7.802330,4.875972,4.508139],[6.754138,-8.275770,-0.295461],[4.380180,-6.369920,7.112664],[-8.748959,-2.571827,2.609465],[3.640125,-2.555578,7.550368],[-7.427964,-7.154634,-6.657683],[-0.214298,4.861819,-7.255060],[7.572512,4.057643,-0.280376],[7.661183,-3.442276,9.953033],[6.014971,6.492203,5.448747]],[[2.122194,-4.964746,0.171598],[5.814790,2.582927,-2.184724],[-8.036060,8.626672,7.887389],[-3.598189,-8.941962,-5.258771],[-5.879819,-2.030795,-7.675252],[1.590724,-2.301492,9.738491],[-6.204470,3.054301,6.131722],[-7.327717,0.454407,5.967435],[4.963476,4.839166,-9.398244],[9.193923,5.191320,6.895248],[3.993515,9.977177,-5.878784],[7.935910,0.471051,-1.822537],[-5.035764,-1.378468,2.523612],[-7.471084,8.958248,-4.320039],[-2.169236,-1.826514,-6.427606],[0.858249,-5.292306,9.492721]],[[0.717231,-1.881040,2.488478],[7.554924,2.092465,1.138763],[-8.433444,1.110901,8.353983],[3.105335,2.165562,5.990824],[9.209750,2.464688,3.687655],[3.500320,2.152510,4.600407],[-2.729060,-0.569651,-7.397673],[-9.706924,0.167755,-1.975704],[-7.214811,-4.216139,0.236721],[0.015736,-5.967952,3.219378],[6.658084,-3.743172,-7.701740],[4.312654,3.687961,-6.748391],[5.301644,-2.726882,-9.998756],[4.960259,-1.919367,-3.841484],[9.177726,8.102293,-8.163213],[6.498874,5.180759,-4.559747]],[[7.287486,-6.756289,0.415633],[-0.655952,-2.027340,8.419263],[3.921240,3.389326,3.679743],[2.239677,1.363287,-7.491107],[8.986263,3.130197,-8.649880],[-3.926611,4.950669,-6.184007],[3.995259,4.279842,2.360944],[-7.255033,2.164656,-0.972891],[9.434779,4.612298,-8.554339],[-5.521156,7.584908,-5.962824],[9.104045,0.430387,-9.466991],[7.604379,-3.842696,1.258683],[-9.108156,-9.060483,1.245543],[-5.744988,-6.924948,8.763573],[0.537581,3.567502,0.168312],[-2.726852,-4.941111,7.144449]]], dtype='float32')
module1.set_input('var_1746', input_1746)
input_1752= np.array([[[2.248264,7.168224,8.948968],[4.549549,-2.314691,7.497956],[5.393311,1.013220,6.152140],[-2.636780,-4.805275,6.576257],[-8.302258,5.681701,8.022501],[9.359766,2.213241,-5.778244],[8.456325,-4.596338,4.633809],[5.325013,6.639224,-7.717741],[8.808044,-5.637774,-6.489955],[3.891334,8.156410,-5.313419],[8.307918,-2.651057,-1.058821],[9.223518,-0.813563,1.423208],[6.358113,-0.326951,4.171793],[3.135061,-8.231907,8.286226],[8.757532,9.624275,1.269420],[4.580027,3.840722,8.537842]],[[-8.528850,3.301288,1.807428],[-5.267900,5.326765,8.778973],[8.797757,9.548460,-2.207849],[2.771162,9.643591,9.226771],[-1.890483,5.662602,-4.725483],[0.482742,2.516807,1.047753],[1.968098,3.994240,0.599002],[-9.110857,2.995205,-6.516483],[-4.093206,5.434695,-1.787238],[-8.397651,4.909641,-9.329097],[-8.131518,-8.341290,-7.031753],[-7.029879,5.126604,-9.082418],[8.526495,0.830578,-0.568139],[-5.101200,0.136159,-1.502320],[-4.573874,-3.748162,-6.742395],[8.904117,-4.519766,-0.598960]],[[-8.689215,0.228314,-7.049201],[-3.278208,-6.811816,-5.333299],[8.824215,1.387196,-3.262164],[-6.116589,-4.708880,-6.831634],[0.376756,7.668133,9.695654],[-8.472779,0.538519,0.545715],[1.932979,7.588463,6.668271],[-1.574201,-8.398124,-0.709473],[6.200407,-7.420130,3.746581],[-5.518163,-6.511648,4.621731],[9.953574,4.753147,9.853448],[-3.102207,-3.136807,-5.038036],[6.017669,-5.492487,-2.314196],[-1.591573,2.826287,5.524999],[0.855306,3.862621,1.250425],[1.694776,-9.375475,0.445447]],[[6.887096,-6.182203,8.332331],[0.647172,7.053898,2.522791],[-0.395598,5.975209,-8.226795],[8.097868,-5.506260,3.817575],[5.224306,-4.209511,-4.550899],[9.639872,4.709821,4.619773],[-5.000230,8.571431,3.639079],[7.805946,-9.382483,4.123313],[4.029669,-4.011016,-0.086354],[9.737274,-4.921148,7.672999],[-7.616888,-3.058182,8.896750],[-2.520399,5.617728,6.826439],[-0.313527,-0.653282,-1.995983],[7.788522,-1.868033,0.300773],[-0.843732,-7.833411,1.802485],[5.378293,-8.888575,4.285165]],[[5.006126,8.851157,-0.961818],[6.765596,5.933294,7.283368],[-1.709121,3.404588,6.582076],[9.236367,-1.907145,-8.595467],[9.375937,8.342742,-8.134992],[7.769736,-5.129072,-7.498330],[-8.734785,-8.333125,9.673986],[3.723466,-6.924454,5.143773],[5.753387,5.512531,-5.872131],[8.248314,3.800025,-0.168096],[-1.658489,-4.277914,-0.035567],[9.095767,-5.400276,8.313686],[3.785769,9.453816,0.028253],[8.610981,4.974701,-8.741733],[8.797291,-0.683483,-1.320720],[-1.377104,-9.841459,-8.346309]],[[5.401013,4.891187,7.255608],[0.675565,5.883080,-2.461115],[0.997270,-6.230229,6.191859],[8.902800,-5.072198,7.459809],[-0.848890,7.457844,-2.047705],[6.945552,7.460711,-1.998393],[5.343610,-6.882928,-5.271421],[-9.708220,-9.825732,8.074225],[8.835050,-1.557498,-0.585130],[2.940434,5.537233,3.420288],[-8.372004,3.019340,-3.518739],[-9.406340,-5.888205,-9.881307],[0.968461,-5.680257,4.719018],[3.218411,-2.759227,6.171155],[-5.544818,9.641856,9.750963],[9.213836,-8.218767,-1.986377]],[[-7.693499,-1.479630,6.016965],[-7.488642,1.570876,-0.926173],[-5.836348,-8.390358,-7.837296],[0.857066,-3.917635,-0.646594],[6.239847,9.345948,8.758004],[-7.062209,-8.089917,-8.111714],[-4.752784,-4.446296,6.964009],[-5.563604,0.308730,2.052520],[-8.783950,-6.843032,9.131157],[7.912899,-4.219953,-5.295033],[-4.265087,-5.105983,-9.209671],[-0.131795,9.570994,-9.549185],[2.384832,-1.868142,6.047916],[-9.877785,-9.634787,-8.361573],[-2.954579,-3.109128,-3.326357],[-6.003148,-8.328405,6.800907]],[[9.840096,7.344383,7.530310],[6.115639,-0.708143,3.573969],[7.429018,-9.124526,-9.961196],[-2.273221,-7.761426,8.785056],[4.164692,2.852252,-8.405180],[-6.314029,-7.362180,-9.956247],[-7.885802,-7.577878,0.311054],[-5.681807,-5.678426,-5.343675],[-8.094992,2.491527,-5.488176],[9.649335,-5.280869,1.009584],[-4.289356,-9.554731,5.382227],[-4.948891,-2.896141,-1.047792],[-6.717326,0.662795,3.518054],[4.339215,-4.473005,-9.080993],[9.054591,-5.888526,-0.753557],[0.784896,-3.015561,6.093964]],[[-9.636528,6.598772,4.667137],[0.394282,-7.366499,-9.433232],[-6.630544,2.798202,-9.716772],[7.745766,-1.589152,1.792224],[9.125009,0.502249,-1.093839],[0.758940,-9.390180,4.157870],[0.084607,0.682949,8.939412],[5.042634,0.693422,-2.204213],[9.517514,7.149239,-3.270820],[-6.674419,1.220493,4.779081],[0.402002,1.700924,8.944847],[6.037874,9.145289,-5.438454],[-4.250864,-9.748985,0.138892],[9.224317,7.377624,-6.473155],[8.931164,4.528119,-3.886682],[-2.505380,-3.106994,5.187650]],[[-9.814644,-6.807481,2.415769],[8.803335,-1.964774,1.059274],[-3.231951,-2.121766,9.353186],[5.800702,-4.109241,-6.028086],[2.831134,-9.504599,-8.521806],[3.111019,-0.222138,-4.342941],[4.134269,4.676831,5.623856],[2.978164,5.382019,1.588810],[-0.778019,4.229259,-4.274890],[8.009542,-0.335127,0.507567],[6.100553,1.449209,-5.005843],[4.825635,9.116534,0.825356],[6.238415,4.703181,7.359063],[6.692936,9.217559,-5.691432],[-5.533794,-8.367647,4.754050],[5.654938,-2.261730,-9.973746]],[[-3.958590,-9.038189,-7.170712],[2.183096,-1.925525,-7.148637],[6.226602,2.900287,-0.112987],[-2.064066,1.960771,8.610658],[-7.606425,3.731729,3.045813],[8.851193,1.058745,-6.174076],[-3.808622,2.826245,-9.019499],[-5.162815,-4.360301,-2.261506],[6.787365,0.548141,1.541817],[1.681406,-5.019970,9.230169],[-3.152493,3.017039,-5.093893],[1.264888,4.663796,-7.822682],[-6.138902,-8.549384,-7.154430],[7.203270,6.547101,6.593056],[1.220970,5.537270,7.376303],[9.561567,2.141918,-5.272904]],[[-4.354344,-4.285911,-5.238756],[0.695496,2.544829,3.443599],[3.733115,1.400293,-3.043742],[8.528077,9.959734,3.369653],[4.872024,-6.236088,9.679840],[2.397957,-2.052130,-7.286580],[-8.667836,6.522538,1.157994],[6.662626,4.634452,1.960483],[8.242324,5.769177,8.898178],[-5.154356,-9.722996,-6.799433],[3.067776,9.806064,1.090147],[-7.793668,-6.166607,-6.589189],[9.065512,-3.848876,-7.783488],[-9.273364,-1.795270,5.615005],[-4.912102,5.341990,-7.339678],[-6.452873,3.941170,-2.807854]],[[5.843703,-8.088211,-5.092799],[2.524180,-7.610514,9.995317],[-2.033044,0.533211,4.956240],[-2.015854,-8.658240,-0.494956],[8.753008,-7.387858,3.863755],[-7.059126,0.385055,-6.553250],[8.523694,6.932322,5.448323],[-4.891816,-4.489026,-5.978901],[-9.045494,0.131425,-0.182567],[-6.861843,4.145303,-0.045896],[1.825893,1.091147,9.576571],[-4.129347,-9.806386,3.639395],[-7.002607,0.365788,4.116076],[2.947877,-6.424636,-2.247282],[-1.431477,4.961737,1.240872],[-3.446337,-3.492445,7.456501]]], dtype='float32')
module1.set_input('var_1752', input_1752)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1746, input_1752, )
res3 = intrp3.evaluate()(input_1746, input_1752, )
res4 = intrp4.evaluate()(input_1746, input_1752, )
res2 = vmobj_to_list(res2)
res3 = vmobj_to_list(res3)
res4 = vmobj_to_list(res4)
res1_0 = module1.get_output(0).asnumpy()
res2_0 = res2[0].asnumpy()
res3_0 = res3[0].asnumpy()
res4_0 = res4[0].asnumpy()
np.testing.assert_allclose(res1_0 ,res2_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_0 ,res3_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_0 ,res4_0, atol=1e-3, rtol=1e-3)
(res1_0 == res2_0).all()
(res1_0 == res3_0).all()
(res1_0 == res4_0).all()
res1_1 = module1.get_output(1).asnumpy()
res2_1 = res2[1].asnumpy()
res3_1 = res3[1].asnumpy()
res4_1 = res4[1].asnumpy()
np.testing.assert_allclose(res1_1 ,res2_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_1 ,res3_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_1 ,res4_1, atol=1e-3, rtol=1e-3)
(res1_1 == res2_1).all()
(res1_1 == res3_1).all()
(res1_1 == res4_1).all()
module5.set_input('var_1746', input_1746)
module5.set_input('var_1752', input_1752)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1746, input_1752, )
res7 = intrp7.evaluate()(input_1746, input_1752, )
res8 = intrp8.evaluate()(input_1746, input_1752, )
res6 = vmobj_to_list(res6)
res7 = vmobj_to_list(res7)
res8 = vmobj_to_list(res8)
res5_0 = module5.get_output(0).asnumpy()
res6_0 = res6[0].asnumpy()
res7_0 = res7[0].asnumpy()
res8_0 = res8[0].asnumpy()
np.testing.assert_allclose(res5_0 ,res6_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_0 ,res7_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_0 ,res8_0, atol=1e-3, rtol=1e-3)
(res5_0 == res6_0).all()
(res5_0 == res7_0).all()
(res5_0 == res8_0).all()
res5_1 = module5.get_output(1).asnumpy()
res6_1 = res6[1].asnumpy()
res7_1 = res7[1].asnumpy()
res8_1 = res8[1].asnumpy()
np.testing.assert_allclose(res5_1 ,res6_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_1 ,res7_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_1 ,res8_1, atol=1e-3, rtol=1e-3)
(res5_1 == res6_1).all()
(res5_1 == res7_1).all()
(res5_1 == res8_1).all()
module9.set_input('var_1746', input_1746)
module9.set_input('var_1752', input_1752)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1746, input_1752, )
res11 = intrp11.evaluate()(input_1746, input_1752, )
res12 = intrp12.evaluate()(input_1746, input_1752, )
res10 = vmobj_to_list(res10)
res11 = vmobj_to_list(res11)
res12 = vmobj_to_list(res12)
res9_0 = module9.get_output(0).asnumpy()
res10_0 = res10[0].asnumpy()
res11_0 = res11[0].asnumpy()
res12_0 = res12[0].asnumpy()
np.testing.assert_allclose(res9_0 ,res10_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_0 ,res11_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_0 ,res12_0, atol=1e-3, rtol=1e-3)
(res9_0 == res10_0).all()
(res9_0 == res11_0).all()
(res9_0 == res12_0).all()
res9_1 = module9.get_output(1).asnumpy()
res10_1 = res10[1].asnumpy()
res11_1 = res11[1].asnumpy()
res12_1 = res12[1].asnumpy()
np.testing.assert_allclose(res9_1 ,res10_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_1 ,res11_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_1 ,res12_1, atol=1e-3, rtol=1e-3)
(res9_1 == res10_1).all()
(res9_1 == res11_1).all()
(res9_1 == res12_1).all()
module13.set_input('var_1746', input_1746)
module13.set_input('var_1752', input_1752)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1746, input_1752, )
res15 = intrp15.evaluate()(input_1746, input_1752, )
res16 = intrp16.evaluate()(input_1746, input_1752, )
res14 = vmobj_to_list(res14)
res15 = vmobj_to_list(res15)
res16 = vmobj_to_list(res16)
res13_0 = module13.get_output(0).asnumpy()
res14_0 = res14[0].asnumpy()
res15_0 = res15[0].asnumpy()
res16_0 = res16[0].asnumpy()
np.testing.assert_allclose(res13_0 ,res14_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_0 ,res15_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_0 ,res16_0, atol=1e-3, rtol=1e-3)
(res13_0 == res14_0).all()
(res13_0 == res15_0).all()
(res13_0 == res16_0).all()
res13_1 = module13.get_output(1).asnumpy()
res14_1 = res14[1].asnumpy()
res15_1 = res15[1].asnumpy()
res16_1 = res16[1].asnumpy()
np.testing.assert_allclose(res13_1 ,res14_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_1 ,res15_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_1 ,res16_1, atol=1e-3, rtol=1e-3)
(res13_1 == res14_1).all()
(res13_1 == res15_1).all()
(res13_1 == res16_1).all()
module17.set_input('var_1746', input_1746)
module17.set_input('var_1752', input_1752)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1746, input_1752, )
res19 = intrp19.evaluate()(input_1746, input_1752, )
res20 = intrp20.evaluate()(input_1746, input_1752, )
res18 = vmobj_to_list(res18)
res19 = vmobj_to_list(res19)
res20 = vmobj_to_list(res20)
res17_0 = module17.get_output(0).asnumpy()
res18_0 = res18[0].asnumpy()
res19_0 = res19[0].asnumpy()
res20_0 = res20[0].asnumpy()
np.testing.assert_allclose(res17_0 ,res18_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_0 ,res19_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_0 ,res20_0, atol=1e-3, rtol=1e-3)
(res17_0 == res18_0).all()
(res17_0 == res19_0).all()
(res17_0 == res20_0).all()
res17_1 = module17.get_output(1).asnumpy()
res18_1 = res18[1].asnumpy()
res19_1 = res19[1].asnumpy()
res20_1 = res20[1].asnumpy()
np.testing.assert_allclose(res17_1 ,res18_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_1 ,res19_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_1 ,res20_1, atol=1e-3, rtol=1e-3)
(res17_1 == res18_1).all()
(res17_1 == res19_1).all()
(res17_1 == res20_1).all()
module21.set_input('var_1746', input_1746)
module21.set_input('var_1752', input_1752)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1746, input_1752, )
res23 = intrp23.evaluate()(input_1746, input_1752, )
res24 = intrp24.evaluate()(input_1746, input_1752, )
res22 = vmobj_to_list(res22)
res23 = vmobj_to_list(res23)
res24 = vmobj_to_list(res24)
res21_0 = module21.get_output(0).asnumpy()
res22_0 = res22[0].asnumpy()
res23_0 = res23[0].asnumpy()
res24_0 = res24[0].asnumpy()
np.testing.assert_allclose(res21_0 ,res22_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_0 ,res23_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_0 ,res24_0, atol=1e-3, rtol=1e-3)
(res21_0 == res22_0).all()
(res21_0 == res23_0).all()
(res21_0 == res24_0).all()
res21_1 = module21.get_output(1).asnumpy()
res22_1 = res22[1].asnumpy()
res23_1 = res23[1].asnumpy()
res24_1 = res24[1].asnumpy()
np.testing.assert_allclose(res21_1 ,res22_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_1 ,res23_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_1 ,res24_1, atol=1e-3, rtol=1e-3)
(res21_1 == res22_1).all()
(res21_1 == res23_1).all()
(res21_1 == res24_1).all()

'''9: TVMFuncCall
8: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
7: tvm::transform::Pass::operator()(tvm::IRModule) const
6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
5: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::LazyGradientInit(tvm::RelayExpr const&, tvm::IRModule)
0: tvm::relay::CheckFeature(tvm::RelayExpr const&, tvm::relay::FeatureSet const&)

'''