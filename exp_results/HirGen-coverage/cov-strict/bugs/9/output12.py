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
var_104 = relay.var("var_104", dtype = "uint32", shape = ())#candidate|104|()|var|uint32
var_105 = relay.var("var_105", dtype = "uint32", shape = (2, 14, 6))#candidate|105|(2, 14, 6)|var|uint32
bop_106 = relay.subtract(var_104.astype('uint32'), var_105.astype('uint32')) # shape=(2, 14, 6)
uop_109 = relay.sigmoid(var_105.astype('float32')) # shape=(2, 14, 6)
output = relay.Tuple([bop_106,uop_109,])
output2 = relay.Tuple([bop_106,uop_109,])
func_112 = relay.Function([var_104,var_105,], output)
mod['func_112'] = func_112
mod = relay.transform.InferType()(mod)
mutated_mod['func_112'] = func_112
mutated_mod = relay.transform.InferType()(mutated_mod)
func_112_call = mutated_mod.get_global_var('func_112')
var_114 = relay.var("var_114", dtype = "uint32", shape = ())#candidate|114|()|var|uint32
var_115 = relay.var("var_115", dtype = "uint32", shape = (2, 14, 6))#candidate|115|(2, 14, 6)|var|uint32
call_113 = func_112_call(var_114,var_115,)
output = call_113
func_116 = relay.Function([var_114,var_115,], output)
mutated_mod['func_116'] = func_116
mutated_mod = relay.transform.InferType()(mutated_mod)
var_134 = relay.var("var_134", dtype = "float32", shape = (7, 8, 15))#candidate|134|(7, 8, 15)|var|float32
uop_135 = relay.tan(var_134.astype('float32')) # shape=(7, 8, 15)
output = relay.Tuple([uop_135,])
output2 = relay.Tuple([uop_135,])
func_143 = relay.Function([var_134,], output)
mod['func_143'] = func_143
mod = relay.transform.InferType()(mod)
mutated_mod['func_143'] = func_143
mutated_mod = relay.transform.InferType()(mutated_mod)
var_144 = relay.var("var_144", dtype = "float32", shape = (7, 8, 15))#candidate|144|(7, 8, 15)|var|float32
func_143_call = mutated_mod.get_global_var('func_143')
call_145 = func_143_call(var_144)
output = call_145
func_146 = relay.Function([var_144], output)
mutated_mod['func_146'] = func_146
mutated_mod = relay.transform.InferType()(mutated_mod)
const_167 = relay.const([[-1.103776,8.185270,-3.874419,-8.512546,-9.753917,2.438772,-8.018017,-9.788032,-0.665204,-9.360188,-9.857630,9.990757,3.257427,5.130136],[9.948534,-2.060764,6.998696,-9.307992,-6.455161,2.885643,-1.328915,5.093213,8.940651,5.880557,2.597746,-5.343535,-4.512137,4.256763],[6.984177,-5.781592,5.505772,5.411566,-8.400123,-9.314665,-4.282989,9.197093,-1.731889,-6.379080,-2.090791,6.186133,7.942517,8.371274],[0.803570,8.666367,4.391033,6.755186,-3.287886,3.902785,-5.825786,-5.842031,-3.829050,6.980038,-4.282972,-9.400405,4.961180,-3.190059]], dtype = "float32")#candidate|167|(4, 14)|const|float32
uop_168 = relay.sigmoid(const_167.astype('float32')) # shape=(4, 14)
func_143_call = mod.get_global_var('func_143')
func_146_call = mutated_mod.get_global_var('func_146')
var_176 = relay.var("var_176", dtype = "float32", shape = (840,))#candidate|176|(840,)|var|float32
call_175 = relay.TupleGetItem(func_143_call(relay.reshape(var_176.astype('float32'), [7, 8, 15])), 0)
call_177 = relay.TupleGetItem(func_146_call(relay.reshape(var_176.astype('float32'), [7, 8, 15])), 0)
output = relay.Tuple([uop_168,call_175,var_176,])
output2 = relay.Tuple([uop_168,call_177,var_176,])
func_178 = relay.Function([var_176,], output)
mod['func_178'] = func_178
mod = relay.transform.InferType()(mod)
mutated_mod['func_178'] = func_178
mutated_mod = relay.transform.InferType()(mutated_mod)
var_179 = relay.var("var_179", dtype = "float32", shape = (840,))#candidate|179|(840,)|var|float32
func_178_call = mutated_mod.get_global_var('func_178')
call_180 = func_178_call(var_179)
output = call_180
func_181 = relay.Function([var_179], output)
mutated_mod['func_181'] = func_181
mutated_mod = relay.transform.InferType()(mutated_mod)
var_224 = relay.var("var_224", dtype = "float32", shape = (14, 16, 1))#candidate|224|(14, 16, 1)|var|float32
uop_225 = relay.acos(var_224.astype('float32')) # shape=(14, 16, 1)
var_228 = relay.var("var_228", dtype = "float32", shape = (14, 16, 1))#candidate|228|(14, 16, 1)|var|float32
bop_229 = relay.power(uop_225.astype('float32'), relay.reshape(var_228.astype('float32'), relay.shape_of(uop_225))) # shape=(14, 16, 1)
output = relay.Tuple([bop_229,])
output2 = relay.Tuple([bop_229,])
func_241 = relay.Function([var_224,var_228,], output)
mod['func_241'] = func_241
mod = relay.transform.InferType()(mod)
var_242 = relay.var("var_242", dtype = "float32", shape = (14, 16, 1))#candidate|242|(14, 16, 1)|var|float32
var_243 = relay.var("var_243", dtype = "float32", shape = (14, 16, 1))#candidate|243|(14, 16, 1)|var|float32
output = func_241(var_242,var_243,)
func_244 = relay.Function([var_242,var_243,], output)
mutated_mod['func_244'] = func_244
mutated_mod = relay.transform.InferType()(mutated_mod)
var_277 = relay.var("var_277", dtype = "float64", shape = (5, 8))#candidate|277|(5, 8)|var|float64
uop_278 = relay.log(var_277.astype('float64')) # shape=(5, 8)
uop_280 = relay.cos(var_277.astype('float64')) # shape=(5, 8)
func_143_call = mod.get_global_var('func_143')
func_146_call = mutated_mod.get_global_var('func_146')
var_292 = relay.var("var_292", dtype = "float32", shape = (840,))#candidate|292|(840,)|var|float32
call_291 = relay.TupleGetItem(func_143_call(relay.reshape(var_292.astype('float32'), [7, 8, 15])), 0)
call_293 = relay.TupleGetItem(func_146_call(relay.reshape(var_292.astype('float32'), [7, 8, 15])), 0)
func_112_call = mod.get_global_var('func_112')
func_116_call = mutated_mod.get_global_var('func_116')
var_304 = relay.var("var_304", dtype = "uint32", shape = ())#candidate|304|()|var|uint32
var_305 = relay.var("var_305", dtype = "uint32", shape = (168,))#candidate|305|(168,)|var|uint32
call_303 = relay.TupleGetItem(func_112_call(relay.reshape(var_304.astype('uint32'), []), relay.reshape(var_305.astype('uint32'), [2, 14, 6]), ), 0)
call_306 = relay.TupleGetItem(func_116_call(relay.reshape(var_304.astype('uint32'), []), relay.reshape(var_305.astype('uint32'), [2, 14, 6]), ), 0)
func_112_call = mod.get_global_var('func_112')
func_116_call = mutated_mod.get_global_var('func_116')
call_307 = relay.TupleGetItem(func_112_call(relay.reshape(var_304.astype('uint32'), []), relay.reshape(var_305.astype('uint32'), [2, 14, 6]), ), 0)
call_308 = relay.TupleGetItem(func_116_call(relay.reshape(var_304.astype('uint32'), []), relay.reshape(var_305.astype('uint32'), [2, 14, 6]), ), 0)
var_315 = relay.var("var_315", dtype = "float64", shape = (5, 8))#candidate|315|(5, 8)|var|float64
bop_316 = relay.greater_equal(uop_280.astype('bool'), relay.reshape(var_315.astype('bool'), relay.shape_of(uop_280))) # shape=(5, 8)
output = relay.Tuple([uop_278,call_291,var_292,call_303,var_304,var_305,call_307,bop_316,])
output2 = relay.Tuple([uop_278,call_293,var_292,call_306,var_304,var_305,call_308,bop_316,])
func_322 = relay.Function([var_277,var_292,var_304,var_305,var_315,], output)
mod['func_322'] = func_322
mod = relay.transform.InferType()(mod)
mutated_mod['func_322'] = func_322
mutated_mod = relay.transform.InferType()(mutated_mod)
func_322_call = mutated_mod.get_global_var('func_322')
var_324 = relay.var("var_324", dtype = "float64", shape = (5, 8))#candidate|324|(5, 8)|var|float64
var_325 = relay.var("var_325", dtype = "float32", shape = (840,))#candidate|325|(840,)|var|float32
var_326 = relay.var("var_326", dtype = "uint32", shape = ())#candidate|326|()|var|uint32
var_327 = relay.var("var_327", dtype = "uint32", shape = (168,))#candidate|327|(168,)|var|uint32
var_328 = relay.var("var_328", dtype = "float64", shape = (5, 8))#candidate|328|(5, 8)|var|float64
call_323 = func_322_call(var_324,var_325,var_326,var_327,var_328,)
output = call_323
func_329 = relay.Function([var_324,var_325,var_326,var_327,var_328,], output)
mutated_mod['func_329'] = func_329
mutated_mod = relay.transform.InferType()(mutated_mod)
var_347 = relay.var("var_347", dtype = "float64", shape = (8,))#candidate|347|(8,)|var|float64
uop_348 = relay.cosh(var_347.astype('float64')) # shape=(8,)
uop_351 = relay.acos(var_347.astype('float64')) # shape=(8,)
output = relay.Tuple([uop_348,uop_351,])
output2 = relay.Tuple([uop_348,uop_351,])
func_365 = relay.Function([var_347,], output)
mod['func_365'] = func_365
mod = relay.transform.InferType()(mod)
var_366 = relay.var("var_366", dtype = "float64", shape = (8,))#candidate|366|(8,)|var|float64
output = func_365(var_366)
func_367 = relay.Function([var_366], output)
mutated_mod['func_367'] = func_367
mutated_mod = relay.transform.InferType()(mutated_mod)
var_417 = relay.var("var_417", dtype = "float64", shape = (4, 11))#candidate|417|(4, 11)|var|float64
uop_418 = relay.erf(var_417.astype('float64')) # shape=(4, 11)
func_178_call = mod.get_global_var('func_178')
func_181_call = mutated_mod.get_global_var('func_181')
var_421 = relay.var("var_421", dtype = "float32", shape = (840,))#candidate|421|(840,)|var|float32
call_420 = relay.TupleGetItem(func_178_call(relay.reshape(var_421.astype('float32'), [840,])), 2)
call_422 = relay.TupleGetItem(func_181_call(relay.reshape(var_421.astype('float32'), [840,])), 2)
var_424 = relay.var("var_424", dtype = "float64", shape = (4, 11))#candidate|424|(4, 11)|var|float64
bop_425 = relay.not_equal(uop_418.astype('bool'), relay.reshape(var_424.astype('bool'), relay.shape_of(uop_418))) # shape=(4, 11)
bop_431 = relay.maximum(uop_418.astype('int32'), relay.reshape(var_417.astype('int32'), relay.shape_of(uop_418))) # shape=(4, 11)
uop_438 = relay.sin(bop_425.astype('float64')) # shape=(4, 11)
uop_441 = relay.atanh(uop_438.astype('float32')) # shape=(4, 11)
bop_447 = relay.logical_or(uop_441.astype('bool'), relay.reshape(bop_425.astype('bool'), relay.shape_of(uop_441))) # shape=(4, 11)
uop_451 = relay.sqrt(uop_441.astype('float32')) # shape=(4, 11)
bop_457 = relay.logical_xor(uop_451.astype('int16'), relay.reshape(uop_441.astype('int16'), relay.shape_of(uop_451))) # shape=(4, 11)
var_461 = relay.var("var_461", dtype = "bool", shape = (4, 11))#candidate|461|(4, 11)|var|bool
bop_462 = relay.less(bop_425.astype('bool'), relay.reshape(var_461.astype('bool'), relay.shape_of(bop_425))) # shape=(4, 11)
func_241_call = mod.get_global_var('func_241')
func_244_call = mutated_mod.get_global_var('func_244')
var_468 = relay.var("var_468", dtype = "float32", shape = (224,))#candidate|468|(224,)|var|float32
call_467 = relay.TupleGetItem(func_241_call(relay.reshape(var_468.astype('float32'), [14, 16, 1]), relay.reshape(var_468.astype('float32'), [14, 16, 1]), ), 0)
call_469 = relay.TupleGetItem(func_244_call(relay.reshape(var_468.astype('float32'), [14, 16, 1]), relay.reshape(var_468.astype('float32'), [14, 16, 1]), ), 0)
const_470 = relay.const([[-5.284287,4.705303,0.860511,-1.547920,-9.067773,1.904841,-4.787382,8.878592,-3.203907,-9.458857,5.502258],[-1.528992,-7.311335,5.658753,-1.206659,-5.687934,7.158906,-8.487523,-4.346901,-3.662370,9.714878,5.011884],[-1.747916,7.444345,-4.456959,0.489603,1.188108,-5.249432,9.925111,9.375920,-4.552147,4.901937,-9.537668],[-2.240775,-8.748755,9.136246,8.088558,7.224477,3.520182,-2.908985,7.723566,3.815006,7.720028,-8.088349]], dtype = "float32")#candidate|470|(4, 11)|const|float32
bop_471 = relay.less_equal(uop_451.astype('bool'), relay.reshape(const_470.astype('bool'), relay.shape_of(uop_451))) # shape=(4, 11)
func_112_call = mod.get_global_var('func_112')
func_116_call = mutated_mod.get_global_var('func_116')
var_475 = relay.var("var_475", dtype = "uint32", shape = ())#candidate|475|()|var|uint32
var_476 = relay.var("var_476", dtype = "uint32", shape = (168,))#candidate|476|(168,)|var|uint32
call_474 = relay.TupleGetItem(func_112_call(relay.reshape(var_475.astype('uint32'), []), relay.reshape(var_476.astype('uint32'), [2, 14, 6]), ), 0)
call_477 = relay.TupleGetItem(func_116_call(relay.reshape(var_475.astype('uint32'), []), relay.reshape(var_476.astype('uint32'), [2, 14, 6]), ), 0)
func_365_call = mod.get_global_var('func_365')
func_367_call = mutated_mod.get_global_var('func_367')
var_479 = relay.var("var_479", dtype = "float64", shape = (8,))#candidate|479|(8,)|var|float64
call_478 = relay.TupleGetItem(func_365_call(relay.reshape(var_479.astype('float64'), [8,])), 1)
call_480 = relay.TupleGetItem(func_367_call(relay.reshape(var_479.astype('float64'), [8,])), 1)
func_178_call = mod.get_global_var('func_178')
func_181_call = mutated_mod.get_global_var('func_181')
call_488 = relay.TupleGetItem(func_178_call(relay.reshape(var_421.astype('float32'), [840,])), 1)
call_489 = relay.TupleGetItem(func_181_call(relay.reshape(var_421.astype('float32'), [840,])), 1)
uop_500 = relay.exp(bop_457.astype('float32')) # shape=(4, 11)
var_502 = relay.var("var_502", dtype = "float32", shape = (4, 11))#candidate|502|(4, 11)|var|float32
bop_503 = relay.left_shift(uop_451.astype('int32'), relay.reshape(var_502.astype('int32'), relay.shape_of(uop_451))) # shape=(4, 11)
bop_508 = relay.mod(uop_418.astype('float64'), relay.reshape(var_461.astype('float64'), relay.shape_of(uop_418))) # shape=(4, 11)
output = relay.Tuple([call_420,var_421,bop_431,bop_447,bop_462,call_467,var_468,bop_471,call_474,var_475,var_476,call_478,var_479,call_488,uop_500,bop_503,bop_508,])
output2 = relay.Tuple([call_422,var_421,bop_431,bop_447,bop_462,call_469,var_468,bop_471,call_477,var_475,var_476,call_480,var_479,call_489,uop_500,bop_503,bop_508,])
func_521 = relay.Function([var_417,var_421,var_424,var_461,var_468,var_475,var_476,var_479,var_502,], output)
mod['func_521'] = func_521
mod = relay.transform.InferType()(mod)
var_522 = relay.var("var_522", dtype = "float64", shape = (4, 11))#candidate|522|(4, 11)|var|float64
var_523 = relay.var("var_523", dtype = "float32", shape = (840,))#candidate|523|(840,)|var|float32
var_524 = relay.var("var_524", dtype = "float64", shape = (4, 11))#candidate|524|(4, 11)|var|float64
var_525 = relay.var("var_525", dtype = "bool", shape = (4, 11))#candidate|525|(4, 11)|var|bool
var_526 = relay.var("var_526", dtype = "float32", shape = (224,))#candidate|526|(224,)|var|float32
var_527 = relay.var("var_527", dtype = "uint32", shape = ())#candidate|527|()|var|uint32
var_528 = relay.var("var_528", dtype = "uint32", shape = (168,))#candidate|528|(168,)|var|uint32
var_529 = relay.var("var_529", dtype = "float64", shape = (8,))#candidate|529|(8,)|var|float64
var_530 = relay.var("var_530", dtype = "float32", shape = (4, 11))#candidate|530|(4, 11)|var|float32
output = func_521(var_522,var_523,var_524,var_525,var_526,var_527,var_528,var_529,var_530,)
func_531 = relay.Function([var_522,var_523,var_524,var_525,var_526,var_527,var_528,var_529,var_530,], output)
mutated_mod['func_531'] = func_531
mutated_mod = relay.transform.InferType()(mutated_mod)
const_559 = relay.const(8.080408, dtype = "float64")#candidate|559|()|const|float64
var_560 = relay.var("var_560", dtype = "float64", shape = (1, 9))#candidate|560|(1, 9)|var|float64
bop_561 = relay.divide(const_559.astype('float64'), var_560.astype('float64')) # shape=(1, 9)
bop_565 = relay.minimum(bop_561.astype('float64'), relay.reshape(var_560.astype('float64'), relay.shape_of(bop_561))) # shape=(1, 9)
var_575 = relay.var("var_575", dtype = "float64", shape = (2, 9))#candidate|575|(2, 9)|var|float64
bop_576 = relay.multiply(var_560.astype('uint16'), var_575.astype('uint16')) # shape=(2, 9)
bop_580 = relay.logical_xor(bop_561.astype('uint16'), relay.reshape(var_560.astype('uint16'), relay.shape_of(bop_561))) # shape=(1, 9)
uop_586 = relay.sigmoid(var_560.astype('float32')) # shape=(1, 9)
uop_596 = relay.atan(uop_586.astype('float64')) # shape=(1, 9)
uop_598 = relay.sqrt(uop_596.astype('float64')) # shape=(1, 9)
output = relay.Tuple([bop_565,bop_576,bop_580,uop_598,])
output2 = relay.Tuple([bop_565,bop_576,bop_580,uop_598,])
func_607 = relay.Function([var_560,var_575,], output)
mod['func_607'] = func_607
mod = relay.transform.InferType()(mod)
mutated_mod['func_607'] = func_607
mutated_mod = relay.transform.InferType()(mutated_mod)
func_607_call = mutated_mod.get_global_var('func_607')
var_609 = relay.var("var_609", dtype = "float64", shape = (1, 9))#candidate|609|(1, 9)|var|float64
var_610 = relay.var("var_610", dtype = "float64", shape = (2, 9))#candidate|610|(2, 9)|var|float64
call_608 = func_607_call(var_609,var_610,)
output = call_608
func_611 = relay.Function([var_609,var_610,], output)
mutated_mod['func_611'] = func_611
mutated_mod = relay.transform.InferType()(mutated_mod)
const_627 = relay.const(True, dtype = "bool")#candidate|627|()|const|bool
const_628 = relay.const([[[True,False,True,False,False,False,False,True,True],[True,False,True,True,True,True,True,False,True]]], dtype = "bool")#candidate|628|(1, 2, 9)|const|bool
bop_629 = relay.logical_or(const_627.astype('bool'), const_628.astype('bool')) # shape=(1, 2, 9)
output = relay.Tuple([bop_629,])
output2 = relay.Tuple([bop_629,])
func_635 = relay.Function([], output)
mod['func_635'] = func_635
mod = relay.transform.InferType()(mod)
output = func_635()
func_636 = relay.Function([], output)
mutated_mod['func_636'] = func_636
mutated_mod = relay.transform.InferType()(mutated_mod)
const_691 = relay.const([[-6,3,9,7,10,-2,7,-1,-10,-9,6,-9,6,-7],[6,-10,10,6,9,-9,-5,2,-2,-6,-2,3,-9,-4],[4,3,3,-2,-5,4,-2,-6,5,6,1,-10,-6,-2],[5,-1,-7,1,-8,-3,-9,-10,2,-7,-2,-4,2,-7],[-7,3,-4,3,1,-3,4,3,-5,-6,8,3,6,-2],[-1,7,7,10,-6,-7,6,7,10,-5,4,-1,5,-4]], dtype = "uint64")#candidate|691|(6, 14)|const|uint64
var_692 = relay.var("var_692", dtype = "uint64", shape = (6, 14))#candidate|692|(6, 14)|var|uint64
bop_693 = relay.maximum(const_691.astype('uint64'), relay.reshape(var_692.astype('uint64'), relay.shape_of(const_691))) # shape=(6, 14)
output = relay.Tuple([bop_693,])
output2 = relay.Tuple([bop_693,])
func_696 = relay.Function([var_692,], output)
mod['func_696'] = func_696
mod = relay.transform.InferType()(mod)
mutated_mod['func_696'] = func_696
mutated_mod = relay.transform.InferType()(mutated_mod)
var_697 = relay.var("var_697", dtype = "uint64", shape = (6, 14))#candidate|697|(6, 14)|var|uint64
func_696_call = mutated_mod.get_global_var('func_696')
call_698 = func_696_call(var_697)
output = call_698
func_699 = relay.Function([var_697], output)
mutated_mod['func_699'] = func_699
mutated_mod = relay.transform.InferType()(mutated_mod)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_760 = relay.TupleGetItem(func_635_call(), 0)
call_761 = relay.TupleGetItem(func_636_call(), 0)
const_767 = relay.const([[[False,False,True,False,False,True,False,True,True],[True,True,False,False,True,False,False,True,True]],[[True,True,False,False,True,True,False,False,False],[False,True,True,True,True,True,False,True,True]],[[True,False,False,True,True,True,True,True,True],[False,False,False,True,True,True,True,True,True]],[[True,True,True,True,False,False,False,True,False],[True,True,True,True,True,True,True,False,False]],[[True,True,False,False,True,False,False,True,True],[True,False,False,False,True,True,False,False,True]],[[False,False,False,False,False,False,False,True,True],[True,False,True,False,True,False,False,True,False]],[[False,True,True,True,True,True,True,True,True],[False,True,True,False,False,False,False,False,False]],[[False,False,True,True,False,True,False,False,True],[True,False,False,True,False,False,False,True,True]],[[True,False,False,False,False,False,False,False,False],[False,True,False,False,True,True,True,True,True]],[[False,True,True,True,False,True,False,True,True],[False,True,False,True,True,False,True,True,False]],[[True,True,True,True,True,False,False,True,True],[True,False,False,False,True,False,True,False,False]],[[False,False,False,True,False,True,True,True,False],[True,True,False,True,False,True,False,False,True]],[[False,False,False,False,True,True,False,True,False],[False,False,False,True,False,True,True,True,True]],[[False,False,True,True,True,False,True,False,False],[False,True,False,True,True,False,False,True,True]],[[True,False,False,True,False,False,True,True,False],[False,False,True,False,True,True,True,False,False]],[[True,True,True,True,True,False,False,True,True],[False,True,True,False,False,True,False,True,True]]], dtype = "bool")#candidate|767|(16, 2, 9)|const|bool
bop_768 = relay.right_shift(call_760.astype('uint32'), const_767.astype('uint32')) # shape=(16, 2, 9)
bop_771 = relay.right_shift(call_761.astype('uint32'), const_767.astype('uint32')) # shape=(16, 2, 9)
func_521_call = mod.get_global_var('func_521')
func_531_call = mutated_mod.get_global_var('func_531')
var_777 = relay.var("var_777", dtype = "float64", shape = (44,))#candidate|777|(44,)|var|float64
var_778 = relay.var("var_778", dtype = "float32", shape = (840,))#candidate|778|(840,)|var|float32
const_779 = relay.const([6.013471,-0.063129,-0.991392,-8.472784,0.102007,9.017163,-2.031607,9.156907,3.616511,2.687906,-7.569702,2.272765,1.577337,7.032515,-9.650438,2.065404,7.865528,-6.755391,1.824035,-9.202949,0.082254,3.705236,5.812392,-3.621377,2.141443,2.111000,-4.176622,-3.348865,0.286392,-5.691700,-6.719868,6.173366,-4.463023,0.877409,-6.386635,-3.382141,9.743077,-8.469374,1.920942,-2.122069,-6.062224,-6.496576,-6.813634,5.956861,-3.988102,5.302126,2.245756,2.086488,-2.591017,3.914630,-2.443242,-8.134212,-5.580624,0.480164,7.985343,-5.721132,3.755121,-4.552555,2.905139,4.328509,6.510870,-8.790192,-2.742843,-4.441542,-6.141124,0.145656,-2.041860,9.064364,-2.503104,-2.469418,-6.324439,-7.681191,9.918271,6.362853,-3.614924,9.296243,5.700699,-3.050460,2.502110,-8.522363,9.636777,-2.536867,8.699091,-9.039325,-4.136407,1.689491,7.259545,-2.780669,-3.909004,7.722279,2.302222,-4.934621,-0.515714,-8.343539,-0.286580,3.133208,-7.860467,9.083662,8.130277,-5.621265,-0.933320,-9.541595,0.348289,-4.828735,-2.729046,-8.511633,-9.429411,-4.143476,9.313244,-9.282452,0.410787,9.249360,-7.007916,3.963588,-0.455842,7.335458,5.584346,-9.684353,1.545001,-3.025279,-1.768497,0.635641,-9.180794,2.908294,-1.766344,1.580668,-6.375613,2.810678,3.690781,-2.116316,-7.597454,-0.674464,-3.555008,-7.786684,4.844677,5.466733,5.797427,9.577467,8.747139,-3.340604,-0.019585,2.038942,-3.761435,0.056448,-1.901146,-9.378595,-0.769028,-8.151095,9.065025,-4.483250,-6.784960,4.105423,-2.672358,3.479216,3.311125,4.586113,8.526335,-2.816438,4.292999,6.844258,-0.713801,8.322336,-3.119057,2.777640,-8.425005,-7.354244,4.420728,3.168562,-4.102867,1.053962,-2.248094,-7.216559,-8.644860,2.697263,-4.392229,4.934688,-9.052754,8.423759,-2.626448,0.943669,-8.740299,0.900269,8.135220,7.903139,6.162963,2.016524,9.277684,4.985480,2.080950,-3.621112,7.475664,-8.065200,-6.031588,-3.019758,-7.214178,3.798922,-5.762176,-8.883320,8.666570,2.443423,5.845651,-3.438386,-3.439992,-7.014206,3.390725,-2.443026,-2.885852,-3.098575,-4.758126,8.306795,-4.987753,-4.429305,-8.697585,-7.372555,-3.085769,-2.871921,8.948583,4.534155,6.338720,9.484988,-8.934117,8.416719,-3.804712,2.009278], dtype = "float32")#candidate|779|(224,)|const|float32
const_780 = relay.const(1, dtype = "uint32")#candidate|780|()|const|uint32
var_781 = relay.var("var_781", dtype = "uint32", shape = (3, 56))#candidate|781|(3, 56)|var|uint32
var_782 = relay.var("var_782", dtype = "float64", shape = (4, 2))#candidate|782|(4, 2)|var|float64
call_776 = relay.TupleGetItem(func_521_call(relay.reshape(var_777.astype('float64'), [4, 11]), relay.reshape(var_778.astype('float32'), [840,]), relay.reshape(var_777.astype('float64'), [4, 11]), relay.reshape(var_777.astype('bool'), [4, 11]), relay.reshape(const_779.astype('float32'), [224,]), relay.reshape(const_780.astype('uint32'), []), relay.reshape(var_781.astype('uint32'), [168,]), relay.reshape(var_782.astype('float64'), [8,]), relay.reshape(var_777.astype('float32'), [4, 11]), ), 16)
call_783 = relay.TupleGetItem(func_531_call(relay.reshape(var_777.astype('float64'), [4, 11]), relay.reshape(var_778.astype('float32'), [840,]), relay.reshape(var_777.astype('float64'), [4, 11]), relay.reshape(var_777.astype('bool'), [4, 11]), relay.reshape(const_779.astype('float32'), [224,]), relay.reshape(const_780.astype('uint32'), []), relay.reshape(var_781.astype('uint32'), [168,]), relay.reshape(var_782.astype('float64'), [8,]), relay.reshape(var_777.astype('float32'), [4, 11]), ), 16)
uop_784 = relay.cos(var_777.astype('float64')) # shape=(44,)
bop_786 = relay.bitwise_and(uop_784.astype('int64'), relay.reshape(call_776.astype('int64'), relay.shape_of(uop_784))) # shape=(44,)
bop_789 = relay.bitwise_and(uop_784.astype('int64'), relay.reshape(call_783.astype('int64'), relay.shape_of(uop_784))) # shape=(44,)
output = relay.Tuple([bop_768,var_778,const_779,const_780,var_781,var_782,bop_786,])
output2 = relay.Tuple([bop_771,var_778,const_779,const_780,var_781,var_782,bop_789,])
func_792 = relay.Function([var_777,var_778,var_781,var_782,], output)
mod['func_792'] = func_792
mod = relay.transform.InferType()(mod)
var_793 = relay.var("var_793", dtype = "float64", shape = (44,))#candidate|793|(44,)|var|float64
var_794 = relay.var("var_794", dtype = "float32", shape = (840,))#candidate|794|(840,)|var|float32
var_795 = relay.var("var_795", dtype = "uint32", shape = (3, 56))#candidate|795|(3, 56)|var|uint32
var_796 = relay.var("var_796", dtype = "float64", shape = (4, 2))#candidate|796|(4, 2)|var|float64
output = func_792(var_793,var_794,var_795,var_796,)
func_797 = relay.Function([var_793,var_794,var_795,var_796,], output)
mutated_mod['func_797'] = func_797
mutated_mod = relay.transform.InferType()(mutated_mod)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_799 = relay.TupleGetItem(func_635_call(), 0)
call_800 = relay.TupleGetItem(func_636_call(), 0)
uop_801 = relay.rsqrt(call_799.astype('float64')) # shape=(1, 2, 9)
uop_803 = relay.rsqrt(call_800.astype('float64')) # shape=(1, 2, 9)
var_804 = relay.var("var_804", dtype = "float64", shape = (15, 2, 9))#candidate|804|(15, 2, 9)|var|float64
bop_805 = relay.mod(uop_801.astype('float32'), var_804.astype('float32')) # shape=(15, 2, 9)
bop_808 = relay.mod(uop_803.astype('float32'), var_804.astype('float32')) # shape=(15, 2, 9)
bop_812 = relay.add(uop_801.astype('int32'), bop_805.astype('int32')) # shape=(15, 2, 9)
bop_815 = relay.add(uop_803.astype('int32'), bop_808.astype('int32')) # shape=(15, 2, 9)
output = bop_812
output2 = bop_815
func_824 = relay.Function([var_804,], output)
mod['func_824'] = func_824
mod = relay.transform.InferType()(mod)
mutated_mod['func_824'] = func_824
mutated_mod = relay.transform.InferType()(mutated_mod)
var_825 = relay.var("var_825", dtype = "float64", shape = (15, 2, 9))#candidate|825|(15, 2, 9)|var|float64
func_824_call = mutated_mod.get_global_var('func_824')
call_826 = func_824_call(var_825)
output = call_826
func_827 = relay.Function([var_825], output)
mutated_mod['func_827'] = func_827
mutated_mod = relay.transform.InferType()(mutated_mod)
var_901 = relay.var("var_901", dtype = "float32", shape = ())#candidate|901|()|var|float32
var_902 = relay.var("var_902", dtype = "float32", shape = (3, 5, 8))#candidate|902|(3, 5, 8)|var|float32
bop_903 = relay.multiply(var_901.astype('float32'), var_902.astype('float32')) # shape=(3, 5, 8)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_909 = relay.TupleGetItem(func_635_call(), 0)
call_910 = relay.TupleGetItem(func_636_call(), 0)
func_322_call = mod.get_global_var('func_322')
func_329_call = mutated_mod.get_global_var('func_329')
var_915 = relay.var("var_915", dtype = "float64", shape = (1, 40))#candidate|915|(1, 40)|var|float64
const_916 = relay.const([-6.513242,-0.846144,-5.939727,-1.121593,-7.105177,-1.806933,-3.411120,5.735058,-0.190071,0.865080,4.791699,-2.073802,6.416166,7.811504,-8.033574,-8.802805,-6.177626,8.752415,7.154145,9.286551,6.839715,-6.709649,-4.780109,-4.103448,-9.710521,8.318986,2.959063,-9.365202,9.134792,2.414748,-7.625535,-4.321588,-4.242645,2.506865,4.331092,8.646433,-1.404268,-4.682229,1.957037,8.959841,-8.383336,6.723728,-9.596875,-4.089651,-4.996158,-1.762034,-9.618222,3.527657,-7.399062,-5.691406,2.268619,2.233205,-6.890413,-0.421074,-0.585181,1.404705,3.561877,-4.418328,-2.759698,2.486532,1.528727,2.542160,8.708947,-2.069086,7.705158,7.986641,5.433358,-5.497503,4.843011,8.194502,6.496523,-1.590874,-7.386983,-7.735564,-8.252246,3.612376,6.135359,6.319158,8.098355,1.105860,-1.917465,-2.405956,1.736476,6.959059,8.315202,-8.138324,-1.749091,2.206818,3.837346,4.761766,-1.364105,5.553577,-4.533009,-3.148717,7.360227,-5.381968,-0.465912,4.201880,-2.394416,-6.409332,2.379787,-1.645764,-6.036945,-3.167681,8.644780,5.329604,-7.869996,-5.776648,4.568722,2.590273,-0.221058,-6.568654,-2.919434,-2.387162,3.566771,-3.431758,5.186401,-6.487941,1.631847,7.914338,-9.352598,-7.767521,-7.490546,7.749768,9.352611,-5.183334,5.650736,-9.606646,-3.015461,-2.052173,-5.699575,-8.040580,5.637934,7.573102,-5.903556,6.760969,1.209798,5.346024,-6.720377,7.781574,2.681951,5.567158,-6.252704,-7.343930,-8.925438,-5.390156,-4.036383,-1.020250,1.819268,-5.169225,0.979724,1.694650,5.298105,-9.439312,4.238369,5.994730,2.818428,1.705316,-1.722148,-5.908767,-8.262614,3.553169,3.261520,-4.731314,6.470365,-2.202817,9.602981,3.399085,8.058870,-7.918871,6.610407,6.603565,6.658980,3.462983,8.972116,-8.946334,-1.618519,-3.139997,-1.334891,9.663752,4.146459,-5.165328,6.653010,5.487359,7.740408,3.131307,7.145462,8.300610,-7.581605,-2.083345,-2.738965,5.486782,9.632914,-1.967004,3.019103,3.423124,-3.701043,4.579948,-0.933719,-4.161017,-9.909089,-2.203068,-8.530456,-9.969210,-3.512974,-0.993014,-9.713031,-7.822708,5.988675,3.704588,-6.408365,1.840873,-4.546410,-8.953712,8.916810,-5.356763,-4.417386,9.868175,-5.151003,9.426909,2.307807,1.661748,-4.662852,3.032753,-5.897521,-2.409464,2.604137,7.355878,-1.543083,-6.909024,-9.671934,2.230025,-0.260090,-1.498467,6.045841,1.265119,4.055066,-6.094885,-0.789775,-5.764591,-6.744297,3.762727,-5.357556,2.589590,-4.625261,2.709586,-1.041237,-7.551900,3.036107,2.233400,-3.631390,3.137403,4.331550,5.395347,7.403727,-2.221643,-3.724369,-0.036453,9.373844,1.204255,8.132274,-4.016026,-7.644607,-7.778285,1.980371,-9.658326,7.062052,-7.504183,8.934962,-2.971100,2.617041,-6.814275,4.107929,-7.168672,-9.230776,-0.872844,-8.699541,2.811996,5.797433,4.751467,8.771457,4.006708,3.086000,7.637006,1.934154,-3.130520,-0.772420,8.894995,9.111764,-8.667828,7.297505,1.510049,-3.043447,-4.520382,9.283259,-8.721847,9.011592,2.600369,6.213782,5.781393,-4.998817,8.763340,-2.952533,-1.043930,9.899296,5.175373,-0.051863,-6.954540,2.373386,-0.497582,-1.877489,9.675478,-2.902998,5.681524,7.200881,4.968303,-5.249174,3.692935,3.220775,-3.324075,-3.217798,5.250016,-2.319104,2.223074,8.714661,-0.323403,0.660032,9.250809,-6.913583,7.150363,-1.832237,-6.741927,-4.676635,0.878567,3.893015,-3.480048,-4.633684,-6.254609,3.534056,-2.807527,-7.422357,3.392174,6.468117,6.640080,1.439617,7.169696,2.945094,-9.089580,9.484597,3.149789,6.095611,1.201735,-0.054416,-3.861567,2.129116,-4.596167,-4.240643,1.913970,-8.750407,8.856379,-6.016219,1.707018,-9.969750,9.860680,-4.071635,1.178483,-7.447898,7.851243,-3.791006,-7.342176,-7.962953,8.321994,-6.232384,-0.102010,-3.771056,-7.105027,8.042423,-5.902484,3.556463,0.205744,1.302317,3.554335,2.947869,-5.913005,7.037177,-5.069109,2.355483,-4.987271,-1.033556,-3.218781,-2.722593,0.168338,7.443494,-3.369409,1.779266,-0.713901,-2.006838,-3.537860,-2.794670,8.437048,-7.305738,6.500713,-0.686969,8.916274,1.068770,7.671439,5.869880,6.137840,-2.975016,1.990861,0.565028,-4.356350,-5.018075,-8.305615,9.754138,-4.957491,8.632640,9.040855,0.408261,-7.243449,0.934124,-1.449691,5.734102,2.330155,4.707363,8.678200,-8.645220,-5.151193,6.369045,7.740040,6.656474,-0.923285,6.898348,-1.550879,-5.571874,0.701922,-1.877513,-4.413904,4.582218,5.465879,2.145418,-9.688912,3.038226,-8.999229,7.504263,7.342055,9.554170,3.375559,-6.113062,6.188657,-1.587160,-7.417037,-0.271017,-2.666990,-5.813632,3.594950,6.178825,0.045412,-9.051889,7.230148,2.960725,-1.772508,-6.710293,2.648489,5.602693,5.815193,-2.483087,-8.424491,2.299408,-5.541602,8.677461,7.056959,1.327181,0.179949,-2.305487,4.213657,3.336550,-2.809100,6.487872,-4.836218,-4.792099,4.978264,8.553084,4.617115,-5.757473,-5.504247,9.931611,-7.538978,-4.358568,6.713311,-7.613892,-2.297098,-3.576651,5.639417,6.708710,2.620576,-9.988071,-7.521227,-0.825729,-3.484042,3.769554,-9.165228,-1.079136,-1.668830,-6.354964,7.982386,-4.128992,3.870421,-0.328808,3.465150,-0.909812,2.985916,2.775548,5.481381,2.335425,-6.834464,-4.900069,-5.368213,-1.479944,-4.886825,-3.865581,8.231428,4.444148,7.456995,-7.184793,2.033912,0.610468,3.991070,-6.916654,4.325061,-2.150983,0.504869,-0.244079,1.423248,-5.420932,-1.060642,9.866781,2.688005,-4.500593,-6.648630,5.233649,3.010073,-2.700731,4.977582,5.272909,-2.097153,4.730276,6.551188,0.412280,-9.181818,-6.487175,-1.573536,-3.806042,-8.896464,6.990922,8.739265,2.137698,5.026360,9.683988,-1.156792,3.552619,-5.734757,-4.950441,2.501855,-7.121064,0.615488,-4.803792,-8.349009,-8.159241,1.137424,6.517494,-6.043876,0.623222,-7.567884,7.140492,-5.442321,-6.345640,3.829918,-5.361904,-5.311573,-1.164774,6.842704,-5.882051,6.781213,-7.183705,-2.567221,-0.596637,-3.304279,-0.436898,-2.361053,-6.352503,3.387122,-3.009446,8.629642,6.794623,1.311178,-4.286322,-0.578739,1.152390,6.161809,-7.007082,4.270027,-4.611471,-9.107744,3.680706,-8.653746,2.128697,-1.075035,-7.121830,-5.163965,7.806910,0.943943,-8.987478,-3.765778,-7.224009,-9.196213,1.395628,4.059832,0.918108,5.642456,-7.085803,-5.198568,1.805854,4.758723,1.344275,5.751272,-8.145273,-3.114890,-9.108512,-9.485113,-0.592065,3.834944,-1.479711,2.420078,-3.098958,-7.690404,8.705494,5.772600,7.317554,-2.770187,-7.212167,-6.356601,1.507047,-7.749484,8.326565,-0.911695,1.728230,-2.241814,-1.925620,4.212798,-2.689184,2.635297,2.116009,-9.537543,-9.674540,2.752392,-4.700101,7.511346,-3.294219,8.378109,-5.238758,1.857801,2.423304,7.140944,7.246775,-6.954454,0.377984,8.789421,-7.255388,4.156623,0.278848,-1.805445,-8.391307,6.131520,6.578881,9.254939,-8.852053,4.044118,-2.625157,3.263611,6.008529,2.501636,8.292350,4.525710,-6.431894,-8.178221,8.169187,9.432544,-2.909726,-9.817759,9.310285,5.095931,-3.558883,2.613216,-1.251025,4.423452,-2.167104,-9.125258,6.915787,-1.813961,-6.513827,9.824098,3.703774,-2.930904,1.443149,3.123656,-4.309887,-3.945380,6.520775,1.455413,-3.051104,7.773998,5.090876,7.548522,-0.611421,-1.050194,-5.838082,-6.638918,-8.321827,-0.710770,-8.214075,-2.436823,-4.707673,2.571283,-4.728294,5.208523,5.945726,0.265017,-9.305742,-5.821913,-1.071676,-6.632041,-4.544490,-1.430735,1.273611,3.808575,-5.948306,-3.222890,-5.189377,-6.089834,-8.864358,-2.817167,8.367221,-2.592370,-8.714509,-0.683362,1.698174,9.643603,9.915809,-8.059703,-0.052522,-1.749317,6.040034,-4.089053,9.687646,-6.718641,-4.387909,1.262656,-4.177479,-0.287349,-4.014960,-8.251445,2.291948,-3.466084,4.712020,0.946495,-1.119657,0.536341,-8.640177,1.185719,-8.902704,9.118580,-0.462080,-1.900285,-1.177092,0.556379,7.690574,-4.485246,-9.202997,-2.480880,7.953144,4.534173,9.395597,5.843290,-1.807323,0.998808,-0.038320,-3.247270,-9.974577,2.758498,2.877805,7.710611,-8.934940,4.570702,-5.901279,0.169303,6.699251,7.974288,2.520491,-6.286318,-4.188695,-3.548444,-5.779298,1.985438,-4.760075,4.586214,-8.946519,-7.351729,-6.187912,5.712484,7.753892,-7.183231,6.576989,-6.678353,-1.481981,1.893919,-5.292443,4.209398,-2.287287,2.341873,2.963235,-2.039491,-1.634106,3.611249,-5.037251,3.737932,4.438469,-6.058426,-2.049731,3.031956,3.751355,4.020788,-6.788666,-5.101005,5.385980,2.693061,-8.992714,-2.048674,4.226557,0.218298], dtype = "float32")#candidate|916|(840,)|const|float32
const_917 = relay.const([-1,9,6,-8,1,7,-10,1,1,3,-5,5,9,9,-4,-8,-5,-2,6,-10,-5,-1,7,6,-10,-1,-9,4,4,5,-2,-5,-1,-6,5,6,-2,7,-10,1,-6,-4,-9,-3,4,2,-3,10,4,-1,-5,2,-3,-9,4,5,-2,-4,-7,-5,5,6,-6,1,6,10,-8,-9,1,-4,-7,6,-7,-2,7,5,5,-1,-10,1,-4,-5,3,-1,-1,-1,5,-8,10,-7,7,1,5,-5,2,-1,8,9,10,-8,-9,9,-8,6,-9,5,9,2,-5,9,2,6,-5,-1,-6,-6,-10,9,-2,-6,-10,-10,6,-2,-8,2,10,10,-6,-10,2,9,1,-1,-3,-5,-7,-3,-7,-8,6,5,-2,3,9,-10,9,4,7,4,-2,4,-6,-4,2,5,6,3,-6,3,10,-6,-5,9,-4,2,4,7], dtype = "uint32")#candidate|917|(168,)|const|uint32
call_914 = relay.TupleGetItem(func_322_call(relay.reshape(var_915.astype('float64'), [5, 8]), relay.reshape(const_916.astype('float32'), [840,]), relay.reshape(var_901.astype('uint32'), []), relay.reshape(const_917.astype('uint32'), [168,]), relay.reshape(var_915.astype('float64'), [5, 8]), ), 3)
call_918 = relay.TupleGetItem(func_329_call(relay.reshape(var_915.astype('float64'), [5, 8]), relay.reshape(const_916.astype('float32'), [840,]), relay.reshape(var_901.astype('uint32'), []), relay.reshape(const_917.astype('uint32'), [168,]), relay.reshape(var_915.astype('float64'), [5, 8]), ), 3)
bop_921 = relay.mod(var_902.astype('float64'), var_901.astype('float64')) # shape=(3, 5, 8)
uop_924 = relay.atan(bop_903.astype('float64')) # shape=(3, 5, 8)
output = relay.Tuple([call_909,call_914,var_915,const_916,const_917,bop_921,uop_924,])
output2 = relay.Tuple([call_910,call_918,var_915,const_916,const_917,bop_921,uop_924,])
func_926 = relay.Function([var_901,var_902,var_915,], output)
mod['func_926'] = func_926
mod = relay.transform.InferType()(mod)
mutated_mod['func_926'] = func_926
mutated_mod = relay.transform.InferType()(mutated_mod)
func_926_call = mutated_mod.get_global_var('func_926')
var_928 = relay.var("var_928", dtype = "float32", shape = ())#candidate|928|()|var|float32
var_929 = relay.var("var_929", dtype = "float32", shape = (3, 5, 8))#candidate|929|(3, 5, 8)|var|float32
var_930 = relay.var("var_930", dtype = "float64", shape = (1, 40))#candidate|930|(1, 40)|var|float64
call_927 = func_926_call(var_928,var_929,var_930,)
output = call_927
func_931 = relay.Function([var_928,var_929,var_930,], output)
mutated_mod['func_931'] = func_931
mutated_mod = relay.transform.InferType()(mutated_mod)
var_952 = relay.var("var_952", dtype = "bool", shape = (5, 2, 15))#candidate|952|(5, 2, 15)|var|bool
var_953 = relay.var("var_953", dtype = "bool", shape = (5, 2, 15))#candidate|953|(5, 2, 15)|var|bool
bop_954 = relay.logical_and(var_952.astype('bool'), relay.reshape(var_953.astype('bool'), relay.shape_of(var_952))) # shape=(5, 2, 15)
output = bop_954
output2 = bop_954
func_973 = relay.Function([var_952,var_953,], output)
mod['func_973'] = func_973
mod = relay.transform.InferType()(mod)
mutated_mod['func_973'] = func_973
mutated_mod = relay.transform.InferType()(mutated_mod)
func_973_call = mutated_mod.get_global_var('func_973')
var_975 = relay.var("var_975", dtype = "bool", shape = (5, 2, 15))#candidate|975|(5, 2, 15)|var|bool
var_976 = relay.var("var_976", dtype = "bool", shape = (5, 2, 15))#candidate|976|(5, 2, 15)|var|bool
call_974 = func_973_call(var_975,var_976,)
output = call_974
func_977 = relay.Function([var_975,var_976,], output)
mutated_mod['func_977'] = func_977
mutated_mod = relay.transform.InferType()(mutated_mod)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_992 = relay.TupleGetItem(func_635_call(), 0)
call_993 = relay.TupleGetItem(func_636_call(), 0)
uop_996 = relay.cosh(call_992.astype('float64')) # shape=(1, 2, 9)
uop_998 = relay.cosh(call_993.astype('float64')) # shape=(1, 2, 9)
uop_1003 = relay.sigmoid(uop_996.astype('float64')) # shape=(1, 2, 9)
uop_1005 = relay.sigmoid(uop_998.astype('float64')) # shape=(1, 2, 9)
bop_1007 = relay.greater_equal(uop_1003.astype('bool'), relay.reshape(call_992.astype('bool'), relay.shape_of(uop_1003))) # shape=(1, 2, 9)
bop_1010 = relay.greater_equal(uop_1005.astype('bool'), relay.reshape(call_993.astype('bool'), relay.shape_of(uop_1005))) # shape=(1, 2, 9)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_1012 = relay.TupleGetItem(func_635_call(), 0)
call_1013 = relay.TupleGetItem(func_636_call(), 0)
const_1018 = relay.const([[[-1.318426,2.959921,6.473895,-2.463027,2.448516,-0.458079,-9.736512,-2.994450,5.857723],[-6.055572,-0.399208,-5.468999,8.257106,1.716386,4.013306,-0.716022,6.302716,9.449262]],[[-6.205877,-9.741310,-8.041341,-9.971001,5.498471,4.581606,-1.470905,-2.159908,1.405219],[7.532613,-3.376848,-9.413769,0.890020,-0.123292,-4.219439,0.383570,5.836146,-4.246147]],[[-3.748940,-4.072054,-4.291013,-3.947108,7.948850,7.573728,8.899913,9.027049,-6.227015],[-0.782853,-8.550524,-8.553244,-8.821426,3.152818,5.685358,9.758558,5.113950,9.019839]]], dtype = "float64")#candidate|1018|(3, 2, 9)|const|float64
bop_1019 = relay.left_shift(uop_996.astype('uint64'), const_1018.astype('uint64')) # shape=(3, 2, 9)
bop_1022 = relay.left_shift(uop_998.astype('uint64'), const_1018.astype('uint64')) # shape=(3, 2, 9)
bop_1030 = relay.logical_xor(bop_1019.astype('uint8'), uop_1003.astype('uint8')) # shape=(3, 2, 9)
bop_1033 = relay.logical_xor(bop_1022.astype('uint8'), uop_1005.astype('uint8')) # shape=(3, 2, 9)
func_824_call = mod.get_global_var('func_824')
func_827_call = mutated_mod.get_global_var('func_827')
const_1044 = relay.const([[6.278936],[-7.705631],[6.872934],[-4.133882],[-6.834551],[9.587011],[8.290664],[-1.013257],[0.103595],[7.390976],[9.602869],[9.648152],[-4.780537],[9.239532],[1.743961],[2.437450],[-7.599061],[-0.171871],[2.792619],[5.013262],[3.199579],[-2.685150],[1.796314],[-7.652260],[3.983143],[8.872917],[-9.824746],[-4.113540],[0.426440],[6.773413],[5.444362],[0.129294],[-0.339794],[5.213663],[-4.563206],[-2.611044],[-1.752393],[5.758372],[5.119223],[-2.745069],[8.418149],[3.307023],[-7.269032],[0.996996],[-6.869045],[5.482809],[0.397317],[7.344163],[-8.991895],[-8.477025],[1.146317],[-3.574942],[9.723549],[3.224173],[8.441895],[0.488712],[-5.603368],[-9.274221],[-6.761469],[-2.352261],[-9.247473],[5.448060],[9.686959],[-3.770536],[1.176006],[-2.596572],[6.356787],[-6.136743],[-5.426505],[2.220393],[-5.983828],[9.012788],[7.951579],[6.034358],[5.060003],[-5.198205],[9.344254],[8.569289],[-1.319075],[-1.375098],[-8.769705],[-4.457720],[-0.150495],[0.048745],[8.667443],[-8.534226],[5.171520],[-0.787001],[-2.685485],[9.152809],[6.794035],[3.837840],[0.161788],[1.808647],[-7.458567],[0.032308],[6.756274],[-3.826805],[3.182166],[-7.635457],[-5.100016],[-6.376757],[-7.509118],[4.790471],[-3.519647],[-9.692564],[-6.932482],[7.268596],[2.008994],[-2.315931],[-2.960984],[-5.127664],[5.290591],[2.515692],[-4.951377],[6.999318],[1.755173],[9.744594],[-0.168596],[-2.763027],[6.132438],[4.956764],[-1.743214],[7.454564],[2.439739],[-3.072116],[3.196043],[8.856996],[6.101191],[2.250474],[5.710001],[3.897300],[-4.773753],[-5.283492],[6.957704],[-4.110131],[-0.306242],[-1.348138],[-8.049249],[-0.678982],[3.661635],[7.013444],[3.309816],[-4.781055],[-9.606184],[7.949490],[-9.642783],[-4.403262],[6.453828],[-6.321903],[9.188422],[2.323104],[-0.998347],[-5.421925],[7.953599],[-4.477480],[4.380619],[5.524028],[2.770203],[-8.339977],[-0.126279],[4.353249],[-8.609355],[5.063852],[-6.103030],[-4.069331],[-0.605977],[1.860864],[0.918036],[1.655757],[-3.613092],[-3.912698],[-8.752554],[9.726959],[3.860209],[5.889450],[2.923462],[-7.316266],[-7.904957],[6.106552],[6.832860],[-8.633880],[1.709516],[-8.534931],[5.123211],[1.291969],[-9.530428],[-7.053390],[2.955319],[-0.432711],[5.272986],[-5.618842],[-2.086886],[-9.272945],[0.269191],[0.032311],[3.401789],[-8.357992],[-2.711115],[6.425784],[3.840319],[5.075650],[-7.272632],[-2.305866],[8.728462],[-6.834195],[-1.606780],[3.042060],[-9.182491],[3.622290],[6.786654],[-6.926942],[-2.019710],[9.995766],[-5.958575],[-1.112350],[2.398749],[3.076792],[3.378783],[-6.518484],[-9.185669],[1.403316],[6.660140],[9.159819],[-1.153837],[4.403337],[6.652152],[-4.877229],[1.577549],[-6.002228],[3.712615],[4.935872],[1.260492],[-4.529369],[-8.968934],[4.081520],[-6.255036],[6.534621],[-4.961387],[9.924514],[4.692143],[-5.870618],[-4.042291],[-9.425515],[2.547156],[0.436786],[2.629113],[4.935023],[2.466720],[-1.725821],[-6.014632],[-6.667061],[-9.661459],[2.330721],[-0.624862],[-6.054305],[3.482390],[1.466051],[5.129789],[3.722888],[-4.528004],[4.764833],[5.779665],[-4.621306],[2.480373],[-3.268047],[5.114485],[-7.042463],[-0.987699],[-3.358328]], dtype = "float64")#candidate|1044|(270, 1)|const|float64
call_1043 = func_824_call(relay.reshape(const_1044.astype('float64'), [15, 2, 9]))
call_1045 = func_824_call(relay.reshape(const_1044.astype('float64'), [15, 2, 9]))
output = relay.Tuple([bop_1007,call_1012,bop_1030,call_1043,const_1044,])
output2 = relay.Tuple([bop_1010,call_1013,bop_1033,call_1045,const_1044,])
func_1054 = relay.Function([], output)
mod['func_1054'] = func_1054
mod = relay.transform.InferType()(mod)
output = func_1054()
func_1055 = relay.Function([], output)
mutated_mod['func_1055'] = func_1055
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1058 = relay.var("var_1058", dtype = "bool", shape = (16, 10))#candidate|1058|(16, 10)|var|bool
var_1059 = relay.var("var_1059", dtype = "bool", shape = (16, 10))#candidate|1059|(16, 10)|var|bool
bop_1060 = relay.logical_or(var_1058.astype('bool'), relay.reshape(var_1059.astype('bool'), relay.shape_of(var_1058))) # shape=(16, 10)
uop_1074 = relay.exp(bop_1060.astype('float64')) # shape=(16, 10)
bop_1076 = relay.right_shift(bop_1060.astype('int64'), relay.reshape(var_1058.astype('int64'), relay.shape_of(bop_1060))) # shape=(16, 10)
bop_1082 = relay.power(uop_1074.astype('float32'), relay.reshape(bop_1060.astype('float32'), relay.shape_of(uop_1074))) # shape=(16, 10)
func_143_call = mod.get_global_var('func_143')
func_146_call = mutated_mod.get_global_var('func_146')
var_1088 = relay.var("var_1088", dtype = "float32", shape = (840,))#candidate|1088|(840,)|var|float32
call_1087 = relay.TupleGetItem(func_143_call(relay.reshape(var_1088.astype('float32'), [7, 8, 15])), 0)
call_1089 = relay.TupleGetItem(func_146_call(relay.reshape(var_1088.astype('float32'), [7, 8, 15])), 0)
output = relay.Tuple([bop_1076,bop_1082,call_1087,var_1088,])
output2 = relay.Tuple([bop_1076,bop_1082,call_1089,var_1088,])
func_1100 = relay.Function([var_1058,var_1059,var_1088,], output)
mod['func_1100'] = func_1100
mod = relay.transform.InferType()(mod)
mutated_mod['func_1100'] = func_1100
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1100_call = mutated_mod.get_global_var('func_1100')
var_1102 = relay.var("var_1102", dtype = "bool", shape = (16, 10))#candidate|1102|(16, 10)|var|bool
var_1103 = relay.var("var_1103", dtype = "bool", shape = (16, 10))#candidate|1103|(16, 10)|var|bool
var_1104 = relay.var("var_1104", dtype = "float32", shape = (840,))#candidate|1104|(840,)|var|float32
call_1101 = func_1100_call(var_1102,var_1103,var_1104,)
output = call_1101
func_1105 = relay.Function([var_1102,var_1103,var_1104,], output)
mutated_mod['func_1105'] = func_1105
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1054_call = mod.get_global_var('func_1054')
func_1055_call = mutated_mod.get_global_var('func_1055')
call_1127 = relay.TupleGetItem(func_1054_call(), 0)
call_1128 = relay.TupleGetItem(func_1055_call(), 0)
var_1143 = relay.var("var_1143", dtype = "bool", shape = (11, 2, 9))#candidate|1143|(11, 2, 9)|var|bool
bop_1144 = relay.less_equal(call_1127.astype('bool'), var_1143.astype('bool')) # shape=(11, 2, 9)
bop_1147 = relay.less_equal(call_1128.astype('bool'), var_1143.astype('bool')) # shape=(11, 2, 9)
output = relay.Tuple([bop_1144,])
output2 = relay.Tuple([bop_1147,])
func_1152 = relay.Function([var_1143,], output)
mod['func_1152'] = func_1152
mod = relay.transform.InferType()(mod)
var_1153 = relay.var("var_1153", dtype = "bool", shape = (11, 2, 9))#candidate|1153|(11, 2, 9)|var|bool
output = func_1152(var_1153)
func_1154 = relay.Function([var_1153], output)
mutated_mod['func_1154'] = func_1154
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1054_call = mod.get_global_var('func_1054')
func_1055_call = mutated_mod.get_global_var('func_1055')
call_1164 = relay.TupleGetItem(func_1054_call(), 2)
call_1165 = relay.TupleGetItem(func_1055_call(), 2)
uop_1171 = relay.tan(call_1164.astype('float64')) # shape=(3, 2, 9)
uop_1173 = relay.tan(call_1165.astype('float64')) # shape=(3, 2, 9)
bop_1175 = relay.floor_divide(uop_1171.astype('float64'), relay.reshape(call_1164.astype('float64'), relay.shape_of(uop_1171))) # shape=(3, 2, 9)
bop_1178 = relay.floor_divide(uop_1173.astype('float64'), relay.reshape(call_1165.astype('float64'), relay.shape_of(uop_1173))) # shape=(3, 2, 9)
uop_1179 = relay.atanh(bop_1175.astype('float64')) # shape=(3, 2, 9)
uop_1181 = relay.atanh(bop_1178.astype('float64')) # shape=(3, 2, 9)
output = uop_1179
output2 = uop_1181
func_1185 = relay.Function([], output)
mod['func_1185'] = func_1185
mod = relay.transform.InferType()(mod)
output = func_1185()
func_1186 = relay.Function([], output)
mutated_mod['func_1186'] = func_1186
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1187 = relay.const([[[-4.076703,4.795759],[-2.138299,0.650700],[-0.831052,3.901508],[0.852839,-4.945937],[9.336037,-7.381951],[1.449504,-0.399400],[5.116744,-1.802533],[-7.206465,9.146804],[-2.352666,0.987156],[-3.876380,9.850326],[0.316069,-8.818510],[5.241696,-3.000841],[5.260039,-3.356685],[3.283987,-7.473744],[-1.654642,-2.240013]],[[-8.431660,-2.686890],[-0.158175,3.238719],[-9.759989,4.376802],[-1.657721,-1.251114],[2.162651,-7.561776],[6.211918,5.324630],[-5.211745,-5.906365],[-1.682554,0.892709],[-7.424767,1.038566],[2.177491,1.172726],[7.211627,6.496357],[-9.195351,-1.201922],[-8.128546,-7.376422],[2.294414,4.985695],[1.993167,0.874733]]], dtype = "float32")#candidate|1187|(2, 15, 2)|const|float32
uop_1188 = relay.sigmoid(const_1187.astype('float32')) # shape=(2, 15, 2)
func_696_call = mod.get_global_var('func_696')
func_699_call = mutated_mod.get_global_var('func_699')
var_1194 = relay.var("var_1194", dtype = "uint64", shape = (84,))#candidate|1194|(84,)|var|uint64
call_1193 = relay.TupleGetItem(func_696_call(relay.reshape(var_1194.astype('uint64'), [6, 14])), 0)
call_1195 = relay.TupleGetItem(func_699_call(relay.reshape(var_1194.astype('uint64'), [6, 14])), 0)
uop_1196 = relay.sin(uop_1188.astype('float32')) # shape=(2, 15, 2)
var_1206 = relay.var("var_1206", dtype = "float32", shape = (2, 15, 2))#candidate|1206|(2, 15, 2)|var|float32
bop_1207 = relay.floor_divide(uop_1188.astype('float32'), relay.reshape(var_1206.astype('float32'), relay.shape_of(uop_1188))) # shape=(2, 15, 2)
uop_1220 = relay.atan(uop_1196.astype('float32')) # shape=(2, 15, 2)
bop_1229 = relay.bitwise_and(uop_1220.astype('int32'), relay.reshape(uop_1188.astype('int32'), relay.shape_of(uop_1220))) # shape=(2, 15, 2)
output = relay.Tuple([call_1193,var_1194,bop_1207,bop_1229,])
output2 = relay.Tuple([call_1195,var_1194,bop_1207,bop_1229,])
func_1233 = relay.Function([var_1194,var_1206,], output)
mod['func_1233'] = func_1233
mod = relay.transform.InferType()(mod)
mutated_mod['func_1233'] = func_1233
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1233_call = mutated_mod.get_global_var('func_1233')
var_1235 = relay.var("var_1235", dtype = "uint64", shape = (84,))#candidate|1235|(84,)|var|uint64
var_1236 = relay.var("var_1236", dtype = "float32", shape = (2, 15, 2))#candidate|1236|(2, 15, 2)|var|float32
call_1234 = func_1233_call(var_1235,var_1236,)
output = call_1234
func_1237 = relay.Function([var_1235,var_1236,], output)
mutated_mod['func_1237'] = func_1237
mutated_mod = relay.transform.InferType()(mutated_mod)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_1264 = relay.TupleGetItem(func_635_call(), 0)
call_1265 = relay.TupleGetItem(func_636_call(), 0)
var_1286 = relay.var("var_1286", dtype = "bool", shape = (14, 2, 9))#candidate|1286|(14, 2, 9)|var|bool
bop_1287 = relay.add(call_1264.astype('uint8'), var_1286.astype('uint8')) # shape=(14, 2, 9)
bop_1290 = relay.add(call_1265.astype('uint8'), var_1286.astype('uint8')) # shape=(14, 2, 9)
output = relay.Tuple([bop_1287,])
output2 = relay.Tuple([bop_1290,])
func_1293 = relay.Function([var_1286,], output)
mod['func_1293'] = func_1293
mod = relay.transform.InferType()(mod)
var_1294 = relay.var("var_1294", dtype = "bool", shape = (14, 2, 9))#candidate|1294|(14, 2, 9)|var|bool
output = func_1293(var_1294)
func_1295 = relay.Function([var_1294], output)
mutated_mod['func_1295'] = func_1295
mutated_mod = relay.transform.InferType()(mutated_mod)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_1314 = relay.TupleGetItem(func_635_call(), 0)
call_1315 = relay.TupleGetItem(func_636_call(), 0)
output = call_1314
output2 = call_1315
func_1319 = relay.Function([], output)
mod['func_1319'] = func_1319
mod = relay.transform.InferType()(mod)
mutated_mod['func_1319'] = func_1319
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1319_call = mutated_mod.get_global_var('func_1319')
call_1320 = func_1319_call()
output = call_1320
func_1321 = relay.Function([], output)
mutated_mod['func_1321'] = func_1321
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1322 = relay.const([[[False,True,False,False,True,True,False,False,False,True,False,False,False,False],[True,False,True,False,True,False,True,True,False,True,True,True,False,True],[False,True,True,False,True,False,False,False,True,False,False,True,True,False],[True,True,False,False,True,True,False,True,True,False,False,False,True,False],[True,False,True,True,True,True,True,False,True,True,True,False,True,True],[True,False,False,False,True,False,False,True,False,False,False,True,False,False],[True,False,False,False,False,False,True,True,True,False,False,False,True,True],[False,False,False,True,False,False,True,False,True,True,True,True,True,True],[False,False,True,False,False,False,False,False,False,False,True,True,False,True],[True,False,False,True,False,True,False,True,True,True,True,False,False,False],[True,True,True,False,True,False,False,True,False,False,True,False,False,True],[True,True,False,False,True,True,True,True,False,True,False,True,False,True],[False,False,True,True,False,False,True,True,True,True,True,True,False,False],[False,False,True,True,True,False,False,False,True,True,False,True,True,False],[False,True,False,False,False,True,False,False,False,True,True,True,True,False],[False,True,True,False,True,False,False,True,False,False,True,True,True,True]],[[False,False,False,True,False,False,True,False,True,True,False,True,True,True],[False,False,False,True,True,True,True,False,True,True,True,False,True,True],[True,True,False,False,True,False,True,True,False,False,True,False,False,True],[True,True,True,False,True,True,True,False,False,False,False,True,False,False],[True,True,True,True,False,True,True,False,False,False,True,False,False,True],[False,False,False,False,True,True,False,False,False,True,False,False,True,True],[False,True,True,True,True,False,False,True,True,True,True,True,True,True],[False,False,False,False,False,False,False,False,False,False,False,False,True,True],[True,True,False,True,False,True,False,True,True,True,True,False,False,False],[False,False,True,False,False,True,False,False,False,True,False,False,True,True],[False,True,False,True,False,False,False,False,True,True,False,False,False,True],[False,False,True,False,False,True,False,False,False,True,True,False,False,True],[False,True,False,True,False,False,False,False,False,True,True,True,False,True],[True,False,False,False,False,True,False,True,False,True,True,True,False,False],[True,False,False,False,True,False,True,False,True,True,False,True,False,True],[True,False,False,False,False,False,False,True,False,True,False,False,False,True]],[[True,False,False,True,False,False,True,True,False,False,True,True,True,False],[True,False,True,False,False,False,False,True,False,True,False,False,False,False],[True,False,True,False,False,True,True,False,True,False,True,False,False,True],[True,False,True,False,False,False,False,False,False,True,True,True,False,True],[True,False,True,False,False,True,True,False,False,False,False,False,True,True],[False,True,False,True,True,True,False,True,False,False,False,False,True,True],[True,True,True,True,True,False,True,True,True,False,True,False,True,True],[False,False,True,False,True,True,True,True,True,True,False,True,False,False],[True,True,False,True,True,True,False,False,True,True,False,True,False,True],[True,True,True,True,True,False,True,False,True,False,True,False,False,False],[True,False,False,True,True,False,False,False,True,False,True,True,True,True],[False,True,False,True,False,True,False,True,True,True,False,True,True,True],[True,True,True,True,True,False,False,True,False,False,True,False,False,False],[True,True,True,True,True,False,False,True,True,False,True,True,True,True],[False,False,False,True,False,False,False,True,False,False,False,False,False,False],[False,False,False,True,False,False,False,True,False,False,False,True,False,True]],[[False,True,False,False,False,True,False,False,True,False,True,True,True,False],[True,True,False,False,False,False,True,False,False,False,True,False,False,True],[False,True,True,False,False,True,True,False,False,True,False,True,True,False],[False,False,False,False,False,False,False,False,False,True,False,True,True,True],[True,False,False,True,True,True,False,True,True,True,False,True,False,False],[True,True,False,True,False,False,True,False,False,True,False,True,True,False],[False,False,True,True,False,True,True,True,True,True,True,False,False,True],[True,False,True,False,True,False,False,True,False,True,True,True,True,True],[False,False,True,False,False,False,True,True,False,False,False,True,True,True],[True,True,False,False,True,False,True,True,False,True,False,False,False,False],[True,True,True,True,True,True,True,False,True,True,True,True,True,True],[False,True,True,True,False,True,False,False,True,True,True,True,False,True],[False,False,True,True,False,True,True,True,False,False,True,True,True,False],[True,True,False,True,False,True,True,False,False,True,False,False,False,True],[True,False,True,True,False,False,True,False,True,False,False,True,False,True],[True,False,False,False,True,False,True,True,True,False,True,True,True,False]],[[True,True,True,True,True,False,False,False,True,True,False,False,True,False],[False,False,False,True,False,False,True,True,False,False,False,True,True,True],[False,False,True,False,False,True,True,True,True,True,True,False,True,False],[True,False,False,True,False,False,False,False,False,True,True,False,True,True],[True,False,True,False,True,False,False,True,True,True,False,True,False,False],[True,True,False,False,False,False,True,False,True,True,True,True,False,False],[False,False,False,True,False,True,True,True,True,True,False,True,False,True],[False,True,True,True,False,True,False,False,True,True,True,False,True,False],[False,True,False,False,True,False,True,False,True,True,True,True,False,False],[False,True,True,False,False,False,True,False,True,True,True,False,True,False],[True,False,False,True,True,False,True,True,True,False,True,False,True,False],[True,False,False,True,True,True,True,True,True,True,True,False,False,False],[True,True,False,False,True,False,True,True,True,False,False,False,False,True],[False,False,True,False,False,False,True,True,True,True,False,True,False,True],[True,False,False,False,False,False,False,True,True,True,False,False,True,False],[False,False,True,False,False,True,False,False,True,False,True,False,True,True]],[[True,True,False,True,True,False,True,True,True,False,True,False,True,True],[False,True,False,False,True,True,False,True,False,True,True,True,True,False],[False,False,True,True,True,True,False,False,False,False,False,True,False,True],[True,True,False,True,False,False,True,True,False,True,False,False,False,True],[False,True,True,False,True,False,True,False,False,False,True,False,False,True],[True,False,False,False,True,False,True,False,True,False,True,True,True,False],[True,False,True,True,True,True,True,True,True,True,True,True,True,False],[True,True,True,False,True,True,False,False,False,True,False,True,True,False],[False,True,False,True,True,True,True,False,False,False,True,False,True,True],[True,False,True,True,True,True,True,False,False,False,True,False,True,True],[True,True,True,True,False,True,True,True,True,False,True,True,False,True],[True,False,False,True,False,True,False,False,False,True,False,True,True,True],[True,True,True,True,False,False,False,False,False,True,True,True,True,False],[False,False,True,False,False,True,True,False,True,True,False,True,False,True],[False,False,False,False,True,True,True,True,False,True,True,False,True,False],[True,False,False,True,False,False,True,False,True,False,True,False,True,True]],[[False,False,False,False,False,True,False,True,False,True,False,False,True,True],[False,False,True,True,False,True,True,True,True,False,True,True,True,False],[True,False,False,True,False,False,False,False,True,False,True,False,False,True],[False,True,False,True,True,True,False,True,True,True,False,False,False,False],[True,True,False,True,True,False,False,False,True,False,False,False,True,False],[False,True,True,True,False,False,False,True,True,False,False,False,False,True],[True,False,True,False,True,True,True,False,False,False,False,True,False,True],[True,True,True,False,False,False,True,False,False,True,True,False,True,False],[False,True,True,True,True,False,False,False,True,True,True,True,True,True],[False,False,False,False,True,True,False,False,False,True,False,False,False,False],[False,True,False,True,True,True,False,False,True,False,True,False,False,False],[False,True,True,False,True,False,False,True,True,False,True,True,True,True],[False,True,True,False,True,True,True,False,False,False,False,True,False,True],[False,False,True,False,False,True,False,True,True,True,False,False,True,True],[False,True,True,False,False,False,False,True,False,False,True,False,False,False],[False,False,True,False,True,True,False,True,False,False,False,True,True,True]],[[True,True,False,True,False,True,True,False,False,False,False,False,False,True],[False,False,True,False,False,True,False,True,False,False,False,False,True,True],[True,False,False,False,True,False,False,True,False,True,False,False,True,False],[False,True,True,False,True,True,True,False,False,True,True,False,False,False],[False,True,True,True,True,True,True,True,True,True,False,True,True,False],[True,False,False,True,False,False,False,True,True,True,True,True,False,True],[True,False,True,True,True,False,False,True,True,True,False,False,True,False],[False,False,True,True,False,True,True,False,True,True,False,False,False,True],[True,False,False,False,True,True,True,False,True,True,True,False,True,True],[True,False,False,True,False,True,False,False,False,True,True,False,False,True],[False,False,False,False,True,True,False,False,False,False,False,False,True,False],[False,False,True,True,False,True,False,False,False,True,True,True,False,False],[True,True,True,True,True,True,True,False,False,False,False,True,False,True],[True,True,True,True,False,False,True,False,False,True,True,False,False,False],[True,True,False,False,False,True,False,True,False,True,False,True,True,False],[False,True,True,True,True,False,False,True,True,True,True,True,True,False]],[[True,True,False,True,False,False,True,False,True,True,False,False,True,False],[True,False,False,True,False,False,False,True,False,False,False,True,False,True],[False,True,False,False,False,False,True,True,True,False,True,False,False,True],[False,True,True,True,True,False,False,True,False,False,False,False,True,False],[False,True,False,False,False,False,False,False,False,True,True,True,False,True],[False,False,False,False,True,False,False,False,False,False,False,False,True,False],[False,False,True,False,True,True,True,True,True,True,True,True,True,True],[True,True,False,True,True,False,True,False,False,True,False,False,False,False],[False,True,True,True,True,False,True,False,True,False,True,False,False,False],[True,True,True,False,False,True,True,True,False,True,True,False,False,True],[True,False,False,True,True,True,False,False,True,False,False,False,False,True],[False,False,False,True,True,True,False,True,True,True,False,True,False,True],[True,True,True,False,True,True,False,True,False,False,True,True,False,False],[True,True,True,True,True,True,False,True,True,False,False,False,False,True],[True,False,False,False,True,True,True,True,False,True,False,False,True,True],[True,False,True,False,True,True,True,False,False,False,True,True,False,False]]], dtype = "bool")#candidate|1322|(9, 16, 14)|const|bool
var_1323 = relay.var("var_1323", dtype = "bool", shape = (9, 16, 14))#candidate|1323|(9, 16, 14)|var|bool
bop_1324 = relay.logical_or(const_1322.astype('bool'), relay.reshape(var_1323.astype('bool'), relay.shape_of(const_1322))) # shape=(9, 16, 14)
output = bop_1324
output2 = bop_1324
func_1327 = relay.Function([var_1323,], output)
mod['func_1327'] = func_1327
mod = relay.transform.InferType()(mod)
var_1328 = relay.var("var_1328", dtype = "bool", shape = (9, 16, 14))#candidate|1328|(9, 16, 14)|var|bool
output = func_1327(var_1328)
func_1329 = relay.Function([var_1328], output)
mutated_mod['func_1329'] = func_1329
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1335 = relay.var("var_1335", dtype = "float32", shape = (15, 15))#candidate|1335|(15, 15)|var|float32
uop_1336 = relay.sinh(var_1335.astype('float32')) # shape=(15, 15)
func_1054_call = mod.get_global_var('func_1054')
func_1055_call = mutated_mod.get_global_var('func_1055')
call_1338 = relay.TupleGetItem(func_1054_call(), 2)
call_1339 = relay.TupleGetItem(func_1055_call(), 2)
uop_1342 = relay.log10(uop_1336.astype('float64')) # shape=(15, 15)
bop_1354 = relay.minimum(uop_1342.astype('int8'), relay.reshape(var_1335.astype('int8'), relay.shape_of(uop_1342))) # shape=(15, 15)
uop_1357 = relay.exp(uop_1336.astype('float32')) # shape=(15, 15)
func_926_call = mod.get_global_var('func_926')
func_931_call = mutated_mod.get_global_var('func_931')
var_1387 = relay.var("var_1387", dtype = "float32", shape = ())#candidate|1387|()|var|float32
var_1388 = relay.var("var_1388", dtype = "float32", shape = (30, 4))#candidate|1388|(30, 4)|var|float32
var_1389 = relay.var("var_1389", dtype = "float64", shape = (40,))#candidate|1389|(40,)|var|float64
call_1386 = relay.TupleGetItem(func_926_call(relay.reshape(var_1387.astype('float32'), []), relay.reshape(var_1388.astype('float32'), [3, 5, 8]), relay.reshape(var_1389.astype('float64'), [1, 40]), ), 6)
call_1390 = relay.TupleGetItem(func_931_call(relay.reshape(var_1387.astype('float32'), []), relay.reshape(var_1388.astype('float32'), [3, 5, 8]), relay.reshape(var_1389.astype('float64'), [1, 40]), ), 6)
var_1393 = relay.var("var_1393", dtype = "float32", shape = (15, 15))#candidate|1393|(15, 15)|var|float32
bop_1394 = relay.logical_or(uop_1336.astype('bool'), relay.reshape(var_1393.astype('bool'), relay.shape_of(uop_1336))) # shape=(15, 15)
output = relay.Tuple([call_1338,bop_1354,uop_1357,call_1386,var_1387,var_1388,var_1389,bop_1394,])
output2 = relay.Tuple([call_1339,bop_1354,uop_1357,call_1390,var_1387,var_1388,var_1389,bop_1394,])
func_1400 = relay.Function([var_1335,var_1387,var_1388,var_1389,var_1393,], output)
mod['func_1400'] = func_1400
mod = relay.transform.InferType()(mod)
mutated_mod['func_1400'] = func_1400
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1400_call = mutated_mod.get_global_var('func_1400')
var_1402 = relay.var("var_1402", dtype = "float32", shape = (15, 15))#candidate|1402|(15, 15)|var|float32
var_1403 = relay.var("var_1403", dtype = "float32", shape = ())#candidate|1403|()|var|float32
var_1404 = relay.var("var_1404", dtype = "float32", shape = (30, 4))#candidate|1404|(30, 4)|var|float32
var_1405 = relay.var("var_1405", dtype = "float64", shape = (40,))#candidate|1405|(40,)|var|float64
var_1406 = relay.var("var_1406", dtype = "float32", shape = (15, 15))#candidate|1406|(15, 15)|var|float32
call_1401 = func_1400_call(var_1402,var_1403,var_1404,var_1405,var_1406,)
output = call_1401
func_1407 = relay.Function([var_1402,var_1403,var_1404,var_1405,var_1406,], output)
mutated_mod['func_1407'] = func_1407
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1409 = relay.var("var_1409", dtype = "int8", shape = (2, 13))#candidate|1409|(2, 13)|var|int8
var_1410 = relay.var("var_1410", dtype = "int8", shape = (2, 13))#candidate|1410|(2, 13)|var|int8
bop_1411 = relay.less(var_1409.astype('bool'), relay.reshape(var_1410.astype('bool'), relay.shape_of(var_1409))) # shape=(2, 13)
output = relay.Tuple([bop_1411,])
output2 = relay.Tuple([bop_1411,])
func_1422 = relay.Function([var_1409,var_1410,], output)
mod['func_1422'] = func_1422
mod = relay.transform.InferType()(mod)
var_1423 = relay.var("var_1423", dtype = "int8", shape = (2, 13))#candidate|1423|(2, 13)|var|int8
var_1424 = relay.var("var_1424", dtype = "int8", shape = (2, 13))#candidate|1424|(2, 13)|var|int8
output = func_1422(var_1423,var_1424,)
func_1425 = relay.Function([var_1423,var_1424,], output)
mutated_mod['func_1425'] = func_1425
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1054_call = mod.get_global_var('func_1054')
func_1055_call = mutated_mod.get_global_var('func_1055')
call_1457 = relay.TupleGetItem(func_1054_call(), 3)
call_1458 = relay.TupleGetItem(func_1055_call(), 3)
func_1327_call = mod.get_global_var('func_1327')
func_1329_call = mutated_mod.get_global_var('func_1329')
var_1467 = relay.var("var_1467", dtype = "bool", shape = (2016,))#candidate|1467|(2016,)|var|bool
call_1466 = func_1327_call(relay.reshape(var_1467.astype('bool'), [9, 16, 14]))
call_1468 = func_1327_call(relay.reshape(var_1467.astype('bool'), [9, 16, 14]))
func_824_call = mod.get_global_var('func_824')
func_827_call = mutated_mod.get_global_var('func_827')
call_1472 = func_824_call(relay.reshape(call_1457.astype('float64'), [15, 2, 9]))
call_1473 = func_824_call(relay.reshape(call_1457.astype('float64'), [15, 2, 9]))
bop_1492 = relay.bitwise_xor(var_1467.astype('int32'), relay.reshape(call_1466.astype('int32'), relay.shape_of(var_1467))) # shape=(2016,)
bop_1495 = relay.bitwise_xor(var_1467.astype('int32'), relay.reshape(call_1468.astype('int32'), relay.shape_of(var_1467))) # shape=(2016,)
func_1400_call = mod.get_global_var('func_1400')
func_1407_call = mutated_mod.get_global_var('func_1407')
const_1497 = relay.const([6.545040,9.952890,-0.072815,2.114421,7.676380,8.639335,4.817840,-5.277115,-6.034938,9.503906,2.865162,2.676212,8.145236,-1.859937,-0.703609,-1.423418,6.208947,0.058006,8.198684,1.656179,1.720282,8.042701,5.155428,-1.451002,-1.325016,5.216901,-6.334939,-4.856948,1.390688,-9.366378,0.634624,7.435034,2.358699,4.906795,3.728571,2.417781,5.377241,1.273800,6.037692,9.674971,-2.017639,7.079010,5.237754,-4.631194,-9.531103,-8.323649,7.773546,-1.386071,5.923558,0.654665,3.576531,3.723641,2.343879,7.281205,-2.402554,-6.074318,0.622926,-4.599108,-0.904767,5.868630,6.149548,-2.308135,8.353060,8.141266,-6.843322,9.389139,-8.004070,0.509497,-8.441493,-5.433376,2.403791,2.982433,-1.592557,4.161381,-8.356981,6.048732,-6.183850,0.270820,-9.119379,-5.250103,7.933328,-2.760515,0.418091,-8.285430,-5.087169,-3.610648,3.525782,5.960631,8.052436,-5.084948,8.468734,-1.318219,-8.590171,-0.220858,-8.525694,4.410276,-3.772891,6.943883,4.907430,0.680722,-6.255531,0.299586,5.174180,5.215877,3.937396,0.226918,5.838152,1.451957,1.922903,7.106017,-4.104156,2.428216,-5.163935,-1.241069,-5.320178,-7.900472,-5.196450,9.762719,-4.023181,-4.110631,-5.985891,-4.122373,3.199013,4.087809,2.719377,5.385895,-6.297460,0.179328,6.667668,-1.098946,4.023714,1.996866,1.749871,-5.854599,6.726227,9.450266,-9.007676,-1.083729,3.315485,-9.659052,-9.805138,4.413378,-6.764014,6.732607,-7.921376,-5.595854,-6.235936,7.806641,-1.739104,9.102956,5.250638,5.597936,9.756093,4.701021,-4.398968,2.175178,6.442523,0.487913,1.966608,-5.733003,-0.745997,-6.316911,-4.669109,-3.143409,0.664877,-6.462536,3.004211,-8.876499,9.020990,5.699759,3.271314,-5.074344,9.342239,-6.780090,2.131997,8.577476,3.338714,-3.706223,7.098231,-8.106272,1.086812,0.974652,7.164624,7.946632,-2.653177,-6.282922,2.850508,-1.779244,-9.960050,-2.505421,5.238703,2.695618,9.168689,-5.932239,4.984937,7.842769,-9.240364,5.024906,-9.296361,-1.302264,6.149074,-5.645805,-9.399956,-9.466388,9.774424,-2.956497,-7.428287,-3.421288,9.956568,-2.528017,8.391519,4.491861,9.320423,9.702085,3.588852,-4.532711,-6.712279,-4.077421,9.457125,0.922610,1.444508,-5.604466,-8.055655,8.007903,3.852033], dtype = "float32")#candidate|1497|(225,)|const|float32
var_1498 = relay.var("var_1498", dtype = "float32", shape = ())#candidate|1498|()|var|float32
const_1499 = relay.const([[-0.981424],[-0.822516],[-0.093122],[4.488940],[9.623971],[5.614262],[3.028268],[-2.608309],[-5.819823],[-9.723263],[-7.863879],[-8.918070],[8.014471],[4.474015],[-2.663351],[6.738272],[5.914970],[7.365791],[2.600007],[-9.980466],[6.299490],[5.065697],[-1.959309],[-0.778345],[-9.874199],[-6.507242],[4.527403],[7.415676],[-4.206845],[-8.321890],[-3.500943],[4.575317],[8.902754],[-2.467081],[0.769558],[4.048180],[-9.865525],[-1.560592],[-2.578790],[-9.751543],[6.433084],[-5.442659],[6.982780],[5.642541],[-4.502870],[-3.903683],[-0.593644],[-1.735889],[-6.710283],[-6.416954],[-0.321015],[-0.138928],[-3.924851],[2.938599],[-4.031145],[-2.544758],[6.336967],[-4.501256],[-4.060486],[-9.063009],[7.642821],[-4.820368],[4.084698],[-9.367640],[7.352281],[4.798884],[-9.508149],[-7.437319],[-9.349991],[-6.768570],[5.470143],[6.829091],[7.202361],[2.589709],[8.296187],[-6.641478],[-0.895639],[-4.328117],[2.121158],[8.383322],[8.022119],[-0.009780],[7.045238],[9.572832],[5.314598],[-0.428934],[-9.346154],[2.792300],[1.880261],[5.245537],[-6.367416],[7.684914],[-3.345112],[-7.893335],[-2.272506],[-3.040230],[1.422257],[-0.928485],[-8.186258],[-8.934655],[7.081010],[-1.655295],[-3.669600],[1.653484],[4.630129],[6.799442],[-3.028787],[-4.363280],[-3.431504],[-7.655658],[9.822927],[5.742295],[-0.349631],[1.209572],[1.051396],[-2.843743],[-8.860398],[-5.093175],[-6.320939],[-2.909071]], dtype = "float32")#candidate|1499|(120, 1)|const|float32
var_1500 = relay.var("var_1500", dtype = "float64", shape = (40,))#candidate|1500|(40,)|var|float64
call_1496 = relay.TupleGetItem(func_1400_call(relay.reshape(const_1497.astype('float32'), [15, 15]), relay.reshape(var_1498.astype('float32'), []), relay.reshape(const_1499.astype('float32'), [30, 4]), relay.reshape(var_1500.astype('float64'), [40,]), relay.reshape(const_1497.astype('float32'), [15, 15]), ), 3)
call_1501 = relay.TupleGetItem(func_1407_call(relay.reshape(const_1497.astype('float32'), [15, 15]), relay.reshape(var_1498.astype('float32'), []), relay.reshape(const_1499.astype('float32'), [30, 4]), relay.reshape(var_1500.astype('float64'), [40,]), relay.reshape(const_1497.astype('float32'), [15, 15]), ), 3)
output = relay.Tuple([call_1457,call_1472,bop_1492,call_1496,const_1497,var_1498,const_1499,var_1500,])
output2 = relay.Tuple([call_1458,call_1473,bop_1495,call_1501,const_1497,var_1498,const_1499,var_1500,])
func_1507 = relay.Function([var_1467,var_1498,var_1500,], output)
mod['func_1507'] = func_1507
mod = relay.transform.InferType()(mod)
var_1508 = relay.var("var_1508", dtype = "bool", shape = (2016,))#candidate|1508|(2016,)|var|bool
var_1509 = relay.var("var_1509", dtype = "float32", shape = ())#candidate|1509|()|var|float32
var_1510 = relay.var("var_1510", dtype = "float64", shape = (40,))#candidate|1510|(40,)|var|float64
output = func_1507(var_1508,var_1509,var_1510,)
func_1511 = relay.Function([var_1508,var_1509,var_1510,], output)
mutated_mod['func_1511'] = func_1511
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1054_call = mod.get_global_var('func_1054')
func_1055_call = mutated_mod.get_global_var('func_1055')
call_1518 = relay.TupleGetItem(func_1054_call(), 2)
call_1519 = relay.TupleGetItem(func_1055_call(), 2)
output = relay.Tuple([call_1518,])
output2 = relay.Tuple([call_1519,])
func_1524 = relay.Function([], output)
mod['func_1524'] = func_1524
mod = relay.transform.InferType()(mod)
output = func_1524()
func_1525 = relay.Function([], output)
mutated_mod['func_1525'] = func_1525
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1542 = relay.const([[[-5,2,-2],[-3,10,-3],[6,7,1],[-2,7,-10],[4,3,-7],[8,3,-6],[-8,-4,-10],[1,10,-2],[7,2,4],[6,5,-3],[-8,-4,4],[-3,-7,-1],[5,7,-10],[-10,2,3],[-6,-5,-8]],[[3,3,6],[7,10,-8],[5,2,3],[-10,3,7],[-2,-4,-5],[-7,-8,-6],[-8,6,9],[6,9,5],[-4,-7,10],[2,8,-9],[10,-1,-6],[3,-2,-9],[-2,6,-7],[-3,7,-7],[-3,-9,6]],[[-6,-3,5],[4,3,-6],[8,1,9],[-6,-8,-2],[1,5,10],[7,-8,-3],[-10,-2,4],[5,-7,-5],[-4,-9,-7],[1,-10,-8],[-10,6,-3],[-2,5,-6],[-3,10,-1],[6,-7,10],[-8,2,7]],[[-10,-4,4],[5,-2,1],[9,1,2],[4,7,-5],[-7,-10,-2],[-8,-4,1],[10,8,-6],[10,-1,-3],[-4,2,-10],[10,1,6],[-9,-8,-5],[-4,4,-6],[9,6,-8],[-6,7,6],[7,7,-8]],[[-1,-5,1],[-7,6,-7],[2,8,-7],[-6,-1,-10],[-2,-4,3],[10,-10,-3],[7,-6,-3],[-5,8,-2],[2,-7,-3],[-9,-2,-2],[5,-10,-8],[-6,5,8],[-6,5,-8],[10,-5,4],[-3,-2,6]],[[-1,7,3],[-6,-10,2],[1,5,-3],[-2,-5,-7],[-6,-2,7],[-4,6,-2],[5,-10,1],[7,10,-9],[-4,-6,-10],[6,5,-4],[-5,-1,-10],[-1,-2,-3],[5,-6,-9],[9,-6,2],[8,-1,7]],[[3,6,2],[8,-2,-6],[-3,5,7],[-4,9,9],[-5,-1,-10],[-2,2,-9],[-5,-10,5],[-1,9,8],[2,-10,2],[-2,8,1],[-2,3,2],[-8,8,5],[-6,-4,2],[10,-3,-5],[-2,10,-10]],[[10,-8,8],[2,9,6],[5,-10,-6],[8,8,-6],[6,2,10],[9,-6,10],[10,-1,-3],[-3,-6,5],[7,8,5],[-2,4,7],[8,3,-6],[9,8,-5],[5,-7,-2],[5,-8,-9],[-6,3,5]],[[-5,9,-8],[-4,3,6],[-1,7,-4],[-2,10,-1],[-4,2,-9],[1,-1,-6],[7,7,-10],[-7,6,10],[-3,-8,-4],[-4,7,8],[-4,-2,9],[-3,7,-6],[4,9,7],[7,-6,5],[-5,-10,6]],[[8,5,4],[-9,-10,-8],[5,-8,-2],[-8,7,6],[-5,2,1],[1,9,-1],[-10,1,-9],[4,10,10],[9,1,-3],[-7,-5,-2],[-8,-8,-9],[-7,3,8],[-6,8,4],[-10,-8,-8],[-7,7,-3]],[[-10,2,-5],[1,9,-1],[-6,9,5],[-1,-8,-5],[10,-1,-3],[6,3,-1],[8,-2,6],[3,-2,-3],[10,1,5],[-4,5,-1],[-10,-2,-2],[3,5,-6],[1,-2,-4],[10,-10,-1],[7,1,-1]],[[-2,7,5],[-7,9,4],[-4,7,-7],[-2,4,-8],[-9,8,-8],[-10,1,-6],[-9,5,-6],[7,5,7],[-4,9,-1],[1,-3,-9],[10,-3,1],[7,2,-9],[-2,-10,10],[-9,8,-10],[-6,-8,10]]], dtype = "int64")#candidate|1542|(12, 15, 3)|const|int64
const_1543 = relay.const([[[1,-2,-1],[1,3,-9],[-7,-4,-6],[4,-1,8],[-9,-8,-9],[5,-8,-2],[-7,10,7],[3,-6,10],[-1,1,7],[-1,10,-3],[10,-10,6],[-9,5,5],[9,-7,-6],[-7,1,-6],[9,-7,9]],[[-9,-10,7],[-3,9,6],[-1,1,-9],[4,3,-6],[-6,-8,-6],[-2,6,-4],[-8,-1,-4],[-2,-9,-4],[1,6,5],[-2,-7,-5],[7,-7,-9],[-6,-7,-5],[-4,-7,-1],[1,1,5],[7,7,6]],[[-6,9,-6],[4,9,-10],[4,3,-7],[2,-6,3],[-9,9,-6],[-9,-9,-8],[1,8,-5],[4,4,-9],[-2,7,6],[-1,2,-10],[-8,5,-9],[2,-6,8],[3,-9,9],[-6,-2,3],[4,-10,6]],[[7,1,1],[6,8,6],[-3,7,-5],[-3,9,3],[1,6,4],[-3,5,-4],[4,10,3],[-7,6,-10],[2,2,-7],[3,-4,1],[-2,10,8],[6,5,-1],[3,-5,-4],[1,-6,-7],[-2,3,1]],[[4,6,-3],[-10,-6,6],[-3,7,-9],[3,-5,7],[-6,-4,4],[10,-1,2],[3,-4,6],[8,8,-9],[6,-10,-3],[-5,-7,7],[-6,-9,-5],[10,4,-9],[-3,-2,-2],[-1,5,9],[8,8,-5]],[[-7,3,-6],[7,-8,-2],[-3,-4,8],[-10,-2,10],[-4,7,6],[-8,-1,-8],[-9,6,8],[-4,3,-8],[2,1,2],[7,-10,9],[-6,-8,5],[-7,1,-2],[9,1,-4],[6,6,3],[-7,-5,7]],[[-4,8,2],[-7,10,2],[8,9,-5],[8,-10,7],[4,6,-4],[3,7,-9],[7,-8,6],[10,-1,-5],[4,10,-10],[9,-2,3],[-7,-3,8],[-3,-9,-1],[-1,1,9],[-2,9,-7],[8,5,1]],[[5,4,10],[9,-6,-4],[-5,7,-1],[-4,-7,3],[-4,-2,-1],[6,-8,-4],[-2,-2,3],[-5,-3,6],[9,6,-5],[-2,10,-9],[-6,6,5],[2,3,8],[-2,-4,-7],[8,10,-4],[3,-3,9]],[[-3,-7,10],[-9,1,-8],[7,4,-6],[10,7,-8],[-5,6,-5],[2,-1,-8],[-10,3,8],[2,3,7],[-7,3,-8],[-2,3,3],[-2,-5,3],[-3,-10,-8],[-2,-4,-4],[-6,9,-4],[2,2,-1]],[[-5,-1,7],[-6,10,-9],[-4,7,-5],[2,-4,3],[-10,4,-2],[8,-8,-8],[4,5,9],[2,-6,5],[4,5,8],[5,-2,-7],[-4,2,5],[-8,3,5],[-9,7,-8],[7,10,1],[9,-6,-7]],[[-5,-10,-4],[6,-4,-7],[-3,5,-4],[-4,6,-2],[-4,5,1],[-4,2,3],[9,-1,-10],[-1,7,-5],[-9,-1,-7],[-2,-3,2],[8,2,10],[-9,9,-1],[10,-4,-7],[7,9,3],[-10,-8,3]],[[-9,-4,2],[-10,-2,-2],[-8,4,-7],[-10,-9,2],[-7,-6,-7],[5,-1,-1],[-2,5,-10],[8,7,-9],[-2,-1,-9],[2,7,-1],[6,-4,-4],[9,6,1],[-1,3,10],[2,-10,-3],[-8,-5,3]]], dtype = "int64")#candidate|1543|(12, 15, 3)|const|int64
bop_1544 = relay.greater_equal(const_1542.astype('bool'), relay.reshape(const_1543.astype('bool'), relay.shape_of(const_1542))) # shape=(12, 15, 3)
bop_1548 = relay.greater(const_1543.astype('bool'), relay.reshape(const_1542.astype('bool'), relay.shape_of(const_1543))) # shape=(12, 15, 3)
bop_1553 = relay.logical_and(const_1543.astype('bool'), relay.reshape(bop_1548.astype('bool'), relay.shape_of(const_1543))) # shape=(12, 15, 3)
bop_1563 = relay.right_shift(const_1543.astype('int16'), relay.reshape(bop_1548.astype('int16'), relay.shape_of(const_1543))) # shape=(12, 15, 3)
func_607_call = mod.get_global_var('func_607')
func_611_call = mutated_mod.get_global_var('func_611')
const_1568 = relay.const([[0.776501,-7.876930,5.033091,-9.756237,-0.830782,2.328021,-4.659119,-4.252945,6.864160]], dtype = "float64")#candidate|1568|(1, 9)|const|float64
var_1569 = relay.var("var_1569", dtype = "float64", shape = (18,))#candidate|1569|(18,)|var|float64
call_1567 = relay.TupleGetItem(func_607_call(relay.reshape(const_1568.astype('float64'), [1, 9]), relay.reshape(var_1569.astype('float64'), [2, 9]), ), 0)
call_1570 = relay.TupleGetItem(func_611_call(relay.reshape(const_1568.astype('float64'), [1, 9]), relay.reshape(var_1569.astype('float64'), [2, 9]), ), 0)
func_824_call = mod.get_global_var('func_824')
func_827_call = mutated_mod.get_global_var('func_827')
var_1576 = relay.var("var_1576", dtype = "float64", shape = (30, 9))#candidate|1576|(30, 9)|var|float64
call_1575 = func_824_call(relay.reshape(var_1576.astype('float64'), [15, 2, 9]))
call_1577 = func_824_call(relay.reshape(var_1576.astype('float64'), [15, 2, 9]))
func_824_call = mod.get_global_var('func_824')
func_827_call = mutated_mod.get_global_var('func_827')
call_1583 = func_824_call(relay.reshape(call_1575.astype('float64'), [15, 2, 9]))
call_1584 = func_824_call(relay.reshape(call_1575.astype('float64'), [15, 2, 9]))
uop_1587 = relay.asinh(call_1575.astype('float64')) # shape=(15, 2, 9)
uop_1589 = relay.asinh(call_1577.astype('float64')) # shape=(15, 2, 9)
const_1592 = relay.const([[[True,True,True],[True,False,False],[True,False,True],[True,False,False],[True,True,True],[True,False,True],[False,True,False],[True,False,True],[False,False,False],[False,True,False],[True,False,False],[False,False,False],[False,True,False],[True,False,True],[True,True,False]],[[True,False,False],[False,True,False],[False,False,False],[False,False,False],[False,False,True],[False,True,False],[False,True,False],[False,True,True],[True,True,True],[False,False,True],[False,True,True],[True,False,False],[True,False,False],[True,False,False],[False,False,True]],[[True,True,False],[True,True,False],[True,False,True],[False,True,False],[False,True,True],[True,True,False],[False,False,False],[True,True,True],[True,True,True],[False,True,False],[True,False,True],[True,False,False],[True,True,False],[True,False,True],[True,False,False]],[[False,True,False],[True,True,False],[True,False,False],[False,False,True],[False,False,False],[False,True,False],[True,False,False],[True,False,False],[False,True,False],[True,False,False],[False,True,True],[False,False,False],[False,True,True],[False,False,True],[True,False,True]],[[True,False,False],[False,True,False],[False,False,False],[False,False,True],[False,False,False],[False,False,True],[True,False,True],[False,False,False],[True,True,False],[False,False,False],[True,False,False],[True,False,True],[True,False,False],[False,True,False],[True,True,False]],[[True,False,False],[False,True,False],[True,True,True],[False,False,False],[False,False,False],[True,True,False],[True,False,False],[True,False,True],[True,False,False],[True,True,True],[False,True,True],[False,True,True],[True,True,False],[False,True,True],[False,True,True]],[[False,False,True],[True,False,True],[True,True,True],[False,False,True],[False,True,True],[False,True,False],[True,False,True],[False,True,False],[True,True,True],[False,True,True],[True,True,True],[False,False,True],[False,False,False],[True,False,False],[True,True,False]],[[False,True,True],[False,False,True],[True,True,False],[False,False,True],[True,False,False],[False,True,False],[False,False,False],[True,False,False],[False,True,True],[False,False,False],[False,False,True],[False,False,True],[True,False,False],[False,False,False],[True,True,False]],[[False,False,False],[False,False,False],[False,True,False],[True,True,True],[False,False,False],[False,False,False],[True,False,True],[False,False,True],[True,False,True],[True,True,False],[False,True,False],[False,True,False],[False,False,False],[False,True,True],[True,True,True]],[[True,True,False],[False,False,False],[True,False,False],[False,True,False],[True,True,False],[True,True,True],[False,True,True],[False,True,True],[False,True,False],[False,False,True],[True,False,False],[True,False,False],[False,True,False],[False,True,False],[True,False,True]],[[True,True,False],[False,True,True],[True,False,False],[False,False,True],[False,False,True],[True,False,True],[True,True,True],[True,True,False],[False,False,True],[False,True,True],[True,False,True],[True,False,False],[False,False,False],[False,False,True],[True,True,True]],[[False,False,True],[False,False,True],[True,True,False],[False,True,False],[True,True,True],[True,False,True],[False,True,False],[False,True,False],[True,True,False],[False,False,True],[True,True,False],[True,True,False],[False,False,True],[False,False,True],[True,False,False]]], dtype = "bool")#candidate|1592|(12, 15, 3)|const|bool
bop_1593 = relay.add(bop_1548.astype('int8'), relay.reshape(const_1592.astype('int8'), relay.shape_of(bop_1548))) # shape=(12, 15, 3)
func_1422_call = mod.get_global_var('func_1422')
func_1425_call = mutated_mod.get_global_var('func_1425')
const_1606 = relay.const([-7,2,5,-5,3,2,-1,-5,-2,-10,3,7,2,4,-8,-4,9,-3,8,-3,1,8,-8,1,-3,-6], dtype = "int8")#candidate|1606|(26,)|const|int8
call_1605 = relay.TupleGetItem(func_1422_call(relay.reshape(const_1606.astype('int8'), [2, 13]), relay.reshape(const_1606.astype('int8'), [2, 13]), ), 0)
call_1607 = relay.TupleGetItem(func_1425_call(relay.reshape(const_1606.astype('int8'), [2, 13]), relay.reshape(const_1606.astype('int8'), [2, 13]), ), 0)
uop_1611 = relay.log10(const_1606.astype('float32')) # shape=(26,)
var_1614 = relay.var("var_1614", dtype = "float64", shape = (15, 2, 9))#candidate|1614|(15, 2, 9)|var|float64
bop_1615 = relay.multiply(uop_1587.astype('int64'), relay.reshape(var_1614.astype('int64'), relay.shape_of(uop_1587))) # shape=(15, 2, 9)
bop_1618 = relay.multiply(uop_1589.astype('int64'), relay.reshape(var_1614.astype('int64'), relay.shape_of(uop_1589))) # shape=(15, 2, 9)
bop_1621 = relay.logical_or(uop_1611.astype('bool'), relay.reshape(const_1606.astype('bool'), relay.shape_of(uop_1611))) # shape=(26,)
uop_1630 = relay.tan(uop_1611.astype('float64')) # shape=(26,)
var_1632 = relay.var("var_1632", dtype = "int8", shape = (12, 15, 3))#candidate|1632|(12, 15, 3)|var|int8
bop_1633 = relay.less(bop_1593.astype('bool'), relay.reshape(var_1632.astype('bool'), relay.shape_of(bop_1593))) # shape=(12, 15, 3)
output = relay.Tuple([bop_1544,bop_1553,bop_1563,call_1567,const_1568,var_1569,var_1576,call_1583,call_1605,bop_1615,bop_1621,uop_1630,bop_1633,])
output2 = relay.Tuple([bop_1544,bop_1553,bop_1563,call_1570,const_1568,var_1569,var_1576,call_1584,call_1607,bop_1618,bop_1621,uop_1630,bop_1633,])
func_1641 = relay.Function([var_1569,var_1576,var_1614,var_1632,], output)
mod['func_1641'] = func_1641
mod = relay.transform.InferType()(mod)
mutated_mod['func_1641'] = func_1641
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1641_call = mutated_mod.get_global_var('func_1641')
var_1643 = relay.var("var_1643", dtype = "float64", shape = (18,))#candidate|1643|(18,)|var|float64
var_1644 = relay.var("var_1644", dtype = "float64", shape = (30, 9))#candidate|1644|(30, 9)|var|float64
var_1645 = relay.var("var_1645", dtype = "float64", shape = (15, 2, 9))#candidate|1645|(15, 2, 9)|var|float64
var_1646 = relay.var("var_1646", dtype = "int8", shape = (12, 15, 3))#candidate|1646|(12, 15, 3)|var|int8
call_1642 = func_1641_call(var_1643,var_1644,var_1645,var_1646,)
output = call_1642
func_1647 = relay.Function([var_1643,var_1644,var_1645,var_1646,], output)
mutated_mod['func_1647'] = func_1647
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1687 = relay.var("var_1687", dtype = "int16", shape = (4, 15, 15))#candidate|1687|(4, 15, 15)|var|int16
var_1688 = relay.var("var_1688", dtype = "int16", shape = (4, 15, 15))#candidate|1688|(4, 15, 15)|var|int16
bop_1689 = relay.less(var_1687.astype('bool'), relay.reshape(var_1688.astype('bool'), relay.shape_of(var_1687))) # shape=(4, 15, 15)
func_1319_call = mod.get_global_var('func_1319')
func_1321_call = mutated_mod.get_global_var('func_1321')
call_1701 = func_1319_call()
call_1702 = func_1319_call()
uop_1703 = relay.log(var_1688.astype('float64')) # shape=(4, 15, 15)
func_1233_call = mod.get_global_var('func_1233')
func_1237_call = mutated_mod.get_global_var('func_1237')
var_1706 = relay.var("var_1706", dtype = "uint64", shape = (84,))#candidate|1706|(84,)|var|uint64
const_1707 = relay.const([[-9.503903,2.545042,-7.793763,8.751656,5.875890,7.265177,1.409474,2.593886,-6.629056,-9.721101,-6.209129,-7.238121,1.158627,5.399413,9.489413,-7.885276,-7.210550,-8.966676,-9.883227,-0.656310,2.678854,2.114754,-1.727681,-3.588630,-2.244422,-6.144215,-1.325621,-1.381397,5.088711,3.474439,-9.491068,-7.937718,7.649601,2.090832,-1.297855,-4.264718,8.210256,-3.951499,-9.090313,-2.927945,4.728810,1.323882,8.891416,-8.201911,2.577019,-8.311041,4.980582,-3.417827,0.093986,-5.890634,-7.118277,5.259524,-8.827050,8.717767,4.252127,1.962230,0.333372,-8.603712,-7.420761,8.331446]], dtype = "float32")#candidate|1707|(1, 60)|const|float32
call_1705 = relay.TupleGetItem(func_1233_call(relay.reshape(var_1706.astype('uint64'), [84,]), relay.reshape(const_1707.astype('float32'), [2, 15, 2]), ), 0)
call_1708 = relay.TupleGetItem(func_1237_call(relay.reshape(var_1706.astype('uint64'), [84,]), relay.reshape(const_1707.astype('float32'), [2, 15, 2]), ), 0)
bop_1711 = relay.equal(uop_1703.astype('bool'), relay.reshape(var_1687.astype('bool'), relay.shape_of(uop_1703))) # shape=(4, 15, 15)
uop_1714 = relay.acosh(var_1687.astype('float32')) # shape=(4, 15, 15)
bop_1716 = relay.floor_divide(uop_1714.astype('float64'), relay.reshape(bop_1711.astype('float64'), relay.shape_of(uop_1714))) # shape=(4, 15, 15)
output = relay.Tuple([bop_1689,call_1701,call_1705,var_1706,const_1707,bop_1716,])
output2 = relay.Tuple([bop_1689,call_1702,call_1708,var_1706,const_1707,bop_1716,])
func_1719 = relay.Function([var_1687,var_1688,var_1706,], output)
mod['func_1719'] = func_1719
mod = relay.transform.InferType()(mod)
var_1720 = relay.var("var_1720", dtype = "int16", shape = (4, 15, 15))#candidate|1720|(4, 15, 15)|var|int16
var_1721 = relay.var("var_1721", dtype = "int16", shape = (4, 15, 15))#candidate|1721|(4, 15, 15)|var|int16
var_1722 = relay.var("var_1722", dtype = "uint64", shape = (84,))#candidate|1722|(84,)|var|uint64
output = func_1719(var_1720,var_1721,var_1722,)
func_1723 = relay.Function([var_1720,var_1721,var_1722,], output)
mutated_mod['func_1723'] = func_1723
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1727 = relay.var("var_1727", dtype = "float32", shape = (2, 10))#candidate|1727|(2, 10)|var|float32
uop_1728 = relay.cosh(var_1727.astype('float32')) # shape=(2, 10)
func_635_call = mod.get_global_var('func_635')
func_636_call = mutated_mod.get_global_var('func_636')
call_1731 = relay.TupleGetItem(func_635_call(), 0)
call_1732 = relay.TupleGetItem(func_636_call(), 0)
uop_1742 = relay.asinh(call_1731.astype('float32')) # shape=(1, 2, 9)
uop_1744 = relay.asinh(call_1732.astype('float32')) # shape=(1, 2, 9)
bop_1745 = relay.right_shift(uop_1742.astype('uint32'), relay.reshape(call_1731.astype('uint32'), relay.shape_of(uop_1742))) # shape=(1, 2, 9)
bop_1748 = relay.right_shift(uop_1744.astype('uint32'), relay.reshape(call_1732.astype('uint32'), relay.shape_of(uop_1744))) # shape=(1, 2, 9)
output = relay.Tuple([uop_1728,bop_1745,])
output2 = relay.Tuple([uop_1728,bop_1748,])
func_1751 = relay.Function([var_1727,], output)
mod['func_1751'] = func_1751
mod = relay.transform.InferType()(mod)
var_1752 = relay.var("var_1752", dtype = "float32", shape = (2, 10))#candidate|1752|(2, 10)|var|float32
output = func_1751(var_1752)
func_1753 = relay.Function([var_1752], output)
mutated_mod['func_1753'] = func_1753
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1755 = relay.var("var_1755", dtype = "float64", shape = (11, 13))#candidate|1755|(11, 13)|var|float64
uop_1756 = relay.sinh(var_1755.astype('float64')) # shape=(11, 13)
output = relay.Tuple([uop_1756,])
output2 = relay.Tuple([uop_1756,])
func_1760 = relay.Function([var_1755,], output)
mod['func_1760'] = func_1760
mod = relay.transform.InferType()(mod)
var_1761 = relay.var("var_1761", dtype = "float64", shape = (11, 13))#candidate|1761|(11, 13)|var|float64
output = func_1760(var_1761)
func_1762 = relay.Function([var_1761], output)
mutated_mod['func_1762'] = func_1762
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1524_call = mod.get_global_var('func_1524')
func_1525_call = mutated_mod.get_global_var('func_1525')
call_1787 = relay.TupleGetItem(func_1524_call(), 0)
call_1788 = relay.TupleGetItem(func_1525_call(), 0)
output = relay.Tuple([call_1787,])
output2 = relay.Tuple([call_1788,])
func_1789 = relay.Function([], output)
mod['func_1789'] = func_1789
mod = relay.transform.InferType()(mod)
mutated_mod['func_1789'] = func_1789
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1789_call = mutated_mod.get_global_var('func_1789')
call_1790 = func_1789_call()
output = call_1790
func_1791 = relay.Function([], output)
mutated_mod['func_1791'] = func_1791
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1054_call = mod.get_global_var('func_1054')
func_1055_call = mutated_mod.get_global_var('func_1055')
call_1849 = relay.TupleGetItem(func_1054_call(), 2)
call_1850 = relay.TupleGetItem(func_1055_call(), 2)
func_1054_call = mod.get_global_var('func_1054')
func_1055_call = mutated_mod.get_global_var('func_1055')
call_1851 = relay.TupleGetItem(func_1054_call(), 3)
call_1852 = relay.TupleGetItem(func_1055_call(), 3)
var_1861 = relay.var("var_1861", dtype = "int32", shape = (15, 2, 9))#candidate|1861|(15, 2, 9)|var|int32
bop_1862 = relay.maximum(call_1851.astype('int16'), relay.reshape(var_1861.astype('int16'), relay.shape_of(call_1851))) # shape=(15, 2, 9)
bop_1865 = relay.maximum(call_1852.astype('int16'), relay.reshape(var_1861.astype('int16'), relay.shape_of(call_1852))) # shape=(15, 2, 9)
var_1878 = relay.var("var_1878", dtype = "int32", shape = (15, 2, 9))#candidate|1878|(15, 2, 9)|var|int32
bop_1879 = relay.bitwise_and(call_1851.astype('int64'), relay.reshape(var_1878.astype('int64'), relay.shape_of(call_1851))) # shape=(15, 2, 9)
bop_1882 = relay.bitwise_and(call_1852.astype('int64'), relay.reshape(var_1878.astype('int64'), relay.shape_of(call_1852))) # shape=(15, 2, 9)
func_112_call = mod.get_global_var('func_112')
func_116_call = mutated_mod.get_global_var('func_116')
var_1885 = relay.var("var_1885", dtype = "uint32", shape = ())#candidate|1885|()|var|uint32
var_1886 = relay.var("var_1886", dtype = "uint32", shape = (168,))#candidate|1886|(168,)|var|uint32
call_1884 = relay.TupleGetItem(func_112_call(relay.reshape(var_1885.astype('uint32'), []), relay.reshape(var_1886.astype('uint32'), [2, 14, 6]), ), 0)
call_1887 = relay.TupleGetItem(func_116_call(relay.reshape(var_1885.astype('uint32'), []), relay.reshape(var_1886.astype('uint32'), [2, 14, 6]), ), 0)
uop_1893 = relay.atan(call_1851.astype('float64')) # shape=(15, 2, 9)
uop_1895 = relay.atan(call_1852.astype('float64')) # shape=(15, 2, 9)
output = relay.Tuple([call_1849,bop_1862,bop_1879,call_1884,var_1885,var_1886,uop_1893,])
output2 = relay.Tuple([call_1850,bop_1865,bop_1882,call_1887,var_1885,var_1886,uop_1895,])
func_1896 = relay.Function([var_1861,var_1878,var_1885,var_1886,], output)
mod['func_1896'] = func_1896
mod = relay.transform.InferType()(mod)
var_1897 = relay.var("var_1897", dtype = "int32", shape = (15, 2, 9))#candidate|1897|(15, 2, 9)|var|int32
var_1898 = relay.var("var_1898", dtype = "int32", shape = (15, 2, 9))#candidate|1898|(15, 2, 9)|var|int32
var_1899 = relay.var("var_1899", dtype = "uint32", shape = ())#candidate|1899|()|var|uint32
var_1900 = relay.var("var_1900", dtype = "uint32", shape = (168,))#candidate|1900|(168,)|var|uint32
output = func_1896(var_1897,var_1898,var_1899,var_1900,)
func_1901 = relay.Function([var_1897,var_1898,var_1899,var_1900,], output)
mutated_mod['func_1901'] = func_1901
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1319_call = mod.get_global_var('func_1319')
func_1321_call = mutated_mod.get_global_var('func_1321')
call_1970 = func_1319_call()
call_1971 = func_1319_call()
var_1974 = relay.var("var_1974", dtype = "bool", shape = (13, 2, 9))#candidate|1974|(13, 2, 9)|var|bool
bop_1975 = relay.floor_divide(call_1970.astype('float64'), var_1974.astype('float64')) # shape=(13, 2, 9)
bop_1978 = relay.floor_divide(call_1971.astype('float64'), var_1974.astype('float64')) # shape=(13, 2, 9)
bop_1979 = relay.floor_mod(bop_1975.astype('float32'), relay.reshape(var_1974.astype('float32'), relay.shape_of(bop_1975))) # shape=(13, 2, 9)
bop_1982 = relay.floor_mod(bop_1978.astype('float32'), relay.reshape(var_1974.astype('float32'), relay.shape_of(bop_1978))) # shape=(13, 2, 9)
func_1100_call = mod.get_global_var('func_1100')
func_1105_call = mutated_mod.get_global_var('func_1105')
const_1990 = relay.const([[False,False,True,True,False,True,True,True,False,False,True,False,True,True,True,True,False,False,True,True],[True,True,False,True,False,True,False,False,True,True,True,True,False,False,False,False,False,True,True,False],[False,False,True,True,True,False,False,True,True,True,False,False,True,True,True,True,False,True,True,True],[True,False,True,True,False,True,True,False,True,False,True,True,True,False,False,False,False,False,False,True],[False,False,True,True,True,False,False,True,False,True,True,True,True,False,False,True,True,True,False,False],[True,True,True,False,True,True,True,True,False,True,True,False,True,False,True,True,True,True,False,True],[False,True,False,True,True,False,False,True,True,False,True,False,True,True,True,False,False,False,False,False],[True,True,False,False,True,True,True,False,False,False,True,False,True,True,True,True,True,False,False,False]], dtype = "bool")#candidate|1990|(8, 20)|const|bool
var_1991 = relay.var("var_1991", dtype = "float32", shape = (840,))#candidate|1991|(840,)|var|float32
call_1989 = relay.TupleGetItem(func_1100_call(relay.reshape(const_1990.astype('bool'), [16, 10]), relay.reshape(const_1990.astype('bool'), [16, 10]), relay.reshape(var_1991.astype('float32'), [840,]), ), 2)
call_1992 = relay.TupleGetItem(func_1105_call(relay.reshape(const_1990.astype('bool'), [16, 10]), relay.reshape(const_1990.astype('bool'), [16, 10]), relay.reshape(var_1991.astype('float32'), [840,]), ), 2)
bop_1999 = relay.logical_xor(bop_1975.astype('uint16'), call_1970.astype('uint16')) # shape=(13, 2, 9)
bop_2002 = relay.logical_xor(bop_1978.astype('uint16'), call_1971.astype('uint16')) # shape=(13, 2, 9)
bop_2004 = relay.bitwise_xor(bop_1999.astype('int8'), relay.reshape(bop_1975.astype('int8'), relay.shape_of(bop_1999))) # shape=(13, 2, 9)
bop_2007 = relay.bitwise_xor(bop_2002.astype('int8'), relay.reshape(bop_1978.astype('int8'), relay.shape_of(bop_2002))) # shape=(13, 2, 9)
func_1319_call = mod.get_global_var('func_1319')
func_1321_call = mutated_mod.get_global_var('func_1321')
call_2009 = func_1319_call()
call_2010 = func_1319_call()
output = relay.Tuple([bop_1979,call_1989,const_1990,var_1991,bop_2004,call_2009,])
output2 = relay.Tuple([bop_1982,call_1992,const_1990,var_1991,bop_2007,call_2010,])
func_2012 = relay.Function([var_1974,var_1991,], output)
mod['func_2012'] = func_2012
mod = relay.transform.InferType()(mod)
mutated_mod['func_2012'] = func_2012
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2012_call = mutated_mod.get_global_var('func_2012')
var_2014 = relay.var("var_2014", dtype = "bool", shape = (13, 2, 9))#candidate|2014|(13, 2, 9)|var|bool
var_2015 = relay.var("var_2015", dtype = "float32", shape = (840,))#candidate|2015|(840,)|var|float32
call_2013 = func_2012_call(var_2014,var_2015,)
output = call_2013
func_2016 = relay.Function([var_2014,var_2015,], output)
mutated_mod['func_2016'] = func_2016
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2030 = relay.var("var_2030", dtype = "int16", shape = (6, 15))#candidate|2030|(6, 15)|var|int16
var_2031 = relay.var("var_2031", dtype = "int16", shape = (6, 15))#candidate|2031|(6, 15)|var|int16
bop_2032 = relay.bitwise_or(var_2030.astype('int16'), relay.reshape(var_2031.astype('int16'), relay.shape_of(var_2030))) # shape=(6, 15)
output = relay.Tuple([bop_2032,])
output2 = relay.Tuple([bop_2032,])
func_2035 = relay.Function([var_2030,var_2031,], output)
mod['func_2035'] = func_2035
mod = relay.transform.InferType()(mod)
var_2036 = relay.var("var_2036", dtype = "int16", shape = (6, 15))#candidate|2036|(6, 15)|var|int16
var_2037 = relay.var("var_2037", dtype = "int16", shape = (6, 15))#candidate|2037|(6, 15)|var|int16
output = func_2035(var_2036,var_2037,)
func_2038 = relay.Function([var_2036,var_2037,], output)
mutated_mod['func_2038'] = func_2038
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1789_call = mod.get_global_var('func_1789')
func_1791_call = mutated_mod.get_global_var('func_1791')
call_2084 = relay.TupleGetItem(func_1789_call(), 0)
call_2085 = relay.TupleGetItem(func_1791_call(), 0)
output = call_2084
output2 = call_2085
func_2086 = relay.Function([], output)
mod['func_2086'] = func_2086
mod = relay.transform.InferType()(mod)
mutated_mod['func_2086'] = func_2086
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2086_call = mutated_mod.get_global_var('func_2086')
call_2087 = func_2086_call()
output = call_2087
func_2088 = relay.Function([], output)
mutated_mod['func_2088'] = func_2088
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2122 = relay.var("var_2122", dtype = "float64", shape = (10, 10))#candidate|2122|(10, 10)|var|float64
uop_2123 = relay.erf(var_2122.astype('float64')) # shape=(10, 10)
output = relay.Tuple([uop_2123,])
output2 = relay.Tuple([uop_2123,])
func_2132 = relay.Function([var_2122,], output)
mod['func_2132'] = func_2132
mod = relay.transform.InferType()(mod)
var_2133 = relay.var("var_2133", dtype = "float64", shape = (10, 10))#candidate|2133|(10, 10)|var|float64
output = func_2132(var_2133)
func_2134 = relay.Function([var_2133], output)
mutated_mod['func_2134'] = func_2134
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2136 = relay.const([[[False],[True],[True],[True],[True],[False],[True],[False],[True],[False],[False],[True],[True],[True],[False],[False]],[[True],[True],[True],[True],[False],[True],[False],[True],[True],[False],[True],[False],[True],[True],[True],[False]]], dtype = "bool")#candidate|2136|(2, 16, 1)|const|bool
const_2137 = relay.const([[[True,True,False,False,True,True,False,True,True,True,True,True,False,False,False],[True,True,False,False,False,False,True,False,True,False,False,False,True,False,True],[True,True,False,False,True,False,True,True,True,False,False,False,True,False,True],[True,True,False,False,False,False,False,True,True,True,True,True,True,False,True],[False,False,False,True,False,True,True,True,False,False,False,False,True,True,True],[False,True,False,False,True,False,True,True,False,False,False,True,True,True,False],[False,False,False,False,True,False,True,False,True,True,False,True,True,True,True],[False,True,False,True,False,True,True,True,False,True,True,False,True,False,True],[True,False,True,True,False,False,True,True,False,False,False,True,False,True,False],[True,False,False,True,True,False,False,False,True,False,False,False,False,True,False],[True,False,False,True,True,False,True,False,True,False,False,True,True,False,False],[True,True,False,True,False,True,True,False,False,False,False,False,False,False,True],[False,False,True,False,True,False,False,False,False,True,False,False,False,True,True],[True,True,False,True,False,True,True,False,True,True,False,False,True,True,False],[False,True,False,True,False,True,True,False,False,True,False,False,True,False,False],[False,True,True,True,True,True,False,False,True,True,True,False,True,False,True]],[[False,False,False,False,True,False,False,False,True,False,True,True,False,False,True],[False,True,True,True,False,False,True,False,False,False,True,True,False,True,True],[True,True,True,False,True,False,False,True,False,True,True,True,False,False,True],[False,False,False,True,False,False,True,True,False,True,True,False,False,False,True],[True,True,False,False,True,True,False,False,True,False,True,False,True,False,False],[False,False,True,True,True,True,True,False,False,False,True,True,False,True,True],[True,False,True,True,False,False,False,False,False,True,False,False,False,True,False],[False,True,False,True,False,True,False,False,True,False,False,False,False,False,True],[True,True,True,False,False,True,True,False,True,True,False,True,True,False,False],[True,False,True,True,False,False,False,False,False,True,True,False,True,True,False],[False,False,True,True,True,True,False,False,True,True,True,True,False,True,True],[False,False,False,False,False,False,False,False,False,False,False,True,False,True,False],[False,False,True,True,True,False,False,False,False,True,True,True,True,False,False],[False,False,True,False,False,True,False,False,True,True,False,True,False,False,True],[True,False,True,False,True,False,False,True,False,False,True,False,True,False,False],[False,False,False,True,True,True,False,True,True,True,False,False,True,True,False]]], dtype = "bool")#candidate|2137|(2, 16, 15)|const|bool
bop_2138 = relay.logical_or(const_2136.astype('bool'), const_2137.astype('bool')) # shape=(2, 16, 15)
output = bop_2138
output2 = bop_2138
F = relay.Function([], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([], output2)
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
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()()
res3 = intrp3.evaluate()()
res4 = intrp4.evaluate()()
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
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()()
res7 = intrp7.evaluate()()
res8 = intrp8.evaluate()()
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
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()()
res11 = intrp11.evaluate()()
res12 = intrp12.evaluate()()
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
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()()
res15 = intrp15.evaluate()()
res16 = intrp16.evaluate()()
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
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()()
res19 = intrp19.evaluate()()
res20 = intrp20.evaluate()()
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
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()()
res23 = intrp23.evaluate()()
res24 = intrp24.evaluate()()
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

'''45: TVMFuncCall
44: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
43: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
42: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
41: tvm::IRModule::FromExpr(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&)
40: tvm::IRModule::FromExprInContext(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&, std::unordered_set<tvm::runtime::String, std::hash<tvm::runtime::String>, std::equal_to<tvm::runtime::String>, std::allocator<tvm::runtime::String> >)
39: tvm::IRModuleNode::Add(tvm::GlobalVar const&, tvm::BaseFunc const&, bool)
38: tvm::WarnIfMalformed(tvm::IRModule const&, tvm::relay::Function)
37: tvm::relay::FreeTypeVars(tvm::RelayExpr const&, tvm::IRModule const&)
36: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
35: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
34: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
33: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
32: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
31: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
30: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
29: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
28: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
27: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
26: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
25: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
24: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
23: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
22: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
21: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
20: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
19: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
16: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
15: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
14: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
13: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
12: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
11: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
10: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
9: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
8: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
7: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
6: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
5: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
4: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9RelayEx
1: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::ConstructorNode const*)
0: tvm::IRModuleNode::LookupTypeDef(tvm::GlobalTypeVar const&) const

'''