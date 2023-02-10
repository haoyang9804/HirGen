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
var_56 = relay.var("var_56", dtype = "int8", shape = (16, 4))#candidate|56|(16, 4)|var|int8
var_57 = relay.var("var_57", dtype = "int8", shape = (16, 4))#candidate|57|(16, 4)|var|int8
bop_58 = relay.logical_xor(var_56.astype('int8'), relay.reshape(var_57.astype('int8'), relay.shape_of(var_56))) # shape=(16, 4)
uop_67 = relay.sigmoid(var_57.astype('float64')) # shape=(16, 4)
output = relay.Tuple([bop_58,uop_67,])
output2 = relay.Tuple([bop_58,uop_67,])
func_71 = relay.Function([var_56,var_57,], output)
mod['func_71'] = func_71
mod = relay.transform.InferType()(mod)
mutated_mod['func_71'] = func_71
mutated_mod = relay.transform.InferType()(mutated_mod)
func_71_call = mutated_mod.get_global_var('func_71')
var_73 = relay.var("var_73", dtype = "int8", shape = (16, 4))#candidate|73|(16, 4)|var|int8
var_74 = relay.var("var_74", dtype = "int8", shape = (16, 4))#candidate|74|(16, 4)|var|int8
call_72 = func_71_call(var_73,var_74,)
output = call_72
func_75 = relay.Function([var_73,var_74,], output)
mutated_mod['func_75'] = func_75
mutated_mod = relay.transform.InferType()(mutated_mod)
var_93 = relay.var("var_93", dtype = "float32", shape = (8, 11))#candidate|93|(8, 11)|var|float32
uop_94 = relay.atanh(var_93.astype('float32')) # shape=(8, 11)
bop_106 = relay.maximum(uop_94.astype('uint16'), relay.reshape(var_93.astype('uint16'), relay.shape_of(uop_94))) # shape=(8, 11)
bop_109 = relay.floor_divide(bop_106.astype('float32'), relay.reshape(uop_94.astype('float32'), relay.shape_of(bop_106))) # shape=(8, 11)
uop_112 = relay.sinh(bop_109.astype('float32')) # shape=(8, 11)
output = uop_112
output2 = uop_112
func_114 = relay.Function([var_93,], output)
mod['func_114'] = func_114
mod = relay.transform.InferType()(mod)
mutated_mod['func_114'] = func_114
mutated_mod = relay.transform.InferType()(mutated_mod)
var_115 = relay.var("var_115", dtype = "float32", shape = (8, 11))#candidate|115|(8, 11)|var|float32
func_114_call = mutated_mod.get_global_var('func_114')
call_116 = func_114_call(var_115)
output = call_116
func_117 = relay.Function([var_115], output)
mutated_mod['func_117'] = func_117
mutated_mod = relay.transform.InferType()(mutated_mod)
const_138 = relay.const([-2.728049,2.131403,2.683160,-1.393109,4.110133,0.443410,-2.580074,4.569863,4.113670,-4.429818,-4.099965,-0.621270,-5.890032,2.702060,0.342035,1.992464], dtype = "float32")#candidate|138|(16,)|const|float32
uop_139 = relay.atan(const_138.astype('float32')) # shape=(16,)
func_114_call = mod.get_global_var('func_114')
func_117_call = mutated_mod.get_global_var('func_117')
var_145 = relay.var("var_145", dtype = "float32", shape = (88,))#candidate|145|(88,)|var|float32
call_144 = func_114_call(relay.reshape(var_145.astype('float32'), [8, 11]))
call_146 = func_114_call(relay.reshape(var_145.astype('float32'), [8, 11]))
bop_147 = relay.maximum(uop_139.astype('uint32'), relay.reshape(const_138.astype('uint32'), relay.shape_of(uop_139))) # shape=(16,)
var_154 = relay.var("var_154", dtype = "uint32", shape = (16,))#candidate|154|(16,)|var|uint32
bop_155 = relay.subtract(bop_147.astype('uint16'), relay.reshape(var_154.astype('uint16'), relay.shape_of(bop_147))) # shape=(16,)
func_114_call = mod.get_global_var('func_114')
func_117_call = mutated_mod.get_global_var('func_117')
call_158 = func_114_call(relay.reshape(var_145.astype('float32'), [8, 11]))
call_159 = func_114_call(relay.reshape(var_145.astype('float32'), [8, 11]))
uop_164 = relay.atanh(bop_155.astype('float32')) # shape=(16,)
var_170 = relay.var("var_170", dtype = "float32", shape = (16,))#candidate|170|(16,)|var|float32
bop_171 = relay.add(uop_164.astype('int16'), relay.reshape(var_170.astype('int16'), relay.shape_of(uop_164))) # shape=(16,)
var_174 = relay.var("var_174", dtype = "float32", shape = (8, 11))#candidate|174|(8, 11)|var|float32
bop_175 = relay.logical_and(call_158.astype('bool'), relay.reshape(var_174.astype('bool'), relay.shape_of(call_158))) # shape=(8, 11)
bop_178 = relay.logical_and(call_159.astype('bool'), relay.reshape(var_174.astype('bool'), relay.shape_of(call_159))) # shape=(8, 11)
const_180 = relay.const([-9,-4,-2,1,-1,2,5,6,-1,1,-1,8,-2,6,-3,-6], dtype = "int16")#candidate|180|(16,)|const|int16
bop_181 = relay.floor_mod(bop_171.astype('float64'), relay.reshape(const_180.astype('float64'), relay.shape_of(bop_171))) # shape=(16,)
var_190 = relay.var("var_190", dtype = "float64", shape = (16,))#candidate|190|(16,)|var|float64
bop_191 = relay.multiply(bop_181.astype('uint8'), relay.reshape(var_190.astype('uint8'), relay.shape_of(bop_181))) # shape=(16,)
bop_194 = relay.logical_and(bop_155.astype('bool'), relay.reshape(uop_164.astype('bool'), relay.shape_of(bop_155))) # shape=(16,)
func_114_call = mod.get_global_var('func_114')
func_117_call = mutated_mod.get_global_var('func_117')
call_197 = func_114_call(relay.reshape(var_174.astype('float32'), [8, 11]))
call_198 = func_114_call(relay.reshape(var_174.astype('float32'), [8, 11]))
bop_199 = relay.mod(bop_181.astype('float32'), relay.reshape(var_190.astype('float32'), relay.shape_of(bop_181))) # shape=(16,)
bop_205 = relay.logical_and(bop_171.astype('bool'), relay.reshape(var_190.astype('bool'), relay.shape_of(bop_171))) # shape=(16,)
uop_209 = relay.erf(bop_147.astype('float64')) # shape=(16,)
var_214 = relay.var("var_214", dtype = "float32", shape = (16,))#candidate|214|(16,)|var|float32
bop_215 = relay.multiply(uop_139.astype('int16'), relay.reshape(var_214.astype('int16'), relay.shape_of(uop_139))) # shape=(16,)
func_71_call = mod.get_global_var('func_71')
func_75_call = mutated_mod.get_global_var('func_75')
var_223 = relay.var("var_223", dtype = "int8", shape = (64,))#candidate|223|(64,)|var|int8
call_222 = relay.TupleGetItem(func_71_call(relay.reshape(var_223.astype('int8'), [16, 4]), relay.reshape(var_223.astype('int8'), [16, 4]), ), 0)
call_224 = relay.TupleGetItem(func_75_call(relay.reshape(var_223.astype('int8'), [16, 4]), relay.reshape(var_223.astype('int8'), [16, 4]), ), 0)
output = relay.Tuple([call_144,var_145,bop_175,bop_191,bop_194,call_197,bop_199,bop_205,uop_209,bop_215,call_222,var_223,])
output2 = relay.Tuple([call_146,var_145,bop_178,bop_191,bop_194,call_198,bop_199,bop_205,uop_209,bop_215,call_224,var_223,])
func_228 = relay.Function([var_145,var_154,var_170,var_174,var_190,var_214,var_223,], output)
mod['func_228'] = func_228
mod = relay.transform.InferType()(mod)
var_229 = relay.var("var_229", dtype = "float32", shape = (88,))#candidate|229|(88,)|var|float32
var_230 = relay.var("var_230", dtype = "uint32", shape = (16,))#candidate|230|(16,)|var|uint32
var_231 = relay.var("var_231", dtype = "float32", shape = (16,))#candidate|231|(16,)|var|float32
var_232 = relay.var("var_232", dtype = "float32", shape = (8, 11))#candidate|232|(8, 11)|var|float32
var_233 = relay.var("var_233", dtype = "float64", shape = (16,))#candidate|233|(16,)|var|float64
var_234 = relay.var("var_234", dtype = "float32", shape = (16,))#candidate|234|(16,)|var|float32
var_235 = relay.var("var_235", dtype = "int8", shape = (64,))#candidate|235|(64,)|var|int8
output = func_228(var_229,var_230,var_231,var_232,var_233,var_234,var_235,)
func_236 = relay.Function([var_229,var_230,var_231,var_232,var_233,var_234,var_235,], output)
mutated_mod['func_236'] = func_236
mutated_mod = relay.transform.InferType()(mutated_mod)
var_280 = relay.var("var_280", dtype = "float32", shape = (14, 9, 7))#candidate|280|(14, 9, 7)|var|float32
uop_281 = relay.sigmoid(var_280.astype('float32')) # shape=(14, 9, 7)
uop_284 = relay.log10(uop_281.astype('float64')) # shape=(14, 9, 7)
uop_288 = relay.acos(uop_284.astype('float32')) # shape=(14, 9, 7)
bop_291 = relay.less(uop_288.astype('bool'), relay.reshape(uop_284.astype('bool'), relay.shape_of(uop_288))) # shape=(14, 9, 7)
output = relay.Tuple([bop_291,])
output2 = relay.Tuple([bop_291,])
func_294 = relay.Function([var_280,], output)
mod['func_294'] = func_294
mod = relay.transform.InferType()(mod)
mutated_mod['func_294'] = func_294
mutated_mod = relay.transform.InferType()(mutated_mod)
var_295 = relay.var("var_295", dtype = "float32", shape = (14, 9, 7))#candidate|295|(14, 9, 7)|var|float32
func_294_call = mutated_mod.get_global_var('func_294')
call_296 = func_294_call(var_295)
output = call_296
func_297 = relay.Function([var_295], output)
mutated_mod['func_297'] = func_297
mutated_mod = relay.transform.InferType()(mutated_mod)
const_306 = relay.const([[[-0.234398],[5.153022],[2.160808],[-4.523918]],[[-5.640143],[-1.886218],[4.342595],[-3.835596]],[[5.834931],[1.848249],[-5.293703],[8.929568]],[[-4.541544],[-1.322405],[8.846013],[4.886455]],[[1.902934],[-6.894392],[-2.135881],[9.028838]],[[-7.041187],[8.894149],[-7.096462],[-8.742913]],[[5.938241],[-1.215155],[4.168933],[7.151607]],[[6.389745],[2.471253],[6.144124],[-3.919095]],[[-1.323543],[6.191208],[9.548156],[1.215630]]], dtype = "float32")#candidate|306|(9, 4, 1)|const|float32
uop_307 = relay.cos(const_306.astype('float32')) # shape=(9, 4, 1)
uop_309 = relay.rsqrt(uop_307.astype('float32')) # shape=(9, 4, 1)
output = relay.Tuple([uop_309,])
output2 = relay.Tuple([uop_309,])
func_314 = relay.Function([], output)
mod['func_314'] = func_314
mod = relay.transform.InferType()(mod)
mutated_mod['func_314'] = func_314
mutated_mod = relay.transform.InferType()(mutated_mod)
func_314_call = mutated_mod.get_global_var('func_314')
call_315 = func_314_call()
output = call_315
func_316 = relay.Function([], output)
mutated_mod['func_316'] = func_316
mutated_mod = relay.transform.InferType()(mutated_mod)
var_328 = relay.var("var_328", dtype = "float32", shape = (7,))#candidate|328|(7,)|var|float32
uop_329 = relay.asin(var_328.astype('float32')) # shape=(7,)
bop_331 = relay.mod(uop_329.astype('float64'), relay.reshape(var_328.astype('float64'), relay.shape_of(uop_329))) # shape=(7,)
uop_336 = relay.log(uop_329.astype('float32')) # shape=(7,)
uop_343 = relay.atan(uop_336.astype('float64')) # shape=(7,)
var_348 = relay.var("var_348", dtype = "float32", shape = (7,))#candidate|348|(7,)|var|float32
bop_349 = relay.less_equal(uop_336.astype('bool'), relay.reshape(var_348.astype('bool'), relay.shape_of(uop_336))) # shape=(7,)
func_114_call = mod.get_global_var('func_114')
func_117_call = mutated_mod.get_global_var('func_117')
var_354 = relay.var("var_354", dtype = "float32", shape = (88,))#candidate|354|(88,)|var|float32
call_353 = func_114_call(relay.reshape(var_354.astype('float32'), [8, 11]))
call_355 = func_114_call(relay.reshape(var_354.astype('float32'), [8, 11]))
bop_356 = relay.power(uop_343.astype('float32'), relay.reshape(uop_336.astype('float32'), relay.shape_of(uop_343))) # shape=(7,)
bop_364 = relay.subtract(bop_349.astype('int16'), relay.reshape(var_348.astype('int16'), relay.shape_of(bop_349))) # shape=(7,)
bop_368 = relay.right_shift(bop_364.astype('int32'), relay.reshape(var_348.astype('int32'), relay.shape_of(bop_364))) # shape=(7,)
bop_373 = relay.multiply(bop_349.astype('int16'), relay.reshape(bop_364.astype('int16'), relay.shape_of(bop_349))) # shape=(7,)
func_228_call = mod.get_global_var('func_228')
func_236_call = mutated_mod.get_global_var('func_236')
const_381 = relay.const([-2,-7,-5,2,-2,-8,-9,-10,-10,1,5,10,-4,9,6,2], dtype = "uint32")#candidate|381|(16,)|const|uint32
var_382 = relay.var("var_382", dtype = "int8", shape = (4, 16))#candidate|382|(4, 16)|var|int8
call_380 = relay.TupleGetItem(func_228_call(relay.reshape(var_354.astype('float32'), [88,]), relay.reshape(const_381.astype('uint32'), [16,]), relay.reshape(const_381.astype('float32'), [16,]), relay.reshape(call_353.astype('float32'), [8, 11]), relay.reshape(const_381.astype('float64'), [16,]), relay.reshape(const_381.astype('float32'), [16,]), relay.reshape(var_382.astype('int8'), [64,]), ), 0)
call_383 = relay.TupleGetItem(func_236_call(relay.reshape(var_354.astype('float32'), [88,]), relay.reshape(const_381.astype('uint32'), [16,]), relay.reshape(const_381.astype('float32'), [16,]), relay.reshape(call_353.astype('float32'), [8, 11]), relay.reshape(const_381.astype('float64'), [16,]), relay.reshape(const_381.astype('float32'), [16,]), relay.reshape(var_382.astype('int8'), [64,]), ), 0)
bop_387 = relay.add(bop_373.astype('float32'), relay.reshape(bop_364.astype('float32'), relay.shape_of(bop_373))) # shape=(7,)
uop_396 = relay.erf(bop_349.astype('float32')) # shape=(7,)
uop_398 = relay.sin(bop_368.astype('float64')) # shape=(7,)
bop_404 = relay.minimum(uop_396.astype('int64'), relay.reshape(uop_398.astype('int64'), relay.shape_of(uop_396))) # shape=(7,)
var_407 = relay.var("var_407", dtype = "float32", shape = (7,))#candidate|407|(7,)|var|float32
bop_408 = relay.not_equal(bop_387.astype('bool'), relay.reshape(var_407.astype('bool'), relay.shape_of(bop_387))) # shape=(7,)
output = relay.Tuple([bop_331,call_353,var_354,bop_356,call_380,const_381,var_382,bop_404,bop_408,])
output2 = relay.Tuple([bop_331,call_355,var_354,bop_356,call_383,const_381,var_382,bop_404,bop_408,])
func_412 = relay.Function([var_328,var_348,var_354,var_382,var_407,], output)
mod['func_412'] = func_412
mod = relay.transform.InferType()(mod)
var_413 = relay.var("var_413", dtype = "float32", shape = (7,))#candidate|413|(7,)|var|float32
var_414 = relay.var("var_414", dtype = "float32", shape = (7,))#candidate|414|(7,)|var|float32
var_415 = relay.var("var_415", dtype = "float32", shape = (88,))#candidate|415|(88,)|var|float32
var_416 = relay.var("var_416", dtype = "int8", shape = (4, 16))#candidate|416|(4, 16)|var|int8
var_417 = relay.var("var_417", dtype = "float32", shape = (7,))#candidate|417|(7,)|var|float32
output = func_412(var_413,var_414,var_415,var_416,var_417,)
func_418 = relay.Function([var_413,var_414,var_415,var_416,var_417,], output)
mutated_mod['func_418'] = func_418
mutated_mod = relay.transform.InferType()(mutated_mod)
func_314_call = mod.get_global_var('func_314')
func_316_call = mutated_mod.get_global_var('func_316')
call_453 = relay.TupleGetItem(func_314_call(), 0)
call_454 = relay.TupleGetItem(func_316_call(), 0)
var_461 = relay.var("var_461", dtype = "float32", shape = (9, 4, 11))#candidate|461|(9, 4, 11)|var|float32
bop_462 = relay.subtract(call_453.astype('uint16'), var_461.astype('uint16')) # shape=(9, 4, 11)
bop_465 = relay.subtract(call_454.astype('uint16'), var_461.astype('uint16')) # shape=(9, 4, 11)
uop_466 = relay.asin(call_453.astype('float64')) # shape=(9, 4, 1)
uop_468 = relay.asin(call_454.astype('float64')) # shape=(9, 4, 1)
var_469 = relay.var("var_469", dtype = "float64", shape = (9, 4, 14))#candidate|469|(9, 4, 14)|var|float64
bop_470 = relay.logical_or(uop_466.astype('bool'), var_469.astype('bool')) # shape=(9, 4, 14)
bop_473 = relay.logical_or(uop_468.astype('bool'), var_469.astype('bool')) # shape=(9, 4, 14)
output = relay.Tuple([bop_462,bop_470,])
output2 = relay.Tuple([bop_465,bop_473,])
func_474 = relay.Function([var_461,var_469,], output)
mod['func_474'] = func_474
mod = relay.transform.InferType()(mod)
var_475 = relay.var("var_475", dtype = "float32", shape = (9, 4, 11))#candidate|475|(9, 4, 11)|var|float32
var_476 = relay.var("var_476", dtype = "float64", shape = (9, 4, 14))#candidate|476|(9, 4, 14)|var|float64
output = func_474(var_475,var_476,)
func_477 = relay.Function([var_475,var_476,], output)
mutated_mod['func_477'] = func_477
mutated_mod = relay.transform.InferType()(mutated_mod)
var_482 = relay.var("var_482", dtype = "float64", shape = (12,))#candidate|482|(12,)|var|float64
const_483 = relay.const([1.815658,9.036204,-0.252076,7.630498,-4.500174,-8.763631,-2.055745,-1.630870,8.859314,4.867120,0.534666,6.449129], dtype = "float64")#candidate|483|(12,)|const|float64
bop_484 = relay.not_equal(var_482.astype('bool'), relay.reshape(const_483.astype('bool'), relay.shape_of(var_482))) # shape=(12,)
func_228_call = mod.get_global_var('func_228')
func_236_call = mutated_mod.get_global_var('func_236')
const_489 = relay.const([-8.966036,3.721494,2.656836,-5.973279,-7.232660,-3.083668,-5.844245,8.507646,0.997887,8.343320,-3.582043,8.659028,-2.020570,8.823603,9.540149,2.450749,8.661765,1.815669,1.249250,0.847615,-4.380370,3.127182,5.557649,-5.362090,1.789143,-9.955019,-2.689873,-7.451758,-6.049167,9.536171,-1.705469,-1.308240,5.252854,9.474068,0.690865,9.343855,-7.509888,3.739411,2.453276,-2.748816,2.863066,-4.258010,3.173820,-3.582553,-6.652377,5.628772,-9.044902,-1.527435,6.333918,-4.544776,-6.632212,-7.105429,-2.322160,4.671552,6.935930,7.751751,-5.072735,7.213311,-7.496961,2.370432,2.220762,-7.732500,-6.021980,6.855275,-1.346337,3.322320,3.716812,1.700267,8.948112,7.136197,7.407420,4.403483,-5.932768,0.804342,-6.221595,0.160964,3.903802,0.119435,-9.518290,8.661565,-6.617375,-3.098814,-2.060665,0.760312,-3.779721,-3.014800,0.998425,2.700880], dtype = "float32")#candidate|489|(88,)|const|float32
var_490 = relay.var("var_490", dtype = "uint32", shape = (16,))#candidate|490|(16,)|var|uint32
const_491 = relay.const([7,5,10,3,-3,8,6,5,-10,1,-6,9,-2,5,10,-9,1,-8,-10,-3,-3,-3,2,-2,-7,-4,9,-6,-10,4,-10,6,7,-4,-4,-9,7,5,-2,-2,9,-1,-9,-9,8,8,8,-9,7,-1,-2,7,2,1,-10,4,-8,8,-2,8,-2,-7,-10,5], dtype = "int8")#candidate|491|(64,)|const|int8
call_488 = relay.TupleGetItem(func_228_call(relay.reshape(const_489.astype('float32'), [88,]), relay.reshape(var_490.astype('uint32'), [16,]), relay.reshape(var_490.astype('float32'), [16,]), relay.reshape(const_489.astype('float32'), [8, 11]), relay.reshape(var_490.astype('float64'), [16,]), relay.reshape(var_490.astype('float32'), [16,]), relay.reshape(const_491.astype('int8'), [64,]), ), 10)
call_492 = relay.TupleGetItem(func_236_call(relay.reshape(const_489.astype('float32'), [88,]), relay.reshape(var_490.astype('uint32'), [16,]), relay.reshape(var_490.astype('float32'), [16,]), relay.reshape(const_489.astype('float32'), [8, 11]), relay.reshape(var_490.astype('float64'), [16,]), relay.reshape(var_490.astype('float32'), [16,]), relay.reshape(const_491.astype('int8'), [64,]), ), 10)
output = relay.Tuple([bop_484,call_488,const_489,var_490,const_491,])
output2 = relay.Tuple([bop_484,call_492,const_489,var_490,const_491,])
func_493 = relay.Function([var_482,var_490,], output)
mod['func_493'] = func_493
mod = relay.transform.InferType()(mod)
var_494 = relay.var("var_494", dtype = "float64", shape = (12,))#candidate|494|(12,)|var|float64
var_495 = relay.var("var_495", dtype = "uint32", shape = (16,))#candidate|495|(16,)|var|uint32
output = func_493(var_494,var_495,)
func_496 = relay.Function([var_494,var_495,], output)
mutated_mod['func_496'] = func_496
mutated_mod = relay.transform.InferType()(mutated_mod)
var_498 = relay.var("var_498", dtype = "float64", shape = (11, 5))#candidate|498|(11, 5)|var|float64
uop_499 = relay.cosh(var_498.astype('float64')) # shape=(11, 5)
bop_504 = relay.right_shift(uop_499.astype('uint64'), relay.reshape(var_498.astype('uint64'), relay.shape_of(uop_499))) # shape=(11, 5)
bop_510 = relay.minimum(bop_504.astype('int8'), relay.reshape(uop_499.astype('int8'), relay.shape_of(bop_504))) # shape=(11, 5)
func_114_call = mod.get_global_var('func_114')
func_117_call = mutated_mod.get_global_var('func_117')
var_514 = relay.var("var_514", dtype = "float32", shape = (88,))#candidate|514|(88,)|var|float32
call_513 = func_114_call(relay.reshape(var_514.astype('float32'), [8, 11]))
call_515 = func_114_call(relay.reshape(var_514.astype('float32'), [8, 11]))
var_521 = relay.var("var_521", dtype = "int8", shape = (11, 5))#candidate|521|(11, 5)|var|int8
bop_522 = relay.logical_xor(bop_510.astype('int16'), relay.reshape(var_521.astype('int16'), relay.shape_of(bop_510))) # shape=(11, 5)
bop_527 = relay.equal(uop_499.astype('bool'), relay.reshape(bop_510.astype('bool'), relay.shape_of(uop_499))) # shape=(11, 5)
uop_531 = relay.log(bop_510.astype('float64')) # shape=(11, 5)
func_493_call = mod.get_global_var('func_493')
func_496_call = mutated_mod.get_global_var('func_496')
var_539 = relay.var("var_539", dtype = "float64", shape = (3, 4))#candidate|539|(3, 4)|var|float64
var_540 = relay.var("var_540", dtype = "uint32", shape = (8, 2))#candidate|540|(8, 2)|var|uint32
call_538 = relay.TupleGetItem(func_493_call(relay.reshape(var_539.astype('float64'), [12,]), relay.reshape(var_540.astype('uint32'), [16,]), ), 1)
call_541 = relay.TupleGetItem(func_496_call(relay.reshape(var_539.astype('float64'), [12,]), relay.reshape(var_540.astype('uint32'), [16,]), ), 1)
output = relay.Tuple([call_513,var_514,bop_522,bop_527,uop_531,call_538,var_539,var_540,])
output2 = relay.Tuple([call_515,var_514,bop_522,bop_527,uop_531,call_541,var_539,var_540,])
func_545 = relay.Function([var_498,var_514,var_521,var_539,var_540,], output)
mod['func_545'] = func_545
mod = relay.transform.InferType()(mod)
mutated_mod['func_545'] = func_545
mutated_mod = relay.transform.InferType()(mutated_mod)
func_545_call = mutated_mod.get_global_var('func_545')
var_547 = relay.var("var_547", dtype = "float64", shape = (11, 5))#candidate|547|(11, 5)|var|float64
var_548 = relay.var("var_548", dtype = "float32", shape = (88,))#candidate|548|(88,)|var|float32
var_549 = relay.var("var_549", dtype = "int8", shape = (11, 5))#candidate|549|(11, 5)|var|int8
var_550 = relay.var("var_550", dtype = "float64", shape = (3, 4))#candidate|550|(3, 4)|var|float64
var_551 = relay.var("var_551", dtype = "uint32", shape = (8, 2))#candidate|551|(8, 2)|var|uint32
call_546 = func_545_call(var_547,var_548,var_549,var_550,var_551,)
output = call_546
func_552 = relay.Function([var_547,var_548,var_549,var_550,var_551,], output)
mutated_mod['func_552'] = func_552
mutated_mod = relay.transform.InferType()(mutated_mod)
var_554 = relay.var("var_554", dtype = "float64", shape = (2,))#candidate|554|(2,)|var|float64
uop_555 = relay.atan(var_554.astype('float64')) # shape=(2,)
output = relay.Tuple([uop_555,])
output2 = relay.Tuple([uop_555,])
func_557 = relay.Function([var_554,], output)
mod['func_557'] = func_557
mod = relay.transform.InferType()(mod)
mutated_mod['func_557'] = func_557
mutated_mod = relay.transform.InferType()(mutated_mod)
var_558 = relay.var("var_558", dtype = "float64", shape = (2,))#candidate|558|(2,)|var|float64
func_557_call = mutated_mod.get_global_var('func_557')
call_559 = func_557_call(var_558)
output = call_559
func_560 = relay.Function([var_558], output)
mutated_mod['func_560'] = func_560
mutated_mod = relay.transform.InferType()(mutated_mod)
func_314_call = mod.get_global_var('func_314')
func_316_call = mutated_mod.get_global_var('func_316')
call_572 = relay.TupleGetItem(func_314_call(), 0)
call_573 = relay.TupleGetItem(func_316_call(), 0)
func_557_call = mod.get_global_var('func_557')
func_560_call = mutated_mod.get_global_var('func_560')
var_575 = relay.var("var_575", dtype = "float64", shape = (2,))#candidate|575|(2,)|var|float64
call_574 = relay.TupleGetItem(func_557_call(relay.reshape(var_575.astype('float64'), [2,])), 0)
call_576 = relay.TupleGetItem(func_560_call(relay.reshape(var_575.astype('float64'), [2,])), 0)
func_314_call = mod.get_global_var('func_314')
func_316_call = mutated_mod.get_global_var('func_316')
call_592 = relay.TupleGetItem(func_314_call(), 0)
call_593 = relay.TupleGetItem(func_316_call(), 0)
output = relay.Tuple([call_572,call_574,var_575,call_592,])
output2 = relay.Tuple([call_573,call_576,var_575,call_593,])
func_597 = relay.Function([var_575,], output)
mod['func_597'] = func_597
mod = relay.transform.InferType()(mod)
mutated_mod['func_597'] = func_597
mutated_mod = relay.transform.InferType()(mutated_mod)
var_598 = relay.var("var_598", dtype = "float64", shape = (2,))#candidate|598|(2,)|var|float64
func_597_call = mutated_mod.get_global_var('func_597')
call_599 = func_597_call(var_598)
output = call_599
func_600 = relay.Function([var_598], output)
mutated_mod['func_600'] = func_600
mutated_mod = relay.transform.InferType()(mutated_mod)
var_631 = relay.var("var_631", dtype = "float32", shape = (4, 6, 2))#candidate|631|(4, 6, 2)|var|float32
uop_632 = relay.asinh(var_631.astype('float32')) # shape=(4, 6, 2)
var_638 = relay.var("var_638", dtype = "float32", shape = (4, 6, 2))#candidate|638|(4, 6, 2)|var|float32
bop_639 = relay.power(var_631.astype('float64'), relay.reshape(var_638.astype('float64'), relay.shape_of(var_631))) # shape=(4, 6, 2)
uop_643 = relay.cos(uop_632.astype('float64')) # shape=(4, 6, 2)
func_294_call = mod.get_global_var('func_294')
func_297_call = mutated_mod.get_global_var('func_297')
const_650 = relay.const([-6.266277,7.116195,2.881496,-9.477113,6.594401,1.692379,0.603569,-2.374647,-8.233118,-5.148014,0.058070,-3.309495,2.783700,0.703038,-5.368057,2.845578,5.539215,-8.122096,3.488606,0.380034,-3.346966,-5.260103,-7.316282,-0.149832,9.811228,4.981163,6.338640,-5.939329,2.072965,0.750910,-9.841495,-0.487920,-4.406237,-0.104663,9.971385,-4.605684,-9.506192,5.985015,2.997120,1.749555,-0.112606,-9.157547,-8.263716,-0.083025,-1.061483,1.447170,6.936386,-1.077396,9.209724,6.353647,-9.673217,-5.152928,-7.596763,-9.687373,3.806646,1.057021,-5.310903,4.940523,-5.726482,-9.869916,-7.953092,7.088176,9.947265,8.095854,-1.367042,9.822978,-1.750301,3.995298,4.655152,1.750752,9.853337,-9.057105,-8.310030,-0.141027,-1.544643,9.978357,6.538277,-9.179741,-8.542308,3.288612,-9.865632,-5.687723,-1.907856,8.195148,9.391042,-0.381854,9.643371,-7.252631,-4.053328,5.882444,3.639533,0.334544,8.308313,9.550218,1.307184,7.459459,2.950041,4.162452,4.463187,5.344379,9.604068,-6.561824,-6.353496,-4.270834,8.739560,6.814433,-8.395846,8.363085,-5.972283,5.958733,3.751933,6.171045,-4.832532,-8.151933,-4.247640,-2.076155,5.298721,-8.780284,5.053639,8.312601,-3.675687,7.351850,7.661033,6.294554,2.831001,-4.087812,8.499697,2.584813,-3.679096,1.123325,3.426557,-2.850062,-4.802267,5.022114,0.572561,8.978846,5.712237,-4.315505,5.358949,3.083469,-5.863836,-3.784633,0.231304,-7.299531,8.188090,-2.444984,8.561136,-3.411436,-4.744389,-5.194189,-5.209552,0.398825,-7.078514,-8.098302,0.626752,6.878658,7.279852,6.667616,4.161828,-9.614030,-4.987087,-6.332292,4.522888,8.645206,-4.895855,2.578507,9.828783,-4.576873,-1.811694,-4.798533,-1.599182,-8.404253,0.441883,0.795143,1.229364,3.588134,-0.124690,0.776817,-1.726074,4.395173,-6.036787,7.257234,-0.952730,-3.829106,8.972213,-1.969093,-3.364405,0.192481,2.690391,-3.137970,7.225918,-8.744557,7.657893,1.094567,-9.976038,-1.591117,-2.322792,8.004446,-2.803313,5.082982,-4.175230,0.182230,-1.834664,-5.995410,9.853414,-2.229820,-4.380825,1.471192,4.026159,-1.816295,8.896429,-4.209548,-8.180360,-8.484553,1.744544,-8.909961,-0.138605,-1.652957,3.463109,-5.436874,-2.365309,9.121471,2.999591,-9.051436,9.074783,-5.969115,-5.275076,-9.738313,-2.761081,1.371251,5.953950,5.715638,5.768279,-4.780039,-5.512663,-0.904071,-3.574107,-2.429121,-7.428966,-5.760417,4.248686,-0.863214,9.606518,-5.547387,-8.820486,-0.094126,-4.353784,4.898084,9.725510,-7.174550,-9.475070,-4.777601,-1.182674,-9.089667,-1.275102,-1.928968,4.115187,8.648518,-7.910354,8.499296,6.882815,-6.412976,5.181578,8.554310,6.845975,2.783673,-6.726378,-0.698423,-3.844216,2.680979,9.495532,-8.226631,2.181232,-9.384712,5.407673,-9.140565,3.245884,-3.420983,-5.545504,5.315182,0.273642,4.972235,4.389911,-3.161936,-0.793935,-6.435716,-0.297750,-1.296493,-7.089729,8.883637,-6.610169,-0.084996,-5.440185,4.341974,-9.157843,-4.299243,7.147217,-7.938372,-4.706748,-9.834459,-2.475283,-8.693133,0.436933,-2.619901,8.964611,2.971323,1.391595,4.608980,9.995285,-8.533036,-6.966548,0.157761,-9.709362,-1.345444,4.338632,6.765382,0.332453,-1.305005,9.883352,-1.340195,-0.290247,-3.833228,8.921936,5.684644,-0.884602,-5.688720,-0.993278,-2.254373,9.884953,6.784826,0.618355,8.293690,3.271299,4.111304,-8.725896,2.191726,-4.947188,1.709709,-2.897290,-9.613190,-5.813668,1.391672,-1.101058,-8.744628,-9.469269,-3.388431,-3.763139,-5.331358,6.601291,-7.072092,8.016970,-3.426145,-6.299262,0.082146,-7.077459,1.839681,-1.805965,-7.757089,6.546746,1.721791,-8.615386,3.341319,9.273531,-9.905236,-3.948081,-0.852654,6.565806,3.406809,-2.933396,-6.376102,-0.896174,-3.312002,-5.379474,3.785303,0.153955,4.585857,1.997112,-4.770750,-2.814493,1.929347,-9.568356,-3.694497,2.046942,1.680936,-1.511906,-4.425521,0.849679,4.066080,3.875410,0.492891,0.149291,1.315014,-7.868393,0.886565,-6.174070,6.482909,9.298978,-6.461739,4.047932,-6.192649,5.946347,-2.598143,4.036031,-7.324143,-7.088898,7.934616,-0.761784,-7.427868,3.864997,7.085479,4.629126,-8.209345,-0.247011,0.212558,-3.139505,7.576612,-5.577588,-0.772674,-3.448611,1.551351,9.903733,-4.821835,2.060192,-1.428475,1.652375,-5.416698,-9.674123,6.943690,-7.844415,-2.754108,-5.991569,-9.641131,2.084921,4.460070,-7.029215,-7.975061,3.540608,3.933825,-8.181524,0.078384,-7.612741,-6.694991,-3.310784,3.129048,-9.290631,3.643204,-5.766441,2.982223,4.293623,1.175060,4.970648,6.809549,-2.337061,2.734902,-9.698359,-3.582942,9.809608,2.975410,-5.451464,-4.100614,-4.431261,-9.254062,9.017611,-1.465709,-6.446440,-6.569684,8.390826,-9.545231,8.239641,-4.005809,4.178595,-4.402811,6.166733,-7.535797,-7.171226,-4.797153,1.934126,0.190651,4.683891,1.339599,6.489976,9.370036,5.900775,4.558309,-8.172658,-1.143926,-8.226651,-6.942528,-8.256143,-1.502463,0.382537,2.056717,7.336747,5.525769,-5.889890,-4.186334,8.777375,-4.813872,5.492952,2.096982,-9.191145,0.074080,-1.911439,-5.228125,7.755734,7.504403,-1.594952,6.297872,-5.979331,-7.434297,5.030505,-0.501647,-1.433968,8.790023,-7.888617,1.612755,-0.833936,-7.213907,-2.686029,7.917912,2.250968,-7.998654,8.565853,8.318925,9.161006,-0.776173,-6.712187,6.952724,4.465695,-2.555913,7.216468,-5.453998,2.599245,-6.573729,-6.532898,9.732832,-3.776042,0.678994,1.775268,0.870000,-7.639776,9.173250,4.195201,8.687481,0.147502,-9.364959,-7.021821,-9.802839,8.350613,1.412380,-3.974804,6.479585,2.935289,-0.639746,5.221160,1.644310,5.443784,-2.680131,5.416231,0.479256,-9.638274,2.932418,2.025199,-4.835075,-5.985125,-4.236899,-7.136644,8.552213,-1.087307,-7.465689,-3.606107,8.850841,-2.361084,-4.560737,3.155205,-1.439213,6.840827,-8.703804,3.632374,-7.559763,9.557528,-2.077775,-3.676987,5.347552,3.848886,3.008233,1.678216,6.550596,-2.704269,9.035446,5.198667,1.379289,6.197169,9.458409,-6.414609,-3.781707,-4.955915,9.139943,-3.676543,9.964219,-5.186395,9.452131,1.562775,2.102441,-4.453716,-4.791794,-2.218671,3.729166,5.013436,-9.547816,-9.591885,0.530957,3.136178,4.493446,6.219806,1.406987,9.231066,3.363950,4.433871,7.017502,7.581400,-7.510023,6.639301,8.427416,-2.055185,2.844539,-5.516050,-1.912740,8.058331,-7.108832,-9.309649,6.600187,-9.295177,3.395238,8.478666,-2.539854,-9.439478,1.144082,-4.246590,-1.019715,-8.653112,1.167410,5.688770,3.953648,4.708019,0.986408,4.710230,-5.150330,-0.833556,-0.812291,-7.798747,-3.985789,9.319503,-1.336527,1.450546,-5.586087,-2.187042,-6.495936,9.793369,5.363018,1.935209,1.179320,8.869039,6.767283,2.225192,9.135740,-6.646933,-2.010365,8.176408,4.232490,7.064791,9.624131,2.854868,0.388999,3.007589,6.849373,-6.585180,9.879089,-0.156838,5.215506,-8.168006,4.938571,-5.036953,-3.251959,4.936847,-5.914003,-2.405063,-5.290235,-9.976357,-9.543371,-8.061674,-1.223655,-3.168797,-4.157636,-8.107544,-6.223424,0.077513,3.176962,4.396615,9.235653,2.931310,-6.920309,4.928582,2.421837,-5.507174,7.121423,-9.417271,8.909857,-9.956715,1.750342,3.667881,0.763196,-5.468175,5.217288,9.341155,-1.517558,-9.153282,-9.887651,2.825379,4.424722,8.024347,3.810897,3.132910,9.932176,-6.006269,-2.876727,-9.251521,-2.862515,5.305599,0.198877,9.078036,4.851605,-8.407938,9.403311,-5.315790,8.095711,-4.193963,2.777889,-8.572719,1.267548,9.911326,-8.518841,5.164034,-1.654671,-9.439023,2.552950,2.451695,-5.778306,-4.466957,-2.309703,-9.113511,2.591603,-7.508908,-0.321895,3.263028,3.302555,-4.042320,-4.919221,9.018652,-5.754458,-2.702907,2.894958,-9.103063,1.594483,5.437445,6.833275,-4.400616,-3.443146,9.632062,2.584835,-4.459230,-8.883608,-8.437130,0.461774,6.111783,-7.311649,0.881726,-5.323900,6.111364,1.613137,2.757973,-4.807585,0.197539,-6.868124,-9.737547,0.085137,-0.882475,8.240807,-2.628819,-1.590949,-6.937241,-5.993436,-9.679441,2.739719,-7.401457,4.093566,-4.593339,5.170421,-1.030545,-3.965565,3.534213,-6.981515,6.459162,5.427418,-8.425116,-4.998196,-3.597645,-0.522946,5.172041,-0.894666,-4.630165,4.450147,2.618081,-1.160378,-2.295401,-5.292802,-5.369789,2.865635,-8.153709,-0.135366,9.643107,2.212600,9.009257,-2.594889,-5.885115,-1.758820,-3.053259,1.860627,3.825009,-4.176220,-5.236290,0.412270,3.920718,1.414057,1.852464,-4.975962,-2.156417,-7.897974,1.742899,-3.867787,8.373084,8.314811,-9.839559,-0.144231,-4.186513,8.590832,9.955105,-0.654647,6.671727,3.751780,3.155574,0.605081,3.589905,7.322323,3.044217,-3.447975,-2.472931,-9.367059,-2.340450,-2.701274,1.333743,-0.091907,5.589806,8.560507,-9.616180,9.653384,-2.199816,4.780249,1.389360,9.981786,-9.580571,7.259932,3.656113,5.748434,-3.881741,-6.028853,6.672140,8.864670,9.923945,2.313050,6.394161,5.187062], dtype = "float32")#candidate|650|(882,)|const|float32
call_649 = relay.TupleGetItem(func_294_call(relay.reshape(const_650.astype('float32'), [14, 9, 7])), 0)
call_651 = relay.TupleGetItem(func_297_call(relay.reshape(const_650.astype('float32'), [14, 9, 7])), 0)
const_652 = relay.const([[[4.773133,5.519004],[-8.639386,-9.396879],[-9.073898,4.368602],[-2.539771,-0.465612],[-5.843277,2.962416],[-3.809030,-2.936086]],[[6.977288,-9.330349],[5.509284,5.017571],[-3.247823,5.247510],[3.304945,1.097620],[5.441515,6.673663],[-3.542221,-4.215593]],[[3.071558,-3.930021],[3.781446,0.698501],[1.375957,-1.560769],[1.049772,9.171151],[-4.745306,8.254925],[-1.898036,2.991490]],[[7.829597,-6.794892],[1.541162,6.952127],[-0.009989,-4.827166],[-9.236002,-8.866578],[0.620167,9.910840],[-4.294384,-0.539961]]], dtype = "float64")#candidate|652|(4, 6, 2)|const|float64
bop_653 = relay.logical_and(uop_643.astype('bool'), relay.reshape(const_652.astype('bool'), relay.shape_of(uop_643))) # shape=(4, 6, 2)
uop_656 = relay.erf(bop_653.astype('float32')) # shape=(4, 6, 2)
output = relay.Tuple([bop_639,call_649,const_650,uop_656,])
output2 = relay.Tuple([bop_639,call_651,const_650,uop_656,])
func_658 = relay.Function([var_631,var_638,], output)
mod['func_658'] = func_658
mod = relay.transform.InferType()(mod)
var_659 = relay.var("var_659", dtype = "float32", shape = (4, 6, 2))#candidate|659|(4, 6, 2)|var|float32
var_660 = relay.var("var_660", dtype = "float32", shape = (4, 6, 2))#candidate|660|(4, 6, 2)|var|float32
output = func_658(var_659,var_660,)
func_661 = relay.Function([var_659,var_660,], output)
mutated_mod['func_661'] = func_661
mutated_mod = relay.transform.InferType()(mutated_mod)
var_663 = relay.var("var_663", dtype = "int8", shape = ())#candidate|663|()|var|int8
const_664 = relay.const([[[-10,6,-6,4,1],[9,-1,-4,7,-10],[3,10,-7,-4,-6],[2,3,-6,4,2],[-2,-9,-5,-7,-2],[-1,-1,4,7,-4],[-7,8,4,-1,-4],[8,-4,-10,-2,1],[8,-9,-2,-8,-1]],[[3,1,-9,9,-5],[-3,-10,2,5,-9],[-5,10,-5,-7,5],[-5,5,2,-1,6],[-4,1,-5,6,-6],[-6,-1,7,1,10],[1,4,3,1,-9],[-2,-5,8,7,-10],[8,-3,4,-6,-3]],[[1,-4,-3,-3,1],[-3,-6,9,10,10],[8,-8,9,-3,9],[3,-2,-6,-4,-1],[-10,1,3,-2,5],[-5,-9,7,6,-5],[8,4,-7,-6,9],[5,2,-7,-5,7],[10,6,-4,-1,3]],[[-7,6,-10,4,-3],[3,-4,3,9,-2],[-9,-10,-4,-7,7],[1,-10,5,-9,8],[-2,9,6,-9,1],[1,2,4,-8,-10],[-1,-8,7,3,-4],[7,1,-4,-7,10],[-2,7,-4,3,4]],[[-8,-6,7,2,1],[-10,4,4,6,10],[-7,-1,-10,7,-9],[-9,-5,-3,-8,2],[-3,-8,6,6,1],[-10,-10,-7,5,6],[-5,1,7,2,-3],[5,-9,-4,4,8],[5,8,-8,10,6]],[[-9,10,-1,-8,-4],[-9,6,3,4,-7],[5,-10,-1,-5,-6],[-4,4,-7,-3,-9],[-5,-6,-8,-8,-6],[-5,-10,3,5,-10],[-3,-8,-10,-3,9],[8,4,-4,4,-7],[8,3,-8,9,8]],[[-6,6,7,8,-7],[-6,2,2,2,-9],[-8,2,-5,10,6],[3,-9,8,3,8],[7,-8,3,7,5],[6,-1,7,-9,-9],[-4,-8,1,1,-4],[-7,-4,-9,6,6],[-10,-7,1,8,-10]],[[6,1,-1,4,10],[10,6,-2,-4,-1],[-10,-1,5,-10,3],[-1,-7,6,9,-9],[-7,-6,-6,-10,-7],[-9,3,5,-5,10],[-8,-8,5,3,4],[-3,-3,-4,7,7],[4,-10,-7,4,1]],[[-9,-6,4,-3,-7],[8,-9,-9,8,-8],[10,3,10,7,4],[-9,7,-4,-7,7],[-5,6,-6,9,-10],[10,-1,-10,-5,-5],[6,-3,-9,3,3],[2,5,-6,-2,6],[3,2,1,5,10]]], dtype = "int8")#candidate|664|(9, 9, 5)|const|int8
bop_665 = relay.left_shift(var_663.astype('int8'), const_664.astype('int8')) # shape=(9, 9, 5)
output = relay.Tuple([bop_665,])
output2 = relay.Tuple([bop_665,])
func_672 = relay.Function([var_663,], output)
mod['func_672'] = func_672
mod = relay.transform.InferType()(mod)
var_673 = relay.var("var_673", dtype = "int8", shape = ())#candidate|673|()|var|int8
output = func_672(var_673)
func_674 = relay.Function([var_673], output)
mutated_mod['func_674'] = func_674
mutated_mod = relay.transform.InferType()(mutated_mod)
func_314_call = mod.get_global_var('func_314')
func_316_call = mutated_mod.get_global_var('func_316')
call_682 = relay.TupleGetItem(func_314_call(), 0)
call_683 = relay.TupleGetItem(func_316_call(), 0)
func_557_call = mod.get_global_var('func_557')
func_560_call = mutated_mod.get_global_var('func_560')
const_685 = relay.const([[-9.190866],[9.057596]], dtype = "float64")#candidate|685|(2, 1)|const|float64
call_684 = relay.TupleGetItem(func_557_call(relay.reshape(const_685.astype('float64'), [2,])), 0)
call_686 = relay.TupleGetItem(func_560_call(relay.reshape(const_685.astype('float64'), [2,])), 0)
func_493_call = mod.get_global_var('func_493')
func_496_call = mutated_mod.get_global_var('func_496')
var_691 = relay.var("var_691", dtype = "float64", shape = (12, 1))#candidate|691|(12, 1)|var|float64
var_692 = relay.var("var_692", dtype = "uint32", shape = (16,))#candidate|692|(16,)|var|uint32
call_690 = relay.TupleGetItem(func_493_call(relay.reshape(var_691.astype('float64'), [12,]), relay.reshape(var_692.astype('uint32'), [16,]), ), 1)
call_693 = relay.TupleGetItem(func_496_call(relay.reshape(var_691.astype('float64'), [12,]), relay.reshape(var_692.astype('uint32'), [16,]), ), 1)
func_672_call = mod.get_global_var('func_672')
func_674_call = mutated_mod.get_global_var('func_674')
var_728 = relay.var("var_728", dtype = "int8", shape = ())#candidate|728|()|var|int8
call_727 = relay.TupleGetItem(func_672_call(relay.reshape(var_728.astype('int8'), [])), 0)
call_729 = relay.TupleGetItem(func_674_call(relay.reshape(var_728.astype('int8'), [])), 0)
output = relay.Tuple([call_682,call_684,const_685,call_690,var_691,var_692,call_727,var_728,])
output2 = relay.Tuple([call_683,call_686,const_685,call_693,var_691,var_692,call_729,var_728,])
func_739 = relay.Function([var_691,var_692,var_728,], output)
mod['func_739'] = func_739
mod = relay.transform.InferType()(mod)
mutated_mod['func_739'] = func_739
mutated_mod = relay.transform.InferType()(mutated_mod)
func_739_call = mutated_mod.get_global_var('func_739')
var_741 = relay.var("var_741", dtype = "float64", shape = (12, 1))#candidate|741|(12, 1)|var|float64
var_742 = relay.var("var_742", dtype = "uint32", shape = (16,))#candidate|742|(16,)|var|uint32
var_743 = relay.var("var_743", dtype = "int8", shape = ())#candidate|743|()|var|int8
call_740 = func_739_call(var_741,var_742,var_743,)
output = call_740
func_744 = relay.Function([var_741,var_742,var_743,], output)
mutated_mod['func_744'] = func_744
mutated_mod = relay.transform.InferType()(mutated_mod)
var_751 = relay.var("var_751", dtype = "uint8", shape = ())#candidate|751|()|var|uint8
var_752 = relay.var("var_752", dtype = "uint8", shape = (13, 4))#candidate|752|(13, 4)|var|uint8
bop_753 = relay.left_shift(var_751.astype('uint8'), var_752.astype('uint8')) # shape=(13, 4)
func_597_call = mod.get_global_var('func_597')
func_600_call = mutated_mod.get_global_var('func_600')
const_758 = relay.const([-8.731642,-7.167593], dtype = "float64")#candidate|758|(2,)|const|float64
call_757 = relay.TupleGetItem(func_597_call(relay.reshape(const_758.astype('float64'), [2,])), 1)
call_759 = relay.TupleGetItem(func_600_call(relay.reshape(const_758.astype('float64'), [2,])), 1)
var_760 = relay.var("var_760", dtype = "uint8", shape = (13, 4))#candidate|760|(13, 4)|var|uint8
bop_761 = relay.bitwise_xor(bop_753.astype('uint8'), relay.reshape(var_760.astype('uint8'), relay.shape_of(bop_753))) # shape=(13, 4)
bop_767 = relay.logical_xor(var_751.astype('int8'), var_760.astype('int8')) # shape=(13, 4)
func_557_call = mod.get_global_var('func_557')
func_560_call = mutated_mod.get_global_var('func_560')
call_770 = relay.TupleGetItem(func_557_call(relay.reshape(call_757.astype('float64'), [2,])), 0)
call_771 = relay.TupleGetItem(func_560_call(relay.reshape(call_757.astype('float64'), [2,])), 0)
output = relay.Tuple([call_757,const_758,bop_761,bop_767,call_770,])
output2 = relay.Tuple([call_759,const_758,bop_761,bop_767,call_771,])
func_772 = relay.Function([var_751,var_752,var_760,], output)
mod['func_772'] = func_772
mod = relay.transform.InferType()(mod)
var_773 = relay.var("var_773", dtype = "uint8", shape = ())#candidate|773|()|var|uint8
var_774 = relay.var("var_774", dtype = "uint8", shape = (13, 4))#candidate|774|(13, 4)|var|uint8
var_775 = relay.var("var_775", dtype = "uint8", shape = (13, 4))#candidate|775|(13, 4)|var|uint8
output = func_772(var_773,var_774,var_775,)
func_776 = relay.Function([var_773,var_774,var_775,], output)
mutated_mod['func_776'] = func_776
mutated_mod = relay.transform.InferType()(mutated_mod)
func_314_call = mod.get_global_var('func_314')
func_316_call = mutated_mod.get_global_var('func_316')
call_786 = relay.TupleGetItem(func_314_call(), 0)
call_787 = relay.TupleGetItem(func_316_call(), 0)
output = relay.Tuple([call_786,])
output2 = relay.Tuple([call_787,])
func_789 = relay.Function([], output)
mod['func_789'] = func_789
mod = relay.transform.InferType()(mod)
mutated_mod['func_789'] = func_789
mutated_mod = relay.transform.InferType()(mutated_mod)
func_789_call = mutated_mod.get_global_var('func_789')
call_790 = func_789_call()
output = call_790
func_791 = relay.Function([], output)
mutated_mod['func_791'] = func_791
mutated_mod = relay.transform.InferType()(mutated_mod)
var_805 = relay.var("var_805", dtype = "int64", shape = (12, 9))#candidate|805|(12, 9)|var|int64
var_806 = relay.var("var_806", dtype = "int64", shape = (12, 9))#candidate|806|(12, 9)|var|int64
bop_807 = relay.minimum(var_805.astype('int64'), relay.reshape(var_806.astype('int64'), relay.shape_of(var_805))) # shape=(12, 9)
output = bop_807
output2 = bop_807
func_811 = relay.Function([var_805,var_806,], output)
mod['func_811'] = func_811
mod = relay.transform.InferType()(mod)
var_812 = relay.var("var_812", dtype = "int64", shape = (12, 9))#candidate|812|(12, 9)|var|int64
var_813 = relay.var("var_813", dtype = "int64", shape = (12, 9))#candidate|813|(12, 9)|var|int64
output = func_811(var_812,var_813,)
func_814 = relay.Function([var_812,var_813,], output)
mutated_mod['func_814'] = func_814
mutated_mod = relay.transform.InferType()(mutated_mod)
var_818 = relay.var("var_818", dtype = "float32", shape = (8, 9))#candidate|818|(8, 9)|var|float32
uop_819 = relay.atan(var_818.astype('float32')) # shape=(8, 9)
func_71_call = mod.get_global_var('func_71')
func_75_call = mutated_mod.get_global_var('func_75')
const_822 = relay.const([[4,5,-1,1,-9,-4,-5,-10,8,10,9,-7,-8,-3,-10,-8],[-7,2,-6,-10,8,-9,-6,4,8,-3,9,7,10,2,5,6],[4,1,-1,-4,-6,1,-4,5,10,1,9,-9,-8,-1,-10,6],[-5,4,8,4,-3,-6,-4,8,-1,3,7,-9,2,9,-6,-8]], dtype = "int8")#candidate|822|(4, 16)|const|int8
call_821 = relay.TupleGetItem(func_71_call(relay.reshape(const_822.astype('int8'), [16, 4]), relay.reshape(const_822.astype('int8'), [16, 4]), ), 1)
call_823 = relay.TupleGetItem(func_75_call(relay.reshape(const_822.astype('int8'), [16, 4]), relay.reshape(const_822.astype('int8'), [16, 4]), ), 1)
func_474_call = mod.get_global_var('func_474')
func_477_call = mutated_mod.get_global_var('func_477')
var_826 = relay.var("var_826", dtype = "float32", shape = (396, 1))#candidate|826|(396, 1)|var|float32
const_827 = relay.const([-9.927052,9.813109,2.222579,2.444291,-8.767329,2.073735,3.599002,-2.503667,-3.261893,5.911930,-0.447227,-9.752048,6.419038,9.344781,3.285062,-8.464988,3.352013,-6.838388,-2.358137,-3.748206,4.678316,2.708284,4.536650,8.060072,0.586637,1.264378,-9.575339,3.595756,-3.853119,0.183104,-0.689261,2.317690,4.590335,-5.227040,0.246591,-2.286586,3.335481,-4.982732,6.743353,-8.779166,-0.116869,-9.757600,-5.966462,-1.658468,8.548035,-8.144771,-9.429987,3.730784,0.162677,9.362337,-5.047242,-3.746103,0.335204,-5.909373,-3.994573,-4.968233,-9.663580,-7.917287,-3.843418,6.334226,-4.317843,3.371827,-0.171395,-6.346989,-0.694222,-1.672409,-5.299481,-6.287298,7.620138,-5.430710,-1.143129,-9.842154,-7.000073,3.327508,-0.466952,-4.215804,-5.254634,3.274879,-6.153533,1.066876,4.837194,9.724420,-7.049689,-5.306334,9.392934,7.148988,1.806451,-8.891315,-2.150470,-0.355765,2.981161,3.669247,-2.294953,-6.401857,5.563176,1.786071,8.579078,-7.164939,8.367702,-3.282293,4.246646,-6.686182,1.655632,-6.607990,-7.496050,-4.031879,3.530447,8.831366,-5.487961,-6.819766,8.697148,4.631895,0.062903,-7.509111,-7.323605,7.895198,7.829242,-9.435172,-4.927273,-3.654023,4.379219,-3.602230,8.029499,-6.484609,8.426149,8.412247,-0.202644,-4.769198,-3.821902,-7.786641,1.199440,-6.301052,5.279801,6.838286,-3.807455,-4.050055,0.452136,-5.657227,1.206886,-7.322746,-6.233307,6.300813,-0.969517,-6.804486,9.367875,4.567846,-2.832680,1.903720,-9.714460,7.953989,-4.763326,-7.719233,8.517953,-4.539359,0.016602,6.596406,-0.178719,9.455030,-3.338707,1.521466,-4.606806,-7.054056,5.023326,2.451579,8.729204,-2.290064,-5.716986,1.724389,4.011546,-9.100598,-6.122108,-5.384744,9.995141,-3.858658,2.301504,6.097181,-5.226707,-8.238911,1.390458,-9.236474,0.212165,-4.952923,3.588130,8.339244,-4.366999,-8.083709,2.794519,2.656615,-2.059620,8.759691,-3.065709,5.368353,-0.685640,-8.567207,2.183327,-9.134509,1.628791,5.991466,-3.810332,7.306603,-7.735311,-1.040620,4.608225,-2.725680,8.360864,3.397034,-4.198256,-7.724377,6.116956,-4.119461,8.670612,6.168853,-3.462162,-7.312313,8.478730,6.617499,8.061541,6.239654,4.448838,-8.232309,4.111325,-6.472840,-8.445568,7.671144,1.113113,9.962819,-2.963655,7.544649,1.548603,-6.560837,-8.766623,-9.815138,2.362119,-7.286371,-6.608449,0.390957,-4.830854,8.251654,-5.615676,-1.948033,2.923040,5.589340,-1.416078,-6.400186,-2.886875,-9.380216,-7.506351,-8.490305,3.060399,-3.771174,2.242480,8.539387,-3.351645,-3.082702,7.949957,-1.001761,-2.443829,1.007092,1.882943,-1.730624,-9.914529,-2.879033,-4.336924,-9.263329,5.947837,0.657724,-3.642170,-9.679433,5.219987,9.104876,-2.217422,1.266141,0.674616,8.998024,-3.577501,-5.234324,2.726571,9.763795,9.808249,5.796684,0.128074,4.088246,8.703948,-4.065666,7.742541,-6.088842,9.931794,-6.450939,-5.454011,2.081020,-9.336251,2.454991,-6.595113,2.554154,-0.324994,-1.519806,8.530459,3.630991,-3.189679,6.819005,7.492390,4.725604,-9.890781,-3.695918,7.079847,1.393139,3.798917,8.286369,-0.425073,-1.966907,3.092596,-8.604474,-9.146860,-8.758575,7.351476,-6.187569,7.706897,5.665245,-5.690283,-2.669135,-6.914393,1.296678,-5.768569,0.648209,1.614458,5.935630,4.727579,8.071828,1.969417,-7.431168,1.748295,8.808518,-6.671292,-4.940298,-7.229703,-2.091278,8.933114,9.343733,5.552509,-5.297690,-9.745730,2.637331,2.987378,9.649239,-9.716525,7.763794,3.197334,-1.111547,3.088770,-9.296207,0.197225,-4.265760,-9.645349,-4.332813,0.594976,-2.884497,-2.276601,8.519027,-7.889600,-4.299922,-0.314346,-2.931309,1.999305,6.012350,-1.161332,-5.779878,-2.724960,-2.768628,-5.273629,-5.897636,1.920255,4.281437,0.715895,9.108830,-8.671599,-1.973987,3.168095,-9.143064,-4.589185,2.741441,2.335598,-1.874575,0.875033,-5.110189,-2.732252,0.815374,6.488451,2.466294,8.904740,-6.776424,2.374091,-0.273292,0.139005,3.069672,-0.495163,-7.656546,-4.447787,9.731018,1.133155,-3.041935,7.823803,7.843118,-1.846789,6.028247,0.068332,4.606703,-7.317086,6.013688,-2.312095,9.293392,-8.656802,-7.000853,8.872521,-9.491914,6.081314,-7.336460,-5.270396,5.823394,-4.330505,-1.980000,-1.715544,-4.880800,2.291065,0.386302,-7.198452,-0.177411,-6.445975,-8.026755,-7.788524,0.978776,1.848706,-7.981397,9.019885,-6.998082,-2.724500,-7.815409,6.288606,-9.997688,-7.937356,-8.055277,-2.977049,2.073015,-4.389191,-3.676120,-7.128648,-3.513652,6.177665,9.977633,5.240308,8.502906,-9.513237,-8.679049,-0.215061,-6.764839,-5.985292,-6.372727,3.650625,-7.878758,1.460494,6.946713,-9.620751,8.720630,-5.058830,-3.138889,-3.731787,-3.129134,8.710767,-3.718695,0.764550,1.006059,-0.188683,-3.260270,0.708107,2.896873,-9.043022,4.949017,-3.362358,-3.134852,6.863250,-6.740878,-9.909845,9.806014,-6.766793,-0.329832,-8.639873,0.206318,8.854205,1.980544,-2.308865,7.142920,2.289342,1.869577,6.337213,-3.840645,3.883600,-7.921918,5.102827,6.057766,-3.989016,6.715144,3.856129,8.654930,4.218416,-2.143445], dtype = "float64")#candidate|827|(504,)|const|float64
call_825 = relay.TupleGetItem(func_474_call(relay.reshape(var_826.astype('float32'), [9, 4, 11]), relay.reshape(const_827.astype('float64'), [9, 4, 14]), ), 0)
call_828 = relay.TupleGetItem(func_477_call(relay.reshape(var_826.astype('float32'), [9, 4, 11]), relay.reshape(const_827.astype('float64'), [9, 4, 14]), ), 0)
output = relay.Tuple([uop_819,call_821,const_822,call_825,var_826,const_827,])
output2 = relay.Tuple([uop_819,call_823,const_822,call_828,var_826,const_827,])
func_833 = relay.Function([var_818,var_826,], output)
mod['func_833'] = func_833
mod = relay.transform.InferType()(mod)
mutated_mod['func_833'] = func_833
mutated_mod = relay.transform.InferType()(mutated_mod)
func_833_call = mutated_mod.get_global_var('func_833')
var_835 = relay.var("var_835", dtype = "float32", shape = (8, 9))#candidate|835|(8, 9)|var|float32
var_836 = relay.var("var_836", dtype = "float32", shape = (396, 1))#candidate|836|(396, 1)|var|float32
call_834 = func_833_call(var_835,var_836,)
output = call_834
func_837 = relay.Function([var_835,var_836,], output)
mutated_mod['func_837'] = func_837
mutated_mod = relay.transform.InferType()(mutated_mod)
func_314_call = mod.get_global_var('func_314')
func_316_call = mutated_mod.get_global_var('func_316')
call_842 = relay.TupleGetItem(func_314_call(), 0)
call_843 = relay.TupleGetItem(func_316_call(), 0)
var_857 = relay.var("var_857", dtype = "float32", shape = (9, 4, 7))#candidate|857|(9, 4, 7)|var|float32
bop_858 = relay.floor_divide(call_842.astype('float32'), var_857.astype('float32')) # shape=(9, 4, 7)
bop_861 = relay.floor_divide(call_843.astype('float32'), var_857.astype('float32')) # shape=(9, 4, 7)
output = relay.Tuple([bop_858,])
output2 = relay.Tuple([bop_861,])
func_864 = relay.Function([var_857,], output)
mod['func_864'] = func_864
mod = relay.transform.InferType()(mod)
var_865 = relay.var("var_865", dtype = "float32", shape = (9, 4, 7))#candidate|865|(9, 4, 7)|var|float32
output = func_864(var_865)
func_866 = relay.Function([var_865], output)
mutated_mod['func_866'] = func_866
mutated_mod = relay.transform.InferType()(mutated_mod)
func_789_call = mod.get_global_var('func_789')
func_791_call = mutated_mod.get_global_var('func_791')
call_894 = relay.TupleGetItem(func_789_call(), 0)
call_895 = relay.TupleGetItem(func_791_call(), 0)
uop_896 = relay.atanh(call_894.astype('float32')) # shape=(9, 4, 1)
uop_898 = relay.atanh(call_895.astype('float32')) # shape=(9, 4, 1)
uop_906 = relay.atan(uop_896.astype('float64')) # shape=(9, 4, 1)
uop_908 = relay.atan(uop_898.astype('float64')) # shape=(9, 4, 1)
bop_909 = relay.greater(uop_896.astype('bool'), relay.reshape(call_894.astype('bool'), relay.shape_of(uop_896))) # shape=(9, 4, 1)
bop_912 = relay.greater(uop_898.astype('bool'), relay.reshape(call_895.astype('bool'), relay.shape_of(uop_898))) # shape=(9, 4, 1)
uop_913 = relay.atan(uop_906.astype('float32')) # shape=(9, 4, 1)
uop_915 = relay.atan(uop_908.astype('float32')) # shape=(9, 4, 1)
func_493_call = mod.get_global_var('func_493')
func_496_call = mutated_mod.get_global_var('func_496')
const_919 = relay.const([-1.239310,-8.522266,6.086988,3.614729,5.872280,1.881104,5.690066,-5.679644,3.696855,7.934604,-5.642834,-5.928073], dtype = "float64")#candidate|919|(12,)|const|float64
var_920 = relay.var("var_920", dtype = "uint32", shape = (1, 16))#candidate|920|(1, 16)|var|uint32
call_918 = relay.TupleGetItem(func_493_call(relay.reshape(const_919.astype('float64'), [12,]), relay.reshape(var_920.astype('uint32'), [16,]), ), 3)
call_921 = relay.TupleGetItem(func_496_call(relay.reshape(const_919.astype('float64'), [12,]), relay.reshape(var_920.astype('uint32'), [16,]), ), 3)
bop_923 = relay.maximum(uop_906.astype('int16'), relay.reshape(uop_896.astype('int16'), relay.shape_of(uop_906))) # shape=(9, 4, 1)
bop_926 = relay.maximum(uop_908.astype('int16'), relay.reshape(uop_898.astype('int16'), relay.shape_of(uop_908))) # shape=(9, 4, 1)
output = relay.Tuple([bop_909,uop_913,call_918,const_919,var_920,bop_923,])
output2 = relay.Tuple([bop_912,uop_915,call_921,const_919,var_920,bop_926,])
func_930 = relay.Function([var_920,], output)
mod['func_930'] = func_930
mod = relay.transform.InferType()(mod)
mutated_mod['func_930'] = func_930
mutated_mod = relay.transform.InferType()(mutated_mod)
var_931 = relay.var("var_931", dtype = "uint32", shape = (1, 16))#candidate|931|(1, 16)|var|uint32
func_930_call = mutated_mod.get_global_var('func_930')
call_932 = func_930_call(var_931)
output = call_932
func_933 = relay.Function([var_931], output)
mutated_mod['func_933'] = func_933
mutated_mod = relay.transform.InferType()(mutated_mod)
func_789_call = mod.get_global_var('func_789')
func_791_call = mutated_mod.get_global_var('func_791')
call_944 = relay.TupleGetItem(func_789_call(), 0)
call_945 = relay.TupleGetItem(func_791_call(), 0)
uop_948 = relay.sigmoid(call_944.astype('float32')) # shape=(9, 4, 1)
uop_950 = relay.sigmoid(call_945.astype('float32')) # shape=(9, 4, 1)
bop_955 = relay.logical_and(call_944.astype('bool'), relay.reshape(uop_948.astype('bool'), relay.shape_of(call_944))) # shape=(9, 4, 1)
bop_958 = relay.logical_and(call_945.astype('bool'), relay.reshape(uop_950.astype('bool'), relay.shape_of(call_945))) # shape=(9, 4, 1)
var_966 = relay.var("var_966", dtype = "bool", shape = (9, 4, 4))#candidate|966|(9, 4, 4)|var|bool
bop_967 = relay.logical_xor(bop_955.astype('uint8'), var_966.astype('uint8')) # shape=(9, 4, 4)
bop_970 = relay.logical_xor(bop_958.astype('uint8'), var_966.astype('uint8')) # shape=(9, 4, 4)
func_597_call = mod.get_global_var('func_597')
func_600_call = mutated_mod.get_global_var('func_600')
const_974 = relay.const([6.566091,2.083323], dtype = "float64")#candidate|974|(2,)|const|float64
call_973 = relay.TupleGetItem(func_597_call(relay.reshape(const_974.astype('float64'), [2,])), 3)
call_975 = relay.TupleGetItem(func_600_call(relay.reshape(const_974.astype('float64'), [2,])), 3)
uop_978 = relay.acos(bop_955.astype('float32')) # shape=(9, 4, 1)
uop_980 = relay.acos(bop_958.astype('float32')) # shape=(9, 4, 1)
bop_981 = relay.greater_equal(uop_978.astype('bool'), relay.reshape(bop_955.astype('bool'), relay.shape_of(uop_978))) # shape=(9, 4, 1)
bop_984 = relay.greater_equal(uop_980.astype('bool'), relay.reshape(bop_958.astype('bool'), relay.shape_of(uop_980))) # shape=(9, 4, 1)
uop_992 = relay.log2(bop_967.astype('float64')) # shape=(9, 4, 4)
uop_994 = relay.log2(bop_970.astype('float64')) # shape=(9, 4, 4)
output = relay.Tuple([call_973,const_974,bop_981,uop_992,])
output2 = relay.Tuple([call_975,const_974,bop_984,uop_994,])
func_995 = relay.Function([var_966,], output)
mod['func_995'] = func_995
mod = relay.transform.InferType()(mod)
var_996 = relay.var("var_996", dtype = "bool", shape = (9, 4, 4))#candidate|996|(9, 4, 4)|var|bool
output = func_995(var_996)
func_997 = relay.Function([var_996], output)
mutated_mod['func_997'] = func_997
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1017 = relay.const([[[-9.974133,2.926458],[-8.267396,-8.380834],[-7.579442,7.350437],[0.437633,-2.189797],[4.321690,-8.645640],[3.079645,5.396144],[-3.562787,8.744553],[-4.296385,-4.617355],[2.358816,-6.683540],[1.368516,4.606881],[-5.295067,-7.714008],[-2.619870,9.670646],[-9.943901,9.861378]],[[-4.786616,-3.027805],[0.829069,-7.366798],[-6.003676,-4.370088],[-7.690356,9.101458],[0.133177,8.013990],[-1.206586,-5.424287],[5.970294,-9.417983],[6.816540,9.537885],[0.932159,8.085563],[4.233985,-9.652823],[-0.972467,2.854145],[5.135547,-3.760095],[5.528076,-9.040228]],[[1.511877,-9.247766],[-4.464144,1.804383],[-8.171672,-8.651375],[-4.182901,-0.245337],[7.888574,4.469196],[3.406623,-8.074327],[2.295843,-5.402599],[-7.141524,-7.004116],[0.473644,-9.588418],[-4.149523,8.821362],[-8.200590,-1.391740],[6.946938,5.273400],[6.012446,-4.026773]],[[1.008135,1.240373],[7.367463,-2.157824],[7.172995,-3.996600],[1.370673,-9.085542],[-5.809349,5.462472],[9.740789,1.473746],[-1.166749,1.596560],[1.282248,1.613460],[-0.354423,4.302173],[-1.862570,-5.282743],[1.263958,-6.682415],[2.775496,-2.333186],[-3.030130,2.619075]],[[0.991806,0.452790],[3.271339,1.181801],[-8.733407,0.962266],[5.570966,5.583894],[-2.672151,9.937313],[-1.792156,7.538139],[0.082201,5.190371],[9.234165,0.933462],[0.716132,-4.011412],[-6.811896,0.685946],[-7.274648,0.004682],[1.808217,7.112425],[-6.853120,7.650224]],[[-7.405425,4.046283],[-3.745743,-7.145407],[4.989396,-3.482672],[-1.843085,2.972339],[-3.907390,2.511562],[6.277647,2.710199],[1.883160,8.542728],[3.431146,-6.367902],[6.678338,9.541017],[-4.305178,1.590325],[-3.476685,-5.828781],[-5.630930,5.477399],[-1.423251,-2.025058]],[[4.891668,7.061285],[-8.479709,6.943587],[0.784597,-1.863513],[7.357961,0.420435],[5.199436,8.153206],[-0.293926,-2.821310],[7.681032,-9.671527],[-7.668627,3.397463],[-4.830576,6.416987],[5.325342,-6.265799],[-0.084372,3.710428],[5.723007,-3.718852],[7.530057,-1.408251]],[[8.216098,6.202991],[-0.743552,9.551508],[-8.243522,-3.933601],[2.061218,6.261325],[-4.826414,1.729944],[-1.586801,-8.174076],[4.163897,-5.230372],[-1.738765,3.966700],[2.082220,-9.332278],[-6.690907,2.396013],[3.725300,2.721722],[1.334139,9.211950],[-8.468310,-3.365437]],[[-5.538763,-4.451485],[4.996009,-2.281128],[3.925392,-2.924752],[-3.651644,-3.210959],[-0.299646,0.146419],[-3.384983,-1.421384],[-2.810667,5.530023],[-2.244479,8.556056],[3.149749,0.854508],[6.020694,-4.345830],[5.808552,8.684775],[-2.014885,-9.477874],[1.693118,-3.113522]],[[-8.988619,3.624620],[-9.030399,3.393620],[4.021374,5.187083],[0.059760,6.756033],[5.837875,9.573890],[-7.548366,-3.680285],[2.544800,5.966928],[6.454031,-4.696683],[-2.999858,-1.626499],[-7.418520,-7.757293],[-1.540219,-4.964385],[-3.703343,-6.308730],[-9.134813,-3.403086]],[[3.777705,7.674822],[2.252454,8.604802],[-0.507837,-0.906540],[-1.830302,2.030320],[-6.566227,-5.056487],[-0.897960,-0.012455],[-6.305765,-6.347440],[4.095976,-8.068775],[0.003772,7.953578],[-7.862596,4.975285],[-0.884379,-5.353848],[-3.874542,1.862856],[0.982976,-8.783089]],[[1.081985,-9.439356],[6.305198,-0.342958],[2.924959,4.338077],[-7.239076,6.118637],[-8.333696,7.345655],[-2.474824,6.274109],[2.095621,5.566229],[-5.262324,-5.884632],[8.482838,0.246235],[8.748422,1.033579],[-9.834925,-0.993858],[6.298377,-1.018006],[-0.362882,-8.174344]]], dtype = "float32")#candidate|1017|(12, 13, 2)|const|float32
const_1018 = relay.const([[[-9.789518,-4.662092],[7.540534,0.548739],[-0.022564,-0.215478],[4.416796,5.572426],[-5.908473,-2.839238],[-2.222199,-6.814726],[5.020185,-0.472162],[4.371320,1.997397],[2.387123,6.183191],[-7.762690,-3.987642],[6.464230,-4.324773],[5.302759,7.983964],[-1.683519,-5.992643]],[[3.374676,3.474537],[9.052508,-2.065731],[8.835600,-3.918152],[-2.996636,8.339431],[-1.677558,8.266800],[-3.335741,9.706375],[-0.455149,0.059347],[-5.055159,6.980890],[-1.540461,-0.719094],[0.620668,-8.401980],[-1.686203,5.157585],[-3.892487,5.158773],[-6.360372,3.031968]],[[-2.771721,-7.188383],[4.232710,2.281229],[7.137674,-8.102891],[2.422667,-5.171836],[1.638956,9.454079],[1.351752,-4.489865],[9.153066,1.352361],[5.216602,-4.861415],[1.312075,-4.336741],[-3.021596,-9.903991],[-4.184034,8.191194],[9.025661,-3.563792],[-8.671390,-4.866350]],[[-9.189798,-1.752443],[-2.724732,-3.087712],[-6.106628,8.195900],[5.258526,-5.005735],[-0.643196,-6.407767],[0.526809,-9.337346],[8.099152,-2.919120],[3.161798,4.062518],[3.746846,9.406511],[5.092772,1.843271],[4.241460,4.439640],[-0.578550,-8.356149],[-5.591519,-5.063105]],[[0.566450,-4.738372],[0.925952,-4.629004],[8.935842,7.323168],[2.758594,-0.724560],[-9.060495,-7.738784],[3.503415,-5.251934],[-1.988352,-9.735258],[8.316771,-6.631746],[4.618348,8.448714],[1.280396,9.435390],[-8.886884,-6.840837],[-4.385350,-4.517190],[-9.784517,-9.294374]],[[1.764063,-9.834069],[9.965436,-2.766961],[9.548188,8.662082],[-0.364652,8.927880],[2.780863,-2.820689],[9.438645,1.817046],[-9.352405,-4.251286],[9.693082,-2.366703],[-5.174930,-3.084285],[4.725781,-1.008951],[2.210713,2.950946],[6.921288,-9.735126],[-9.419407,-1.842779]],[[6.726622,3.705216],[4.124760,-9.966333],[-2.846090,6.186162],[9.998188,-5.269618],[5.483874,8.598941],[5.811700,-6.282611],[-2.544460,5.039197],[3.707820,-0.881353],[6.837279,-5.262727],[-0.332048,0.617974],[9.895568,2.101716],[-2.569271,9.966820],[-8.776817,9.010543]],[[-6.676352,-6.571393],[-9.878416,2.732680],[5.718466,-3.109742],[5.210397,-8.658954],[1.323588,-2.744772],[-5.295969,-7.232015],[7.337856,-0.852757],[-6.978824,7.715633],[4.315393,9.802032],[8.716925,-8.001674],[2.530542,-3.168907],[1.317700,-1.296874],[-9.557888,-1.317398]],[[-2.082444,-4.773196],[-2.939168,4.266733],[-4.946868,5.060410],[4.385573,5.970569],[6.931093,5.582478],[-3.334622,8.874112],[-0.338658,-9.692610],[-8.877766,-7.490178],[6.282139,7.985436],[9.601753,7.833468],[3.182557,-4.294090],[4.303207,-3.009190],[9.040569,5.524145]],[[7.007619,3.057999],[-0.156610,-2.988833],[-7.107476,5.378256],[4.951127,-6.983660],[-9.334552,-8.852505],[-8.253405,-2.903884],[-9.168302,9.102941],[-7.785654,3.629440],[9.791873,-4.872082],[2.841683,1.601072],[1.866989,1.857072],[2.953997,7.734891],[1.503497,6.198745]],[[-1.833770,-2.352043],[-6.265567,-4.914824],[-5.522986,-1.968218],[8.392706,-0.424796],[-2.706821,-4.390640],[-7.697128,-7.952892],[2.240503,3.009138],[-4.274656,5.118050],[-8.310303,4.077596],[8.733917,-2.024418],[-8.160097,4.843206],[9.673811,-7.278883],[2.896150,9.376184]],[[-5.290807,5.359012],[4.862966,5.709995],[-2.791945,7.896272],[-6.923503,3.720336],[-3.529831,5.788745],[1.321176,-9.187181],[-2.510393,-2.690467],[-4.182490,-0.140573],[3.912238,-5.789840],[-4.235026,-4.163278],[-4.859812,-7.610470],[3.509563,-0.748453],[4.876764,-3.029099]]], dtype = "float32")#candidate|1018|(12, 13, 2)|const|float32
bop_1019 = relay.divide(const_1017.astype('float32'), relay.reshape(const_1018.astype('float32'), relay.shape_of(const_1017))) # shape=(12, 13, 2)
bop_1027 = relay.less_equal(const_1017.astype('bool'), relay.reshape(const_1018.astype('bool'), relay.shape_of(const_1017))) # shape=(12, 13, 2)
output = relay.Tuple([bop_1019,bop_1027,])
output2 = relay.Tuple([bop_1019,bop_1027,])
func_1032 = relay.Function([], output)
mod['func_1032'] = func_1032
mod = relay.transform.InferType()(mod)
output = func_1032()
func_1033 = relay.Function([], output)
mutated_mod['func_1033'] = func_1033
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1065 = relay.var("var_1065", dtype = "uint32", shape = (11,))#candidate|1065|(11,)|var|uint32
var_1066 = relay.var("var_1066", dtype = "uint32", shape = (11,))#candidate|1066|(11,)|var|uint32
bop_1067 = relay.bitwise_or(var_1065.astype('uint32'), relay.reshape(var_1066.astype('uint32'), relay.shape_of(var_1065))) # shape=(11,)
uop_1070 = relay.asin(var_1066.astype('float64')) # shape=(11,)
output = relay.Tuple([bop_1067,uop_1070,])
output2 = relay.Tuple([bop_1067,uop_1070,])
func_1073 = relay.Function([var_1065,var_1066,], output)
mod['func_1073'] = func_1073
mod = relay.transform.InferType()(mod)
var_1074 = relay.var("var_1074", dtype = "uint32", shape = (11,))#candidate|1074|(11,)|var|uint32
var_1075 = relay.var("var_1075", dtype = "uint32", shape = (11,))#candidate|1075|(11,)|var|uint32
output = func_1073(var_1074,var_1075,)
func_1076 = relay.Function([var_1074,var_1075,], output)
mutated_mod['func_1076'] = func_1076
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1096 = relay.const([[6.569100,-0.335592,-3.226561,8.057965,-3.850403,-9.097467],[8.270298,-4.040047,-5.397116,-1.456116,6.735132,7.770303],[0.710866,2.614087,-4.356561,9.146261,8.999027,0.812587],[2.262382,4.009606,3.998372,-2.125164,7.422228,-9.053222],[5.844858,-2.428284,1.036428,-3.963142,2.313853,-4.709550],[-1.817742,-4.025534,-4.303812,7.099076,-1.030147,3.191131],[1.794150,-2.989539,-1.673350,-5.536998,3.218602,-5.254730],[1.574412,5.103277,5.204299,-3.037404,-0.834857,5.330523]], dtype = "float32")#candidate|1096|(8, 6)|const|float32
uop_1097 = relay.acos(const_1096.astype('float32')) # shape=(8, 6)
uop_1102 = relay.log10(uop_1097.astype('float32')) # shape=(8, 6)
output = relay.Tuple([uop_1102,])
output2 = relay.Tuple([uop_1102,])
func_1108 = relay.Function([], output)
mod['func_1108'] = func_1108
mod = relay.transform.InferType()(mod)
output = func_1108()
func_1109 = relay.Function([], output)
mutated_mod['func_1109'] = func_1109
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1117 = relay.var("var_1117", dtype = "float64", shape = (4, 16, 10))#candidate|1117|(4, 16, 10)|var|float64
uop_1118 = relay.atanh(var_1117.astype('float64')) # shape=(4, 16, 10)
const_1121 = relay.const([[[5.129643,3.902192,8.128486,0.631273,-9.551935,-0.320487,-8.977821,-0.015944,-6.625509,8.695164],[1.377917,-7.874207,8.087951,0.847766,0.305148,-0.049340,-8.391067,-1.367451,-3.782655,-1.107618],[-7.228515,7.896217,9.625680,2.208918,-7.180392,4.575565,-4.212343,7.833605,3.082313,9.694314],[8.635680,7.997574,0.288634,-2.827145,-7.769274,5.739575,0.084860,5.801389,-2.361106,9.552252],[4.254111,-7.626795,-5.392096,7.654246,-0.912478,2.042500,4.580393,2.559325,-0.862993,-9.487896],[-6.855743,-0.284169,4.400197,-3.671077,-3.435084,-5.341482,1.800119,3.639693,-4.103391,6.023148],[4.404011,6.389830,-6.686802,3.812246,-1.718897,7.595873,5.391328,-9.855419,1.810487,-9.059358],[-4.136028,9.200569,0.788259,-2.781317,0.269616,-5.443761,-9.498610,2.887331,9.056092,7.532324],[2.622611,-9.024729,-8.960966,8.632608,-7.998911,9.401979,-6.698817,-4.361583,0.937798,-4.293731],[-2.176629,2.963558,-9.598451,5.026180,-6.021333,8.212150,2.500654,-8.536127,-2.826937,7.595962],[-6.461246,-1.130871,1.390912,-8.109442,1.802855,-5.337606,-5.458398,-8.365127,-6.658393,-4.680999],[-3.140871,-1.926147,7.586076,-4.141340,-4.691245,-3.715385,-4.490584,-3.908881,9.346954,6.528149],[6.818089,5.038329,5.661396,-4.308270,7.391117,-4.206599,-5.331479,-4.414617,-2.351143,3.727210],[-7.692779,6.786743,-0.800255,2.138478,-3.690379,-4.104692,-5.363840,6.242022,8.234674,6.650014],[-0.607388,9.712797,0.180438,4.161585,-0.441722,-3.670421,-8.958092,8.302271,0.983609,9.883861],[0.362258,2.125859,4.067102,-7.894206,-7.265090,0.932496,5.594818,7.072057,-4.732547,-6.972831]],[[-6.350464,-1.923730,-6.876280,8.663426,-7.467667,-1.583163,7.278022,5.694856,-2.887851,6.119642],[1.154591,-2.652222,0.220404,9.009053,-3.193949,-2.168306,1.450463,9.082640,8.579017,0.163003],[7.731862,9.864137,-0.519074,2.787854,-0.773964,2.538202,-0.957210,6.551532,-4.939864,6.551839],[-5.192078,2.732304,-5.627752,-3.493667,6.881751,9.963271,7.002880,9.113064,0.613556,-8.664189],[-4.545534,-4.046227,3.671139,-1.576129,-8.681868,-2.567691,-6.327345,-2.003647,3.331256,-2.822055],[4.169509,5.292124,-7.157834,-3.351229,-3.512346,-3.422497,-8.734527,0.877109,-6.021065,7.319396],[6.748099,6.750406,0.495793,3.422506,-1.511449,-4.676766,3.450500,-1.272373,-5.198622,-4.717677],[-9.212862,9.389767,9.399652,-5.368347,8.414546,-0.243860,-3.120077,-7.496521,-3.572040,-4.340153],[0.450955,-1.176931,-9.398983,-3.277988,8.050353,1.483298,7.075614,-3.307554,7.283028,-1.888798],[2.483824,-0.325930,8.369534,2.700270,-7.278089,-0.899371,-5.629293,-2.105395,-3.804459,-7.692491],[9.126163,0.292902,-1.644774,3.087925,-8.300342,-4.861063,4.140998,6.112397,2.460304,-7.578568],[-3.252734,3.371595,8.760692,-0.252786,4.004291,7.970365,-8.381539,5.347923,-0.676805,0.122145],[6.542055,-8.519983,-9.636606,1.227760,4.164471,1.518195,-4.990563,-2.910582,-1.491346,6.887881],[1.074385,9.559412,3.280806,6.278309,4.715268,4.444172,-8.269855,1.320989,4.765660,0.892087],[-5.257323,1.624618,7.760422,-6.065094,-0.179616,1.834419,-2.124675,8.982366,-6.097272,-6.605837],[5.650408,-8.392333,0.375305,-2.547222,-7.717015,0.676385,-7.783547,6.333654,0.319992,-3.924262]],[[-9.997609,8.527422,-0.738601,-9.600082,3.476207,-3.727965,-7.886140,3.257501,7.621832,3.254270],[6.735050,-5.437537,9.783750,2.802867,-0.913902,-3.497423,-6.407482,3.592187,-6.761707,6.645187],[-4.841762,4.401677,-1.372873,8.030655,-0.692751,8.295782,4.072933,-8.980699,2.158418,8.102611],[4.416931,-1.340228,-3.452030,-1.120654,-5.998377,-4.905134,6.136700,7.423233,-1.695367,2.255073],[3.843984,-1.539895,2.811276,1.724130,5.606298,-2.748846,-5.516128,-1.430398,-2.553733,-6.148454],[-2.646592,-7.562853,-3.655319,0.957375,-7.275375,6.947700,6.573739,-6.867781,-8.606564,1.149367],[1.296382,-0.560280,9.470339,-9.356211,7.148074,-9.343755,6.913893,-1.852151,-7.246915,9.870708],[-6.392187,6.416731,-9.843541,-6.159491,8.039407,-0.369210,9.880739,1.285742,6.570771,1.914385],[-2.327737,-5.728248,-2.721784,4.910318,-8.859529,7.521459,3.625084,-7.652541,-4.758994,9.894808],[1.221140,-0.259817,-8.302219,7.624950,6.030814,4.667901,-6.054424,-1.811309,-8.529228,1.907270],[-8.133281,5.050079,1.585069,4.105687,2.318344,-2.756551,4.463038,5.773876,-4.003046,0.427916],[8.629111,0.382601,6.302788,-5.708886,0.677965,-0.162955,-3.355768,1.559959,-4.582766,-0.036420],[7.494949,-8.077085,0.255096,4.928653,-5.700638,-9.713699,0.431508,9.263803,4.115719,-5.738473],[2.500593,-0.001544,-4.366424,-7.604158,-3.546549,-7.256173,8.591318,-1.025398,2.675108,-0.293565],[7.258343,6.405993,-3.609340,3.768485,0.348774,-3.249631,-3.648249,-4.820716,-9.751400,5.998966],[6.080688,7.894456,-6.943040,-1.782116,3.212813,7.160527,1.605750,6.134064,9.710679,6.284285]],[[-6.611201,-1.881076,-5.610889,6.206853,2.531363,8.866574,5.329088,6.095114,5.126497,4.106943],[-7.763921,5.485860,6.975204,4.525037,4.980459,9.156958,-7.858184,-5.848218,1.599418,3.074324],[6.370943,8.701239,5.091668,-5.327094,-6.939554,0.442747,2.208606,-9.087912,6.739162,7.947004],[-4.575578,-0.299478,-8.061273,8.974946,1.830201,-8.672570,-1.039803,0.092439,-6.452159,-5.361439],[1.170449,3.112432,-1.644305,7.715392,8.739408,-6.763900,-6.064439,-3.200226,3.220409,-7.691094],[8.775903,2.642183,-9.993767,8.354158,0.350976,-0.089445,-3.322828,3.980791,9.241848,-3.998221],[4.372693,7.588134,-5.684583,9.359144,0.172168,-5.940713,-1.393711,-4.532030,-2.533802,-7.246717],[7.363465,-4.590694,-6.647467,-3.695922,-8.613795,1.920745,5.892628,7.168630,-1.949866,-6.157126],[-6.886560,-9.644945,-9.848711,-8.146451,2.703410,-8.197484,6.812527,-7.001874,-0.285539,7.287462],[0.953307,7.237850,4.253679,-4.070139,5.807633,-8.166212,5.454423,-8.381926,0.348211,-6.226457],[4.877969,-3.120371,5.982304,1.949145,3.693648,0.891942,5.150860,-7.321584,5.116274,-0.771431],[9.825879,1.825941,-6.172253,-3.789346,-9.256819,-2.911943,-9.536775,-4.826948,2.479238,0.030780],[-7.856429,1.288164,6.151307,7.227923,5.180128,5.001585,-5.334085,4.882596,2.686984,-5.204646],[3.649754,9.524874,1.152903,6.103070,4.451003,-6.360905,7.185551,0.246326,5.100753,-4.485670],[1.347439,6.854203,-1.407946,-9.381044,-3.253778,1.267126,6.662491,7.741842,-2.847923,4.550055],[-0.402484,1.901232,9.319446,-1.285701,7.224945,8.379613,3.884768,-4.072163,-4.459217,9.966001]]], dtype = "float64")#candidate|1121|(4, 16, 10)|const|float64
bop_1122 = relay.divide(uop_1118.astype('float32'), relay.reshape(const_1121.astype('float32'), relay.shape_of(uop_1118))) # shape=(4, 16, 10)
output = bop_1122
output2 = bop_1122
func_1127 = relay.Function([var_1117,], output)
mod['func_1127'] = func_1127
mod = relay.transform.InferType()(mod)
mutated_mod['func_1127'] = func_1127
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1128 = relay.var("var_1128", dtype = "float64", shape = (4, 16, 10))#candidate|1128|(4, 16, 10)|var|float64
func_1127_call = mutated_mod.get_global_var('func_1127')
call_1129 = func_1127_call(var_1128)
output = call_1129
func_1130 = relay.Function([var_1128], output)
mutated_mod['func_1130'] = func_1130
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1160 = relay.var("var_1160", dtype = "int32", shape = (14, 13, 8))#candidate|1160|(14, 13, 8)|var|int32
var_1161 = relay.var("var_1161", dtype = "int32", shape = (14, 13, 8))#candidate|1161|(14, 13, 8)|var|int32
bop_1162 = relay.right_shift(var_1160.astype('int32'), relay.reshape(var_1161.astype('int32'), relay.shape_of(var_1160))) # shape=(14, 13, 8)
func_1032_call = mod.get_global_var('func_1032')
func_1033_call = mutated_mod.get_global_var('func_1033')
call_1166 = relay.TupleGetItem(func_1032_call(), 1)
call_1167 = relay.TupleGetItem(func_1033_call(), 1)
func_597_call = mod.get_global_var('func_597')
func_600_call = mutated_mod.get_global_var('func_600')
const_1182 = relay.const([[-4.652248],[6.015799]], dtype = "float64")#candidate|1182|(2, 1)|const|float64
call_1181 = relay.TupleGetItem(func_597_call(relay.reshape(const_1182.astype('float64'), [2,])), 1)
call_1183 = relay.TupleGetItem(func_600_call(relay.reshape(const_1182.astype('float64'), [2,])), 1)
var_1191 = relay.var("var_1191", dtype = "int32", shape = (14, 13, 8))#candidate|1191|(14, 13, 8)|var|int32
bop_1192 = relay.multiply(var_1160.astype('int8'), relay.reshape(var_1191.astype('int8'), relay.shape_of(var_1160))) # shape=(14, 13, 8)
output = relay.Tuple([bop_1162,call_1166,call_1181,const_1182,bop_1192,])
output2 = relay.Tuple([bop_1162,call_1167,call_1183,const_1182,bop_1192,])
func_1198 = relay.Function([var_1160,var_1161,var_1191,], output)
mod['func_1198'] = func_1198
mod = relay.transform.InferType()(mod)
mutated_mod['func_1198'] = func_1198
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1198_call = mutated_mod.get_global_var('func_1198')
var_1200 = relay.var("var_1200", dtype = "int32", shape = (14, 13, 8))#candidate|1200|(14, 13, 8)|var|int32
var_1201 = relay.var("var_1201", dtype = "int32", shape = (14, 13, 8))#candidate|1201|(14, 13, 8)|var|int32
var_1202 = relay.var("var_1202", dtype = "int32", shape = (14, 13, 8))#candidate|1202|(14, 13, 8)|var|int32
call_1199 = func_1198_call(var_1200,var_1201,var_1202,)
output = call_1199
func_1203 = relay.Function([var_1200,var_1201,var_1202,], output)
mutated_mod['func_1203'] = func_1203
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1207 = relay.const([[[4.860905,2.222425,-9.783392,-2.936669,-5.922628,-7.019132,-1.283083,4.268093,-6.872039,-9.924243,4.546889],[-9.444714,6.225382,2.466420,-3.887147,9.055394,4.089111,8.522973,0.728249,1.704106,7.051953,6.877913]],[[3.117407,8.881426,8.839023,7.373755,2.044734,2.517645,-6.043454,9.712518,5.746487,4.629350,-2.583229],[2.178973,-4.687978,6.344179,-2.479318,0.463395,3.121205,2.988518,7.494796,-0.243053,-5.295694,8.237678]],[[-8.406745,5.065024,-7.404359,-6.283008,-2.899308,-3.402802,-5.446206,-6.523175,-9.808917,-7.187674,8.732234],[0.692336,3.161270,5.686849,7.266144,-6.468296,7.466762,-7.794961,-4.857322,-7.660506,-2.127193,-8.958205]],[[-5.813984,8.260849,-2.865134,7.142945,9.594326,-5.307404,-6.163095,2.990193,2.573346,1.288037,2.909614],[-5.689083,-2.216417,-8.365244,1.131064,1.424073,-9.659491,8.295418,1.441270,-1.385012,8.754172,4.896869]],[[-7.407566,7.116336,-0.028922,-4.113682,-6.207950,-1.020473,-7.875682,0.940871,0.898784,5.517207,4.264528],[-4.669914,0.380835,-9.943980,-9.093362,-4.349302,-0.226799,-1.811184,-5.879293,-9.891290,0.979662,2.225738]]], dtype = "float64")#candidate|1207|(5, 2, 11)|const|float64
var_1208 = relay.var("var_1208", dtype = "float64", shape = (5, 2, 11))#candidate|1208|(5, 2, 11)|var|float64
bop_1209 = relay.floor_mod(const_1207.astype('float64'), relay.reshape(var_1208.astype('float64'), relay.shape_of(const_1207))) # shape=(5, 2, 11)
uop_1214 = relay.sin(bop_1209.astype('float32')) # shape=(5, 2, 11)
bop_1216 = relay.logical_xor(uop_1214.astype('int16'), relay.reshape(const_1207.astype('int16'), relay.shape_of(uop_1214))) # shape=(5, 2, 11)
bop_1221 = relay.right_shift(bop_1216.astype('int32'), relay.reshape(bop_1209.astype('int32'), relay.shape_of(bop_1216))) # shape=(5, 2, 11)
uop_1224 = relay.asin(bop_1221.astype('float64')) # shape=(5, 2, 11)
bop_1227 = relay.logical_and(var_1208.astype('bool'), relay.reshape(uop_1214.astype('bool'), relay.shape_of(var_1208))) # shape=(5, 2, 11)
uop_1231 = relay.atanh(uop_1224.astype('float64')) # shape=(5, 2, 11)
uop_1234 = relay.rsqrt(uop_1224.astype('float32')) # shape=(5, 2, 11)
var_1236 = relay.var("var_1236", dtype = "float32", shape = (5, 2, 11))#candidate|1236|(5, 2, 11)|var|float32
bop_1237 = relay.less_equal(uop_1234.astype('bool'), relay.reshape(var_1236.astype('bool'), relay.shape_of(uop_1234))) # shape=(5, 2, 11)
bop_1241 = relay.not_equal(uop_1231.astype('bool'), relay.reshape(uop_1234.astype('bool'), relay.shape_of(uop_1231))) # shape=(5, 2, 11)
var_1245 = relay.var("var_1245", dtype = "float64", shape = (5, 2, 11))#candidate|1245|(5, 2, 11)|var|float64
bop_1246 = relay.bitwise_and(uop_1224.astype('uint32'), relay.reshape(var_1245.astype('uint32'), relay.shape_of(uop_1224))) # shape=(5, 2, 11)
output = relay.Tuple([bop_1227,bop_1237,bop_1241,bop_1246,])
output2 = relay.Tuple([bop_1227,bop_1237,bop_1241,bop_1246,])
func_1251 = relay.Function([var_1208,var_1236,var_1245,], output)
mod['func_1251'] = func_1251
mod = relay.transform.InferType()(mod)
var_1252 = relay.var("var_1252", dtype = "float64", shape = (5, 2, 11))#candidate|1252|(5, 2, 11)|var|float64
var_1253 = relay.var("var_1253", dtype = "float32", shape = (5, 2, 11))#candidate|1253|(5, 2, 11)|var|float32
var_1254 = relay.var("var_1254", dtype = "float64", shape = (5, 2, 11))#candidate|1254|(5, 2, 11)|var|float64
output = func_1251(var_1252,var_1253,var_1254,)
func_1255 = relay.Function([var_1252,var_1253,var_1254,], output)
mutated_mod['func_1255'] = func_1255
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1289 = relay.var("var_1289", dtype = "float64", shape = (15, 7))#candidate|1289|(15, 7)|var|float64
uop_1290 = relay.erf(var_1289.astype('float64')) # shape=(15, 7)
bop_1292 = relay.right_shift(var_1289.astype('int64'), relay.reshape(uop_1290.astype('int64'), relay.shape_of(var_1289))) # shape=(15, 7)
uop_1296 = relay.log10(uop_1290.astype('float64')) # shape=(15, 7)
uop_1301 = relay.sin(uop_1296.astype('float32')) # shape=(15, 7)
output = relay.Tuple([bop_1292,uop_1301,])
output2 = relay.Tuple([bop_1292,uop_1301,])
func_1311 = relay.Function([var_1289,], output)
mod['func_1311'] = func_1311
mod = relay.transform.InferType()(mod)
mutated_mod['func_1311'] = func_1311
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1312 = relay.var("var_1312", dtype = "float64", shape = (15, 7))#candidate|1312|(15, 7)|var|float64
func_1311_call = mutated_mod.get_global_var('func_1311')
call_1313 = func_1311_call(var_1312)
output = call_1313
func_1314 = relay.Function([var_1312], output)
mutated_mod['func_1314'] = func_1314
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1108_call = mod.get_global_var('func_1108')
func_1109_call = mutated_mod.get_global_var('func_1109')
call_1318 = relay.TupleGetItem(func_1108_call(), 0)
call_1319 = relay.TupleGetItem(func_1109_call(), 0)
func_772_call = mod.get_global_var('func_772')
func_776_call = mutated_mod.get_global_var('func_776')
const_1321 = relay.const(-7, dtype = "uint8")#candidate|1321|()|const|uint8
const_1322 = relay.const([10,-9,2,5,7,9,1,-9,-3,9,7,1,7,3,5,-2,-6,7,5,7,-7,9,4,-1,10,-6,1,-2,7,10,3,-8,-7,6,8,-3,4,5,10,2,-5,-7,-8,-3,-4,9,-9,8,-4,5,10,9], dtype = "uint8")#candidate|1322|(52,)|const|uint8
call_1320 = relay.TupleGetItem(func_772_call(relay.reshape(const_1321.astype('uint8'), []), relay.reshape(const_1322.astype('uint8'), [13, 4]), relay.reshape(const_1322.astype('uint8'), [13, 4]), ), 1)
call_1323 = relay.TupleGetItem(func_776_call(relay.reshape(const_1321.astype('uint8'), []), relay.reshape(const_1322.astype('uint8'), [13, 4]), relay.reshape(const_1322.astype('uint8'), [13, 4]), ), 1)
output = relay.Tuple([call_1318,call_1320,const_1321,const_1322,])
output2 = relay.Tuple([call_1319,call_1323,const_1321,const_1322,])
func_1330 = relay.Function([], output)
mod['func_1330'] = func_1330
mod = relay.transform.InferType()(mod)
output = func_1330()
func_1331 = relay.Function([], output)
mutated_mod['func_1331'] = func_1331
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1335 = relay.var("var_1335", dtype = "int32", shape = (8, 10))#candidate|1335|(8, 10)|var|int32
const_1336 = relay.const([[9,6,10,5,10,6,-9,7,-5,9],[7,-7,5,6,-10,-1,8,9,3,10],[5,7,4,-5,6,-2,9,-1,3,6],[-4,4,-4,10,6,-1,5,-6,-2,10],[8,1,1,-10,1,-8,2,-9,-5,-6],[-10,-4,4,3,-3,-4,-3,8,10,10],[5,8,-10,-6,5,-7,-8,9,-3,10],[3,8,-10,4,4,1,2,7,-6,6]], dtype = "int32")#candidate|1336|(8, 10)|const|int32
bop_1337 = relay.left_shift(var_1335.astype('int32'), relay.reshape(const_1336.astype('int32'), relay.shape_of(var_1335))) # shape=(8, 10)
bop_1347 = relay.less(const_1336.astype('bool'), relay.reshape(bop_1337.astype('bool'), relay.shape_of(const_1336))) # shape=(8, 10)
output = relay.Tuple([bop_1347,])
output2 = relay.Tuple([bop_1347,])
F = relay.Function([var_1335,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1335,], output2)
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
	relay.transform.ToANormalForm(),
	relay.transform.ToGraphNormalForm(),
	relay.transform.SimplifyInference(),
	relay.transform.ToBasicBlockNormalForm(),
	relay.transform.FuseOps(3),
	relay.transform.DefuseOps(),
	relay.transform.SimplifyExpr(),
	relay.transform.InferType(),
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
input_1335= np.array([[2,-5,4,7,-2,10,-3,2,-5,-1],[-5,-2,6,5,-2,1,5,-8,-7,-6],[2,-4,8,-10,-2,-1,1,-3,-7,-4],[-1,5,-3,4,2,1,-6,10,1,-9],[1,10,-5,1,-3,-3,4,-10,-2,6],[-6,7,-2,-8,-9,7,-8,3,10,-7],[1,-9,9,1,-4,5,-6,4,9,2],[-10,7,-6,5,-6,-10,4,3,9,-5]], dtype='int32')
module1.set_input('var_1335', input_1335)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1335, )
res3 = intrp3.evaluate()(input_1335, )
res4 = intrp4.evaluate()(input_1335, )
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
module5.set_input('var_1335', input_1335)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1335, )
res7 = intrp7.evaluate()(input_1335, )
res8 = intrp8.evaluate()(input_1335, )
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
module9.set_input('var_1335', input_1335)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1335, )
res11 = intrp11.evaluate()(input_1335, )
res12 = intrp12.evaluate()(input_1335, )
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
module13.set_input('var_1335', input_1335)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1335, )
res15 = intrp15.evaluate()(input_1335, )
res16 = intrp16.evaluate()(input_1335, )
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
module17.set_input('var_1335', input_1335)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1335, )
res19 = intrp19.evaluate()(input_1335, )
res20 = intrp20.evaluate()(input_1335, )
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
module21.set_input('var_1335', input_1335)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1335, )
res23 = intrp23.evaluate()(input_1335, )
res24 = intrp24.evaluate()(input_1335, )
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

'''22: TVMFuncCall
21: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
20: tvm::transform::Pass::operator()(tvm::IRModule) const
19: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
18: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
17: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
16: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
15: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
14: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
13: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
12: _ZNSt17_Function_handlerIFSt10sha
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::TupleNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''