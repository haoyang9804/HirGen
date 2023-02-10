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
const_26 = relay.const(-8.092647, dtype = "float64")#candidate|26|()|const|float64
var_27 = relay.var("var_27", dtype = "float64", shape = (9, 9))#candidate|27|(9, 9)|var|float64
bop_28 = relay.greater_equal(const_26.astype('bool'), var_27.astype('bool')) # shape=(9, 9)
uop_31 = relay.cosh(bop_28.astype('float64')) # shape=(9, 9)
var_36 = relay.var("var_36", dtype = "float64", shape = (9, 9))#candidate|36|(9, 9)|var|float64
bop_37 = relay.floor_divide(uop_31.astype('float64'), relay.reshape(var_36.astype('float64'), relay.shape_of(uop_31))) # shape=(9, 9)
var_40 = relay.var("var_40", dtype = "float64", shape = (9, 9))#candidate|40|(9, 9)|var|float64
bop_41 = relay.bitwise_xor(uop_31.astype('int64'), relay.reshape(var_40.astype('int64'), relay.shape_of(uop_31))) # shape=(9, 9)
uop_47 = relay.acosh(bop_41.astype('float32')) # shape=(9, 9)
uop_53 = relay.log10(uop_47.astype('float32')) # shape=(9, 9)
bop_55 = relay.bitwise_and(uop_53.astype('int8'), relay.reshape(var_36.astype('int8'), relay.shape_of(uop_53))) # shape=(9, 9)
var_58 = relay.var("var_58", dtype = "float32", shape = (9, 9))#candidate|58|(9, 9)|var|float32
bop_59 = relay.minimum(uop_47.astype('uint64'), relay.reshape(var_58.astype('uint64'), relay.shape_of(uop_47))) # shape=(9, 9)
bop_62 = relay.bitwise_or(bop_55.astype('uint32'), relay.reshape(bop_41.astype('uint32'), relay.shape_of(bop_55))) # shape=(9, 9)
uop_66 = relay.sigmoid(uop_53.astype('float32')) # shape=(9, 9)
var_69 = relay.var("var_69", dtype = "float32", shape = (9, 9))#candidate|69|(9, 9)|var|float32
bop_70 = relay.subtract(uop_47.astype('float32'), relay.reshape(var_69.astype('float32'), relay.shape_of(uop_47))) # shape=(9, 9)
bop_73 = relay.less(uop_66.astype('bool'), relay.reshape(uop_31.astype('bool'), relay.shape_of(uop_66))) # shape=(9, 9)
uop_76 = relay.exp(uop_66.astype('float64')) # shape=(9, 9)
uop_79 = relay.log2(uop_76.astype('float32')) # shape=(9, 9)
bop_83 = relay.maximum(uop_79.astype('float64'), relay.reshape(bop_55.astype('float64'), relay.shape_of(uop_79))) # shape=(9, 9)
bop_89 = relay.left_shift(uop_79.astype('uint16'), relay.reshape(var_40.astype('uint16'), relay.shape_of(uop_79))) # shape=(9, 9)
uop_96 = relay.sqrt(uop_76.astype('float32')) # shape=(9, 9)
var_102 = relay.var("var_102", dtype = "float32", shape = (9, 9))#candidate|102|(9, 9)|var|float32
bop_103 = relay.not_equal(uop_96.astype('bool'), relay.reshape(var_102.astype('bool'), relay.shape_of(uop_96))) # shape=(9, 9)
uop_109 = relay.sinh(bop_62.astype('float32')) # shape=(9, 9)
bop_111 = relay.logical_and(uop_66.astype('bool'), relay.reshape(var_69.astype('bool'), relay.shape_of(uop_66))) # shape=(9, 9)
output = relay.Tuple([bop_37,bop_59,bop_70,bop_73,bop_83,bop_89,bop_103,uop_109,bop_111,])
output2 = relay.Tuple([bop_37,bop_59,bop_70,bop_73,bop_83,bop_89,bop_103,uop_109,bop_111,])
func_120 = relay.Function([var_27,var_36,var_40,var_58,var_69,var_102,], output)
mod['func_120'] = func_120
mod = relay.transform.InferType()(mod)
mutated_mod['func_120'] = func_120
mutated_mod = relay.transform.InferType()(mutated_mod)
func_120_call = mutated_mod.get_global_var('func_120')
var_122 = relay.var("var_122", dtype = "float64", shape = (9, 9))#candidate|122|(9, 9)|var|float64
var_123 = relay.var("var_123", dtype = "float64", shape = (9, 9))#candidate|123|(9, 9)|var|float64
var_124 = relay.var("var_124", dtype = "float64", shape = (9, 9))#candidate|124|(9, 9)|var|float64
var_125 = relay.var("var_125", dtype = "float32", shape = (9, 9))#candidate|125|(9, 9)|var|float32
var_126 = relay.var("var_126", dtype = "float32", shape = (9, 9))#candidate|126|(9, 9)|var|float32
var_127 = relay.var("var_127", dtype = "float32", shape = (9, 9))#candidate|127|(9, 9)|var|float32
call_121 = func_120_call(var_122,var_123,var_124,var_125,var_126,var_127,)
output = call_121
func_128 = relay.Function([var_122,var_123,var_124,var_125,var_126,var_127,], output)
mutated_mod['func_128'] = func_128
mutated_mod = relay.transform.InferType()(mutated_mod)
const_177 = relay.const([[-6.162339,7.114975,7.656640,8.609955],[-8.560811,0.356286,2.728243,1.712767],[-9.499508,-7.540001,9.765174,-3.421343],[8.437189,-1.021861,6.146629,7.222046]], dtype = "float64")#candidate|177|(4, 4)|const|float64
uop_178 = relay.cosh(const_177.astype('float64')) # shape=(4, 4)
var_181 = relay.var("var_181", dtype = "float64", shape = (4, 4))#candidate|181|(4, 4)|var|float64
bop_182 = relay.not_equal(uop_178.astype('bool'), relay.reshape(var_181.astype('bool'), relay.shape_of(uop_178))) # shape=(4, 4)
bop_187 = relay.greater(bop_182.astype('bool'), relay.reshape(uop_178.astype('bool'), relay.shape_of(bop_182))) # shape=(4, 4)
output = relay.Tuple([bop_187,])
output2 = relay.Tuple([bop_187,])
func_190 = relay.Function([var_181,], output)
mod['func_190'] = func_190
mod = relay.transform.InferType()(mod)
var_191 = relay.var("var_191", dtype = "float64", shape = (4, 4))#candidate|191|(4, 4)|var|float64
output = func_190(var_191)
func_192 = relay.Function([var_191], output)
mutated_mod['func_192'] = func_192
mutated_mod = relay.transform.InferType()(mutated_mod)
var_200 = relay.var("var_200", dtype = "float64", shape = (12, 4))#candidate|200|(12, 4)|var|float64
uop_201 = relay.atan(var_200.astype('float64')) # shape=(12, 4)
bop_203 = relay.subtract(uop_201.astype('int32'), relay.reshape(var_200.astype('int32'), relay.shape_of(uop_201))) # shape=(12, 4)
bop_206 = relay.left_shift(uop_201.astype('int64'), relay.reshape(var_200.astype('int64'), relay.shape_of(uop_201))) # shape=(12, 4)
func_190_call = mod.get_global_var('func_190')
func_192_call = mutated_mod.get_global_var('func_192')
const_210 = relay.const([[-2.637675,3.683067],[8.480500,0.821895],[7.121725,-0.855838],[-3.421673,-3.441374],[3.578391,9.552986],[-4.987582,-5.289706],[-6.702480,-4.520640],[8.504087,-8.590919]], dtype = "float64")#candidate|210|(8, 2)|const|float64
call_209 = relay.TupleGetItem(func_190_call(relay.reshape(const_210.astype('float64'), [4, 4])), 0)
call_211 = relay.TupleGetItem(func_192_call(relay.reshape(const_210.astype('float64'), [4, 4])), 0)
output = relay.Tuple([bop_203,bop_206,call_209,const_210,])
output2 = relay.Tuple([bop_203,bop_206,call_211,const_210,])
func_212 = relay.Function([var_200,], output)
mod['func_212'] = func_212
mod = relay.transform.InferType()(mod)
var_213 = relay.var("var_213", dtype = "float64", shape = (12, 4))#candidate|213|(12, 4)|var|float64
output = func_212(var_213)
func_214 = relay.Function([var_213], output)
mutated_mod['func_214'] = func_214
mutated_mod = relay.transform.InferType()(mutated_mod)
var_218 = relay.var("var_218", dtype = "int8", shape = (15,))#candidate|218|(15,)|var|int8
const_219 = relay.const([1,-8,-2,4,2,6,6,8,7,-1,10,-6,6,4,5], dtype = "int8")#candidate|219|(15,)|const|int8
bop_220 = relay.greater(var_218.astype('bool'), relay.reshape(const_219.astype('bool'), relay.shape_of(var_218))) # shape=(15,)
var_225 = relay.var("var_225", dtype = "int8", shape = (15,))#candidate|225|(15,)|var|int8
bop_226 = relay.multiply(const_219.astype('float32'), relay.reshape(var_225.astype('float32'), relay.shape_of(const_219))) # shape=(15,)
uop_229 = relay.acos(var_218.astype('float64')) # shape=(15,)
output = relay.Tuple([bop_220,bop_226,uop_229,])
output2 = relay.Tuple([bop_220,bop_226,uop_229,])
func_234 = relay.Function([var_218,var_225,], output)
mod['func_234'] = func_234
mod = relay.transform.InferType()(mod)
var_235 = relay.var("var_235", dtype = "int8", shape = (15,))#candidate|235|(15,)|var|int8
var_236 = relay.var("var_236", dtype = "int8", shape = (15,))#candidate|236|(15,)|var|int8
output = func_234(var_235,var_236,)
func_237 = relay.Function([var_235,var_236,], output)
mutated_mod['func_237'] = func_237
mutated_mod = relay.transform.InferType()(mutated_mod)
var_256 = relay.var("var_256", dtype = "float32", shape = (5, 12, 16))#candidate|256|(5, 12, 16)|var|float32
uop_257 = relay.tan(var_256.astype('float32')) # shape=(5, 12, 16)
uop_259 = relay.sigmoid(uop_257.astype('float32')) # shape=(5, 12, 16)
uop_264 = relay.acos(uop_259.astype('float64')) # shape=(5, 12, 16)
bop_267 = relay.bitwise_xor(uop_264.astype('uint8'), relay.reshape(uop_259.astype('uint8'), relay.shape_of(uop_264))) # shape=(5, 12, 16)
var_273 = relay.var("var_273", dtype = "float32", shape = (5, 12, 16))#candidate|273|(5, 12, 16)|var|float32
bop_274 = relay.logical_or(uop_259.astype('bool'), relay.reshape(var_273.astype('bool'), relay.shape_of(uop_259))) # shape=(5, 12, 16)
bop_279 = relay.greater_equal(uop_264.astype('bool'), relay.reshape(uop_257.astype('bool'), relay.shape_of(uop_264))) # shape=(5, 12, 16)
bop_282 = relay.greater_equal(bop_274.astype('bool'), relay.reshape(uop_259.astype('bool'), relay.shape_of(bop_274))) # shape=(5, 12, 16)
bop_291 = relay.floor_divide(bop_279.astype('float32'), relay.reshape(bop_274.astype('float32'), relay.shape_of(bop_279))) # shape=(5, 12, 16)
output = relay.Tuple([bop_267,bop_282,bop_291,])
output2 = relay.Tuple([bop_267,bop_282,bop_291,])
func_294 = relay.Function([var_256,var_273,], output)
mod['func_294'] = func_294
mod = relay.transform.InferType()(mod)
var_295 = relay.var("var_295", dtype = "float32", shape = (5, 12, 16))#candidate|295|(5, 12, 16)|var|float32
var_296 = relay.var("var_296", dtype = "float32", shape = (5, 12, 16))#candidate|296|(5, 12, 16)|var|float32
output = func_294(var_295,var_296,)
func_297 = relay.Function([var_295,var_296,], output)
mutated_mod['func_297'] = func_297
mutated_mod = relay.transform.InferType()(mutated_mod)
var_322 = relay.var("var_322", dtype = "float32", shape = (6, 1))#candidate|322|(6, 1)|var|float32
var_323 = relay.var("var_323", dtype = "float32", shape = (6, 4))#candidate|323|(6, 4)|var|float32
bop_324 = relay.subtract(var_322.astype('float32'), var_323.astype('float32')) # shape=(6, 4)
uop_327 = relay.acos(var_323.astype('float32')) # shape=(6, 4)
bop_329 = relay.bitwise_or(uop_327.astype('uint32'), relay.reshape(bop_324.astype('uint32'), relay.shape_of(uop_327))) # shape=(6, 4)
uop_338 = relay.sin(var_323.astype('float32')) # shape=(6, 4)
bop_340 = relay.minimum(bop_324.astype('int32'), var_322.astype('int32')) # shape=(6, 4)
uop_345 = relay.log2(uop_327.astype('float32')) # shape=(6, 4)
bop_348 = relay.not_equal(uop_345.astype('bool'), relay.reshape(var_323.astype('bool'), relay.shape_of(uop_345))) # shape=(6, 4)
bop_354 = relay.left_shift(uop_345.astype('int32'), relay.reshape(bop_329.astype('int32'), relay.shape_of(uop_345))) # shape=(6, 4)
const_357 = relay.const([[-1.992393,1.726161,-0.551572,5.885546],[-8.423176,-7.240354,8.830583,4.955012],[1.796523,7.294385,3.077693,7.468525],[3.755455,-4.618313,-6.824987,-1.258114],[-7.144474,9.378294,-6.211094,-3.466529],[9.927740,-8.254316,0.237738,8.902587]], dtype = "float32")#candidate|357|(6, 4)|const|float32
bop_358 = relay.multiply(uop_338.astype('int8'), relay.reshape(const_357.astype('int8'), relay.shape_of(uop_338))) # shape=(6, 4)
output = relay.Tuple([bop_340,bop_348,bop_354,bop_358,])
output2 = relay.Tuple([bop_340,bop_348,bop_354,bop_358,])
func_365 = relay.Function([var_322,var_323,], output)
mod['func_365'] = func_365
mod = relay.transform.InferType()(mod)
mutated_mod['func_365'] = func_365
mutated_mod = relay.transform.InferType()(mutated_mod)
func_365_call = mutated_mod.get_global_var('func_365')
var_367 = relay.var("var_367", dtype = "float32", shape = (6, 1))#candidate|367|(6, 1)|var|float32
var_368 = relay.var("var_368", dtype = "float32", shape = (6, 4))#candidate|368|(6, 4)|var|float32
call_366 = func_365_call(var_367,var_368,)
output = call_366
func_369 = relay.Function([var_367,var_368,], output)
mutated_mod['func_369'] = func_369
mutated_mod = relay.transform.InferType()(mutated_mod)
const_388 = relay.const([[True,False,False,True,True],[False,False,True,True,True],[True,True,False,True,False],[True,False,False,False,False],[False,True,True,True,False],[False,False,False,True,False],[False,True,False,False,False],[True,True,False,False,False]], dtype = "bool")#candidate|388|(8, 5)|const|bool
var_389 = relay.var("var_389", dtype = "bool", shape = (8, 5))#candidate|389|(8, 5)|var|bool
bop_390 = relay.logical_and(const_388.astype('bool'), relay.reshape(var_389.astype('bool'), relay.shape_of(const_388))) # shape=(8, 5)
func_365_call = mod.get_global_var('func_365')
func_369_call = mutated_mod.get_global_var('func_369')
var_394 = relay.var("var_394", dtype = "float32", shape = (6,))#candidate|394|(6,)|var|float32
var_395 = relay.var("var_395", dtype = "float32", shape = (6, 4))#candidate|395|(6, 4)|var|float32
call_393 = relay.TupleGetItem(func_365_call(relay.reshape(var_394.astype('float32'), [6, 1]), relay.reshape(var_395.astype('float32'), [6, 4]), ), 3)
call_396 = relay.TupleGetItem(func_369_call(relay.reshape(var_394.astype('float32'), [6, 1]), relay.reshape(var_395.astype('float32'), [6, 4]), ), 3)
bop_403 = relay.right_shift(var_395.astype('int64'), relay.reshape(call_393.astype('int64'), relay.shape_of(var_395))) # shape=(6, 4)
bop_406 = relay.right_shift(var_395.astype('int64'), relay.reshape(call_396.astype('int64'), relay.shape_of(var_395))) # shape=(6, 4)
output = relay.Tuple([bop_390,var_394,bop_403,])
output2 = relay.Tuple([bop_390,var_394,bop_406,])
func_408 = relay.Function([var_389,var_394,var_395,], output)
mod['func_408'] = func_408
mod = relay.transform.InferType()(mod)
mutated_mod['func_408'] = func_408
mutated_mod = relay.transform.InferType()(mutated_mod)
func_408_call = mutated_mod.get_global_var('func_408')
var_410 = relay.var("var_410", dtype = "bool", shape = (8, 5))#candidate|410|(8, 5)|var|bool
var_411 = relay.var("var_411", dtype = "float32", shape = (6,))#candidate|411|(6,)|var|float32
var_412 = relay.var("var_412", dtype = "float32", shape = (6, 4))#candidate|412|(6, 4)|var|float32
call_409 = func_408_call(var_410,var_411,var_412,)
output = call_409
func_413 = relay.Function([var_410,var_411,var_412,], output)
mutated_mod['func_413'] = func_413
mutated_mod = relay.transform.InferType()(mutated_mod)
var_452 = relay.var("var_452", dtype = "float64", shape = (15,))#candidate|452|(15,)|var|float64
uop_453 = relay.atanh(var_452.astype('float64')) # shape=(15,)
uop_458 = relay.log(uop_453.astype('float64')) # shape=(15,)
var_477 = relay.var("var_477", dtype = "float64", shape = (15,))#candidate|477|(15,)|var|float64
bop_478 = relay.greater(uop_453.astype('bool'), relay.reshape(var_477.astype('bool'), relay.shape_of(uop_453))) # shape=(15,)
func_408_call = mod.get_global_var('func_408')
func_413_call = mutated_mod.get_global_var('func_413')
const_490 = relay.const([True,True,False,True,True,False,True,False,True,False,True,False,True,True,False,True,False,False,False,True,True,True,False,False,True,True,True,False,False,True,False,False,False,False,True,False,True,True,False,False], dtype = "bool")#candidate|490|(40,)|const|bool
var_491 = relay.var("var_491", dtype = "float32", shape = (6,))#candidate|491|(6,)|var|float32
const_492 = relay.const([1.749030,-3.752333,1.776257,8.528612,-8.419696,7.405516,8.870019,3.990938,8.714466,3.415533,-3.995585,-7.227705,-7.444266,8.921926,8.567478,8.713939,-6.792688,7.232598,9.868522,9.809743,-4.715075,-6.114682,7.014013,8.153641], dtype = "float32")#candidate|492|(24,)|const|float32
call_489 = relay.TupleGetItem(func_408_call(relay.reshape(const_490.astype('bool'), [8, 5]), relay.reshape(var_491.astype('float32'), [6,]), relay.reshape(const_492.astype('float32'), [6, 4]), ), 1)
call_493 = relay.TupleGetItem(func_413_call(relay.reshape(const_490.astype('bool'), [8, 5]), relay.reshape(var_491.astype('float32'), [6,]), relay.reshape(const_492.astype('float32'), [6, 4]), ), 1)
bop_495 = relay.left_shift(uop_453.astype('int8'), relay.reshape(bop_478.astype('int8'), relay.shape_of(uop_453))) # shape=(15,)
uop_498 = relay.sinh(uop_458.astype('float64')) # shape=(15,)
output = relay.Tuple([call_489,const_490,var_491,const_492,bop_495,uop_498,])
output2 = relay.Tuple([call_493,const_490,var_491,const_492,bop_495,uop_498,])
func_502 = relay.Function([var_452,var_477,var_491,], output)
mod['func_502'] = func_502
mod = relay.transform.InferType()(mod)
mutated_mod['func_502'] = func_502
mutated_mod = relay.transform.InferType()(mutated_mod)
func_502_call = mutated_mod.get_global_var('func_502')
var_504 = relay.var("var_504", dtype = "float64", shape = (15,))#candidate|504|(15,)|var|float64
var_505 = relay.var("var_505", dtype = "float64", shape = (15,))#candidate|505|(15,)|var|float64
var_506 = relay.var("var_506", dtype = "float32", shape = (6,))#candidate|506|(6,)|var|float32
call_503 = func_502_call(var_504,var_505,var_506,)
output = call_503
func_507 = relay.Function([var_504,var_505,var_506,], output)
mutated_mod['func_507'] = func_507
mutated_mod = relay.transform.InferType()(mutated_mod)
var_531 = relay.var("var_531", dtype = "int64", shape = ())#candidate|531|()|var|int64
const_532 = relay.const([[[1,-2,-7,6,-9],[-8,-1,-10,9,-3],[-5,5,-8,-2,-4],[-6,-6,1,-3,7]],[[-9,-7,-3,-7,8],[-9,-10,-8,5,-7],[3,-4,-6,-2,-2],[-10,-6,-3,-3,-1]],[[-2,-5,-5,1,5],[9,-5,-8,-7,-7],[1,-2,-4,8,-7],[10,9,4,-2,-4]]], dtype = "int64")#candidate|532|(3, 4, 5)|const|int64
bop_533 = relay.subtract(var_531.astype('int64'), const_532.astype('int64')) # shape=(3, 4, 5)
func_365_call = mod.get_global_var('func_365')
func_369_call = mutated_mod.get_global_var('func_369')
const_537 = relay.const([[5.595365,1.229351],[4.069512,6.896467],[8.068858,6.239145]], dtype = "float32")#candidate|537|(3, 2)|const|float32
var_538 = relay.var("var_538", dtype = "float32", shape = (24,))#candidate|538|(24,)|var|float32
call_536 = relay.TupleGetItem(func_365_call(relay.reshape(const_537.astype('float32'), [6, 1]), relay.reshape(var_538.astype('float32'), [6, 4]), ), 2)
call_539 = relay.TupleGetItem(func_369_call(relay.reshape(const_537.astype('float32'), [6, 1]), relay.reshape(var_538.astype('float32'), [6, 4]), ), 2)
output = relay.Tuple([bop_533,call_536,const_537,var_538,])
output2 = relay.Tuple([bop_533,call_539,const_537,var_538,])
func_540 = relay.Function([var_531,var_538,], output)
mod['func_540'] = func_540
mod = relay.transform.InferType()(mod)
mutated_mod['func_540'] = func_540
mutated_mod = relay.transform.InferType()(mutated_mod)
func_540_call = mutated_mod.get_global_var('func_540')
var_542 = relay.var("var_542", dtype = "int64", shape = ())#candidate|542|()|var|int64
var_543 = relay.var("var_543", dtype = "float32", shape = (24,))#candidate|543|(24,)|var|float32
call_541 = func_540_call(var_542,var_543,)
output = call_541
func_544 = relay.Function([var_542,var_543,], output)
mutated_mod['func_544'] = func_544
mutated_mod = relay.transform.InferType()(mutated_mod)
const_587 = relay.const([[-3,-1,-5,4,5,-9,7,-4,5,9,-1,-7,-4,9],[7,9,4,9,-5,7,7,8,3,7,9,3,6,-7],[8,-2,-3,5,1,9,8,-5,3,1,7,-3,3,8],[-3,-2,3,6,-4,-5,6,10,7,1,1,5,9,6],[-3,3,-9,9,-8,-8,-7,-7,5,8,-5,4,-2,-3],[7,5,4,-6,-3,-3,-9,3,-3,10,8,-10,1,-8],[-2,-6,-10,9,-8,-7,1,2,-10,-7,-5,-1,-6,-7],[-6,2,6,7,10,10,-4,-9,-6,4,-7,-10,-9,-9],[-3,-4,9,-9,3,-1,-5,-10,-9,-3,-3,-2,9,7],[5,2,10,9,3,6,-5,8,-9,8,-1,-2,7,7],[6,-3,-7,10,-3,1,-8,-2,4,-2,2,1,-8,-8],[-9,-10,-10,-7,-10,-6,-8,8,4,9,-2,10,-1,-7],[-8,-6,1,-3,1,-10,6,1,-3,-2,-9,-4,1,4],[7,-3,-8,1,4,-2,3,-5,8,-5,-7,7,-7,8],[4,-2,-5,5,3,-5,-9,4,-9,-3,5,7,4,-7]], dtype = "uint64")#candidate|587|(15, 14)|const|uint64
var_588 = relay.var("var_588", dtype = "uint64", shape = (15, 14))#candidate|588|(15, 14)|var|uint64
bop_589 = relay.multiply(const_587.astype('uint64'), relay.reshape(var_588.astype('uint64'), relay.shape_of(const_587))) # shape=(15, 14)
output = relay.Tuple([bop_589,])
output2 = relay.Tuple([bop_589,])
func_600 = relay.Function([var_588,], output)
mod['func_600'] = func_600
mod = relay.transform.InferType()(mod)
mutated_mod['func_600'] = func_600
mutated_mod = relay.transform.InferType()(mutated_mod)
var_601 = relay.var("var_601", dtype = "uint64", shape = (15, 14))#candidate|601|(15, 14)|var|uint64
func_600_call = mutated_mod.get_global_var('func_600')
call_602 = func_600_call(var_601)
output = call_602
func_603 = relay.Function([var_601], output)
mutated_mod['func_603'] = func_603
mutated_mod = relay.transform.InferType()(mutated_mod)
const_658 = relay.const([[[-2.624476,-9.964665,4.491629,-2.068417,9.972987,-2.080278,2.222467,-3.042339,-7.424462,-0.457329,7.870569,1.856403,5.462930,-8.316284,0.120156,-2.328780]],[[-1.030133,-6.356216,4.358043,5.977191,-6.671092,-1.660969,-2.564125,9.252062,-0.042201,-8.989364,3.010199,-6.893241,-0.211423,-2.075707,-5.386037,-0.335414]],[[6.600438,1.197735,8.121462,-7.643717,-1.776907,-5.259443,1.838702,-3.346694,5.941438,5.418061,-2.760012,-2.525120,0.831727,-4.070604,5.540328,-5.267635]],[[3.468036,0.012503,-2.716917,9.919148,-2.340741,0.734669,9.918835,7.663706,-7.486716,0.664836,8.254432,-5.756609,-4.850053,-0.680072,1.990664,5.705082]],[[-6.399341,5.019494,-9.433059,-5.440341,-6.754217,-3.384530,6.003496,5.876981,-3.272535,-7.438203,-0.958392,-0.991537,2.725697,-3.586516,-6.187456,-8.103173]],[[-6.218468,2.916266,-7.887238,4.412432,-6.579586,-7.113215,3.288218,4.823961,4.692327,4.462645,-6.086815,-5.949894,4.691317,-1.678120,-7.471290,-2.766263]],[[3.270549,9.586473,9.886376,-9.670076,-1.029853,4.671087,-4.598573,-5.350792,-5.970472,4.454300,6.944568,1.914887,-2.222167,-2.625169,-6.246341,5.194197]],[[5.661421,6.336351,3.481430,-5.608770,-9.354091,-7.629441,-7.071589,-6.975638,-0.757566,-8.196605,-7.036568,-6.248756,4.994399,1.022165,-5.280952,8.494952]],[[-9.880825,-9.827321,7.170936,9.471827,8.506614,-6.588448,-8.833982,-6.043893,4.766303,4.881417,-6.070130,-9.490678,-7.044399,5.482621,5.056711,-8.725594]]], dtype = "float64")#candidate|658|(9, 1, 16)|const|float64
var_659 = relay.var("var_659", dtype = "float64", shape = (9, 3, 16))#candidate|659|(9, 3, 16)|var|float64
bop_660 = relay.mod(const_658.astype('float64'), var_659.astype('float64')) # shape=(9, 3, 16)
bop_667 = relay.bitwise_or(var_659.astype('int64'), relay.reshape(bop_660.astype('int64'), relay.shape_of(var_659))) # shape=(9, 3, 16)
bop_670 = relay.divide(bop_667.astype('float32'), relay.reshape(var_659.astype('float32'), relay.shape_of(bop_667))) # shape=(9, 3, 16)
func_294_call = mod.get_global_var('func_294')
func_297_call = mutated_mod.get_global_var('func_297')
const_674 = relay.const([2.145457,0.747519,4.255823,2.087791,-8.972173,9.714822,3.641217,7.159098,-9.228921,2.137555,2.729884,-4.463989,6.549360,6.336884,7.365984,6.819210,-9.148645,-6.303467,-2.871186,2.622963,8.207824,7.046527,-3.545614,-9.761523,8.389398,-5.712890,-3.048279,9.195863,1.617838,2.495827,7.858270,3.656840,8.279734,0.444302,2.037325,-4.158826,6.776075,-6.323370,-1.964773,9.518009,-5.062274,-5.661267,-9.310547,1.972790,4.180843,-7.071476,9.177156,-9.834767,6.061002,-6.127497,-3.127501,4.590610,-5.094326,4.738019,7.569207,-5.996712,-6.597138,-9.711061,3.712108,5.532049,-4.170725,1.916077,2.423136,-5.278982,-8.103085,-3.679671,2.145983,-5.230507,2.616472,-8.888675,-8.428113,-6.763033,-4.267167,6.443887,-1.303696,-7.904348,-9.889571,-0.579780,7.141779,-0.840475,1.430160,-2.313730,4.641590,6.258112,-1.513848,3.104215,-6.562481,9.348864,-4.301035,-6.917728,5.535227,-7.957821,1.909604,-4.453436,9.069533,-8.603108,9.235884,-7.249743,5.870066,6.365329,8.774090,1.149508,-0.008458,-1.801042,2.274553,-5.659811,2.425301,8.849184,7.945729,0.682724,-5.166104,-9.774306,-7.976950,-8.456358,-6.030754,3.255024,-3.835157,0.211598,-4.657976,-7.643052,9.485994,-6.196685,8.567671,0.575729,9.017177,1.879830,-7.373974,6.977557,-0.977184,-0.869087,4.274560,3.480516,-1.899859,7.389215,-9.050342,4.718839,-1.976816,1.723279,7.317171,-6.963555,4.213788,-6.773180,6.230124,6.993780,-8.993382,-9.888418,0.908454,1.751265,2.363844,-7.896709,-8.855143,8.645350,-3.863212,-5.800483,1.395582,-2.546327,2.584047,-4.916452,0.443862,-1.463572,4.460956,6.890704,-9.245388,1.409545,0.917709,-0.560583,3.535562,-4.320170,-2.108534,2.700487,-4.431897,-7.068437,-3.659837,-9.978015,-2.130338,3.788031,-7.614283,4.307532,-8.194010,-4.019383,-9.610115,5.560232,-7.249164,9.637038,8.502948,-5.367507,5.548421,8.994654,9.888800,7.435978,-3.635003,3.219859,0.877609,-0.653911,1.217249,-0.755343,2.101218,6.650294,9.253141,-0.755843,9.171030,-7.775115,0.096693,6.841760,-8.491424,9.226672,3.323537,2.997906,0.115902,-2.810642,-9.338042,-8.269942,7.690482,0.850578,7.477153,4.624811,-8.214054,0.893438,4.100049,-9.789375,0.169234,-7.599314,-2.377408,5.790960,9.026954,7.007022,-6.031360,4.412186,1.550328,3.329952,-0.085980,-8.241033,-2.411712,7.175940,-8.685565,0.875701,2.559898,-0.702196,0.674039,7.601058,-9.480738,-6.138667,-5.353801,-7.398342,-8.710029,-9.955499,-8.594969,4.131762,-9.857442,-7.724166,-8.311970,-4.890426,1.927146,-5.248714,1.168036,-1.093654,-7.611935,-4.512114,-8.202910,-3.445425,-1.191153,-7.167026,9.946375,-4.977209,-0.570576,-0.856120,-0.357036,-2.831202,-5.101978,4.038499,-1.320432,0.255196,-4.774447,5.091895,-0.510877,-9.596848,8.104822,7.304619,-5.816017,-3.989631,-5.258370,-9.896186,-1.091253,-9.551685,9.744429,-8.358098,6.366599,-7.874705,7.044515,4.908121,6.103534,3.974103,-2.864968,-1.655515,-0.103859,8.385771,4.825781,7.208745,-1.322677,-9.051378,7.324980,1.782886,-6.541049,6.761187,8.320920,3.237432,-3.427961,7.694888,1.018991,3.036926,1.695722,-5.612286,-0.231694,-6.228611,-5.256349,-4.381955,6.050192,-9.743573,8.384632,4.037801,-2.152021,-1.025016,8.004490,7.968783,3.475458,-8.881463,4.915139,8.606938,1.185108,-2.678328,6.869700,2.717131,-0.352912,-6.345607,9.141836,6.193295,7.932304,-0.568498,5.910929,6.407357,1.805476,4.241180,6.484049,-3.231796,-6.743098,-7.874940,-4.131933,-3.152756,0.253338,-3.625656,9.129300,6.840842,4.826981,2.308758,3.373321,5.899196,2.955541,-8.897233,-2.519374,7.435430,4.678306,-3.798821,2.357944,9.365700,1.551535,-1.402031,5.861685,8.296277,9.694674,2.263548,-5.015769,9.479831,-4.981924,-1.129573,3.913748,8.826742,9.955123,-5.302654,1.335759,8.196410,6.710943,9.984874,7.248258,-1.427111,7.347913,-6.247268,-3.366915,0.793326,2.675515,1.368721,2.343363,4.746011,-6.436191,-1.887182,7.780693,4.971855,3.687883,7.968853,7.854441,1.495831,-1.767957,-6.897711,-6.157691,2.825677,8.979949,-1.410234,7.250496,3.109134,0.270020,9.824396,-8.366930,3.337920,4.777945,-3.161308,-3.357513,-3.577529,8.411365,-5.229152,-9.962252,5.068608,-9.374900,2.485220,-2.250486,-3.352272,-2.858779,-2.447050,-4.870140,-4.174989,0.045942,-0.183628,-0.163035,-4.324228,-7.981698,9.747903,-5.856169,-0.120324,-3.162087,-1.639697,7.008808,-4.742599,5.871062,-0.991269,5.207447,0.749978,8.356661,-3.543582,-2.288142,-1.739092,8.802435,-2.193302,-5.206645,0.984978,-2.655381,0.679436,-7.963630,1.253628,6.710724,9.444176,-2.405267,3.242371,-7.233561,2.456378,-1.798803,7.352413,-2.610183,1.069714,3.050942,3.898853,-9.105300,4.121991,5.057288,-6.900856,-8.621158,8.801905,2.327275,-7.619186,-9.284101,-2.717299,3.241478,5.969154,2.616421,5.056907,2.400645,-2.711648,4.360808,-4.608030,-9.684403,9.783346,-9.192801,9.241306,0.524426,8.874314,-7.206160,-1.180224,-4.671547,3.949393,-2.780768,-6.640989,3.766082,-1.560670,-9.304778,2.289930,2.070100,6.501857,5.696190,8.424214,-9.037758,6.644447,5.552249,6.425430,3.426835,7.035965,1.310623,9.468557,7.564109,6.322583,-4.049047,-5.757360,-6.627603,2.034584,3.628538,-6.322439,6.360832,-4.516928,-9.371321,-3.686120,-2.124926,1.397740,9.788658,-8.083052,-1.469316,-7.619131,5.271987,0.531129,-2.154785,0.868479,4.610131,-1.794844,8.296560,0.924810,-6.154669,8.235790,-7.230320,-2.236184,4.899334,-4.753314,-6.877409,-0.101685,8.236356,4.627112,-2.333552,0.332254,5.761226,7.785423,-4.948896,-5.596506,-1.238933,3.778668,-3.070937,5.085520,6.728977,-4.838891,-1.615409,5.053120,-1.563518,-6.238615,2.445166,-3.034362,7.296412,-0.137023,7.109867,-8.514820,-3.228180,-7.522564,0.451784,3.080310,6.925127,-1.138887,-5.530774,7.251875,3.934206,-3.928035,-9.430549,3.711420,-8.643538,7.003025,-0.149686,0.826836,-2.239296,5.238660,3.487592,-6.911087,-0.749204,1.962242,-1.770212,8.136197,-3.637275,-4.661506,1.417958,-3.072790,5.438932,7.984464,2.864155,2.830047,-7.517179,3.058906,7.443437,-7.137263,-4.004677,-8.054503,-9.798532,-9.498461,4.385562,8.729533,-1.196159,-8.025107,-9.349057,9.538095,1.587707,2.239752,-9.981408,4.415849,-3.831707,8.474573,8.253213,7.668474,-4.755985,-1.139237,-4.222065,-8.800277,7.592622,5.713351,-4.749182,3.580310,6.322229,4.738831,9.547286,7.139765,-9.085370,1.775267,-4.870064,5.056892,-4.505095,7.621382,-9.936229,-2.607063,-3.911698,6.897986,-3.957183,-3.494661,1.318496,6.132233,2.395174,-7.212869,-4.957432,6.450752,-2.279323,0.139753,5.791115,9.930421,-6.452091,9.234247,2.005964,9.781741,-8.211121,1.393353,4.877188,-2.484888,3.283824,-8.557160,-7.300114,3.862444,-0.526613,-7.020096,7.340386,3.135685,5.382395,-0.974047,4.552198,-4.646055,-8.330873,8.234971,1.917878,-1.127932,-3.743245,-9.088679,-8.099315,9.155906,-9.865973,-1.489099,-1.244158,1.476276,2.732240,-2.284757,9.033478,-9.476024,3.525156,7.504669,6.691716,7.413487,-0.056557,6.501601,8.602794,5.853570,-8.199580,1.780012,-9.230112,-4.566139,1.311000,-3.221211,6.375222,-1.735164,6.674139,4.576384,4.298829,-8.219306,-5.567701,-5.436071,6.218035,4.525655,-7.170655,-2.093317,6.331763,-4.151838,5.237045,0.753795,1.445228,3.122556,6.061276,1.305301,-0.874456,-6.799707,4.146775,-2.166331,6.773462,-5.003771,4.609266,-6.483179,5.077006,5.223647,2.037194,-0.365565,-1.999891,1.339664,-2.616193,-7.170395,4.902758,-7.831261,5.835774,8.439485,-5.826848,-0.285120,-4.245797,9.409901,1.276174,-5.530800,6.573654,4.743858,5.855139,-1.702896,-8.614317,-9.285603,7.205826,-6.852056,9.626174,2.805552,-0.944193,7.783912,-7.014994,-4.900798,2.583348,4.792993,7.020142,-3.086655,2.129771,2.198745,5.804210,0.546361,-2.199986,-4.638655,4.586386,4.935491,2.315819,-1.525726,8.695453,-5.418545,-7.385746,-0.088138,-9.163034,-5.570450,7.873645,5.100836,9.638007,1.860912,9.956220,0.603117,-4.408923,-7.807958,-1.048109,9.112072,6.822639,-9.845132,5.852694,-6.682232,-3.675782,-9.654677,-1.678471,-6.368613,-9.129004,3.562826,6.146484,7.617239,0.981601,-7.210798,-9.649409,-2.636844,6.586208,7.572133,3.970326,-0.066773,-6.263865,9.474724,0.457312,-4.703356,7.446947,2.838690,5.978285,-9.896620,-3.815677,-2.027375,4.547971,-1.303861,0.702192,5.594095,5.357294,-7.878074,-5.685994,-4.250530,-4.021289,-2.842394,-0.937798,-5.945323,-6.150717,3.828763,4.665646,2.617104,-0.198452,0.124988,-9.561459,-8.684326,7.013393,1.790456,-3.944036,4.833059,2.233838,-8.655174,4.034978,2.849581,-7.730160,-5.482277,-4.396098,-8.018649,1.064453,-8.840943,-8.703990,-8.981188,-3.492904,-1.271539,5.392775,1.839137,7.135438,7.249659,-4.672318,-0.709344,6.227050,5.381815,-8.920310,-3.725792,4.670172,3.403664,2.739375,9.917745,-8.974587,-3.368325,1.991135,-4.929926,8.954598,3.272894,-0.124098,-3.066580,-8.264217,8.444986,2.194191,7.112850,-2.562590,0.258360,-5.248067,1.606626,-4.427245,-4.805305,1.071929,-8.915419,-9.661554,2.521329,3.102945,-3.028122,9.571835,-0.022675,1.596227,7.745962,1.593615,5.220499,-4.369856,-2.998543,-5.622430,-1.210703,3.729323,7.279107,1.062912,-7.256997,-9.321607,-8.527647,-3.776417,8.421926,-6.501520,3.220669,8.479954,5.765983,-8.766194,-9.986618,2.956059,-4.332944,3.027742,-3.162921,-3.344397,1.954046,-2.196573,-0.814493,-2.092829,-4.872719,2.979262,-0.986757,7.582805,-9.947016,2.404278,-8.495232,7.926943,9.731522,-0.306533,8.967277,-5.563919,-0.892867,8.929648,0.336286,-9.424702,-8.085808,-9.950914,-2.348200,2.794440,2.030872], dtype = "float32")#candidate|674|(960,)|const|float32
call_673 = relay.TupleGetItem(func_294_call(relay.reshape(const_674.astype('float32'), [5, 12, 16]), relay.reshape(const_674.astype('float32'), [5, 12, 16]), ), 2)
call_675 = relay.TupleGetItem(func_297_call(relay.reshape(const_674.astype('float32'), [5, 12, 16]), relay.reshape(const_674.astype('float32'), [5, 12, 16]), ), 2)
var_677 = relay.var("var_677", dtype = "float32", shape = (960,))#candidate|677|(960,)|var|float32
bop_678 = relay.floor_mod(const_674.astype('float64'), relay.reshape(var_677.astype('float64'), relay.shape_of(const_674))) # shape=(960,)
uop_687 = relay.sin(bop_670.astype('float64')) # shape=(9, 3, 16)
uop_689 = relay.sqrt(uop_687.astype('float64')) # shape=(9, 3, 16)
bop_691 = relay.equal(uop_687.astype('bool'), const_658.astype('bool')) # shape=(9, 3, 16)
uop_695 = relay.log2(uop_687.astype('float32')) # shape=(9, 3, 16)
bop_697 = relay.floor_divide(uop_689.astype('float64'), relay.reshape(uop_695.astype('float64'), relay.shape_of(uop_689))) # shape=(9, 3, 16)
output = relay.Tuple([call_673,bop_678,bop_691,bop_697,])
output2 = relay.Tuple([call_675,bop_678,bop_691,bop_697,])
func_701 = relay.Function([var_659,var_677,], output)
mod['func_701'] = func_701
mod = relay.transform.InferType()(mod)
var_702 = relay.var("var_702", dtype = "float64", shape = (9, 3, 16))#candidate|702|(9, 3, 16)|var|float64
var_703 = relay.var("var_703", dtype = "float32", shape = (960,))#candidate|703|(960,)|var|float32
output = func_701(var_702,var_703,)
func_704 = relay.Function([var_702,var_703,], output)
mutated_mod['func_704'] = func_704
mutated_mod = relay.transform.InferType()(mutated_mod)
var_741 = relay.var("var_741", dtype = "float64", shape = (12,))#candidate|741|(12,)|var|float64
uop_742 = relay.cos(var_741.astype('float64')) # shape=(12,)
bop_747 = relay.less_equal(uop_742.astype('bool'), relay.reshape(var_741.astype('bool'), relay.shape_of(uop_742))) # shape=(12,)
func_294_call = mod.get_global_var('func_294')
func_297_call = mutated_mod.get_global_var('func_297')
const_751 = relay.const([-2.983576,-7.787760,-7.503728,-1.095211,1.127180,-4.521543,2.123650,8.119033,-1.428182,9.055471,-7.728677,-9.558445,1.526746,-5.273051,5.066066,-7.017945,-2.802587,-3.733339,8.106545,-8.073168,-5.648816,9.721314,5.791813,-3.468073,-0.575789,-0.229211,-0.201668,2.357824,0.581771,-8.028025,-2.884782,-9.659430,5.889527,1.336790,3.617065,-8.828503,-0.844993,-3.140708,-3.251447,2.338099,-9.182255,6.127710,-4.028115,-4.065303,8.550006,-1.663643,0.599742,-2.885663,-3.307315,6.481587,1.640633,7.677354,-1.651152,2.783454,9.755714,0.107512,6.787939,-1.013074,-4.708457,1.111247,4.128661,5.608197,4.724842,-1.535013,9.377134,1.914269,2.478739,8.392403,-5.345250,-7.418159,5.534852,5.802252,5.248934,8.384481,-7.532437,4.733528,-1.944340,9.728686,-8.029333,6.123439,8.485913,3.685706,-1.378913,-0.515798,2.157955,-3.614457,8.116040,5.860199,1.656640,-4.800694,-9.379251,0.065863,-5.390114,-0.407267,-5.391056,-0.034382,-5.416577,-1.051335,-6.294292,-0.985822,-9.245073,3.957384,-7.322515,-3.638770,3.629572,-3.924468,-6.409746,-4.769352,-1.633913,1.669003,1.508570,-3.942492,9.277728,-8.734835,-4.637556,-3.231878,1.166079,-5.236575,-4.897838,1.575102,0.960635,7.155710,2.464451,-3.910313,-1.561862,5.586600,1.388926,-4.083657,-7.512564,4.182046,6.460101,0.597693,6.689949,-5.792820,-1.022929,4.684556,9.464709,6.288325,-2.574486,-0.710118,-4.197034,-7.554177,-4.932126,-3.772180,6.050370,9.360551,3.910881,6.498201,-6.250374,2.554958,9.922324,-0.278149,4.322870,-5.544470,2.174014,0.530194,-5.843368,7.415636,-7.770145,3.841308,6.368513,-2.267523,-5.489639,8.601081,8.664651,7.563083,1.353113,-7.538966,-4.750074,4.488835,6.577999,2.901878,-5.579355,-2.173799,-2.117586,3.863742,-8.217886,8.627070,2.290186,1.975452,9.064026,-4.123356,-9.412089,4.060009,6.269162,-3.085761,9.922431,-4.238739,9.135641,-6.578837,-6.701816,7.570402,1.085736,0.648166,-6.359033,-7.924885,-4.700922,-4.338740,4.176099,9.306085,-2.624678,4.991956,4.544646,-7.199512,7.994464,-9.626078,-2.144826,-7.271276,-8.105945,-0.500959,5.638697,-0.432560,9.158316,-2.199416,4.558004,7.449970,-8.947028,0.866091,-4.525351,-2.183005,7.251433,-2.124249,-1.684086,4.811156,-7.104672,0.508570,-7.513893,0.204374,9.339899,-1.444387,5.328540,-1.388908,7.918572,3.513024,4.663780,6.560732,3.972417,-4.087737,-3.189504,-7.728610,-0.220949,-1.302741,-4.236793,-1.968122,-2.436086,-8.533401,5.144941,0.571785,-7.940270,-1.020918,-9.396995,-3.390062,2.216090,3.004455,6.150184,5.678000,-3.010454,-4.303984,9.319544,-4.386429,-6.839508,-4.397699,7.528693,-9.202906,-4.629489,-3.512161,-2.388247,-7.493093,5.505167,8.654407,-4.325903,-1.027858,4.960036,8.729046,-4.334272,-4.825130,-8.185339,-0.317214,-1.633830,0.911958,6.828282,2.971939,2.207737,-4.009769,9.082829,8.580439,-5.540796,-0.831126,-7.016050,-6.250589,8.994295,1.926791,1.035005,-2.187183,-4.638279,2.925878,9.364904,9.291106,-4.311682,0.842667,-5.274708,2.725895,8.972505,-0.086282,5.928152,-3.233112,2.946476,-4.847386,4.037970,-1.192957,2.109387,7.536404,-0.619390,4.128495,-5.809893,-2.868257,1.045143,0.583059,-4.315796,-0.046872,8.937437,-4.838035,7.688088,7.189293,6.775632,-0.318863,8.995367,7.425002,9.805189,7.346917,-3.309066,-2.722459,5.525802,-4.850627,7.874503,-9.453724,8.632811,5.659036,6.352485,9.725631,-2.710938,8.335595,7.832812,8.419531,5.952837,8.037680,-0.515715,7.498791,8.198212,7.588769,9.549833,5.381971,8.441865,5.772932,-7.271703,0.474585,-1.115349,-8.860624,-7.263829,-4.280046,-3.241006,2.053498,8.584705,4.253247,-8.697661,7.630939,-9.071139,-5.624692,0.915355,-0.994448,-4.954054,-3.811700,1.958607,-4.410971,-0.332204,4.123514,-3.698087,-7.229239,-0.088303,8.247193,-0.625077,-1.303511,2.376809,-6.571196,-2.058027,-6.570529,7.934892,2.423574,-1.109729,-4.117976,3.886632,7.471307,-3.892246,8.169969,3.761206,5.173006,-7.533486,-6.213022,-5.724578,-3.069484,-2.319664,9.087493,-1.579876,-1.005810,-8.519858,0.986565,-7.985053,-3.166555,-8.205149,5.940464,-1.404887,-6.474262,-1.331802,-5.722727,2.423030,1.249807,-4.884824,7.077804,-7.078548,6.466445,-2.723587,-0.703102,8.384254,1.020387,-6.594995,-2.963812,1.995724,2.821362,4.624765,-3.560302,7.419707,-8.405986,-3.783025,-4.495154,3.075782,-1.016513,-2.642440,3.902345,0.165878,-9.783159,4.706890,-3.370436,5.402412,7.280580,0.010773,0.235914,2.004463,-9.059011,-9.560254,-0.286756,-4.883654,6.775874,5.008303,7.369248,-0.495247,4.430228,-1.216131,-0.620865,7.916401,-8.505002,-2.417621,3.016938,-6.727465,8.314740,-8.170525,-4.302137,-2.136700,-8.022376,-2.329501,-5.044646,9.099235,2.523335,9.337696,7.956759,8.081332,-8.710183,4.803863,6.087425,2.937477,-0.511317,0.561156,2.418347,6.825906,8.878812,2.744184,-5.215153,2.178154,-6.020373,2.275832,-9.897672,5.905841,-1.056211,-8.828947,-5.986145,4.065045,-0.974182,-5.580248,-7.298352,0.983054,9.196588,-9.429226,7.152315,0.625567,-1.950838,5.752804,0.676996,2.338382,2.823506,-3.859311,-0.099197,3.533504,-0.728205,-4.142805,9.246252,-6.323120,-0.514626,-4.859913,1.688489,5.788139,-1.784760,-3.208377,5.627099,-8.080850,5.604932,9.479603,-4.261079,-9.474932,-6.275697,0.130771,-8.275047,-7.559854,-4.980439,-6.661618,4.482575,-2.750695,-9.771970,1.682001,-3.922039,5.889600,-9.247736,9.780558,7.383032,2.301015,-9.392455,9.226243,7.825128,4.106424,-8.272542,8.514533,5.464547,5.381422,-2.853045,-8.701628,-6.034094,-0.309507,-7.721087,5.391418,-8.358484,0.970445,-4.197187,2.059034,-2.557176,4.119929,5.692953,-1.529418,-3.443080,-6.015758,8.287994,-8.882939,-7.302734,-2.009147,-7.685024,-6.584996,6.462716,-2.220746,-2.057732,8.749620,-3.151036,5.588732,2.960553,5.083493,-8.064407,-6.210295,-8.404056,9.634700,6.073509,1.089850,-1.569543,-6.603844,8.358638,5.667923,7.265464,-1.333519,5.245151,5.527541,-6.021651,5.670955,-4.464836,-7.838166,-5.084080,-4.810807,5.456601,3.102378,7.532571,1.053892,-0.102274,1.171538,0.993633,-7.047922,7.982722,0.709224,-1.671400,-1.104464,-6.131126,2.152972,2.269480,9.537674,3.621256,-6.462432,-6.798536,3.164450,-1.955746,-4.222556,-8.314442,-6.365225,-1.509097,4.936589,-4.286403,-0.256237,-6.296532,2.652586,7.387519,6.571229,-2.644796,3.479324,3.672242,9.917011,7.483835,-8.872776,-6.224597,-9.385664,-2.820747,2.920945,1.985019,-3.666835,6.644695,7.135512,-0.055712,4.612932,-3.459642,4.150498,0.866048,4.841621,4.943315,4.168864,3.375913,-6.244692,1.855100,-8.788585,5.048498,8.389364,5.351954,7.352271,1.745683,-8.226815,-7.769348,1.791154,-4.141698,-2.393793,-4.329930,-2.798352,8.361682,-2.219240,-4.335502,-3.405766,4.112702,9.442567,-8.996594,-5.242657,5.498883,-7.374576,-9.462638,-0.805693,-4.885375,-7.459721,-8.025261,-5.372180,-4.428896,-5.095649,-3.035185,-1.061834,-2.432686,-2.526278,6.920980,-5.583155,7.278639,-6.582410,1.820053,-8.588623,1.575185,-9.295730,6.470187,-0.365639,8.909914,4.596517,2.040272,5.694330,5.441598,-7.043314,-7.529402,0.428195,3.898615,-6.825668,-6.636041,9.483719,0.354119,9.488873,-5.027599,-5.243795,5.966392,6.324661,-1.921831,-3.960176,-3.202477,-4.730673,-5.259579,-2.535512,8.674521,-1.516387,-1.907691,8.183015,0.639211,-1.073250,-5.487504,-1.272799,-5.556535,-6.150873,-3.139908,-3.891024,-1.659634,-6.850552,5.976657,-5.684995,9.948280,-6.552137,-2.344533,0.415114,1.942664,-7.799456,3.013544,5.822359,-4.111399,-7.581877,-9.542606,-5.832761,7.384069,4.594780,0.523232,0.639358,-5.278576,-7.040846,9.939274,3.751173,-3.178016,5.858909,-0.515047,5.804958,-0.577547,3.276181,3.226137,1.294757,-0.429265,6.671131,6.940046,-9.031818,-5.810962,-8.290128,5.318118,-0.351470,1.605347,7.992910,-3.654673,4.261490,-9.120125,-9.197510,-1.275661,-7.651481,-2.363578,-2.292425,-0.701908,-2.298535,-2.014345,-6.151212,-4.565346,3.093959,-9.390581,9.831788,5.304341,-1.985712,3.464679,6.763534,-3.679170,5.859649,9.858622,-2.166385,-2.769660,8.210347,-3.710355,-4.153175,-7.454423,-7.040858,-5.862893,-7.634955,-3.696055,7.196047,0.473092,0.280511,-8.796034,-6.769401,6.991841,7.146710,5.889300,-7.426327,6.276812,-8.649660,-0.376392,-8.255049,-5.344641,-3.377490,1.577857,4.937064,8.091128,1.367869,9.162724,2.791387,-8.530536,-4.482122,4.402616,5.290836,-8.595954,7.545611,4.947299,2.585033,-1.472069,-3.646832,-5.745118,-1.151412,4.929513,6.061199,1.887941,5.506007,-1.070361,-0.023666,-7.414862,-9.419394,8.006769,5.107596,4.318053,8.928210,-8.052040,-9.086897,5.320564,-1.202538,-3.220759,3.200805,5.180282,1.429097,-5.668603,4.826900,-2.450929,6.073736,2.698497,-9.961282,-3.895053,-2.636258,5.563968,1.732042,2.179073,3.048683,2.626905,-3.933437,4.022623,-2.901906,9.958556,-5.692502,9.532873,5.117128,-9.798986,-8.576646,7.031730,-8.223784,1.470800,9.302099,3.122006,-8.832962,3.764348,4.191213,-9.383693,7.845096,-0.626292,3.741628,2.922598,-9.085460,-3.471120,-4.961429,-2.474178,3.618250,-4.803857,-3.367065,5.741963,4.632753,7.789240,-4.857902,-0.036848,8.078818,7.945279,4.572690,-8.172983,4.878454,-4.365326,-5.009562,-2.642728,-3.049561,5.654441,-4.332089,3.743087,9.507341,5.829820,-9.401296,0.008302,-4.955098,9.222110,-1.239446,4.994062,-4.339253,7.930039,-0.047788,-2.231776,3.628375,-2.273024,2.848789,3.623500,-0.166163,-3.004173,-0.281994,6.499659,-3.261464,-1.893093,-2.868344,-6.022951,4.179274,9.964812,5.476515,-6.953745,6.651677,6.683411,-4.479240,-3.679229,9.033177,-9.563068], dtype = "float32")#candidate|751|(960,)|const|float32
call_750 = relay.TupleGetItem(func_294_call(relay.reshape(const_751.astype('float32'), [5, 12, 16]), relay.reshape(const_751.astype('float32'), [5, 12, 16]), ), 0)
call_752 = relay.TupleGetItem(func_297_call(relay.reshape(const_751.astype('float32'), [5, 12, 16]), relay.reshape(const_751.astype('float32'), [5, 12, 16]), ), 0)
var_758 = relay.var("var_758", dtype = "float32", shape = (960,))#candidate|758|(960,)|var|float32
bop_759 = relay.right_shift(const_751.astype('uint8'), relay.reshape(var_758.astype('uint8'), relay.shape_of(const_751))) # shape=(960,)
bop_764 = relay.not_equal(bop_759.astype('bool'), relay.reshape(const_751.astype('bool'), relay.shape_of(bop_759))) # shape=(960,)
bop_767 = relay.less(uop_742.astype('bool'), relay.reshape(bop_747.astype('bool'), relay.shape_of(uop_742))) # shape=(12,)
bop_770 = relay.minimum(bop_747.astype('int8'), relay.reshape(uop_742.astype('int8'), relay.shape_of(bop_747))) # shape=(12,)
uop_776 = relay.atanh(uop_742.astype('float64')) # shape=(12,)
uop_778 = relay.log2(uop_776.astype('float32')) # shape=(12,)
func_502_call = mod.get_global_var('func_502')
func_507_call = mutated_mod.get_global_var('func_507')
var_787 = relay.var("var_787", dtype = "float64", shape = (15,))#candidate|787|(15,)|var|float64
var_788 = relay.var("var_788", dtype = "float32", shape = (3, 2))#candidate|788|(3, 2)|var|float32
call_786 = relay.TupleGetItem(func_502_call(relay.reshape(var_787.astype('float64'), [15,]), relay.reshape(var_787.astype('float64'), [15,]), relay.reshape(var_788.astype('float32'), [6,]), ), 1)
call_789 = relay.TupleGetItem(func_507_call(relay.reshape(var_787.astype('float64'), [15,]), relay.reshape(var_787.astype('float64'), [15,]), relay.reshape(var_788.astype('float32'), [6,]), ), 1)
uop_797 = relay.acos(uop_776.astype('float32')) # shape=(12,)
output = relay.Tuple([call_750,bop_764,bop_767,bop_770,uop_778,call_786,var_787,var_788,uop_797,])
output2 = relay.Tuple([call_752,bop_764,bop_767,bop_770,uop_778,call_789,var_787,var_788,uop_797,])
func_799 = relay.Function([var_741,var_758,var_787,var_788,], output)
mod['func_799'] = func_799
mod = relay.transform.InferType()(mod)
mutated_mod['func_799'] = func_799
mutated_mod = relay.transform.InferType()(mutated_mod)
func_799_call = mutated_mod.get_global_var('func_799')
var_801 = relay.var("var_801", dtype = "float64", shape = (12,))#candidate|801|(12,)|var|float64
var_802 = relay.var("var_802", dtype = "float32", shape = (960,))#candidate|802|(960,)|var|float32
var_803 = relay.var("var_803", dtype = "float64", shape = (15,))#candidate|803|(15,)|var|float64
var_804 = relay.var("var_804", dtype = "float32", shape = (3, 2))#candidate|804|(3, 2)|var|float32
call_800 = func_799_call(var_801,var_802,var_803,var_804,)
output = call_800
func_805 = relay.Function([var_801,var_802,var_803,var_804,], output)
mutated_mod['func_805'] = func_805
mutated_mod = relay.transform.InferType()(mutated_mod)
var_822 = relay.var("var_822", dtype = "uint64", shape = (9, 6, 8))#candidate|822|(9, 6, 8)|var|uint64
const_823 = relay.const([[[-6,-8,-2,4,-2,6,-7,-7],[-1,-2,3,1,4,9,8,9],[4,8,-8,2,3,3,-4,-6],[-3,2,-6,-7,-6,2,10,5],[3,-9,-6,-1,8,-8,6,-7],[-2,-4,2,3,-8,10,-8,-3]],[[10,2,-5,-4,8,-5,-8,3],[3,-10,-9,-1,-8,6,-7,9],[-4,-3,7,-2,-3,-9,-1,-1],[-10,-9,6,7,-10,3,5,-5],[4,-8,7,-6,-10,1,9,-8],[-5,-9,7,-6,-10,-4,-1,-4]],[[7,4,3,-4,4,4,9,2],[9,4,9,2,-9,2,5,4],[7,8,9,3,-7,-9,-1,10],[-4,7,-4,-1,-5,-4,-1,4],[6,8,-4,-1,9,9,10,-9],[4,8,-5,3,-5,2,-5,4]],[[8,9,2,-5,7,-1,3,-3],[-6,-7,3,-2,-8,4,5,-5],[1,-5,1,6,-1,-6,6,1],[4,6,-9,6,5,1,2,-2],[-9,8,6,9,-10,4,-6,-2],[5,8,6,-1,5,-10,-7,10]],[[1,5,1,-9,-9,-8,9,7],[-6,8,-10,3,-1,4,7,-4],[-1,8,-1,4,-9,6,2,2],[10,-10,-10,-8,7,-1,-8,-5],[-1,3,-1,3,9,9,-5,-9],[-2,-5,-3,-7,7,-9,2,5]],[[8,-5,-7,10,-10,-9,-10,9],[-9,-8,9,7,3,2,9,-9],[5,5,-4,10,-3,-6,7,-9],[-10,-3,2,7,-3,9,6,8],[-3,5,-9,1,4,-9,3,-9],[4,-5,1,10,-1,-7,-8,-8]],[[1,8,10,4,-7,3,8,6],[-4,-8,1,1,8,8,-5,7],[-4,6,4,-6,-2,2,2,3],[9,3,9,3,-3,3,1,-4],[5,-7,-1,-2,1,-4,-4,-1],[8,6,-9,3,9,-2,6,-5]],[[7,1,-4,-2,9,-10,-4,-7],[-6,-8,-1,6,6,2,5,2],[9,1,-6,-10,8,-9,10,5],[-8,-10,-3,-1,-6,1,-2,3],[-6,-5,-7,7,-5,-10,-2,-1],[-5,1,1,6,-7,-9,10,8]],[[-6,8,10,-1,-10,9,3,4],[5,7,-7,3,2,1,-6,7],[6,7,3,-8,-8,-6,-9,2],[-4,-7,-4,5,4,7,6,6],[-3,7,-3,-8,-4,-4,3,-6],[-7,-9,5,6,-9,3,-8,9]]], dtype = "uint64")#candidate|823|(9, 6, 8)|const|uint64
bop_824 = relay.bitwise_xor(var_822.astype('uint64'), relay.reshape(const_823.astype('uint64'), relay.shape_of(var_822))) # shape=(9, 6, 8)
var_829 = relay.var("var_829", dtype = "uint64", shape = (9, 6, 8))#candidate|829|(9, 6, 8)|var|uint64
bop_830 = relay.not_equal(bop_824.astype('bool'), relay.reshape(var_829.astype('bool'), relay.shape_of(bop_824))) # shape=(9, 6, 8)
bop_853 = relay.floor_mod(const_823.astype('float32'), relay.reshape(bop_824.astype('float32'), relay.shape_of(const_823))) # shape=(9, 6, 8)
func_120_call = mod.get_global_var('func_120')
func_128_call = mutated_mod.get_global_var('func_128')
const_864 = relay.const([2.540653,7.051389,8.888186,-0.997326,-9.670933,9.819040,9.461093,-7.202217,9.851938,-0.987453,4.691851,-9.731992,6.868163,-9.557669,8.468693,-3.481880,3.203070,3.752085,0.021385,9.468121,-4.173082,-7.334100,4.691985,-6.559312,-5.066558,-4.846175,-9.365900,-2.642260,2.309890,-5.688631,-3.323086,4.860679,-8.830300,-1.188858,9.384040,-9.140025,-1.390380,1.364044,0.175313,-0.069673,3.015197,6.701074,-1.445330,5.427347,9.728298,-7.124164,1.485824,9.776208,0.861111,4.077814,-8.333674,-1.328851,-6.765494,-1.574358,5.369139,9.532451,0.874417,-2.817006,-0.599579,7.176945,2.702367,9.200635,-6.867950,1.366328,-2.863181,-8.880753,-5.986526,-4.556729,-9.418855,-6.779268,1.423631,-6.846932,-8.310300,-6.811838,4.576492,-5.707876,-8.253281,4.085109,7.537930,-4.760331,4.491341], dtype = "float64")#candidate|864|(81,)|const|float64
call_863 = relay.TupleGetItem(func_120_call(relay.reshape(const_864.astype('float64'), [9, 9]), relay.reshape(const_864.astype('float64'), [9, 9]), relay.reshape(const_864.astype('float64'), [9, 9]), relay.reshape(const_864.astype('float32'), [9, 9]), relay.reshape(const_864.astype('float32'), [9, 9]), relay.reshape(const_864.astype('float32'), [9, 9]), ), 4)
call_865 = relay.TupleGetItem(func_128_call(relay.reshape(const_864.astype('float64'), [9, 9]), relay.reshape(const_864.astype('float64'), [9, 9]), relay.reshape(const_864.astype('float64'), [9, 9]), relay.reshape(const_864.astype('float32'), [9, 9]), relay.reshape(const_864.astype('float32'), [9, 9]), relay.reshape(const_864.astype('float32'), [9, 9]), ), 4)
bop_866 = relay.greater(var_829.astype('bool'), relay.reshape(bop_853.astype('bool'), relay.shape_of(var_829))) # shape=(9, 6, 8)
uop_872 = relay.asinh(bop_830.astype('float32')) # shape=(9, 6, 8)
output = relay.Tuple([call_863,const_864,bop_866,uop_872,])
output2 = relay.Tuple([call_865,const_864,bop_866,uop_872,])
func_874 = relay.Function([var_822,var_829,], output)
mod['func_874'] = func_874
mod = relay.transform.InferType()(mod)
var_875 = relay.var("var_875", dtype = "uint64", shape = (9, 6, 8))#candidate|875|(9, 6, 8)|var|uint64
var_876 = relay.var("var_876", dtype = "uint64", shape = (9, 6, 8))#candidate|876|(9, 6, 8)|var|uint64
output = func_874(var_875,var_876,)
func_877 = relay.Function([var_875,var_876,], output)
mutated_mod['func_877'] = func_877
mutated_mod = relay.transform.InferType()(mutated_mod)
var_894 = relay.var("var_894", dtype = "uint32", shape = (6, 11))#candidate|894|(6, 11)|var|uint32
var_895 = relay.var("var_895", dtype = "uint32", shape = (6, 11))#candidate|895|(6, 11)|var|uint32
bop_896 = relay.subtract(var_894.astype('uint32'), relay.reshape(var_895.astype('uint32'), relay.shape_of(var_894))) # shape=(6, 11)
bop_899 = relay.equal(bop_896.astype('bool'), relay.reshape(var_895.astype('bool'), relay.shape_of(bop_896))) # shape=(6, 11)
uop_902 = relay.sqrt(bop_896.astype('float64')) # shape=(6, 11)
uop_905 = relay.asinh(var_895.astype('float64')) # shape=(6, 11)
var_910 = relay.var("var_910", dtype = "float64", shape = (6, 11))#candidate|910|(6, 11)|var|float64
bop_911 = relay.mod(uop_902.astype('float32'), relay.reshape(var_910.astype('float32'), relay.shape_of(uop_902))) # shape=(6, 11)
bop_915 = relay.mod(uop_905.astype('float32'), relay.reshape(bop_899.astype('float32'), relay.shape_of(uop_905))) # shape=(6, 11)
uop_922 = relay.asin(bop_915.astype('float32')) # shape=(6, 11)
func_408_call = mod.get_global_var('func_408')
func_413_call = mutated_mod.get_global_var('func_413')
const_927 = relay.const([False,True,True,False,False,False,False,True,False,True,True,False,False,True,True,True,True,False,False,False,False,False,False,False,True,False,True,True,False,False,True,False,True,False,True,False,False,True,True,True], dtype = "bool")#candidate|927|(40,)|const|bool
var_928 = relay.var("var_928", dtype = "float32", shape = (6,))#candidate|928|(6,)|var|float32
var_929 = relay.var("var_929", dtype = "float32", shape = (2, 12))#candidate|929|(2, 12)|var|float32
call_926 = relay.TupleGetItem(func_408_call(relay.reshape(const_927.astype('bool'), [8, 5]), relay.reshape(var_928.astype('float32'), [6,]), relay.reshape(var_929.astype('float32'), [6, 4]), ), 0)
call_930 = relay.TupleGetItem(func_413_call(relay.reshape(const_927.astype('bool'), [8, 5]), relay.reshape(var_928.astype('float32'), [6,]), relay.reshape(var_929.astype('float32'), [6, 4]), ), 0)
uop_931 = relay.exp(bop_899.astype('float32')) # shape=(6, 11)
bop_935 = relay.greater(bop_899.astype('bool'), relay.reshape(uop_931.astype('bool'), relay.shape_of(bop_899))) # shape=(6, 11)
func_600_call = mod.get_global_var('func_600')
func_603_call = mutated_mod.get_global_var('func_603')
var_939 = relay.var("var_939", dtype = "uint64", shape = (210,))#candidate|939|(210,)|var|uint64
call_938 = relay.TupleGetItem(func_600_call(relay.reshape(var_939.astype('uint64'), [15, 14])), 0)
call_940 = relay.TupleGetItem(func_603_call(relay.reshape(var_939.astype('uint64'), [15, 14])), 0)
uop_942 = relay.atanh(uop_922.astype('float32')) # shape=(6, 11)
bop_945 = relay.bitwise_xor(uop_942.astype('int16'), relay.reshape(bop_911.astype('int16'), relay.shape_of(uop_942))) # shape=(6, 11)
var_948 = relay.var("var_948", dtype = "float32", shape = (6, 11))#candidate|948|(6, 11)|var|float32
bop_949 = relay.power(uop_922.astype('float64'), relay.reshape(var_948.astype('float64'), relay.shape_of(uop_922))) # shape=(6, 11)
bop_952 = relay.multiply(bop_949.astype('float64'), relay.reshape(bop_896.astype('float64'), relay.shape_of(bop_949))) # shape=(6, 11)
uop_958 = relay.cosh(uop_942.astype('float64')) # shape=(6, 11)
func_234_call = mod.get_global_var('func_234')
func_237_call = mutated_mod.get_global_var('func_237')
const_962 = relay.const([-10,8,-5,10,-2,-3,-5,4,9,3,4,4,7,8,3], dtype = "int8")#candidate|962|(15,)|const|int8
call_961 = relay.TupleGetItem(func_234_call(relay.reshape(const_962.astype('int8'), [15,]), relay.reshape(const_962.astype('int8'), [15,]), ), 2)
call_963 = relay.TupleGetItem(func_237_call(relay.reshape(const_962.astype('int8'), [15,]), relay.reshape(const_962.astype('int8'), [15,]), ), 2)
output = relay.Tuple([call_926,const_927,var_928,var_929,bop_935,call_938,var_939,bop_945,bop_952,uop_958,call_961,const_962,])
output2 = relay.Tuple([call_930,const_927,var_928,var_929,bop_935,call_940,var_939,bop_945,bop_952,uop_958,call_963,const_962,])
func_964 = relay.Function([var_894,var_895,var_910,var_928,var_929,var_939,var_948,], output)
mod['func_964'] = func_964
mod = relay.transform.InferType()(mod)
var_965 = relay.var("var_965", dtype = "uint32", shape = (6, 11))#candidate|965|(6, 11)|var|uint32
var_966 = relay.var("var_966", dtype = "uint32", shape = (6, 11))#candidate|966|(6, 11)|var|uint32
var_967 = relay.var("var_967", dtype = "float64", shape = (6, 11))#candidate|967|(6, 11)|var|float64
var_968 = relay.var("var_968", dtype = "float32", shape = (6,))#candidate|968|(6,)|var|float32
var_969 = relay.var("var_969", dtype = "float32", shape = (2, 12))#candidate|969|(2, 12)|var|float32
var_970 = relay.var("var_970", dtype = "uint64", shape = (210,))#candidate|970|(210,)|var|uint64
var_971 = relay.var("var_971", dtype = "float32", shape = (6, 11))#candidate|971|(6, 11)|var|float32
output = func_964(var_965,var_966,var_967,var_968,var_969,var_970,var_971,)
func_972 = relay.Function([var_965,var_966,var_967,var_968,var_969,var_970,var_971,], output)
mutated_mod['func_972'] = func_972
mutated_mod = relay.transform.InferType()(mutated_mod)
var_980 = relay.var("var_980", dtype = "float64", shape = (8, 13, 11))#candidate|980|(8, 13, 11)|var|float64
const_981 = relay.const([[[8.432366,0.167312,0.454560,-1.910115,-4.919331,0.473726,-7.626449,-0.078517,-1.835104,6.534452,5.444501],[-0.966174,4.611327,0.264017,5.674046,-2.466747,-1.206001,-6.891005,5.892705,-8.014612,2.191930,-9.000950],[-1.621226,-0.221439,-9.247592,0.596838,9.720693,-9.543663,-1.088304,-0.245119,4.158925,-7.302615,9.782873],[0.720842,-5.875303,-2.636550,2.114093,1.580904,1.864732,-6.942418,9.205134,1.160919,6.888061,7.580667],[7.774512,3.027136,-9.725868,-4.630452,-8.251050,-5.184112,9.930299,-1.966386,4.695010,4.726740,3.450398],[-5.321340,4.365096,1.334272,-0.553661,-3.813952,7.032635,-0.929809,-3.245363,6.211678,-5.890096,9.282661],[-8.554155,-3.292172,2.331267,-9.857354,3.818474,8.217462,-0.499211,6.864100,-1.709294,-0.388598,-4.697976],[4.992182,9.866725,2.014462,4.190395,-3.372348,4.988540,8.380934,-0.048581,6.105217,7.724387,-0.355107],[-2.556790,-1.160913,-4.284225,-3.001618,6.270032,0.780634,-0.395335,-7.594852,1.316376,6.882069,9.242291],[7.912148,-3.625802,6.223867,2.083390,5.790887,2.782618,-3.915939,-3.134183,-6.231428,0.599089,-6.507013],[2.316937,3.409846,2.995827,5.023897,8.974205,-3.854028,5.321601,7.229196,-7.150799,-1.451370,-6.792934],[8.101531,6.515902,-8.997966,7.477996,-9.688996,6.450655,5.100137,-1.363440,3.509692,1.967068,6.188748],[5.779307,2.358755,7.421497,-3.899213,5.345082,-2.682485,4.019655,8.154534,-8.899155,9.074971,8.565472]],[[-8.559554,4.539214,6.467295,-0.272395,-9.164680,-9.820617,4.303018,9.578030,3.494570,2.144574,-9.434352],[3.704874,2.188324,4.817548,3.476543,3.917055,3.295832,-7.222509,6.165996,2.560572,-8.313738,5.367466],[-4.986601,8.963823,1.434379,9.974861,-5.509524,9.101294,-6.707650,-9.895144,7.068316,-2.473937,3.409174],[-6.464022,1.676752,-7.851689,5.758372,6.852612,-8.324919,4.265532,5.148918,1.554886,4.702511,7.745731],[-8.425456,-8.426076,9.777403,-2.725308,9.021729,8.238448,-2.823396,-8.780962,1.210096,3.764133,7.202356],[-2.769642,-6.402465,-1.594373,-1.275855,3.318858,5.262182,0.452955,-8.086053,4.851601,-2.805416,-6.864839],[-6.951085,-9.181914,-3.289702,-8.960272,-7.007375,-1.513441,-7.493064,7.263338,-7.851467,7.150219,-2.197254],[-3.780516,-6.286827,-0.444741,-5.379424,-2.157170,6.208384,9.046971,-7.889338,1.670199,-2.909504,2.683002],[-2.220094,5.647711,-1.021368,5.947876,8.709470,8.718802,3.057916,-8.815661,2.360118,4.186673,0.248293],[-0.951439,-8.356155,8.587943,-1.540075,1.507289,8.323242,5.870367,-6.538139,-2.742769,9.813213,2.335851],[3.349577,-8.690703,-0.298152,8.026351,-7.291322,-7.319500,-4.718444,3.091370,-1.734114,3.977734,-9.431139],[8.482524,1.875248,-3.800813,-0.035107,-9.992419,-6.418303,-0.523735,-1.880628,-2.702766,0.802143,5.583299],[-8.255919,-7.743805,2.373490,-7.330028,2.645103,-2.296291,8.660262,3.262212,5.707752,1.414958,-7.367813]],[[-2.883217,-2.361112,-6.321153,0.783396,4.737095,-8.026001,5.743917,7.034990,4.040521,2.064623,-9.040433],[3.904274,-3.493234,6.178575,-3.803538,4.414170,-3.694710,4.900964,-2.622631,-9.449093,6.617048,2.033367],[-1.521453,-7.000814,4.986563,-9.395748,1.338526,7.958569,-6.838698,4.481429,9.247197,3.313453,-7.779974],[-4.329497,-9.148978,-2.472360,8.970703,-4.625211,4.539734,-0.795487,-3.234106,-2.086585,-2.870209,-3.606151],[-3.425753,5.084836,-1.303559,9.714024,6.022583,5.814899,-2.745834,7.102212,-4.092094,7.410673,4.655509],[-9.199554,-4.351216,5.589881,8.271605,2.501552,5.755283,3.022604,-0.144601,5.320409,-1.284190,0.345597],[-0.887901,-0.086254,-7.001449,-2.084717,8.733056,6.689980,7.312399,-0.316867,-0.496762,-0.569692,-8.258351],[-4.268795,-5.784311,5.910789,-7.043792,2.579408,5.685617,3.695639,-7.172534,0.361163,9.452516,6.802106],[6.598160,5.032844,-1.412705,-4.317639,6.900962,6.812485,-7.368474,-8.751462,4.545016,0.032137,5.888841],[-6.046799,0.822294,-4.302185,-3.775989,0.010410,7.904862,-3.041422,7.301942,8.150096,5.112731,-8.382229],[6.758371,-8.999375,4.362913,4.288768,8.341288,-5.195976,-0.172351,6.940407,-6.820582,-8.765267,9.701133],[5.219929,8.501362,7.104295,2.305252,-2.536213,-4.661746,9.355961,5.087051,-0.283791,-0.738826,7.195723],[2.109955,-1.348343,4.721752,-7.508667,3.439346,4.314165,-8.492713,5.543438,8.490579,-6.309094,-5.026822]],[[9.208867,-8.733529,-2.911062,-9.709991,3.252216,-3.197118,8.456222,4.655900,-5.859647,-6.729880,1.725126],[-0.237183,-3.302254,5.101231,6.971958,8.242765,5.555522,4.817761,-4.537470,2.491433,3.478885,0.108299],[1.355948,-9.852748,4.854059,-2.945829,6.528014,1.968040,-7.926094,-9.330390,5.015536,-2.919298,-4.798324],[-2.772121,6.938455,-0.496264,-2.309731,2.451481,-4.792607,5.635387,-2.129974,5.048074,0.696011,-3.806268],[4.682583,0.725231,-0.806266,7.752119,1.088562,6.017408,0.598008,-1.374216,-5.541781,3.891481,2.321624],[-5.313545,-7.390662,-2.987626,1.201467,4.732939,1.527498,2.592901,-8.289571,-9.405425,-3.606674,-3.367531],[-1.202797,7.355499,3.084548,2.402824,-5.028192,-8.655125,6.207482,-9.034687,0.611611,9.359264,-4.611461],[-6.119404,6.867806,5.570028,9.872723,0.286646,-4.409912,-5.284047,-4.201977,-9.068301,7.076801,0.470511],[8.902163,1.855794,-1.516219,-9.632015,0.486756,2.378052,0.082223,4.515254,6.891531,-2.278873,8.490400],[0.179555,0.912160,-4.216723,-5.419304,1.464442,4.615161,3.736688,-7.297035,2.880261,-3.482002,-5.583930],[9.197955,-4.971807,7.524898,-4.778886,1.911226,-9.297474,1.696145,-1.481736,-1.009209,-5.675427,-0.370203],[7.842937,0.211063,3.455029,-0.664558,2.348475,3.286847,-2.617687,1.186487,2.686152,-7.011612,4.110768],[2.376273,6.414513,6.832725,8.876106,5.762859,-6.844332,7.182745,3.829300,-4.266023,7.780288,-5.075776]],[[-7.750233,-8.008059,9.534827,0.002989,6.284999,-5.898329,-1.645570,-7.467796,7.231740,4.076876,-1.013317],[2.453733,-3.966194,-6.561769,2.013320,0.047988,-0.420001,-2.849153,-9.529095,-7.106167,-6.971862,-5.888427],[6.118385,-6.469015,5.201272,-9.166427,9.768667,7.113637,8.967549,8.399072,-9.454112,-1.353576,7.819980],[-5.750869,-3.884342,8.334506,5.140833,-7.305987,-6.326078,-7.841345,-5.847713,9.916472,-8.026743,1.784729],[-2.037756,4.651947,8.207026,-8.809057,-5.347595,-7.977471,0.021091,5.411572,9.650843,9.441179,-4.548575],[8.766046,3.339701,-0.526629,7.424324,9.294758,5.511465,5.913483,6.096465,0.682306,-8.583892,-1.716280],[3.344126,4.655495,2.701242,-4.319813,-5.259187,-7.885991,-4.816978,1.766884,-8.982763,3.106549,2.071274],[-9.589373,-3.951236,-3.642263,9.116218,0.793128,-1.692054,-0.760267,5.787516,-6.566602,-1.774880,3.033022],[3.613633,3.562077,-1.223957,8.140289,9.224766,-2.904404,3.971172,-8.453540,0.965159,9.152805,-8.454178],[-4.399257,-7.519429,9.467744,5.251531,9.477625,-0.698116,3.181874,-3.253392,-4.345609,-4.819284,4.139271],[-5.530481,1.420198,4.172996,6.988188,-8.979809,-3.596772,-5.720219,1.822884,-7.392072,1.414659,5.553586],[4.201526,7.361676,4.005216,-3.284576,-5.472844,5.309280,8.440342,-2.726270,-4.700547,-5.704342,-9.270133],[1.254937,-7.174469,-8.922652,-9.898704,-3.266172,6.996519,7.775818,-1.134983,-7.389277,-5.912876,-8.962172]],[[-2.811150,8.079706,0.562991,9.618915,-9.224129,7.975169,0.280691,-6.565864,-3.400492,6.975202,1.429211],[-1.996278,0.219761,-5.574704,-3.139014,1.083398,8.665712,-5.327057,-2.696713,-9.516161,-0.446863,6.615653],[-5.343876,1.434436,5.090998,-6.827564,1.608299,-1.692746,6.375454,9.647290,3.143121,7.883094,3.951958],[2.225659,1.993678,-7.163229,3.088126,2.059388,-0.379435,-2.040386,-0.191376,-4.639724,-9.190088,-0.132281],[9.403103,2.959320,-1.514362,9.373056,-6.613647,8.120096,2.404380,2.331808,1.688826,-1.947257,0.558438],[2.544324,5.745492,-9.761035,-0.860615,-7.664919,-3.482519,-0.067352,-8.865664,-8.466958,-6.516312,-8.637785],[4.982002,4.269224,7.103322,-3.313506,-1.449322,-1.183123,4.707246,2.475175,1.773750,0.665915,-5.767821],[-8.845199,-4.841743,1.908424,-0.042460,-2.542189,-9.524229,-9.750544,2.577076,-9.457199,2.103433,7.019265],[9.975983,-9.494897,-7.797160,-2.722310,-0.452374,7.159350,4.818152,5.093401,0.451060,-4.066013,4.478973],[0.079284,1.307430,5.752103,-9.522762,-4.302500,4.157926,-6.270228,-2.661787,-2.469839,5.716134,0.470970],[0.070197,-6.595075,-2.086954,-5.801113,9.318833,-6.615344,6.902398,8.786916,6.669729,-3.038462,0.663238],[-2.439548,1.233728,-7.899426,1.824936,-4.576737,-1.576603,2.772131,1.087290,6.471572,9.218510,9.872526],[6.341865,-7.035736,-2.230256,2.378589,7.698269,-4.180917,8.977005,7.874931,-2.379210,3.774021,-8.160139]],[[-3.259506,-8.686828,-0.657017,5.579069,-9.476889,6.372839,8.484808,0.046245,8.374745,6.783566,-9.236330],[-4.062177,6.470058,8.542745,7.258100,9.478474,6.955972,5.712713,1.530310,1.100488,2.242546,-6.165072],[6.277135,-2.306678,-5.678862,-6.811506,-1.304716,-3.136086,0.685527,-9.495098,3.856710,-1.394636,6.272559],[-7.964886,6.117725,-3.209507,-3.410954,0.798776,7.127627,-8.660027,2.770737,-2.619079,1.994014,7.351838],[-0.467079,-3.325532,4.979646,-7.333834,-1.063405,1.896975,-5.489984,2.741443,-7.825972,-8.851159,-8.548792],[0.690177,5.999393,4.043075,-2.912958,-4.967021,-0.173106,-2.026952,8.233232,-2.633226,-3.534619,5.888910],[0.217135,7.369355,-1.518026,5.249394,9.015598,-1.624250,-2.336323,0.173053,2.711872,4.441719,-6.404452],[1.577821,-7.216538,-6.693532,-8.172118,-4.413108,-3.999911,3.403830,2.036658,3.303549,-1.482092,-2.335330],[7.349535,9.887201,9.958111,-8.434218,6.301072,0.849448,6.115594,-2.079131,-6.210472,2.400617,-9.126164],[1.357984,-3.983082,-5.129476,-4.877904,5.936545,6.054359,-8.565535,-3.928252,-9.776110,1.906797,4.339845],[6.090798,-8.439850,-1.674675,-0.289623,-8.829806,8.851972,-3.699544,2.172439,-9.728852,-8.606032,-1.343132],[-2.048312,6.317863,-5.426204,-7.861695,8.348948,1.751801,4.832156,-7.888233,-5.250320,4.753881,4.721507],[-3.212309,0.201658,1.965887,-1.672805,-0.239865,-9.757029,5.948514,-0.365236,6.654631,-7.585290,2.403035]],[[-8.544475,7.337612,2.593354,-0.165042,3.468163,8.517526,6.334378,-8.322867,-5.746749,-4.855412,7.662081],[5.992092,-0.836276,4.610698,5.074808,-6.417488,-4.956566,2.441704,4.451228,-5.468192,-0.420069,3.969874],[4.933236,7.262147,-8.191349,-0.582381,4.615411,-5.048214,-6.064591,-0.981163,-3.678589,-3.759087,8.608787],[-6.822174,8.632669,1.735173,7.672453,-8.571002,-7.852629,-0.829935,1.743403,-4.489766,7.689205,-1.355784],[2.413588,7.551685,-6.331301,-2.766618,4.336033,-8.043191,-1.792366,-0.013173,-5.486594,-5.335156,-4.360615],[2.261373,6.914602,2.347455,-7.879397,0.310710,-7.135470,-4.661759,-4.081278,-3.090762,7.805427,-7.926067],[-0.642675,9.419906,8.298648,-3.193022,4.244950,-7.563013,8.813990,5.967383,3.035504,5.592481,0.703099],[-6.318406,6.734777,-9.824215,0.893082,-5.517052,3.329433,-2.703358,0.645481,9.196046,8.747292,7.199014],[1.896848,-0.181439,3.834127,-4.889143,6.031405,-0.893311,-3.448824,-3.970443,-8.332036,5.730960,-1.599103],[-9.323346,-1.204974,-7.018760,1.912481,3.133717,-3.242722,-3.894283,2.953128,1.498683,5.774099,5.813823],[-7.647572,-4.057301,-9.484545,-3.454506,3.922545,-2.114136,-2.758261,3.813352,0.117719,-5.807120,6.550951],[8.780542,-9.393425,0.807862,2.824696,1.712811,-0.380609,-3.315746,4.934590,-7.973078,4.778993,0.358647],[0.513034,-6.664971,4.594188,5.932669,3.709129,0.620079,0.263701,0.824052,-8.631020,-2.718136,-8.580439]]], dtype = "float64")#candidate|981|(8, 13, 11)|const|float64
bop_982 = relay.mod(var_980.astype('float64'), relay.reshape(const_981.astype('float64'), relay.shape_of(var_980))) # shape=(8, 13, 11)
func_212_call = mod.get_global_var('func_212')
func_214_call = mutated_mod.get_global_var('func_214')
const_986 = relay.const([3.865641,-7.677369,5.630244,5.629935,2.265349,-1.922984,2.973919,7.218659,9.619797,-1.348833,-2.001232,-2.499462,6.745537,-9.348265,3.273226,-2.955241,-3.500712,3.924826,-8.178250,-6.193108,5.992667,-2.515587,9.263549,-0.253241,-8.287439,-7.474662,-5.832612,-5.281862,-1.310336,-4.732076,-2.322991,-3.197223,-6.286661,7.359269,-0.994493,-6.998854,-8.696017,6.322008,5.878107,-4.745424,-2.996206,-2.818630,-0.138788,-2.057029,4.792011,-1.017172,4.392329,-4.606858], dtype = "float64")#candidate|986|(48,)|const|float64
call_985 = relay.TupleGetItem(func_212_call(relay.reshape(const_986.astype('float64'), [12, 4])), 2)
call_987 = relay.TupleGetItem(func_214_call(relay.reshape(const_986.astype('float64'), [12, 4])), 2)
uop_991 = relay.cos(bop_982.astype('float32')) # shape=(8, 13, 11)
bop_995 = relay.floor_divide(uop_991.astype('float32'), relay.reshape(const_981.astype('float32'), relay.shape_of(uop_991))) # shape=(8, 13, 11)
bop_998 = relay.logical_xor(const_981.astype('int64'), relay.reshape(bop_995.astype('int64'), relay.shape_of(const_981))) # shape=(8, 13, 11)
bop_1003 = relay.less(bop_998.astype('bool'), relay.reshape(const_981.astype('bool'), relay.shape_of(bop_998))) # shape=(8, 13, 11)
uop_1006 = relay.acos(bop_982.astype('float64')) # shape=(8, 13, 11)
bop_1008 = relay.multiply(bop_995.astype('uint64'), relay.reshape(const_981.astype('uint64'), relay.shape_of(bop_995))) # shape=(8, 13, 11)
func_964_call = mod.get_global_var('func_964')
func_972_call = mutated_mod.get_global_var('func_972')
const_1013 = relay.const([-9,-6,7,-7,8,3,-10,7,-6,-2,-1,-3,3,-5,6,-6,1,1,6,7,9,-7,-6,10,5,2,10,-7,3,-1,-9,3,9,-10,-6,-8,-10,6,1,-8,8,2,-7,1,2,10,-1,3,-9,-6,3,-9,-10,7,-1,-10,6,6,-3,-5,3,6,8,-5,-6,3], dtype = "uint32")#candidate|1013|(66,)|const|uint32
var_1014 = relay.var("var_1014", dtype = "float32", shape = (6,))#candidate|1014|(6,)|var|float32
const_1015 = relay.const([2.033995,2.553825,3.822633,-7.742875,-2.669353,2.682039,7.414472,-8.581836,0.354068,6.321738,3.711457,0.144667,-9.104749,2.574926,1.192979,-8.033821,1.846276,1.471191,-4.075078,0.189584,-0.304631,4.395273,-5.522864,1.836522], dtype = "float32")#candidate|1015|(24,)|const|float32
const_1016 = relay.const([10,-8,-10,9,-4,-7,3,-8,3,-6,2,6,5,2,-7,-5,-4,-8,-2,10,-3,-10,10,-7,-3,3,-10,3,-3,9,6,9,-9,-6,3,6,2,4,-1,7,4,9,-1,1,5,-3,6,-3,-8,-5,-7,-9,-6,-3,4,2,9,3,-4,-5,-6,4,-10,-6,-7,3,3,-3,5,-7,5,10,-3,-4,-9,7,-7,5,7,-8,-2,8,-3,4,6,-8,1,5,-2,2,-7,-5,10,-2,3,3,2,-1,-2,-4,8,-9,1,4,2,5,5,-4,3,-5,-7,-2,-10,8,1,10,6,-8,-3,2,-10,-7,-10,1,-6,-10,5,-5,-6,-4,-5,-6,-6,-9,9,-6,-2,-4,-1,-6,3,1,-7,-2,-7,1,3,-4,2,-7,7,-9,-8,1,-10,-8,4,-1,-6,-5,-9,7,-4,10,4,-6,-3,-2,-10,10,-2,4,-8,7,8,-8,-8,9,-4,7,-6,-1,-10,2,5,-1,-2,3,10,5,3,4,-6,5,-2,10,-7,-9,-3,10,-6,-3,6,10,5,5,-7,-9,-10,7], dtype = "uint64")#candidate|1016|(210,)|const|uint64
call_1012 = relay.TupleGetItem(func_964_call(relay.reshape(const_1013.astype('uint32'), [6, 11]), relay.reshape(const_1013.astype('uint32'), [6, 11]), relay.reshape(const_1013.astype('float64'), [6, 11]), relay.reshape(var_1014.astype('float32'), [6,]), relay.reshape(const_1015.astype('float32'), [2, 12]), relay.reshape(const_1016.astype('uint64'), [210,]), relay.reshape(const_1013.astype('float32'), [6, 11]), ), 9)
call_1017 = relay.TupleGetItem(func_972_call(relay.reshape(const_1013.astype('uint32'), [6, 11]), relay.reshape(const_1013.astype('uint32'), [6, 11]), relay.reshape(const_1013.astype('float64'), [6, 11]), relay.reshape(var_1014.astype('float32'), [6,]), relay.reshape(const_1015.astype('float32'), [2, 12]), relay.reshape(const_1016.astype('uint64'), [210,]), relay.reshape(const_1013.astype('float32'), [6, 11]), ), 9)
func_120_call = mod.get_global_var('func_120')
func_128_call = mutated_mod.get_global_var('func_128')
const_1019 = relay.const([9.747252,5.089303,-7.849781,-3.566499,7.560405,-3.153771,-3.313427,8.017683,0.952022,7.205911,9.774456,-9.391593,0.725363,2.602259,-7.480412,6.943193,-4.807458,-1.369026,-7.613994,-2.950467,-7.231644,1.252764,3.886086,-9.524871,2.251987,7.973810,5.537636,-8.204123,-1.415858,8.635188,-5.586267,-4.520694,4.453305,-6.502723,0.895281,0.242038,3.349932,-6.870125,-8.981830,0.040667,-6.788038,-5.218342,-1.487225,-9.457013,4.752845,-1.683141,-9.687751,-2.127256,6.644841,-2.696479,9.434544,-2.699577,-1.056962,-8.758430,1.132604,0.968871,-0.230775,9.755188,-1.630167,9.518257,2.624884,0.151063,-9.011802,8.705906,-9.165640,2.024477,-6.940702,-1.346505,-0.797852,4.360501,-6.061157,-6.611797,-5.706615,-1.460378,9.780065,-4.399800,6.334181,2.541458,6.724060,7.250357,-2.634223], dtype = "float64")#candidate|1019|(81,)|const|float64
call_1018 = relay.TupleGetItem(func_120_call(relay.reshape(const_1019.astype('float64'), [9, 9]), relay.reshape(const_1019.astype('float64'), [9, 9]), relay.reshape(const_1019.astype('float64'), [9, 9]), relay.reshape(const_1019.astype('float32'), [9, 9]), relay.reshape(const_1019.astype('float32'), [9, 9]), relay.reshape(const_1019.astype('float32'), [9, 9]), ), 1)
call_1020 = relay.TupleGetItem(func_128_call(relay.reshape(const_1019.astype('float64'), [9, 9]), relay.reshape(const_1019.astype('float64'), [9, 9]), relay.reshape(const_1019.astype('float64'), [9, 9]), relay.reshape(const_1019.astype('float32'), [9, 9]), relay.reshape(const_1019.astype('float32'), [9, 9]), relay.reshape(const_1019.astype('float32'), [9, 9]), ), 1)
output = relay.Tuple([call_985,const_986,bop_1003,uop_1006,bop_1008,call_1012,const_1013,var_1014,const_1015,const_1016,call_1018,const_1019,])
output2 = relay.Tuple([call_987,const_986,bop_1003,uop_1006,bop_1008,call_1017,const_1013,var_1014,const_1015,const_1016,call_1020,const_1019,])
func_1028 = relay.Function([var_980,var_1014,], output)
mod['func_1028'] = func_1028
mod = relay.transform.InferType()(mod)
var_1029 = relay.var("var_1029", dtype = "float64", shape = (8, 13, 11))#candidate|1029|(8, 13, 11)|var|float64
var_1030 = relay.var("var_1030", dtype = "float32", shape = (6,))#candidate|1030|(6,)|var|float32
output = func_1028(var_1029,var_1030,)
func_1031 = relay.Function([var_1029,var_1030,], output)
mutated_mod['func_1031'] = func_1031
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1049 = relay.var("var_1049", dtype = "int32", shape = (4, 15, 2))#candidate|1049|(4, 15, 2)|var|int32
var_1050 = relay.var("var_1050", dtype = "int32", shape = (4, 15, 2))#candidate|1050|(4, 15, 2)|var|int32
bop_1051 = relay.right_shift(var_1049.astype('int32'), relay.reshape(var_1050.astype('int32'), relay.shape_of(var_1049))) # shape=(4, 15, 2)
output = relay.Tuple([bop_1051,])
output2 = relay.Tuple([bop_1051,])
F = relay.Function([var_1049,var_1050,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1049,var_1050,], output2)
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
input_1049= np.array([[[10,5],[1,10],[4,-7],[2,-1],[5,-2],[-1,-9],[-4,5],[-9,-9],[10,6],[-7,9],[-8,2],[-6,-4],[5,7],[-6,7],[1,6]],[[-9,-10],[-10,10],[3,-9],[-4,3],[-5,8],[5,-8],[-2,-3],[-8,-5],[-3,-6],[3,10],[-3,-9],[1,-6],[-10,-8],[2,6],[-10,-8]],[[-7,5],[-10,-10],[-5,1],[1,-7],[-1,-9],[-9,-6],[1,-4],[-2,-9],[-7,-6],[3,2],[-8,8],[-10,-2],[-9,-1],[-7,9],[10,1]],[[10,-8],[6,4],[4,9],[-8,3],[-3,-10],[-7,-9],[2,4],[-5,-4],[7,-10],[10,2],[1,-5],[-10,10],[5,-4],[-2,7],[-2,6]]], dtype='int32')
module1.set_input('var_1049', input_1049)
input_1050= np.array([[[8,-2],[3,-7],[7,6],[3,-8],[10,-4],[9,7],[10,-7],[-5,-2],[6,9],[4,-4],[-9,1],[-2,5],[-9,8],[7,-1],[-9,3]],[[-3,10],[-7,-9],[-5,3],[3,-6],[-5,8],[-9,3],[-2,-7],[5,9],[-3,9],[6,-2],[-1,6],[9,1],[-4,9],[5,-3],[8,-10]],[[-10,-3],[7,-2],[10,-5],[8,-6],[-7,2],[-6,-5],[3,-10],[8,4],[-10,-10],[8,2],[-7,-10],[-10,4],[5,-9],[5,4],[1,-8]],[[5,-8],[8,-1],[-3,6],[-10,7],[-5,7],[4,-1],[-9,-8],[-8,-3],[10,-7],[3,-7],[-1,-8],[-9,2],[5,2],[-7,-10],[-9,-9]]], dtype='int32')
module1.set_input('var_1050', input_1050)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1049, input_1050, )
res3 = intrp3.evaluate()(input_1049, input_1050, )
res4 = intrp4.evaluate()(input_1049, input_1050, )
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
module5.set_input('var_1049', input_1049)
module5.set_input('var_1050', input_1050)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1049, input_1050, )
res7 = intrp7.evaluate()(input_1049, input_1050, )
res8 = intrp8.evaluate()(input_1049, input_1050, )
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
module9.set_input('var_1049', input_1049)
module9.set_input('var_1050', input_1050)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1049, input_1050, )
res11 = intrp11.evaluate()(input_1049, input_1050, )
res12 = intrp12.evaluate()(input_1049, input_1050, )
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
module13.set_input('var_1049', input_1049)
module13.set_input('var_1050', input_1050)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1049, input_1050, )
res15 = intrp15.evaluate()(input_1049, input_1050, )
res16 = intrp16.evaluate()(input_1049, input_1050, )
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
module17.set_input('var_1049', input_1049)
module17.set_input('var_1050', input_1050)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1049, input_1050, )
res19 = intrp19.evaluate()(input_1049, input_1050, )
res20 = intrp20.evaluate()(input_1049, input_1050, )
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
module21.set_input('var_1049', input_1049)
module21.set_input('var_1050', input_1050)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1049, input_1050, )
res23 = intrp23.evaluate()(input_1049, input_1050, )
res24 = intrp24.evaluate()(input_1049, input_1050, )
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

'''49: TVMFuncCall
48: _ZNSt17_Function_handlerIFvN3tvm7run
47: tvm::runtime::TypedPackedFunc<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>::AssignTypedLambda<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>(tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target), std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const
46: tvm::relay::EvalFunction(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)
45: tvm::relay::Prepare(tvm::IRModule, tvm::CompilationConfig)
44: tvm::transform::Pass::operator()(tvm::IRModule) const
43: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
42: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
41: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
40: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
36: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
35: tvm::transform::Pass::operator()(tvm::IRModule) const
34: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
31: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
30: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
29: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
28: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
27: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
26: _ZN3tvm5relay9transform22Devic
25: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
24: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
23: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
22: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
21: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::LetNode const*)
20: tvm::relay::tec::LowerTensorExprMutator::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
19: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
16: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
15: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
14: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
13: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
12: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
11: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
9: tvm::relay::tec::TECompilerImpl::LowerShapeFunc(tvm::relay::tec::CCacheKey const&)
8: tvm::relay::tec::TECompilerImpl::LowerShapeFuncInternal(tvm::relay::tec::CCacheKey const&)
7: tvm::relay::tec::ShapeFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
6: tvm::relay::tec::MakeShapeFunc::Create(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
5: tvm::relay::tec::MakeShapeFunc::VisitExpr(tvm::RelayExpr const&)
4: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFNS_7runtime
1: tvm::relay::tec::MakeShapeFunc::VisitExpr_(tvm::relay::CallNode const*)
0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
3: TVMFuncCall
2: _ZNSt17_Function_handlerIFvN3tvm7run
1: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
0: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
4: TVMFuncCall
3: _ZNSt17_Function_handlerIFvN3tvm7run
2: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
1: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
0: tvm::runtime::Array<tvm::PrimExpr, void> tvm::runtime::TVMPODValue_::AsObjectRef<tvm::runtime::Array<tvm::PrimExpr, void> >() const

'''