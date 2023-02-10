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
var_56 = relay.var("var_56", dtype = "int16", shape = (13, 11))#candidate|56|(13, 11)|var|int16
var_57 = relay.var("var_57", dtype = "int16", shape = (13, 11))#candidate|57|(13, 11)|var|int16
bop_58 = relay.maximum(var_56.astype('int16'), relay.reshape(var_57.astype('int16'), relay.shape_of(var_56))) # shape=(13, 11)
output = relay.Tuple([bop_58,])
output2 = relay.Tuple([bop_58,])
func_84 = relay.Function([var_56,var_57,], output)
mod['func_84'] = func_84
mod = relay.transform.InferType()(mod)
var_85 = relay.var("var_85", dtype = "int16", shape = (13, 11))#candidate|85|(13, 11)|var|int16
var_86 = relay.var("var_86", dtype = "int16", shape = (13, 11))#candidate|86|(13, 11)|var|int16
output = func_84(var_85,var_86,)
func_87 = relay.Function([var_85,var_86,], output)
mutated_mod['func_87'] = func_87
mutated_mod = relay.transform.InferType()(mutated_mod)
var_153 = relay.var("var_153", dtype = "float64", shape = (2, 15))#candidate|153|(2, 15)|var|float64
uop_154 = relay.sin(var_153.astype('float64')) # shape=(2, 15)
uop_160 = relay.tan(uop_154.astype('float64')) # shape=(2, 15)
output = uop_160
output2 = uop_160
func_162 = relay.Function([var_153,], output)
mod['func_162'] = func_162
mod = relay.transform.InferType()(mod)
mutated_mod['func_162'] = func_162
mutated_mod = relay.transform.InferType()(mutated_mod)
var_163 = relay.var("var_163", dtype = "float64", shape = (2, 15))#candidate|163|(2, 15)|var|float64
func_162_call = mutated_mod.get_global_var('func_162')
call_164 = func_162_call(var_163)
output = call_164
func_165 = relay.Function([var_163], output)
mutated_mod['func_165'] = func_165
mutated_mod = relay.transform.InferType()(mutated_mod)
const_191 = relay.const(5, dtype = "int32")#candidate|191|()|const|int32
var_192 = relay.var("var_192", dtype = "int32", shape = (12, 6, 4))#candidate|192|(12, 6, 4)|var|int32
bop_193 = relay.maximum(const_191.astype('int32'), var_192.astype('int32')) # shape=(12, 6, 4)
bop_209 = relay.less_equal(const_191.astype('bool'), bop_193.astype('bool')) # shape=(12, 6, 4)
const_217 = relay.const([[[-7,-4,8,2],[-8,-3,-7,3],[1,8,-4,-7],[4,-9,-10,-3],[-7,-7,-8,8],[-1,9,4,-7]],[[7,-4,-4,1],[1,-10,-3,7],[1,-7,-2,-7],[10,2,7,8],[9,-7,3,8],[2,-6,-3,6]],[[-6,3,2,2],[5,5,-7,5],[4,-8,2,2],[6,-10,10,4],[-1,-3,-6,-4],[-1,2,1,-6]],[[1,-6,10,8],[6,6,4,-2],[-7,-9,-10,-4],[1,2,3,-8],[10,9,2,7],[-3,-5,-4,-3]],[[2,6,6,-6],[7,-10,-4,5],[2,7,1,-5],[-10,3,-7,-5],[-10,-4,2,-3],[8,9,2,9]],[[10,-10,-1,-5],[-5,-4,-7,-3],[1,-3,-9,-2],[-10,10,-5,-5],[-6,-3,-10,-7],[9,9,-10,4]],[[10,-4,4,8],[7,5,7,-4],[-7,-10,-7,1],[-10,-3,3,-4],[5,-3,-6,-4],[-9,2,10,-10]],[[2,3,-9,-5],[-7,1,-6,-4],[-5,-3,-8,-3],[-7,-4,10,4],[-4,-3,-3,3],[-7,-1,-1,5]],[[2,6,-4,-8],[5,-7,-3,-9],[10,-3,-10,-3],[1,1,9,-10],[5,-2,5,3],[-1,5,3,-5]],[[5,6,-4,-6],[4,-9,-5,-5],[-4,-3,-8,-1],[9,5,1,7],[-7,3,3,-6],[-7,4,-3,6]],[[8,6,-8,-7],[2,-10,-1,1],[9,7,3,3],[-10,6,5,-9],[-5,-1,2,1],[9,7,6,9]],[[-3,-7,2,8],[6,3,8,-6],[-9,-6,-6,-6],[-8,-6,2,10],[-2,2,9,-5],[-6,6,-2,-8]]], dtype = "int32")#candidate|217|(12, 6, 4)|const|int32
bop_218 = relay.multiply(bop_193.astype('uint8'), relay.reshape(const_217.astype('uint8'), relay.shape_of(bop_193))) # shape=(12, 6, 4)
var_224 = relay.var("var_224", dtype = "uint8", shape = (12, 6, 4))#candidate|224|(12, 6, 4)|var|uint8
bop_225 = relay.bitwise_and(bop_218.astype('uint8'), relay.reshape(var_224.astype('uint8'), relay.shape_of(bop_218))) # shape=(12, 6, 4)
bop_228 = relay.greater(bop_225.astype('bool'), relay.reshape(bop_209.astype('bool'), relay.shape_of(bop_225))) # shape=(12, 6, 4)
var_238 = relay.var("var_238", dtype = "uint8", shape = (12, 6, 4))#candidate|238|(12, 6, 4)|var|uint8
bop_239 = relay.bitwise_or(bop_218.astype('int64'), relay.reshape(var_238.astype('int64'), relay.shape_of(bop_218))) # shape=(12, 6, 4)
func_84_call = mod.get_global_var('func_84')
func_87_call = mutated_mod.get_global_var('func_87')
var_243 = relay.var("var_243", dtype = "int16", shape = (143,))#candidate|243|(143,)|var|int16
call_242 = relay.TupleGetItem(func_84_call(relay.reshape(var_243.astype('int16'), [13, 11]), relay.reshape(var_243.astype('int16'), [13, 11]), ), 0)
call_244 = relay.TupleGetItem(func_87_call(relay.reshape(var_243.astype('int16'), [13, 11]), relay.reshape(var_243.astype('int16'), [13, 11]), ), 0)
uop_246 = relay.asinh(var_238.astype('float64')) # shape=(12, 6, 4)
func_84_call = mod.get_global_var('func_84')
func_87_call = mutated_mod.get_global_var('func_87')
call_249 = relay.TupleGetItem(func_84_call(relay.reshape(call_242.astype('int16'), [13, 11]), relay.reshape(var_243.astype('int16'), [13, 11]), ), 0)
call_250 = relay.TupleGetItem(func_87_call(relay.reshape(call_242.astype('int16'), [13, 11]), relay.reshape(var_243.astype('int16'), [13, 11]), ), 0)
var_255 = relay.var("var_255", dtype = "float64", shape = (12, 6, 4))#candidate|255|(12, 6, 4)|var|float64
bop_256 = relay.logical_and(uop_246.astype('bool'), relay.reshape(var_255.astype('bool'), relay.shape_of(uop_246))) # shape=(12, 6, 4)
uop_264 = relay.sinh(uop_246.astype('float32')) # shape=(12, 6, 4)
func_84_call = mod.get_global_var('func_84')
func_87_call = mutated_mod.get_global_var('func_87')
call_267 = relay.TupleGetItem(func_84_call(relay.reshape(var_243.astype('int16'), [13, 11]), relay.reshape(call_242.astype('int16'), [13, 11]), ), 0)
call_268 = relay.TupleGetItem(func_87_call(relay.reshape(var_243.astype('int16'), [13, 11]), relay.reshape(call_242.astype('int16'), [13, 11]), ), 0)
output = relay.Tuple([bop_228,bop_239,call_242,var_243,call_249,bop_256,uop_264,call_267,])
output2 = relay.Tuple([bop_228,bop_239,call_244,var_243,call_250,bop_256,uop_264,call_268,])
func_269 = relay.Function([var_192,var_224,var_238,var_243,var_255,], output)
mod['func_269'] = func_269
mod = relay.transform.InferType()(mod)
mutated_mod['func_269'] = func_269
mutated_mod = relay.transform.InferType()(mutated_mod)
func_269_call = mutated_mod.get_global_var('func_269')
var_271 = relay.var("var_271", dtype = "int32", shape = (12, 6, 4))#candidate|271|(12, 6, 4)|var|int32
var_272 = relay.var("var_272", dtype = "uint8", shape = (12, 6, 4))#candidate|272|(12, 6, 4)|var|uint8
var_273 = relay.var("var_273", dtype = "uint8", shape = (12, 6, 4))#candidate|273|(12, 6, 4)|var|uint8
var_274 = relay.var("var_274", dtype = "int16", shape = (143,))#candidate|274|(143,)|var|int16
var_275 = relay.var("var_275", dtype = "float64", shape = (12, 6, 4))#candidate|275|(12, 6, 4)|var|float64
call_270 = func_269_call(var_271,var_272,var_273,var_274,var_275,)
output = call_270
func_276 = relay.Function([var_271,var_272,var_273,var_274,var_275,], output)
mutated_mod['func_276'] = func_276
mutated_mod = relay.transform.InferType()(mutated_mod)
var_326 = relay.var("var_326", dtype = "bool", shape = (10, 3))#candidate|326|(10, 3)|var|bool
const_327 = relay.const([[True,False,False],[False,True,False],[False,True,False],[True,True,False],[False,True,False],[True,False,False],[False,False,True],[False,True,False],[True,False,False],[False,False,True]], dtype = "bool")#candidate|327|(10, 3)|const|bool
bop_328 = relay.logical_and(var_326.astype('bool'), relay.reshape(const_327.astype('bool'), relay.shape_of(var_326))) # shape=(10, 3)
func_84_call = mod.get_global_var('func_84')
func_87_call = mutated_mod.get_global_var('func_87')
var_338 = relay.var("var_338", dtype = "int16", shape = (143,))#candidate|338|(143,)|var|int16
call_337 = relay.TupleGetItem(func_84_call(relay.reshape(var_338.astype('int16'), [13, 11]), relay.reshape(var_338.astype('int16'), [13, 11]), ), 0)
call_339 = relay.TupleGetItem(func_87_call(relay.reshape(var_338.astype('int16'), [13, 11]), relay.reshape(var_338.astype('int16'), [13, 11]), ), 0)
var_360 = relay.var("var_360", dtype = "int16", shape = (13, 11))#candidate|360|(13, 11)|var|int16
bop_361 = relay.mod(call_337.astype('float64'), relay.reshape(var_360.astype('float64'), relay.shape_of(call_337))) # shape=(13, 11)
bop_364 = relay.mod(call_339.astype('float64'), relay.reshape(var_360.astype('float64'), relay.shape_of(call_339))) # shape=(13, 11)
uop_366 = relay.atan(bop_328.astype('float32')) # shape=(10, 3)
uop_375 = relay.cosh(uop_366.astype('float32')) # shape=(10, 3)
uop_378 = relay.exp(uop_375.astype('float64')) # shape=(10, 3)
func_269_call = mod.get_global_var('func_269')
func_276_call = mutated_mod.get_global_var('func_276')
const_381 = relay.const([-3,-3,10,-8,-2,-9,-7,-8,-2,-4,10,-8,-1,-10,-2,7,3,-4,3,-4,-6,-7,1,-5,-6,4,1,10,10,-10,-10,-8,7,8,3,7,4,10,1,2,2,-3,6,4,-3,2,1,6,1,-8,-7,4,10,9,-9,9,7,-6,-5,-10,-7,4,2,6,6,-10,10,4,-9,-3,4,-2,4,-3,-8,-9,-7,3,-5,-5,-4,3,8,-7,-2,5,-2,6,-5,-7,4,-6,-1,10,-7,-6,-9,-6,-6,-6,-3,4,-10,-9,1,-7,-4,-8,1,10,3,8,4,-1,-9,9,-1,-7,-5,-5,-4,3,-5,-2,1,2,-7,7,8,-2,-8,9,5,-8,2,-9,8,-4,-5,7,8,1,4,7,-10,7,3,7,1,10,-1,-10,3,9,-10,-5,8,6,-7,-3,3,5,7,-5,7,4,4,-6,7,4,-4,8,6,7,6,-10,-10,-6,-9,5,-6,-6,2,7,9,-9,-2,-8,-10,4,5,-8,1,1,-4,-7,-7,1,5,3,1,1,-10,4,4,-7,-3,3,8,5,-4,4,-3,3,5,4,-7,-3,-5,-8,-6,9,1,-5,-8,2,-1,5,3,4,3,8,-4,-8,8,8,-2,-5,-1,-1,1,2,2,-5,-5,-7,-8,10,-3,-1,-6,-7,5,1,-4,-4,-8,6,4,-6,-2,-9,5,-2,-2,3,1,-6,4,10,-2,1,-3,-9,-10,-7,-10,-3,-8,5,-8,-7,8,3,3,-8,-2,-6], dtype = "int32")#candidate|381|(288,)|const|int32
call_380 = relay.TupleGetItem(func_269_call(relay.reshape(const_381.astype('int32'), [12, 6, 4]), relay.reshape(const_381.astype('uint8'), [12, 6, 4]), relay.reshape(const_381.astype('uint8'), [12, 6, 4]), relay.reshape(var_338.astype('int16'), [143,]), relay.reshape(const_381.astype('float64'), [12, 6, 4]), ), 0)
call_382 = relay.TupleGetItem(func_276_call(relay.reshape(const_381.astype('int32'), [12, 6, 4]), relay.reshape(const_381.astype('uint8'), [12, 6, 4]), relay.reshape(const_381.astype('uint8'), [12, 6, 4]), relay.reshape(var_338.astype('int16'), [143,]), relay.reshape(const_381.astype('float64'), [12, 6, 4]), ), 0)
var_385 = relay.var("var_385", dtype = "float32", shape = (10, 3))#candidate|385|(10, 3)|var|float32
bop_386 = relay.bitwise_and(uop_375.astype('uint16'), relay.reshape(var_385.astype('uint16'), relay.shape_of(uop_375))) # shape=(10, 3)
var_392 = relay.var("var_392", dtype = "float32", shape = (10, 3))#candidate|392|(10, 3)|var|float32
bop_393 = relay.multiply(uop_366.astype('float32'), relay.reshape(var_392.astype('float32'), relay.shape_of(uop_366))) # shape=(10, 3)
bop_396 = relay.maximum(uop_378.astype('uint16'), relay.reshape(bop_393.astype('uint16'), relay.shape_of(uop_378))) # shape=(10, 3)
uop_408 = relay.log10(bop_386.astype('float32')) # shape=(10, 3)
const_414 = relay.const([[9.813779,7.227135,-0.129990],[-0.614661,-1.612960,4.420294],[8.073726,5.645628,8.492469],[-9.131129,-0.011189,8.000201],[1.229281,5.547511,7.415151],[8.071823,0.097423,-0.977511],[6.191698,4.581437,-5.718170],[-7.282705,4.659676,-9.771751],[-6.839180,7.341988,6.118051],[0.190738,1.819243,0.100067]], dtype = "float32")#candidate|414|(10, 3)|const|float32
bop_415 = relay.not_equal(uop_408.astype('bool'), relay.reshape(const_414.astype('bool'), relay.shape_of(uop_408))) # shape=(10, 3)
const_421 = relay.const([[-3.819720,6.130720,4.213064],[-4.700179,9.889071,-9.891327],[-6.917810,3.014159,4.021542],[-5.581486,9.053027,1.530401],[-5.071299,-0.477339,9.303519],[-5.181342,0.243188,-4.106068],[-1.017668,-0.262076,-3.499013],[6.770015,-8.524052,-3.849475],[-9.775831,2.875241,-7.555920],[7.472800,0.958419,-0.447179]], dtype = "float64")#candidate|421|(10, 3)|const|float64
bop_422 = relay.add(uop_378.astype('int16'), relay.reshape(const_421.astype('int16'), relay.shape_of(uop_378))) # shape=(10, 3)
uop_425 = relay.asin(bop_386.astype('float32')) # shape=(10, 3)
output = relay.Tuple([var_338,bop_361,call_380,const_381,bop_396,bop_415,bop_422,uop_425,])
output2 = relay.Tuple([var_338,bop_364,call_382,const_381,bop_396,bop_415,bop_422,uop_425,])
func_429 = relay.Function([var_326,var_338,var_360,var_385,var_392,], output)
mod['func_429'] = func_429
mod = relay.transform.InferType()(mod)
mutated_mod['func_429'] = func_429
mutated_mod = relay.transform.InferType()(mutated_mod)
func_429_call = mutated_mod.get_global_var('func_429')
var_431 = relay.var("var_431", dtype = "bool", shape = (10, 3))#candidate|431|(10, 3)|var|bool
var_432 = relay.var("var_432", dtype = "int16", shape = (143,))#candidate|432|(143,)|var|int16
var_433 = relay.var("var_433", dtype = "int16", shape = (13, 11))#candidate|433|(13, 11)|var|int16
var_434 = relay.var("var_434", dtype = "float32", shape = (10, 3))#candidate|434|(10, 3)|var|float32
var_435 = relay.var("var_435", dtype = "float32", shape = (10, 3))#candidate|435|(10, 3)|var|float32
call_430 = func_429_call(var_431,var_432,var_433,var_434,var_435,)
output = call_430
func_436 = relay.Function([var_431,var_432,var_433,var_434,var_435,], output)
mutated_mod['func_436'] = func_436
mutated_mod = relay.transform.InferType()(mutated_mod)
var_443 = relay.var("var_443", dtype = "float64", shape = (4, 10, 5))#candidate|443|(4, 10, 5)|var|float64
uop_444 = relay.log10(var_443.astype('float64')) # shape=(4, 10, 5)
const_448 = relay.const([[[7.196274,-9.872079,3.215650,-6.589625,2.580273],[0.500982,5.188330,-8.934833,5.953359,-4.815808],[2.980816,2.315215,-1.171573,-7.746631,1.649777],[-4.872027,0.687735,2.293226,4.472282,2.780825],[0.472261,-7.725238,6.270324,4.736623,-3.491157],[-2.441284,-0.339589,-4.423579,7.115191,2.276548],[7.216138,8.281608,-3.728157,9.021984,-1.851356],[-9.617728,2.869824,8.412003,9.536515,1.456570],[-7.868569,-3.025701,5.735624,-8.166496,4.088117],[6.561241,7.624207,5.003618,-4.587686,-4.646359]],[[0.610910,-1.425068,4.332861,-2.264405,-1.886073],[1.205549,3.173463,-4.633422,-7.922673,-5.880935],[-8.962674,-5.290088,-1.559948,6.742767,-2.937072],[-4.132839,-1.825870,-6.799487,-1.454062,6.114641],[9.235951,9.249514,-0.401085,6.872418,-0.058460],[-1.262504,-8.092147,0.352491,6.923653,1.428789],[-1.185158,2.765463,-9.579227,-4.897288,-8.237831],[4.269974,-2.326821,1.209956,-3.096094,-7.062617],[-0.497190,8.816767,4.680448,1.908511,4.352804],[9.620323,-6.317387,1.356810,-2.871749,7.927874]],[[9.730987,-4.247814,-3.945203,-9.735662,0.718006],[6.070980,0.015089,-6.892494,0.566817,0.503612],[-4.499082,-3.415441,-0.095632,1.402195,-3.363719],[7.087238,4.333913,9.857490,2.731361,7.025564],[-3.715005,-8.257266,5.055215,4.079539,3.077937],[-6.465316,7.071615,-3.544722,-9.885010,-2.003569],[2.907433,-3.598299,-2.252823,-9.887816,5.805865],[-1.026442,-3.641423,6.778873,1.791245,0.362433],[-7.029435,-2.653628,-5.464168,8.995332,-4.364460],[6.695266,4.627373,-3.543853,-8.389808,-7.374119]],[[-9.000096,6.620952,6.222792,9.488156,2.449731],[6.340827,-5.821148,9.107179,9.266273,6.762487],[5.599865,7.220767,-5.477973,-6.975690,-7.298587],[-4.425338,8.575958,5.179295,-3.064803,1.066357],[2.216429,4.619139,-8.668379,2.779118,5.884422],[1.188161,3.451218,-3.601587,-2.361667,-9.910970],[4.901528,-1.376369,3.475973,2.677730,-7.760411],[5.870972,-8.990015,0.764760,-4.212641,-4.742596],[5.170599,-1.540851,-1.297580,-7.163623,4.460886],[8.419249,6.576340,9.889781,8.209295,-2.188546]]], dtype = "float64")#candidate|448|(4, 10, 5)|const|float64
bop_449 = relay.bitwise_or(uop_444.astype('uint64'), relay.reshape(const_448.astype('uint64'), relay.shape_of(uop_444))) # shape=(4, 10, 5)
bop_452 = relay.floor_mod(bop_449.astype('float64'), relay.reshape(uop_444.astype('float64'), relay.shape_of(bop_449))) # shape=(4, 10, 5)
bop_458 = relay.logical_and(bop_452.astype('bool'), relay.reshape(uop_444.astype('bool'), relay.shape_of(bop_452))) # shape=(4, 10, 5)
bop_462 = relay.floor_divide(bop_458.astype('float32'), relay.reshape(bop_452.astype('float32'), relay.shape_of(bop_458))) # shape=(4, 10, 5)
output = relay.Tuple([bop_462,])
output2 = relay.Tuple([bop_462,])
func_470 = relay.Function([var_443,], output)
mod['func_470'] = func_470
mod = relay.transform.InferType()(mod)
mutated_mod['func_470'] = func_470
mutated_mod = relay.transform.InferType()(mutated_mod)
var_471 = relay.var("var_471", dtype = "float64", shape = (4, 10, 5))#candidate|471|(4, 10, 5)|var|float64
func_470_call = mutated_mod.get_global_var('func_470')
call_472 = func_470_call(var_471)
output = call_472
func_473 = relay.Function([var_471], output)
mutated_mod['func_473'] = func_473
mutated_mod = relay.transform.InferType()(mutated_mod)
const_624 = relay.const([[-7,-8,-9,1,3,-4],[5,2,2,2,-6,-2],[5,-3,-7,-10,7,7],[-10,-8,8,-1,-7,10],[6,-1,10,5,7,10],[8,2,7,-1,-10,1],[2,2,-4,-9,4,-5],[7,2,9,-10,1,4],[3,-6,-4,-2,-10,6],[-7,-6,-8,7,-2,-9],[-10,-2,3,8,1,9],[-4,2,5,10,4,5],[9,-5,-1,-6,5,2],[1,-2,-6,8,3,-4],[7,-2,-9,2,1,1]], dtype = "uint64")#candidate|624|(15, 6)|const|uint64
var_625 = relay.var("var_625", dtype = "uint64", shape = (15, 6))#candidate|625|(15, 6)|var|uint64
bop_626 = relay.equal(const_624.astype('bool'), relay.reshape(var_625.astype('bool'), relay.shape_of(const_624))) # shape=(15, 6)
bop_631 = relay.less(const_624.astype('bool'), relay.reshape(bop_626.astype('bool'), relay.shape_of(const_624))) # shape=(15, 6)
bop_634 = relay.mod(bop_626.astype('float64'), relay.reshape(bop_631.astype('float64'), relay.shape_of(bop_626))) # shape=(15, 6)
bop_641 = relay.logical_xor(bop_631.astype('uint16'), relay.reshape(bop_634.astype('uint16'), relay.shape_of(bop_631))) # shape=(15, 6)
bop_648 = relay.bitwise_or(bop_626.astype('int64'), relay.reshape(bop_631.astype('int64'), relay.shape_of(bop_626))) # shape=(15, 6)
output = relay.Tuple([bop_641,bop_648,])
output2 = relay.Tuple([bop_641,bop_648,])
func_651 = relay.Function([var_625,], output)
mod['func_651'] = func_651
mod = relay.transform.InferType()(mod)
var_652 = relay.var("var_652", dtype = "uint64", shape = (15, 6))#candidate|652|(15, 6)|var|uint64
output = func_651(var_652)
func_653 = relay.Function([var_652], output)
mutated_mod['func_653'] = func_653
mutated_mod = relay.transform.InferType()(mutated_mod)
const_708 = relay.const([[[-1.988190,-1.038480,-4.205774,7.147386,8.018323],[5.504740,1.503488,-1.939707,-5.306361,8.742611],[-1.126412,-0.878247,7.510108,8.481547,8.862451],[-7.330546,3.385554,5.159018,6.700866,-2.777736],[0.702553,1.240913,-7.019062,9.867866,-9.478602],[-3.573664,-7.474417,-2.835177,-7.626659,-0.240894],[-7.099116,6.033295,6.204908,1.117842,7.319778],[-5.462030,-6.833429,0.979362,3.292953,9.747756],[7.056617,0.678302,1.201810,1.426963,2.531946],[9.128298,-9.891755,2.477391,9.207260,-4.045031],[-9.301575,0.231493,7.182894,7.599616,8.638366],[-4.022990,-4.168869,-9.937838,0.115619,5.366502],[7.833159,-4.112313,4.216676,4.998230,-3.153893]],[[3.398086,-2.730859,9.613898,9.279708,8.843167],[-0.379565,6.318836,9.401861,-5.094733,-8.405693],[6.224626,-1.069526,0.162089,5.261015,-2.673606],[1.501162,3.333684,-1.675340,6.216179,4.669135],[-0.291077,-7.315952,0.668652,3.244967,-8.379266],[5.750514,3.004856,2.784685,-5.828913,0.685980],[-1.098643,-0.340009,3.699398,1.510520,3.187220],[4.966484,-5.357993,4.172388,0.221350,3.903795],[-4.989425,6.080500,-0.718622,4.744885,7.646837],[0.314321,3.427929,2.421377,-7.785001,3.386008],[-9.309825,8.352394,-0.860598,-2.800570,9.509904],[0.190510,-4.936601,3.200629,2.786590,0.379522],[5.480552,-5.493315,6.538126,2.159360,-5.510538]],[[7.600134,4.891423,1.394269,4.583226,3.811804],[8.772691,5.286244,5.917279,-0.369180,8.174274],[-7.743809,-0.254376,1.644225,-8.226871,0.797564],[9.513892,7.431898,7.502941,-2.870860,-9.426407],[2.225348,-2.207157,-1.788714,0.932081,-6.909270],[-0.682827,-5.405627,3.190063,-8.066213,-8.696110],[9.902281,-4.957998,3.560518,9.489785,1.064605],[6.370781,-8.492979,8.621002,-8.785879,8.189570],[3.278860,4.534932,8.505305,4.440230,-5.863994],[5.362522,3.597016,4.695081,-9.150601,-9.590411],[-1.203663,-4.517743,9.571500,7.040010,9.235178],[-3.112352,0.404208,-6.631877,-9.508965,-5.011725],[1.373343,-8.023329,-7.047079,9.495943,0.609737]],[[-9.819439,-8.600159,-9.138106,-4.488100,3.252774],[-6.325277,-5.769545,5.301339,-2.339810,-9.085950],[-9.831776,3.653727,-2.620252,-6.374929,-4.471396],[0.512053,-0.263325,1.219938,3.932613,-9.528568],[2.063428,-4.958067,-7.896001,-7.544384,0.733378],[-3.730602,4.559194,3.687315,5.176808,2.465734],[0.245610,-2.250163,-2.209751,-5.601970,6.453722],[-2.408115,7.088618,-2.441830,-5.905165,9.865509],[1.644063,-7.080986,5.536521,-2.771808,0.442121],[-5.727323,-5.772432,-5.884303,-7.546767,-9.062894],[-3.358393,1.317135,2.752057,0.386668,-3.349970],[9.714418,-7.700882,-0.154206,8.992067,-4.124562],[5.587670,6.005467,6.915270,-9.496143,-2.643067]],[[1.144382,-0.602981,4.120467,-2.040577,4.332113],[2.903943,-0.247754,-3.797621,4.820668,0.404144],[9.590630,-7.959542,-3.248086,-4.266088,7.711150],[5.203408,4.266003,7.238595,2.809037,-5.831738],[0.146244,-4.900330,4.929080,-8.614047,-3.914710],[3.179716,-1.776600,1.134953,0.901897,-3.355445],[-8.013805,-6.268107,3.908184,-4.339606,7.415295],[3.267281,-1.974691,3.754496,4.191726,-6.833979],[3.688447,6.135828,8.459652,2.360441,-1.392701],[-3.086145,8.800325,6.325038,-4.106878,-8.034710],[-0.440775,-9.247436,0.387560,-8.509622,-6.705895],[-8.546770,-4.458565,3.640654,-1.758578,5.148456],[-0.950292,-8.043924,-2.383760,-3.258003,1.293746]]], dtype = "float64")#candidate|708|(5, 13, 5)|const|float64
var_709 = relay.var("var_709", dtype = "float64", shape = (5, 13, 5))#candidate|709|(5, 13, 5)|var|float64
bop_710 = relay.less(const_708.astype('bool'), relay.reshape(var_709.astype('bool'), relay.shape_of(const_708))) # shape=(5, 13, 5)
output = relay.Tuple([bop_710,])
output2 = relay.Tuple([bop_710,])
func_720 = relay.Function([var_709,], output)
mod['func_720'] = func_720
mod = relay.transform.InferType()(mod)
mutated_mod['func_720'] = func_720
mutated_mod = relay.transform.InferType()(mutated_mod)
var_721 = relay.var("var_721", dtype = "float64", shape = (5, 13, 5))#candidate|721|(5, 13, 5)|var|float64
func_720_call = mutated_mod.get_global_var('func_720')
call_722 = func_720_call(var_721)
output = call_722
func_723 = relay.Function([var_721], output)
mutated_mod['func_723'] = func_723
mutated_mod = relay.transform.InferType()(mutated_mod)
var_773 = relay.var("var_773", dtype = "float32", shape = (7, 4, 15))#candidate|773|(7, 4, 15)|var|float32
var_774 = relay.var("var_774", dtype = "float32", shape = (7, 4, 15))#candidate|774|(7, 4, 15)|var|float32
bop_775 = relay.power(var_773.astype('float32'), relay.reshape(var_774.astype('float32'), relay.shape_of(var_773))) # shape=(7, 4, 15)
func_269_call = mod.get_global_var('func_269')
func_276_call = mutated_mod.get_global_var('func_276')
var_780 = relay.var("var_780", dtype = "int32", shape = (288,))#candidate|780|(288,)|var|int32
const_781 = relay.const([-9,2,1,8,-8,-8,1,-6,-3,5,9,-10,5,-4,9,6,10,1,-4,5,-9,1,-2,-8,1,-10,9,-9,-8,-5,-4,-6,3,-4,-6,8,-3,-7,6,-3,-1,7,2,4,9,-10,-2,-7,-4,1,1,9,-10,2,-2,-5,5,2,-10,-9,8,3,-6,7,3,-3,-9,10,-7,-5,7,-3,-2,2,-3,-10,-2,5,3,-9,-1,-1,-3,10,-3,-2,-5,-3,-1,-8,7,7,10,5,-3,-10,-7,5,-1,-3,-1,5,9,2,2,-4,7,8,1,-3,-10,-9,1,8,-2,-6,-3,9,-6,1,3,9,-5,9,10,2,-2,-6,6,7,-4,-1,2,8,10,-8,-5,5,3,7,6,3,2], dtype = "int16")#candidate|781|(143,)|const|int16
call_779 = relay.TupleGetItem(func_269_call(relay.reshape(var_780.astype('int32'), [12, 6, 4]), relay.reshape(var_780.astype('uint8'), [12, 6, 4]), relay.reshape(var_780.astype('uint8'), [12, 6, 4]), relay.reshape(const_781.astype('int16'), [143,]), relay.reshape(var_780.astype('float64'), [12, 6, 4]), ), 0)
call_782 = relay.TupleGetItem(func_276_call(relay.reshape(var_780.astype('int32'), [12, 6, 4]), relay.reshape(var_780.astype('uint8'), [12, 6, 4]), relay.reshape(var_780.astype('uint8'), [12, 6, 4]), relay.reshape(const_781.astype('int16'), [143,]), relay.reshape(var_780.astype('float64'), [12, 6, 4]), ), 0)
func_429_call = mod.get_global_var('func_429')
func_436_call = mutated_mod.get_global_var('func_436')
var_784 = relay.var("var_784", dtype = "bool", shape = (30,))#candidate|784|(30,)|var|bool
call_783 = relay.TupleGetItem(func_429_call(relay.reshape(var_784.astype('bool'), [10, 3]), relay.reshape(const_781.astype('int16'), [143,]), relay.reshape(const_781.astype('int16'), [13, 11]), relay.reshape(var_784.astype('float32'), [10, 3]), relay.reshape(var_784.astype('float32'), [10, 3]), ), 4)
call_785 = relay.TupleGetItem(func_436_call(relay.reshape(var_784.astype('bool'), [10, 3]), relay.reshape(const_781.astype('int16'), [143,]), relay.reshape(const_781.astype('int16'), [13, 11]), relay.reshape(var_784.astype('float32'), [10, 3]), relay.reshape(var_784.astype('float32'), [10, 3]), ), 4)
output = relay.Tuple([bop_775,call_779,var_780,const_781,call_783,var_784,])
output2 = relay.Tuple([bop_775,call_782,var_780,const_781,call_785,var_784,])
func_789 = relay.Function([var_773,var_774,var_780,var_784,], output)
mod['func_789'] = func_789
mod = relay.transform.InferType()(mod)
mutated_mod['func_789'] = func_789
mutated_mod = relay.transform.InferType()(mutated_mod)
func_789_call = mutated_mod.get_global_var('func_789')
var_791 = relay.var("var_791", dtype = "float32", shape = (7, 4, 15))#candidate|791|(7, 4, 15)|var|float32
var_792 = relay.var("var_792", dtype = "float32", shape = (7, 4, 15))#candidate|792|(7, 4, 15)|var|float32
var_793 = relay.var("var_793", dtype = "int32", shape = (288,))#candidate|793|(288,)|var|int32
var_794 = relay.var("var_794", dtype = "bool", shape = (30,))#candidate|794|(30,)|var|bool
call_790 = func_789_call(var_791,var_792,var_793,var_794,)
output = call_790
func_795 = relay.Function([var_791,var_792,var_793,var_794,], output)
mutated_mod['func_795'] = func_795
mutated_mod = relay.transform.InferType()(mutated_mod)
var_816 = relay.var("var_816", dtype = "float32", shape = (14, 4))#candidate|816|(14, 4)|var|float32
uop_817 = relay.cos(var_816.astype('float32')) # shape=(14, 4)
uop_821 = relay.sinh(var_816.astype('float32')) # shape=(14, 4)
bop_824 = relay.bitwise_xor(uop_821.astype('uint8'), relay.reshape(var_816.astype('uint8'), relay.shape_of(uop_821))) # shape=(14, 4)
bop_827 = relay.logical_xor(var_816.astype('uint32'), relay.reshape(uop_817.astype('uint32'), relay.shape_of(var_816))) # shape=(14, 4)
var_838 = relay.var("var_838", dtype = "float32", shape = (14, 4))#candidate|838|(14, 4)|var|float32
bop_839 = relay.floor_divide(uop_817.astype('float64'), relay.reshape(var_838.astype('float64'), relay.shape_of(uop_817))) # shape=(14, 4)
uop_847 = relay.asin(uop_817.astype('float64')) # shape=(14, 4)
bop_852 = relay.mod(uop_847.astype('float32'), relay.reshape(uop_817.astype('float32'), relay.shape_of(uop_847))) # shape=(14, 4)
output = relay.Tuple([bop_824,bop_827,bop_839,bop_852,])
output2 = relay.Tuple([bop_824,bop_827,bop_839,bop_852,])
func_862 = relay.Function([var_816,var_838,], output)
mod['func_862'] = func_862
mod = relay.transform.InferType()(mod)
mutated_mod['func_862'] = func_862
mutated_mod = relay.transform.InferType()(mutated_mod)
func_862_call = mutated_mod.get_global_var('func_862')
var_864 = relay.var("var_864", dtype = "float32", shape = (14, 4))#candidate|864|(14, 4)|var|float32
var_865 = relay.var("var_865", dtype = "float32", shape = (14, 4))#candidate|865|(14, 4)|var|float32
call_863 = func_862_call(var_864,var_865,)
output = call_863
func_866 = relay.Function([var_864,var_865,], output)
mutated_mod['func_866'] = func_866
mutated_mod = relay.transform.InferType()(mutated_mod)
var_875 = relay.var("var_875", dtype = "float32", shape = (1, 6, 4))#candidate|875|(1, 6, 4)|var|float32
uop_876 = relay.rsqrt(var_875.astype('float32')) # shape=(1, 6, 4)
output = relay.Tuple([uop_876,])
output2 = relay.Tuple([uop_876,])
func_878 = relay.Function([var_875,], output)
mod['func_878'] = func_878
mod = relay.transform.InferType()(mod)
mutated_mod['func_878'] = func_878
mutated_mod = relay.transform.InferType()(mutated_mod)
var_879 = relay.var("var_879", dtype = "float32", shape = (1, 6, 4))#candidate|879|(1, 6, 4)|var|float32
func_878_call = mutated_mod.get_global_var('func_878')
call_880 = func_878_call(var_879)
output = call_880
func_881 = relay.Function([var_879], output)
mutated_mod['func_881'] = func_881
mutated_mod = relay.transform.InferType()(mutated_mod)
var_926 = relay.var("var_926", dtype = "int64", shape = ())#candidate|926|()|var|int64
var_927 = relay.var("var_927", dtype = "int64", shape = (8, 9, 7))#candidate|927|(8, 9, 7)|var|int64
bop_928 = relay.logical_xor(var_926.astype('int64'), var_927.astype('int64')) # shape=(8, 9, 7)
func_429_call = mod.get_global_var('func_429')
func_436_call = mutated_mod.get_global_var('func_436')
const_932 = relay.const([True,False,False,True,True,True,False,True,False,True,False,True,True,True,True,False,False,False,True,True,True,False,True,True,True,False,True,True,True,True], dtype = "bool")#candidate|932|(30,)|const|bool
const_933 = relay.const([[1,4,6,1,1,-2,4,-4,6,-3,-8],[5,-6,2,6,1,-4,5,6,-1,3,4],[9,-6,4,9,2,7,7,-6,-9,-10,-2],[4,3,7,-9,3,2,2,-9,-10,-6,-1],[-6,8,3,-10,-5,5,10,-5,-4,8,10],[-4,10,9,10,-3,-8,4,8,-8,-6,2],[6,4,6,8,1,-4,-3,-9,5,-9,-2],[4,-7,-5,-5,9,-2,1,-7,3,-10,-9],[2,-6,-1,-1,10,-4,-8,-9,4,6,3],[6,-8,-5,8,1,-3,8,-4,8,-4,10],[-1,-9,-5,3,-2,-7,7,-7,6,-6,7],[-5,-3,-4,7,-3,2,-5,4,-6,-6,7],[3,-10,-3,1,10,-2,1,4,-1,8,6]], dtype = "int16")#candidate|933|(13, 11)|const|int16
call_931 = relay.TupleGetItem(func_429_call(relay.reshape(const_932.astype('bool'), [10, 3]), relay.reshape(const_933.astype('int16'), [143,]), relay.reshape(const_933.astype('int16'), [13, 11]), relay.reshape(const_932.astype('float32'), [10, 3]), relay.reshape(const_932.astype('float32'), [10, 3]), ), 3)
call_934 = relay.TupleGetItem(func_436_call(relay.reshape(const_932.astype('bool'), [10, 3]), relay.reshape(const_933.astype('int16'), [143,]), relay.reshape(const_933.astype('int16'), [13, 11]), relay.reshape(const_932.astype('float32'), [10, 3]), relay.reshape(const_932.astype('float32'), [10, 3]), ), 3)
output = relay.Tuple([bop_928,call_931,const_932,const_933,])
output2 = relay.Tuple([bop_928,call_934,const_932,const_933,])
func_940 = relay.Function([var_926,var_927,], output)
mod['func_940'] = func_940
mod = relay.transform.InferType()(mod)
var_941 = relay.var("var_941", dtype = "int64", shape = ())#candidate|941|()|var|int64
var_942 = relay.var("var_942", dtype = "int64", shape = (8, 9, 7))#candidate|942|(8, 9, 7)|var|int64
output = func_940(var_941,var_942,)
func_943 = relay.Function([var_941,var_942,], output)
mutated_mod['func_943'] = func_943
mutated_mod = relay.transform.InferType()(mutated_mod)
var_966 = relay.var("var_966", dtype = "float32", shape = (14, 2, 7))#candidate|966|(14, 2, 7)|var|float32
uop_967 = relay.sqrt(var_966.astype('float32')) # shape=(14, 2, 7)
bop_969 = relay.divide(uop_967.astype('float32'), relay.reshape(var_966.astype('float32'), relay.shape_of(uop_967))) # shape=(14, 2, 7)
uop_972 = relay.asinh(uop_967.astype('float64')) # shape=(14, 2, 7)
bop_974 = relay.equal(uop_972.astype('bool'), relay.reshape(uop_967.astype('bool'), relay.shape_of(uop_972))) # shape=(14, 2, 7)
bop_977 = relay.power(bop_974.astype('float64'), relay.reshape(uop_972.astype('float64'), relay.shape_of(bop_974))) # shape=(14, 2, 7)
bop_985 = relay.greater(uop_972.astype('bool'), relay.reshape(uop_967.astype('bool'), relay.shape_of(uop_972))) # shape=(14, 2, 7)
func_470_call = mod.get_global_var('func_470')
func_473_call = mutated_mod.get_global_var('func_473')
var_997 = relay.var("var_997", dtype = "float64", shape = (200,))#candidate|997|(200,)|var|float64
call_996 = relay.TupleGetItem(func_470_call(relay.reshape(var_997.astype('float64'), [4, 10, 5])), 0)
call_998 = relay.TupleGetItem(func_473_call(relay.reshape(var_997.astype('float64'), [4, 10, 5])), 0)
bop_999 = relay.logical_xor(bop_985.astype('uint16'), relay.reshape(uop_972.astype('uint16'), relay.shape_of(bop_985))) # shape=(14, 2, 7)
uop_1004 = relay.log2(bop_969.astype('float64')) # shape=(14, 2, 7)
bop_1011 = relay.logical_and(uop_967.astype('bool'), relay.reshape(uop_1004.astype('bool'), relay.shape_of(uop_967))) # shape=(14, 2, 7)
uop_1014 = relay.sin(bop_969.astype('float32')) # shape=(14, 2, 7)
var_1017 = relay.var("var_1017", dtype = "uint16", shape = (14, 2, 7))#candidate|1017|(14, 2, 7)|var|uint16
bop_1018 = relay.subtract(bop_999.astype('int32'), relay.reshape(var_1017.astype('int32'), relay.shape_of(bop_999))) # shape=(14, 2, 7)
var_1043 = relay.var("var_1043", dtype = "float64", shape = (14, 2, 7))#candidate|1043|(14, 2, 7)|var|float64
bop_1044 = relay.greater_equal(uop_972.astype('bool'), relay.reshape(var_1043.astype('bool'), relay.shape_of(uop_972))) # shape=(14, 2, 7)
var_1047 = relay.var("var_1047", dtype = "float32", shape = (14, 2, 7))#candidate|1047|(14, 2, 7)|var|float32
bop_1048 = relay.floor_mod(uop_967.astype('float32'), relay.reshape(var_1047.astype('float32'), relay.shape_of(uop_967))) # shape=(14, 2, 7)
output = relay.Tuple([bop_977,call_996,var_997,bop_1011,uop_1014,bop_1018,bop_1044,bop_1048,])
output2 = relay.Tuple([bop_977,call_998,var_997,bop_1011,uop_1014,bop_1018,bop_1044,bop_1048,])
func_1059 = relay.Function([var_966,var_997,var_1017,var_1043,var_1047,], output)
mod['func_1059'] = func_1059
mod = relay.transform.InferType()(mod)
mutated_mod['func_1059'] = func_1059
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1059_call = mutated_mod.get_global_var('func_1059')
var_1061 = relay.var("var_1061", dtype = "float32", shape = (14, 2, 7))#candidate|1061|(14, 2, 7)|var|float32
var_1062 = relay.var("var_1062", dtype = "float64", shape = (200,))#candidate|1062|(200,)|var|float64
var_1063 = relay.var("var_1063", dtype = "uint16", shape = (14, 2, 7))#candidate|1063|(14, 2, 7)|var|uint16
var_1064 = relay.var("var_1064", dtype = "float64", shape = (14, 2, 7))#candidate|1064|(14, 2, 7)|var|float64
var_1065 = relay.var("var_1065", dtype = "float32", shape = (14, 2, 7))#candidate|1065|(14, 2, 7)|var|float32
call_1060 = func_1059_call(var_1061,var_1062,var_1063,var_1064,var_1065,)
output = call_1060
func_1066 = relay.Function([var_1061,var_1062,var_1063,var_1064,var_1065,], output)
mutated_mod['func_1066'] = func_1066
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1112 = relay.var("var_1112", dtype = "uint32", shape = ())#candidate|1112|()|var|uint32
var_1113 = relay.var("var_1113", dtype = "uint32", shape = (6, 15))#candidate|1113|(6, 15)|var|uint32
bop_1114 = relay.greater_equal(var_1112.astype('bool'), var_1113.astype('bool')) # shape=(6, 15)
output = relay.Tuple([bop_1114,])
output2 = relay.Tuple([bop_1114,])
func_1117 = relay.Function([var_1112,var_1113,], output)
mod['func_1117'] = func_1117
mod = relay.transform.InferType()(mod)
mutated_mod['func_1117'] = func_1117
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1117_call = mutated_mod.get_global_var('func_1117')
var_1119 = relay.var("var_1119", dtype = "uint32", shape = ())#candidate|1119|()|var|uint32
var_1120 = relay.var("var_1120", dtype = "uint32", shape = (6, 15))#candidate|1120|(6, 15)|var|uint32
call_1118 = func_1117_call(var_1119,var_1120,)
output = call_1118
func_1121 = relay.Function([var_1119,var_1120,], output)
mutated_mod['func_1121'] = func_1121
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1156 = relay.var("var_1156", dtype = "float32", shape = (4, 3, 6))#candidate|1156|(4, 3, 6)|var|float32
uop_1157 = relay.log(var_1156.astype('float32')) # shape=(4, 3, 6)
func_429_call = mod.get_global_var('func_429')
func_436_call = mutated_mod.get_global_var('func_436')
const_1162 = relay.const([[False,False,False,False,False,True],[True,False,False,False,False,True],[True,False,True,True,False,True],[True,True,False,True,True,True],[False,True,True,True,False,False]], dtype = "bool")#candidate|1162|(5, 6)|const|bool
const_1163 = relay.const([2,-10,5,4,-2,9,-3,4,1,-6,5,1,-7,10,9,-7,-5,7,5,-2,-6,-8,-3,-10,-8,-1,6,-5,-5,8,4,6,-2,-1,-4,-3,7,-9,6,7,8,9,3,7,-10,-3,-10,-6,-4,9,-2,-8,4,1,7,-5,9,-8,-9,10,10,4,4,2,8,6,4,1,3,9,2,-4,-10,-3,9,-6,7,-1,-1,9,-4,1,10,5,8,5,8,2,4,-6,-7,4,-4,6,-7,9,1,9,-4,9,8,3,-3,-5,8,-2,6,2,-2,-7,3,-10,6,10,-4,3,10,8,-10,-2,4,7,3,-10,-6,-4,5,-7,-5,-2,-3,9,-2,9,-9,2,9,3,-7,-9,-2,-5,-8], dtype = "int16")#candidate|1163|(143,)|const|int16
call_1161 = relay.TupleGetItem(func_429_call(relay.reshape(const_1162.astype('bool'), [10, 3]), relay.reshape(const_1163.astype('int16'), [143,]), relay.reshape(const_1163.astype('int16'), [13, 11]), relay.reshape(const_1162.astype('float32'), [10, 3]), relay.reshape(const_1162.astype('float32'), [10, 3]), ), 4)
call_1164 = relay.TupleGetItem(func_436_call(relay.reshape(const_1162.astype('bool'), [10, 3]), relay.reshape(const_1163.astype('int16'), [143,]), relay.reshape(const_1163.astype('int16'), [13, 11]), relay.reshape(const_1162.astype('float32'), [10, 3]), relay.reshape(const_1162.astype('float32'), [10, 3]), ), 4)
output = relay.Tuple([uop_1157,call_1161,const_1162,const_1163,])
output2 = relay.Tuple([uop_1157,call_1164,const_1162,const_1163,])
func_1165 = relay.Function([var_1156,], output)
mod['func_1165'] = func_1165
mod = relay.transform.InferType()(mod)
var_1166 = relay.var("var_1166", dtype = "float32", shape = (4, 3, 6))#candidate|1166|(4, 3, 6)|var|float32
output = func_1165(var_1166)
func_1167 = relay.Function([var_1166], output)
mutated_mod['func_1167'] = func_1167
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1240 = relay.var("var_1240", dtype = "float32", shape = (1,))#candidate|1240|(1,)|var|float32
uop_1241 = relay.log(var_1240.astype('float32')) # shape=(1,)
func_862_call = mod.get_global_var('func_862')
func_866_call = mutated_mod.get_global_var('func_866')
const_1247 = relay.const([[-0.269200,-7.933338,-1.756965,9.523110,-4.862873,3.006436,-1.220689,-7.362940,-4.866177,-8.497589,-6.861745,5.434535,-2.769170,-3.094979,1.577864,4.138772,6.368711,2.891953,0.231115,-7.262362,-5.806539,-0.052091,-6.095165,5.888061,2.775134,-5.192224,6.259023,-5.446240],[-3.866798,3.817291,-5.586921,-1.043320,-2.476062,-2.936528,-4.209875,5.843003,-8.656743,2.712703,-7.295511,8.603629,-7.798007,-1.790584,9.319438,8.020107,1.231106,-9.944923,7.271533,4.789248,-8.586520,5.345789,-3.618795,2.620737,7.641840,3.702837,0.488361,1.084425]], dtype = "float32")#candidate|1247|(2, 28)|const|float32
call_1246 = relay.TupleGetItem(func_862_call(relay.reshape(const_1247.astype('float32'), [14, 4]), relay.reshape(const_1247.astype('float32'), [14, 4]), ), 3)
call_1248 = relay.TupleGetItem(func_866_call(relay.reshape(const_1247.astype('float32'), [14, 4]), relay.reshape(const_1247.astype('float32'), [14, 4]), ), 3)
output = relay.Tuple([uop_1241,call_1246,const_1247,])
output2 = relay.Tuple([uop_1241,call_1248,const_1247,])
func_1251 = relay.Function([var_1240,], output)
mod['func_1251'] = func_1251
mod = relay.transform.InferType()(mod)
mutated_mod['func_1251'] = func_1251
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1252 = relay.var("var_1252", dtype = "float32", shape = (1,))#candidate|1252|(1,)|var|float32
func_1251_call = mutated_mod.get_global_var('func_1251')
call_1253 = func_1251_call(var_1252)
output = call_1253
func_1254 = relay.Function([var_1252], output)
mutated_mod['func_1254'] = func_1254
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1317 = relay.var("var_1317", dtype = "int32", shape = ())#candidate|1317|()|var|int32
var_1318 = relay.var("var_1318", dtype = "int32", shape = (8, 8, 1))#candidate|1318|(8, 8, 1)|var|int32
bop_1319 = relay.bitwise_and(var_1317.astype('int32'), var_1318.astype('int32')) # shape=(8, 8, 1)
output = relay.Tuple([bop_1319,])
output2 = relay.Tuple([bop_1319,])
func_1324 = relay.Function([var_1317,var_1318,], output)
mod['func_1324'] = func_1324
mod = relay.transform.InferType()(mod)
mutated_mod['func_1324'] = func_1324
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1324_call = mutated_mod.get_global_var('func_1324')
var_1326 = relay.var("var_1326", dtype = "int32", shape = ())#candidate|1326|()|var|int32
var_1327 = relay.var("var_1327", dtype = "int32", shape = (8, 8, 1))#candidate|1327|(8, 8, 1)|var|int32
call_1325 = func_1324_call(var_1326,var_1327,)
output = call_1325
func_1328 = relay.Function([var_1326,var_1327,], output)
mutated_mod['func_1328'] = func_1328
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1344 = relay.var("var_1344", dtype = "float64", shape = (2, 1))#candidate|1344|(2, 1)|var|float64
const_1345 = relay.const([[5.108552,5.914302,-5.627046,7.469867,-1.708146,-7.014088,8.769624],[-8.412927,9.960866,-6.020720,-0.501669,6.695802,1.916566,3.104768]], dtype = "float64")#candidate|1345|(2, 7)|const|float64
bop_1346 = relay.floor_divide(var_1344.astype('float64'), const_1345.astype('float64')) # shape=(2, 7)
output = bop_1346
output2 = bop_1346
func_1355 = relay.Function([var_1344,], output)
mod['func_1355'] = func_1355
mod = relay.transform.InferType()(mod)
mutated_mod['func_1355'] = func_1355
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1356 = relay.var("var_1356", dtype = "float64", shape = (2, 1))#candidate|1356|(2, 1)|var|float64
func_1355_call = mutated_mod.get_global_var('func_1355')
call_1357 = func_1355_call(var_1356)
output = call_1357
func_1358 = relay.Function([var_1356], output)
mutated_mod['func_1358'] = func_1358
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1465 = relay.var("var_1465", dtype = "float64", shape = (5, 11, 1))#candidate|1465|(5, 11, 1)|var|float64
uop_1466 = relay.atanh(var_1465.astype('float64')) # shape=(5, 11, 1)
const_1470 = relay.const([[[-6.081301,-6.161178,-9.213400,-6.723813,-4.989278,-8.117123,-7.316257,4.831526,8.926896,-4.681554,8.698538,0.993108],[-8.926752,4.511190,-7.111180,5.477823,-3.739032,-0.801031,-0.784156,-9.512554,-2.479756,0.707386,1.595144,2.048017],[-7.490045,2.662484,-4.584532,-9.844828,5.965664,0.456083,0.346984,9.775791,-4.583644,-0.174870,-9.822317,-1.728486],[0.310442,3.782692,-3.808577,7.462486,-9.421963,-8.922360,-6.089106,7.507884,-0.299704,5.985427,-9.993361,1.010117],[-5.976126,9.305169,-9.753848,-3.649477,0.216358,-7.429521,-0.371747,-4.323647,8.341512,2.231181,-2.774518,0.393379],[-8.586069,-8.154057,0.370641,2.742400,6.628832,-8.648529,-8.172562,4.591461,-6.828833,0.003473,3.182907,-3.635734],[-8.844505,-1.206470,-0.053955,4.211707,-8.018074,7.889554,1.922176,-2.948867,0.270197,-5.057660,0.797077,-1.876200],[-9.704248,-9.854835,9.726287,-9.210194,-2.700995,-3.366470,-0.506482,7.457783,3.354568,4.696245,-9.544349,5.367731],[-4.312896,3.105428,0.355755,-2.740245,-9.629943,-5.500184,3.743993,9.326209,3.488319,-5.244365,-8.148400,0.954468],[0.946592,-2.745335,-2.767521,9.601891,-9.408200,-4.243794,-8.293824,8.649464,4.254838,1.981295,-3.128154,-6.764659],[-2.056301,0.044087,4.023205,5.464161,8.202204,-9.585906,6.444989,1.194266,-1.702038,-1.495652,-9.455008,5.758814]],[[6.354033,-9.723534,-8.241516,5.185195,6.661303,9.692590,8.301983,9.672416,-7.205246,-3.426377,-9.587407,5.179768],[3.060697,7.382437,6.879999,4.220741,3.353682,-5.992582,0.631517,5.560878,5.133228,5.201718,6.705907,7.567410],[-4.895908,3.310394,6.530126,3.966831,-4.247596,-2.625567,-6.651048,-4.384926,1.257839,9.286353,7.787989,-8.711752],[3.485555,3.645439,-6.227270,-6.492732,0.236006,3.724090,7.686748,5.249480,-2.548475,6.491421,-2.341584,-2.231326],[9.681994,-9.126487,-8.833052,4.755370,-1.923214,-1.655898,5.777057,6.955148,0.648732,-0.542888,7.601374,-0.269325],[3.191137,-5.581934,-2.801551,-1.274179,-0.750386,-5.566314,7.394343,-8.594629,-5.169023,-6.376891,-1.496427,-6.092283],[9.288224,-3.379365,5.873007,-9.944522,-1.445315,-4.562470,6.505748,-3.860635,4.018871,0.940898,-7.202304,-2.038371],[-2.556291,6.599202,-3.750812,-4.926738,-1.636271,-3.376293,-1.057616,-4.679855,-8.157273,6.906459,6.216533,9.462534],[-6.586810,9.048777,2.760897,3.897456,4.854422,-3.153142,0.756195,0.984426,0.062785,5.151629,-3.022124,-8.803522],[-9.643700,-4.733063,-9.090075,-4.587046,-6.062105,3.764921,4.743666,-3.737364,4.400282,-8.255967,3.500369,-5.930877],[0.134225,-8.015719,-1.150694,-3.452144,-3.681834,-7.447215,9.749475,3.678117,6.417888,9.456028,-4.400450,2.519252]],[[-5.691721,-9.862559,9.651275,-4.450214,-2.423547,-8.108996,9.340768,2.156601,8.711281,-9.620856,-8.005213,5.316222],[1.845318,-1.726151,-1.368536,3.386534,9.237155,-9.918278,8.739416,2.485946,-7.967827,-3.673096,0.056095,4.768118],[9.666017,8.497518,-1.812252,4.936836,6.561181,3.664295,7.225877,-7.077018,-4.659152,5.522810,6.225079,3.096743],[-2.870232,6.963070,4.598457,0.995832,9.882248,-6.777421,9.072186,5.536752,1.651675,-4.350592,6.778895,2.545841],[7.598821,9.469040,2.251796,-7.503224,-1.608732,1.107149,9.199307,2.285971,0.062082,-8.730902,0.520202,9.992470],[-2.767173,-3.579812,-3.983875,4.812236,-4.601983,-1.556574,5.652218,-2.015788,8.561240,6.820201,-3.724315,7.294815],[-4.088256,-1.940013,7.501040,-3.726815,0.118374,1.824067,-9.316052,7.758228,-5.537554,-4.923601,0.379303,1.260504],[4.486296,2.099378,9.504799,3.080455,-4.734813,-6.050360,-1.189096,2.942535,4.872878,6.527374,7.182300,-3.694641],[5.180614,5.730831,5.275701,0.378245,5.599953,-3.211191,7.365727,-3.777213,1.246188,5.395132,9.488012,7.914636],[5.409617,-7.310115,-0.774388,-1.661688,9.030183,-4.292733,9.096222,8.792589,-6.057677,0.532518,-0.220698,4.910916],[-3.249877,3.437934,6.350789,-2.068313,8.585847,0.854685,-2.480608,4.954107,2.046648,-5.778944,1.954970,4.135530]],[[-7.530238,5.603257,-0.030733,-6.783476,2.964015,-6.390862,2.285832,-4.236767,0.448946,-8.126075,1.946473,2.207832],[6.183273,8.183840,-3.833850,-8.769094,-5.214494,2.863135,-5.803837,2.024276,4.782233,-9.060114,1.746162,8.177834],[-2.689961,-8.594239,-7.299106,-7.766482,0.661326,-3.639647,-5.166282,-3.717635,8.044624,7.432672,-2.368473,6.929571],[-6.641063,-5.705606,-2.166039,-6.850992,2.300723,9.476824,-1.584169,4.990723,4.845341,5.155865,-9.277700,-4.893041],[5.238275,7.661708,1.402766,4.530935,-6.661307,0.148008,-7.561473,7.227560,-8.151031,5.120591,-9.853228,-2.956101],[-6.253218,5.785860,-1.964776,-1.534929,7.474304,3.142864,8.745040,-3.956287,-8.351880,-5.247984,4.481225,-6.123137],[-3.374557,-6.999953,-3.986945,6.894682,8.386517,5.517366,-2.131107,5.516579,3.531576,-1.162714,-3.163112,-4.441212],[-5.163118,3.382996,-9.710121,8.782403,2.762856,7.551350,0.377081,-9.899042,5.225681,-8.962184,-3.545993,4.592111],[-8.593225,-4.952151,-1.922120,0.662459,4.659489,-7.369334,-1.348561,4.542625,1.147321,6.795259,2.965466,7.073120],[8.968515,-2.831137,-1.408378,6.847319,-6.583164,0.662254,-3.801694,2.838674,8.645374,4.170665,-5.584463,0.977375],[5.523312,-0.551938,-1.037175,4.803935,8.288046,9.479835,7.755821,-8.726399,-1.937223,-5.675867,-2.202215,4.493493]],[[0.673560,0.458371,3.576463,9.840416,1.558129,1.626275,-7.782208,9.025794,-6.038332,3.239502,5.011915,5.947904],[6.897615,-1.691992,-7.598293,8.771523,-5.394498,5.945403,1.975260,-0.858143,-6.079445,-2.969346,-4.787213,-5.404292],[8.171287,-1.683023,7.620331,-6.103843,9.114802,-7.764735,-4.921603,9.769370,-1.261662,-1.266290,3.211913,-3.801354],[-4.933487,-0.231587,-3.689972,-8.387438,-5.793838,-6.437689,-1.493169,-7.749100,0.360300,9.358247,-5.715713,-1.442093],[1.843680,-5.997192,6.356600,-5.533756,-2.626447,-7.728044,0.307676,-8.789093,7.056085,6.262445,7.656856,2.979382],[-3.847888,2.897990,2.616233,3.545365,-3.147436,0.059171,-0.324145,-5.671763,6.053740,3.667428,2.924078,-4.215674],[-7.130641,-4.791003,-8.346791,-1.149001,0.525118,1.372383,8.004905,2.698029,0.047078,9.308753,0.827646,-2.461081],[-9.026629,1.996628,1.918917,-9.307765,-1.055872,6.500630,-3.821157,9.869761,1.333080,-7.746639,-5.356855,3.390501],[4.914030,5.911610,-9.503767,0.745504,-5.621813,-3.709856,-4.298014,4.961066,5.268855,2.011466,8.941269,4.326605],[-9.164527,0.258795,-3.324390,-9.684517,-2.482958,-6.996557,3.088050,7.254567,-5.171989,1.134962,9.169560,6.910205],[8.374135,-5.071195,-9.509320,-6.458030,-3.222788,1.909204,-5.761190,-3.174766,6.478221,8.397183,-8.850975,-7.344284]]], dtype = "float64")#candidate|1470|(5, 11, 12)|const|float64
bop_1471 = relay.logical_or(uop_1466.astype('bool'), const_1470.astype('bool')) # shape=(5, 11, 12)
bop_1474 = relay.maximum(bop_1471.astype('uint32'), relay.reshape(const_1470.astype('uint32'), relay.shape_of(bop_1471))) # shape=(5, 11, 12)
func_878_call = mod.get_global_var('func_878')
func_881_call = mutated_mod.get_global_var('func_881')
var_1478 = relay.var("var_1478", dtype = "float32", shape = (24,))#candidate|1478|(24,)|var|float32
call_1477 = relay.TupleGetItem(func_878_call(relay.reshape(var_1478.astype('float32'), [1, 6, 4])), 0)
call_1479 = relay.TupleGetItem(func_881_call(relay.reshape(var_1478.astype('float32'), [1, 6, 4])), 0)
bop_1487 = relay.not_equal(var_1478.astype('bool'), uop_1466.astype('bool')) # shape=(5, 11, 24)
bop_1498 = relay.bitwise_and(bop_1471.astype('int64'), relay.reshape(bop_1474.astype('int64'), relay.shape_of(bop_1471))) # shape=(5, 11, 12)
bop_1507 = relay.subtract(var_1465.astype('int8'), relay.reshape(uop_1466.astype('int8'), relay.shape_of(var_1465))) # shape=(5, 11, 1)
bop_1513 = relay.not_equal(bop_1507.astype('bool'), const_1470.astype('bool')) # shape=(5, 11, 12)
const_1516 = relay.const([[[3,10,2,-5,9,5,-1,-8,7,3,-6,-3],[-5,-7,9,8,10,8,-7,8,-9,6,6,-6],[1,-9,6,4,-6,-2,8,9,-10,-7,9,3],[-8,9,4,7,7,-2,7,4,10,-10,-8,6],[5,-4,-1,-6,7,5,3,7,-7,2,-9,-7],[4,7,6,9,9,8,10,8,-1,1,4,-3],[8,-7,4,6,-3,10,-8,-1,-2,-6,-2,4],[-7,9,-4,4,-10,1,-7,3,-1,9,4,8],[-8,-10,9,6,-8,9,7,-4,-6,-2,1,3],[10,3,2,10,4,-4,4,8,7,6,1,-3],[-4,-10,-2,-8,4,4,3,-8,4,1,-2,-1]],[[-6,-4,-6,4,4,6,-6,8,-8,-6,6,1],[7,-1,-2,-7,1,7,-8,9,6,8,2,4],[8,-2,-5,6,6,1,2,-8,1,2,6,-1],[-7,-6,3,-8,1,5,4,8,-3,-7,8,7],[1,-9,1,1,7,4,7,-10,3,-4,1,8],[-10,-6,-2,-8,-4,-10,-6,-10,8,2,-9,4],[10,-2,10,2,7,-8,2,-10,2,-1,2,-1],[9,-6,6,2,-7,-4,4,-9,-3,-8,5,-3],[-9,-7,-8,-1,-2,7,8,3,-4,4,-4,-9],[-10,-3,-8,-1,-1,-9,-7,-7,-9,6,-4,5],[-5,-7,2,5,6,6,-4,-2,7,7,-10,-10]],[[9,-6,-4,-4,-10,9,9,-5,-2,1,-1,-5],[2,2,-8,-3,-8,-1,-2,3,-6,-5,6,1],[4,-8,-5,-1,6,-3,6,1,-6,10,9,-9],[8,9,-7,-6,4,-4,-8,8,-10,10,6,4],[6,-10,-7,-8,-5,1,-9,1,-2,2,1,3],[10,9,-5,-8,7,5,-6,-1,4,-3,10,8],[-10,-9,2,-4,7,8,6,-4,3,7,4,10],[-1,-1,-8,10,-9,1,-4,6,10,-3,10,-9],[7,-2,-5,9,5,4,8,3,6,-3,-9,-2],[-2,5,-4,-7,-10,4,-1,-1,2,2,-3,1],[9,2,10,-6,2,-6,-6,9,-10,-3,-5,-9]],[[-2,-1,8,-6,4,4,3,8,-1,-2,-7,-5],[4,-2,-2,8,-4,3,-10,10,7,2,3,-3],[-6,3,6,2,6,8,5,-6,-6,-2,5,2],[5,2,-6,4,10,5,7,-3,-1,3,-1,-6],[5,7,7,4,5,-2,2,1,5,-5,5,-5],[3,2,-4,-2,4,3,-9,7,-4,5,5,6],[-3,-2,3,3,-3,1,-3,8,-4,-3,7,-9],[9,7,-1,3,7,10,-5,7,10,-7,5,-2],[-8,6,-5,8,-8,-5,9,-2,-7,8,7,-8],[-5,10,2,-1,-7,6,9,-3,9,7,-7,-6],[-1,-4,10,-5,-1,-5,7,3,-10,-3,3,9]],[[3,3,4,6,-4,6,-6,-8,-6,-10,-8,4],[-2,-8,-2,-10,-2,-8,-10,6,7,-7,4,2],[10,-2,4,8,-4,2,5,4,-3,-8,7,9],[5,-8,3,9,-1,4,9,-3,10,1,-7,7],[-7,5,8,-1,1,-3,1,-10,5,8,-7,8],[-6,7,4,-3,8,9,3,-2,-1,-1,-9,5],[-6,-9,-3,4,-6,3,4,2,-6,-1,-1,3],[10,-4,-6,6,-1,-9,-1,-6,-4,8,-4,2],[-3,9,8,8,8,8,9,10,-8,-6,1,-10],[8,-3,-10,4,-9,-7,-5,-4,7,10,5,-7],[8,6,5,4,-3,8,3,-7,5,7,9,4]]], dtype = "int64")#candidate|1516|(5, 11, 12)|const|int64
bop_1517 = relay.less_equal(bop_1498.astype('bool'), relay.reshape(const_1516.astype('bool'), relay.shape_of(bop_1498))) # shape=(5, 11, 12)
uop_1524 = relay.rsqrt(bop_1507.astype('float64')) # shape=(5, 11, 1)
output = relay.Tuple([call_1477,bop_1487,bop_1513,bop_1517,uop_1524,])
output2 = relay.Tuple([call_1479,bop_1487,bop_1513,bop_1517,uop_1524,])
func_1526 = relay.Function([var_1465,var_1478,], output)
mod['func_1526'] = func_1526
mod = relay.transform.InferType()(mod)
var_1527 = relay.var("var_1527", dtype = "float64", shape = (5, 11, 1))#candidate|1527|(5, 11, 1)|var|float64
var_1528 = relay.var("var_1528", dtype = "float32", shape = (24,))#candidate|1528|(24,)|var|float32
output = func_1526(var_1527,var_1528,)
func_1529 = relay.Function([var_1527,var_1528,], output)
mutated_mod['func_1529'] = func_1529
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1705 = relay.var("var_1705", dtype = "bool", shape = (8, 10))#candidate|1705|(8, 10)|var|bool
var_1706 = relay.var("var_1706", dtype = "bool", shape = (8, 10))#candidate|1706|(8, 10)|var|bool
bop_1707 = relay.logical_or(var_1705.astype('bool'), relay.reshape(var_1706.astype('bool'), relay.shape_of(var_1705))) # shape=(8, 10)
func_162_call = mod.get_global_var('func_162')
func_165_call = mutated_mod.get_global_var('func_165')
const_1718 = relay.const([[-6.229625,7.845028,5.278776,-7.459220,-8.708849,7.738300,-9.940678,-3.345012,5.267625,1.039870,5.730060,-9.310133,-5.070162,-9.889855,-5.841099,0.357271,-9.914026,6.953350,8.122975,-1.356393,-2.802789,8.356795,-4.173244,-3.444360,3.503348,6.548836,-6.282746,-7.110418,0.665151,-9.132382]], dtype = "float64")#candidate|1718|(1, 30)|const|float64
call_1717 = func_162_call(relay.reshape(const_1718.astype('float64'), [2, 15]))
call_1719 = func_162_call(relay.reshape(const_1718.astype('float64'), [2, 15]))
uop_1724 = relay.acos(var_1706.astype('float64')) # shape=(8, 10)
bop_1728 = relay.logical_xor(const_1718.astype('uint8'), relay.reshape(call_1717.astype('uint8'), relay.shape_of(const_1718))) # shape=(1, 30)
bop_1731 = relay.logical_xor(const_1718.astype('uint8'), relay.reshape(call_1719.astype('uint8'), relay.shape_of(const_1718))) # shape=(1, 30)
output = relay.Tuple([bop_1707,uop_1724,bop_1728,])
output2 = relay.Tuple([bop_1707,uop_1724,bop_1731,])
func_1733 = relay.Function([var_1705,var_1706,], output)
mod['func_1733'] = func_1733
mod = relay.transform.InferType()(mod)
var_1734 = relay.var("var_1734", dtype = "bool", shape = (8, 10))#candidate|1734|(8, 10)|var|bool
var_1735 = relay.var("var_1735", dtype = "bool", shape = (8, 10))#candidate|1735|(8, 10)|var|bool
output = func_1733(var_1734,var_1735,)
func_1736 = relay.Function([var_1734,var_1735,], output)
mutated_mod['func_1736'] = func_1736
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1753 = relay.const([[[-6.978859,-7.472698],[7.576137,-9.066705],[4.075102,-9.375586],[-2.252677,9.950190],[7.713043,6.375608],[-1.616596,-8.050555],[-5.434786,-6.490044],[2.846932,-6.030327],[-6.470350,5.587891],[4.825456,-3.299183]]], dtype = "float32")#candidate|1753|(1, 10, 2)|const|float32
uop_1754 = relay.asin(const_1753.astype('float32')) # shape=(1, 10, 2)
uop_1761 = relay.sinh(uop_1754.astype('float64')) # shape=(1, 10, 2)
bop_1763 = relay.logical_or(const_1753.astype('bool'), relay.reshape(uop_1761.astype('bool'), relay.shape_of(const_1753))) # shape=(1, 10, 2)
uop_1776 = relay.sin(uop_1761.astype('float32')) # shape=(1, 10, 2)
bop_1778 = relay.power(uop_1776.astype('float32'), relay.reshape(uop_1761.astype('float32'), relay.shape_of(uop_1776))) # shape=(1, 10, 2)
output = relay.Tuple([bop_1763,bop_1778,])
output2 = relay.Tuple([bop_1763,bop_1778,])
func_1781 = relay.Function([], output)
mod['func_1781'] = func_1781
mod = relay.transform.InferType()(mod)
output = func_1781()
func_1782 = relay.Function([], output)
mutated_mod['func_1782'] = func_1782
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1795 = relay.const([[-4.797913,-9.429040,-9.429555,2.059702],[-6.911016,-2.601067,3.444246,-4.713981],[8.052499,9.080301,-6.668732,4.275902],[-5.627016,8.538208,-3.852472,-3.133162],[-6.598278,4.173388,1.385540,-7.972774],[0.988741,-3.953896,-2.148688,3.640682],[-4.152025,-8.312190,-1.449641,-0.383610],[8.877953,5.933575,-8.910518,-5.139574],[-2.987276,0.469653,4.835925,2.977748],[-8.539444,1.725103,-2.257931,-7.427135],[5.125468,-6.757878,4.996337,-5.728585],[-3.472164,-4.348617,0.797208,9.844109]], dtype = "float32")#candidate|1795|(12, 4)|const|float32
uop_1796 = relay.atan(const_1795.astype('float32')) # shape=(12, 4)
uop_1798 = relay.log(const_1795.astype('float64')) # shape=(12, 4)
bop_1801 = relay.greater_equal(uop_1796.astype('bool'), relay.reshape(uop_1798.astype('bool'), relay.shape_of(uop_1796))) # shape=(12, 4)
func_720_call = mod.get_global_var('func_720')
func_723_call = mutated_mod.get_global_var('func_723')
var_1805 = relay.var("var_1805", dtype = "float64", shape = (325,))#candidate|1805|(325,)|var|float64
call_1804 = relay.TupleGetItem(func_720_call(relay.reshape(var_1805.astype('float64'), [5, 13, 5])), 0)
call_1806 = relay.TupleGetItem(func_723_call(relay.reshape(var_1805.astype('float64'), [5, 13, 5])), 0)
output = relay.Tuple([bop_1801,call_1804,var_1805,])
output2 = relay.Tuple([bop_1801,call_1806,var_1805,])
func_1810 = relay.Function([var_1805,], output)
mod['func_1810'] = func_1810
mod = relay.transform.InferType()(mod)
var_1811 = relay.var("var_1811", dtype = "float64", shape = (325,))#candidate|1811|(325,)|var|float64
output = func_1810(var_1811)
func_1812 = relay.Function([var_1811], output)
mutated_mod['func_1812'] = func_1812
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1781_call = mod.get_global_var('func_1781')
func_1782_call = mutated_mod.get_global_var('func_1782')
call_1829 = relay.TupleGetItem(func_1781_call(), 1)
call_1830 = relay.TupleGetItem(func_1782_call(), 1)
func_1526_call = mod.get_global_var('func_1526')
func_1529_call = mutated_mod.get_global_var('func_1529')
var_1832 = relay.var("var_1832", dtype = "float64", shape = (55,))#candidate|1832|(55,)|var|float64
var_1833 = relay.var("var_1833", dtype = "float32", shape = (2, 12))#candidate|1833|(2, 12)|var|float32
call_1831 = relay.TupleGetItem(func_1526_call(relay.reshape(var_1832.astype('float64'), [5, 11, 1]), relay.reshape(var_1833.astype('float32'), [24,]), ), 4)
call_1834 = relay.TupleGetItem(func_1529_call(relay.reshape(var_1832.astype('float64'), [5, 11, 1]), relay.reshape(var_1833.astype('float32'), [24,]), ), 4)
uop_1845 = relay.atanh(call_1829.astype('float64')) # shape=(1, 10, 2)
uop_1847 = relay.atanh(call_1830.astype('float64')) # shape=(1, 10, 2)
bop_1849 = relay.logical_xor(uop_1845.astype('int64'), relay.reshape(call_1829.astype('int64'), relay.shape_of(uop_1845))) # shape=(1, 10, 2)
bop_1852 = relay.logical_xor(uop_1847.astype('int64'), relay.reshape(call_1830.astype('int64'), relay.shape_of(uop_1847))) # shape=(1, 10, 2)
uop_1853 = relay.erf(uop_1845.astype('float32')) # shape=(1, 10, 2)
uop_1855 = relay.erf(uop_1847.astype('float32')) # shape=(1, 10, 2)
const_1874 = relay.const([[[-1.908826,4.514492],[2.133806,6.256187],[-6.405122,-3.298920],[5.610612,6.406742],[4.381819,-8.499053],[6.036029,-4.940136],[3.577809,1.977478],[1.612507,-2.268067],[4.091157,2.129266],[-7.036200,-4.940323]],[[-4.392469,-1.549926],[9.548634,-0.188151],[-9.087829,1.614772],[9.138002,-9.414737],[1.012812,-2.667928],[3.319817,-2.084346],[-4.067943,5.184547],[6.559520,5.219175],[2.870698,6.756260],[8.508473,7.889272]],[[5.817946,2.801697],[-6.628681,5.163608],[6.629067,-9.239629],[1.617881,7.302878],[-4.223385,-4.526339],[-3.419759,-1.349396],[2.898951,8.600562],[5.164449,-4.384095],[4.651381,-2.166265],[-0.094886,-8.246203]],[[9.107685,-0.698283],[9.321545,-3.038392],[-3.083780,1.479496],[-1.798319,-2.175796],[-8.124609,-9.348990],[-9.065688,-0.076995],[-7.461928,1.766099],[-4.049214,-7.138741],[0.063983,-8.535600],[-5.104756,-4.265123]],[[-6.104610,5.034157],[1.953634,-6.639798],[3.705389,-2.653513],[-1.087706,-8.979800],[1.787512,2.707099],[1.795566,0.918035],[-2.080149,3.121909],[-4.382183,2.650803],[-5.937005,8.404204],[1.400449,2.125530]],[[-8.149226,-3.337157],[-9.388108,6.678759],[0.919728,1.488985],[-5.357535,9.636267],[4.827524,0.086454],[-2.622887,-8.664517],[3.355611,8.166471],[4.734248,-5.845357],[-6.718912,-3.817610],[7.920928,-8.972661]],[[-0.083796,0.364339],[-3.533623,4.979185],[7.931263,9.550114],[-1.119247,-7.399513],[9.757178,-7.179180],[-9.765274,-0.774634],[7.391489,8.789948],[-8.523035,5.029123],[-3.167095,7.987874],[4.249978,-7.084399]],[[3.967202,6.542140],[7.292979,0.359739],[6.769947,7.028602],[-8.669198,-1.578178],[4.253146,-8.927698],[7.829855,-8.241668],[-2.923323,-8.681762],[5.611221,9.538738],[-2.457461,-5.268764],[-1.625111,-6.037290]],[[9.565543,-5.511857],[4.862924,6.241724],[-0.419599,-4.485995],[-2.285348,1.647056],[-6.236369,-4.072101],[-8.312787,-4.767396],[-7.443400,-5.413319],[-3.715886,-4.086520],[0.508440,-7.456140],[6.303309,8.835712]],[[3.632273,7.762980],[-5.543098,1.090008],[-0.437406,-0.003913],[2.798556,0.488091],[2.236521,-5.873784],[-3.127140,0.950430],[4.290541,-2.423007],[-9.133977,2.215985],[3.542948,9.710122],[0.090247,-2.301230]],[[-6.167276,3.216937],[1.682788,-6.462282],[-9.131069,7.151196],[4.138235,-1.963050],[-0.210427,-4.020554],[-7.662162,-8.289627],[-5.378793,-7.386808],[-5.677821,-0.926485],[2.339395,-2.924093],[-3.189140,-8.278014]],[[-3.284353,-0.768743],[-0.188474,-4.863382],[3.886388,9.813180],[-1.469147,-8.746418],[7.812700,8.618180],[5.790874,9.163211],[2.284806,-4.997079],[-3.606674,-3.506827],[-8.330489,-5.287458],[-7.575167,2.258733]],[[1.058332,2.368196],[-9.088695,8.810282],[-9.570831,-3.687167],[-7.432005,8.836005],[8.669034,7.777630],[-1.380070,-1.858899],[-8.109495,6.423785],[-0.245681,5.190613],[-1.752102,4.486073],[-5.372633,-8.684771]],[[6.017701,-1.594966],[-4.060827,8.694167],[5.719969,9.218016],[0.542593,-9.630201],[2.604187,-7.696409],[-7.724079,7.935008],[-4.637342,-4.355240],[7.906157,-2.024573],[9.064684,-2.562167],[-2.101500,3.586078]],[[7.037269,6.964263],[3.457620,-5.868691],[0.799283,-7.123187],[-2.263699,-3.799677],[-5.620449,3.684991],[-9.738910,6.459403],[9.961155,5.853273],[0.588788,6.154100],[-6.652175,7.360129],[7.972937,-8.941871]]], dtype = "float64")#candidate|1874|(15, 10, 2)|const|float64
bop_1875 = relay.bitwise_xor(uop_1845.astype('int32'), const_1874.astype('int32')) # shape=(15, 10, 2)
bop_1878 = relay.bitwise_xor(uop_1847.astype('int32'), const_1874.astype('int32')) # shape=(15, 10, 2)
bop_1881 = relay.greater(uop_1853.astype('bool'), relay.reshape(bop_1849.astype('bool'), relay.shape_of(uop_1853))) # shape=(1, 10, 2)
bop_1884 = relay.greater(uop_1855.astype('bool'), relay.reshape(bop_1852.astype('bool'), relay.shape_of(uop_1855))) # shape=(1, 10, 2)
output = relay.Tuple([call_1831,var_1832,var_1833,bop_1875,bop_1881,])
output2 = relay.Tuple([call_1834,var_1832,var_1833,bop_1878,bop_1884,])
func_1886 = relay.Function([var_1832,var_1833,], output)
mod['func_1886'] = func_1886
mod = relay.transform.InferType()(mod)
mutated_mod['func_1886'] = func_1886
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1886_call = mutated_mod.get_global_var('func_1886')
var_1888 = relay.var("var_1888", dtype = "float64", shape = (55,))#candidate|1888|(55,)|var|float64
var_1889 = relay.var("var_1889", dtype = "float32", shape = (2, 12))#candidate|1889|(2, 12)|var|float32
call_1887 = func_1886_call(var_1888,var_1889,)
output = call_1887
func_1890 = relay.Function([var_1888,var_1889,], output)
mutated_mod['func_1890'] = func_1890
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1914 = relay.var("var_1914", dtype = "float32", shape = (14, 5))#candidate|1914|(14, 5)|var|float32
uop_1915 = relay.sin(var_1914.astype('float32')) # shape=(14, 5)
func_651_call = mod.get_global_var('func_651')
func_653_call = mutated_mod.get_global_var('func_653')
const_1920 = relay.const([8,-7,-7,-2,-3,-5,3,-10,10,6,-4,-6,4,8,-6,2,-9,10,-3,-4,-9,-1,-2,7,-10,-9,3,-1,-1,-7,-5,8,8,-9,-2,1,1,-2,-4,-9,4,-9,3,8,-10,-10,3,-1,10,6,-3,1,-9,-1,6,5,7,-8,5,-6,-2,10,-6,10,10,-8,-1,9,4,2,7,9,-1,-4,-7,-1,-7,-6,-9,1,-2,-2,-7,-4,-10,-2,-2,-1,-2,-1], dtype = "uint64")#candidate|1920|(90,)|const|uint64
call_1919 = relay.TupleGetItem(func_651_call(relay.reshape(const_1920.astype('uint64'), [15, 6])), 1)
call_1921 = relay.TupleGetItem(func_653_call(relay.reshape(const_1920.astype('uint64'), [15, 6])), 1)
bop_1922 = relay.mod(uop_1915.astype('float32'), relay.reshape(var_1914.astype('float32'), relay.shape_of(uop_1915))) # shape=(14, 5)
uop_1932 = relay.acosh(var_1914.astype('float64')) # shape=(14, 5)
bop_1939 = relay.less(uop_1932.astype('bool'), relay.reshape(uop_1915.astype('bool'), relay.shape_of(uop_1932))) # shape=(14, 5)
bop_1952 = relay.logical_or(uop_1932.astype('bool'), relay.reshape(uop_1915.astype('bool'), relay.shape_of(uop_1932))) # shape=(14, 5)
var_1957 = relay.var("var_1957", dtype = "float64", shape = (14, 5))#candidate|1957|(14, 5)|var|float64
bop_1958 = relay.add(uop_1932.astype('uint64'), relay.reshape(var_1957.astype('uint64'), relay.shape_of(uop_1932))) # shape=(14, 5)
bop_1961 = relay.floor_mod(bop_1939.astype('float64'), relay.reshape(uop_1932.astype('float64'), relay.shape_of(bop_1939))) # shape=(14, 5)
output = relay.Tuple([call_1919,const_1920,bop_1922,bop_1952,bop_1958,bop_1961,])
output2 = relay.Tuple([call_1921,const_1920,bop_1922,bop_1952,bop_1958,bop_1961,])
func_1964 = relay.Function([var_1914,var_1957,], output)
mod['func_1964'] = func_1964
mod = relay.transform.InferType()(mod)
mutated_mod['func_1964'] = func_1964
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1964_call = mutated_mod.get_global_var('func_1964')
var_1966 = relay.var("var_1966", dtype = "float32", shape = (14, 5))#candidate|1966|(14, 5)|var|float32
var_1967 = relay.var("var_1967", dtype = "float64", shape = (14, 5))#candidate|1967|(14, 5)|var|float64
call_1965 = func_1964_call(var_1966,var_1967,)
output = call_1965
func_1968 = relay.Function([var_1966,var_1967,], output)
mutated_mod['func_1968'] = func_1968
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1781_call = mod.get_global_var('func_1781')
func_1782_call = mutated_mod.get_global_var('func_1782')
call_1982 = relay.TupleGetItem(func_1781_call(), 1)
call_1983 = relay.TupleGetItem(func_1782_call(), 1)
output = call_1982
output2 = call_1983
func_2005 = relay.Function([], output)
mod['func_2005'] = func_2005
mod = relay.transform.InferType()(mod)
output = func_2005()
func_2006 = relay.Function([], output)
mutated_mod['func_2006'] = func_2006
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2034 = relay.const([[8.751766,-2.671796,0.412482,-3.907355,-9.677990,9.833016,1.931579,3.333330,-4.125624],[1.976001,2.257177,6.081396,-2.151013,8.228289,2.984851,-7.924475,-7.251963,1.857049],[-9.718954,-6.757442,-2.603561,2.729360,7.257282,-4.934460,-6.477194,8.457250,2.526276],[-4.050463,6.987175,7.232892,-7.363592,5.001300,6.633612,-1.425263,-8.707866,-7.238967],[0.084615,-1.530357,-7.591447,-7.292914,-9.099583,-2.673451,-9.134308,4.910144,-4.551731],[-7.837540,5.370195,6.576188,-2.320098,-0.208006,-6.400048,-1.995131,-3.437511,5.891290],[0.634429,1.600492,5.181583,3.828679,4.928197,3.994327,1.646608,-7.105422,3.535747]], dtype = "float32")#candidate|2034|(7, 9)|const|float32
uop_2035 = relay.atan(const_2034.astype('float32')) # shape=(7, 9)
output = uop_2035
output2 = uop_2035
func_2040 = relay.Function([], output)
mod['func_2040'] = func_2040
mod = relay.transform.InferType()(mod)
mutated_mod['func_2040'] = func_2040
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2040_call = mutated_mod.get_global_var('func_2040')
call_2041 = func_2040_call()
output = call_2041
func_2042 = relay.Function([], output)
mutated_mod['func_2042'] = func_2042
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2043 = relay.var("var_2043", dtype = "float32", shape = (10, 4))#candidate|2043|(10, 4)|var|float32
uop_2044 = relay.sqrt(var_2043.astype('float32')) # shape=(10, 4)
output = uop_2044
output2 = uop_2044
func_2047 = relay.Function([var_2043,], output)
mod['func_2047'] = func_2047
mod = relay.transform.InferType()(mod)
var_2048 = relay.var("var_2048", dtype = "float32", shape = (10, 4))#candidate|2048|(10, 4)|var|float32
output = func_2047(var_2048)
func_2049 = relay.Function([var_2048], output)
mutated_mod['func_2049'] = func_2049
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2084 = relay.var("var_2084", dtype = "float32", shape = (5, 9, 4))#candidate|2084|(5, 9, 4)|var|float32
uop_2085 = relay.cosh(var_2084.astype('float32')) # shape=(5, 9, 4)
output = uop_2085
output2 = uop_2085
func_2087 = relay.Function([var_2084,], output)
mod['func_2087'] = func_2087
mod = relay.transform.InferType()(mod)
var_2088 = relay.var("var_2088", dtype = "float32", shape = (5, 9, 4))#candidate|2088|(5, 9, 4)|var|float32
output = func_2087(var_2088)
func_2089 = relay.Function([var_2088], output)
mutated_mod['func_2089'] = func_2089
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2096 = relay.var("var_2096", dtype = "float32", shape = (10, 14))#candidate|2096|(10, 14)|var|float32
uop_2097 = relay.asin(var_2096.astype('float32')) # shape=(10, 14)
func_1117_call = mod.get_global_var('func_1117')
func_1121_call = mutated_mod.get_global_var('func_1121')
const_2102 = relay.const(-4, dtype = "uint32")#candidate|2102|()|const|uint32
const_2103 = relay.const([[9,-1,4,-10,3,4,9,-1,9,-3,-6,-8,-5,-5,-8,6,-5,-10,6,3,-4,-9,2,-1,10,7,-8,-8,-10,1],[-3,5,4,6,-5,-1,7,1,-10,4,8,-5,-6,3,-7,4,-3,-8,4,9,7,8,10,5,2,9,1,-6,8,-5],[-4,-4,7,4,-8,-3,-2,8,2,1,3,4,5,-2,-1,-7,5,-4,5,-6,-2,7,3,8,-9,9,-1,7,-5,1]], dtype = "uint32")#candidate|2103|(3, 30)|const|uint32
call_2101 = relay.TupleGetItem(func_1117_call(relay.reshape(const_2102.astype('uint32'), []), relay.reshape(const_2103.astype('uint32'), [6, 15]), ), 0)
call_2104 = relay.TupleGetItem(func_1121_call(relay.reshape(const_2102.astype('uint32'), []), relay.reshape(const_2103.astype('uint32'), [6, 15]), ), 0)
const_2106 = relay.const([[6.775503,3.473371,-4.784910,1.282202,-3.184392,-8.825885,-2.064098,-3.850068,-2.295328,-2.421824,-2.697169,-6.633200,-3.019737,-5.681741],[-6.131880,-2.842290,-2.843578,0.036162,-8.876690,-6.319677,-4.647744,6.798312,-9.828645,-8.791579,1.022277,3.735531,-2.117591,-6.429451],[-8.911446,-0.824455,8.625054,-3.883012,-9.623258,-0.018439,7.379074,9.572638,0.910227,-3.298268,-4.822221,-4.099694,0.673712,-2.621814],[5.810622,-7.879637,3.598896,-1.810589,-7.558733,-3.684971,-3.798601,7.764555,7.616508,-8.116432,6.196715,8.911791,5.781176,-3.858324],[1.471443,-6.213634,4.741197,1.151050,5.267141,4.543953,-3.531059,0.032480,-8.849322,-3.436721,6.148997,-8.116706,-1.498229,9.936474],[0.310451,-6.830086,-9.243270,3.015680,-8.250192,-5.051302,-5.081542,2.202253,6.503091,-0.463024,2.152978,0.572068,2.880244,-0.752753],[9.316104,4.963292,5.558272,-0.474265,-3.509042,3.805836,-4.512450,4.542841,-6.421124,2.764758,5.853450,6.294883,2.924737,7.484140],[9.662543,-3.777346,6.401390,3.741319,9.520242,-7.104428,-7.367251,-6.097504,8.966433,1.350990,-4.042761,6.811964,-2.623266,2.832887],[-2.244045,-4.690615,-9.131740,-9.533070,-4.894698,-7.011519,0.283713,-1.452906,-9.467189,-0.409316,-1.315155,9.741875,7.282532,-7.713946],[-1.445555,4.468242,-9.658693,-6.857779,-6.791019,7.299749,-2.696642,-6.964527,-6.121077,1.918963,-9.253123,8.705419,-6.384641,8.676678]], dtype = "float32")#candidate|2106|(10, 14)|const|float32
bop_2107 = relay.less(uop_2097.astype('bool'), relay.reshape(const_2106.astype('bool'), relay.shape_of(uop_2097))) # shape=(10, 14)
output = relay.Tuple([call_2101,const_2102,const_2103,bop_2107,])
output2 = relay.Tuple([call_2104,const_2102,const_2103,bop_2107,])
F = relay.Function([var_2096,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_2096,], output2)
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
input_2096= np.array([[7.061859,-9.386172,4.461666,-5.418395,-9.679766,-9.736794,3.191972,-8.382635,3.574321,4.114678,-0.408721,-7.253532,8.279303,-8.381766],[4.610759,-0.288816,-9.477978,2.837079,6.765043,6.962967,9.590418,9.267822,-4.260065,-7.328364,7.331639,9.670982,-0.467825,-7.575231],[-7.323720,6.412759,1.447503,-4.445306,-2.093938,0.067032,-0.425070,7.831181,-2.462172,-6.962910,4.029137,6.969993,-6.065397,-9.084977],[-3.412820,6.638638,0.795876,1.376528,-2.746390,-0.871941,0.988346,3.442018,-7.560036,6.458601,0.098361,6.608060,-7.981986,7.850940],[-2.188356,-1.454812,1.511228,0.808922,-0.389875,9.859583,-6.541787,1.266023,7.818563,8.948139,-9.386284,-3.686027,4.225662,-1.847315],[-6.026646,-0.285303,5.400488,-2.695231,-2.300120,4.972859,-6.460215,-8.788328,-0.351029,-0.113323,-2.647712,0.788114,-1.428010,-0.693365],[0.253822,3.236331,-8.512586,-6.471957,-8.703854,-7.776110,-8.804644,8.871254,-9.173628,-2.104465,-5.605675,-8.350129,-1.878425,1.000063],[-4.720687,-8.460514,-4.696516,7.108023,-6.186886,-5.643084,-6.802349,4.727044,-8.504795,4.757802,-6.459008,4.491299,-0.347106,8.411199],[3.504411,6.612855,-5.219177,7.934453,-1.687247,-2.466824,4.555174,1.907740,8.961966,-9.296368,-6.992261,7.271388,8.145313,1.083516],[-7.224812,7.153853,5.829881,2.646875,-4.085698,-6.319750,-1.612623,8.787700,3.339750,7.716790,-8.559991,-7.987984,-7.909259,5.862213]], dtype='float32')
module1.set_input('var_2096', input_2096)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_2096, )
res3 = intrp3.evaluate()(input_2096, )
res4 = intrp4.evaluate()(input_2096, )
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
res1_2 = module1.get_output(2).asnumpy()
res2_2 = res2[2].asnumpy()
res3_2 = res3[2].asnumpy()
res4_2 = res4[2].asnumpy()
np.testing.assert_allclose(res1_2 ,res2_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res3_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res4_2, atol=1e-3, rtol=1e-3)
(res1_2 == res2_2).all()
(res1_2 == res3_2).all()
(res1_2 == res4_2).all()
res1_3 = module1.get_output(3).asnumpy()
res2_3 = res2[3].asnumpy()
res3_3 = res3[3].asnumpy()
res4_3 = res4[3].asnumpy()
np.testing.assert_allclose(res1_3 ,res2_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_3 ,res3_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_3 ,res4_3, atol=1e-3, rtol=1e-3)
(res1_3 == res2_3).all()
(res1_3 == res3_3).all()
(res1_3 == res4_3).all()
module5.set_input('var_2096', input_2096)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_2096, )
res7 = intrp7.evaluate()(input_2096, )
res8 = intrp8.evaluate()(input_2096, )
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
res5_2 = module5.get_output(2).asnumpy()
res6_2 = res6[2].asnumpy()
res7_2 = res7[2].asnumpy()
res8_2 = res8[2].asnumpy()
np.testing.assert_allclose(res5_2 ,res6_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res7_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res8_2, atol=1e-3, rtol=1e-3)
(res5_2 == res6_2).all()
(res5_2 == res7_2).all()
(res5_2 == res8_2).all()
res5_3 = module5.get_output(3).asnumpy()
res6_3 = res6[3].asnumpy()
res7_3 = res7[3].asnumpy()
res8_3 = res8[3].asnumpy()
np.testing.assert_allclose(res5_3 ,res6_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_3 ,res7_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_3 ,res8_3, atol=1e-3, rtol=1e-3)
(res5_3 == res6_3).all()
(res5_3 == res7_3).all()
(res5_3 == res8_3).all()
module9.set_input('var_2096', input_2096)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_2096, )
res11 = intrp11.evaluate()(input_2096, )
res12 = intrp12.evaluate()(input_2096, )
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
res9_2 = module9.get_output(2).asnumpy()
res10_2 = res10[2].asnumpy()
res11_2 = res11[2].asnumpy()
res12_2 = res12[2].asnumpy()
np.testing.assert_allclose(res9_2 ,res10_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res11_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res12_2, atol=1e-3, rtol=1e-3)
(res9_2 == res10_2).all()
(res9_2 == res11_2).all()
(res9_2 == res12_2).all()
res9_3 = module9.get_output(3).asnumpy()
res10_3 = res10[3].asnumpy()
res11_3 = res11[3].asnumpy()
res12_3 = res12[3].asnumpy()
np.testing.assert_allclose(res9_3 ,res10_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_3 ,res11_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_3 ,res12_3, atol=1e-3, rtol=1e-3)
(res9_3 == res10_3).all()
(res9_3 == res11_3).all()
(res9_3 == res12_3).all()
module13.set_input('var_2096', input_2096)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_2096, )
res15 = intrp15.evaluate()(input_2096, )
res16 = intrp16.evaluate()(input_2096, )
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
res13_2 = module13.get_output(2).asnumpy()
res14_2 = res14[2].asnumpy()
res15_2 = res15[2].asnumpy()
res16_2 = res16[2].asnumpy()
np.testing.assert_allclose(res13_2 ,res14_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res15_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res16_2, atol=1e-3, rtol=1e-3)
(res13_2 == res14_2).all()
(res13_2 == res15_2).all()
(res13_2 == res16_2).all()
res13_3 = module13.get_output(3).asnumpy()
res14_3 = res14[3].asnumpy()
res15_3 = res15[3].asnumpy()
res16_3 = res16[3].asnumpy()
np.testing.assert_allclose(res13_3 ,res14_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_3 ,res15_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_3 ,res16_3, atol=1e-3, rtol=1e-3)
(res13_3 == res14_3).all()
(res13_3 == res15_3).all()
(res13_3 == res16_3).all()
module17.set_input('var_2096', input_2096)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_2096, )
res19 = intrp19.evaluate()(input_2096, )
res20 = intrp20.evaluate()(input_2096, )
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
res17_2 = module17.get_output(2).asnumpy()
res18_2 = res18[2].asnumpy()
res19_2 = res19[2].asnumpy()
res20_2 = res20[2].asnumpy()
np.testing.assert_allclose(res17_2 ,res18_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res19_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res20_2, atol=1e-3, rtol=1e-3)
(res17_2 == res18_2).all()
(res17_2 == res19_2).all()
(res17_2 == res20_2).all()
res17_3 = module17.get_output(3).asnumpy()
res18_3 = res18[3].asnumpy()
res19_3 = res19[3].asnumpy()
res20_3 = res20[3].asnumpy()
np.testing.assert_allclose(res17_3 ,res18_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_3 ,res19_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_3 ,res20_3, atol=1e-3, rtol=1e-3)
(res17_3 == res18_3).all()
(res17_3 == res19_3).all()
(res17_3 == res20_3).all()
module21.set_input('var_2096', input_2096)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_2096, )
res23 = intrp23.evaluate()(input_2096, )
res24 = intrp24.evaluate()(input_2096, )
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
res21_2 = module21.get_output(2).asnumpy()
res22_2 = res22[2].asnumpy()
res23_2 = res23[2].asnumpy()
res24_2 = res24[2].asnumpy()
np.testing.assert_allclose(res21_2 ,res22_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res23_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res24_2, atol=1e-3, rtol=1e-3)
(res21_2 == res22_2).all()
(res21_2 == res23_2).all()
(res21_2 == res24_2).all()
res21_3 = module21.get_output(3).asnumpy()
res22_3 = res22[3].asnumpy()
res23_3 = res23[3].asnumpy()
res24_3 = res24[3].asnumpy()
np.testing.assert_allclose(res21_3 ,res22_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_3 ,res23_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_3 ,res24_3, atol=1e-3, rtol=1e-3)
(res21_3 == res22_3).all()
(res21_3 == res23_3).all()
(res21_3 == res24_3).all()

'''6: TVMFuncCall
5: _ZNSt17_Function_handlerIFvN3tvm7runtime7
4: tvm::runtime::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const [clone .isra.808]
3: tvm::runtime::GraphExecutorCreate(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module const&, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
2: tvm::runtime::GraphExecutor::Init(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
1: tvm::runtime::GraphExecutor::SetupOpExecs()
0: tvm::runtime::GraphExecutor::CreateTVMOp(tvm::runtime::TVMOpParam const&, std::vector<DLTensor, std::allocator<DLTensor> > const&)

'''