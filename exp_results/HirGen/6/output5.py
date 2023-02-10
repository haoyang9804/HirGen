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
var_2 = relay.var("var_2", dtype = "float64", shape = (8, 2, 7))#candidate|2|(8, 2, 7)|var|float64
uop_3 = relay.log2(var_2.astype('float64')) # shape=(8, 2, 7)
output = relay.Tuple([uop_3,])
output2 = relay.Tuple([uop_3,])
func_5 = relay.Function([var_2,], output)
mod['func_5'] = func_5
mod = relay.transform.InferType()(mod)
var_6 = relay.var("var_6", dtype = "float64", shape = (8, 2, 7))#candidate|6|(8, 2, 7)|var|float64
output = func_5(var_6)
func_7 = relay.Function([var_6], output)
mutated_mod['func_7'] = func_7
mutated_mod = relay.transform.InferType()(mutated_mod)
var_100 = relay.var("var_100", dtype = "int32", shape = (1, 10))#candidate|100|(1, 10)|var|int32
const_101 = relay.const([[9,-7,5,6,-10,8,-9,5,10,4],[-3,10,-5,5,-8,-1,3,7,-9,-5],[-9,-5,6,6,6,-4,2,1,9,-5],[9,5,-5,1,5,-2,-6,3,-7,-10],[3,-3,1,2,-2,-6,4,-2,-6,-10],[-8,7,1,8,9,10,1,6,-9,-2],[3,1,-1,-6,-7,-8,-3,9,-5,2],[2,-5,-9,-6,7,-9,-8,-10,7,-6],[4,-7,8,-1,-2,6,-6,8,-2,-2],[-6,1,-5,-3,-9,-7,4,10,-2,7]], dtype = "int32")#candidate|101|(10, 10)|const|int32
bop_102 = relay.not_equal(var_100.astype('bool'), const_101.astype('bool')) # shape=(10, 10)
output = bop_102
output2 = bop_102
func_112 = relay.Function([var_100,], output)
mod['func_112'] = func_112
mod = relay.transform.InferType()(mod)
var_113 = relay.var("var_113", dtype = "int32", shape = (1, 10))#candidate|113|(1, 10)|var|int32
output = func_112(var_113)
func_114 = relay.Function([var_113], output)
mutated_mod['func_114'] = func_114
mutated_mod = relay.transform.InferType()(mutated_mod)
const_138 = relay.const(2.928474, dtype = "float32")#candidate|138|()|const|float32
var_139 = relay.var("var_139", dtype = "float32", shape = (12,))#candidate|139|(12,)|var|float32
bop_140 = relay.multiply(const_138.astype('float32'), var_139.astype('float32')) # shape=(12,)
output = bop_140
output2 = bop_140
func_148 = relay.Function([var_139,], output)
mod['func_148'] = func_148
mod = relay.transform.InferType()(mod)
mutated_mod['func_148'] = func_148
mutated_mod = relay.transform.InferType()(mutated_mod)
var_149 = relay.var("var_149", dtype = "float32", shape = (12,))#candidate|149|(12,)|var|float32
func_148_call = mutated_mod.get_global_var('func_148')
call_150 = func_148_call(var_149)
output = call_150
func_151 = relay.Function([var_149], output)
mutated_mod['func_151'] = func_151
mutated_mod = relay.transform.InferType()(mutated_mod)
var_179 = relay.var("var_179", dtype = "uint32", shape = (12, 6, 11))#candidate|179|(12, 6, 11)|var|uint32
var_180 = relay.var("var_180", dtype = "uint32", shape = (12, 6, 11))#candidate|180|(12, 6, 11)|var|uint32
bop_181 = relay.minimum(var_179.astype('uint32'), relay.reshape(var_180.astype('uint32'), relay.shape_of(var_179))) # shape=(12, 6, 11)
func_5_call = mod.get_global_var('func_5')
func_7_call = mutated_mod.get_global_var('func_7')
const_190 = relay.const([1.037515,2.337758,-2.161465,6.836725,-7.755202,2.314235,5.039831,6.975354,-8.733084,1.605056,7.641703,6.787784,9.041294,8.057509,-3.897077,9.484790,0.872315,9.371177,5.782602,0.127291,-0.372550,-6.499637,6.323740,-4.069410,-6.357053,7.492252,8.070889,-7.902342,5.386015,9.517953,0.771691,-6.034162,1.686228,-0.404510,-6.486186,-0.049790,1.729328,-5.581561,-8.448365,-3.357637,6.659023,-9.395412,-7.115504,9.967685,-0.106593,-1.549019,9.428829,-5.818180,-4.051443,8.329533,8.974627,1.825374,-2.656769,4.306776,-1.908030,7.920504,2.276733,-3.937918,-1.833554,1.365551,-0.188080,-2.063829,-0.543055,-7.895256,4.471970,-3.733967,3.864079,8.664119,9.934615,-9.605391,-7.348494,1.621162,0.151343,-3.790548,2.743111,1.807926,4.884419,6.689980,-8.806807,-3.826673,7.176340,-9.599526,-7.688069,-8.218943,-0.141273,7.361304,-3.357909,-6.332803,1.276366,6.733312,9.887054,-2.929122,-6.971670,-5.542435,-8.280442,6.985205,3.688049,-5.133129,5.464960,-1.871255,-0.870036,9.720754,-8.061008,-9.188223,-2.300172,8.499137,-8.683871,-2.649368,0.384115,-6.782219,3.012145,-9.757269], dtype = "float64")#candidate|190|(112,)|const|float64
call_189 = relay.TupleGetItem(func_5_call(relay.reshape(const_190.astype('float64'), [8, 2, 7])), 0)
call_191 = relay.TupleGetItem(func_7_call(relay.reshape(const_190.astype('float64'), [8, 2, 7])), 0)
const_196 = relay.const([3.955750,5.416398,0.888204,-7.256047,-4.597097,-5.313669,7.197794,-4.023626,-3.804236,2.378304,4.186560,8.556678,7.544091,5.555197,-4.191648,4.868100,-4.001570,-3.405751,5.885684,7.340629,-5.282731,-4.581976,-4.425253,-1.284256,-6.641253,-5.037438,-1.337392,-9.894846,-8.341865,-2.257608,-6.601339,-2.950916,-8.117078,0.396472,-5.999984,0.618165,-2.420894,-3.495511,-1.756685,0.033182,-2.191689,0.222927,-9.506369,-1.943779,9.442344,2.958105,8.475288,-3.334312,8.226564,0.655642,-3.691880,-5.676290,-4.478670,1.189521,9.646983,-0.757100,-0.665499,1.345767,7.040291,5.978820,-1.727454,-9.893548,6.529497,8.277876,-8.754295,-2.559320,-1.740095,-1.564144,2.206800,5.181224,-8.654956,-9.993118,4.827145,0.083327,2.371483,1.404572,-0.290632,7.842490,-5.969818,-2.576241,-0.893175,-2.083955,7.912378,8.606416,-7.392110,6.127177,-8.776210,-4.803147,2.638561,-9.824202,-6.448306,-1.357106,-0.275293,-5.271066,0.353178,8.171467,5.363961,-9.616396,-3.590364,1.586233,-6.657303,-3.862299,-3.161778,5.330715,-6.278793,7.526794,5.274794,5.549268,8.059736,0.445309,-4.741352,-9.515798], dtype = "float64")#candidate|196|(112,)|const|float64
bop_197 = relay.left_shift(const_190.astype('int16'), relay.reshape(const_196.astype('int16'), relay.shape_of(const_190))) # shape=(112,)
func_148_call = mod.get_global_var('func_148')
func_151_call = mutated_mod.get_global_var('func_151')
const_201 = relay.const([-0.558795,-9.011829,2.085701,-9.908540,-1.168999,1.518080,0.868507,3.233136,1.000745,1.468131,-9.995210,-6.895932], dtype = "float32")#candidate|201|(12,)|const|float32
call_200 = func_148_call(relay.reshape(const_201.astype('float32'), [12,]))
call_202 = func_148_call(relay.reshape(const_201.astype('float32'), [12,]))
var_203 = relay.var("var_203", dtype = "int16", shape = (112,))#candidate|203|(112,)|var|int16
bop_204 = relay.power(bop_197.astype('float32'), relay.reshape(var_203.astype('float32'), relay.shape_of(bop_197))) # shape=(112,)
bop_213 = relay.bitwise_or(const_201.astype('uint64'), relay.reshape(call_200.astype('uint64'), relay.shape_of(const_201))) # shape=(12,)
bop_216 = relay.bitwise_or(const_201.astype('uint64'), relay.reshape(call_202.astype('uint64'), relay.shape_of(const_201))) # shape=(12,)
uop_222 = relay.atanh(bop_204.astype('float64')) # shape=(112,)
uop_224 = relay.asin(uop_222.astype('float32')) # shape=(112,)
func_148_call = mod.get_global_var('func_148')
func_151_call = mutated_mod.get_global_var('func_151')
call_226 = func_148_call(relay.reshape(const_201.astype('float32'), [12,]))
call_227 = func_148_call(relay.reshape(const_201.astype('float32'), [12,]))
output = relay.Tuple([bop_181,call_189,bop_213,uop_224,call_226,])
output2 = relay.Tuple([bop_181,call_191,bop_216,uop_224,call_227,])
func_228 = relay.Function([var_179,var_180,var_203,], output)
mod['func_228'] = func_228
mod = relay.transform.InferType()(mod)
mutated_mod['func_228'] = func_228
mutated_mod = relay.transform.InferType()(mutated_mod)
func_228_call = mutated_mod.get_global_var('func_228')
var_230 = relay.var("var_230", dtype = "uint32", shape = (12, 6, 11))#candidate|230|(12, 6, 11)|var|uint32
var_231 = relay.var("var_231", dtype = "uint32", shape = (12, 6, 11))#candidate|231|(12, 6, 11)|var|uint32
var_232 = relay.var("var_232", dtype = "int16", shape = (112,))#candidate|232|(112,)|var|int16
call_229 = func_228_call(var_230,var_231,var_232,)
output = call_229
func_233 = relay.Function([var_230,var_231,var_232,], output)
mutated_mod['func_233'] = func_233
mutated_mod = relay.transform.InferType()(mutated_mod)
var_240 = relay.var("var_240", dtype = "uint8", shape = (15, 15, 16))#candidate|240|(15, 15, 16)|var|uint8
var_241 = relay.var("var_241", dtype = "uint8", shape = (15, 15, 16))#candidate|241|(15, 15, 16)|var|uint8
bop_242 = relay.left_shift(var_240.astype('uint8'), relay.reshape(var_241.astype('uint8'), relay.shape_of(var_240))) # shape=(15, 15, 16)
uop_250 = relay.sigmoid(bop_242.astype('float64')) # shape=(15, 15, 16)
uop_252 = relay.sin(uop_250.astype('float64')) # shape=(15, 15, 16)
uop_254 = relay.sinh(uop_252.astype('float32')) # shape=(15, 15, 16)
func_5_call = mod.get_global_var('func_5')
func_7_call = mutated_mod.get_global_var('func_7')
var_261 = relay.var("var_261", dtype = "float64", shape = (112,))#candidate|261|(112,)|var|float64
call_260 = relay.TupleGetItem(func_5_call(relay.reshape(var_261.astype('float64'), [8, 2, 7])), 0)
call_262 = relay.TupleGetItem(func_7_call(relay.reshape(var_261.astype('float64'), [8, 2, 7])), 0)
bop_267 = relay.left_shift(uop_250.astype('int64'), relay.reshape(uop_254.astype('int64'), relay.shape_of(uop_250))) # shape=(15, 15, 16)
bop_270 = relay.logical_and(uop_252.astype('bool'), relay.reshape(bop_242.astype('bool'), relay.shape_of(uop_252))) # shape=(15, 15, 16)
var_284 = relay.var("var_284", dtype = "bool", shape = (15, 15, 16))#candidate|284|(15, 15, 16)|var|bool
bop_285 = relay.floor_divide(bop_270.astype('float64'), relay.reshape(var_284.astype('float64'), relay.shape_of(bop_270))) # shape=(15, 15, 16)
uop_288 = relay.log2(uop_254.astype('float32')) # shape=(15, 15, 16)
uop_290 = relay.rsqrt(uop_288.astype('float32')) # shape=(15, 15, 16)
bop_292 = relay.greater(uop_290.astype('bool'), relay.reshape(var_284.astype('bool'), relay.shape_of(uop_290))) # shape=(15, 15, 16)
bop_295 = relay.less(uop_290.astype('bool'), relay.reshape(bop_270.astype('bool'), relay.shape_of(uop_290))) # shape=(15, 15, 16)
bop_298 = relay.floor_divide(uop_288.astype('float32'), relay.reshape(bop_270.astype('float32'), relay.shape_of(uop_288))) # shape=(15, 15, 16)
uop_301 = relay.log(bop_285.astype('float64')) # shape=(15, 15, 16)
output = relay.Tuple([call_260,var_261,bop_267,bop_292,bop_295,bop_298,uop_301,])
output2 = relay.Tuple([call_262,var_261,bop_267,bop_292,bop_295,bop_298,uop_301,])
func_303 = relay.Function([var_240,var_241,var_261,var_284,], output)
mod['func_303'] = func_303
mod = relay.transform.InferType()(mod)
var_304 = relay.var("var_304", dtype = "uint8", shape = (15, 15, 16))#candidate|304|(15, 15, 16)|var|uint8
var_305 = relay.var("var_305", dtype = "uint8", shape = (15, 15, 16))#candidate|305|(15, 15, 16)|var|uint8
var_306 = relay.var("var_306", dtype = "float64", shape = (112,))#candidate|306|(112,)|var|float64
var_307 = relay.var("var_307", dtype = "bool", shape = (15, 15, 16))#candidate|307|(15, 15, 16)|var|bool
output = func_303(var_304,var_305,var_306,var_307,)
func_308 = relay.Function([var_304,var_305,var_306,var_307,], output)
mutated_mod['func_308'] = func_308
mutated_mod = relay.transform.InferType()(mutated_mod)
var_381 = relay.var("var_381", dtype = "uint8", shape = (1, 16))#candidate|381|(1, 16)|var|uint8
var_382 = relay.var("var_382", dtype = "uint8", shape = (12, 16))#candidate|382|(12, 16)|var|uint8
bop_383 = relay.not_equal(var_381.astype('bool'), var_382.astype('bool')) # shape=(12, 16)
output = bop_383
output2 = bop_383
func_390 = relay.Function([var_381,var_382,], output)
mod['func_390'] = func_390
mod = relay.transform.InferType()(mod)
var_391 = relay.var("var_391", dtype = "uint8", shape = (1, 16))#candidate|391|(1, 16)|var|uint8
var_392 = relay.var("var_392", dtype = "uint8", shape = (12, 16))#candidate|392|(12, 16)|var|uint8
output = func_390(var_391,var_392,)
func_393 = relay.Function([var_391,var_392,], output)
mutated_mod['func_393'] = func_393
mutated_mod = relay.transform.InferType()(mutated_mod)
var_420 = relay.var("var_420", dtype = "uint64", shape = (7, 8, 9))#candidate|420|(7, 8, 9)|var|uint64
var_421 = relay.var("var_421", dtype = "uint64", shape = (7, 8, 9))#candidate|421|(7, 8, 9)|var|uint64
bop_422 = relay.minimum(var_420.astype('uint64'), relay.reshape(var_421.astype('uint64'), relay.shape_of(var_420))) # shape=(7, 8, 9)
output = bop_422
output2 = bop_422
func_428 = relay.Function([var_420,var_421,], output)
mod['func_428'] = func_428
mod = relay.transform.InferType()(mod)
var_429 = relay.var("var_429", dtype = "uint64", shape = (7, 8, 9))#candidate|429|(7, 8, 9)|var|uint64
var_430 = relay.var("var_430", dtype = "uint64", shape = (7, 8, 9))#candidate|430|(7, 8, 9)|var|uint64
output = func_428(var_429,var_430,)
func_431 = relay.Function([var_429,var_430,], output)
mutated_mod['func_431'] = func_431
mutated_mod = relay.transform.InferType()(mutated_mod)
var_436 = relay.var("var_436", dtype = "float64", shape = (13, 6))#candidate|436|(13, 6)|var|float64
uop_437 = relay.cos(var_436.astype('float64')) # shape=(13, 6)
bop_440 = relay.logical_xor(uop_437.astype('uint64'), relay.reshape(var_436.astype('uint64'), relay.shape_of(uop_437))) # shape=(13, 6)
var_447 = relay.var("var_447", dtype = "float64", shape = (13, 6))#candidate|447|(13, 6)|var|float64
bop_448 = relay.bitwise_and(uop_437.astype('uint32'), relay.reshape(var_447.astype('uint32'), relay.shape_of(uop_437))) # shape=(13, 6)
output = relay.Tuple([bop_440,bop_448,])
output2 = relay.Tuple([bop_440,bop_448,])
func_451 = relay.Function([var_436,var_447,], output)
mod['func_451'] = func_451
mod = relay.transform.InferType()(mod)
mutated_mod['func_451'] = func_451
mutated_mod = relay.transform.InferType()(mutated_mod)
func_451_call = mutated_mod.get_global_var('func_451')
var_453 = relay.var("var_453", dtype = "float64", shape = (13, 6))#candidate|453|(13, 6)|var|float64
var_454 = relay.var("var_454", dtype = "float64", shape = (13, 6))#candidate|454|(13, 6)|var|float64
call_452 = func_451_call(var_453,var_454,)
output = call_452
func_455 = relay.Function([var_453,var_454,], output)
mutated_mod['func_455'] = func_455
mutated_mod = relay.transform.InferType()(mutated_mod)
var_482 = relay.var("var_482", dtype = "uint8", shape = (8, 6, 2))#candidate|482|(8, 6, 2)|var|uint8
var_483 = relay.var("var_483", dtype = "uint8", shape = (8, 6, 2))#candidate|483|(8, 6, 2)|var|uint8
bop_484 = relay.multiply(var_482.astype('uint8'), relay.reshape(var_483.astype('uint8'), relay.shape_of(var_482))) # shape=(8, 6, 2)
var_498 = relay.var("var_498", dtype = "uint8", shape = (8, 6, 2))#candidate|498|(8, 6, 2)|var|uint8
bop_499 = relay.subtract(bop_484.astype('uint32'), relay.reshape(var_498.astype('uint32'), relay.shape_of(bop_484))) # shape=(8, 6, 2)
bop_511 = relay.logical_and(var_483.astype('bool'), relay.reshape(bop_499.astype('bool'), relay.shape_of(var_483))) # shape=(8, 6, 2)
uop_516 = relay.sinh(bop_511.astype('float64')) # shape=(8, 6, 2)
uop_521 = relay.log10(bop_499.astype('float64')) # shape=(8, 6, 2)
bop_524 = relay.floor_divide(uop_521.astype('float32'), relay.reshape(bop_484.astype('float32'), relay.shape_of(uop_521))) # shape=(8, 6, 2)
var_528 = relay.var("var_528", dtype = "float64", shape = (8, 6, 2))#candidate|528|(8, 6, 2)|var|float64
bop_529 = relay.greater_equal(uop_521.astype('bool'), relay.reshape(var_528.astype('bool'), relay.shape_of(uop_521))) # shape=(8, 6, 2)
bop_534 = relay.add(bop_524.astype('float32'), relay.reshape(bop_529.astype('float32'), relay.shape_of(bop_524))) # shape=(8, 6, 2)
uop_538 = relay.log2(bop_529.astype('float64')) # shape=(8, 6, 2)
func_390_call = mod.get_global_var('func_390')
func_393_call = mutated_mod.get_global_var('func_393')
var_541 = relay.var("var_541", dtype = "uint8", shape = (16,))#candidate|541|(16,)|var|uint8
const_542 = relay.const([-6,-6,1,5,7,-1,-10,-2,-2,10,8,-6,-6,3,6,9,-9,4,9,4,-5,-2,10,7,4,5,-3,-7,9,9,8,-7,5,4,1,7,-7,-10,4,6,-7,8,-4,-8,-6,8,5,2,-7,1,6,-3,-10,9,-2,-1,-6,10,-7,10,-3,9,4,-7,10,8,-7,-1,5,2,3,-8,-4,6,5,5,6,6,-7,-10,1,3,8,7,-3,3,-4,-4,9,10,-8,2,10,4,2,-2,-2,3,2,5,10,-1,1,3,-5,-4,5,-7,-5,-2,-4,2,-3,10,-6,9,5,10,-5,-5,-3,3,4,-5,-2,-8,3,-6,6,10,-3,7,10,3,10,6,-5,-5,3,-2,1,3,1,-10,7,6,-10,-5,2,-8,6,-10,-4,-1,-7,2,-2,-1,8,7,-3,3,-8,6,8,4,7,-4,9,8,8,-3,8,-3,-5,-10,-7,-2,3,5,-1,5,3,10,7,-5,9,7,-1,-4,5,7], dtype = "uint8")#candidate|542|(192,)|const|uint8
call_540 = func_390_call(relay.reshape(var_541.astype('uint8'), [1, 16]), relay.reshape(const_542.astype('uint8'), [12, 16]), )
call_543 = func_390_call(relay.reshape(var_541.astype('uint8'), [1, 16]), relay.reshape(const_542.astype('uint8'), [12, 16]), )
bop_544 = relay.logical_xor(uop_538.astype('uint64'), relay.reshape(bop_499.astype('uint64'), relay.shape_of(uop_538))) # shape=(8, 6, 2)
bop_549 = relay.floor_divide(bop_534.astype('float64'), relay.reshape(bop_499.astype('float64'), relay.shape_of(bop_534))) # shape=(8, 6, 2)
bop_553 = relay.divide(bop_544.astype('float32'), relay.reshape(bop_529.astype('float32'), relay.shape_of(bop_544))) # shape=(8, 6, 2)
bop_556 = relay.left_shift(uop_538.astype('uint32'), relay.reshape(uop_521.astype('uint32'), relay.shape_of(uop_538))) # shape=(8, 6, 2)
output = relay.Tuple([uop_516,call_540,var_541,const_542,bop_549,bop_553,bop_556,])
output2 = relay.Tuple([uop_516,call_543,var_541,const_542,bop_549,bop_553,bop_556,])
func_564 = relay.Function([var_482,var_483,var_498,var_528,var_541,], output)
mod['func_564'] = func_564
mod = relay.transform.InferType()(mod)
mutated_mod['func_564'] = func_564
mutated_mod = relay.transform.InferType()(mutated_mod)
func_564_call = mutated_mod.get_global_var('func_564')
var_566 = relay.var("var_566", dtype = "uint8", shape = (8, 6, 2))#candidate|566|(8, 6, 2)|var|uint8
var_567 = relay.var("var_567", dtype = "uint8", shape = (8, 6, 2))#candidate|567|(8, 6, 2)|var|uint8
var_568 = relay.var("var_568", dtype = "uint8", shape = (8, 6, 2))#candidate|568|(8, 6, 2)|var|uint8
var_569 = relay.var("var_569", dtype = "float64", shape = (8, 6, 2))#candidate|569|(8, 6, 2)|var|float64
var_570 = relay.var("var_570", dtype = "uint8", shape = (16,))#candidate|570|(16,)|var|uint8
call_565 = func_564_call(var_566,var_567,var_568,var_569,var_570,)
output = call_565
func_571 = relay.Function([var_566,var_567,var_568,var_569,var_570,], output)
mutated_mod['func_571'] = func_571
mutated_mod = relay.transform.InferType()(mutated_mod)
var_607 = relay.var("var_607", dtype = "float32", shape = (7, 11))#candidate|607|(7, 11)|var|float32
uop_608 = relay.rsqrt(var_607.astype('float32')) # shape=(7, 11)
output = uop_608
output2 = uop_608
func_612 = relay.Function([var_607,], output)
mod['func_612'] = func_612
mod = relay.transform.InferType()(mod)
var_613 = relay.var("var_613", dtype = "float32", shape = (7, 11))#candidate|613|(7, 11)|var|float32
output = func_612(var_613)
func_614 = relay.Function([var_613], output)
mutated_mod['func_614'] = func_614
mutated_mod = relay.transform.InferType()(mutated_mod)
var_687 = relay.var("var_687", dtype = "uint32", shape = (6, 6, 16))#candidate|687|(6, 6, 16)|var|uint32
var_688 = relay.var("var_688", dtype = "uint32", shape = (6, 6, 16))#candidate|688|(6, 6, 16)|var|uint32
bop_689 = relay.multiply(var_687.astype('uint32'), relay.reshape(var_688.astype('uint32'), relay.shape_of(var_687))) # shape=(6, 6, 16)
func_148_call = mod.get_global_var('func_148')
func_151_call = mutated_mod.get_global_var('func_151')
const_701 = relay.const([-1.976583,-5.097276,6.849880,-6.097151,-4.516620,6.299290,1.520781,-3.705308,-3.789906,-0.887738,0.839560,-4.707511], dtype = "float32")#candidate|701|(12,)|const|float32
call_700 = func_148_call(relay.reshape(const_701.astype('float32'), [12,]))
call_702 = func_148_call(relay.reshape(const_701.astype('float32'), [12,]))
bop_703 = relay.less(var_688.astype('bool'), relay.reshape(bop_689.astype('bool'), relay.shape_of(var_688))) # shape=(6, 6, 16)
bop_714 = relay.less(bop_703.astype('bool'), relay.reshape(bop_689.astype('bool'), relay.shape_of(bop_703))) # shape=(6, 6, 16)
uop_720 = relay.sqrt(bop_714.astype('float32')) # shape=(6, 6, 16)
bop_723 = relay.bitwise_or(uop_720.astype('int8'), relay.reshape(bop_714.astype('int8'), relay.shape_of(uop_720))) # shape=(6, 6, 16)
uop_728 = relay.cos(bop_723.astype('float32')) # shape=(6, 6, 16)
bop_730 = relay.add(uop_728.astype('int16'), relay.reshape(var_688.astype('int16'), relay.shape_of(uop_728))) # shape=(6, 6, 16)
bop_734 = relay.logical_or(uop_720.astype('bool'), relay.reshape(var_688.astype('bool'), relay.shape_of(uop_720))) # shape=(6, 6, 16)
uop_741 = relay.sigmoid(bop_734.astype('float32')) # shape=(6, 6, 16)
uop_743 = relay.sin(uop_720.astype('float64')) # shape=(6, 6, 16)
var_745 = relay.var("var_745", dtype = "float32", shape = (6, 6, 16))#candidate|745|(6, 6, 16)|var|float32
bop_746 = relay.maximum(uop_741.astype('uint16'), relay.reshape(var_745.astype('uint16'), relay.shape_of(uop_741))) # shape=(6, 6, 16)
uop_749 = relay.exp(bop_723.astype('float64')) # shape=(6, 6, 16)
bop_755 = relay.equal(bop_746.astype('bool'), relay.reshape(bop_689.astype('bool'), relay.shape_of(bop_746))) # shape=(6, 6, 16)
bop_760 = relay.left_shift(bop_723.astype('uint16'), relay.reshape(uop_728.astype('uint16'), relay.shape_of(bop_723))) # shape=(6, 6, 16)
var_764 = relay.var("var_764", dtype = "uint16", shape = (6, 6, 16))#candidate|764|(6, 6, 16)|var|uint16
bop_765 = relay.not_equal(bop_760.astype('bool'), relay.reshape(var_764.astype('bool'), relay.shape_of(bop_760))) # shape=(6, 6, 16)
uop_772 = relay.log10(uop_728.astype('float64')) # shape=(6, 6, 16)
bop_774 = relay.subtract(uop_772.astype('int8'), relay.reshape(var_687.astype('int8'), relay.shape_of(uop_772))) # shape=(6, 6, 16)
bop_777 = relay.power(bop_774.astype('float64'), relay.reshape(bop_760.astype('float64'), relay.shape_of(bop_774))) # shape=(6, 6, 16)
bop_783 = relay.logical_xor(uop_772.astype('int64'), relay.reshape(uop_741.astype('int64'), relay.shape_of(uop_772))) # shape=(6, 6, 16)
output = relay.Tuple([call_700,const_701,bop_730,uop_743,uop_749,bop_755,bop_765,bop_777,bop_783,])
output2 = relay.Tuple([call_702,const_701,bop_730,uop_743,uop_749,bop_755,bop_765,bop_777,bop_783,])
func_786 = relay.Function([var_687,var_688,var_745,var_764,], output)
mod['func_786'] = func_786
mod = relay.transform.InferType()(mod)
mutated_mod['func_786'] = func_786
mutated_mod = relay.transform.InferType()(mutated_mod)
func_786_call = mutated_mod.get_global_var('func_786')
var_788 = relay.var("var_788", dtype = "uint32", shape = (6, 6, 16))#candidate|788|(6, 6, 16)|var|uint32
var_789 = relay.var("var_789", dtype = "uint32", shape = (6, 6, 16))#candidate|789|(6, 6, 16)|var|uint32
var_790 = relay.var("var_790", dtype = "float32", shape = (6, 6, 16))#candidate|790|(6, 6, 16)|var|float32
var_791 = relay.var("var_791", dtype = "uint16", shape = (6, 6, 16))#candidate|791|(6, 6, 16)|var|uint16
call_787 = func_786_call(var_788,var_789,var_790,var_791,)
output = call_787
func_792 = relay.Function([var_788,var_789,var_790,var_791,], output)
mutated_mod['func_792'] = func_792
mutated_mod = relay.transform.InferType()(mutated_mod)
const_814 = relay.const([[-9.277580],[9.164220],[-8.412375],[3.587912],[-6.660771],[-8.343874],[-7.297474],[-4.874341]], dtype = "float64")#candidate|814|(8, 1)|const|float64
uop_815 = relay.tan(const_814.astype('float64')) # shape=(8, 1)
func_451_call = mod.get_global_var('func_451')
func_455_call = mutated_mod.get_global_var('func_455')
var_820 = relay.var("var_820", dtype = "float64", shape = (78,))#candidate|820|(78,)|var|float64
call_819 = relay.TupleGetItem(func_451_call(relay.reshape(var_820.astype('float64'), [13, 6]), relay.reshape(var_820.astype('float64'), [13, 6]), ), 0)
call_821 = relay.TupleGetItem(func_455_call(relay.reshape(var_820.astype('float64'), [13, 6]), relay.reshape(var_820.astype('float64'), [13, 6]), ), 0)
var_823 = relay.var("var_823", dtype = "float64", shape = (8, 3))#candidate|823|(8, 3)|var|float64
bop_824 = relay.equal(uop_815.astype('bool'), var_823.astype('bool')) # shape=(8, 3)
bop_827 = relay.multiply(uop_815.astype('int16'), var_823.astype('int16')) # shape=(8, 3)
bop_832 = relay.bitwise_and(uop_815.astype('int8'), bop_824.astype('int8')) # shape=(8, 3)
output = relay.Tuple([call_819,var_820,bop_827,bop_832,])
output2 = relay.Tuple([call_821,var_820,bop_827,bop_832,])
func_835 = relay.Function([var_820,var_823,], output)
mod['func_835'] = func_835
mod = relay.transform.InferType()(mod)
mutated_mod['func_835'] = func_835
mutated_mod = relay.transform.InferType()(mutated_mod)
func_835_call = mutated_mod.get_global_var('func_835')
var_837 = relay.var("var_837", dtype = "float64", shape = (78,))#candidate|837|(78,)|var|float64
var_838 = relay.var("var_838", dtype = "float64", shape = (8, 3))#candidate|838|(8, 3)|var|float64
call_836 = func_835_call(var_837,var_838,)
output = call_836
func_839 = relay.Function([var_837,var_838,], output)
mutated_mod['func_839'] = func_839
mutated_mod = relay.transform.InferType()(mutated_mod)
var_854 = relay.var("var_854", dtype = "float32", shape = (12, 12, 13))#candidate|854|(12, 12, 13)|var|float32
uop_855 = relay.acos(var_854.astype('float32')) # shape=(12, 12, 13)
bop_857 = relay.floor_divide(var_854.astype('float32'), relay.reshape(uop_855.astype('float32'), relay.shape_of(var_854))) # shape=(12, 12, 13)
bop_860 = relay.add(bop_857.astype('float64'), relay.reshape(var_854.astype('float64'), relay.shape_of(bop_857))) # shape=(12, 12, 13)
bop_863 = relay.bitwise_xor(bop_857.astype('uint8'), relay.reshape(uop_855.astype('uint8'), relay.shape_of(bop_857))) # shape=(12, 12, 13)
bop_870 = relay.logical_xor(bop_857.astype('uint32'), relay.reshape(bop_860.astype('uint32'), relay.shape_of(bop_857))) # shape=(12, 12, 13)
output = relay.Tuple([bop_863,bop_870,])
output2 = relay.Tuple([bop_863,bop_870,])
func_873 = relay.Function([var_854,], output)
mod['func_873'] = func_873
mod = relay.transform.InferType()(mod)
mutated_mod['func_873'] = func_873
mutated_mod = relay.transform.InferType()(mutated_mod)
var_874 = relay.var("var_874", dtype = "float32", shape = (12, 12, 13))#candidate|874|(12, 12, 13)|var|float32
func_873_call = mutated_mod.get_global_var('func_873')
call_875 = func_873_call(var_874)
output = call_875
func_876 = relay.Function([var_874], output)
mutated_mod['func_876'] = func_876
mutated_mod = relay.transform.InferType()(mutated_mod)
const_921 = relay.const([[6.354367,-9.786599,-4.913186],[-8.324383,2.386799,-7.826004],[2.620896,4.772198,1.908474],[-9.083943,-7.846844,-2.531224],[-2.622167,2.712950,2.244258],[0.278421,5.625398,8.928418],[-1.723301,5.735179,-2.121857],[2.467065,-6.842799,-0.368299],[0.061931,-4.890595,8.930602],[8.445549,-3.524242,0.238916],[-1.147883,-1.237445,-0.213206]], dtype = "float32")#candidate|921|(11, 3)|const|float32
var_922 = relay.var("var_922", dtype = "float32", shape = (11, 3))#candidate|922|(11, 3)|var|float32
bop_923 = relay.less_equal(const_921.astype('bool'), relay.reshape(var_922.astype('bool'), relay.shape_of(const_921))) # shape=(11, 3)
output = bop_923
output2 = bop_923
func_931 = relay.Function([var_922,], output)
mod['func_931'] = func_931
mod = relay.transform.InferType()(mod)
mutated_mod['func_931'] = func_931
mutated_mod = relay.transform.InferType()(mutated_mod)
var_932 = relay.var("var_932", dtype = "float32", shape = (11, 3))#candidate|932|(11, 3)|var|float32
func_931_call = mutated_mod.get_global_var('func_931')
call_933 = func_931_call(var_932)
output = call_933
func_934 = relay.Function([var_932], output)
mutated_mod['func_934'] = func_934
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1000 = relay.var("var_1000", dtype = "float32", shape = (9, 11))#candidate|1000|(9, 11)|var|float32
var_1001 = relay.var("var_1001", dtype = "float32", shape = (9, 11))#candidate|1001|(9, 11)|var|float32
bop_1002 = relay.floor_divide(var_1000.astype('float32'), relay.reshape(var_1001.astype('float32'), relay.shape_of(var_1000))) # shape=(9, 11)
const_1011 = relay.const([[-5.000118,-6.858555,5.602528,-8.418250,-6.101142,-0.772989,-2.490909,-1.110481,9.205004,3.208241,9.577353],[-6.987441,2.413040,1.195119,4.627963,0.272092,9.433262,3.148449,0.500093,-5.899883,-7.792425,5.391975],[-3.155622,-2.132160,-1.652990,2.349990,-3.173595,-7.873801,-0.006588,-2.594568,3.895923,0.395469,0.687783],[7.268395,-4.301417,-9.884143,-5.222108,-8.242474,7.332938,-7.993402,-1.278862,-0.539842,5.513997,-4.723016],[-5.039617,5.357424,0.770023,-6.638590,-3.428317,9.873875,7.209280,9.695389,8.407598,6.937294,-8.523756],[9.303186,1.393999,-3.533913,-5.475928,-5.655493,8.580661,-8.606943,8.076550,0.580263,-9.250628,-4.802930],[-5.310510,5.891492,4.746931,-0.455221,-9.172179,-2.985668,5.908885,-1.401575,-7.135209,-8.277555,5.933000],[-4.494557,-2.929219,-0.077195,-6.591697,-4.817645,-0.781412,6.669151,-9.756817,-6.696109,6.400945,-5.434787],[6.973778,-5.158848,-8.137763,3.011519,-3.081898,-7.264277,9.178485,8.874341,1.593811,-8.270898,-8.464944]], dtype = "float32")#candidate|1011|(9, 11)|const|float32
bop_1012 = relay.floor_mod(bop_1002.astype('float64'), relay.reshape(const_1011.astype('float64'), relay.shape_of(bop_1002))) # shape=(9, 11)
bop_1015 = relay.floor_divide(const_1011.astype('float64'), relay.reshape(bop_1012.astype('float64'), relay.shape_of(const_1011))) # shape=(9, 11)
bop_1018 = relay.multiply(var_1001.astype('int16'), relay.reshape(bop_1015.astype('int16'), relay.shape_of(var_1001))) # shape=(9, 11)
func_303_call = mod.get_global_var('func_303')
func_308_call = mutated_mod.get_global_var('func_308')
var_1026 = relay.var("var_1026", dtype = "uint8", shape = (3600,))#candidate|1026|(3600,)|var|uint8
const_1027 = relay.const([-4.960160,2.639742,4.374388,8.673165,9.987294,-4.703926,5.200551,-3.976527,9.359100,7.291812,-7.927871,-6.754033,1.358013,-0.708148,-6.134414,-3.509100,3.849801,9.888161,8.759822,5.594326,9.486497,-2.389501,-0.615556,9.302182,-3.086454,7.677690,-5.711475,-3.445594,6.469641,-2.078443,7.438447,8.262524,0.542633,7.412264,5.249493,1.025183,7.709156,2.079256,-6.631765,4.911538,7.442064,9.442831,-1.757650,4.608698,5.712160,4.432711,-7.878101,3.706837,-1.635496,-5.406263,-9.806236,-0.176486,7.644675,2.293332,-0.426682,-1.474240,7.359450,-7.025920,-3.256471,-2.038914,4.213630,-5.345966,-7.094555,2.057094,-2.814598,-2.842487,4.734742,-9.218709,-2.017618,-1.602947,-0.458647,-8.170948,5.386165,-9.532114,-1.110346,8.600120,-1.344642,7.333080,9.320134,7.819631,4.013419,-2.535457,-2.203900,0.024482,-7.519674,-5.624159,6.427411,5.446272,6.392049,-2.539192,-5.680106,4.498686,6.543740,0.020452,-1.076777,-3.732375,-9.885533,-5.207944,-7.774044,8.540009,-8.320465,8.950080,-1.002964,-7.344419,-0.510018,1.339296,-5.012262,-7.692512,1.933877,-2.609365,-5.666791,6.467225], dtype = "float64")#candidate|1027|(112,)|const|float64
call_1025 = relay.TupleGetItem(func_303_call(relay.reshape(var_1026.astype('uint8'), [15, 15, 16]), relay.reshape(var_1026.astype('uint8'), [15, 15, 16]), relay.reshape(const_1027.astype('float64'), [112,]), relay.reshape(var_1026.astype('bool'), [15, 15, 16]), ), 3)
call_1028 = relay.TupleGetItem(func_308_call(relay.reshape(var_1026.astype('uint8'), [15, 15, 16]), relay.reshape(var_1026.astype('uint8'), [15, 15, 16]), relay.reshape(const_1027.astype('float64'), [112,]), relay.reshape(var_1026.astype('bool'), [15, 15, 16]), ), 3)
output = relay.Tuple([bop_1018,call_1025,var_1026,const_1027,])
output2 = relay.Tuple([bop_1018,call_1028,var_1026,const_1027,])
func_1029 = relay.Function([var_1000,var_1001,var_1026,], output)
mod['func_1029'] = func_1029
mod = relay.transform.InferType()(mod)
var_1030 = relay.var("var_1030", dtype = "float32", shape = (9, 11))#candidate|1030|(9, 11)|var|float32
var_1031 = relay.var("var_1031", dtype = "float32", shape = (9, 11))#candidate|1031|(9, 11)|var|float32
var_1032 = relay.var("var_1032", dtype = "uint8", shape = (3600,))#candidate|1032|(3600,)|var|uint8
output = func_1029(var_1030,var_1031,var_1032,)
func_1033 = relay.Function([var_1030,var_1031,var_1032,], output)
mutated_mod['func_1033'] = func_1033
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1056 = relay.const([[0.995248],[9.598932],[-7.948386],[-8.092527],[9.499869],[-2.986927],[3.001436],[5.776927],[3.930767],[3.731786]], dtype = "float32")#candidate|1056|(10, 1)|const|float32
uop_1057 = relay.exp(const_1056.astype('float32')) # shape=(10, 1)
var_1061 = relay.var("var_1061", dtype = "float32", shape = (10, 7))#candidate|1061|(10, 7)|var|float32
bop_1062 = relay.divide(uop_1057.astype('float64'), var_1061.astype('float64')) # shape=(10, 7)
func_564_call = mod.get_global_var('func_564')
func_571_call = mutated_mod.get_global_var('func_571')
var_1066 = relay.var("var_1066", dtype = "uint8", shape = (96,))#candidate|1066|(96,)|var|uint8
const_1067 = relay.const([10,-4,7,7,-9,-1,-9,-5,7,-6,-4,-6,-5,3,7,-5], dtype = "uint8")#candidate|1067|(16,)|const|uint8
call_1065 = relay.TupleGetItem(func_564_call(relay.reshape(var_1066.astype('uint8'), [8, 6, 2]), relay.reshape(var_1066.astype('uint8'), [8, 6, 2]), relay.reshape(var_1066.astype('uint8'), [8, 6, 2]), relay.reshape(var_1066.astype('float64'), [8, 6, 2]), relay.reshape(const_1067.astype('uint8'), [16,]), ), 6)
call_1068 = relay.TupleGetItem(func_571_call(relay.reshape(var_1066.astype('uint8'), [8, 6, 2]), relay.reshape(var_1066.astype('uint8'), [8, 6, 2]), relay.reshape(var_1066.astype('uint8'), [8, 6, 2]), relay.reshape(var_1066.astype('float64'), [8, 6, 2]), relay.reshape(const_1067.astype('uint8'), [16,]), ), 6)
uop_1069 = relay.log(bop_1062.astype('float32')) # shape=(10, 7)
func_1029_call = mod.get_global_var('func_1029')
func_1033_call = mutated_mod.get_global_var('func_1033')
var_1074 = relay.var("var_1074", dtype = "float32", shape = (99,))#candidate|1074|(99,)|var|float32
const_1075 = relay.const([-6,4,-10,5,-5,-5,6,6,-4,9,1,2,9,-2,-8,-9,-9,-6,-1,-6,-5,3,-8,1,5,-4,1,10,-3,6,-7,-6,7,-5,2,-7,2,-8,-1,4,8,-4,-5,-8,-5,2,1,-8,1,-7,5,-1,5,9,8,-6,5,-5,4,-9,-10,-3,-8,-9,4,-3,9,-8,-3,4,2,10,-10,6,-10,5,5,-1,8,-3,6,-3,-1,-3,8,10,7,-7,-4,-5,7,4,-9,-6,-5,-9,-8,7,-3,-8,8,-1,7,-4,-9,7,10,-10,6,-5,-9,-3,-9,6,9,-3,1,3,-9,1,7,-6,7,7,-3,6,9,9,-10,-9,9,7,3,-10,-9,10,6,5,3,-1,-4,-8,-6,2,2,-6,-3,-4,5,1,-4,-1,9,6,-10,8,-3,9,-3,-7,4,7,10,-6,-5,8,7,10,7,-8,4,-3,4,8,4,-4,3,-6,5,8,6,-9,-7,-9,-6,3,4,-9,-7,8,6,-10,5,2,5,-2,-7,-5,2,3,4,-10,-2,-9,-7,2,10,4,5,7,-7,-4,10,3,7,1,-1,-10,-1,-6,7,-4,8,-4,6,-7,-8,9,4,1,-7,-3,-3,-3,-3,5,-7,-8,-8,1,2,9,1,-10,-10,-4,-3,-7,-1,3,1,5,8,2,4,2,-1,7,-8,5,-10,-7,8,-3,4,5,5,-3,3,-1,-9,4,3,-2,-10,1,9,-5,3,8,-6,-5,2,6,-3,2,-9,2,6,6,-2,-8,-6,8,2,7,6,9,-7,-3,3,2,10,5,-9,-3,-8,2,3,1,3,-5,9,-8,10,-7,-5,-4,-5,2,-2,5,-6,-3,-6,-6,4,1,-7,10,-10,-10,-5,1,-5,-6,-8,6,7,-2,-8,3,2,-5,-2,-6,-9,-9,-1,6,1,-3,-2,-2,-10,3,7,-10,-1,7,10,-8,-10,-5,7,-8,-5,7,10,8,6,-10,9,-1,4,6,3,-3,10,5,1,-6,-4,8,4,6,7,-2,-5,-9,-1,1,8,9,3,-7,-7,5,-8,5,-6,2,6,-4,-5,8,-7,-1,4,-6,4,-3,8,2,8,-8,8,-2,3,3,-10,-5,-1,-3,-6,4,-4,-3,-5,-8,4,-9,-5,-9,8,8,5,4,-10,3,-8,8,-1,4,-5,9,8,4,-1,-4,4,-9,-5,-7,10,1,-4,-9,9,3,4,3,-3,7,-9,5,3,2,-6,1,6,3,-10,-9,4,2,5,1,-4,4,-9,5,-2,10,-6,4,-4,6,-9,-4,8,-10,3,2,10,-1,2,3,-4,2,3,-4,2,10,4,4,-9,-4,8,10,-10,8,-8,8,1,2,7,-8,5,-10,-9,-10,9,-3,-9,-6,5,-3,-3,6,2,-10,3,9,6,-6,6,-5,9,-3,-2,-7,10,-9,2,1,-3,-4,8,4,-5,-1,6,5,6,4,-6,5,1,-1,9,-9,-10,9,-10,-2,-5,5,10,7,-1,-4,-1,3,2,7,7,-4,-10,-6,5,-5,-2,-2,-4,10,9,2,-4,-2,-10,-8,4,5,4,-6,9,1,-6,6,3,5,2,2,4,-2,4,1,-5,-3,8,-10,-1,-8,6,-8,2,1,-3,-9,8,1,-5,-9,1,-9,-1,5,2,10,-9,-8,-2,-4,3,-1,-7,10,-9,2,1,10,-10,6,2,1,4,4,-10,10,2,-2,-8,-10,-5,8,2,10,6,-8,-1,10,10,-3,2,4,7,-2,-10,3,-5,-1,-10,2,8,4,9,-7,10,9,-5,6,9,-8,-2,-6,3,-5,10,-9,-5,10,-9,-5,8,10,7,7,8,6,-6,7,2,4,-7,-10,6,-8,-8,-3,-7,10,-3,-9,-2,-10,-9,-10,9,3,-7,-10,-5,-7,9,9,-4,-9,4,7,7,6,2,-7,10,-7,-4,-9,3,-10,-5,-10,8,6,-9,9,10,-2,-3,-3,8,9,9,9,-3,6,-9,-9,-8,-10,9,3,-7,2,-1,-3,7,8,-10,-7,7,-3,2,3,8,-6,-7,6,3,3,3,2,-10,5,7,-4,-7,-1,10,8,-1,5,-5,-9,-7,2,-2,1,5,3,7,-6,-6,-5,3,4,-10,-10,-8,-3,5,6,-10,-9,-4,-6,-10,7,-1,2,-6,8,5,-2,8,9,-5,-1,3,-9,-2,8,-9,-10,7,-4,10,2,-1,8,2,-5,4,1,7,6,4,5,-1,4,9,-3,-2,-4,-9,-2,-2,9,4,5,9,5,9,9,-7,10,-2,6,-9,-5,-5,-1,-8,2,8,5,1,3,2,-10,6,4,4,-2,-2,-1,-7,-1,1,-7,3,10,-4,5,-6,4,6,2,9,3,10,-8,4,-2,-5,8,10,5,3,-9,-7,-1,-2,3,6,5,-9,4,4,-10,2,-8,-8,-6,3,7,8,-10,-2,4,-10,3,1,3,-1,6,-3,-6,7,9,7,9,6,6,4,-6,-8,-2,-6,6,-7,3,-7,10,-2,4,-9,-3,-4,-4,8,3,-9,-6,-1,6,-5,10,9,7,-1,1,6,6,4,8,-8,-9,-7,3,7,8,10,5,-9,1,3,-1,-7,-7,-5,-2,-6,-1,5,-1,-5,6,6,-5,2,-7,1,4,-3,8,7,-2,-10,-6,-8,9,5,-1,5,-5,5,3,-8,-3,-9,-2,8,9,-8,4,3,3,9,8,1,2,-5,-1,4,-5,-2,1,-2,1,7,-8,-1,8,-3,7,-10,-10,-5,-10,-5,-4,-4,1,7,-6,7,7,-4,-1,3,-3,-6,10,5,10,6,9,1,6,1,-2,-7,-8,4,-10,-5,2,7,3,-3,-4,-9,9,-4,-5,10,-5,6,9,-5,1,-3,3,6,-4,1,9,4,9,4,4,5,10,5,3,10,1,-8,5,-10,-4,8,2,-1,1,9,9,7,5,-1,-3,4,-4,10,-6,-10,-3,4,-10,8,-10,9,9,1,9,4,-2,-9,1,-8,10,-9,-6,-10,-8,10,-5,9,6,-10,1,8,-2,-2,6,10,-7,3,-2,10,3,-9,-8,4,8,10,8,5,-6,-3,-6,-2,-1,-4,8,-2,3,-1,2,4,7,-1,3,5,4,2,10,-3,-9,-4,3,9,-9,-7,-4,-3,7,8,-2,9,-2,8,-9,2,9,4,-4,-8,-3,-4,-7,10,2,8,-4,4,2,-7,-2,-6,7,-3,-4,3,-7,6,5,2,5,-5,-6,6,-8,6,2,-3,3,9,10,-7,-9,2,9,3,-5,5,-7,9,4,7,9,-6,-7,-2,-6,-5,1,1,3,-6,-4,2,-1,-9,8,2,8,-4,4,-4,-4,-5,6,7,-2,10,5,7,4,-3,-2,2,5,-6,1,-7,-9,-1,10,8,2,-6,-1,3,9,-10,10,-2,7,4,-8,-6,-7,5,7,-9,-5,-4,-10,-4,2,9,-5,9,-8,1,-5,-6,3,10,-7,7,-4,8,7,-2,-5,2,-1,-1,3,-7,-8,1,-4,-1,7,-3,-10,-4,5,7,6,-8,-8,6,-1,-10,-5,10,1,10,-1,1,-8,-10,9,-10,5,-2,7,6,10,-8,-8,-4,-2,-7,1,-6,-2,-9,2,-6,9,3,5,-5,5,9,4,2,-2,-2,-8,-2,-3,-3,-10,3,6,-5,-5,-7,-6,6,4,9,-1,-6,7,1,-9,-6,-1,10,7,2,10,6,8,7,2,-5,-8,1,-1,7,7,2,-4,-7,10,7,-9,-5,5,-8,-1,-9,5,4,4,-9,5,-1,4,2,-7,1,1,4,9,2,-4,8,-5,-7,-4,-2,-6,7,-10,1,9,-7,9,4,-1,9,-9,5,9,-5,-6,1,-9,-3,9,-10,7,7,-7,7,9,-5,3,4,-10,5,7,2,-4,7,-9,2,-4,-6,-1,-1,9,1,-6,5,6,-3,-8,9,2,-7,-2,-8,9,-9,5,3,6,7,8,9,6,9,9,-7,-5,-1,-5,-5,-1,5,6,1,9,-5,3,-5,8,-4,3,7,6,-6,4,-3,8,2,-5,10,-10,5,9,-4,-5,3,8,5,-7,-4,4,8,7,-8,3,-8,-2,-8,-4,-7,9,-6,-1,2,2,9,-3,-8,8,-5,-3,-6,8,6,-6,-1,-5,3,-8,4,-1,5,-1,-6,-10,-2,8,8,1,9,8,-2,9,-9,-10,-8,-5,5,-8,6,6,3,-5,-9,6,4,10,-9,5,5,-2,-6,-5,-6,6,3,10,-3,6,3,-9,-5,7,7,3,-7,6,3,-8,6,-1,-5,4,9,-1,-9,-1,1,-10,10,-2,-10,2,-2,4,-2,9,-7,-7,-3,7,1,8,-6,2,-4,2,-7,-1,3,-8,-8,6,2,-10,-4,-6,-6,8,-3,-7,7,4,4,-2,1,-5,9,-6,5,-3,-4,-3,-3,4,1,4,-3,-1,-6,-5,5,5,1,4,-9,6,-7,-6,10,-6,-1,1,1,-2,-5,3,9,3,-9,-10,5,2,3,-2,-7,-3,9,6,-6,-6,-7,2,-6,10,5,6,3,1,7,10,10,-6,-7,-10,1,5,-6,6,5,1,-3,7,-4,-4,2,7,-7,-4,5,1,-1,-6,5,-10,5,10,6,-5,-8,5,3,3,-7,8,10,-10,6,-7,-7,3,8,-1,1,-9,7,8,-6,-3,7,-4,1,-7,9,9,-2,-7,6,-7,6,-5,-9,-6,-5,9,10,-10,-9,-5,-2,4,7,-6,-1,-1,10,1,-2,-6,6,8,9,5,6,6,-6,-3,4,-9,4,-4,-5,-6,4,3,-5,5,10,2,4,3,-10,5,8,9,-4,-8,1,-8,-4,-6,4,3,-3,-2,-4,-9,3,-8,-9,-7,-1,-4,9,-7,4,3,7,4,2,-9,2,1,-3,8,-6,-7,-10,-1,-4,3,4,-8,4,2,-7,-2,-10,-6,-2,7,9,-8,6,9,2,7,-10,8,6,-9,3,-4,6,8,-6,7,8,4,7,-3,7,-6,7,-8,5,8,1,-3,-5,5,7,-5,-9,10,-4,6,4,-6,-5,-4,4,-8,-7,3,3,1,2,-4,7,-1,-8,-4,-9,3,-7,-2,-2,-7,1,-1,8,4,-10,-4,-4,-9,2,-5,6,-2,-7,-5,7,8,-3,-4,-10,2,-9,-8,1,-5,3,-1,10,-1,2,-5,-9,8,-1,5,-3,3,3,-6,9,7,-3,-10,-6,1,-10,-6,1,-2,-8,-8,-4,-4,-2,-3,-4,-4,1,-4,-6,2,3,-4,-3,9,-3,-4,-5,6,-1,-10,-9,3,-6,-6,2,-8,-7,-9,1,4,4,1,-8,-9,-5,-3,-1,6,-3,-2,2,10,-4,-4,7,-9,7,2,-6,-4,8,-7,8,-8,8,8,2,3,-8,-3,4,8,1,-7,-9,-7,-2,7,10,1,-7,-9,-8,9,9,3,-3,7,10,-10,-4,-7,-5,-6,-2,-4,5,-2,-8,-7,2,-2,-10,8,2,-1,8,3,-4,8,-6,-7,-6,4,-9,-2,-2,5,-3,1,2,2,-2,1,5,-4,-2,4,1,3,-8,10,-9,-9,9,-8,4,8,8,-1,-1,-6,7,10,-5,-9,-7,-10,-3,1,-2,2,-10,1,-2,-10,7,-5,-3,4,9,10,2,4,-10,-7,3,-8,2,10,-4,-8,-6,7,9,5,5,-3,7,2,4,2,1,-2,-1,-8,5,-6,-10,-8,4,-6,-1,4,8,-5,-7,10,7,2,6,-3,-7,-2,9,5,-5,2,10,4,-2,-10,6,-9,-4,2,-3,3,-1,-7,6,4,1,-9,-2,8,-2,6,-10,4,6,-6,-9,8,-5,-2,7,4,4,10,6,3,-8,-7,-4,2,7,8,8,-7,1,8,9,-7,-5,10,-6,-9,-2,1,-7,3,-3,-1,-6,-6,10,-6,-5,-2,1,10,-5,-2,-5,-2,-7,6,5,-3,-8,2,2,-8,1,5,-2,-1,-4,-5,-7,-1,1,4,-7,-8,5,-4,8,-2,-2,2,4,2,-6,6,3,8,7,9,-1,2,9,-3,-7,-7,-3,-2,4,-4,-10,-6,8,-4,-7,2,2,-4,-7,7,3,1,-2,-5,-7,-9,-9,-10,8,-3,-7,-3,-6,-9,5,9,9,-5,4,-9,9,10,2,5,3,7,5,-10,-6,-6,-6,-10,-4,6,-3,1,-3,-9,10,-7,4,-7,-1,-9,-8,1,-9,-7,-3,3,7,9,-6,7,4,10,-7,6,4,8,-2,5,7,8,-6,8,-6,5,5,7,7,7,2,3,-1,-9,-9,8,-6,-6,7,8,-10,3,-9,5,8,2,-7,8,-6,-3,-1,1,-7,-2,2,8,-4,-7,3,-1,-4,-9,-7,-1,3,-10,-6,2,9,-10,3,-4,10,10,-3,-2,-3,2,-9,-5,-8,5,-7,9,-10,-8,8,-1,-3,-10,3,7,-5,6,-1,-6,-5,5,9,-7,-3,-3,-6,9,-5,-10,-3,4,6,-3,10,-8,-2,6,-6,-2,7,10,-10,-7,-10,-5,1,-4,-7,6,6,-2,-4,10,1,-9,-7,-6,-9,-4,-9,5,-8,9,-6,5,-8,10,7,-3,4,-9,-2,2,7,4,4,-5,8,10,-10,6,3,-1,-8,9,8,-4,2,-6,1,4,-9,2,5,-3,-8,-3,5,-9,10,-6,5,-9,-8,-5,-4,8,10,-3,1,-9,5,8,-8,6,7,6,-3,-7,9,2,3,-1,10,-7,1,10,2,7,-6,-7,6,3,8,4,3,3,6,10,-9,8,5,-4,-5,6,-6,2,-6,-3,6,4,-3,-7,2,-8,-6,-8,2,9,-7,-9,-10,6,6,2,9,5,-6,5,-10,10,-9,9,2,-1,3,-3,10,10,-3,10,-10,-7,7,-4,-5,5,3,-7,1,8,-9,6,-4,-5,-1,-10,2,-7,9,8,-8,-2,-6,-7,-9,1,-8,7,-3,9,-9,-5,8,8,3,-5,-8,1,-6,-10,-5,-2,6,-5,2,-3,4,-2,9,-2,-5,-3,-2,8,5,4,7,-5,7,-3,5,-10,5,-1,-1,9,10,10,8,-3,-5,5,4,6,1,10,7,-4,1,9,8,9,-9,3,10,2,-2,-7,8,-6,3,4,-6,-2,-2,1,6,2,-3,-5,-9,-8,6,5,-9,5,-9,-9,5,-3,2,-9,-8,-6,-5,3,9,-5,7,1,1,5,2,4,-5,4,-3,-8,5,-1,6,-4,-5,10,-9,7,4,1,-6,-3,5,2,4,8,-5,-4,3,6,9,10,-6,9,9,-7,-8,-9,2,9,-1,-9,-6,-7,-1,6,8,7,3,7,-7,-10,-9,-5,10,8,3,-8,-10,-7,-8,10,-7,4,-4,-5,-5,5,-3,4,-8,-5,-2,9,1,-3,-7,7,-4,7,1,-9,4,-10,-6,-10,-7,-5,-9,-10,-10,-1,8,4,-10,9,10,8,4,1,-10,-6,6,-5,8,5,-8,7,-4,-9,-10,6,8,4,-9,6,3,-8,3,-4,-9,-9,7,5,4,6,4,-4,5,6,2,-5,4,-4,7,8,-8,4,-2,7,-5,-10,6,-1,1,-4,1,-7,-3,4,8,-9,-8,4,-2,-5,-5,-5,-8,-1,-4,-9,-6,-2,7,10,-7,-4,2,4,-8,-10,-6,1,1,-3,8,-2,8,3,9,1,-4,8,4,9,-4,2,-6,-6,-6,2,10,-8,7,4,-7,-1,-1,-7,2,6,9,-8,2,9,-2,1,6,-10,3,10,-10,-2,-9,-7,-9,9,-10,10,-2,9,-2,3,2,-3,-9,4,4,-10,7,6,-10,-3,-3,-2,3,-8,-4,3,6,10,10,3,-1,9,-2,-5,1,5,-10,8,2,8,-8,-1,-10,-10,-6,4,-9,9,-7,8,-2,-1,-1,8,6,-1,-9,-1,-3,9,7,9,8,9,-2,-5,-9,-9,-7,8,-4,-2,-9,-2,-7,4,-4,-9,-2,-8,-5,-1,1,-9,-8,6,3,-5,-10,-9,-5,10,-7,-2,9,-5,4,3,7,-6,8,-7,9,-5,8,-7,-6,-4,7,-6,-7,3,7,1,-9,-1,8,-7,-1,-5,6,-5,8,-9,-5,-1,4,-8,-1,8,-5,-6,8,-9,8,1,-5,-8,6,-7,-10,5,-4,8,-1,5,5,6,-5,-6,10,-6,-4,7,-4,10,3,-10,8,2,4,3,7,7,-6,-6,8,1,-8,1,3,4,7,-4,-3,4,2,-9,9,-4,8,-1,-9,8,1,9,-1,9,-8,2,-9,-1,4,-1,-5,-7,-2,9,1,1,9,5,3,-7,-8,4,-5,-8,4,-1,-6,-2,-7,4,-10,-7,4,-10,-5,-10,6,-10,-9,-2,-10,2,1,4,4,-5,2,-5,5,5,-2,5,-1,1,-3,-2,7,-9,-5,8,1,-8,-7,-1,-8,-4,-5,10,-4,-7,5,-2,-1,-3,-3,10,4,9,2,-4,10,3,-7,-1,-6,-9,-8,-7,9,-9,-1,-10,-6,-8,1,-1,2,-6,8,1,8,9,9,6,6,-4,-4,-3,3,1,5,-7,-7,-7,4,3,8,-9,9,-8,-1,-1,-7,7,8,-6,-1,2,-7,7,7,-3,-3,5,-8,-7,5,3,-4,4,8,5,-4,-9,5,2,5,8,-10,2,-6,-8,-8,-3,-3,6,-6,3,9,-10,-8,-6,-10,1,-7,-7,-3,-4,-5,-5,3,-10,-4,8,4,8,10,3,8,1,10,-10,2,3,-10,-6,2,9,-10,9,-5,-4,-9,-9,4,6,-8,6,-10,-9,-7,-3,-2,5,9,4,1,3,8,8,-1,-5,-3,-9,9,-3,-3,-7,9,5,-6,-4,-8,8,8,-9,2,-8,-4,-8,-4,-1,-9,2,-2,8,7,-4,10,2,-10,8,4,7,-4,-2,9,-7,-4,6,5,4,7,-10,2,6,-2,3,2,-7,6,7,5,-4,1,4,6,-2,8,10,4,1,2,-6,-4,3,10,-5,-3,5,1,-2,7,-9,-5,-4,-7,-7,-9,-5,3,2,9,-10,8,-8,5,5,-5,-9,-3,9,8,-1,8,9,-8,-7,-10,-9,2,-9,-7,-9,1,3,10,6,-10,6,-4,9,6,9,3,-6,5,-2,-2,-7,8,4,3,-8,6,-1,9,-2,-6,-4,-9,3,-6,8,-3,-8,4,1,-4,8,-4,-2,-9,-9,3,3,4,6,9,7,-9,-5,10,10,-6,6,9,3,10,2,-5,-5,-1,10,-7,1,3,8,-3,-9,-9,9,1,-4,3,-4,3,1,10,-1,10,2,-10,-9,9,10,-1,3,-10,-2,-5,5,-6,1,8,3,-3,-3,5,6,10,-1,-6,2,5,10,10,9,4,2,-9,-1,-1,-7,-6,2,8,8,10,-9,6,5,7,-2,4,-2,-10,-7,-7,7,5,3,3,-5,-1,-7,4,-5,-8,9,-10,6,5,-10,8,-10,-7,6,4,4,-8,4,-7,-2,-5,-8,3,6,-9,-2,4,-2,5,-4,1,-4], dtype = "uint8")#candidate|1075|(3600,)|const|uint8
call_1073 = relay.TupleGetItem(func_1029_call(relay.reshape(var_1074.astype('float32'), [9, 11]), relay.reshape(var_1074.astype('float32'), [9, 11]), relay.reshape(const_1075.astype('uint8'), [3600,]), ), 3)
call_1076 = relay.TupleGetItem(func_1033_call(relay.reshape(var_1074.astype('float32'), [9, 11]), relay.reshape(var_1074.astype('float32'), [9, 11]), relay.reshape(const_1075.astype('uint8'), [3600,]), ), 3)
uop_1078 = relay.sqrt(uop_1057.astype('float32')) # shape=(10, 1)
output = relay.Tuple([call_1065,var_1066,const_1067,uop_1069,call_1073,var_1074,const_1075,uop_1078,])
output2 = relay.Tuple([call_1068,var_1066,const_1067,uop_1069,call_1076,var_1074,const_1075,uop_1078,])
func_1080 = relay.Function([var_1061,var_1066,var_1074,], output)
mod['func_1080'] = func_1080
mod = relay.transform.InferType()(mod)
mutated_mod['func_1080'] = func_1080
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1080_call = mutated_mod.get_global_var('func_1080')
var_1082 = relay.var("var_1082", dtype = "float32", shape = (10, 7))#candidate|1082|(10, 7)|var|float32
var_1083 = relay.var("var_1083", dtype = "uint8", shape = (96,))#candidate|1083|(96,)|var|uint8
var_1084 = relay.var("var_1084", dtype = "float32", shape = (99,))#candidate|1084|(99,)|var|float32
call_1081 = func_1080_call(var_1082,var_1083,var_1084,)
output = call_1081
func_1085 = relay.Function([var_1082,var_1083,var_1084,], output)
mutated_mod['func_1085'] = func_1085
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1149 = relay.var("var_1149", dtype = "bool", shape = (7, 4, 3))#candidate|1149|(7, 4, 3)|var|bool
var_1150 = relay.var("var_1150", dtype = "bool", shape = (7, 4, 3))#candidate|1150|(7, 4, 3)|var|bool
bop_1151 = relay.logical_or(var_1149.astype('bool'), relay.reshape(var_1150.astype('bool'), relay.shape_of(var_1149))) # shape=(7, 4, 3)
bop_1157 = relay.multiply(bop_1151.astype('float64'), relay.reshape(var_1149.astype('float64'), relay.shape_of(bop_1151))) # shape=(7, 4, 3)
const_1160 = relay.const([[[4.719031,0.208614,1.872112],[7.311452,5.271495,5.119044],[4.477552,1.650193,9.403240],[8.660852,-0.410981,-9.545895]],[[1.370466,3.930806,-0.960240],[6.156148,4.112166,-0.580296],[-7.782220,-6.295490,5.718049],[7.267976,3.401738,0.963064]],[[-0.228981,1.612246,8.674173],[0.053201,-6.769108,6.755974],[7.610539,-3.127957,5.617341],[-5.235967,-2.416407,-2.826028]],[[-8.529289,-0.487872,-4.902289],[-6.579734,9.938568,8.372237],[5.984035,5.915459,5.289573],[9.520834,6.643263,-1.637360]],[[-8.249962,-2.022828,3.569037],[-3.557170,5.080141,0.624238],[-7.237567,-5.810869,8.760421],[1.900615,7.923569,8.180444]],[[2.333078,7.974320,-7.034309],[3.718675,-3.169635,-7.466382],[-5.555564,6.526649,0.062949],[6.624484,0.512868,4.476526]],[[-9.037377,3.387799,-2.388936],[-4.243697,2.100948,-3.148414],[5.488532,-6.712980,-0.063499],[5.904668,8.033176,-1.565991]]], dtype = "float64")#candidate|1160|(7, 4, 3)|const|float64
bop_1161 = relay.mod(bop_1157.astype('float64'), relay.reshape(const_1160.astype('float64'), relay.shape_of(bop_1157))) # shape=(7, 4, 3)
bop_1169 = relay.add(bop_1157.astype('float64'), relay.reshape(const_1160.astype('float64'), relay.shape_of(bop_1157))) # shape=(7, 4, 3)
var_1182 = relay.var("var_1182", dtype = "bool", shape = (7, 4, 3))#candidate|1182|(7, 4, 3)|var|bool
bop_1183 = relay.right_shift(bop_1151.astype('uint16'), relay.reshape(var_1182.astype('uint16'), relay.shape_of(bop_1151))) # shape=(7, 4, 3)
bop_1186 = relay.bitwise_and(bop_1157.astype('uint16'), relay.reshape(bop_1183.astype('uint16'), relay.shape_of(bop_1157))) # shape=(7, 4, 3)
uop_1197 = relay.rsqrt(bop_1151.astype('float64')) # shape=(7, 4, 3)
func_5_call = mod.get_global_var('func_5')
func_7_call = mutated_mod.get_global_var('func_7')
var_1201 = relay.var("var_1201", dtype = "float64", shape = (112,))#candidate|1201|(112,)|var|float64
call_1200 = relay.TupleGetItem(func_5_call(relay.reshape(var_1201.astype('float64'), [8, 2, 7])), 0)
call_1202 = relay.TupleGetItem(func_7_call(relay.reshape(var_1201.astype('float64'), [8, 2, 7])), 0)
output = relay.Tuple([bop_1161,bop_1169,bop_1186,uop_1197,call_1200,var_1201,])
output2 = relay.Tuple([bop_1161,bop_1169,bop_1186,uop_1197,call_1202,var_1201,])
func_1203 = relay.Function([var_1149,var_1150,var_1182,var_1201,], output)
mod['func_1203'] = func_1203
mod = relay.transform.InferType()(mod)
mutated_mod['func_1203'] = func_1203
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1203_call = mutated_mod.get_global_var('func_1203')
var_1205 = relay.var("var_1205", dtype = "bool", shape = (7, 4, 3))#candidate|1205|(7, 4, 3)|var|bool
var_1206 = relay.var("var_1206", dtype = "bool", shape = (7, 4, 3))#candidate|1206|(7, 4, 3)|var|bool
var_1207 = relay.var("var_1207", dtype = "bool", shape = (7, 4, 3))#candidate|1207|(7, 4, 3)|var|bool
var_1208 = relay.var("var_1208", dtype = "float64", shape = (112,))#candidate|1208|(112,)|var|float64
call_1204 = func_1203_call(var_1205,var_1206,var_1207,var_1208,)
output = call_1204
func_1209 = relay.Function([var_1205,var_1206,var_1207,var_1208,], output)
mutated_mod['func_1209'] = func_1209
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1236 = relay.const([-5.355553,1.723849,-5.851424,-3.144394,2.631054,9.787965,4.349687,2.728528,-4.697411,-1.592823], dtype = "float64")#candidate|1236|(10,)|const|float64
uop_1237 = relay.exp(const_1236.astype('float64')) # shape=(10,)
bop_1239 = relay.floor_divide(uop_1237.astype('float64'), relay.reshape(const_1236.astype('float64'), relay.shape_of(uop_1237))) # shape=(10,)
bop_1242 = relay.minimum(const_1236.astype('float32'), relay.reshape(uop_1237.astype('float32'), relay.shape_of(const_1236))) # shape=(10,)
func_1029_call = mod.get_global_var('func_1029')
func_1033_call = mutated_mod.get_global_var('func_1033')
var_1246 = relay.var("var_1246", dtype = "float32", shape = (99,))#candidate|1246|(99,)|var|float32
const_1247 = relay.const([-4,-5,-1,-3,3,-2,-7,-4,-4,8,5,-4,2,5,9,-1,-1,1,1,8,1,-6,2,7,2,-7,-9,-8,-7,8,7,10,-6,-9,-4,10,6,-4,-5,6,6,-4,6,-8,3,10,3,-7,10,-3,-2,10,8,10,-6,-3,-2,9,-9,1,-1,3,3,-3,7,-7,-1,-8,-2,-4,-3,-3,4,-2,-1,6,-8,-7,10,-3,-8,-5,-9,7,9,2,-8,8,-6,-6,7,-8,4,3,-7,-10,-1,1,-8,-1,-10,1,-7,2,1,-2,-2,3,-8,-10,5,10,-7,-5,-1,-3,1,-1,-5,4,10,6,8,10,6,-3,-1,6,-5,-2,9,-9,9,7,-10,-2,8,-4,5,-4,-3,-3,-1,-2,-2,-5,10,-9,2,3,-8,-4,-2,5,-2,8,10,-2,10,9,10,1,8,9,10,3,8,6,-8,5,2,7,-5,-4,5,-9,7,-5,1,-6,5,-9,-9,-2,8,6,3,1,-6,-6,-8,6,-4,2,-4,-8,-10,-3,-2,5,-4,-9,6,-6,-4,-3,-5,10,1,-5,-2,5,10,2,-5,-3,8,4,3,-8,10,1,-10,5,5,-6,4,1,-1,7,10,3,-3,5,6,-3,3,-4,-3,6,-10,4,-6,-10,-8,1,9,9,10,8,-5,4,2,-3,-6,2,-3,-10,3,-6,-8,-10,-10,9,-1,5,5,-4,-7,2,7,-9,-2,1,-10,-6,-1,-5,5,10,8,-2,5,10,6,-5,-7,5,-3,3,-1,-2,-5,-3,9,-6,9,2,7,-5,-7,6,-5,3,5,-3,-7,1,2,-4,-7,9,-2,9,-6,-10,7,5,7,1,-6,-10,-1,-4,-5,3,-1,-8,-5,10,-3,2,-6,-3,-7,-7,-3,5,-8,-7,9,6,2,3,-3,7,-5,10,-8,-6,-9,7,10,6,-2,2,2,-3,4,-10,-1,6,5,-1,9,7,2,2,5,1,-10,1,3,-6,10,-3,-2,10,5,2,1,-5,-3,-5,4,10,-9,8,2,8,-10,4,-1,1,6,-3,-3,9,3,8,10,-5,-6,-2,7,-2,-4,-10,3,-5,7,7,6,7,10,9,1,2,7,6,7,-2,8,-5,5,1,8,1,-6,3,2,-7,-8,-10,4,-8,6,1,1,-7,8,8,6,1,-6,7,8,10,-4,8,8,10,-7,-10,-10,7,-4,5,-4,3,1,-7,5,-3,-10,8,1,4,10,-8,-10,8,10,6,-10,-6,-9,7,2,5,-3,10,3,-1,7,-5,6,10,7,-2,5,7,10,-7,4,-4,7,2,-9,4,2,-4,-3,4,10,-6,-3,8,6,6,7,8,6,-5,3,-5,-7,5,3,9,-7,-8,-7,-4,2,-3,-9,-5,1,4,-7,10,5,-5,7,2,8,-4,6,8,-7,-7,9,3,-1,-3,5,-4,1,9,-3,1,10,-6,5,-2,-2,1,10,3,-3,10,2,-6,-3,-10,-7,8,6,4,8,6,-7,-3,-2,-10,7,-3,-6,-3,-6,-7,-3,10,1,-3,-4,-10,5,-9,2,-4,-9,-9,5,-6,-4,-6,8,-6,-2,9,-7,6,-2,1,-4,4,-1,10,4,9,-9,-5,-2,7,2,-1,-2,-3,4,10,-2,-3,7,10,-9,1,3,4,-9,-9,-4,-9,9,-3,-9,-1,7,-3,7,-6,-2,6,-3,3,1,2,-4,4,-5,4,-9,-7,9,1,-3,4,-5,3,3,9,-7,1,-2,-3,-3,-2,-2,-1,9,8,-10,-9,5,3,5,-1,-3,-2,-7,-3,-1,-7,4,-3,9,-6,6,-5,7,-6,-4,-3,-9,-3,9,9,2,3,3,10,-9,-4,-4,-5,-1,-1,10,-1,10,-7,1,-1,1,9,-1,-7,-7,-7,-2,-2,-5,-5,9,-6,-1,8,-9,4,-2,-6,-2,-5,-6,10,9,8,-4,-1,-10,-4,7,4,2,6,1,-8,4,3,-8,-7,-4,-10,-9,-1,-7,-3,-6,10,5,1,7,10,-5,-3,-4,9,-5,-6,2,-5,-4,-5,5,10,1,-5,-1,2,-8,10,-5,-9,1,-5,5,-4,-8,9,2,3,-1,6,5,-9,-4,4,9,10,-9,3,-6,-10,3,10,-9,1,-5,-1,6,3,10,-9,-3,6,-5,5,-5,-10,6,-4,-4,4,-5,-2,-8,10,4,-4,-10,8,-9,-9,-10,3,-8,1,-5,1,-4,8,10,8,9,3,-4,5,4,10,-2,1,1,9,-9,-5,-7,-4,2,3,6,9,6,5,-7,-5,-7,6,-8,-5,9,-1,4,-5,9,-8,-3,7,2,-1,-7,9,-5,-4,-3,-2,-3,-1,10,5,-4,-6,-1,-8,5,3,8,-5,-8,7,-2,-4,-6,1,-10,7,9,-1,-8,-5,-8,4,2,-10,2,2,10,8,2,4,-8,-6,-6,-9,-7,5,5,-3,1,-7,-5,-7,-6,1,-6,10,7,1,6,-6,-4,-10,10,-6,-9,4,-4,8,-5,8,5,-3,7,-4,6,5,2,-9,-9,3,-7,-1,-7,-9,-10,-2,-6,-7,-3,-7,10,-6,-9,5,9,-7,9,7,7,-6,4,1,-7,4,-9,6,3,4,-9,10,-10,-6,5,3,-9,8,-1,5,7,-5,-10,-8,4,-5,-10,3,-3,7,2,1,-2,-7,10,-8,-1,5,-3,4,-2,10,-7,-2,-2,10,10,5,2,3,-4,-1,4,-8,-7,-6,-3,-1,-1,-3,-1,-9,4,-10,-8,2,-6,10,-9,8,-1,8,-5,6,-1,1,2,4,-5,-4,-9,-6,-9,6,-5,7,1,1,4,3,6,9,3,-4,8,-3,4,4,8,-5,-4,-6,-8,2,4,-9,-9,-8,2,3,2,8,7,-1,-10,5,-9,-1,-10,8,8,6,-6,2,-8,9,-2,6,-7,-2,3,2,-10,8,-6,8,6,3,-10,-9,9,8,4,-9,-3,1,2,-10,6,4,6,8,-6,3,6,-6,-1,6,-10,-8,8,9,-3,-4,-6,-5,-10,4,3,-6,9,-6,6,-6,7,-9,5,6,-6,-1,9,-7,-8,-9,-6,2,6,-6,9,9,7,-4,-6,4,-1,6,-8,2,-2,9,-7,3,-8,-8,-2,5,-4,6,1,6,3,-3,5,-9,8,8,8,9,-2,-4,-3,-6,10,-9,-1,3,8,-9,3,3,5,-9,3,-8,4,-1,-3,-5,8,3,10,-5,4,-3,-1,10,-6,-8,6,5,-2,-8,3,9,1,1,10,10,-2,-2,-9,-9,-1,5,-7,-9,7,3,3,1,3,-5,-10,8,-10,-6,-5,6,-4,-7,-3,7,4,-2,-1,10,7,-7,-2,10,7,3,-1,-2,-1,-5,5,-6,3,5,1,7,3,-4,9,7,9,10,8,-6,-10,3,4,-4,7,2,9,3,-1,6,-10,8,4,1,9,-1,10,-6,-3,-3,2,-6,3,2,6,-4,3,-10,-9,9,8,-8,-3,-3,5,-7,-1,7,-2,-8,8,-9,-5,10,-9,3,-8,-4,-5,7,4,8,-7,-7,-5,1,8,-6,8,8,-4,3,-2,-5,7,-9,-8,-4,2,-7,-8,-2,-8,-1,6,-4,-2,4,-5,1,5,-10,-6,9,8,9,-4,-4,9,1,-7,-6,1,3,1,7,-7,-8,-3,-3,4,-7,-2,3,8,-9,-8,-6,-6,-4,-4,-2,9,-3,8,-3,9,1,1,-7,9,-4,10,7,-8,5,-3,4,-7,-4,5,7,7,9,3,7,-1,-10,-2,9,7,3,-10,-10,3,2,-3,-4,-7,10,-9,4,9,-1,1,2,-3,-6,1,-10,3,-4,8,-7,-3,5,-4,-1,2,-6,3,9,-3,9,-4,-2,-10,1,6,-3,3,-10,8,1,-5,-1,5,6,-6,3,5,-6,2,5,5,7,-7,-6,10,6,-4,-8,8,6,1,5,7,-3,9,8,-7,4,2,10,7,9,-3,-8,-1,7,6,-9,3,3,-1,6,6,3,-9,-1,-1,8,9,6,-2,3,-7,2,-6,4,6,7,-9,9,5,4,2,1,10,1,-2,-10,7,-1,-7,-2,6,-4,-2,-6,-4,-3,4,6,-1,4,1,4,-5,3,-3,-4,4,5,10,-3,2,1,-1,-10,5,-8,7,6,-9,4,2,1,6,2,-7,-2,-2,-7,-3,-5,-10,-7,-5,10,2,3,2,1,-10,-7,9,-8,-4,2,-3,-8,-10,-1,-9,-6,-5,-8,5,-6,-4,3,-2,-6,7,6,5,-4,-10,-9,-8,-4,5,1,-1,4,1,1,-8,2,5,5,5,-6,-6,5,-9,6,-9,4,3,-7,5,10,5,9,3,9,-7,2,-9,5,6,-3,-8,6,7,8,1,7,1,5,2,-9,-8,-2,9,-6,1,-3,8,-2,5,9,9,2,-2,5,1,-4,-6,10,6,-2,7,7,5,2,-3,-7,-7,-4,6,-4,3,9,-6,7,-5,2,4,-7,1,10,2,-6,-6,9,-10,-2,3,-4,-10,-5,-2,5,6,-9,-3,1,-6,-4,9,-6,-8,-4,10,2,-3,-5,5,2,-3,-10,-10,2,10,-10,-1,8,7,-10,3,8,-10,10,-2,9,-3,-2,10,3,-1,-8,1,2,9,-8,-4,8,-1,1,1,9,-1,-9,8,-10,6,1,-6,-7,3,-7,4,-4,4,-8,-7,10,-1,-8,2,-8,-8,6,-10,-8,10,3,-1,-1,-2,9,-8,-7,8,8,5,-8,2,8,-9,-8,-2,-5,8,5,6,7,-2,3,-10,-2,4,10,4,-2,2,3,1,-6,-5,8,6,-5,2,8,8,8,10,10,4,8,-1,2,1,2,-3,-7,-8,7,10,-1,-5,-10,7,-10,3,-8,9,8,-10,-5,-3,-1,-5,-10,-4,-3,9,6,5,-6,-6,-3,-6,6,-5,10,3,8,10,5,10,4,3,-5,8,7,8,10,4,2,6,7,5,-9,5,10,-4,-5,4,5,1,-7,-3,3,-2,-5,8,10,-9,-8,-5,-8,-3,-5,-2,9,2,2,-3,-3,1,3,-1,-3,-6,-3,-9,9,5,-9,6,-10,-3,6,-9,-1,10,-2,-1,-1,7,-7,6,-10,-1,10,6,8,4,7,4,6,6,-4,4,-1,9,9,5,8,8,-7,-10,6,-8,7,-5,-5,4,7,5,1,7,1,1,1,-9,-4,3,3,6,-5,-5,-1,2,-7,7,10,9,-7,-3,10,7,3,5,-9,-5,2,-5,-9,5,3,-7,6,-6,-6,1,3,1,6,-9,7,-1,6,-5,4,-2,-2,2,4,-10,7,8,1,-5,5,4,-9,-2,3,-9,-5,-6,9,4,-5,8,8,-5,7,-6,5,-4,-6,-9,-2,-10,1,-6,3,8,4,4,-3,-3,-9,3,-10,10,3,-7,-10,-6,2,-5,-1,6,8,9,1,2,-10,5,3,-6,-5,-5,-1,-9,-2,1,-5,9,9,7,-2,7,5,2,7,8,4,-10,4,-9,2,-10,-8,-4,-8,4,-9,2,5,6,3,-6,7,1,5,-10,3,-6,8,6,1,8,5,6,3,3,-5,-2,-7,6,6,-9,-5,-7,-4,-10,7,7,-7,3,4,-7,-7,5,-3,-5,6,-5,-8,-5,9,7,5,-7,8,4,8,-3,10,10,-9,-2,7,-8,-4,-7,-2,7,10,-2,6,10,-9,2,1,5,-6,-6,6,10,-9,-3,3,10,5,-2,10,-8,7,8,-1,9,-2,10,1,8,-9,2,-10,-5,-4,8,9,3,-6,-3,6,2,-8,-5,-4,-8,-1,-5,3,-6,5,1,9,-7,6,2,3,-1,-10,-8,9,-7,6,-1,-10,-6,-2,10,-10,10,8,-8,-6,9,-5,-10,3,-5,1,-10,5,-7,-2,2,-3,-7,-4,1,-6,-6,7,3,7,-3,-4,-9,-10,9,5,-3,4,1,-7,-8,3,1,-1,-4,-4,7,9,-8,4,8,-9,-8,-2,-8,3,9,-1,-8,9,-3,8,-3,-5,5,-10,-9,-5,-1,7,-6,-7,-6,7,1,5,-1,3,9,4,-9,6,-1,-6,8,2,-2,8,7,4,-3,3,4,-7,8,-9,-6,-5,-8,9,-4,9,-4,-6,6,1,10,5,-2,1,4,4,-4,6,2,7,10,2,7,-3,9,-4,7,-7,2,6,-7,-5,-3,-3,6,-9,-10,-8,-8,-1,6,7,7,-2,-9,6,-3,7,9,9,3,-2,8,3,-3,-4,-2,-2,-5,9,-5,2,-3,3,2,-7,-9,-10,-9,4,7,-10,-1,7,9,3,-10,-3,5,-5,6,-4,1,4,6,1,-6,-2,3,4,-9,7,2,-2,-6,6,9,8,3,-9,10,2,-8,4,8,-7,-6,4,-6,5,-5,5,10,-3,7,-9,9,4,8,3,7,2,-4,-1,3,4,8,2,-5,10,7,1,-3,-7,3,6,10,5,-4,-6,-4,-7,-10,-10,-5,6,-10,-2,-2,-3,-9,-6,8,-6,-6,3,3,-2,-2,7,-5,-1,-2,7,2,-8,10,3,-5,-4,9,1,-9,4,-9,9,6,6,-3,-3,-4,7,4,6,8,10,-7,-1,8,-8,-8,9,-6,-3,10,5,6,-10,3,10,5,-9,3,-7,-5,7,1,-7,-5,-6,-6,-10,3,5,-5,-7,4,-1,4,-9,-4,-2,4,-1,-4,3,8,-4,-3,-6,3,7,9,-8,7,-4,4,7,9,-7,3,-3,-3,-1,-6,-3,10,8,-9,4,4,10,-6,-5,2,-6,-4,-7,5,-3,7,-10,7,6,-4,5,-8,3,-9,9,-4,8,1,5,3,7,-2,-7,-4,-7,5,4,7,9,10,9,10,-10,-3,9,10,8,-2,5,9,-4,-5,10,-8,10,-7,-9,-6,-10,-5,5,-6,1,9,5,7,-2,10,6,1,3,5,-4,3,10,-4,5,-9,1,-2,3,10,4,9,5,6,-1,3,1,-1,6,-5,9,-7,1,-9,-9,-2,-6,-2,5,2,-10,1,-3,-2,10,-9,-1,-9,6,-7,-6,7,-3,6,1,-4,-3,4,-9,-6,-6,-8,-3,-1,7,-5,8,7,-3,1,-1,6,-4,-1,-9,-2,1,-1,-8,-3,-10,-8,-10,-4,-7,8,-2,1,7,6,6,8,-9,10,-1,4,1,2,10,6,-10,3,-1,5,6,-1,-1,-10,7,-2,4,2,2,9,-9,-9,-9,1,7,-5,9,-7,-4,5,-8,-2,6,10,-7,10,-3,-6,-1,10,5,3,1,8,-9,-7,-1,3,-2,-5,-3,-4,4,-9,-9,5,4,9,9,3,10,7,8,1,5,8,3,-4,-3,3,2,5,-5,-3,2,4,-7,-5,5,4,2,-9,5,-1,6,8,2,-8,-6,5,9,9,-2,-1,-2,8,9,8,-1,-2,-5,-9,-4,5,-9,7,10,4,-2,-1,-1,8,-4,-6,9,4,-8,10,9,1,4,6,8,7,6,-6,-7,-1,-3,8,4,-6,2,-4,9,-8,5,-8,3,-5,-1,8,-5,9,-5,-2,8,3,1,-10,3,7,2,-3,9,6,8,10,-2,-5,-5,-7,-4,4,8,-8,-3,-10,-10,-2,2,-1,-1,-8,-9,3,8,6,10,7,-5,10,-2,3,-5,1,2,-2,10,-9,10,2,3,10,-9,3,4,1,4,-7,8,10,7,-6,-1,7,-8,9,-8,-5,6,2,1,-5,9,6,10,-9,10,-8,-2,-10,4,10,-8,1,-4,7,6,10,6,5,4,4,1,2,-7,-2,4,2,-10,8,-6,7,9,-9,5,-10,8,-10,-6,-10,7,-3,-5,-2,10,9,7,9,-3,-10,2,5,5,-9,-10,-5,3,9,-5,-5,3,10,-9,-7,-9,1,8,-5,5,-6,-5,3,9,7,3,8,7,-9,8,1,6,1,1,-8,10,-1,-5,4,-8,-7,-5,10,-4,10,-6,9,4,5,8,2,-7,4,1,5,8,-4,-8,9,-5,4,-2,3,-5,-3,-2,-2,10,-3,5,-2,-4,8,-1,4,7,-7,-10,-9,9,-9,-9,-3,8,-1,6,6,10,-1,-2,-6,2,-10,-4,-7,-10,-5,7,7,-4,8,2,9,9,-4,5,-2,5,-9,-2,3,3,1,6,7,6,9,1,-9,-5,3,-5,5,8,9,10,-2,-10,5,10,-3,9,-4,9,3,8,-4,-3,-3,4,-1,-1,-1,5,-6,-10,-9,2,-1,9,-5,-10,-10,-9,-3,4,-9,-6,-6,6,6,-3,-7,10,-4,-10,8,9,10,-8,2,7,-6,2,6,6,-6,-8,-1,-8,9,-7,5,-4,3,8,-3,9,-9,-5,-8,6,-9,-4,-5,7,-10,1,-10,10,7,1,4,3,-10,8,-9,-2,1,9,10,-1,2,-6,7,8,-6,7,-10,5,1,7,-8,-8,-7,-4,-7,-7,-9,3,-1,-1,1,-6,10,-10,5,-1,-8,-1,-8,-6,-1,5,-6,-5,9,-3,-6,1,2,-7,1,-1,6,8,8,10,4,1,-2,-8,-6,1,-2,8,-4,-3,10,3,-6,10,4,-5,-7,-10,-5,-2,-9,-3,-5,-7,3,-3,-1,-8,9,1,2,-2,2,-8,10,10,-10,2,-3,4,6,10,6,-4,-10,6,1,-10,10,2,5,9,6,-3,-10,-8,6,6,7,-5,-5,10,-1,-8,6,10,5,-1,3,-1,-10,-1,-6,8,-7,4,-6,-5,-5,-7,-5,6,-10,-1,10,-6,-1,-7,-4,-4,-8,2,-3,10,-9,10,-7,4,-1,10,6,-2,-5,-8,5,-9,-10,-8,7,10,-6,2,7,1,-10,9,-7,5,-1,-10,-2,-1,1,5,7,5,-8,-9,2,2,10,-8,-2,8,5,-2,-8,-1,-2,8,4,5,10,-6,5,-4,7,4,6,-7,-4,10,-1,3,-9,-5,2,-6,5,5,-8,-6,5,-8,-4,2,3,-1,4,8,-3,-10,-2,-9,8,6,5,5,-4,10,8,-7,5,5,-2,-7,-7,-9,-5,7,-4,-1,7,-10,-9,-8,-2,5,-2,1,5,-8,6,1,-8,5,1,1,-2,-8,1,9,2,-3,6,8,-3,-2,-6,8,-3,-5,9,-2,-7,5,8,-3,9,5,-1,5,-1,-3,-1,-10,6,-9,-1,9,5,8,-7,4,-9,1,3,-6,-4,3,1,-10,-9,1,3,-10,3,10,2,-6,-2,-3,-3,-1,-9,2,8,-7,4,8,10,-4,9,5,-9,-2,-9,7,-3,9,-3,-2,-7,-9,-8,8,9,1,7,9,6,-10,6,7,6,-3,-8,-3,5,6,10,-9,6,-7,5,-3,-7,6,-8,3,8,6,-1,1,-10,-7,-10,9,8,1,2,-4,6,-10,-2,2,-9,6,-7,5,-3,-8,-3,-1,1,-4,-7,-1,8,-9,4,-10,8,-4,2,-4,4,-10,-7,7,6,-2,-10,7,-6,-10,-6,5,4,-2,-1,-3,-6,-3,7,-4,1,-5], dtype = "uint8")#candidate|1247|(3600,)|const|uint8
call_1245 = relay.TupleGetItem(func_1029_call(relay.reshape(var_1246.astype('float32'), [9, 11]), relay.reshape(var_1246.astype('float32'), [9, 11]), relay.reshape(const_1247.astype('uint8'), [3600,]), ), 1)
call_1248 = relay.TupleGetItem(func_1033_call(relay.reshape(var_1246.astype('float32'), [9, 11]), relay.reshape(var_1246.astype('float32'), [9, 11]), relay.reshape(const_1247.astype('uint8'), [3600,]), ), 1)
bop_1251 = relay.bitwise_or(bop_1242.astype('int32'), relay.reshape(bop_1239.astype('int32'), relay.shape_of(bop_1242))) # shape=(10,)
output = relay.Tuple([call_1245,var_1246,const_1247,bop_1251,])
output2 = relay.Tuple([call_1248,var_1246,const_1247,bop_1251,])
func_1255 = relay.Function([var_1246,], output)
mod['func_1255'] = func_1255
mod = relay.transform.InferType()(mod)
mutated_mod['func_1255'] = func_1255
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1256 = relay.var("var_1256", dtype = "float32", shape = (99,))#candidate|1256|(99,)|var|float32
func_1255_call = mutated_mod.get_global_var('func_1255')
call_1257 = func_1255_call(var_1256)
output = call_1257
func_1258 = relay.Function([var_1256], output)
mutated_mod['func_1258'] = func_1258
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1315 = relay.var("var_1315", dtype = "float32", shape = (14, 15, 7))#candidate|1315|(14, 15, 7)|var|float32
uop_1316 = relay.sqrt(var_1315.astype('float32')) # shape=(14, 15, 7)
uop_1319 = relay.sigmoid(uop_1316.astype('float64')) # shape=(14, 15, 7)
func_931_call = mod.get_global_var('func_931')
func_934_call = mutated_mod.get_global_var('func_934')
const_1322 = relay.const([2.106244,0.204849,4.084301,3.135948,3.071502,-1.325898,8.683870,9.434851,5.640487,-7.651008,4.652917,1.596999,-6.997850,-1.023545,-2.351783,-5.553999,-2.496700,-1.891440,-2.050261,-5.780235,2.908786,-2.405869,-4.455398,6.831655,8.859531,6.694068,6.119128,5.382138,-1.988750,3.488202,-2.185935,-2.433076,-0.084696], dtype = "float32")#candidate|1322|(33,)|const|float32
call_1321 = func_931_call(relay.reshape(const_1322.astype('float32'), [11, 3]))
call_1323 = func_931_call(relay.reshape(const_1322.astype('float32'), [11, 3]))
func_1080_call = mod.get_global_var('func_1080')
func_1085_call = mutated_mod.get_global_var('func_1085')
var_1329 = relay.var("var_1329", dtype = "float32", shape = (70, 1))#candidate|1329|(70, 1)|var|float32
var_1330 = relay.var("var_1330", dtype = "uint8", shape = (1, 96))#candidate|1330|(1, 96)|var|uint8
const_1331 = relay.const([-1.937658,4.676694,2.351819,6.495035,-1.970077,3.363124,-0.618117,-8.991386,8.917707,-1.677073,-2.749688,5.589832,1.665142,-5.611006,-5.871254,2.879759,-5.643590,1.608274,-2.801716,-4.783312,-6.351106,9.988648,-0.515278,6.212335,-2.536572,1.955939,0.102021,0.319263,5.726588,5.098792,-6.608931,-4.716352,3.890880,4.385273,2.554267,3.498613,8.502822,-6.290692,-2.802278,0.604653,0.537037,2.942918,6.683989,9.657470,7.566831,-3.614694,3.418330,-6.324714,4.097822,-3.943178,1.604570,-1.634706,-5.861999,-6.811334,-6.455331,2.440208,0.280069,-5.872039,1.376674,5.369950,1.639781,9.690831,0.176695,6.811397,0.890247,7.393143,4.330016,2.680361,-3.057608,-3.831910,5.468264,-6.911959,9.756442,1.473216,0.607189,7.738726,-2.421134,4.010051,-2.282284,-2.299259,7.019334,-8.254711,2.014114,-9.139936,-4.319537,1.314383,2.132592,-7.701039,-7.981290,6.388493,-3.969270,-2.062066,5.434515,-6.440196,1.329990,3.448031,-9.725729,3.974739,1.231694], dtype = "float32")#candidate|1331|(99,)|const|float32
call_1328 = relay.TupleGetItem(func_1080_call(relay.reshape(var_1329.astype('float32'), [10, 7]), relay.reshape(var_1330.astype('uint8'), [96,]), relay.reshape(const_1331.astype('float32'), [99,]), ), 7)
call_1332 = relay.TupleGetItem(func_1085_call(relay.reshape(var_1329.astype('float32'), [10, 7]), relay.reshape(var_1330.astype('uint8'), [96,]), relay.reshape(const_1331.astype('float32'), [99,]), ), 7)
output = relay.Tuple([uop_1319,call_1321,const_1322,call_1328,var_1329,var_1330,const_1331,])
output2 = relay.Tuple([uop_1319,call_1323,const_1322,call_1332,var_1329,var_1330,const_1331,])
func_1333 = relay.Function([var_1315,var_1329,var_1330,], output)
mod['func_1333'] = func_1333
mod = relay.transform.InferType()(mod)
mutated_mod['func_1333'] = func_1333
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1333_call = mutated_mod.get_global_var('func_1333')
var_1335 = relay.var("var_1335", dtype = "float32", shape = (14, 15, 7))#candidate|1335|(14, 15, 7)|var|float32
var_1336 = relay.var("var_1336", dtype = "float32", shape = (70, 1))#candidate|1336|(70, 1)|var|float32
var_1337 = relay.var("var_1337", dtype = "uint8", shape = (1, 96))#candidate|1337|(1, 96)|var|uint8
call_1334 = func_1333_call(var_1335,var_1336,var_1337,)
output = call_1334
func_1338 = relay.Function([var_1335,var_1336,var_1337,], output)
mutated_mod['func_1338'] = func_1338
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1395 = relay.const([[[6,5,-6,-10,9,6,-3,-10,2,-8,5,2,-8],[-8,-2,6,-1,-1,1,-4,3,2,-8,10,6,-7],[-5,-2,9,-10,3,8,9,10,1,10,-3,-10,5],[-3,10,-4,10,8,6,-10,-3,2,10,-4,6,-6],[8,-10,-8,6,4,4,5,-3,7,-1,-4,6,5]],[[-4,6,9,-8,7,-3,-5,-2,-1,1,10,-5,-2],[-4,-2,8,1,8,6,-1,3,5,-1,6,-3,-3],[-7,-4,9,7,6,-9,-7,-8,-2,-7,3,2,-10],[5,9,6,10,5,7,-3,5,-1,1,-7,-8,-10],[8,-4,-2,2,1,-7,10,-2,-2,-2,-9,1,-10]],[[-2,5,-9,4,1,-6,-9,-3,-8,8,9,-5,9],[2,3,-8,-2,5,-8,-7,-8,4,-8,4,9,5],[4,-9,-5,10,-4,-1,3,-8,-8,10,5,-1,-10],[-10,9,-6,-9,-3,-7,10,-3,-6,8,5,-8,-2],[-8,6,10,-8,-5,-1,1,-7,-8,10,8,8,-1]],[[-4,10,1,-8,6,5,6,-10,4,-10,-10,8,10],[5,3,-2,4,-2,-10,2,-8,-1,6,-1,3,-1],[-2,2,-4,1,-2,6,-5,-8,9,-10,-6,-3,-10],[-5,8,5,-5,-1,1,-8,-4,-8,-5,3,-6,-2],[4,7,10,9,9,7,1,5,-1,-7,-9,4,-9]],[[5,10,10,-5,-7,10,-4,-10,-3,10,-2,-10,5],[-2,-1,-8,10,8,-1,6,7,-6,5,3,-6,5],[-5,7,-4,1,-3,6,8,-3,4,-8,5,1,1],[-7,6,-7,9,-2,-6,-10,1,-2,7,-9,5,1],[3,-10,-7,-3,3,3,5,-6,-8,10,2,10,7]]], dtype = "uint16")#candidate|1395|(5, 5, 13)|const|uint16
var_1396 = relay.var("var_1396", dtype = "uint16", shape = (5, 5, 13))#candidate|1396|(5, 5, 13)|var|uint16
bop_1397 = relay.equal(const_1395.astype('bool'), relay.reshape(var_1396.astype('bool'), relay.shape_of(const_1395))) # shape=(5, 5, 13)
uop_1400 = relay.log10(bop_1397.astype('float64')) # shape=(5, 5, 13)
func_1029_call = mod.get_global_var('func_1029')
func_1033_call = mutated_mod.get_global_var('func_1033')
var_1403 = relay.var("var_1403", dtype = "float32", shape = (99,))#candidate|1403|(99,)|var|float32
var_1404 = relay.var("var_1404", dtype = "uint8", shape = (3600,))#candidate|1404|(3600,)|var|uint8
call_1402 = relay.TupleGetItem(func_1029_call(relay.reshape(var_1403.astype('float32'), [9, 11]), relay.reshape(var_1403.astype('float32'), [9, 11]), relay.reshape(var_1404.astype('uint8'), [3600,]), ), 3)
call_1405 = relay.TupleGetItem(func_1033_call(relay.reshape(var_1403.astype('float32'), [9, 11]), relay.reshape(var_1403.astype('float32'), [9, 11]), relay.reshape(var_1404.astype('uint8'), [3600,]), ), 3)
output = relay.Tuple([uop_1400,call_1402,var_1403,var_1404,])
output2 = relay.Tuple([uop_1400,call_1405,var_1403,var_1404,])
func_1411 = relay.Function([var_1396,var_1403,var_1404,], output)
mod['func_1411'] = func_1411
mod = relay.transform.InferType()(mod)
mutated_mod['func_1411'] = func_1411
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1411_call = mutated_mod.get_global_var('func_1411')
var_1413 = relay.var("var_1413", dtype = "uint16", shape = (5, 5, 13))#candidate|1413|(5, 5, 13)|var|uint16
var_1414 = relay.var("var_1414", dtype = "float32", shape = (99,))#candidate|1414|(99,)|var|float32
var_1415 = relay.var("var_1415", dtype = "uint8", shape = (3600,))#candidate|1415|(3600,)|var|uint8
call_1412 = func_1411_call(var_1413,var_1414,var_1415,)
output = call_1412
func_1416 = relay.Function([var_1413,var_1414,var_1415,], output)
mutated_mod['func_1416'] = func_1416
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1440 = relay.var("var_1440", dtype = "bool", shape = (13, 2, 3))#candidate|1440|(13, 2, 3)|var|bool
var_1441 = relay.var("var_1441", dtype = "bool", shape = (13, 2, 3))#candidate|1441|(13, 2, 3)|var|bool
bop_1442 = relay.logical_or(var_1440.astype('bool'), relay.reshape(var_1441.astype('bool'), relay.shape_of(var_1440))) # shape=(13, 2, 3)
output = bop_1442
output2 = bop_1442
func_1445 = relay.Function([var_1440,var_1441,], output)
mod['func_1445'] = func_1445
mod = relay.transform.InferType()(mod)
var_1446 = relay.var("var_1446", dtype = "bool", shape = (13, 2, 3))#candidate|1446|(13, 2, 3)|var|bool
var_1447 = relay.var("var_1447", dtype = "bool", shape = (13, 2, 3))#candidate|1447|(13, 2, 3)|var|bool
output = func_1445(var_1446,var_1447,)
func_1448 = relay.Function([var_1446,var_1447,], output)
mutated_mod['func_1448'] = func_1448
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1484 = relay.var("var_1484", dtype = "uint8", shape = (5, 11))#candidate|1484|(5, 11)|var|uint8
const_1485 = relay.const([[6,10,-4,-3,3,7,7,-2,-6,9,-3],[-1,-7,9,2,-5,-5,-6,7,4,-1,-1],[10,1,8,-5,-2,-7,5,8,-7,-6,8],[-1,-1,-3,4,3,1,-2,-4,9,-3,-5],[-6,6,8,6,-3,-10,-10,9,1,9,-5]], dtype = "uint8")#candidate|1485|(5, 11)|const|uint8
bop_1486 = relay.bitwise_xor(var_1484.astype('uint8'), relay.reshape(const_1485.astype('uint8'), relay.shape_of(var_1484))) # shape=(5, 11)
output = relay.Tuple([bop_1486,])
output2 = relay.Tuple([bop_1486,])
func_1491 = relay.Function([var_1484,], output)
mod['func_1491'] = func_1491
mod = relay.transform.InferType()(mod)
mutated_mod['func_1491'] = func_1491
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1492 = relay.var("var_1492", dtype = "uint8", shape = (5, 11))#candidate|1492|(5, 11)|var|uint8
func_1491_call = mutated_mod.get_global_var('func_1491')
call_1493 = func_1491_call(var_1492)
output = call_1493
func_1494 = relay.Function([var_1492], output)
mutated_mod['func_1494'] = func_1494
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1504 = relay.const([[-0.861221],[5.094203],[-6.847413]], dtype = "float32")#candidate|1504|(3, 1)|const|float32
uop_1505 = relay.asinh(const_1504.astype('float32')) # shape=(3, 1)
uop_1507 = relay.log(uop_1505.astype('float32')) # shape=(3, 1)
bop_1510 = relay.subtract(uop_1507.astype('int8'), relay.reshape(uop_1505.astype('int8'), relay.shape_of(uop_1507))) # shape=(3, 1)
bop_1513 = relay.maximum(uop_1507.astype('uint32'), relay.reshape(uop_1505.astype('uint32'), relay.shape_of(uop_1507))) # shape=(3, 1)
uop_1516 = relay.atan(bop_1513.astype('float64')) # shape=(3, 1)
output = relay.Tuple([bop_1510,uop_1516,])
output2 = relay.Tuple([bop_1510,uop_1516,])
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
	relay.transform.PartialEvaluate(),
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