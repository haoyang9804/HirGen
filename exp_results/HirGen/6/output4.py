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
var_0 = relay.var("var_0", dtype = "float32", shape = (12,))#candidate|0|(12,)|var|float32
uop_1 = relay.asinh(var_0.astype('float32')) # shape=(12,)
uop_5 = relay.acos(uop_1.astype('float32')) # shape=(12,)
const_9 = relay.const([-9.431325,0.184045,2.004480,6.986755,8.201221,9.927439,-4.848378,9.921228,3.868355,4.972297,-5.680844,-0.341277], dtype = "float32")#candidate|9|(12,)|const|float32
bop_10 = relay.logical_and(uop_5.astype('bool'), relay.reshape(const_9.astype('bool'), relay.shape_of(uop_5))) # shape=(12,)
bop_13 = relay.less(uop_5.astype('bool'), relay.reshape(const_9.astype('bool'), relay.shape_of(uop_5))) # shape=(12,)
bop_18 = relay.less_equal(var_0.astype('bool'), relay.reshape(uop_5.astype('bool'), relay.shape_of(var_0))) # shape=(12,)
bop_26 = relay.mod(var_0.astype('float64'), relay.reshape(bop_13.astype('float64'), relay.shape_of(var_0))) # shape=(12,)
var_29 = relay.var("var_29", dtype = "bool", shape = (12,))#candidate|29|(12,)|var|bool
bop_30 = relay.floor_mod(bop_10.astype('float64'), relay.reshape(var_29.astype('float64'), relay.shape_of(bop_10))) # shape=(12,)
uop_34 = relay.sinh(bop_10.astype('float32')) # shape=(12,)
bop_36 = relay.subtract(uop_1.astype('float32'), relay.reshape(uop_5.astype('float32'), relay.shape_of(uop_1))) # shape=(12,)
var_43 = relay.var("var_43", dtype = "bool", shape = (12,))#candidate|43|(12,)|var|bool
bop_44 = relay.logical_or(bop_13.astype('bool'), relay.reshape(var_43.astype('bool'), relay.shape_of(bop_13))) # shape=(12,)
bop_52 = relay.greater_equal(uop_1.astype('bool'), relay.reshape(bop_36.astype('bool'), relay.shape_of(uop_1))) # shape=(12,)
const_56 = relay.const([False,True,False,False,True,True,False,False,False,False,True,False], dtype = "bool")#candidate|56|(12,)|const|bool
bop_57 = relay.floor_divide(bop_18.astype('float32'), relay.reshape(const_56.astype('float32'), relay.shape_of(bop_18))) # shape=(12,)
var_62 = relay.var("var_62", dtype = "float32", shape = (12,))#candidate|62|(12,)|var|float32
bop_63 = relay.logical_xor(uop_1.astype('int16'), relay.reshape(var_62.astype('int16'), relay.shape_of(uop_1))) # shape=(12,)
bop_72 = relay.left_shift(bop_63.astype('uint16'), relay.reshape(uop_5.astype('uint16'), relay.shape_of(bop_63))) # shape=(12,)
output = relay.Tuple([bop_26,bop_30,uop_34,bop_44,bop_52,bop_57,bop_72,])
output2 = relay.Tuple([bop_26,bop_30,uop_34,bop_44,bop_52,bop_57,bop_72,])
func_78 = relay.Function([var_0,var_29,var_43,var_62,], output)
mod['func_78'] = func_78
mod = relay.transform.InferType()(mod)
mutated_mod['func_78'] = func_78
mutated_mod = relay.transform.InferType()(mutated_mod)
func_78_call = mutated_mod.get_global_var('func_78')
var_80 = relay.var("var_80", dtype = "float32", shape = (12,))#candidate|80|(12,)|var|float32
var_81 = relay.var("var_81", dtype = "bool", shape = (12,))#candidate|81|(12,)|var|bool
var_82 = relay.var("var_82", dtype = "bool", shape = (12,))#candidate|82|(12,)|var|bool
var_83 = relay.var("var_83", dtype = "float32", shape = (12,))#candidate|83|(12,)|var|float32
call_79 = func_78_call(var_80,var_81,var_82,var_83,)
output = call_79
func_84 = relay.Function([var_80,var_81,var_82,var_83,], output)
mutated_mod['func_84'] = func_84
mutated_mod = relay.transform.InferType()(mutated_mod)
const_99 = relay.const(6, dtype = "uint64")#candidate|99|()|const|uint64
var_100 = relay.var("var_100", dtype = "uint64", shape = (2, 12, 10))#candidate|100|(2, 12, 10)|var|uint64
bop_101 = relay.equal(const_99.astype('bool'), var_100.astype('bool')) # shape=(2, 12, 10)
func_78_call = mod.get_global_var('func_78')
func_84_call = mutated_mod.get_global_var('func_84')
var_106 = relay.var("var_106", dtype = "float32", shape = (12,))#candidate|106|(12,)|var|float32
call_105 = relay.TupleGetItem(func_78_call(relay.reshape(var_106.astype('float32'), [12,]), relay.reshape(var_106.astype('bool'), [12,]), relay.reshape(var_106.astype('bool'), [12,]), relay.reshape(var_106.astype('float32'), [12,]), ), 2)
call_107 = relay.TupleGetItem(func_84_call(relay.reshape(var_106.astype('float32'), [12,]), relay.reshape(var_106.astype('bool'), [12,]), relay.reshape(var_106.astype('bool'), [12,]), relay.reshape(var_106.astype('float32'), [12,]), ), 2)
func_78_call = mod.get_global_var('func_78')
func_84_call = mutated_mod.get_global_var('func_84')
call_108 = relay.TupleGetItem(func_78_call(relay.reshape(call_105.astype('float32'), [12,]), relay.reshape(call_105.astype('bool'), [12,]), relay.reshape(call_105.astype('bool'), [12,]), relay.reshape(var_106.astype('float32'), [12,]), ), 2)
call_109 = relay.TupleGetItem(func_84_call(relay.reshape(call_105.astype('float32'), [12,]), relay.reshape(call_105.astype('bool'), [12,]), relay.reshape(call_105.astype('bool'), [12,]), relay.reshape(var_106.astype('float32'), [12,]), ), 2)
var_121 = relay.var("var_121", dtype = "bool", shape = (2, 12, 10))#candidate|121|(2, 12, 10)|var|bool
bop_122 = relay.left_shift(bop_101.astype('int16'), relay.reshape(var_121.astype('int16'), relay.shape_of(bop_101))) # shape=(2, 12, 10)
output = relay.Tuple([call_105,var_106,call_108,bop_122,])
output2 = relay.Tuple([call_107,var_106,call_109,bop_122,])
func_131 = relay.Function([var_100,var_106,var_121,], output)
mod['func_131'] = func_131
mod = relay.transform.InferType()(mod)
mutated_mod['func_131'] = func_131
mutated_mod = relay.transform.InferType()(mutated_mod)
func_131_call = mutated_mod.get_global_var('func_131')
var_133 = relay.var("var_133", dtype = "uint64", shape = (2, 12, 10))#candidate|133|(2, 12, 10)|var|uint64
var_134 = relay.var("var_134", dtype = "float32", shape = (12,))#candidate|134|(12,)|var|float32
var_135 = relay.var("var_135", dtype = "bool", shape = (2, 12, 10))#candidate|135|(2, 12, 10)|var|bool
call_132 = func_131_call(var_133,var_134,var_135,)
output = call_132
func_136 = relay.Function([var_133,var_134,var_135,], output)
mutated_mod['func_136'] = func_136
mutated_mod = relay.transform.InferType()(mutated_mod)
const_141 = relay.const([[8.443113,-5.179691,1.449563,-2.576380,1.081021,9.503134,4.973794,-7.759004],[-4.429041,-1.341067,3.044930,5.851585,-2.407767,-7.507170,-6.755800,-9.387958],[-0.821562,4.199966,3.094789,-1.418776,-6.412651,6.253139,-4.653644,-2.503801],[3.228827,5.363529,4.450834,-6.685613,9.798370,-3.274810,-5.686694,0.159974],[5.652350,-5.790628,3.536544,6.256972,9.525415,6.240155,7.348846,-0.917820],[-9.352206,5.198654,-5.120562,2.281493,-9.123987,3.376833,-6.392170,-8.476783],[4.554572,1.371027,5.398271,-6.091535,6.807524,6.117407,5.430092,-7.058336],[8.894186,4.861637,7.993479,1.414021,7.441759,-0.080356,3.666126,9.907521],[1.302154,5.014946,-0.435622,5.360379,5.865331,-1.925575,-4.240844,-7.307026],[-7.189999,2.757249,-5.478968,-1.541013,-5.244388,-9.626913,5.015383,-6.270319],[-1.946894,2.603718,-4.733847,-8.469136,0.379562,-7.471789,3.094986,4.350693],[5.240533,1.925529,5.597809,6.940632,-8.625120,-3.880014,7.907293,3.796565],[-6.700873,-7.470982,7.484150,4.510778,6.263483,9.177593,-3.595941,5.736241]], dtype = "float32")#candidate|141|(13, 8)|const|float32
var_142 = relay.var("var_142", dtype = "float32", shape = (13, 8))#candidate|142|(13, 8)|var|float32
bop_143 = relay.multiply(const_141.astype('float32'), relay.reshape(var_142.astype('float32'), relay.shape_of(const_141))) # shape=(13, 8)
bop_146 = relay.power(var_142.astype('float32'), relay.reshape(bop_143.astype('float32'), relay.shape_of(var_142))) # shape=(13, 8)
var_149 = relay.var("var_149", dtype = "float32", shape = (13, 8))#candidate|149|(13, 8)|var|float32
bop_150 = relay.minimum(bop_146.astype('int8'), relay.reshape(var_149.astype('int8'), relay.shape_of(bop_146))) # shape=(13, 8)
output = bop_150
output2 = bop_150
func_155 = relay.Function([var_142,var_149,], output)
mod['func_155'] = func_155
mod = relay.transform.InferType()(mod)
mutated_mod['func_155'] = func_155
mutated_mod = relay.transform.InferType()(mutated_mod)
func_155_call = mutated_mod.get_global_var('func_155')
var_157 = relay.var("var_157", dtype = "float32", shape = (13, 8))#candidate|157|(13, 8)|var|float32
var_158 = relay.var("var_158", dtype = "float32", shape = (13, 8))#candidate|158|(13, 8)|var|float32
call_156 = func_155_call(var_157,var_158,)
output = call_156
func_159 = relay.Function([var_157,var_158,], output)
mutated_mod['func_159'] = func_159
mutated_mod = relay.transform.InferType()(mutated_mod)
var_185 = relay.var("var_185", dtype = "float64", shape = (16, 4, 5))#candidate|185|(16, 4, 5)|var|float64
uop_186 = relay.sinh(var_185.astype('float64')) # shape=(16, 4, 5)
bop_189 = relay.left_shift(uop_186.astype('int64'), relay.reshape(var_185.astype('int64'), relay.shape_of(uop_186))) # shape=(16, 4, 5)
bop_195 = relay.equal(bop_189.astype('bool'), relay.reshape(uop_186.astype('bool'), relay.shape_of(bop_189))) # shape=(16, 4, 5)
func_78_call = mod.get_global_var('func_78')
func_84_call = mutated_mod.get_global_var('func_84')
const_200 = relay.const([-0.862428,-1.390308,-2.312575,-7.054581,8.838186,7.978259,2.953123,-2.025901,4.751766,0.677184,-9.037221,5.881852], dtype = "float32")#candidate|200|(12,)|const|float32
call_199 = relay.TupleGetItem(func_78_call(relay.reshape(const_200.astype('float32'), [12,]), relay.reshape(const_200.astype('bool'), [12,]), relay.reshape(const_200.astype('bool'), [12,]), relay.reshape(const_200.astype('float32'), [12,]), ), 3)
call_201 = relay.TupleGetItem(func_84_call(relay.reshape(const_200.astype('float32'), [12,]), relay.reshape(const_200.astype('bool'), [12,]), relay.reshape(const_200.astype('bool'), [12,]), relay.reshape(const_200.astype('float32'), [12,]), ), 3)
bop_203 = relay.right_shift(uop_186.astype('int32'), relay.reshape(bop_195.astype('int32'), relay.shape_of(uop_186))) # shape=(16, 4, 5)
bop_211 = relay.minimum(const_200.astype('uint64'), relay.reshape(call_199.astype('uint64'), relay.shape_of(const_200))) # shape=(12,)
bop_214 = relay.minimum(const_200.astype('uint64'), relay.reshape(call_201.astype('uint64'), relay.shape_of(const_200))) # shape=(12,)
uop_215 = relay.acosh(bop_203.astype('float64')) # shape=(16, 4, 5)
uop_219 = relay.cosh(uop_215.astype('float64')) # shape=(16, 4, 5)
var_221 = relay.var("var_221", dtype = "float64", shape = (16, 4, 5))#candidate|221|(16, 4, 5)|var|float64
bop_222 = relay.mod(uop_219.astype('float64'), relay.reshape(var_221.astype('float64'), relay.shape_of(uop_219))) # shape=(16, 4, 5)
bop_225 = relay.greater(uop_215.astype('bool'), relay.reshape(var_185.astype('bool'), relay.shape_of(uop_215))) # shape=(16, 4, 5)
var_231 = relay.var("var_231", dtype = "float64", shape = (16, 4, 5))#candidate|231|(16, 4, 5)|var|float64
bop_232 = relay.greater_equal(bop_222.astype('bool'), relay.reshape(var_231.astype('bool'), relay.shape_of(bop_222))) # shape=(16, 4, 5)
func_155_call = mod.get_global_var('func_155')
func_159_call = mutated_mod.get_global_var('func_159')
var_239 = relay.var("var_239", dtype = "float32", shape = (104,))#candidate|239|(104,)|var|float32
call_238 = func_155_call(relay.reshape(var_239.astype('float32'), [13, 8]), relay.reshape(var_239.astype('float32'), [13, 8]), )
call_240 = func_155_call(relay.reshape(var_239.astype('float32'), [13, 8]), relay.reshape(var_239.astype('float32'), [13, 8]), )
const_241 = relay.const([[[True,True,False,True,True],[False,True,False,False,True],[False,True,True,True,False],[False,True,True,True,False]],[[True,False,True,True,True],[False,True,False,False,True],[True,False,False,True,True],[True,True,False,True,True]],[[False,True,False,True,False],[False,False,True,True,True],[False,False,False,True,True],[True,False,True,False,False]],[[False,True,False,False,False],[True,False,True,False,True],[False,False,True,False,True],[True,False,True,True,True]],[[True,True,False,True,False],[True,False,False,False,False],[True,False,True,True,True],[True,True,True,False,True]],[[False,False,True,True,False],[False,True,False,False,False],[False,True,True,False,False],[True,True,False,False,False]],[[True,True,False,False,False],[True,False,True,False,False],[False,True,True,True,False],[True,False,True,False,False]],[[True,False,True,False,False],[True,False,True,True,False],[True,False,True,False,True],[True,True,True,True,False]],[[True,True,True,False,True],[True,False,True,True,False],[True,False,False,False,True],[False,True,True,True,False]],[[True,True,True,False,True],[False,True,False,True,False],[False,False,False,True,True],[True,True,True,False,False]],[[True,True,False,True,True],[True,True,False,False,False],[False,True,True,True,True],[False,True,True,True,False]],[[True,True,True,True,True],[False,False,False,True,False],[False,False,True,False,True],[False,False,False,False,False]],[[False,True,False,False,False],[True,False,False,False,True],[False,False,True,True,True],[False,True,False,False,False]],[[False,False,False,False,False],[True,False,False,True,True],[True,False,False,True,False],[False,False,False,False,True]],[[False,True,True,True,False],[False,True,False,False,True],[False,True,True,True,True],[True,False,True,False,False]],[[False,True,False,False,False],[False,True,False,False,True],[True,False,False,False,True],[True,True,False,True,True]]], dtype = "bool")#candidate|241|(16, 4, 5)|const|bool
bop_242 = relay.left_shift(bop_232.astype('int8'), relay.reshape(const_241.astype('int8'), relay.shape_of(bop_232))) # shape=(16, 4, 5)
func_155_call = mod.get_global_var('func_155')
func_159_call = mutated_mod.get_global_var('func_159')
call_246 = func_155_call(relay.reshape(call_238.astype('float32'), [13, 8]), relay.reshape(call_238.astype('float32'), [13, 8]), )
call_247 = func_155_call(relay.reshape(call_238.astype('float32'), [13, 8]), relay.reshape(call_238.astype('float32'), [13, 8]), )
bop_249 = relay.bitwise_and(uop_219.astype('int64'), relay.reshape(var_221.astype('int64'), relay.shape_of(uop_219))) # shape=(16, 4, 5)
bop_256 = relay.subtract(uop_219.astype('uint32'), relay.reshape(uop_186.astype('uint32'), relay.shape_of(uop_219))) # shape=(16, 4, 5)
uop_272 = relay.erf(uop_215.astype('float64')) # shape=(16, 4, 5)
uop_274 = relay.log10(bop_249.astype('float32')) # shape=(16, 4, 5)
var_277 = relay.var("var_277", dtype = "int8", shape = (16, 4, 5))#candidate|277|(16, 4, 5)|var|int8
bop_278 = relay.less_equal(bop_242.astype('bool'), relay.reshape(var_277.astype('bool'), relay.shape_of(bop_242))) # shape=(16, 4, 5)
uop_281 = relay.asin(uop_219.astype('float64')) # shape=(16, 4, 5)
uop_286 = relay.log(bop_242.astype('float32')) # shape=(16, 4, 5)
func_155_call = mod.get_global_var('func_155')
func_159_call = mutated_mod.get_global_var('func_159')
call_293 = func_155_call(relay.reshape(call_238.astype('float32'), [13, 8]), relay.reshape(call_238.astype('float32'), [13, 8]), )
call_294 = func_155_call(relay.reshape(call_238.astype('float32'), [13, 8]), relay.reshape(call_238.astype('float32'), [13, 8]), )
uop_297 = relay.log2(uop_215.astype('float64')) # shape=(16, 4, 5)
bop_301 = relay.bitwise_or(uop_281.astype('uint16'), relay.reshape(bop_256.astype('uint16'), relay.shape_of(uop_281))) # shape=(16, 4, 5)
var_306 = relay.var("var_306", dtype = "float64", shape = (16, 4, 5))#candidate|306|(16, 4, 5)|var|float64
bop_307 = relay.power(uop_215.astype('float32'), relay.reshape(var_306.astype('float32'), relay.shape_of(uop_215))) # shape=(16, 4, 5)
bop_313 = relay.maximum(uop_281.astype('int32'), relay.reshape(bop_195.astype('int32'), relay.shape_of(uop_281))) # shape=(16, 4, 5)
output = relay.Tuple([bop_211,bop_225,call_238,var_239,call_246,uop_272,uop_274,bop_278,uop_286,call_293,uop_297,bop_301,bop_307,bop_313,])
output2 = relay.Tuple([bop_214,bop_225,call_240,var_239,call_247,uop_272,uop_274,bop_278,uop_286,call_294,uop_297,bop_301,bop_307,bop_313,])
func_317 = relay.Function([var_185,var_221,var_231,var_239,var_277,var_306,], output)
mod['func_317'] = func_317
mod = relay.transform.InferType()(mod)
var_318 = relay.var("var_318", dtype = "float64", shape = (16, 4, 5))#candidate|318|(16, 4, 5)|var|float64
var_319 = relay.var("var_319", dtype = "float64", shape = (16, 4, 5))#candidate|319|(16, 4, 5)|var|float64
var_320 = relay.var("var_320", dtype = "float64", shape = (16, 4, 5))#candidate|320|(16, 4, 5)|var|float64
var_321 = relay.var("var_321", dtype = "float32", shape = (104,))#candidate|321|(104,)|var|float32
var_322 = relay.var("var_322", dtype = "int8", shape = (16, 4, 5))#candidate|322|(16, 4, 5)|var|int8
var_323 = relay.var("var_323", dtype = "float64", shape = (16, 4, 5))#candidate|323|(16, 4, 5)|var|float64
output = func_317(var_318,var_319,var_320,var_321,var_322,var_323,)
func_324 = relay.Function([var_318,var_319,var_320,var_321,var_322,var_323,], output)
mutated_mod['func_324'] = func_324
mutated_mod = relay.transform.InferType()(mutated_mod)
var_326 = relay.var("var_326", dtype = "int32", shape = (8,))#candidate|326|(8,)|var|int32
var_327 = relay.var("var_327", dtype = "int32", shape = (8,))#candidate|327|(8,)|var|int32
bop_328 = relay.multiply(var_326.astype('int32'), relay.reshape(var_327.astype('int32'), relay.shape_of(var_326))) # shape=(8,)
func_131_call = mod.get_global_var('func_131')
func_136_call = mutated_mod.get_global_var('func_136')
var_336 = relay.var("var_336", dtype = "uint64", shape = (240,))#candidate|336|(240,)|var|uint64
var_337 = relay.var("var_337", dtype = "float32", shape = (12,))#candidate|337|(12,)|var|float32
call_335 = relay.TupleGetItem(func_131_call(relay.reshape(var_336.astype('uint64'), [2, 12, 10]), relay.reshape(var_337.astype('float32'), [12,]), relay.reshape(var_336.astype('bool'), [2, 12, 10]), ), 1)
call_338 = relay.TupleGetItem(func_136_call(relay.reshape(var_336.astype('uint64'), [2, 12, 10]), relay.reshape(var_337.astype('float32'), [12,]), relay.reshape(var_336.astype('bool'), [2, 12, 10]), ), 1)
output = relay.Tuple([bop_328,call_335,var_336,var_337,])
output2 = relay.Tuple([bop_328,call_338,var_336,var_337,])
func_347 = relay.Function([var_326,var_327,var_336,var_337,], output)
mod['func_347'] = func_347
mod = relay.transform.InferType()(mod)
mutated_mod['func_347'] = func_347
mutated_mod = relay.transform.InferType()(mutated_mod)
func_347_call = mutated_mod.get_global_var('func_347')
var_349 = relay.var("var_349", dtype = "int32", shape = (8,))#candidate|349|(8,)|var|int32
var_350 = relay.var("var_350", dtype = "int32", shape = (8,))#candidate|350|(8,)|var|int32
var_351 = relay.var("var_351", dtype = "uint64", shape = (240,))#candidate|351|(240,)|var|uint64
var_352 = relay.var("var_352", dtype = "float32", shape = (12,))#candidate|352|(12,)|var|float32
call_348 = func_347_call(var_349,var_350,var_351,var_352,)
output = call_348
func_353 = relay.Function([var_349,var_350,var_351,var_352,], output)
mutated_mod['func_353'] = func_353
mutated_mod = relay.transform.InferType()(mutated_mod)
var_390 = relay.var("var_390", dtype = "float32", shape = (4, 9, 13))#candidate|390|(4, 9, 13)|var|float32
uop_391 = relay.acos(var_390.astype('float32')) # shape=(4, 9, 13)
uop_394 = relay.sqrt(uop_391.astype('float32')) # shape=(4, 9, 13)
uop_396 = relay.tan(uop_391.astype('float64')) # shape=(4, 9, 13)
bop_398 = relay.minimum(uop_391.astype('int16'), relay.reshape(uop_396.astype('int16'), relay.shape_of(uop_391))) # shape=(4, 9, 13)
func_78_call = mod.get_global_var('func_78')
func_84_call = mutated_mod.get_global_var('func_84')
var_402 = relay.var("var_402", dtype = "float32", shape = (12,))#candidate|402|(12,)|var|float32
call_401 = relay.TupleGetItem(func_78_call(relay.reshape(var_402.astype('float32'), [12,]), relay.reshape(var_402.astype('bool'), [12,]), relay.reshape(var_402.astype('bool'), [12,]), relay.reshape(var_402.astype('float32'), [12,]), ), 1)
call_403 = relay.TupleGetItem(func_84_call(relay.reshape(var_402.astype('float32'), [12,]), relay.reshape(var_402.astype('bool'), [12,]), relay.reshape(var_402.astype('bool'), [12,]), relay.reshape(var_402.astype('float32'), [12,]), ), 1)
bop_406 = relay.logical_and(uop_391.astype('bool'), relay.reshape(uop_396.astype('bool'), relay.shape_of(uop_391))) # shape=(4, 9, 13)
func_155_call = mod.get_global_var('func_155')
func_159_call = mutated_mod.get_global_var('func_159')
const_412 = relay.const([0.995748,-4.943871,-0.070871,6.432690,0.206461,4.168191,6.805266,-0.453787,3.383950,-2.636476,5.686373,7.332713,9.574273,1.356872,-3.855646,-5.057855,2.113309,-3.495270,5.700978,7.868071,-0.522976,7.787577,-5.413145,-6.030841,3.074149,7.783136,8.171849,0.144776,3.338060,7.074390,7.285905,0.883007,-9.790792,-7.779546,0.365688,1.971377,6.130273,7.760047,-3.717723,9.402134,3.933009,-4.107825,1.009242,4.407472,0.265094,-5.612467,-6.564245,7.664527,-0.753567,9.972305,1.350007,0.921791,6.249984,-9.628600,4.027074,0.061189,-9.050678,2.243814,1.970412,9.347452,-8.584974,8.520753,7.876995,2.641923,3.461654,-8.935231,1.179082,5.595899,-9.468742,3.263560,6.612143,-1.710945,-0.723115,9.915747,0.283503,-9.179967,9.443384,-9.104439,-3.636973,-6.501759,-3.343532,-5.185413,-0.390956,0.678496,8.502832,-5.238269,6.828610,7.361091,5.984717,-5.759078,2.993073,-5.483864,-2.077365,2.099317,-3.523184,-6.270795,-1.025029,1.090410,-8.046589,-3.465962,7.343732,-7.836522,0.752886,-4.463724], dtype = "float32")#candidate|412|(104,)|const|float32
call_411 = func_155_call(relay.reshape(const_412.astype('float32'), [13, 8]), relay.reshape(const_412.astype('float32'), [13, 8]), )
call_413 = func_155_call(relay.reshape(const_412.astype('float32'), [13, 8]), relay.reshape(const_412.astype('float32'), [13, 8]), )
func_317_call = mod.get_global_var('func_317')
func_324_call = mutated_mod.get_global_var('func_324')
const_415 = relay.const([-5.029436,-0.625579,4.215288,5.052643,-8.952521,0.063295,3.208955,9.999310,9.949418,9.992012,-0.604993,1.693924,-1.729573,2.601239,-3.355583,2.593126,5.819939,3.018520,2.704580,-0.197738,-5.714927,8.765640,-9.293203,5.489849,1.612015,-8.960389,7.482002,-7.192599,-9.862940,3.583621,-6.230337,7.968107,3.601680,-4.673462,8.090004,-2.413223,0.759243,2.733186,-6.998683,2.643721,0.106595,9.221630,-0.650684,-6.022903,-8.868203,6.451349,1.628809,-4.489856,9.454458,7.779075,6.245434,3.065509,-0.429482,-8.222019,4.810205,-3.363230,0.128202,6.676676,3.113915,-6.797174,-0.823532,-7.958960,-7.434290,-0.079111,-1.225624,9.751071,8.647761,-5.421559,9.486677,2.211858,-1.051344,-1.595781,9.702598,2.260854,0.391956,-6.436290,7.779595,0.351295,1.714281,-5.058184,-3.170576,-0.215446,-8.081598,4.493290,6.758169,-8.768663,-9.322398,-9.573896,-4.729510,-7.115970,2.764832,0.626536,3.262107,4.647336,-9.967633,-0.862127,9.525782,4.795847,8.571457,1.082043,1.446806,6.896731,-8.242717,2.460116,6.291841,6.364940,6.170721,-0.468716,5.567182,3.074827,3.254824,-1.153486,-5.734630,-8.820125,-2.469812,0.595600,2.061887,1.854387,-6.754574,-9.982842,-3.407137,-7.835649,0.129341,8.122056,-0.217812,-1.690413,-1.977241,-9.600892,-0.258805,8.773206,1.580498,-1.432002,1.351706,-1.416842,2.948546,3.892266,-8.250398,3.030699,-0.212506,8.785353,4.552590,-6.444122,-8.364719,5.267577,8.458887,-9.098331,6.375649,0.897460,-2.736516,-8.578789,5.995418,-3.784258,6.959768,4.845611,-4.801177,0.638293,-7.087727,8.709884,-3.489084,-2.739592,3.432919,-4.687136,2.516402,5.983824,-2.115933,9.123692,-3.806818,2.069816,6.134704,3.661509,8.149661,-6.943755,0.436912,8.247182,-2.401562,4.402624,-8.576360,-9.989816,-9.415006,3.467915,5.898230,-5.611877,-8.659631,5.965889,-0.485667,5.231991,1.664366,4.264082,-8.341845,5.057313,-0.214341,-4.963573,-9.667945,7.104853,6.716066,8.011228,8.560222,-7.618979,-6.789623,-4.685373,-1.187300,-3.865550,2.564181,-1.302465,-0.528900,-8.886786,0.880948,0.134840,6.660130,-8.487672,-0.904752,9.873043,2.345757,2.474618,1.682669,-3.092623,-1.487437,-0.019758,-5.282131,5.816616,-7.967838,-4.845033,-4.493832,1.723568,-0.487079,-9.245479,-3.554454,-6.880939,5.472228,7.596919,6.772745,7.075616,-1.575673,0.885915,-2.038864,-2.630137,-6.049863,-1.253489,-8.264822,-0.820524,7.519958,0.798106,-3.969771,0.394773,-1.276254,4.722784,-3.650603,4.336791,-9.463933,-1.278729,6.391860,2.587843,9.750647,-1.704525,6.602362,-8.734438,-3.826522,-2.059106,0.830483,-5.909458,-4.593956,-6.911360,-0.698056,-7.153595,-8.907825,-1.218455,3.694375,-5.162661,4.701206,-0.244010,1.105362,-1.454420,2.703197,3.748553,-0.868084,-7.078973,9.485846,-4.624075,-7.026077,-8.687304,1.166166,-8.728078,-1.053792,5.870658,9.694882,-2.468607,-5.593806,-0.669416,-8.091949,-7.302025,-1.081544,3.264153,6.228765,9.880510,-6.203461,1.567219,-6.340680,-8.017412,-2.098692,0.161462,-4.537205,-2.217767,-6.504692,-7.020322,-4.973955,8.623295,6.282672,-7.545598,-0.961180,-6.826859,5.214810,-0.057405,9.606267,-8.927344,-7.925302,4.405029,-2.187570,3.848490,4.724194,7.116115], dtype = "float64")#candidate|415|(320,)|const|float64
call_414 = relay.TupleGetItem(func_317_call(relay.reshape(const_415.astype('float64'), [16, 4, 5]), relay.reshape(const_415.astype('float64'), [16, 4, 5]), relay.reshape(const_415.astype('float64'), [16, 4, 5]), relay.reshape(const_412.astype('float32'), [104,]), relay.reshape(const_415.astype('int8'), [16, 4, 5]), relay.reshape(const_415.astype('float64'), [16, 4, 5]), ), 2)
call_416 = relay.TupleGetItem(func_324_call(relay.reshape(const_415.astype('float64'), [16, 4, 5]), relay.reshape(const_415.astype('float64'), [16, 4, 5]), relay.reshape(const_415.astype('float64'), [16, 4, 5]), relay.reshape(const_412.astype('float32'), [104,]), relay.reshape(const_415.astype('int8'), [16, 4, 5]), relay.reshape(const_415.astype('float64'), [16, 4, 5]), ), 2)
bop_417 = relay.logical_or(uop_396.astype('bool'), relay.reshape(var_390.astype('bool'), relay.shape_of(uop_396))) # shape=(4, 9, 13)
uop_420 = relay.log(uop_391.astype('float32')) # shape=(4, 9, 13)
func_155_call = mod.get_global_var('func_155')
func_159_call = mutated_mod.get_global_var('func_159')
call_422 = func_155_call(relay.reshape(call_414.astype('float32'), [13, 8]), relay.reshape(const_412.astype('float32'), [13, 8]), )
call_423 = func_155_call(relay.reshape(call_414.astype('float32'), [13, 8]), relay.reshape(const_412.astype('float32'), [13, 8]), )
output = relay.Tuple([uop_394,bop_398,call_401,var_402,bop_406,call_411,const_412,call_414,const_415,bop_417,uop_420,call_422,])
output2 = relay.Tuple([uop_394,bop_398,call_403,var_402,bop_406,call_413,const_412,call_416,const_415,bop_417,uop_420,call_423,])
func_434 = relay.Function([var_390,var_402,], output)
mod['func_434'] = func_434
mod = relay.transform.InferType()(mod)
var_435 = relay.var("var_435", dtype = "float32", shape = (4, 9, 13))#candidate|435|(4, 9, 13)|var|float32
var_436 = relay.var("var_436", dtype = "float32", shape = (12,))#candidate|436|(12,)|var|float32
output = func_434(var_435,var_436,)
func_437 = relay.Function([var_435,var_436,], output)
mutated_mod['func_437'] = func_437
mutated_mod = relay.transform.InferType()(mutated_mod)
var_529 = relay.var("var_529", dtype = "float64", shape = (1, 16, 10))#candidate|529|(1, 16, 10)|var|float64
uop_530 = relay.asinh(var_529.astype('float64')) # shape=(1, 16, 10)
func_434_call = mod.get_global_var('func_434')
func_437_call = mutated_mod.get_global_var('func_437')
const_538 = relay.const([4.919486,-9.560568,-6.481322,-7.742349,7.344630,-8.062246,-0.103797,-2.360130,4.607020,-4.738603,8.248350,6.215130,2.620868,-6.970131,8.476937,-5.916739,9.340667,-6.576840,4.328580,6.658554,5.307953,7.431545,0.309262,-8.182239,-0.251329,-0.041816,-3.877710,7.042449,1.636397,-5.378191,-0.557444,-9.000736,-5.274031,1.561324,2.529792,-5.662525,-2.919562,-8.619880,5.710864,-5.183045,6.376718,-5.695679,-3.191983,2.896977,0.694351,-5.763352,-8.026726,-8.408425,9.604215,-1.393626,-8.939339,4.466122,7.951765,4.946565,-2.848414,-0.766316,-3.451764,0.350628,8.018085,-2.783711,8.446308,-6.165917,8.897756,-0.229096,-1.831420,4.652595,0.042353,1.326653,6.892200,-9.494562,-6.918954,8.463196,7.582139,7.655569,-1.716025,9.423176,4.273688,7.126548,-0.180362,4.600614,6.167052,-4.063411,-7.453250,0.599432,-7.095679,2.927320,-6.074153,5.145400,-0.753826,-8.439357,-4.336190,-7.179774,8.372075,0.134523,7.440330,-9.212504,2.017676,6.898284,-9.638695,7.452406,-6.787163,4.342962,-3.936860,-3.376299,-4.305620,0.242695,9.941045,-9.152599,9.972870,-2.239913,-1.725961,1.489174,6.347499,0.078286,5.294545,-3.222211,-2.251238,1.602671,-6.530755,3.562177,-6.858954,-3.104898,-8.818979,8.741201,-3.864920,-6.838276,-0.616485,-2.074900,-7.360843,-5.865702,6.442400,-0.568890,-1.851953,-6.469413,9.975575,2.036628,-9.906659,5.493895,-0.321003,-0.891932,4.101295,-3.686971,6.564369,4.326363,-8.496091,3.913261,4.569140,7.962723,9.793509,-2.579562,5.024458,2.779992,0.517947,-1.036743,6.774239,-6.926061,9.833638,-1.436802,-2.581035,0.257850,4.037743,-4.989278,-1.626339,1.977570,7.653199,-5.009688,1.412199,-6.403380,0.717869,3.000865,8.778612,7.767649,-3.755919,6.501436,7.636804,1.914917,-9.434985,1.982150,-7.188572,-4.554517,0.229757,-4.716181,-9.079532,2.146623,-8.894845,-9.785320,2.547223,-0.753015,-0.927297,-7.750170,-8.684325,-3.494742,0.593720,0.957124,-1.028080,8.893659,5.484446,9.793014,-9.140765,0.698382,4.134552,-1.026562,1.979025,-7.766358,9.225715,7.177167,-3.383757,-6.196235,-5.980564,6.423776,8.412685,-8.457959,4.754125,9.609847,-7.044885,-6.691739,-7.391077,-2.069394,3.760213,4.413879,2.925419,9.899429,4.796464,6.011291,0.321689,-2.288743,7.792270,9.074470,-4.941889,-2.173180,8.560509,-8.861821,-4.325514,6.916110,4.021323,-9.369192,-6.892792,0.201130,6.505167,0.962147,3.741349,5.118011,-9.047324,7.008760,2.641350,0.663232,-6.575873,2.579045,3.383745,8.920462,1.778491,8.703742,8.983274,3.753783,-2.726794,7.407206,1.061344,-5.529346,-7.963065,-1.100037,-3.695781,-8.734671,-4.557479,6.535040,3.811709,-4.071296,-0.950439,-3.993073,-4.909658,-8.685001,-8.872248,2.920612,-5.581502,4.545688,0.847105,4.981114,0.647941,6.912226,1.877721,8.634274,0.159595,5.181789,-7.527412,0.445367,-9.295060,6.079863,1.718006,-6.967578,-9.884581,-7.619398,9.175749,4.099224,-3.805750,-9.208709,3.972411,-5.474235,-1.713713,-4.926721,7.831458,8.771970,3.117498,-4.236803,9.609101,1.538098,-9.020191,-6.191801,8.006522,-6.716119,-1.951607,2.572387,5.325504,-3.825683,9.197785,-4.076652,-9.227707,-7.424440,-9.678548,-4.564301,4.804573,2.220056,-6.056265,-6.716862,-0.019030,5.508463,8.470100,-7.059871,-4.641438,-7.699260,-2.793995,-7.284032,1.871330,-4.610836,5.981233,0.253401,-7.488387,-4.333734,-5.149421,-1.404152,7.323227,3.996887,-0.732273,5.988127,-8.743230,9.491123,-3.172989,2.819500,-0.174758,9.491598,9.081140,3.588882,-8.164419,-1.569751,-6.634097,1.199757,-3.864026,-4.866194,-0.397379,2.218658,-9.069128,9.720997,-3.642891,-9.646269,4.589646,-4.669186,-0.350252,-9.953819,4.035644,-9.541263,-5.885524,4.940084,-3.534961,4.037314,-3.727699,-6.829467,3.170420,2.464408,-1.096078,7.105857,-4.165149,-6.992426,6.226313,8.237123,1.868291,-7.330444,1.683515,-9.364820,8.156173,-6.323658,-2.622674,-7.777237,4.353620,3.282543,6.765386,0.973773,-9.591167,7.854019,-1.278999,-4.600730,-2.574921,6.971116,0.291139,8.553484,-3.395002,-1.478406,-9.079132,-5.742605,-5.817580,-1.438082,-1.556566,6.999087,5.090653,-0.387924,0.848536,-1.232541,8.690749,9.948335,5.911094,-3.329249,8.486875,8.883445,3.204048,7.090876,9.291752,-8.735847,0.664358,2.697713,-1.746437,-5.262670,-0.425478,8.327430,3.778640,-5.516597,3.470822,-6.139704,2.756278,1.766941,3.357610,3.868843,-1.379231,2.788082,4.236591,-0.014939,-6.509684,-6.525259,8.230386,5.711788,-4.016959,-8.945787,-1.654984,-0.184240,-5.709203,-5.123852,-1.407836,-8.471664,3.685083,-6.007451,-6.607982,-5.244308,-0.706014,-3.109289,-9.635621,-9.981617,4.044737,3.689834,0.691601,-9.239828,3.413909,6.040652], dtype = "float32")#candidate|538|(468,)|const|float32
var_539 = relay.var("var_539", dtype = "float32", shape = (1, 12))#candidate|539|(1, 12)|var|float32
call_537 = relay.TupleGetItem(func_434_call(relay.reshape(const_538.astype('float32'), [4, 9, 13]), relay.reshape(var_539.astype('float32'), [12,]), ), 7)
call_540 = relay.TupleGetItem(func_437_call(relay.reshape(const_538.astype('float32'), [4, 9, 13]), relay.reshape(var_539.astype('float32'), [12,]), ), 7)
output = relay.Tuple([uop_530,call_537,const_538,var_539,])
output2 = relay.Tuple([uop_530,call_540,const_538,var_539,])
func_541 = relay.Function([var_529,var_539,], output)
mod['func_541'] = func_541
mod = relay.transform.InferType()(mod)
mutated_mod['func_541'] = func_541
mutated_mod = relay.transform.InferType()(mutated_mod)
func_541_call = mutated_mod.get_global_var('func_541')
var_543 = relay.var("var_543", dtype = "float64", shape = (1, 16, 10))#candidate|543|(1, 16, 10)|var|float64
var_544 = relay.var("var_544", dtype = "float32", shape = (1, 12))#candidate|544|(1, 12)|var|float32
call_542 = func_541_call(var_543,var_544,)
output = call_542
func_545 = relay.Function([var_543,var_544,], output)
mutated_mod['func_545'] = func_545
mutated_mod = relay.transform.InferType()(mutated_mod)
var_557 = relay.var("var_557", dtype = "int16", shape = ())#candidate|557|()|var|int16
const_558 = relay.const([-5,-9,-4,-6,-2,-7,-1], dtype = "int16")#candidate|558|(7,)|const|int16
bop_559 = relay.greater_equal(var_557.astype('bool'), const_558.astype('bool')) # shape=(7,)
func_434_call = mod.get_global_var('func_434')
func_437_call = mutated_mod.get_global_var('func_437')
var_564 = relay.var("var_564", dtype = "float32", shape = (3, 156))#candidate|564|(3, 156)|var|float32
const_565 = relay.const([-0.077898,4.205852,-0.425442,-5.231197,-9.317109,6.368815,1.386367,6.454583,-5.139901,-8.876898,-0.946819,-0.831098], dtype = "float32")#candidate|565|(12,)|const|float32
call_563 = relay.TupleGetItem(func_434_call(relay.reshape(var_564.astype('float32'), [4, 9, 13]), relay.reshape(const_565.astype('float32'), [12,]), ), 11)
call_566 = relay.TupleGetItem(func_437_call(relay.reshape(var_564.astype('float32'), [4, 9, 13]), relay.reshape(const_565.astype('float32'), [12,]), ), 11)
output = relay.Tuple([bop_559,call_563,var_564,const_565,])
output2 = relay.Tuple([bop_559,call_566,var_564,const_565,])
func_578 = relay.Function([var_557,var_564,], output)
mod['func_578'] = func_578
mod = relay.transform.InferType()(mod)
mutated_mod['func_578'] = func_578
mutated_mod = relay.transform.InferType()(mutated_mod)
func_578_call = mutated_mod.get_global_var('func_578')
var_580 = relay.var("var_580", dtype = "int16", shape = ())#candidate|580|()|var|int16
var_581 = relay.var("var_581", dtype = "float32", shape = (3, 156))#candidate|581|(3, 156)|var|float32
call_579 = func_578_call(var_580,var_581,)
output = call_579
func_582 = relay.Function([var_580,var_581,], output)
mutated_mod['func_582'] = func_582
mutated_mod = relay.transform.InferType()(mutated_mod)
const_584 = relay.const([[7.256952,-4.578725,0.023620,-1.162304],[0.534170,-1.581927,8.256142,6.907155],[-8.803854,-9.383694,7.246639,-7.535245],[4.455221,-5.077692,0.046809,0.725136],[3.181245,9.957540,2.837803,0.122858]], dtype = "float64")#candidate|584|(5, 4)|const|float64
uop_585 = relay.erf(const_584.astype('float64')) # shape=(5, 4)
func_578_call = mod.get_global_var('func_578')
func_582_call = mutated_mod.get_global_var('func_582')
var_590 = relay.var("var_590", dtype = "int16", shape = ())#candidate|590|()|var|int16
var_591 = relay.var("var_591", dtype = "float32", shape = (468,))#candidate|591|(468,)|var|float32
call_589 = relay.TupleGetItem(func_578_call(relay.reshape(var_590.astype('int16'), []), relay.reshape(var_591.astype('float32'), [3, 156]), ), 0)
call_592 = relay.TupleGetItem(func_582_call(relay.reshape(var_590.astype('int16'), []), relay.reshape(var_591.astype('float32'), [3, 156]), ), 0)
output = relay.Tuple([uop_585,call_589,var_590,var_591,])
output2 = relay.Tuple([uop_585,call_592,var_590,var_591,])
func_594 = relay.Function([var_590,var_591,], output)
mod['func_594'] = func_594
mod = relay.transform.InferType()(mod)
var_595 = relay.var("var_595", dtype = "int16", shape = ())#candidate|595|()|var|int16
var_596 = relay.var("var_596", dtype = "float32", shape = (468,))#candidate|596|(468,)|var|float32
output = func_594(var_595,var_596,)
func_597 = relay.Function([var_595,var_596,], output)
mutated_mod['func_597'] = func_597
mutated_mod = relay.transform.InferType()(mutated_mod)
var_612 = relay.var("var_612", dtype = "float32", shape = (6,))#candidate|612|(6,)|var|float32
var_613 = relay.var("var_613", dtype = "float32", shape = (6,))#candidate|613|(6,)|var|float32
bop_614 = relay.floor_mod(var_612.astype('float32'), relay.reshape(var_613.astype('float32'), relay.shape_of(var_612))) # shape=(6,)
uop_626 = relay.erf(var_612.astype('float32')) # shape=(6,)
uop_629 = relay.cos(uop_626.astype('float32')) # shape=(6,)
bop_631 = relay.subtract(uop_629.astype('float64'), relay.reshape(var_612.astype('float64'), relay.shape_of(uop_629))) # shape=(6,)
bop_634 = relay.minimum(bop_631.astype('int16'), relay.reshape(uop_626.astype('int16'), relay.shape_of(bop_631))) # shape=(6,)
var_638 = relay.var("var_638", dtype = "int16", shape = (6,))#candidate|638|(6,)|var|int16
bop_639 = relay.logical_and(bop_634.astype('bool'), relay.reshape(var_638.astype('bool'), relay.shape_of(bop_634))) # shape=(6,)
uop_646 = relay.tan(bop_634.astype('float32')) # shape=(6,)
uop_648 = relay.exp(uop_646.astype('float32')) # shape=(6,)
bop_651 = relay.mod(uop_648.astype('float64'), relay.reshape(uop_629.astype('float64'), relay.shape_of(uop_648))) # shape=(6,)
bop_658 = relay.logical_and(uop_646.astype('bool'), relay.reshape(uop_629.astype('bool'), relay.shape_of(uop_646))) # shape=(6,)
func_594_call = mod.get_global_var('func_594')
func_597_call = mutated_mod.get_global_var('func_597')
const_662 = relay.const(1, dtype = "int16")#candidate|662|()|const|int16
var_663 = relay.var("var_663", dtype = "float32", shape = (468,))#candidate|663|(468,)|var|float32
call_661 = relay.TupleGetItem(func_594_call(relay.reshape(const_662.astype('int16'), []), relay.reshape(var_663.astype('float32'), [468,]), ), 0)
call_664 = relay.TupleGetItem(func_597_call(relay.reshape(const_662.astype('int16'), []), relay.reshape(var_663.astype('float32'), [468,]), ), 0)
bop_666 = relay.bitwise_and(uop_648.astype('int16'), relay.reshape(uop_626.astype('int16'), relay.shape_of(uop_648))) # shape=(6,)
uop_669 = relay.log10(bop_651.astype('float64')) # shape=(6,)
bop_672 = relay.divide(bop_651.astype('float64'), relay.reshape(bop_614.astype('float64'), relay.shape_of(bop_651))) # shape=(6,)
bop_675 = relay.greater_equal(uop_669.astype('bool'), relay.reshape(uop_648.astype('bool'), relay.shape_of(uop_669))) # shape=(6,)
uop_684 = relay.asinh(bop_675.astype('float64')) # shape=(6,)
const_686 = relay.const([6.374298,1.782559,-8.074223,-9.568647,-9.421834,-9.330557], dtype = "float64")#candidate|686|(6,)|const|float64
bop_687 = relay.right_shift(uop_684.astype('uint16'), relay.reshape(const_686.astype('uint16'), relay.shape_of(uop_684))) # shape=(6,)
func_594_call = mod.get_global_var('func_594')
func_597_call = mutated_mod.get_global_var('func_597')
call_691 = relay.TupleGetItem(func_594_call(relay.reshape(const_662.astype('int16'), []), relay.reshape(var_663.astype('float32'), [468,]), ), 2)
call_692 = relay.TupleGetItem(func_597_call(relay.reshape(const_662.astype('int16'), []), relay.reshape(var_663.astype('float32'), [468,]), ), 2)
bop_694 = relay.add(uop_648.astype('float64'), relay.reshape(uop_669.astype('float64'), relay.shape_of(uop_648))) # shape=(6,)
uop_702 = relay.log2(bop_687.astype('float64')) # shape=(6,)
uop_704 = relay.atan(uop_702.astype('float32')) # shape=(6,)
output = relay.Tuple([bop_639,bop_658,call_661,const_662,var_663,bop_666,bop_672,call_691,bop_694,uop_704,])
output2 = relay.Tuple([bop_639,bop_658,call_664,const_662,var_663,bop_666,bop_672,call_692,bop_694,uop_704,])
func_707 = relay.Function([var_612,var_613,var_638,var_663,], output)
mod['func_707'] = func_707
mod = relay.transform.InferType()(mod)
mutated_mod['func_707'] = func_707
mutated_mod = relay.transform.InferType()(mutated_mod)
func_707_call = mutated_mod.get_global_var('func_707')
var_709 = relay.var("var_709", dtype = "float32", shape = (6,))#candidate|709|(6,)|var|float32
var_710 = relay.var("var_710", dtype = "float32", shape = (6,))#candidate|710|(6,)|var|float32
var_711 = relay.var("var_711", dtype = "int16", shape = (6,))#candidate|711|(6,)|var|int16
var_712 = relay.var("var_712", dtype = "float32", shape = (468,))#candidate|712|(468,)|var|float32
call_708 = func_707_call(var_709,var_710,var_711,var_712,)
output = call_708
func_713 = relay.Function([var_709,var_710,var_711,var_712,], output)
mutated_mod['func_713'] = func_713
mutated_mod = relay.transform.InferType()(mutated_mod)
var_755 = relay.var("var_755", dtype = "uint8", shape = ())#candidate|755|()|var|uint8
var_756 = relay.var("var_756", dtype = "uint8", shape = (13, 16))#candidate|756|(13, 16)|var|uint8
bop_757 = relay.right_shift(var_755.astype('uint8'), var_756.astype('uint8')) # shape=(13, 16)
output = relay.Tuple([bop_757,])
output2 = relay.Tuple([bop_757,])
func_762 = relay.Function([var_755,var_756,], output)
mod['func_762'] = func_762
mod = relay.transform.InferType()(mod)
mutated_mod['func_762'] = func_762
mutated_mod = relay.transform.InferType()(mutated_mod)
func_762_call = mutated_mod.get_global_var('func_762')
var_764 = relay.var("var_764", dtype = "uint8", shape = ())#candidate|764|()|var|uint8
var_765 = relay.var("var_765", dtype = "uint8", shape = (13, 16))#candidate|765|(13, 16)|var|uint8
call_763 = func_762_call(var_764,var_765,)
output = call_763
func_766 = relay.Function([var_764,var_765,], output)
mutated_mod['func_766'] = func_766
mutated_mod = relay.transform.InferType()(mutated_mod)
var_771 = relay.var("var_771", dtype = "float32", shape = ())#candidate|771|()|var|float32
var_772 = relay.var("var_772", dtype = "float32", shape = (6,))#candidate|772|(6,)|var|float32
bop_773 = relay.power(var_771.astype('float32'), var_772.astype('float32')) # shape=(6,)
func_347_call = mod.get_global_var('func_347')
func_353_call = mutated_mod.get_global_var('func_353')
const_779 = relay.const([-4,9,3,-2,7,10,9,4], dtype = "int32")#candidate|779|(8,)|const|int32
var_780 = relay.var("var_780", dtype = "uint64", shape = (2, 120))#candidate|780|(2, 120)|var|uint64
var_781 = relay.var("var_781", dtype = "float32", shape = (12,))#candidate|781|(12,)|var|float32
call_778 = relay.TupleGetItem(func_347_call(relay.reshape(const_779.astype('int32'), [8,]), relay.reshape(const_779.astype('int32'), [8,]), relay.reshape(var_780.astype('uint64'), [240,]), relay.reshape(var_781.astype('float32'), [12,]), ), 1)
call_782 = relay.TupleGetItem(func_353_call(relay.reshape(const_779.astype('int32'), [8,]), relay.reshape(const_779.astype('int32'), [8,]), relay.reshape(var_780.astype('uint64'), [240,]), relay.reshape(var_781.astype('float32'), [12,]), ), 1)
output = relay.Tuple([bop_773,call_778,const_779,var_780,var_781,])
output2 = relay.Tuple([bop_773,call_782,const_779,var_780,var_781,])
func_796 = relay.Function([var_771,var_772,var_780,var_781,], output)
mod['func_796'] = func_796
mod = relay.transform.InferType()(mod)
var_797 = relay.var("var_797", dtype = "float32", shape = ())#candidate|797|()|var|float32
var_798 = relay.var("var_798", dtype = "float32", shape = (6,))#candidate|798|(6,)|var|float32
var_799 = relay.var("var_799", dtype = "uint64", shape = (2, 120))#candidate|799|(2, 120)|var|uint64
var_800 = relay.var("var_800", dtype = "float32", shape = (12,))#candidate|800|(12,)|var|float32
output = func_796(var_797,var_798,var_799,var_800,)
func_801 = relay.Function([var_797,var_798,var_799,var_800,], output)
mutated_mod['func_801'] = func_801
mutated_mod = relay.transform.InferType()(mutated_mod)
const_817 = relay.const([[[-0.735676,9.405657,-0.876983],[-1.647914,2.870509,-2.296052],[-3.462763,8.307190,9.746087],[-1.112130,-8.695598,-8.785626],[-7.464921,4.746552,-5.724941],[0.733868,5.495128,3.847348],[-7.266877,7.851499,8.507854],[-5.159881,-4.987218,1.699776],[-6.167605,-4.998497,5.314466],[-9.797911,0.661974,0.907253],[-9.328913,-3.191275,2.360117],[9.279823,5.839768,7.767100],[2.298993,-7.026086,0.776791],[3.969557,-1.746060,-5.636636]],[[-4.414179,4.135534,-0.448458],[7.571549,-9.367832,6.483928],[4.810655,1.695516,1.430934],[-3.104127,-0.453618,-4.238802],[-0.080857,-6.369737,-2.756198],[-2.120864,4.832148,-9.455726],[-3.500489,6.361183,-8.400488],[-1.862015,2.557027,-5.800504],[-3.664736,4.059663,-6.340270],[-0.248352,6.603187,4.330047],[8.045802,-1.974130,7.124206],[5.834767,5.948991,-6.455684],[2.105400,0.033206,-8.172136],[7.324316,-3.549048,-4.343777]],[[3.634971,-2.609093,3.883196],[-2.547535,6.761552,4.511541],[3.152778,7.217046,0.148784],[-2.367014,-7.209181,-2.658164],[-3.133034,-6.875711,-4.073268],[4.567914,1.437267,4.151875],[3.842981,-4.889696,-7.376234],[1.534687,-5.849291,-0.673945],[3.192000,-0.522441,5.482777],[-7.012100,-3.542329,-6.411383],[2.475807,-6.963496,2.500827],[7.842022,4.241552,-7.243574],[-6.186836,-6.033690,-4.700220],[-6.202132,0.012464,6.822754]],[[3.995584,-8.956360,-9.624623],[5.507831,-7.562023,1.719176],[1.836569,-9.959828,-3.481248],[-0.502299,7.217540,6.597932],[-3.221130,6.947086,-3.908068],[-6.363033,-3.076809,9.820017],[3.883119,5.714505,8.855179],[-4.907331,4.265747,7.113179],[2.860129,-5.279894,-7.385861],[-9.789754,-1.006818,7.262178],[-0.234435,-1.022843,-3.876470],[-8.969716,-3.418627,-8.612493],[-4.076404,-0.688393,6.694099],[5.834107,-2.924528,8.680331]],[[7.188104,2.138105,3.645207],[-4.488426,-0.863912,1.484333],[-9.851508,-4.425554,2.254044],[-2.468826,4.492821,1.001237],[3.204432,1.731694,3.539636],[-1.970097,-2.538703,-4.751060],[-6.282685,-6.668927,3.352657],[-8.734813,3.878714,4.914329],[5.614110,-1.502621,-6.796715],[0.259227,-4.808863,8.191656],[-3.671936,-8.721268,2.853420],[-8.981582,1.145539,-4.404218],[-3.080885,-9.872561,-2.549467],[-6.659363,-9.229107,-9.762183]],[[3.125412,-3.630873,6.648736],[8.678296,9.518827,-9.399943],[-6.108876,7.646348,8.498573],[-6.229800,-7.348347,0.608548],[-8.721531,-8.586706,-2.841564],[-8.538069,-4.095724,3.379851],[-8.265665,1.550694,-7.195561],[9.243803,6.547867,-0.610967],[-7.577122,-4.949277,-5.527191],[2.038215,-7.566373,5.494647],[7.592279,-0.250978,-0.361600],[0.647422,-7.376732,-2.766174],[-4.479910,8.671157,-2.113989],[8.635077,-1.174600,2.506869]],[[7.307126,-6.339444,-3.648831],[3.204773,7.765481,-4.507035],[6.064187,-6.156928,-9.883204],[1.303510,-0.252468,-4.205558],[-3.522197,-6.392800,-9.657809],[7.167946,-1.620829,0.623363],[2.640680,-6.950382,-4.410496],[-0.524282,-0.826333,0.438683],[-1.840328,-8.859090,3.350078],[5.634317,-7.199679,-0.478078],[-2.913043,-0.341552,8.328941],[1.640037,-7.302442,6.859866],[-5.569413,6.668960,7.979180],[-7.047862,1.949397,-4.307010]],[[1.640683,-2.684188,2.612199],[-4.129698,9.660921,1.005658],[-5.814748,2.813608,-9.968421],[5.947826,0.134225,-6.200015],[9.391591,-9.933011,6.639186],[5.303229,-7.966002,-3.609701],[2.035976,4.932601,4.282416],[8.015393,-2.289465,4.172602],[-2.794570,2.727720,1.888232],[-9.034264,-2.174622,0.036587],[0.271089,2.988235,-6.148254],[-3.056149,-0.980801,-2.505915],[-1.101256,-5.033851,2.490683],[6.429735,-7.761949,8.776366]],[[-7.686043,5.005994,-5.368103],[-6.154003,9.985834,8.367494],[4.059771,9.563465,-6.704370],[-7.374135,-2.752793,0.168490],[-2.377829,2.875941,5.897658],[-5.310018,-0.192001,7.518148],[-9.216154,-9.912723,5.001694],[4.303486,0.980908,-5.251681],[2.309912,-6.826475,9.541502],[6.916298,7.485885,1.283932],[4.596032,-9.682439,4.855035],[-0.007645,-6.460592,2.540992],[-2.826608,-1.792261,-0.883925],[1.201393,2.944104,0.498720]],[[2.116243,-4.528829,8.439949],[-0.662505,6.734099,-0.462965],[-7.995507,-7.428669,3.837473],[-3.939493,4.586231,-9.332107],[0.662114,3.834988,5.451729],[-6.769214,1.583932,4.172228],[-3.156439,8.544717,0.911989],[-8.044046,-3.161366,-7.170893],[-1.716126,0.561626,3.605947],[2.495184,-3.068958,-0.534635],[3.597358,-4.666127,6.827273],[5.731721,6.833801,-8.178508],[-1.608266,-2.773505,8.875507],[0.002188,6.541672,6.270171]],[[-1.222779,-8.766307,0.975732],[-8.243458,1.650505,3.758023],[2.864922,-5.475635,6.254919],[6.167104,4.612565,-3.959050],[-2.901624,-2.868678,6.807423],[-5.601829,9.690093,-6.297592],[1.932701,9.880047,1.042630],[-8.284533,-7.992481,-5.852704],[0.937250,6.717509,6.489880],[1.839751,3.665616,-2.460981],[-9.264115,-6.265439,6.043161],[-2.523881,-1.877225,9.807995],[6.399940,-7.846153,-8.850143],[7.045705,-1.616045,-7.293437]]], dtype = "float64")#candidate|817|(11, 14, 3)|const|float64
uop_818 = relay.sin(const_817.astype('float64')) # shape=(11, 14, 3)
bop_820 = relay.bitwise_and(uop_818.astype('int32'), relay.reshape(const_817.astype('int32'), relay.shape_of(uop_818))) # shape=(11, 14, 3)
var_823 = relay.var("var_823", dtype = "int32", shape = (11, 14, 3))#candidate|823|(11, 14, 3)|var|int32
bop_824 = relay.logical_and(bop_820.astype('bool'), relay.reshape(var_823.astype('bool'), relay.shape_of(bop_820))) # shape=(11, 14, 3)
const_832 = relay.const([[[-1,10,2],[-6,-4,-7],[9,-9,-1],[-6,-2,9],[-10,-5,-2],[6,1,-3],[9,6,-4],[-2,-4,-9],[-3,-1,-5],[10,-1,-6],[2,-6,9],[-8,9,-2],[6,8,-6],[-5,6,-3]],[[-8,-4,7],[-10,3,-2],[5,1,-3],[-1,-6,8],[-7,-10,-1],[-4,3,6],[10,-9,-5],[-7,10,-6],[-2,1,-7],[-9,-2,-9],[-9,9,-3],[-4,-4,7],[-3,-10,4],[-7,5,4]],[[6,8,2],[10,2,-5],[-8,-9,-1],[2,3,-10],[-1,5,6],[-10,7,-2],[-9,2,4],[8,8,6],[6,-5,1],[-9,-4,-8],[6,-5,-8],[-1,9,-10],[-3,-7,-6],[10,1,10]],[[-1,-4,-5],[-5,2,3],[-7,9,-10],[-1,5,-3],[10,-2,9],[3,-3,4],[9,9,9],[5,10,2],[-3,4,10],[1,-1,1],[7,9,-8],[7,4,3],[10,9,-2],[-10,5,-6]],[[-1,5,2],[-3,9,-10],[-7,-6,-5],[-6,9,-6],[8,7,-10],[-8,-10,5],[8,-2,2],[-4,2,-1],[1,3,5],[-10,-7,-6],[-9,-10,-7],[4,4,4],[9,6,10],[-10,7,8]],[[5,4,-6],[-9,-5,6],[7,-9,6],[-9,-3,-10],[-2,-1,-6],[-10,-10,2],[-5,1,-3],[1,-6,3],[8,-9,-6],[-2,-9,-7],[6,5,10],[-10,-2,4],[-7,-7,9],[-5,1,6]],[[4,6,-2],[8,10,4],[5,-6,4],[-2,7,3],[9,-1,-5],[8,6,3],[-8,-6,8],[-2,-5,-10],[10,-1,3],[-3,4,-3],[-2,8,-4],[5,-6,4],[-4,-7,-5],[-9,10,2]],[[-4,4,-10],[8,-7,-1],[1,-4,-1],[-1,5,4],[-6,6,1],[-9,-4,4],[9,7,10],[8,-4,-8],[-3,9,-8],[9,-6,-2],[-5,-3,-2],[-5,8,-9],[-10,2,9],[10,-1,-9]],[[4,5,-3],[8,-5,-8],[-7,-10,-5],[-5,-5,-10],[9,4,2],[8,-1,5],[-7,-3,3],[1,-4,-7],[1,-5,-1],[1,-6,-7],[-10,4,8],[-7,-2,9],[-4,5,3],[-7,4,-1]],[[7,-2,-8],[-1,3,-2],[-5,-6,-1],[-2,5,-3],[9,-3,1],[7,-7,-1],[7,-5,-7],[3,-8,-7],[-2,-4,-6],[5,8,-8],[1,7,4],[3,1,-4],[-7,-10,-4],[5,-8,4]],[[7,-9,5],[-2,-3,-4],[-8,-8,7],[9,7,-10],[-8,10,4],[-10,-1,-6],[8,-5,-8],[8,6,-6],[8,-6,5],[-9,-10,9],[-10,5,-9],[2,-8,10],[5,-3,-6],[5,-6,-10]]], dtype = "int32")#candidate|832|(11, 14, 3)|const|int32
bop_833 = relay.floor_divide(bop_820.astype('float64'), relay.reshape(const_832.astype('float64'), relay.shape_of(bop_820))) # shape=(11, 14, 3)
bop_837 = relay.less(var_823.astype('bool'), relay.reshape(bop_824.astype('bool'), relay.shape_of(var_823))) # shape=(11, 14, 3)
const_842 = relay.const([[[True,False,True],[False,False,True],[True,True,False],[True,False,True],[True,False,True],[False,False,True],[True,True,True],[False,True,False],[False,True,False],[False,True,True],[False,False,True],[True,True,True],[False,False,False],[False,False,False]],[[False,True,False],[True,True,True],[False,False,False],[True,True,True],[False,True,False],[False,True,True],[True,True,False],[False,True,True],[False,True,True],[False,False,True],[True,False,False],[True,True,False],[False,True,False],[False,True,True]],[[True,True,True],[False,True,False],[True,False,False],[True,False,True],[False,False,False],[False,True,False],[True,False,False],[False,True,True],[False,False,True],[False,False,False],[False,False,True],[True,False,False],[True,True,False],[True,True,False]],[[False,True,True],[True,True,False],[True,True,False],[False,True,True],[True,True,True],[False,True,False],[False,True,False],[True,False,False],[True,False,True],[True,True,False],[False,False,False],[True,True,True],[True,False,False],[True,False,True]],[[False,False,False],[False,False,False],[False,True,True],[False,False,False],[False,False,False],[True,True,True],[False,True,True],[False,False,False],[True,True,True],[False,False,True],[True,True,True],[False,True,False],[False,True,True],[True,True,True]],[[True,True,True],[True,False,True],[True,False,False],[False,False,True],[True,False,False],[False,False,True],[True,True,False],[True,True,True],[True,True,False],[False,True,True],[True,False,False],[True,False,False],[False,True,True],[False,True,True]],[[True,False,True],[False,False,True],[True,False,True],[True,True,False],[False,False,False],[False,False,True],[True,True,True],[True,False,True],[True,False,False],[False,True,False],[False,False,False],[True,False,True],[True,True,True],[False,False,False]],[[False,False,False],[False,False,False],[True,True,True],[True,False,False],[False,False,False],[True,False,True],[True,False,False],[True,False,False],[False,True,False],[True,True,False],[True,True,True],[True,False,True],[True,True,True],[True,False,True]],[[True,True,True],[True,False,False],[True,True,False],[True,False,False],[True,True,True],[True,False,False],[False,False,False],[True,True,False],[False,True,True],[True,False,False],[True,True,True],[False,False,True],[False,True,False],[True,False,False]],[[True,False,True],[True,True,False],[True,True,False],[True,False,True],[True,True,False],[True,False,False],[True,True,True],[False,False,False],[True,False,True],[True,True,False],[True,True,False],[True,False,True],[True,True,True],[True,True,True]],[[False,False,False],[True,True,True],[True,False,False],[True,False,False],[True,True,True],[False,False,False],[False,False,True],[False,True,True],[False,False,True],[True,True,False],[False,True,False],[True,False,False],[False,False,False],[False,True,True]]], dtype = "bool")#candidate|842|(11, 14, 3)|const|bool
bop_843 = relay.maximum(bop_824.astype('int32'), relay.reshape(const_842.astype('int32'), relay.shape_of(bop_824))) # shape=(11, 14, 3)
uop_850 = relay.sqrt(bop_837.astype('float32')) # shape=(11, 14, 3)
output = relay.Tuple([bop_833,bop_843,uop_850,])
output2 = relay.Tuple([bop_833,bop_843,uop_850,])
func_854 = relay.Function([var_823,], output)
mod['func_854'] = func_854
mod = relay.transform.InferType()(mod)
var_855 = relay.var("var_855", dtype = "int32", shape = (11, 14, 3))#candidate|855|(11, 14, 3)|var|int32
output = func_854(var_855)
func_856 = relay.Function([var_855], output)
mutated_mod['func_856'] = func_856
mutated_mod = relay.transform.InferType()(mutated_mod)
var_858 = relay.var("var_858", dtype = "uint64", shape = (9, 11))#candidate|858|(9, 11)|var|uint64
var_859 = relay.var("var_859", dtype = "uint64", shape = (9, 11))#candidate|859|(9, 11)|var|uint64
bop_860 = relay.bitwise_and(var_858.astype('uint64'), relay.reshape(var_859.astype('uint64'), relay.shape_of(var_858))) # shape=(9, 11)
bop_869 = relay.greater_equal(var_858.astype('bool'), relay.reshape(bop_860.astype('bool'), relay.shape_of(var_858))) # shape=(9, 11)
uop_873 = relay.cosh(bop_860.astype('float32')) # shape=(9, 11)
uop_875 = relay.sinh(uop_873.astype('float32')) # shape=(9, 11)
uop_877 = relay.erf(uop_875.astype('float64')) # shape=(9, 11)
bop_879 = relay.power(uop_877.astype('float64'), relay.reshape(var_858.astype('float64'), relay.shape_of(uop_877))) # shape=(9, 11)
output = relay.Tuple([bop_869,bop_879,])
output2 = relay.Tuple([bop_869,bop_879,])
func_882 = relay.Function([var_858,var_859,], output)
mod['func_882'] = func_882
mod = relay.transform.InferType()(mod)
mutated_mod['func_882'] = func_882
mutated_mod = relay.transform.InferType()(mutated_mod)
func_882_call = mutated_mod.get_global_var('func_882')
var_884 = relay.var("var_884", dtype = "uint64", shape = (9, 11))#candidate|884|(9, 11)|var|uint64
var_885 = relay.var("var_885", dtype = "uint64", shape = (9, 11))#candidate|885|(9, 11)|var|uint64
call_883 = func_882_call(var_884,var_885,)
output = call_883
func_886 = relay.Function([var_884,var_885,], output)
mutated_mod['func_886'] = func_886
mutated_mod = relay.transform.InferType()(mutated_mod)
const_920 = relay.const([5.805815,9.233662,7.367537,9.601609,1.813433,-8.358723,-7.885233,-2.215671,0.381285,4.864317], dtype = "float64")#candidate|920|(10,)|const|float64
uop_921 = relay.cosh(const_920.astype('float64')) # shape=(10,)
output = uop_921
output2 = uop_921
func_926 = relay.Function([], output)
mod['func_926'] = func_926
mod = relay.transform.InferType()(mod)
mutated_mod['func_926'] = func_926
mutated_mod = relay.transform.InferType()(mutated_mod)
func_926_call = mutated_mod.get_global_var('func_926')
call_927 = func_926_call()
output = call_927
func_928 = relay.Function([], output)
mutated_mod['func_928'] = func_928
mutated_mod = relay.transform.InferType()(mutated_mod)
var_939 = relay.var("var_939", dtype = "float32", shape = (9,))#candidate|939|(9,)|var|float32
uop_940 = relay.asin(var_939.astype('float32')) # shape=(9,)
bop_947 = relay.bitwise_xor(uop_940.astype('uint16'), relay.reshape(var_939.astype('uint16'), relay.shape_of(uop_940))) # shape=(9,)
const_950 = relay.const([10,-8,2,8,2,-2,2,4,-10], dtype = "uint16")#candidate|950|(9,)|const|uint16
bop_951 = relay.floor_mod(bop_947.astype('float32'), relay.reshape(const_950.astype('float32'), relay.shape_of(bop_947))) # shape=(9,)
uop_960 = relay.sqrt(var_939.astype('float32')) # shape=(9,)
uop_962 = relay.sigmoid(var_939.astype('float64')) # shape=(9,)
bop_965 = relay.less_equal(const_950.astype('bool'), relay.reshape(bop_951.astype('bool'), relay.shape_of(const_950))) # shape=(9,)
func_541_call = mod.get_global_var('func_541')
func_545_call = mutated_mod.get_global_var('func_545')
var_972 = relay.var("var_972", dtype = "float64", shape = (160,))#candidate|972|(160,)|var|float64
const_973 = relay.const([-2.434751,-5.861830,7.583642,7.107756,-2.195604,8.962817,1.092809,-6.941350,-3.836519,-1.660500,7.155243,7.955169], dtype = "float32")#candidate|973|(12,)|const|float32
call_971 = relay.TupleGetItem(func_541_call(relay.reshape(var_972.astype('float64'), [1, 16, 10]), relay.reshape(const_973.astype('float32'), [1, 12]), ), 1)
call_974 = relay.TupleGetItem(func_545_call(relay.reshape(var_972.astype('float64'), [1, 16, 10]), relay.reshape(const_973.astype('float32'), [1, 12]), ), 1)
output = relay.Tuple([uop_960,uop_962,bop_965,call_971,var_972,const_973,])
output2 = relay.Tuple([uop_960,uop_962,bop_965,call_974,var_972,const_973,])
func_977 = relay.Function([var_939,var_972,], output)
mod['func_977'] = func_977
mod = relay.transform.InferType()(mod)
mutated_mod['func_977'] = func_977
mutated_mod = relay.transform.InferType()(mutated_mod)
func_977_call = mutated_mod.get_global_var('func_977')
var_979 = relay.var("var_979", dtype = "float32", shape = (9,))#candidate|979|(9,)|var|float32
var_980 = relay.var("var_980", dtype = "float64", shape = (160,))#candidate|980|(160,)|var|float64
call_978 = func_977_call(var_979,var_980,)
output = call_978
func_981 = relay.Function([var_979,var_980,], output)
mutated_mod['func_981'] = func_981
mutated_mod = relay.transform.InferType()(mutated_mod)
func_926_call = mod.get_global_var('func_926')
func_928_call = mutated_mod.get_global_var('func_928')
call_1004 = func_926_call()
call_1005 = func_926_call()
func_578_call = mod.get_global_var('func_578')
func_582_call = mutated_mod.get_global_var('func_582')
var_1015 = relay.var("var_1015", dtype = "int16", shape = ())#candidate|1015|()|var|int16
const_1016 = relay.const([[5.713764],[-8.018832],[2.397649],[7.938721],[8.187485],[7.342159],[7.250141],[8.877620],[-6.188070],[-3.529015],[7.033365],[-8.446853],[1.468081],[-4.477846],[-6.020048],[7.655694],[5.491801],[-4.087111],[1.521311],[-6.777017],[-1.683861],[8.930917],[-6.634923],[2.785440],[-6.490087],[7.820797],[1.314392],[0.689382],[7.677997],[-6.234312],[2.012532],[9.099523],[-6.325126],[2.477532],[0.059941],[-9.644074],[7.867176],[9.715737],[4.418062],[-2.658135],[-2.511253],[3.224367],[7.231815],[-1.067457],[2.893822],[-4.169356],[0.763805],[2.622738],[6.729167],[3.121985],[-7.059776],[3.836276],[8.713223],[0.797316],[1.407955],[-4.368497],[8.337175],[3.305136],[5.083116],[9.616689],[-6.406011],[8.826836],[8.794569],[7.646711],[1.462577],[9.748523],[0.893188],[5.095519],[5.805480],[6.897880],[-2.313659],[-8.232377],[7.630168],[-3.712097],[-9.599219],[8.298502],[2.868179],[-6.051887],[7.810597],[-0.478924],[-7.265898],[3.938910],[-6.002393],[0.878353],[2.008417],[-1.319326],[-9.621747],[-9.237807],[-0.700460],[-7.886336],[-2.669616],[1.521368],[-0.600553],[5.761560],[-0.077495],[-8.650293],[0.113305],[-7.888617],[-2.737947],[-2.760891],[-7.519943],[2.371324],[5.685693],[-5.193073],[-8.745752],[-7.789180],[6.007821],[4.024316],[5.121970],[9.549717],[-6.318887],[1.525381],[8.930517],[3.437910],[1.923435],[-7.014887],[-8.359206],[-6.582712],[2.198818],[7.642120],[-1.336198],[-7.769108],[2.367075],[6.886787],[-8.185090],[-5.497487],[4.189310],[7.145894],[5.057042],[-1.495338],[-4.975248],[-4.532583],[6.130665],[-5.017745],[-8.251355],[-1.362365],[-5.550187],[-7.382733],[-5.761161],[-8.576439],[-0.423530],[4.600524],[-2.884325],[6.469365],[0.420977],[-8.388299],[-3.180531],[-7.627113],[-9.109221],[-7.063587],[-5.313103],[7.124993],[0.243905],[2.900988],[3.069002],[6.469336],[-5.296550],[1.632808],[8.422239],[0.806529],[2.315637],[-3.360950],[-7.860612],[2.581165],[2.209760],[4.368355],[-8.411959],[-5.943986],[-9.575721],[-7.372723],[-0.816022],[-0.895004],[-9.080764],[2.629893],[-1.124629],[6.698965],[3.852852],[0.208473],[-5.699750],[-1.521778],[-3.790449],[-5.210060],[-1.518555],[-2.858615],[-4.300491],[-1.059553],[-8.688613],[3.434668],[7.914448],[-2.460564],[4.794946],[2.134697],[3.260132],[8.961372],[4.799558],[-9.023245],[2.716651],[-5.752133],[5.947473],[5.154204],[3.342151],[-4.751666],[-2.590054],[0.523735],[9.182174],[7.886218],[1.660758],[7.484791],[3.723207],[-1.746807],[-1.439279],[-6.762115],[4.677628],[1.967803],[5.344931],[-7.947144],[-2.667588],[0.691234],[5.851068],[-2.339427],[-4.500187],[3.916823],[-9.524733],[8.189554],[8.208020],[-7.960631],[-5.394356],[-6.555110],[-0.899594],[-7.114551],[-6.564487],[-0.799804],[-8.790082],[1.467927],[-5.370726],[-3.913904],[8.952872],[-2.766289],[-6.959374],[4.939319],[-5.438596],[9.148679],[-4.343215],[-6.735182],[0.723139],[-1.251296],[-2.591848],[-9.178854],[-0.314400],[-4.632064],[-9.792393],[-5.431340],[-8.191263],[7.257066],[-7.312101],[4.688748],[7.357371],[4.281887],[-6.894961],[-2.811153],[-2.979352],[8.143456],[0.754871],[7.502587],[7.765512],[5.725247],[-7.295096],[8.806091],[-0.970378],[5.216423],[7.022616],[6.497989],[5.000035],[-1.124796],[9.315256],[4.010498],[1.516987],[-0.249172],[2.107105],[0.382313],[4.009892],[-1.552865],[-4.948109],[-4.105935],[-9.583929],[6.066270],[-4.707989],[-8.700707],[1.596270],[8.170865],[9.469742],[-5.037690],[5.148521],[-1.617884],[6.460886],[7.619422],[-1.526954],[-7.898984],[5.036688],[5.455157],[3.083099],[2.984617],[-5.534857],[6.834892],[-0.445857],[-4.625042],[-1.940057],[5.627422],[7.266825],[-0.737016],[1.758633],[5.861550],[9.956861],[-3.485952],[-0.925473],[-9.656300],[-3.157328],[-8.927892],[-1.628649],[-2.667894],[-8.415857],[-3.283557],[8.372599],[0.194805],[1.942386],[6.972075],[0.004853],[-4.135642],[4.230179],[-4.247397],[0.521461],[0.412259],[-1.424707],[-3.838714],[9.275227],[-5.525262],[3.130760],[-5.572637],[-2.359550],[6.950457],[6.694220],[-9.392452],[3.326173],[0.767131],[0.406149],[-3.933602],[2.546747],[-7.561441],[5.698971],[-0.949100],[8.853584],[6.709578],[1.397058],[0.053213],[5.832160],[-3.386490],[-1.164723],[7.424582],[0.407175],[-9.671603],[-8.092788],[8.077733],[-7.292179],[-9.490780],[-1.740076],[-4.840119],[4.019357],[-6.328056],[-6.069934],[-5.476167],[-8.869040],[-5.901978],[2.671721],[-6.320898],[6.714376],[5.481646],[-1.172177],[2.489155],[-3.021617],[-6.565931],[-6.692519],[-4.134061],[-6.277359],[-8.506671],[-6.140092],[-1.739388],[7.164238],[1.197314],[7.626990],[-9.223000],[-6.052522],[-3.419454],[-9.251231],[5.355103],[8.910065],[-1.590722],[8.033723],[6.829072],[7.956958],[-0.373396],[-4.566965],[6.060971],[7.199638],[-2.520753],[-0.008838],[2.761417],[-9.721931],[2.934291],[3.556122],[3.084752],[9.407832],[-3.855097],[-1.482330],[-7.614680],[-8.684001],[-9.586483],[-0.763664],[-7.683676],[-9.332071],[-6.917180],[9.710643],[-5.858981],[9.394386],[8.278020],[-0.592507],[-5.717437],[-7.536847],[-8.049906],[-2.469462],[-6.729147],[4.721629],[-1.617541],[-2.790227],[-9.511047],[6.269883],[-1.655418],[-7.246129],[5.347653],[-6.864679],[8.168563],[0.471388],[-8.758905],[-7.633579],[9.751845],[-8.554444],[1.031691],[-1.475402],[8.587876],[6.937834],[1.667691],[-0.946474],[4.092577],[-8.573654],[-4.968868],[0.018962],[-1.253787],[9.482285],[-8.182245],[7.136827],[2.087422],[-4.187337],[-8.822589],[-3.222208],[9.501422],[6.336178],[-4.604211],[-7.107521],[-2.574580]], dtype = "float32")#candidate|1016|(468, 1)|const|float32
call_1014 = relay.TupleGetItem(func_578_call(relay.reshape(var_1015.astype('int16'), []), relay.reshape(const_1016.astype('float32'), [3, 156]), ), 0)
call_1017 = relay.TupleGetItem(func_582_call(relay.reshape(var_1015.astype('int16'), []), relay.reshape(const_1016.astype('float32'), [3, 156]), ), 0)
output = relay.Tuple([call_1004,call_1014,var_1015,const_1016,])
output2 = relay.Tuple([call_1005,call_1017,var_1015,const_1016,])
func_1033 = relay.Function([var_1015,], output)
mod['func_1033'] = func_1033
mod = relay.transform.InferType()(mod)
mutated_mod['func_1033'] = func_1033
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1034 = relay.var("var_1034", dtype = "int16", shape = ())#candidate|1034|()|var|int16
func_1033_call = mutated_mod.get_global_var('func_1033')
call_1035 = func_1033_call(var_1034)
output = call_1035
func_1036 = relay.Function([var_1034], output)
mutated_mod['func_1036'] = func_1036
mutated_mod = relay.transform.InferType()(mutated_mod)
func_926_call = mod.get_global_var('func_926')
func_928_call = mutated_mod.get_global_var('func_928')
call_1140 = func_926_call()
call_1141 = func_926_call()
func_78_call = mod.get_global_var('func_78')
func_84_call = mutated_mod.get_global_var('func_84')
var_1146 = relay.var("var_1146", dtype = "float32", shape = (12,))#candidate|1146|(12,)|var|float32
call_1145 = relay.TupleGetItem(func_78_call(relay.reshape(var_1146.astype('float32'), [12,]), relay.reshape(var_1146.astype('bool'), [12,]), relay.reshape(var_1146.astype('bool'), [12,]), relay.reshape(var_1146.astype('float32'), [12,]), ), 3)
call_1147 = relay.TupleGetItem(func_84_call(relay.reshape(var_1146.astype('float32'), [12,]), relay.reshape(var_1146.astype('bool'), [12,]), relay.reshape(var_1146.astype('bool'), [12,]), relay.reshape(var_1146.astype('float32'), [12,]), ), 3)
output = relay.Tuple([call_1140,call_1145,var_1146,])
output2 = relay.Tuple([call_1141,call_1147,var_1146,])
func_1150 = relay.Function([var_1146,], output)
mod['func_1150'] = func_1150
mod = relay.transform.InferType()(mod)
mutated_mod['func_1150'] = func_1150
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1151 = relay.var("var_1151", dtype = "float32", shape = (12,))#candidate|1151|(12,)|var|float32
func_1150_call = mutated_mod.get_global_var('func_1150')
call_1152 = func_1150_call(var_1151)
output = call_1152
func_1153 = relay.Function([var_1151], output)
mutated_mod['func_1153'] = func_1153
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1175 = relay.var("var_1175", dtype = "int64", shape = (14, 10, 16))#candidate|1175|(14, 10, 16)|var|int64
var_1176 = relay.var("var_1176", dtype = "int64", shape = (14, 10, 16))#candidate|1176|(14, 10, 16)|var|int64
bop_1177 = relay.subtract(var_1175.astype('int64'), relay.reshape(var_1176.astype('int64'), relay.shape_of(var_1175))) # shape=(14, 10, 16)
bop_1183 = relay.less_equal(bop_1177.astype('bool'), relay.reshape(var_1176.astype('bool'), relay.shape_of(bop_1177))) # shape=(14, 10, 16)
func_541_call = mod.get_global_var('func_541')
func_545_call = mutated_mod.get_global_var('func_545')
var_1188 = relay.var("var_1188", dtype = "float64", shape = (8, 20))#candidate|1188|(8, 20)|var|float64
const_1189 = relay.const([9.173581,6.764240,2.526983,-9.627802,9.824366,6.544161,-8.985605,-8.254718,0.502748,-8.369865,-2.120296,-2.367013], dtype = "float32")#candidate|1189|(12,)|const|float32
call_1187 = relay.TupleGetItem(func_541_call(relay.reshape(var_1188.astype('float64'), [1, 16, 10]), relay.reshape(const_1189.astype('float32'), [1, 12]), ), 2)
call_1190 = relay.TupleGetItem(func_545_call(relay.reshape(var_1188.astype('float64'), [1, 16, 10]), relay.reshape(const_1189.astype('float32'), [1, 12]), ), 2)
output = relay.Tuple([bop_1183,call_1187,var_1188,const_1189,])
output2 = relay.Tuple([bop_1183,call_1190,var_1188,const_1189,])
func_1198 = relay.Function([var_1175,var_1176,var_1188,], output)
mod['func_1198'] = func_1198
mod = relay.transform.InferType()(mod)
mutated_mod['func_1198'] = func_1198
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1198_call = mutated_mod.get_global_var('func_1198')
var_1200 = relay.var("var_1200", dtype = "int64", shape = (14, 10, 16))#candidate|1200|(14, 10, 16)|var|int64
var_1201 = relay.var("var_1201", dtype = "int64", shape = (14, 10, 16))#candidate|1201|(14, 10, 16)|var|int64
var_1202 = relay.var("var_1202", dtype = "float64", shape = (8, 20))#candidate|1202|(8, 20)|var|float64
call_1199 = func_1198_call(var_1200,var_1201,var_1202,)
output = call_1199
func_1203 = relay.Function([var_1200,var_1201,var_1202,], output)
mutated_mod['func_1203'] = func_1203
mutated_mod = relay.transform.InferType()(mutated_mod)
func_926_call = mod.get_global_var('func_926')
func_928_call = mutated_mod.get_global_var('func_928')
call_1207 = func_926_call()
call_1208 = func_926_call()
uop_1230 = relay.log(call_1207.astype('float32')) # shape=(10,)
uop_1232 = relay.log(call_1208.astype('float32')) # shape=(10,)
uop_1233 = relay.acos(uop_1230.astype('float32')) # shape=(10,)
uop_1235 = relay.acos(uop_1232.astype('float32')) # shape=(10,)
uop_1236 = relay.asin(uop_1230.astype('float64')) # shape=(10,)
uop_1238 = relay.asin(uop_1232.astype('float64')) # shape=(10,)
bop_1241 = relay.not_equal(uop_1230.astype('bool'), relay.reshape(uop_1236.astype('bool'), relay.shape_of(uop_1230))) # shape=(10,)
bop_1244 = relay.not_equal(uop_1232.astype('bool'), relay.reshape(uop_1238.astype('bool'), relay.shape_of(uop_1232))) # shape=(10,)
bop_1246 = relay.less(uop_1233.astype('bool'), relay.reshape(uop_1230.astype('bool'), relay.shape_of(uop_1233))) # shape=(10,)
bop_1249 = relay.less(uop_1235.astype('bool'), relay.reshape(uop_1232.astype('bool'), relay.shape_of(uop_1235))) # shape=(10,)
output = relay.Tuple([bop_1241,bop_1246,])
output2 = relay.Tuple([bop_1244,bop_1249,])
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

'''21: TVMFuncCall
20: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
19: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
18: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
17: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
16: tvm::transform::Pass::operator()(tvm::IRModule) const
15: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
14: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
13: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
12: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
11: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
10: tvm::relay::DynamicToStatic(tvm::relay::Function, tvm::IRModule)
9: tvm::relay::DynamicToStaticMutator::PrepareInput(tvm::RelayExpr const&)
8: tvm::transform::Pass::operator()(tvm::IRModule) const
7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
6: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
5: tvm::transform::Pass::operator()(tvm::IRModule) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}::operator()(tvm::IRModule, tvm::transform::PassContext const&) const [clone .isra.813]
0: tvm::DiagnosticContext::Render()

'''