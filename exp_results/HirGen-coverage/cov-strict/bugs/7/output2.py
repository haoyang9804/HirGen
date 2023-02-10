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
const_3 = relay.const([-5.863715], dtype = "float32")#candidate|3|(1,)|const|float32
var_4 = relay.var("var_4", dtype = "float32", shape = (12,))#candidate|4|(12,)|var|float32
bop_5 = relay.floor_mod(const_3.astype('float32'), var_4.astype('float32')) # shape=(12,)
bop_12 = relay.multiply(bop_5.astype('float64'), relay.reshape(var_4.astype('float64'), relay.shape_of(bop_5))) # shape=(12,)
bop_15 = relay.equal(var_4.astype('bool'), const_3.astype('bool')) # shape=(12,)
bop_18 = relay.subtract(bop_5.astype('int16'), relay.reshape(var_4.astype('int16'), relay.shape_of(bop_5))) # shape=(12,)
output = relay.Tuple([bop_12,bop_15,bop_18,])
output2 = relay.Tuple([bop_12,bop_15,bop_18,])
func_28 = relay.Function([var_4,], output)
mod['func_28'] = func_28
mod = relay.transform.InferType()(mod)
var_29 = relay.var("var_29", dtype = "float32", shape = (12,))#candidate|29|(12,)|var|float32
output = func_28(var_29)
func_30 = relay.Function([var_29], output)
mutated_mod['func_30'] = func_30
mutated_mod = relay.transform.InferType()(mutated_mod)
var_43 = relay.var("var_43", dtype = "float64", shape = (8, 11))#candidate|43|(8, 11)|var|float64
uop_44 = relay.rsqrt(var_43.astype('float64')) # shape=(8, 11)
uop_49 = relay.acos(uop_44.astype('float32')) # shape=(8, 11)
uop_52 = relay.sqrt(uop_49.astype('float32')) # shape=(8, 11)
uop_54 = relay.atan(uop_49.astype('float64')) # shape=(8, 11)
var_56 = relay.var("var_56", dtype = "float64", shape = (8, 11))#candidate|56|(8, 11)|var|float64
bop_57 = relay.multiply(uop_54.astype('int64'), relay.reshape(var_56.astype('int64'), relay.shape_of(uop_54))) # shape=(8, 11)
func_28_call = mod.get_global_var('func_28')
func_30_call = mutated_mod.get_global_var('func_30')
const_63 = relay.const([8.926573,7.517056,-1.926500,-1.970334,-4.507979,-6.288672,-2.020830,8.721674,2.544907,0.812955,6.338131,-0.745906], dtype = "float32")#candidate|63|(12,)|const|float32
call_62 = relay.TupleGetItem(func_28_call(relay.reshape(const_63.astype('float32'), [12,])), 0)
call_64 = relay.TupleGetItem(func_30_call(relay.reshape(const_63.astype('float32'), [12,])), 0)
func_28_call = mod.get_global_var('func_28')
func_30_call = mutated_mod.get_global_var('func_30')
call_67 = relay.TupleGetItem(func_28_call(relay.reshape(const_63.astype('float32'), [12,])), 2)
call_68 = relay.TupleGetItem(func_30_call(relay.reshape(const_63.astype('float32'), [12,])), 2)
uop_69 = relay.asin(bop_57.astype('float32')) # shape=(8, 11)
bop_71 = relay.mod(uop_54.astype('float32'), relay.reshape(bop_57.astype('float32'), relay.shape_of(uop_54))) # shape=(8, 11)
bop_76 = relay.add(uop_69.astype('float32'), relay.reshape(bop_71.astype('float32'), relay.shape_of(uop_69))) # shape=(8, 11)
bop_79 = relay.logical_xor(bop_76.astype('int32'), relay.reshape(uop_52.astype('int32'), relay.shape_of(bop_76))) # shape=(8, 11)
uop_85 = relay.erf(uop_54.astype('float64')) # shape=(8, 11)
bop_88 = relay.left_shift(bop_76.astype('uint16'), relay.reshape(uop_69.astype('uint16'), relay.shape_of(bop_76))) # shape=(8, 11)
bop_95 = relay.power(bop_88.astype('float32'), relay.reshape(uop_49.astype('float32'), relay.shape_of(bop_88))) # shape=(8, 11)
var_99 = relay.var("var_99", dtype = "float32", shape = (8, 11))#candidate|99|(8, 11)|var|float32
bop_100 = relay.equal(uop_69.astype('bool'), relay.reshape(var_99.astype('bool'), relay.shape_of(uop_69))) # shape=(8, 11)
bop_107 = relay.greater(bop_95.astype('bool'), relay.reshape(bop_79.astype('bool'), relay.shape_of(bop_95))) # shape=(8, 11)
uop_111 = relay.cos(bop_100.astype('float64')) # shape=(8, 11)
uop_119 = relay.exp(uop_54.astype('float64')) # shape=(8, 11)
func_28_call = mod.get_global_var('func_28')
func_30_call = mutated_mod.get_global_var('func_30')
call_124 = relay.TupleGetItem(func_28_call(relay.reshape(call_62.astype('float32'), [12,])), 2)
call_125 = relay.TupleGetItem(func_30_call(relay.reshape(call_62.astype('float32'), [12,])), 2)
var_126 = relay.var("var_126", dtype = "float64", shape = (8, 11))#candidate|126|(8, 11)|var|float64
bop_127 = relay.less_equal(uop_111.astype('bool'), relay.reshape(var_126.astype('bool'), relay.shape_of(uop_111))) # shape=(8, 11)
bop_130 = relay.mod(bop_95.astype('float64'), relay.reshape(bop_57.astype('float64'), relay.shape_of(bop_95))) # shape=(8, 11)
output = relay.Tuple([call_62,const_63,call_67,uop_85,bop_107,uop_119,call_124,bop_127,bop_130,])
output2 = relay.Tuple([call_64,const_63,call_68,uop_85,bop_107,uop_119,call_125,bop_127,bop_130,])
func_133 = relay.Function([var_43,var_56,var_99,var_126,], output)
mod['func_133'] = func_133
mod = relay.transform.InferType()(mod)
var_134 = relay.var("var_134", dtype = "float64", shape = (8, 11))#candidate|134|(8, 11)|var|float64
var_135 = relay.var("var_135", dtype = "float64", shape = (8, 11))#candidate|135|(8, 11)|var|float64
var_136 = relay.var("var_136", dtype = "float32", shape = (8, 11))#candidate|136|(8, 11)|var|float32
var_137 = relay.var("var_137", dtype = "float64", shape = (8, 11))#candidate|137|(8, 11)|var|float64
output = func_133(var_134,var_135,var_136,var_137,)
func_138 = relay.Function([var_134,var_135,var_136,var_137,], output)
mutated_mod['func_138'] = func_138
mutated_mod = relay.transform.InferType()(mutated_mod)
var_156 = relay.var("var_156", dtype = "float32", shape = (5,))#candidate|156|(5,)|var|float32
uop_157 = relay.exp(var_156.astype('float32')) # shape=(5,)
bop_161 = relay.logical_or(uop_157.astype('bool'), relay.reshape(var_156.astype('bool'), relay.shape_of(uop_157))) # shape=(5,)
uop_164 = relay.sqrt(bop_161.astype('float32')) # shape=(5,)
func_28_call = mod.get_global_var('func_28')
func_30_call = mutated_mod.get_global_var('func_30')
const_168 = relay.const([[-1.505914,4.053923,-1.965735,-6.674029,0.707059,7.843002,-6.023188,7.989759,4.146424,-0.550399,-2.671860,-6.295549]], dtype = "float32")#candidate|168|(1, 12)|const|float32
call_167 = relay.TupleGetItem(func_28_call(relay.reshape(const_168.astype('float32'), [12,])), 2)
call_169 = relay.TupleGetItem(func_30_call(relay.reshape(const_168.astype('float32'), [12,])), 2)
uop_170 = relay.log10(bop_161.astype('float64')) # shape=(5,)
var_172 = relay.var("var_172", dtype = "bool", shape = (5,))#candidate|172|(5,)|var|bool
bop_173 = relay.bitwise_or(bop_161.astype('uint32'), relay.reshape(var_172.astype('uint32'), relay.shape_of(bop_161))) # shape=(5,)
bop_176 = relay.bitwise_or(uop_164.astype('uint8'), relay.reshape(bop_173.astype('uint8'), relay.shape_of(uop_164))) # shape=(5,)
uop_182 = relay.log(uop_170.astype('float64')) # shape=(5,)
var_185 = relay.var("var_185", dtype = "float64", shape = (5,))#candidate|185|(5,)|var|float64
bop_186 = relay.subtract(uop_182.astype('float64'), relay.reshape(var_185.astype('float64'), relay.shape_of(uop_182))) # shape=(5,)
var_190 = relay.var("var_190", dtype = "bool", shape = (5,))#candidate|190|(5,)|var|bool
bop_191 = relay.greater_equal(bop_161.astype('bool'), relay.reshape(var_190.astype('bool'), relay.shape_of(bop_161))) # shape=(5,)
output = relay.Tuple([call_167,const_168,bop_176,bop_186,bop_191,])
output2 = relay.Tuple([call_169,const_168,bop_176,bop_186,bop_191,])
func_194 = relay.Function([var_156,var_172,var_185,var_190,], output)
mod['func_194'] = func_194
mod = relay.transform.InferType()(mod)
mutated_mod['func_194'] = func_194
mutated_mod = relay.transform.InferType()(mutated_mod)
func_194_call = mutated_mod.get_global_var('func_194')
var_196 = relay.var("var_196", dtype = "float32", shape = (5,))#candidate|196|(5,)|var|float32
var_197 = relay.var("var_197", dtype = "bool", shape = (5,))#candidate|197|(5,)|var|bool
var_198 = relay.var("var_198", dtype = "float64", shape = (5,))#candidate|198|(5,)|var|float64
var_199 = relay.var("var_199", dtype = "bool", shape = (5,))#candidate|199|(5,)|var|bool
call_195 = func_194_call(var_196,var_197,var_198,var_199,)
output = call_195
func_200 = relay.Function([var_196,var_197,var_198,var_199,], output)
mutated_mod['func_200'] = func_200
mutated_mod = relay.transform.InferType()(mutated_mod)
var_222 = relay.var("var_222", dtype = "float64", shape = (6, 1))#candidate|222|(6, 1)|var|float64
uop_223 = relay.sinh(var_222.astype('float64')) # shape=(6, 1)
uop_226 = relay.sin(uop_223.astype('float64')) # shape=(6, 1)
output = relay.Tuple([uop_226,])
output2 = relay.Tuple([uop_226,])
func_228 = relay.Function([var_222,], output)
mod['func_228'] = func_228
mod = relay.transform.InferType()(mod)
mutated_mod['func_228'] = func_228
mutated_mod = relay.transform.InferType()(mutated_mod)
var_229 = relay.var("var_229", dtype = "float64", shape = (6, 1))#candidate|229|(6, 1)|var|float64
func_228_call = mutated_mod.get_global_var('func_228')
call_230 = func_228_call(var_229)
output = call_230
func_231 = relay.Function([var_229], output)
mutated_mod['func_231'] = func_231
mutated_mod = relay.transform.InferType()(mutated_mod)
var_246 = relay.var("var_246", dtype = "float64", shape = (12, 15))#candidate|246|(12, 15)|var|float64
uop_247 = relay.log(var_246.astype('float64')) # shape=(12, 15)
var_250 = relay.var("var_250", dtype = "float64", shape = (12, 15))#candidate|250|(12, 15)|var|float64
bop_251 = relay.logical_and(uop_247.astype('bool'), relay.reshape(var_250.astype('bool'), relay.shape_of(uop_247))) # shape=(12, 15)
bop_256 = relay.not_equal(uop_247.astype('bool'), relay.reshape(var_246.astype('bool'), relay.shape_of(uop_247))) # shape=(12, 15)
uop_266 = relay.asin(bop_251.astype('float32')) # shape=(12, 15)
bop_270 = relay.bitwise_or(uop_266.astype('uint8'), relay.reshape(uop_247.astype('uint8'), relay.shape_of(uop_266))) # shape=(12, 15)
var_273 = relay.var("var_273", dtype = "uint8", shape = (12, 15))#candidate|273|(12, 15)|var|uint8
bop_274 = relay.maximum(bop_270.astype('int8'), relay.reshape(var_273.astype('int8'), relay.shape_of(bop_270))) # shape=(12, 15)
func_194_call = mod.get_global_var('func_194')
func_200_call = mutated_mod.get_global_var('func_200')
const_281 = relay.const([-3.470594,0.476808,-1.783147,6.703906,7.769650], dtype = "float32")#candidate|281|(5,)|const|float32
call_280 = relay.TupleGetItem(func_194_call(relay.reshape(const_281.astype('float32'), [5,]), relay.reshape(const_281.astype('bool'), [5,]), relay.reshape(const_281.astype('float64'), [5,]), relay.reshape(const_281.astype('bool'), [5,]), ), 4)
call_282 = relay.TupleGetItem(func_200_call(relay.reshape(const_281.astype('float32'), [5,]), relay.reshape(const_281.astype('bool'), [5,]), relay.reshape(const_281.astype('float64'), [5,]), relay.reshape(const_281.astype('bool'), [5,]), ), 4)
bop_284 = relay.logical_xor(uop_266.astype('uint64'), relay.reshape(bop_256.astype('uint64'), relay.shape_of(uop_266))) # shape=(12, 15)
bop_287 = relay.multiply(bop_270.astype('uint16'), relay.reshape(bop_274.astype('uint16'), relay.shape_of(bop_270))) # shape=(12, 15)
const_291 = relay.const([[-5,-5,-6,1,5,9,-2,7,2,2,6,-9,6,-4,-2],[-3,3,-1,6,-10,-6,7,1,-10,3,-1,9,-1,-7,-8],[-9,5,8,-4,8,3,10,-3,-5,-6,-8,-7,-6,-2,3],[-4,-10,6,-7,-2,2,-10,-9,-7,-10,3,9,-7,-6,8],[10,2,10,-9,8,-1,-9,5,6,10,-7,5,3,-1,-10],[6,2,9,-8,1,10,-8,-5,-6,-7,8,-1,9,3,-1],[-2,3,9,-8,-2,4,-2,-10,-5,9,-5,7,-10,-10,-10],[-7,1,6,2,8,-4,-6,3,3,-4,-4,-9,-9,-9,3],[-1,2,10,4,4,8,2,-4,1,5,-10,-8,-9,-10,-5],[8,3,10,9,-7,-5,1,-10,4,-4,9,-9,-5,-4,-9],[-7,-7,-6,-6,-2,-6,-10,8,-3,1,3,-1,9,-1,10],[1,6,2,8,4,2,9,-10,2,-9,-1,5,5,8,6]], dtype = "int8")#candidate|291|(12, 15)|const|int8
bop_292 = relay.bitwise_and(bop_274.astype('int64'), relay.reshape(const_291.astype('int64'), relay.shape_of(bop_274))) # shape=(12, 15)
uop_296 = relay.sqrt(bop_284.astype('float64')) # shape=(12, 15)
uop_299 = relay.log2(uop_296.astype('float64')) # shape=(12, 15)
bop_302 = relay.minimum(bop_287.astype('float64'), relay.reshape(var_246.astype('float64'), relay.shape_of(bop_287))) # shape=(12, 15)
uop_311 = relay.asinh(uop_299.astype('float64')) # shape=(12, 15)
var_313 = relay.var("var_313", dtype = "float64", shape = (12, 15))#candidate|313|(12, 15)|var|float64
bop_314 = relay.right_shift(uop_299.astype('uint8'), relay.reshape(var_313.astype('uint8'), relay.shape_of(uop_299))) # shape=(12, 15)
uop_319 = relay.atan(uop_311.astype('float64')) # shape=(12, 15)
bop_321 = relay.equal(uop_319.astype('bool'), relay.reshape(bop_284.astype('bool'), relay.shape_of(uop_319))) # shape=(12, 15)
func_194_call = mod.get_global_var('func_194')
func_200_call = mutated_mod.get_global_var('func_200')
call_325 = relay.TupleGetItem(func_194_call(relay.reshape(call_280.astype('float32'), [5,]), relay.reshape(const_281.astype('bool'), [5,]), relay.reshape(call_280.astype('float64'), [5,]), relay.reshape(call_280.astype('bool'), [5,]), ), 0)
call_326 = relay.TupleGetItem(func_200_call(relay.reshape(call_280.astype('float32'), [5,]), relay.reshape(const_281.astype('bool'), [5,]), relay.reshape(call_280.astype('float64'), [5,]), relay.reshape(call_280.astype('bool'), [5,]), ), 0)
output = relay.Tuple([call_280,const_281,bop_292,bop_302,bop_314,bop_321,call_325,])
output2 = relay.Tuple([call_282,const_281,bop_292,bop_302,bop_314,bop_321,call_326,])
func_327 = relay.Function([var_246,var_250,var_273,var_313,], output)
mod['func_327'] = func_327
mod = relay.transform.InferType()(mod)
var_328 = relay.var("var_328", dtype = "float64", shape = (12, 15))#candidate|328|(12, 15)|var|float64
var_329 = relay.var("var_329", dtype = "float64", shape = (12, 15))#candidate|329|(12, 15)|var|float64
var_330 = relay.var("var_330", dtype = "uint8", shape = (12, 15))#candidate|330|(12, 15)|var|uint8
var_331 = relay.var("var_331", dtype = "float64", shape = (12, 15))#candidate|331|(12, 15)|var|float64
output = func_327(var_328,var_329,var_330,var_331,)
func_332 = relay.Function([var_328,var_329,var_330,var_331,], output)
mutated_mod['func_332'] = func_332
mutated_mod = relay.transform.InferType()(mutated_mod)
const_371 = relay.const([[-7.727543,-9.989615,-8.786451,-1.477365,-7.458135,-0.244869,-7.095868,6.828654]], dtype = "float64")#candidate|371|(1, 8)|const|float64
var_372 = relay.var("var_372", dtype = "float64", shape = (10, 8))#candidate|372|(10, 8)|var|float64
bop_373 = relay.mod(const_371.astype('float64'), var_372.astype('float64')) # shape=(10, 8)
bop_376 = relay.divide(const_371.astype('float64'), bop_373.astype('float64')) # shape=(10, 8)
func_228_call = mod.get_global_var('func_228')
func_231_call = mutated_mod.get_global_var('func_231')
const_383 = relay.const([-3.011409,8.593108,5.773695,-0.404672,3.861413,9.061531], dtype = "float64")#candidate|383|(6,)|const|float64
call_382 = relay.TupleGetItem(func_228_call(relay.reshape(const_383.astype('float64'), [6, 1])), 0)
call_384 = relay.TupleGetItem(func_231_call(relay.reshape(const_383.astype('float64'), [6, 1])), 0)
uop_385 = relay.sinh(bop_376.astype('float32')) # shape=(10, 8)
uop_388 = relay.log10(var_372.astype('float32')) # shape=(10, 8)
bop_390 = relay.not_equal(uop_388.astype('bool'), relay.reshape(uop_385.astype('bool'), relay.shape_of(uop_388))) # shape=(10, 8)
output = relay.Tuple([call_382,const_383,bop_390,])
output2 = relay.Tuple([call_384,const_383,bop_390,])
func_393 = relay.Function([var_372,], output)
mod['func_393'] = func_393
mod = relay.transform.InferType()(mod)
mutated_mod['func_393'] = func_393
mutated_mod = relay.transform.InferType()(mutated_mod)
var_394 = relay.var("var_394", dtype = "float64", shape = (10, 8))#candidate|394|(10, 8)|var|float64
func_393_call = mutated_mod.get_global_var('func_393')
call_395 = func_393_call(var_394)
output = call_395
func_396 = relay.Function([var_394], output)
mutated_mod['func_396'] = func_396
mutated_mod = relay.transform.InferType()(mutated_mod)
var_398 = relay.var("var_398", dtype = "float32", shape = (11, 3))#candidate|398|(11, 3)|var|float32
const_399 = relay.const([[5.489292,1.976830,8.572090],[3.375355,9.185549,6.405369],[-7.754896,7.727583,9.948196],[-3.446056,-2.873522,-8.197074],[5.442368,9.610296,-3.068880],[-5.247899,8.019648,-3.335413],[-5.848647,0.635147,7.920537],[-2.991995,-8.013710,0.116619],[-8.132101,-7.913308,4.943193],[0.406455,-7.524314,5.264278],[3.389798,-2.543989,7.072802]], dtype = "float32")#candidate|399|(11, 3)|const|float32
bop_400 = relay.floor_divide(var_398.astype('float32'), relay.reshape(const_399.astype('float32'), relay.shape_of(var_398))) # shape=(11, 3)
output = relay.Tuple([bop_400,])
output2 = relay.Tuple([bop_400,])
func_403 = relay.Function([var_398,], output)
mod['func_403'] = func_403
mod = relay.transform.InferType()(mod)
var_404 = relay.var("var_404", dtype = "float32", shape = (11, 3))#candidate|404|(11, 3)|var|float32
output = func_403(var_404)
func_405 = relay.Function([var_404], output)
mutated_mod['func_405'] = func_405
mutated_mod = relay.transform.InferType()(mutated_mod)
var_414 = relay.var("var_414", dtype = "float32", shape = (4,))#candidate|414|(4,)|var|float32
uop_415 = relay.cosh(var_414.astype('float32')) # shape=(4,)
bop_418 = relay.power(var_414.astype('float32'), relay.reshape(uop_415.astype('float32'), relay.shape_of(var_414))) # shape=(4,)
uop_421 = relay.sqrt(uop_415.astype('float32')) # shape=(4,)
const_425 = relay.const([-8.242732,-3.579042,-2.177921,-1.080875], dtype = "float32")#candidate|425|(4,)|const|float32
bop_426 = relay.bitwise_and(uop_421.astype('uint64'), relay.reshape(const_425.astype('uint64'), relay.shape_of(uop_421))) # shape=(4,)
uop_429 = relay.erf(bop_418.astype('float32')) # shape=(4,)
uop_436 = relay.sinh(uop_421.astype('float64')) # shape=(4,)
var_439 = relay.var("var_439", dtype = "float32", shape = (4,))#candidate|439|(4,)|var|float32
bop_440 = relay.not_equal(bop_418.astype('bool'), relay.reshape(var_439.astype('bool'), relay.shape_of(bop_418))) # shape=(4,)
bop_443 = relay.bitwise_xor(uop_436.astype('int16'), relay.reshape(bop_426.astype('int16'), relay.shape_of(uop_436))) # shape=(4,)
bop_450 = relay.not_equal(uop_421.astype('bool'), relay.reshape(uop_436.astype('bool'), relay.shape_of(uop_421))) # shape=(4,)
uop_462 = relay.sigmoid(bop_418.astype('float64')) # shape=(4,)
bop_465 = relay.power(uop_421.astype('float32'), relay.reshape(bop_443.astype('float32'), relay.shape_of(uop_421))) # shape=(4,)
const_468 = relay.const([7,-2,7,-7], dtype = "uint64")#candidate|468|(4,)|const|uint64
bop_469 = relay.bitwise_and(bop_426.astype('uint16'), relay.reshape(const_468.astype('uint16'), relay.shape_of(bop_426))) # shape=(4,)
bop_472 = relay.maximum(bop_440.astype('uint32'), relay.reshape(bop_450.astype('uint32'), relay.shape_of(bop_440))) # shape=(4,)
bop_475 = relay.multiply(bop_426.astype('float64'), relay.reshape(uop_436.astype('float64'), relay.shape_of(bop_426))) # shape=(4,)
uop_481 = relay.log(bop_450.astype('float32')) # shape=(4,)
bop_485 = relay.power(uop_481.astype('float64'), relay.reshape(bop_475.astype('float64'), relay.shape_of(uop_481))) # shape=(4,)
const_489 = relay.const([4.696274,2.450548,1.042792,9.396086], dtype = "float32")#candidate|489|(4,)|const|float32
bop_490 = relay.less(uop_415.astype('bool'), relay.reshape(const_489.astype('bool'), relay.shape_of(uop_415))) # shape=(4,)
bop_499 = relay.less(uop_481.astype('bool'), relay.reshape(bop_440.astype('bool'), relay.shape_of(uop_481))) # shape=(4,)
uop_502 = relay.exp(bop_440.astype('float32')) # shape=(4,)
bop_505 = relay.logical_and(uop_436.astype('bool'), relay.reshape(bop_485.astype('bool'), relay.shape_of(uop_436))) # shape=(4,)
bop_511 = relay.multiply(uop_429.astype('int64'), relay.reshape(uop_421.astype('int64'), relay.shape_of(uop_429))) # shape=(4,)
bop_514 = relay.divide(uop_481.astype('float32'), relay.reshape(bop_490.astype('float32'), relay.shape_of(uop_481))) # shape=(4,)
uop_518 = relay.cos(bop_465.astype('float32')) # shape=(4,)
bop_522 = relay.right_shift(bop_499.astype('uint32'), relay.reshape(bop_514.astype('uint32'), relay.shape_of(bop_499))) # shape=(4,)
uop_525 = relay.asinh(bop_485.astype('float32')) # shape=(4,)
bop_527 = relay.floor_divide(uop_518.astype('float64'), relay.reshape(bop_514.astype('float64'), relay.shape_of(uop_518))) # shape=(4,)
func_228_call = mod.get_global_var('func_228')
func_231_call = mutated_mod.get_global_var('func_231')
var_531 = relay.var("var_531", dtype = "float64", shape = (6,))#candidate|531|(6,)|var|float64
call_530 = relay.TupleGetItem(func_228_call(relay.reshape(var_531.astype('float64'), [6, 1])), 0)
call_532 = relay.TupleGetItem(func_231_call(relay.reshape(var_531.astype('float64'), [6, 1])), 0)
bop_540 = relay.mod(uop_525.astype('float64'), relay.reshape(uop_421.astype('float64'), relay.shape_of(uop_525))) # shape=(4,)
func_28_call = mod.get_global_var('func_28')
func_30_call = mutated_mod.get_global_var('func_30')
var_544 = relay.var("var_544", dtype = "float32", shape = (12,))#candidate|544|(12,)|var|float32
call_543 = relay.TupleGetItem(func_28_call(relay.reshape(var_544.astype('float32'), [12,])), 0)
call_545 = relay.TupleGetItem(func_30_call(relay.reshape(var_544.astype('float32'), [12,])), 0)
bop_547 = relay.greater(uop_525.astype('bool'), relay.reshape(bop_443.astype('bool'), relay.shape_of(uop_525))) # shape=(4,)
bop_553 = relay.logical_xor(uop_518.astype('uint8'), relay.reshape(bop_490.astype('uint8'), relay.shape_of(uop_518))) # shape=(4,)
bop_556 = relay.logical_xor(bop_540.astype('int16'), relay.reshape(bop_440.astype('int16'), relay.shape_of(bop_540))) # shape=(4,)
func_194_call = mod.get_global_var('func_194')
func_200_call = mutated_mod.get_global_var('func_200')
const_561 = relay.const([8.342119,-3.272851,-4.521359,-3.851113,-4.438513], dtype = "float32")#candidate|561|(5,)|const|float32
call_560 = relay.TupleGetItem(func_194_call(relay.reshape(const_561.astype('float32'), [5,]), relay.reshape(const_561.astype('bool'), [5,]), relay.reshape(const_561.astype('float64'), [5,]), relay.reshape(const_561.astype('bool'), [5,]), ), 3)
call_562 = relay.TupleGetItem(func_200_call(relay.reshape(const_561.astype('float32'), [5,]), relay.reshape(const_561.astype('bool'), [5,]), relay.reshape(const_561.astype('float64'), [5,]), relay.reshape(const_561.astype('bool'), [5,]), ), 3)
uop_563 = relay.atanh(uop_525.astype('float64')) # shape=(4,)
uop_567 = relay.rsqrt(bop_540.astype('float32')) # shape=(4,)
bop_569 = relay.left_shift(uop_563.astype('uint64'), relay.reshape(bop_490.astype('uint64'), relay.shape_of(uop_563))) # shape=(4,)
output = relay.Tuple([uop_462,bop_469,bop_472,uop_502,bop_505,bop_511,bop_522,bop_527,call_530,var_531,call_543,var_544,bop_547,bop_553,bop_556,call_560,const_561,uop_567,bop_569,])
output2 = relay.Tuple([uop_462,bop_469,bop_472,uop_502,bop_505,bop_511,bop_522,bop_527,call_532,var_531,call_545,var_544,bop_547,bop_553,bop_556,call_562,const_561,uop_567,bop_569,])
func_575 = relay.Function([var_414,var_439,var_531,var_544,], output)
mod['func_575'] = func_575
mod = relay.transform.InferType()(mod)
mutated_mod['func_575'] = func_575
mutated_mod = relay.transform.InferType()(mutated_mod)
func_575_call = mutated_mod.get_global_var('func_575')
var_577 = relay.var("var_577", dtype = "float32", shape = (4,))#candidate|577|(4,)|var|float32
var_578 = relay.var("var_578", dtype = "float32", shape = (4,))#candidate|578|(4,)|var|float32
var_579 = relay.var("var_579", dtype = "float64", shape = (6,))#candidate|579|(6,)|var|float64
var_580 = relay.var("var_580", dtype = "float32", shape = (12,))#candidate|580|(12,)|var|float32
call_576 = func_575_call(var_577,var_578,var_579,var_580,)
output = call_576
func_581 = relay.Function([var_577,var_578,var_579,var_580,], output)
mutated_mod['func_581'] = func_581
mutated_mod = relay.transform.InferType()(mutated_mod)
const_600 = relay.const([[5.255663,6.259975,0.346589,-2.796533,7.785224,3.859685],[3.458284,5.293668,1.860170,-8.099496,9.207946,-0.235487],[-8.590466,3.396388,-1.851841,8.461034,-8.972416,-0.236981],[-5.148917,3.062093,-3.269490,8.850947,1.738696,6.051154],[-2.288422,9.053562,-7.906260,-0.213722,-4.188839,-8.743988],[-8.072692,-8.880881,-5.708208,1.445983,4.719712,-2.997486],[4.227912,-6.976314,-0.944034,7.651961,2.484517,2.198715],[-1.126829,-9.338114,9.238753,7.998881,-8.642178,-9.467432],[4.756152,-1.950191,-9.374437,-0.630270,-4.026252,9.939731],[-4.980059,-9.478883,6.657963,-3.820102,2.076259,7.288627],[2.272304,0.261059,-0.413486,-8.572005,9.101924,5.125536],[7.027273,5.018823,-6.540241,9.721833,-6.295157,-6.218852]], dtype = "float32")#candidate|600|(12, 6)|const|float32
var_601 = relay.var("var_601", dtype = "float32", shape = (12, 6))#candidate|601|(12, 6)|var|float32
bop_602 = relay.floor_divide(const_600.astype('float32'), relay.reshape(var_601.astype('float32'), relay.shape_of(const_600))) # shape=(12, 6)
bop_605 = relay.power(bop_602.astype('float64'), relay.reshape(var_601.astype('float64'), relay.shape_of(bop_602))) # shape=(12, 6)
bop_608 = relay.mod(bop_602.astype('float64'), relay.reshape(const_600.astype('float64'), relay.shape_of(bop_602))) # shape=(12, 6)
output = relay.Tuple([bop_605,bop_608,])
output2 = relay.Tuple([bop_605,bop_608,])
func_618 = relay.Function([var_601,], output)
mod['func_618'] = func_618
mod = relay.transform.InferType()(mod)
mutated_mod['func_618'] = func_618
mutated_mod = relay.transform.InferType()(mutated_mod)
var_619 = relay.var("var_619", dtype = "float32", shape = (12, 6))#candidate|619|(12, 6)|var|float32
func_618_call = mutated_mod.get_global_var('func_618')
call_620 = func_618_call(var_619)
output = call_620
func_621 = relay.Function([var_619], output)
mutated_mod['func_621'] = func_621
mutated_mod = relay.transform.InferType()(mutated_mod)
var_635 = relay.var("var_635", dtype = "float64", shape = (2, 5))#candidate|635|(2, 5)|var|float64
var_636 = relay.var("var_636", dtype = "float64", shape = (2, 5))#candidate|636|(2, 5)|var|float64
bop_637 = relay.divide(var_635.astype('float64'), relay.reshape(var_636.astype('float64'), relay.shape_of(var_635))) # shape=(2, 5)
output = relay.Tuple([bop_637,])
output2 = relay.Tuple([bop_637,])
func_641 = relay.Function([var_635,var_636,], output)
mod['func_641'] = func_641
mod = relay.transform.InferType()(mod)
mutated_mod['func_641'] = func_641
mutated_mod = relay.transform.InferType()(mutated_mod)
func_641_call = mutated_mod.get_global_var('func_641')
var_643 = relay.var("var_643", dtype = "float64", shape = (2, 5))#candidate|643|(2, 5)|var|float64
var_644 = relay.var("var_644", dtype = "float64", shape = (2, 5))#candidate|644|(2, 5)|var|float64
call_642 = func_641_call(var_643,var_644,)
output = call_642
func_645 = relay.Function([var_643,var_644,], output)
mutated_mod['func_645'] = func_645
mutated_mod = relay.transform.InferType()(mutated_mod)
const_649 = relay.const([6], dtype = "int64")#candidate|649|(1,)|const|int64
var_650 = relay.var("var_650", dtype = "int64", shape = (11,))#candidate|650|(11,)|var|int64
bop_651 = relay.less_equal(const_649.astype('bool'), var_650.astype('bool')) # shape=(11,)
func_327_call = mod.get_global_var('func_327')
func_332_call = mutated_mod.get_global_var('func_332')
var_656 = relay.var("var_656", dtype = "float64", shape = (180,))#candidate|656|(180,)|var|float64
call_655 = relay.TupleGetItem(func_327_call(relay.reshape(var_656.astype('float64'), [12, 15]), relay.reshape(var_656.astype('float64'), [12, 15]), relay.reshape(var_656.astype('uint8'), [12, 15]), relay.reshape(var_656.astype('float64'), [12, 15]), ), 2)
call_657 = relay.TupleGetItem(func_332_call(relay.reshape(var_656.astype('float64'), [12, 15]), relay.reshape(var_656.astype('float64'), [12, 15]), relay.reshape(var_656.astype('uint8'), [12, 15]), relay.reshape(var_656.astype('float64'), [12, 15]), ), 2)
bop_658 = relay.multiply(var_650.astype('uint8'), relay.reshape(bop_651.astype('uint8'), relay.shape_of(var_650))) # shape=(11,)
output = relay.Tuple([call_655,var_656,bop_658,])
output2 = relay.Tuple([call_657,var_656,bop_658,])
func_661 = relay.Function([var_650,var_656,], output)
mod['func_661'] = func_661
mod = relay.transform.InferType()(mod)
mutated_mod['func_661'] = func_661
mutated_mod = relay.transform.InferType()(mutated_mod)
func_661_call = mutated_mod.get_global_var('func_661')
var_663 = relay.var("var_663", dtype = "int64", shape = (11,))#candidate|663|(11,)|var|int64
var_664 = relay.var("var_664", dtype = "float64", shape = (180,))#candidate|664|(180,)|var|float64
call_662 = func_661_call(var_663,var_664,)
output = call_662
func_665 = relay.Function([var_663,var_664,], output)
mutated_mod['func_665'] = func_665
mutated_mod = relay.transform.InferType()(mutated_mod)
var_685 = relay.var("var_685", dtype = "int8", shape = (9, 2))#candidate|685|(9, 2)|var|int8
var_686 = relay.var("var_686", dtype = "int8", shape = (9, 2))#candidate|686|(9, 2)|var|int8
bop_687 = relay.multiply(var_685.astype('int8'), relay.reshape(var_686.astype('int8'), relay.shape_of(var_685))) # shape=(9, 2)
bop_690 = relay.add(var_685.astype('uint16'), relay.reshape(bop_687.astype('uint16'), relay.shape_of(var_685))) # shape=(9, 2)
output = relay.Tuple([bop_690,])
output2 = relay.Tuple([bop_690,])
func_694 = relay.Function([var_685,var_686,], output)
mod['func_694'] = func_694
mod = relay.transform.InferType()(mod)
mutated_mod['func_694'] = func_694
mutated_mod = relay.transform.InferType()(mutated_mod)
func_694_call = mutated_mod.get_global_var('func_694')
var_696 = relay.var("var_696", dtype = "int8", shape = (9, 2))#candidate|696|(9, 2)|var|int8
var_697 = relay.var("var_697", dtype = "int8", shape = (9, 2))#candidate|697|(9, 2)|var|int8
call_695 = func_694_call(var_696,var_697,)
output = call_695
func_698 = relay.Function([var_696,var_697,], output)
mutated_mod['func_698'] = func_698
mutated_mod = relay.transform.InferType()(mutated_mod)
const_703 = relay.const([[[-1.212403,0.819999,-9.119503,2.066573,-0.560988,-1.067871,-3.508635,-8.549562,-8.105328],[3.195854,0.228069,1.381740,5.874777,-5.948282,-5.368348,1.045145,5.410009,-7.340926],[-2.874247,2.131678,2.141971,9.152197,4.276389,-2.807364,-1.201281,4.248374,7.198672],[7.585441,-3.155459,2.829228,8.154350,5.800728,3.549276,-6.301549,6.802258,-1.606607],[-3.063704,3.674449,-6.169856,2.563347,0.931120,-1.073991,8.542289,5.283224,-2.853434],[8.467392,9.741605,-3.380919,-6.535775,-8.030265,-2.745653,-1.769265,-5.463014,-4.215386],[1.718394,-1.863230,-9.201655,2.027765,-1.449264,5.811395,-5.986188,-0.079537,6.613671],[-0.102784,-1.939525,-6.109219,-9.590988,1.322742,-5.231700,3.217903,2.029128,-9.897972],[-4.263781,4.513012,-4.248128,2.702578,7.088725,-3.280667,-5.980527,-1.005687,9.100809],[7.649630,-8.922142,-3.613854,1.927621,-9.604082,5.471524,-0.689583,6.352735,-5.007186],[6.346309,-1.129639,-6.255293,9.218632,2.427609,0.748269,-4.990823,-6.280325,1.385022],[2.143875,-2.725568,0.966143,-4.884678,-2.111207,1.626721,2.002057,-1.447875,-5.924596],[-2.665110,-5.623096,3.653063,-8.670699,-8.021859,-6.582801,-7.960665,-8.452994,-3.459580],[-1.918716,-4.410741,-2.737649,-2.618233,-6.878765,2.047133,3.559197,-8.083631,1.172666]],[[6.558641,-8.203023,1.150868,1.659864,-9.518168,-4.518115,8.737720,-0.626776,7.279350],[9.435582,-9.052294,6.082618,-9.942783,-6.385034,-7.079242,6.934530,5.856863,0.030293],[-6.058145,-4.983208,4.321097,1.760697,-6.065188,-0.175347,-4.576755,4.843040,4.973394],[-7.230097,4.809477,-5.774030,1.263294,7.413613,5.428103,7.527818,5.550504,7.427026],[-2.732329,7.651921,-9.351021,7.621267,8.515542,8.194543,7.353195,7.070563,-0.829449],[-3.771171,8.642742,9.656012,-0.825181,-2.459378,-8.235403,-9.987657,9.678671,9.387988],[6.951651,-2.267447,8.806295,5.842714,-4.114479,8.533857,5.772866,-8.789150,1.293779],[6.450225,-4.710177,6.377796,8.444043,9.398866,6.379477,-2.053176,8.607686,-7.957675],[9.154833,6.818499,-8.665829,-5.480789,-6.846757,-0.498052,-4.951468,-4.114866,2.150763],[-6.556341,5.028461,2.646052,-7.763425,-7.901261,5.787378,7.967400,-7.850762,-7.246386],[5.256718,-9.627789,0.484088,-0.745136,9.550845,-3.695659,-1.257712,7.139310,-9.332484],[6.492762,3.916912,4.360355,-9.637638,9.174942,8.369113,2.640333,-0.233831,6.029332],[-6.322117,-2.424845,-7.030741,-3.312857,5.098539,-6.465832,3.870059,-4.078937,8.065363],[3.238606,-3.067444,9.030179,-8.051273,5.571565,-2.763777,-3.975776,-6.048257,-5.174486]],[[6.794837,1.217369,1.974584,-3.963281,8.557397,9.935008,-5.213895,-3.781423,3.600191],[0.095533,-6.022373,-3.858148,0.234817,3.721489,8.507011,-8.923748,9.092355,3.438567],[5.869740,2.101673,5.381407,-6.075413,1.009448,-6.879439,-2.319603,-1.168302,-6.398865],[3.299556,5.649136,3.396669,4.444168,2.990970,-3.794206,7.730124,-3.962349,-6.393419],[3.324665,-4.322755,-2.491930,8.772946,-8.586436,3.675792,-3.076793,-1.499169,-5.864287],[0.401637,4.851638,-2.615161,3.939164,8.365214,-6.400813,2.946943,-3.322541,-7.433187],[0.105671,4.645727,7.520726,-4.447238,-1.580146,0.815340,1.503847,-2.259833,6.846871],[-9.026672,-3.184601,-6.232167,1.605506,-1.469025,3.791797,-6.737654,-4.891913,-9.458202],[1.913406,4.133322,2.309403,-9.390498,0.206854,1.200049,1.472417,-1.307935,-0.510146],[-9.268120,-5.455692,4.704761,2.180087,-5.456294,-6.796790,2.333855,-4.851045,1.077113],[9.977638,-8.808993,-0.926526,5.908561,1.835841,1.929162,-1.104645,2.415650,-3.062968],[5.224193,0.411730,9.577160,-5.986914,9.533465,9.667578,0.230645,-7.755713,-2.778020],[8.155580,-4.037495,9.170352,-4.218910,6.263813,-1.502085,5.230819,4.481268,-1.759234],[-8.599922,7.428297,3.463915,-5.127838,5.102387,-9.189493,3.604639,4.710747,-1.801986]],[[4.379013,-8.695430,-9.182416,-8.801033,-8.484721,-1.226060,-6.218071,-8.732249,-2.417212],[-1.346020,-5.805602,4.399428,5.668212,-1.880452,-6.099164,4.265230,-1.745339,-6.679475],[4.613578,-6.486873,-9.257733,-4.350859,6.643978,-2.410864,-2.596800,-3.711514,-4.633505],[6.639442,-8.551946,-1.480532,-3.901781,-0.533896,-6.065417,7.191640,8.871452,6.963760],[1.081352,-6.038245,-7.507391,-6.709463,2.369719,6.778390,9.694592,-2.352468,7.865472],[-9.170116,-1.829087,-6.998975,-7.343650,-0.955959,2.865973,5.607455,8.979149,0.966668],[6.036647,-7.504604,-1.964628,-1.350750,-9.207244,5.173659,-7.716458,0.313990,-3.718493],[8.570553,-7.538947,-3.149209,0.349383,-4.132915,4.211404,1.271395,8.449506,1.868391],[2.067903,1.424190,-2.570496,6.356033,1.804288,-4.181543,-5.601272,-7.458066,9.516569],[-1.133139,-0.443614,-6.501411,-6.398474,-2.824317,2.036759,0.814374,-2.079029,6.692184],[9.194306,2.665904,2.286753,-9.272949,8.629365,-1.329991,4.789353,8.549605,1.627988],[4.431559,6.409280,-9.716914,-2.974936,-1.700911,-1.497098,3.212311,-8.210384,-3.710575],[-4.867245,-4.288290,9.029510,4.202801,2.696611,9.405629,-3.697879,6.895287,-5.437935],[-9.810487,7.875813,0.321722,-2.305154,-2.573410,8.900131,6.507856,5.218707,6.818154]],[[-4.086392,-7.006686,-4.493460,0.892599,-9.546760,-5.907144,-3.674512,8.371208,0.895438],[2.083167,-5.854314,9.740685,0.971471,0.104610,-8.913883,1.479474,4.587045,-1.575117],[7.766227,-9.289133,-4.221485,1.802823,-7.388277,1.307025,9.824820,1.254687,-7.187272],[-4.600525,2.440068,-8.077290,-7.057694,8.363702,2.753696,9.088734,0.034288,2.666238],[5.695628,-2.563528,4.192397,1.490206,-6.113621,-6.461513,0.791262,-0.988317,0.678545],[-8.964723,6.514634,6.162841,-5.128914,3.809926,9.552732,7.928088,6.039841,-6.171788],[3.974726,-4.917639,-1.606849,-0.046389,7.203292,5.063627,-8.975211,-4.852920,0.420211],[-4.851968,4.595829,8.983932,-2.738248,3.078497,-8.144018,3.124397,2.520316,7.728629],[-9.414262,-2.143977,6.838451,-5.458243,8.355385,5.068001,3.036932,-7.457157,2.505101],[3.122123,5.961041,-0.350084,-2.676757,5.709721,1.312835,2.439446,4.050555,3.230873],[0.088064,-4.077988,-6.950121,2.835571,5.931375,-9.402227,0.396963,2.252767,-0.190288],[-9.570779,8.757920,0.567343,5.744285,-1.641178,-2.684733,1.422003,-8.726914,-1.161607],[9.807132,-6.892738,-1.575648,2.268023,-3.393241,-0.529802,-8.556892,4.496432,-5.849593],[-5.983692,5.256324,-9.920742,-5.154920,-4.198719,7.200375,-2.131165,-3.372062,-1.659882]],[[5.777914,5.083286,-5.033940,9.689653,-1.478540,-9.113704,5.033885,-4.055381,5.417458],[-0.765513,-7.108917,-0.980532,2.850425,-2.102993,4.887400,-6.455026,1.021802,-6.204228],[3.617239,-3.777875,-2.132744,2.380016,-8.493685,3.944388,-5.414089,5.483286,-9.205590],[5.159324,-6.197871,6.444190,-2.783415,1.105056,0.079220,1.041496,8.798950,5.188568],[-6.194452,8.713775,-9.984578,-0.251420,-9.477610,-6.544846,-0.718799,-3.815296,3.598038],[-0.333381,-5.231707,8.238865,-9.112299,-6.307551,6.786544,8.339951,-6.397928,-4.442563],[3.643194,-1.916420,-0.331051,-6.450241,-1.710788,-3.817886,3.842320,0.604927,-5.761463],[3.291139,5.990166,-0.384749,3.907498,-0.261461,-7.108277,2.141336,8.744717,8.635665],[-5.182157,-8.101720,-5.690385,-2.407920,-8.331194,8.990834,8.350941,4.019429,9.552003],[6.131793,2.437610,-8.642726,-4.412713,-5.270861,5.528912,2.597530,5.024779,-1.275968],[-9.399165,7.786454,-4.726082,2.675388,-8.282246,-4.454087,-4.995689,-5.473743,3.008886],[9.565699,-6.071966,-6.429668,6.787486,6.597945,1.487322,0.071299,9.381116,-2.794817],[1.044654,2.348901,-3.797879,9.679026,4.740184,0.782431,5.859169,-9.916432,-6.939149],[5.589137,-6.053118,-0.207855,7.242240,-4.717406,6.036889,-5.591995,-6.907724,-7.986488]],[[-4.654834,-1.051589,8.072588,-2.107992,-6.133270,2.770454,8.530091,3.449836,1.715400],[-7.694027,2.687817,1.709273,8.096847,5.654208,6.176234,6.797163,-7.471145,9.599641],[7.759627,-5.510173,-5.136111,-2.392967,-8.384140,-5.914144,8.783132,-6.712377,6.853205],[0.287932,-9.383111,-6.779924,-9.916942,2.816012,2.913478,2.997179,-7.968348,0.238791],[4.284050,3.621229,4.427069,7.035168,-8.067246,6.587359,-7.567574,-4.155737,0.122824],[-7.027400,-3.088807,-1.488061,6.003041,-2.437332,7.031415,-9.094195,6.001832,8.576207],[4.326040,-1.392154,-2.080787,0.632947,-9.162120,7.851860,2.859598,-9.706794,4.175752],[8.767371,-5.886088,1.398114,6.895979,-6.221425,6.669695,3.150780,-1.955816,-1.949205],[7.957516,7.223785,4.570921,-2.044215,2.518243,7.377564,3.339932,1.934306,-7.258452],[-8.311263,-2.674811,7.368810,-1.025087,0.963177,-7.265383,5.500774,5.668167,7.231488],[-0.502304,6.742154,-3.221124,-0.435700,-1.794505,7.558594,-6.750189,-6.357994,-7.550264],[-0.917642,9.623673,-5.715481,9.035904,2.740850,-8.774427,-3.079589,-4.150454,-7.206031],[-8.804099,1.698295,-4.532045,5.981482,-8.048757,-1.000383,6.222680,3.562299,-5.781036],[-4.821276,1.079809,-7.962633,2.867665,-1.509884,2.496794,-6.495510,-5.038538,-9.965819]],[[3.389265,2.215386,5.323625,-3.262560,8.515973,4.554455,0.372839,1.950563,-9.336423],[7.314338,-3.662958,8.739784,-2.699259,5.042851,2.020942,-1.063545,-6.556966,-7.724507],[3.645881,-6.397641,-5.177086,-1.440972,-2.895640,0.382653,6.488035,-2.865540,-0.896994],[6.609116,3.799952,6.957784,-1.116567,3.541721,-3.743849,4.199821,9.467494,-8.652576],[6.945774,8.923561,-4.879252,-4.716495,9.572639,-9.339624,5.964536,0.387722,-2.755070],[3.590575,-0.892668,8.154826,9.168792,-3.567124,7.937918,-4.203697,1.710591,-6.175545],[-6.378203,0.110415,8.703425,-6.560980,-8.898734,3.028842,-3.317056,8.947484,-4.453903],[2.960601,1.146385,-2.746804,-3.477039,-4.091877,-6.471824,-9.598406,0.551761,0.847054],[6.416907,5.034090,7.053876,-3.792334,-6.475401,-4.504722,6.679042,-7.776834,8.039119],[0.300228,4.455239,7.847477,-2.477067,3.900294,0.673576,-3.817846,-9.094478,-1.808372],[-8.143978,3.206665,7.285006,9.543218,-9.393031,6.631796,-3.117017,5.038975,-8.549780],[-5.347331,-5.003188,9.246250,-0.284380,-4.163052,4.057070,-3.439600,-4.485828,-9.406667],[7.973822,-0.168173,9.807251,5.674394,-2.127709,-5.370404,0.531622,6.706111,-9.006169],[8.799462,-9.209535,8.481189,4.135839,6.543190,3.832876,4.087332,-8.706264,-4.984334]],[[9.302570,3.514412,6.042128,6.681175,-9.413244,-1.000104,4.912056,7.040099,-2.045384],[8.764488,-2.627405,8.291883,7.135605,-2.496637,-4.001217,5.749828,-1.553934,0.207995],[-2.142491,-3.298457,7.833321,5.211684,3.236913,-9.418736,1.118109,4.126114,-7.519622],[-7.055661,3.380375,9.985627,6.631436,4.421254,-1.271466,4.314173,2.521983,5.422302],[3.757731,7.045776,-0.663465,-6.444402,-6.239765,8.221656,-4.714677,-3.679084,8.843688],[1.334919,-9.299367,-1.753904,6.949261,-3.877469,-6.949920,8.120273,-1.492356,-9.184603],[-4.768759,1.720829,-3.350482,1.418996,-3.622251,-6.489694,-0.352887,-1.746665,-0.641367],[7.124646,8.692914,1.918469,0.412517,3.438832,6.392777,-1.642365,-7.261900,-9.764297],[9.084438,0.715131,7.796090,7.661847,3.308156,5.260195,-0.221447,6.892255,-5.099414],[0.398744,4.519767,1.819936,-1.865998,-4.371656,1.711269,8.820002,-0.901062,-5.807413],[-7.954674,3.141099,-9.715111,-3.456912,8.201453,-9.153079,-0.937004,-6.017756,-0.249839],[-9.574427,7.622305,7.588667,-5.975054,0.075390,8.073954,-1.406037,4.628940,0.347329],[-3.965294,1.059071,4.902070,2.722302,8.878669,3.261478,-2.130789,7.672812,-9.729551],[-6.503349,6.949026,9.962435,-6.722505,2.866746,-7.257363,6.226605,-8.433277,-0.860324]],[[-5.545237,3.889408,5.268929,6.595954,-4.949959,-4.809455,1.511126,6.287247,-1.524715],[0.346383,4.011801,4.352428,8.389478,-8.715141,3.759371,6.283759,5.725274,5.942910],[0.199169,-2.301517,2.599853,-1.313160,1.364613,-1.119274,-9.669903,-3.040813,-3.794171],[8.848608,-5.491721,3.636044,6.824175,5.171547,1.575003,2.755387,6.153935,-8.662517],[-9.973573,8.910509,4.161292,-5.966455,1.491835,3.054097,-6.745128,6.861012,-9.137167],[-5.173523,7.057643,-8.519695,-2.137843,0.350321,8.586165,2.829664,-8.601361,-5.842560],[-9.968492,-0.621799,2.814559,1.113348,-9.493538,8.628627,7.201507,-0.986584,6.421954],[1.951034,3.178857,-1.803692,-4.424017,-1.927086,-9.600520,-6.800515,5.243179,7.681131],[-3.872034,-9.822872,6.958748,-8.518710,2.264017,-0.751934,8.425273,2.953675,-0.648098],[-9.703160,3.460991,-0.695365,8.067005,-9.391024,6.514295,0.907651,-7.424339,-1.355325],[1.450523,-6.314587,-9.434886,-5.174499,-3.584817,-8.879895,7.518452,8.633644,5.436456],[-0.912847,6.129557,-6.760894,-8.772526,-7.342245,3.610878,6.412151,8.821979,-4.936257],[-0.636204,-5.978245,-7.757458,6.517598,0.230702,-7.005609,-2.694376,8.053076,7.018198],[-7.307581,4.105746,-4.945405,-1.651612,-5.075167,1.866014,7.296843,0.439703,-3.220368]],[[-9.673552,2.483033,-7.947714,-8.185783,-4.825109,-4.693797,2.510160,-8.750070,-9.044105],[-8.900209,-0.267199,-0.706363,3.087736,3.292021,8.532206,-5.804427,5.856843,9.662714],[6.856074,3.475062,6.980714,6.303162,-8.011921,-3.249239,-0.648648,6.298898,0.938574],[-4.186984,8.814007,7.710564,7.174172,-4.212312,0.781877,-1.160903,-4.843611,-6.783244],[-6.180902,-3.448518,8.109306,8.615259,6.635746,-6.250270,-0.944782,6.946979,-7.856818],[7.105214,-6.471088,4.801183,4.163040,-7.801769,1.335510,-6.461952,-7.508120,-2.651092],[0.081022,8.711613,5.738483,5.938092,-6.803589,6.112763,2.180068,-2.253752,0.951591],[2.546428,-7.441548,-8.303444,-6.828024,0.684094,6.473135,0.089709,6.561892,-9.675691],[-5.735390,-9.441313,-1.657534,-8.709987,-4.802662,-6.864659,-9.679202,0.799998,4.048305],[-4.848132,-4.308706,3.416227,5.138651,-8.168154,5.527284,3.362336,-2.744190,2.963769],[-5.808310,-3.595334,-6.421359,-5.473682,-7.131810,7.929004,0.911966,9.601373,8.264163],[3.704977,5.229980,4.152785,-3.043517,3.280545,4.818895,-2.258420,-9.938354,-9.787257],[0.215470,4.248916,-7.092622,7.233510,-0.804090,3.760792,9.760248,-2.449300,-9.452137],[-3.189750,1.179140,5.095664,-7.113139,-9.391184,-1.642154,3.759908,6.009898,0.996867]],[[-7.157251,3.071490,-1.516669,-1.162320,4.789650,-6.691821,8.450880,7.526762,-2.104382],[1.263115,-0.376129,6.549965,-7.820206,-6.730452,5.251960,-8.586707,-5.057541,-0.776890],[-3.754075,9.657160,-4.255298,-9.137521,-4.446006,-3.013739,-1.369076,0.619319,7.289459],[-4.134407,2.342696,6.409646,-6.164046,-5.914764,6.199463,5.820556,-1.511513,3.725604],[2.642238,-7.193442,7.175624,-8.893425,-5.512583,-4.769748,1.520210,3.341632,0.013498],[8.525652,-9.201631,0.320976,4.152198,-1.483670,3.091249,-3.744731,9.794869,-6.055401],[7.221206,-0.289550,-4.546894,4.390360,5.109747,-6.508862,9.467108,8.975902,-0.352782],[0.201778,1.689991,3.661608,9.829513,2.706825,-5.660329,-3.570008,8.840685,7.151479],[4.323868,0.433711,0.748074,-3.481038,2.524630,3.833972,-5.134255,1.384415,-0.130667],[-7.902758,-4.760406,-4.842035,4.494161,-4.440372,4.859412,-2.845976,-1.059696,-3.346223],[8.768182,-2.496358,-4.306376,-6.961846,-1.942812,-2.169567,-8.786675,4.102244,3.686199],[0.890007,-5.607313,8.434662,8.641913,-5.237144,9.986774,-3.065112,4.195825,4.566520],[-6.282063,-7.709237,-8.329596,9.683593,0.643089,8.317400,-3.495297,1.381535,0.769891],[-5.978113,-5.235125,-4.157937,7.762487,9.426656,-9.740362,-3.959221,-1.351519,6.766177]],[[-5.651585,-5.604816,2.308685,-7.837024,-4.326606,-0.946240,-7.692119,0.955766,-7.606902],[-1.413915,-3.915968,-1.983611,-7.433904,6.762481,-1.464779,9.599437,1.769643,-6.851456],[-1.880832,-9.440760,2.653937,-4.337718,8.821094,-3.152485,6.888362,-4.648550,-0.365680],[-7.883094,1.317657,-8.412624,1.245616,-8.263564,6.385926,8.493075,6.757216,-6.716674],[2.293964,-0.735442,-2.332016,-2.606748,1.619446,8.883435,1.389305,6.288405,1.151436],[-7.203830,9.621410,8.060109,-4.702298,-9.719055,-2.547894,-3.032974,8.618804,1.166414],[9.515960,-3.730759,-5.971494,5.604027,2.477368,-9.557591,-9.307336,2.437567,-2.306239],[-7.609281,-2.070677,-3.035235,2.156559,-8.926530,-0.767182,5.225119,7.328138,-7.286067],[7.644366,4.000765,-8.669790,-3.275484,2.880915,2.544880,-4.340423,-1.161350,-5.569237],[-8.941510,9.707614,3.910356,-1.439306,0.253122,-9.429317,7.508596,9.676114,-8.931966],[-5.855201,-0.191616,6.808235,9.427862,9.803289,-7.949615,2.582023,-8.450946,0.553254],[-5.799286,-9.898956,6.801332,3.665834,-1.034095,8.127729,3.199028,1.155445,0.447908],[3.388004,6.501925,-6.209437,-7.057589,-8.691748,-4.370389,-1.732361,0.720056,4.792154],[5.608670,9.577231,6.642826,1.971484,-2.504747,0.903687,9.535280,4.333091,8.806833]]], dtype = "float32")#candidate|703|(13, 14, 9)|const|float32
uop_704 = relay.atan(const_703.astype('float32')) # shape=(13, 14, 9)
bop_708 = relay.add(uop_704.astype('int32'), relay.reshape(const_703.astype('int32'), relay.shape_of(uop_704))) # shape=(13, 14, 9)
var_711 = relay.var("var_711", dtype = "int32", shape = (13, 14, 9))#candidate|711|(13, 14, 9)|var|int32
bop_712 = relay.power(bop_708.astype('float32'), relay.reshape(var_711.astype('float32'), relay.shape_of(bop_708))) # shape=(13, 14, 9)
uop_715 = relay.tan(bop_712.astype('float32')) # shape=(13, 14, 9)
output = relay.Tuple([uop_715,])
output2 = relay.Tuple([uop_715,])
F = relay.Function([var_711,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_711,], output2)
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
input_711= np.array([[[4,-3,-10,5,-7,-7,4,2,-8],[2,9,-7,-10,1,7,-10,-9,-9],[7,10,-9,-4,-5,-8,-5,10,-2],[6,-8,-7,-6,-6,-9,4,1,1],[-1,4,1,8,2,9,-5,-9,4],[9,-4,-5,-2,-10,-2,-4,6,-8],[9,3,4,-7,-9,-3,4,-10,-7],[-9,5,3,-10,10,-4,-3,4,8],[9,6,-4,8,-1,4,7,-6,-5],[-10,5,-3,10,-5,5,2,-9,-2],[7,10,-10,5,-8,-10,7,1,1],[2,1,6,1,8,6,4,9,3],[-8,9,-9,-2,10,-10,-5,-2,2],[-2,6,4,9,-9,-1,10,-3,7]],[[7,-8,-1,9,3,-4,-1,-7,-6],[1,6,-8,2,-6,6,-7,-3,3],[5,1,5,-2,-2,3,7,-5,-4],[2,10,1,1,10,-4,7,-5,9],[4,4,10,3,-1,-2,1,1,2],[-9,1,-7,2,-6,2,7,-9,-1],[-8,6,-3,4,-2,7,-8,-7,8],[6,9,7,9,9,-4,-7,-5,4],[3,3,-5,-1,9,-2,1,-6,-1],[-4,-4,-8,-3,3,-10,-5,-6,-2],[1,4,1,-6,5,5,-5,-4,-3],[-9,9,-4,8,-6,10,-7,-1,-10],[8,-5,9,7,-10,-3,5,-9,4],[5,-5,-8,9,5,-6,-2,-2,-5]],[[8,5,5,1,9,-6,3,7,-4],[-4,-8,-3,-5,-7,2,10,-6,-8],[-4,-2,-8,8,-2,9,9,9,2],[4,-6,8,4,8,4,10,-6,3],[-9,-3,-5,6,-6,1,-6,-7,5],[10,5,9,5,-9,10,-5,10,-1],[-10,9,4,6,-5,2,-10,9,-4],[5,8,2,-6,8,-3,-3,5,-9],[-3,9,-9,8,-5,5,7,-9,-3],[-2,6,-7,8,3,3,4,-8,2],[6,-7,10,10,-1,2,-10,5,6],[-6,-1,10,3,-9,-9,-8,-2,-5],[4,-10,6,9,8,-5,-5,9,-2],[-1,4,-5,3,-7,-8,-7,-2,7]],[[1,-8,-1,3,6,5,2,-2,-6],[-9,-2,-3,-3,-3,1,2,8,-1],[3,-7,6,8,8,4,6,-2,-2],[-9,9,-9,-1,-10,-3,10,-5,-6],[8,8,-1,-9,-8,9,8,-6,-9],[-5,7,9,-8,-5,-7,3,2,-9],[-6,4,-9,-3,-1,4,-5,10,6],[-2,-10,7,-8,1,2,7,-8,9],[3,6,4,-4,9,-10,3,9,-7],[-1,7,-6,-3,4,-3,-1,-8,2],[7,8,10,-10,8,-6,-8,-1,-10],[3,-10,-7,1,-9,4,-8,7,-6],[-5,-10,8,9,5,-4,-9,-10,-5],[9,-2,-4,2,-7,9,8,10,7]],[[5,7,-8,-7,-7,5,2,5,-6],[2,2,-9,-3,-4,-4,-2,-1,-9],[9,5,10,5,5,8,10,9,1],[2,2,6,9,5,8,-8,-1,-2],[-4,-5,-8,10,4,9,-5,9,7],[5,5,7,-5,2,-9,-7,3,1],[6,9,-1,5,-9,5,-10,-5,2],[8,-6,-5,7,7,-5,-9,-2,-3],[7,8,10,4,-6,5,-1,-7,-5],[2,-6,10,-9,-4,-9,3,5,-4],[7,-3,8,-4,2,6,6,7,-6],[5,10,-6,8,-7,-3,6,-10,6],[8,7,-1,-7,-8,9,3,-1,7],[2,-9,8,9,-3,-6,6,-8,4]],[[-5,-6,-7,-9,3,3,-7,5,5],[5,9,4,-8,-9,-5,-6,4,-3],[9,1,-2,-1,-3,-3,10,-4,-10],[4,1,4,6,4,4,-3,1,-6],[-2,4,-1,-6,6,10,8,-4,7],[3,-10,3,8,10,2,4,10,9],[-4,-4,5,-6,10,-2,-10,8,3],[7,-9,-1,-10,7,5,-3,3,9],[3,-10,7,-8,6,10,-9,-1,3],[-3,10,-5,8,-2,-5,-6,8,9],[-8,4,-5,3,3,-6,9,9,9],[-4,2,-5,-9,-5,-7,-4,-1,8],[5,2,7,-4,7,-6,7,-3,-8],[-6,-1,5,7,-2,-10,-6,2,-2]],[[-8,-7,-10,-3,-5,-8,7,9,6],[6,3,-2,10,7,8,3,6,10],[5,6,10,-2,-9,4,6,9,-10],[4,2,4,-3,-5,6,-6,9,-10],[6,-5,-6,-10,-9,-4,3,8,9],[5,-8,-10,8,-7,10,7,3,10],[-10,-2,3,-9,3,-2,-9,6,-1],[-6,-4,7,2,9,5,3,-2,-6],[2,-1,-5,1,4,-5,-4,4,-4],[4,-3,9,4,5,10,3,1,9],[9,-1,-5,-9,-7,-1,-4,-5,9],[6,10,1,7,2,-10,-3,8,3],[2,6,-5,-10,-3,-7,-1,-8,-6],[-8,5,-10,-10,-3,1,-4,9,2]],[[-7,4,-7,2,4,8,-1,5,-10],[-10,-10,-9,-2,-3,10,-7,-8,-7],[-2,7,-1,7,3,7,-7,-4,-1],[-7,3,-3,5,-2,-1,-2,-9,10],[-3,-9,9,2,2,-4,-6,-5,-10],[5,-8,-2,7,-1,-10,4,6,-2],[9,2,7,-6,6,5,10,5,7],[5,9,10,-5,-5,-10,-7,-10,-2],[-6,1,10,1,-7,5,2,-10,6],[-2,-6,3,5,-5,-10,1,5,-9],[1,-2,6,4,7,-9,-4,3,2],[-6,-5,-2,-7,-2,-7,7,3,-8],[-7,7,4,7,8,10,6,-7,-6],[-8,1,7,-5,8,4,-4,-8,-4]],[[-6,-2,1,-6,-9,-7,-1,-10,-7],[2,9,-9,-8,3,6,8,-10,-8],[-4,7,-9,-10,1,3,-9,1,7],[-3,3,-8,-5,2,-1,-6,5,10],[-10,-4,-2,-1,2,-1,2,-1,-9],[-5,-9,10,7,5,-1,-8,5,4],[-7,-10,-6,7,4,4,-10,8,4],[-7,3,-5,-7,5,-2,-3,3,9],[4,-7,7,2,10,5,5,7,6],[-2,2,6,6,7,3,-9,4,-5],[-9,8,-4,-3,9,-9,-1,-9,-2],[-9,4,-10,-3,-6,-6,10,8,5],[1,5,8,-4,-6,-4,-7,7,-10],[-10,7,-10,7,-10,-7,6,-3,-7]],[[-4,-9,-3,-1,5,-4,6,-8,10],[-7,7,9,-2,-2,-8,5,-2,-4],[-7,-9,-4,9,-7,-2,6,-3,7],[-9,-8,4,-1,-10,3,-10,-1,-7],[2,-5,-5,-5,6,-6,-9,-4,5],[5,-6,9,-10,-4,-9,-6,-6,-9],[9,4,3,10,6,-1,9,6,-2],[7,-7,-5,-1,-7,-3,10,2,7],[5,-7,1,-1,-5,-8,-6,-1,7],[7,-8,-4,3,-10,7,10,7,4],[-7,-8,4,1,10,-10,7,-4,-4],[-5,2,1,5,3,9,9,2,8],[3,3,1,-2,-3,-3,-5,-1,4],[-8,3,2,-1,-8,8,9,5,-5]],[[-7,7,3,6,-10,5,-9,4,6],[-4,7,-10,-7,-4,-4,1,8,-3],[7,9,6,-1,6,-9,-1,9,-4],[-3,-1,3,8,9,1,-4,-9,6],[-6,-2,-2,7,5,-3,9,2,-3],[-5,1,4,4,-10,1,-10,-3,-9],[5,-2,4,-1,-1,10,8,-2,1],[-10,-6,-10,6,-3,-2,6,10,8],[-1,-6,-2,6,-4,-8,6,10,6],[-10,9,1,-8,9,-5,4,7,-8],[-8,8,5,-9,-6,5,-10,-6,8],[10,-4,-6,9,-5,-1,7,9,3],[3,-5,2,1,-4,-9,-4,-8,10],[-9,3,-7,-7,2,-10,-10,-7,-4]],[[-6,9,1,2,-10,6,-2,1,9],[8,10,3,-6,-5,-9,-2,-3,7],[-2,3,-9,2,-6,-7,-8,-10,-6],[10,-6,-5,-5,2,-9,-6,-7,-10],[8,8,-2,-8,9,-1,-2,8,-8],[-7,-1,3,-7,10,-5,-10,2,-8],[10,-7,7,5,10,9,6,3,4],[-5,2,-7,-6,2,3,-4,-8,-4],[-10,-6,2,-2,-3,4,-10,8,-4],[1,-2,-10,4,9,-5,-1,-2,7],[2,2,4,-3,-5,-6,-8,1,-1],[4,-2,-1,6,4,-7,-3,2,-4],[-1,1,-10,7,6,-7,-8,4,5],[-9,-5,10,7,3,-9,8,-6,6]],[[-7,-3,10,2,2,-7,9,-8,10],[8,-3,9,-8,2,7,9,2,9],[-1,-7,4,9,-10,10,5,9,9],[-10,4,-2,-1,-7,4,-4,-8,9],[6,7,-3,8,-7,5,-5,-4,6],[-2,-8,9,-8,6,-5,7,-10,-7],[-1,2,2,-3,-2,-7,-4,5,-1],[-1,-1,-2,1,8,-4,-8,-10,3],[8,10,-5,-8,8,-9,2,-6,9],[9,-7,-5,2,6,-4,3,-6,-3],[9,-10,-9,-6,2,-3,-9,-5,7],[-7,-3,-10,-3,10,7,-2,7,5],[-2,-6,-2,7,1,-8,-10,-9,8],[-2,4,2,-5,10,-6,-7,-6,3]]], dtype='int32')
module1.set_input('var_711', input_711)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_711, )
res3 = intrp3.evaluate()(input_711, )
res4 = intrp4.evaluate()(input_711, )
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
module5.set_input('var_711', input_711)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_711, )
res7 = intrp7.evaluate()(input_711, )
res8 = intrp8.evaluate()(input_711, )
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
module9.set_input('var_711', input_711)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_711, )
res11 = intrp11.evaluate()(input_711, )
res12 = intrp12.evaluate()(input_711, )
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
module13.set_input('var_711', input_711)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_711, )
res15 = intrp15.evaluate()(input_711, )
res16 = intrp16.evaluate()(input_711, )
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
module17.set_input('var_711', input_711)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_711, )
res19 = intrp19.evaluate()(input_711, )
res20 = intrp20.evaluate()(input_711, )
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
module21.set_input('var_711', input_711)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_711, )
res23 = intrp23.evaluate()(input_711, )
res24 = intrp24.evaluate()(input_711, )
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

'''28: TVMFuncCall
27: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
26: tvm::transform::Pass::operator()(tvm::IRModule) const
25: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
24: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
23: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
22: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
21: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
20: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
19: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
18: _ZNSt17_Function_handlerIFSt10sha
17: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
16: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
15: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
14: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::TupleNode const*)
13: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''