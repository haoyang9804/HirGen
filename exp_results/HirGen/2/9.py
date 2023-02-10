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
const_32 = relay.const([[7.929061,-7.749843,-8.689723,-3.453226,6.802972],[1.281562,9.986092,-3.223099,-8.315408,-1.160550],[8.042205,-4.288188,-2.865290,1.507249,5.156565],[-1.555064,-7.105994,-5.819074,7.378049,-6.717805],[-5.957625,6.873538,7.971491,8.097758,9.652341],[-3.652820,7.904217,-4.116925,0.050511,-9.146064],[-8.712019,1.485042,7.003862,4.022104,-0.748387],[4.951375,8.989897,-6.046402,-8.144938,-3.699259],[9.946647,7.752628,1.310592,1.650714,6.875047],[7.136399,-2.591802,-9.553997,3.693484,2.983765]], dtype = "float64")#candidate|32|(10, 5)|const|float64
uop_33 = relay.erf(const_32.astype('float64')) # shape=(10, 5)
uop_41 = relay.log10(const_32.astype('float32')) # shape=(10, 5)
output = relay.Tuple([uop_33,uop_41,])
output2 = relay.Tuple([uop_33,uop_41,])
func_51 = relay.Function([], output)
mod['func_51'] = func_51
mod = relay.transform.InferType()(mod)
output = func_51()
func_52 = relay.Function([], output)
mutated_mod['func_52'] = func_52
mutated_mod = relay.transform.InferType()(mutated_mod)
const_60 = relay.const([7.612766,5.299240,-2.759523,9.599302,-8.218623,3.842389,-1.747124,5.199438,-2.182924,-7.804608,-9.752656,-3.096530,7.795955,0.246064,8.772928,-0.186253], dtype = "float64")#candidate|60|(16,)|const|float64
uop_61 = relay.asin(const_60.astype('float64')) # shape=(16,)
bop_64 = relay.floor_mod(uop_61.astype('float32'), relay.reshape(const_60.astype('float32'), relay.shape_of(uop_61))) # shape=(16,)
output = bop_64
output2 = bop_64
func_72 = relay.Function([], output)
mod['func_72'] = func_72
mod = relay.transform.InferType()(mod)
mutated_mod['func_72'] = func_72
mutated_mod = relay.transform.InferType()(mutated_mod)
func_72_call = mutated_mod.get_global_var('func_72')
call_73 = func_72_call()
output = call_73
func_74 = relay.Function([], output)
mutated_mod['func_74'] = func_74
mutated_mod = relay.transform.InferType()(mutated_mod)
var_175 = relay.var("var_175", dtype = "uint32", shape = (10, 3))#candidate|175|(10, 3)|var|uint32
var_176 = relay.var("var_176", dtype = "uint32", shape = (10, 3))#candidate|176|(10, 3)|var|uint32
bop_177 = relay.less(var_175.astype('bool'), relay.reshape(var_176.astype('bool'), relay.shape_of(var_175))) # shape=(10, 3)
bop_183 = relay.bitwise_or(bop_177.astype('uint16'), relay.reshape(var_176.astype('uint16'), relay.shape_of(bop_177))) # shape=(10, 3)
bop_188 = relay.bitwise_xor(var_176.astype('int8'), relay.reshape(bop_183.astype('int8'), relay.shape_of(var_176))) # shape=(10, 3)
bop_202 = relay.maximum(var_175.astype('int64'), relay.reshape(bop_183.astype('int64'), relay.shape_of(var_175))) # shape=(10, 3)
bop_205 = relay.equal(bop_202.astype('bool'), relay.reshape(bop_183.astype('bool'), relay.shape_of(bop_202))) # shape=(10, 3)
output = relay.Tuple([bop_188,bop_205,])
output2 = relay.Tuple([bop_188,bop_205,])
func_209 = relay.Function([var_175,var_176,], output)
mod['func_209'] = func_209
mod = relay.transform.InferType()(mod)
var_210 = relay.var("var_210", dtype = "uint32", shape = (10, 3))#candidate|210|(10, 3)|var|uint32
var_211 = relay.var("var_211", dtype = "uint32", shape = (10, 3))#candidate|211|(10, 3)|var|uint32
output = func_209(var_210,var_211,)
func_212 = relay.Function([var_210,var_211,], output)
mutated_mod['func_212'] = func_212
mutated_mod = relay.transform.InferType()(mutated_mod)
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_214 = func_72_call()
call_215 = func_72_call()
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_218 = relay.TupleGetItem(func_51_call(), 1)
call_219 = relay.TupleGetItem(func_52_call(), 1)
output = relay.Tuple([call_214,call_218,])
output2 = relay.Tuple([call_215,call_219,])
func_222 = relay.Function([], output)
mod['func_222'] = func_222
mod = relay.transform.InferType()(mod)
output = func_222()
func_223 = relay.Function([], output)
mutated_mod['func_223'] = func_223
mutated_mod = relay.transform.InferType()(mutated_mod)
const_224 = relay.const([[[7,8,-10,2,9,-9,4,-1,6,1,-5,3],[-4,-10,3,6,9,-6,6,-2,-5,3,9,-10],[-2,-6,2,10,-6,-10,4,9,-7,-2,-4,9],[-7,7,-8,1,-9,-9,-6,-8,9,1,1,9],[-6,-2,7,-10,-6,8,-7,4,-6,9,-1,-6]],[[7,6,-7,8,5,-4,3,-7,-2,-3,-6,4],[-1,5,4,-2,9,8,2,9,-10,-2,-8,4],[-5,-8,-5,4,7,-5,3,1,5,7,-7,-7],[-1,2,7,9,4,-5,-10,-2,-10,-10,-6,4],[2,1,10,-10,6,-2,7,-6,-9,-9,-10,9]],[[-10,5,-4,9,2,7,5,-5,-9,-5,8,-9],[-2,9,-1,-6,-9,-1,9,9,8,-3,5,-8],[-5,-3,4,-4,-4,10,1,-10,-6,1,9,-5],[6,-3,-6,-5,2,3,-4,4,-10,-5,2,-9],[-4,-2,-8,1,-4,5,5,-10,2,5,-2,3]],[[-5,1,4,4,10,-1,4,6,-6,10,3,-2],[-3,10,-8,-7,3,-8,10,5,1,-2,-7,5],[-1,9,-6,10,8,7,5,8,-4,5,10,5],[5,6,-8,-7,7,3,-5,-5,3,8,5,-5],[2,-2,1,4,9,-8,-3,3,9,5,7,8]],[[1,-7,1,5,6,-6,9,3,-9,2,-5,4],[-1,9,-9,9,2,6,1,-7,-9,4,-10,-5],[-7,1,-3,6,-3,5,4,10,1,-9,7,9],[-8,5,7,9,6,6,4,-5,-8,9,-4,10],[-6,10,-7,-9,-2,-5,4,-5,-3,-2,-4,3]],[[5,6,2,-5,-4,-10,-3,-6,-5,6,-2,-10],[3,-3,8,10,10,1,9,10,10,-2,1,10],[8,5,-8,-6,-8,-9,-7,-10,-3,8,-8,5],[10,-2,7,9,9,-5,1,-9,2,-7,-7,3],[7,-1,-2,10,-6,-3,4,8,5,2,9,-7]],[[9,8,-9,3,-6,9,-3,-10,-5,3,4,-5],[-9,7,6,8,7,1,-9,-4,-3,3,-5,-6],[-5,-10,-4,-9,1,10,-1,-5,4,-6,2,-8],[-4,10,3,8,7,-6,-4,-4,3,2,-2,-2],[-5,-1,-8,-2,9,10,8,-7,-7,-5,-2,-5]],[[5,-8,-8,-5,-9,-10,5,-7,-1,-6,6,-5],[-8,5,-3,6,-2,7,5,1,-10,10,10,-2],[6,-4,-1,9,-3,-1,7,-4,-10,-6,-7,-10],[7,-4,9,2,2,-2,-6,2,5,6,9,-8],[-3,7,-3,-6,-6,9,-4,3,-6,-1,-3,-2]],[[6,-8,-7,7,5,9,8,-8,6,2,-6,-8],[-7,5,-9,8,-4,-2,7,3,1,7,7,-6],[3,-3,-2,-3,7,10,-4,3,9,1,5,7],[5,-7,-1,5,10,3,8,-10,-4,1,1,8],[-1,5,4,-2,-5,-3,-2,10,-6,-4,1,-7]],[[6,-4,6,-6,-2,1,-2,7,1,-6,8,-7],[-3,9,6,-8,6,9,4,8,-6,-4,-3,-8],[10,-9,1,3,6,-7,-9,-5,7,-8,-1,8],[7,-10,4,-7,1,5,2,6,-1,8,-6,-5],[-1,3,-1,-1,-5,-9,8,-5,1,-1,-1,9]],[[-7,-2,-4,-5,1,-6,1,1,10,-8,5,7],[5,-9,2,8,8,2,8,6,-7,4,-2,7],[2,-10,9,2,-3,2,-8,-5,-4,-8,-4,4],[7,-1,-10,10,-2,-10,8,-8,-3,-8,-5,-9],[2,1,-4,-7,-2,9,-7,-5,-1,-1,-7,4]]], dtype = "uint64")#candidate|224|(11, 5, 12)|const|uint64
var_225 = relay.var("var_225", dtype = "uint64", shape = (11, 5, 12))#candidate|225|(11, 5, 12)|var|uint64
bop_226 = relay.maximum(const_224.astype('uint64'), relay.reshape(var_225.astype('uint64'), relay.shape_of(const_224))) # shape=(11, 5, 12)
uop_232 = relay.atanh(bop_226.astype('float32')) # shape=(11, 5, 12)
bop_234 = relay.power(uop_232.astype('float32'), relay.reshape(bop_226.astype('float32'), relay.shape_of(uop_232))) # shape=(11, 5, 12)
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_246 = func_72_call()
call_247 = func_72_call()
bop_253 = relay.logical_xor(bop_234.astype('uint8'), relay.reshape(var_225.astype('uint8'), relay.shape_of(bop_234))) # shape=(11, 5, 12)
uop_262 = relay.acosh(uop_232.astype('float32')) # shape=(11, 5, 12)
var_264 = relay.var("var_264", dtype = "float32", shape = (11, 5, 12))#candidate|264|(11, 5, 12)|var|float32
bop_265 = relay.less(uop_262.astype('bool'), relay.reshape(var_264.astype('bool'), relay.shape_of(uop_262))) # shape=(11, 5, 12)
output = relay.Tuple([call_246,bop_253,bop_265,])
output2 = relay.Tuple([call_247,bop_253,bop_265,])
func_268 = relay.Function([var_225,var_264,], output)
mod['func_268'] = func_268
mod = relay.transform.InferType()(mod)
mutated_mod['func_268'] = func_268
mutated_mod = relay.transform.InferType()(mutated_mod)
func_268_call = mutated_mod.get_global_var('func_268')
var_270 = relay.var("var_270", dtype = "uint64", shape = (11, 5, 12))#candidate|270|(11, 5, 12)|var|uint64
var_271 = relay.var("var_271", dtype = "float32", shape = (11, 5, 12))#candidate|271|(11, 5, 12)|var|float32
call_269 = func_268_call(var_270,var_271,)
output = call_269
func_272 = relay.Function([var_270,var_271,], output)
mutated_mod['func_272'] = func_272
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_344 = relay.TupleGetItem(func_51_call(), 0)
call_345 = relay.TupleGetItem(func_52_call(), 0)
func_222_call = mod.get_global_var('func_222')
func_223_call = mutated_mod.get_global_var('func_223')
call_348 = relay.TupleGetItem(func_222_call(), 1)
call_349 = relay.TupleGetItem(func_223_call(), 1)
uop_350 = relay.asinh(call_348.astype('float64')) # shape=(10, 5)
uop_352 = relay.asinh(call_349.astype('float64')) # shape=(10, 5)
bop_354 = relay.less(uop_350.astype('bool'), relay.reshape(call_348.astype('bool'), relay.shape_of(uop_350))) # shape=(10, 5)
bop_357 = relay.less(uop_352.astype('bool'), relay.reshape(call_349.astype('bool'), relay.shape_of(uop_352))) # shape=(10, 5)
func_268_call = mod.get_global_var('func_268')
func_272_call = mutated_mod.get_global_var('func_272')
var_363 = relay.var("var_363", dtype = "uint64", shape = (110, 6))#candidate|363|(110, 6)|var|uint64
call_362 = relay.TupleGetItem(func_268_call(relay.reshape(var_363.astype('uint64'), [11, 5, 12]), relay.reshape(var_363.astype('float32'), [11, 5, 12]), ), 2)
call_364 = relay.TupleGetItem(func_272_call(relay.reshape(var_363.astype('uint64'), [11, 5, 12]), relay.reshape(var_363.astype('float32'), [11, 5, 12]), ), 2)
output = relay.Tuple([call_344,bop_354,call_362,var_363,])
output2 = relay.Tuple([call_345,bop_357,call_364,var_363,])
func_367 = relay.Function([var_363,], output)
mod['func_367'] = func_367
mod = relay.transform.InferType()(mod)
var_368 = relay.var("var_368", dtype = "uint64", shape = (110, 6))#candidate|368|(110, 6)|var|uint64
output = func_367(var_368)
func_369 = relay.Function([var_368], output)
mutated_mod['func_369'] = func_369
mutated_mod = relay.transform.InferType()(mutated_mod)
var_374 = relay.var("var_374", dtype = "int64", shape = ())#candidate|374|()|var|int64
var_375 = relay.var("var_375", dtype = "int64", shape = (12, 16, 2))#candidate|375|(12, 16, 2)|var|int64
bop_376 = relay.bitwise_xor(var_374.astype('int64'), var_375.astype('int64')) # shape=(12, 16, 2)
bop_380 = relay.left_shift(var_374.astype('uint8'), var_375.astype('uint8')) # shape=(12, 16, 2)
output = relay.Tuple([bop_376,bop_380,])
output2 = relay.Tuple([bop_376,bop_380,])
func_389 = relay.Function([var_374,var_375,], output)
mod['func_389'] = func_389
mod = relay.transform.InferType()(mod)
var_390 = relay.var("var_390", dtype = "int64", shape = ())#candidate|390|()|var|int64
var_391 = relay.var("var_391", dtype = "int64", shape = (12, 16, 2))#candidate|391|(12, 16, 2)|var|int64
output = func_389(var_390,var_391,)
func_392 = relay.Function([var_390,var_391,], output)
mutated_mod['func_392'] = func_392
mutated_mod = relay.transform.InferType()(mutated_mod)
func_222_call = mod.get_global_var('func_222')
func_223_call = mutated_mod.get_global_var('func_223')
call_394 = relay.TupleGetItem(func_222_call(), 1)
call_395 = relay.TupleGetItem(func_223_call(), 1)
func_268_call = mod.get_global_var('func_268')
func_272_call = mutated_mod.get_global_var('func_272')
var_401 = relay.var("var_401", dtype = "uint64", shape = (660,))#candidate|401|(660,)|var|uint64
call_400 = relay.TupleGetItem(func_268_call(relay.reshape(var_401.astype('uint64'), [11, 5, 12]), relay.reshape(var_401.astype('float32'), [11, 5, 12]), ), 2)
call_402 = relay.TupleGetItem(func_272_call(relay.reshape(var_401.astype('uint64'), [11, 5, 12]), relay.reshape(var_401.astype('float32'), [11, 5, 12]), ), 2)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_403 = relay.TupleGetItem(func_51_call(), 0)
call_404 = relay.TupleGetItem(func_52_call(), 0)
func_209_call = mod.get_global_var('func_209')
func_212_call = mutated_mod.get_global_var('func_212')
var_406 = relay.var("var_406", dtype = "uint32", shape = (30,))#candidate|406|(30,)|var|uint32
call_405 = relay.TupleGetItem(func_209_call(relay.reshape(var_406.astype('uint32'), [10, 3]), relay.reshape(var_406.astype('uint32'), [10, 3]), ), 1)
call_407 = relay.TupleGetItem(func_212_call(relay.reshape(var_406.astype('uint32'), [10, 3]), relay.reshape(var_406.astype('uint32'), [10, 3]), ), 1)
bop_411 = relay.logical_or(var_401.astype('bool'), relay.reshape(call_400.astype('bool'), relay.shape_of(var_401))) # shape=(660,)
bop_414 = relay.logical_or(var_401.astype('bool'), relay.reshape(call_402.astype('bool'), relay.shape_of(var_401))) # shape=(660,)
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_421 = func_72_call()
call_422 = func_72_call()
output = relay.Tuple([call_394,call_403,call_405,var_406,bop_411,call_421,])
output2 = relay.Tuple([call_395,call_404,call_407,var_406,bop_414,call_422,])
func_427 = relay.Function([var_401,var_406,], output)
mod['func_427'] = func_427
mod = relay.transform.InferType()(mod)
var_428 = relay.var("var_428", dtype = "uint64", shape = (660,))#candidate|428|(660,)|var|uint64
var_429 = relay.var("var_429", dtype = "uint32", shape = (30,))#candidate|429|(30,)|var|uint32
output = func_427(var_428,var_429,)
func_430 = relay.Function([var_428,var_429,], output)
mutated_mod['func_430'] = func_430
mutated_mod = relay.transform.InferType()(mutated_mod)
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_438 = func_72_call()
call_439 = func_72_call()
output = call_438
output2 = call_439
func_444 = relay.Function([], output)
mod['func_444'] = func_444
mod = relay.transform.InferType()(mod)
output = func_444()
func_445 = relay.Function([], output)
mutated_mod['func_445'] = func_445
mutated_mod = relay.transform.InferType()(mutated_mod)
func_222_call = mod.get_global_var('func_222')
func_223_call = mutated_mod.get_global_var('func_223')
call_460 = relay.TupleGetItem(func_222_call(), 0)
call_461 = relay.TupleGetItem(func_223_call(), 0)
func_209_call = mod.get_global_var('func_209')
func_212_call = mutated_mod.get_global_var('func_212')
const_467 = relay.const([8,-8,2,-4,5,-3,8,-10,4,-6,-5,-10,9,5,10,10,1,-3,-9,-2,-2,7,-8,-4,2,-5,-5,3,8,7], dtype = "uint32")#candidate|467|(30,)|const|uint32
call_466 = relay.TupleGetItem(func_209_call(relay.reshape(const_467.astype('uint32'), [10, 3]), relay.reshape(const_467.astype('uint32'), [10, 3]), ), 1)
call_468 = relay.TupleGetItem(func_212_call(relay.reshape(const_467.astype('uint32'), [10, 3]), relay.reshape(const_467.astype('uint32'), [10, 3]), ), 1)
uop_493 = relay.asinh(call_460.astype('float64')) # shape=(16,)
uop_495 = relay.asinh(call_461.astype('float64')) # shape=(16,)
output = relay.Tuple([call_466,const_467,uop_493,])
output2 = relay.Tuple([call_468,const_467,uop_495,])
func_500 = relay.Function([], output)
mod['func_500'] = func_500
mod = relay.transform.InferType()(mod)
output = func_500()
func_501 = relay.Function([], output)
mutated_mod['func_501'] = func_501
mutated_mod = relay.transform.InferType()(mutated_mod)
func_444_call = mod.get_global_var('func_444')
func_445_call = mutated_mod.get_global_var('func_445')
call_505 = func_444_call()
call_506 = func_444_call()
output = call_505
output2 = call_506
func_515 = relay.Function([], output)
mod['func_515'] = func_515
mod = relay.transform.InferType()(mod)
mutated_mod['func_515'] = func_515
mutated_mod = relay.transform.InferType()(mutated_mod)
func_515_call = mutated_mod.get_global_var('func_515')
call_516 = func_515_call()
output = call_516
func_517 = relay.Function([], output)
mutated_mod['func_517'] = func_517
mutated_mod = relay.transform.InferType()(mutated_mod)
func_515_call = mod.get_global_var('func_515')
func_517_call = mutated_mod.get_global_var('func_517')
call_521 = func_515_call()
call_522 = func_515_call()
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_523 = func_72_call()
call_524 = func_72_call()
func_222_call = mod.get_global_var('func_222')
func_223_call = mutated_mod.get_global_var('func_223')
call_532 = relay.TupleGetItem(func_222_call(), 0)
call_533 = relay.TupleGetItem(func_223_call(), 0)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_551 = relay.TupleGetItem(func_51_call(), 0)
call_552 = relay.TupleGetItem(func_52_call(), 0)
output = relay.Tuple([call_521,call_523,call_532,call_551,])
output2 = relay.Tuple([call_522,call_524,call_533,call_552,])
func_558 = relay.Function([], output)
mod['func_558'] = func_558
mod = relay.transform.InferType()(mod)
mutated_mod['func_558'] = func_558
mutated_mod = relay.transform.InferType()(mutated_mod)
func_558_call = mutated_mod.get_global_var('func_558')
call_559 = func_558_call()
output = call_559
func_560 = relay.Function([], output)
mutated_mod['func_560'] = func_560
mutated_mod = relay.transform.InferType()(mutated_mod)
func_515_call = mod.get_global_var('func_515')
func_517_call = mutated_mod.get_global_var('func_517')
call_571 = func_515_call()
call_572 = func_515_call()
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_590 = relay.TupleGetItem(func_51_call(), 0)
call_591 = relay.TupleGetItem(func_52_call(), 0)
func_389_call = mod.get_global_var('func_389')
func_392_call = mutated_mod.get_global_var('func_392')
const_594 = relay.const(-8, dtype = "int64")#candidate|594|()|const|int64
const_595 = relay.const([-4,5,7,-6,-5,7,-7,-1,3,1,2,-10,-5,-4,-7,6,5,7,5,8,-4,-9,4,-2,-6,6,2,6,6,4,-7,3,4,-6,7,-8,9,-4,7,-9,-8,1,-3,-1,9,2,-6,9,-9,6,1,7,-1,-4,-6,3,1,5,-1,5,-10,7,4,-3,-1,8,-4,3,-5,-6,-3,4,3,8,-4,5,1,-9,5,7,4,9,2,5,-2,-10,-10,-9,-4,6,-9,3,10,-5,7,-5,9,10,8,1,-6,5,7,-6,10,-10,-2,8,-8,-1,7,5,7,-9,1,-5,3,3,-3,-3,-3,-1,-10,9,4,-3,3,8,5,7,9,5,3,4,-8,-6,-1,-8,-5,-1,6,-9,3,-10,-5,6,-10,-6,2,-8,8,-3,-10,2,-7,-4,8,-1,1,1,10,-7,-10,-3,6,5,7,5,-9,3,-1,-1,2,-9,-3,6,8,6,-10,-2,-2,10,-3,7,1,9,8,-10,2,-3,10,10,7,-5,5,-1,4,-6,-6,10,2,-1,1,-1,3,-10,5,10,-8,-9,4,-9,1,5,6,-7,7,9,1,1,-5,-6,10,4,-3,-10,3,5,-7,-3,8,-1,8,-2,-3,-1,-5,1,-3,5,8,-2,7,4,8,6,7,-10,9,-4,-9,1,1,9,-8,9,10,-8,-7,-10,2,-2,3,-4,5,5,10,-8,-9,-7,1,-3,-7,7,-3,-10,1,4,3,-6,-5,-4,8,-7,9,4,-10,-1,9,-1,-5,9,-5,1,10,-8,-10,-9,-6,-3,-2,7,5,-9,7,7,2,-9,-9,-5,-10,4,-7,8,1,8,-10,5,2,10,-3,3,-2,-4,-5,-10,4,-6,-10,6,-8,5,1,5,-6,-5,1,6,-10,-4,1,-1,9,8,4,1,10,1,-7,-5,9,-4,7,4,2,2,2,-10,3,-8,-4,8,-6,3,3,-6,7,-3,-2,2,4,7,1,7,-7,-10,6,3,5,5,6,-2,-3,-9], dtype = "int64")#candidate|595|(384,)|const|int64
call_593 = relay.TupleGetItem(func_389_call(relay.reshape(const_594.astype('int64'), []), relay.reshape(const_595.astype('int64'), [12, 16, 2]), ), 0)
call_596 = relay.TupleGetItem(func_392_call(relay.reshape(const_594.astype('int64'), []), relay.reshape(const_595.astype('int64'), [12, 16, 2]), ), 0)
uop_597 = relay.sin(call_590.astype('float32')) # shape=(10, 5)
uop_599 = relay.sin(call_591.astype('float32')) # shape=(10, 5)
output = relay.Tuple([call_571,call_593,const_594,const_595,uop_597,])
output2 = relay.Tuple([call_572,call_596,const_594,const_595,uop_599,])
func_600 = relay.Function([], output)
mod['func_600'] = func_600
mod = relay.transform.InferType()(mod)
mutated_mod['func_600'] = func_600
mutated_mod = relay.transform.InferType()(mutated_mod)
func_600_call = mutated_mod.get_global_var('func_600')
call_601 = func_600_call()
output = call_601
func_602 = relay.Function([], output)
mutated_mod['func_602'] = func_602
mutated_mod = relay.transform.InferType()(mutated_mod)
var_638 = relay.var("var_638", dtype = "float32", shape = (2, 15, 15))#candidate|638|(2, 15, 15)|var|float32
uop_639 = relay.log(var_638.astype('float32')) # shape=(2, 15, 15)
uop_643 = relay.rsqrt(uop_639.astype('float64')) # shape=(2, 15, 15)
uop_649 = relay.tan(uop_643.astype('float64')) # shape=(2, 15, 15)
uop_651 = relay.asin(uop_643.astype('float32')) # shape=(2, 15, 15)
bop_654 = relay.logical_xor(uop_639.astype('int16'), relay.reshape(uop_643.astype('int16'), relay.shape_of(uop_639))) # shape=(2, 15, 15)
var_657 = relay.var("var_657", dtype = "int16", shape = (2, 15, 15))#candidate|657|(2, 15, 15)|var|int16
bop_658 = relay.bitwise_xor(bop_654.astype('int64'), relay.reshape(var_657.astype('int64'), relay.shape_of(bop_654))) # shape=(2, 15, 15)
output = relay.Tuple([uop_649,uop_651,bop_658,])
output2 = relay.Tuple([uop_649,uop_651,bop_658,])
func_662 = relay.Function([var_638,var_657,], output)
mod['func_662'] = func_662
mod = relay.transform.InferType()(mod)
var_663 = relay.var("var_663", dtype = "float32", shape = (2, 15, 15))#candidate|663|(2, 15, 15)|var|float32
var_664 = relay.var("var_664", dtype = "int16", shape = (2, 15, 15))#candidate|664|(2, 15, 15)|var|int16
output = func_662(var_663,var_664,)
func_665 = relay.Function([var_663,var_664,], output)
mutated_mod['func_665'] = func_665
mutated_mod = relay.transform.InferType()(mutated_mod)
var_694 = relay.var("var_694", dtype = "uint8", shape = (15, 7))#candidate|694|(15, 7)|var|uint8
const_695 = relay.const([[-7,-8,-10,9,5,3,8],[9,3,7,-4,-3,-7,-2],[10,-2,9,-8,-10,-9,10],[10,3,6,-3,9,5,-8],[-2,-1,-10,-1,-6,8,-5],[1,-8,-3,-10,8,-3,-3],[-1,-3,3,7,-8,4,5],[-9,8,6,-1,-6,10,-6],[-1,1,10,-6,7,-10,-10],[-8,4,-7,-7,8,-2,9],[-6,-4,4,10,-9,-10,-10],[3,3,-8,9,1,6,4],[-2,1,9,-9,-1,8,-7],[-10,9,8,-7,10,10,3],[-9,-8,-3,-2,-2,-7,10]], dtype = "uint8")#candidate|695|(15, 7)|const|uint8
bop_696 = relay.right_shift(var_694.astype('uint8'), relay.reshape(const_695.astype('uint8'), relay.shape_of(var_694))) # shape=(15, 7)
func_268_call = mod.get_global_var('func_268')
func_272_call = mutated_mod.get_global_var('func_272')
var_710 = relay.var("var_710", dtype = "uint64", shape = (660,))#candidate|710|(660,)|var|uint64
call_709 = relay.TupleGetItem(func_268_call(relay.reshape(var_710.astype('uint64'), [11, 5, 12]), relay.reshape(var_710.astype('float32'), [11, 5, 12]), ), 0)
call_711 = relay.TupleGetItem(func_272_call(relay.reshape(var_710.astype('uint64'), [11, 5, 12]), relay.reshape(var_710.astype('float32'), [11, 5, 12]), ), 0)
func_500_call = mod.get_global_var('func_500')
func_501_call = mutated_mod.get_global_var('func_501')
call_712 = relay.TupleGetItem(func_500_call(), 2)
call_713 = relay.TupleGetItem(func_501_call(), 2)
func_209_call = mod.get_global_var('func_209')
func_212_call = mutated_mod.get_global_var('func_212')
var_723 = relay.var("var_723", dtype = "uint32", shape = (30,))#candidate|723|(30,)|var|uint32
call_722 = relay.TupleGetItem(func_209_call(relay.reshape(var_723.astype('uint32'), [10, 3]), relay.reshape(var_723.astype('uint32'), [10, 3]), ), 0)
call_724 = relay.TupleGetItem(func_212_call(relay.reshape(var_723.astype('uint32'), [10, 3]), relay.reshape(var_723.astype('uint32'), [10, 3]), ), 0)
output = relay.Tuple([bop_696,call_709,var_710,call_712,call_722,var_723,])
output2 = relay.Tuple([bop_696,call_711,var_710,call_713,call_724,var_723,])
func_738 = relay.Function([var_694,var_710,var_723,], output)
mod['func_738'] = func_738
mod = relay.transform.InferType()(mod)
var_739 = relay.var("var_739", dtype = "uint8", shape = (15, 7))#candidate|739|(15, 7)|var|uint8
var_740 = relay.var("var_740", dtype = "uint64", shape = (660,))#candidate|740|(660,)|var|uint64
var_741 = relay.var("var_741", dtype = "uint32", shape = (30,))#candidate|741|(30,)|var|uint32
output = func_738(var_739,var_740,var_741,)
func_742 = relay.Function([var_739,var_740,var_741,], output)
mutated_mod['func_742'] = func_742
mutated_mod = relay.transform.InferType()(mutated_mod)
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_744 = func_72_call()
call_745 = func_72_call()
output = call_744
output2 = call_745
func_748 = relay.Function([], output)
mod['func_748'] = func_748
mod = relay.transform.InferType()(mod)
output = func_748()
func_749 = relay.Function([], output)
mutated_mod['func_749'] = func_749
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_763 = relay.TupleGetItem(func_51_call(), 0)
call_764 = relay.TupleGetItem(func_52_call(), 0)
output = call_763
output2 = call_764
func_765 = relay.Function([], output)
mod['func_765'] = func_765
mod = relay.transform.InferType()(mod)
mutated_mod['func_765'] = func_765
mutated_mod = relay.transform.InferType()(mutated_mod)
func_765_call = mutated_mod.get_global_var('func_765')
call_766 = func_765_call()
output = call_766
func_767 = relay.Function([], output)
mutated_mod['func_767'] = func_767
mutated_mod = relay.transform.InferType()(mutated_mod)
var_779 = relay.var("var_779", dtype = "float32", shape = (4, 10))#candidate|779|(4, 10)|var|float32
uop_780 = relay.rsqrt(var_779.astype('float32')) # shape=(4, 10)
bop_790 = relay.not_equal(uop_780.astype('bool'), relay.reshape(var_779.astype('bool'), relay.shape_of(uop_780))) # shape=(4, 10)
uop_798 = relay.sinh(uop_780.astype('float64')) # shape=(4, 10)
bop_803 = relay.greater_equal(uop_798.astype('bool'), relay.reshape(bop_790.astype('bool'), relay.shape_of(uop_798))) # shape=(4, 10)
bop_806 = relay.minimum(uop_798.astype('uint32'), relay.reshape(bop_790.astype('uint32'), relay.shape_of(uop_798))) # shape=(4, 10)
const_809 = relay.const([[-9.656140,6.306932,-0.704761,1.457372,-0.004545,3.166790,7.029885,9.413067,0.100070,7.575068],[-9.660612,-0.685104,3.702546,1.306804,-8.357059,-1.742885,9.377098,0.413025,-4.024321,-7.495780],[4.111587,-3.381314,8.198409,6.878400,5.902559,4.874521,9.685665,0.430521,0.343411,-8.768144],[2.809442,3.708025,3.857787,-5.809395,9.378386,-2.691079,-2.129447,3.036253,1.004996,-8.514880]], dtype = "float64")#candidate|809|(4, 10)|const|float64
bop_810 = relay.logical_or(uop_798.astype('bool'), relay.reshape(const_809.astype('bool'), relay.shape_of(uop_798))) # shape=(4, 10)
func_209_call = mod.get_global_var('func_209')
func_212_call = mutated_mod.get_global_var('func_212')
var_815 = relay.var("var_815", dtype = "uint32", shape = (30,))#candidate|815|(30,)|var|uint32
call_814 = relay.TupleGetItem(func_209_call(relay.reshape(var_815.astype('uint32'), [10, 3]), relay.reshape(var_815.astype('uint32'), [10, 3]), ), 1)
call_816 = relay.TupleGetItem(func_212_call(relay.reshape(var_815.astype('uint32'), [10, 3]), relay.reshape(var_815.astype('uint32'), [10, 3]), ), 1)
func_222_call = mod.get_global_var('func_222')
func_223_call = mutated_mod.get_global_var('func_223')
call_818 = relay.TupleGetItem(func_222_call(), 1)
call_819 = relay.TupleGetItem(func_223_call(), 1)
output = relay.Tuple([bop_803,bop_806,bop_810,call_814,var_815,call_818,])
output2 = relay.Tuple([bop_803,bop_806,bop_810,call_816,var_815,call_819,])
func_827 = relay.Function([var_779,var_815,], output)
mod['func_827'] = func_827
mod = relay.transform.InferType()(mod)
var_828 = relay.var("var_828", dtype = "float32", shape = (4, 10))#candidate|828|(4, 10)|var|float32
var_829 = relay.var("var_829", dtype = "uint32", shape = (30,))#candidate|829|(30,)|var|uint32
output = func_827(var_828,var_829,)
func_830 = relay.Function([var_828,var_829,], output)
mutated_mod['func_830'] = func_830
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_837 = relay.TupleGetItem(func_51_call(), 0)
call_838 = relay.TupleGetItem(func_52_call(), 0)
func_738_call = mod.get_global_var('func_738')
func_742_call = mutated_mod.get_global_var('func_742')
var_864 = relay.var("var_864", dtype = "uint8", shape = (105,))#candidate|864|(105,)|var|uint8
var_865 = relay.var("var_865", dtype = "uint64", shape = (660,))#candidate|865|(660,)|var|uint64
var_866 = relay.var("var_866", dtype = "uint32", shape = (30,))#candidate|866|(30,)|var|uint32
call_863 = relay.TupleGetItem(func_738_call(relay.reshape(var_864.astype('uint8'), [15, 7]), relay.reshape(var_865.astype('uint64'), [660,]), relay.reshape(var_866.astype('uint32'), [30,]), ), 3)
call_867 = relay.TupleGetItem(func_742_call(relay.reshape(var_864.astype('uint8'), [15, 7]), relay.reshape(var_865.astype('uint64'), [660,]), relay.reshape(var_866.astype('uint32'), [30,]), ), 3)
output = relay.Tuple([call_837,call_863,var_864,var_865,var_866,])
output2 = relay.Tuple([call_838,call_867,var_864,var_865,var_866,])
func_879 = relay.Function([var_864,var_865,var_866,], output)
mod['func_879'] = func_879
mod = relay.transform.InferType()(mod)
mutated_mod['func_879'] = func_879
mutated_mod = relay.transform.InferType()(mutated_mod)
func_879_call = mutated_mod.get_global_var('func_879')
var_881 = relay.var("var_881", dtype = "uint8", shape = (105,))#candidate|881|(105,)|var|uint8
var_882 = relay.var("var_882", dtype = "uint64", shape = (660,))#candidate|882|(660,)|var|uint64
var_883 = relay.var("var_883", dtype = "uint32", shape = (30,))#candidate|883|(30,)|var|uint32
call_880 = func_879_call(var_881,var_882,var_883,)
output = call_880
func_884 = relay.Function([var_881,var_882,var_883,], output)
mutated_mod['func_884'] = func_884
mutated_mod = relay.transform.InferType()(mutated_mod)
const_888 = relay.const(9.559716, dtype = "float64")#candidate|888|()|const|float64
const_889 = relay.const([[-2.217760,-4.396021,6.783850,-1.287716,3.991221,5.834983,2.378553,5.919011,0.462157,-7.092485,9.048543,4.906941,5.254933,-0.075796],[-2.855631,0.453888,0.472246,-8.255746,-2.986411,-1.621707,-4.781482,-1.931345,7.510523,-8.508107,-2.209983,5.508594,-0.561749,-7.399705],[-5.249582,5.382077,6.474605,-5.236989,-2.612708,3.353265,9.401541,-5.218591,6.258612,4.818185,-9.209462,8.135951,-7.630681,-4.051287],[4.873200,-7.160945,-5.453559,-3.311381,0.650554,-2.383111,9.060266,-2.814199,9.573448,-7.321540,2.391340,1.106531,5.131984,8.110189],[2.119494,-0.055531,-9.628368,3.194278,-8.886339,-3.401423,-3.360950,-1.654304,9.303070,-9.203770,9.690157,5.769293,6.611738,7.870843],[-8.031836,-0.896168,-7.177079,-4.129682,-2.890473,4.720997,-9.240472,-6.072781,-3.901741,-5.531030,7.557206,8.231788,-0.135781,6.365266],[3.756420,-7.358598,4.530336,-0.691949,-4.402527,9.267966,-4.763773,2.222545,-5.539777,6.985358,-8.008711,-2.647681,-2.452286,9.063777],[8.223415,-4.294044,3.434653,-0.704904,-3.553867,-9.995866,5.485787,2.196471,4.663331,-5.038330,-9.420833,-3.535921,-4.620072,-1.541109],[-3.191094,3.866379,-7.623649,9.720696,-2.406004,3.735556,-2.478451,7.763239,6.465247,4.750520,-9.845231,-6.367255,4.108641,6.532501],[-7.473848,-6.730276,2.007184,-4.789912,-5.643019,7.021522,4.938673,7.095843,2.481202,3.922574,4.554696,4.717604,4.112537,-2.283219],[-0.686566,-2.522216,3.935316,-9.849067,-6.494858,-5.529182,-1.123917,0.294787,-9.463276,4.787436,-6.337925,-5.571705,4.545080,2.462067],[-9.450685,8.527355,7.352900,0.702827,-6.218220,-5.454016,-1.928152,8.191955,-2.109773,-7.804790,7.017232,-8.289809,-3.599941,2.928904],[-4.057593,-7.927617,-8.508117,-2.634319,9.573155,-1.003695,9.173350,-3.216171,-3.039613,-0.600088,8.476806,9.311886,-9.580289,-4.086589],[8.331286,-6.335865,-7.851503,-8.275769,-1.871138,-4.253449,-7.682911,0.129051,0.875060,5.127273,-1.976495,2.483160,-3.439553,-3.733837],[-5.116159,3.031571,2.873277,-5.308992,7.315913,-9.835888,2.943358,5.949156,-0.070274,7.315540,1.744945,5.963437,-5.651092,6.861788],[3.564905,4.096549,-8.349638,4.570330,-3.456007,2.807388,-4.306331,7.020320,-1.253071,1.477694,9.872457,7.663969,3.180839,-5.203069]], dtype = "float64")#candidate|889|(16, 14)|const|float64
bop_890 = relay.floor_mod(const_888.astype('float64'), const_889.astype('float64')) # shape=(16, 14)
output = bop_890
output2 = bop_890
func_894 = relay.Function([], output)
mod['func_894'] = func_894
mod = relay.transform.InferType()(mod)
mutated_mod['func_894'] = func_894
mutated_mod = relay.transform.InferType()(mutated_mod)
func_894_call = mutated_mod.get_global_var('func_894')
call_895 = func_894_call()
output = call_895
func_896 = relay.Function([], output)
mutated_mod['func_896'] = func_896
mutated_mod = relay.transform.InferType()(mutated_mod)
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_901 = func_72_call()
call_902 = func_72_call()
output = relay.Tuple([call_901,])
output2 = relay.Tuple([call_902,])
func_904 = relay.Function([], output)
mod['func_904'] = func_904
mod = relay.transform.InferType()(mod)
mutated_mod['func_904'] = func_904
mutated_mod = relay.transform.InferType()(mutated_mod)
func_904_call = mutated_mod.get_global_var('func_904')
call_905 = func_904_call()
output = call_905
func_906 = relay.Function([], output)
mutated_mod['func_906'] = func_906
mutated_mod = relay.transform.InferType()(mutated_mod)
const_968 = relay.const([[False,True,True,True,True,False,False,True,False,False,False],[True,True,False,False,True,False,True,False,False,False,False],[False,True,True,False,True,False,False,True,False,True,False]], dtype = "bool")#candidate|968|(3, 11)|const|bool
var_969 = relay.var("var_969", dtype = "bool", shape = (3, 11))#candidate|969|(3, 11)|var|bool
bop_970 = relay.logical_and(const_968.astype('bool'), relay.reshape(var_969.astype('bool'), relay.shape_of(const_968))) # shape=(3, 11)
output = bop_970
output2 = bop_970
func_973 = relay.Function([var_969,], output)
mod['func_973'] = func_973
mod = relay.transform.InferType()(mod)
var_974 = relay.var("var_974", dtype = "bool", shape = (3, 11))#candidate|974|(3, 11)|var|bool
output = func_973(var_974)
func_975 = relay.Function([var_974], output)
mutated_mod['func_975'] = func_975
mutated_mod = relay.transform.InferType()(mutated_mod)
func_904_call = mod.get_global_var('func_904')
func_906_call = mutated_mod.get_global_var('func_906')
call_1067 = relay.TupleGetItem(func_904_call(), 0)
call_1068 = relay.TupleGetItem(func_906_call(), 0)
output = relay.Tuple([call_1067,])
output2 = relay.Tuple([call_1068,])
func_1069 = relay.Function([], output)
mod['func_1069'] = func_1069
mod = relay.transform.InferType()(mod)
output = func_1069()
func_1070 = relay.Function([], output)
mutated_mod['func_1070'] = func_1070
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1074 = relay.var("var_1074", dtype = "uint64", shape = (9, 16, 3))#candidate|1074|(9, 16, 3)|var|uint64
const_1075 = relay.const([[[5,2,-8],[-10,5,-8],[-10,1,-3],[-5,-5,5],[-7,-1,2],[7,-6,1],[-6,-6,-4],[3,7,7],[2,-9,6],[5,2,8],[-6,3,6],[7,4,4],[-6,-3,2],[2,-2,1],[6,1,3],[-4,-8,10]],[[1,-7,-9],[-5,9,-9],[-8,7,2],[9,-7,6],[4,7,-6],[-6,-4,-2],[-6,-9,4],[10,-9,-3],[-6,-10,5],[-10,-7,8],[6,10,9],[-3,8,-1],[-2,-4,7],[-5,-2,-7],[3,3,-6],[5,9,2]],[[-7,2,-10],[8,2,-4],[-1,-6,-6],[-7,2,-10],[5,2,3],[1,-10,-5],[-5,-4,-7],[8,7,-9],[6,-3,10],[-7,-9,8],[-5,-1,-1],[5,1,5],[-8,-4,-6],[10,-10,2],[-6,-4,10],[-8,10,-9]],[[-6,-7,4],[3,-4,10],[1,-5,3],[-3,-8,6],[1,-10,3],[-4,-6,6],[-4,3,-7],[9,3,2],[-1,8,4],[-7,8,-7],[-9,-10,1],[-4,-6,-2],[-9,2,5],[1,1,-7],[-3,-1,2],[-10,1,7]],[[-5,-9,-7],[5,-2,-2],[-2,-6,2],[1,8,6],[2,-2,-7],[6,-9,4],[-6,3,-5],[8,10,4],[7,-9,7],[-4,3,-1],[6,-4,-2],[10,7,-5],[-10,2,7],[2,-1,-7],[-8,7,-1],[-2,6,2]],[[4,6,-9],[4,4,-9],[-3,-4,-4],[3,4,-3],[-8,1,-1],[-6,7,6],[2,5,-7],[-8,-7,-7],[1,-3,1],[1,-1,5],[-7,6,-2],[-2,-2,2],[-1,-4,-3],[7,2,-1],[6,-2,4],[8,10,-5]],[[-6,-9,-2],[-10,3,1],[-4,10,-10],[6,-2,-4],[8,6,-4],[10,-8,8],[2,-4,1],[-1,-6,2],[-1,-4,9],[-9,-10,2],[-8,5,1],[-5,-3,-6],[-4,1,7],[8,2,-10],[1,-5,-6],[4,9,-9]],[[3,3,-3],[-10,7,7],[-5,-3,8],[-9,5,-8],[2,7,1],[4,-10,-8],[8,-7,9],[1,2,2],[-7,9,-4],[-3,10,10],[5,-5,-10],[5,-3,5],[10,-8,7],[-2,-9,-6],[-9,1,-10],[3,2,-8]],[[3,2,-10],[9,-4,2],[-7,6,7],[-10,3,-8],[-10,-6,-10],[-9,-2,10],[10,9,6],[-5,-6,-8],[-5,1,-10],[10,-9,6],[-3,-4,-8],[6,-4,1],[-8,5,-9],[9,3,-3],[9,7,-10],[-9,8,9]]], dtype = "uint64")#candidate|1075|(9, 16, 3)|const|uint64
bop_1076 = relay.less_equal(var_1074.astype('bool'), relay.reshape(const_1075.astype('bool'), relay.shape_of(var_1074))) # shape=(9, 16, 3)
output = bop_1076
output2 = bop_1076
func_1088 = relay.Function([var_1074,], output)
mod['func_1088'] = func_1088
mod = relay.transform.InferType()(mod)
mutated_mod['func_1088'] = func_1088
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1089 = relay.var("var_1089", dtype = "uint64", shape = (9, 16, 3))#candidate|1089|(9, 16, 3)|var|uint64
func_1088_call = mutated_mod.get_global_var('func_1088')
call_1090 = func_1088_call(var_1089)
output = call_1090
func_1091 = relay.Function([var_1089], output)
mutated_mod['func_1091'] = func_1091
mutated_mod = relay.transform.InferType()(mutated_mod)
func_894_call = mod.get_global_var('func_894')
func_896_call = mutated_mod.get_global_var('func_896')
call_1100 = func_894_call()
call_1101 = func_894_call()
output = relay.Tuple([call_1100,])
output2 = relay.Tuple([call_1101,])
func_1104 = relay.Function([], output)
mod['func_1104'] = func_1104
mod = relay.transform.InferType()(mod)
output = func_1104()
func_1105 = relay.Function([], output)
mutated_mod['func_1105'] = func_1105
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1138 = relay.var("var_1138", dtype = "float64", shape = (8,))#candidate|1138|(8,)|var|float64
uop_1139 = relay.acos(var_1138.astype('float64')) # shape=(8,)
bop_1141 = relay.right_shift(uop_1139.astype('int64'), relay.reshape(var_1138.astype('int64'), relay.shape_of(uop_1139))) # shape=(8,)
func_444_call = mod.get_global_var('func_444')
func_445_call = mutated_mod.get_global_var('func_445')
call_1150 = func_444_call()
call_1151 = func_444_call()
output = relay.Tuple([bop_1141,call_1150,])
output2 = relay.Tuple([bop_1141,call_1151,])
func_1157 = relay.Function([var_1138,], output)
mod['func_1157'] = func_1157
mod = relay.transform.InferType()(mod)
mutated_mod['func_1157'] = func_1157
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1158 = relay.var("var_1158", dtype = "float64", shape = (8,))#candidate|1158|(8,)|var|float64
func_1157_call = mutated_mod.get_global_var('func_1157')
call_1159 = func_1157_call(var_1158)
output = call_1159
func_1160 = relay.Function([var_1158], output)
mutated_mod['func_1160'] = func_1160
mutated_mod = relay.transform.InferType()(mutated_mod)
func_515_call = mod.get_global_var('func_515')
func_517_call = mutated_mod.get_global_var('func_517')
call_1195 = func_515_call()
call_1196 = func_515_call()
output = call_1195
output2 = call_1196
func_1197 = relay.Function([], output)
mod['func_1197'] = func_1197
mod = relay.transform.InferType()(mod)
output = func_1197()
func_1198 = relay.Function([], output)
mutated_mod['func_1198'] = func_1198
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1205 = relay.const([8.117030,7.695069,6.963819,-2.479255,-8.989933,3.869922,-7.333601,-3.032387,0.716614,-2.448965,-9.599028,1.676340,6.380418,2.521668,1.276788], dtype = "float32")#candidate|1205|(15,)|const|float32
uop_1206 = relay.tan(const_1205.astype('float32')) # shape=(15,)
func_1069_call = mod.get_global_var('func_1069')
func_1070_call = mutated_mod.get_global_var('func_1070')
call_1226 = relay.TupleGetItem(func_1069_call(), 0)
call_1227 = relay.TupleGetItem(func_1070_call(), 0)
func_1088_call = mod.get_global_var('func_1088')
func_1091_call = mutated_mod.get_global_var('func_1091')
var_1229 = relay.var("var_1229", dtype = "uint64", shape = (432,))#candidate|1229|(432,)|var|uint64
call_1228 = func_1088_call(relay.reshape(var_1229.astype('uint64'), [9, 16, 3]))
call_1230 = func_1088_call(relay.reshape(var_1229.astype('uint64'), [9, 16, 3]))
output = relay.Tuple([uop_1206,call_1226,call_1228,var_1229,])
output2 = relay.Tuple([uop_1206,call_1227,call_1230,var_1229,])
func_1232 = relay.Function([var_1229,], output)
mod['func_1232'] = func_1232
mod = relay.transform.InferType()(mod)
mutated_mod['func_1232'] = func_1232
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1233 = relay.var("var_1233", dtype = "uint64", shape = (432,))#candidate|1233|(432,)|var|uint64
func_1232_call = mutated_mod.get_global_var('func_1232')
call_1234 = func_1232_call(var_1233)
output = call_1234
func_1235 = relay.Function([var_1233], output)
mutated_mod['func_1235'] = func_1235
mutated_mod = relay.transform.InferType()(mutated_mod)
func_748_call = mod.get_global_var('func_748')
func_749_call = mutated_mod.get_global_var('func_749')
call_1241 = func_748_call()
call_1242 = func_748_call()
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_1258 = func_72_call()
call_1259 = func_72_call()
func_1197_call = mod.get_global_var('func_1197')
func_1198_call = mutated_mod.get_global_var('func_1198')
call_1262 = func_1197_call()
call_1263 = func_1197_call()
output = relay.Tuple([call_1241,call_1258,call_1262,])
output2 = relay.Tuple([call_1242,call_1259,call_1263,])
func_1267 = relay.Function([], output)
mod['func_1267'] = func_1267
mod = relay.transform.InferType()(mod)
output = func_1267()
func_1268 = relay.Function([], output)
mutated_mod['func_1268'] = func_1268
mutated_mod = relay.transform.InferType()(mutated_mod)
func_72_call = mod.get_global_var('func_72')
func_74_call = mutated_mod.get_global_var('func_74')
call_1293 = func_72_call()
call_1294 = func_72_call()
output = relay.Tuple([call_1293,])
output2 = relay.Tuple([call_1294,])
func_1316 = relay.Function([], output)
mod['func_1316'] = func_1316
mod = relay.transform.InferType()(mod)
mutated_mod['func_1316'] = func_1316
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1316_call = mutated_mod.get_global_var('func_1316')
call_1317 = func_1316_call()
output = call_1317
func_1318 = relay.Function([], output)
mutated_mod['func_1318'] = func_1318
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1354 = relay.var("var_1354", dtype = "float32", shape = (6, 16))#candidate|1354|(6, 16)|var|float32
var_1355 = relay.var("var_1355", dtype = "float32", shape = (6, 16))#candidate|1355|(6, 16)|var|float32
bop_1356 = relay.power(var_1354.astype('float32'), relay.reshape(var_1355.astype('float32'), relay.shape_of(var_1354))) # shape=(6, 16)
var_1359 = relay.var("var_1359", dtype = "float32", shape = (6, 16))#candidate|1359|(6, 16)|var|float32
bop_1360 = relay.less_equal(bop_1356.astype('bool'), relay.reshape(var_1359.astype('bool'), relay.shape_of(bop_1356))) # shape=(6, 16)
output = bop_1360
output2 = bop_1360
func_1369 = relay.Function([var_1354,var_1355,var_1359,], output)
mod['func_1369'] = func_1369
mod = relay.transform.InferType()(mod)
var_1370 = relay.var("var_1370", dtype = "float32", shape = (6, 16))#candidate|1370|(6, 16)|var|float32
var_1371 = relay.var("var_1371", dtype = "float32", shape = (6, 16))#candidate|1371|(6, 16)|var|float32
var_1372 = relay.var("var_1372", dtype = "float32", shape = (6, 16))#candidate|1372|(6, 16)|var|float32
output = func_1369(var_1370,var_1371,var_1372,)
func_1373 = relay.Function([var_1370,var_1371,var_1372,], output)
mutated_mod['func_1373'] = func_1373
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1378 = relay.var("var_1378", dtype = "int8", shape = (4, 1, 10))#candidate|1378|(4, 1, 10)|var|int8
var_1379 = relay.var("var_1379", dtype = "int8", shape = (4, 15, 10))#candidate|1379|(4, 15, 10)|var|int8
bop_1380 = relay.minimum(var_1378.astype('int8'), var_1379.astype('int8')) # shape=(4, 15, 10)
bop_1383 = relay.power(bop_1380.astype('float32'), relay.reshape(var_1379.astype('float32'), relay.shape_of(bop_1380))) # shape=(4, 15, 10)
output = relay.Tuple([bop_1383,])
output2 = relay.Tuple([bop_1383,])
func_1387 = relay.Function([var_1378,var_1379,], output)
mod['func_1387'] = func_1387
mod = relay.transform.InferType()(mod)
var_1388 = relay.var("var_1388", dtype = "int8", shape = (4, 1, 10))#candidate|1388|(4, 1, 10)|var|int8
var_1389 = relay.var("var_1389", dtype = "int8", shape = (4, 15, 10))#candidate|1389|(4, 15, 10)|var|int8
output = func_1387(var_1388,var_1389,)
func_1390 = relay.Function([var_1388,var_1389,], output)
mutated_mod['func_1390'] = func_1390
mutated_mod = relay.transform.InferType()(mutated_mod)
func_765_call = mod.get_global_var('func_765')
func_767_call = mutated_mod.get_global_var('func_767')
call_1448 = func_765_call()
call_1449 = func_765_call()
func_662_call = mod.get_global_var('func_662')
func_665_call = mutated_mod.get_global_var('func_665')
const_1452 = relay.const([-5.234751,9.774915,4.015990,-7.581704,-2.568389,6.090659,5.860279,-9.926090,6.468451,-1.850554,6.163852,3.769774,8.178557,-3.576400,3.386337,8.462768,7.817165,-5.547596,-1.530785,-1.744697,7.024299,-8.722497,-3.329229,-7.900872,-7.084924,-2.125314,-3.285245,8.556891,8.093804,-6.924855,-2.064886,-2.127779,-5.336914,-4.635156,4.697848,7.812031,9.756888,8.511837,-9.062979,9.967426,-7.911690,4.505059,7.706611,8.695073,-3.342304,-4.933217,0.234026,-1.718001,6.267896,-4.836855,8.250853,2.249365,-9.329091,-1.822807,-5.309564,1.164234,-7.034710,8.779700,-0.077772,-0.866851,-0.670602,6.543940,-0.225841,-6.041555,4.327484,-3.430340,-2.235251,4.959152,9.964055,-3.554772,9.649626,-3.218111,-6.139993,-5.313729,-4.535598,2.852425,1.317471,-6.067118,-9.534945,2.748146,-5.327937,5.473483,-2.172754,4.968779,-2.677853,8.847452,4.638824,6.597053,-5.842944,-2.503714,8.430478,4.308769,5.145969,-5.123928,-9.042749,9.154698,2.313814,-5.271242,9.781585,3.319787,6.581882,1.601275,4.214676,-3.295374,-8.760922,-2.456337,-1.932450,-4.191826,-4.048234,6.715866,0.759196,7.915090,7.527320,-0.863142,4.466651,5.953610,-8.878447,0.123867,3.480698,6.464993,5.287930,0.613140,-4.442988,2.393148,-1.811189,4.169651,3.130600,6.095572,-8.104684,5.417118,0.543448,-9.950067,1.765553,9.440059,7.132051,3.872087,2.276544,-1.631038,-5.366177,-9.831283,5.328972,-4.274491,-0.654584,9.938310,6.297493,-6.639432,0.060440,5.137298,-4.555697,-8.139414,9.781734,-0.180326,-4.991299,-8.023079,-9.692028,-2.798067,4.972378,-3.229881,-1.114799,-0.101200,-7.128860,-8.693358,-4.422689,6.810112,7.695936,-1.958459,5.150144,4.943111,-8.162924,-8.310540,-4.707583,-8.350983,-1.494686,8.114896,-9.030697,-9.655706,-2.204833,1.213878,-6.803526,2.924557,-0.182827,8.143392,5.461154,2.316136,-9.075625,-7.252125,-5.333821,-7.525344,-5.767637,1.250180,-8.728241,2.681236,6.236532,3.748413,2.117845,4.657806,-1.440952,5.325201,2.937047,8.261324,2.138920,-0.909199,-9.369550,-0.320867,-1.990055,-0.459313,5.033590,0.402679,-6.934682,2.742017,6.896539,2.679945,2.062927,0.539553,2.411137,-3.004071,4.450095,5.991461,-9.433040,-1.923729,4.598500,6.407356,-0.164675,0.003679,2.020772,9.930729,0.196370,-9.106978,1.587174,1.110203,6.072582,3.651472,-1.702724,-0.507425,-0.053563,2.072390,9.221092,0.989492,8.886341,-8.375086,9.228762,-2.419496,-7.505518,4.102968,-1.518627,4.305761,-6.318997,-6.278124,3.121007,7.279528,8.393667,6.078459,2.815692,6.105519,-5.082419,0.160046,5.771066,-4.477018,4.281304,3.820969,8.049174,0.351210,-9.669980,-2.914447,-6.051117,9.342817,7.579350,1.458007,4.289834,0.481883,-8.906895,9.740639,-9.580234,-6.898463,0.650444,-8.968481,-3.774269,0.869557,2.965706,-9.087256,5.828638,-4.070265,-9.716189,-9.953296,-0.182308,9.135964,-2.332604,-1.368788,1.565847,-4.597952,-9.967261,4.578815,-7.550339,7.767854,-5.703158,-9.994035,-4.090465,-2.410203,-7.086275,-2.744666,2.514437,4.256432,4.103786,-6.188813,-3.133086,6.721835,8.555252,4.307377,7.996788,-6.999819,-2.924356,-7.379670,2.464749,-8.665319,9.911846,-1.364569,-2.212770,-5.105706,7.021681,-8.330046,-1.079321,-9.495318,-9.552201,5.503743,6.307785,-5.905563,-0.122853,-1.761697,1.979044,1.123422,5.037711,9.373796,-5.845898,1.459904,0.254947,4.631105,-5.729899,9.182683,-6.406287,-5.598458,-5.199708,7.840176,3.482849,-8.059382,-9.620278,4.716193,1.786765,-5.429244,2.552414,4.933120,6.377824,-1.957391,7.075852,2.572226,-6.270822,0.723082,-4.115218,-6.955021,-9.427845,-5.423473,-5.565776,-4.643892,6.359610,2.490168,8.037506,2.662789,-1.700615,-1.449444,6.235526,1.197198,5.309080,0.634183,0.014398,-0.302061,-3.269508,3.354856,-3.172196,9.979020,-3.091431,-6.412682,0.774709,-3.910602,5.147038,9.171082,-2.783346,1.246588,1.208106,6.415849,-5.795470,-9.286188,8.872940,9.312096,-4.850217,8.480954,-0.637051,-3.923815,0.258705,1.852104,0.852748,4.399802,3.884416,1.153143,1.014759,7.494766,-8.948963,4.662349,9.725161,-1.860003,5.091160,-8.608082,1.815583,4.503673,-6.010105,-7.893288,-1.138638,9.715583,-9.969241,-9.691522,-5.444522,8.887113,-1.144479,-2.816727,9.773887,8.517139,-6.869156,-8.877581,-5.966996,-3.694782,-9.348310,-5.395964,-1.067145,-0.420794,-4.843507,1.102180,4.007406,-3.368731,8.639730,0.380562,6.609985,3.404092,8.181633,6.925158,5.864289,-6.087062,-9.367721,-4.924126,-3.608414,0.344720,-8.853435,9.030731], dtype = "float32")#candidate|1452|(450,)|const|float32
call_1451 = relay.TupleGetItem(func_662_call(relay.reshape(const_1452.astype('float32'), [2, 15, 15]), relay.reshape(const_1452.astype('int16'), [2, 15, 15]), ), 1)
call_1453 = relay.TupleGetItem(func_665_call(relay.reshape(const_1452.astype('float32'), [2, 15, 15]), relay.reshape(const_1452.astype('int16'), [2, 15, 15]), ), 1)
output = relay.Tuple([call_1448,call_1451,const_1452,])
output2 = relay.Tuple([call_1449,call_1453,const_1452,])
func_1455 = relay.Function([], output)
mod['func_1455'] = func_1455
mod = relay.transform.InferType()(mod)
mutated_mod['func_1455'] = func_1455
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1455_call = mutated_mod.get_global_var('func_1455')
call_1456 = func_1455_call()
output = call_1456
func_1457 = relay.Function([], output)
mutated_mod['func_1457'] = func_1457
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1458 = relay.const([[[-3.491111,8.406223,9.421872,-2.133096,7.557040,-5.565956,-6.988292,-0.335229],[8.403185,-0.801574,8.912123,-5.132789,9.534306,-0.115375,-6.460245,4.145404]],[[7.450753,7.818889,4.213668,-8.012958,-1.929773,-1.264645,3.806818,-1.893611],[-4.298642,-0.776762,4.829232,2.457742,1.986131,1.866405,-8.950083,-5.677894]],[[6.460431,-0.528300,2.543358,5.825063,8.489664,4.871556,-0.259934,-8.714772],[-0.997229,-8.638553,7.282260,-6.313882,4.572777,-7.501989,-9.737163,-9.333580]],[[7.296016,-5.220788,4.895385,-3.726549,2.704562,6.782337,4.662288,-9.175171],[4.847487,5.738455,-8.136915,9.940352,9.095696,7.830022,2.157926,3.675871]],[[3.623165,0.866300,8.656593,1.353649,8.234684,-2.115699,-6.308879,-4.930350],[-9.783055,-5.821723,3.072766,-3.761905,-4.999837,-5.339895,9.427296,-5.711375]],[[5.742054,7.867974,8.847563,6.009067,-9.683658,2.168194,-8.578799,-7.746722],[1.681576,-0.821701,-9.837818,-2.270713,-2.445622,-1.528960,-3.436668,-5.830045]],[[3.251189,1.151426,8.033509,5.130592,-7.504552,-7.408510,5.159965,3.031011],[-9.294529,5.708526,1.832090,3.750070,3.408515,7.947057,-4.978105,9.879932]],[[-7.508107,-8.342938,-5.054091,7.263384,-5.044443,1.018626,-5.511736,-8.647533],[-3.946463,7.086185,0.436675,-8.113706,2.743568,5.959190,7.455702,9.180705]],[[0.734791,8.938747,1.522065,-3.601599,5.792499,6.737132,7.191143,-8.795041],[-4.646971,6.187974,-0.330220,-4.361984,4.751411,4.950340,-4.515650,-8.770587]],[[-0.371342,8.051490,-5.024494,8.650065,-6.698976,4.386574,-9.646714,1.584684],[-7.169061,8.317518,-1.415060,-8.207188,4.426475,3.869841,-7.667123,9.198168]],[[2.148960,-1.360157,-8.730135,9.692483,1.737358,-9.639360,-4.671279,-8.772245],[-2.747157,2.350934,6.987377,4.366479,2.476955,-9.745299,-7.572953,-3.726586]],[[9.885711,-3.874224,-2.654608,-2.395603,-7.823905,-1.247927,-4.466242,0.399345],[-3.938336,-6.504147,-2.974681,-0.318840,-0.238310,1.698477,-6.912081,-7.661639]],[[4.762720,4.569273,2.489976,8.479074,7.846024,-0.664583,-6.687367,-0.443625],[0.539767,-5.554526,-7.876763,4.543342,7.000517,-1.344655,0.354022,7.887509]],[[0.543234,-8.362391,-0.884599,-5.223221,8.298398,-8.589779,5.972003,-8.109981],[2.157090,-2.470334,0.412106,-9.679430,4.647700,-5.346673,2.810809,5.493554]]], dtype = "float32")#candidate|1458|(14, 2, 8)|const|float32
uop_1459 = relay.log(const_1458.astype('float32')) # shape=(14, 2, 8)
bop_1474 = relay.add(uop_1459.astype('uint16'), relay.reshape(const_1458.astype('uint16'), relay.shape_of(uop_1459))) # shape=(14, 2, 8)
uop_1482 = relay.asinh(const_1458.astype('float64')) # shape=(14, 2, 8)
bop_1488 = relay.bitwise_and(bop_1474.astype('int8'), relay.reshape(const_1458.astype('int8'), relay.shape_of(bop_1474))) # shape=(14, 2, 8)
bop_1492 = relay.less(uop_1482.astype('bool'), relay.reshape(uop_1459.astype('bool'), relay.shape_of(uop_1482))) # shape=(14, 2, 8)
const_1496 = relay.const([[[-7,-2,-2,8,-8,2,4,-10],[8,8,6,-7,5,8,2,-5]],[[6,-10,2,6,5,-9,-9,-7],[-6,-7,10,-1,6,6,5,8]],[[9,-10,8,-3,7,-1,-5,-5],[5,5,-8,2,-8,-5,-2,-6]],[[1,-3,1,-1,10,-5,2,5],[-4,5,-9,-7,9,-9,4,-3]],[[-1,-1,4,1,6,10,7,-5],[8,-4,-9,3,2,-9,9,-1]],[[-8,-3,4,8,-1,8,-6,1],[-5,10,-6,1,6,6,10,8]],[[-10,4,-6,-6,3,-6,-8,-2],[2,4,6,7,5,-9,-1,-5]],[[-9,-6,1,3,5,10,6,4],[-5,-5,-4,-3,6,-4,-1,7]],[[-7,-5,3,2,-7,1,6,1],[-4,2,-9,-6,7,3,10,7]],[[2,-8,10,-5,4,-10,1,-4],[10,-4,-1,5,2,-7,7,-2]],[[8,6,-9,4,-4,6,10,2],[-5,10,-7,-7,4,-3,-10,1]],[[-6,3,-5,1,-5,-7,-6,-4],[5,9,10,3,10,8,3,-5]],[[6,4,-8,3,1,-8,-8,10],[-8,-5,4,4,-4,-1,2,5]],[[5,9,4,7,-1,4,9,3],[10,-6,8,8,4,10,-7,-4]]], dtype = "int8")#candidate|1496|(14, 2, 8)|const|int8
bop_1497 = relay.floor_mod(bop_1488.astype('float64'), relay.reshape(const_1496.astype('float64'), relay.shape_of(bop_1488))) # shape=(14, 2, 8)
bop_1501 = relay.left_shift(bop_1492.astype('int32'), relay.reshape(bop_1497.astype('int32'), relay.shape_of(bop_1492))) # shape=(14, 2, 8)
uop_1504 = relay.log10(bop_1497.astype('float64')) # shape=(14, 2, 8)
uop_1508 = relay.erf(uop_1504.astype('float64')) # shape=(14, 2, 8)
var_1511 = relay.var("var_1511", dtype = "float64", shape = (14, 2, 8))#candidate|1511|(14, 2, 8)|var|float64
bop_1512 = relay.right_shift(uop_1504.astype('uint8'), relay.reshape(var_1511.astype('uint8'), relay.shape_of(uop_1504))) # shape=(14, 2, 8)
output = relay.Tuple([bop_1501,uop_1508,bop_1512,])
output2 = relay.Tuple([bop_1501,uop_1508,bop_1512,])
func_1515 = relay.Function([var_1511,], output)
mod['func_1515'] = func_1515
mod = relay.transform.InferType()(mod)
var_1516 = relay.var("var_1516", dtype = "float64", shape = (14, 2, 8))#candidate|1516|(14, 2, 8)|var|float64
output = func_1515(var_1516)
func_1517 = relay.Function([var_1516], output)
mutated_mod['func_1517'] = func_1517
mutated_mod = relay.transform.InferType()(mutated_mod)
func_894_call = mod.get_global_var('func_894')
func_896_call = mutated_mod.get_global_var('func_896')
call_1563 = func_894_call()
call_1564 = func_894_call()
uop_1568 = relay.tan(call_1563.astype('float32')) # shape=(16, 14)
uop_1570 = relay.tan(call_1564.astype('float32')) # shape=(16, 14)
output = uop_1568
output2 = uop_1570
func_1572 = relay.Function([], output)
mod['func_1572'] = func_1572
mod = relay.transform.InferType()(mod)
output = func_1572()
func_1573 = relay.Function([], output)
mutated_mod['func_1573'] = func_1573
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1198_call = mutated_mod.get_global_var('func_1198')
call_1604 = func_1197_call()
call_1605 = func_1197_call()
func_1232_call = mod.get_global_var('func_1232')
func_1235_call = mutated_mod.get_global_var('func_1235')
var_1616 = relay.var("var_1616", dtype = "uint64", shape = (432,))#candidate|1616|(432,)|var|uint64
call_1615 = relay.TupleGetItem(func_1232_call(relay.reshape(var_1616.astype('uint64'), [432,])), 1)
call_1617 = relay.TupleGetItem(func_1235_call(relay.reshape(var_1616.astype('uint64'), [432,])), 1)
var_1644 = relay.var("var_1644", dtype = "uint64", shape = (432,))#candidate|1644|(432,)|var|uint64
bop_1645 = relay.greater(var_1616.astype('bool'), relay.reshape(var_1644.astype('bool'), relay.shape_of(var_1616))) # shape=(432,)
output = relay.Tuple([call_1604,call_1615,bop_1645,])
output2 = relay.Tuple([call_1605,call_1617,bop_1645,])
func_1649 = relay.Function([var_1616,var_1644,], output)
mod['func_1649'] = func_1649
mod = relay.transform.InferType()(mod)
var_1650 = relay.var("var_1650", dtype = "uint64", shape = (432,))#candidate|1650|(432,)|var|uint64
var_1651 = relay.var("var_1651", dtype = "uint64", shape = (432,))#candidate|1651|(432,)|var|uint64
output = func_1649(var_1650,var_1651,)
func_1652 = relay.Function([var_1650,var_1651,], output)
mutated_mod['func_1652'] = func_1652
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1455_call = mod.get_global_var('func_1455')
func_1457_call = mutated_mod.get_global_var('func_1457')
call_1664 = relay.TupleGetItem(func_1455_call(), 0)
call_1665 = relay.TupleGetItem(func_1457_call(), 0)
output = call_1664
output2 = call_1665
func_1666 = relay.Function([], output)
mod['func_1666'] = func_1666
mod = relay.transform.InferType()(mod)
mutated_mod['func_1666'] = func_1666
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1666_call = mutated_mod.get_global_var('func_1666')
call_1667 = func_1666_call()
output = call_1667
func_1668 = relay.Function([], output)
mutated_mod['func_1668'] = func_1668
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1681 = relay.var("var_1681", dtype = "float32", shape = ())#candidate|1681|()|var|float32
var_1682 = relay.var("var_1682", dtype = "float32", shape = (10, 9))#candidate|1682|(10, 9)|var|float32
bop_1683 = relay.mod(var_1681.astype('float32'), var_1682.astype('float32')) # shape=(10, 9)
func_209_call = mod.get_global_var('func_209')
func_212_call = mutated_mod.get_global_var('func_212')
var_1693 = relay.var("var_1693", dtype = "uint32", shape = (30,))#candidate|1693|(30,)|var|uint32
call_1692 = relay.TupleGetItem(func_209_call(relay.reshape(var_1693.astype('uint32'), [10, 3]), relay.reshape(var_1693.astype('uint32'), [10, 3]), ), 0)
call_1694 = relay.TupleGetItem(func_212_call(relay.reshape(var_1693.astype('uint32'), [10, 3]), relay.reshape(var_1693.astype('uint32'), [10, 3]), ), 0)
func_367_call = mod.get_global_var('func_367')
func_369_call = mutated_mod.get_global_var('func_369')
var_1707 = relay.var("var_1707", dtype = "uint64", shape = (660,))#candidate|1707|(660,)|var|uint64
call_1706 = relay.TupleGetItem(func_367_call(relay.reshape(var_1707.astype('uint64'), [110, 6])), 2)
call_1708 = relay.TupleGetItem(func_369_call(relay.reshape(var_1707.astype('uint64'), [110, 6])), 2)
uop_1717 = relay.acosh(call_1692.astype('float32')) # shape=(10, 3)
uop_1719 = relay.acosh(call_1694.astype('float32')) # shape=(10, 3)
bop_1726 = relay.logical_and(uop_1717.astype('bool'), relay.reshape(var_1693.astype('bool'), relay.shape_of(uop_1717))) # shape=(10, 3)
bop_1729 = relay.logical_and(uop_1719.astype('bool'), relay.reshape(var_1693.astype('bool'), relay.shape_of(uop_1719))) # shape=(10, 3)
uop_1731 = relay.asin(call_1692.astype('float64')) # shape=(10, 3)
uop_1733 = relay.asin(call_1694.astype('float64')) # shape=(10, 3)
bop_1736 = relay.subtract(uop_1717.astype('int64'), var_1681.astype('int64')) # shape=(10, 3)
bop_1739 = relay.subtract(uop_1719.astype('int64'), var_1681.astype('int64')) # shape=(10, 3)
func_1387_call = mod.get_global_var('func_1387')
func_1390_call = mutated_mod.get_global_var('func_1390')
var_1741 = relay.var("var_1741", dtype = "int8", shape = (40,))#candidate|1741|(40,)|var|int8
const_1742 = relay.const([-10,10,-3,-8,-5,10,-4,10,-7,7,2,-3,10,-5,6,-8,7,10,4,8,6,9,-2,-6,-8,2,5,-3,7,-9,-10,-9,-6,-4,-6,-10,-2,-2,10,7,3,9,5,4,2,4,2,-8,6,-1,-6,-7,-9,-3,7,8,9,9,-9,-3,8,-2,5,-1,2,3,10,-6,-9,10,9,7,7,-10,-8,-7,4,-9,9,-10,-1,-1,-5,3,-6,-1,3,4,3,-6,1,-9,5,4,9,7,10,-4,4,10,-10,-8,-10,-3,-8,-6,-4,-4,7,5,5,-8,1,-2,7,9,-5,-10,-9,8,2,6,10,-5,9,3,-5,9,-10,4,-4,-7,-10,-4,8,10,-5,-7,8,4,-3,1,9,-3,2,-8,-10,5,-10,3,-10,-1,9,2,10,-5,4,-1,6,3,3,6,4,7,2,-5,3,7,-4,9,6,6,-10,4,3,7,7,-3,-7,-9,10,-9,-10,7,7,7,-2,3,2,-1,3,2,1,2,-6,7,5,-1,4,10,-5,-5,1,-2,-7,9,1,9,9,-2,-2,-2,-8,-3,-10,6,-9,-6,-10,1,-1,7,-4,-4,10,7,-6,1,2,8,5,1,4,-2,6,-2,-6,9,-5,-2,-10,7,7,-1,1,3,3,10,9,-2,7,-2,-10,4,5,-10,-4,4,-4,10,-6,1,-7,7,-3,3,-10,3,-7,7,3,-10,4,-10,-10,-4,-3,6,-6,-6,-9,-8,-5,2,10,-5,-7,-1,8,1,-2,-2,2,5,-10,-4,1,1,3,-7,3,5,6,2,7,-1,10,4,-1,9,-1,2,5,9,5,2,-9,-9,-4,8,-6,-2,2,10,6,-5,-2,-10,-10,-8,6,-4,-8,-9,-8,8,4,2,-4,1,2,7,-3,-3,-8,10,1,-7,9,-3,-3,-10,-1,-4,5,-6,-1,10,-2,9,-2,-1,2,-3,-4,-5,9,5,-8,-4,-4,9,-3,-1,10,4,8,-6,-5,-1,-5,-7,2,4,-1,-10,-9,10,7,-9,-1,10,10,10,-5,8,7,-4,-9,8,7,5,-6,9,4,10,-5,-5,-6,-2,-7,-1,-8,-5,3,5,1,5,-4,-3,-6,-6,-5,6,7,2,-10,10,8,-3,9,1,9,-9,2,-3,-5,9,-1,1,5,-2,4,3,7,3,5,6,-9,10,7,2,-4,3,-8,-3,-8,7,9,4,-3,4,10,-5,6,-2,9,-6,7,-6,4,7,4,-3,-2,-1,-5,-7,4,6,5,5,-4,-7,-2,7,4,-9,-7,-3,3,-6,-8,5,-4,4,-1,-5,1,7,7,-9,-7,1,-3,5,-8,-4,3,-8,10,-6,2,9,3,5,9,3,4,-2,-6,-1,-7,1,-4,-4,-3,-7,-2,-8,-2,9,9,-1,-5,10,-9,8,-5,5,7,4,-10,7,-9,8,-8,-5,8,9,-9,5,-7,-6,-1,2,2,2,-8,5,1,-1,2,-5,-3,4,-3,-2,3,5,-10,-3,-2,4,3,-9,3,-3,-7,-5,-3,2,1,8,-1,7,-7,-2,7,-3,5,7,4,-3,8,-1,1,9,-7,-4], dtype = "int8")#candidate|1742|(600,)|const|int8
call_1740 = relay.TupleGetItem(func_1387_call(relay.reshape(var_1741.astype('int8'), [4, 1, 10]), relay.reshape(const_1742.astype('int8'), [4, 15, 10]), ), 0)
call_1743 = relay.TupleGetItem(func_1390_call(relay.reshape(var_1741.astype('int8'), [4, 1, 10]), relay.reshape(const_1742.astype('int8'), [4, 15, 10]), ), 0)
uop_1744 = relay.atanh(bop_1736.astype('float32')) # shape=(10, 3)
uop_1746 = relay.atanh(bop_1739.astype('float32')) # shape=(10, 3)
func_600_call = mod.get_global_var('func_600')
func_602_call = mutated_mod.get_global_var('func_602')
call_1751 = relay.TupleGetItem(func_600_call(), 1)
call_1752 = relay.TupleGetItem(func_602_call(), 1)
output = relay.Tuple([bop_1683,call_1706,var_1707,bop_1726,uop_1731,call_1740,var_1741,const_1742,uop_1744,call_1751,])
output2 = relay.Tuple([bop_1683,call_1708,var_1707,bop_1729,uop_1733,call_1743,var_1741,const_1742,uop_1746,call_1752,])
func_1754 = relay.Function([var_1681,var_1682,var_1693,var_1707,var_1741,], output)
mod['func_1754'] = func_1754
mod = relay.transform.InferType()(mod)
var_1755 = relay.var("var_1755", dtype = "float32", shape = ())#candidate|1755|()|var|float32
var_1756 = relay.var("var_1756", dtype = "float32", shape = (10, 9))#candidate|1756|(10, 9)|var|float32
var_1757 = relay.var("var_1757", dtype = "uint32", shape = (30,))#candidate|1757|(30,)|var|uint32
var_1758 = relay.var("var_1758", dtype = "uint64", shape = (660,))#candidate|1758|(660,)|var|uint64
var_1759 = relay.var("var_1759", dtype = "int8", shape = (40,))#candidate|1759|(40,)|var|int8
output = func_1754(var_1755,var_1756,var_1757,var_1758,var_1759,)
func_1760 = relay.Function([var_1755,var_1756,var_1757,var_1758,var_1759,], output)
mutated_mod['func_1760'] = func_1760
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1783 = relay.var("var_1783", dtype = "int16", shape = (15, 6, 9))#candidate|1783|(15, 6, 9)|var|int16
var_1784 = relay.var("var_1784", dtype = "int16", shape = (15, 6, 9))#candidate|1784|(15, 6, 9)|var|int16
bop_1785 = relay.minimum(var_1783.astype('int16'), relay.reshape(var_1784.astype('int16'), relay.shape_of(var_1783))) # shape=(15, 6, 9)
uop_1795 = relay.sinh(var_1783.astype('float32')) # shape=(15, 6, 9)
output = relay.Tuple([bop_1785,uop_1795,])
output2 = relay.Tuple([bop_1785,uop_1795,])
func_1802 = relay.Function([var_1783,var_1784,], output)
mod['func_1802'] = func_1802
mod = relay.transform.InferType()(mod)
mutated_mod['func_1802'] = func_1802
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1802_call = mutated_mod.get_global_var('func_1802')
var_1804 = relay.var("var_1804", dtype = "int16", shape = (15, 6, 9))#candidate|1804|(15, 6, 9)|var|int16
var_1805 = relay.var("var_1805", dtype = "int16", shape = (15, 6, 9))#candidate|1805|(15, 6, 9)|var|int16
call_1803 = func_1802_call(var_1804,var_1805,)
output = call_1803
func_1806 = relay.Function([var_1804,var_1805,], output)
mutated_mod['func_1806'] = func_1806
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1198_call = mutated_mod.get_global_var('func_1198')
call_1831 = func_1197_call()
call_1832 = func_1197_call()
output = call_1831
output2 = call_1832
func_1852 = relay.Function([], output)
mod['func_1852'] = func_1852
mod = relay.transform.InferType()(mod)
output = func_1852()
func_1853 = relay.Function([], output)
mutated_mod['func_1853'] = func_1853
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1455_call = mod.get_global_var('func_1455')
func_1457_call = mutated_mod.get_global_var('func_1457')
call_1854 = relay.TupleGetItem(func_1455_call(), 1)
call_1855 = relay.TupleGetItem(func_1457_call(), 1)
output = call_1854
output2 = call_1855
func_1856 = relay.Function([], output)
mod['func_1856'] = func_1856
mod = relay.transform.InferType()(mod)
output = func_1856()
func_1857 = relay.Function([], output)
mutated_mod['func_1857'] = func_1857
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1888 = relay.var("var_1888", dtype = "uint8", shape = (9, 3, 9))#candidate|1888|(9, 3, 9)|var|uint8
var_1889 = relay.var("var_1889", dtype = "uint8", shape = (9, 3, 9))#candidate|1889|(9, 3, 9)|var|uint8
bop_1890 = relay.less_equal(var_1888.astype('bool'), relay.reshape(var_1889.astype('bool'), relay.shape_of(var_1888))) # shape=(9, 3, 9)
bop_1893 = relay.multiply(var_1888.astype('float32'), relay.reshape(bop_1890.astype('float32'), relay.shape_of(var_1888))) # shape=(9, 3, 9)
uop_1898 = relay.cosh(var_1889.astype('float64')) # shape=(9, 3, 9)
func_1316_call = mod.get_global_var('func_1316')
func_1318_call = mutated_mod.get_global_var('func_1318')
call_1900 = relay.TupleGetItem(func_1316_call(), 0)
call_1901 = relay.TupleGetItem(func_1318_call(), 0)
uop_1913 = relay.acosh(uop_1898.astype('float32')) # shape=(9, 3, 9)
output = relay.Tuple([bop_1893,call_1900,uop_1913,])
output2 = relay.Tuple([bop_1893,call_1901,uop_1913,])
func_1917 = relay.Function([var_1888,var_1889,], output)
mod['func_1917'] = func_1917
mod = relay.transform.InferType()(mod)
mutated_mod['func_1917'] = func_1917
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1917_call = mutated_mod.get_global_var('func_1917')
var_1919 = relay.var("var_1919", dtype = "uint8", shape = (9, 3, 9))#candidate|1919|(9, 3, 9)|var|uint8
var_1920 = relay.var("var_1920", dtype = "uint8", shape = (9, 3, 9))#candidate|1920|(9, 3, 9)|var|uint8
call_1918 = func_1917_call(var_1919,var_1920,)
output = call_1918
func_1921 = relay.Function([var_1919,var_1920,], output)
mutated_mod['func_1921'] = func_1921
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1926 = relay.var("var_1926", dtype = "int16", shape = ())#candidate|1926|()|var|int16
const_1927 = relay.const([[-10,-2,-1,9,7],[-1,-4,-9,9,10],[5,1,9,7,1],[-5,-4,9,-7,1],[-6,-4,-1,3,-1],[2,-10,-3,-1,2],[-2,-2,5,4,-2],[9,-9,10,-8,-2]], dtype = "int16")#candidate|1927|(8, 5)|const|int16
bop_1928 = relay.logical_xor(var_1926.astype('int16'), const_1927.astype('int16')) # shape=(8, 5)
bop_1931 = relay.left_shift(bop_1928.astype('uint8'), var_1926.astype('uint8')) # shape=(8, 5)
output = relay.Tuple([bop_1931,])
output2 = relay.Tuple([bop_1931,])
func_1935 = relay.Function([var_1926,], output)
mod['func_1935'] = func_1935
mod = relay.transform.InferType()(mod)
mutated_mod['func_1935'] = func_1935
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1936 = relay.var("var_1936", dtype = "int16", shape = ())#candidate|1936|()|var|int16
func_1935_call = mutated_mod.get_global_var('func_1935')
call_1937 = func_1935_call(var_1936)
output = call_1937
func_1938 = relay.Function([var_1936], output)
mutated_mod['func_1938'] = func_1938
mutated_mod = relay.transform.InferType()(mutated_mod)
func_894_call = mod.get_global_var('func_894')
func_896_call = mutated_mod.get_global_var('func_896')
call_1942 = func_894_call()
call_1943 = func_894_call()
func_1455_call = mod.get_global_var('func_1455')
func_1457_call = mutated_mod.get_global_var('func_1457')
call_1957 = relay.TupleGetItem(func_1455_call(), 1)
call_1958 = relay.TupleGetItem(func_1457_call(), 1)
uop_1959 = relay.exp(call_1957.astype('float32')) # shape=(2, 15, 15)
uop_1961 = relay.exp(call_1958.astype('float32')) # shape=(2, 15, 15)
uop_1962 = relay.asinh(call_1942.astype('float64')) # shape=(16, 14)
uop_1964 = relay.asinh(call_1943.astype('float64')) # shape=(16, 14)
func_268_call = mod.get_global_var('func_268')
func_272_call = mutated_mod.get_global_var('func_272')
var_1968 = relay.var("var_1968", dtype = "uint64", shape = (660,))#candidate|1968|(660,)|var|uint64
call_1967 = relay.TupleGetItem(func_268_call(relay.reshape(var_1968.astype('uint64'), [11, 5, 12]), relay.reshape(var_1968.astype('float32'), [11, 5, 12]), ), 0)
call_1969 = relay.TupleGetItem(func_272_call(relay.reshape(var_1968.astype('uint64'), [11, 5, 12]), relay.reshape(var_1968.astype('float32'), [11, 5, 12]), ), 0)
output = relay.Tuple([uop_1959,uop_1962,call_1967,var_1968,])
output2 = relay.Tuple([uop_1961,uop_1964,call_1969,var_1968,])
func_1973 = relay.Function([var_1968,], output)
mod['func_1973'] = func_1973
mod = relay.transform.InferType()(mod)
mutated_mod['func_1973'] = func_1973
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1974 = relay.var("var_1974", dtype = "uint64", shape = (660,))#candidate|1974|(660,)|var|uint64
func_1973_call = mutated_mod.get_global_var('func_1973')
call_1975 = func_1973_call(var_1974)
output = call_1975
func_1976 = relay.Function([var_1974], output)
mutated_mod['func_1976'] = func_1976
mutated_mod = relay.transform.InferType()(mutated_mod)
func_904_call = mod.get_global_var('func_904')
func_906_call = mutated_mod.get_global_var('func_906')
call_1989 = relay.TupleGetItem(func_904_call(), 0)
call_1990 = relay.TupleGetItem(func_906_call(), 0)
output = call_1989
output2 = call_1990
func_1999 = relay.Function([], output)
mod['func_1999'] = func_1999
mod = relay.transform.InferType()(mod)
output = func_1999()
func_2000 = relay.Function([], output)
mutated_mod['func_2000'] = func_2000
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2024 = relay.const([[[-3.714169,-6.380063,5.749095,6.469704,-1.825563,1.037809,2.740357,4.410879,8.601634,-3.948513,-1.879927,3.383474,-7.221358,-4.461896,8.935827],[8.903110,0.802803,-0.926496,5.626505,-2.060723,7.685388,-2.356118,-5.855814,-1.073415,-1.204615,-3.894085,-2.356194,3.892613,-1.410523,-3.398100],[-9.042643,4.713937,-3.224733,-5.454005,-8.570762,7.368275,6.618244,-7.147074,1.796436,8.342118,4.414961,5.056436,9.821616,3.605237,-7.607626],[4.428822,7.315554,0.226418,-0.703542,-1.196229,-1.467333,9.651939,4.965344,3.080569,-4.605957,-9.022786,3.101371,6.124394,9.904871,-3.135076],[9.470426,-5.309236,-6.048534,-8.136095,0.600520,0.651141,-2.905723,3.578510,-1.062035,-6.338081,-3.434153,3.772060,-7.507812,-0.785538,2.470469],[-4.112919,8.239945,3.479138,-0.394154,-0.352467,-1.674830,1.824108,-4.975286,-7.704212,3.002846,-0.596054,-8.579250,5.181331,-1.617706,-3.463934],[-5.264636,-3.920996,3.210314,-1.744716,8.291700,2.827670,-1.543506,-1.211996,-2.708846,-7.168815,-8.000713,0.382957,-8.605191,7.319438,-3.281173],[-3.239839,-2.081706,-4.901728,-3.375157,8.762211,-3.270239,-0.242509,7.025362,3.707211,3.082509,-0.147228,-2.718813,-0.951537,-2.474254,-5.354413],[1.420462,-1.047596,8.564776,5.020569,6.272786,-1.630904,-7.093506,-5.151311,5.368695,3.482520,9.734029,-7.680958,9.583330,-6.963186,4.604341],[-1.215054,9.469964,-5.705552,-6.903328,5.091667,6.273689,7.951372,-7.008397,8.397945,8.725304,-5.743437,-2.471424,9.534476,-7.092346,-2.234738],[-4.629037,-6.196311,0.167929,-2.052009,8.860535,1.129591,3.393035,-2.171150,-3.897332,-1.721435,-1.277706,2.518252,7.312262,-2.855381,-4.845194],[6.521218,6.795498,5.218418,6.440035,-4.207410,3.797226,5.349488,-9.639877,-0.162618,6.133100,2.080825,8.331501,4.230437,4.148496,-6.862372],[-9.813556,-7.859565,-4.923296,-2.958967,9.389588,-9.365263,-4.105650,-6.010835,1.399375,2.041753,8.466638,-3.354604,8.567850,8.690960,7.969312],[0.134739,-6.174702,8.855894,2.278137,-4.739231,9.498451,3.426187,-4.172084,-2.574132,-7.869677,-8.011159,-0.568255,-5.786680,-2.335600,-4.208582]],[[-0.804788,8.271040,2.861346,9.261613,-2.392882,-8.495577,-2.030900,5.965899,5.811131,-5.781560,-4.739136,3.701259,1.292411,-4.849327,0.832414],[4.429236,-5.615813,-2.776670,4.593570,-5.843266,2.474603,5.012771,-2.533021,-4.382060,3.713889,5.925367,-7.866275,-3.079092,8.918142,9.903447],[-9.043715,-9.917896,-7.132621,-2.734981,-2.433232,1.987748,2.414475,7.864747,-6.249030,9.256680,-1.734390,2.260736,3.589984,-0.138753,-8.269215],[-8.321323,-4.687048,-6.481431,6.354350,-3.272725,0.478154,-7.373969,7.610146,1.600359,0.003684,1.644140,2.219802,-1.282115,-7.328802,5.305291],[3.803369,5.650924,-4.886072,-7.387983,4.381464,5.406305,-3.915711,3.168542,6.175607,9.963815,-1.412506,-8.758811,3.300680,-8.796234,6.996100],[6.894761,5.508985,-5.277960,-8.332992,-3.844689,-0.353139,-2.951463,0.234070,-6.437923,-6.300490,9.641801,-0.847377,9.862954,0.277982,9.429962],[0.506617,6.719785,-5.717865,8.253107,-3.691126,-4.291808,5.000319,4.632910,-4.926887,-4.363493,-9.768186,0.526829,6.119082,-2.877891,9.398053],[-8.246777,-8.346145,9.399192,9.542498,8.233474,-9.629270,-7.171236,5.937405,-8.824910,9.051910,-3.612849,1.067562,-7.648009,-9.784940,3.144549],[4.156635,5.175611,1.741052,-8.155053,1.101615,-8.605563,4.421968,3.638852,-5.737986,-5.750470,-7.435379,7.043179,7.699566,4.952248,2.978484],[3.887766,-1.288931,-2.346933,1.723019,-1.050976,8.080792,3.701761,-4.632654,7.582888,-0.288136,-1.261749,-1.239804,-7.109563,-8.495447,-9.163032],[-5.990898,-8.315610,-8.774708,-6.989344,3.659586,-8.976442,5.173459,6.311076,-5.452614,8.522989,-1.362869,-0.031368,5.329862,-5.936557,-6.603115],[8.810118,-9.121128,3.209131,1.287804,7.008070,-0.573080,6.731266,7.496965,2.975816,0.768534,-3.250976,8.110702,-2.519300,-2.061137,-7.971039],[8.411188,9.016073,1.285721,-8.746183,3.883206,6.558389,1.666439,1.696125,3.172054,8.994160,5.278823,-8.912524,-4.227379,-2.168406,-8.060745],[2.737513,-0.693479,0.486008,1.757988,-8.907941,8.321490,-5.678901,-1.796850,2.550523,4.008909,-7.022236,1.586881,7.700397,-7.619252,-6.919818]],[[-2.857620,-5.309187,-7.126142,-7.475000,-5.844410,-4.213948,-9.693021,-3.523672,9.510866,-4.530698,-7.657607,6.144844,8.084911,8.156364,-8.279882],[-8.136176,0.733658,-9.288853,1.941983,7.066071,0.714674,4.787972,0.639697,4.980189,-1.374251,8.408773,-1.676127,-3.012873,1.979578,7.921482],[9.242366,-7.216634,1.352332,-3.029268,6.286270,-2.536769,-2.804539,-0.606212,6.827592,-5.100798,0.229261,1.153671,-0.405307,3.575930,4.673031],[-5.345403,0.498532,-9.171479,9.755234,9.123105,-5.883311,1.269284,9.854461,-3.352217,9.288390,-7.674913,-2.433426,3.254053,1.750569,-3.421491],[6.887366,6.802028,0.899513,6.095448,3.535316,1.829350,-0.387658,-4.577665,-4.983840,9.477412,8.313773,-6.844640,5.685490,-3.023910,2.722397],[7.298510,3.021750,8.751677,-8.001110,9.128533,8.316770,-7.253391,-0.626013,7.012694,5.907782,-7.432353,2.889111,7.831636,-7.859301,1.488547],[-2.522124,3.524773,-9.177084,0.754335,7.841564,-1.347663,5.835054,3.626949,1.121674,8.350060,7.223766,7.743892,-9.181574,8.684128,-4.322615],[-7.983840,1.660002,3.286675,-6.831070,8.853348,0.270720,-6.608442,-8.823331,-0.947629,3.408810,-9.440024,-4.653923,7.787999,-9.833393,2.183996],[8.394680,-2.690147,2.314205,-1.242358,-7.113827,-7.901105,5.682816,-3.147640,-8.420362,2.149655,-6.563411,-5.373784,4.997633,-2.667317,-4.604940],[5.283318,7.738516,-7.000621,3.011461,4.400886,-8.339034,1.858499,-9.262501,7.047236,7.493465,-8.760289,0.922484,-7.649481,-7.099552,-0.400973],[-8.262470,2.530999,1.918907,0.078251,-1.664496,-0.333005,-4.463225,0.139590,-1.662897,9.851385,9.328479,4.488939,-2.357417,-9.229070,-4.844748],[0.648234,-6.164925,3.983414,-9.967392,-7.772266,1.812264,-7.774661,-8.268949,-1.361404,-1.489683,-7.225348,-7.402575,-6.917245,1.676337,0.768932],[-9.454615,5.696105,7.051418,-9.759970,8.167254,1.042803,-3.969267,-5.197815,9.351654,7.368404,-3.620937,8.495703,8.834099,2.082917,-1.834678],[5.437934,8.812961,4.673664,8.628634,-8.537297,-9.495878,0.863730,6.647295,-9.585146,-0.511465,7.113625,3.637045,0.055901,-7.852819,-2.116012]],[[-6.054707,8.072451,-0.503010,-3.227973,9.272511,-0.433124,-6.143964,-0.048353,9.047896,-7.595136,-9.760678,-8.002527,6.611065,-5.930336,-0.684787],[-1.209143,-1.040388,-4.058802,9.874880,-4.214409,3.890797,-8.008777,-7.565210,-4.418927,7.781310,-9.547371,-8.229500,-8.442037,-7.176981,8.422498],[3.608969,8.403062,-0.695075,-3.323852,1.616120,-4.039817,7.083989,-1.139346,3.490047,0.612988,-5.778448,-0.104025,-0.283143,-3.246141,8.692672],[8.482235,5.923235,0.667963,-4.349176,2.476120,-0.725375,-6.378980,6.867397,-1.129221,2.410065,2.663329,5.951244,-2.909034,-2.012055,-8.140491],[1.650627,-0.862755,8.560091,-4.266637,6.867335,4.291861,2.121325,3.201576,-2.869152,4.234437,8.790793,7.482575,-6.100926,-2.999141,-3.999399],[-0.084927,-5.218248,9.636979,-8.933350,9.637494,5.806112,-6.712574,0.674809,1.566768,1.389200,7.299490,-9.418347,9.406810,8.388487,4.099739],[-8.351091,6.362733,2.355783,-6.454804,-2.204349,7.227550,-0.491259,-1.298559,-4.576506,-4.164670,-7.805093,-9.393833,-2.014857,7.178006,-4.769078],[-2.051645,-6.177609,7.592903,-5.386513,-0.122916,-3.772412,-3.603374,1.043931,-5.709261,2.662455,9.723177,-6.148830,-3.836291,4.559342,-8.164836],[-4.646335,-6.279565,-9.587030,-7.447688,-9.130576,2.075677,8.904764,7.964649,-3.708470,-3.815331,-2.414962,5.906171,5.374598,-0.341811,4.884219],[-6.138878,7.519428,0.307685,7.970165,-6.657318,-3.187020,-5.983198,7.449259,-7.761336,1.689071,5.558997,-5.139695,-3.822348,-2.091911,-9.264311],[-8.111060,-4.233421,6.672333,4.223753,7.234332,-1.514566,-2.107400,4.217814,6.205095,6.752124,-7.208426,-9.117709,-7.509892,3.698525,3.792589],[-5.450433,-9.950566,0.559127,-5.762855,-2.816943,-6.664834,5.380861,7.980303,5.131708,-0.227232,0.124237,9.549220,8.098231,5.558390,6.687528],[-2.233249,-6.141468,-3.392019,4.528481,-8.126258,-0.375907,-0.339228,4.472533,5.356507,-6.535008,-6.825436,-1.548297,-5.373808,1.168071,-2.895219],[-2.465941,-3.810928,-7.685538,4.738391,-1.720714,7.337927,-2.111528,-3.496258,-1.889348,-2.460876,-8.998156,-6.312085,8.966229,-2.968611,-0.039730]],[[4.968406,-2.465158,5.509698,-3.527003,-6.609755,5.969943,0.916090,0.509735,5.443674,7.913435,0.996979,-1.614381,0.994448,-5.285657,-0.273391],[-1.266279,3.166581,-8.263246,5.420490,3.698377,8.118749,-8.847080,2.442925,-8.694767,5.094599,-2.647956,6.095627,-0.224966,-3.959088,8.892288],[-9.007242,-2.217801,-6.930416,-5.039602,7.799078,4.300911,-6.349513,-3.900259,8.418408,6.996432,-0.831204,5.245743,-0.129348,-8.349846,-3.135840],[-8.482036,-6.378475,9.653512,0.310803,-6.748021,3.924937,0.860116,9.491702,6.846952,-2.043488,4.617755,2.783144,6.520904,0.461246,-9.288694],[3.663074,-2.177779,-3.146838,-6.645859,5.141538,-8.198645,-3.807492,3.385318,2.203290,-0.403284,2.267415,9.551971,9.123364,-9.846448,-2.916466],[-2.934864,2.962275,5.568958,-6.686960,0.575093,9.319495,0.554454,2.523331,4.374639,4.479114,4.471091,-3.343683,2.508138,0.886310,-5.990054],[1.417065,0.902025,3.534712,-4.958437,-2.328415,4.741188,6.763591,2.620676,6.650938,-9.502720,-6.718293,7.642867,-6.612394,-7.174850,3.986516],[3.923835,-0.062710,7.861230,-9.059621,1.916930,-2.842736,-1.319434,3.736404,-8.150419,-0.496935,-8.389410,-1.841081,5.670079,-6.968409,-6.631377],[-9.465522,-1.652157,-8.408294,-5.527517,8.657172,0.065983,7.783817,-3.663268,1.141720,0.882069,0.937114,3.901166,-5.750955,-2.600143,-6.701828],[-5.300662,-1.329729,0.712347,-5.317257,2.557148,-9.167075,6.143046,-9.272051,-7.488959,9.473505,-6.598880,9.378712,-4.213585,-5.442553,1.909509],[-3.128393,-0.886853,-4.815440,5.916691,7.075774,0.320986,-5.692978,0.955845,-3.204961,-3.287210,-4.956107,-6.095965,-7.400477,2.817666,5.092700],[-7.245985,-0.492067,-4.356839,-4.463316,6.892456,9.462087,1.432357,-5.370721,-6.126296,7.934473,-3.581934,1.027829,0.208110,5.908029,0.723709],[2.510629,-7.491969,-3.442292,8.423694,6.157295,-7.620229,2.599468,-5.722055,0.622472,-2.105704,9.445483,-8.365010,4.916636,7.500819,-3.845943],[5.898820,-3.667081,5.015485,-9.165552,4.839711,9.872852,-8.454268,-5.562900,-3.231047,-4.744183,-0.347251,8.089480,-1.055158,-6.065471,-9.825230]],[[-0.656492,8.126033,7.278149,9.171384,9.131524,2.450319,2.396117,3.170139,-1.733030,9.816936,-7.660949,9.609834,-7.968080,-2.696027,4.847883],[7.351700,-5.411608,-6.496394,5.060659,9.107764,-2.463186,0.285515,6.207005,5.071280,6.841966,-2.176589,-6.761232,2.136084,2.731529,-4.500965],[-8.017840,7.964548,-3.120689,7.226017,7.742118,-2.869963,3.303385,2.558854,-1.350806,5.077345,3.441186,2.013968,1.285360,-1.350724,-6.623944],[9.034792,-8.269798,4.409339,-0.651608,-8.720061,2.757306,4.481297,1.874240,1.005622,-6.778829,7.374982,-7.110660,0.852106,4.984202,1.437655],[-3.215988,6.647749,8.482020,7.039808,5.282816,3.771030,-5.147412,3.271194,-1.440342,7.553887,2.339579,-2.950602,2.218325,6.992983,-8.374787],[-8.862476,2.411027,6.137314,-5.323829,-7.157594,9.083986,7.393817,7.842720,6.016679,4.536460,-3.332023,1.923270,6.117775,-3.109167,7.080273],[5.033326,-8.979918,8.103483,8.399200,-8.527724,1.513733,1.197466,-1.148610,-8.095678,-3.578478,-0.623849,-7.287293,-4.905340,9.007175,1.710090],[6.374721,-3.375958,5.743016,8.624877,-0.575043,1.318377,4.958538,-4.004508,7.848715,7.812024,-6.133402,-4.165436,1.616892,-1.186376,2.099448],[-9.915313,-3.762607,-5.469984,-3.220549,-1.420648,-9.003594,3.890098,5.353562,8.024669,-8.926002,-4.887807,-5.414119,1.592691,6.530663,-4.739855],[8.498377,6.715287,-7.061109,-0.735005,-6.994534,7.997297,-0.092653,-3.861900,-8.725376,-7.320310,2.805366,8.540780,3.910420,9.880070,6.373411],[4.763056,-2.599647,-2.203588,7.636090,-7.216510,-8.786078,-4.399619,-7.150438,-2.880467,-8.800884,5.704120,6.226246,4.284146,4.864242,-4.305651],[3.780131,-5.165446,-2.410074,-7.220235,7.741777,2.744773,-1.066579,-7.501684,5.008625,5.822611,1.548027,0.098244,7.474756,4.818274,4.894750],[7.131793,0.490653,2.896296,3.033134,5.450025,-3.144107,-1.354203,-0.361370,8.546712,3.852821,4.905563,7.699530,3.575481,-5.048456,4.949196],[-9.095937,-7.027075,-4.491710,-7.661700,2.601848,-0.948846,-6.660466,-0.009395,2.198582,-6.609873,-3.347705,2.622458,-7.591904,-8.006158,5.150955]],[[9.831998,5.661064,-1.597115,4.605182,7.385380,-9.953342,-5.730494,-9.570597,-9.001656,-9.263669,-2.153799,-8.263407,-3.288889,-8.501231,7.260452],[-7.445886,-4.954574,-3.209866,-0.243373,2.886176,-5.326731,7.197506,-3.861241,5.141338,-6.827000,-3.703360,-0.536211,-8.952945,3.329247,-9.871288],[4.811427,6.167489,-5.701529,-6.523679,7.688425,-9.321948,3.560007,-4.348643,7.363669,-2.605933,9.858064,-0.341079,-0.612723,-0.622135,0.201015],[-8.219366,-5.288002,-2.592982,-7.983684,5.048900,-9.602878,5.758483,5.467343,-7.510024,-6.884360,-0.221490,7.780792,0.597028,-9.417560,8.260066],[-3.671767,-4.163503,-6.951510,4.635651,5.643338,-6.805019,-0.347401,9.746580,-4.599417,-6.779291,1.123444,-6.808617,9.304917,-9.846119,-0.039684],[-7.838076,9.755608,5.108789,7.193144,-4.339989,7.993759,1.377180,6.680915,6.505648,-1.312155,-7.155227,0.983037,-0.997605,-4.919276,-7.791594],[-1.960801,-9.288355,6.443293,3.131201,-7.841949,7.616976,9.067009,-1.453164,8.746663,-5.174899,5.564515,0.982946,3.197141,-0.104233,-2.890315],[-7.990679,8.600653,3.622800,-6.007412,2.823616,-2.616164,-6.000790,0.767802,-5.366450,3.489916,-6.176470,1.749802,7.568802,0.803828,3.965171],[-3.128269,0.997020,-5.205946,-2.616179,8.887656,3.867655,-4.063428,4.144774,-3.381789,-4.695923,-8.126685,9.269963,6.986309,2.061450,-3.166704],[-2.514701,-5.152095,-0.830897,-0.796146,-2.406306,-0.500465,-8.890176,5.529305,4.034165,4.778504,3.960266,-3.403607,-8.863818,4.030779,0.014432],[-9.957337,-5.982448,8.207751,3.765829,4.280351,1.199598,5.987675,-0.081656,-4.871098,7.499413,-7.782487,2.247222,6.637828,-4.661960,8.727874],[-8.248188,7.776795,2.098451,2.860726,8.752169,-4.410772,-6.485969,-6.225827,-4.979731,-6.247586,-1.775610,0.697071,-6.562668,5.204312,2.289466],[-7.705830,-8.010456,6.021158,-0.154938,-6.783381,4.764041,-4.520549,1.718766,3.672371,0.354711,4.053376,-9.539266,-1.248847,7.176415,3.595783],[-6.599276,-4.590463,2.600910,-4.582747,-1.559149,9.878509,4.838892,9.865708,-6.078861,3.746718,8.509431,-7.131749,1.188938,-9.187775,-7.316907]],[[8.840777,9.185292,-2.348311,3.673129,-6.511432,5.612482,-4.136059,4.335195,7.515575,-8.983319,-8.639795,2.681713,6.251500,0.125793,7.930353],[8.591590,-3.276701,-7.780297,2.522716,-8.018691,9.791186,-1.253202,-9.764322,-4.713147,9.431836,-2.999321,-4.886965,-1.778710,-0.435822,6.065662],[-0.380001,1.491134,5.356687,4.718609,3.795871,-7.244365,1.098174,-9.188813,-0.683572,-0.982663,4.466840,8.171212,-1.212984,2.262018,6.472879],[8.614487,9.095384,2.588081,-6.259513,-0.455805,-9.815282,5.586772,4.308133,5.386073,-7.912563,2.216750,1.060693,2.766083,0.690117,-2.689141],[7.208496,-8.254475,-0.437743,9.473994,-2.746131,4.108611,-0.295473,-3.564145,-1.802608,3.791409,0.840381,2.633808,-9.767216,3.903251,3.602693],[-9.821384,9.749919,-4.238072,-7.997624,-8.619692,3.987381,5.798563,-0.845686,0.008199,-9.883446,5.969283,4.445344,-3.058226,0.159197,2.564407],[0.774143,3.021440,-0.911340,4.771405,0.634857,1.042496,1.992808,-4.257992,8.615929,-6.754655,2.401368,6.051855,-1.248722,-1.983107,7.602845],[9.279444,4.382623,0.314924,-5.053106,7.658908,5.799318,-7.061538,4.430398,-7.124724,-8.943061,-0.532588,1.452204,-2.021761,2.365940,0.764846],[9.706516,-3.927071,-4.764474,-3.483125,-9.042780,-1.442558,8.238995,-4.091514,-1.042795,3.944947,4.133831,-4.560602,7.970977,7.362216,0.071659],[-9.437100,3.301592,8.054003,-3.474137,-5.065225,5.492513,-7.590802,5.317230,-4.257645,-2.840379,-7.265369,7.418841,-4.925905,5.827771,-3.487876],[-6.262976,1.859159,7.779833,8.296624,7.973501,-8.358979,4.850211,8.303586,9.423470,-7.061076,3.089350,4.971730,9.243342,4.631787,0.477004],[8.040591,-9.146770,5.815938,8.721123,-6.086825,-9.253402,-8.995997,0.608726,-1.834871,4.580055,-8.952931,0.222114,2.361938,9.385220,-3.248143],[-5.199760,0.194121,-6.269782,-2.991900,4.070701,-8.150753,5.722498,2.956773,-9.774769,7.932564,-0.874946,-4.345737,-9.273264,-0.273451,5.789287],[-3.454504,-8.850075,-7.546695,6.208561,2.032504,7.651689,-6.574482,-2.074925,-2.417179,-5.229654,-1.581941,5.743627,-7.993587,-5.166971,-2.255409]],[[2.506958,-3.620831,-2.063632,2.543442,1.821483,-0.827496,-6.583469,-3.054121,9.483312,-5.787986,-9.789738,-0.539800,3.461703,8.989716,-5.219172],[9.065394,4.050279,2.711447,0.804157,8.826015,7.209466,5.281160,-6.471870,-4.405437,-2.871905,3.270358,2.697182,7.879332,6.314711,4.333235],[-8.558184,9.443643,-7.696575,6.065309,7.114854,-0.799700,-6.380506,2.288293,3.975599,-5.518246,-7.564796,6.122715,-5.635501,-1.540729,5.125173],[1.400913,9.027143,0.073729,4.999902,7.455115,6.453248,-1.483230,2.255458,1.894005,0.249071,8.912654,6.032914,-5.369236,9.120845,9.038282],[-5.972123,-0.127881,-1.042247,4.566240,-6.383593,-3.176128,-9.387023,4.988264,-4.343448,0.634052,9.798159,7.000071,5.386784,-2.712164,9.414995],[8.481908,-1.093749,7.558819,-0.956619,-8.928694,7.653823,-4.751971,5.482287,8.114960,-3.023727,-2.193326,1.606806,3.638192,5.866150,6.091152],[1.736582,-2.211735,-6.902997,-4.939159,-2.725046,2.971634,-4.627290,6.406226,-3.502052,1.216445,9.853571,-5.650727,-8.900690,8.948141,7.633545],[-6.569299,-9.301654,-7.035824,-4.153645,8.895777,1.229877,-8.039569,-0.294825,-1.850588,-8.254402,-2.500587,4.221698,2.520762,-1.198896,1.380246],[4.865445,-4.174919,-3.921184,0.212636,-2.020758,8.665926,3.557731,9.468102,9.158654,-2.159375,5.066279,-2.347029,9.949560,-5.047650,-2.555866],[8.761986,2.608295,4.264986,-2.051620,-6.482256,5.708800,1.967063,7.551997,3.058142,-4.123730,2.496347,-6.017111,-9.928937,5.930213,9.046596],[-1.229438,3.877728,-1.454694,2.085299,1.678135,4.333840,6.987874,-4.984545,3.223957,6.743773,2.670847,-6.806180,-4.923693,-2.595352,0.645052],[-2.605887,-5.168273,-7.880039,-3.420338,-7.391423,6.810700,0.114558,-6.027515,-7.544787,8.288397,1.873090,-9.657623,-5.335794,-0.904631,-4.241397],[-7.159580,3.830263,-8.949306,8.554527,1.440681,4.904829,-6.202034,6.797678,-2.106076,-4.168066,1.165924,-4.968014,8.156775,4.288133,1.126648],[-8.525322,-6.607010,4.810065,1.341098,9.203759,-9.196506,5.682147,9.545125,3.076047,-8.743442,0.397471,2.801235,3.843993,-7.888813,-8.227270]],[[-7.337232,9.830781,-2.344581,-9.822334,-1.310539,-1.382127,-5.647645,2.141229,-9.491970,8.720384,-4.647144,-5.135367,-9.382233,7.248522,-8.861503],[6.867901,-1.398147,9.465447,7.535237,-4.683941,-6.405551,-5.151155,-3.451079,-6.097082,-6.966656,-0.683812,9.030607,-4.928586,-4.700821,-9.736906],[3.485199,-6.998510,6.800738,-8.155352,6.191312,-4.035544,8.978544,-4.834105,-1.537563,-7.100748,-4.176006,4.134020,4.668653,2.483204,9.698767],[9.378431,5.149672,-0.423901,9.027848,3.718390,7.473897,-6.781515,-7.710819,6.420202,-7.719670,6.657341,-6.421970,-2.025898,-2.968161,8.488168],[3.435135,-5.977870,-6.357315,0.909044,3.152230,-7.616228,-1.156968,3.440746,8.660919,4.266806,-1.863666,-5.871691,-4.021840,3.383671,2.767563],[-6.219560,7.696930,3.757858,-9.861455,3.321215,-5.549635,-5.147417,8.065244,2.150348,-5.741820,-0.624955,-3.831484,1.989924,2.763677,5.885924],[-6.543056,6.555484,5.882080,5.811814,-1.424268,8.391632,3.476217,1.561282,-4.986041,6.708398,-4.508017,5.066808,9.942968,3.409056,2.855495],[4.824150,-2.353196,4.062065,4.333432,4.984975,-2.713337,-8.028225,4.232293,-5.203868,4.779960,-1.628748,0.283746,-4.711173,0.353144,6.510556],[-0.894367,-7.937842,8.043765,-8.971040,-9.874212,-9.400113,1.567641,-9.699500,8.831115,-8.554559,8.228990,-7.261661,-0.975011,5.474033,-6.625341],[3.278868,7.809407,8.276648,1.065064,5.160686,-9.240060,-5.158397,1.076157,-2.810242,3.386167,3.616306,0.255351,3.753373,-9.435023,4.249317],[8.125339,1.007851,-2.741886,-3.823867,8.503203,-5.154853,9.584603,5.480105,4.610156,6.669790,-8.030437,8.672492,-9.902142,0.815779,8.100029],[6.366182,6.780235,4.821832,-8.172227,-4.240194,-1.316715,8.204791,9.882546,-2.268232,-3.481120,-2.919651,-5.661060,-1.870951,-2.529133,0.985327],[0.112999,8.336623,5.016748,-5.987511,-4.848133,5.369196,7.929276,-8.399549,6.884864,-7.036283,3.916697,-9.781499,-0.433595,-5.117107,7.633702],[1.879984,2.073148,6.904631,2.182504,-7.376341,-8.718180,-5.429195,-1.011818,7.386946,3.621986,8.828243,-6.546869,-6.619469,-2.805818,2.448819]],[[-1.586076,-3.687647,9.084836,3.298777,-2.958292,-7.477300,4.223596,-6.637317,-3.534268,-5.401691,1.110341,-0.193495,0.688854,5.122560,6.704507],[2.074989,6.230644,8.128974,-7.201233,0.334331,-1.611994,0.718210,9.428888,3.056502,6.539531,4.614359,0.004151,-9.990222,7.422347,4.078613],[3.175634,-3.292517,-4.704987,4.805910,-7.589651,7.282682,0.680145,8.597957,8.505741,-6.565469,-6.961912,4.343324,2.115078,-7.705410,0.994702],[8.258991,8.008808,7.563016,-4.947448,-4.018244,-1.908989,8.488949,-1.634565,-4.325763,-9.553696,3.064560,-3.890747,-0.939911,-6.924589,-7.953585],[-1.445866,-0.991646,-9.352585,-2.935700,8.811093,4.398570,-0.732436,1.728505,1.495042,-1.503087,-7.173742,-3.013283,5.422229,-8.866528,-7.866295],[6.279830,-5.202728,-6.545436,-1.908128,0.857308,6.779377,0.898424,-6.244960,-5.158337,-4.277933,4.944702,-4.408551,3.590863,-4.291512,-6.834671],[6.681443,-1.292986,7.981087,-9.589303,8.992254,0.978997,-3.825144,-3.601716,0.058959,9.348951,-7.473658,-2.984684,-1.412180,-5.191165,2.909227],[1.222124,-5.044761,2.954017,-7.031196,1.040140,5.314105,-2.281813,7.050105,-0.184233,-9.251876,-4.331368,-7.341482,4.478098,9.329540,-0.960611],[-3.817352,2.462995,0.766086,3.452776,-6.484586,3.012915,9.114608,6.087492,-1.346350,-8.579543,5.734939,3.248381,2.004693,7.346435,-9.919336],[-3.792914,9.100150,-3.926912,-7.031407,1.629335,-4.577815,-1.002774,-1.727300,3.477441,-3.636555,4.730820,-2.687747,5.762181,-0.556422,-8.749599],[-8.627529,5.219233,-9.247235,-6.307386,6.866185,3.154740,-4.420542,-2.466806,-4.425019,-7.632957,6.037670,-9.021584,-0.081146,7.621375,-6.778813],[-6.183473,-0.090960,9.482610,1.648330,0.348052,8.480844,-5.852169,3.386749,-0.763345,-3.391474,-0.501441,6.844683,-1.571363,-6.984672,-6.176995],[2.413993,-7.739170,-9.802076,9.538681,4.643608,6.585082,0.620207,8.358121,-6.142776,-5.514634,-7.518641,-8.747779,-8.118264,8.722463,-5.775722],[1.387877,-6.251585,-6.695367,9.216983,-5.226867,-8.539778,6.584032,-3.220274,-0.144250,2.613373,1.980191,8.818623,9.226668,-5.524307,9.009433]],[[-0.900028,9.057320,-5.510685,9.183948,6.949091,-2.360053,1.138365,9.024585,-6.190986,-3.128826,9.211009,7.628107,4.026278,-5.049972,-4.095230],[0.637049,-3.540898,7.736495,-6.158077,6.827420,7.642210,9.918891,1.325006,-7.297771,0.885938,3.063935,-8.794874,9.908916,5.813746,-3.816518],[5.804451,-2.649691,-7.375561,-3.688544,1.198887,-6.250299,-2.975677,0.189945,-7.152668,1.855963,9.556352,-2.994922,-3.267370,0.566367,5.828375],[2.189402,2.239925,2.629091,2.876773,-1.709543,2.989489,7.719483,6.340596,6.423287,-4.665558,0.953597,-5.574651,-4.852563,8.787817,-1.319388],[0.728141,-7.423315,-0.516184,-5.766405,2.975196,-3.910651,8.932456,-2.292176,-1.511627,-7.586939,0.832340,-9.907002,-7.695998,-4.480873,6.092718],[-8.357148,-3.050828,2.502906,-4.446542,-4.619745,9.463910,-7.332415,0.534133,5.592969,7.312222,-0.896306,8.609396,-8.497816,5.247445,9.254076],[8.743402,7.765701,-8.028169,-9.947007,-2.813318,2.346016,4.196499,-1.273073,-2.231455,0.628810,-7.157024,-7.216103,-0.802653,6.089277,6.649169],[-2.506094,-1.667747,-5.266579,4.904210,-3.205237,-3.456534,7.374675,5.459874,-8.563241,5.842500,2.639371,6.791331,1.947153,-8.353081,-0.510899],[6.944653,6.541756,-8.910739,7.278855,-6.130043,-6.790853,5.616727,0.823264,3.331029,-2.463698,1.642119,-4.030016,-8.463376,-4.807696,-6.968349],[-5.960734,-0.433940,-2.559166,-8.808741,-4.360134,1.148279,6.538907,-2.321687,2.982465,-7.065798,-2.795171,-2.948321,-3.184795,-4.710029,8.972488],[0.372021,-9.246825,5.165770,8.168895,-1.948829,7.625175,-7.615463,-9.413293,3.177725,-4.960203,-8.201366,6.608825,-9.400437,3.128606,-5.896693],[4.377209,-6.071654,4.023320,-6.338834,0.093644,-1.427070,-4.212769,5.607813,6.858949,7.596041,-5.048958,-2.725469,-9.321027,-1.281661,4.533441],[-3.524893,-5.054897,0.636256,3.498367,-0.691199,-4.619366,8.758764,2.704242,-2.987398,1.051371,6.774649,4.400694,8.494849,-8.219719,9.080134],[-0.129463,-4.471280,-3.094089,8.101452,-9.660315,1.328251,-2.520859,7.102697,3.720016,8.456290,4.977856,7.760622,-0.240441,-5.951919,-1.584335]]], dtype = "float64")#candidate|2024|(12, 14, 15)|const|float64
uop_2025 = relay.cos(const_2024.astype('float64')) # shape=(12, 14, 15)
uop_2032 = relay.log10(const_2024.astype('float64')) # shape=(12, 14, 15)
bop_2034 = relay.greater(const_2024.astype('bool'), relay.reshape(uop_2025.astype('bool'), relay.shape_of(const_2024))) # shape=(12, 14, 15)
uop_2042 = relay.atanh(bop_2034.astype('float32')) # shape=(12, 14, 15)
bop_2044 = relay.floor_divide(uop_2042.astype('float32'), relay.reshape(uop_2025.astype('float32'), relay.shape_of(uop_2042))) # shape=(12, 14, 15)
output = relay.Tuple([uop_2032,bop_2044,])
output2 = relay.Tuple([uop_2032,bop_2044,])
func_2048 = relay.Function([], output)
mod['func_2048'] = func_2048
mod = relay.transform.InferType()(mod)
mutated_mod['func_2048'] = func_2048
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2048_call = mutated_mod.get_global_var('func_2048')
call_2049 = func_2048_call()
output = call_2049
func_2050 = relay.Function([], output)
mutated_mod['func_2050'] = func_2050
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2058 = relay.const(5.615320, dtype = "float32")#candidate|2058|()|const|float32
var_2059 = relay.var("var_2059", dtype = "float32", shape = (9, 11, 1))#candidate|2059|(9, 11, 1)|var|float32
bop_2060 = relay.divide(const_2058.astype('float32'), var_2059.astype('float32')) # shape=(9, 11, 1)
const_2078 = relay.const([[[0.262684,8.466404,-8.303042,0.444999,9.339904,8.356983,-2.616057,7.403673,3.680966,-0.692649,9.612607,6.866705],[4.379887,-7.683008,-7.528270,-7.816394,2.452857,3.456847,-7.478295,-4.412965,-5.180789,2.736589,9.714160,8.467870],[-8.302924,4.080284,-7.848423,3.816877,9.029935,7.838509,2.377613,7.483559,0.206317,-2.214389,4.477837,-8.052237],[8.478357,-2.519386,-3.943036,8.167408,9.586138,-8.387152,5.243805,1.180326,2.950766,-6.680058,-1.269666,-7.161690],[-2.268811,-0.026051,3.304434,-0.473651,-7.457140,5.653965,-4.125473,0.164627,3.261541,-9.874037,0.664152,2.288264],[-0.716555,7.226362,-0.841985,0.589898,3.035733,-6.732525,-5.169222,7.231107,-6.964503,5.331967,1.033284,-2.518873],[-8.917496,3.656227,5.536308,-5.054007,1.789550,-0.877452,4.852806,-4.673129,6.411815,-2.865576,1.262631,-8.886816],[-7.484577,7.067766,-6.850569,5.417255,-0.698851,-4.940781,0.888329,1.611524,5.698549,-7.073868,2.016254,1.749497],[-1.305505,-9.311977,4.341536,1.525012,-4.294723,2.171156,-3.182759,5.621149,7.884108,-1.455518,9.095172,1.088379],[-1.195712,4.617585,7.973373,-3.860996,-9.712894,1.600985,-5.370760,-0.221067,1.452509,0.115049,-7.853911,-0.464586],[6.850010,-6.664827,4.444040,7.275408,-5.755359,7.703672,9.321801,2.910239,2.005734,5.586459,0.815588,-6.975500]],[[8.283110,-3.067210,4.794031,7.679638,-9.316522,2.491497,-3.275586,6.748335,4.576176,-9.740924,-3.404140,6.071073],[-8.770578,4.197058,-3.562370,8.647333,-0.407047,3.837357,-6.656063,-3.897324,-2.425203,-7.062256,-8.661759,6.908304],[6.433181,-1.482523,4.990076,0.294934,-2.159842,3.639064,-0.046134,-3.232116,7.723466,-6.264082,-5.501430,-4.675610],[-1.006431,1.672432,-1.300959,9.146135,-2.052585,8.177060,1.291407,-3.016243,-7.987631,-7.883986,-1.657232,6.199817],[-1.314761,1.441438,-9.023812,7.924250,1.896765,4.482271,-5.861114,6.036850,-3.974707,-6.553968,6.607013,6.632660],[1.071593,-5.633066,-7.771625,8.762298,-0.547581,2.971105,-0.129029,3.032697,-5.017585,3.490903,7.658810,4.496875],[4.456969,5.317639,-6.287279,-2.989193,-0.099512,1.685255,-0.803611,1.868489,-3.694278,7.515679,-8.396362,-0.046021],[6.720214,1.517696,-1.346169,-9.673013,-5.348679,-6.681230,-3.708136,-0.406008,1.190832,4.805181,-3.562660,-0.932904],[-5.138826,5.956336,3.157219,7.831623,4.857743,8.713209,4.915973,-0.266879,-2.935136,7.089756,2.293582,6.598146],[0.839514,1.589227,-7.274755,-2.387581,1.858429,5.482966,7.731391,-1.512349,2.506340,-3.079461,1.736949,-7.621786],[2.815801,2.351938,-1.254928,-0.604922,1.080022,2.318653,7.112765,-6.228135,-4.090255,2.439246,-0.751476,-6.944651]],[[-2.088126,3.209009,6.866912,0.777868,8.770018,-0.968305,1.676953,2.678368,-4.133274,-4.676382,9.072819,-3.889993],[-4.522385,6.197584,8.263838,-9.067770,3.745534,-5.788073,-8.397362,0.017629,-6.672167,2.979216,-1.893480,-8.857779],[4.070535,3.414890,-3.271845,-4.823621,8.729842,7.079711,-0.441962,-6.096005,4.951277,-9.044711,-6.455468,6.354922],[9.076507,7.423494,-8.385081,-3.509101,-0.785620,-5.515112,-0.897974,-2.695643,-5.177275,-0.976527,7.284877,-5.673566],[3.552346,-3.305090,7.286537,-9.596109,-5.868390,-6.343840,-8.661115,9.706943,-0.511306,5.747525,-2.947552,1.509009],[9.687682,7.262966,-2.965692,-9.446913,4.310805,-9.858732,-0.792474,2.008659,-9.841977,2.162918,9.881544,6.693363],[-8.258575,-9.425387,-5.975449,-3.024464,7.095062,2.009049,5.516109,-8.348747,-1.208507,4.307871,5.037117,-7.404037],[7.334902,-7.890410,-4.859982,-6.387099,-9.581440,7.475369,-5.658796,9.157301,-1.889493,-2.365854,-5.146575,-8.238293],[4.316006,8.204976,-3.914167,-3.848252,-7.219992,-8.037799,-8.561480,5.275166,3.370694,-7.179509,-7.409262,-9.974771],[-0.893094,2.933828,2.816451,1.324213,6.250588,3.020077,-3.880565,0.818416,5.624811,9.373382,1.868283,-8.453765],[-4.553869,-8.314462,-4.719168,6.116215,1.981886,7.490476,-3.483176,6.498167,-7.137900,3.984486,-6.555469,9.239031]],[[3.673450,-7.319808,6.499839,6.729594,8.767537,9.608904,-5.086448,-5.893890,6.286677,-5.343463,-6.075522,1.935078],[1.662017,-8.653464,-8.355392,4.059184,-9.026896,-6.031895,-6.954222,9.262123,0.452541,-5.005441,4.130083,-6.868398],[-2.085552,-7.464291,-4.145120,4.191632,5.780804,-5.405706,2.775432,2.117904,9.244509,-8.057088,0.645278,4.150125],[-7.453628,5.853293,-5.994749,-5.620755,-8.782001,6.338498,-1.096568,-9.062269,-1.719977,5.601494,-0.341782,-1.740775],[-0.584418,-6.336912,-7.059782,7.564405,2.271690,2.902629,-4.707185,-7.110322,-6.835873,1.391429,0.272580,-5.383491],[-7.128455,2.389977,-1.920510,-7.361657,-4.057678,9.296315,-5.951762,1.461503,6.888652,8.287923,-9.552599,-8.729923],[-8.935831,-1.716984,-6.873152,2.472115,-1.009633,8.668648,-7.586938,-2.485820,-8.359169,-9.367025,-9.902896,-0.362122],[-8.882309,9.758047,-9.756088,1.464919,3.847155,-0.209531,5.968420,0.029694,-9.962541,-7.404816,-2.393394,-9.804048],[-3.580466,-5.083486,-0.444788,-1.504894,-2.978699,-1.561425,-4.593457,8.494974,8.771576,4.047967,-3.892238,-5.466293],[3.814001,0.053067,7.262987,9.021994,-0.232113,-5.992336,-2.939919,1.362389,-4.525059,6.258675,-3.603695,9.123096],[-2.757414,-5.830587,5.754763,-1.237970,8.660495,-0.580985,8.342541,7.650632,9.007362,-3.095548,-4.002956,8.746504]],[[8.655596,6.315408,3.674414,7.194643,0.627274,-5.498015,5.079071,-8.804888,5.653243,7.969117,-6.136435,-7.878902],[6.742941,6.224316,3.988795,-1.399103,-3.118354,-2.990976,-2.041812,3.827726,4.848915,-1.614624,-5.995175,5.148476],[-3.414869,-0.976640,4.962671,-4.823930,-5.812761,3.618567,-4.971428,-3.692841,7.876220,9.969039,8.209746,-6.814506],[-7.059754,3.297727,7.853082,-8.051561,1.527642,-0.794838,-9.340967,6.320198,-3.043113,-6.679902,7.492597,4.381463],[4.059539,-1.026573,-2.946592,6.672267,-5.246615,7.108764,7.370362,8.874361,0.611608,-7.213922,-7.147747,-5.615373],[-5.861510,-8.322819,9.670719,-2.403229,-9.957303,9.753664,2.912924,2.079563,4.732110,8.220596,2.105364,-7.193056],[-1.265501,-5.058441,-3.686392,-9.708817,1.383764,-4.257544,-5.979960,3.829544,-3.655724,9.516046,2.131928,4.590871],[-5.958295,1.376749,5.063231,-2.089491,-0.766065,-6.758896,1.732092,3.878851,7.097701,5.883935,8.115467,-4.954611],[-4.932302,0.354408,7.968237,-9.290447,-0.937082,3.371670,1.449248,5.602813,-6.775018,7.903452,-3.099515,-6.186458],[-5.097186,-6.630993,-8.960723,2.967298,-7.809847,-2.246543,1.566291,9.232265,-7.016651,8.932610,-1.613092,-1.329951],[7.054006,-4.701784,-7.358919,0.207697,6.251340,-2.508355,9.904120,-3.499780,1.666375,1.396973,4.838462,-0.253536]],[[1.056793,-0.171935,-4.349723,-0.912407,-1.868994,-9.276678,5.190965,2.315754,3.276095,-3.520167,-2.100632,-9.039603],[1.845837,-6.407213,2.696275,-4.425407,8.696834,-7.711668,5.292236,7.225561,-7.855904,3.418908,4.789063,-6.123161],[-4.404604,9.145919,8.088997,-7.212703,1.724626,5.760311,6.097170,9.664493,-0.409035,6.758790,-3.935452,6.064547],[-7.484429,-0.392531,-9.832426,-5.954139,7.532640,0.535338,3.085934,3.089754,-0.998339,7.871870,5.360568,-6.455449],[3.193144,5.937393,-8.387446,-0.518223,9.796329,9.037110,4.599254,0.707763,0.890373,9.153002,-1.381044,-0.329671],[4.575778,5.761077,-3.750745,3.840113,3.039532,-3.029043,9.865185,-8.239803,4.978162,-9.768988,4.142181,-3.448013],[-9.760890,7.596792,-6.448510,-8.419975,-1.998021,-3.989623,4.110713,-9.192924,3.256472,0.826848,7.030651,7.340245],[6.660304,2.317825,5.551951,6.869232,8.893469,7.895027,-8.728815,4.400744,6.020311,-7.119515,7.745064,6.053887],[5.741102,-1.707073,-7.225785,9.218118,5.633399,-3.185107,-5.468135,7.224072,-3.176034,-6.801895,-1.343506,6.902174],[2.867871,-6.731263,-9.309934,8.055641,-3.838220,-2.989984,-8.630793,-1.516858,-1.655523,-8.934871,7.090811,-3.016793],[-7.622240,-0.823779,5.689020,-6.560963,3.092717,9.212048,-5.248220,2.891735,-2.681855,4.408751,-2.555926,8.110764]],[[-4.436930,9.154114,0.993990,-1.629715,-2.661940,-1.262666,-9.354457,2.534230,4.605837,-4.008677,4.045377,9.152532],[3.203650,-9.897836,-4.314911,9.335561,-6.576921,-5.798392,-0.011093,3.308921,-1.940163,-8.441707,4.761933,-8.421032],[8.607087,-7.184425,6.603425,-8.104731,4.848060,-7.275071,-7.898765,-2.483525,-2.910272,-1.912934,7.348842,-7.188499],[4.862353,2.305777,-3.012814,-6.542794,-7.518447,-6.867820,-3.796618,-3.714814,-9.059866,6.108031,-0.640383,0.434923],[4.230408,-8.296233,1.217164,-5.115700,1.997961,5.822253,6.315731,2.415056,4.946166,8.357313,2.268771,1.890560],[-0.558680,-3.429691,-4.005467,-8.543653,-2.284644,0.104330,5.409042,-7.496244,0.580995,-2.755701,3.973032,-8.884585],[7.274306,-2.892226,1.823642,-7.362527,3.284356,3.004158,-7.783919,1.570305,-4.069810,1.194473,-3.104644,-0.644990],[5.965739,-2.657533,5.066269,6.832659,5.344014,0.053145,2.315371,-7.461734,-7.539603,8.414006,3.782253,6.210860],[7.428597,9.372631,3.885351,2.987895,-6.721417,3.803687,-0.157006,5.562418,9.386297,-3.851613,-3.717079,9.491144],[-7.634756,-7.214949,8.979039,1.104444,-9.728147,-9.599174,0.151720,-5.682127,3.373665,5.588201,6.299531,3.488928],[6.108572,1.584451,5.305138,-5.606010,2.879575,4.078897,4.015500,4.871116,-4.792424,1.265985,6.392066,7.628454]],[[1.828810,-9.287050,7.017000,1.466164,-9.834748,9.972658,0.754284,7.303170,-1.259221,1.030108,-6.024084,8.442942],[-8.610694,-4.902578,-7.740080,1.571927,-5.016233,-9.051022,9.411843,-6.066293,3.170758,8.404038,5.470235,2.945119],[-1.268431,7.425664,-7.591998,0.433731,1.883208,-5.836489,9.336373,-9.646417,-3.891177,-2.600119,-8.559090,4.119891],[-5.560435,4.781622,8.637145,9.315459,1.531732,-5.235467,4.529847,0.384261,2.109042,-9.547107,4.505698,-6.994215],[-0.883041,4.899333,1.570742,-8.819104,-6.926593,-5.370344,-5.273326,1.271359,-0.133998,7.877373,7.262757,7.441101],[0.113892,9.266093,-6.064728,-6.073559,-5.040962,1.973278,-0.687670,2.743767,7.829596,-4.736667,4.882425,-9.014090],[-2.258835,2.937745,-8.064847,-8.501011,0.064136,-3.773927,-6.307696,-5.666765,-6.486838,-7.815607,4.277504,9.286489],[0.020316,-1.156651,1.600666,2.434212,5.320192,0.625543,7.681843,3.475817,-1.737737,-5.594857,-5.152742,8.932688],[-1.167520,0.731343,-5.442124,5.655647,-2.677485,-5.426749,-0.616453,3.391148,3.221340,6.126286,4.006771,6.355993],[-9.414520,-2.633037,-2.277139,1.627330,8.377766,-7.381613,-5.741853,-6.989850,7.835357,-4.007196,-6.559514,5.316218],[7.217916,-4.999602,2.667259,-4.107948,4.477811,-7.763875,-7.601922,-3.628916,-9.881286,8.001018,-8.956089,-5.726607]],[[5.404299,3.917821,-3.214208,8.973209,4.734936,-6.593887,5.069370,-2.700832,7.351714,-0.949966,2.290399,-5.484322],[0.180354,-8.711055,6.596352,3.388430,8.620602,-7.943602,-4.744507,9.416594,8.916341,-7.020778,-4.229792,4.673855],[-3.883880,-5.167803,-4.263365,8.423112,-3.102216,-8.324300,-3.469507,-9.617240,-8.880810,-5.030461,-5.490652,7.934202],[7.147331,-0.534284,-6.746242,-3.935640,1.815698,-7.432368,9.676303,1.373079,9.350513,5.593706,-5.475537,-1.943589],[-7.872224,4.072522,-1.922105,-6.186872,-3.299103,-1.874718,-6.021283,9.196805,7.353418,1.562484,-8.847534,-6.356905],[-8.674386,4.252869,-2.624381,-8.385722,-2.621248,4.280728,-0.749982,4.773487,4.793934,6.783057,-3.262526,5.288131],[4.645695,-3.008077,-5.579932,6.169645,-7.192634,8.732269,-1.513467,3.157922,9.957339,-0.271710,-8.687040,-2.248665],[1.061212,9.345437,0.231899,3.011749,3.131590,-5.046653,-3.434934,1.950921,-2.565304,9.502552,-4.470556,-5.859075],[-4.603544,4.696547,-0.737412,-9.683283,7.369004,3.886316,-6.757072,1.020469,-2.748652,3.496814,-4.947581,1.866711],[3.494007,2.219374,1.405710,7.240953,-9.799686,-3.617158,9.735171,-5.178508,-3.433263,8.820642,-5.211252,-0.422440],[-4.246459,-5.147380,-3.659176,1.562646,-1.147318,9.594537,-1.807927,0.015174,0.967984,6.720287,-2.968057,-4.804500]]], dtype = "float32")#candidate|2078|(9, 11, 12)|const|float32
bop_2079 = relay.less_equal(var_2059.astype('bool'), const_2078.astype('bool')) # shape=(9, 11, 12)
output = relay.Tuple([bop_2060,bop_2079,])
output2 = relay.Tuple([bop_2060,bop_2079,])
func_2082 = relay.Function([var_2059,], output)
mod['func_2082'] = func_2082
mod = relay.transform.InferType()(mod)
var_2083 = relay.var("var_2083", dtype = "float32", shape = (9, 11, 1))#candidate|2083|(9, 11, 1)|var|float32
output = func_2082(var_2083)
func_2084 = relay.Function([var_2083], output)
mutated_mod['func_2084'] = func_2084
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1999_call = mod.get_global_var('func_1999')
func_2000_call = mutated_mod.get_global_var('func_2000')
call_2155 = func_1999_call()
call_2156 = func_1999_call()
func_765_call = mod.get_global_var('func_765')
func_767_call = mutated_mod.get_global_var('func_767')
call_2159 = func_765_call()
call_2160 = func_765_call()
output = relay.Tuple([call_2155,call_2159,])
output2 = relay.Tuple([call_2156,call_2160,])
func_2172 = relay.Function([], output)
mod['func_2172'] = func_2172
mod = relay.transform.InferType()(mod)
mutated_mod['func_2172'] = func_2172
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2172_call = mutated_mod.get_global_var('func_2172')
call_2173 = func_2172_call()
output = call_2173
func_2174 = relay.Function([], output)
mutated_mod['func_2174'] = func_2174
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2181 = relay.const([[-8.486499,5.687703,1.323962,-8.136810,6.721110,2.817526,-6.918660,9.246447,6.741502,-5.024145,4.992870,-6.279603,7.048502,-9.642057,-8.040915],[-0.727995,7.278403,5.766417,3.825986,6.134944,-2.692428,-7.991752,-3.944055,4.037645,-0.490057,3.588984,-8.449712,3.451448,-7.068593,-9.316394],[6.100160,3.522184,-8.812166,-5.958123,3.475498,7.801256,-8.635915,-8.764301,5.740994,9.258527,8.774023,-0.497387,-0.307933,6.079475,3.224611],[2.429047,6.092935,2.260029,4.911890,4.675732,-0.574412,-2.617948,7.199328,8.387247,-0.669769,-8.909164,9.433055,-3.097497,-5.683126,-1.663339],[6.377295,9.578289,9.584380,4.948900,-6.650976,1.096276,3.967798,0.961144,-1.138666,2.012891,9.300521,-4.562178,-1.194972,2.579568,4.503521],[-0.888235,9.882175,4.773164,9.656168,-3.149561,4.951629,-2.376928,0.202167,4.895815,-7.528847,4.089747,8.120225,-7.479197,-2.987643,-9.786605],[5.194545,-2.875688,-5.343666,4.816495,-1.506156,8.092881,0.129891,-7.646361,-6.611886,-7.939148,4.100310,-0.122089,-6.858396,-1.993633,-3.591418],[-1.942956,-2.343729,7.328109,-7.986614,-4.604548,-0.398782,-8.748448,5.185918,-4.427654,4.169439,-8.987352,-1.411740,9.444643,6.631667,7.262937],[-2.635843,9.220267,6.661238,-6.739186,-2.595723,-5.394823,-3.562440,-5.681627,-2.697211,2.001915,-0.335883,-8.238526,1.551603,7.607176,7.892052],[-8.360048,-7.626481,-6.387619,-6.182905,9.628771,1.328921,8.751919,-9.839955,1.681743,8.756268,-1.593261,-2.762005,-4.116853,-5.233976,-2.785077],[9.243408,-4.307968,2.885616,0.563382,5.859031,-7.911112,-0.257894,-0.579676,8.169143,6.778501,8.196584,-1.082891,-1.447227,7.186634,-5.396021],[9.593383,-9.979146,8.236769,-6.762301,-7.816021,0.519108,9.453812,-7.053974,5.438950,4.217195,-6.008571,-1.398097,1.773075,-9.373755,-0.238436],[-0.937052,-4.547522,9.938268,-9.428869,-9.180254,6.300140,9.157643,0.052230,-3.524103,-4.682754,-7.615030,-2.758359,5.590376,8.006567,-1.509029],[4.566858,5.134582,-3.048048,4.889185,9.653150,0.040545,-9.694437,0.390352,-1.308980,-5.487069,9.607268,-4.221034,5.391253,5.643513,3.284924]], dtype = "float32")#candidate|2181|(14, 15)|const|float32
uop_2182 = relay.erf(const_2181.astype('float32')) # shape=(14, 15)
func_222_call = mod.get_global_var('func_222')
func_223_call = mutated_mod.get_global_var('func_223')
call_2187 = relay.TupleGetItem(func_222_call(), 0)
call_2188 = relay.TupleGetItem(func_223_call(), 0)
output = relay.Tuple([uop_2182,call_2187,])
output2 = relay.Tuple([uop_2182,call_2188,])
func_2189 = relay.Function([], output)
mod['func_2189'] = func_2189
mod = relay.transform.InferType()(mod)
mutated_mod['func_2189'] = func_2189
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2189_call = mutated_mod.get_global_var('func_2189')
call_2190 = func_2189_call()
output = call_2190
func_2191 = relay.Function([], output)
mutated_mod['func_2191'] = func_2191
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1104_call = mod.get_global_var('func_1104')
func_1105_call = mutated_mod.get_global_var('func_1105')
call_2192 = relay.TupleGetItem(func_1104_call(), 0)
call_2193 = relay.TupleGetItem(func_1105_call(), 0)
output = relay.Tuple([call_2192,])
output2 = relay.Tuple([call_2193,])
func_2195 = relay.Function([], output)
mod['func_2195'] = func_2195
mod = relay.transform.InferType()(mod)
output = func_2195()
func_2196 = relay.Function([], output)
mutated_mod['func_2196'] = func_2196
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2209 = relay.var("var_2209", dtype = "float64", shape = (4, 7))#candidate|2209|(4, 7)|var|float64
uop_2210 = relay.rsqrt(var_2209.astype('float64')) # shape=(4, 7)
bop_2213 = relay.power(uop_2210.astype('float64'), relay.reshape(var_2209.astype('float64'), relay.shape_of(uop_2210))) # shape=(4, 7)
output = bop_2213
output2 = bop_2213
func_2216 = relay.Function([var_2209,], output)
mod['func_2216'] = func_2216
mod = relay.transform.InferType()(mod)
mutated_mod['func_2216'] = func_2216
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2217 = relay.var("var_2217", dtype = "float64", shape = (4, 7))#candidate|2217|(4, 7)|var|float64
func_2216_call = mutated_mod.get_global_var('func_2216')
call_2218 = func_2216_call(var_2217)
output = call_2218
func_2219 = relay.Function([var_2217], output)
mutated_mod['func_2219'] = func_2219
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2245 = relay.var("var_2245", dtype = "float64", shape = (4, 7))#candidate|2245|(4, 7)|var|float64
uop_2246 = relay.sinh(var_2245.astype('float64')) # shape=(4, 7)
var_2249 = relay.var("var_2249", dtype = "float64", shape = (4, 7))#candidate|2249|(4, 7)|var|float64
bop_2250 = relay.maximum(uop_2246.astype('int64'), relay.reshape(var_2249.astype('int64'), relay.shape_of(uop_2246))) # shape=(4, 7)
bop_2255 = relay.bitwise_and(bop_2250.astype('uint32'), relay.reshape(var_2245.astype('uint32'), relay.shape_of(bop_2250))) # shape=(4, 7)
bop_2259 = relay.floor_divide(var_2249.astype('float64'), relay.reshape(bop_2250.astype('float64'), relay.shape_of(var_2249))) # shape=(4, 7)
var_2267 = relay.var("var_2267", dtype = "uint32", shape = (4, 7))#candidate|2267|(4, 7)|var|uint32
bop_2268 = relay.greater(bop_2255.astype('bool'), relay.reshape(var_2267.astype('bool'), relay.shape_of(bop_2255))) # shape=(4, 7)
uop_2285 = relay.asinh(uop_2246.astype('float64')) # shape=(4, 7)
func_1666_call = mod.get_global_var('func_1666')
func_1668_call = mutated_mod.get_global_var('func_1668')
call_2288 = func_1666_call()
call_2289 = func_1666_call()
uop_2292 = relay.atan(uop_2285.astype('float64')) # shape=(4, 7)
uop_2294 = relay.log10(bop_2259.astype('float32')) # shape=(4, 7)
bop_2299 = relay.minimum(uop_2285.astype('int16'), relay.reshape(uop_2292.astype('int16'), relay.shape_of(uop_2285))) # shape=(4, 7)
uop_2305 = relay.log(uop_2294.astype('float32')) # shape=(4, 7)
uop_2307 = relay.acosh(uop_2294.astype('float32')) # shape=(4, 7)
func_973_call = mod.get_global_var('func_973')
func_975_call = mutated_mod.get_global_var('func_975')
var_2310 = relay.var("var_2310", dtype = "bool", shape = (33,))#candidate|2310|(33,)|var|bool
call_2309 = func_973_call(relay.reshape(var_2310.astype('bool'), [3, 11]))
call_2311 = func_973_call(relay.reshape(var_2310.astype('bool'), [3, 11]))
bop_2314 = relay.divide(bop_2255.astype('float32'), relay.reshape(var_2267.astype('float32'), relay.shape_of(bop_2255))) # shape=(4, 7)
func_1802_call = mod.get_global_var('func_1802')
func_1806_call = mutated_mod.get_global_var('func_1806')
var_2319 = relay.var("var_2319", dtype = "int16", shape = (810,))#candidate|2319|(810,)|var|int16
call_2318 = relay.TupleGetItem(func_1802_call(relay.reshape(var_2319.astype('int16'), [15, 6, 9]), relay.reshape(var_2319.astype('int16'), [15, 6, 9]), ), 0)
call_2320 = relay.TupleGetItem(func_1806_call(relay.reshape(var_2319.astype('int16'), [15, 6, 9]), relay.reshape(var_2319.astype('int16'), [15, 6, 9]), ), 0)
func_558_call = mod.get_global_var('func_558')
func_560_call = mutated_mod.get_global_var('func_560')
call_2321 = relay.TupleGetItem(func_558_call(), 2)
call_2322 = relay.TupleGetItem(func_560_call(), 2)
bop_2328 = relay.logical_and(uop_2246.astype('bool'), relay.reshape(bop_2314.astype('bool'), relay.shape_of(uop_2246))) # shape=(4, 7)
bop_2333 = relay.logical_or(bop_2299.astype('bool'), relay.reshape(bop_2314.astype('bool'), relay.shape_of(bop_2299))) # shape=(4, 7)
output = relay.Tuple([bop_2268,call_2288,uop_2305,uop_2307,call_2309,var_2310,call_2318,var_2319,call_2321,bop_2328,bop_2333,])
output2 = relay.Tuple([bop_2268,call_2289,uop_2305,uop_2307,call_2311,var_2310,call_2320,var_2319,call_2322,bop_2328,bop_2333,])
func_2336 = relay.Function([var_2245,var_2249,var_2267,var_2310,var_2319,], output)
mod['func_2336'] = func_2336
mod = relay.transform.InferType()(mod)
mutated_mod['func_2336'] = func_2336
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2336_call = mutated_mod.get_global_var('func_2336')
var_2338 = relay.var("var_2338", dtype = "float64", shape = (4, 7))#candidate|2338|(4, 7)|var|float64
var_2339 = relay.var("var_2339", dtype = "float64", shape = (4, 7))#candidate|2339|(4, 7)|var|float64
var_2340 = relay.var("var_2340", dtype = "uint32", shape = (4, 7))#candidate|2340|(4, 7)|var|uint32
var_2341 = relay.var("var_2341", dtype = "bool", shape = (33,))#candidate|2341|(33,)|var|bool
var_2342 = relay.var("var_2342", dtype = "int16", shape = (810,))#candidate|2342|(810,)|var|int16
call_2337 = func_2336_call(var_2338,var_2339,var_2340,var_2341,var_2342,)
output = call_2337
func_2343 = relay.Function([var_2338,var_2339,var_2340,var_2341,var_2342,], output)
mutated_mod['func_2343'] = func_2343
mutated_mod = relay.transform.InferType()(mutated_mod)
func_600_call = mod.get_global_var('func_600')
func_602_call = mutated_mod.get_global_var('func_602')
call_2372 = relay.TupleGetItem(func_600_call(), 3)
call_2373 = relay.TupleGetItem(func_602_call(), 3)
func_2216_call = mod.get_global_var('func_2216')
func_2219_call = mutated_mod.get_global_var('func_2219')
const_2376 = relay.const([-9.559749,9.652900,-4.624873,-0.024464,-7.703407,4.426079,1.459117,-6.434970,-7.451941,0.229130,-9.290631,-0.790462,-5.695762,-0.104973,-6.505606,3.693684,-9.665306,4.971360,1.460160,1.072858,3.231177,7.005337,2.902919,4.242563,9.139369,2.386968,5.206289,9.426036], dtype = "float64")#candidate|2376|(28,)|const|float64
call_2375 = func_2216_call(relay.reshape(const_2376.astype('float64'), [4, 7]))
call_2377 = func_2216_call(relay.reshape(const_2376.astype('float64'), [4, 7]))
var_2386 = relay.var("var_2386", dtype = "int64", shape = (384,))#candidate|2386|(384,)|var|int64
bop_2387 = relay.floor_mod(call_2372.astype('float32'), relay.reshape(var_2386.astype('float32'), relay.shape_of(call_2372))) # shape=(384,)
bop_2390 = relay.floor_mod(call_2373.astype('float32'), relay.reshape(var_2386.astype('float32'), relay.shape_of(call_2373))) # shape=(384,)
output = relay.Tuple([call_2375,const_2376,bop_2387,])
output2 = relay.Tuple([call_2377,const_2376,bop_2390,])
func_2393 = relay.Function([var_2386,], output)
mod['func_2393'] = func_2393
mod = relay.transform.InferType()(mod)
var_2394 = relay.var("var_2394", dtype = "int64", shape = (384,))#candidate|2394|(384,)|var|int64
output = func_2393(var_2394)
func_2395 = relay.Function([var_2394], output)
mutated_mod['func_2395'] = func_2395
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1198_call = mutated_mod.get_global_var('func_1198')
call_2397 = func_1197_call()
call_2398 = func_1197_call()
output = call_2397
output2 = call_2398
func_2399 = relay.Function([], output)
mod['func_2399'] = func_2399
mod = relay.transform.InferType()(mod)
mutated_mod['func_2399'] = func_2399
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2399_call = mutated_mod.get_global_var('func_2399')
call_2400 = func_2399_call()
output = call_2400
func_2401 = relay.Function([], output)
mutated_mod['func_2401'] = func_2401
mutated_mod = relay.transform.InferType()(mutated_mod)
func_558_call = mod.get_global_var('func_558')
func_560_call = mutated_mod.get_global_var('func_560')
call_2433 = relay.TupleGetItem(func_558_call(), 3)
call_2434 = relay.TupleGetItem(func_560_call(), 3)
uop_2436 = relay.asin(call_2433.astype('float32')) # shape=(10, 5)
uop_2438 = relay.asin(call_2434.astype('float32')) # shape=(10, 5)
uop_2439 = relay.cos(uop_2436.astype('float32')) # shape=(10, 5)
uop_2441 = relay.cos(uop_2438.astype('float32')) # shape=(10, 5)
output = uop_2439
output2 = uop_2441
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
	relay.transform.Legalize(),
	relay.transform.FoldConstant(),
	relay.transform.ToANormalForm(),
	relay.transform.ToGraphNormalForm(),
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

'''39: TVMFuncCall
38: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
37: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
36: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
35: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
34: tvm::transform::Pass::operator()(tvm::IRModule) const
33: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
30: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
29: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
28: tvm::relay::DynamicToStatic(tvm::relay::Function, tvm::IRModule)
27: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
26: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
25: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
24: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
23: _ZN3tvm5relay16MixedModeMutato
22: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
21: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
20: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
18: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
17: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
16: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
15: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
14: _ZN3tvm5relay16MixedModeMutato
13: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
12: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
11: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
10: tvm::relay::MixedModeMutator::VisitExpr_(tvm::relay::CallNode const*)
9: tvm::relay::DynamicToStaticMutator::Rewrite_(tvm::relay::CallNode const*, tvm::RelayExpr const&)
8: std::_Function_handler<tvm::RelayExpr (tvm::relay::CallNode const*), tvm::relay::DynamicToStaticMutator::DynamicToStaticMutator(tvm::IRModule, tvm::relay::Function)::{lambda(tvm::relay::CallNode const*)#1}>::_M_invoke(std::_Any_data const&, tvm::relay::CallNode const*&&)
7: tvm::relay::DynamicToStaticMutator::PrepareArgs(tvm::relay::CallNode const*)
6: tvm::relay::DynamicToStaticMutator::PrepareInput(tvm::RelayExpr const&)
5: tvm::transform::Pass::operator()(tvm::IRModule) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}::operator()(tvm::IRModule, tvm::transform::PassContext const&) const [clone .isra.813]
0: tvm::DiagnosticContext::Render()

'''