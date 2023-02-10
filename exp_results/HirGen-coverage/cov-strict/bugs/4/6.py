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
var_53 = relay.var("var_53", dtype = "float64", shape = (15, 9))#candidate|53|(15, 9)|var|float64
uop_54 = relay.cosh(var_53.astype('float64')) # shape=(15, 9)
var_68 = relay.var("var_68", dtype = "float64", shape = (15, 9))#candidate|68|(15, 9)|var|float64
bop_69 = relay.minimum(uop_54.astype('int64'), relay.reshape(var_68.astype('int64'), relay.shape_of(uop_54))) # shape=(15, 9)
bop_75 = relay.subtract(var_68.astype('float32'), relay.reshape(uop_54.astype('float32'), relay.shape_of(var_68))) # shape=(15, 9)
output = relay.Tuple([bop_69,bop_75,])
output2 = relay.Tuple([bop_69,bop_75,])
func_80 = relay.Function([var_53,var_68,], output)
mod['func_80'] = func_80
mod = relay.transform.InferType()(mod)
mutated_mod['func_80'] = func_80
mutated_mod = relay.transform.InferType()(mutated_mod)
func_80_call = mutated_mod.get_global_var('func_80')
var_82 = relay.var("var_82", dtype = "float64", shape = (15, 9))#candidate|82|(15, 9)|var|float64
var_83 = relay.var("var_83", dtype = "float64", shape = (15, 9))#candidate|83|(15, 9)|var|float64
call_81 = func_80_call(var_82,var_83,)
output = call_81
func_84 = relay.Function([var_82,var_83,], output)
mutated_mod['func_84'] = func_84
mutated_mod = relay.transform.InferType()(mutated_mod)
var_86 = relay.var("var_86", dtype = "float32", shape = (10, 3))#candidate|86|(10, 3)|var|float32
uop_87 = relay.cosh(var_86.astype('float32')) # shape=(10, 3)
var_89 = relay.var("var_89", dtype = "float32", shape = (10, 3))#candidate|89|(10, 3)|var|float32
bop_90 = relay.equal(uop_87.astype('bool'), relay.reshape(var_89.astype('bool'), relay.shape_of(uop_87))) # shape=(10, 3)
output = bop_90
output2 = bop_90
func_93 = relay.Function([var_86,var_89,], output)
mod['func_93'] = func_93
mod = relay.transform.InferType()(mod)
mutated_mod['func_93'] = func_93
mutated_mod = relay.transform.InferType()(mutated_mod)
func_93_call = mutated_mod.get_global_var('func_93')
var_95 = relay.var("var_95", dtype = "float32", shape = (10, 3))#candidate|95|(10, 3)|var|float32
var_96 = relay.var("var_96", dtype = "float32", shape = (10, 3))#candidate|96|(10, 3)|var|float32
call_94 = func_93_call(var_95,var_96,)
output = call_94
func_97 = relay.Function([var_95,var_96,], output)
mutated_mod['func_97'] = func_97
mutated_mod = relay.transform.InferType()(mutated_mod)
const_212 = relay.const(-7, dtype = "uint8")#candidate|212|()|const|uint8
var_213 = relay.var("var_213", dtype = "uint8", shape = (7, 7, 1))#candidate|213|(7, 7, 1)|var|uint8
bop_214 = relay.logical_xor(const_212.astype('uint8'), var_213.astype('uint8')) # shape=(7, 7, 1)
uop_217 = relay.sqrt(bop_214.astype('float64')) # shape=(7, 7, 1)
bop_219 = relay.add(uop_217.astype('uint64'), relay.reshape(bop_214.astype('uint64'), relay.shape_of(uop_217))) # shape=(7, 7, 1)
var_225 = relay.var("var_225", dtype = "uint64", shape = (7, 7, 13))#candidate|225|(7, 7, 13)|var|uint64
bop_226 = relay.logical_or(bop_219.astype('bool'), var_225.astype('bool')) # shape=(7, 7, 13)
output = bop_226
output2 = bop_226
func_229 = relay.Function([var_213,var_225,], output)
mod['func_229'] = func_229
mod = relay.transform.InferType()(mod)
mutated_mod['func_229'] = func_229
mutated_mod = relay.transform.InferType()(mutated_mod)
func_229_call = mutated_mod.get_global_var('func_229')
var_231 = relay.var("var_231", dtype = "uint8", shape = (7, 7, 1))#candidate|231|(7, 7, 1)|var|uint8
var_232 = relay.var("var_232", dtype = "uint64", shape = (7, 7, 13))#candidate|232|(7, 7, 13)|var|uint64
call_230 = func_229_call(var_231,var_232,)
output = call_230
func_233 = relay.Function([var_231,var_232,], output)
mutated_mod['func_233'] = func_233
mutated_mod = relay.transform.InferType()(mutated_mod)
var_267 = relay.var("var_267", dtype = "int32", shape = ())#candidate|267|()|var|int32
const_268 = relay.const([[[-4,9,-2,7,-10,-10,9,-6,9]],[[-5,-1,4,2,5,9,-5,-6,-8]],[[-9,-6,2,-7,6,-3,1,8,6]],[[8,-1,2,-4,-4,-3,3,-5,-2]],[[7,-8,-3,7,-1,-8,7,3,-10]],[[10,3,-6,-6,10,-8,-10,2,-9]],[[1,10,3,6,-3,-9,-7,-9,-5]],[[-9,-4,4,-3,-8,-2,-1,2,-2]]], dtype = "int32")#candidate|268|(8, 1, 9)|const|int32
bop_269 = relay.left_shift(var_267.astype('int32'), const_268.astype('int32')) # shape=(8, 1, 9)
uop_274 = relay.log(const_268.astype('float32')) # shape=(8, 1, 9)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
var_277 = relay.var("var_277", dtype = "float64", shape = (135,))#candidate|277|(135,)|var|float64
call_276 = relay.TupleGetItem(func_80_call(relay.reshape(var_277.astype('float64'), [15, 9]), relay.reshape(var_277.astype('float64'), [15, 9]), ), 1)
call_278 = relay.TupleGetItem(func_84_call(relay.reshape(var_277.astype('float64'), [15, 9]), relay.reshape(var_277.astype('float64'), [15, 9]), ), 1)
func_93_call = mod.get_global_var('func_93')
func_97_call = mutated_mod.get_global_var('func_97')
var_281 = relay.var("var_281", dtype = "float32", shape = (5, 6))#candidate|281|(5, 6)|var|float32
call_280 = func_93_call(relay.reshape(var_281.astype('float32'), [10, 3]), relay.reshape(var_281.astype('float32'), [10, 3]), )
call_282 = func_93_call(relay.reshape(var_281.astype('float32'), [10, 3]), relay.reshape(var_281.astype('float32'), [10, 3]), )
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
const_287 = relay.const([4,7,-8,-10,10,3,3,-5,3,6,-5,-10,9,9,4,6,-9,5,-8,6,-6,5,-10,-4,10,-10,-8,-1,-6,4,-6,9,5,-5,-8,-1,1,-4,10,6,9,-5,8,3,8,-3,2,-2,-5], dtype = "uint8")#candidate|287|(49,)|const|uint8
var_288 = relay.var("var_288", dtype = "uint64", shape = (637,))#candidate|288|(637,)|var|uint64
call_286 = func_229_call(relay.reshape(const_287.astype('uint8'), [7, 7, 1]), relay.reshape(var_288.astype('uint64'), [7, 7, 13]), )
call_289 = func_229_call(relay.reshape(const_287.astype('uint8'), [7, 7, 1]), relay.reshape(var_288.astype('uint64'), [7, 7, 13]), )
uop_290 = relay.sin(uop_274.astype('float32')) # shape=(8, 1, 9)
uop_292 = relay.atan(uop_290.astype('float32')) # shape=(8, 1, 9)
bop_297 = relay.floor_divide(uop_292.astype('float64'), call_276.astype('float64')) # shape=(8, 15, 9)
bop_300 = relay.floor_divide(uop_292.astype('float64'), call_278.astype('float64')) # shape=(8, 15, 9)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
call_304 = func_229_call(relay.reshape(const_287.astype('uint8'), [7, 7, 1]), relay.reshape(call_286.astype('uint64'), [7, 7, 13]), )
call_305 = func_229_call(relay.reshape(const_287.astype('uint8'), [7, 7, 1]), relay.reshape(call_286.astype('uint64'), [7, 7, 13]), )
uop_310 = relay.log2(bop_297.astype('float32')) # shape=(8, 15, 9)
uop_312 = relay.log2(bop_300.astype('float32')) # shape=(8, 15, 9)
uop_315 = relay.erf(uop_310.astype('float64')) # shape=(8, 15, 9)
uop_317 = relay.erf(uop_312.astype('float64')) # shape=(8, 15, 9)
output = relay.Tuple([bop_269,var_277,call_280,var_281,call_286,const_287,var_288,call_304,uop_315,])
output2 = relay.Tuple([bop_269,var_277,call_282,var_281,call_289,const_287,var_288,call_305,uop_317,])
func_318 = relay.Function([var_267,var_277,var_281,var_288,], output)
mod['func_318'] = func_318
mod = relay.transform.InferType()(mod)
var_319 = relay.var("var_319", dtype = "int32", shape = ())#candidate|319|()|var|int32
var_320 = relay.var("var_320", dtype = "float64", shape = (135,))#candidate|320|(135,)|var|float64
var_321 = relay.var("var_321", dtype = "float32", shape = (5, 6))#candidate|321|(5, 6)|var|float32
var_322 = relay.var("var_322", dtype = "uint64", shape = (637,))#candidate|322|(637,)|var|uint64
output = func_318(var_319,var_320,var_321,var_322,)
func_323 = relay.Function([var_319,var_320,var_321,var_322,], output)
mutated_mod['func_323'] = func_323
mutated_mod = relay.transform.InferType()(mutated_mod)
var_328 = relay.var("var_328", dtype = "float64", shape = (9, 15, 7))#candidate|328|(9, 15, 7)|var|float64
uop_329 = relay.asinh(var_328.astype('float64')) # shape=(9, 15, 7)
bop_332 = relay.less_equal(uop_329.astype('bool'), relay.reshape(var_328.astype('bool'), relay.shape_of(uop_329))) # shape=(9, 15, 7)
bop_335 = relay.bitwise_or(bop_332.astype('uint16'), relay.reshape(uop_329.astype('uint16'), relay.shape_of(bop_332))) # shape=(9, 15, 7)
uop_338 = relay.cos(bop_335.astype('float32')) # shape=(9, 15, 7)
var_341 = relay.var("var_341", dtype = "float32", shape = (9, 15, 7))#candidate|341|(9, 15, 7)|var|float32
bop_342 = relay.power(uop_338.astype('float32'), relay.reshape(var_341.astype('float32'), relay.shape_of(uop_338))) # shape=(9, 15, 7)
output = bop_342
output2 = bop_342
func_349 = relay.Function([var_328,var_341,], output)
mod['func_349'] = func_349
mod = relay.transform.InferType()(mod)
mutated_mod['func_349'] = func_349
mutated_mod = relay.transform.InferType()(mutated_mod)
func_349_call = mutated_mod.get_global_var('func_349')
var_351 = relay.var("var_351", dtype = "float64", shape = (9, 15, 7))#candidate|351|(9, 15, 7)|var|float64
var_352 = relay.var("var_352", dtype = "float32", shape = (9, 15, 7))#candidate|352|(9, 15, 7)|var|float32
call_350 = func_349_call(var_351,var_352,)
output = call_350
func_353 = relay.Function([var_351,var_352,], output)
mutated_mod['func_353'] = func_353
mutated_mod = relay.transform.InferType()(mutated_mod)
var_383 = relay.var("var_383", dtype = "float64", shape = (9, 6, 6))#candidate|383|(9, 6, 6)|var|float64
uop_384 = relay.acos(var_383.astype('float64')) # shape=(9, 6, 6)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
var_388 = relay.var("var_388", dtype = "uint8", shape = (49,))#candidate|388|(49,)|var|uint8
var_389 = relay.var("var_389", dtype = "uint64", shape = (637,))#candidate|389|(637,)|var|uint64
call_387 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(var_389.astype('uint64'), [7, 7, 13]), )
call_390 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(var_389.astype('uint64'), [7, 7, 13]), )
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
call_393 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(var_389.astype('uint64'), [7, 7, 13]), )
call_394 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(var_389.astype('uint64'), [7, 7, 13]), )
func_318_call = mod.get_global_var('func_318')
func_323_call = mutated_mod.get_global_var('func_323')
var_396 = relay.var("var_396", dtype = "int32", shape = ())#candidate|396|()|var|int32
const_397 = relay.const([2.418523,-6.696740,5.886188,7.823364,-6.269682,4.735500,-2.101959,0.422892,-6.117826,1.273654,3.213272,5.441407,-5.635840,2.129866,-0.414809,0.804473,1.413782,-2.150925,1.711664,7.311643,-0.767228,5.377372,8.378822,-8.377336,7.665665,-3.461474,-5.520737,2.026273,4.148134,6.994201,0.507434,3.013927,-8.230365,8.841167,2.623642,8.094973,-9.567161,-4.907627,4.078483,7.285125,-9.659538,3.160552,9.834043,1.653256,4.558888,6.488486,8.941279,-9.551355,-4.432474,5.417656,-5.040603,0.169674,0.256059,7.553621,1.157285,8.873141,-1.859511,-8.775382,-8.496830,-2.376112,-2.342186,-0.952984,-9.092321,-2.072892,-5.638697,-0.373118,-0.364024,8.338965,5.807939,5.637335,-5.524674,-9.177745,4.745114,-7.519494,-8.547456,-2.114351,-2.712311,-5.584157,8.098419,-6.497027,1.936356,-2.409601,-7.792361,8.107144,-1.562049,-7.089624,-0.010882,0.759480,-0.070123,0.033590,-9.313401,6.331053,-2.466450,8.748807,-1.233962,2.445136,-4.674667,-6.156005,-1.761847,-6.734943,-7.678667,7.431799,0.208630,-6.307891,-1.872932,-6.212857,4.548540,-0.114364,-3.212926,6.125425,5.474190,-1.289850,6.841139,-6.798736,-3.600674,-1.523638,-9.088752,1.579893,6.601981,2.057168,6.560610,-4.950767,3.259026,-7.185302,5.742027,4.817282,2.879511,-2.680267,-0.935264,-8.769972,-4.313513,9.099452,-1.952493,-4.837016,-3.050801], dtype = "float64")#candidate|397|(135,)|const|float64
const_398 = relay.const([[-3.435485,7.250596,5.299518,-3.359476,-4.925168,5.183476,-6.244728,5.389066,6.249457,7.164608,-2.436394,2.332736,1.077551,-3.363551,-1.194413,4.952444,-6.676455,-9.996035,2.431505,-6.456976,-5.459342,-5.012757,2.955780,7.156213,-0.007785,-7.959968,-5.229125,-8.605119,-5.457878,5.653901]], dtype = "float32")#candidate|398|(1, 30)|const|float32
call_395 = relay.TupleGetItem(func_318_call(relay.reshape(var_396.astype('int32'), []), relay.reshape(const_397.astype('float64'), [135,]), relay.reshape(const_398.astype('float32'), [5, 6]), relay.reshape(call_387.astype('uint64'), [637,]), ), 8)
call_399 = relay.TupleGetItem(func_323_call(relay.reshape(var_396.astype('int32'), []), relay.reshape(const_397.astype('float64'), [135,]), relay.reshape(const_398.astype('float32'), [5, 6]), relay.reshape(call_387.astype('uint64'), [637,]), ), 8)
func_349_call = mod.get_global_var('func_349')
func_353_call = mutated_mod.get_global_var('func_353')
var_401 = relay.var("var_401", dtype = "float64", shape = (945,))#candidate|401|(945,)|var|float64
call_400 = func_349_call(relay.reshape(var_401.astype('float64'), [9, 15, 7]), relay.reshape(var_401.astype('float32'), [9, 15, 7]), )
call_402 = func_349_call(relay.reshape(var_401.astype('float64'), [9, 15, 7]), relay.reshape(var_401.astype('float32'), [9, 15, 7]), )
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
call_403 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(var_389.astype('uint64'), [7, 7, 13]), )
call_404 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(var_389.astype('uint64'), [7, 7, 13]), )
bop_407 = relay.not_equal(uop_384.astype('bool'), relay.reshape(var_383.astype('bool'), relay.shape_of(uop_384))) # shape=(9, 6, 6)
uop_411 = relay.erf(bop_407.astype('float64')) # shape=(9, 6, 6)
func_93_call = mod.get_global_var('func_93')
func_97_call = mutated_mod.get_global_var('func_97')
call_414 = func_93_call(relay.reshape(const_398.astype('float32'), [10, 3]), relay.reshape(const_398.astype('float32'), [10, 3]), )
call_415 = func_93_call(relay.reshape(const_398.astype('float32'), [10, 3]), relay.reshape(const_398.astype('float32'), [10, 3]), )
var_417 = relay.var("var_417", dtype = "float64", shape = (9, 6, 6))#candidate|417|(9, 6, 6)|var|float64
bop_418 = relay.floor_divide(uop_411.astype('float32'), relay.reshape(var_417.astype('float32'), relay.shape_of(uop_411))) # shape=(9, 6, 6)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
call_426 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(call_393.astype('uint64'), [7, 7, 13]), )
call_427 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(call_393.astype('uint64'), [7, 7, 13]), )
func_93_call = mod.get_global_var('func_93')
func_97_call = mutated_mod.get_global_var('func_97')
call_428 = func_93_call(relay.reshape(const_398.astype('float32'), [10, 3]), relay.reshape(const_398.astype('float32'), [10, 3]), )
call_429 = func_93_call(relay.reshape(const_398.astype('float32'), [10, 3]), relay.reshape(const_398.astype('float32'), [10, 3]), )
const_430 = relay.const([[[-3.481109,1.498926,-8.721437,1.849842,1.636577,-4.526259],[-1.729528,-0.061720,-9.377413,-8.686095,1.325177,-7.277994],[1.349051,-5.439900,-2.167443,2.991798,0.349682,-7.505460],[-6.910411,9.629546,-6.057452,-3.221146,-7.342304,-4.760481],[-7.675796,-6.541917,6.208957,6.780292,6.418714,-9.027011],[5.797349,-9.912169,-0.554349,-0.014312,1.579660,-8.650546]],[[-3.214188,-1.924401,3.609458,4.008398,2.086897,-5.459780],[-0.136164,0.596485,1.403465,-5.383824,3.491970,4.609313],[0.931930,-2.690162,7.486509,-6.669572,2.288409,-1.982532],[-0.383872,5.882848,6.328766,-6.224802,1.619931,-5.836117],[-5.752392,-8.504725,4.849024,5.792433,0.816811,0.111557],[3.418086,-4.377917,-2.447336,1.084192,-3.402393,-9.408417]],[[-6.411465,-7.604172,-1.935540,3.387395,5.121768,5.783020],[6.955430,1.245359,-9.531319,-1.999493,-6.317233,5.722453],[2.543784,7.197324,3.615834,-2.026530,5.755608,8.890005],[2.262118,-4.342002,-2.076142,8.051654,-4.227847,6.549563],[-7.860372,-1.737944,-1.087226,3.331690,4.413113,-5.187683],[0.524264,-2.458465,1.884996,-8.753208,-2.794512,9.473213]],[[-1.201022,9.540591,8.632985,-2.400478,-4.231861,-2.946893],[-0.953992,-8.696380,8.090574,-3.156707,0.534803,2.944821],[-8.633532,6.219649,-6.157922,-6.342713,-6.537830,8.297069],[-7.343377,3.805400,6.615596,1.400101,2.286576,9.338726],[-0.436679,2.560405,-3.665083,-5.917929,8.877663,-6.997888],[-4.703887,-2.398532,-2.416530,-6.296253,8.420123,1.386704]],[[-9.294940,-1.272791,-0.045649,-2.024166,9.660042,7.712799],[-7.458235,5.751364,9.436658,3.313123,-3.051732,-2.899573],[4.704727,9.470110,-2.547273,7.128563,3.959943,2.035686],[1.796066,1.194017,3.646998,6.524277,-2.557489,4.458280],[-0.606662,-9.463020,6.622352,6.465789,7.472006,7.967331],[5.864797,-3.569336,6.787449,-2.623536,7.159998,2.542371]],[[-6.880554,-6.737992,0.503596,1.335719,-4.884730,-0.779459],[-3.375528,4.751504,-9.346403,-2.696207,2.051312,6.870601],[-7.705398,-1.280612,0.038531,-7.471546,-9.882107,-1.818369],[4.294244,9.273991,8.131813,9.624392,8.977447,-0.450214],[7.812817,5.707819,8.074151,7.814742,-1.905924,-2.221119],[-3.009759,-1.004489,7.067276,-2.409040,-1.147552,9.300315]],[[-5.563832,4.906854,1.614443,9.471431,-0.046800,1.088765],[1.101800,6.298364,-6.836979,7.604400,0.936247,7.412005],[9.758788,7.735840,5.691903,6.957027,2.421179,6.006215],[-1.078321,0.289184,4.001718,7.367291,-0.697575,9.250221],[-0.919809,5.757891,-2.762777,-5.377661,-4.145138,1.767100],[-4.561127,4.588073,5.481483,-2.930540,-4.591246,5.269823]],[[9.696331,-5.526988,9.708699,-4.675319,7.741385,4.665074],[8.465636,6.501813,3.187479,2.658458,5.646122,8.046987],[-5.562167,7.352590,-7.535040,-9.688270,4.044172,3.954608],[-5.081768,-6.985753,5.191541,7.953972,2.765317,-8.829447],[8.946636,2.368360,8.932037,-8.715752,-4.522768,-1.753003],[-6.221231,-0.236337,8.186974,-5.219892,-3.695569,7.348971]],[[2.508809,-2.236986,3.466641,4.014718,-2.759663,-1.932871],[-2.201740,3.802921,-6.851834,8.994900,3.934115,-8.456719],[5.146843,-9.691164,0.252838,0.882226,9.749481,2.728617],[-4.109343,2.985966,2.497727,-1.286809,-5.466322,4.920182],[1.053044,3.436275,-5.250467,-5.588883,9.531451,6.678637],[-9.000308,8.424821,-9.717566,1.345757,4.553101,-3.140061]]], dtype = "float32")#candidate|430|(9, 6, 6)|const|float32
bop_431 = relay.floor_mod(bop_418.astype('float32'), relay.reshape(const_430.astype('float32'), relay.shape_of(bop_418))) # shape=(9, 6, 6)
bop_434 = relay.add(bop_407.astype('uint8'), relay.reshape(uop_411.astype('uint8'), relay.shape_of(bop_407))) # shape=(9, 6, 6)
uop_437 = relay.cos(bop_407.astype('float32')) # shape=(9, 6, 6)
bop_439 = relay.power(bop_434.astype('float64'), relay.reshape(bop_407.astype('float64'), relay.shape_of(bop_434))) # shape=(9, 6, 6)
uop_442 = relay.sinh(uop_384.astype('float64')) # shape=(9, 6, 6)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
call_447 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(call_403.astype('uint64'), [7, 7, 13]), )
call_448 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(call_403.astype('uint64'), [7, 7, 13]), )
bop_453 = relay.greater_equal(bop_439.astype('bool'), relay.reshape(bop_418.astype('bool'), relay.shape_of(bop_439))) # shape=(9, 6, 6)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
call_457 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(call_426.astype('uint64'), [7, 7, 13]), )
call_458 = func_229_call(relay.reshape(var_388.astype('uint8'), [7, 7, 1]), relay.reshape(call_426.astype('uint64'), [7, 7, 13]), )
func_93_call = mod.get_global_var('func_93')
func_97_call = mutated_mod.get_global_var('func_97')
call_460 = func_93_call(relay.reshape(call_414.astype('float32'), [10, 3]), relay.reshape(const_398.astype('float32'), [10, 3]), )
call_461 = func_93_call(relay.reshape(call_414.astype('float32'), [10, 3]), relay.reshape(const_398.astype('float32'), [10, 3]), )
bop_462 = relay.equal(uop_411.astype('bool'), relay.reshape(bop_434.astype('bool'), relay.shape_of(uop_411))) # shape=(9, 6, 6)
output = relay.Tuple([call_387,var_388,var_389,call_393,call_395,var_396,const_397,const_398,call_400,var_401,call_403,call_414,call_426,call_428,bop_431,uop_437,uop_442,call_447,bop_453,call_457,call_460,bop_462,])
output2 = relay.Tuple([call_390,var_388,var_389,call_394,call_399,var_396,const_397,const_398,call_402,var_401,call_404,call_415,call_427,call_429,bop_431,uop_437,uop_442,call_448,bop_453,call_458,call_461,bop_462,])
func_465 = relay.Function([var_383,var_388,var_389,var_396,var_401,var_417,], output)
mod['func_465'] = func_465
mod = relay.transform.InferType()(mod)
var_466 = relay.var("var_466", dtype = "float64", shape = (9, 6, 6))#candidate|466|(9, 6, 6)|var|float64
var_467 = relay.var("var_467", dtype = "uint8", shape = (49,))#candidate|467|(49,)|var|uint8
var_468 = relay.var("var_468", dtype = "uint64", shape = (637,))#candidate|468|(637,)|var|uint64
var_469 = relay.var("var_469", dtype = "int32", shape = ())#candidate|469|()|var|int32
var_470 = relay.var("var_470", dtype = "float64", shape = (945,))#candidate|470|(945,)|var|float64
var_471 = relay.var("var_471", dtype = "float64", shape = (9, 6, 6))#candidate|471|(9, 6, 6)|var|float64
output = func_465(var_466,var_467,var_468,var_469,var_470,var_471,)
func_472 = relay.Function([var_466,var_467,var_468,var_469,var_470,var_471,], output)
mutated_mod['func_472'] = func_472
mutated_mod = relay.transform.InferType()(mutated_mod)
const_523 = relay.const([-6.885895,0.686722,9.299625,-0.352528,-0.499931,0.045291,4.619449,8.437325,-9.309410,0.056482,-1.753560,6.128748], dtype = "float64")#candidate|523|(12,)|const|float64
uop_524 = relay.sin(const_523.astype('float64')) # shape=(12,)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
var_529 = relay.var("var_529", dtype = "float64", shape = (135,))#candidate|529|(135,)|var|float64
call_528 = relay.TupleGetItem(func_80_call(relay.reshape(var_529.astype('float64'), [15, 9]), relay.reshape(var_529.astype('float64'), [15, 9]), ), 1)
call_530 = relay.TupleGetItem(func_84_call(relay.reshape(var_529.astype('float64'), [15, 9]), relay.reshape(var_529.astype('float64'), [15, 9]), ), 1)
uop_542 = relay.log10(uop_524.astype('float32')) # shape=(12,)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
var_553 = relay.var("var_553", dtype = "uint8", shape = (49,))#candidate|553|(49,)|var|uint8
const_554 = relay.const([2,-2,-10,-2,6,3,-9,7,-4,8,-4,-7,5,8,5,9,5,-6,2,3,7,9,-8,5,9,-8,-4,1,3,1,7,7,-5,8,3,5,-7,6,-1,9,-6,-3,2,-10,-1,2,2,1,4,-5,1,7,-4,-7,-5,5,3,-10,1,-9,10,4,3,-3,5,4,-8,5,-7,7,9,1,-1,7,10,8,-9,6,-8,-9,-7,-8,-7,6,-7,3,-8,-10,-7,5,-4,1,-5,-7,6,-3,-10,-9,-8,-2,10,3,-2,-9,1,-8,6,8,6,2,1,4,-8,2,-8,1,2,7,6,-2,-9,1,10,5,-7,6,2,-1,5,-6,-10,2,-10,3,-3,2,7,-7,3,1,-7,-7,-10,-1,4,5,-4,-3,6,-10,-4,10,6,-6,-9,10,-5,7,7,-10,7,6,-4,8,6,-8,-2,2,8,-8,7,3,5,9,1,-5,-7,-10,10,1,-5,8,7,-5,2,-2,-1,2,8,-10,-5,5,-7,-2,-2,5,-2,-5,7,-6,3,-7,-4,2,2,-2,8,8,-2,-2,-3,-2,-2,-4,6,7,7,-6,2,-6,4,3,5,7,4,-4,8,2,2,10,-5,9,-10,-4,5,-9,7,-10,5,-2,1,-6,1,-3,-3,2,2,9,-8,-6,-4,-1,9,3,-3,5,-10,-2,-8,-3,8,8,-9,-3,3,-10,-4,-9,-6,-9,9,10,-4,8,9,-4,9,8,2,-4,1,-1,7,-10,-4,10,4,1,-4,1,-4,-10,-8,2,-10,-4,-10,-5,-6,-8,7,-4,-9,8,-7,5,-5,-4,-3,-4,3,-3,-4,-8,5,-5,-5,3,5,7,2,1,9,2,-4,-7,6,-9,-8,3,4,-6,-6,-3,2,-10,2,-4,1,-4,-4,-3,9,-1,9,-2,3,-4,-7,-9,-8,3,-4,-2,-3,-1,7,7,6,-5,-3,9,-3,4,-1,6,4,1,6,-5,10,-6,-10,3,1,10,-2,-10,2,-4,-9,-2,2,2,-7,-7,-5,-3,7,10,1,4,6,8,-9,3,-5,-2,-10,-4,-3,3,-7,-4,2,4,1,7,-1,-9,7,10,-7,-8,-9,4,-5,-4,5,9,-10,-3,1,2,4,4,-1,5,9,7,-8,-1,-9,-7,3,-4,-9,3,6,3,7,3,-4,-8,-4,10,-7,4,8,-2,7,1,4,5,-1,4,-6,8,7,-2,4,-10,-4,6,-8,-10,-8,-3,7,4,8,10,5,5,-7,-3,-8,-1,-9,-4,-9,4,-7,10,6,-6,4,-10,3,2,-2,-10,-10,-5,-9,-10,3,-4,2,4,1,-6,-5,5,2,9,-4,6,-9,-5,-8,-7,-10,1,6,-1,7,-4,-4,10,8,-1,10,6,-6,8,-7,-9,7,-8,9,-1,-7,-9,-4,8,-10,-4,-4,-10,3,-8,-4,-7,9,3,7,2,-9,-6,2,6,-10,-2,3,7,-9,-8,4,-3,3,8,-2,-3,-3,-5,-10,7,-3,6,-1,6,10,5,9,6,8,1,6,2,-2,4,-9,-4,10,-10,5,-2,-10,-10,7,-3,1,4,8,7,-2,-2,4,3,5,6,9,-4,5,10,-1,-8,4,10,-9,-2,9,-1,-9,8,2,-4,2,-2,2,-2,2,5,-3,8,1,-2,-1,3,9,-3,-5,-2,5,-5,-3], dtype = "uint64")#candidate|554|(637,)|const|uint64
call_552 = func_229_call(relay.reshape(var_553.astype('uint8'), [7, 7, 1]), relay.reshape(const_554.astype('uint64'), [7, 7, 13]), )
call_555 = func_229_call(relay.reshape(var_553.astype('uint8'), [7, 7, 1]), relay.reshape(const_554.astype('uint64'), [7, 7, 13]), )
bop_556 = relay.logical_and(uop_524.astype('bool'), relay.reshape(uop_542.astype('bool'), relay.shape_of(uop_524))) # shape=(12,)
var_559 = relay.var("var_559", dtype = "bool", shape = (12,))#candidate|559|(12,)|var|bool
bop_560 = relay.bitwise_xor(bop_556.astype('int16'), relay.reshape(var_559.astype('int16'), relay.shape_of(bop_556))) # shape=(12,)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
call_574 = func_229_call(relay.reshape(var_553.astype('uint8'), [7, 7, 1]), relay.reshape(const_554.astype('uint64'), [7, 7, 13]), )
call_575 = func_229_call(relay.reshape(var_553.astype('uint8'), [7, 7, 1]), relay.reshape(const_554.astype('uint64'), [7, 7, 13]), )
const_583 = relay.const([7.960561,0.682475,-4.392774,-2.424052,8.987638,-7.423081,9.058587,3.415022,9.345433,-6.611664,-8.476890,-3.839003], dtype = "float64")#candidate|583|(12,)|const|float64
bop_584 = relay.subtract(uop_524.astype('float32'), relay.reshape(const_583.astype('float32'), relay.shape_of(uop_524))) # shape=(12,)
func_318_call = mod.get_global_var('func_318')
func_323_call = mutated_mod.get_global_var('func_323')
const_590 = relay.const(-8, dtype = "int32")#candidate|590|()|const|int32
var_591 = relay.var("var_591", dtype = "float32", shape = (30,))#candidate|591|(30,)|var|float32
call_589 = relay.TupleGetItem(func_318_call(relay.reshape(const_590.astype('int32'), []), relay.reshape(var_529.astype('float64'), [135,]), relay.reshape(var_591.astype('float32'), [5, 6]), relay.reshape(call_574.astype('uint64'), [637,]), ), 7)
call_592 = relay.TupleGetItem(func_323_call(relay.reshape(const_590.astype('int32'), []), relay.reshape(var_529.astype('float64'), [135,]), relay.reshape(var_591.astype('float32'), [5, 6]), relay.reshape(call_574.astype('uint64'), [637,]), ), 7)
output = relay.Tuple([call_528,var_529,call_552,var_553,const_554,bop_560,call_574,bop_584,call_589,const_590,var_591,])
output2 = relay.Tuple([call_530,var_529,call_555,var_553,const_554,bop_560,call_575,bop_584,call_592,const_590,var_591,])
func_601 = relay.Function([var_529,var_553,var_559,var_591,], output)
mod['func_601'] = func_601
mod = relay.transform.InferType()(mod)
var_602 = relay.var("var_602", dtype = "float64", shape = (135,))#candidate|602|(135,)|var|float64
var_603 = relay.var("var_603", dtype = "uint8", shape = (49,))#candidate|603|(49,)|var|uint8
var_604 = relay.var("var_604", dtype = "bool", shape = (12,))#candidate|604|(12,)|var|bool
var_605 = relay.var("var_605", dtype = "float32", shape = (30,))#candidate|605|(30,)|var|float32
output = func_601(var_602,var_603,var_604,var_605,)
func_606 = relay.Function([var_602,var_603,var_604,var_605,], output)
mutated_mod['func_606'] = func_606
mutated_mod = relay.transform.InferType()(mutated_mod)
const_678 = relay.const([[8.545504,-7.982402]], dtype = "float32")#candidate|678|(1, 2)|const|float32
uop_679 = relay.erf(const_678.astype('float32')) # shape=(1, 2)
var_681 = relay.var("var_681", dtype = "float32", shape = (14, 2))#candidate|681|(14, 2)|var|float32
bop_682 = relay.minimum(uop_679.astype('uint8'), var_681.astype('uint8')) # shape=(14, 2)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
const_687 = relay.const([-5,3,-6,10,10,3,3,-2,7,-1,-2,1,4,-2,-3,5,3,6,6,-5,8,-6,3,5,-3,10,-1,9,-6,10,-9,-7,-3,-6,-1,-5,-10,-9,1,2,2,1,-2,1,-9,-7,-5,3,-9], dtype = "uint8")#candidate|687|(49,)|const|uint8
const_688 = relay.const([9,-8,-3,1,-1,-7,-2,8,-7,1,-9,-8,-3,2,-9,2,9,8,-5,3,1,-2,-8,-8,9,-3,-10,-3,-2,-4,4,-3,-4,-4,-2,-8,8,-8,10,1,9,-10,-5,-9,9,-7,9,-5,-1,-3,-5,5,-3,-10,-3,4,3,9,8,4,-3,-10,3,5,-6,6,-3,5,-6,-7,-5,10,-7,-5,-7,2,-10,-2,2,6,10,-9,4,7,10,10,-6,-1,-9,8,5,-8,4,9,-7,-2,-8,-10,-7,-4,-10,10,7,9,-2,-4,6,-4,-9,-6,8,5,-1,8,-9,-6,-9,7,2,3,-2,10,6,10,-9,-6,-6,-4,-6,-6,-7,6,-9,-3,8,4,1,-9,8,-4,8,5,-1,-9,-1,1,-7,5,9,-6,4,-8,4,1,1,-10,-7,10,-3,-2,6,-6,2,9,-9,-10,-7,-7,-8,-4,8,-7,-8,-1,-6,7,8,-4,-5,4,2,-7,2,1,-6,10,-8,7,4,6,2,-1,-5,5,-2,3,10,-2,-8,4,-5,-6,-1,-8,2,4,-5,8,-4,1,-9,7,-8,4,1,8,2,8,8,10,-9,-7,3,-6,8,10,-6,9,-7,10,10,-3,-3,-10,8,2,-4,2,4,-4,-7,-2,-3,-7,-5,-8,4,-9,-6,-9,-9,-10,9,10,-1,-8,9,1,10,-4,-4,9,-6,6,-7,4,5,-1,-7,-10,5,7,7,-8,6,1,4,2,10,2,1,7,4,-2,9,9,9,10,4,4,-9,1,-7,-5,-1,8,5,-9,1,9,-3,-4,2,8,-1,7,-7,5,-1,5,-7,3,-8,10,4,-1,-8,9,8,-8,-10,-1,3,-5,8,8,2,-9,8,2,-1,-8,8,10,5,-7,3,3,5,4,-5,1,6,10,3,5,-6,2,-10,-2,-8,-1,-10,-9,-10,10,-7,-7,-3,6,-10,10,6,2,4,-3,6,5,8,-10,-9,8,-3,9,-1,-4,10,8,7,1,5,-6,-8,4,-7,6,7,1,8,-10,7,-1,9,2,-9,-5,-9,-3,2,-4,-6,-6,-4,-4,9,1,-9,4,2,-2,-10,-5,-10,7,10,2,4,8,6,-2,8,-4,8,5,-7,2,5,-8,10,8,9,-5,-1,4,-6,6,-4,3,10,3,-4,-5,-6,6,8,-5,8,-3,2,-6,-6,4,-3,-1,4,10,-2,-1,-8,6,2,7,3,9,-9,-6,7,4,-7,-5,-2,-1,2,3,-10,9,10,2,-6,-9,-5,-10,4,-6,-1,-4,-8,5,8,1,7,-7,-6,10,-1,-6,1,3,-10,-8,8,3,-8,5,6,-1,-5,-2,9,-4,-9,9,6,7,8,6,9,3,8,1,-3,-9,-3,1,2,-4,-6,1,4,-8,7,-2,5,-2,-3,-9,5,9,10,10,7,-3,9,-5,-3,-7,10,7,-5,10,-9,-8,-8,2,7,2,-8,-7,8,-1,-6,3,10,-9,6,6,-8,-2,9,8,10,2,-2,6,-8,3,5,-9,-6,3,8,1,-2,-6,-9,-1,-2,-6,9,-4,-8,2,-3,-2,9,-2,3,-5,-6,-7,3,5,-10,-10,4,-7,-3,8,10,-4,3,6,-9,2,1,-10,9,2,6,-8,5,3,-6,3,-10,5,-5,9,3,7,3,10,5,-3,3,-1,9], dtype = "uint64")#candidate|688|(637,)|const|uint64
call_686 = func_229_call(relay.reshape(const_687.astype('uint8'), [7, 7, 1]), relay.reshape(const_688.astype('uint64'), [7, 7, 13]), )
call_689 = func_229_call(relay.reshape(const_687.astype('uint8'), [7, 7, 1]), relay.reshape(const_688.astype('uint64'), [7, 7, 13]), )
uop_693 = relay.asinh(uop_679.astype('float32')) # shape=(1, 2)
uop_696 = relay.atan(uop_693.astype('float64')) # shape=(1, 2)
bop_705 = relay.add(uop_693.astype('float32'), relay.reshape(uop_679.astype('float32'), relay.shape_of(uop_693))) # shape=(1, 2)
bop_709 = relay.left_shift(bop_682.astype('uint16'), uop_693.astype('uint16')) # shape=(14, 2)
var_716 = relay.var("var_716", dtype = "uint8", shape = (14, 2))#candidate|716|(14, 2)|var|uint8
bop_717 = relay.greater_equal(bop_682.astype('bool'), relay.reshape(var_716.astype('bool'), relay.shape_of(bop_682))) # shape=(14, 2)
bop_721 = relay.less_equal(uop_696.astype('bool'), bop_717.astype('bool')) # shape=(14, 2)
var_724 = relay.var("var_724", dtype = "float64", shape = (1, 2))#candidate|724|(1, 2)|var|float64
bop_725 = relay.floor_mod(uop_696.astype('float32'), relay.reshape(var_724.astype('float32'), relay.shape_of(uop_696))) # shape=(1, 2)
bop_732 = relay.divide(uop_679.astype('float64'), relay.reshape(bop_725.astype('float64'), relay.shape_of(uop_679))) # shape=(1, 2)
output = relay.Tuple([call_686,const_687,const_688,bop_705,bop_709,bop_721,bop_732,])
output2 = relay.Tuple([call_689,const_687,const_688,bop_705,bop_709,bop_721,bop_732,])
func_736 = relay.Function([var_681,var_716,var_724,], output)
mod['func_736'] = func_736
mod = relay.transform.InferType()(mod)
var_737 = relay.var("var_737", dtype = "float32", shape = (14, 2))#candidate|737|(14, 2)|var|float32
var_738 = relay.var("var_738", dtype = "uint8", shape = (14, 2))#candidate|738|(14, 2)|var|uint8
var_739 = relay.var("var_739", dtype = "float64", shape = (1, 2))#candidate|739|(1, 2)|var|float64
output = func_736(var_737,var_738,var_739,)
func_740 = relay.Function([var_737,var_738,var_739,], output)
mutated_mod['func_740'] = func_740
mutated_mod = relay.transform.InferType()(mutated_mod)
var_744 = relay.var("var_744", dtype = "float64", shape = (4, 12))#candidate|744|(4, 12)|var|float64
uop_745 = relay.atan(var_744.astype('float64')) # shape=(4, 12)
bop_751 = relay.less(uop_745.astype('bool'), relay.reshape(var_744.astype('bool'), relay.shape_of(uop_745))) # shape=(4, 12)
uop_756 = relay.rsqrt(bop_751.astype('float64')) # shape=(4, 12)
bop_761 = relay.add(bop_751.astype('float32'), relay.reshape(uop_756.astype('float32'), relay.shape_of(bop_751))) # shape=(4, 12)
func_229_call = mod.get_global_var('func_229')
func_233_call = mutated_mod.get_global_var('func_233')
var_768 = relay.var("var_768", dtype = "uint8", shape = (49,))#candidate|768|(49,)|var|uint8
const_769 = relay.const([7,-7,-6,-3,9,4,2,-10,8,5,1,7,-1,-1,-1,-9,4,7,10,-6,1,5,-9,-2,-5,7,2,-2,-1,3,-7,-4,-8,-2,-10,7,8,7,-7,6,-6,1,-9,3,6,-5,10,-8,2,-7,8,-6,-8,-8,1,-8,-9,7,10,1,7,10,7,6,-8,4,10,6,4,-2,8,-4,-9,-6,-2,-4,4,6,4,1,-8,-9,4,6,-9,-4,10,-5,-10,5,-9,1,-7,-3,3,-10,-8,9,4,1,-6,10,5,-5,6,9,3,9,10,-8,7,-4,-6,-8,-10,4,7,4,-9,1,10,-3,1,-2,5,4,7,6,2,-6,6,-2,5,-5,6,-4,-5,1,10,-6,3,-4,9,6,5,-6,3,3,4,4,5,10,4,7,5,6,5,-10,1,6,1,8,9,-2,1,-5,-2,8,6,-10,1,3,-6,-10,3,5,-7,-8,6,9,5,-6,1,-5,-1,10,-10,10,7,5,6,-2,2,-1,-9,8,4,-3,9,-2,2,4,6,-10,-6,3,-4,6,6,-6,8,8,-7,-9,-8,7,-4,2,7,-2,-6,6,1,-9,9,-1,9,-1,2,-3,10,10,2,-5,-5,-3,-2,-2,1,2,-7,9,4,-4,-1,-9,-6,-6,-3,9,-7,-1,-9,4,6,-10,-8,-1,-10,-1,2,3,-8,-8,9,1,8,-5,-6,-9,-4,7,9,-4,-5,-1,4,-6,-2,-3,2,4,-7,10,-7,-6,-7,3,-3,6,-4,6,10,10,6,2,-9,-1,7,-10,10,5,8,-7,9,-8,2,-10,-6,4,-4,-4,-3,9,-8,-4,2,-8,-1,3,10,-2,-7,-7,-1,-7,-7,9,-7,2,5,4,7,-10,-8,-8,-8,1,3,10,-8,10,-7,-7,-5,-10,5,10,5,-10,-4,-1,9,8,1,5,-5,9,3,-8,5,1,-9,1,10,9,2,-6,4,-2,-2,-6,-1,10,-1,4,-4,7,5,-1,1,-9,-10,1,-1,-10,-9,9,-3,6,5,1,7,7,4,-9,8,8,-4,-1,-4,-5,9,-5,-1,6,2,-7,3,-1,2,4,7,-7,10,6,-8,7,5,10,4,1,-3,-5,2,5,-7,1,8,-6,5,2,-1,3,-2,1,8,2,-10,-8,-4,-6,10,4,7,-8,-2,-8,-1,8,-3,-9,-10,-2,7,-5,-3,-1,-8,4,-7,-10,3,9,-8,-9,-5,-1,1,-7,3,9,-3,-2,-7,8,7,10,-5,-7,-7,3,-7,1,8,-8,10,-9,-10,-3,5,1,-9,-5,-2,-9,-1,3,9,-7,1,-7,3,-1,-7,1,6,-8,-3,-3,-7,5,7,5,-3,-2,-8,-3,-5,3,1,-5,-6,-3,-6,-5,1,1,10,-3,-10,7,-3,8,-7,9,-5,-6,-7,-1,7,8,-9,-5,6,-4,-2,4,-1,1,-5,7,-7,2,-2,-9,-2,7,5,-2,10,9,7,5,-8,-6,-1,5,-6,1,2,4,-6,-5,3,-6,-3,2,10,8,8,-3,8,4,-4,9,5,-2,-6,-4,4,9,4,-9,-3,-4,-2,-7,-6,-2,7,6,5,-5,-5,-10,-8,10,4,-8,-6,-5,1,7,-4,8,10,-5,2,-8,10,9,10,-10,7,-1,-2,1,10,-9,-7,-8,-10,2,3,-2,9], dtype = "uint64")#candidate|769|(637,)|const|uint64
call_767 = func_229_call(relay.reshape(var_768.astype('uint8'), [7, 7, 1]), relay.reshape(const_769.astype('uint64'), [7, 7, 13]), )
call_770 = func_229_call(relay.reshape(var_768.astype('uint8'), [7, 7, 1]), relay.reshape(const_769.astype('uint64'), [7, 7, 13]), )
output = relay.Tuple([bop_761,call_767,var_768,const_769,])
output2 = relay.Tuple([bop_761,call_770,var_768,const_769,])
func_777 = relay.Function([var_744,var_768,], output)
mod['func_777'] = func_777
mod = relay.transform.InferType()(mod)
var_778 = relay.var("var_778", dtype = "float64", shape = (4, 12))#candidate|778|(4, 12)|var|float64
var_779 = relay.var("var_779", dtype = "uint8", shape = (49,))#candidate|779|(49,)|var|uint8
output = func_777(var_778,var_779,)
func_780 = relay.Function([var_778,var_779,], output)
mutated_mod['func_780'] = func_780
mutated_mod = relay.transform.InferType()(mutated_mod)
var_810 = relay.var("var_810", dtype = "uint32", shape = (4, 7, 8))#candidate|810|(4, 7, 8)|var|uint32
const_811 = relay.const([[[6,4,-1,5,5,8,7,3],[5,5,-6,5,-9,2,9,-3],[-8,-1,7,-10,3,8,-7,-6],[-4,-8,1,8,8,4,-5,-2],[-4,-1,6,5,9,-3,-2,-1],[-10,-9,-4,-3,-6,-8,3,-6],[-3,-10,-8,2,6,-4,-4,7]],[[3,7,10,8,6,-6,7,6],[9,-1,-10,-4,10,-7,2,10],[-9,-3,-6,6,10,-1,-6,-7],[2,3,2,-2,-5,7,-4,-3],[4,3,-10,2,-5,-4,-1,-7],[10,-6,-7,-3,7,2,-8,6],[2,2,2,-9,-8,-5,4,3]],[[-2,-6,-7,-3,-10,-8,9,-10],[5,2,-7,-2,4,7,1,-3],[9,6,-8,-2,4,-7,-6,-6],[9,8,-9,-3,10,-2,-10,-6],[4,-4,-10,1,8,1,-2,-5],[-4,-9,3,-5,-1,-6,4,9],[5,-10,-2,1,10,-2,-8,5]],[[9,-5,5,-7,-1,-9,-6,-2],[-4,6,-5,5,-8,8,-5,-9],[7,9,8,-10,1,-1,-2,8],[7,-3,2,8,7,7,10,-8],[-2,1,-3,6,-4,10,1,-1],[3,-9,-4,-4,-4,4,-9,-3],[1,8,6,-9,2,-6,7,-1]]], dtype = "uint32")#candidate|811|(4, 7, 8)|const|uint32
bop_812 = relay.bitwise_xor(var_810.astype('uint32'), relay.reshape(const_811.astype('uint32'), relay.shape_of(var_810))) # shape=(4, 7, 8)
output = bop_812
output2 = bop_812
func_815 = relay.Function([var_810,], output)
mod['func_815'] = func_815
mod = relay.transform.InferType()(mod)
var_816 = relay.var("var_816", dtype = "uint32", shape = (4, 7, 8))#candidate|816|(4, 7, 8)|var|uint32
output = func_815(var_816)
func_817 = relay.Function([var_816], output)
mutated_mod['func_817'] = func_817
mutated_mod = relay.transform.InferType()(mutated_mod)
var_821 = relay.var("var_821", dtype = "int32", shape = (4,))#candidate|821|(4,)|var|int32
const_822 = relay.const([-6,10,-7,-9], dtype = "int32")#candidate|822|(4,)|const|int32
bop_823 = relay.not_equal(var_821.astype('bool'), relay.reshape(const_822.astype('bool'), relay.shape_of(var_821))) # shape=(4,)
func_93_call = mod.get_global_var('func_93')
func_97_call = mutated_mod.get_global_var('func_97')
const_831 = relay.const([3.843991,-7.354571,8.717332,8.264397,-8.829340,0.271177,0.298398,-5.232840,-9.799960,-1.013306,-0.496226,1.284220,0.316162,0.947440,4.001949,-6.777209,-1.963187,9.744260,7.550665,-8.077279,-2.084773,6.865292,2.123633,-0.196004,-3.667310,3.343645,4.197709,0.516778,-2.563719,4.131377], dtype = "float32")#candidate|831|(30,)|const|float32
call_830 = func_93_call(relay.reshape(const_831.astype('float32'), [10, 3]), relay.reshape(const_831.astype('float32'), [10, 3]), )
call_832 = func_93_call(relay.reshape(const_831.astype('float32'), [10, 3]), relay.reshape(const_831.astype('float32'), [10, 3]), )
output = relay.Tuple([bop_823,call_830,const_831,])
output2 = relay.Tuple([bop_823,call_832,const_831,])
func_839 = relay.Function([var_821,], output)
mod['func_839'] = func_839
mod = relay.transform.InferType()(mod)
mutated_mod['func_839'] = func_839
mutated_mod = relay.transform.InferType()(mutated_mod)
var_840 = relay.var("var_840", dtype = "int32", shape = (4,))#candidate|840|(4,)|var|int32
func_839_call = mutated_mod.get_global_var('func_839')
call_841 = func_839_call(var_840)
output = call_841
func_842 = relay.Function([var_840], output)
mutated_mod['func_842'] = func_842
mutated_mod = relay.transform.InferType()(mutated_mod)
var_847 = relay.var("var_847", dtype = "int16", shape = (13, 3, 12))#candidate|847|(13, 3, 12)|var|int16
var_848 = relay.var("var_848", dtype = "int16", shape = (13, 3, 12))#candidate|848|(13, 3, 12)|var|int16
bop_849 = relay.left_shift(var_847.astype('int16'), relay.reshape(var_848.astype('int16'), relay.shape_of(var_847))) # shape=(13, 3, 12)
var_855 = relay.var("var_855", dtype = "int16", shape = (13, 3, 12))#candidate|855|(13, 3, 12)|var|int16
bop_856 = relay.floor_divide(bop_849.astype('float64'), relay.reshape(var_855.astype('float64'), relay.shape_of(bop_849))) # shape=(13, 3, 12)
output = bop_856
output2 = bop_856
func_870 = relay.Function([var_847,var_848,var_855,], output)
mod['func_870'] = func_870
mod = relay.transform.InferType()(mod)
var_871 = relay.var("var_871", dtype = "int16", shape = (13, 3, 12))#candidate|871|(13, 3, 12)|var|int16
var_872 = relay.var("var_872", dtype = "int16", shape = (13, 3, 12))#candidate|872|(13, 3, 12)|var|int16
var_873 = relay.var("var_873", dtype = "int16", shape = (13, 3, 12))#candidate|873|(13, 3, 12)|var|int16
output = func_870(var_871,var_872,var_873,)
func_874 = relay.Function([var_871,var_872,var_873,], output)
mutated_mod['func_874'] = func_874
mutated_mod = relay.transform.InferType()(mutated_mod)
var_949 = relay.var("var_949", dtype = "float32", shape = (2, 12))#candidate|949|(2, 12)|var|float32
uop_950 = relay.rsqrt(var_949.astype('float32')) # shape=(2, 12)
bop_955 = relay.bitwise_or(uop_950.astype('uint16'), relay.reshape(var_949.astype('uint16'), relay.shape_of(uop_950))) # shape=(2, 12)
const_958 = relay.const([[8,-1,-8,-2,-1,5,7,8,1,7,-10,-3],[1,-1,-5,-6,6,-7,2,7,-4,-9,-8,2]], dtype = "uint16")#candidate|958|(2, 12)|const|uint16
bop_959 = relay.bitwise_and(bop_955.astype('uint64'), relay.reshape(const_958.astype('uint64'), relay.shape_of(bop_955))) # shape=(2, 12)
uop_964 = relay.log(bop_959.astype('float32')) # shape=(2, 12)
bop_969 = relay.bitwise_or(uop_964.astype('uint8'), relay.reshape(bop_955.astype('uint8'), relay.shape_of(uop_964))) # shape=(2, 12)
func_601_call = mod.get_global_var('func_601')
func_606_call = mutated_mod.get_global_var('func_606')
const_979 = relay.const([[5.372361,-8.823897,-5.878173,-2.125459,-5.927796,-0.836807,-1.509913,-2.479908,-1.059014],[-3.924106,-5.060836,-5.382243,-9.135002,0.631353,1.787135,-6.311476,9.962535,2.740946],[8.894602,-4.056852,6.473320,5.345862,6.079245,3.948740,2.248019,-6.387491,0.280595],[2.172555,-4.745799,-1.923989,-8.010052,0.650383,7.557713,0.896540,-5.492740,-1.340330],[-4.191604,-8.371231,-5.814781,4.107826,5.338950,2.596793,3.677551,-1.233708,-7.126949],[0.874070,-2.926118,-9.485283,7.651429,-3.599690,-2.674460,7.701738,7.627353,-6.656506],[-4.642035,9.651998,3.374485,-2.786379,0.179468,-4.484638,-2.780671,-5.081480,-8.070410],[-0.646184,3.380307,-5.596554,-1.702847,-3.718324,3.217287,4.874932,6.978881,7.457070],[0.401760,1.797981,5.860521,-2.863395,2.597174,-5.409090,-1.301763,0.429874,8.831672],[8.154482,-6.721612,3.926845,-0.729899,1.977906,6.440188,-8.668471,-8.540634,-9.097296],[7.780429,4.616978,0.415531,4.923132,3.515939,1.999784,-3.096316,-6.745049,6.943547],[-7.629280,-0.919147,1.331773,-1.575800,-1.122532,0.669854,-7.930328,-2.662267,9.901853],[8.271843,5.006459,9.727196,-0.655983,7.648689,-6.980875,-4.690953,-5.433177,-8.090598],[1.220441,2.280357,-0.523597,-1.320498,1.687403,-2.375888,-3.145435,4.852662,-8.666132],[2.795712,-8.547874,3.280384,3.079016,2.954799,-4.991466,2.968123,1.461855,-2.872132]], dtype = "float64")#candidate|979|(15, 9)|const|float64
const_980 = relay.const([[5,3,2,3,-4,-3,-7,2,2,-10,4,10,-6,7,-4,-7,10,10,-1,-10,-10,-9,6,8,-3,-8,6,10,-6,2,-2,8,9,4,5,-7,8,1,9,9,-8,3,-8,7,-10,3,-8,-5,-3]], dtype = "uint8")#candidate|980|(1, 49)|const|uint8
var_981 = relay.var("var_981", dtype = "bool", shape = (12,))#candidate|981|(12,)|var|bool
var_982 = relay.var("var_982", dtype = "float32", shape = (5, 6))#candidate|982|(5, 6)|var|float32
call_978 = relay.TupleGetItem(func_601_call(relay.reshape(const_979.astype('float64'), [135,]), relay.reshape(const_980.astype('uint8'), [49,]), relay.reshape(var_981.astype('bool'), [12,]), relay.reshape(var_982.astype('float32'), [30,]), ), 0)
call_983 = relay.TupleGetItem(func_606_call(relay.reshape(const_979.astype('float64'), [135,]), relay.reshape(const_980.astype('uint8'), [49,]), relay.reshape(var_981.astype('bool'), [12,]), relay.reshape(var_982.astype('float32'), [30,]), ), 0)
bop_989 = relay.add(uop_950.astype('uint64'), relay.reshape(bop_955.astype('uint64'), relay.shape_of(uop_950))) # shape=(2, 12)
bop_993 = relay.less(const_958.astype('bool'), relay.reshape(uop_950.astype('bool'), relay.shape_of(const_958))) # shape=(2, 12)
func_601_call = mod.get_global_var('func_601')
func_606_call = mutated_mod.get_global_var('func_606')
call_998 = relay.TupleGetItem(func_601_call(relay.reshape(call_978.astype('float64'), [135,]), relay.reshape(const_980.astype('uint8'), [49,]), relay.reshape(var_981.astype('bool'), [12,]), relay.reshape(var_982.astype('float32'), [30,]), ), 2)
call_999 = relay.TupleGetItem(func_606_call(relay.reshape(call_978.astype('float64'), [135,]), relay.reshape(const_980.astype('uint8'), [49,]), relay.reshape(var_981.astype('bool'), [12,]), relay.reshape(var_982.astype('float32'), [30,]), ), 2)
uop_1008 = relay.sigmoid(bop_969.astype('float64')) # shape=(2, 12)
const_1021 = relay.const([[0.878067,-8.496424,6.872937,4.642767,-0.886268,-1.845805,3.617513,-1.723647,-7.215318,2.984144,-7.621040,-2.971445],[5.860654,-6.186379,-4.423771,0.516051,8.574123,-8.784101,2.621709,-0.913110,9.847522,-7.620849,1.292460,1.406681]], dtype = "float64")#candidate|1021|(2, 12)|const|float64
bop_1022 = relay.multiply(uop_1008.astype('uint16'), relay.reshape(const_1021.astype('uint16'), relay.shape_of(uop_1008))) # shape=(2, 12)
output = relay.Tuple([call_978,const_979,const_980,var_981,var_982,bop_989,bop_993,call_998,bop_1022,])
output2 = relay.Tuple([call_983,const_979,const_980,var_981,var_982,bop_989,bop_993,call_999,bop_1022,])
func_1027 = relay.Function([var_949,var_981,var_982,], output)
mod['func_1027'] = func_1027
mod = relay.transform.InferType()(mod)
mutated_mod['func_1027'] = func_1027
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1027_call = mutated_mod.get_global_var('func_1027')
var_1029 = relay.var("var_1029", dtype = "float32", shape = (2, 12))#candidate|1029|(2, 12)|var|float32
var_1030 = relay.var("var_1030", dtype = "bool", shape = (12,))#candidate|1030|(12,)|var|bool
var_1031 = relay.var("var_1031", dtype = "float32", shape = (5, 6))#candidate|1031|(5, 6)|var|float32
call_1028 = func_1027_call(var_1029,var_1030,var_1031,)
output = call_1028
func_1032 = relay.Function([var_1029,var_1030,var_1031,], output)
mutated_mod['func_1032'] = func_1032
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1063 = relay.var("var_1063", dtype = "uint64", shape = ())#candidate|1063|()|var|uint64
var_1064 = relay.var("var_1064", dtype = "uint64", shape = (10, 2, 12))#candidate|1064|(10, 2, 12)|var|uint64
bop_1065 = relay.right_shift(var_1063.astype('uint64'), var_1064.astype('uint64')) # shape=(10, 2, 12)
var_1076 = relay.var("var_1076", dtype = "uint64", shape = (10, 2, 12))#candidate|1076|(10, 2, 12)|var|uint64
bop_1077 = relay.logical_xor(bop_1065.astype('uint64'), relay.reshape(var_1076.astype('uint64'), relay.shape_of(bop_1065))) # shape=(10, 2, 12)
var_1084 = relay.var("var_1084", dtype = "uint64", shape = (10, 2, 12))#candidate|1084|(10, 2, 12)|var|uint64
bop_1085 = relay.bitwise_or(var_1076.astype('uint32'), relay.reshape(var_1084.astype('uint32'), relay.shape_of(var_1076))) # shape=(10, 2, 12)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
const_1092 = relay.const([-7.899522,7.397577,8.160745,-9.812652,2.318448,-4.641060,6.830492,-9.424711,-6.417851,0.327713,-1.040694,-6.972321,7.752042,3.112914,5.408512,4.396977,-9.408377,6.684853,-9.479511,6.572347,-0.234032,7.539514,3.899887,0.581738,5.490913,2.194903,-7.383048,0.584502,-3.396345,-7.032887,-4.936259,4.776182,7.466375,-4.949573,-4.680327,4.682567,2.815989,9.550487,-5.644830,-3.663116,6.109068,4.732431,9.661191,0.439239,0.343511,7.920529,-6.475460,4.586652,2.474093,7.248907,7.763057,-3.376339,5.016419,5.271543,-7.196048,-7.302368,-1.596428,1.565990,-1.238240,5.494378,7.266013,7.911839,0.030824,4.574982,-9.347415,-0.050260,4.299005,-4.264182,-4.029654,-9.893642,7.326503,-5.446603,-6.379612,2.488609,4.074070,4.615714,8.719692,-9.551311,4.678923,-8.342384,-8.344513,7.247827,9.498989,-9.200328,-9.478954,8.590431,5.126728,3.127862,4.836722,-0.439523,1.790907,5.519375,-5.351904,8.679651,6.033556,4.342795,-5.047780,6.465275,-5.342341,-5.294551,-9.935842,-1.506976,5.377261,-3.501711,-5.074296,6.172195,-1.981918,0.590618,-9.851762,-1.125178,-1.710412,2.626249,-1.408505,-7.824867,1.235430,1.885015,-1.719471,6.550373,8.092952,3.284664,3.155250,8.158802,0.931861,-8.261854,2.583824,-4.345958,0.951613,4.141779,-5.196523,-2.269858,-9.587627,-8.518059,-2.504907,-1.885244,7.743095], dtype = "float64")#candidate|1092|(135,)|const|float64
call_1091 = relay.TupleGetItem(func_80_call(relay.reshape(const_1092.astype('float64'), [15, 9]), relay.reshape(const_1092.astype('float64'), [15, 9]), ), 0)
call_1093 = relay.TupleGetItem(func_84_call(relay.reshape(const_1092.astype('float64'), [15, 9]), relay.reshape(const_1092.astype('float64'), [15, 9]), ), 0)
uop_1100 = relay.log(bop_1085.astype('float32')) # shape=(10, 2, 12)
output = relay.Tuple([bop_1077,call_1091,const_1092,uop_1100,])
output2 = relay.Tuple([bop_1077,call_1093,const_1092,uop_1100,])
func_1103 = relay.Function([var_1063,var_1064,var_1076,var_1084,], output)
mod['func_1103'] = func_1103
mod = relay.transform.InferType()(mod)
mutated_mod['func_1103'] = func_1103
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1103_call = mutated_mod.get_global_var('func_1103')
var_1105 = relay.var("var_1105", dtype = "uint64", shape = ())#candidate|1105|()|var|uint64
var_1106 = relay.var("var_1106", dtype = "uint64", shape = (10, 2, 12))#candidate|1106|(10, 2, 12)|var|uint64
var_1107 = relay.var("var_1107", dtype = "uint64", shape = (10, 2, 12))#candidate|1107|(10, 2, 12)|var|uint64
var_1108 = relay.var("var_1108", dtype = "uint64", shape = (10, 2, 12))#candidate|1108|(10, 2, 12)|var|uint64
call_1104 = func_1103_call(var_1105,var_1106,var_1107,var_1108,)
output = call_1104
func_1109 = relay.Function([var_1105,var_1106,var_1107,var_1108,], output)
mutated_mod['func_1109'] = func_1109
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1127 = relay.var("var_1127", dtype = "float32", shape = (11, 1, 7))#candidate|1127|(11, 1, 7)|var|float32
uop_1128 = relay.acosh(var_1127.astype('float32')) # shape=(11, 1, 7)
uop_1132 = relay.log10(uop_1128.astype('float64')) # shape=(11, 1, 7)
func_601_call = mod.get_global_var('func_601')
func_606_call = mutated_mod.get_global_var('func_606')
var_1136 = relay.var("var_1136", dtype = "float64", shape = (3, 45))#candidate|1136|(3, 45)|var|float64
const_1137 = relay.const([4,-7,6,-5,6,1,-8,1,5,6,-6,5,4,8,-10,10,10,-4,-9,5,4,-5,7,6,5,2,-10,1,-4,-6,2,10,-3,-10,3,10,-1,10,2,10,5,-1,8,10,10,8,-2,-4,5], dtype = "uint8")#candidate|1137|(49,)|const|uint8
const_1138 = relay.const([False,True,False,True,False,True,True,False,True,False,False,True], dtype = "bool")#candidate|1138|(12,)|const|bool
const_1139 = relay.const([0.081426,-8.172547,2.212718,4.734290,4.448322,-3.392869,8.669459,-2.088766,2.956981,-8.719071,1.374406,-8.648091,-3.212502,8.688515,-8.280157,-4.565979,-7.358727,1.943762,-6.800327,-9.506987,-6.971492,-0.857096,-4.256560,7.303722,-2.202036,2.017838,-4.014226,-0.334149,8.605430,-5.151113], dtype = "float32")#candidate|1139|(30,)|const|float32
call_1135 = relay.TupleGetItem(func_601_call(relay.reshape(var_1136.astype('float64'), [135,]), relay.reshape(const_1137.astype('uint8'), [49,]), relay.reshape(const_1138.astype('bool'), [12,]), relay.reshape(const_1139.astype('float32'), [30,]), ), 2)
call_1140 = relay.TupleGetItem(func_606_call(relay.reshape(var_1136.astype('float64'), [135,]), relay.reshape(const_1137.astype('uint8'), [49,]), relay.reshape(const_1138.astype('bool'), [12,]), relay.reshape(const_1139.astype('float32'), [30,]), ), 2)
bop_1145 = relay.floor_divide(uop_1128.astype('float64'), relay.reshape(uop_1132.astype('float64'), relay.shape_of(uop_1128))) # shape=(11, 1, 7)
bop_1149 = relay.greater(bop_1145.astype('bool'), relay.reshape(uop_1132.astype('bool'), relay.shape_of(bop_1145))) # shape=(11, 1, 7)
const_1152 = relay.const([[[-3.337006,2.200301,-3.377591,6.567858,9.425339,-1.432861,-1.793708],[0.271781,7.981593,8.333025,-3.854990,-6.306644,7.595264,-0.985073],[-1.459968,-5.813555,-0.348104,-4.972554,2.521398,-2.339178,-5.380413],[-0.662874,-2.818598,-7.927516,-3.238015,-5.775367,-9.425356,8.880658],[5.832069,0.648215,-2.889541,-4.142847,-3.644969,3.127125,9.948653],[-4.772833,-1.239886,-6.418244,-3.216168,-9.128747,-7.776738,2.488034],[3.256474,-4.681504,8.436589,-3.967793,1.593178,-8.438351,3.623382],[5.109410,8.913156,9.245747,8.748922,4.556835,-4.058479,-9.686534],[-1.411604,7.966958,7.642412,-0.928449,8.346983,-7.480662,-3.324050],[-2.156183,-5.104210,-5.793228,-8.074277,2.472223,2.767095,1.360230],[-0.142686,-9.350593,-4.560001,-7.596748,8.492170,-1.295625,-9.523588],[7.420983,9.308438,-5.487139,-5.993272,-7.892375,-1.270100,-0.737322],[-8.658939,7.322765,1.086208,6.264415,2.396682,0.671409,-7.317552],[-9.756565,-4.787427,0.788042,4.279543,7.745522,-6.798495,-2.508152]],[[0.083359,-1.076359,0.946815,-8.861935,5.225770,-4.073659,-3.854432],[0.043665,6.499291,9.878348,-6.169233,5.899804,-6.186740,0.474844],[-9.778180,-6.536536,-0.443777,4.268052,-0.380821,-4.494501,4.698034],[7.664535,3.514591,6.223456,-1.005149,-5.702209,0.137046,-4.480623],[2.777079,2.864891,-0.662103,2.448557,-1.041094,-9.408938,9.960902],[3.796708,6.451024,-9.004298,7.923504,2.021535,-0.253178,-7.636832],[5.555425,2.968572,-8.450636,3.527696,-8.401466,4.083698,5.868226],[-8.386026,0.570201,-8.327850,-6.257906,2.498021,-9.537133,4.514830],[-8.222156,-1.295110,-4.227731,-0.567161,-4.790561,4.073860,6.593112],[-5.022509,-4.854169,9.231757,-7.646348,-8.024239,-0.410040,-8.799850],[-1.914244,0.630003,-4.960250,0.745941,0.215857,8.735547,1.531589],[6.654693,-0.435692,8.344921,-1.948282,-6.371286,-1.081158,1.454410],[-8.971609,5.708771,6.893498,7.370276,4.013816,-8.948445,3.075200],[-7.621209,0.807478,1.102933,0.818690,-3.345375,5.029598,-2.810698]],[[-1.132745,-2.346592,-8.765590,8.144124,-9.530125,1.689682,0.004976],[-7.898405,8.000122,3.281961,-9.676973,-6.732438,-0.231430,8.952792],[3.773758,2.644889,7.553517,-9.018940,-1.541284,4.012159,2.396387],[-5.781414,0.477415,0.973083,7.063231,6.177739,3.591502,-4.780985],[-9.052653,0.696219,8.944174,6.005071,-9.424778,-3.065320,-6.815198],[1.644820,-6.538855,-5.974349,-6.881437,6.716758,9.619465,-1.016835],[-2.056428,-2.936694,-8.556248,4.966958,-8.317790,5.161259,-8.355805],[2.384909,1.688088,-8.639360,-3.620189,2.126790,-2.165998,9.882627],[6.297455,-2.583241,-4.008342,5.372220,-0.683775,-3.707117,1.831369],[3.156192,-3.417160,-9.722007,5.184588,9.921828,-8.143791,3.228289],[5.874602,-1.201184,-6.791487,6.451958,2.891679,5.626142,0.500749],[-3.596251,3.291893,-3.539180,0.147406,-0.670830,-1.824089,-1.954943],[-9.070946,8.193326,-6.567519,8.847308,1.367993,5.296906,9.477525],[-2.818361,6.173317,4.087401,9.774828,-8.692524,6.160083,2.387333]],[[-4.677566,-8.046753,9.604681,1.331584,9.229822,-1.225946,1.737355],[4.796946,8.168491,3.820597,-6.585795,3.816024,-9.654803,-5.346958],[-1.859075,1.340927,-0.873940,-7.297422,-5.017647,9.912608,-3.460012],[-0.846281,-5.318184,-7.983302,6.984423,-7.315019,-3.904763,2.057535],[2.838345,-6.629768,-9.229403,-7.902437,6.677101,-3.093463,-2.013770],[9.918458,-3.998307,-6.253889,8.269212,-0.723392,-5.184453,1.201478],[-5.502551,-9.178502,-7.757675,-1.642963,-7.869293,2.989842,8.840513],[5.083516,-6.580016,-5.454445,-7.985032,9.524501,0.016766,1.905721],[-3.611207,-9.878637,2.918736,-8.416875,-7.042457,-2.260627,7.954563],[-3.918211,-5.357930,8.250288,6.483436,3.544415,-1.236767,8.106235],[-6.581323,-7.744543,7.608600,6.452667,8.083905,2.626348,-0.999481],[-1.393250,4.686484,-9.278685,5.837329,-8.568961,8.839171,-7.171040],[-7.016849,3.475731,-0.380188,2.388015,0.505760,5.313610,-1.328146],[-7.255036,-2.492000,5.151168,5.784509,1.926714,9.412370,3.135877]],[[4.591467,-4.016089,-3.051381,-0.453288,8.916348,-5.827516,4.842255],[9.867697,-4.059761,-7.198507,7.410378,3.108036,-7.782023,2.616713],[-9.455500,-6.839300,-4.727195,-5.959901,9.797490,-4.108199,-4.147818],[6.549257,-0.002999,-1.785272,-1.221861,9.458371,-2.225773,-9.767802],[-0.367192,-7.702869,7.868146,-3.058158,2.060038,-0.100420,-9.850879],[-9.492939,8.477620,-9.714447,-3.086627,2.450157,1.815142,-1.770405],[9.102321,-5.165003,-2.636848,-2.495906,-3.378675,-1.684753,7.550828],[-8.936963,0.684444,5.251170,-5.531489,-2.975681,-0.760939,0.384994],[-9.442246,3.176667,-7.893682,6.928029,-6.346721,-8.884552,3.609029],[-4.856410,5.499929,-3.699042,7.548538,-9.833233,7.236356,8.463309],[-3.861363,7.278682,-6.124505,-0.936014,5.872085,-6.183444,1.326548],[-2.612839,6.317871,-0.687777,6.231106,3.060182,-7.959011,-0.734830],[-2.351201,3.576267,-9.092820,2.824375,8.502293,9.353397,-0.707649],[-5.174584,7.319195,-8.297322,3.626463,5.640802,4.728349,-5.920814]],[[-1.546413,4.317954,-4.527957,9.715936,3.371290,5.313564,-3.586184],[2.024917,-0.449571,3.943283,-5.213193,-0.507902,-5.687747,-7.569958],[-3.360777,3.434765,9.840954,0.389434,0.755543,-7.567129,-6.105167],[5.359094,2.830132,8.199305,-8.023726,6.267156,-9.960601,-8.183739],[6.413828,-9.518837,8.372465,1.055351,-5.190020,-2.340336,3.316052],[-5.966283,2.098092,2.561116,8.178005,-0.849544,9.017878,-2.656773],[-4.217134,-4.552815,-4.568885,6.674078,-8.449226,-6.245123,-5.106117],[-4.819952,-7.924228,2.498045,-9.820796,-7.086121,4.988380,8.445957],[-6.740442,-6.609612,-9.481801,-3.838404,7.636897,4.908285,-4.821381],[-8.634280,-2.425829,4.558214,6.281122,-2.943081,1.471048,7.708266],[-5.910101,5.140420,4.904661,9.563803,-0.456945,-8.918858,1.968126],[8.242255,7.123670,-8.191683,-4.733753,-0.196780,-1.655815,7.598940],[8.090607,-1.759254,8.612283,-8.820992,4.435079,-6.592908,-5.067038],[6.064149,-6.880761,2.892662,-2.027932,-1.782304,1.924086,6.020433]],[[2.854592,5.328696,5.191519,-5.769984,7.158240,7.437217,-2.792587],[-8.842058,-7.227254,8.740874,-0.933203,8.881013,-4.707524,4.046011],[-6.240936,-4.530399,5.648384,9.930864,7.305711,3.675013,-9.661030],[-7.479016,1.579588,2.528333,-6.445688,-3.770337,-8.562903,-8.304017],[0.142151,-2.385131,-2.389442,-0.579396,0.733832,-8.252748,2.318788],[4.324763,1.817064,3.547515,4.887369,-9.806679,8.512013,-7.022219],[-5.936145,-4.969317,4.492770,-6.727763,5.976644,2.041473,-5.138961],[-9.983506,-1.262182,2.784320,7.385717,-3.131217,-9.069448,-4.316632],[4.749436,9.057860,6.122807,-9.273641,-8.874768,-3.238917,4.572167],[-4.919882,-4.098913,0.867543,7.680696,-1.504173,3.013721,3.108453],[9.562573,7.107705,-3.288373,3.269874,8.977078,-3.692656,-0.438100],[1.605249,5.731908,0.148853,-1.497418,-3.181624,7.121007,-1.868152],[7.725932,-1.981429,-2.623676,1.864907,-0.239750,-7.754688,-8.643315],[-0.137720,-2.215062,-6.092119,1.946222,2.142470,0.648992,8.477370]],[[-5.166020,8.239057,8.403651,-0.562318,-6.045025,0.927295,-7.746062],[-4.211747,-4.844986,5.641886,-0.017490,-0.008007,2.196194,-0.745757],[8.633325,-9.135324,5.877062,7.642409,8.566819,-5.328091,-9.242791],[9.107815,-0.975510,0.363801,-9.963110,9.730904,9.655895,-2.807500],[8.306800,3.182081,2.481025,-8.719498,1.001310,5.788975,9.325917],[-0.333796,8.395513,7.197988,3.194452,9.766412,2.592496,6.462629],[-0.029196,-0.547085,-3.942803,4.398831,7.003250,-3.806651,3.066192],[0.539221,-1.883381,-3.189770,7.275882,8.198487,8.279557,-6.402639],[0.324130,8.325970,-7.441439,2.024392,9.139695,-3.214376,-0.564062],[4.959202,-6.173046,-9.751373,3.781946,4.420051,-5.845809,6.114245],[9.357759,8.967058,7.529154,-5.502575,-4.199567,-3.608840,-7.327125],[-5.903591,-0.162168,6.730131,-5.514432,5.285498,-9.032853,2.418952],[4.284970,-2.158017,-6.848619,0.863094,-6.682047,7.627741,0.088624],[-7.293820,-6.047926,-2.489043,9.503904,-1.362774,5.695321,0.645068]],[[-8.831613,0.713164,1.124839,-6.421534,2.247886,-2.810163,1.512655],[-2.993328,2.491243,5.832724,8.188105,-7.737924,4.395956,-0.360650],[-2.575762,-0.691185,-0.216552,-2.595787,-0.410920,-0.678844,-3.135355],[4.944982,-9.494450,-6.103349,8.555912,-9.331624,0.201060,0.397696],[0.128878,-8.354375,8.745012,-7.982042,-6.170879,7.030660,7.519973],[-4.051053,-1.879495,7.081197,4.339520,5.040398,3.003763,7.851006],[-3.154936,5.166915,-3.886930,-7.749230,9.909035,-7.547428,8.941013],[-4.725620,1.925665,0.958247,-6.386261,-2.566523,-7.062491,-0.418271],[7.616999,2.088755,-5.846015,7.775108,-2.578150,-7.227739,-8.973620],[5.526234,9.046526,-6.662273,-6.831204,-2.056415,6.529793,-2.795463],[-9.920251,4.453287,-9.206792,-3.419060,-6.699977,6.809845,0.058204],[-5.485094,-8.692637,8.615694,2.361261,-1.010835,7.559499,-3.525180],[-9.567462,2.662760,-4.317921,-8.643945,4.324168,3.628410,0.271684],[-9.905726,3.887697,-1.824240,-8.422054,9.051607,-0.222899,7.679944]],[[-3.093832,-8.418929,-4.120513,-4.340364,-3.482204,5.240513,4.154337],[1.136325,2.987865,9.421466,-3.122856,2.436772,0.506644,-9.352735],[-2.597613,-7.559801,-9.143910,8.816917,-1.731277,3.683300,3.296935],[-1.960163,8.509144,4.869746,4.284986,4.719197,-4.068337,-3.289786],[1.918867,1.761821,-3.322404,9.786337,-0.454204,3.189467,-4.160769],[-0.582612,8.810249,-8.140223,-9.693117,7.030991,-1.614468,-7.594492],[7.606749,-4.198322,7.492214,7.857606,0.741709,-3.330738,0.552014],[-9.260420,-7.262522,7.544949,8.300268,8.130131,7.106465,1.308379],[0.285858,7.395199,-2.294734,-0.207372,1.458546,9.223110,-2.871005],[0.286488,6.146504,8.908028,-8.781139,3.963672,-9.130369,3.067072],[-6.687119,3.510447,-1.153973,-6.707430,-2.201500,2.743620,-8.655351],[5.130381,-9.815665,4.034803,-7.779747,-9.702623,3.102963,9.634293],[-7.491540,-9.125469,6.363933,-9.515702,-9.154478,2.826362,5.471047],[-8.173854,4.492237,-0.941995,-9.451481,-0.621569,-5.009913,4.411910]],[[-9.187102,8.626674,-1.220909,1.705174,1.654169,0.514806,2.289251],[8.214486,2.340310,-9.813322,6.406042,7.612267,-3.628415,-6.932935],[-9.819203,1.571715,-8.881051,0.188537,-5.049621,6.419559,-4.109499],[7.540084,7.237263,7.300034,-8.560719,8.476387,-2.749159,-2.526020],[-3.638445,-3.429513,-4.717674,-5.100651,-0.146522,8.479793,-2.267412],[0.165458,-7.419690,-8.716392,5.695792,-2.271997,-5.608173,3.202737],[-2.759211,1.357076,-5.517863,0.093464,2.578448,-4.117759,4.845680],[-0.509444,-9.844506,5.513245,-0.441641,-9.060829,1.471996,-5.758710],[-6.385763,-8.438535,8.424126,-6.346160,0.685553,-4.394076,8.953148],[9.516420,4.036387,-8.098391,4.920099,-3.458943,-5.812929,1.506575],[4.528528,8.485677,2.444515,1.066999,8.642850,-5.012308,3.269064],[-5.620087,3.116083,-6.209838,3.284766,-1.914125,-3.295171,1.107039],[3.699891,-0.258834,-2.338639,5.322482,2.111167,-0.568127,-7.363032],[-9.477277,0.432502,-6.446457,-8.989662,1.712170,1.178574,-3.211976]]], dtype = "float64")#candidate|1152|(11, 14, 7)|const|float64
bop_1153 = relay.bitwise_xor(uop_1132.astype('uint64'), const_1152.astype('uint64')) # shape=(11, 14, 7)
uop_1156 = relay.cosh(bop_1153.astype('float32')) # shape=(11, 14, 7)
var_1163 = relay.var("var_1163", dtype = "float64", shape = (11, 12, 7))#candidate|1163|(11, 12, 7)|var|float64
bop_1164 = relay.subtract(bop_1145.astype('int16'), var_1163.astype('int16')) # shape=(11, 12, 7)
func_1103_call = mod.get_global_var('func_1103')
func_1109_call = mutated_mod.get_global_var('func_1109')
const_1168 = relay.const(1, dtype = "uint64")#candidate|1168|()|const|uint64
var_1169 = relay.var("var_1169", dtype = "uint64", shape = (240,))#candidate|1169|(240,)|var|uint64
call_1167 = relay.TupleGetItem(func_1103_call(relay.reshape(const_1168.astype('uint64'), []), relay.reshape(var_1169.astype('uint64'), [10, 2, 12]), relay.reshape(var_1169.astype('uint64'), [10, 2, 12]), relay.reshape(var_1169.astype('uint64'), [10, 2, 12]), ), 2)
call_1170 = relay.TupleGetItem(func_1109_call(relay.reshape(const_1168.astype('uint64'), []), relay.reshape(var_1169.astype('uint64'), [10, 2, 12]), relay.reshape(var_1169.astype('uint64'), [10, 2, 12]), relay.reshape(var_1169.astype('uint64'), [10, 2, 12]), ), 2)
bop_1173 = relay.not_equal(uop_1156.astype('bool'), uop_1128.astype('bool')) # shape=(11, 14, 7)
bop_1176 = relay.less_equal(uop_1156.astype('bool'), bop_1149.astype('bool')) # shape=(11, 14, 7)
uop_1182 = relay.atan(bop_1153.astype('float32')) # shape=(11, 14, 7)
output = relay.Tuple([call_1135,var_1136,const_1137,const_1138,const_1139,bop_1164,call_1167,const_1168,var_1169,bop_1173,bop_1176,uop_1182,])
output2 = relay.Tuple([call_1140,var_1136,const_1137,const_1138,const_1139,bop_1164,call_1170,const_1168,var_1169,bop_1173,bop_1176,uop_1182,])
func_1185 = relay.Function([var_1127,var_1136,var_1163,var_1169,], output)
mod['func_1185'] = func_1185
mod = relay.transform.InferType()(mod)
mutated_mod['func_1185'] = func_1185
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1185_call = mutated_mod.get_global_var('func_1185')
var_1187 = relay.var("var_1187", dtype = "float32", shape = (11, 1, 7))#candidate|1187|(11, 1, 7)|var|float32
var_1188 = relay.var("var_1188", dtype = "float64", shape = (3, 45))#candidate|1188|(3, 45)|var|float64
var_1189 = relay.var("var_1189", dtype = "float64", shape = (11, 12, 7))#candidate|1189|(11, 12, 7)|var|float64
var_1190 = relay.var("var_1190", dtype = "uint64", shape = (240,))#candidate|1190|(240,)|var|uint64
call_1186 = func_1185_call(var_1187,var_1188,var_1189,var_1190,)
output = call_1186
func_1191 = relay.Function([var_1187,var_1188,var_1189,var_1190,], output)
mutated_mod['func_1191'] = func_1191
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1256 = relay.const(6, dtype = "uint16")#candidate|1256|()|const|uint16
var_1257 = relay.var("var_1257", dtype = "uint16", shape = (14, 4, 12))#candidate|1257|(14, 4, 12)|var|uint16
bop_1258 = relay.bitwise_and(const_1256.astype('uint16'), var_1257.astype('uint16')) # shape=(14, 4, 12)
func_93_call = mod.get_global_var('func_93')
func_97_call = mutated_mod.get_global_var('func_97')
var_1272 = relay.var("var_1272", dtype = "float32", shape = (30,))#candidate|1272|(30,)|var|float32
call_1271 = func_93_call(relay.reshape(var_1272.astype('float32'), [10, 3]), relay.reshape(var_1272.astype('float32'), [10, 3]), )
call_1273 = func_93_call(relay.reshape(var_1272.astype('float32'), [10, 3]), relay.reshape(var_1272.astype('float32'), [10, 3]), )
bop_1283 = relay.minimum(const_1256.astype('uint16'), bop_1258.astype('uint16')) # shape=(14, 4, 12)
func_93_call = mod.get_global_var('func_93')
func_97_call = mutated_mod.get_global_var('func_97')
call_1293 = func_93_call(relay.reshape(var_1272.astype('float32'), [10, 3]), relay.reshape(call_1271.astype('float32'), [10, 3]), )
call_1294 = func_93_call(relay.reshape(var_1272.astype('float32'), [10, 3]), relay.reshape(call_1271.astype('float32'), [10, 3]), )
output = relay.Tuple([call_1271,var_1272,bop_1283,call_1293,])
output2 = relay.Tuple([call_1273,var_1272,bop_1283,call_1294,])
func_1296 = relay.Function([var_1257,var_1272,], output)
mod['func_1296'] = func_1296
mod = relay.transform.InferType()(mod)
var_1297 = relay.var("var_1297", dtype = "uint16", shape = (14, 4, 12))#candidate|1297|(14, 4, 12)|var|uint16
var_1298 = relay.var("var_1298", dtype = "float32", shape = (30,))#candidate|1298|(30,)|var|float32
output = func_1296(var_1297,var_1298,)
func_1299 = relay.Function([var_1297,var_1298,], output)
mutated_mod['func_1299'] = func_1299
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1330 = relay.var("var_1330", dtype = "float32", shape = (6, 2, 1))#candidate|1330|(6, 2, 1)|var|float32
uop_1331 = relay.sinh(var_1330.astype('float32')) # shape=(6, 2, 1)
const_1334 = relay.const([[[-7.504470,-9.686023,1.857704,2.399469,-4.370487,-0.396079,-2.536412,0.254114,2.685937,-0.126619],[-1.295880,-9.828282,2.197148,-7.803763,4.401771,-2.424367,-9.127828,6.768467,0.938292,4.343930]],[[-5.069657,7.737858,7.691321,0.852398,1.469201,7.515484,-2.978153,-9.297033,-5.336478,-5.754231],[-2.727224,5.157463,0.283780,9.519471,8.258913,-6.401571,-5.723079,-1.892572,-4.534390,-5.090803]],[[7.308973,-3.918427,1.088498,1.247104,4.777158,8.555430,-9.284220,-9.122241,1.640951,-2.069308],[-7.172575,6.544736,0.062556,-5.407780,1.407791,2.269331,0.700327,-4.601375,-6.805172,-8.960627]],[[1.518659,-0.139210,7.609943,7.685670,-2.575401,7.095075,5.346256,5.460973,0.882191,-1.933394],[3.115034,7.092339,5.682025,6.289942,-2.414035,2.494355,-2.526238,1.929474,-3.432576,-7.410517]],[[8.833745,9.212718,-0.322744,7.296629,7.446063,-1.407924,3.300533,7.729198,-5.662335,0.281464],[-7.376160,-3.915440,-9.016782,-3.682941,7.274001,2.149041,7.775741,0.191204,1.138266,6.401146]],[[-1.233532,1.961176,0.075648,2.681332,-8.963247,-9.161772,0.719162,1.038319,2.712609,8.398232],[-6.527432,9.513244,7.242915,-7.999573,5.306259,3.236155,5.168703,4.569809,1.310385,-7.427543]]], dtype = "float32")#candidate|1334|(6, 2, 10)|const|float32
bop_1335 = relay.right_shift(var_1330.astype('int32'), const_1334.astype('int32')) # shape=(6, 2, 10)
func_318_call = mod.get_global_var('func_318')
func_323_call = mutated_mod.get_global_var('func_323')
const_1341 = relay.const(1, dtype = "int32")#candidate|1341|()|const|int32
var_1342 = relay.var("var_1342", dtype = "float64", shape = (1, 135))#candidate|1342|(1, 135)|var|float64
var_1343 = relay.var("var_1343", dtype = "float32", shape = (30,))#candidate|1343|(30,)|var|float32
var_1344 = relay.var("var_1344", dtype = "uint64", shape = (637, 1))#candidate|1344|(637, 1)|var|uint64
call_1340 = relay.TupleGetItem(func_318_call(relay.reshape(const_1341.astype('int32'), []), relay.reshape(var_1342.astype('float64'), [135,]), relay.reshape(var_1343.astype('float32'), [5, 6]), relay.reshape(var_1344.astype('uint64'), [637,]), ), 5)
call_1345 = relay.TupleGetItem(func_323_call(relay.reshape(const_1341.astype('int32'), []), relay.reshape(var_1342.astype('float64'), [135,]), relay.reshape(var_1343.astype('float32'), [5, 6]), relay.reshape(var_1344.astype('uint64'), [637,]), ), 5)
uop_1350 = relay.cos(var_1344.astype('float32')) # shape=(637, 1)
var_1354 = relay.var("var_1354", dtype = "float32", shape = (637, 15))#candidate|1354|(637, 15)|var|float32
bop_1355 = relay.maximum(uop_1350.astype('int64'), var_1354.astype('int64')) # shape=(637, 15)
uop_1359 = relay.sqrt(uop_1350.astype('float64')) # shape=(637, 1)
bop_1362 = relay.bitwise_and(uop_1359.astype('int32'), const_1341.astype('int32')) # shape=(637, 1)
bop_1370 = relay.power(bop_1355.astype('float32'), const_1341.astype('float32')) # shape=(637, 15)
bop_1379 = relay.greater_equal(uop_1359.astype('bool'), var_1342.astype('bool')) # shape=(637, 135)
output = relay.Tuple([uop_1331,bop_1335,call_1340,var_1343,bop_1362,bop_1370,bop_1379,])
output2 = relay.Tuple([uop_1331,bop_1335,call_1345,var_1343,bop_1362,bop_1370,bop_1379,])
func_1382 = relay.Function([var_1330,var_1342,var_1343,var_1344,var_1354,], output)
mod['func_1382'] = func_1382
mod = relay.transform.InferType()(mod)
mutated_mod['func_1382'] = func_1382
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1382_call = mutated_mod.get_global_var('func_1382')
var_1384 = relay.var("var_1384", dtype = "float32", shape = (6, 2, 1))#candidate|1384|(6, 2, 1)|var|float32
var_1385 = relay.var("var_1385", dtype = "float64", shape = (1, 135))#candidate|1385|(1, 135)|var|float64
var_1386 = relay.var("var_1386", dtype = "float32", shape = (30,))#candidate|1386|(30,)|var|float32
var_1387 = relay.var("var_1387", dtype = "uint64", shape = (637, 1))#candidate|1387|(637, 1)|var|uint64
var_1388 = relay.var("var_1388", dtype = "float32", shape = (637, 15))#candidate|1388|(637, 15)|var|float32
call_1383 = func_1382_call(var_1384,var_1385,var_1386,var_1387,var_1388,)
output = call_1383
func_1389 = relay.Function([var_1384,var_1385,var_1386,var_1387,var_1388,], output)
mutated_mod['func_1389'] = func_1389
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1451 = relay.var("var_1451", dtype = "uint16", shape = (14, 13, 10))#candidate|1451|(14, 13, 10)|var|uint16
var_1452 = relay.var("var_1452", dtype = "uint16", shape = (14, 13, 10))#candidate|1452|(14, 13, 10)|var|uint16
bop_1453 = relay.bitwise_xor(var_1451.astype('uint16'), relay.reshape(var_1452.astype('uint16'), relay.shape_of(var_1451))) # shape=(14, 13, 10)
output = bop_1453
output2 = bop_1453
func_1456 = relay.Function([var_1451,var_1452,], output)
mod['func_1456'] = func_1456
mod = relay.transform.InferType()(mod)
mutated_mod['func_1456'] = func_1456
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1456_call = mutated_mod.get_global_var('func_1456')
var_1458 = relay.var("var_1458", dtype = "uint16", shape = (14, 13, 10))#candidate|1458|(14, 13, 10)|var|uint16
var_1459 = relay.var("var_1459", dtype = "uint16", shape = (14, 13, 10))#candidate|1459|(14, 13, 10)|var|uint16
call_1457 = func_1456_call(var_1458,var_1459,)
output = call_1457
func_1460 = relay.Function([var_1458,var_1459,], output)
mutated_mod['func_1460'] = func_1460
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1502 = relay.var("var_1502", dtype = "float32", shape = (10, 6))#candidate|1502|(10, 6)|var|float32
var_1503 = relay.var("var_1503", dtype = "float32", shape = (10, 6))#candidate|1503|(10, 6)|var|float32
bop_1504 = relay.floor_mod(var_1502.astype('float32'), relay.reshape(var_1503.astype('float32'), relay.shape_of(var_1502))) # shape=(10, 6)
uop_1510 = relay.sigmoid(bop_1504.astype('float64')) # shape=(10, 6)
uop_1513 = relay.asin(uop_1510.astype('float64')) # shape=(10, 6)
bop_1521 = relay.logical_xor(uop_1510.astype('uint32'), relay.reshape(var_1503.astype('uint32'), relay.shape_of(uop_1510))) # shape=(10, 6)
bop_1533 = relay.greater_equal(bop_1521.astype('bool'), relay.reshape(uop_1510.astype('bool'), relay.shape_of(bop_1521))) # shape=(10, 6)
func_1456_call = mod.get_global_var('func_1456')
func_1460_call = mutated_mod.get_global_var('func_1460')
var_1546 = relay.var("var_1546", dtype = "uint16", shape = (1820,))#candidate|1546|(1820,)|var|uint16
call_1545 = func_1456_call(relay.reshape(var_1546.astype('uint16'), [14, 13, 10]), relay.reshape(var_1546.astype('uint16'), [14, 13, 10]), )
call_1547 = func_1456_call(relay.reshape(var_1546.astype('uint16'), [14, 13, 10]), relay.reshape(var_1546.astype('uint16'), [14, 13, 10]), )
var_1548 = relay.var("var_1548", dtype = "float64", shape = (10, 6))#candidate|1548|(10, 6)|var|float64
bop_1549 = relay.logical_and(uop_1513.astype('bool'), relay.reshape(var_1548.astype('bool'), relay.shape_of(uop_1513))) # shape=(10, 6)
bop_1552 = relay.subtract(bop_1521.astype('int8'), relay.reshape(bop_1504.astype('int8'), relay.shape_of(bop_1521))) # shape=(10, 6)
uop_1557 = relay.asinh(bop_1549.astype('float32')) # shape=(10, 6)
func_1296_call = mod.get_global_var('func_1296')
func_1299_call = mutated_mod.get_global_var('func_1299')
const_1560 = relay.const([-5,7,1,-7,-4,-9,3,-4,8,3,-9,-2,2,2,-5,7,7,-1,-7,-1,-1,9,-10,-4,-4,-10,2,1,4,-3,10,10,3,6,-2,-1,9,8,-5,-8,7,-4,-9,-2,-8,-3,4,-6,9,-7,2,3,-9,1,-1,-3,-6,-9,1,10,-8,-10,10,-1,9,-7,7,-4,-1,8,-5,-4,9,-8,7,-10,10,-9,-7,9,-10,8,-8,-2,7,-3,-5,3,-10,-4,6,-2,-5,9,9,7,6,-7,-6,-3,2,-3,-3,-9,-7,10,9,4,-2,6,-2,-3,8,1,8,7,10,-5,8,7,-10,-1,-8,-5,-7,-2,-3,5,4,5,10,1,4,4,-3,9,10,2,-2,-8,9,-8,-6,6,-1,-1,-5,-2,10,2,10,7,-9,8,8,-2,10,9,-5,6,-8,-8,9,3,-7,-1,6,7,-6,9,-2,1,3,8,5,-5,-10,5,2,-7,2,-9,5,-4,-2,-5,5,8,3,5,9,2,-2,9,6,-6,3,7,-1,8,-3,8,6,6,-2,-1,3,-2,-10,-1,-4,-5,4,4,-2,-9,-5,2,2,3,9,5,-7,6,-6,4,-8,9,-10,7,7,-6,2,-6,-6,-8,-2,8,10,9,-5,-9,3,-4,5,4,7,4,7,-4,8,-1,-7,-2,-5,6,-1,2,7,6,7,-1,9,4,5,1,3,6,1,8,6,-5,-1,2,10,7,7,2,5,-3,7,6,8,-10,9,-6,-4,-1,-9,-2,-10,-6,10,6,-7,8,5,-2,7,5,-1,-10,-3,1,-5,-10,-5,3,-8,1,-7,7,-6,-6,-2,8,3,10,8,-7,6,9,6,5,2,-7,-6,3,-5,4,10,-5,-9,-9,6,-1,3,-2,-3,-3,6,3,6,5,-5,-1,10,-7,8,-7,2,-7,-7,4,-9,-2,-6,-6,-1,4,-9,-10,-10,9,3,-4,-5,9,-1,1,-1,6,9,10,4,-3,-8,3,10,-5,9,7,2,3,3,5,8,1,-3,7,-6,10,3,-3,-7,-7,10,8,4,-7,-9,10,6,-5,2,-8,-4,4,4,-2,8,7,5,-9,-9,3,1,-5,5,-5,4,7,2,7,1,-6,9,6,-9,1,3,7,-9,10,4,3,9,-6,-3,8,2,-4,9,5,7,7,-3,-4,9,-6,-9,1,-9,2,1,8,8,-6,-2,-5,10,8,10,-4,6,-2,-9,7,-1,1,-9,-7,3,-3,4,-3,-4,6,3,-3,-1,5,9,1,-1,3,-3,8,-2,2,-9,-10,6,-6,-7,-9,3,2,6,-1,-6,-6,3,-7,-4,-10,-6,5,6,2,-10,4,6,3,-9,-8,9,6,-9,8,3,-8,8,-10,7,-2,2,-10,1,-2,-10,-7,-6,2,5,-1,-7,7,2,-2,9,6,9,2,1,-4,4,-1,-2,8,-10,8,-9,3,-2,2,10,-5,-6,-10,-10,-10,-4,7,-2,5,-5,9,8,5,8,1,5,-6,-2,-10,1,-5,8,2,8,-2,-2,8,3,-10,-1,7,-9,-1,-9,1,8,4,-9,-6,6,9,8,-8,1,10,-2,-1,9,-10,-2,6,-7,6,9,-3,1,-4,1,8,3,5,-4,-4,8,-1,5,10,10,-9,9,5,-6,-9,-3,9,6,1,10,-4,1,9,-1,-9,-5,1,2,6,-4,-5,1,-5,9,6,5,-2,-3,2,9,6,-2,2,2,-2,-10,-5,2,-7,-10,-4,4,1,6,3,9,-3], dtype = "uint16")#candidate|1560|(672,)|const|uint16
var_1561 = relay.var("var_1561", dtype = "float32", shape = (30,))#candidate|1561|(30,)|var|float32
call_1559 = relay.TupleGetItem(func_1296_call(relay.reshape(const_1560.astype('uint16'), [14, 4, 12]), relay.reshape(var_1561.astype('float32'), [30,]), ), 3)
call_1562 = relay.TupleGetItem(func_1299_call(relay.reshape(const_1560.astype('uint16'), [14, 4, 12]), relay.reshape(var_1561.astype('float32'), [30,]), ), 3)
output = relay.Tuple([bop_1533,call_1545,var_1546,bop_1552,uop_1557,call_1559,const_1560,var_1561,])
output2 = relay.Tuple([bop_1533,call_1547,var_1546,bop_1552,uop_1557,call_1562,const_1560,var_1561,])
func_1564 = relay.Function([var_1502,var_1503,var_1546,var_1548,var_1561,], output)
mod['func_1564'] = func_1564
mod = relay.transform.InferType()(mod)
var_1565 = relay.var("var_1565", dtype = "float32", shape = (10, 6))#candidate|1565|(10, 6)|var|float32
var_1566 = relay.var("var_1566", dtype = "float32", shape = (10, 6))#candidate|1566|(10, 6)|var|float32
var_1567 = relay.var("var_1567", dtype = "uint16", shape = (1820,))#candidate|1567|(1820,)|var|uint16
var_1568 = relay.var("var_1568", dtype = "float64", shape = (10, 6))#candidate|1568|(10, 6)|var|float64
var_1569 = relay.var("var_1569", dtype = "float32", shape = (30,))#candidate|1569|(30,)|var|float32
output = func_1564(var_1565,var_1566,var_1567,var_1568,var_1569,)
func_1570 = relay.Function([var_1565,var_1566,var_1567,var_1568,var_1569,], output)
mutated_mod['func_1570'] = func_1570
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1576 = relay.const(4, dtype = "int16")#candidate|1576|()|const|int16
const_1577 = relay.const([[-5,-9,-3,-8,-10,-9,-2,3,10],[6,-5,9,8,-3,6,-7,4,-8],[8,3,-8,2,7,-2,5,10,10],[3,5,-4,-1,-2,10,10,-7,-5],[7,5,-1,10,-6,10,-10,-5,-1],[7,-4,7,-8,-10,-1,-2,-2,-2],[-7,-3,-1,-10,-5,7,5,-8,-3],[-3,-1,10,9,2,-10,-6,-5,9],[-9,-10,-3,2,2,-7,2,-10,4],[-7,-1,-4,-3,9,7,-5,-8,-5]], dtype = "int16")#candidate|1577|(10, 9)|const|int16
bop_1578 = relay.equal(const_1576.astype('bool'), const_1577.astype('bool')) # shape=(10, 9)
uop_1583 = relay.acosh(bop_1578.astype('float64')) # shape=(10, 9)
const_1592 = relay.const([[-3.098010,-6.255737,-2.652694,-3.931865,-8.672310,7.135921,-0.654693,5.530139,-5.411334],[-3.552427,-2.528503,-0.780792,-4.925276,-0.822734,-3.412123,-8.416083,9.958912,4.392159],[2.298654,4.183563,6.714343,-5.035961,1.235097,4.076669,-5.343035,1.333843,0.809565],[6.549569,3.100937,-8.099467,6.549470,5.102735,-4.455121,0.331311,-3.986524,3.369600],[7.543532,-2.441708,9.919051,-5.947755,4.096112,1.284768,0.507669,-4.400417,-8.414311],[5.600216,5.277169,6.744859,2.484496,-4.482292,9.579857,3.794084,4.357246,0.230240],[2.705949,-9.506820,-5.184085,2.334136,-1.681487,-0.004523,6.706060,2.602392,-7.686720],[-7.805822,1.217952,-0.866913,-6.645738,9.478048,5.034725,-1.295133,3.021684,7.482788],[-2.744754,2.898827,-8.066747,-2.231858,8.807904,-1.282940,-1.262629,6.459159,1.536152],[-8.094990,6.892370,8.174717,-6.984043,-4.943854,-5.672817,6.383711,-8.533774,-3.842769]], dtype = "float64")#candidate|1592|(10, 9)|const|float64
bop_1593 = relay.less_equal(uop_1583.astype('bool'), relay.reshape(const_1592.astype('bool'), relay.shape_of(uop_1583))) # shape=(10, 9)
output = relay.Tuple([bop_1593,])
output2 = relay.Tuple([bop_1593,])
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

'''61: TVMFuncCall
60: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
59: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
58: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
57: tvm::IRModule::FromExpr(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&)
56: tvm::IRModule::FromExprInContext(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&, std::unordered_set<tvm::runtime::String, std::hash<tvm::runtime::String>, std::equal_to<tvm::runtime::String>, std::allocator<tvm::runtime::String> >)
55: tvm::IRModuleNode::Add(tvm::GlobalVar const&, tvm::BaseFunc const&, bool)
54: tvm::WarnIfMalformed(tvm::IRModule const&, tvm::relay::Function)
53: tvm::relay::FreeTypeVars(tvm::RelayExpr const&, tvm::IRModule const&)
52: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
51: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
50: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
49: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
48: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
47: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
46: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
45: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
44: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
43: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
42: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
41: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
40: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
39: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
38: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
37: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
36: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
35: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
34: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
33: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
32: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
31: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
30: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
29: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
28: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
27: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
26: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
25: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
24: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
23: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
22: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
21: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
20: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
19: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
18: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
17: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
16: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
15: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
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