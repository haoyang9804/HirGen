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
var_28 = relay.var("var_28", dtype = "float64", shape = ())#candidate|28|()|var|float64
const_29 = relay.const([[[-4.204685],[5.336266],[-9.514345],[1.366430],[3.375826],[-2.072101],[-0.619420],[-5.204091],[7.354086],[3.636944],[2.089494]],[[9.633225],[7.673488],[-0.931437],[-2.813855],[5.655785],[5.619806],[4.439581],[-0.219899],[3.048941],[-0.569320],[0.027832]],[[9.463243],[-7.728181],[-6.492604],[-7.123426],[-2.107915],[9.879992],[-5.100980],[-6.505912],[-7.388059],[6.441135],[-1.846363]],[[9.784632],[8.033990],[-6.714942],[-5.466752],[2.136255],[6.212437],[-2.304559],[2.718655],[-9.567961],[-1.749028],[8.092475]],[[3.437527],[-8.200418],[4.315087],[-7.796920],[-2.864734],[9.453531],[-0.325278],[-3.607498],[3.509331],[-0.866006],[2.777993]],[[-2.537731],[2.068860],[-0.154326],[6.877533],[-8.985835],[6.272986],[-5.612542],[-2.062500],[-6.579340],[-2.047938],[-6.492818]],[[8.093662],[2.903393],[-9.262930],[-0.239251],[-2.369661],[-8.676194],[7.804847],[-3.075191],[7.827025],[1.446148],[7.183847]],[[2.581399],[-6.432214],[-3.158540],[0.124306],[9.323341],[8.017562],[-0.744032],[-0.657268],[-0.112894],[2.262769],[-2.133346]],[[4.552314],[7.784534],[8.133260],[-8.572023],[-0.058680],[-1.206764],[6.948436],[7.843605],[-2.925080],[-8.587392],[5.944759]],[[1.597061],[6.836044],[7.714153],[8.964750],[8.018485],[-9.703099],[8.903801],[5.078462],[-5.807693],[8.749622],[5.243625]],[[5.587230],[-4.953209],[2.558608],[-9.408069],[0.613292],[3.908185],[8.166571],[7.337275],[-2.914213],[4.117858],[-8.768802]],[[8.319720],[-9.896829],[1.783185],[-5.427919],[6.275298],[-7.871298],[7.734782],[-2.903743],[-0.299697],[-4.897026],[-3.215717]],[[-0.397710],[-0.683647],[-8.460172],[7.566384],[-1.076053],[-1.049333],[-8.507214],[-1.769208],[-4.014514],[3.188567],[6.955419]],[[5.694744],[-6.099364],[4.473178],[1.253453],[-8.067559],[-9.220174],[-2.500337],[-7.714564],[3.012474],[-9.329252],[-7.119723]],[[9.884574],[-5.062150],[-7.045253],[-8.617982],[-3.929846],[1.565398],[-7.860974],[0.953584],[-4.327124],[9.026181],[-2.838491]],[[8.240006],[-2.454157],[6.357862],[-2.774652],[0.604827],[9.015995],[-8.095390],[4.340924],[1.570586],[-7.898280],[-3.684713]]], dtype = "float64")#candidate|29|(16, 11, 1)|const|float64
bop_30 = relay.mod(var_28.astype('float64'), const_29.astype('float64')) # shape=(16, 11, 1)
output = bop_30
output2 = bop_30
func_35 = relay.Function([var_28,], output)
mod['func_35'] = func_35
mod = relay.transform.InferType()(mod)
mutated_mod['func_35'] = func_35
mutated_mod = relay.transform.InferType()(mutated_mod)
var_36 = relay.var("var_36", dtype = "float64", shape = ())#candidate|36|()|var|float64
func_35_call = mutated_mod.get_global_var('func_35')
call_37 = func_35_call(var_36)
output = call_37
func_38 = relay.Function([var_36], output)
mutated_mod['func_38'] = func_38
mutated_mod = relay.transform.InferType()(mutated_mod)
var_50 = relay.var("var_50", dtype = "float64", shape = (12, 10))#candidate|50|(12, 10)|var|float64
uop_51 = relay.erf(var_50.astype('float64')) # shape=(12, 10)
bop_53 = relay.maximum(uop_51.astype('uint8'), relay.reshape(var_50.astype('uint8'), relay.shape_of(uop_51))) # shape=(12, 10)
func_35_call = mod.get_global_var('func_35')
func_38_call = mutated_mod.get_global_var('func_38')
var_57 = relay.var("var_57", dtype = "float64", shape = ())#candidate|57|()|var|float64
call_56 = func_35_call(relay.reshape(var_57.astype('float64'), []))
call_58 = func_35_call(relay.reshape(var_57.astype('float64'), []))
bop_61 = relay.less(bop_53.astype('bool'), relay.reshape(uop_51.astype('bool'), relay.shape_of(bop_53))) # shape=(12, 10)
bop_65 = relay.equal(bop_53.astype('bool'), relay.reshape(uop_51.astype('bool'), relay.shape_of(bop_53))) # shape=(12, 10)
output = relay.Tuple([call_56,var_57,bop_61,bop_65,])
output2 = relay.Tuple([call_58,var_57,bop_61,bop_65,])
func_69 = relay.Function([var_50,var_57,], output)
mod['func_69'] = func_69
mod = relay.transform.InferType()(mod)
mutated_mod['func_69'] = func_69
mutated_mod = relay.transform.InferType()(mutated_mod)
func_69_call = mutated_mod.get_global_var('func_69')
var_71 = relay.var("var_71", dtype = "float64", shape = (12, 10))#candidate|71|(12, 10)|var|float64
var_72 = relay.var("var_72", dtype = "float64", shape = ())#candidate|72|()|var|float64
call_70 = func_69_call(var_71,var_72,)
output = call_70
func_73 = relay.Function([var_71,var_72,], output)
mutated_mod['func_73'] = func_73
mutated_mod = relay.transform.InferType()(mutated_mod)
var_97 = relay.var("var_97", dtype = "int16", shape = ())#candidate|97|()|var|int16
var_98 = relay.var("var_98", dtype = "int16", shape = (4, 11))#candidate|98|(4, 11)|var|int16
bop_99 = relay.greater_equal(var_97.astype('bool'), var_98.astype('bool')) # shape=(4, 11)
uop_103 = relay.asin(bop_99.astype('float32')) # shape=(4, 11)
output = uop_103
output2 = uop_103
func_105 = relay.Function([var_97,var_98,], output)
mod['func_105'] = func_105
mod = relay.transform.InferType()(mod)
var_106 = relay.var("var_106", dtype = "int16", shape = ())#candidate|106|()|var|int16
var_107 = relay.var("var_107", dtype = "int16", shape = (4, 11))#candidate|107|(4, 11)|var|int16
output = func_105(var_106,var_107,)
func_108 = relay.Function([var_106,var_107,], output)
mutated_mod['func_108'] = func_108
mutated_mod = relay.transform.InferType()(mutated_mod)
var_142 = relay.var("var_142", dtype = "int8", shape = (3, 10, 4))#candidate|142|(3, 10, 4)|var|int8
var_143 = relay.var("var_143", dtype = "int8", shape = (3, 10, 4))#candidate|143|(3, 10, 4)|var|int8
bop_144 = relay.right_shift(var_142.astype('int8'), relay.reshape(var_143.astype('int8'), relay.shape_of(var_142))) # shape=(3, 10, 4)
output = bop_144
output2 = bop_144
func_149 = relay.Function([var_142,var_143,], output)
mod['func_149'] = func_149
mod = relay.transform.InferType()(mod)
var_150 = relay.var("var_150", dtype = "int8", shape = (3, 10, 4))#candidate|150|(3, 10, 4)|var|int8
var_151 = relay.var("var_151", dtype = "int8", shape = (3, 10, 4))#candidate|151|(3, 10, 4)|var|int8
output = func_149(var_150,var_151,)
func_152 = relay.Function([var_150,var_151,], output)
mutated_mod['func_152'] = func_152
mutated_mod = relay.transform.InferType()(mutated_mod)
const_193 = relay.const([[[-0.461182,-3.991705],[-9.470712,-4.886412],[7.195610,-2.089892],[1.010141,-1.302271],[-7.200602,-6.271096],[-9.732286,4.906022]],[[1.775397,8.873244],[-1.536715,-3.029551],[-6.571908,0.791352],[-5.266828,5.118871],[-5.084295,-6.977666],[9.942909,8.822485]],[[8.016309,4.587090],[-7.676050,-9.226200],[-1.321988,9.569105],[-8.402012,3.074329],[-4.570870,3.175564],[1.383401,7.029248]],[[3.961247,8.588823],[-4.825323,-2.474513],[8.147581,7.080060],[-8.683633,3.712549],[2.696628,4.743534],[-3.968941,-3.757996]],[[-4.664538,2.520354],[-5.589071,9.533155],[-9.919814,-1.851861],[-8.120251,-7.051165],[3.428000,1.681334],[5.110318,-1.991285]],[[-1.140348,1.030970],[8.603482,6.173965],[7.424797,4.199784],[1.925282,2.845838],[1.751840,0.319198],[-8.142160,2.926574]],[[7.880427,1.431181],[8.418016,-1.759787],[-3.899274,-6.326962],[6.954715,0.822721],[-2.109486,8.925712],[6.759819,-5.846764]],[[-5.542851,-2.005572],[5.559161,-1.543946],[3.611424,-8.597561],[-4.445526,-5.521848],[-3.233947,-0.718898],[-3.196287,-4.002149]],[[-6.252515,2.667663],[7.888731,-3.541111],[7.511713,-2.218082],[7.219534,5.953184],[2.163013,7.342219],[-6.192619,-1.037922]],[[4.022118,9.076993],[-6.028806,-3.380254],[4.406414,-5.450340],[1.376798,-3.397415],[-0.828691,-6.539023],[-8.302579,-3.097675]],[[1.878583,8.210674],[9.686100,-0.273694],[-4.212903,-6.391818],[6.846411,0.947350],[-6.989520,6.438863],[-7.787897,-4.767459]],[[1.573482,-9.753775],[8.380635,4.086225],[5.101568,-1.475784],[-9.182399,9.665074],[-1.004020,6.898118],[-9.380129,-3.226063]],[[-6.348224,4.090467],[-9.859171,0.670720],[-4.392723,6.793966],[-9.617004,-4.143145],[4.562377,-3.356734],[4.511392,-1.382169]],[[-1.824796,5.494036],[-9.221996,5.159477],[-1.835410,4.399641],[9.399716,-1.982185],[8.759993,8.932178],[-2.707953,9.508419]]], dtype = "float64")#candidate|193|(14, 6, 2)|const|float64
uop_194 = relay.erf(const_193.astype('float64')) # shape=(14, 6, 2)
func_35_call = mod.get_global_var('func_35')
func_38_call = mutated_mod.get_global_var('func_38')
var_197 = relay.var("var_197", dtype = "float64", shape = ())#candidate|197|()|var|float64
call_196 = func_35_call(relay.reshape(var_197.astype('float64'), []))
call_198 = func_35_call(relay.reshape(var_197.astype('float64'), []))
var_200 = relay.var("var_200", dtype = "float64", shape = (14, 6, 2))#candidate|200|(14, 6, 2)|var|float64
bop_201 = relay.logical_or(uop_194.astype('bool'), relay.reshape(var_200.astype('bool'), relay.shape_of(uop_194))) # shape=(14, 6, 2)
func_35_call = mod.get_global_var('func_35')
func_38_call = mutated_mod.get_global_var('func_38')
call_209 = func_35_call(relay.reshape(var_197.astype('float64'), []))
call_210 = func_35_call(relay.reshape(var_197.astype('float64'), []))
uop_212 = relay.asinh(uop_194.astype('float64')) # shape=(14, 6, 2)
uop_217 = relay.acosh(uop_212.astype('float64')) # shape=(14, 6, 2)
func_149_call = mod.get_global_var('func_149')
func_152_call = mutated_mod.get_global_var('func_152')
var_221 = relay.var("var_221", dtype = "int8", shape = (30, 4))#candidate|221|(30, 4)|var|int8
call_220 = func_149_call(relay.reshape(var_221.astype('int8'), [3, 10, 4]), relay.reshape(var_221.astype('int8'), [3, 10, 4]), )
call_222 = func_149_call(relay.reshape(var_221.astype('int8'), [3, 10, 4]), relay.reshape(var_221.astype('int8'), [3, 10, 4]), )
output = relay.Tuple([call_196,var_197,bop_201,call_209,uop_217,call_220,var_221,])
output2 = relay.Tuple([call_198,var_197,bop_201,call_210,uop_217,call_222,var_221,])
func_223 = relay.Function([var_197,var_200,var_221,], output)
mod['func_223'] = func_223
mod = relay.transform.InferType()(mod)
var_224 = relay.var("var_224", dtype = "float64", shape = ())#candidate|224|()|var|float64
var_225 = relay.var("var_225", dtype = "float64", shape = (14, 6, 2))#candidate|225|(14, 6, 2)|var|float64
var_226 = relay.var("var_226", dtype = "int8", shape = (30, 4))#candidate|226|(30, 4)|var|int8
output = func_223(var_224,var_225,var_226,)
func_227 = relay.Function([var_224,var_225,var_226,], output)
mutated_mod['func_227'] = func_227
mutated_mod = relay.transform.InferType()(mutated_mod)
var_337 = relay.var("var_337", dtype = "float32", shape = (6, 14, 9))#candidate|337|(6, 14, 9)|var|float32
uop_338 = relay.log(var_337.astype('float32')) # shape=(6, 14, 9)
output = uop_338
output2 = uop_338
func_340 = relay.Function([var_337,], output)
mod['func_340'] = func_340
mod = relay.transform.InferType()(mod)
mutated_mod['func_340'] = func_340
mutated_mod = relay.transform.InferType()(mutated_mod)
var_341 = relay.var("var_341", dtype = "float32", shape = (6, 14, 9))#candidate|341|(6, 14, 9)|var|float32
func_340_call = mutated_mod.get_global_var('func_340')
call_342 = func_340_call(var_341)
output = call_342
func_343 = relay.Function([var_341], output)
mutated_mod['func_343'] = func_343
mutated_mod = relay.transform.InferType()(mutated_mod)
var_360 = relay.var("var_360", dtype = "uint16", shape = (10, 16))#candidate|360|(10, 16)|var|uint16
var_361 = relay.var("var_361", dtype = "uint16", shape = (10, 16))#candidate|361|(10, 16)|var|uint16
bop_362 = relay.multiply(var_360.astype('uint16'), relay.reshape(var_361.astype('uint16'), relay.shape_of(var_360))) # shape=(10, 16)
uop_368 = relay.sin(var_361.astype('float64')) # shape=(10, 16)
uop_370 = relay.sigmoid(uop_368.astype('float64')) # shape=(10, 16)
func_149_call = mod.get_global_var('func_149')
func_152_call = mutated_mod.get_global_var('func_152')
var_373 = relay.var("var_373", dtype = "int8", shape = (1, 120))#candidate|373|(1, 120)|var|int8
call_372 = func_149_call(relay.reshape(var_373.astype('int8'), [3, 10, 4]), relay.reshape(var_373.astype('int8'), [3, 10, 4]), )
call_374 = func_149_call(relay.reshape(var_373.astype('int8'), [3, 10, 4]), relay.reshape(var_373.astype('int8'), [3, 10, 4]), )
bop_375 = relay.greater_equal(uop_370.astype('bool'), relay.reshape(var_360.astype('bool'), relay.shape_of(uop_370))) # shape=(10, 16)
uop_378 = relay.erf(bop_375.astype('float64')) # shape=(10, 16)
bop_387 = relay.logical_and(uop_378.astype('bool'), relay.reshape(uop_368.astype('bool'), relay.shape_of(uop_378))) # shape=(10, 16)
bop_390 = relay.logical_xor(bop_387.astype('uint8'), relay.reshape(bop_362.astype('uint8'), relay.shape_of(bop_387))) # shape=(10, 16)
uop_395 = relay.cos(bop_362.astype('float64')) # shape=(10, 16)
var_397 = relay.var("var_397", dtype = "bool", shape = (10, 16))#candidate|397|(10, 16)|var|bool
bop_398 = relay.less_equal(bop_387.astype('bool'), relay.reshape(var_397.astype('bool'), relay.shape_of(bop_387))) # shape=(10, 16)
uop_402 = relay.asin(uop_378.astype('float64')) # shape=(10, 16)
uop_406 = relay.tan(uop_402.astype('float64')) # shape=(10, 16)
uop_409 = relay.atanh(uop_406.astype('float64')) # shape=(10, 16)
uop_411 = relay.acosh(uop_395.astype('float32')) # shape=(10, 16)
bop_413 = relay.power(uop_409.astype('float32'), relay.reshape(bop_398.astype('float32'), relay.shape_of(uop_409))) # shape=(10, 16)
bop_416 = relay.maximum(bop_413.astype('int16'), relay.reshape(bop_390.astype('int16'), relay.shape_of(bop_413))) # shape=(10, 16)
output = relay.Tuple([call_372,var_373,uop_411,bop_416,])
output2 = relay.Tuple([call_374,var_373,uop_411,bop_416,])
func_419 = relay.Function([var_360,var_361,var_373,var_397,], output)
mod['func_419'] = func_419
mod = relay.transform.InferType()(mod)
mutated_mod['func_419'] = func_419
mutated_mod = relay.transform.InferType()(mutated_mod)
func_419_call = mutated_mod.get_global_var('func_419')
var_421 = relay.var("var_421", dtype = "uint16", shape = (10, 16))#candidate|421|(10, 16)|var|uint16
var_422 = relay.var("var_422", dtype = "uint16", shape = (10, 16))#candidate|422|(10, 16)|var|uint16
var_423 = relay.var("var_423", dtype = "int8", shape = (1, 120))#candidate|423|(1, 120)|var|int8
var_424 = relay.var("var_424", dtype = "bool", shape = (10, 16))#candidate|424|(10, 16)|var|bool
call_420 = func_419_call(var_421,var_422,var_423,var_424,)
output = call_420
func_425 = relay.Function([var_421,var_422,var_423,var_424,], output)
mutated_mod['func_425'] = func_425
mutated_mod = relay.transform.InferType()(mutated_mod)
var_438 = relay.var("var_438", dtype = "int64", shape = (1, 9, 4))#candidate|438|(1, 9, 4)|var|int64
var_439 = relay.var("var_439", dtype = "int64", shape = (16, 9, 4))#candidate|439|(16, 9, 4)|var|int64
bop_440 = relay.right_shift(var_438.astype('int64'), var_439.astype('int64')) # shape=(16, 9, 4)
var_443 = relay.var("var_443", dtype = "int64", shape = (16, 9, 4))#candidate|443|(16, 9, 4)|var|int64
bop_444 = relay.not_equal(bop_440.astype('bool'), relay.reshape(var_443.astype('bool'), relay.shape_of(bop_440))) # shape=(16, 9, 4)
output = bop_444
output2 = bop_444
func_452 = relay.Function([var_438,var_439,var_443,], output)
mod['func_452'] = func_452
mod = relay.transform.InferType()(mod)
var_453 = relay.var("var_453", dtype = "int64", shape = (1, 9, 4))#candidate|453|(1, 9, 4)|var|int64
var_454 = relay.var("var_454", dtype = "int64", shape = (16, 9, 4))#candidate|454|(16, 9, 4)|var|int64
var_455 = relay.var("var_455", dtype = "int64", shape = (16, 9, 4))#candidate|455|(16, 9, 4)|var|int64
output = func_452(var_453,var_454,var_455,)
func_456 = relay.Function([var_453,var_454,var_455,], output)
mutated_mod['func_456'] = func_456
mutated_mod = relay.transform.InferType()(mutated_mod)
const_493 = relay.const([False,False,False,False,False,False,True], dtype = "bool")#candidate|493|(7,)|const|bool
const_494 = relay.const([False,True,False,True,False,False,True], dtype = "bool")#candidate|494|(7,)|const|bool
bop_495 = relay.logical_or(const_493.astype('bool'), relay.reshape(const_494.astype('bool'), relay.shape_of(const_493))) # shape=(7,)
output = bop_495
output2 = bop_495
func_502 = relay.Function([], output)
mod['func_502'] = func_502
mod = relay.transform.InferType()(mod)
output = func_502()
func_503 = relay.Function([], output)
mutated_mod['func_503'] = func_503
mutated_mod = relay.transform.InferType()(mutated_mod)
func_502_call = mod.get_global_var('func_502')
func_503_call = mutated_mod.get_global_var('func_503')
call_518 = func_502_call()
call_519 = func_502_call()
func_35_call = mod.get_global_var('func_35')
func_38_call = mutated_mod.get_global_var('func_38')
const_540 = relay.const(6.374908, dtype = "float64")#candidate|540|()|const|float64
call_539 = func_35_call(relay.reshape(const_540.astype('float64'), []))
call_541 = func_35_call(relay.reshape(const_540.astype('float64'), []))
output = relay.Tuple([call_518,call_539,const_540,])
output2 = relay.Tuple([call_519,call_541,const_540,])
func_557 = relay.Function([], output)
mod['func_557'] = func_557
mod = relay.transform.InferType()(mod)
output = func_557()
func_558 = relay.Function([], output)
mutated_mod['func_558'] = func_558
mutated_mod = relay.transform.InferType()(mutated_mod)
func_502_call = mod.get_global_var('func_502')
func_503_call = mutated_mod.get_global_var('func_503')
call_577 = func_502_call()
call_578 = func_502_call()
func_419_call = mod.get_global_var('func_419')
func_425_call = mutated_mod.get_global_var('func_425')
var_580 = relay.var("var_580", dtype = "uint16", shape = (160,))#candidate|580|(160,)|var|uint16
var_581 = relay.var("var_581", dtype = "int8", shape = (120, 1))#candidate|581|(120, 1)|var|int8
call_579 = relay.TupleGetItem(func_419_call(relay.reshape(var_580.astype('uint16'), [10, 16]), relay.reshape(var_580.astype('uint16'), [10, 16]), relay.reshape(var_581.astype('int8'), [1, 120]), relay.reshape(var_580.astype('bool'), [10, 16]), ), 3)
call_582 = relay.TupleGetItem(func_425_call(relay.reshape(var_580.astype('uint16'), [10, 16]), relay.reshape(var_580.astype('uint16'), [10, 16]), relay.reshape(var_581.astype('int8'), [1, 120]), relay.reshape(var_580.astype('bool'), [10, 16]), ), 3)
func_340_call = mod.get_global_var('func_340')
func_343_call = mutated_mod.get_global_var('func_343')
var_593 = relay.var("var_593", dtype = "float32", shape = (756,))#candidate|593|(756,)|var|float32
call_592 = func_340_call(relay.reshape(var_593.astype('float32'), [6, 14, 9]))
call_594 = func_340_call(relay.reshape(var_593.astype('float32'), [6, 14, 9]))
uop_601 = relay.rsqrt(call_577.astype('float32')) # shape=(7,)
uop_603 = relay.rsqrt(call_578.astype('float32')) # shape=(7,)
uop_607 = relay.sin(uop_601.astype('float64')) # shape=(7,)
uop_609 = relay.sin(uop_603.astype('float64')) # shape=(7,)
bop_610 = relay.right_shift(uop_607.astype('int8'), relay.reshape(uop_601.astype('int8'), relay.shape_of(uop_607))) # shape=(7,)
bop_613 = relay.right_shift(uop_609.astype('int8'), relay.reshape(uop_603.astype('int8'), relay.shape_of(uop_609))) # shape=(7,)
uop_615 = relay.log(uop_601.astype('float64')) # shape=(7,)
uop_617 = relay.log(uop_603.astype('float64')) # shape=(7,)
output = relay.Tuple([call_579,var_580,var_581,call_592,var_593,bop_610,uop_615,])
output2 = relay.Tuple([call_582,var_580,var_581,call_594,var_593,bop_613,uop_617,])
func_618 = relay.Function([var_580,var_581,var_593,], output)
mod['func_618'] = func_618
mod = relay.transform.InferType()(mod)
var_619 = relay.var("var_619", dtype = "uint16", shape = (160,))#candidate|619|(160,)|var|uint16
var_620 = relay.var("var_620", dtype = "int8", shape = (120, 1))#candidate|620|(120, 1)|var|int8
var_621 = relay.var("var_621", dtype = "float32", shape = (756,))#candidate|621|(756,)|var|float32
output = func_618(var_619,var_620,var_621,)
func_622 = relay.Function([var_619,var_620,var_621,], output)
mutated_mod['func_622'] = func_622
mutated_mod = relay.transform.InferType()(mutated_mod)
var_642 = relay.var("var_642", dtype = "int8", shape = (7, 3))#candidate|642|(7, 3)|var|int8
var_643 = relay.var("var_643", dtype = "int8", shape = (7, 3))#candidate|643|(7, 3)|var|int8
bop_644 = relay.less(var_642.astype('bool'), relay.reshape(var_643.astype('bool'), relay.shape_of(var_642))) # shape=(7, 3)
bop_649 = relay.greater(bop_644.astype('bool'), relay.reshape(var_642.astype('bool'), relay.shape_of(bop_644))) # shape=(7, 3)
bop_654 = relay.bitwise_or(bop_644.astype('uint32'), relay.reshape(bop_649.astype('uint32'), relay.shape_of(bop_644))) # shape=(7, 3)
func_69_call = mod.get_global_var('func_69')
func_73_call = mutated_mod.get_global_var('func_73')
const_660 = relay.const([5.523061,4.922307,4.365016,2.578729,6.872432,-7.024939,-3.741753,-2.581207,8.332810,4.986087,-6.171142,6.737822,7.412710,0.513301,1.180930,-3.846339,9.868884,-7.596429,-1.764497,8.961076,-3.919773,-2.772880,-5.940795,-7.641425,-1.839037,-4.122664,9.818076,2.814543,8.603775,6.580692,5.645156,3.652555,3.347778,-8.715594,2.767864,-6.381363,0.009127,0.823445,7.067046,6.528197,5.489090,-5.534305,-3.171475,1.617241,-6.803203,-6.562333,4.576994,0.647293,-9.803784,-3.390651,7.561590,6.658592,1.313011,3.911643,-6.977441,-5.137716,-2.448898,9.079322,-6.797106,-1.573530,2.509132,0.955411,-2.817996,-9.455629,-3.552407,5.863186,4.242650,-0.085735,8.357690,4.914559,-3.548149,4.092728,0.384556,2.473423,0.505710,-2.664661,6.838121,0.317386,-6.621871,-9.494466,5.699909,8.882348,-6.419085,0.479755,5.496960,-9.159930,2.616826,-6.648002,-0.102852,-6.971099,3.421493,-6.261670,-5.705675,-8.089416,6.532546,0.237744,-3.442924,8.371005,-8.108170,-4.235942,-2.973276,-9.842874,-4.503797,-8.341334,-3.847140,3.588292,0.106852,-6.162972,2.664812,5.086442,5.108964,9.473997,0.932779,4.212016,-9.489568,8.142331,-7.695218,-8.569973,-1.302098,5.326934], dtype = "float64")#candidate|660|(120,)|const|float64
const_661 = relay.const(-9.639206, dtype = "float64")#candidate|661|()|const|float64
call_659 = relay.TupleGetItem(func_69_call(relay.reshape(const_660.astype('float64'), [12, 10]), relay.reshape(const_661.astype('float64'), []), ), 1)
call_662 = relay.TupleGetItem(func_73_call(relay.reshape(const_660.astype('float64'), [12, 10]), relay.reshape(const_661.astype('float64'), []), ), 1)
func_105_call = mod.get_global_var('func_105')
func_108_call = mutated_mod.get_global_var('func_108')
var_669 = relay.var("var_669", dtype = "int16", shape = (44,))#candidate|669|(44,)|var|int16
call_668 = func_105_call(relay.reshape(const_661.astype('int16'), []), relay.reshape(var_669.astype('int16'), [4, 11]), )
call_670 = func_105_call(relay.reshape(const_661.astype('int16'), []), relay.reshape(var_669.astype('int16'), [4, 11]), )
bop_673 = relay.bitwise_or(call_668.astype('int16'), call_659.astype('int16')) # shape=(4, 11)
bop_676 = relay.bitwise_or(call_670.astype('int16'), call_662.astype('int16')) # shape=(4, 11)
output = relay.Tuple([bop_654,const_660,const_661,var_669,bop_673,])
output2 = relay.Tuple([bop_654,const_660,const_661,var_669,bop_676,])
func_680 = relay.Function([var_642,var_643,var_669,], output)
mod['func_680'] = func_680
mod = relay.transform.InferType()(mod)
var_681 = relay.var("var_681", dtype = "int8", shape = (7, 3))#candidate|681|(7, 3)|var|int8
var_682 = relay.var("var_682", dtype = "int8", shape = (7, 3))#candidate|682|(7, 3)|var|int8
var_683 = relay.var("var_683", dtype = "int16", shape = (44,))#candidate|683|(44,)|var|int16
output = func_680(var_681,var_682,var_683,)
func_684 = relay.Function([var_681,var_682,var_683,], output)
mutated_mod['func_684'] = func_684
mutated_mod = relay.transform.InferType()(mutated_mod)
var_696 = relay.var("var_696", dtype = "int8", shape = (12, 5, 11))#candidate|696|(12, 5, 11)|var|int8
var_697 = relay.var("var_697", dtype = "int8", shape = (12, 5, 11))#candidate|697|(12, 5, 11)|var|int8
bop_698 = relay.right_shift(var_696.astype('int8'), relay.reshape(var_697.astype('int8'), relay.shape_of(var_696))) # shape=(12, 5, 11)
bop_702 = relay.minimum(var_697.astype('int8'), relay.reshape(var_696.astype('int8'), relay.shape_of(var_697))) # shape=(12, 5, 11)
bop_707 = relay.power(bop_698.astype('float32'), relay.reshape(bop_702.astype('float32'), relay.shape_of(bop_698))) # shape=(12, 5, 11)
output = relay.Tuple([bop_707,])
output2 = relay.Tuple([bop_707,])
func_712 = relay.Function([var_696,var_697,], output)
mod['func_712'] = func_712
mod = relay.transform.InferType()(mod)
mutated_mod['func_712'] = func_712
mutated_mod = relay.transform.InferType()(mutated_mod)
func_712_call = mutated_mod.get_global_var('func_712')
var_714 = relay.var("var_714", dtype = "int8", shape = (12, 5, 11))#candidate|714|(12, 5, 11)|var|int8
var_715 = relay.var("var_715", dtype = "int8", shape = (12, 5, 11))#candidate|715|(12, 5, 11)|var|int8
call_713 = func_712_call(var_714,var_715,)
output = call_713
func_716 = relay.Function([var_714,var_715,], output)
mutated_mod['func_716'] = func_716
mutated_mod = relay.transform.InferType()(mutated_mod)
var_724 = relay.var("var_724", dtype = "float64", shape = (5,))#candidate|724|(5,)|var|float64
uop_725 = relay.sinh(var_724.astype('float64')) # shape=(5,)
func_419_call = mod.get_global_var('func_419')
func_425_call = mutated_mod.get_global_var('func_425')
var_730 = relay.var("var_730", dtype = "uint16", shape = (160,))#candidate|730|(160,)|var|uint16
const_731 = relay.const([[5,10],[-8,-1],[7,-2],[-3,7],[1,-5],[-3,1],[-7,9],[-3,3],[-2,-9],[7,5],[-1,-7],[7,-10],[-9,7],[-5,5],[7,7],[10,-7],[4,5],[5,-7],[4,3],[-7,10],[-8,-8],[-3,-5],[7,-2],[2,1],[5,-9],[-5,7],[10,1],[-4,5],[-7,-7],[10,-3],[-3,-3],[-4,3],[10,2],[-8,-5],[10,8],[-6,-8],[-10,5],[4,-6],[-5,-2],[-1,8],[6,-7],[10,3],[-7,-6],[5,-7],[-8,4],[4,-10],[7,6],[8,-1],[6,7],[9,8],[10,-10],[5,-10],[4,4],[6,-5],[-8,2],[8,8],[-3,-2],[4,10],[-1,9],[-2,10]], dtype = "int8")#candidate|731|(60, 2)|const|int8
call_729 = relay.TupleGetItem(func_419_call(relay.reshape(var_730.astype('uint16'), [10, 16]), relay.reshape(var_730.astype('uint16'), [10, 16]), relay.reshape(const_731.astype('int8'), [1, 120]), relay.reshape(var_730.astype('bool'), [10, 16]), ), 3)
call_732 = relay.TupleGetItem(func_425_call(relay.reshape(var_730.astype('uint16'), [10, 16]), relay.reshape(var_730.astype('uint16'), [10, 16]), relay.reshape(const_731.astype('int8'), [1, 120]), relay.reshape(var_730.astype('bool'), [10, 16]), ), 3)
bop_735 = relay.minimum(uop_725.astype('uint16'), relay.reshape(var_724.astype('uint16'), relay.shape_of(uop_725))) # shape=(5,)
output = relay.Tuple([call_729,var_730,const_731,bop_735,])
output2 = relay.Tuple([call_732,var_730,const_731,bop_735,])
func_738 = relay.Function([var_724,var_730,], output)
mod['func_738'] = func_738
mod = relay.transform.InferType()(mod)
var_739 = relay.var("var_739", dtype = "float64", shape = (5,))#candidate|739|(5,)|var|float64
var_740 = relay.var("var_740", dtype = "uint16", shape = (160,))#candidate|740|(160,)|var|uint16
output = func_738(var_739,var_740,)
func_741 = relay.Function([var_739,var_740,], output)
mutated_mod['func_741'] = func_741
mutated_mod = relay.transform.InferType()(mutated_mod)
var_754 = relay.var("var_754", dtype = "uint8", shape = (11, 10, 4))#candidate|754|(11, 10, 4)|var|uint8
var_755 = relay.var("var_755", dtype = "uint8", shape = (11, 10, 4))#candidate|755|(11, 10, 4)|var|uint8
bop_756 = relay.greater(var_754.astype('bool'), relay.reshape(var_755.astype('bool'), relay.shape_of(var_754))) # shape=(11, 10, 4)
bop_761 = relay.left_shift(bop_756.astype('int8'), relay.reshape(var_754.astype('int8'), relay.shape_of(bop_756))) # shape=(11, 10, 4)
bop_765 = relay.equal(var_754.astype('bool'), relay.reshape(bop_756.astype('bool'), relay.shape_of(var_754))) # shape=(11, 10, 4)
uop_768 = relay.sin(var_754.astype('float64')) # shape=(11, 10, 4)
output = relay.Tuple([bop_761,bop_765,uop_768,])
output2 = relay.Tuple([bop_761,bop_765,uop_768,])
func_770 = relay.Function([var_754,var_755,], output)
mod['func_770'] = func_770
mod = relay.transform.InferType()(mod)
var_771 = relay.var("var_771", dtype = "uint8", shape = (11, 10, 4))#candidate|771|(11, 10, 4)|var|uint8
var_772 = relay.var("var_772", dtype = "uint8", shape = (11, 10, 4))#candidate|772|(11, 10, 4)|var|uint8
output = func_770(var_771,var_772,)
func_773 = relay.Function([var_771,var_772,], output)
mutated_mod['func_773'] = func_773
mutated_mod = relay.transform.InferType()(mutated_mod)
var_778 = relay.var("var_778", dtype = "int64", shape = (11, 1))#candidate|778|(11, 1)|var|int64
const_779 = relay.const([[-3,-5,-6,-4,7,-5,6,5,-3,2,-4,-5,4,-8],[-3,7,9,-6,-5,3,10,6,10,-3,10,-4,-10,9],[4,-7,6,-5,-7,9,3,-5,9,-10,5,-1,-1,-7],[-1,-4,-8,4,-6,8,-1,-5,-10,-5,4,-6,-1,1],[-10,-2,9,-8,9,-10,10,-8,8,3,-6,6,8,-8],[-9,-10,1,1,-9,6,10,3,-5,-1,6,10,-6,3],[-6,-4,2,-3,-7,-3,-3,6,-5,5,2,3,-4,2],[-9,4,8,2,4,-2,-10,-9,4,-8,6,10,-1,-6],[6,-3,-2,10,-6,9,8,3,8,4,-7,2,-6,-1],[-5,-10,-6,-6,3,-8,8,1,1,-7,3,-3,-4,6],[-5,-6,6,1,-10,-10,-5,-2,-8,-1,4,-9,7,-9]], dtype = "int64")#candidate|779|(11, 14)|const|int64
bop_780 = relay.equal(var_778.astype('bool'), const_779.astype('bool')) # shape=(11, 14)
uop_784 = relay.sigmoid(var_778.astype('float64')) # shape=(11, 1)
bop_786 = relay.multiply(bop_780.astype('uint64'), uop_784.astype('uint64')) # shape=(11, 14)
func_149_call = mod.get_global_var('func_149')
func_152_call = mutated_mod.get_global_var('func_152')
const_794 = relay.const([[5,6,-10,8,-3,-4,9,1,4,-9,5,-7,1,6,7,3,8,10,2,-3,-9,7,3,-5,8,-9,7,-3,-7,8,10,8,3,-5,-7,9,7,4,-7,-7,5,5,2,-2,-8,-5,-4,-4,2,-1,8,-7,-9,5,1,9,7,-10,6,8,-2,2,-7,-2,-8,7,4,-10,-6,9,8,3,-3,-4,5,-3,-6,-3,-3,-9,-8,6,-10,-7,5,4,-5,7,-2,-9,5,-3,8,2,-1,-2,-9,4,-9,6,8,-7,8,1,7,6,3,-2,5,2,-5,8,-5,3,-3,7,-10,-4,-8,-8]], dtype = "int8")#candidate|794|(1, 120)|const|int8
call_793 = func_149_call(relay.reshape(const_794.astype('int8'), [3, 10, 4]), relay.reshape(const_794.astype('int8'), [3, 10, 4]), )
call_795 = func_149_call(relay.reshape(const_794.astype('int8'), [3, 10, 4]), relay.reshape(const_794.astype('int8'), [3, 10, 4]), )
output = relay.Tuple([bop_786,call_793,const_794,])
output2 = relay.Tuple([bop_786,call_795,const_794,])
func_797 = relay.Function([var_778,], output)
mod['func_797'] = func_797
mod = relay.transform.InferType()(mod)
mutated_mod['func_797'] = func_797
mutated_mod = relay.transform.InferType()(mutated_mod)
var_798 = relay.var("var_798", dtype = "int64", shape = (11, 1))#candidate|798|(11, 1)|var|int64
func_797_call = mutated_mod.get_global_var('func_797')
call_799 = func_797_call(var_798)
output = call_799
func_800 = relay.Function([var_798], output)
mutated_mod['func_800'] = func_800
mutated_mod = relay.transform.InferType()(mutated_mod)
func_557_call = mod.get_global_var('func_557')
func_558_call = mutated_mod.get_global_var('func_558')
call_826 = relay.TupleGetItem(func_557_call(), 0)
call_827 = relay.TupleGetItem(func_558_call(), 0)
output = call_826
output2 = call_827
func_835 = relay.Function([], output)
mod['func_835'] = func_835
mod = relay.transform.InferType()(mod)
output = func_835()
func_836 = relay.Function([], output)
mutated_mod['func_836'] = func_836
mutated_mod = relay.transform.InferType()(mutated_mod)
var_843 = relay.var("var_843", dtype = "float32", shape = (4,))#candidate|843|(4,)|var|float32
var_844 = relay.var("var_844", dtype = "float32", shape = (4,))#candidate|844|(4,)|var|float32
bop_845 = relay.mod(var_843.astype('float32'), relay.reshape(var_844.astype('float32'), relay.shape_of(var_843))) # shape=(4,)
var_853 = relay.var("var_853", dtype = "float32", shape = (4,))#candidate|853|(4,)|var|float32
bop_854 = relay.logical_or(var_844.astype('bool'), relay.reshape(var_853.astype('bool'), relay.shape_of(var_844))) # shape=(4,)
func_105_call = mod.get_global_var('func_105')
func_108_call = mutated_mod.get_global_var('func_108')
const_860 = relay.const(-9, dtype = "int16")#candidate|860|()|const|int16
const_861 = relay.const([-7,3,-10,-1,-1,8,6,3,10,-4,6,7,-2,7,4,-4,-9,-1,5,-10,6,2,-6,5,9,-4,-3,-8,-8,8,-6,-4,-9,-6,6,5,-5,-4,-3,9,-8,-10,6,10], dtype = "int16")#candidate|861|(44,)|const|int16
call_859 = func_105_call(relay.reshape(const_860.astype('int16'), []), relay.reshape(const_861.astype('int16'), [4, 11]), )
call_862 = func_105_call(relay.reshape(const_860.astype('int16'), []), relay.reshape(const_861.astype('int16'), [4, 11]), )
func_149_call = mod.get_global_var('func_149')
func_152_call = mutated_mod.get_global_var('func_152')
const_864 = relay.const([-2,5,-4,8,-8,2,9,6,-6,-6,8,9,-2,7,9,-10,4,2,6,-8,2,-8,9,-5,-9,-1,-6,10,-1,2,2,9,-7,4,1,-10,3,6,-3,6,1,-10,10,-5,-8,1,8,6,2,-3,-1,10,2,-5,-8,-7,-2,7,9,-5,8,-9,-7,-4,-6,-4,9,-2,-9,-5,9,7,1,5,4,8,-3,10,5,-9,-8,-7,-6,-7,-7,-5,4,-10,-3,-6,-3,1,-5,-8,-5,-1,-1,-2,-9,-5,1,7,-9,2,-1,1,5,4,4,-10,8,-2,2,-5,2,9,7,3,-2,7], dtype = "int8")#candidate|864|(120,)|const|int8
call_863 = func_149_call(relay.reshape(const_864.astype('int8'), [3, 10, 4]), relay.reshape(const_864.astype('int8'), [3, 10, 4]), )
call_865 = func_149_call(relay.reshape(const_864.astype('int8'), [3, 10, 4]), relay.reshape(const_864.astype('int8'), [3, 10, 4]), )
output = relay.Tuple([bop_845,bop_854,call_859,const_860,const_861,call_863,const_864,])
output2 = relay.Tuple([bop_845,bop_854,call_862,const_860,const_861,call_865,const_864,])
func_869 = relay.Function([var_843,var_844,var_853,], output)
mod['func_869'] = func_869
mod = relay.transform.InferType()(mod)
mutated_mod['func_869'] = func_869
mutated_mod = relay.transform.InferType()(mutated_mod)
func_869_call = mutated_mod.get_global_var('func_869')
var_871 = relay.var("var_871", dtype = "float32", shape = (4,))#candidate|871|(4,)|var|float32
var_872 = relay.var("var_872", dtype = "float32", shape = (4,))#candidate|872|(4,)|var|float32
var_873 = relay.var("var_873", dtype = "float32", shape = (4,))#candidate|873|(4,)|var|float32
call_870 = func_869_call(var_871,var_872,var_873,)
output = call_870
func_874 = relay.Function([var_871,var_872,var_873,], output)
mutated_mod['func_874'] = func_874
mutated_mod = relay.transform.InferType()(mutated_mod)
func_835_call = mod.get_global_var('func_835')
func_836_call = mutated_mod.get_global_var('func_836')
call_883 = func_835_call()
call_884 = func_835_call()
output = relay.Tuple([call_883,])
output2 = relay.Tuple([call_884,])
func_892 = relay.Function([], output)
mod['func_892'] = func_892
mod = relay.transform.InferType()(mod)
output = func_892()
func_893 = relay.Function([], output)
mutated_mod['func_893'] = func_893
mutated_mod = relay.transform.InferType()(mutated_mod)
func_557_call = mod.get_global_var('func_557')
func_558_call = mutated_mod.get_global_var('func_558')
call_894 = relay.TupleGetItem(func_557_call(), 1)
call_895 = relay.TupleGetItem(func_558_call(), 1)
uop_900 = relay.acos(call_894.astype('float32')) # shape=(16, 11, 1)
uop_902 = relay.acos(call_895.astype('float32')) # shape=(16, 11, 1)
uop_903 = relay.log2(call_894.astype('float64')) # shape=(16, 11, 1)
uop_905 = relay.log2(call_895.astype('float64')) # shape=(16, 11, 1)
uop_910 = relay.sin(uop_900.astype('float32')) # shape=(16, 11, 1)
uop_912 = relay.sin(uop_902.astype('float32')) # shape=(16, 11, 1)
bop_915 = relay.minimum(uop_910.astype('int32'), relay.reshape(uop_903.astype('int32'), relay.shape_of(uop_910))) # shape=(16, 11, 1)
bop_918 = relay.minimum(uop_912.astype('int32'), relay.reshape(uop_905.astype('int32'), relay.shape_of(uop_912))) # shape=(16, 11, 1)
bop_919 = relay.power(uop_903.astype('float64'), relay.reshape(uop_900.astype('float64'), relay.shape_of(uop_903))) # shape=(16, 11, 1)
bop_922 = relay.power(uop_905.astype('float64'), relay.reshape(uop_902.astype('float64'), relay.shape_of(uop_905))) # shape=(16, 11, 1)
bop_923 = relay.add(bop_919.astype('int64'), relay.reshape(uop_910.astype('int64'), relay.shape_of(bop_919))) # shape=(16, 11, 1)
bop_926 = relay.add(bop_922.astype('int64'), relay.reshape(uop_912.astype('int64'), relay.shape_of(bop_922))) # shape=(16, 11, 1)
output = relay.Tuple([bop_915,bop_923,])
output2 = relay.Tuple([bop_918,bop_926,])
func_927 = relay.Function([], output)
mod['func_927'] = func_927
mod = relay.transform.InferType()(mod)
mutated_mod['func_927'] = func_927
mutated_mod = relay.transform.InferType()(mutated_mod)
func_927_call = mutated_mod.get_global_var('func_927')
call_928 = func_927_call()
output = call_928
func_929 = relay.Function([], output)
mutated_mod['func_929'] = func_929
mutated_mod = relay.transform.InferType()(mutated_mod)
var_972 = relay.var("var_972", dtype = "float64", shape = (9, 13))#candidate|972|(9, 13)|var|float64
var_973 = relay.var("var_973", dtype = "float64", shape = (9, 13))#candidate|973|(9, 13)|var|float64
bop_974 = relay.floor_mod(var_972.astype('float64'), relay.reshape(var_973.astype('float64'), relay.shape_of(var_972))) # shape=(9, 13)
func_502_call = mod.get_global_var('func_502')
func_503_call = mutated_mod.get_global_var('func_503')
call_978 = func_502_call()
call_979 = func_502_call()
func_738_call = mod.get_global_var('func_738')
func_741_call = mutated_mod.get_global_var('func_741')
const_985 = relay.const([8.586077,5.433054,-5.511576,4.390752,-0.313656], dtype = "float64")#candidate|985|(5,)|const|float64
var_986 = relay.var("var_986", dtype = "uint16", shape = (160,))#candidate|986|(160,)|var|uint16
call_984 = relay.TupleGetItem(func_738_call(relay.reshape(const_985.astype('float64'), [5,]), relay.reshape(var_986.astype('uint16'), [160,]), ), 3)
call_987 = relay.TupleGetItem(func_741_call(relay.reshape(const_985.astype('float64'), [5,]), relay.reshape(var_986.astype('uint16'), [160,]), ), 3)
func_680_call = mod.get_global_var('func_680')
func_684_call = mutated_mod.get_global_var('func_684')
const_992 = relay.const([-8,8,5,7,7,-5,-6,-10,10,8,8,-3,-1,-9,9,8,10,-9,-3,3,6], dtype = "int8")#candidate|992|(21,)|const|int8
const_993 = relay.const([[-2],[8],[2],[-4],[-3],[10],[-10],[9],[-8],[5],[-5],[4],[10],[-8],[3],[-1],[-4],[7],[-5],[1],[-5],[-3],[3],[4],[10],[-1],[-8],[3],[-4],[-6],[-2],[-1],[8],[-2],[-8],[2],[-7],[-10],[-4],[2],[-4],[8],[-7],[-7]], dtype = "int16")#candidate|993|(44, 1)|const|int16
call_991 = relay.TupleGetItem(func_680_call(relay.reshape(const_992.astype('int8'), [7, 3]), relay.reshape(const_992.astype('int8'), [7, 3]), relay.reshape(const_993.astype('int16'), [44,]), ), 0)
call_994 = relay.TupleGetItem(func_684_call(relay.reshape(const_992.astype('int8'), [7, 3]), relay.reshape(const_992.astype('int8'), [7, 3]), relay.reshape(const_993.astype('int16'), [44,]), ), 0)
var_1003 = relay.var("var_1003", dtype = "float64", shape = (9, 13))#candidate|1003|(9, 13)|var|float64
bop_1004 = relay.equal(var_972.astype('bool'), relay.reshape(var_1003.astype('bool'), relay.shape_of(var_972))) # shape=(9, 13)
func_223_call = mod.get_global_var('func_223')
func_227_call = mutated_mod.get_global_var('func_227')
const_1010 = relay.const(1.960732, dtype = "float64")#candidate|1010|()|const|float64
var_1011 = relay.var("var_1011", dtype = "float64", shape = (168,))#candidate|1011|(168,)|var|float64
var_1012 = relay.var("var_1012", dtype = "int8", shape = (3, 40))#candidate|1012|(3, 40)|var|int8
call_1009 = relay.TupleGetItem(func_223_call(relay.reshape(const_1010.astype('float64'), []), relay.reshape(var_1011.astype('float64'), [14, 6, 2]), relay.reshape(var_1012.astype('int8'), [30, 4]), ), 2)
call_1013 = relay.TupleGetItem(func_227_call(relay.reshape(const_1010.astype('float64'), []), relay.reshape(var_1011.astype('float64'), [14, 6, 2]), relay.reshape(var_1012.astype('int8'), [30, 4]), ), 2)
func_869_call = mod.get_global_var('func_869')
func_874_call = mutated_mod.get_global_var('func_874')
const_1015 = relay.const([9.322095,6.524118,3.339017,-9.675540], dtype = "float32")#candidate|1015|(4,)|const|float32
call_1014 = relay.TupleGetItem(func_869_call(relay.reshape(const_1015.astype('float32'), [4,]), relay.reshape(const_1015.astype('float32'), [4,]), relay.reshape(const_1015.astype('float32'), [4,]), ), 3)
call_1016 = relay.TupleGetItem(func_874_call(relay.reshape(const_1015.astype('float32'), [4,]), relay.reshape(const_1015.astype('float32'), [4,]), relay.reshape(const_1015.astype('float32'), [4,]), ), 3)
func_892_call = mod.get_global_var('func_892')
func_893_call = mutated_mod.get_global_var('func_893')
call_1019 = relay.TupleGetItem(func_892_call(), 0)
call_1020 = relay.TupleGetItem(func_893_call(), 0)
uop_1031 = relay.sqrt(bop_1004.astype('float64')) # shape=(9, 13)
uop_1038 = relay.asinh(uop_1031.astype('float64')) # shape=(9, 13)
bop_1040 = relay.divide(uop_1038.astype('float64'), relay.reshape(bop_1004.astype('float64'), relay.shape_of(uop_1038))) # shape=(9, 13)
output = relay.Tuple([bop_974,call_978,call_984,const_985,var_986,call_991,const_992,const_993,call_1009,const_1010,var_1011,var_1012,call_1014,const_1015,call_1019,bop_1040,])
output2 = relay.Tuple([bop_974,call_979,call_987,const_985,var_986,call_994,const_992,const_993,call_1013,const_1010,var_1011,var_1012,call_1016,const_1015,call_1020,bop_1040,])
func_1045 = relay.Function([var_972,var_973,var_986,var_1003,var_1011,var_1012,], output)
mod['func_1045'] = func_1045
mod = relay.transform.InferType()(mod)
mutated_mod['func_1045'] = func_1045
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1045_call = mutated_mod.get_global_var('func_1045')
var_1047 = relay.var("var_1047", dtype = "float64", shape = (9, 13))#candidate|1047|(9, 13)|var|float64
var_1048 = relay.var("var_1048", dtype = "float64", shape = (9, 13))#candidate|1048|(9, 13)|var|float64
var_1049 = relay.var("var_1049", dtype = "uint16", shape = (160,))#candidate|1049|(160,)|var|uint16
var_1050 = relay.var("var_1050", dtype = "float64", shape = (9, 13))#candidate|1050|(9, 13)|var|float64
var_1051 = relay.var("var_1051", dtype = "float64", shape = (168,))#candidate|1051|(168,)|var|float64
var_1052 = relay.var("var_1052", dtype = "int8", shape = (3, 40))#candidate|1052|(3, 40)|var|int8
call_1046 = func_1045_call(var_1047,var_1048,var_1049,var_1050,var_1051,var_1052,)
output = call_1046
func_1053 = relay.Function([var_1047,var_1048,var_1049,var_1050,var_1051,var_1052,], output)
mutated_mod['func_1053'] = func_1053
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1055 = relay.var("var_1055", dtype = "uint64", shape = (11, 3))#candidate|1055|(11, 3)|var|uint64
const_1056 = relay.const([[-3,10,-10],[7,-9,-1],[-9,6,-10],[9,1,4],[7,8,-9],[5,-7,-9],[7,-6,7],[3,-6,10],[-8,-9,-9],[5,-7,1],[3,6,-2]], dtype = "uint64")#candidate|1056|(11, 3)|const|uint64
bop_1057 = relay.right_shift(var_1055.astype('uint64'), relay.reshape(const_1056.astype('uint64'), relay.shape_of(var_1055))) # shape=(11, 3)
output = bop_1057
output2 = bop_1057
func_1062 = relay.Function([var_1055,], output)
mod['func_1062'] = func_1062
mod = relay.transform.InferType()(mod)
var_1063 = relay.var("var_1063", dtype = "uint64", shape = (11, 3))#candidate|1063|(11, 3)|var|uint64
output = func_1062(var_1063)
func_1064 = relay.Function([var_1063], output)
mutated_mod['func_1064'] = func_1064
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1079 = relay.var("var_1079", dtype = "float32", shape = ())#candidate|1079|()|var|float32
var_1080 = relay.var("var_1080", dtype = "float32", shape = (7,))#candidate|1080|(7,)|var|float32
bop_1081 = relay.subtract(var_1079.astype('float32'), var_1080.astype('float32')) # shape=(7,)
func_419_call = mod.get_global_var('func_419')
func_425_call = mutated_mod.get_global_var('func_425')
var_1085 = relay.var("var_1085", dtype = "uint16", shape = (160,))#candidate|1085|(160,)|var|uint16
const_1086 = relay.const([10,4,6,5,-1,10,8,9,3,-9,-1,2,-4,-10,-8,-7,8,10,-5,3,-1,-9,10,-6,-4,-2,9,-8,2,-7,-8,3,-6,-1,10,-6,8,-2,5,-3,-3,6,1,-5,10,5,6,6,-3,-2,-3,-3,-4,-1,-3,10,1,5,7,-5,2,9,-10,-10,3,5,10,9,3,5,-4,1,8,-5,8,6,-4,9,-1,6,-1,-2,-8,-5,10,-6,-2,10,-3,-6,9,-2,-6,-6,10,-1,-2,9,5,-1,-5,-1,-6,-8,-8,-1,-8,-9,-3,-1,6,1,-6,-7,8,6,-9,-10,-8,3], dtype = "int8")#candidate|1086|(120,)|const|int8
call_1084 = relay.TupleGetItem(func_419_call(relay.reshape(var_1085.astype('uint16'), [10, 16]), relay.reshape(var_1085.astype('uint16'), [10, 16]), relay.reshape(const_1086.astype('int8'), [1, 120]), relay.reshape(var_1085.astype('bool'), [10, 16]), ), 3)
call_1087 = relay.TupleGetItem(func_425_call(relay.reshape(var_1085.astype('uint16'), [10, 16]), relay.reshape(var_1085.astype('uint16'), [10, 16]), relay.reshape(const_1086.astype('int8'), [1, 120]), relay.reshape(var_1085.astype('bool'), [10, 16]), ), 3)
func_680_call = mod.get_global_var('func_680')
func_684_call = mutated_mod.get_global_var('func_684')
var_1090 = relay.var("var_1090", dtype = "int8", shape = (21,))#candidate|1090|(21,)|var|int8
var_1091 = relay.var("var_1091", dtype = "int16", shape = (44, 1))#candidate|1091|(44, 1)|var|int16
call_1089 = relay.TupleGetItem(func_680_call(relay.reshape(var_1090.astype('int8'), [7, 3]), relay.reshape(var_1090.astype('int8'), [7, 3]), relay.reshape(var_1091.astype('int16'), [44,]), ), 3)
call_1092 = relay.TupleGetItem(func_684_call(relay.reshape(var_1090.astype('int8'), [7, 3]), relay.reshape(var_1090.astype('int8'), [7, 3]), relay.reshape(var_1091.astype('int16'), [44,]), ), 3)
var_1093 = relay.var("var_1093", dtype = "int16", shape = (44,))#candidate|1093|(44,)|var|int16
bop_1094 = relay.multiply(call_1089.astype('uint32'), relay.reshape(var_1093.astype('uint32'), relay.shape_of(call_1089))) # shape=(44,)
bop_1097 = relay.multiply(call_1092.astype('uint32'), relay.reshape(var_1093.astype('uint32'), relay.shape_of(call_1092))) # shape=(44,)
output = relay.Tuple([bop_1081,call_1084,var_1085,const_1086,var_1090,var_1091,bop_1094,])
output2 = relay.Tuple([bop_1081,call_1087,var_1085,const_1086,var_1090,var_1091,bop_1097,])
func_1098 = relay.Function([var_1079,var_1080,var_1085,var_1090,var_1091,var_1093,], output)
mod['func_1098'] = func_1098
mod = relay.transform.InferType()(mod)
mutated_mod['func_1098'] = func_1098
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1098_call = mutated_mod.get_global_var('func_1098')
var_1100 = relay.var("var_1100", dtype = "float32", shape = ())#candidate|1100|()|var|float32
var_1101 = relay.var("var_1101", dtype = "float32", shape = (7,))#candidate|1101|(7,)|var|float32
var_1102 = relay.var("var_1102", dtype = "uint16", shape = (160,))#candidate|1102|(160,)|var|uint16
var_1103 = relay.var("var_1103", dtype = "int8", shape = (21,))#candidate|1103|(21,)|var|int8
var_1104 = relay.var("var_1104", dtype = "int16", shape = (44, 1))#candidate|1104|(44, 1)|var|int16
var_1105 = relay.var("var_1105", dtype = "int16", shape = (44,))#candidate|1105|(44,)|var|int16
call_1099 = func_1098_call(var_1100,var_1101,var_1102,var_1103,var_1104,var_1105,)
output = call_1099
func_1106 = relay.Function([var_1100,var_1101,var_1102,var_1103,var_1104,var_1105,], output)
mutated_mod['func_1106'] = func_1106
mutated_mod = relay.transform.InferType()(mutated_mod)
func_835_call = mod.get_global_var('func_835')
func_836_call = mutated_mod.get_global_var('func_836')
call_1144 = func_835_call()
call_1145 = func_835_call()
const_1153 = relay.const([True,True,True,False,True,False,True], dtype = "bool")#candidate|1153|(7,)|const|bool
bop_1154 = relay.mod(call_1144.astype('float64'), relay.reshape(const_1153.astype('float64'), relay.shape_of(call_1144))) # shape=(7,)
bop_1157 = relay.mod(call_1145.astype('float64'), relay.reshape(const_1153.astype('float64'), relay.shape_of(call_1145))) # shape=(7,)
bop_1161 = relay.left_shift(call_1144.astype('uint64'), relay.reshape(const_1153.astype('uint64'), relay.shape_of(call_1144))) # shape=(7,)
bop_1164 = relay.left_shift(call_1145.astype('uint64'), relay.reshape(const_1153.astype('uint64'), relay.shape_of(call_1145))) # shape=(7,)
output = relay.Tuple([bop_1154,bop_1161,])
output2 = relay.Tuple([bop_1157,bop_1164,])
func_1166 = relay.Function([], output)
mod['func_1166'] = func_1166
mod = relay.transform.InferType()(mod)
output = func_1166()
func_1167 = relay.Function([], output)
mutated_mod['func_1167'] = func_1167
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1178 = relay.var("var_1178", dtype = "float64", shape = (6, 6))#candidate|1178|(6, 6)|var|float64
uop_1179 = relay.sin(var_1178.astype('float64')) # shape=(6, 6)
const_1181 = relay.const([[0.591116,5.596263,-8.229338,-4.524915,2.910678,6.280103],[-0.250708,1.527366,-1.163974,-2.954385,-4.577973,4.549722],[-6.253894,-6.973546,-8.882915,-3.909063,7.673108,6.418631],[-3.152345,4.446483,4.082062,-6.179553,-5.791732,-4.933037],[-3.674081,0.725050,-8.045285,-6.419532,5.479526,0.734156],[-4.067106,-2.422339,3.064942,5.136233,6.411232,5.017453]], dtype = "float64")#candidate|1181|(6, 6)|const|float64
bop_1182 = relay.bitwise_xor(uop_1179.astype('int32'), relay.reshape(const_1181.astype('int32'), relay.shape_of(uop_1179))) # shape=(6, 6)
bop_1185 = relay.less(var_1178.astype('bool'), relay.reshape(uop_1179.astype('bool'), relay.shape_of(var_1178))) # shape=(6, 6)
uop_1191 = relay.sqrt(bop_1182.astype('float64')) # shape=(6, 6)
bop_1194 = relay.not_equal(uop_1191.astype('bool'), relay.reshape(var_1178.astype('bool'), relay.shape_of(uop_1191))) # shape=(6, 6)
output = relay.Tuple([bop_1185,bop_1194,])
output2 = relay.Tuple([bop_1185,bop_1194,])
func_1197 = relay.Function([var_1178,], output)
mod['func_1197'] = func_1197
mod = relay.transform.InferType()(mod)
mutated_mod['func_1197'] = func_1197
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1198 = relay.var("var_1198", dtype = "float64", shape = (6, 6))#candidate|1198|(6, 6)|var|float64
func_1197_call = mutated_mod.get_global_var('func_1197')
call_1199 = func_1197_call(var_1198)
output = call_1199
func_1200 = relay.Function([var_1198], output)
mutated_mod['func_1200'] = func_1200
mutated_mod = relay.transform.InferType()(mutated_mod)
func_892_call = mod.get_global_var('func_892')
func_893_call = mutated_mod.get_global_var('func_893')
call_1222 = relay.TupleGetItem(func_892_call(), 0)
call_1223 = relay.TupleGetItem(func_893_call(), 0)
func_105_call = mod.get_global_var('func_105')
func_108_call = mutated_mod.get_global_var('func_108')
var_1226 = relay.var("var_1226", dtype = "int16", shape = ())#candidate|1226|()|var|int16
const_1227 = relay.const([-6,-9,8,-4,10,-2,-9,-8,-5,-8,-4,1,-7,8,-10,9,4,10,-9,5,-6,-4,9,-4,2,5,9,3,-3,-5,6,8,4,-3,2,4,-4,5,-8,-8,-2,-6,9,1], dtype = "int16")#candidate|1227|(44,)|const|int16
call_1225 = func_105_call(relay.reshape(var_1226.astype('int16'), []), relay.reshape(const_1227.astype('int16'), [4, 11]), )
call_1228 = func_105_call(relay.reshape(var_1226.astype('int16'), []), relay.reshape(const_1227.astype('int16'), [4, 11]), )
output = relay.Tuple([call_1222,call_1225,var_1226,const_1227,])
output2 = relay.Tuple([call_1223,call_1228,var_1226,const_1227,])
func_1234 = relay.Function([var_1226,], output)
mod['func_1234'] = func_1234
mod = relay.transform.InferType()(mod)
var_1235 = relay.var("var_1235", dtype = "int16", shape = ())#candidate|1235|()|var|int16
output = func_1234(var_1235)
func_1236 = relay.Function([var_1235], output)
mutated_mod['func_1236'] = func_1236
mutated_mod = relay.transform.InferType()(mutated_mod)
func_502_call = mod.get_global_var('func_502')
func_503_call = mutated_mod.get_global_var('func_503')
call_1241 = func_502_call()
call_1242 = func_502_call()
output = call_1241
output2 = call_1242
func_1243 = relay.Function([], output)
mod['func_1243'] = func_1243
mod = relay.transform.InferType()(mod)
mutated_mod['func_1243'] = func_1243
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1243_call = mutated_mod.get_global_var('func_1243')
call_1244 = func_1243_call()
output = call_1244
func_1245 = relay.Function([], output)
mutated_mod['func_1245'] = func_1245
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1284 = relay.const([[[10,1,6,10,-2,-2,-1,6,8,10,-8,9,3,-9],[-8,6,2,9,-1,-4,7,7,10,-9,-9,-10,7,-9],[3,-9,2,8,5,3,-8,1,5,-3,-1,-5,3,-7],[5,1,3,-1,-5,8,6,-8,-2,-10,-6,4,-5,-8],[-4,-9,1,7,7,-1,-6,-4,2,-7,-10,-3,-9,-4],[4,1,6,-7,9,3,1,-8,-2,-7,6,-3,2,3],[-1,-6,-6,4,-6,-9,4,-10,-7,-2,-5,-5,5,-5],[4,-8,2,1,-10,2,-8,2,-4,-9,-5,9,-9,8],[4,1,-3,2,-3,-5,-10,3,-9,-1,9,5,-10,3],[1,-3,9,5,-5,7,-4,4,-3,-3,2,-7,-6,5],[9,3,-1,8,-2,-2,-2,8,-5,-9,7,-1,7,-10]],[[-4,-7,2,4,-8,-5,-7,-9,8,-4,-8,-1,4,6],[2,4,-8,9,-3,-7,-9,-3,-1,-5,10,4,-2,-6],[9,-10,5,-6,-5,5,2,-10,6,7,-6,8,2,7],[-9,10,3,8,4,-3,1,-8,1,3,-9,-5,-3,-4],[-7,8,-4,-1,-10,6,-2,-2,6,5,-8,-3,9,4],[2,-6,1,-2,-4,-3,-1,8,-8,6,2,1,8,-7],[-5,10,2,-4,-1,9,9,-10,3,-4,2,-6,1,-2],[-5,-7,-1,8,2,6,10,9,-9,2,10,-4,-5,-5],[-10,-5,-10,-7,-7,4,-7,-3,8,-6,2,9,-3,-4],[-3,-5,8,10,-4,8,-2,-3,7,-3,9,-1,7,-2],[-5,9,5,-3,-7,10,-9,-3,1,9,-1,2,-4,-10]],[[7,1,8,9,7,-6,-6,4,-5,8,2,-3,-2,2],[1,6,5,9,-9,-9,-10,-9,-3,2,1,-3,7,-7],[-2,4,-2,-1,-10,-2,-9,-5,8,-7,-8,-7,3,1],[8,1,-8,5,-7,-7,-7,8,-10,8,9,-8,-8,-8],[-7,-1,9,-6,8,5,8,-4,2,4,7,1,1,4],[6,-10,-3,9,4,-8,-9,-8,6,-3,-4,-2,10,8],[5,-3,-6,-4,-6,2,10,-10,5,-5,10,1,6,-4],[9,10,-3,-2,7,-1,-1,-1,-5,2,-1,-2,7,-4],[9,1,-9,9,-5,-5,9,3,1,4,9,5,-3,2],[10,-2,3,6,-6,7,6,-2,-8,-8,1,2,-10,-1],[-3,2,8,-4,-7,4,-3,-5,-3,-7,-1,9,-9,-3]],[[9,1,-4,9,6,2,2,-9,7,8,-8,-8,-2,-3],[9,9,8,1,-2,6,4,3,4,-2,5,-8,5,-4],[-4,-7,4,2,-7,-6,3,-8,9,-2,-5,-1,-2,-5],[-4,3,7,2,9,3,-9,4,-2,1,-6,-2,7,3],[-2,-8,9,-6,-3,-8,-4,-8,-4,5,5,-5,8,3],[-3,-5,-5,1,4,10,8,2,1,-5,2,-10,5,-1],[-5,6,-1,10,4,7,3,-9,-7,4,-3,-4,-8,-4],[-10,-4,1,10,3,-7,-3,6,4,-6,-8,-9,-9,6],[9,-8,8,3,-6,-9,-5,5,-1,-6,8,-5,2,9],[-1,9,-8,-9,1,-4,-5,5,1,9,-2,-6,7,-9],[9,2,-5,-8,-8,-7,-6,-5,6,-4,9,-9,-8,-1]],[[8,8,8,-5,-1,-3,-9,7,3,-10,-3,4,-8,10],[9,5,-1,-5,2,-1,-5,1,7,1,1,9,-6,5],[7,3,9,3,10,-3,10,-5,-10,-10,6,9,5,-1],[-2,-6,-4,-4,7,-6,-5,-9,-3,-1,1,-4,-3,4],[10,10,6,2,4,-4,5,-10,-9,7,-5,8,8,-5],[-2,8,6,5,-3,-1,-5,2,9,10,2,1,-6,5],[1,-6,9,-7,3,9,1,4,9,7,7,5,7,-7],[4,-8,10,-8,-9,7,10,-7,-8,-8,7,2,-3,3],[2,4,2,3,-7,2,-7,-7,7,-8,6,-9,-1,7],[9,-3,-2,8,-8,-5,-1,2,-7,7,-4,-6,-2,3],[7,9,-2,8,-9,5,5,-4,-6,-6,-5,5,-9,2]],[[-8,-5,7,-3,-6,2,-7,-5,-8,1,-5,-8,-2,-4],[6,2,-10,-2,-1,-7,-6,-2,-7,-7,8,-10,-7,-5],[-1,-9,4,-2,-10,-3,-9,-8,1,-10,-1,-3,5,1],[-2,-10,-7,2,4,-8,-8,3,-3,-2,5,3,9,2],[-9,-5,-1,-4,7,10,7,4,-10,3,-2,-8,-9,2],[6,1,8,4,4,-8,3,-2,9,-8,-5,-2,7,3],[-10,-5,10,2,3,-7,-4,-10,6,-9,-8,-4,7,-1],[8,10,10,4,3,-7,-4,5,-9,4,5,-10,10,7],[10,2,1,-7,7,-10,-9,-9,1,-10,-9,8,10,8],[-2,5,-2,-1,9,5,-1,-8,-3,-1,3,5,-8,7],[-10,5,-1,-7,7,7,6,-4,8,-5,-1,3,10,7]],[[2,-3,10,-6,10,3,1,-7,-1,-4,-5,9,-9,-3],[2,-6,-5,-9,-1,-5,8,10,9,8,10,-10,-5,4],[-1,3,-6,-10,-10,-5,3,1,9,3,-9,-9,-5,4],[8,-3,5,-4,5,-2,-2,5,7,7,7,6,-2,-5],[-4,8,8,-10,-7,4,10,-6,-2,-6,-7,9,-2,10],[-9,-3,-5,-2,-4,8,9,-3,7,-9,-2,-2,-9,-3],[-3,7,9,3,-2,-8,9,8,-5,9,2,-5,-5,3],[5,5,10,-9,3,-4,-1,3,-6,10,-2,5,8,-8],[4,-7,1,8,5,10,-1,6,7,-4,10,4,4,8],[9,5,9,-4,-7,2,10,-1,-8,-5,-6,-6,2,10],[9,-9,6,-2,7,5,-8,5,9,-1,3,-10,4,7]],[[5,-3,7,-3,-6,-8,-10,-7,8,-8,-10,-3,-4,3],[-8,3,10,-10,-8,2,7,-8,-4,7,9,-10,4,-1],[8,-6,4,2,7,-3,-4,-10,-6,9,-5,2,2,2],[-7,6,-7,-2,6,-5,5,-10,4,9,10,9,-4,9],[5,6,4,-7,-6,10,-2,-5,-4,5,6,-7,2,4],[-9,2,4,-5,-9,7,-1,5,3,-10,6,-4,-9,-1],[-9,7,-7,3,6,-8,-2,2,4,-7,-9,4,2,8],[-1,2,-3,9,7,-1,-1,2,4,7,1,-4,10,-8],[8,-8,6,-10,-1,10,2,-3,-10,6,-10,3,6,-4],[5,-1,-9,-3,-8,-5,-5,-7,8,-2,8,4,1,-9],[-5,5,6,-5,4,-9,3,1,1,-5,-9,-7,6,-5]],[[9,4,-9,-3,5,-2,-8,-1,-7,8,-10,6,1,8],[-7,4,7,9,9,4,2,4,7,10,6,1,-5,7],[10,-5,8,-2,-1,4,-5,-5,2,-8,5,-2,-9,8],[10,10,-1,-7,-6,6,10,-9,-8,-5,-6,-1,10,-9],[-8,3,-4,-8,-9,6,3,-7,1,1,10,2,-8,4],[-8,-9,2,4,6,-1,-6,4,-8,-6,5,-10,3,3],[-5,-9,4,9,9,-1,5,10,10,5,-3,9,-8,4],[-1,3,6,8,10,-3,1,4,10,7,7,8,-5,-9],[5,6,-10,9,3,-5,5,3,-4,1,-10,5,-2,-9],[-9,-10,4,10,5,2,-9,2,-5,-1,2,4,-10,-1],[-2,-9,-1,-9,5,10,1,7,-7,5,5,-2,5,9]],[[6,8,-2,-1,6,-7,9,8,-7,1,7,7,-1,-9],[6,-4,7,2,-8,4,5,4,9,4,-8,-2,-3,7],[10,10,3,10,-7,1,5,1,7,4,7,1,-4,-2],[-4,1,-4,-1,5,1,2,-9,-7,7,-4,-9,-8,6],[-3,-9,9,10,10,-2,-3,-9,5,3,8,6,-2,3],[-5,-4,-9,-5,2,2,9,1,7,10,-9,5,-5,-9],[8,-7,-3,1,-5,-10,6,-10,-8,-8,2,8,-2,-9],[4,6,-4,-4,5,8,-4,6,10,4,7,-2,1,-2],[-3,-10,1,-8,8,-1,-7,-2,7,10,3,5,7,1],[-5,1,-4,-4,8,1,4,-10,10,2,7,3,8,-2],[-5,-3,7,-10,-4,7,6,9,-9,3,-8,-7,8,6]]], dtype = "uint32")#candidate|1284|(10, 11, 14)|const|uint32
const_1285 = relay.const([[[9,-4,5,1,-4,2,-1,7,2,10,-1,-8,8,-6],[-10,6,6,-4,-5,8,8,-1,4,10,8,-7,-10,6],[8,-6,-8,-10,2,3,7,-9,3,5,-8,7,-5,-4],[9,5,-5,8,1,6,-9,1,2,2,-9,-7,2,-2],[-8,1,6,7,-5,5,-2,-1,9,1,-3,-9,10,1],[9,6,-10,5,5,9,8,7,-2,2,-4,5,4,2],[-9,5,-6,5,9,4,-1,-3,-5,-8,-1,-10,-3,-4],[10,-2,-1,-7,7,-5,-2,-8,1,2,4,-5,-2,9],[2,4,-1,1,3,-1,9,-7,-2,-2,6,9,7,4],[10,5,-9,4,-5,-4,-2,-7,8,2,8,-4,-9,6],[-8,-5,6,-2,1,6,2,2,-4,7,-4,-3,2,-9]],[[-2,-8,7,-1,-5,9,9,-2,5,7,7,-3,-2,3],[6,6,9,10,-7,-9,4,-10,3,-5,6,-8,-7,-3],[-7,-2,10,-3,-2,-7,4,3,5,-2,-5,-7,-1,5],[-7,-7,-5,-9,9,-6,-7,-9,-8,-10,6,-1,10,-2],[-4,1,3,8,-8,-5,-1,7,-1,-10,9,-1,-3,-3],[-10,-6,-2,-8,6,5,2,-3,10,-10,4,-1,9,7],[-4,-5,-7,-9,-9,-6,-2,10,-2,7,-1,-2,-7,10],[-4,-8,-7,-3,3,3,-10,6,1,-5,9,2,1,6],[8,-5,-3,9,5,6,3,-9,-10,4,-9,2,-3,7],[-2,-3,9,3,1,3,3,6,-9,-10,8,-10,-4,-6],[-1,-10,-7,3,1,9,-9,1,-2,-9,-10,1,4,-6]],[[6,-5,4,-1,-10,-9,2,4,10,3,-2,2,1,9],[-5,-4,-4,9,4,-1,6,-5,7,8,-7,-9,1,-3],[7,10,7,-4,-2,-1,4,-3,1,7,-1,5,-5,-9],[6,-1,-3,6,6,1,6,6,-5,10,-1,3,-3,-2],[-1,-7,4,-1,-10,7,10,5,1,5,2,-1,2,7],[-10,-4,-1,-1,10,1,10,5,7,5,-1,5,4,8],[-2,1,-3,5,5,-6,10,2,-7,-2,-9,6,5,-5],[9,-5,10,3,1,7,5,-9,-3,7,-2,-8,-3,-10],[6,-9,1,-6,-10,-4,10,-6,-5,2,-10,9,7,2],[2,10,9,2,3,-3,-10,5,8,-7,-10,2,9,1],[-9,-5,5,8,5,6,-2,1,7,4,-9,8,-1,9]],[[-2,-5,1,1,-5,2,8,8,-8,-2,6,10,-8,-7],[1,-6,-6,3,10,5,-4,5,-8,2,3,1,-1,-7],[1,7,-1,4,-3,-7,2,10,3,-8,8,7,7,3],[8,8,-5,1,7,5,2,-4,-4,-10,10,5,9,1],[1,3,-5,1,5,8,-7,3,3,-6,2,3,6,-9],[-2,-3,4,4,-9,9,-7,6,-10,-3,-4,9,7,-7],[-3,-9,7,-10,10,-10,9,2,8,7,-7,-10,-3,7],[2,4,10,-4,9,-10,8,-7,1,10,-5,5,1,-3],[4,-6,-7,6,6,-6,4,4,-6,-10,8,6,10,7],[9,-6,1,-1,-4,6,7,-5,-8,4,9,2,-5,-3],[10,-6,10,-1,-6,1,-1,10,-1,-10,3,4,9,9]],[[6,9,-2,4,9,-4,2,-2,2,-1,-6,-3,-5,3],[5,-8,-4,7,-3,9,-10,6,-8,-5,2,-5,-4,-9],[9,5,-6,7,9,7,6,-4,6,-10,-10,-7,10,-1],[2,5,4,5,2,-9,5,5,-1,-2,8,-3,7,-1],[2,3,-5,-5,-8,-8,-2,3,2,-3,-6,-2,-6,-5],[3,-2,-9,6,3,3,-2,2,-10,3,3,-7,-2,-5],[1,3,-5,8,-10,-8,-6,6,3,-3,-5,5,6,-8],[8,-7,1,-3,-9,-2,5,-3,2,2,7,8,-6,1],[4,10,-1,1,1,2,-6,8,-4,-6,-10,4,4,3],[-5,-1,4,-1,10,-8,-9,-8,5,-6,-9,-7,-1,-8],[3,4,-6,6,-8,-4,10,-7,4,-4,3,8,-4,-1]],[[-8,-9,-2,10,10,9,-5,-6,7,-4,1,10,7,8],[-6,6,1,4,8,3,-10,-8,5,9,-9,5,-10,6],[4,1,-4,-4,2,-8,2,10,4,-8,-4,-3,10,5],[-2,9,6,9,-7,-2,-3,-9,2,8,7,-6,-2,4],[-6,4,-1,-7,-3,6,9,8,8,6,7,7,9,5],[-2,-5,-9,8,10,-8,4,-4,1,3,-1,1,-5,4],[-2,5,-1,2,-9,-1,-8,3,-2,-4,-3,4,8,-4],[4,6,10,-9,2,-10,-4,2,-7,-3,-4,-1,4,2],[3,-4,7,-4,-1,-5,-6,6,5,1,2,2,3,-4],[-1,10,4,2,3,-8,8,-7,-10,2,10,-7,1,8],[-3,4,1,-7,-9,-8,-4,7,2,4,-9,-7,-4,-1]],[[6,-1,-4,-4,-1,1,1,-2,-6,4,-5,1,-9,-2],[7,-7,7,-1,-3,7,7,3,7,-2,6,-4,6,-6],[-6,-7,-6,-5,2,-3,2,-8,7,-6,9,-9,3,8],[9,-4,4,7,8,2,-7,-7,-4,7,-8,-8,-8,-4],[10,-10,-9,5,8,5,3,7,2,-10,-9,-8,9,-6],[2,-8,3,5,-4,5,-2,1,-8,5,10,-6,2,8],[9,8,3,8,-6,7,-1,6,-5,-9,3,2,-1,4],[2,7,-5,7,-3,-3,-7,-10,7,4,-3,4,10,1],[4,-5,7,-2,-6,-10,5,9,6,-4,5,6,2,-6],[-5,4,-4,8,2,4,8,-9,7,-6,-7,-9,-8,-2],[2,5,-7,-9,-9,-2,8,8,-10,-10,7,4,-6,-6]],[[-6,7,6,-9,10,-2,6,7,-10,9,-6,-8,7,-4],[-3,-9,9,-8,7,-8,5,5,5,-3,-1,-6,-5,9],[-4,-7,5,7,4,-1,-3,10,-10,-9,-9,2,7,8],[-2,-4,-1,-5,-6,2,-8,-1,-1,-8,7,8,10,-7],[-5,10,1,-4,-6,-9,-1,-4,-4,-9,7,-8,5,-10],[3,3,-2,7,-9,-8,-2,-9,9,-2,-8,10,8,-8],[1,7,5,3,-5,3,-3,-8,9,6,9,-1,-8,-2],[8,-9,-8,-2,-8,-2,-5,-6,-7,3,-2,-6,-8,-9],[7,-3,1,-3,-8,-4,9,-9,5,5,10,10,9,-6],[-2,-3,8,3,8,10,2,-4,2,-7,1,-10,2,-4],[3,-7,-6,-3,6,6,5,-4,2,8,-3,10,1,-6]],[[10,-3,1,8,-10,-4,9,2,7,-1,-6,-2,5,5],[-4,8,-1,6,1,-1,-7,9,6,9,9,-2,-8,4],[-8,1,7,5,10,-10,-7,4,-8,9,-6,7,10,-10],[4,4,-4,6,-4,-1,-4,7,-4,5,-4,3,-5,-5],[2,5,-1,9,-6,7,8,6,9,-5,-4,4,-7,10],[-7,-8,10,6,6,9,2,4,3,2,-3,6,9,5],[-7,9,10,5,-5,-3,8,-3,-7,-8,8,6,-9,7],[2,-4,-7,-5,10,-6,2,-6,-1,-9,-7,-1,-1,4],[-9,7,-5,2,-4,6,8,-1,-1,-3,5,3,10,-8],[-10,10,5,1,-10,1,-2,10,-9,2,-5,2,-6,-3],[-8,-9,5,-8,-9,-10,3,-3,-8,-7,-1,7,-4,7]],[[6,8,2,4,7,10,4,4,-3,2,-9,8,-8,9],[8,3,7,9,9,7,-9,1,-6,1,-2,-1,-4,5],[8,5,10,-9,9,6,-5,-3,1,8,-7,-9,3,-3],[2,8,-10,5,1,6,-6,-8,-7,-5,-5,8,-3,-4],[-8,3,3,-8,-4,-2,9,-7,-10,5,-7,4,5,-5],[-8,7,4,-3,-9,9,-6,-8,-8,-9,9,-4,6,8],[9,-2,-2,2,-8,2,-8,3,-1,10,-8,-6,4,-3],[7,2,7,-6,4,3,-4,1,6,-7,-7,-4,7,3],[-9,4,10,-5,-1,-5,8,-8,5,-4,5,5,10,8],[1,-3,-8,10,2,-6,-8,-7,-9,9,-4,4,-6,-3],[10,8,-5,-4,7,-2,1,9,6,-6,-8,-10,9,4]]], dtype = "uint32")#candidate|1285|(10, 11, 14)|const|uint32
bop_1286 = relay.multiply(const_1284.astype('uint32'), relay.reshape(const_1285.astype('uint32'), relay.shape_of(const_1284))) # shape=(10, 11, 14)
output = relay.Tuple([bop_1286,])
output2 = relay.Tuple([bop_1286,])
func_1289 = relay.Function([], output)
mod['func_1289'] = func_1289
mod = relay.transform.InferType()(mod)
output = func_1289()
func_1290 = relay.Function([], output)
mutated_mod['func_1290'] = func_1290
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1316 = relay.var("var_1316", dtype = "float64", shape = (4, 7))#candidate|1316|(4, 7)|var|float64
uop_1317 = relay.tan(var_1316.astype('float64')) # shape=(4, 7)
bop_1328 = relay.greater(uop_1317.astype('bool'), relay.reshape(var_1316.astype('bool'), relay.shape_of(uop_1317))) # shape=(4, 7)
bop_1332 = relay.bitwise_or(bop_1328.astype('uint32'), relay.reshape(var_1316.astype('uint32'), relay.shape_of(bop_1328))) # shape=(4, 7)
var_1338 = relay.var("var_1338", dtype = "bool", shape = (4, 7))#candidate|1338|(4, 7)|var|bool
bop_1339 = relay.logical_and(bop_1328.astype('bool'), relay.reshape(var_1338.astype('bool'), relay.shape_of(bop_1328))) # shape=(4, 7)
bop_1342 = relay.mod(var_1338.astype('float64'), relay.reshape(uop_1317.astype('float64'), relay.shape_of(var_1338))) # shape=(4, 7)
bop_1346 = relay.less_equal(var_1338.astype('bool'), relay.reshape(bop_1328.astype('bool'), relay.shape_of(var_1338))) # shape=(4, 7)
uop_1350 = relay.tan(bop_1342.astype('float32')) # shape=(4, 7)
bop_1353 = relay.less(uop_1350.astype('bool'), relay.reshape(bop_1328.astype('bool'), relay.shape_of(uop_1350))) # shape=(4, 7)
uop_1361 = relay.cosh(bop_1353.astype('float32')) # shape=(4, 7)
output = relay.Tuple([bop_1332,bop_1339,bop_1346,uop_1361,])
output2 = relay.Tuple([bop_1332,bop_1339,bop_1346,uop_1361,])
func_1363 = relay.Function([var_1316,var_1338,], output)
mod['func_1363'] = func_1363
mod = relay.transform.InferType()(mod)
var_1364 = relay.var("var_1364", dtype = "float64", shape = (4, 7))#candidate|1364|(4, 7)|var|float64
var_1365 = relay.var("var_1365", dtype = "bool", shape = (4, 7))#candidate|1365|(4, 7)|var|bool
output = func_1363(var_1364,var_1365,)
func_1366 = relay.Function([var_1364,var_1365,], output)
mutated_mod['func_1366'] = func_1366
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1289_call = mod.get_global_var('func_1289')
func_1290_call = mutated_mod.get_global_var('func_1290')
call_1373 = relay.TupleGetItem(func_1289_call(), 0)
call_1374 = relay.TupleGetItem(func_1290_call(), 0)
output = call_1373
output2 = call_1374
func_1387 = relay.Function([], output)
mod['func_1387'] = func_1387
mod = relay.transform.InferType()(mod)
output = func_1387()
func_1388 = relay.Function([], output)
mutated_mod['func_1388'] = func_1388
mutated_mod = relay.transform.InferType()(mutated_mod)
func_557_call = mod.get_global_var('func_557')
func_558_call = mutated_mod.get_global_var('func_558')
call_1411 = relay.TupleGetItem(func_557_call(), 1)
call_1412 = relay.TupleGetItem(func_558_call(), 1)
uop_1415 = relay.tan(call_1411.astype('float64')) # shape=(16, 11, 1)
uop_1417 = relay.tan(call_1412.astype('float64')) # shape=(16, 11, 1)
uop_1421 = relay.log10(uop_1415.astype('float64')) # shape=(16, 11, 1)
uop_1423 = relay.log10(uop_1417.astype('float64')) # shape=(16, 11, 1)
uop_1425 = relay.log(uop_1421.astype('float64')) # shape=(16, 11, 1)
uop_1427 = relay.log(uop_1423.astype('float64')) # shape=(16, 11, 1)
uop_1428 = relay.acosh(uop_1421.astype('float32')) # shape=(16, 11, 1)
uop_1430 = relay.acosh(uop_1423.astype('float32')) # shape=(16, 11, 1)
func_1387_call = mod.get_global_var('func_1387')
func_1388_call = mutated_mod.get_global_var('func_1388')
call_1435 = func_1387_call()
call_1436 = func_1387_call()
var_1439 = relay.var("var_1439", dtype = "float64", shape = (16, 11, 7))#candidate|1439|(16, 11, 7)|var|float64
bop_1440 = relay.bitwise_or(uop_1425.astype('int16'), var_1439.astype('int16')) # shape=(16, 11, 7)
bop_1443 = relay.bitwise_or(uop_1427.astype('int16'), var_1439.astype('int16')) # shape=(16, 11, 7)
bop_1447 = relay.floor_mod(uop_1415.astype('float64'), relay.reshape(uop_1428.astype('float64'), relay.shape_of(uop_1415))) # shape=(16, 11, 1)
bop_1450 = relay.floor_mod(uop_1417.astype('float64'), relay.reshape(uop_1430.astype('float64'), relay.shape_of(uop_1417))) # shape=(16, 11, 1)
bop_1453 = relay.divide(uop_1428.astype('float64'), var_1439.astype('float64')) # shape=(16, 11, 7)
bop_1456 = relay.divide(uop_1430.astype('float64'), var_1439.astype('float64')) # shape=(16, 11, 7)
func_149_call = mod.get_global_var('func_149')
func_152_call = mutated_mod.get_global_var('func_152')
const_1458 = relay.const([-2,-1,-9,-9,-8,-6,-1,-5,-3,3,3,-5,-3,-9,-10,9,1,10,-4,2,-8,-9,5,7,9,1,-10,-1,3,-5,6,-7,-5,-8,-10,1,1,6,3,5,3,-6,-4,7,-6,-2,-8,1,8,4,-10,-5,-6,8,-4,3,-10,-2,7,4,2,8,-7,-5,8,2,-6,-10,7,2,-10,-1,-6,-7,-2,8,6,-6,-3,-7,-2,-1,-5,-1,-1,-6,-9,-3,9,10,10,-5,-2,-1,-4,7,7,-4,-1,9,-7,-4,1,3,-9,1,-9,-8,-9,-7,9,-1,-6,-1,10,-10,4,-5,-3,-6], dtype = "int8")#candidate|1458|(120,)|const|int8
call_1457 = func_149_call(relay.reshape(const_1458.astype('int8'), [3, 10, 4]), relay.reshape(const_1458.astype('int8'), [3, 10, 4]), )
call_1459 = func_149_call(relay.reshape(const_1458.astype('int8'), [3, 10, 4]), relay.reshape(const_1458.astype('int8'), [3, 10, 4]), )
output = relay.Tuple([call_1435,bop_1440,bop_1447,bop_1453,call_1457,const_1458,])
output2 = relay.Tuple([call_1436,bop_1443,bop_1450,bop_1456,call_1459,const_1458,])
func_1460 = relay.Function([var_1439,], output)
mod['func_1460'] = func_1460
mod = relay.transform.InferType()(mod)
var_1461 = relay.var("var_1461", dtype = "float64", shape = (16, 11, 7))#candidate|1461|(16, 11, 7)|var|float64
output = func_1460(var_1461)
func_1462 = relay.Function([var_1461], output)
mutated_mod['func_1462'] = func_1462
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1243_call = mod.get_global_var('func_1243')
func_1245_call = mutated_mod.get_global_var('func_1245')
call_1501 = func_1243_call()
call_1502 = func_1243_call()
output = call_1501
output2 = call_1502
func_1526 = relay.Function([], output)
mod['func_1526'] = func_1526
mod = relay.transform.InferType()(mod)
output = func_1526()
func_1527 = relay.Function([], output)
mutated_mod['func_1527'] = func_1527
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1166_call = mod.get_global_var('func_1166')
func_1167_call = mutated_mod.get_global_var('func_1167')
call_1554 = relay.TupleGetItem(func_1166_call(), 0)
call_1555 = relay.TupleGetItem(func_1167_call(), 0)
uop_1557 = relay.tan(call_1554.astype('float64')) # shape=(7,)
uop_1559 = relay.tan(call_1555.astype('float64')) # shape=(7,)
bop_1564 = relay.greater_equal(uop_1557.astype('bool'), relay.reshape(call_1554.astype('bool'), relay.shape_of(uop_1557))) # shape=(7,)
bop_1567 = relay.greater_equal(uop_1559.astype('bool'), relay.reshape(call_1555.astype('bool'), relay.shape_of(uop_1559))) # shape=(7,)
bop_1568 = relay.floor_mod(call_1554.astype('float64'), relay.reshape(uop_1557.astype('float64'), relay.shape_of(call_1554))) # shape=(7,)
bop_1571 = relay.floor_mod(call_1555.astype('float64'), relay.reshape(uop_1559.astype('float64'), relay.shape_of(call_1555))) # shape=(7,)
uop_1580 = relay.log10(bop_1568.astype('float32')) # shape=(7,)
uop_1582 = relay.log10(bop_1571.astype('float32')) # shape=(7,)
uop_1585 = relay.acos(call_1554.astype('float32')) # shape=(7,)
uop_1587 = relay.acos(call_1555.astype('float32')) # shape=(7,)
func_1166_call = mod.get_global_var('func_1166')
func_1167_call = mutated_mod.get_global_var('func_1167')
call_1592 = relay.TupleGetItem(func_1166_call(), 0)
call_1593 = relay.TupleGetItem(func_1167_call(), 0)
var_1596 = relay.var("var_1596", dtype = "float32", shape = (7,))#candidate|1596|(7,)|var|float32
bop_1597 = relay.multiply(uop_1580.astype('float32'), relay.reshape(var_1596.astype('float32'), relay.shape_of(uop_1580))) # shape=(7,)
bop_1600 = relay.multiply(uop_1582.astype('float32'), relay.reshape(var_1596.astype('float32'), relay.shape_of(uop_1582))) # shape=(7,)
func_1062_call = mod.get_global_var('func_1062')
func_1064_call = mutated_mod.get_global_var('func_1064')
var_1607 = relay.var("var_1607", dtype = "uint64", shape = (11, 3))#candidate|1607|(11, 3)|var|uint64
call_1606 = func_1062_call(relay.reshape(var_1607.astype('uint64'), [11, 3]))
call_1608 = func_1062_call(relay.reshape(var_1607.astype('uint64'), [11, 3]))
bop_1613 = relay.right_shift(uop_1580.astype('int16'), relay.reshape(var_1596.astype('int16'), relay.shape_of(uop_1580))) # shape=(7,)
bop_1616 = relay.right_shift(uop_1582.astype('int16'), relay.reshape(var_1596.astype('int16'), relay.shape_of(uop_1582))) # shape=(7,)
output = relay.Tuple([bop_1564,uop_1585,call_1592,bop_1597,call_1606,var_1607,bop_1613,])
output2 = relay.Tuple([bop_1567,uop_1587,call_1593,bop_1600,call_1608,var_1607,bop_1616,])
func_1617 = relay.Function([var_1596,var_1607,], output)
mod['func_1617'] = func_1617
mod = relay.transform.InferType()(mod)
var_1618 = relay.var("var_1618", dtype = "float32", shape = (7,))#candidate|1618|(7,)|var|float32
var_1619 = relay.var("var_1619", dtype = "uint64", shape = (11, 3))#candidate|1619|(11, 3)|var|uint64
output = func_1617(var_1618,var_1619,)
func_1620 = relay.Function([var_1618,var_1619,], output)
mutated_mod['func_1620'] = func_1620
mutated_mod = relay.transform.InferType()(mutated_mod)
func_927_call = mod.get_global_var('func_927')
func_929_call = mutated_mod.get_global_var('func_929')
call_1665 = relay.TupleGetItem(func_927_call(), 0)
call_1666 = relay.TupleGetItem(func_929_call(), 0)
output = call_1665
output2 = call_1666
func_1668 = relay.Function([], output)
mod['func_1668'] = func_1668
mod = relay.transform.InferType()(mod)
mutated_mod['func_1668'] = func_1668
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1668_call = mutated_mod.get_global_var('func_1668')
call_1669 = func_1668_call()
output = call_1669
func_1670 = relay.Function([], output)
mutated_mod['func_1670'] = func_1670
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1685 = relay.var("var_1685", dtype = "float32", shape = (5,))#candidate|1685|(5,)|var|float32
const_1686 = relay.const([6.917929,2.092171,-6.804694,1.640266,-5.868437], dtype = "float32")#candidate|1686|(5,)|const|float32
bop_1687 = relay.maximum(var_1685.astype('float32'), relay.reshape(const_1686.astype('float32'), relay.shape_of(var_1685))) # shape=(5,)
func_69_call = mod.get_global_var('func_69')
func_73_call = mutated_mod.get_global_var('func_73')
const_1693 = relay.const([5.273267,-3.411506,-0.223319,-3.057202,-3.983637,-3.437055,1.337263,-2.729829,6.608121,-6.968700,-8.188019,-4.783949,3.945385,6.271687,-1.768818,7.772908,4.956244,-5.200334,1.811314,-8.641477,6.639637,-0.341685,9.728430,-1.126567,-6.605508,9.103794,4.482299,9.695453,-3.486694,4.731536,5.897621,-4.381299,-6.528882,-7.476056,-4.635038,4.983327,5.869294,-7.539667,8.097959,-2.625001,1.676260,-8.514500,3.012049,6.833873,-0.421454,6.632415,-6.519376,-0.079310,-9.882231,-6.265428,-8.272969,-5.596386,1.746144,-0.865178,9.011423,-4.728055,8.806917,7.352773,0.960095,-0.661347,1.206160,5.748582,-1.231933,-3.488917,-5.661636,2.184918,-7.776524,8.675636,-7.953684,-0.583847,-6.163508,-9.165848,8.654049,-2.327575,8.642507,-1.491398,-4.393773,-0.100268,-3.416504,7.635431,4.799813,8.004130,-2.781513,-3.074172,-3.488157,-3.295051,7.441090,-7.122287,-1.396728,8.254217,-9.455202,-9.716360,2.031150,7.435223,4.858173,6.313605,1.876429,5.876077,2.720547,-5.566838,-6.282723,-7.636807,0.161475,3.288055,6.240082,6.090298,-5.343700,7.848892,-8.331701,7.112749,4.691350,0.781482,0.365832,1.472294,-8.576670,7.723363,-9.874426,-9.915529,-2.406483,-3.614438], dtype = "float64")#candidate|1693|(120,)|const|float64
const_1694 = relay.const(7.683932, dtype = "float64")#candidate|1694|()|const|float64
call_1692 = relay.TupleGetItem(func_69_call(relay.reshape(const_1693.astype('float64'), [12, 10]), relay.reshape(const_1694.astype('float64'), []), ), 3)
call_1695 = relay.TupleGetItem(func_73_call(relay.reshape(const_1693.astype('float64'), [12, 10]), relay.reshape(const_1694.astype('float64'), []), ), 3)
var_1698 = relay.var("var_1698", dtype = "float32", shape = (5,))#candidate|1698|(5,)|var|float32
bop_1699 = relay.equal(var_1685.astype('bool'), relay.reshape(var_1698.astype('bool'), relay.shape_of(var_1685))) # shape=(5,)
output = relay.Tuple([bop_1687,call_1692,const_1693,const_1694,bop_1699,])
output2 = relay.Tuple([bop_1687,call_1695,const_1693,const_1694,bop_1699,])
F = relay.Function([var_1685,var_1698,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1685,var_1698,], output2)
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
input_1685= np.array([3.428942,9.849181,-5.736655,8.967442,-1.392835], dtype='float32')
module1.set_input('var_1685', input_1685)
input_1698= np.array([-4.567964,9.720301,5.002336,9.469524,-2.306452], dtype='float32')
module1.set_input('var_1698', input_1698)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1685, input_1698, )
res3 = intrp3.evaluate()(input_1685, input_1698, )
res4 = intrp4.evaluate()(input_1685, input_1698, )
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
res1_4 = module1.get_output(4).asnumpy()
res2_4 = res2[4].asnumpy()
res3_4 = res3[4].asnumpy()
res4_4 = res4[4].asnumpy()
np.testing.assert_allclose(res1_4 ,res2_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_4 ,res3_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_4 ,res4_4, atol=1e-3, rtol=1e-3)
(res1_4 == res2_4).all()
(res1_4 == res3_4).all()
(res1_4 == res4_4).all()
module5.set_input('var_1685', input_1685)
module5.set_input('var_1698', input_1698)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1685, input_1698, )
res7 = intrp7.evaluate()(input_1685, input_1698, )
res8 = intrp8.evaluate()(input_1685, input_1698, )
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
res5_4 = module5.get_output(4).asnumpy()
res6_4 = res6[4].asnumpy()
res7_4 = res7[4].asnumpy()
res8_4 = res8[4].asnumpy()
np.testing.assert_allclose(res5_4 ,res6_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_4 ,res7_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_4 ,res8_4, atol=1e-3, rtol=1e-3)
(res5_4 == res6_4).all()
(res5_4 == res7_4).all()
(res5_4 == res8_4).all()
module9.set_input('var_1685', input_1685)
module9.set_input('var_1698', input_1698)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1685, input_1698, )
res11 = intrp11.evaluate()(input_1685, input_1698, )
res12 = intrp12.evaluate()(input_1685, input_1698, )
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
res9_4 = module9.get_output(4).asnumpy()
res10_4 = res10[4].asnumpy()
res11_4 = res11[4].asnumpy()
res12_4 = res12[4].asnumpy()
np.testing.assert_allclose(res9_4 ,res10_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_4 ,res11_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_4 ,res12_4, atol=1e-3, rtol=1e-3)
(res9_4 == res10_4).all()
(res9_4 == res11_4).all()
(res9_4 == res12_4).all()
module13.set_input('var_1685', input_1685)
module13.set_input('var_1698', input_1698)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1685, input_1698, )
res15 = intrp15.evaluate()(input_1685, input_1698, )
res16 = intrp16.evaluate()(input_1685, input_1698, )
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
res13_4 = module13.get_output(4).asnumpy()
res14_4 = res14[4].asnumpy()
res15_4 = res15[4].asnumpy()
res16_4 = res16[4].asnumpy()
np.testing.assert_allclose(res13_4 ,res14_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_4 ,res15_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_4 ,res16_4, atol=1e-3, rtol=1e-3)
(res13_4 == res14_4).all()
(res13_4 == res15_4).all()
(res13_4 == res16_4).all()
module17.set_input('var_1685', input_1685)
module17.set_input('var_1698', input_1698)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1685, input_1698, )
res19 = intrp19.evaluate()(input_1685, input_1698, )
res20 = intrp20.evaluate()(input_1685, input_1698, )
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
res17_4 = module17.get_output(4).asnumpy()
res18_4 = res18[4].asnumpy()
res19_4 = res19[4].asnumpy()
res20_4 = res20[4].asnumpy()
np.testing.assert_allclose(res17_4 ,res18_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_4 ,res19_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_4 ,res20_4, atol=1e-3, rtol=1e-3)
(res17_4 == res18_4).all()
(res17_4 == res19_4).all()
(res17_4 == res20_4).all()
module21.set_input('var_1685', input_1685)
module21.set_input('var_1698', input_1698)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1685, input_1698, )
res23 = intrp23.evaluate()(input_1685, input_1698, )
res24 = intrp24.evaluate()(input_1685, input_1698, )
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
res21_4 = module21.get_output(4).asnumpy()
res22_4 = res22[4].asnumpy()
res23_4 = res23[4].asnumpy()
res24_4 = res24[4].asnumpy()
np.testing.assert_allclose(res21_4 ,res22_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_4 ,res23_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_4 ,res24_4, atol=1e-3, rtol=1e-3)
(res21_4 == res22_4).all()
(res21_4 == res23_4).all()
(res21_4 == res24_4).all()

'''6: TVMFuncCall
5: _ZNSt17_Function_handlerIFvN3tvm7runtime7
4: tvm::runtime::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const [clone .isra.808]
3: tvm::runtime::GraphExecutorCreate(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module const&, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
2: tvm::runtime::GraphExecutor::Init(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
1: tvm::runtime::GraphExecutor::SetupOpExecs()
0: tvm::runtime::GraphExecutor::CreateTVMOp(tvm::runtime::TVMOpParam const&, std::vector<DLTensor, std::allocator<DLTensor> > const&)

'''