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
var_40 = relay.var("var_40", dtype = "float32", shape = ())#candidate|40|()|var|float32
var_41 = relay.var("var_41", dtype = "float32", shape = (1, 5, 4))#candidate|41|(1, 5, 4)|var|float32
bop_42 = relay.divide(var_40.astype('float32'), var_41.astype('float32')) # shape=(1, 5, 4)
uop_51 = relay.sigmoid(bop_42.astype('float32')) # shape=(1, 5, 4)
const_54 = relay.const([[[-5.839043,6.383135,-0.743586,-0.975430],[8.652859,6.741462,8.612809,-7.818889],[-2.250638,-2.034546,-7.765085,1.599711],[0.701122,-1.385742,-9.586611,3.185680],[1.991107,-2.193005,7.420282,-8.395201]],[[4.634564,-8.015831,-6.922549,-0.432859],[-7.753345,-8.408870,-3.173664,-0.098787],[8.441590,4.986639,-0.752840,-0.042896],[6.010029,4.956699,-3.803776,9.351502],[8.302224,-7.740942,7.426067,4.546817]]], dtype = "float32")#candidate|54|(2, 5, 4)|const|float32
bop_55 = relay.maximum(bop_42.astype('float64'), const_54.astype('float64')) # shape=(2, 5, 4)
const_60 = relay.const([[[0.163085,8.139957,-0.331113,-7.700740],[8.929827,1.158075,3.675843,-6.107986],[-2.423974,-0.499644,-5.834866,-9.150685],[6.565202,-4.793363,-2.109024,-1.814634],[4.664729,5.140727,-3.770829,7.175652]],[[-2.968932,6.068425,-9.086404,0.511649],[7.496793,6.207765,5.026974,-0.456277],[-3.851498,1.454557,-7.934241,4.177440],[2.719908,-0.124438,-4.604316,-2.119067],[-8.789612,-4.046913,5.663026,7.545826]],[[-5.063393,-4.159357,4.127542,-3.011749],[-3.194237,-3.779832,0.446243,-6.272299],[-7.430155,3.008627,8.256552,-1.807670],[0.567179,-5.164383,-4.757716,1.533369],[-8.686737,-4.800121,-0.969444,-2.340265]],[[-1.074726,-7.446108,-7.584206,3.172745],[7.555372,3.578648,-5.923635,-4.214321],[-6.157233,-9.757858,0.285971,2.163685],[5.951975,5.484895,9.024816,-8.214810],[-7.234567,8.646639,-3.087771,-1.842434]],[[0.610489,-5.986377,2.694272,-0.054440],[3.163075,-1.658629,-6.836729,-8.616558],[9.309767,-7.514569,9.049079,7.062757],[3.394833,-8.854862,2.252770,7.053985],[-3.762142,-0.728695,-3.480952,-2.557521]],[[4.238776,-0.236487,-1.007097,9.232527],[-7.657855,-5.293330,3.486014,-3.988043],[1.417564,-9.062588,1.613802,1.693465],[6.001103,8.542874,-2.555043,6.453097],[-0.291961,-2.956292,-3.193498,-4.489662]]], dtype = "float32")#candidate|60|(6, 5, 4)|const|float32
bop_61 = relay.logical_and(uop_51.astype('bool'), const_60.astype('bool')) # shape=(6, 5, 4)
uop_64 = relay.asinh(bop_61.astype('float32')) # shape=(6, 5, 4)
uop_69 = relay.exp(uop_51.astype('float64')) # shape=(1, 5, 4)
bop_76 = relay.bitwise_xor(uop_51.astype('int16'), const_60.astype('int16')) # shape=(6, 5, 4)
uop_79 = relay.exp(uop_64.astype('float64')) # shape=(6, 5, 4)
output = relay.Tuple([bop_55,uop_69,bop_76,uop_79,])
output2 = relay.Tuple([bop_55,uop_69,bop_76,uop_79,])
func_86 = relay.Function([var_40,var_41,], output)
mod['func_86'] = func_86
mod = relay.transform.InferType()(mod)
var_87 = relay.var("var_87", dtype = "float32", shape = ())#candidate|87|()|var|float32
var_88 = relay.var("var_88", dtype = "float32", shape = (1, 5, 4))#candidate|88|(1, 5, 4)|var|float32
output = func_86(var_87,var_88,)
func_89 = relay.Function([var_87,var_88,], output)
mutated_mod['func_89'] = func_89
mutated_mod = relay.transform.InferType()(mutated_mod)
var_91 = relay.var("var_91", dtype = "float64", shape = (2, 6, 3))#candidate|91|(2, 6, 3)|var|float64
uop_92 = relay.log(var_91.astype('float64')) # shape=(2, 6, 3)
bop_101 = relay.less_equal(var_91.astype('bool'), relay.reshape(uop_92.astype('bool'), relay.shape_of(var_91))) # shape=(2, 6, 3)
output = bop_101
output2 = bop_101
func_106 = relay.Function([var_91,], output)
mod['func_106'] = func_106
mod = relay.transform.InferType()(mod)
mutated_mod['func_106'] = func_106
mutated_mod = relay.transform.InferType()(mutated_mod)
var_107 = relay.var("var_107", dtype = "float64", shape = (2, 6, 3))#candidate|107|(2, 6, 3)|var|float64
func_106_call = mutated_mod.get_global_var('func_106')
call_108 = func_106_call(var_107)
output = call_108
func_109 = relay.Function([var_107], output)
mutated_mod['func_109'] = func_109
mutated_mod = relay.transform.InferType()(mutated_mod)
var_173 = relay.var("var_173", dtype = "float64", shape = (6, 9))#candidate|173|(6, 9)|var|float64
uop_174 = relay.sinh(var_173.astype('float64')) # shape=(6, 9)
func_106_call = mod.get_global_var('func_106')
func_109_call = mutated_mod.get_global_var('func_109')
var_179 = relay.var("var_179", dtype = "float64", shape = (36,))#candidate|179|(36,)|var|float64
call_178 = func_106_call(relay.reshape(var_179.astype('float64'), [2, 6, 3]))
call_180 = func_106_call(relay.reshape(var_179.astype('float64'), [2, 6, 3]))
output = relay.Tuple([uop_174,call_178,var_179,])
output2 = relay.Tuple([uop_174,call_180,var_179,])
func_190 = relay.Function([var_173,var_179,], output)
mod['func_190'] = func_190
mod = relay.transform.InferType()(mod)
mutated_mod['func_190'] = func_190
mutated_mod = relay.transform.InferType()(mutated_mod)
func_190_call = mutated_mod.get_global_var('func_190')
var_192 = relay.var("var_192", dtype = "float64", shape = (6, 9))#candidate|192|(6, 9)|var|float64
var_193 = relay.var("var_193", dtype = "float64", shape = (36,))#candidate|193|(36,)|var|float64
call_191 = func_190_call(var_192,var_193,)
output = call_191
func_194 = relay.Function([var_192,var_193,], output)
mutated_mod['func_194'] = func_194
mutated_mod = relay.transform.InferType()(mutated_mod)
var_240 = relay.var("var_240", dtype = "float64", shape = (7, 11))#candidate|240|(7, 11)|var|float64
uop_241 = relay.atan(var_240.astype('float64')) # shape=(7, 11)
uop_243 = relay.cosh(uop_241.astype('float64')) # shape=(7, 11)
func_106_call = mod.get_global_var('func_106')
func_109_call = mutated_mod.get_global_var('func_109')
var_248 = relay.var("var_248", dtype = "float64", shape = (36,))#candidate|248|(36,)|var|float64
call_247 = func_106_call(relay.reshape(var_248.astype('float64'), [2, 6, 3]))
call_249 = func_106_call(relay.reshape(var_248.astype('float64'), [2, 6, 3]))
func_190_call = mod.get_global_var('func_190')
func_194_call = mutated_mod.get_global_var('func_194')
const_254 = relay.const([8.145175,-7.273402,-9.716951,-1.616779,-0.458161,-7.818025,-0.658583,-7.680088,-8.251125,5.354412,-4.653405,6.220774,9.352579,-7.374848,-6.966601,0.218595,-0.205189,3.075248,-0.053386,8.280659,1.330094,5.498898,2.789183,9.135430,-0.274359,-3.275788,-9.577053,-9.681869,-5.714374,8.435481,-4.174286,-2.160982,-2.732017,-2.152954,-8.451582,-6.614822,8.327006,6.074025,1.282584,2.192197,-6.096337,-2.316752,6.313375,-9.573579,1.799351,-8.309633,4.105531,-0.765555,-7.882500,-8.017800,-1.265493,-8.283967,1.155154,5.482564], dtype = "float64")#candidate|254|(54,)|const|float64
call_253 = relay.TupleGetItem(func_190_call(relay.reshape(const_254.astype('float64'), [6, 9]), relay.reshape(call_247.astype('float64'), [36,]), ), 2)
call_255 = relay.TupleGetItem(func_194_call(relay.reshape(const_254.astype('float64'), [6, 9]), relay.reshape(call_247.astype('float64'), [36,]), ), 2)
output = relay.Tuple([uop_243,call_247,var_248,call_253,const_254,])
output2 = relay.Tuple([uop_243,call_249,var_248,call_255,const_254,])
func_257 = relay.Function([var_240,var_248,], output)
mod['func_257'] = func_257
mod = relay.transform.InferType()(mod)
mutated_mod['func_257'] = func_257
mutated_mod = relay.transform.InferType()(mutated_mod)
func_257_call = mutated_mod.get_global_var('func_257')
var_259 = relay.var("var_259", dtype = "float64", shape = (7, 11))#candidate|259|(7, 11)|var|float64
var_260 = relay.var("var_260", dtype = "float64", shape = (36,))#candidate|260|(36,)|var|float64
call_258 = func_257_call(var_259,var_260,)
output = call_258
func_261 = relay.Function([var_259,var_260,], output)
mutated_mod['func_261'] = func_261
mutated_mod = relay.transform.InferType()(mutated_mod)
const_374 = relay.const([[2,-9,2,-1,-3,-5,1],[7,-6,3,-1,-4,-2,-5],[5,-5,2,7,5,-8,-5],[5,3,-8,-6,-6,8,1],[-2,-4,-6,5,5,4,2],[-10,5,4,7,8,3,-10],[3,10,6,4,5,2,3],[-1,9,1,7,1,-7,5],[2,8,7,-3,-4,-6,6],[8,-3,-2,10,-6,9,9],[8,-6,1,6,-4,10,-7],[6,2,3,-10,-10,10,-7],[3,7,-4,-2,8,3,-10],[-9,-7,10,-8,-9,-10,1],[4,3,6,1,-4,4,-2],[4,-10,-8,-10,8,-3,-6]], dtype = "uint64")#candidate|374|(16, 7)|const|uint64
var_375 = relay.var("var_375", dtype = "uint64", shape = (16, 7))#candidate|375|(16, 7)|var|uint64
bop_376 = relay.multiply(const_374.astype('uint64'), relay.reshape(var_375.astype('uint64'), relay.shape_of(const_374))) # shape=(16, 7)
func_106_call = mod.get_global_var('func_106')
func_109_call = mutated_mod.get_global_var('func_109')
const_383 = relay.const([1.911170,6.736093,-0.750493,5.693853,-3.180145,7.470203,-0.895762,2.185618,0.138698,-3.461647,-9.645584,-8.174502,-8.367031,8.839049,5.538167,2.521147,2.554883,7.019690,9.083741,4.823702,6.959579,-6.338823,2.887982,8.730894,8.900632,-1.726324,-7.933376,-0.331872,-4.038220,-3.697138,-5.303980,-7.146875,-9.174845,0.993229,-5.866920,5.588216], dtype = "float64")#candidate|383|(36,)|const|float64
call_382 = func_106_call(relay.reshape(const_383.astype('float64'), [2, 6, 3]))
call_384 = func_106_call(relay.reshape(const_383.astype('float64'), [2, 6, 3]))
uop_388 = relay.sqrt(const_374.astype('float64')) # shape=(16, 7)
bop_399 = relay.greater(call_382.astype('bool'), relay.reshape(const_383.astype('bool'), relay.shape_of(call_382))) # shape=(2, 6, 3)
bop_402 = relay.greater(call_384.astype('bool'), relay.reshape(const_383.astype('bool'), relay.shape_of(call_384))) # shape=(2, 6, 3)
output = relay.Tuple([bop_376,uop_388,bop_399,])
output2 = relay.Tuple([bop_376,uop_388,bop_402,])
func_403 = relay.Function([var_375,], output)
mod['func_403'] = func_403
mod = relay.transform.InferType()(mod)
mutated_mod['func_403'] = func_403
mutated_mod = relay.transform.InferType()(mutated_mod)
var_404 = relay.var("var_404", dtype = "uint64", shape = (16, 7))#candidate|404|(16, 7)|var|uint64
func_403_call = mutated_mod.get_global_var('func_403')
call_405 = func_403_call(var_404)
output = call_405
func_406 = relay.Function([var_404], output)
mutated_mod['func_406'] = func_406
mutated_mod = relay.transform.InferType()(mutated_mod)
var_459 = relay.var("var_459", dtype = "uint32", shape = (6, 9, 6))#candidate|459|(6, 9, 6)|var|uint32
const_460 = relay.const([[[4,-10,8,5,-4,1],[10,-2,-2,-9,7,6],[9,4,-5,-7,-6,5],[4,-5,-7,-6,10,3],[-4,-7,-6,-10,5,10],[-6,-10,-3,-8,2,-7],[-10,8,-1,-3,3,-6],[5,7,-10,9,-3,2],[-3,-3,4,6,3,5]],[[6,10,10,-2,2,4],[-6,9,1,5,10,8],[6,-7,9,4,9,-9],[-10,3,-2,-5,9,10],[-5,-5,1,-8,-9,-9],[-6,6,-6,6,7,1],[8,8,10,-9,-2,-10],[8,-1,3,-7,-2,-8],[-9,-8,8,6,5,-3]],[[-3,-8,7,-8,-5,3],[-1,-9,-3,-3,3,-1],[-9,7,2,-7,-2,-7],[-4,-5,-6,-7,9,9],[-4,6,-2,-8,-5,-4],[-5,-9,-9,-8,-5,-2],[-3,7,-6,4,9,-9],[9,3,-4,-1,2,-8],[8,4,10,4,6,5]],[[-10,-3,-1,1,4,5],[2,7,-8,5,-2,1],[6,5,-9,-3,1,9],[10,3,-3,-2,-6,1],[5,-2,-6,2,8,-5],[10,-5,3,3,3,7],[-6,-5,7,8,1,-2],[-1,-2,-2,6,-4,-10],[-7,2,1,-10,-8,1]],[[8,-1,9,9,-6,1],[-7,-9,-9,7,4,-8],[1,-1,-10,8,4,-10],[-1,-5,9,6,-4,-6],[-5,-1,-4,-3,-8,6],[1,-7,-3,9,5,2],[1,-1,-4,2,-1,10],[3,-10,2,-8,-10,-7],[1,9,-10,-1,1,-8]],[[8,-6,-4,-4,-4,7],[-2,8,-9,7,10,9],[6,-1,-9,-2,10,5],[6,10,-4,5,-1,10],[-7,1,2,7,9,5],[-8,6,-6,-3,-8,4],[9,8,2,1,4,7],[3,1,2,9,-2,-10],[-6,3,-6,-6,6,-5]]], dtype = "uint32")#candidate|460|(6, 9, 6)|const|uint32
bop_461 = relay.maximum(var_459.astype('uint32'), relay.reshape(const_460.astype('uint32'), relay.shape_of(var_459))) # shape=(6, 9, 6)
func_106_call = mod.get_global_var('func_106')
func_109_call = mutated_mod.get_global_var('func_109')
const_475 = relay.const([[9.491702,-8.140561],[2.973249,7.126248],[-3.311989,-4.664196],[4.781642,-7.397061],[6.722795,9.311730],[-5.231275,-4.544439],[-9.955792,-0.411046],[-3.123601,1.533645],[-5.018684,9.203963],[9.264069,-7.446976],[-5.356785,-8.659168],[-2.315973,-3.108925],[4.106900,1.618681],[-8.922184,5.956090],[8.102179,0.510531],[-2.027201,0.726512],[6.941053,-1.967581],[2.828217,-3.300683]], dtype = "float64")#candidate|475|(18, 2)|const|float64
call_474 = func_106_call(relay.reshape(const_475.astype('float64'), [2, 6, 3]))
call_476 = func_106_call(relay.reshape(const_475.astype('float64'), [2, 6, 3]))
output = relay.Tuple([bop_461,call_474,const_475,])
output2 = relay.Tuple([bop_461,call_476,const_475,])
func_492 = relay.Function([var_459,], output)
mod['func_492'] = func_492
mod = relay.transform.InferType()(mod)
mutated_mod['func_492'] = func_492
mutated_mod = relay.transform.InferType()(mutated_mod)
var_493 = relay.var("var_493", dtype = "uint32", shape = (6, 9, 6))#candidate|493|(6, 9, 6)|var|uint32
func_492_call = mutated_mod.get_global_var('func_492')
call_494 = func_492_call(var_493)
output = call_494
func_495 = relay.Function([var_493], output)
mutated_mod['func_495'] = func_495
mutated_mod = relay.transform.InferType()(mutated_mod)
const_575 = relay.const([[[6,-1,6,-9,2,-10],[5,-4,3,4,2,-6],[-4,-2,7,2,7,-6],[-3,8,6,-2,10,-1],[-6,1,-3,-7,-1,-1],[-2,4,8,2,4,9],[-10,2,-8,4,-2,-2],[-9,-3,9,-1,-8,-8]],[[-6,2,-10,1,-9,-1],[9,-1,-8,7,3,8],[-5,2,-3,-4,3,-10],[-4,-7,5,1,1,3],[4,-7,-10,-8,-5,10],[5,8,7,-4,-8,-5],[-7,4,-8,-7,9,-9],[10,-5,2,-2,9,-4]],[[7,-9,-9,5,1,-3],[-8,-3,-6,-5,-2,1],[6,-10,5,-3,-3,7],[-9,5,-2,10,9,-7],[8,3,-10,5,-10,-1],[10,-9,-8,-5,9,-8],[-10,-10,-2,6,-7,-1],[3,-5,8,10,-3,2]],[[-5,8,-9,-4,-3,-10],[1,6,-4,9,10,2],[3,9,-1,5,6,-10],[10,-3,-10,9,-8,-3],[3,-2,5,4,-8,-4],[-1,-6,2,3,-10,4],[-1,3,-1,2,8,6],[7,-3,-2,5,4,2]],[[5,-8,2,-9,7,2],[6,-2,9,-10,-10,3],[-8,5,8,-9,-3,-2],[-3,9,10,-9,-7,-3],[-9,-5,5,-10,4,6],[3,7,-4,-2,-9,-6],[4,9,6,-8,-6,5],[10,-2,-3,8,-4,5]],[[-5,7,-9,5,10,-7],[-8,3,-9,-3,8,-3],[6,10,-8,9,1,7],[1,-5,10,7,-8,-9],[-3,-4,-2,1,-10,-9],[3,10,-6,-7,-3,-7],[-6,-10,-9,1,-3,-10],[-4,2,-1,10,1,-9]],[[1,3,8,-8,8,-9],[6,6,5,-6,-8,-4],[-9,4,-6,-6,-9,-2],[3,-3,7,8,2,-5],[-2,7,-9,-1,-5,10],[-6,-10,-7,3,-4,3],[-8,9,4,10,-4,8],[-10,-4,1,-8,-10,3]],[[-5,-1,5,-4,-2,-7],[-4,-4,4,2,10,8],[-9,10,7,9,4,-2],[-2,3,7,2,-7,-2],[6,-6,1,8,2,-1],[3,-4,-10,5,-3,7],[10,-5,-4,9,-2,-8],[1,-5,4,-4,8,4]],[[-4,-1,7,1,9,-2],[3,-5,-8,-2,-1,-2],[4,1,-3,9,4,-8],[-10,-6,-5,4,-4,6],[3,9,-4,-10,6,-10],[8,10,-1,-3,-7,2],[9,-9,4,-5,3,3],[-4,8,4,-2,-9,-4]],[[-1,3,-8,-10,-10,10],[-6,3,10,9,-10,-5],[-6,9,-9,10,-10,10],[8,4,10,-2,-10,6],[-4,3,3,-10,-5,8],[2,6,1,-1,-7,-6],[3,-3,-6,3,7,6],[6,-9,-7,-6,5,2]],[[-1,4,9,5,10,7],[1,6,7,-6,-9,5],[3,4,7,-3,-10,6],[-7,-1,1,2,9,-7],[-1,-9,-1,-4,-8,-4],[4,-3,6,-4,-8,-9],[5,10,1,-10,2,3],[-6,10,7,10,1,6]]], dtype = "int16")#candidate|575|(11, 8, 6)|const|int16
var_576 = relay.var("var_576", dtype = "int16", shape = (11, 8, 6))#candidate|576|(11, 8, 6)|var|int16
bop_577 = relay.logical_xor(const_575.astype('int16'), relay.reshape(var_576.astype('int16'), relay.shape_of(const_575))) # shape=(11, 8, 6)
func_492_call = mod.get_global_var('func_492')
func_495_call = mutated_mod.get_global_var('func_495')
const_581 = relay.const([[5,7,-7,10,-6,5,-2,-7,8,8,9,7,-5,2,10,-6,-2,-3,-10,2,-6,10,-1,4,-4,-9,10,4,-6,-9,-8,1,6,3,-7,-3,-2,-10,2,2,-8,-1,-10,-4,-3,3,5,-5,10,-10,3,-9,-8,7,2,-8,-2,-1,-5,-9,7,2,-7,-5,-5,5,8,-10,-2,-8,8,-7,6,3,4,-2,5,2,-10,-1,2,10,10,8,-7,-6,-4,-5,2,2,1,6,-7,-10,-5,-2,6,9,-2,-8,3,5,-9,-8,10,9,-5,9],[-1,-8,1,3,4,3,-9,6,-5,1,-3,4,-8,-9,8,-4,-7,7,-6,5,-4,-5,-9,7,4,-5,7,-2,-4,10,9,3,10,-5,-4,-4,9,3,10,2,-10,-1,-8,-1,-6,-6,-7,-2,-1,-1,7,5,-5,-6,10,7,4,-6,4,7,-8,-4,4,-2,-3,1,-8,-8,7,-8,-3,-9,-4,1,2,-10,1,-5,-2,-9,2,4,8,-3,-7,4,-7,-9,5,10,8,-8,-4,1,2,10,1,3,8,-2,6,-7,8,3,5,-1,-3,-1],[1,-2,3,-6,8,2,5,2,-2,4,6,3,3,4,4,5,-7,-10,-1,-5,-6,-4,2,2,-7,-4,-6,-6,-3,-6,-6,-2,-10,-3,1,-7,-4,-7,6,-10,8,8,3,4,4,4,-3,-3,1,1,4,8,-10,4,-6,1,-8,-1,-2,-8,-7,6,-1,2,-5,-5,-2,8,-9,9,7,-1,-10,-8,9,1,-10,-7,-6,-4,-3,-1,-1,-8,1,6,-9,3,-9,-6,-6,-6,1,-7,4,9,8,2,-9,10,8,-6,-2,10,7,-2,10,9]], dtype = "uint32")#candidate|581|(3, 108)|const|uint32
call_580 = relay.TupleGetItem(func_492_call(relay.reshape(const_581.astype('uint32'), [6, 9, 6])), 1)
call_582 = relay.TupleGetItem(func_495_call(relay.reshape(const_581.astype('uint32'), [6, 9, 6])), 1)
func_492_call = mod.get_global_var('func_492')
func_495_call = mutated_mod.get_global_var('func_495')
call_589 = relay.TupleGetItem(func_492_call(relay.reshape(const_581.astype('uint32'), [6, 9, 6])), 1)
call_590 = relay.TupleGetItem(func_495_call(relay.reshape(const_581.astype('uint32'), [6, 9, 6])), 1)
var_593 = relay.var("var_593", dtype = "int16", shape = (11, 8, 6))#candidate|593|(11, 8, 6)|var|int16
bop_594 = relay.not_equal(var_576.astype('bool'), relay.reshape(var_593.astype('bool'), relay.shape_of(var_576))) # shape=(11, 8, 6)
uop_599 = relay.exp(bop_577.astype('float64')) # shape=(11, 8, 6)
uop_612 = relay.sin(uop_599.astype('float32')) # shape=(11, 8, 6)
bop_614 = relay.equal(uop_599.astype('bool'), relay.reshape(const_575.astype('bool'), relay.shape_of(uop_599))) # shape=(11, 8, 6)
bop_618 = relay.equal(uop_612.astype('bool'), relay.reshape(const_575.astype('bool'), relay.shape_of(uop_612))) # shape=(11, 8, 6)
output = relay.Tuple([call_580,const_581,call_589,bop_594,bop_614,bop_618,])
output2 = relay.Tuple([call_582,const_581,call_590,bop_594,bop_614,bop_618,])
func_621 = relay.Function([var_576,var_593,], output)
mod['func_621'] = func_621
mod = relay.transform.InferType()(mod)
var_622 = relay.var("var_622", dtype = "int16", shape = (11, 8, 6))#candidate|622|(11, 8, 6)|var|int16
var_623 = relay.var("var_623", dtype = "int16", shape = (11, 8, 6))#candidate|623|(11, 8, 6)|var|int16
output = func_621(var_622,var_623,)
func_624 = relay.Function([var_622,var_623,], output)
mutated_mod['func_624'] = func_624
mutated_mod = relay.transform.InferType()(mutated_mod)
var_638 = relay.var("var_638", dtype = "float32", shape = (10, 13, 16))#candidate|638|(10, 13, 16)|var|float32
uop_639 = relay.log(var_638.astype('float32')) # shape=(10, 13, 16)
uop_643 = relay.atan(uop_639.astype('float32')) # shape=(10, 13, 16)
bop_649 = relay.logical_or(uop_643.astype('bool'), relay.reshape(uop_639.astype('bool'), relay.shape_of(uop_643))) # shape=(10, 13, 16)
output = bop_649
output2 = bop_649
func_656 = relay.Function([var_638,], output)
mod['func_656'] = func_656
mod = relay.transform.InferType()(mod)
var_657 = relay.var("var_657", dtype = "float32", shape = (10, 13, 16))#candidate|657|(10, 13, 16)|var|float32
output = func_656(var_657)
func_658 = relay.Function([var_657], output)
mutated_mod['func_658'] = func_658
mutated_mod = relay.transform.InferType()(mutated_mod)
var_673 = relay.var("var_673", dtype = "float32", shape = (5, 16, 14))#candidate|673|(5, 16, 14)|var|float32
var_674 = relay.var("var_674", dtype = "float32", shape = (5, 16, 14))#candidate|674|(5, 16, 14)|var|float32
bop_675 = relay.power(var_673.astype('float32'), relay.reshape(var_674.astype('float32'), relay.shape_of(var_673))) # shape=(5, 16, 14)
func_86_call = mod.get_global_var('func_86')
func_89_call = mutated_mod.get_global_var('func_89')
var_680 = relay.var("var_680", dtype = "float32", shape = ())#candidate|680|()|var|float32
const_681 = relay.const([-0.104642,5.579190,-6.561313,-3.048244,-0.471849,9.468332,7.311014,8.905978,-3.364531,6.588487,-6.164741,-9.092226,1.470289,-4.206565,-9.505245,-3.165818,5.143886,-1.384120,-2.374934,1.322347], dtype = "float32")#candidate|681|(20,)|const|float32
call_679 = relay.TupleGetItem(func_86_call(relay.reshape(var_680.astype('float32'), []), relay.reshape(const_681.astype('float32'), [1, 5, 4]), ), 1)
call_682 = relay.TupleGetItem(func_89_call(relay.reshape(var_680.astype('float32'), []), relay.reshape(const_681.astype('float32'), [1, 5, 4]), ), 1)
uop_699 = relay.atanh(var_673.astype('float64')) # shape=(5, 16, 14)
output = relay.Tuple([bop_675,call_679,var_680,const_681,uop_699,])
output2 = relay.Tuple([bop_675,call_682,var_680,const_681,uop_699,])
func_702 = relay.Function([var_673,var_674,var_680,], output)
mod['func_702'] = func_702
mod = relay.transform.InferType()(mod)
mutated_mod['func_702'] = func_702
mutated_mod = relay.transform.InferType()(mutated_mod)
func_702_call = mutated_mod.get_global_var('func_702')
var_704 = relay.var("var_704", dtype = "float32", shape = (5, 16, 14))#candidate|704|(5, 16, 14)|var|float32
var_705 = relay.var("var_705", dtype = "float32", shape = (5, 16, 14))#candidate|705|(5, 16, 14)|var|float32
var_706 = relay.var("var_706", dtype = "float32", shape = ())#candidate|706|()|var|float32
call_703 = func_702_call(var_704,var_705,var_706,)
output = call_703
func_707 = relay.Function([var_704,var_705,var_706,], output)
mutated_mod['func_707'] = func_707
mutated_mod = relay.transform.InferType()(mutated_mod)
var_826 = relay.var("var_826", dtype = "float32", shape = (1, 3, 13))#candidate|826|(1, 3, 13)|var|float32
const_827 = relay.const([[[3.138905,-9.446295,7.270949,-4.726812,6.306899,-3.563213,-6.017480,3.436917,-7.549416,0.517418,5.852105,2.056259,0.900931],[-9.682446,-5.553882,-4.078165,5.475115,-6.911426,-0.803433,3.968149,-7.138724,-1.927388,0.052759,4.104608,9.083541,-7.225175],[6.171745,-1.527146,9.552364,-1.366068,-5.506971,5.842767,8.126238,-7.765648,-9.574690,-1.242654,1.210206,-1.558965,-7.341558]],[[-8.237672,-4.096975,-1.134440,6.333470,-4.910084,-2.538484,-7.958589,7.356396,-0.812585,1.774971,-2.203287,-7.240968,-4.528886],[-4.858381,-3.432198,-0.196382,-0.046864,2.015133,0.444712,2.602727,-4.800318,-4.493112,2.903477,-2.500458,3.272839,-8.958713],[-1.774939,-7.183363,8.181326,6.801899,1.875070,-1.182216,-9.343683,-1.363423,-3.759812,8.495849,9.273347,6.091945,-9.487153]]], dtype = "float32")#candidate|827|(2, 3, 13)|const|float32
bop_828 = relay.greater_equal(var_826.astype('bool'), const_827.astype('bool')) # shape=(2, 3, 13)
func_492_call = mod.get_global_var('func_492')
func_495_call = mutated_mod.get_global_var('func_495')
var_837 = relay.var("var_837", dtype = "uint32", shape = (18, 18))#candidate|837|(18, 18)|var|uint32
call_836 = relay.TupleGetItem(func_492_call(relay.reshape(var_837.astype('uint32'), [6, 9, 6])), 1)
call_838 = relay.TupleGetItem(func_495_call(relay.reshape(var_837.astype('uint32'), [6, 9, 6])), 1)
output = relay.Tuple([bop_828,call_836,var_837,])
output2 = relay.Tuple([bop_828,call_838,var_837,])
func_840 = relay.Function([var_826,var_837,], output)
mod['func_840'] = func_840
mod = relay.transform.InferType()(mod)
var_841 = relay.var("var_841", dtype = "float32", shape = (1, 3, 13))#candidate|841|(1, 3, 13)|var|float32
var_842 = relay.var("var_842", dtype = "uint32", shape = (18, 18))#candidate|842|(18, 18)|var|uint32
output = func_840(var_841,var_842,)
func_843 = relay.Function([var_841,var_842,], output)
mutated_mod['func_843'] = func_843
mutated_mod = relay.transform.InferType()(mutated_mod)
const_847 = relay.const([[6.005277,-7.555267],[9.185906,6.107810],[0.444222,-9.382933]], dtype = "float64")#candidate|847|(3, 2)|const|float64
const_848 = relay.const([[-9.492150,-5.163276],[0.060854,1.440539],[5.970620,0.097285]], dtype = "float64")#candidate|848|(3, 2)|const|float64
bop_849 = relay.less(const_847.astype('bool'), relay.reshape(const_848.astype('bool'), relay.shape_of(const_847))) # shape=(3, 2)
output = relay.Tuple([bop_849,])
output2 = relay.Tuple([bop_849,])
func_863 = relay.Function([], output)
mod['func_863'] = func_863
mod = relay.transform.InferType()(mod)
mutated_mod['func_863'] = func_863
mutated_mod = relay.transform.InferType()(mutated_mod)
func_863_call = mutated_mod.get_global_var('func_863')
call_864 = func_863_call()
output = call_864
func_865 = relay.Function([], output)
mutated_mod['func_865'] = func_865
mutated_mod = relay.transform.InferType()(mutated_mod)
func_863_call = mod.get_global_var('func_863')
func_865_call = mutated_mod.get_global_var('func_865')
call_914 = relay.TupleGetItem(func_863_call(), 0)
call_915 = relay.TupleGetItem(func_865_call(), 0)
var_935 = relay.var("var_935", dtype = "bool", shape = (3, 2))#candidate|935|(3, 2)|var|bool
bop_936 = relay.floor_divide(call_914.astype('float32'), relay.reshape(var_935.astype('float32'), relay.shape_of(call_914))) # shape=(3, 2)
bop_939 = relay.floor_divide(call_915.astype('float32'), relay.reshape(var_935.astype('float32'), relay.shape_of(call_915))) # shape=(3, 2)
output = relay.Tuple([bop_936,])
output2 = relay.Tuple([bop_939,])
func_942 = relay.Function([var_935,], output)
mod['func_942'] = func_942
mod = relay.transform.InferType()(mod)
var_943 = relay.var("var_943", dtype = "bool", shape = (3, 2))#candidate|943|(3, 2)|var|bool
output = func_942(var_943)
func_944 = relay.Function([var_943], output)
mutated_mod['func_944'] = func_944
mutated_mod = relay.transform.InferType()(mutated_mod)
func_863_call = mod.get_global_var('func_863')
func_865_call = mutated_mod.get_global_var('func_865')
call_993 = relay.TupleGetItem(func_863_call(), 0)
call_994 = relay.TupleGetItem(func_865_call(), 0)
uop_995 = relay.exp(call_993.astype('float64')) # shape=(3, 2)
uop_997 = relay.exp(call_994.astype('float64')) # shape=(3, 2)
output = relay.Tuple([uop_995,])
output2 = relay.Tuple([uop_997,])
func_1001 = relay.Function([], output)
mod['func_1001'] = func_1001
mod = relay.transform.InferType()(mod)
output = func_1001()
func_1002 = relay.Function([], output)
mutated_mod['func_1002'] = func_1002
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1006 = relay.var("var_1006", dtype = "float64", shape = (7, 9))#candidate|1006|(7, 9)|var|float64
uop_1007 = relay.erf(var_1006.astype('float64')) # shape=(7, 9)
output = uop_1007
output2 = uop_1007
func_1009 = relay.Function([var_1006,], output)
mod['func_1009'] = func_1009
mod = relay.transform.InferType()(mod)
mutated_mod['func_1009'] = func_1009
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1010 = relay.var("var_1010", dtype = "float64", shape = (7, 9))#candidate|1010|(7, 9)|var|float64
func_1009_call = mutated_mod.get_global_var('func_1009')
call_1011 = func_1009_call(var_1010)
output = call_1011
func_1012 = relay.Function([var_1010], output)
mutated_mod['func_1012'] = func_1012
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1014 = relay.var("var_1014", dtype = "int16", shape = (3, 14))#candidate|1014|(3, 14)|var|int16
var_1015 = relay.var("var_1015", dtype = "int16", shape = (3, 14))#candidate|1015|(3, 14)|var|int16
bop_1016 = relay.bitwise_and(var_1014.astype('int16'), relay.reshape(var_1015.astype('int16'), relay.shape_of(var_1014))) # shape=(3, 14)
output = relay.Tuple([bop_1016,])
output2 = relay.Tuple([bop_1016,])
func_1035 = relay.Function([var_1014,var_1015,], output)
mod['func_1035'] = func_1035
mod = relay.transform.InferType()(mod)
mutated_mod['func_1035'] = func_1035
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1035_call = mutated_mod.get_global_var('func_1035')
var_1037 = relay.var("var_1037", dtype = "int16", shape = (3, 14))#candidate|1037|(3, 14)|var|int16
var_1038 = relay.var("var_1038", dtype = "int16", shape = (3, 14))#candidate|1038|(3, 14)|var|int16
call_1036 = func_1035_call(var_1037,var_1038,)
output = call_1036
func_1039 = relay.Function([var_1037,var_1038,], output)
mutated_mod['func_1039'] = func_1039
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1001_call = mod.get_global_var('func_1001')
func_1002_call = mutated_mod.get_global_var('func_1002')
call_1050 = relay.TupleGetItem(func_1001_call(), 0)
call_1051 = relay.TupleGetItem(func_1002_call(), 0)
uop_1053 = relay.erf(call_1050.astype('float64')) # shape=(3, 2)
uop_1055 = relay.erf(call_1051.astype('float64')) # shape=(3, 2)
uop_1059 = relay.rsqrt(uop_1053.astype('float64')) # shape=(3, 2)
uop_1061 = relay.rsqrt(uop_1055.astype('float64')) # shape=(3, 2)
bop_1062 = relay.floor_mod(uop_1059.astype('float32'), relay.reshape(call_1050.astype('float32'), relay.shape_of(uop_1059))) # shape=(3, 2)
bop_1065 = relay.floor_mod(uop_1061.astype('float32'), relay.reshape(call_1051.astype('float32'), relay.shape_of(uop_1061))) # shape=(3, 2)
func_403_call = mod.get_global_var('func_403')
func_406_call = mutated_mod.get_global_var('func_406')
const_1067 = relay.const([-9,-10,3,-3,-8,-2,2,-1,-6,7,-5,10,4,10,-10,3,-5,8,-9,-9,-5,-1,3,-3,2,2,-4,-5,-10,10,-7,1,8,-10,-10,7,3,6,10,4,-9,9,1,-4,8,-2,8,7,-1,-4,-7,8,-9,-6,4,-2,10,-6,7,1,5,10,-7,1,5,10,-6,-4,5,-3,1,9,7,2,7,-9,9,8,-8,5,-6,-7,-5,9,-10,6,8,-3,-3,10,2,-2,1,-8,-1,-4,10,-9,-10,-5,-1,-2,-8,-7,-1,1,-10,-4,-5,2,7,-2], dtype = "uint64")#candidate|1067|(112,)|const|uint64
call_1066 = relay.TupleGetItem(func_403_call(relay.reshape(const_1067.astype('uint64'), [16, 7])), 0)
call_1068 = relay.TupleGetItem(func_406_call(relay.reshape(const_1067.astype('uint64'), [16, 7])), 0)
output = relay.Tuple([bop_1062,call_1066,const_1067,])
output2 = relay.Tuple([bop_1065,call_1068,const_1067,])
func_1069 = relay.Function([], output)
mod['func_1069'] = func_1069
mod = relay.transform.InferType()(mod)
mutated_mod['func_1069'] = func_1069
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1069_call = mutated_mod.get_global_var('func_1069')
call_1070 = func_1069_call()
output = call_1070
func_1071 = relay.Function([], output)
mutated_mod['func_1071'] = func_1071
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1081 = relay.var("var_1081", dtype = "int32", shape = (2, 5))#candidate|1081|(2, 5)|var|int32
const_1082 = relay.const([[4,-1,7,9,7],[-1,-6,9,-4,-9]], dtype = "int32")#candidate|1082|(2, 5)|const|int32
bop_1083 = relay.bitwise_and(var_1081.astype('int32'), relay.reshape(const_1082.astype('int32'), relay.shape_of(var_1081))) # shape=(2, 5)
output = bop_1083
output2 = bop_1083
func_1087 = relay.Function([var_1081,], output)
mod['func_1087'] = func_1087
mod = relay.transform.InferType()(mod)
mutated_mod['func_1087'] = func_1087
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1088 = relay.var("var_1088", dtype = "int32", shape = (2, 5))#candidate|1088|(2, 5)|var|int32
func_1087_call = mutated_mod.get_global_var('func_1087')
call_1089 = func_1087_call(var_1088)
output = call_1089
func_1090 = relay.Function([var_1088], output)
mutated_mod['func_1090'] = func_1090
mutated_mod = relay.transform.InferType()(mutated_mod)
func_863_call = mod.get_global_var('func_863')
func_865_call = mutated_mod.get_global_var('func_865')
call_1092 = relay.TupleGetItem(func_863_call(), 0)
call_1093 = relay.TupleGetItem(func_865_call(), 0)
output = relay.Tuple([call_1092,])
output2 = relay.Tuple([call_1093,])
func_1096 = relay.Function([], output)
mod['func_1096'] = func_1096
mod = relay.transform.InferType()(mod)
output = func_1096()
func_1097 = relay.Function([], output)
mutated_mod['func_1097'] = func_1097
mutated_mod = relay.transform.InferType()(mutated_mod)
func_863_call = mod.get_global_var('func_863')
func_865_call = mutated_mod.get_global_var('func_865')
call_1121 = relay.TupleGetItem(func_863_call(), 0)
call_1122 = relay.TupleGetItem(func_865_call(), 0)
output = call_1121
output2 = call_1122
func_1132 = relay.Function([], output)
mod['func_1132'] = func_1132
mod = relay.transform.InferType()(mod)
mutated_mod['func_1132'] = func_1132
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1132_call = mutated_mod.get_global_var('func_1132')
call_1133 = func_1132_call()
output = call_1133
func_1134 = relay.Function([], output)
mutated_mod['func_1134'] = func_1134
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1148 = relay.const([[[2.485627,4.138591,-5.381469,4.459629,3.066942,2.260478,-7.444427],[-7.390391,2.617194,-4.607219,6.238992,-8.691022,-3.765938,-8.176471],[-0.416398,-5.991228,-2.600950,6.119907,1.446243,-9.010409,-9.554775],[-6.258135,9.758913,-8.321593,7.964634,-3.759753,7.392112,7.456553],[-8.686205,-0.568485,3.291127,-5.607823,-0.935402,-9.700154,-9.705459],[-0.275769,-8.913133,2.044654,-5.339642,9.236656,9.203785,-3.717242],[-6.350503,-5.452698,3.280768,6.595283,-6.178870,-1.683532,8.990208],[-2.650869,0.759058,-7.892297,-5.464821,2.917625,3.222367,-7.190474]],[[4.722807,7.078491,-7.937709,-4.287968,-3.831148,-9.185973,0.961075],[-2.509568,7.991854,2.743986,6.641360,-5.128743,8.772793,-0.403544],[-7.878489,6.467370,0.182508,3.688612,5.653697,-0.234541,3.742742],[-9.551366,6.807746,4.810378,9.875341,-1.939863,-0.506724,-0.118007],[-7.838506,-0.351824,3.144184,-1.720803,9.773434,-2.365867,8.222632],[-3.531651,4.236689,-0.458575,0.220602,3.898975,-1.302379,-9.377066],[6.162914,-2.850279,5.565162,-4.221717,6.153160,-3.850800,6.554184],[8.344267,6.032030,2.612934,-6.362850,6.463356,-3.648953,-2.697245]],[[-5.036233,2.847860,-8.195851,-6.638778,2.718058,-5.885772,6.805324],[8.582970,-7.411227,-6.340213,-0.711718,-6.159121,7.995807,2.958111],[0.232698,-1.283257,-4.638788,2.465702,7.818948,-6.045348,-5.329071],[8.909375,7.918287,-7.300446,0.593048,4.794288,5.086071,-7.301496],[-3.076395,-5.813367,-1.769210,2.208390,-5.741936,5.642137,-6.249474],[-7.185209,-3.328909,0.727081,-6.349559,5.854145,-0.584772,-6.505321],[8.667376,-9.999479,-0.508035,9.569165,7.796427,4.203196,3.119110],[3.281044,-6.319163,-4.151397,-1.034650,5.050476,-3.093977,-2.500466]],[[0.902201,-8.904885,-0.090406,-9.759160,1.330886,9.274505,1.376179],[-8.625374,0.651188,-2.662311,-9.990735,-4.294396,-5.244625,3.399848],[5.176617,5.130834,-5.591230,4.917031,-5.044386,7.820032,3.860861],[-0.751712,6.966605,1.038889,8.636997,8.310618,-0.765685,-4.803220],[1.382366,8.076172,3.073657,5.043123,-8.392654,-5.923859,6.020571],[-9.974617,-5.760697,1.887326,-6.840566,-3.433265,-5.954974,0.250235],[-0.958336,-6.745884,7.016407,-2.898975,-3.349616,1.096876,-5.829704],[-1.490137,4.410783,-5.378194,5.952636,-7.093640,-4.002102,8.110731]],[[-0.243144,5.169993,-5.375382,-9.647638,-6.595256,4.413130,-6.967438],[1.961920,-4.369021,-9.397713,2.988008,-5.717714,-2.977389,-8.279817],[1.901149,-3.520474,-1.402540,-1.109469,9.036834,-4.794393,8.008363],[-8.406821,-8.459579,-1.071863,6.830108,8.489593,1.269355,-3.479156],[-3.728035,-7.922712,-3.203680,4.432299,3.567583,-3.348474,-5.947141],[-7.700528,-0.732776,9.535329,-3.989089,2.990991,7.919671,-0.727675],[-6.128958,-9.878058,6.024606,2.524479,7.552465,-5.087385,-4.256614],[5.500020,-9.151053,9.509086,1.831590,7.262528,-3.598925,7.789255]],[[7.138046,8.842467,-1.188343,-5.499016,-9.288113,-4.394863,-2.609000],[-6.821931,3.146722,6.498181,8.776949,-4.020191,-4.139600,3.230174],[-8.816510,-1.020744,-3.558654,-8.605886,8.881224,1.242854,-2.793080],[-3.699317,-7.662046,7.149556,-5.640948,5.850239,5.300778,-4.063727],[-6.503225,3.586018,5.581150,-5.911030,3.628463,-0.640590,-7.091692],[-3.676553,6.289953,2.321567,-0.157527,4.323777,-8.749726,-0.581877],[-9.992092,2.937621,-7.361697,-1.920696,-0.740683,-9.061545,0.557209],[-5.003103,-8.333964,-0.008649,3.070810,1.786790,-0.073141,2.253479]],[[-1.183613,6.548607,-8.823402,3.806939,4.045643,-8.445455,-3.517143],[2.508189,-5.593288,0.546175,-5.880546,-9.625413,-1.252111,3.082880],[7.559369,5.751208,7.164674,-0.033678,6.285264,-0.422123,3.991257],[-4.944378,7.684898,3.000177,7.185103,-6.032353,-4.923607,-6.695130],[0.802159,-6.473931,-9.340967,-8.708458,-3.960738,5.745404,-6.308467],[-3.080498,7.495064,6.080090,-3.727096,-8.779161,-9.367729,0.889319],[2.300106,9.248051,-9.721507,6.171130,0.601114,-4.339690,5.709230],[2.268622,9.205271,2.432696,-2.606599,-4.344976,-5.580540,-3.667585]],[[4.861568,0.680097,3.586312,2.302540,-6.032272,-4.440529,-9.043392],[0.153095,5.082936,1.319657,-6.453879,0.993521,-0.642727,4.798630],[-4.913605,9.125615,-5.194705,-2.800910,3.855328,1.510931,1.262704],[8.027203,-5.878046,4.490992,8.689376,-6.214059,2.162988,-7.829076],[2.791000,-4.743709,7.278995,-7.640420,5.755651,-6.597861,-2.537934],[-2.618234,5.222820,-7.820905,1.682076,6.890383,-7.849270,-3.812347],[9.021634,7.984984,4.506444,-3.951690,3.412605,-5.277306,-6.316251],[9.028006,-9.478972,-9.646814,7.089508,7.510456,-5.346942,7.437213]],[[-1.503810,-7.580357,-7.688831,9.539203,-2.626962,2.382959,-6.749371],[-3.574080,-3.538360,5.511327,8.522242,-8.437556,-0.967030,9.964231],[0.687610,-4.088392,7.535326,6.138009,-1.060838,7.097551,3.854495],[6.875821,-9.864331,-2.543466,7.726514,1.412070,8.324985,-9.904887],[8.292066,-2.105628,-6.318348,-4.034112,8.720950,-6.381694,-5.285112],[-2.089164,-8.748613,8.086429,4.341405,9.781483,-1.690614,-3.336795],[-3.049406,2.805973,-2.975539,-3.436767,-7.877151,-2.941384,3.010024],[-1.284949,-3.555388,6.545691,-5.323018,8.215588,-3.033458,-6.081779]]], dtype = "float32")#candidate|1148|(9, 8, 7)|const|float32
const_1149 = relay.const([[[-3.586395,6.784130,0.164908,1.439931,2.739160,6.177148,3.748773],[-8.775244,-2.788721,9.344702,9.422521,1.427457,2.827514,2.235330],[-2.042582,-5.237269,-8.232402,-2.565111,-3.432264,-0.595896,-3.845959],[3.895622,-8.023646,1.609609,2.960811,-5.576899,-0.689990,3.112133],[1.079918,6.620520,-6.657733,-7.104675,-9.138553,-2.839261,-0.788648],[-3.236748,-5.941577,-9.201725,1.456516,-6.262176,-3.447325,3.378911],[-0.112651,3.815159,1.978237,9.381649,-5.869555,-2.636769,-9.754300],[0.162498,2.684646,4.968111,-4.949995,-7.454196,6.678688,-8.209716]],[[3.339436,-1.131220,-0.649907,7.894670,3.696287,4.471219,-6.640010],[-6.395913,1.565121,3.131380,6.021171,-3.723278,7.515730,6.378653],[9.747946,-3.515377,5.944979,-1.365134,-1.126937,8.319099,-3.720706],[-1.847720,-1.275647,-8.285312,-2.067648,2.644823,-9.982506,3.811922],[-7.131325,3.295616,-3.893135,0.626973,-7.340688,-9.382640,7.286726],[-8.865757,-9.435262,3.072120,7.218788,-0.674842,3.681738,5.131218],[-4.972177,6.930476,3.956944,-6.885198,3.350039,2.497198,-0.524081],[-5.068011,1.322514,7.028875,4.671891,7.502259,-3.986308,4.250753]],[[0.797885,-0.158219,9.877204,-8.687931,-7.895546,6.165711,-8.904186],[7.535972,-0.490429,6.992350,0.255134,6.029945,7.899198,4.200666],[7.080831,1.452022,-0.481140,-2.233145,-7.317361,-4.613130,7.886753],[-9.672953,2.880677,-3.675692,1.572229,1.944763,4.129943,-4.046328],[7.320875,-4.893659,-1.138790,9.042557,8.522870,-0.346427,8.186366],[-4.017228,-9.406222,-6.493142,2.508728,9.096783,-6.694278,8.858282],[-8.841627,-1.527300,1.261302,-0.891077,4.837134,-6.359474,4.752435],[-2.126713,-7.423581,0.735024,7.168190,-5.063006,1.810058,3.615034]],[[-1.078589,9.972989,-2.543356,5.348065,-2.068512,3.169073,8.102617],[7.141637,9.449521,0.089329,-7.724354,-6.250828,2.667398,-8.665238],[8.580103,-7.857928,-8.762730,-9.745778,-2.564595,2.110989,5.308492],[-1.392117,-2.482939,4.228690,0.843118,9.796087,5.303119,9.892096],[-7.420469,8.212239,0.234938,-3.970937,4.586042,3.001447,-9.998916],[9.586745,4.979803,-0.255630,1.347501,5.272267,4.180468,-4.763875],[6.781451,7.297490,-0.230917,8.848037,-0.080440,4.043619,-0.197434],[0.323300,3.473405,-2.444799,-3.135063,7.909718,-2.900095,-8.208153]],[[1.441358,-2.194679,-1.274751,9.545513,7.320710,4.747138,-9.810402],[2.702089,1.520466,3.463307,9.782236,6.965967,-4.549775,7.965322],[-5.883132,-3.285442,7.842647,-4.947764,-8.560099,-9.266947,-6.872339],[-2.450441,-6.901057,5.967835,-1.080323,8.561554,-9.616864,-6.133789],[0.462397,3.356241,3.315549,5.829708,-4.755028,2.097104,3.634225],[-0.231718,5.529556,-9.101681,1.431189,9.956144,-4.823331,6.053504],[7.325076,-6.416305,-3.259840,-0.125695,4.348913,-8.074666,3.587775],[9.748206,0.703149,-9.729231,-1.208123,-8.755317,-4.070482,4.806590]],[[-4.336835,0.541138,-9.254223,-2.255206,2.386963,6.132718,3.627166],[7.681424,9.981687,1.921403,-0.798969,0.270326,7.496619,-7.719383],[8.029416,-6.645360,7.469023,6.807757,7.811675,-6.672837,2.760944],[-3.806933,9.566849,7.535189,-5.912022,2.608904,3.209175,8.242993],[-3.023504,-8.345892,4.329120,-1.067117,6.745872,8.169116,2.092090],[1.152986,5.123113,9.185652,9.580147,8.867682,-7.047141,2.111508],[5.521184,-5.372489,-8.956664,2.638175,6.434547,-8.264824,6.190830],[-9.742641,0.200959,5.916304,7.584516,1.757327,-0.641667,-4.666210]],[[4.035110,-5.020096,-7.643776,3.618318,0.160925,-6.310624,-7.395977],[7.644327,-1.520608,-5.469555,-9.813932,-3.728832,4.017960,-2.578394],[-1.876532,-9.482448,-9.452947,-5.575970,7.945115,-9.052658,-2.877106],[0.883236,3.474354,1.485857,-1.484923,-8.417136,6.460656,-4.281832],[-5.919178,-3.921664,2.567077,5.927971,1.862653,-8.538492,6.147713],[-0.198524,-3.322687,-0.940992,-2.836027,8.254581,5.643074,9.387008],[6.374766,6.772145,9.354163,-4.267207,-1.009711,-2.293721,-5.354477],[-8.160048,1.213130,8.778718,7.241697,-5.162242,2.631444,-2.860772]],[[9.677140,-7.878005,-7.659280,-5.648385,8.101220,-7.251243,-2.549149],[-9.478732,-8.268870,-0.599496,0.323743,0.384537,3.481945,-9.630743],[6.672745,-5.269375,8.438755,4.411691,-0.938749,2.970739,9.485570],[-5.854095,-5.191549,-9.246462,2.587152,4.148574,-1.256866,9.478473],[-9.910093,6.627047,1.161696,-8.803973,-6.776359,-1.064576,1.207389],[-9.380063,6.282336,6.285491,3.155268,-4.873629,-7.741376,5.070365],[-7.207202,0.901015,1.610646,0.518295,-1.116464,3.694941,1.823940],[1.673853,5.347373,9.620796,-3.456596,-5.982328,-5.991693,2.764897]],[[-9.655093,-4.967435,3.576893,8.435141,-5.934522,1.028179,-1.530976],[2.043532,-0.981525,1.370785,-9.967173,9.111451,7.029802,9.598916],[-9.400967,-5.264525,7.055662,-0.401916,-2.112381,0.292764,-2.294289],[7.777370,8.079689,-7.401354,-1.539303,2.503625,3.232041,3.111927],[4.142917,-2.423729,3.014480,-0.469659,0.965793,-1.609075,7.465700],[-6.319206,-8.054476,-0.225786,-7.383743,-5.228143,-8.583600,-8.415983],[-6.231886,-0.893122,-1.714074,6.642731,-4.119637,-9.902314,-2.724927],[1.426390,8.502369,0.391402,-2.044815,7.427155,2.202142,3.450316]]], dtype = "float32")#candidate|1149|(9, 8, 7)|const|float32
bop_1150 = relay.mod(const_1148.astype('float32'), relay.reshape(const_1149.astype('float32'), relay.shape_of(const_1148))) # shape=(9, 8, 7)
bop_1155 = relay.bitwise_or(const_1148.astype('int64'), relay.reshape(bop_1150.astype('int64'), relay.shape_of(const_1148))) # shape=(9, 8, 7)
output = bop_1155
output2 = bop_1155
func_1165 = relay.Function([], output)
mod['func_1165'] = func_1165
mod = relay.transform.InferType()(mod)
mutated_mod['func_1165'] = func_1165
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mutated_mod.get_global_var('func_1165')
call_1166 = func_1165_call()
output = call_1166
func_1167 = relay.Function([], output)
mutated_mod['func_1167'] = func_1167
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1186 = relay.const([[-1.608174,6.590495,-6.293388,6.033987,-1.339624,5.184795,5.589714,-3.144338,4.034070,8.756134,5.694732,-7.803624,-4.629471],[-8.989968,5.545369,7.982616,0.487499,9.009432,9.160925,9.501731,4.715605,-9.416142,2.609383,-8.276479,-6.381987,1.203028],[-6.031820,-0.247707,9.035224,-5.627103,-9.198572,-4.648257,-7.529451,-2.028057,-2.317555,-0.021279,-6.398287,-6.152149,1.498368],[-9.642867,4.835527,2.077078,8.208183,-9.482234,8.868631,3.721916,-8.810062,1.613597,4.888798,-8.974064,-7.491119,0.956336],[4.569085,-0.625251,7.657994,-8.429287,-8.757525,-4.223310,5.397829,5.704695,-9.812841,0.947490,-3.594832,6.673652,-7.586096],[-2.548655,-0.292042,4.686206,9.323668,1.909530,-2.862208,2.152351,8.737212,4.593887,-0.391881,3.652479,7.224798,-0.912204],[-8.918733,-4.159871,0.130544,7.848060,-5.054160,-2.354017,-0.711822,-3.618004,8.534396,-3.858291,-6.937581,-6.899250,-8.096912]], dtype = "float64")#candidate|1186|(7, 13)|const|float64
const_1187 = relay.const([[-7.196180,2.389395,9.609395,7.265047,-1.299516,5.100836,9.994249,9.216102,6.914149,-1.102688,6.081610,2.518263,-2.319409],[2.930211,-2.747781,1.989743,-4.157498,9.628615,-7.899627,-1.945552,2.654006,1.196234,7.235442,-3.614042,7.842858,-0.025822],[8.033650,-7.370525,3.018729,-8.974939,-0.849962,-0.354128,1.006963,-0.142379,-8.678438,-2.557596,-7.764602,-5.229526,-1.085205],[-7.422375,0.642632,9.271530,2.058371,0.667513,2.240207,-3.305572,0.590967,-6.866413,-8.774867,-0.956189,6.226491,-1.582701],[-3.814053,-7.928998,8.983364,-7.714300,9.645216,-3.586809,3.952285,3.923786,9.502659,-4.439307,-5.499091,6.182279,-0.983885],[9.863690,2.762882,5.239470,-1.869720,-7.317201,9.960820,-7.767473,-1.487538,9.668620,-8.586805,-2.926446,-5.120400,-2.095273],[6.423897,9.688445,9.559663,-4.862227,-2.405832,6.931962,6.836369,-2.646650,-7.678063,0.533772,7.208343,5.143183,2.019464]], dtype = "float64")#candidate|1187|(7, 13)|const|float64
bop_1188 = relay.subtract(const_1186.astype('float64'), relay.reshape(const_1187.astype('float64'), relay.shape_of(const_1186))) # shape=(7, 13)
var_1192 = relay.var("var_1192", dtype = "float64", shape = (7, 13))#candidate|1192|(7, 13)|var|float64
bop_1193 = relay.right_shift(const_1186.astype('uint64'), relay.reshape(var_1192.astype('uint64'), relay.shape_of(const_1186))) # shape=(7, 13)
bop_1197 = relay.greater_equal(bop_1188.astype('bool'), relay.reshape(var_1192.astype('bool'), relay.shape_of(bop_1188))) # shape=(7, 13)
var_1206 = relay.var("var_1206", dtype = "bool", shape = (7, 13))#candidate|1206|(7, 13)|var|bool
bop_1207 = relay.logical_or(bop_1197.astype('bool'), relay.reshape(var_1206.astype('bool'), relay.shape_of(bop_1197))) # shape=(7, 13)
uop_1213 = relay.log10(var_1192.astype('float32')) # shape=(7, 13)
uop_1215 = relay.erf(bop_1197.astype('float64')) # shape=(7, 13)
bop_1217 = relay.equal(uop_1215.astype('bool'), relay.reshape(bop_1188.astype('bool'), relay.shape_of(uop_1215))) # shape=(7, 13)
func_403_call = mod.get_global_var('func_403')
func_406_call = mutated_mod.get_global_var('func_406')
var_1222 = relay.var("var_1222", dtype = "uint64", shape = (112,))#candidate|1222|(112,)|var|uint64
call_1221 = relay.TupleGetItem(func_403_call(relay.reshape(var_1222.astype('uint64'), [16, 7])), 0)
call_1223 = relay.TupleGetItem(func_406_call(relay.reshape(var_1222.astype('uint64'), [16, 7])), 0)
func_257_call = mod.get_global_var('func_257')
func_261_call = mutated_mod.get_global_var('func_261')
var_1235 = relay.var("var_1235", dtype = "float64", shape = (77,))#candidate|1235|(77,)|var|float64
var_1236 = relay.var("var_1236", dtype = "float64", shape = (18, 2))#candidate|1236|(18, 2)|var|float64
call_1234 = relay.TupleGetItem(func_257_call(relay.reshape(var_1235.astype('float64'), [7, 11]), relay.reshape(var_1236.astype('float64'), [36,]), ), 3)
call_1237 = relay.TupleGetItem(func_261_call(relay.reshape(var_1235.astype('float64'), [7, 11]), relay.reshape(var_1236.astype('float64'), [36,]), ), 3)
output = relay.Tuple([bop_1193,bop_1207,uop_1213,bop_1217,call_1221,var_1222,call_1234,var_1235,var_1236,])
output2 = relay.Tuple([bop_1193,bop_1207,uop_1213,bop_1217,call_1223,var_1222,call_1237,var_1235,var_1236,])
func_1246 = relay.Function([var_1192,var_1206,var_1222,var_1235,var_1236,], output)
mod['func_1246'] = func_1246
mod = relay.transform.InferType()(mod)
var_1247 = relay.var("var_1247", dtype = "float64", shape = (7, 13))#candidate|1247|(7, 13)|var|float64
var_1248 = relay.var("var_1248", dtype = "bool", shape = (7, 13))#candidate|1248|(7, 13)|var|bool
var_1249 = relay.var("var_1249", dtype = "uint64", shape = (112,))#candidate|1249|(112,)|var|uint64
var_1250 = relay.var("var_1250", dtype = "float64", shape = (77,))#candidate|1250|(77,)|var|float64
var_1251 = relay.var("var_1251", dtype = "float64", shape = (18, 2))#candidate|1251|(18, 2)|var|float64
output = func_1246(var_1247,var_1248,var_1249,var_1250,var_1251,)
func_1252 = relay.Function([var_1247,var_1248,var_1249,var_1250,var_1251,], output)
mutated_mod['func_1252'] = func_1252
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1132_call = mod.get_global_var('func_1132')
func_1134_call = mutated_mod.get_global_var('func_1134')
call_1330 = func_1132_call()
call_1331 = func_1132_call()
const_1336 = relay.const([[False,False],[True,True],[False,True]], dtype = "bool")#candidate|1336|(3, 2)|const|bool
bop_1337 = relay.maximum(call_1330.astype('uint8'), relay.reshape(const_1336.astype('uint8'), relay.shape_of(call_1330))) # shape=(3, 2)
bop_1340 = relay.maximum(call_1331.astype('uint8'), relay.reshape(const_1336.astype('uint8'), relay.shape_of(call_1331))) # shape=(3, 2)
output = relay.Tuple([bop_1337,])
output2 = relay.Tuple([bop_1340,])
func_1352 = relay.Function([], output)
mod['func_1352'] = func_1352
mod = relay.transform.InferType()(mod)
mutated_mod['func_1352'] = func_1352
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1352_call = mutated_mod.get_global_var('func_1352')
call_1353 = func_1352_call()
output = call_1353
func_1354 = relay.Function([], output)
mutated_mod['func_1354'] = func_1354
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1355 = relay.const(False, dtype = "bool")#candidate|1355|()|const|bool
var_1356 = relay.var("var_1356", dtype = "bool", shape = (5, 7))#candidate|1356|(5, 7)|var|bool
bop_1357 = relay.logical_and(const_1355.astype('bool'), var_1356.astype('bool')) # shape=(5, 7)
bop_1360 = relay.minimum(var_1356.astype('uint8'), const_1355.astype('uint8')) # shape=(5, 7)
func_86_call = mod.get_global_var('func_86')
func_89_call = mutated_mod.get_global_var('func_89')
const_1365 = relay.const([[9.960951,-6.559293,0.243494,-6.747021,-7.102408,-5.800674,-0.286309,-5.374551,-5.165031,-2.193249,2.623823,9.038058,1.180090,6.434320,3.923542,-5.105159,2.091736,4.017589,-2.947133,1.239826]], dtype = "float32")#candidate|1365|(1, 20)|const|float32
call_1364 = relay.TupleGetItem(func_86_call(relay.reshape(const_1355.astype('float32'), []), relay.reshape(const_1365.astype('float32'), [1, 5, 4]), ), 3)
call_1366 = relay.TupleGetItem(func_89_call(relay.reshape(const_1355.astype('float32'), []), relay.reshape(const_1365.astype('float32'), [1, 5, 4]), ), 3)
uop_1367 = relay.atan(bop_1360.astype('float32')) # shape=(5, 7)
uop_1372 = relay.tan(uop_1367.astype('float64')) # shape=(5, 7)
const_1376 = relay.const([[8.753853,4.819543,0.108589,-8.715744,2.645134,7.826148,-1.774052],[-3.497104,2.949315,-8.480241,2.338752,4.986925,-9.516705,-7.246188],[1.302627,8.714886,9.845891,4.199364,-4.144494,-7.800938,4.812895],[2.879767,-3.450960,-0.872107,-6.244900,7.792674,-5.537967,1.117709],[7.749208,-3.662350,-0.447832,2.754351,-4.931419,-4.556830,-9.189118]], dtype = "float64")#candidate|1376|(5, 7)|const|float64
bop_1377 = relay.maximum(uop_1372.astype('float64'), relay.reshape(const_1376.astype('float64'), relay.shape_of(uop_1372))) # shape=(5, 7)
output = relay.Tuple([bop_1357,call_1364,const_1365,bop_1377,])
output2 = relay.Tuple([bop_1357,call_1366,const_1365,bop_1377,])
func_1380 = relay.Function([var_1356,], output)
mod['func_1380'] = func_1380
mod = relay.transform.InferType()(mod)
mutated_mod['func_1380'] = func_1380
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1381 = relay.var("var_1381", dtype = "bool", shape = (5, 7))#candidate|1381|(5, 7)|var|bool
func_1380_call = mutated_mod.get_global_var('func_1380')
call_1382 = func_1380_call(var_1381)
output = call_1382
func_1383 = relay.Function([var_1381], output)
mutated_mod['func_1383'] = func_1383
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1413 = relay.const([8,4,5,9,3,8,-3,9,7,9,-2,-6,6], dtype = "int16")#candidate|1413|(13,)|const|int16
var_1414 = relay.var("var_1414", dtype = "int16", shape = (13,))#candidate|1414|(13,)|var|int16
bop_1415 = relay.bitwise_or(const_1413.astype('int16'), relay.reshape(var_1414.astype('int16'), relay.shape_of(const_1413))) # shape=(13,)
bop_1424 = relay.floor_divide(const_1413.astype('float64'), relay.reshape(bop_1415.astype('float64'), relay.shape_of(const_1413))) # shape=(13,)
output = bop_1424
output2 = bop_1424
func_1435 = relay.Function([var_1414,], output)
mod['func_1435'] = func_1435
mod = relay.transform.InferType()(mod)
mutated_mod['func_1435'] = func_1435
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1436 = relay.var("var_1436", dtype = "int16", shape = (13,))#candidate|1436|(13,)|var|int16
func_1435_call = mutated_mod.get_global_var('func_1435')
call_1437 = func_1435_call(var_1436)
output = call_1437
func_1438 = relay.Function([var_1436], output)
mutated_mod['func_1438'] = func_1438
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1443 = relay.var("var_1443", dtype = "float32", shape = ())#candidate|1443|()|var|float32
var_1444 = relay.var("var_1444", dtype = "float32", shape = (9, 6))#candidate|1444|(9, 6)|var|float32
bop_1445 = relay.floor_mod(var_1443.astype('float32'), var_1444.astype('float32')) # shape=(9, 6)
func_106_call = mod.get_global_var('func_106')
func_109_call = mutated_mod.get_global_var('func_109')
const_1456 = relay.const([-7.812240,2.854182,9.653929,-0.387348,4.036258,-5.321262,-9.893002,2.337869,3.055793,0.071289,5.993994,-1.400940,1.714328,-7.182116,-2.538299,5.937333,-4.106578,6.917908,-2.596725,-5.138904,-1.601623,4.509110,1.918304,2.036183,5.590471,1.277228,8.575676,-2.672672,-0.109046,6.562258,-7.123458,-1.117119,-5.055645,9.042076,2.045642,-0.903043], dtype = "float64")#candidate|1456|(36,)|const|float64
call_1455 = func_106_call(relay.reshape(const_1456.astype('float64'), [2, 6, 3]))
call_1457 = func_106_call(relay.reshape(const_1456.astype('float64'), [2, 6, 3]))
bop_1463 = relay.bitwise_and(const_1456.astype('int32'), relay.reshape(call_1455.astype('int32'), relay.shape_of(const_1456))) # shape=(36,)
bop_1466 = relay.bitwise_and(const_1456.astype('int32'), relay.reshape(call_1457.astype('int32'), relay.shape_of(const_1456))) # shape=(36,)
uop_1467 = relay.sigmoid(const_1456.astype('float64')) # shape=(36,)
bop_1470 = relay.mod(uop_1467.astype('float32'), relay.reshape(const_1456.astype('float32'), relay.shape_of(uop_1467))) # shape=(36,)
bop_1474 = relay.power(bop_1445.astype('float32'), var_1443.astype('float32')) # shape=(9, 6)
func_1435_call = mod.get_global_var('func_1435')
func_1438_call = mutated_mod.get_global_var('func_1438')
const_1479 = relay.const([[10,-7,-7,3,4,4,-1,6,2,1,8,4,-1]], dtype = "int16")#candidate|1479|(1, 13)|const|int16
call_1478 = func_1435_call(relay.reshape(const_1479.astype('int16'), [13,]))
call_1480 = func_1435_call(relay.reshape(const_1479.astype('int16'), [13,]))
func_863_call = mod.get_global_var('func_863')
func_865_call = mutated_mod.get_global_var('func_865')
call_1485 = relay.TupleGetItem(func_863_call(), 0)
call_1486 = relay.TupleGetItem(func_865_call(), 0)
bop_1487 = relay.bitwise_xor(bop_1463.astype('uint64'), relay.reshape(bop_1470.astype('uint64'), relay.shape_of(bop_1463))) # shape=(36,)
bop_1490 = relay.bitwise_xor(bop_1466.astype('uint64'), relay.reshape(bop_1470.astype('uint64'), relay.shape_of(bop_1466))) # shape=(36,)
bop_1493 = relay.bitwise_or(uop_1467.astype('uint16'), relay.reshape(bop_1463.astype('uint16'), relay.shape_of(uop_1467))) # shape=(36,)
bop_1496 = relay.bitwise_or(uop_1467.astype('uint16'), relay.reshape(bop_1466.astype('uint16'), relay.shape_of(uop_1467))) # shape=(36,)
output = relay.Tuple([bop_1474,call_1478,const_1479,call_1485,bop_1487,bop_1493,])
output2 = relay.Tuple([bop_1474,call_1480,const_1479,call_1486,bop_1490,bop_1496,])
func_1499 = relay.Function([var_1443,var_1444,], output)
mod['func_1499'] = func_1499
mod = relay.transform.InferType()(mod)
mutated_mod['func_1499'] = func_1499
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1499_call = mutated_mod.get_global_var('func_1499')
var_1501 = relay.var("var_1501", dtype = "float32", shape = ())#candidate|1501|()|var|float32
var_1502 = relay.var("var_1502", dtype = "float32", shape = (9, 6))#candidate|1502|(9, 6)|var|float32
call_1500 = func_1499_call(var_1501,var_1502,)
output = call_1500
func_1503 = relay.Function([var_1501,var_1502,], output)
mutated_mod['func_1503'] = func_1503
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1526 = relay.var("var_1526", dtype = "float32", shape = (16, 5, 14))#candidate|1526|(16, 5, 14)|var|float32
uop_1527 = relay.sqrt(var_1526.astype('float32')) # shape=(16, 5, 14)
bop_1546 = relay.less_equal(var_1526.astype('bool'), relay.reshape(uop_1527.astype('bool'), relay.shape_of(var_1526))) # shape=(16, 5, 14)
bop_1549 = relay.not_equal(uop_1527.astype('bool'), relay.reshape(bop_1546.astype('bool'), relay.shape_of(uop_1527))) # shape=(16, 5, 14)
func_840_call = mod.get_global_var('func_840')
func_843_call = mutated_mod.get_global_var('func_843')
const_1560 = relay.const([6.668562,-4.223707,6.854419,9.828602,-0.534728,1.097188,6.462196,9.254982,0.462627,5.248712,-0.737759,-0.324990,-8.149855,5.265469,-9.858977,0.967838,9.543980,-6.509374,-9.872089,1.855848,-6.028897,-2.938683,3.667952,8.785944,-7.406191,-9.803503,-5.961285,-8.145039,1.631062,-4.471113,-9.652804,6.125781,-0.772538,-7.565083,6.320359,1.786487,-6.060877,5.191623,-9.769931], dtype = "float32")#candidate|1560|(39,)|const|float32
var_1561 = relay.var("var_1561", dtype = "uint32", shape = (162, 2))#candidate|1561|(162, 2)|var|uint32
call_1559 = relay.TupleGetItem(func_840_call(relay.reshape(const_1560.astype('float32'), [1, 3, 13]), relay.reshape(var_1561.astype('uint32'), [18, 18]), ), 1)
call_1562 = relay.TupleGetItem(func_843_call(relay.reshape(const_1560.astype('float32'), [1, 3, 13]), relay.reshape(var_1561.astype('uint32'), [18, 18]), ), 1)
output = relay.Tuple([bop_1549,call_1559,const_1560,var_1561,])
output2 = relay.Tuple([bop_1549,call_1562,const_1560,var_1561,])
func_1569 = relay.Function([var_1526,var_1561,], output)
mod['func_1569'] = func_1569
mod = relay.transform.InferType()(mod)
mutated_mod['func_1569'] = func_1569
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1569_call = mutated_mod.get_global_var('func_1569')
var_1571 = relay.var("var_1571", dtype = "float32", shape = (16, 5, 14))#candidate|1571|(16, 5, 14)|var|float32
var_1572 = relay.var("var_1572", dtype = "uint32", shape = (162, 2))#candidate|1572|(162, 2)|var|uint32
call_1570 = func_1569_call(var_1571,var_1572,)
output = call_1570
func_1573 = relay.Function([var_1571,var_1572,], output)
mutated_mod['func_1573'] = func_1573
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1585 = relay.var("var_1585", dtype = "uint32", shape = (14, 6))#candidate|1585|(14, 6)|var|uint32
var_1586 = relay.var("var_1586", dtype = "uint32", shape = (14, 6))#candidate|1586|(14, 6)|var|uint32
bop_1587 = relay.right_shift(var_1585.astype('uint32'), relay.reshape(var_1586.astype('uint32'), relay.shape_of(var_1585))) # shape=(14, 6)
output = bop_1587
output2 = bop_1587
func_1592 = relay.Function([var_1585,var_1586,], output)
mod['func_1592'] = func_1592
mod = relay.transform.InferType()(mod)
var_1593 = relay.var("var_1593", dtype = "uint32", shape = (14, 6))#candidate|1593|(14, 6)|var|uint32
var_1594 = relay.var("var_1594", dtype = "uint32", shape = (14, 6))#candidate|1594|(14, 6)|var|uint32
output = func_1592(var_1593,var_1594,)
func_1595 = relay.Function([var_1593,var_1594,], output)
mutated_mod['func_1595'] = func_1595
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1614 = relay.var("var_1614", dtype = "int32", shape = ())#candidate|1614|()|var|int32
var_1615 = relay.var("var_1615", dtype = "int32", shape = (10, 9))#candidate|1615|(10, 9)|var|int32
bop_1616 = relay.greater_equal(var_1614.astype('bool'), var_1615.astype('bool')) # shape=(10, 9)
output = bop_1616
output2 = bop_1616
func_1623 = relay.Function([var_1614,var_1615,], output)
mod['func_1623'] = func_1623
mod = relay.transform.InferType()(mod)
mutated_mod['func_1623'] = func_1623
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1623_call = mutated_mod.get_global_var('func_1623')
var_1625 = relay.var("var_1625", dtype = "int32", shape = ())#candidate|1625|()|var|int32
var_1626 = relay.var("var_1626", dtype = "int32", shape = (10, 9))#candidate|1626|(10, 9)|var|int32
call_1624 = func_1623_call(var_1625,var_1626,)
output = call_1624
func_1627 = relay.Function([var_1625,var_1626,], output)
mutated_mod['func_1627'] = func_1627
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1132_call = mod.get_global_var('func_1132')
func_1134_call = mutated_mod.get_global_var('func_1134')
call_1638 = func_1132_call()
call_1639 = func_1132_call()
output = call_1638
output2 = call_1639
func_1644 = relay.Function([], output)
mod['func_1644'] = func_1644
mod = relay.transform.InferType()(mod)
output = func_1644()
func_1645 = relay.Function([], output)
mutated_mod['func_1645'] = func_1645
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mod.get_global_var('func_1165')
func_1167_call = mutated_mod.get_global_var('func_1167')
call_1649 = func_1165_call()
call_1650 = func_1165_call()
output = relay.Tuple([call_1649,])
output2 = relay.Tuple([call_1650,])
func_1659 = relay.Function([], output)
mod['func_1659'] = func_1659
mod = relay.transform.InferType()(mod)
output = func_1659()
func_1660 = relay.Function([], output)
mutated_mod['func_1660'] = func_1660
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mod.get_global_var('func_1165')
func_1167_call = mutated_mod.get_global_var('func_1167')
call_1669 = func_1165_call()
call_1670 = func_1165_call()
func_621_call = mod.get_global_var('func_621')
func_624_call = mutated_mod.get_global_var('func_624')
var_1682 = relay.var("var_1682", dtype = "int16", shape = (528,))#candidate|1682|(528,)|var|int16
call_1681 = relay.TupleGetItem(func_621_call(relay.reshape(var_1682.astype('int16'), [11, 8, 6]), relay.reshape(var_1682.astype('int16'), [11, 8, 6]), ), 5)
call_1683 = relay.TupleGetItem(func_624_call(relay.reshape(var_1682.astype('int16'), [11, 8, 6]), relay.reshape(var_1682.astype('int16'), [11, 8, 6]), ), 5)
output = relay.Tuple([call_1669,call_1681,var_1682,])
output2 = relay.Tuple([call_1670,call_1683,var_1682,])
func_1684 = relay.Function([var_1682,], output)
mod['func_1684'] = func_1684
mod = relay.transform.InferType()(mod)
var_1685 = relay.var("var_1685", dtype = "int16", shape = (528,))#candidate|1685|(528,)|var|int16
output = func_1684(var_1685)
func_1686 = relay.Function([var_1685], output)
mutated_mod['func_1686'] = func_1686
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1096_call = mod.get_global_var('func_1096')
func_1097_call = mutated_mod.get_global_var('func_1097')
call_1688 = relay.TupleGetItem(func_1096_call(), 0)
call_1689 = relay.TupleGetItem(func_1097_call(), 0)
output = relay.Tuple([call_1688,])
output2 = relay.Tuple([call_1689,])
func_1697 = relay.Function([], output)
mod['func_1697'] = func_1697
mod = relay.transform.InferType()(mod)
output = func_1697()
func_1698 = relay.Function([], output)
mutated_mod['func_1698'] = func_1698
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1714 = relay.const([[[-2.439399,1.966011,3.234873,-8.767723,-6.084091,0.551821],[7.131617,9.588070,-1.723964,-8.588765,-6.530595,-3.011481],[4.811336,-4.846191,6.748924,4.875340,-0.375043,6.891218],[6.307306,-9.495956,4.536325,-5.900057,3.612610,-3.815439],[-0.298413,-1.636340,-6.403978,1.535201,8.928458,5.301260],[-8.883687,-2.483512,-1.577602,-7.265303,-3.195987,6.394199]],[[-4.714568,-0.308130,9.633128,7.875097,-8.629276,-6.258176],[-0.880495,-5.118921,-4.595119,-4.618642,-8.492300,8.372714],[8.349264,-1.424690,-8.286944,8.667336,-7.758714,9.132299],[-8.209058,6.618394,-3.396631,4.681897,-3.460718,-4.660948],[-8.966426,2.380355,-8.900539,5.042905,-8.585915,6.538302],[7.003900,-9.345054,-2.036410,8.324173,-9.809880,0.917047]],[[-2.856860,-8.685423,2.618739,5.882311,1.158733,-4.915172],[5.570698,8.752722,-1.910094,8.053320,3.767819,3.725438],[6.904088,9.118325,4.763337,-3.887603,2.610502,-9.077629],[8.673925,-9.127995,-5.352180,-0.307430,-8.493283,-8.699537],[-7.477484,-9.551185,4.230740,-7.920438,-1.625079,-4.154311],[-4.127298,-2.077621,7.280786,-0.560257,-9.370154,-2.639711]],[[-6.235260,0.531796,-3.844687,8.826858,3.027791,1.171274],[6.903224,4.461049,0.493119,0.359277,-7.383995,8.547830],[0.055165,-8.453456,2.504124,-6.090066,-7.302928,-1.821164],[8.923367,0.708096,-0.552418,0.265111,-4.723953,6.005639],[5.862753,-5.644492,-9.673182,0.104119,9.331098,-3.646235],[-4.423385,-0.446498,-5.126205,-1.544991,-4.938664,-1.057647]],[[-3.011479,-4.844374,-6.671403,-6.025564,8.300899,7.707021],[-5.547041,4.799956,-6.617642,-4.476729,-8.921090,9.815427],[-4.091853,6.468771,-0.965410,1.560854,4.267142,8.832654],[5.315092,2.172932,-9.029563,4.363040,-7.414524,0.051042],[-1.022600,5.754875,-4.066217,6.960535,-4.269330,-6.909640],[-6.057409,4.782997,1.754736,0.783909,-6.516891,6.279691]],[[6.008922,-7.662609,4.761251,-4.901810,7.589004,2.221071],[0.138538,-6.443517,0.079377,-1.753847,-2.361565,4.930488],[-5.691028,6.091365,-9.629278,-3.342985,5.097387,7.869398],[-7.269041,-4.668738,3.392705,4.039728,-3.791802,7.819506],[-5.209430,3.991728,8.190225,7.013931,8.705734,5.354313],[2.245571,4.502864,-1.510194,-9.676958,-0.473601,-8.200305]]], dtype = "float64")#candidate|1714|(6, 6, 6)|const|float64
uop_1715 = relay.cosh(const_1714.astype('float64')) # shape=(6, 6, 6)
uop_1721 = relay.sin(uop_1715.astype('float64')) # shape=(6, 6, 6)
bop_1723 = relay.logical_and(uop_1721.astype('bool'), relay.reshape(uop_1715.astype('bool'), relay.shape_of(uop_1721))) # shape=(6, 6, 6)
bop_1736 = relay.not_equal(uop_1715.astype('bool'), relay.reshape(const_1714.astype('bool'), relay.shape_of(uop_1715))) # shape=(6, 6, 6)
var_1745 = relay.var("var_1745", dtype = "float64", shape = (6, 6, 6))#candidate|1745|(6, 6, 6)|var|float64
bop_1746 = relay.less(uop_1721.astype('bool'), relay.reshape(var_1745.astype('bool'), relay.shape_of(uop_1721))) # shape=(6, 6, 6)
bop_1754 = relay.power(uop_1715.astype('float32'), relay.reshape(bop_1736.astype('float32'), relay.shape_of(uop_1715))) # shape=(6, 6, 6)
func_1659_call = mod.get_global_var('func_1659')
func_1660_call = mutated_mod.get_global_var('func_1660')
call_1757 = relay.TupleGetItem(func_1659_call(), 0)
call_1758 = relay.TupleGetItem(func_1660_call(), 0)
var_1760 = relay.var("var_1760", dtype = "bool", shape = (6, 6, 6))#candidate|1760|(6, 6, 6)|var|bool
bop_1761 = relay.greater_equal(bop_1723.astype('bool'), relay.reshape(var_1760.astype('bool'), relay.shape_of(bop_1723))) # shape=(6, 6, 6)
output = relay.Tuple([bop_1746,bop_1754,call_1757,bop_1761,])
output2 = relay.Tuple([bop_1746,bop_1754,call_1758,bop_1761,])
func_1764 = relay.Function([var_1745,var_1760,], output)
mod['func_1764'] = func_1764
mod = relay.transform.InferType()(mod)
var_1765 = relay.var("var_1765", dtype = "float64", shape = (6, 6, 6))#candidate|1765|(6, 6, 6)|var|float64
var_1766 = relay.var("var_1766", dtype = "bool", shape = (6, 6, 6))#candidate|1766|(6, 6, 6)|var|bool
output = func_1764(var_1765,var_1766,)
func_1767 = relay.Function([var_1765,var_1766,], output)
mutated_mod['func_1767'] = func_1767
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1792 = relay.var("var_1792", dtype = "float32", shape = (16, 16))#candidate|1792|(16, 16)|var|float32
var_1793 = relay.var("var_1793", dtype = "float32", shape = (16, 16))#candidate|1793|(16, 16)|var|float32
bop_1794 = relay.minimum(var_1792.astype('float32'), relay.reshape(var_1793.astype('float32'), relay.shape_of(var_1792))) # shape=(16, 16)
output = relay.Tuple([bop_1794,])
output2 = relay.Tuple([bop_1794,])
func_1801 = relay.Function([var_1792,var_1793,], output)
mod['func_1801'] = func_1801
mod = relay.transform.InferType()(mod)
var_1802 = relay.var("var_1802", dtype = "float32", shape = (16, 16))#candidate|1802|(16, 16)|var|float32
var_1803 = relay.var("var_1803", dtype = "float32", shape = (16, 16))#candidate|1803|(16, 16)|var|float32
output = func_1801(var_1802,var_1803,)
func_1804 = relay.Function([var_1802,var_1803,], output)
mutated_mod['func_1804'] = func_1804
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1872 = relay.var("var_1872", dtype = "float32", shape = (13, 9))#candidate|1872|(13, 9)|var|float32
uop_1873 = relay.rsqrt(var_1872.astype('float32')) # shape=(13, 9)
func_1569_call = mod.get_global_var('func_1569')
func_1573_call = mutated_mod.get_global_var('func_1573')
var_1877 = relay.var("var_1877", dtype = "float32", shape = (1120, 1))#candidate|1877|(1120, 1)|var|float32
var_1878 = relay.var("var_1878", dtype = "uint32", shape = (324,))#candidate|1878|(324,)|var|uint32
call_1876 = relay.TupleGetItem(func_1569_call(relay.reshape(var_1877.astype('float32'), [16, 5, 14]), relay.reshape(var_1878.astype('uint32'), [162, 2]), ), 0)
call_1879 = relay.TupleGetItem(func_1573_call(relay.reshape(var_1877.astype('float32'), [16, 5, 14]), relay.reshape(var_1878.astype('uint32'), [162, 2]), ), 0)
output = relay.Tuple([uop_1873,call_1876,var_1877,var_1878,])
output2 = relay.Tuple([uop_1873,call_1879,var_1877,var_1878,])
func_1883 = relay.Function([var_1872,var_1877,var_1878,], output)
mod['func_1883'] = func_1883
mod = relay.transform.InferType()(mod)
mutated_mod['func_1883'] = func_1883
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1883_call = mutated_mod.get_global_var('func_1883')
var_1885 = relay.var("var_1885", dtype = "float32", shape = (13, 9))#candidate|1885|(13, 9)|var|float32
var_1886 = relay.var("var_1886", dtype = "float32", shape = (1120, 1))#candidate|1886|(1120, 1)|var|float32
var_1887 = relay.var("var_1887", dtype = "uint32", shape = (324,))#candidate|1887|(324,)|var|uint32
call_1884 = func_1883_call(var_1885,var_1886,var_1887,)
output = call_1884
func_1888 = relay.Function([var_1885,var_1886,var_1887,], output)
mutated_mod['func_1888'] = func_1888
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mod.get_global_var('func_1165')
func_1167_call = mutated_mod.get_global_var('func_1167')
call_1890 = func_1165_call()
call_1891 = func_1165_call()
var_1897 = relay.var("var_1897", dtype = "int64", shape = (9, 8, 7))#candidate|1897|(9, 8, 7)|var|int64
bop_1898 = relay.less_equal(call_1890.astype('bool'), relay.reshape(var_1897.astype('bool'), relay.shape_of(call_1890))) # shape=(9, 8, 7)
bop_1901 = relay.less_equal(call_1891.astype('bool'), relay.reshape(var_1897.astype('bool'), relay.shape_of(call_1891))) # shape=(9, 8, 7)
uop_1902 = relay.atanh(bop_1898.astype('float32')) # shape=(9, 8, 7)
uop_1904 = relay.atanh(bop_1901.astype('float32')) # shape=(9, 8, 7)
bop_1906 = relay.greater(uop_1902.astype('bool'), relay.reshape(bop_1898.astype('bool'), relay.shape_of(uop_1902))) # shape=(9, 8, 7)
bop_1909 = relay.greater(uop_1904.astype('bool'), relay.reshape(bop_1901.astype('bool'), relay.shape_of(uop_1904))) # shape=(9, 8, 7)
bop_1910 = relay.maximum(bop_1906.astype('float64'), relay.reshape(var_1897.astype('float64'), relay.shape_of(bop_1906))) # shape=(9, 8, 7)
bop_1913 = relay.maximum(bop_1909.astype('float64'), relay.reshape(var_1897.astype('float64'), relay.shape_of(bop_1909))) # shape=(9, 8, 7)
bop_1914 = relay.multiply(bop_1906.astype('uint64'), relay.reshape(bop_1910.astype('uint64'), relay.shape_of(bop_1906))) # shape=(9, 8, 7)
bop_1917 = relay.multiply(bop_1909.astype('uint64'), relay.reshape(bop_1913.astype('uint64'), relay.shape_of(bop_1909))) # shape=(9, 8, 7)
uop_1918 = relay.log(call_1890.astype('float64')) # shape=(9, 8, 7)
uop_1920 = relay.log(call_1891.astype('float64')) # shape=(9, 8, 7)
bop_1924 = relay.logical_xor(bop_1914.astype('uint32'), relay.reshape(bop_1910.astype('uint32'), relay.shape_of(bop_1914))) # shape=(9, 8, 7)
bop_1927 = relay.logical_xor(bop_1917.astype('uint32'), relay.reshape(bop_1913.astype('uint32'), relay.shape_of(bop_1917))) # shape=(9, 8, 7)
bop_1928 = relay.bitwise_and(uop_1902.astype('uint16'), relay.reshape(uop_1918.astype('uint16'), relay.shape_of(uop_1902))) # shape=(9, 8, 7)
bop_1931 = relay.bitwise_and(uop_1904.astype('uint16'), relay.reshape(uop_1920.astype('uint16'), relay.shape_of(uop_1904))) # shape=(9, 8, 7)
output = relay.Tuple([bop_1924,bop_1928,])
output2 = relay.Tuple([bop_1927,bop_1931,])
func_1933 = relay.Function([var_1897,], output)
mod['func_1933'] = func_1933
mod = relay.transform.InferType()(mod)
var_1934 = relay.var("var_1934", dtype = "int64", shape = (9, 8, 7))#candidate|1934|(9, 8, 7)|var|int64
output = func_1933(var_1934)
func_1935 = relay.Function([var_1934], output)
mutated_mod['func_1935'] = func_1935
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1001_call = mod.get_global_var('func_1001')
func_1002_call = mutated_mod.get_global_var('func_1002')
call_1937 = relay.TupleGetItem(func_1001_call(), 0)
call_1938 = relay.TupleGetItem(func_1002_call(), 0)
output = call_1937
output2 = call_1938
func_1956 = relay.Function([], output)
mod['func_1956'] = func_1956
mod = relay.transform.InferType()(mod)
mutated_mod['func_1956'] = func_1956
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1956_call = mutated_mod.get_global_var('func_1956')
call_1957 = func_1956_call()
output = call_1957
func_1958 = relay.Function([], output)
mutated_mod['func_1958'] = func_1958
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1969 = relay.var("var_1969", dtype = "bool", shape = (5, 10))#candidate|1969|(5, 10)|var|bool
const_1970 = relay.const([[False,True,False,True,False,True,False,False,True,False],[True,False,False,True,True,False,False,False,True,True],[True,True,True,True,False,True,True,False,True,False],[False,True,True,False,False,True,False,False,False,True],[False,True,True,False,True,True,False,True,True,False]], dtype = "bool")#candidate|1970|(5, 10)|const|bool
bop_1971 = relay.logical_or(var_1969.astype('bool'), relay.reshape(const_1970.astype('bool'), relay.shape_of(var_1969))) # shape=(5, 10)
const_1977 = relay.const([[False,True,False,True,False,True,True,True,True,True],[True,False,True,False,False,False,False,False,False,False],[False,True,False,True,False,True,False,True,False,False],[True,False,False,False,False,False,True,True,False,False],[False,True,False,True,False,False,True,False,False,True]], dtype = "bool")#candidate|1977|(5, 10)|const|bool
bop_1978 = relay.greater_equal(bop_1971.astype('bool'), relay.reshape(const_1977.astype('bool'), relay.shape_of(bop_1971))) # shape=(5, 10)
const_1981 = relay.const([[True,False,False,True,True,False,True,True,False,False],[False,True,False,False,False,True,True,True,True,True],[True,False,False,False,True,True,True,False,True,True],[False,False,True,False,False,True,False,True,False,True],[True,True,False,True,True,False,True,False,False,False]], dtype = "bool")#candidate|1981|(5, 10)|const|bool
bop_1982 = relay.add(bop_1971.astype('float64'), relay.reshape(const_1981.astype('float64'), relay.shape_of(bop_1971))) # shape=(5, 10)
output = relay.Tuple([bop_1978,bop_1982,])
output2 = relay.Tuple([bop_1978,bop_1982,])
func_2004 = relay.Function([var_1969,], output)
mod['func_2004'] = func_2004
mod = relay.transform.InferType()(mod)
var_2005 = relay.var("var_2005", dtype = "bool", shape = (5, 10))#candidate|2005|(5, 10)|var|bool
output = func_2004(var_2005)
func_2006 = relay.Function([var_2005], output)
mutated_mod['func_2006'] = func_2006
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2040 = relay.var("var_2040", dtype = "uint32", shape = ())#candidate|2040|()|var|uint32
var_2041 = relay.var("var_2041", dtype = "uint32", shape = (11, 7))#candidate|2041|(11, 7)|var|uint32
bop_2042 = relay.greater_equal(var_2040.astype('bool'), var_2041.astype('bool')) # shape=(11, 7)
output = relay.Tuple([bop_2042,])
output2 = relay.Tuple([bop_2042,])
func_2052 = relay.Function([var_2040,var_2041,], output)
mod['func_2052'] = func_2052
mod = relay.transform.InferType()(mod)
var_2053 = relay.var("var_2053", dtype = "uint32", shape = ())#candidate|2053|()|var|uint32
var_2054 = relay.var("var_2054", dtype = "uint32", shape = (11, 7))#candidate|2054|(11, 7)|var|uint32
output = func_2052(var_2053,var_2054,)
func_2055 = relay.Function([var_2053,var_2054,], output)
mutated_mod['func_2055'] = func_2055
mutated_mod = relay.transform.InferType()(mutated_mod)
func_863_call = mod.get_global_var('func_863')
func_865_call = mutated_mod.get_global_var('func_865')
call_2080 = relay.TupleGetItem(func_863_call(), 0)
call_2081 = relay.TupleGetItem(func_865_call(), 0)
func_1684_call = mod.get_global_var('func_1684')
func_1686_call = mutated_mod.get_global_var('func_1686')
var_2092 = relay.var("var_2092", dtype = "int16", shape = (528,))#candidate|2092|(528,)|var|int16
call_2091 = relay.TupleGetItem(func_1684_call(relay.reshape(var_2092.astype('int16'), [528,])), 2)
call_2093 = relay.TupleGetItem(func_1686_call(relay.reshape(var_2092.astype('int16'), [528,])), 2)
func_1684_call = mod.get_global_var('func_1684')
func_1686_call = mutated_mod.get_global_var('func_1686')
call_2098 = relay.TupleGetItem(func_1684_call(relay.reshape(call_2091.astype('int16'), [528,])), 2)
call_2099 = relay.TupleGetItem(func_1686_call(relay.reshape(call_2091.astype('int16'), [528,])), 2)
output = relay.Tuple([call_2080,call_2091,var_2092,call_2098,])
output2 = relay.Tuple([call_2081,call_2093,var_2092,call_2099,])
func_2113 = relay.Function([var_2092,], output)
mod['func_2113'] = func_2113
mod = relay.transform.InferType()(mod)
mutated_mod['func_2113'] = func_2113
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2114 = relay.var("var_2114", dtype = "int16", shape = (528,))#candidate|2114|(528,)|var|int16
func_2113_call = mutated_mod.get_global_var('func_2113')
call_2115 = func_2113_call(var_2114)
output = call_2115
func_2116 = relay.Function([var_2114], output)
mutated_mod['func_2116'] = func_2116
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1352_call = mod.get_global_var('func_1352')
func_1354_call = mutated_mod.get_global_var('func_1354')
call_2118 = relay.TupleGetItem(func_1352_call(), 0)
call_2119 = relay.TupleGetItem(func_1354_call(), 0)
const_2124 = relay.const([[-9,-8],[-3,6],[9,-5]], dtype = "uint8")#candidate|2124|(3, 2)|const|uint8
bop_2125 = relay.logical_or(call_2118.astype('bool'), relay.reshape(const_2124.astype('bool'), relay.shape_of(call_2118))) # shape=(3, 2)
bop_2128 = relay.logical_or(call_2119.astype('bool'), relay.reshape(const_2124.astype('bool'), relay.shape_of(call_2119))) # shape=(3, 2)
func_1001_call = mod.get_global_var('func_1001')
func_1002_call = mutated_mod.get_global_var('func_1002')
call_2136 = relay.TupleGetItem(func_1001_call(), 0)
call_2137 = relay.TupleGetItem(func_1002_call(), 0)
uop_2142 = relay.tan(const_2124.astype('float64')) # shape=(3, 2)
func_1933_call = mod.get_global_var('func_1933')
func_1935_call = mutated_mod.get_global_var('func_1935')
var_2152 = relay.var("var_2152", dtype = "int64", shape = (252, 2))#candidate|2152|(252, 2)|var|int64
call_2151 = relay.TupleGetItem(func_1933_call(relay.reshape(var_2152.astype('int64'), [9, 8, 7])), 0)
call_2153 = relay.TupleGetItem(func_1935_call(relay.reshape(var_2152.astype('int64'), [9, 8, 7])), 0)
bop_2155 = relay.left_shift(uop_2142.astype('int8'), relay.reshape(call_2136.astype('int8'), relay.shape_of(uop_2142))) # shape=(3, 2)
bop_2158 = relay.left_shift(uop_2142.astype('int8'), relay.reshape(call_2137.astype('int8'), relay.shape_of(uop_2142))) # shape=(3, 2)
uop_2164 = relay.asin(uop_2142.astype('float64')) # shape=(3, 2)
bop_2168 = relay.floor_divide(call_2151.astype('float64'), relay.reshape(var_2152.astype('float64'), relay.shape_of(call_2151))) # shape=(9, 8, 7)
bop_2171 = relay.floor_divide(call_2153.astype('float64'), relay.reshape(var_2152.astype('float64'), relay.shape_of(call_2153))) # shape=(9, 8, 7)
uop_2175 = relay.atanh(uop_2164.astype('float32')) # shape=(3, 2)
bop_2177 = relay.not_equal(uop_2175.astype('bool'), relay.reshape(uop_2142.astype('bool'), relay.shape_of(uop_2175))) # shape=(3, 2)
bop_2180 = relay.bitwise_xor(uop_2164.astype('uint8'), relay.reshape(uop_2175.astype('uint8'), relay.shape_of(uop_2164))) # shape=(3, 2)
output = relay.Tuple([bop_2125,bop_2155,bop_2168,bop_2177,bop_2180,])
output2 = relay.Tuple([bop_2128,bop_2158,bop_2171,bop_2177,bop_2180,])
func_2183 = relay.Function([var_2152,], output)
mod['func_2183'] = func_2183
mod = relay.transform.InferType()(mod)
var_2184 = relay.var("var_2184", dtype = "int64", shape = (252, 2))#candidate|2184|(252, 2)|var|int64
output = func_2183(var_2184)
func_2185 = relay.Function([var_2184], output)
mutated_mod['func_2185'] = func_2185
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2193 = relay.var("var_2193", dtype = "float64", shape = (5, 13, 9))#candidate|2193|(5, 13, 9)|var|float64
var_2194 = relay.var("var_2194", dtype = "float64", shape = (5, 13, 9))#candidate|2194|(5, 13, 9)|var|float64
bop_2195 = relay.floor_mod(var_2193.astype('float64'), relay.reshape(var_2194.astype('float64'), relay.shape_of(var_2193))) # shape=(5, 13, 9)
uop_2206 = relay.log(bop_2195.astype('float32')) # shape=(5, 13, 9)
output = uop_2206
output2 = uop_2206
func_2209 = relay.Function([var_2193,var_2194,], output)
mod['func_2209'] = func_2209
mod = relay.transform.InferType()(mod)
var_2210 = relay.var("var_2210", dtype = "float64", shape = (5, 13, 9))#candidate|2210|(5, 13, 9)|var|float64
var_2211 = relay.var("var_2211", dtype = "float64", shape = (5, 13, 9))#candidate|2211|(5, 13, 9)|var|float64
output = func_2209(var_2210,var_2211,)
func_2212 = relay.Function([var_2210,var_2211,], output)
mutated_mod['func_2212'] = func_2212
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1132_call = mod.get_global_var('func_1132')
func_1134_call = mutated_mod.get_global_var('func_1134')
call_2226 = func_1132_call()
call_2227 = func_1132_call()
output = call_2226
output2 = call_2227
func_2232 = relay.Function([], output)
mod['func_2232'] = func_2232
mod = relay.transform.InferType()(mod)
output = func_2232()
func_2233 = relay.Function([], output)
mutated_mod['func_2233'] = func_2233
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2247 = relay.var("var_2247", dtype = "uint64", shape = (2, 15))#candidate|2247|(2, 15)|var|uint64
var_2248 = relay.var("var_2248", dtype = "uint64", shape = (2, 15))#candidate|2248|(2, 15)|var|uint64
bop_2249 = relay.bitwise_and(var_2247.astype('uint64'), relay.reshape(var_2248.astype('uint64'), relay.shape_of(var_2247))) # shape=(2, 15)
output = bop_2249
output2 = bop_2249
func_2252 = relay.Function([var_2247,var_2248,], output)
mod['func_2252'] = func_2252
mod = relay.transform.InferType()(mod)
mutated_mod['func_2252'] = func_2252
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2252_call = mutated_mod.get_global_var('func_2252')
var_2254 = relay.var("var_2254", dtype = "uint64", shape = (2, 15))#candidate|2254|(2, 15)|var|uint64
var_2255 = relay.var("var_2255", dtype = "uint64", shape = (2, 15))#candidate|2255|(2, 15)|var|uint64
call_2253 = func_2252_call(var_2254,var_2255,)
output = call_2253
func_2256 = relay.Function([var_2254,var_2255,], output)
mutated_mod['func_2256'] = func_2256
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2275 = relay.var("var_2275", dtype = "float64", shape = (2, 2, 10))#candidate|2275|(2, 2, 10)|var|float64
uop_2276 = relay.sinh(var_2275.astype('float64')) # shape=(2, 2, 10)
output = uop_2276
output2 = uop_2276
F = relay.Function([var_2275,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_2275,], output2)
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
input_2275= np.array([[[-2.978758,0.066825,-8.754888,-7.698129,2.292354,7.123978,9.158082,-6.358058,9.716337,-8.050844],[7.182404,-4.617329,-2.130258,-1.741990,2.386786,-6.037910,-6.826571,-5.326790,7.847593,7.816300]],[[1.529314,-2.543148,2.680133,-3.099907,-2.942851,-4.265314,1.953852,-6.365265,-8.253030,2.294527],[2.836444,2.058558,8.880301,-5.877353,-2.129083,-3.381968,7.568285,-4.489116,3.007816,1.603092]]], dtype='float64')
module1.set_input('var_2275', input_2275)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_2275, )
res3 = intrp3.evaluate()(input_2275, )
res4 = intrp4.evaluate()(input_2275, )
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
module5.set_input('var_2275', input_2275)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_2275, )
res7 = intrp7.evaluate()(input_2275, )
res8 = intrp8.evaluate()(input_2275, )
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
module9.set_input('var_2275', input_2275)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_2275, )
res11 = intrp11.evaluate()(input_2275, )
res12 = intrp12.evaluate()(input_2275, )
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
module13.set_input('var_2275', input_2275)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_2275, )
res15 = intrp15.evaluate()(input_2275, )
res16 = intrp16.evaluate()(input_2275, )
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
module17.set_input('var_2275', input_2275)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_2275, )
res19 = intrp19.evaluate()(input_2275, )
res20 = intrp20.evaluate()(input_2275, )
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
module21.set_input('var_2275', input_2275)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_2275, )
res23 = intrp23.evaluate()(input_2275, )
res24 = intrp24.evaluate()(input_2275, )
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

'''43: TVMFuncCall
42: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
41: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
40: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
39: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
38: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
37: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
36: tvm::transform::Pass::operator()(tvm::IRModule) const
35: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
34: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
30: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
29: tvm::transform::Pass::operator()(tvm::IRModule) const
28: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
27: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
26: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
25: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
24: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
22: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
21: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
20: _ZN3tvm5relay9transform22Devic
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
18: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
17: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
15: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
14: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
11: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
9: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
8: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
7: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
6: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
5: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
3: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
2: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
1: tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
0: tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)

'''