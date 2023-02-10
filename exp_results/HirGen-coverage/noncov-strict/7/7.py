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
var_0 = relay.var("var_0", dtype = "float32", shape = ())#candidate|0|()|var|float32
uop_1 = relay.cosh(var_0.astype('float32')) # shape=()
var_3 = relay.var("var_3", dtype = "float32", shape = ())#candidate|3|()|var|float32
bop_4 = relay.maximum(uop_1.astype('int16'), var_3.astype('int16')) # shape=()
uop_7 = relay.tan(bop_4.astype('float32')) # shape=()
uop_9 = relay.cosh(uop_7.astype('float64')) # shape=()
uop_11 = relay.cosh(uop_7.astype('float32')) # shape=()
bop_13 = relay.mod(uop_11.astype('float32'), var_3.astype('float32')) # shape=()
bop_16 = relay.multiply(uop_9.astype('uint8'), uop_7.astype('uint8')) # shape=()
var_19 = relay.var("var_19", dtype = "float32", shape = (2, 10, 12))#candidate|19|(2, 10, 12)|var|float32
bop_20 = relay.not_equal(var_3.astype('bool'), var_19.astype('bool')) # shape=(2, 10, 12)
bop_23 = relay.greater(bop_16.astype('bool'), uop_1.astype('bool')) # shape=()
uop_26 = relay.erf(var_3.astype('float64')) # shape=()
var_28 = relay.var("var_28", dtype = "float32", shape = ())#candidate|28|()|var|float32
bop_29 = relay.not_equal(uop_11.astype('bool'), var_28.astype('bool')) # shape=()
bop_32 = relay.not_equal(uop_7.astype('bool'), uop_26.astype('bool')) # shape=()
uop_35 = relay.log2(uop_11.astype('float64')) # shape=()
uop_37 = relay.sinh(bop_32.astype('float32')) # shape=()
uop_39 = relay.acosh(uop_35.astype('float32')) # shape=()
bop_41 = relay.power(uop_11.astype('float64'), bop_16.astype('float64')) # shape=()
var_44 = relay.var("var_44", dtype = "float64", shape = (5, 8))#candidate|44|(5, 8)|var|float64
bop_45 = relay.greater_equal(uop_35.astype('bool'), var_44.astype('bool')) # shape=(5, 8)
bop_48 = relay.logical_or(uop_35.astype('bool'), uop_37.astype('bool')) # shape=()
uop_51 = relay.rsqrt(uop_39.astype('float32')) # shape=()
bop_53 = relay.multiply(uop_51.astype('int32'), uop_35.astype('int32')) # shape=()
var_56 = relay.var("var_56", dtype = "float32", shape = (11,))#candidate|56|(11,)|var|float32
bop_57 = relay.less(uop_39.astype('bool'), var_56.astype('bool')) # shape=(11,)
bop_60 = relay.floor_mod(bop_48.astype('float64'), var_19.astype('float64')) # shape=(2, 10, 12)
bop_63 = relay.floor_mod(bop_16.astype('float64'), var_56.astype('float64')) # shape=(11,)
bop_66 = relay.less(uop_51.astype('bool'), bop_57.astype('bool')) # shape=(11,)
bop_69 = relay.greater(uop_51.astype('bool'), uop_7.astype('bool')) # shape=()
bop_72 = relay.floor_mod(bop_63.astype('float64'), uop_37.astype('float64')) # shape=(11,)
uop_75 = relay.sqrt(bop_53.astype('float64')) # shape=()
bop_77 = relay.maximum(uop_75.astype('int64'), bop_66.astype('int64')) # shape=(11,)
uop_80 = relay.sin(bop_77.astype('float32')) # shape=(11,)
bop_82 = relay.maximum(uop_80.astype('int16'), uop_39.astype('int16')) # shape=(11,)
bop_85 = relay.floor_mod(bop_72.astype('float64'), uop_1.astype('float64')) # shape=(11,)
bop_88 = relay.equal(bop_82.astype('bool'), uop_9.astype('bool')) # shape=(11,)
bop_91 = relay.mod(bop_77.astype('float32'), relay.reshape(bop_72.astype('float32'), relay.shape_of(bop_77))) # shape=(11,)
uop_94 = relay.tan(bop_88.astype('float64')) # shape=(11,)
uop_96 = relay.acosh(uop_94.astype('float64')) # shape=(11,)
uop_98 = relay.sqrt(uop_96.astype('float64')) # shape=(11,)
bop_100 = relay.not_equal(uop_94.astype('bool'), relay.reshape(uop_80.astype('bool'), relay.shape_of(uop_94))) # shape=(11,)
bop_103 = relay.bitwise_and(uop_96.astype('uint64'), bop_48.astype('uint64')) # shape=(11,)
uop_106 = relay.cos(uop_98.astype('float64')) # shape=(11,)
uop_108 = relay.rsqrt(uop_106.astype('float64')) # shape=(11,)
uop_110 = relay.rsqrt(uop_106.astype('float32')) # shape=(11,)
uop_112 = relay.log10(uop_110.astype('float32')) # shape=(11,)
var_114 = relay.var("var_114", dtype = "float32", shape = (11,))#candidate|114|(11,)|var|float32
bop_115 = relay.maximum(uop_112.astype('uint32'), relay.reshape(var_114.astype('uint32'), relay.shape_of(uop_112))) # shape=(11,)
bop_118 = relay.floor_mod(bop_115.astype('float32'), bop_48.astype('float32')) # shape=(11,)
uop_121 = relay.log2(uop_108.astype('float32')) # shape=(11,)
bop_123 = relay.bitwise_and(uop_108.astype('int16'), var_3.astype('int16')) # shape=(11,)
uop_126 = relay.sin(bop_123.astype('float32')) # shape=(11,)
bop_128 = relay.bitwise_and(uop_110.astype('uint8'), relay.reshape(uop_126.astype('uint8'), relay.shape_of(uop_110))) # shape=(11,)
bop_131 = relay.divide(uop_112.astype('float32'), relay.reshape(bop_103.astype('float32'), relay.shape_of(uop_112))) # shape=(11,)
bop_134 = relay.greater_equal(uop_121.astype('bool'), relay.reshape(bop_77.astype('bool'), relay.shape_of(uop_121))) # shape=(11,)
uop_137 = relay.acos(uop_108.astype('float64')) # shape=(11,)
var_139 = relay.var("var_139", dtype = "float32", shape = (11,))#candidate|139|(11,)|var|float32
bop_140 = relay.multiply(bop_118.astype('float64'), relay.reshape(var_139.astype('float64'), relay.shape_of(bop_118))) # shape=(11,)
bop_143 = relay.logical_and(uop_126.astype('bool'), var_0.astype('bool')) # shape=(11,)
uop_146 = relay.rsqrt(bop_131.astype('float32')) # shape=(11,)
var_148 = relay.var("var_148", dtype = "float32", shape = (11,))#candidate|148|(11,)|var|float32
bop_149 = relay.bitwise_and(uop_146.astype('uint64'), relay.reshape(var_148.astype('uint64'), relay.shape_of(uop_146))) # shape=(11,)
bop_152 = relay.not_equal(uop_98.astype('bool'), var_0.astype('bool')) # shape=(11,)
bop_155 = relay.left_shift(uop_106.astype('uint8'), var_0.astype('uint8')) # shape=(11,)
uop_158 = relay.asin(bop_134.astype('float64')) # shape=(11,)
uop_160 = relay.cos(uop_146.astype('float64')) # shape=(11,)
bop_162 = relay.left_shift(uop_160.astype('int32'), uop_35.astype('int32')) # shape=(11,)
bop_165 = relay.logical_or(uop_160.astype('bool'), relay.reshape(bop_149.astype('bool'), relay.shape_of(uop_160))) # shape=(11,)
uop_168 = relay.atan(uop_110.astype('float32')) # shape=(11,)
uop_170 = relay.rsqrt(bop_162.astype('float64')) # shape=(11,)
var_172 = relay.var("var_172", dtype = "float64", shape = (11,))#candidate|172|(11,)|var|float64
bop_173 = relay.add(uop_170.astype('uint64'), relay.reshape(var_172.astype('uint64'), relay.shape_of(uop_170))) # shape=(11,)
uop_176 = relay.sinh(uop_158.astype('float64')) # shape=(11,)
bop_178 = relay.add(uop_176.astype('float64'), relay.reshape(var_172.astype('float64'), relay.shape_of(uop_176))) # shape=(11,)
bop_181 = relay.divide(uop_110.astype('float32'), uop_37.astype('float32')) # shape=(11,)
bop_184 = relay.add(uop_160.astype('float32'), relay.reshape(bop_181.astype('float32'), relay.shape_of(uop_160))) # shape=(11,)
uop_187 = relay.log(bop_162.astype('float64')) # shape=(11,)
uop_189 = relay.atan(uop_170.astype('float32')) # shape=(11,)
var_191 = relay.var("var_191", dtype = "float32", shape = (11,))#candidate|191|(11,)|var|float32
bop_192 = relay.logical_and(uop_189.astype('bool'), relay.reshape(var_191.astype('bool'), relay.shape_of(uop_189))) # shape=(11,)
uop_195 = relay.sin(bop_192.astype('float32')) # shape=(11,)
const_197 = relay.const([-2.739637,-2.189355,6.916444,-8.513802,-2.814204,9.311849,-8.304596,-0.821931,-4.702148,3.280049,1.785388], dtype = "float32")#candidate|197|(11,)|const|float32
bop_198 = relay.greater_equal(uop_189.astype('bool'), relay.reshape(const_197.astype('bool'), relay.shape_of(uop_189))) # shape=(11,)
bop_201 = relay.logical_and(uop_195.astype('bool'), relay.reshape(bop_165.astype('bool'), relay.shape_of(uop_195))) # shape=(11,)
var_204 = relay.var("var_204", dtype = "bool", shape = (11,))#candidate|204|(11,)|var|bool
bop_205 = relay.bitwise_and(bop_201.astype('uint16'), relay.reshape(var_204.astype('uint16'), relay.shape_of(bop_201))) # shape=(11,)
var_208 = relay.var("var_208", dtype = "uint64", shape = (11,))#candidate|208|(11,)|var|uint64
bop_209 = relay.logical_xor(bop_173.astype('uint64'), relay.reshape(var_208.astype('uint64'), relay.shape_of(bop_173))) # shape=(11,)
bop_212 = relay.maximum(bop_205.astype('float64'), relay.reshape(bop_201.astype('float64'), relay.shape_of(bop_205))) # shape=(11,)
bop_215 = relay.maximum(bop_149.astype('float32'), bop_13.astype('float32')) # shape=(11,)
var_218 = relay.var("var_218", dtype = "bool", shape = (11,))#candidate|218|(11,)|var|bool
bop_219 = relay.floor_divide(bop_192.astype('float64'), relay.reshape(var_218.astype('float64'), relay.shape_of(bop_192))) # shape=(11,)
uop_222 = relay.rsqrt(uop_189.astype('float32')) # shape=(11,)
bop_224 = relay.maximum(bop_205.astype('uint32'), relay.reshape(bop_192.astype('uint32'), relay.shape_of(bop_205))) # shape=(11,)
uop_227 = relay.log10(bop_219.astype('float32')) # shape=(11,)
uop_229 = relay.atanh(bop_219.astype('float32')) # shape=(11,)
uop_231 = relay.erf(bop_198.astype('float32')) # shape=(11,)
var_233 = relay.var("var_233", dtype = "float32", shape = (11,))#candidate|233|(11,)|var|float32
bop_234 = relay.maximum(uop_227.astype('int8'), relay.reshape(var_233.astype('int8'), relay.shape_of(uop_227))) # shape=(11,)
output = relay.Tuple([bop_20,bop_23,bop_29,bop_41,bop_45,bop_60,bop_69,bop_85,bop_91,bop_100,bop_128,uop_137,bop_140,bop_143,bop_152,bop_155,uop_168,bop_178,bop_184,uop_187,bop_209,bop_212,bop_215,uop_222,bop_224,uop_229,uop_231,bop_234,])
output2 = relay.Tuple([bop_20,bop_23,bop_29,bop_41,bop_45,bop_60,bop_69,bop_85,bop_91,bop_100,bop_128,uop_137,bop_140,bop_143,bop_152,bop_155,uop_168,bop_178,bop_184,uop_187,bop_209,bop_212,bop_215,uop_222,bop_224,uop_229,uop_231,bop_234,])
func_237 = relay.Function([var_0,var_3,var_19,var_28,var_44,var_56,var_114,var_139,var_148,var_172,var_191,var_204,var_208,var_218,var_233,], output)
mod['func_237'] = func_237
mod = relay.transform.InferType()(mod)
var_238 = relay.var("var_238", dtype = "float32", shape = ())#candidate|238|()|var|float32
var_239 = relay.var("var_239", dtype = "float32", shape = ())#candidate|239|()|var|float32
var_240 = relay.var("var_240", dtype = "float32", shape = (2, 10, 12))#candidate|240|(2, 10, 12)|var|float32
var_241 = relay.var("var_241", dtype = "float32", shape = ())#candidate|241|()|var|float32
var_242 = relay.var("var_242", dtype = "float64", shape = (5, 8))#candidate|242|(5, 8)|var|float64
var_243 = relay.var("var_243", dtype = "float32", shape = (11,))#candidate|243|(11,)|var|float32
var_244 = relay.var("var_244", dtype = "float32", shape = (11,))#candidate|244|(11,)|var|float32
var_245 = relay.var("var_245", dtype = "float32", shape = (11,))#candidate|245|(11,)|var|float32
var_246 = relay.var("var_246", dtype = "float32", shape = (11,))#candidate|246|(11,)|var|float32
var_247 = relay.var("var_247", dtype = "float64", shape = (11,))#candidate|247|(11,)|var|float64
var_248 = relay.var("var_248", dtype = "float32", shape = (11,))#candidate|248|(11,)|var|float32
var_249 = relay.var("var_249", dtype = "bool", shape = (11,))#candidate|249|(11,)|var|bool
var_250 = relay.var("var_250", dtype = "uint64", shape = (11,))#candidate|250|(11,)|var|uint64
var_251 = relay.var("var_251", dtype = "bool", shape = (11,))#candidate|251|(11,)|var|bool
var_252 = relay.var("var_252", dtype = "float32", shape = (11,))#candidate|252|(11,)|var|float32
output = func_237(var_238,var_239,var_240,var_241,var_242,var_243,var_244,var_245,var_246,var_247,var_248,var_249,var_250,var_251,var_252,)
func_253 = relay.Function([var_238,var_239,var_240,var_241,var_242,var_243,var_244,var_245,var_246,var_247,var_248,var_249,var_250,var_251,var_252,], output)
mutated_mod['func_253'] = func_253
mutated_mod = relay.transform.InferType()(mutated_mod)
const_255 = relay.const([[[-6,-9,-8,1,-6,-6,8,2,-6]],[[-9,-8,7,10,-4,7,-3,8,1]],[[-2,8,3,-9,-5,7,2,4,4]],[[4,3,7,3,4,-3,-7,-7,-5]],[[9,1,4,5,5,8,3,1,3]],[[-5,-9,8,-5,-7,2,-5,5,-6]],[[4,-5,-3,8,6,-3,1,-5,-10]],[[-2,-10,9,2,-2,10,4,-9,4]],[[-1,-9,4,-7,-3,-5,7,-8,-8]],[[-8,3,-7,-6,-8,9,7,4,8]],[[2,-5,2,-6,9,8,4,-10,-6]],[[1,-8,-6,-1,-7,-3,-5,2,7]],[[-9,-3,-8,-8,6,1,-4,1,3]]], dtype = "uint64")#candidate|255|(13, 1, 9)|const|uint64
var_256 = relay.var("var_256", dtype = "uint64", shape = (13, 1, 9))#candidate|256|(13, 1, 9)|var|uint64
bop_257 = relay.greater_equal(const_255.astype('bool'), relay.reshape(var_256.astype('bool'), relay.shape_of(const_255))) # shape=(13, 1, 9)
uop_260 = relay.acos(bop_257.astype('float64')) # shape=(13, 1, 9)
output = relay.Tuple([uop_260,])
output2 = relay.Tuple([uop_260,])
func_262 = relay.Function([var_256,], output)
mod['func_262'] = func_262
mod = relay.transform.InferType()(mod)
mutated_mod['func_262'] = func_262
mutated_mod = relay.transform.InferType()(mutated_mod)
var_263 = relay.var("var_263", dtype = "uint64", shape = (13, 1, 9))#candidate|263|(13, 1, 9)|var|uint64
func_262_call = mutated_mod.get_global_var('func_262')
call_264 = func_262_call(var_263)
output = call_264
func_265 = relay.Function([var_263], output)
mutated_mod['func_265'] = func_265
mutated_mod = relay.transform.InferType()(mutated_mod)
const_267 = relay.const([[5.175913,2.602939,-4.398583,1.528721,6.400705,-6.287511,3.487327,5.604635,1.536792,-3.229487,5.683488,-7.017892,-0.237815,3.727165],[-0.441030,1.657270,7.394614,-4.352413,6.089001,5.016943,-4.555491,0.233629,-5.238702,-5.159404,-9.219449,8.587545,5.988615,4.220898],[2.255976,-4.554526,8.002212,3.121871,0.112119,-2.607105,7.304754,8.106094,5.858457,-4.728828,6.215887,-9.850999,7.467607,-7.825592],[0.604431,-7.858599,-1.519698,-6.926740,-0.664700,7.036291,-2.421983,-3.948717,0.400383,6.689573,1.732304,-1.414270,4.744939,-1.257420],[-4.887143,1.051426,-6.396644,-5.790765,-5.887488,-4.857615,5.443539,-6.553628,5.907894,0.677079,2.018434,-8.629186,-4.141011,6.837473],[5.207245,-3.312691,-5.410442,1.984938,9.850905,-1.594646,-3.196170,8.400891,-5.434232,3.180157,-0.274655,-7.697055,-8.500463,-3.043900],[2.563875,-7.211469,-7.916447,-0.265171,-7.943821,7.809760,-2.516073,5.452012,5.029505,-2.398737,4.042774,9.506345,-6.331692,6.174942],[-9.520906,-7.930702,4.608086,-1.496110,2.527739,2.825437,-3.840388,-6.254055,-6.521363,-6.696720,-2.677412,4.275699,-3.128443,3.247568],[-8.562435,6.610082,3.401680,-2.627075,-3.118908,-3.215518,6.388394,3.338546,-2.834025,-1.453654,-9.003837,-1.542558,4.167499,-8.317321],[5.438260,3.489902,-2.707732,-1.595331,-7.954747,-5.455351,-3.274611,-2.777049,-2.153372,0.867735,8.178241,6.814902,-6.878225,-7.127802],[4.915564,-5.961806,6.325190,-9.869491,-7.060288,9.816102,6.676072,-0.932182,-8.195940,-9.310720,-7.674743,3.236206,1.092049,3.133123],[0.880203,-3.315600,-9.230427,8.489496,-0.042031,1.764916,-1.995834,7.434377,7.471966,1.856495,-3.755330,-5.524362,-8.962719,7.293068]], dtype = "float64")#candidate|267|(12, 14)|const|float64
uop_268 = relay.log(const_267.astype('float64')) # shape=(12, 14)
uop_270 = relay.exp(uop_268.astype('float32')) # shape=(12, 14)
uop_272 = relay.rsqrt(const_267.astype('float32')) # shape=(12, 14)
func_262_call = mod.get_global_var('func_262')
func_265_call = mutated_mod.get_global_var('func_265')
var_275 = relay.var("var_275", dtype = "uint64", shape = (117,))#candidate|275|(117,)|var|uint64
call_274 = relay.TupleGetItem(func_262_call(relay.reshape(var_275.astype('uint64'), [13, 1, 9])), 0)
call_276 = relay.TupleGetItem(func_265_call(relay.reshape(var_275.astype('uint64'), [13, 1, 9])), 0)
uop_277 = relay.erf(uop_270.astype('float32')) # shape=(12, 14)
uop_279 = relay.erf(uop_268.astype('float64')) # shape=(12, 14)
bop_281 = relay.logical_or(uop_277.astype('bool'), relay.reshape(uop_270.astype('bool'), relay.shape_of(uop_277))) # shape=(12, 14)
uop_284 = relay.acosh(uop_277.astype('float64')) # shape=(12, 14)
uop_286 = relay.sin(const_267.astype('float64')) # shape=(12, 14)
uop_288 = relay.sinh(const_267.astype('float64')) # shape=(12, 14)
func_262_call = mod.get_global_var('func_262')
func_265_call = mutated_mod.get_global_var('func_265')
call_290 = relay.TupleGetItem(func_262_call(relay.reshape(call_274.astype('uint64'), [13, 1, 9])), 0)
call_291 = relay.TupleGetItem(func_265_call(relay.reshape(call_274.astype('uint64'), [13, 1, 9])), 0)
uop_292 = relay.atan(uop_288.astype('float64')) # shape=(12, 14)
uop_294 = relay.rsqrt(uop_284.astype('float64')) # shape=(12, 14)
uop_296 = relay.sqrt(uop_294.astype('float32')) # shape=(12, 14)
output = relay.Tuple([uop_272,call_274,var_275,uop_279,bop_281,uop_286,call_290,uop_292,uop_296,])
output2 = relay.Tuple([uop_272,call_276,var_275,uop_279,bop_281,uop_286,call_291,uop_292,uop_296,])
func_298 = relay.Function([var_275,], output)
mod['func_298'] = func_298
mod = relay.transform.InferType()(mod)
var_299 = relay.var("var_299", dtype = "uint64", shape = (117,))#candidate|299|(117,)|var|uint64
output = func_298(var_299)
func_300 = relay.Function([var_299], output)
mutated_mod['func_300'] = func_300
mutated_mod = relay.transform.InferType()(mutated_mod)
const_302 = relay.const([-6.404477,6.950683], dtype = "float32")#candidate|302|(2,)|const|float32
uop_303 = relay.exp(const_302.astype('float32')) # shape=(2,)
output = relay.Tuple([uop_303,])
output2 = relay.Tuple([uop_303,])
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

'''45: TVMFuncCall
44: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
43: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
42: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
41: tvm::IRModule::FromExpr(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&)
40: tvm::IRModule::FromExprInContext(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&, std::unordered_set<tvm::runtime::String, std::hash<tvm::runtime::String>, std::equal_to<tvm::runtime::String>, std::allocator<tvm::runtime::String> >)
39: tvm::IRModuleNode::Add(tvm::GlobalVar const&, tvm::BaseFunc const&, bool)
38: tvm::WarnIfMalformed(tvm::IRModule const&, tvm::relay::Function)
37: tvm::relay::FreeTypeVars(tvm::RelayExpr const&, tvm::IRModule const&)
36: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
35: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
34: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
33: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
32: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
31: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
30: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
29: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
28: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
27: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
26: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
25: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
24: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
23: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
22: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
21: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
20: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
19: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
16: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
15: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
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