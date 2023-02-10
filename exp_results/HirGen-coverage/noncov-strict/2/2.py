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
var_0 = relay.var("var_0", dtype = "float64", shape = (9,))#candidate|0|(9,)|var|float64
uop_1 = relay.cosh(var_0.astype('float64')) # shape=(9,)
bop_3 = relay.greater_equal(uop_1.astype('bool'), relay.reshape(var_0.astype('bool'), relay.shape_of(uop_1))) # shape=(9,)
var_6 = relay.var("var_6", dtype = "bool", shape = (9,))#candidate|6|(9,)|var|bool
bop_7 = relay.less(bop_3.astype('bool'), relay.reshape(var_6.astype('bool'), relay.shape_of(bop_3))) # shape=(9,)
const_10 = relay.const([False,False,True,True,True,False,True,False,True], dtype = "bool")#candidate|10|(9,)|const|bool
bop_11 = relay.power(bop_7.astype('float32'), relay.reshape(const_10.astype('float32'), relay.shape_of(bop_7))) # shape=(9,)
var_14 = relay.var("var_14", dtype = "float64", shape = (9,))#candidate|14|(9,)|var|float64
bop_15 = relay.left_shift(uop_1.astype('uint32'), relay.reshape(var_14.astype('uint32'), relay.shape_of(uop_1))) # shape=(9,)
var_18 = relay.var("var_18", dtype = "float64", shape = (9,))#candidate|18|(9,)|var|float64
bop_19 = relay.add(var_14.astype('uint8'), relay.reshape(var_18.astype('uint8'), relay.shape_of(var_14))) # shape=(9,)
bop_22 = relay.logical_and(uop_1.astype('bool'), relay.reshape(bop_15.astype('bool'), relay.shape_of(uop_1))) # shape=(9,)
var_25 = relay.var("var_25", dtype = "uint32", shape = (9,))#candidate|25|(9,)|var|uint32
bop_26 = relay.power(bop_15.astype('float64'), relay.reshape(var_25.astype('float64'), relay.shape_of(bop_15))) # shape=(9,)
uop_29 = relay.atan(var_25.astype('float64')) # shape=(9,)
uop_31 = relay.log(bop_22.astype('float64')) # shape=(9,)
uop_33 = relay.atanh(uop_31.astype('float64')) # shape=(9,)
bop_35 = relay.maximum(bop_15.astype('int64'), relay.reshape(uop_1.astype('int64'), relay.shape_of(bop_15))) # shape=(9,)
bop_38 = relay.add(uop_33.astype('float32'), relay.reshape(var_18.astype('float32'), relay.shape_of(uop_33))) # shape=(9,)
bop_41 = relay.bitwise_and(var_0.astype('uint32'), relay.reshape(bop_7.astype('uint32'), relay.shape_of(var_0))) # shape=(9,)
uop_44 = relay.cos(bop_38.astype('float64')) # shape=(9,)
output = relay.Tuple([bop_11,bop_19,bop_26,uop_29,bop_35,bop_41,uop_44,])
output2 = relay.Tuple([bop_11,bop_19,bop_26,uop_29,bop_35,bop_41,uop_44,])
func_46 = relay.Function([var_0,var_6,var_14,var_18,var_25,], output)
mod['func_46'] = func_46
mod = relay.transform.InferType()(mod)
var_47 = relay.var("var_47", dtype = "float64", shape = (9,))#candidate|47|(9,)|var|float64
var_48 = relay.var("var_48", dtype = "bool", shape = (9,))#candidate|48|(9,)|var|bool
var_49 = relay.var("var_49", dtype = "float64", shape = (9,))#candidate|49|(9,)|var|float64
var_50 = relay.var("var_50", dtype = "float64", shape = (9,))#candidate|50|(9,)|var|float64
var_51 = relay.var("var_51", dtype = "uint32", shape = (9,))#candidate|51|(9,)|var|uint32
output = func_46(var_47,var_48,var_49,var_50,var_51,)
func_52 = relay.Function([var_47,var_48,var_49,var_50,var_51,], output)
mutated_mod['func_52'] = func_52
mutated_mod = relay.transform.InferType()(mutated_mod)
var_54 = relay.var("var_54", dtype = "float64", shape = (5,))#candidate|54|(5,)|var|float64
uop_55 = relay.sigmoid(var_54.astype('float64')) # shape=(5,)
uop_57 = relay.log2(uop_55.astype('float32')) # shape=(5,)
bop_59 = relay.bitwise_or(var_54.astype('uint32'), relay.reshape(uop_57.astype('uint32'), relay.shape_of(var_54))) # shape=(5,)
bop_62 = relay.greater(var_54.astype('bool'), relay.reshape(uop_55.astype('bool'), relay.shape_of(var_54))) # shape=(5,)
var_65 = relay.var("var_65", dtype = "float64", shape = (5,))#candidate|65|(5,)|var|float64
bop_66 = relay.minimum(uop_55.astype('uint64'), relay.reshape(var_65.astype('uint64'), relay.shape_of(uop_55))) # shape=(5,)
uop_69 = relay.atan(uop_57.astype('float32')) # shape=(5,)
bop_71 = relay.equal(uop_69.astype('bool'), relay.reshape(bop_66.astype('bool'), relay.shape_of(uop_69))) # shape=(5,)
bop_74 = relay.greater_equal(var_54.astype('bool'), relay.reshape(bop_71.astype('bool'), relay.shape_of(var_54))) # shape=(5,)
uop_77 = relay.log(bop_74.astype('float64')) # shape=(5,)
var_79 = relay.var("var_79", dtype = "uint32", shape = (5,))#candidate|79|(5,)|var|uint32
bop_80 = relay.logical_and(bop_59.astype('bool'), relay.reshape(var_79.astype('bool'), relay.shape_of(bop_59))) # shape=(5,)
bop_83 = relay.minimum(bop_80.astype('float32'), relay.reshape(uop_69.astype('float32'), relay.shape_of(bop_80))) # shape=(5,)
var_86 = relay.var("var_86", dtype = "float64", shape = (5,))#candidate|86|(5,)|var|float64
bop_87 = relay.greater(uop_77.astype('bool'), relay.reshape(var_86.astype('bool'), relay.shape_of(uop_77))) # shape=(5,)
uop_90 = relay.rsqrt(uop_55.astype('float64')) # shape=(5,)
var_92 = relay.var("var_92", dtype = "float32", shape = (5,))#candidate|92|(5,)|var|float32
bop_93 = relay.add(uop_57.astype('uint16'), relay.reshape(var_92.astype('uint16'), relay.shape_of(uop_57))) # shape=(5,)
uop_96 = relay.log2(uop_77.astype('float64')) # shape=(5,)
bop_98 = relay.logical_and(bop_87.astype('bool'), relay.reshape(bop_71.astype('bool'), relay.shape_of(bop_87))) # shape=(5,)
bop_101 = relay.less(bop_87.astype('bool'), relay.reshape(var_86.astype('bool'), relay.shape_of(bop_87))) # shape=(5,)
uop_104 = relay.log(uop_96.astype('float32')) # shape=(5,)
var_106 = relay.var("var_106", dtype = "float64", shape = (5,))#candidate|106|(5,)|var|float64
bop_107 = relay.less_equal(uop_96.astype('bool'), relay.reshape(var_106.astype('bool'), relay.shape_of(uop_96))) # shape=(5,)
uop_110 = relay.rsqrt(bop_101.astype('float32')) # shape=(5,)
uop_112 = relay.log10(uop_104.astype('float32')) # shape=(5,)
uop_114 = relay.asin(bop_101.astype('float32')) # shape=(5,)
uop_116 = relay.exp(uop_110.astype('float32')) # shape=(5,)
uop_118 = relay.acos(uop_112.astype('float64')) # shape=(5,)
uop_120 = relay.cosh(uop_116.astype('float64')) # shape=(5,)
bop_122 = relay.minimum(uop_104.astype('int16'), relay.reshape(var_54.astype('int16'), relay.shape_of(uop_104))) # shape=(5,)
bop_125 = relay.bitwise_and(bop_87.astype('int8'), relay.reshape(bop_74.astype('int8'), relay.shape_of(bop_87))) # shape=(5,)
bop_128 = relay.bitwise_and(uop_118.astype('uint64'), relay.reshape(var_92.astype('uint64'), relay.shape_of(uop_118))) # shape=(5,)
var_131 = relay.var("var_131", dtype = "float64", shape = (5,))#candidate|131|(5,)|var|float64
bop_132 = relay.add(uop_118.astype('int32'), relay.reshape(var_131.astype('int32'), relay.shape_of(uop_118))) # shape=(5,)
var_135 = relay.var("var_135", dtype = "uint64", shape = (5,))#candidate|135|(5,)|var|uint64
bop_136 = relay.greater(bop_128.astype('bool'), relay.reshape(var_135.astype('bool'), relay.shape_of(bop_128))) # shape=(5,)
var_139 = relay.var("var_139", dtype = "float32", shape = (5,))#candidate|139|(5,)|var|float32
bop_140 = relay.mod(uop_104.astype('float64'), relay.reshape(var_139.astype('float64'), relay.shape_of(uop_104))) # shape=(5,)
uop_143 = relay.sigmoid(bop_87.astype('float64')) # shape=(5,)
bop_145 = relay.greater(bop_136.astype('bool'), relay.reshape(uop_143.astype('bool'), relay.shape_of(bop_136))) # shape=(5,)
bop_148 = relay.logical_xor(uop_112.astype('int16'), relay.reshape(bop_125.astype('int16'), relay.shape_of(uop_112))) # shape=(5,)
const_151 = relay.const([2,-6,4,-3,2], dtype = "int8")#candidate|151|(5,)|const|int8
bop_152 = relay.bitwise_and(bop_125.astype('uint8'), relay.reshape(const_151.astype('uint8'), relay.shape_of(bop_125))) # shape=(5,)
uop_155 = relay.tan(bop_145.astype('float32')) # shape=(5,)
uop_157 = relay.asinh(uop_155.astype('float64')) # shape=(5,)
var_159 = relay.var("var_159", dtype = "float32", shape = (5,))#candidate|159|(5,)|var|float32
bop_160 = relay.equal(uop_155.astype('bool'), relay.reshape(var_159.astype('bool'), relay.shape_of(uop_155))) # shape=(5,)
bop_163 = relay.minimum(bop_160.astype('int8'), relay.reshape(var_159.astype('int8'), relay.shape_of(bop_160))) # shape=(5,)
uop_166 = relay.sqrt(bop_163.astype('float32')) # shape=(5,)
bop_168 = relay.power(uop_155.astype('float32'), relay.reshape(var_86.astype('float32'), relay.shape_of(uop_155))) # shape=(5,)
bop_171 = relay.maximum(uop_118.astype('int32'), relay.reshape(uop_112.astype('int32'), relay.shape_of(uop_118))) # shape=(5,)
uop_174 = relay.log(bop_132.astype('float64')) # shape=(5,)
bop_176 = relay.logical_or(uop_157.astype('bool'), relay.reshape(bop_148.astype('bool'), relay.shape_of(uop_157))) # shape=(5,)
bop_179 = relay.floor_mod(uop_120.astype('float32'), relay.reshape(bop_98.astype('float32'), relay.shape_of(uop_120))) # shape=(5,)
uop_182 = relay.log2(bop_122.astype('float64')) # shape=(5,)
uop_184 = relay.sinh(bop_176.astype('float32')) # shape=(5,)
bop_186 = relay.bitwise_or(uop_184.astype('uint16'), relay.reshape(uop_112.astype('uint16'), relay.shape_of(uop_184))) # shape=(5,)
uop_189 = relay.erf(uop_184.astype('float64')) # shape=(5,)
uop_191 = relay.rsqrt(uop_189.astype('float64')) # shape=(5,)
uop_193 = relay.sqrt(uop_166.astype('float64')) # shape=(5,)
var_195 = relay.var("var_195", dtype = "float64", shape = (5,))#candidate|195|(5,)|var|float64
bop_196 = relay.minimum(uop_191.astype('uint8'), relay.reshape(var_195.astype('uint8'), relay.shape_of(uop_191))) # shape=(5,)
const_199 = relay.const([4.241620,9.821375,-7.961053,4.617392,8.312699], dtype = "float64")#candidate|199|(5,)|const|float64
bop_200 = relay.not_equal(uop_189.astype('bool'), relay.reshape(const_199.astype('bool'), relay.shape_of(uop_189))) # shape=(5,)
uop_203 = relay.sqrt(bop_200.astype('float32')) # shape=(5,)
bop_205 = relay.not_equal(uop_193.astype('bool'), relay.reshape(bop_83.astype('bool'), relay.shape_of(uop_193))) # shape=(5,)
var_208 = relay.var("var_208", dtype = "float32", shape = (5,))#candidate|208|(5,)|var|float32
bop_209 = relay.minimum(uop_203.astype('float64'), relay.reshape(var_208.astype('float64'), relay.shape_of(uop_203))) # shape=(5,)
bop_212 = relay.bitwise_or(uop_193.astype('uint64'), relay.reshape(uop_96.astype('uint64'), relay.shape_of(uop_193))) # shape=(5,)
bop_215 = relay.subtract(bop_200.astype('int32'), relay.reshape(uop_166.astype('int32'), relay.shape_of(bop_200))) # shape=(5,)
var_218 = relay.var("var_218", dtype = "float64", shape = (5,))#candidate|218|(5,)|var|float64
bop_219 = relay.greater(uop_191.astype('bool'), relay.reshape(var_218.astype('bool'), relay.shape_of(uop_191))) # shape=(5,)
uop_222 = relay.asinh(uop_191.astype('float32')) # shape=(5,)
var_224 = relay.var("var_224", dtype = "float32", shape = (5,))#candidate|224|(5,)|var|float32
bop_225 = relay.equal(uop_222.astype('bool'), relay.reshape(var_224.astype('bool'), relay.shape_of(uop_222))) # shape=(5,)
uop_228 = relay.sinh(uop_222.astype('float32')) # shape=(5,)
bop_230 = relay.greater_equal(uop_228.astype('bool'), relay.reshape(uop_222.astype('bool'), relay.shape_of(uop_228))) # shape=(5,)
var_233 = relay.var("var_233", dtype = "float32", shape = (5,))#candidate|233|(5,)|var|float32
bop_234 = relay.logical_xor(uop_203.astype('int32'), relay.reshape(var_233.astype('int32'), relay.shape_of(uop_203))) # shape=(5,)
bop_237 = relay.multiply(uop_228.astype('uint64'), relay.reshape(uop_112.astype('uint64'), relay.shape_of(uop_228))) # shape=(5,)
bop_240 = relay.left_shift(bop_230.astype('uint16'), relay.reshape(uop_118.astype('uint16'), relay.shape_of(bop_230))) # shape=(5,)
var_243 = relay.var("var_243", dtype = "uint16", shape = (5,))#candidate|243|(5,)|var|uint16
bop_244 = relay.bitwise_xor(bop_240.astype('int32'), relay.reshape(var_243.astype('int32'), relay.shape_of(bop_240))) # shape=(5,)
uop_247 = relay.acosh(uop_184.astype('float64')) # shape=(5,)
var_249 = relay.var("var_249", dtype = "float32", shape = (5,))#candidate|249|(5,)|var|float32
bop_250 = relay.subtract(uop_222.astype('int64'), relay.reshape(var_249.astype('int64'), relay.shape_of(uop_222))) # shape=(5,)
var_253 = relay.var("var_253", dtype = "float32", shape = (5,))#candidate|253|(5,)|var|float32
bop_254 = relay.add(uop_222.astype('uint64'), relay.reshape(var_253.astype('uint64'), relay.shape_of(uop_222))) # shape=(5,)
bop_257 = relay.less(bop_240.astype('bool'), relay.reshape(uop_104.astype('bool'), relay.shape_of(bop_240))) # shape=(5,)
bop_260 = relay.logical_xor(bop_240.astype('uint16'), relay.reshape(uop_118.astype('uint16'), relay.shape_of(bop_240))) # shape=(5,)
bop_263 = relay.right_shift(bop_225.astype('uint64'), relay.reshape(uop_96.astype('uint64'), relay.shape_of(bop_225))) # shape=(5,)
bop_266 = relay.bitwise_xor(bop_225.astype('int16'), relay.reshape(uop_57.astype('int16'), relay.shape_of(bop_225))) # shape=(5,)
uop_269 = relay.rsqrt(bop_225.astype('float64')) # shape=(5,)
var_271 = relay.var("var_271", dtype = "float64", shape = (5,))#candidate|271|(5,)|var|float64
bop_272 = relay.bitwise_xor(uop_247.astype('uint64'), relay.reshape(var_271.astype('uint64'), relay.shape_of(uop_247))) # shape=(5,)
bop_275 = relay.less(uop_228.astype('bool'), relay.reshape(bop_71.astype('bool'), relay.shape_of(uop_228))) # shape=(5,)
uop_278 = relay.asinh(bop_250.astype('float32')) # shape=(5,)
uop_280 = relay.acos(bop_237.astype('float32')) # shape=(5,)
bop_282 = relay.right_shift(uop_280.astype('int16'), relay.reshape(bop_87.astype('int16'), relay.shape_of(uop_280))) # shape=(5,)
output = relay.Tuple([bop_62,uop_90,bop_93,bop_107,uop_114,bop_140,bop_152,bop_168,bop_171,uop_174,bop_179,uop_182,bop_186,bop_196,bop_205,bop_209,bop_212,bop_215,bop_219,bop_234,bop_244,bop_254,bop_257,bop_260,bop_263,bop_266,uop_269,bop_272,bop_275,uop_278,bop_282,])
output2 = relay.Tuple([bop_62,uop_90,bop_93,bop_107,uop_114,bop_140,bop_152,bop_168,bop_171,uop_174,bop_179,uop_182,bop_186,bop_196,bop_205,bop_209,bop_212,bop_215,bop_219,bop_234,bop_244,bop_254,bop_257,bop_260,bop_263,bop_266,uop_269,bop_272,bop_275,uop_278,bop_282,])
func_285 = relay.Function([var_54,var_65,var_79,var_86,var_92,var_106,var_131,var_135,var_139,var_159,var_195,var_208,var_218,var_224,var_233,var_243,var_249,var_253,var_271,], output)
mod['func_285'] = func_285
mod = relay.transform.InferType()(mod)
var_286 = relay.var("var_286", dtype = "float64", shape = (5,))#candidate|286|(5,)|var|float64
var_287 = relay.var("var_287", dtype = "float64", shape = (5,))#candidate|287|(5,)|var|float64
var_288 = relay.var("var_288", dtype = "uint32", shape = (5,))#candidate|288|(5,)|var|uint32
var_289 = relay.var("var_289", dtype = "float64", shape = (5,))#candidate|289|(5,)|var|float64
var_290 = relay.var("var_290", dtype = "float32", shape = (5,))#candidate|290|(5,)|var|float32
var_291 = relay.var("var_291", dtype = "float64", shape = (5,))#candidate|291|(5,)|var|float64
var_292 = relay.var("var_292", dtype = "float64", shape = (5,))#candidate|292|(5,)|var|float64
var_293 = relay.var("var_293", dtype = "uint64", shape = (5,))#candidate|293|(5,)|var|uint64
var_294 = relay.var("var_294", dtype = "float32", shape = (5,))#candidate|294|(5,)|var|float32
var_295 = relay.var("var_295", dtype = "float32", shape = (5,))#candidate|295|(5,)|var|float32
var_296 = relay.var("var_296", dtype = "float64", shape = (5,))#candidate|296|(5,)|var|float64
var_297 = relay.var("var_297", dtype = "float32", shape = (5,))#candidate|297|(5,)|var|float32
var_298 = relay.var("var_298", dtype = "float64", shape = (5,))#candidate|298|(5,)|var|float64
var_299 = relay.var("var_299", dtype = "float32", shape = (5,))#candidate|299|(5,)|var|float32
var_300 = relay.var("var_300", dtype = "float32", shape = (5,))#candidate|300|(5,)|var|float32
var_301 = relay.var("var_301", dtype = "uint16", shape = (5,))#candidate|301|(5,)|var|uint16
var_302 = relay.var("var_302", dtype = "float32", shape = (5,))#candidate|302|(5,)|var|float32
var_303 = relay.var("var_303", dtype = "float32", shape = (5,))#candidate|303|(5,)|var|float32
var_304 = relay.var("var_304", dtype = "float64", shape = (5,))#candidate|304|(5,)|var|float64
output = func_285(var_286,var_287,var_288,var_289,var_290,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,var_299,var_300,var_301,var_302,var_303,var_304,)
func_305 = relay.Function([var_286,var_287,var_288,var_289,var_290,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,var_299,var_300,var_301,var_302,var_303,var_304,], output)
mutated_mod['func_305'] = func_305
mutated_mod = relay.transform.InferType()(mutated_mod)
var_307 = relay.var("var_307", dtype = "int16", shape = (5, 14, 16))#candidate|307|(5, 14, 16)|var|int16
var_308 = relay.var("var_308", dtype = "int16", shape = (5, 14, 16))#candidate|308|(5, 14, 16)|var|int16
bop_309 = relay.less(var_307.astype('bool'), relay.reshape(var_308.astype('bool'), relay.shape_of(var_307))) # shape=(5, 14, 16)
uop_312 = relay.acos(var_308.astype('float64')) # shape=(5, 14, 16)
const_314 = relay.const([[[4.120484,9.743199,-4.153346,0.612536,-6.140838,6.466173,0.938132,-4.798955,-8.892711,8.729049,3.514246,2.724811,0.079870,-7.762634,0.917035,4.520031],[4.761381,-5.256124,7.668148,3.083704,-4.383486,-5.381241,-0.345756,8.708852,0.686369,-5.881523,-0.864884,8.952417,-9.000060,-2.470710,0.024840,-1.224857],[3.315013,-6.119077,1.309545,-6.603864,-2.475174,5.673236,1.871248,-7.973636,3.239178,8.571339,-6.442857,-4.584516,-9.137384,-3.743259,-5.362415,4.289761],[-3.748579,-8.593467,7.852191,6.813150,1.238030,-5.476031,7.441621,-2.800152,4.292721,-6.145984,-2.917918,-1.541076,-7.784484,-3.393201,-9.710601,-7.445475],[-2.783498,-4.989374,2.846559,8.663381,-6.165612,9.089067,-2.147660,-0.801771,7.162094,9.408515,4.372969,-8.379939,0.299066,-3.573287,-4.333346,-3.516681],[-3.047974,-5.384294,-2.228600,-0.479718,-5.017556,8.086748,5.486047,-5.174717,-3.339960,-1.733300,-6.241483,5.942327,-0.597122,1.858693,-6.524301,3.808960],[-7.713263,-5.787317,3.208347,9.280096,-7.521906,6.048552,-0.289451,-1.018171,3.663509,6.076951,-8.726838,6.337283,-6.347820,5.930606,-9.641040,7.721048],[1.684838,-1.344758,4.921028,-3.269601,-2.581981,9.169644,7.875237,9.843721,7.620931,-9.814906,-2.270472,-9.049488,-6.648733,3.225469,-2.338078,7.783444],[6.511851,-1.454972,5.397431,-6.316957,-3.131754,1.850782,-9.908518,9.574923,6.236732,9.591263,9.280023,-3.948264,8.186656,9.538041,0.224052,0.697401],[-5.787010,9.930126,-9.546675,6.235943,7.484065,6.549670,1.066327,0.142675,-6.254039,-4.100368,-4.008005,-1.253219,8.946442,7.500424,-7.423510,3.711571],[5.709797,6.037023,-9.492899,-1.589514,3.993687,0.556613,-4.647193,5.772639,4.364011,9.226200,-0.044921,-5.371601,9.104434,-0.083024,-0.806222,-3.299124],[-6.257254,-9.661263,-3.406587,1.949378,-9.525581,-3.433572,6.429068,1.695738,-3.252633,5.579056,-7.585046,-4.552705,3.768318,-4.087494,-7.212104,9.645157],[-2.707687,0.157863,6.584703,-4.300301,-5.743325,-2.303705,-1.540751,-8.346255,-5.608866,7.524252,1.342030,-8.821749,1.479981,9.891372,9.896555,-1.479400],[-9.750588,-6.650776,-6.650934,4.983646,-3.382579,9.622375,8.522959,7.990515,6.174428,3.129416,-8.341134,9.432856,-9.852340,0.831138,-6.456130,8.166700]],[[5.780770,-0.610751,-9.859543,-4.275845,1.517711,9.171333,-3.129858,4.052298,3.758516,-1.981365,-5.631047,5.642889,-8.321541,3.401719,8.858954,9.302067],[-4.396832,-2.520480,4.558649,6.738499,-1.859158,-9.472435,9.072252,-3.462654,-4.082563,-6.485311,8.286896,2.766523,-4.010885,-1.460319,6.380251,3.552417],[2.861591,-0.228476,-1.304041,3.094842,-8.708519,9.980497,1.564245,-1.866919,9.104634,3.393858,0.676094,-4.171764,2.304789,7.079304,4.878749,-9.996971],[-8.779442,0.876226,2.814959,-9.440635,-2.934211,4.480936,4.484494,8.881160,6.262341,7.409305,4.336733,-9.391495,-8.079069,9.493546,-4.243548,-3.745795],[-9.808827,-6.470732,1.402643,8.348125,-9.852618,0.572581,4.048167,-3.276304,2.835234,2.521686,-7.320936,5.724292,0.053893,-1.447411,0.784162,-1.165284],[-0.329274,9.814579,1.589739,-8.680814,6.408411,-8.767794,-0.452465,0.835282,-6.388237,4.579313,5.487792,-1.553550,5.993079,-1.735580,1.461216,9.543893],[9.243152,4.505883,9.352824,9.295789,-1.896587,0.437029,-9.723373,-2.126838,9.856466,1.149427,2.658191,7.842224,-1.851967,3.005478,7.235953,-1.491443],[2.844096,9.119072,7.837494,7.071231,-3.592087,-2.046410,-4.156005,-0.477566,-9.890711,-4.245831,4.055118,7.443087,9.827744,8.189622,3.987349,-2.475754],[-2.981720,0.356803,-3.402967,6.198799,-9.360953,7.363600,-9.596810,-6.708599,-0.521100,-7.411494,1.111918,4.492779,-6.400676,2.638995,-9.476025,-2.639256],[9.288140,-4.618154,3.452953,3.384070,-4.353893,-4.098812,-6.652656,8.509551,1.721493,-4.136701,0.280716,8.854963,-2.954574,5.944587,-5.815693,1.889792],[-8.468687,3.396572,1.318509,8.188679,-3.708353,6.065335,-8.754065,5.077129,-3.746008,-6.002090,8.054525,-6.704763,-3.309643,9.445758,7.579724,2.725727],[-1.494950,6.682800,2.663807,7.644300,-2.160705,-7.837017,-8.759396,-2.650829,9.295832,-4.654819,4.847281,-2.341137,3.594409,-3.782373,2.092841,-1.539588],[2.862228,8.642727,-5.175096,-6.438374,8.679774,2.374241,-9.101463,7.171503,3.376771,-2.933016,-9.733542,-8.600334,-4.937486,-1.646976,-8.414509,-7.220844],[6.417836,-2.334056,-9.470443,-3.012401,6.930054,9.037702,-4.827314,5.855937,6.116768,-9.604869,7.362213,7.300317,3.923354,2.532647,-2.752493,4.799184]],[[8.074585,7.184373,-6.938842,-7.680397,7.565727,4.021534,-8.700237,7.736767,0.815823,1.846993,6.817456,-0.621582,-0.491351,-5.785006,7.488871,-9.122811],[-8.255061,0.967188,-8.423056,-7.368564,9.582997,0.375787,9.301734,7.461316,-4.982063,4.216176,1.478247,-0.564867,0.495106,-5.383845,6.385258,7.800590],[-9.890986,-6.353597,-4.456115,9.036963,2.229661,5.193415,2.946885,2.743567,8.639744,3.552133,-5.996793,3.163436,1.391529,1.542595,-8.007593,-1.848671],[1.580016,5.683211,6.043922,9.901426,4.201720,5.662359,-7.628535,-0.745766,8.603467,-9.259411,-6.709445,3.236578,-9.535378,-2.587827,-3.272032,2.066189],[6.149421,3.210641,2.584053,9.338289,-8.907925,-9.583417,1.059572,-8.228150,-8.397171,0.441429,-9.072869,-3.499460,5.544201,-6.505895,6.238021,4.855790],[-0.997547,-3.046061,8.581763,0.299992,-1.852340,-4.862805,-8.013398,8.092318,-0.435842,1.001143,4.001821,2.918860,-7.625277,-6.445974,7.616297,9.576170],[8.784143,-3.943786,8.538256,-7.129983,3.118034,-7.258341,2.088365,-7.174368,-9.422918,1.935168,-0.808693,-1.124465,-9.704306,-8.765581,-0.703626,4.559035],[0.307909,-9.684745,8.106981,6.010467,-4.562454,0.563881,0.853644,1.247356,7.713508,5.509634,2.255831,3.391712,7.960207,5.132354,4.650837,0.909843],[-4.726381,4.300790,-1.717245,-4.198013,-5.560699,-4.492385,-2.381529,8.549351,-2.403291,1.477575,-1.993289,0.280944,-1.779533,1.787006,-4.271475,6.227841],[9.897484,-4.500575,5.888900,-7.799675,-4.980105,6.471135,-5.756111,5.132507,1.804731,-8.873529,5.736789,0.259547,-9.737587,4.024973,9.944017,-2.613936],[9.543092,-8.215986,5.685802,4.784874,-4.492830,-9.733432,8.536609,2.048289,-1.570776,9.675493,-6.581584,-9.249284,0.324492,2.514311,-5.594006,7.114355],[-5.359529,-3.426366,-8.519696,-4.185800,-3.639484,0.696877,-1.210459,-7.915009,-0.452218,-5.213820,6.860226,-4.437172,-7.495265,9.794766,6.031663,-8.347310],[0.491761,7.618686,-5.293587,-6.303544,5.246038,5.174643,-3.425769,-0.671068,7.828183,2.536756,5.559687,7.845848,2.013599,-1.059565,-0.238592,3.419745],[8.438683,9.364486,-6.886931,-4.622339,-4.191006,6.561168,-5.928401,8.519627,-4.466022,-8.564654,-0.336128,6.255996,-4.678923,-8.553693,-9.174803,-2.184058]],[[9.946280,-8.926232,1.948566,6.307776,-9.037017,5.693501,-7.279280,-8.143536,3.078494,3.319453,-2.911580,-0.984309,-9.037460,4.946820,-5.666151,4.939254],[-3.614488,3.150313,-1.404787,4.325565,5.343739,6.608998,2.917695,-0.295534,-9.922946,1.653732,-4.586092,-7.034518,-8.938157,6.522769,2.099470,-1.576409],[-9.720912,-6.874331,-5.761259,3.647057,-2.712939,-1.590797,-6.195957,-8.077845,-4.806040,-2.175001,8.137005,7.917230,-3.792900,-0.689385,7.711086,6.747541],[-9.371849,7.745387,1.661457,-6.263341,8.239985,-8.625032,4.117183,7.062424,-2.994912,-4.512876,-2.089740,9.558810,1.971583,-2.529388,6.063682,-5.784115],[8.967525,-7.197717,8.383420,-8.642072,8.428310,-9.529133,-8.460190,-2.294599,9.816248,-1.110960,3.425184,9.271003,-6.834919,1.116593,6.017259,8.476065],[9.590898,-0.511876,4.466289,-9.932362,3.285588,2.057116,-6.169654,-5.035512,2.743628,8.765404,-8.571846,8.981902,-1.522968,-7.990468,-0.704636,1.497124],[-5.912817,-3.521552,-8.397967,-7.042174,-1.805261,4.201019,-1.521175,-6.144541,9.268130,-4.361290,-0.923674,-0.212879,-3.781623,9.969926,-5.972646,5.824004],[-9.588301,-1.632673,-2.534407,-9.429842,2.046310,5.589803,3.906374,6.430919,7.404125,-4.314227,3.544656,-1.004696,-5.753801,-1.767638,-6.773508,2.753227],[-4.347175,-0.451042,-6.018780,9.221853,-6.246111,8.856028,-3.777145,-7.546985,-1.358742,-4.127967,6.524447,-7.761567,-5.631301,3.887981,-4.385759,8.543892],[-1.658412,1.865389,-9.439504,-8.035882,8.708876,-4.046178,-4.295549,6.694873,0.447470,-4.077793,-8.332178,-0.712745,-0.929732,-8.062508,-5.624516,4.918953],[0.200155,-8.679865,4.032197,-0.524989,-4.087967,-4.209834,-1.347207,-4.234896,5.518836,-3.284490,8.760723,-3.142292,-3.937981,-0.119701,-0.527256,-1.655946],[-2.479232,6.629863,9.155167,-4.649279,-4.531202,8.748589,-2.005137,7.076759,-9.776746,8.660907,-1.750152,9.673083,-3.671486,-2.739974,2.361825,-4.891174],[1.576684,2.124504,-1.755333,-0.381569,4.622277,7.316394,2.187095,8.296680,-0.427262,2.940259,7.147770,-7.864980,-2.026523,-4.666054,9.660864,6.780379],[5.446445,7.997565,-6.802920,-6.361275,2.977546,0.888285,-9.455335,8.600465,-4.007336,-7.057847,8.555038,6.912084,5.734284,-8.510409,5.291518,9.804896]],[[-0.095732,-6.610047,-1.203530,-6.968995,-5.583657,6.158198,5.097669,9.140352,1.227285,8.087292,-1.663662,8.524914,-3.965248,-9.337326,7.933259,-3.887181],[4.183645,2.613566,-0.875618,2.250117,8.392688,5.184157,6.452426,-8.697232,-5.740361,-5.073481,5.288647,-4.809568,7.219210,1.913294,-0.234097,-8.438151],[-0.628139,-9.304272,-5.943221,7.485949,-7.458496,-3.337766,6.670765,-2.174435,-9.925891,8.035352,-5.858321,0.482746,-0.069419,0.445112,2.140039,-0.330952],[4.239551,-8.134479,1.494610,6.204329,-9.169599,7.568291,3.974414,-7.508061,-9.335324,0.156634,6.341501,-4.165531,6.417015,8.165367,-0.604373,-2.809776],[-7.661145,9.733011,-6.902553,2.011751,-3.055369,-7.153170,-5.886991,9.682957,-9.695853,-5.166935,2.370879,-1.015245,-1.490917,5.209121,3.630862,-0.714769],[-5.422266,2.168175,-9.111410,-0.807401,-1.642111,-8.427878,5.069920,5.493208,-5.633269,6.985653,5.056254,-6.334811,5.044657,-8.342564,7.390612,-8.689515],[-6.953543,-8.469797,-3.462686,3.461145,-3.648713,-3.524927,6.722690,8.505441,6.026639,2.740464,-8.551855,-0.715409,-3.738289,0.547068,2.649928,4.106816],[5.123224,-5.176834,-2.489215,-3.479962,-2.603135,-0.369198,7.789160,4.487748,1.797444,0.712484,2.818469,7.148342,-6.542108,8.758371,7.699224,8.777958],[9.590250,-8.842188,2.579714,-3.479294,4.376154,0.551938,-2.262926,7.313083,5.658801,-8.068903,-6.736224,-3.166275,-7.374026,-2.492506,6.570504,-8.313016],[-6.974788,-4.531671,-7.536708,3.759278,-2.838738,-4.534842,9.905506,-5.763403,-0.310131,5.224636,-8.822388,2.909283,-1.544991,-3.210523,9.086242,-0.419735],[-6.316401,3.605165,2.938602,-6.249845,4.896887,-1.802057,9.529388,8.983122,5.829708,-0.581173,8.019766,-8.882690,-1.898694,6.720619,-8.324034,9.638449],[-7.411876,9.195932,8.669668,9.029123,-2.921131,0.635440,-5.599610,7.455509,-9.269970,-6.180236,0.035167,3.593994,2.955209,3.092730,-6.948019,-0.656926],[5.706074,-8.079254,-5.255264,4.818644,7.372389,-1.083039,4.274023,-5.969316,2.138432,4.999806,8.192823,-1.609564,8.051079,-5.004158,-0.211445,8.167259],[-4.524648,-6.816426,5.175127,9.709520,3.697108,8.678402,1.193661,-3.953926,-0.108755,9.681962,-7.827435,-2.373901,5.942049,4.386978,-0.240350,0.378199]]], dtype = "float64")#candidate|314|(5, 14, 16)|const|float64
bop_315 = relay.equal(uop_312.astype('bool'), relay.reshape(const_314.astype('bool'), relay.shape_of(uop_312))) # shape=(5, 14, 16)
uop_318 = relay.acosh(bop_315.astype('float64')) # shape=(5, 14, 16)
uop_320 = relay.acosh(const_314.astype('float64')) # shape=(5, 14, 16)
output = relay.Tuple([bop_309,uop_318,uop_320,])
output2 = relay.Tuple([bop_309,uop_318,uop_320,])
F = relay.Function([var_307,var_308,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_307,var_308,], output2)
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
input_307= np.array([[[5,1,9,9,1,-10,-8,-9,1,7,4,6,2,8,8,-10],[1,4,5,4,9,8,-10,-10,-6,5,-4,7,-6,9,6,9],[-1,9,-6,3,-4,-5,-9,-7,1,-6,-4,2,-8,6,-2,-3],[-5,10,-5,7,3,6,-6,-8,7,-3,-5,-3,-7,8,1,2],[-4,-7,5,-6,7,-2,-5,1,4,-4,-8,3,5,7,-1,-4],[-3,-3,9,9,-1,9,3,1,4,8,4,9,1,7,-5,8],[-6,-6,-10,10,-8,-1,-1,-8,-8,6,8,-3,-9,2,5,-2],[2,-5,5,-6,-6,-5,9,-9,10,9,1,8,4,6,1,-10],[7,5,-3,-6,8,-6,-7,-6,-2,-6,8,-7,-3,2,2,-8],[9,3,-4,5,-2,-3,-7,7,-3,-6,-6,-3,-5,-4,-9,4],[-4,4,-7,4,-9,8,3,7,8,-6,6,3,6,-6,8,5],[-8,3,-7,5,-5,3,-8,-5,4,9,1,-9,9,-2,-5,10],[7,9,2,9,4,-5,-2,9,2,-6,-1,-1,-10,7,1,1],[5,-1,3,-4,2,1,2,10,7,-10,-2,-9,8,-10,-7,9]],[[2,-9,-10,3,-6,5,-4,5,-2,-10,-8,7,-8,6,-9,-5],[5,-7,9,-8,-4,-5,10,2,5,-6,-2,10,1,3,-9,4],[-6,-8,1,-2,-10,5,-6,-4,8,5,-3,10,-9,6,5,1],[-3,8,6,10,6,-6,4,-5,-9,2,7,8,9,5,5,3],[6,3,1,8,1,-2,-3,7,1,-6,5,2,1,6,-3,6],[-2,3,-2,-3,-4,-8,-3,-1,-6,3,-2,-3,-4,10,10,9],[3,-7,3,-8,6,6,-2,-6,-7,-10,-3,3,-4,5,7,7],[-5,6,6,-5,-1,4,1,2,6,-8,-8,6,-1,7,-6,9],[1,-7,9,-9,-6,-6,10,6,3,-7,-2,1,8,1,-4,-3],[-6,-2,-6,-9,-6,9,4,-4,-7,6,-7,-5,6,-7,6,5],[-2,-1,2,1,7,-3,-7,-8,9,-6,-10,1,6,-8,8,8],[-4,10,-1,1,-3,6,3,-4,7,-1,2,-5,-10,3,-8,-1],[10,-6,10,-8,4,-3,-8,-1,9,-4,-6,9,1,5,3,-2],[5,3,-7,-10,-4,10,-2,-9,-3,-1,-1,4,-2,-3,-6,9]],[[1,9,-2,-9,-2,2,-9,4,-1,2,7,6,-6,-4,2,-4],[-5,2,-1,-8,9,-8,9,2,-9,4,10,-3,8,7,-3,4],[-3,7,-7,-9,-5,7,-9,7,-3,1,-10,-8,-4,-10,-7,4],[1,3,8,-6,-7,8,2,-3,8,-8,8,9,-2,-8,-3,8],[4,2,3,7,-3,-7,2,-9,4,9,3,4,6,6,-4,3],[-10,-2,-5,5,5,2,-2,8,-8,9,-4,1,-7,-1,-3,-3],[-1,6,-5,3,-4,7,-6,3,-3,-7,5,7,7,5,2,5],[-2,-9,-3,-8,5,-5,2,5,8,2,-2,-9,7,-10,-1,-2],[7,1,2,-7,5,-10,-7,2,-8,3,3,10,-6,-9,-5,-2],[-2,10,-10,-3,-8,4,7,-5,10,5,-1,-5,5,-4,5,-3],[-6,-3,2,5,-8,-9,7,-9,3,-4,1,1,-3,8,-10,-7],[4,5,7,-7,-1,9,2,8,-6,-10,-10,-4,-6,1,3,-2],[-9,6,1,2,-2,1,-10,9,-1,-2,-2,4,-9,3,4,-3],[-4,-10,2,6,-2,8,-8,-8,10,2,6,-7,2,1,7,10]],[[10,-6,-1,-8,3,-4,7,-10,2,6,-7,-3,4,6,7,-3],[-7,1,5,-10,7,9,-6,10,2,-5,8,-9,-1,6,-10,-5],[9,-6,7,-5,2,10,-3,-5,4,-4,-8,-9,-8,9,3,7],[-1,-10,-9,-5,-7,8,-5,4,4,4,-3,-3,9,10,-5,5],[2,-6,-8,-3,-6,-6,-5,-10,10,-1,10,-5,5,10,4,7],[1,-8,-4,4,3,1,-2,-7,9,-5,-3,6,6,9,1,8],[9,1,-9,3,4,-10,-7,9,6,-4,6,8,-6,-6,2,7],[1,3,2,2,2,-4,-10,6,1,-6,10,1,-2,2,-10,-3],[8,1,-6,10,6,8,9,1,-3,-6,-2,10,-3,-5,-5,6],[-8,5,-1,-1,-9,10,-2,7,-9,-10,-2,-2,-3,10,7,1],[-4,-1,-1,1,-3,7,3,3,-7,-10,1,3,-10,4,-10,7],[3,-10,-10,6,1,1,7,3,-4,2,6,-3,-2,-6,-6,-8],[-10,-5,7,-6,-3,2,10,4,2,-1,-3,7,-3,-10,5,-10],[-6,7,-4,-4,9,-9,-3,3,10,6,-10,3,-5,4,8,-6]],[[7,-10,8,2,-9,-6,-5,-10,5,-2,-6,-10,-5,2,8,-10],[3,8,1,9,-10,-4,4,-1,7,-5,8,5,-2,9,-5,5],[-8,-9,2,2,-1,-8,-10,7,4,-2,9,-8,-4,-9,7,-7],[6,5,-2,-9,-3,9,7,1,-7,-6,10,-5,10,-3,9,-1],[4,2,-4,10,-7,-1,2,-8,6,-5,-5,-1,8,9,2,2],[-7,-3,-3,-6,-1,-7,-10,-3,-8,-3,1,-7,5,-6,-5,4],[-1,10,-1,6,-5,4,7,-2,-6,-7,8,-5,9,2,8,-7],[7,2,-6,-8,3,-8,7,-7,-6,9,-8,9,-4,3,-10,-2],[10,5,-4,10,-4,10,-10,-6,1,2,-8,9,3,5,-4,10],[-2,-3,7,4,2,5,-10,10,-4,3,5,-7,7,4,8,-7],[-1,7,-2,1,4,7,7,-5,10,-5,1,-3,-4,10,-1,4],[-7,-9,-1,-4,6,-7,7,-4,-2,-1,-10,9,-9,10,6,-9],[-5,2,9,-9,7,8,9,-10,10,4,-4,-8,10,3,1,8],[2,-10,-8,-6,8,-9,-5,-10,10,6,4,9,1,-1,3,6]]], dtype='int16')
module1.set_input('var_307', input_307)
input_308= np.array([[[1,-6,8,4,-5,3,-1,4,-6,7,4,6,-7,-10,8,8],[4,10,6,-8,-8,1,-5,-1,7,10,4,-9,-5,-3,-4,6],[-7,-2,2,-5,2,-10,-3,5,-10,3,3,9,10,-2,-5,4],[7,-6,-7,-9,-1,5,3,5,-7,9,10,-7,-2,-7,-9,8],[1,5,8,-7,5,4,2,-3,-8,10,-2,10,-5,-8,10,-10],[-3,8,-9,10,-7,-3,-4,1,-1,-1,7,5,-10,8,-3,2],[-1,7,5,5,-4,4,-7,1,3,10,-5,2,10,-7,8,10],[10,-1,-9,-9,-7,10,-8,-1,-4,3,-8,5,6,-2,1,-6],[-1,6,-3,10,1,-1,2,-2,-5,-8,-1,-10,4,3,-3,-6],[6,-2,-10,7,4,-10,2,-5,-8,-9,-5,-8,-8,-4,-2,5],[4,-4,-10,4,9,-2,-8,9,1,5,-4,7,-5,-4,3,2],[-5,-2,7,10,-1,-8,-1,-3,2,10,-2,-9,8,-6,3,8],[4,4,5,-9,6,5,4,-2,7,-6,9,-10,9,9,3,9],[-7,-1,6,-9,7,2,4,-2,5,2,3,2,7,5,9,9]],[[8,-8,-10,-7,-3,-2,2,-7,-2,-3,-2,-9,6,-1,8,1],[-10,10,-8,-10,-7,-5,6,8,-3,-5,5,4,-7,-6,-6,3],[-10,9,4,-8,-10,2,6,-4,-10,-4,4,-6,-6,10,-1,-10],[-7,-8,6,4,8,-4,7,-5,-8,1,6,4,10,-4,6,2],[2,-6,6,6,7,10,7,-2,2,-8,-1,-2,6,-3,-1,8],[5,-2,6,-4,-6,-9,-10,10,6,-4,-10,-1,-4,1,-9,7],[-2,4,-6,5,3,-7,-6,-6,-6,-1,-7,-7,7,6,-4,2],[3,8,-4,-3,8,1,10,-8,8,-5,2,-5,9,-4,6,10],[2,1,2,5,6,4,5,-2,-1,-9,-10,-3,8,1,3,-5],[1,3,10,3,-5,-1,9,5,-2,-5,1,-3,-5,-4,8,-7],[5,-2,7,-2,2,-5,9,-3,4,-5,-3,-3,5,-1,7,3],[8,-9,-3,-9,4,-6,-6,6,-6,-5,1,9,5,10,8,2],[-9,10,1,8,5,-2,2,1,1,9,3,-10,10,10,1,3],[-4,3,4,3,-5,10,-3,7,6,5,-6,8,3,5,10,6]],[[-4,-9,10,8,7,5,9,-3,10,4,3,3,10,1,-9,-3],[-8,1,-6,-3,9,10,-2,4,-1,7,10,1,1,1,-3,-2],[9,-8,6,-6,3,-4,-1,4,-7,-7,2,-1,1,2,-6,-4],[-10,8,8,-1,-1,4,5,5,9,-1,1,-2,3,6,-8,4],[3,-4,-7,-1,-3,-10,1,10,-7,-8,4,1,-4,9,-7,10],[7,2,2,-2,-6,-1,10,-6,-6,-7,6,8,-5,2,9,4],[7,-8,9,3,-4,4,-8,1,5,-4,6,-1,-8,4,4,-3],[-4,-9,5,-8,-6,1,3,6,9,-10,-2,7,3,-9,-7,-5],[7,6,4,9,8,-4,3,10,-1,-9,1,-5,5,-9,-8,7],[-10,5,7,-4,5,8,9,2,-6,3,6,2,-1,8,-7,5],[6,-1,-8,-9,-4,-3,-3,-5,7,-8,7,6,1,9,9,-5],[-6,7,1,8,-3,10,-7,8,-2,-1,-8,-7,3,7,10,4],[-8,1,-8,-7,-2,-10,-10,7,5,-1,4,-2,-3,-6,2,-10],[-4,-2,-2,5,3,-6,-6,-5,8,-3,-3,9,7,9,-9,2]],[[4,6,-10,-1,-8,6,-8,4,8,-10,4,6,-9,3,-8,9],[-6,7,8,8,2,8,4,10,1,2,-1,9,5,-6,-6,4],[-2,7,-6,-5,-9,-2,7,5,-10,-5,7,-1,-7,-7,3,-2],[-10,-4,4,-7,-2,-5,9,-9,9,1,10,-2,10,8,1,-10],[-1,-9,6,7,5,9,4,-4,-1,-10,-1,-6,2,-6,-5,3],[-3,-2,-8,-8,-1,-8,4,9,-1,8,1,-10,-9,-5,-5,2],[-1,-8,3,4,3,-10,-6,-4,-6,-6,8,10,-4,-4,3,8],[10,5,-2,-9,2,3,8,4,2,3,-8,7,-7,-5,-7,-4],[-9,-10,7,7,1,-7,9,-3,1,-1,8,-5,9,5,10,-10],[-6,9,2,-3,6,-9,9,-8,-9,-1,6,-4,-3,8,-6,-4],[1,-6,-1,-3,6,4,6,-8,-10,4,5,9,-6,-5,7,5],[-5,8,-1,-5,-7,-4,10,9,-3,5,10,-7,-5,9,2,-8],[6,-2,-9,3,5,-8,-9,8,10,1,-9,8,-1,-6,2,1],[-1,6,10,-5,-9,4,2,2,-5,-5,1,10,2,-6,-9,-4]],[[-2,10,2,-8,10,2,4,-7,-7,5,6,2,9,-2,-7,-9],[-9,-2,4,-3,-4,10,9,6,10,5,-5,3,9,4,-4,-6],[10,5,7,-10,-2,-8,-8,8,4,-3,4,-6,-5,-8,1,-1],[1,-7,-3,1,7,-6,-4,2,-2,10,5,7,-4,9,1,8],[-4,-2,4,-2,-7,-8,5,-10,7,10,-5,1,4,-7,-10,6],[-2,2,-4,-8,5,1,-6,-6,2,-6,-7,-6,-9,-3,-10,4],[9,-9,-10,-5,1,-4,8,1,-10,-7,-3,5,8,10,8,1],[10,2,6,-7,-3,5,3,-1,2,9,1,-7,3,-6,-3,-8],[-8,7,8,-5,4,-2,8,1,1,5,-3,2,10,-7,-6,8],[-7,-10,-8,6,4,1,1,-2,-2,-6,-7,-1,8,-4,2,10],[6,2,4,-7,6,-2,2,1,1,8,-8,7,-4,-2,-2,6],[4,-3,8,9,-8,2,5,3,-5,-3,10,10,-2,10,9,-1],[6,1,4,7,-1,3,6,-1,6,3,4,10,10,-10,-8,9],[-7,-7,10,-4,-6,-6,-10,1,-1,-2,2,2,1,-10,-3,4]]], dtype='int16')
module1.set_input('var_308', input_308)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_307, input_308, )
res3 = intrp3.evaluate()(input_307, input_308, )
res4 = intrp4.evaluate()(input_307, input_308, )
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
module5.set_input('var_307', input_307)
module5.set_input('var_308', input_308)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_307, input_308, )
res7 = intrp7.evaluate()(input_307, input_308, )
res8 = intrp8.evaluate()(input_307, input_308, )
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
module9.set_input('var_307', input_307)
module9.set_input('var_308', input_308)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_307, input_308, )
res11 = intrp11.evaluate()(input_307, input_308, )
res12 = intrp12.evaluate()(input_307, input_308, )
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
module13.set_input('var_307', input_307)
module13.set_input('var_308', input_308)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_307, input_308, )
res15 = intrp15.evaluate()(input_307, input_308, )
res16 = intrp16.evaluate()(input_307, input_308, )
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
module17.set_input('var_307', input_307)
module17.set_input('var_308', input_308)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_307, input_308, )
res19 = intrp19.evaluate()(input_307, input_308, )
res20 = intrp20.evaluate()(input_307, input_308, )
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
module21.set_input('var_307', input_307)
module21.set_input('var_308', input_308)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_307, input_308, )
res23 = intrp23.evaluate()(input_307, input_308, )
res24 = intrp24.evaluate()(input_307, input_308, )
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

'''22: TVMFuncCall
21: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
20: tvm::transform::Pass::operator()(tvm::IRModule) const
19: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
18: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
17: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
16: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
15: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
14: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
13: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
12: _ZNSt17_Function_handlerIFSt10sha
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::TupleNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''