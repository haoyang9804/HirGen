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
uop_1 = relay.log(var_0.astype('float32')) # shape=()
bop_3 = relay.bitwise_xor(uop_1.astype('uint8'), var_0.astype('uint8')) # shape=()
var_6 = relay.var("var_6", dtype = "float32", shape = (1,))#candidate|6|(1,)|var|float32
bop_7 = relay.power(uop_1.astype('float64'), var_6.astype('float64')) # shape=()
uop_10 = relay.cosh(bop_3.astype('float32')) # shape=()
bop_12 = relay.bitwise_xor(bop_7.astype('uint16'), uop_1.astype('uint16')) # shape=()
uop_15 = relay.cosh(uop_10.astype('float64')) # shape=()
bop_17 = relay.bitwise_xor(uop_10.astype('int32'), uop_1.astype('int32')) # shape=()
uop_20 = relay.atan(uop_15.astype('float64')) # shape=()
bop_22 = relay.right_shift(uop_20.astype('uint16'), bop_7.astype('uint16')) # shape=()
var_25 = relay.var("var_25", dtype = "float64", shape = (8, 2, 12))#candidate|25|(8, 2, 12)|var|float64
bop_26 = relay.bitwise_xor(uop_15.astype('uint64'), var_25.astype('uint64')) # shape=(8, 2, 12)
bop_29 = relay.subtract(bop_26.astype('int8'), uop_20.astype('int8')) # shape=(8, 2, 12)
uop_32 = relay.atanh(bop_22.astype('float32')) # shape=()
uop_34 = relay.cos(uop_32.astype('float64')) # shape=()
const_36 = relay.const([-0.938747], dtype = "float64")#candidate|36|(1,)|const|float64
bop_37 = relay.logical_and(uop_34.astype('bool'), const_36.astype('bool')) # shape=()
bop_40 = relay.maximum(uop_34.astype('uint16'), uop_10.astype('uint16')) # shape=()
const_43 = relay.const([1.939417,-1.129055,-8.741195,9.807461,-6.497251,-1.974782,2.204033], dtype = "float32")#candidate|43|(7,)|const|float32
bop_44 = relay.right_shift(uop_32.astype('uint16'), const_43.astype('uint16')) # shape=(7,)
output = relay.Tuple([bop_12,bop_17,bop_29,bop_37,bop_40,bop_44,])
output2 = relay.Tuple([bop_12,bop_17,bop_29,bop_37,bop_40,bop_44,])
func_47 = relay.Function([var_0,var_6,var_25,], output)
mod['func_47'] = func_47
mod = relay.transform.InferType()(mod)
mutated_mod['func_47'] = func_47
mutated_mod = relay.transform.InferType()(mutated_mod)
func_47_call = mutated_mod.get_global_var('func_47')
var_49 = relay.var("var_49", dtype = "float32", shape = ())#candidate|49|()|var|float32
var_50 = relay.var("var_50", dtype = "float32", shape = (1,))#candidate|50|(1,)|var|float32
var_51 = relay.var("var_51", dtype = "float64", shape = (8, 2, 12))#candidate|51|(8, 2, 12)|var|float64
call_48 = func_47_call(var_49,var_50,var_51,)
output = call_48
func_52 = relay.Function([var_49,var_50,var_51,], output)
mutated_mod['func_52'] = func_52
mutated_mod = relay.transform.InferType()(mutated_mod)
var_54 = relay.var("var_54", dtype = "float32", shape = (10, 11))#candidate|54|(10, 11)|var|float32
const_55 = relay.const([[7.407247,-1.944603,7.102113,9.672320,8.027242,7.628222,-8.092649,3.651406,-8.273440,8.615685,-8.798850],[9.260322,2.612760,3.764225,-9.962558,0.170064,-8.728256,0.692980,-8.598986,1.811248,-1.666672,-1.444597],[-8.350443,-1.492238,8.032063,4.748472,-3.047534,-0.236840,-4.124612,0.606751,-3.418653,-4.040138,-0.667905],[-9.869140,-3.456930,-1.251255,-3.722383,-9.318181,-7.565253,0.791552,-4.659611,-2.128927,-3.750639,-2.137428],[-4.752408,0.247205,7.665503,2.497558,2.465963,-1.036836,-4.651678,-4.396681,3.519834,8.119004,6.896954],[1.475754,6.914587,-8.289104,-0.393432,-8.837191,3.958315,1.327944,-9.621066,-1.546895,-6.407297,3.421372],[-8.784334,7.426386,-0.282113,3.074213,9.501424,-5.385115,-0.885443,-8.526396,1.008694,0.462537,-6.519718],[0.376288,-9.219787,-1.977283,-8.991917,4.909522,6.863274,-9.609187,4.656506,7.603219,-6.739020,3.389115],[6.799143,9.880763,4.427952,6.370768,4.557484,-3.698033,-8.825876,-6.423170,-5.102898,6.735739,-3.770412],[-6.224035,-4.358042,8.142911,0.839305,3.401257,-6.936835,5.659743,-8.502633,-5.806203,-8.173051,0.179288]], dtype = "float32")#candidate|55|(10, 11)|const|float32
bop_56 = relay.mod(var_54.astype('float32'), relay.reshape(const_55.astype('float32'), relay.shape_of(var_54))) # shape=(10, 11)
uop_59 = relay.sigmoid(var_54.astype('float64')) # shape=(10, 11)
const_61 = relay.const([[-1.282329,1.866780,-0.543615,-6.904298,6.663634,-6.705011,-8.075075,-5.806285,0.845929,8.662090,-8.978143],[2.934936,-5.357758,-9.453486,3.707654,5.574826,0.424555,-3.693924,0.932022,9.809661,4.858678,3.188751],[-6.179311,7.043391,2.082785,-8.368664,-8.280637,6.742656,6.519834,-8.337461,-3.448484,8.699327,7.703994],[3.272587,-1.717778,5.245293,-2.505630,3.642675,-6.345745,-0.715662,-6.288968,-2.199214,-8.332211,7.165290],[6.175576,6.035617,1.642428,4.537972,-6.321986,9.970682,5.269114,-8.630771,9.985579,9.632907,-2.395347],[5.893032,-8.892592,-1.945211,1.275597,-2.726543,2.054772,-9.489290,-3.300945,8.326487,7.728253,7.467355],[1.151566,-7.189787,-9.111924,6.628627,-0.685906,0.784571,-1.019172,4.558934,0.485504,-2.306443,0.101212],[2.411641,-3.429722,5.288863,1.600876,-5.103080,-1.701812,9.892003,-2.364416,-5.887601,1.782307,1.310262],[8.976218,7.903572,8.541246,1.345555,4.697429,-1.486695,-2.741388,1.611135,0.636636,-8.404275,-0.305141],[-0.787831,-0.362403,8.528402,-7.667831,3.801813,-0.686736,-0.808998,-3.035308,-4.188441,-7.651264,9.634133]], dtype = "float64")#candidate|61|(10, 11)|const|float64
bop_62 = relay.less_equal(uop_59.astype('bool'), relay.reshape(const_61.astype('bool'), relay.shape_of(uop_59))) # shape=(10, 11)
bop_65 = relay.bitwise_xor(const_61.astype('int16'), relay.reshape(bop_56.astype('int16'), relay.shape_of(const_61))) # shape=(10, 11)
var_68 = relay.var("var_68", dtype = "int16", shape = (10, 11))#candidate|68|(10, 11)|var|int16
bop_69 = relay.multiply(bop_65.astype('int16'), relay.reshape(var_68.astype('int16'), relay.shape_of(bop_65))) # shape=(10, 11)
uop_72 = relay.acos(bop_65.astype('float32')) # shape=(10, 11)
uop_74 = relay.rsqrt(var_54.astype('float64')) # shape=(10, 11)
uop_76 = relay.cos(uop_59.astype('float64')) # shape=(10, 11)
uop_78 = relay.acosh(uop_59.astype('float32')) # shape=(10, 11)
uop_80 = relay.acos(uop_72.astype('float64')) # shape=(10, 11)
var_82 = relay.var("var_82", dtype = "float64", shape = (10, 11))#candidate|82|(10, 11)|var|float64
bop_83 = relay.bitwise_xor(uop_80.astype('int64'), relay.reshape(var_82.astype('int64'), relay.shape_of(uop_80))) # shape=(10, 11)
uop_86 = relay.atanh(bop_62.astype('float64')) # shape=(10, 11)
var_88 = relay.var("var_88", dtype = "float64", shape = (10, 11))#candidate|88|(10, 11)|var|float64
bop_89 = relay.bitwise_and(uop_76.astype('int32'), relay.reshape(var_88.astype('int32'), relay.shape_of(uop_76))) # shape=(10, 11)
uop_92 = relay.sigmoid(uop_78.astype('float32')) # shape=(10, 11)
bop_94 = relay.add(uop_92.astype('int16'), relay.reshape(uop_80.astype('int16'), relay.shape_of(uop_92))) # shape=(10, 11)
bop_97 = relay.maximum(bop_94.astype('uint16'), relay.reshape(uop_76.astype('uint16'), relay.shape_of(bop_94))) # shape=(10, 11)
var_100 = relay.var("var_100", dtype = "float32", shape = (10, 11))#candidate|100|(10, 11)|var|float32
bop_101 = relay.right_shift(uop_72.astype('int8'), relay.reshape(var_100.astype('int8'), relay.shape_of(uop_72))) # shape=(10, 11)
var_104 = relay.var("var_104", dtype = "int16", shape = (10, 11))#candidate|104|(10, 11)|var|int16
bop_105 = relay.greater(bop_94.astype('bool'), relay.reshape(var_104.astype('bool'), relay.shape_of(bop_94))) # shape=(10, 11)
uop_108 = relay.tan(bop_105.astype('float32')) # shape=(10, 11)
uop_110 = relay.asinh(uop_108.astype('float32')) # shape=(10, 11)
bop_112 = relay.equal(uop_110.astype('bool'), relay.reshape(bop_65.astype('bool'), relay.shape_of(uop_110))) # shape=(10, 11)
func_47_call = mod.get_global_var('func_47')
func_52_call = mutated_mod.get_global_var('func_52')
var_116 = relay.var("var_116", dtype = "float32", shape = ())#candidate|116|()|var|float32
var_117 = relay.var("var_117", dtype = "float64", shape = (192,))#candidate|117|(192,)|var|float64
call_115 = relay.TupleGetItem(func_47_call(relay.reshape(var_116.astype('float32'), []), relay.reshape(var_116.astype('float32'), [1,]), relay.reshape(var_117.astype('float64'), [8, 2, 12]), ), 1)
call_118 = relay.TupleGetItem(func_52_call(relay.reshape(var_116.astype('float32'), []), relay.reshape(var_116.astype('float32'), [1,]), relay.reshape(var_117.astype('float64'), [8, 2, 12]), ), 1)
uop_119 = relay.atanh(uop_80.astype('float32')) # shape=(10, 11)
var_121 = relay.var("var_121", dtype = "bool", shape = (10, 11))#candidate|121|(10, 11)|var|bool
bop_122 = relay.less(bop_112.astype('bool'), relay.reshape(var_121.astype('bool'), relay.shape_of(bop_112))) # shape=(10, 11)
output = relay.Tuple([bop_69,uop_74,bop_83,uop_86,bop_89,bop_97,bop_101,call_115,var_116,var_117,uop_119,bop_122,])
output2 = relay.Tuple([bop_69,uop_74,bop_83,uop_86,bop_89,bop_97,bop_101,call_118,var_116,var_117,uop_119,bop_122,])
func_125 = relay.Function([var_54,var_68,var_82,var_88,var_100,var_104,var_116,var_117,var_121,], output)
mod['func_125'] = func_125
mod = relay.transform.InferType()(mod)
mutated_mod['func_125'] = func_125
mutated_mod = relay.transform.InferType()(mutated_mod)
func_125_call = mutated_mod.get_global_var('func_125')
var_127 = relay.var("var_127", dtype = "float32", shape = (10, 11))#candidate|127|(10, 11)|var|float32
var_128 = relay.var("var_128", dtype = "int16", shape = (10, 11))#candidate|128|(10, 11)|var|int16
var_129 = relay.var("var_129", dtype = "float64", shape = (10, 11))#candidate|129|(10, 11)|var|float64
var_130 = relay.var("var_130", dtype = "float64", shape = (10, 11))#candidate|130|(10, 11)|var|float64
var_131 = relay.var("var_131", dtype = "float32", shape = (10, 11))#candidate|131|(10, 11)|var|float32
var_132 = relay.var("var_132", dtype = "int16", shape = (10, 11))#candidate|132|(10, 11)|var|int16
var_133 = relay.var("var_133", dtype = "float32", shape = ())#candidate|133|()|var|float32
var_134 = relay.var("var_134", dtype = "float64", shape = (192,))#candidate|134|(192,)|var|float64
var_135 = relay.var("var_135", dtype = "bool", shape = (10, 11))#candidate|135|(10, 11)|var|bool
call_126 = func_125_call(var_127,var_128,var_129,var_130,var_131,var_132,var_133,var_134,var_135,)
output = call_126
func_136 = relay.Function([var_127,var_128,var_129,var_130,var_131,var_132,var_133,var_134,var_135,], output)
mutated_mod['func_136'] = func_136
mutated_mod = relay.transform.InferType()(mutated_mod)
var_138 = relay.var("var_138", dtype = "uint16", shape = ())#candidate|138|()|var|uint16
const_139 = relay.const([[5,6,4],[-10,-3,9]], dtype = "uint16")#candidate|139|(2, 3)|const|uint16
bop_140 = relay.bitwise_and(var_138.astype('uint16'), const_139.astype('uint16')) # shape=(2, 3)
const_143 = relay.const([[-5,-8,4],[2,-5,9]], dtype = "uint16")#candidate|143|(2, 3)|const|uint16
bop_144 = relay.logical_xor(const_139.astype('int32'), relay.reshape(const_143.astype('int32'), relay.shape_of(const_139))) # shape=(2, 3)
bop_147 = relay.greater(const_143.astype('bool'), relay.reshape(const_139.astype('bool'), relay.shape_of(const_143))) # shape=(2, 3)
uop_150 = relay.erf(const_139.astype('float64')) # shape=(2, 3)
uop_152 = relay.cos(const_143.astype('float32')) # shape=(2, 3)
uop_154 = relay.erf(uop_152.astype('float32')) # shape=(2, 3)
const_156 = relay.const([[-9.423527,-6.306119,-2.493747],[-9.470715,-6.226824,0.926595]], dtype = "float32")#candidate|156|(2, 3)|const|float32
bop_157 = relay.bitwise_and(uop_154.astype('uint8'), relay.reshape(const_156.astype('uint8'), relay.shape_of(uop_154))) # shape=(2, 3)
uop_160 = relay.log2(bop_157.astype('float64')) # shape=(2, 3)
uop_162 = relay.acos(uop_160.astype('float64')) # shape=(2, 3)
bop_164 = relay.power(uop_160.astype('float64'), relay.reshape(uop_154.astype('float64'), relay.shape_of(uop_160))) # shape=(2, 3)
bop_167 = relay.less(uop_154.astype('bool'), relay.reshape(bop_140.astype('bool'), relay.shape_of(uop_154))) # shape=(2, 3)
uop_170 = relay.asin(bop_167.astype('float64')) # shape=(2, 3)
bop_172 = relay.less_equal(uop_162.astype('bool'), relay.reshape(const_156.astype('bool'), relay.shape_of(uop_162))) # shape=(2, 3)
uop_175 = relay.sigmoid(uop_170.astype('float32')) # shape=(2, 3)
bop_177 = relay.logical_or(bop_164.astype('bool'), relay.reshape(const_143.astype('bool'), relay.shape_of(bop_164))) # shape=(2, 3)
bop_180 = relay.bitwise_xor(uop_160.astype('uint8'), relay.reshape(const_156.astype('uint8'), relay.shape_of(uop_160))) # shape=(2, 3)
var_183 = relay.var("var_183", dtype = "uint16", shape = (2, 3))#candidate|183|(2, 3)|var|uint16
bop_184 = relay.multiply(const_143.astype('uint64'), relay.reshape(var_183.astype('uint64'), relay.shape_of(const_143))) # shape=(2, 3)
uop_187 = relay.erf(uop_152.astype('float64')) # shape=(2, 3)
bop_189 = relay.multiply(uop_162.astype('int16'), relay.reshape(uop_150.astype('int16'), relay.shape_of(uop_162))) # shape=(2, 3)
bop_192 = relay.not_equal(bop_172.astype('bool'), relay.reshape(uop_187.astype('bool'), relay.shape_of(bop_172))) # shape=(2, 3)
bop_195 = relay.divide(uop_162.astype('float32'), relay.reshape(uop_175.astype('float32'), relay.shape_of(uop_162))) # shape=(2, 3)
output = relay.Tuple([bop_144,bop_147,bop_177,bop_180,bop_184,bop_189,bop_192,bop_195,])
output2 = relay.Tuple([bop_144,bop_147,bop_177,bop_180,bop_184,bop_189,bop_192,bop_195,])
func_198 = relay.Function([var_138,var_183,], output)
mod['func_198'] = func_198
mod = relay.transform.InferType()(mod)
var_199 = relay.var("var_199", dtype = "uint16", shape = ())#candidate|199|()|var|uint16
var_200 = relay.var("var_200", dtype = "uint16", shape = (2, 3))#candidate|200|(2, 3)|var|uint16
output = func_198(var_199,var_200,)
func_201 = relay.Function([var_199,var_200,], output)
mutated_mod['func_201'] = func_201
mutated_mod = relay.transform.InferType()(mutated_mod)
const_203 = relay.const([[6.444516,1.342281,-1.909841,3.863526],[6.932940,4.049482,4.396788,0.784897],[-4.684469,5.066387,-6.243932,1.938503],[-7.415973,2.137501,6.575762,2.186077],[4.732708,-2.552394,-0.184321,1.969901],[-3.364203,-0.335703,8.951685,-0.663770],[0.707741,-5.279728,0.516477,8.050904],[8.959787,-4.710452,4.069597,0.379892],[6.538686,-6.403655,-2.798869,3.293370],[8.423914,-2.648422,0.233513,-5.912341],[-9.036376,-1.243153,-3.206759,-7.570504],[-6.984923,9.956298,-2.257273,-9.895677],[-6.485163,-7.654060,-6.660702,-5.172190],[2.746092,8.466915,7.796328,2.550848],[-4.257149,8.803607,-6.731098,4.901387],[1.028878,-3.204054,0.850860,6.543283]], dtype = "float32")#candidate|203|(16, 4)|const|float32
uop_204 = relay.atanh(const_203.astype('float32')) # shape=(16, 4)
uop_206 = relay.rsqrt(uop_204.astype('float64')) # shape=(16, 4)
uop_208 = relay.atanh(uop_204.astype('float64')) # shape=(16, 4)
bop_210 = relay.floor_divide(uop_208.astype('float32'), relay.reshape(uop_204.astype('float32'), relay.shape_of(uop_208))) # shape=(16, 4)
bop_213 = relay.divide(uop_208.astype('float64'), relay.reshape(uop_206.astype('float64'), relay.shape_of(uop_208))) # shape=(16, 4)
uop_216 = relay.asin(uop_208.astype('float64')) # shape=(16, 4)
var_218 = relay.var("var_218", dtype = "float32", shape = (16, 4))#candidate|218|(16, 4)|var|float32
bop_219 = relay.divide(bop_210.astype('float64'), relay.reshape(var_218.astype('float64'), relay.shape_of(bop_210))) # shape=(16, 4)
func_198_call = mod.get_global_var('func_198')
func_201_call = mutated_mod.get_global_var('func_201')
var_223 = relay.var("var_223", dtype = "uint16", shape = ())#candidate|223|()|var|uint16
var_224 = relay.var("var_224", dtype = "uint16", shape = (1, 6))#candidate|224|(1, 6)|var|uint16
call_222 = relay.TupleGetItem(func_198_call(relay.reshape(var_223.astype('uint16'), []), relay.reshape(var_224.astype('uint16'), [2, 3]), ), 1)
call_225 = relay.TupleGetItem(func_201_call(relay.reshape(var_223.astype('uint16'), []), relay.reshape(var_224.astype('uint16'), [2, 3]), ), 1)
uop_226 = relay.cos(bop_213.astype('float32')) # shape=(16, 4)
uop_228 = relay.exp(var_218.astype('float32')) # shape=(16, 4)
bop_230 = relay.less(uop_204.astype('bool'), relay.reshape(bop_213.astype('bool'), relay.shape_of(uop_204))) # shape=(16, 4)
bop_233 = relay.maximum(uop_206.astype('uint64'), relay.reshape(bop_219.astype('uint64'), relay.shape_of(uop_206))) # shape=(16, 4)
output = relay.Tuple([uop_216,call_222,var_223,var_224,uop_226,uop_228,bop_230,bop_233,])
output2 = relay.Tuple([uop_216,call_225,var_223,var_224,uop_226,uop_228,bop_230,bop_233,])
func_236 = relay.Function([var_218,var_223,var_224,], output)
mod['func_236'] = func_236
mod = relay.transform.InferType()(mod)
mutated_mod['func_236'] = func_236
mutated_mod = relay.transform.InferType()(mutated_mod)
func_236_call = mutated_mod.get_global_var('func_236')
var_238 = relay.var("var_238", dtype = "float32", shape = (16, 4))#candidate|238|(16, 4)|var|float32
var_239 = relay.var("var_239", dtype = "uint16", shape = ())#candidate|239|()|var|uint16
var_240 = relay.var("var_240", dtype = "uint16", shape = (1, 6))#candidate|240|(1, 6)|var|uint16
call_237 = func_236_call(var_238,var_239,var_240,)
output = call_237
func_241 = relay.Function([var_238,var_239,var_240,], output)
mutated_mod['func_241'] = func_241
mutated_mod = relay.transform.InferType()(mutated_mod)
var_243 = relay.var("var_243", dtype = "float32", shape = (2,))#candidate|243|(2,)|var|float32
uop_244 = relay.sqrt(var_243.astype('float32')) # shape=(2,)
bop_246 = relay.subtract(uop_244.astype('int32'), relay.reshape(var_243.astype('int32'), relay.shape_of(uop_244))) # shape=(2,)
uop_249 = relay.exp(bop_246.astype('float32')) # shape=(2,)
bop_251 = relay.logical_xor(uop_244.astype('uint16'), relay.reshape(var_243.astype('uint16'), relay.shape_of(uop_244))) # shape=(2,)
bop_254 = relay.bitwise_or(uop_249.astype('int64'), relay.reshape(bop_246.astype('int64'), relay.shape_of(uop_249))) # shape=(2,)
uop_257 = relay.asin(uop_244.astype('float32')) # shape=(2,)
uop_259 = relay.sqrt(uop_257.astype('float64')) # shape=(2,)
uop_261 = relay.exp(uop_259.astype('float64')) # shape=(2,)
func_236_call = mod.get_global_var('func_236')
func_241_call = mutated_mod.get_global_var('func_241')
const_264 = relay.const([[-6.603733,-7.277343,-7.777164,-6.374520],[-3.211436,2.691254,7.816391,-9.696794],[3.499449,-8.825794,3.929764,-9.297342],[5.992207,-2.542008,7.924449,8.719471],[-2.188641,1.616189,9.246080,-1.640895],[-2.721801,-9.484381,-0.637601,7.154929],[2.313876,3.632773,5.130240,-2.484843],[3.719707,-9.513930,-2.746144,2.057305],[3.706588,6.252521,-8.153048,7.899426],[9.414131,-5.695171,-0.342187,9.351140],[-4.210560,-0.782955,-0.226259,-5.902511],[-2.898004,-1.893367,-3.105979,0.309708],[-8.083516,7.460688,-2.438940,-9.824724],[-0.049900,-3.929897,-5.549486,-8.717926],[-0.691144,-8.939510,7.730535,4.419192],[3.849893,9.156722,8.005566,-6.294740]], dtype = "float32")#candidate|264|(16, 4)|const|float32
const_265 = relay.const(-8, dtype = "uint16")#candidate|265|()|const|uint16
var_266 = relay.var("var_266", dtype = "uint16", shape = (6, 1))#candidate|266|(6, 1)|var|uint16
call_263 = relay.TupleGetItem(func_236_call(relay.reshape(const_264.astype('float32'), [16, 4]), relay.reshape(const_265.astype('uint16'), []), relay.reshape(var_266.astype('uint16'), [1, 6]), ), 1)
call_267 = relay.TupleGetItem(func_241_call(relay.reshape(const_264.astype('float32'), [16, 4]), relay.reshape(const_265.astype('uint16'), []), relay.reshape(var_266.astype('uint16'), [1, 6]), ), 1)
bop_268 = relay.power(uop_261.astype('float64'), relay.reshape(uop_249.astype('float64'), relay.shape_of(uop_261))) # shape=(2,)
output = relay.Tuple([bop_251,bop_254,call_263,const_264,const_265,var_266,bop_268,])
output2 = relay.Tuple([bop_251,bop_254,call_267,const_264,const_265,var_266,bop_268,])
func_271 = relay.Function([var_243,var_266,], output)
mod['func_271'] = func_271
mod = relay.transform.InferType()(mod)
mutated_mod['func_271'] = func_271
mutated_mod = relay.transform.InferType()(mutated_mod)
func_271_call = mutated_mod.get_global_var('func_271')
var_273 = relay.var("var_273", dtype = "float32", shape = (2,))#candidate|273|(2,)|var|float32
var_274 = relay.var("var_274", dtype = "uint16", shape = (6, 1))#candidate|274|(6, 1)|var|uint16
call_272 = func_271_call(var_273,var_274,)
output = call_272
func_275 = relay.Function([var_273,var_274,], output)
mutated_mod['func_275'] = func_275
mutated_mod = relay.transform.InferType()(mutated_mod)
var_277 = relay.var("var_277", dtype = "bool", shape = (13, 3))#candidate|277|(13, 3)|var|bool
var_278 = relay.var("var_278", dtype = "bool", shape = (13, 3))#candidate|278|(13, 3)|var|bool
bop_279 = relay.logical_or(var_277.astype('bool'), relay.reshape(var_278.astype('bool'), relay.shape_of(var_277))) # shape=(13, 3)
output = bop_279
output2 = bop_279
func_282 = relay.Function([var_277,var_278,], output)
mod['func_282'] = func_282
mod = relay.transform.InferType()(mod)
mutated_mod['func_282'] = func_282
mutated_mod = relay.transform.InferType()(mutated_mod)
func_282_call = mutated_mod.get_global_var('func_282')
var_284 = relay.var("var_284", dtype = "bool", shape = (13, 3))#candidate|284|(13, 3)|var|bool
var_285 = relay.var("var_285", dtype = "bool", shape = (13, 3))#candidate|285|(13, 3)|var|bool
call_283 = func_282_call(var_284,var_285,)
output = call_283
func_286 = relay.Function([var_284,var_285,], output)
mutated_mod['func_286'] = func_286
mutated_mod = relay.transform.InferType()(mutated_mod)
var_288 = relay.var("var_288", dtype = "int64", shape = (12, 3, 8))#candidate|288|(12, 3, 8)|var|int64
var_289 = relay.var("var_289", dtype = "int64", shape = (12, 3, 8))#candidate|289|(12, 3, 8)|var|int64
bop_290 = relay.logical_xor(var_288.astype('int64'), relay.reshape(var_289.astype('int64'), relay.shape_of(var_288))) # shape=(12, 3, 8)
uop_293 = relay.sigmoid(var_288.astype('float64')) # shape=(12, 3, 8)
output = relay.Tuple([bop_290,uop_293,])
output2 = relay.Tuple([bop_290,uop_293,])
func_295 = relay.Function([var_288,var_289,], output)
mod['func_295'] = func_295
mod = relay.transform.InferType()(mod)
var_296 = relay.var("var_296", dtype = "int64", shape = (12, 3, 8))#candidate|296|(12, 3, 8)|var|int64
var_297 = relay.var("var_297", dtype = "int64", shape = (12, 3, 8))#candidate|297|(12, 3, 8)|var|int64
output = func_295(var_296,var_297,)
func_298 = relay.Function([var_296,var_297,], output)
mutated_mod['func_298'] = func_298
mutated_mod = relay.transform.InferType()(mutated_mod)
var_300 = relay.var("var_300", dtype = "float32", shape = (11, 6))#candidate|300|(11, 6)|var|float32
uop_301 = relay.sqrt(var_300.astype('float32')) # shape=(11, 6)
var_303 = relay.var("var_303", dtype = "float32", shape = (11, 6))#candidate|303|(11, 6)|var|float32
bop_304 = relay.less(uop_301.astype('bool'), relay.reshape(var_303.astype('bool'), relay.shape_of(uop_301))) # shape=(11, 6)
func_295_call = mod.get_global_var('func_295')
func_298_call = mutated_mod.get_global_var('func_298')
const_308 = relay.const([-5,-8,-9,6,-8,1,-8,8,-7,-7,-7,1,-1,-8,-2,-5,-7,6,5,9,6,-7,1,6,-2,4,-10,-9,-4,3,3,4,2,5,5,1,8,-4,3,10,5,-5,10,7,2,-1,9,-4,-7,-8,-6,2,5,2,7,3,5,8,-1,-9,-6,-7,-6,-8,9,5,-3,9,4,5,-3,-5,6,4,-8,-10,5,-1,-8,6,-3,5,8,3,5,-5,-8,-7,-4,4,5,-6,3,1,-6,-1,8,4,-9,-9,-9,-3,5,-3,-4,-10,-3,9,-4,9,9,-9,-5,-4,2,-9,10,-7,-5,-3,-9,5,-7,-2,-7,1,3,5,-2,4,-4,6,6,-8,-6,-9,-2,-9,-1,4,-1,5,4,2,1,-2,1,-9,-3,-10,-6,6,-3,10,7,3,-6,4,10,-10,3,6,-1,-3,-9,-5,-9,-4,-9,8,6,-6,10,6,7,1,-7,-4,-1,5,9,7,-5,2,-3,10,10,6,7,9,6,-3,-6,-3,3,7,8,-2,2,6,4,9,-3,5,-6,3,-4,-5,5,-7,6,2,7,8,8,7,10,-10,-10,5,9,8,-6,6,-2,2,7,3,-8,3,5,-1,-7,-1,6,-5,-6,2,7,-5,-1,9,3,-1,-3,8,-6,4,-10,4,-1,-2,-3,2,2,-1,6,-10,3,9,-8,5,-8,9,4,-3,-2,3,-9,5,2,-4,-5,-9,-8,4,7,-9,-8,2,-2,6,-2,-7,-4,2,-8,8], dtype = "int64")#candidate|308|(288,)|const|int64
call_307 = relay.TupleGetItem(func_295_call(relay.reshape(const_308.astype('int64'), [12, 3, 8]), relay.reshape(const_308.astype('int64'), [12, 3, 8]), ), 1)
call_309 = relay.TupleGetItem(func_298_call(relay.reshape(const_308.astype('int64'), [12, 3, 8]), relay.reshape(const_308.astype('int64'), [12, 3, 8]), ), 1)
uop_310 = relay.cos(bop_304.astype('float32')) # shape=(11, 6)
func_295_call = mod.get_global_var('func_295')
func_298_call = mutated_mod.get_global_var('func_298')
call_312 = relay.TupleGetItem(func_295_call(relay.reshape(const_308.astype('int64'), [12, 3, 8]), relay.reshape(call_307.astype('int64'), [12, 3, 8]), ), 1)
call_313 = relay.TupleGetItem(func_298_call(relay.reshape(const_308.astype('int64'), [12, 3, 8]), relay.reshape(call_307.astype('int64'), [12, 3, 8]), ), 1)
uop_314 = relay.atan(uop_310.astype('float64')) # shape=(11, 6)
uop_316 = relay.sqrt(uop_314.astype('float32')) # shape=(11, 6)
uop_318 = relay.cos(uop_310.astype('float32')) # shape=(11, 6)
bop_320 = relay.bitwise_xor(uop_316.astype('int16'), relay.reshape(uop_301.astype('int16'), relay.shape_of(uop_316))) # shape=(11, 6)
bop_323 = relay.less(uop_310.astype('bool'), relay.reshape(var_300.astype('bool'), relay.shape_of(uop_310))) # shape=(11, 6)
uop_326 = relay.acosh(bop_320.astype('float32')) # shape=(11, 6)
output = relay.Tuple([call_307,const_308,call_312,uop_318,bop_323,uop_326,])
output2 = relay.Tuple([call_309,const_308,call_313,uop_318,bop_323,uop_326,])
func_328 = relay.Function([var_300,var_303,], output)
mod['func_328'] = func_328
mod = relay.transform.InferType()(mod)
mutated_mod['func_328'] = func_328
mutated_mod = relay.transform.InferType()(mutated_mod)
func_328_call = mutated_mod.get_global_var('func_328')
var_330 = relay.var("var_330", dtype = "float32", shape = (11, 6))#candidate|330|(11, 6)|var|float32
var_331 = relay.var("var_331", dtype = "float32", shape = (11, 6))#candidate|331|(11, 6)|var|float32
call_329 = func_328_call(var_330,var_331,)
output = call_329
func_332 = relay.Function([var_330,var_331,], output)
mutated_mod['func_332'] = func_332
mutated_mod = relay.transform.InferType()(mutated_mod)
const_334 = relay.const([[[-4.374152,6.061776,-5.838458],[-7.656725,-3.167096,2.125777]],[[-5.539799,7.895837,7.875954],[0.367717,-7.919228,-3.717502]],[[-0.681226,5.290023,0.823123],[-1.752623,0.941718,5.502694]],[[-5.414436,-0.004670,-9.524776],[2.781177,7.063087,-2.531171]],[[3.847085,-8.940865,9.000016],[7.904348,0.035141,-1.358822]],[[-6.873184,7.663894,-3.272503],[9.553364,-0.893322,2.607530]],[[3.137613,6.094728,-6.403873],[8.193102,-6.944198,4.732160]]], dtype = "float32")#candidate|334|(7, 2, 3)|const|float32
uop_335 = relay.exp(const_334.astype('float32')) # shape=(7, 2, 3)
output = relay.Tuple([uop_335,])
output2 = relay.Tuple([uop_335,])
func_337 = relay.Function([], output)
mod['func_337'] = func_337
mod = relay.transform.InferType()(mod)
output = func_337()
func_338 = relay.Function([], output)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
var_339 = relay.var("var_339", dtype = "float32", shape = (11,))#candidate|339|(11,)|var|float32
uop_340 = relay.log10(var_339.astype('float32')) # shape=(11,)
uop_342 = relay.sin(var_339.astype('float64')) # shape=(11,)
uop_344 = relay.sinh(uop_342.astype('float32')) # shape=(11,)
uop_346 = relay.sinh(uop_340.astype('float32')) # shape=(11,)
uop_348 = relay.cos(uop_346.astype('float64')) # shape=(11,)
uop_350 = relay.asinh(uop_348.astype('float32')) # shape=(11,)
uop_352 = relay.atan(uop_350.astype('float32')) # shape=(11,)
output = relay.Tuple([uop_344,uop_352,])
output2 = relay.Tuple([uop_344,uop_352,])
F = relay.Function([var_339,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_339,], output2)
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
input_339= np.array([9.770670,2.294972,-8.346079,1.084558,8.688035,5.752772,8.913287,-0.392939,1.132671,2.877421,-8.188787], dtype='float32')
module1.set_input('var_339', input_339)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_339, )
res3 = intrp3.evaluate()(input_339, )
res4 = intrp4.evaluate()(input_339, )
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
module5.set_input('var_339', input_339)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_339, )
res7 = intrp7.evaluate()(input_339, )
res8 = intrp8.evaluate()(input_339, )
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
module9.set_input('var_339', input_339)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_339, )
res11 = intrp11.evaluate()(input_339, )
res12 = intrp12.evaluate()(input_339, )
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
module13.set_input('var_339', input_339)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_339, )
res15 = intrp15.evaluate()(input_339, )
res16 = intrp16.evaluate()(input_339, )
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
module17.set_input('var_339', input_339)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_339, )
res19 = intrp19.evaluate()(input_339, )
res20 = intrp20.evaluate()(input_339, )
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
module21.set_input('var_339', input_339)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_339, )
res23 = intrp23.evaluate()(input_339, )
res24 = intrp24.evaluate()(input_339, )
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

'''193: TVMFuncCall
192: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::vm::VMCompiler::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
191: tvm::relay::vm::VMCompiler::Lower(tvm::IRModule, tvm::runtime::Map<tvm::Integer, tvm::Target, void, void>, tvm::Target)
190: tvm::relay::vm::VMFunctionCompiler::Compile(tvm::GlobalVar const&, tvm::relay::Function const&)
189: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
188: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
187: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::FunctionNode const*)
186: tvm::relay::vm::VMFunctionCompiler::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
185: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
184: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
183: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
182: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
181: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
180: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
179: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
178: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
177: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
176: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
175: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
174: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
173: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
172: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
171: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
170: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
169: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
168: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
167: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
166: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
165: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
164: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
163: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
162: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
161: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
160: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
159: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
158: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
157: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
156: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
155: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
154: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
153: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
152: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
151: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
150: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
149: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
148: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
147: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
146: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
145: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
144: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
143: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
142: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
141: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
140: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
139: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
138: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
137: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
136: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
135: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
134: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
133: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
132: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
131: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
130: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
129: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
128: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
127: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
126: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
125: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
124: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
123: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
122: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
121: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
120: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
119: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
118: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
117: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
116: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
115: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
114: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
113: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
112: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
111: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
110: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
109: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
108: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
107: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
106: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
105: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
104: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
103: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
102: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
101: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
100: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
99: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
98: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
97: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
96: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
95: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
94: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
93: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
92: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
91: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
90: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
89: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
88: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
87: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
86: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
85: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
84: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
83: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
82: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
81: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
80: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
79: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
78: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
77: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
76: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
75: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
74: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
73: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
72: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
71: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
70: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
69: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
68: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
67: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
66: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
65: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
64: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
63: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
62: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
61: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
60: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
59: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
58: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
57: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
56: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
55: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
54: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
53: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
52: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
51: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
50: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
49: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
48: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
47: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
46: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
45: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
44: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
43: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
42: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
41: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
40: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
39: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
38: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
37: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
36: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
35: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
34: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
33: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
32: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
31: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
30: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
29: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
28: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
27: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
26: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
25: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
24: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
22: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
21: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
20: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
19: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
18: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
17: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
15: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
14: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
11: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
8: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
7: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
6: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
5: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
4: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
3: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
1: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
0: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)

'''