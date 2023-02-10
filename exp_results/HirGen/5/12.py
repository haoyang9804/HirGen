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
var_0 = relay.var("var_0", dtype = "int32", shape = ())#candidate|0|()|var|int32
var_1 = relay.var("var_1", dtype = "int32", shape = (3, 9))#candidate|1|(3, 9)|var|int32
bop_2 = relay.greater(var_0.astype('bool'), var_1.astype('bool')) # shape=(3, 9)
uop_5 = relay.cos(var_1.astype('float32')) # shape=(3, 9)
var_7 = relay.var("var_7", dtype = "int32", shape = (15, 13, 6))#candidate|7|(15, 13, 6)|var|int32
bop_8 = relay.floor_mod(var_0.astype('float64'), var_7.astype('float64')) # shape=(15, 13, 6)
bop_11 = relay.mod(uop_5.astype('float64'), relay.reshape(bop_2.astype('float64'), relay.shape_of(uop_5))) # shape=(3, 9)
uop_14 = relay.log2(bop_11.astype('float32')) # shape=(3, 9)
bop_16 = relay.greater_equal(uop_14.astype('bool'), relay.reshape(bop_2.astype('bool'), relay.shape_of(uop_14))) # shape=(3, 9)
bop_19 = relay.maximum(uop_14.astype('float64'), relay.reshape(bop_16.astype('float64'), relay.shape_of(uop_14))) # shape=(3, 9)
uop_22 = relay.log(bop_2.astype('float64')) # shape=(3, 9)
var_24 = relay.var("var_24", dtype = "float64", shape = (3, 9))#candidate|24|(3, 9)|var|float64
bop_25 = relay.right_shift(uop_22.astype('int64'), relay.reshape(var_24.astype('int64'), relay.shape_of(uop_22))) # shape=(3, 9)
uop_28 = relay.erf(bop_16.astype('float32')) # shape=(3, 9)
bop_30 = relay.maximum(uop_28.astype('int64'), relay.reshape(bop_25.astype('int64'), relay.shape_of(uop_28))) # shape=(3, 9)
uop_33 = relay.log10(bop_11.astype('float64')) # shape=(3, 9)
var_35 = relay.var("var_35", dtype = "float64", shape = (3, 9))#candidate|35|(3, 9)|var|float64
bop_36 = relay.power(uop_33.astype('float64'), relay.reshape(var_35.astype('float64'), relay.shape_of(uop_33))) # shape=(3, 9)
var_39 = relay.var("var_39", dtype = "float64", shape = (3, 9))#candidate|39|(3, 9)|var|float64
bop_40 = relay.bitwise_or(uop_33.astype('uint8'), relay.reshape(var_39.astype('uint8'), relay.shape_of(uop_33))) # shape=(3, 9)
bop_43 = relay.bitwise_and(uop_33.astype('int16'), relay.reshape(uop_5.astype('int16'), relay.shape_of(uop_33))) # shape=(3, 9)
bop_46 = relay.logical_xor(bop_30.astype('int64'), relay.reshape(uop_5.astype('int64'), relay.shape_of(bop_30))) # shape=(3, 9)
uop_49 = relay.atanh(bop_43.astype('float32')) # shape=(3, 9)
var_51 = relay.var("var_51", dtype = "int64", shape = (3, 9))#candidate|51|(3, 9)|var|int64
bop_52 = relay.mod(bop_30.astype('float64'), relay.reshape(var_51.astype('float64'), relay.shape_of(bop_30))) # shape=(3, 9)
bop_55 = relay.add(bop_30.astype('float64'), relay.reshape(uop_5.astype('float64'), relay.shape_of(bop_30))) # shape=(3, 9)
bop_58 = relay.power(uop_14.astype('float64'), relay.reshape(bop_43.astype('float64'), relay.shape_of(uop_14))) # shape=(3, 9)
output = relay.Tuple([bop_8,bop_19,bop_36,bop_40,bop_46,uop_49,bop_52,bop_55,bop_58,])
output2 = relay.Tuple([bop_8,bop_19,bop_36,bop_40,bop_46,uop_49,bop_52,bop_55,bop_58,])
func_61 = relay.Function([var_0,var_1,var_7,var_24,var_35,var_39,var_51,], output)
mod['func_61'] = func_61
mod = relay.transform.InferType()(mod)
var_62 = relay.var("var_62", dtype = "int32", shape = ())#candidate|62|()|var|int32
var_63 = relay.var("var_63", dtype = "int32", shape = (3, 9))#candidate|63|(3, 9)|var|int32
var_64 = relay.var("var_64", dtype = "int32", shape = (15, 13, 6))#candidate|64|(15, 13, 6)|var|int32
var_65 = relay.var("var_65", dtype = "float64", shape = (3, 9))#candidate|65|(3, 9)|var|float64
var_66 = relay.var("var_66", dtype = "float64", shape = (3, 9))#candidate|66|(3, 9)|var|float64
var_67 = relay.var("var_67", dtype = "float64", shape = (3, 9))#candidate|67|(3, 9)|var|float64
var_68 = relay.var("var_68", dtype = "int64", shape = (3, 9))#candidate|68|(3, 9)|var|int64
output = func_61(var_62,var_63,var_64,var_65,var_66,var_67,var_68,)
func_69 = relay.Function([var_62,var_63,var_64,var_65,var_66,var_67,var_68,], output)
mutated_mod['func_69'] = func_69
mutated_mod = relay.transform.InferType()(mutated_mod)
var_71 = relay.var("var_71", dtype = "float32", shape = (1, 4, 7))#candidate|71|(1, 4, 7)|var|float32
uop_72 = relay.asinh(var_71.astype('float32')) # shape=(1, 4, 7)
uop_74 = relay.cos(uop_72.astype('float64')) # shape=(1, 4, 7)
bop_76 = relay.equal(uop_72.astype('bool'), relay.reshape(uop_74.astype('bool'), relay.shape_of(uop_72))) # shape=(1, 4, 7)
uop_79 = relay.asin(uop_74.astype('float32')) # shape=(1, 4, 7)
output = relay.Tuple([bop_76,uop_79,])
output2 = relay.Tuple([bop_76,uop_79,])
func_81 = relay.Function([var_71,], output)
mod['func_81'] = func_81
mod = relay.transform.InferType()(mod)
mutated_mod['func_81'] = func_81
mutated_mod = relay.transform.InferType()(mutated_mod)
var_82 = relay.var("var_82", dtype = "float32", shape = (1, 4, 7))#candidate|82|(1, 4, 7)|var|float32
func_81_call = mutated_mod.get_global_var('func_81')
call_83 = func_81_call(var_82)
output = call_83
func_84 = relay.Function([var_82], output)
mutated_mod['func_84'] = func_84
mutated_mod = relay.transform.InferType()(mutated_mod)
var_86 = relay.var("var_86", dtype = "uint8", shape = ())#candidate|86|()|var|uint8
var_87 = relay.var("var_87", dtype = "uint8", shape = (16, 2, 12))#candidate|87|(16, 2, 12)|var|uint8
bop_88 = relay.less(var_86.astype('bool'), var_87.astype('bool')) # shape=(16, 2, 12)
var_91 = relay.var("var_91", dtype = "bool", shape = (16, 2, 12))#candidate|91|(16, 2, 12)|var|bool
bop_92 = relay.subtract(bop_88.astype('uint32'), relay.reshape(var_91.astype('uint32'), relay.shape_of(bop_88))) # shape=(16, 2, 12)
var_95 = relay.var("var_95", dtype = "uint8", shape = (16, 2, 12))#candidate|95|(16, 2, 12)|var|uint8
bop_96 = relay.greater_equal(var_87.astype('bool'), relay.reshape(var_95.astype('bool'), relay.shape_of(var_87))) # shape=(16, 2, 12)
func_81_call = mod.get_global_var('func_81')
func_84_call = mutated_mod.get_global_var('func_84')
const_100 = relay.const([-3.048230,7.675644,-5.642678,3.559808,4.524989,-9.833245,3.515734,-6.436284,4.814111,-6.990186,4.530402,8.474229,6.181763,8.831596,2.666154,5.068456,-3.875271,3.453367,0.190734,-3.400461,3.466281,-0.191811,2.670223,-3.963017,-0.727443,1.856479,-0.458785,-3.021364], dtype = "float32")#candidate|100|(28,)|const|float32
call_99 = relay.TupleGetItem(func_81_call(relay.reshape(const_100.astype('float32'), [1, 4, 7])), 0)
call_101 = relay.TupleGetItem(func_84_call(relay.reshape(const_100.astype('float32'), [1, 4, 7])), 0)
func_81_call = mod.get_global_var('func_81')
func_84_call = mutated_mod.get_global_var('func_84')
call_102 = relay.TupleGetItem(func_81_call(relay.reshape(const_100.astype('float32'), [1, 4, 7])), 1)
call_103 = relay.TupleGetItem(func_84_call(relay.reshape(const_100.astype('float32'), [1, 4, 7])), 1)
uop_104 = relay.atan(const_100.astype('float64')) # shape=(28,)
uop_106 = relay.acos(uop_104.astype('float64')) # shape=(28,)
uop_108 = relay.log2(uop_104.astype('float64')) # shape=(28,)
bop_110 = relay.less_equal(uop_104.astype('bool'), relay.reshape(uop_108.astype('bool'), relay.shape_of(uop_104))) # shape=(28,)
uop_113 = relay.cos(uop_108.astype('float32')) # shape=(28,)
uop_115 = relay.log(uop_104.astype('float32')) # shape=(28,)
uop_117 = relay.erf(uop_113.astype('float64')) # shape=(28,)
output = relay.Tuple([bop_92,bop_96,call_99,call_102,uop_106,bop_110,uop_115,uop_117,])
output2 = relay.Tuple([bop_92,bop_96,call_101,call_103,uop_106,bop_110,uop_115,uop_117,])
func_119 = relay.Function([var_86,var_87,var_91,var_95,], output)
mod['func_119'] = func_119
mod = relay.transform.InferType()(mod)
var_120 = relay.var("var_120", dtype = "uint8", shape = ())#candidate|120|()|var|uint8
var_121 = relay.var("var_121", dtype = "uint8", shape = (16, 2, 12))#candidate|121|(16, 2, 12)|var|uint8
var_122 = relay.var("var_122", dtype = "bool", shape = (16, 2, 12))#candidate|122|(16, 2, 12)|var|bool
var_123 = relay.var("var_123", dtype = "uint8", shape = (16, 2, 12))#candidate|123|(16, 2, 12)|var|uint8
output = func_119(var_120,var_121,var_122,var_123,)
func_124 = relay.Function([var_120,var_121,var_122,var_123,], output)
mutated_mod['func_124'] = func_124
mutated_mod = relay.transform.InferType()(mutated_mod)
const_126 = relay.const(2.007695, dtype = "float64")#candidate|126|()|const|float64
uop_127 = relay.tan(const_126.astype('float64')) # shape=()
uop_129 = relay.exp(uop_127.astype('float32')) # shape=()
bop_131 = relay.maximum(uop_129.astype('float64'), uop_127.astype('float64')) # shape=()
uop_134 = relay.sigmoid(uop_127.astype('float64')) # shape=()
bop_136 = relay.right_shift(uop_129.astype('int32'), uop_127.astype('int32')) # shape=()
uop_139 = relay.cos(bop_131.astype('float64')) # shape=()
uop_141 = relay.tan(uop_139.astype('float64')) # shape=()
bop_143 = relay.add(uop_139.astype('float32'), bop_131.astype('float32')) # shape=()
var_146 = relay.var("var_146", dtype = "float32", shape = ())#candidate|146|()|var|float32
bop_147 = relay.left_shift(uop_129.astype('int32'), var_146.astype('int32')) # shape=()
uop_150 = relay.acos(uop_141.astype('float64')) # shape=()
uop_152 = relay.sigmoid(uop_150.astype('float64')) # shape=()
uop_154 = relay.sinh(uop_150.astype('float64')) # shape=()
var_156 = relay.var("var_156", dtype = "float64", shape = (3,))#candidate|156|(3,)|var|float64
bop_157 = relay.floor_mod(uop_141.astype('float32'), var_156.astype('float32')) # shape=(3,)
bop_160 = relay.divide(uop_139.astype('float32'), const_126.astype('float32')) # shape=()
uop_163 = relay.erf(uop_154.astype('float64')) # shape=()
const_165 = relay.const([[-5.166118,-0.625511,1.112783,-0.535901],[-0.042107,5.381763,3.404809,2.463748],[7.492165,-6.461182,6.788261,9.922862],[8.584151,-2.190207,-5.231489,-8.078722],[-2.190303,7.150513,-4.676634,8.243402],[-5.027265,2.379456,3.480376,6.641207],[0.674744,8.201193,-7.104074,6.474830],[-4.536015,1.320836,3.850985,7.291576],[-2.506414,-8.072388,-6.711772,0.034145],[-2.895775,8.192425,7.272049,-1.736920],[-5.218309,-4.280730,2.154888,0.528436],[0.137277,-2.837040,-5.310981,9.288909]], dtype = "float64")#candidate|165|(12, 4)|const|float64
bop_166 = relay.bitwise_and(uop_150.astype('uint32'), const_165.astype('uint32')) # shape=(12, 4)
var_169 = relay.var("var_169", dtype = "float64", shape = ())#candidate|169|()|var|float64
bop_170 = relay.multiply(uop_163.astype('float64'), var_169.astype('float64')) # shape=()
var_173 = relay.var("var_173", dtype = "float32", shape = (3,))#candidate|173|(3,)|var|float32
bop_174 = relay.minimum(bop_157.astype('float64'), relay.reshape(var_173.astype('float64'), relay.shape_of(bop_157))) # shape=(3,)
bop_177 = relay.logical_xor(bop_170.astype('int32'), bop_160.astype('int32')) # shape=()
uop_180 = relay.cosh(bop_170.astype('float64')) # shape=()
bop_182 = relay.mod(bop_174.astype('float64'), bop_131.astype('float64')) # shape=(3,)
bop_185 = relay.floor_divide(uop_180.astype('float32'), uop_141.astype('float32')) # shape=()
uop_188 = relay.cosh(bop_185.astype('float32')) # shape=()
bop_190 = relay.less_equal(bop_185.astype('bool'), uop_134.astype('bool')) # shape=()
const_193 = relay.const([-5.668786,-9.297411,6.190923,8.250676,-0.420758,-1.698829,-6.960887,-1.834073,6.607732,5.919087,6.331428,-8.000586,2.597798], dtype = "float64")#candidate|193|(13,)|const|float64
bop_194 = relay.power(uop_180.astype('float32'), const_193.astype('float32')) # shape=(13,)
var_197 = relay.var("var_197", dtype = "float32", shape = (15,))#candidate|197|(15,)|var|float32
bop_198 = relay.not_equal(uop_188.astype('bool'), var_197.astype('bool')) # shape=(15,)
uop_201 = relay.log2(uop_188.astype('float64')) # shape=()
output = relay.Tuple([bop_136,bop_143,bop_147,uop_152,bop_166,bop_177,bop_182,bop_190,bop_194,bop_198,uop_201,])
output2 = relay.Tuple([bop_136,bop_143,bop_147,uop_152,bop_166,bop_177,bop_182,bop_190,bop_194,bop_198,uop_201,])
func_203 = relay.Function([var_146,var_156,var_169,var_173,var_197,], output)
mod['func_203'] = func_203
mod = relay.transform.InferType()(mod)
var_204 = relay.var("var_204", dtype = "float32", shape = ())#candidate|204|()|var|float32
var_205 = relay.var("var_205", dtype = "float64", shape = (3,))#candidate|205|(3,)|var|float64
var_206 = relay.var("var_206", dtype = "float64", shape = ())#candidate|206|()|var|float64
var_207 = relay.var("var_207", dtype = "float32", shape = (3,))#candidate|207|(3,)|var|float32
var_208 = relay.var("var_208", dtype = "float32", shape = (15,))#candidate|208|(15,)|var|float32
output = func_203(var_204,var_205,var_206,var_207,var_208,)
func_209 = relay.Function([var_204,var_205,var_206,var_207,var_208,], output)
mutated_mod['func_209'] = func_209
mutated_mod = relay.transform.InferType()(mutated_mod)
var_211 = relay.var("var_211", dtype = "float32", shape = (5, 5, 16))#candidate|211|(5, 5, 16)|var|float32
uop_212 = relay.log2(var_211.astype('float32')) # shape=(5, 5, 16)
output = uop_212
output2 = uop_212
func_214 = relay.Function([var_211,], output)
mod['func_214'] = func_214
mod = relay.transform.InferType()(mod)
mutated_mod['func_214'] = func_214
mutated_mod = relay.transform.InferType()(mutated_mod)
var_215 = relay.var("var_215", dtype = "float32", shape = (5, 5, 16))#candidate|215|(5, 5, 16)|var|float32
func_214_call = mutated_mod.get_global_var('func_214')
call_216 = func_214_call(var_215)
output = call_216
func_217 = relay.Function([var_215], output)
mutated_mod['func_217'] = func_217
mutated_mod = relay.transform.InferType()(mutated_mod)
const_219 = relay.const([[2.175851,-1.596749,9.324127,8.055951,-5.130722,2.625232,7.135001,7.615420,2.802565,0.073294,6.370703,7.307289,4.669151,-7.260696,-4.628124],[9.511400,-7.595959,-2.202644,8.859111,1.416910,0.927235,-6.749096,4.761694,-2.550066,-1.133072,-2.774909,2.673903,1.628695,-5.547144,1.341562],[3.871591,9.064633,-8.935687,-0.277067,8.985312,6.553464,-6.881768,2.907005,-6.975928,-0.044998,4.113040,-0.931033,9.573972,-8.425126,2.783540],[-2.554872,-4.255025,-1.652384,7.974261,-5.087897,-2.066974,-7.088559,-3.266435,-0.251188,9.723377,9.282475,-2.741698,7.797268,0.275850,8.020149],[3.709138,-4.375623,4.520030,1.768463,7.271477,1.016988,1.310841,-0.597007,7.433119,-7.450413,8.727981,-5.619921,0.046077,1.750716,0.706025],[-6.952053,-5.788446,-6.320342,1.363647,7.289602,2.025960,3.912448,-2.519614,-3.328547,-8.369450,-5.417479,6.093300,1.974199,-8.970108,4.968163],[-5.455492,-9.353395,-2.030017,-5.997561,-9.990561,-2.849888,7.061588,3.083077,4.047126,1.643524,-9.657736,-1.321565,-3.528051,1.187710,-2.879288],[-1.763841,-2.124724,-2.679852,4.910323,-4.281290,-5.931392,-7.977065,-6.940808,5.405272,8.122345,-0.792824,-3.055842,9.863762,-5.993852,4.640774],[-9.831900,5.460142,-8.157481,0.446698,5.410838,-2.199326,9.624573,8.087284,-2.641450,8.474556,1.462223,0.012041,-6.933095,2.360304,6.230902],[-1.364781,-8.881814,-8.671240,6.984763,0.868022,-0.395247,-6.270351,-4.987018,5.169381,-5.815267,-2.570008,-3.553540,9.364076,9.264283,-4.749204],[5.356837,-6.665317,-6.026872,-5.131523,-1.905842,-8.926324,0.623133,-0.511288,-9.578605,-7.037419,3.399439,0.594027,4.182838,-7.916840,2.214180],[6.691514,0.904687,-9.430041,4.254762,9.388680,-1.177717,-5.031907,-7.437737,1.728048,-2.709399,1.641077,4.849783,-6.557881,4.927529,4.748877],[8.206262,3.191985,-8.165718,-4.521603,8.199850,4.934983,-3.974525,1.081619,5.700417,3.472003,6.679495,1.151729,0.133421,4.860617,-9.343241]], dtype = "float32")#candidate|219|(13, 15)|const|float32
uop_220 = relay.sin(const_219.astype('float32')) # shape=(13, 15)
uop_222 = relay.sinh(const_219.astype('float64')) # shape=(13, 15)
uop_224 = relay.erf(const_219.astype('float32')) # shape=(13, 15)
output = relay.Tuple([uop_220,uop_222,uop_224,])
output2 = relay.Tuple([uop_220,uop_222,uop_224,])
func_226 = relay.Function([], output)
mod['func_226'] = func_226
mod = relay.transform.InferType()(mod)
mutated_mod['func_226'] = func_226
mutated_mod = relay.transform.InferType()(mutated_mod)
func_226_call = mutated_mod.get_global_var('func_226')
call_227 = func_226_call()
output = call_227
func_228 = relay.Function([], output)
mutated_mod['func_228'] = func_228
mutated_mod = relay.transform.InferType()(mutated_mod)
var_229 = relay.var("var_229", dtype = "float64", shape = (7, 1, 14))#candidate|229|(7, 1, 14)|var|float64
uop_230 = relay.acos(var_229.astype('float64')) # shape=(7, 1, 14)
output = relay.Tuple([uop_230,])
output2 = relay.Tuple([uop_230,])
func_232 = relay.Function([var_229,], output)
mod['func_232'] = func_232
mod = relay.transform.InferType()(mod)
mutated_mod['func_232'] = func_232
mutated_mod = relay.transform.InferType()(mutated_mod)
var_233 = relay.var("var_233", dtype = "float64", shape = (7, 1, 14))#candidate|233|(7, 1, 14)|var|float64
func_232_call = mutated_mod.get_global_var('func_232')
call_234 = func_232_call(var_233)
output = call_234
func_235 = relay.Function([var_233], output)
mutated_mod['func_235'] = func_235
mutated_mod = relay.transform.InferType()(mutated_mod)
var_237 = relay.var("var_237", dtype = "uint64", shape = (5,))#candidate|237|(5,)|var|uint64
var_238 = relay.var("var_238", dtype = "uint64", shape = (5,))#candidate|238|(5,)|var|uint64
bop_239 = relay.left_shift(var_237.astype('uint64'), relay.reshape(var_238.astype('uint64'), relay.shape_of(var_237))) # shape=(5,)
output = relay.Tuple([bop_239,])
output2 = relay.Tuple([bop_239,])
func_242 = relay.Function([var_237,var_238,], output)
mod['func_242'] = func_242
mod = relay.transform.InferType()(mod)
mutated_mod['func_242'] = func_242
mutated_mod = relay.transform.InferType()(mutated_mod)
func_242_call = mutated_mod.get_global_var('func_242')
var_244 = relay.var("var_244", dtype = "uint64", shape = (5,))#candidate|244|(5,)|var|uint64
var_245 = relay.var("var_245", dtype = "uint64", shape = (5,))#candidate|245|(5,)|var|uint64
call_243 = func_242_call(var_244,var_245,)
output = call_243
func_246 = relay.Function([var_244,var_245,], output)
mutated_mod['func_246'] = func_246
mutated_mod = relay.transform.InferType()(mutated_mod)
var_248 = relay.var("var_248", dtype = "float32", shape = (16, 12))#candidate|248|(16, 12)|var|float32
uop_249 = relay.tan(var_248.astype('float32')) # shape=(16, 12)
uop_251 = relay.cosh(var_248.astype('float32')) # shape=(16, 12)
bop_253 = relay.maximum(uop_251.astype('uint8'), relay.reshape(uop_249.astype('uint8'), relay.shape_of(uop_251))) # shape=(16, 12)
bop_256 = relay.maximum(uop_249.astype('int64'), relay.reshape(var_248.astype('int64'), relay.shape_of(uop_249))) # shape=(16, 12)
bop_259 = relay.power(var_248.astype('float32'), relay.reshape(bop_253.astype('float32'), relay.shape_of(var_248))) # shape=(16, 12)
const_262 = relay.const([[-3.474705,2.822673,0.098984,-7.810647,-9.396813,6.335987,9.182318,-6.103956,-8.338657,7.662741,1.168249,-2.386832],[0.007543,-1.201967,-7.843487,2.860830,-3.702849,-7.402797,1.765097,-5.879113,-1.000188,-9.848980,-3.128486,-7.982407],[-4.164415,-6.479670,8.848542,3.702483,-1.886657,-6.803112,5.015631,-0.782513,-3.182849,4.053507,3.657820,-4.262458],[9.484549,5.508310,2.140867,-0.550842,2.468724,8.516614,-9.370651,-6.354438,3.575470,6.935712,-7.201631,-4.430046],[8.048185,-9.178892,7.219338,-6.450733,3.802522,8.046735,-3.847469,-9.704007,-2.963976,-9.632584,-5.428585,-8.119369],[8.734337,3.627869,3.374593,2.311243,0.919804,1.573682,-6.941720,-2.525250,2.480577,-4.671870,0.136582,-9.764876],[-9.572012,-3.386275,9.087116,-0.082034,0.754948,-1.126964,3.642552,6.500500,-7.521286,-9.505209,-8.975044,-6.198229],[-1.405819,3.932862,-2.435828,7.220631,-0.863331,3.005282,-8.161368,1.188982,6.980051,-0.008313,0.764173,-7.139714],[-8.375696,-9.040499,1.530216,4.768702,5.402868,5.500365,-3.396194,-8.621721,7.876859,-3.815020,-1.854327,4.554618],[7.864605,-2.089849,3.144647,-0.696548,3.317025,-4.113025,-9.346556,-8.049536,7.796545,6.413002,8.002283,-2.639314],[8.682465,-1.397085,-0.082812,-8.775792,-8.899219,-6.108678,0.744712,1.229271,2.069336,5.592339,9.963929,2.215821],[2.998546,-2.851982,5.646403,2.483321,5.213748,-8.076970,3.427545,-7.980831,4.063963,0.257720,2.430445,-0.135747],[8.355071,-9.035102,2.339664,1.909142,-8.944168,4.495126,-3.719777,3.651118,2.999526,-8.739883,-1.148961,-3.190445],[4.239268,-7.000919,-5.721731,-5.159007,-3.402132,0.083060,-0.749905,-2.204359,-7.072153,-0.882628,9.037921,5.338850],[9.045899,8.509479,-2.646472,-2.365867,-5.848763,-2.681538,0.406186,-3.591194,-1.323288,3.069181,-7.797914,4.493805],[-3.042834,0.038589,1.601856,-1.755790,-7.827606,-4.683860,6.751251,5.421044,7.850537,-4.081995,-3.198763,-9.852826]], dtype = "float32")#candidate|262|(16, 12)|const|float32
bop_263 = relay.logical_and(var_248.astype('bool'), relay.reshape(const_262.astype('bool'), relay.shape_of(var_248))) # shape=(16, 12)
output = relay.Tuple([bop_256,bop_259,bop_263,])
output2 = relay.Tuple([bop_256,bop_259,bop_263,])
func_266 = relay.Function([var_248,], output)
mod['func_266'] = func_266
mod = relay.transform.InferType()(mod)
mutated_mod['func_266'] = func_266
mutated_mod = relay.transform.InferType()(mutated_mod)
var_267 = relay.var("var_267", dtype = "float32", shape = (16, 12))#candidate|267|(16, 12)|var|float32
func_266_call = mutated_mod.get_global_var('func_266')
call_268 = func_266_call(var_267)
output = call_268
func_269 = relay.Function([var_267], output)
mutated_mod['func_269'] = func_269
mutated_mod = relay.transform.InferType()(mutated_mod)
var_271 = relay.var("var_271", dtype = "uint64", shape = (2, 8))#candidate|271|(2, 8)|var|uint64
var_272 = relay.var("var_272", dtype = "uint64", shape = (2, 8))#candidate|272|(2, 8)|var|uint64
bop_273 = relay.bitwise_or(var_271.astype('uint64'), relay.reshape(var_272.astype('uint64'), relay.shape_of(var_271))) # shape=(2, 8)
func_242_call = mod.get_global_var('func_242')
func_246_call = mutated_mod.get_global_var('func_246')
var_277 = relay.var("var_277", dtype = "uint64", shape = (5, 1))#candidate|277|(5, 1)|var|uint64
call_276 = relay.TupleGetItem(func_242_call(relay.reshape(var_277.astype('uint64'), [5,]), relay.reshape(var_277.astype('uint64'), [5,]), ), 0)
call_278 = relay.TupleGetItem(func_246_call(relay.reshape(var_277.astype('uint64'), [5,]), relay.reshape(var_277.astype('uint64'), [5,]), ), 0)
bop_279 = relay.logical_or(var_277.astype('bool'), relay.reshape(call_276.astype('bool'), relay.shape_of(var_277))) # shape=(5, 1)
bop_282 = relay.logical_or(var_277.astype('bool'), relay.reshape(call_278.astype('bool'), relay.shape_of(var_277))) # shape=(5, 1)
output = relay.Tuple([bop_273,bop_279,])
output2 = relay.Tuple([bop_273,bop_282,])
func_283 = relay.Function([var_271,var_272,var_277,], output)
mod['func_283'] = func_283
mod = relay.transform.InferType()(mod)
mutated_mod['func_283'] = func_283
mutated_mod = relay.transform.InferType()(mutated_mod)
func_283_call = mutated_mod.get_global_var('func_283')
var_285 = relay.var("var_285", dtype = "uint64", shape = (2, 8))#candidate|285|(2, 8)|var|uint64
var_286 = relay.var("var_286", dtype = "uint64", shape = (2, 8))#candidate|286|(2, 8)|var|uint64
var_287 = relay.var("var_287", dtype = "uint64", shape = (5, 1))#candidate|287|(5, 1)|var|uint64
call_284 = func_283_call(var_285,var_286,var_287,)
output = call_284
func_288 = relay.Function([var_285,var_286,var_287,], output)
mutated_mod['func_288'] = func_288
mutated_mod = relay.transform.InferType()(mutated_mod)
var_290 = relay.var("var_290", dtype = "float32", shape = (1, 10))#candidate|290|(1, 10)|var|float32
uop_291 = relay.asin(var_290.astype('float32')) # shape=(1, 10)
bop_293 = relay.divide(uop_291.astype('float32'), relay.reshape(var_290.astype('float32'), relay.shape_of(uop_291))) # shape=(1, 10)
bop_296 = relay.logical_or(uop_291.astype('bool'), relay.reshape(bop_293.astype('bool'), relay.shape_of(uop_291))) # shape=(1, 10)
bop_299 = relay.subtract(bop_293.astype('float32'), relay.reshape(var_290.astype('float32'), relay.shape_of(bop_293))) # shape=(1, 10)
uop_302 = relay.log2(uop_291.astype('float64')) # shape=(1, 10)
bop_304 = relay.mod(uop_302.astype('float64'), relay.reshape(bop_293.astype('float64'), relay.shape_of(uop_302))) # shape=(1, 10)
uop_307 = relay.exp(bop_304.astype('float32')) # shape=(1, 10)
bop_309 = relay.right_shift(uop_307.astype('int64'), relay.reshape(var_290.astype('int64'), relay.shape_of(uop_307))) # shape=(1, 10)
const_312 = relay.const([[-4.778500,3.044054,9.354131,-9.532178,5.652241,-1.214133,-5.249546,2.483621,-0.307699,5.187924]], dtype = "float32")#candidate|312|(1, 10)|const|float32
bop_313 = relay.logical_xor(uop_307.astype('uint8'), relay.reshape(const_312.astype('uint8'), relay.shape_of(uop_307))) # shape=(1, 10)
uop_316 = relay.atan(bop_313.astype('float32')) # shape=(1, 10)
bop_318 = relay.right_shift(uop_316.astype('uint32'), relay.reshape(bop_309.astype('uint32'), relay.shape_of(uop_316))) # shape=(1, 10)
output = relay.Tuple([bop_296,bop_299,bop_318,])
output2 = relay.Tuple([bop_296,bop_299,bop_318,])
func_321 = relay.Function([var_290,], output)
mod['func_321'] = func_321
mod = relay.transform.InferType()(mod)
var_322 = relay.var("var_322", dtype = "float32", shape = (1, 10))#candidate|322|(1, 10)|var|float32
output = func_321(var_322)
func_323 = relay.Function([var_322], output)
mutated_mod['func_323'] = func_323
mutated_mod = relay.transform.InferType()(mutated_mod)
var_325 = relay.var("var_325", dtype = "float64", shape = ())#candidate|325|()|var|float64
uop_326 = relay.rsqrt(var_325.astype('float64')) # shape=()
bop_328 = relay.power(var_325.astype('float64'), uop_326.astype('float64')) # shape=()
var_331 = relay.var("var_331", dtype = "float64", shape = ())#candidate|331|()|var|float64
bop_332 = relay.mod(bop_328.astype('float32'), var_331.astype('float32')) # shape=()
bop_335 = relay.greater_equal(var_331.astype('bool'), uop_326.astype('bool')) # shape=()
uop_338 = relay.log2(var_331.astype('float32')) # shape=()
uop_340 = relay.sin(var_325.astype('float64')) # shape=()
bop_342 = relay.multiply(uop_338.astype('uint8'), uop_340.astype('uint8')) # shape=()
output = relay.Tuple([bop_332,bop_335,bop_342,])
output2 = relay.Tuple([bop_332,bop_335,bop_342,])
func_345 = relay.Function([var_325,var_331,], output)
mod['func_345'] = func_345
mod = relay.transform.InferType()(mod)
var_346 = relay.var("var_346", dtype = "float64", shape = ())#candidate|346|()|var|float64
var_347 = relay.var("var_347", dtype = "float64", shape = ())#candidate|347|()|var|float64
output = func_345(var_346,var_347,)
func_348 = relay.Function([var_346,var_347,], output)
mutated_mod['func_348'] = func_348
mutated_mod = relay.transform.InferType()(mutated_mod)
const_350 = relay.const(5.201025, dtype = "float64")#candidate|350|()|const|float64
var_351 = relay.var("var_351", dtype = "float64", shape = (3, 9, 2))#candidate|351|(3, 9, 2)|var|float64
bop_352 = relay.multiply(const_350.astype('float64'), var_351.astype('float64')) # shape=(3, 9, 2)
bop_355 = relay.add(bop_352.astype('int16'), relay.reshape(var_351.astype('int16'), relay.shape_of(bop_352))) # shape=(3, 9, 2)
output = bop_355
output2 = bop_355
func_358 = relay.Function([var_351,], output)
mod['func_358'] = func_358
mod = relay.transform.InferType()(mod)
var_359 = relay.var("var_359", dtype = "float64", shape = (3, 9, 2))#candidate|359|(3, 9, 2)|var|float64
output = func_358(var_359)
func_360 = relay.Function([var_359], output)
mutated_mod['func_360'] = func_360
mutated_mod = relay.transform.InferType()(mutated_mod)
var_362 = relay.var("var_362", dtype = "float64", shape = ())#candidate|362|()|var|float64
uop_363 = relay.cosh(var_362.astype('float64')) # shape=()
bop_365 = relay.floor_mod(var_362.astype('float64'), uop_363.astype('float64')) # shape=()
bop_368 = relay.power(uop_363.astype('float32'), bop_365.astype('float32')) # shape=()
uop_371 = relay.atan(uop_363.astype('float64')) # shape=()
var_373 = relay.var("var_373", dtype = "float64", shape = (2, 15))#candidate|373|(2, 15)|var|float64
bop_374 = relay.logical_xor(uop_371.astype('uint8'), var_373.astype('uint8')) # shape=(2, 15)
output = relay.Tuple([bop_368,bop_374,])
output2 = relay.Tuple([bop_368,bop_374,])
F = relay.Function([var_362,var_373,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_362,var_373,], output2)
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
input_362= np.array(-2.013634, dtype='float64')
module1.set_input('var_362', input_362)
input_373= np.array([[4.490382,2.307871,-9.101246,-1.673384,2.884632,8.819956,8.193376,-8.609675,-8.119770,-0.199568,-5.594802,1.171236,-5.877919,-0.992618,-1.898364],[7.793420,-0.641043,-7.647125,1.020103,-0.266472,0.359918,-5.196367,-6.418351,5.134816,7.266908,-4.343441,-1.165606,5.188666,5.561927,6.315489]], dtype='float64')
module1.set_input('var_373', input_373)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_362, input_373, )
res3 = intrp3.evaluate()(input_362, input_373, )
res4 = intrp4.evaluate()(input_362, input_373, )
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
module5.set_input('var_362', input_362)
module5.set_input('var_373', input_373)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_362, input_373, )
res7 = intrp7.evaluate()(input_362, input_373, )
res8 = intrp8.evaluate()(input_362, input_373, )
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
module9.set_input('var_362', input_362)
module9.set_input('var_373', input_373)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_362, input_373, )
res11 = intrp11.evaluate()(input_362, input_373, )
res12 = intrp12.evaluate()(input_362, input_373, )
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
module13.set_input('var_362', input_362)
module13.set_input('var_373', input_373)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_362, input_373, )
res15 = intrp15.evaluate()(input_362, input_373, )
res16 = intrp16.evaluate()(input_362, input_373, )
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
module17.set_input('var_362', input_362)
module17.set_input('var_373', input_373)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_362, input_373, )
res19 = intrp19.evaluate()(input_362, input_373, )
res20 = intrp20.evaluate()(input_362, input_373, )
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
module21.set_input('var_362', input_362)
module21.set_input('var_373', input_373)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_362, input_373, )
res23 = intrp23.evaluate()(input_362, input_373, )
res24 = intrp24.evaluate()(input_362, input_373, )
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

'''1, 254],
4,   7]], dtype=uint8)

'''