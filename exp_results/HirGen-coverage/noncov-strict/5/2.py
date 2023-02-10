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
var_0 = relay.var("var_0", dtype = "float64", shape = (2, 4))#candidate|0|(2, 4)|var|float64
uop_1 = relay.exp(var_0.astype('float64')) # shape=(2, 4)
uop_3 = relay.sinh(uop_1.astype('float32')) # shape=(2, 4)
bop_5 = relay.floor_mod(uop_1.astype('float64'), relay.reshape(var_0.astype('float64'), relay.shape_of(uop_1))) # shape=(2, 4)
output = relay.Tuple([uop_3,bop_5,])
output2 = relay.Tuple([uop_3,bop_5,])
func_8 = relay.Function([var_0,], output)
mod['func_8'] = func_8
mod = relay.transform.InferType()(mod)
var_9 = relay.var("var_9", dtype = "float64", shape = (2, 4))#candidate|9|(2, 4)|var|float64
output = func_8(var_9)
func_10 = relay.Function([var_9], output)
mutated_mod['func_10'] = func_10
mutated_mod = relay.transform.InferType()(mutated_mod)
var_12 = relay.var("var_12", dtype = "float32", shape = (9,))#candidate|12|(9,)|var|float32
const_13 = relay.const([6.926758,-1.359243,0.723200,2.464702,-1.726918,3.008633,-0.613327,-5.472957,6.254233], dtype = "float32")#candidate|13|(9,)|const|float32
bop_14 = relay.minimum(var_12.astype('float32'), relay.reshape(const_13.astype('float32'), relay.shape_of(var_12))) # shape=(9,)
uop_17 = relay.tan(const_13.astype('float64')) # shape=(9,)
bop_19 = relay.right_shift(uop_17.astype('int64'), relay.reshape(const_13.astype('int64'), relay.shape_of(uop_17))) # shape=(9,)
bop_22 = relay.maximum(bop_19.astype('uint64'), relay.reshape(bop_14.astype('uint64'), relay.shape_of(bop_19))) # shape=(9,)
uop_25 = relay.sin(bop_22.astype('float64')) # shape=(9,)
bop_27 = relay.bitwise_and(bop_19.astype('uint16'), relay.reshape(bop_14.astype('uint16'), relay.shape_of(bop_19))) # shape=(9,)
uop_30 = relay.atanh(uop_25.astype('float64')) # shape=(9,)
uop_32 = relay.erf(var_12.astype('float64')) # shape=(9,)
const_34 = relay.const([-4.173364,2.773963,8.792648,9.224946,-5.748593,-7.033748,6.407429,4.015965,8.792989], dtype = "float64")#candidate|34|(9,)|const|float64
bop_35 = relay.greater_equal(uop_25.astype('bool'), relay.reshape(const_34.astype('bool'), relay.shape_of(uop_25))) # shape=(9,)
uop_38 = relay.log(uop_30.astype('float32')) # shape=(9,)
uop_40 = relay.acosh(uop_38.astype('float64')) # shape=(9,)
var_42 = relay.var("var_42", dtype = "float64", shape = (9,))#candidate|42|(9,)|var|float64
bop_43 = relay.bitwise_or(uop_40.astype('int8'), relay.reshape(var_42.astype('int8'), relay.shape_of(uop_40))) # shape=(9,)
uop_46 = relay.asinh(bop_27.astype('float32')) # shape=(9,)
uop_48 = relay.sigmoid(bop_43.astype('float32')) # shape=(9,)
uop_50 = relay.sigmoid(uop_48.astype('float64')) # shape=(9,)
uop_52 = relay.log10(uop_50.astype('float64')) # shape=(9,)
uop_54 = relay.log(uop_50.astype('float64')) # shape=(9,)
output = relay.Tuple([uop_32,bop_35,uop_46,uop_52,uop_54,])
output2 = relay.Tuple([uop_32,bop_35,uop_46,uop_52,uop_54,])
func_56 = relay.Function([var_12,var_42,], output)
mod['func_56'] = func_56
mod = relay.transform.InferType()(mod)
var_57 = relay.var("var_57", dtype = "float32", shape = (9,))#candidate|57|(9,)|var|float32
var_58 = relay.var("var_58", dtype = "float64", shape = (9,))#candidate|58|(9,)|var|float64
output = func_56(var_57,var_58,)
func_59 = relay.Function([var_57,var_58,], output)
mutated_mod['func_59'] = func_59
mutated_mod = relay.transform.InferType()(mutated_mod)
var_61 = relay.var("var_61", dtype = "float64", shape = ())#candidate|61|()|var|float64
uop_62 = relay.cosh(var_61.astype('float64')) # shape=()
var_64 = relay.var("var_64", dtype = "float64", shape = (5, 12, 5))#candidate|64|(5, 12, 5)|var|float64
bop_65 = relay.maximum(uop_62.astype('float64'), var_64.astype('float64')) # shape=(5, 12, 5)
uop_68 = relay.log(uop_62.astype('float32')) # shape=()
bop_70 = relay.bitwise_or(uop_68.astype('uint32'), var_61.astype('uint32')) # shape=()
uop_73 = relay.log2(uop_68.astype('float64')) # shape=()
var_75 = relay.var("var_75", dtype = "float64", shape = (5, 4, 6))#candidate|75|(5, 4, 6)|var|float64
bop_76 = relay.greater(uop_73.astype('bool'), var_75.astype('bool')) # shape=(5, 4, 6)
uop_79 = relay.log2(bop_70.astype('float32')) # shape=()
uop_81 = relay.atanh(bop_76.astype('float32')) # shape=(5, 4, 6)
bop_83 = relay.right_shift(bop_65.astype('uint32'), relay.reshape(var_64.astype('uint32'), relay.shape_of(bop_65))) # shape=(5, 12, 5)
const_86 = relay.const([[[8.720836,-8.619379,-9.431191,0.169738,4.428920,-3.607662],[-9.126936,4.421925,3.983140,0.408288,5.218653,-1.951082],[7.689820,-1.058269,-4.765788,-9.944416,9.188207,5.765954],[1.969037,-9.163838,-8.599622,1.203851,0.417935,6.315228]],[[6.171676,-6.796629,-2.133544,-8.650002,-1.771585,0.299710],[-2.648889,-3.966451,-5.811379,1.669934,-5.176927,-3.860752],[-5.604600,9.933701,-3.229383,9.517689,-3.149560,-5.790481],[-8.024168,-0.958862,-9.667748,-3.968033,-2.152329,-0.816072]],[[-7.797816,-1.381291,6.389842,0.497609,-4.196001,-5.630263],[6.124297,-0.778697,2.026243,-4.965529,8.134899,-9.983386],[-6.028126,-4.795428,9.529169,0.463090,9.377987,-0.049791],[9.339179,-1.202384,0.504412,-5.879938,-4.076180,-5.261116]],[[-2.003410,-7.386981,4.310039,4.027389,4.807358,-8.621740],[-2.594422,-0.557761,9.169828,1.114554,-4.873095,-7.252412],[-6.414503,2.771921,-6.260410,-2.518729,-3.288913,-1.819219],[9.816130,-9.514962,-3.542352,0.239938,-3.852213,-3.156995]],[[-1.048713,-0.203212,6.381033,-1.422710,-5.237272,5.794144],[4.843981,1.266737,-9.722164,-5.619602,0.721662,-7.251852],[7.164636,-3.904469,5.354569,-8.499574,-4.915173,-5.436153],[2.383159,2.799005,-3.503481,-6.151766,-2.626776,-3.205823]]], dtype = "float32")#candidate|86|(5, 4, 6)|const|float32
bop_87 = relay.floor_divide(uop_81.astype('float32'), relay.reshape(const_86.astype('float32'), relay.shape_of(uop_81))) # shape=(5, 4, 6)
func_8_call = mod.get_global_var('func_8')
func_10_call = mutated_mod.get_global_var('func_10')
var_91 = relay.var("var_91", dtype = "float64", shape = (2, 4))#candidate|91|(2, 4)|var|float64
call_90 = relay.TupleGetItem(func_8_call(relay.reshape(var_91.astype('float64'), [2, 4])), 0)
call_92 = relay.TupleGetItem(func_10_call(relay.reshape(var_91.astype('float64'), [2, 4])), 0)
const_93 = relay.const([[[1.747966,-2.624134,2.108024,-6.504263,-4.880093],[-8.670367,-8.587468,-6.276956,5.613429,-2.923256],[4.981527,-8.411988,-2.694220,-1.327615,6.910075]],[[5.328760,-0.917544,0.933921,6.888218,-4.898505],[-9.862439,-7.257749,2.089754,-4.711334,-1.779480],[3.324615,5.384572,9.822319,1.765708,-7.367726]],[[6.820604,8.707020,4.120663,-4.446382,-6.819410],[0.848372,4.210628,1.920091,-5.689783,9.876565],[6.503628,-1.497059,7.382836,-9.328112,-4.471758]],[[7.428254,-2.694980,7.511384,-5.806460,-1.836869],[-4.521589,9.366756,7.233494,-4.368308,-7.031728],[3.861074,-6.629360,1.509559,3.271972,-5.864514]]], dtype = "float64")#candidate|93|(4, 3, 5)|const|float64
bop_94 = relay.bitwise_and(var_61.astype('uint8'), const_93.astype('uint8')) # shape=(4, 3, 5)
uop_97 = relay.sqrt(uop_81.astype('float32')) # shape=(5, 4, 6)
uop_99 = relay.asin(uop_97.astype('float64')) # shape=(5, 4, 6)
uop_101 = relay.exp(uop_97.astype('float64')) # shape=(5, 4, 6)
uop_103 = relay.log2(uop_97.astype('float64')) # shape=(5, 4, 6)
uop_105 = relay.atan(uop_101.astype('float32')) # shape=(5, 4, 6)
var_107 = relay.var("var_107", dtype = "float32", shape = (5, 4, 6))#candidate|107|(5, 4, 6)|var|float32
bop_108 = relay.not_equal(uop_97.astype('bool'), relay.reshape(var_107.astype('bool'), relay.shape_of(uop_97))) # shape=(5, 4, 6)
output = relay.Tuple([uop_79,bop_83,bop_87,call_90,var_91,bop_94,uop_99,uop_103,uop_105,bop_108,])
output2 = relay.Tuple([uop_79,bop_83,bop_87,call_92,var_91,bop_94,uop_99,uop_103,uop_105,bop_108,])
func_111 = relay.Function([var_61,var_64,var_75,var_91,var_107,], output)
mod['func_111'] = func_111
mod = relay.transform.InferType()(mod)
var_112 = relay.var("var_112", dtype = "float64", shape = ())#candidate|112|()|var|float64
var_113 = relay.var("var_113", dtype = "float64", shape = (5, 12, 5))#candidate|113|(5, 12, 5)|var|float64
var_114 = relay.var("var_114", dtype = "float64", shape = (5, 4, 6))#candidate|114|(5, 4, 6)|var|float64
var_115 = relay.var("var_115", dtype = "float64", shape = (2, 4))#candidate|115|(2, 4)|var|float64
var_116 = relay.var("var_116", dtype = "float32", shape = (5, 4, 6))#candidate|116|(5, 4, 6)|var|float32
output = func_111(var_112,var_113,var_114,var_115,var_116,)
func_117 = relay.Function([var_112,var_113,var_114,var_115,var_116,], output)
mutated_mod['func_117'] = func_117
mutated_mod = relay.transform.InferType()(mutated_mod)
var_119 = relay.var("var_119", dtype = "uint8", shape = (11, 5, 3))#candidate|119|(11, 5, 3)|var|uint8
var_120 = relay.var("var_120", dtype = "uint8", shape = (11, 5, 3))#candidate|120|(11, 5, 3)|var|uint8
bop_121 = relay.bitwise_or(var_119.astype('uint8'), relay.reshape(var_120.astype('uint8'), relay.shape_of(var_119))) # shape=(11, 5, 3)
uop_124 = relay.tan(var_119.astype('float32')) # shape=(11, 5, 3)
var_126 = relay.var("var_126", dtype = "float32", shape = (11, 5, 3))#candidate|126|(11, 5, 3)|var|float32
bop_127 = relay.logical_and(uop_124.astype('bool'), relay.reshape(var_126.astype('bool'), relay.shape_of(uop_124))) # shape=(11, 5, 3)
output = relay.Tuple([bop_121,bop_127,])
output2 = relay.Tuple([bop_121,bop_127,])
func_130 = relay.Function([var_119,var_120,var_126,], output)
mod['func_130'] = func_130
mod = relay.transform.InferType()(mod)
var_131 = relay.var("var_131", dtype = "uint8", shape = (11, 5, 3))#candidate|131|(11, 5, 3)|var|uint8
var_132 = relay.var("var_132", dtype = "uint8", shape = (11, 5, 3))#candidate|132|(11, 5, 3)|var|uint8
var_133 = relay.var("var_133", dtype = "float32", shape = (11, 5, 3))#candidate|133|(11, 5, 3)|var|float32
output = func_130(var_131,var_132,var_133,)
func_134 = relay.Function([var_131,var_132,var_133,], output)
mutated_mod['func_134'] = func_134
mutated_mod = relay.transform.InferType()(mutated_mod)
var_136 = relay.var("var_136", dtype = "int32", shape = (7,))#candidate|136|(7,)|var|int32
const_137 = relay.const([-2,10,-7,10,6,4,-4], dtype = "int32")#candidate|137|(7,)|const|int32
bop_138 = relay.equal(var_136.astype('bool'), relay.reshape(const_137.astype('bool'), relay.shape_of(var_136))) # shape=(7,)
bop_141 = relay.logical_or(const_137.astype('bool'), relay.reshape(bop_138.astype('bool'), relay.shape_of(const_137))) # shape=(7,)
bop_144 = relay.multiply(var_136.astype('uint32'), relay.reshape(const_137.astype('uint32'), relay.shape_of(var_136))) # shape=(7,)
uop_147 = relay.acos(var_136.astype('float32')) # shape=(7,)
uop_149 = relay.sinh(uop_147.astype('float32')) # shape=(7,)
uop_151 = relay.acos(uop_147.astype('float32')) # shape=(7,)
const_153 = relay.const([-6.333550,1.439576,4.760058,7.662684,8.938376,-5.915930,4.459028], dtype = "float32")#candidate|153|(7,)|const|float32
bop_154 = relay.power(uop_147.astype('float64'), relay.reshape(const_153.astype('float64'), relay.shape_of(uop_147))) # shape=(7,)
bop_157 = relay.greater_equal(uop_147.astype('bool'), relay.reshape(const_153.astype('bool'), relay.shape_of(uop_147))) # shape=(7,)
func_56_call = mod.get_global_var('func_56')
func_59_call = mutated_mod.get_global_var('func_59')
const_161 = relay.const([7.583710,-0.249119,-7.302390,-4.334535,0.447445,-6.524341,9.314006,-6.821256,7.473726], dtype = "float32")#candidate|161|(9,)|const|float32
call_160 = relay.TupleGetItem(func_56_call(relay.reshape(const_161.astype('float32'), [9,]), relay.reshape(const_161.astype('float64'), [9,]), ), 1)
call_162 = relay.TupleGetItem(func_59_call(relay.reshape(const_161.astype('float32'), [9,]), relay.reshape(const_161.astype('float64'), [9,]), ), 1)
bop_163 = relay.add(bop_141.astype('int8'), relay.reshape(var_136.astype('int8'), relay.shape_of(bop_141))) # shape=(7,)
bop_166 = relay.logical_and(uop_149.astype('bool'), relay.reshape(bop_144.astype('bool'), relay.shape_of(uop_149))) # shape=(7,)
uop_169 = relay.cos(uop_147.astype('float32')) # shape=(7,)
bop_171 = relay.logical_and(uop_151.astype('bool'), relay.reshape(bop_166.astype('bool'), relay.shape_of(uop_151))) # shape=(7,)
bop_174 = relay.less_equal(uop_169.astype('bool'), relay.reshape(const_153.astype('bool'), relay.shape_of(uop_169))) # shape=(7,)
uop_177 = relay.asinh(uop_149.astype('float32')) # shape=(7,)
uop_179 = relay.sqrt(uop_177.astype('float32')) # shape=(7,)
bop_181 = relay.right_shift(uop_179.astype('uint32'), relay.reshape(var_136.astype('uint32'), relay.shape_of(uop_179))) # shape=(7,)
uop_184 = relay.cos(call_160.astype('float32')) # shape=(9,)
uop_186 = relay.cos(call_162.astype('float32')) # shape=(9,)
var_187 = relay.var("var_187", dtype = "float32", shape = (7,))#candidate|187|(7,)|var|float32
bop_188 = relay.multiply(uop_179.astype('uint64'), relay.reshape(var_187.astype('uint64'), relay.shape_of(uop_179))) # shape=(7,)
bop_191 = relay.floor_mod(uop_177.astype('float32'), relay.reshape(var_187.astype('float32'), relay.shape_of(uop_177))) # shape=(7,)
output = relay.Tuple([bop_154,bop_157,const_161,bop_163,bop_171,bop_174,bop_181,uop_184,bop_188,bop_191,])
output2 = relay.Tuple([bop_154,bop_157,const_161,bop_163,bop_171,bop_174,bop_181,uop_186,bop_188,bop_191,])
func_194 = relay.Function([var_136,var_187,], output)
mod['func_194'] = func_194
mod = relay.transform.InferType()(mod)
var_195 = relay.var("var_195", dtype = "int32", shape = (7,))#candidate|195|(7,)|var|int32
var_196 = relay.var("var_196", dtype = "float32", shape = (7,))#candidate|196|(7,)|var|float32
output = func_194(var_195,var_196,)
func_197 = relay.Function([var_195,var_196,], output)
mutated_mod['func_197'] = func_197
mutated_mod = relay.transform.InferType()(mutated_mod)
var_199 = relay.var("var_199", dtype = "bool", shape = (15,))#candidate|199|(15,)|var|bool
const_200 = relay.const([False,True,False,True,False,False,False,False,True,False,True,False,True,False,False], dtype = "bool")#candidate|200|(15,)|const|bool
bop_201 = relay.logical_and(var_199.astype('bool'), relay.reshape(const_200.astype('bool'), relay.shape_of(var_199))) # shape=(15,)
uop_204 = relay.cosh(var_199.astype('float32')) # shape=(15,)
var_206 = relay.var("var_206", dtype = "float32", shape = (15,))#candidate|206|(15,)|var|float32
bop_207 = relay.logical_xor(uop_204.astype('int8'), relay.reshape(var_206.astype('int8'), relay.shape_of(uop_204))) # shape=(15,)
uop_210 = relay.sin(uop_204.astype('float32')) # shape=(15,)
uop_212 = relay.asinh(uop_210.astype('float32')) # shape=(15,)
bop_214 = relay.power(uop_210.astype('float32'), relay.reshape(uop_204.astype('float32'), relay.shape_of(uop_210))) # shape=(15,)
bop_217 = relay.not_equal(uop_204.astype('bool'), relay.reshape(bop_207.astype('bool'), relay.shape_of(uop_204))) # shape=(15,)
uop_220 = relay.tan(const_200.astype('float32')) # shape=(15,)
bop_222 = relay.add(uop_212.astype('int8'), relay.reshape(bop_201.astype('int8'), relay.shape_of(uop_212))) # shape=(15,)
var_225 = relay.var("var_225", dtype = "int8", shape = (15,))#candidate|225|(15,)|var|int8
bop_226 = relay.power(bop_222.astype('float32'), relay.reshape(var_225.astype('float32'), relay.shape_of(bop_222))) # shape=(15,)
uop_229 = relay.atanh(bop_222.astype('float64')) # shape=(15,)
bop_231 = relay.minimum(bop_226.astype('int16'), relay.reshape(bop_207.astype('int16'), relay.shape_of(bop_226))) # shape=(15,)
uop_234 = relay.atanh(uop_229.astype('float64')) # shape=(15,)
uop_236 = relay.sin(bop_231.astype('float64')) # shape=(15,)
var_238 = relay.var("var_238", dtype = "float64", shape = (15,))#candidate|238|(15,)|var|float64
bop_239 = relay.divide(uop_229.astype('float32'), relay.reshape(var_238.astype('float32'), relay.shape_of(uop_229))) # shape=(15,)
output = relay.Tuple([bop_214,bop_217,uop_220,uop_234,uop_236,bop_239,])
output2 = relay.Tuple([bop_214,bop_217,uop_220,uop_234,uop_236,bop_239,])
func_242 = relay.Function([var_199,var_206,var_225,var_238,], output)
mod['func_242'] = func_242
mod = relay.transform.InferType()(mod)
mutated_mod['func_242'] = func_242
mutated_mod = relay.transform.InferType()(mutated_mod)
func_242_call = mutated_mod.get_global_var('func_242')
var_244 = relay.var("var_244", dtype = "bool", shape = (15,))#candidate|244|(15,)|var|bool
var_245 = relay.var("var_245", dtype = "float32", shape = (15,))#candidate|245|(15,)|var|float32
var_246 = relay.var("var_246", dtype = "int8", shape = (15,))#candidate|246|(15,)|var|int8
var_247 = relay.var("var_247", dtype = "float64", shape = (15,))#candidate|247|(15,)|var|float64
call_243 = func_242_call(var_244,var_245,var_246,var_247,)
output = call_243
func_248 = relay.Function([var_244,var_245,var_246,var_247,], output)
mutated_mod['func_248'] = func_248
mutated_mod = relay.transform.InferType()(mutated_mod)
const_250 = relay.const(3, dtype = "int8")#candidate|250|()|const|int8
var_251 = relay.var("var_251", dtype = "int8", shape = (9, 3, 16))#candidate|251|(9, 3, 16)|var|int8
bop_252 = relay.less(const_250.astype('bool'), var_251.astype('bool')) # shape=(9, 3, 16)
bop_255 = relay.subtract(var_251.astype('uint64'), relay.reshape(bop_252.astype('uint64'), relay.shape_of(var_251))) # shape=(9, 3, 16)
bop_258 = relay.maximum(var_251.astype('uint8'), const_250.astype('uint8')) # shape=(9, 3, 16)
bop_261 = relay.maximum(var_251.astype('float32'), relay.reshape(bop_255.astype('float32'), relay.shape_of(var_251))) # shape=(9, 3, 16)
uop_264 = relay.sigmoid(const_250.astype('float32')) # shape=()
var_266 = relay.var("var_266", dtype = "float32", shape = (9, 3, 16))#candidate|266|(9, 3, 16)|var|float32
bop_267 = relay.add(bop_261.astype('uint32'), relay.reshape(var_266.astype('uint32'), relay.shape_of(bop_261))) # shape=(9, 3, 16)
uop_270 = relay.acosh(uop_264.astype('float32')) # shape=()
uop_272 = relay.asin(uop_270.astype('float32')) # shape=()
bop_274 = relay.not_equal(bop_267.astype('bool'), uop_264.astype('bool')) # shape=(9, 3, 16)
var_277 = relay.var("var_277", dtype = "float32", shape = (10,))#candidate|277|(10,)|var|float32
bop_278 = relay.floor_divide(uop_272.astype('float32'), var_277.astype('float32')) # shape=(10,)
uop_281 = relay.tan(uop_270.astype('float32')) # shape=()
uop_283 = relay.cos(bop_278.astype('float32')) # shape=(10,)
var_285 = relay.var("var_285", dtype = "float32", shape = (10,))#candidate|285|(10,)|var|float32
bop_286 = relay.logical_xor(bop_278.astype('int8'), relay.reshape(var_285.astype('int8'), relay.shape_of(bop_278))) # shape=(10,)
bop_289 = relay.mod(uop_283.astype('float64'), uop_270.astype('float64')) # shape=(10,)
bop_292 = relay.not_equal(uop_283.astype('bool'), uop_272.astype('bool')) # shape=(10,)
output = relay.Tuple([bop_258,bop_274,uop_281,bop_286,bop_289,bop_292,])
output2 = relay.Tuple([bop_258,bop_274,uop_281,bop_286,bop_289,bop_292,])
func_295 = relay.Function([var_251,var_266,var_277,var_285,], output)
mod['func_295'] = func_295
mod = relay.transform.InferType()(mod)
var_296 = relay.var("var_296", dtype = "int8", shape = (9, 3, 16))#candidate|296|(9, 3, 16)|var|int8
var_297 = relay.var("var_297", dtype = "float32", shape = (9, 3, 16))#candidate|297|(9, 3, 16)|var|float32
var_298 = relay.var("var_298", dtype = "float32", shape = (10,))#candidate|298|(10,)|var|float32
var_299 = relay.var("var_299", dtype = "float32", shape = (10,))#candidate|299|(10,)|var|float32
output = func_295(var_296,var_297,var_298,var_299,)
func_300 = relay.Function([var_296,var_297,var_298,var_299,], output)
mutated_mod['func_300'] = func_300
mutated_mod = relay.transform.InferType()(mutated_mod)
const_302 = relay.const([[[-2,1,6,8,-6,-5,-6,2,-10,1,5,-9,-6,4,8,10],[-1,-6,7,-3,1,-7,5,-8,-2,-9,9,-10,1,-2,5,8]],[[7,1,2,-6,-3,9,4,3,6,-5,-9,7,6,-8,-3,7],[4,3,-5,8,2,4,-5,-2,-8,-4,-1,-5,-4,9,1,7]],[[-5,1,1,-8,7,1,-10,2,1,4,-1,-8,-4,-10,5,2],[2,-2,5,6,-1,-3,-9,10,3,1,-5,-6,-7,5,8,4]],[[3,10,3,9,2,-6,6,-2,-9,2,-1,-2,5,-10,5,10],[9,3,8,5,-3,-5,-8,8,8,-1,8,-9,1,5,-5,-3]],[[7,-6,9,6,-4,-10,8,2,-8,6,8,-9,9,6,-9,8],[-6,-2,8,4,-1,-4,4,4,1,-10,-9,8,2,-2,3,-8]],[[6,4,10,-6,4,-7,-8,-2,2,5,-8,-9,-7,3,5,-10],[3,-5,-4,3,2,8,-2,-6,10,-8,-6,-1,10,-4,-7,-5]],[[10,3,-2,-4,6,-2,-2,-3,-4,3,9,-5,-1,-5,-3,9],[2,-7,-6,-9,-3,-1,3,-7,6,-7,7,-6,-10,5,3,9]],[[5,2,-7,6,-3,-6,10,3,5,-2,1,9,9,5,9,1],[-9,5,-5,-9,7,-7,9,4,-2,-6,-4,-4,-10,-9,2,5]],[[3,1,-3,8,-4,-9,-5,-6,-1,1,9,2,-4,-10,-1,2],[-7,1,3,-9,4,-1,1,7,-10,-9,4,10,-2,-6,5,6]],[[1,9,-7,-1,6,-10,-9,-1,5,-10,1,7,-10,6,-3,7],[-10,-7,5,-9,-2,-2,-8,-4,4,2,-3,1,-1,9,9,-7]],[[-2,-7,-6,10,5,-1,-4,6,8,-3,6,-4,-1,1,9,-1],[4,-8,4,8,-1,-4,5,-2,7,-4,1,10,-1,4,-8,-10]],[[4,-3,-7,10,10,-2,-4,-8,2,-6,9,7,-6,9,7,-6],[-9,-9,-8,2,-3,-4,9,-1,-5,-3,2,4,2,-4,7,2]]], dtype = "int32")#candidate|302|(12, 2, 16)|const|int32
const_303 = relay.const([[[-10,-6,-4,4,1,-7,-3,8,2,4,7,6,-8,-9,7,-2],[5,-9,-4,8,-6,10,5,-3,-2,-10,-2,2,-1,5,-4,-6]],[[7,-9,-8,-8,1,6,-8,3,-5,8,7,7,10,5,1,3],[7,-2,-9,9,-10,8,-3,-5,-3,-10,-10,10,9,4,-9,9]],[[-10,-10,3,1,-4,9,-9,10,9,2,-6,-1,-10,-6,-3,2],[8,-2,-1,2,9,5,-10,-2,6,8,8,7,-4,4,8,1]],[[-8,-4,-6,-4,-2,5,-7,-10,-2,-3,10,8,-10,3,10,3],[7,-9,-6,7,4,-9,-10,-2,-7,-5,-6,1,7,4,-8,-9]],[[4,7,1,-7,6,7,-3,8,-5,-9,8,-7,-3,-6,9,7],[3,3,-6,10,6,-9,8,1,2,6,-9,7,-7,-5,6,10]],[[-4,3,5,5,5,-2,-1,8,-2,-3,1,-5,1,-5,-3,-8],[1,-8,-9,10,7,6,-3,-3,-1,8,-6,-10,-4,-8,1,-4]],[[9,1,-7,1,10,-1,-2,-2,1,-6,-2,6,3,-10,9,-1],[8,-6,2,8,-5,-7,-3,3,-8,1,-7,-3,-10,-9,9,-9]],[[1,-1,-5,4,-9,-8,-9,-6,-1,-6,-5,-1,4,-6,-1,1],[10,-2,2,-6,9,1,6,7,-6,6,-2,6,8,-10,-9,-10]],[[2,6,9,4,6,8,-5,-1,-9,-2,-5,-8,4,-3,-6,2],[6,3,7,-7,-5,9,1,-10,-5,1,3,-5,7,10,3,-5]],[[-7,8,6,-1,10,10,7,2,-2,-4,-4,4,8,-7,5,2],[10,7,5,6,7,-7,-6,6,5,7,-10,-5,6,-1,-3,10]],[[-3,1,7,3,3,-6,-2,9,4,-9,8,5,5,2,1,3],[-10,-3,-2,-9,-8,-6,-9,10,-10,10,-4,-1,-2,-8,4,6]],[[8,1,8,6,6,8,8,3,-10,5,9,-4,-8,2,8,6],[-1,-7,-10,7,-7,-3,5,-3,8,-1,-4,8,9,-4,9,-2]]], dtype = "int32")#candidate|303|(12, 2, 16)|const|int32
bop_304 = relay.not_equal(const_302.astype('bool'), relay.reshape(const_303.astype('bool'), relay.shape_of(const_302))) # shape=(12, 2, 16)
bop_307 = relay.divide(const_302.astype('float64'), relay.reshape(const_303.astype('float64'), relay.shape_of(const_302))) # shape=(12, 2, 16)
uop_310 = relay.cos(const_302.astype('float32')) # shape=(12, 2, 16)
uop_312 = relay.asinh(uop_310.astype('float32')) # shape=(12, 2, 16)
var_314 = relay.var("var_314", dtype = "float32", shape = (12, 2, 16))#candidate|314|(12, 2, 16)|var|float32
bop_315 = relay.greater(uop_312.astype('bool'), relay.reshape(var_314.astype('bool'), relay.shape_of(uop_312))) # shape=(12, 2, 16)
output = relay.Tuple([bop_304,bop_307,bop_315,])
output2 = relay.Tuple([bop_304,bop_307,bop_315,])
func_318 = relay.Function([var_314,], output)
mod['func_318'] = func_318
mod = relay.transform.InferType()(mod)
var_319 = relay.var("var_319", dtype = "float32", shape = (12, 2, 16))#candidate|319|(12, 2, 16)|var|float32
output = func_318(var_319)
func_320 = relay.Function([var_319], output)
mutated_mod['func_320'] = func_320
mutated_mod = relay.transform.InferType()(mutated_mod)
const_322 = relay.const([[[-8.173931,2.341250,-7.219350,1.189037,-8.442067,1.417393,6.525734],[4.228006,1.784520,6.315488,-5.992057,-1.639566,-3.274031,2.566953],[0.443533,-2.460663,-2.487768,2.661733,6.631080,0.397894,4.302262],[-5.383606,8.630724,7.916511,5.845394,1.224440,-8.235988,-6.671903]],[[-7.068648,0.418423,-4.465354,2.804500,1.759859,4.654327,0.428485],[-2.735181,4.946298,8.894124,-9.267409,0.370928,6.040747,-1.781966],[-5.208468,7.768112,-1.173371,-4.077246,-2.336521,8.805926,-4.050911],[-0.698500,-0.943113,2.402964,9.036699,-4.205408,8.517201,-9.458315]],[[-9.866490,-7.728282,0.772652,9.461249,-8.810594,-9.593270,4.292717],[8.696779,-1.872370,-1.639438,-4.324317,-3.026983,-4.457401,-9.383571],[-9.229066,4.892373,5.148073,-1.604167,5.138678,-6.329765,6.436740],[-8.736668,0.272590,3.767268,1.000400,-4.539310,5.744237,2.450893]]], dtype = "float64")#candidate|322|(3, 4, 7)|const|float64
uop_323 = relay.erf(const_322.astype('float64')) # shape=(3, 4, 7)
var_325 = relay.var("var_325", dtype = "float64", shape = (3, 4, 7))#candidate|325|(3, 4, 7)|var|float64
bop_326 = relay.floor_mod(const_322.astype('float32'), relay.reshape(var_325.astype('float32'), relay.shape_of(const_322))) # shape=(3, 4, 7)
bop_329 = relay.equal(uop_323.astype('bool'), relay.reshape(bop_326.astype('bool'), relay.shape_of(uop_323))) # shape=(3, 4, 7)
bop_332 = relay.greater_equal(uop_323.astype('bool'), relay.reshape(var_325.astype('bool'), relay.shape_of(uop_323))) # shape=(3, 4, 7)
uop_335 = relay.acos(uop_323.astype('float64')) # shape=(3, 4, 7)
uop_337 = relay.log2(uop_323.astype('float32')) # shape=(3, 4, 7)
output = relay.Tuple([bop_329,bop_332,uop_335,uop_337,])
output2 = relay.Tuple([bop_329,bop_332,uop_335,uop_337,])
F = relay.Function([var_325,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_325,], output2)
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
input_325= np.array([[[5.961342,-0.934314,-2.472108,-6.233953,-3.690597,7.268831,6.311722],[7.474531,-7.354570,-0.273277,-9.776249,6.936793,5.296208,-7.210929],[-4.521920,3.765504,-8.756855,-7.670427,0.018546,5.387240,8.229712],[9.801348,-8.218806,-8.090063,-3.416547,1.793875,-8.289488,8.808870]],[[6.783247,5.338545,-8.310798,6.915202,7.679600,6.903463,-8.715720],[-8.153973,-3.481426,-3.712563,-6.408971,8.231210,-0.980019,4.660574],[2.302067,-8.756384,-0.583679,-1.742611,-3.572062,4.286705,3.763354],[-1.942820,-5.418574,0.903036,-5.438537,1.594944,3.685979,1.359866]],[[-7.225420,-7.104637,5.973328,-1.089394,-9.158709,-3.340548,9.782364],[-9.254077,6.990702,-3.218913,4.380278,-8.985567,-7.222498,-0.530330],[-2.027333,7.001803,-2.323486,-5.747411,-9.602978,-9.724730,-0.801547],[3.504316,-4.684477,-7.000266,-7.250887,6.627679,1.518719,0.946963]]], dtype='float64')
module1.set_input('var_325', input_325)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_325, )
res3 = intrp3.evaluate()(input_325, )
res4 = intrp4.evaluate()(input_325, )
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
module5.set_input('var_325', input_325)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_325, )
res7 = intrp7.evaluate()(input_325, )
res8 = intrp8.evaluate()(input_325, )
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
module9.set_input('var_325', input_325)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_325, )
res11 = intrp11.evaluate()(input_325, )
res12 = intrp12.evaluate()(input_325, )
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
module13.set_input('var_325', input_325)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_325, )
res15 = intrp15.evaluate()(input_325, )
res16 = intrp16.evaluate()(input_325, )
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
module17.set_input('var_325', input_325)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_325, )
res19 = intrp19.evaluate()(input_325, )
res20 = intrp20.evaluate()(input_325, )
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
module21.set_input('var_325', input_325)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_325, )
res23 = intrp23.evaluate()(input_325, )
res24 = intrp24.evaluate()(input_325, )
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