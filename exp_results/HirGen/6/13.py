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
var_1 = relay.var("var_1", dtype = "float32", shape = ())#candidate|1|()|var|float32
bop_2 = relay.multiply(var_0.astype('float32'), var_1.astype('float32')) # shape=()
bop_5 = relay.less(var_1.astype('bool'), bop_2.astype('bool')) # shape=()
bop_8 = relay.subtract(var_0.astype('uint8'), var_1.astype('uint8')) # shape=()
bop_11 = relay.greater_equal(var_1.astype('bool'), bop_8.astype('bool')) # shape=()
bop_14 = relay.bitwise_xor(bop_5.astype('uint64'), bop_11.astype('uint64')) # shape=()
bop_17 = relay.equal(bop_8.astype('bool'), bop_14.astype('bool')) # shape=()
uop_20 = relay.acos(bop_2.astype('float64')) # shape=()
bop_22 = relay.divide(bop_8.astype('float64'), bop_14.astype('float64')) # shape=()
uop_25 = relay.log10(uop_20.astype('float32')) # shape=()
var_27 = relay.var("var_27", dtype = "float64", shape = (16,))#candidate|27|(16,)|var|float64
bop_28 = relay.logical_xor(uop_20.astype('int16'), var_27.astype('int16')) # shape=(16,)
bop_31 = relay.right_shift(uop_25.astype('int16'), var_1.astype('int16')) # shape=()
bop_34 = relay.bitwise_and(bop_28.astype('uint8'), bop_11.astype('uint8')) # shape=(16,)
uop_37 = relay.log(uop_25.astype('float32')) # shape=()
uop_39 = relay.acosh(bop_22.astype('float32')) # shape=()
uop_41 = relay.cos(uop_37.astype('float32')) # shape=()
uop_43 = relay.sin(uop_37.astype('float64')) # shape=()
var_45 = relay.var("var_45", dtype = "float32", shape = (1, 2))#candidate|45|(1, 2)|var|float32
bop_46 = relay.bitwise_xor(uop_41.astype('int64'), var_45.astype('int64')) # shape=(1, 2)
var_49 = relay.var("var_49", dtype = "float32", shape = (3, 5, 7))#candidate|49|(3, 5, 7)|var|float32
bop_50 = relay.floor_divide(uop_25.astype('float64'), var_49.astype('float64')) # shape=(3, 5, 7)
bop_53 = relay.floor_divide(uop_43.astype('float64'), bop_50.astype('float64')) # shape=(3, 5, 7)
var_56 = relay.var("var_56", dtype = "int16", shape = (9, 14, 16))#candidate|56|(9, 14, 16)|var|int16
bop_57 = relay.greater_equal(bop_31.astype('bool'), var_56.astype('bool')) # shape=(9, 14, 16)
uop_60 = relay.sqrt(bop_46.astype('float32')) # shape=(1, 2)
bop_62 = relay.less_equal(uop_37.astype('bool'), bop_34.astype('bool')) # shape=(16,)
uop_65 = relay.sqrt(bop_53.astype('float64')) # shape=(3, 5, 7)
output = relay.Tuple([bop_17,uop_39,bop_57,uop_60,bop_62,uop_65,])
output2 = relay.Tuple([bop_17,uop_39,bop_57,uop_60,bop_62,uop_65,])
func_67 = relay.Function([var_0,var_1,var_27,var_45,var_49,var_56,], output)
mod['func_67'] = func_67
mod = relay.transform.InferType()(mod)
var_68 = relay.var("var_68", dtype = "float32", shape = ())#candidate|68|()|var|float32
var_69 = relay.var("var_69", dtype = "float32", shape = ())#candidate|69|()|var|float32
var_70 = relay.var("var_70", dtype = "float64", shape = (16,))#candidate|70|(16,)|var|float64
var_71 = relay.var("var_71", dtype = "float32", shape = (1, 2))#candidate|71|(1, 2)|var|float32
var_72 = relay.var("var_72", dtype = "float32", shape = (3, 5, 7))#candidate|72|(3, 5, 7)|var|float32
var_73 = relay.var("var_73", dtype = "int16", shape = (9, 14, 16))#candidate|73|(9, 14, 16)|var|int16
output = func_67(var_68,var_69,var_70,var_71,var_72,var_73,)
func_74 = relay.Function([var_68,var_69,var_70,var_71,var_72,var_73,], output)
mutated_mod['func_74'] = func_74
mutated_mod = relay.transform.InferType()(mutated_mod)
var_76 = relay.var("var_76", dtype = "float32", shape = (5, 11))#candidate|76|(5, 11)|var|float32
const_77 = relay.const([[5.309083,-1.371105,5.522774,-5.590317,9.319926,-8.214010,6.855158,-6.217933,1.847932,-7.127106,-0.875545],[1.436519,5.123500,-5.123307,-6.278128,1.839257,-2.064752,0.390774,4.310238,-5.232228,-7.311453,2.968178],[-3.010185,-6.642031,-9.540932,-6.197442,-3.590494,-2.245388,-4.101424,-8.908641,-3.372374,-9.902162,1.810315],[-5.403211,-7.931564,2.917644,-2.061809,-4.850007,-8.888439,-8.090580,9.267166,-8.638451,-7.068275,6.435567],[3.854584,-1.166059,-6.501577,-8.189142,-9.827156,-2.240659,9.380931,-3.698729,-0.087808,3.073885,9.645429]], dtype = "float32")#candidate|77|(5, 11)|const|float32
bop_78 = relay.less_equal(var_76.astype('bool'), relay.reshape(const_77.astype('bool'), relay.shape_of(var_76))) # shape=(5, 11)
var_81 = relay.var("var_81", dtype = "float32", shape = (5, 11))#candidate|81|(5, 11)|var|float32
bop_82 = relay.bitwise_or(const_77.astype('uint16'), relay.reshape(var_81.astype('uint16'), relay.shape_of(const_77))) # shape=(5, 11)
uop_85 = relay.sqrt(var_76.astype('float32')) # shape=(5, 11)
uop_87 = relay.exp(uop_85.astype('float64')) # shape=(5, 11)
uop_89 = relay.asin(uop_87.astype('float32')) # shape=(5, 11)
var_91 = relay.var("var_91", dtype = "float32", shape = (5, 11))#candidate|91|(5, 11)|var|float32
bop_92 = relay.logical_and(uop_89.astype('bool'), relay.reshape(var_91.astype('bool'), relay.shape_of(uop_89))) # shape=(5, 11)
const_95 = relay.const([[True,False,False,False,False,True,True,True,False,True,True],[False,True,True,False,False,False,True,False,True,False,True],[False,True,False,True,False,False,False,False,False,True,False],[True,True,True,False,True,False,False,False,False,False,True],[True,False,True,True,False,True,True,False,True,True,True]], dtype = "bool")#candidate|95|(5, 11)|const|bool
bop_96 = relay.subtract(bop_92.astype('uint16'), relay.reshape(const_95.astype('uint16'), relay.shape_of(bop_92))) # shape=(5, 11)
var_99 = relay.var("var_99", dtype = "float32", shape = (5, 11))#candidate|99|(5, 11)|var|float32
bop_100 = relay.minimum(uop_85.astype('uint64'), relay.reshape(var_99.astype('uint64'), relay.shape_of(uop_85))) # shape=(5, 11)
bop_103 = relay.logical_and(uop_89.astype('bool'), relay.reshape(bop_96.astype('bool'), relay.shape_of(uop_89))) # shape=(5, 11)
const_106 = relay.const([[False,False,True,False,False,True,True,False,False,False,True],[True,True,False,False,True,True,False,True,False,True,False],[True,False,False,False,True,False,True,True,False,True,True],[False,True,True,True,False,True,True,True,False,False,False],[False,False,True,True,True,False,True,False,False,False,True]], dtype = "bool")#candidate|106|(5, 11)|const|bool
bop_107 = relay.less_equal(bop_103.astype('bool'), relay.reshape(const_106.astype('bool'), relay.shape_of(bop_103))) # shape=(5, 11)
bop_110 = relay.bitwise_or(const_106.astype('int16'), relay.reshape(bop_78.astype('int16'), relay.shape_of(const_106))) # shape=(5, 11)
var_113 = relay.var("var_113", dtype = "bool", shape = (5, 11))#candidate|113|(5, 11)|var|bool
bop_114 = relay.maximum(bop_107.astype('uint16'), relay.reshape(var_113.astype('uint16'), relay.shape_of(bop_107))) # shape=(5, 11)
uop_117 = relay.acos(bop_96.astype('float32')) # shape=(5, 11)
var_119 = relay.var("var_119", dtype = "float32", shape = (5, 11))#candidate|119|(5, 11)|var|float32
bop_120 = relay.subtract(uop_117.astype('float64'), relay.reshape(var_119.astype('float64'), relay.shape_of(uop_117))) # shape=(5, 11)
var_123 = relay.var("var_123", dtype = "float32", shape = (5, 11))#candidate|123|(5, 11)|var|float32
bop_124 = relay.multiply(uop_117.astype('uint64'), relay.reshape(var_123.astype('uint64'), relay.shape_of(uop_117))) # shape=(5, 11)
uop_127 = relay.asinh(bop_124.astype('float32')) # shape=(5, 11)
func_67_call = mod.get_global_var('func_67')
func_74_call = mutated_mod.get_global_var('func_74')
var_130 = relay.var("var_130", dtype = "float32", shape = ())#candidate|130|()|var|float32
const_131 = relay.const([[-6.223947,-5.684951],[9.086051,-4.708186],[5.341707,2.687098],[9.521109,5.616477],[-0.690684,-5.531507],[-4.101447,-7.615469],[8.604315,-5.227754],[6.732172,9.380189]], dtype = "float64")#candidate|131|(8, 2)|const|float64
var_132 = relay.var("var_132", dtype = "float32", shape = (2,))#candidate|132|(2,)|var|float32
const_133 = relay.const([-0.735787,-0.927348,-7.621092,1.937768,0.350052,1.359103,-4.186237,-2.159521,2.976941,-6.273515,-6.551225,-1.263411,3.768937,-9.649609,-3.294076,6.005566,-1.398468,-3.872287,-7.623462,3.380178,2.420671,5.199613,0.701663,-3.168625,-3.587384,2.421839,-1.897921,-2.618232,-8.376060,-3.079852,-7.098640,2.203999,2.068329,3.843047,4.465304,8.805853,1.305620,1.596494,-0.641991,-8.671546,-3.087619,9.233265,5.685854,-9.286271,-7.829114,-7.508431,-0.295166,2.015818,7.820667,7.396822,-2.707571,-1.110658,-2.624586,-0.317536,7.708666,9.380750,-3.242002,-1.723241,-1.072179,0.285582,7.432843,-3.511672,9.264163,3.947148,-4.137135,7.664778,-2.351351,3.894963,5.683369,4.055817,-0.229770,-3.424247,-8.184442,-0.546723,4.550973,5.208219,-7.923990,0.039100,-0.278030,-3.026733,7.946102,5.949954,5.061227,-0.233480,-8.700381,0.564020,7.145222,9.932643,-4.803810,-6.540884,4.771683,-2.089593,1.729613,4.768829,4.685132,5.799430,6.671878,-1.985065,4.302768,-3.420897,-7.387668,-6.578727,1.992807,5.428389,9.148718], dtype = "float32")#candidate|133|(105,)|const|float32
var_134 = relay.var("var_134", dtype = "int16", shape = (1, 2016))#candidate|134|(1, 2016)|var|int16
call_129 = relay.TupleGetItem(func_67_call(relay.reshape(var_130.astype('float32'), []), relay.reshape(var_130.astype('float32'), []), relay.reshape(const_131.astype('float64'), [16,]), relay.reshape(var_132.astype('float32'), [1, 2]), relay.reshape(const_133.astype('float32'), [3, 5, 7]), relay.reshape(var_134.astype('int16'), [9, 14, 16]), ), 5)
call_135 = relay.TupleGetItem(func_74_call(relay.reshape(var_130.astype('float32'), []), relay.reshape(var_130.astype('float32'), []), relay.reshape(const_131.astype('float64'), [16,]), relay.reshape(var_132.astype('float32'), [1, 2]), relay.reshape(const_133.astype('float32'), [3, 5, 7]), relay.reshape(var_134.astype('int16'), [9, 14, 16]), ), 5)
bop_136 = relay.right_shift(uop_127.astype('int32'), relay.reshape(bop_110.astype('int32'), relay.shape_of(uop_127))) # shape=(5, 11)
bop_139 = relay.floor_divide(bop_136.astype('float32'), relay.reshape(bop_103.astype('float32'), relay.shape_of(bop_136))) # shape=(5, 11)
var_142 = relay.var("var_142", dtype = "float32", shape = (5, 11))#candidate|142|(5, 11)|var|float32
bop_143 = relay.subtract(bop_139.astype('uint64'), relay.reshape(var_142.astype('uint64'), relay.shape_of(bop_139))) # shape=(5, 11)
uop_146 = relay.log10(bop_96.astype('float64')) # shape=(5, 11)
output = relay.Tuple([bop_82,bop_100,bop_114,bop_120,call_129,var_130,const_131,var_132,const_133,var_134,bop_143,uop_146,])
output2 = relay.Tuple([bop_82,bop_100,bop_114,bop_120,call_135,var_130,const_131,var_132,const_133,var_134,bop_143,uop_146,])
func_148 = relay.Function([var_76,var_81,var_91,var_99,var_113,var_119,var_123,var_130,var_132,var_134,var_142,], output)
mod['func_148'] = func_148
mod = relay.transform.InferType()(mod)
mutated_mod['func_148'] = func_148
mutated_mod = relay.transform.InferType()(mutated_mod)
func_148_call = mutated_mod.get_global_var('func_148')
var_150 = relay.var("var_150", dtype = "float32", shape = (5, 11))#candidate|150|(5, 11)|var|float32
var_151 = relay.var("var_151", dtype = "float32", shape = (5, 11))#candidate|151|(5, 11)|var|float32
var_152 = relay.var("var_152", dtype = "float32", shape = (5, 11))#candidate|152|(5, 11)|var|float32
var_153 = relay.var("var_153", dtype = "float32", shape = (5, 11))#candidate|153|(5, 11)|var|float32
var_154 = relay.var("var_154", dtype = "bool", shape = (5, 11))#candidate|154|(5, 11)|var|bool
var_155 = relay.var("var_155", dtype = "float32", shape = (5, 11))#candidate|155|(5, 11)|var|float32
var_156 = relay.var("var_156", dtype = "float32", shape = (5, 11))#candidate|156|(5, 11)|var|float32
var_157 = relay.var("var_157", dtype = "float32", shape = ())#candidate|157|()|var|float32
var_158 = relay.var("var_158", dtype = "float32", shape = (2,))#candidate|158|(2,)|var|float32
var_159 = relay.var("var_159", dtype = "int16", shape = (1, 2016))#candidate|159|(1, 2016)|var|int16
var_160 = relay.var("var_160", dtype = "float32", shape = (5, 11))#candidate|160|(5, 11)|var|float32
call_149 = func_148_call(var_150,var_151,var_152,var_153,var_154,var_155,var_156,var_157,var_158,var_159,var_160,)
output = call_149
func_161 = relay.Function([var_150,var_151,var_152,var_153,var_154,var_155,var_156,var_157,var_158,var_159,var_160,], output)
mutated_mod['func_161'] = func_161
mutated_mod = relay.transform.InferType()(mutated_mod)
var_163 = relay.var("var_163", dtype = "float64", shape = (14,))#candidate|163|(14,)|var|float64
uop_164 = relay.sqrt(var_163.astype('float64')) # shape=(14,)
var_166 = relay.var("var_166", dtype = "float64", shape = (14,))#candidate|166|(14,)|var|float64
bop_167 = relay.less_equal(var_163.astype('bool'), relay.reshape(var_166.astype('bool'), relay.shape_of(var_163))) # shape=(14,)
uop_170 = relay.asinh(uop_164.astype('float32')) # shape=(14,)
uop_172 = relay.exp(uop_164.astype('float32')) # shape=(14,)
bop_174 = relay.floor_divide(uop_170.astype('float32'), relay.reshape(uop_164.astype('float32'), relay.shape_of(uop_170))) # shape=(14,)
bop_177 = relay.divide(uop_164.astype('float64'), relay.reshape(var_163.astype('float64'), relay.shape_of(uop_164))) # shape=(14,)
bop_180 = relay.maximum(uop_172.astype('uint32'), relay.reshape(bop_174.astype('uint32'), relay.shape_of(uop_172))) # shape=(14,)
uop_183 = relay.cos(uop_164.astype('float64')) # shape=(14,)
func_148_call = mod.get_global_var('func_148')
func_161_call = mutated_mod.get_global_var('func_161')
const_186 = relay.const([-0.092374,-0.623528,5.185604,9.791941,-6.173385,-2.314240,-4.494597,-1.846431,-5.445514,9.234511,-9.659535,-1.466627,0.361197,1.389212,3.992102,1.130033,-8.781774,1.696636,-3.146251,9.476408,7.962923,3.303280,-2.609757,2.641917,-1.234196,-8.548483,6.381811,-6.262788,-5.014446,-4.323194,-4.712220,4.903972,9.555056,5.929085,-4.647385,2.293294,-1.591824,8.448665,-6.104396,-1.080601,8.990532,4.519406,1.239918,0.307890,-8.889699,5.609353,9.548193,0.461928,-4.432326,-7.433059,0.724904,-3.492236,-0.576289,-1.338627,-4.814462], dtype = "float32")#candidate|186|(55,)|const|float32
var_187 = relay.var("var_187", dtype = "float32", shape = ())#candidate|187|()|var|float32
const_188 = relay.const([[-0.324187],[-6.649038]], dtype = "float32")#candidate|188|(2, 1)|const|float32
var_189 = relay.var("var_189", dtype = "int16", shape = (252, 8))#candidate|189|(252, 8)|var|int16
call_185 = relay.TupleGetItem(func_148_call(relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('bool'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(var_187.astype('float32'), []), relay.reshape(const_188.astype('float32'), [2,]), relay.reshape(var_189.astype('int16'), [1, 2016]), relay.reshape(const_186.astype('float32'), [5, 11]), ), 7)
call_190 = relay.TupleGetItem(func_161_call(relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('bool'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(const_186.astype('float32'), [5, 11]), relay.reshape(var_187.astype('float32'), []), relay.reshape(const_188.astype('float32'), [2,]), relay.reshape(var_189.astype('int16'), [1, 2016]), relay.reshape(const_186.astype('float32'), [5, 11]), ), 7)
bop_191 = relay.divide(uop_170.astype('float32'), relay.reshape(bop_167.astype('float32'), relay.shape_of(uop_170))) # shape=(14,)
uop_194 = relay.sinh(bop_180.astype('float64')) # shape=(14,)
bop_196 = relay.divide(uop_194.astype('float64'), relay.reshape(bop_167.astype('float64'), relay.shape_of(uop_194))) # shape=(14,)
uop_199 = relay.rsqrt(bop_196.astype('float64')) # shape=(14,)
bop_201 = relay.not_equal(uop_199.astype('bool'), relay.reshape(var_166.astype('bool'), relay.shape_of(uop_199))) # shape=(14,)
bop_204 = relay.bitwise_xor(bop_201.astype('uint64'), const_188.astype('uint64')) # shape=(2, 14)
bop_207 = relay.floor_divide(uop_199.astype('float64'), relay.reshape(var_166.astype('float64'), relay.shape_of(uop_199))) # shape=(14,)
bop_210 = relay.multiply(uop_194.astype('uint8'), relay.reshape(bop_180.astype('uint8'), relay.shape_of(uop_194))) # shape=(14,)
uop_213 = relay.tan(bop_204.astype('float32')) # shape=(2, 14)
uop_215 = relay.atan(uop_213.astype('float32')) # shape=(2, 14)
uop_217 = relay.cos(uop_215.astype('float32')) # shape=(2, 14)
bop_219 = relay.add(uop_213.astype('uint32'), var_166.astype('uint32')) # shape=(2, 14)
uop_222 = relay.sqrt(bop_207.astype('float32')) # shape=(14,)
output = relay.Tuple([bop_177,uop_183,call_185,const_186,var_187,var_189,bop_191,bop_210,uop_217,bop_219,uop_222,])
output2 = relay.Tuple([bop_177,uop_183,call_190,const_186,var_187,var_189,bop_191,bop_210,uop_217,bop_219,uop_222,])
func_224 = relay.Function([var_163,var_166,var_187,var_189,], output)
mod['func_224'] = func_224
mod = relay.transform.InferType()(mod)
mutated_mod['func_224'] = func_224
mutated_mod = relay.transform.InferType()(mutated_mod)
func_224_call = mutated_mod.get_global_var('func_224')
var_226 = relay.var("var_226", dtype = "float64", shape = (14,))#candidate|226|(14,)|var|float64
var_227 = relay.var("var_227", dtype = "float64", shape = (14,))#candidate|227|(14,)|var|float64
var_228 = relay.var("var_228", dtype = "float32", shape = ())#candidate|228|()|var|float32
var_229 = relay.var("var_229", dtype = "int16", shape = (252, 8))#candidate|229|(252, 8)|var|int16
call_225 = func_224_call(var_226,var_227,var_228,var_229,)
output = call_225
func_230 = relay.Function([var_226,var_227,var_228,var_229,], output)
mutated_mod['func_230'] = func_230
mutated_mod = relay.transform.InferType()(mutated_mod)
var_232 = relay.var("var_232", dtype = "float32", shape = (10, 7, 1))#candidate|232|(10, 7, 1)|var|float32
uop_233 = relay.rsqrt(var_232.astype('float32')) # shape=(10, 7, 1)
uop_235 = relay.acosh(var_232.astype('float64')) # shape=(10, 7, 1)
var_237 = relay.var("var_237", dtype = "float64", shape = (10, 7, 7))#candidate|237|(10, 7, 7)|var|float64
bop_238 = relay.logical_and(uop_235.astype('bool'), var_237.astype('bool')) # shape=(10, 7, 7)
uop_241 = relay.atan(bop_238.astype('float32')) # shape=(10, 7, 7)
bop_243 = relay.greater_equal(uop_241.astype('bool'), var_232.astype('bool')) # shape=(10, 7, 7)
var_246 = relay.var("var_246", dtype = "float32", shape = (10, 7, 7))#candidate|246|(10, 7, 7)|var|float32
bop_247 = relay.not_equal(uop_241.astype('bool'), relay.reshape(var_246.astype('bool'), relay.shape_of(uop_241))) # shape=(10, 7, 7)
func_148_call = mod.get_global_var('func_148')
func_161_call = mutated_mod.get_global_var('func_161')
const_251 = relay.const([7.245938,4.563919,-1.242114,-8.774332,6.996910,-9.115036,-2.122578,7.316178,5.092107,-9.628703,-6.123879,-7.703867,-1.766971,7.615095,0.526497,4.745157,-9.276964,6.913156,9.126249,9.003370,-5.992792,0.644270,6.968976,7.166017,-1.916124,-2.939779,-7.024916,-5.028379,-5.422452,2.996608,-3.077556,-8.308772,0.873078,0.860775,0.976794,4.929322,-2.577540,-9.464947,0.835688,-1.441992,6.837339,-7.983792,2.624882,0.288471,-9.879501,-4.752274,2.030152,-5.060936,-5.358465,1.527133,-3.335625,5.271796,-9.472457,-2.070785,2.410682], dtype = "float32")#candidate|251|(55,)|const|float32
const_252 = relay.const(3.626395, dtype = "float32")#candidate|252|()|const|float32
const_253 = relay.const([8.186239,3.541195], dtype = "float32")#candidate|253|(2,)|const|float32
const_254 = relay.const([3,-3,9,-4,7,-7,10,5,4,-2,7,2,-10,-10,6,-2,-3,3,-10,-1,6,3,4,2,2,-10,9,-6,-8,10,8,-10,-8,6,-10,4,3,-2,9,-6,9,1,7,-9,-4,8,2,6,2,9,-7,8,2,8,5,4,1,5,7,-4,4,4,3,-5,9,7,-9,-7,-6,-4,-10,-8,-8,-10,7,-5,-9,7,8,-5,8,-7,8,7,9,-6,4,2,6,-4,-10,-4,-9,-1,2,7,9,-6,8,-2,7,-9,-1,-2,5,-2,8,-5,6,5,3,8,6,8,9,2,-7,8,2,-7,-7,-3,6,1,4,-1,-10,9,-3,9,-7,10,1,-2,-10,-1,1,-2,-2,8,-3,-2,-8,-8,-8,-4,9,6,2,5,10,-8,8,7,-7,5,-4,2,8,4,8,-8,4,6,-6,7,-5,-5,9,4,-7,-5,-7,-4,8,2,7,-4,6,8,-1,7,2,-2,6,-8,-6,-5,-1,-8,10,-6,2,10,1,9,5,-9,-8,7,-8,8,-7,5,-3,-10,4,9,-7,3,9,10,-10,10,-2,3,-3,4,-3,9,-3,-8,-6,-5,8,-8,4,8,7,-3,6,-10,8,-8,-2,-6,7,-6,-10,3,-10,4,9,7,3,-6,9,-5,-2,7,-1,1,-6,-8,-7,-6,-2,-1,6,8,10,-6,1,-6,-6,7,6,5,6,-10,-2,1,9,-6,4,4,-7,-2,6,1,3,-1,-7,-10,-8,9,-9,6,6,2,-4,-3,1,-9,-5,10,10,6,-6,4,7,5,-6,3,-3,-10,-5,9,-8,2,-1,-1,4,10,-4,-3,8,8,-9,5,-9,-10,-6,10,-1,9,-7,-5,-6,8,-1,-2,8,-2,2,-10,2,5,3,4,-8,-6,-1,-5,-9,-10,5,-9,3,-2,1,-8,-4,-3,-8,10,9,-2,-8,1,1,6,-3,-1,-10,-5,-6,4,-8,6,-3,6,2,-4,7,6,7,4,-10,-8,7,-1,-8,5,-3,4,7,10,7,8,9,2,9,-8,2,-7,-1,3,5,-7,-9,4,10,3,-10,-1,-8,-7,-2,6,-10,4,3,-3,7,5,-1,-10,-3,-8,8,1,4,-3,-2,-10,-3,-6,-4,-3,10,-5,-8,-4,1,-5,-1,7,4,-2,10,-1,-6,9,7,-1,1,10,2,-2,5,-4,-4,6,10,6,-1,2,3,9,5,-9,-5,-8,4,-2,7,-9,-6,1,-2,5,-4,-1,8,9,10,-10,8,4,7,-10,3,-7,4,-6,-5,-10,-9,7,10,3,-5,-5,-2,-4,-6,-5,-6,7,-10,-6,3,3,-4,-9,8,4,-4,6,-9,-5,5,-1,2,-10,-10,-7,-10,-5,3,-1,-6,6,-4,-6,-5,8,-4,-2,9,-9,6,-2,-8,-3,10,6,8,8,6,-7,-6,2,5,6,-4,5,-3,-8,-5,-4,4,-1,-1,4,-2,10,-7,-9,6,6,-2,-2,7,6,1,-7,9,7,8,9,4,9,2,2,5,10,1,4,1,10,-9,9,6,10,-9,3,8,5,-3,-10,6,-8,9,5,-2,-4,-3,-1,-10,2,-7,5,10,-7,-6,1,-4,-9,9,-5,-8,-3,5,-9,-2,6,-9,-7,8,9,-1,-6,-9,10,-8,2,7,8,-7,6,-9,3,3,6,-5,7,-7,-1,-9,-8,-7,4,4,-3,-10,-4,-3,-8,6,10,-1,5,10,-8,-6,-8,7,-6,-4,5,10,-3,-4,-1,-5,10,-8,5,3,-8,10,-4,10,9,1,-7,3,5,-3,8,7,-5,2,-2,-3,-4,6,1,7,9,-6,-10,10,-6,2,-4,-1,4,9,8,6,-9,9,-10,2,6,-10,6,-2,5,-8,4,8,-1,-4,10,-1,4,-9,-8,-4,8,1,-3,10,-10,7,-6,-2,3,-7,7,4,-10,6,3,-2,-3,8,-9,-6,-7,-3,-9,1,2,-6,6,10,2,-3,7,9,10,8,8,-8,-10,-9,5,-2,9,-6,8,2,4,-9,3,-1,-7,-4,-6,1,3,-4,3,-5,-5,-9,9,-5,1,5,3,-8,-6,5,-5,7,-10,6,6,6,-4,-4,-8,-7,-4,2,-8,-6,-5,6,1,9,4,-3,-7,-4,5,6,-7,-4,-5,8,9,10,9,5,1,1,9,6,4,9,-7,5,-9,-7,-5,10,3,9,7,-1,-8,-4,6,-7,1,10,-2,-4,-1,-5,-7,10,6,3,-3,-2,7,-2,-7,3,-4,3,-7,-4,9,-5,-6,-10,6,6,6,-2,5,5,-2,7,1,-1,9,7,7,-8,-8,-5,-10,4,10,-8,-2,6,-1,-10,-5,1,6,-3,9,-8,-10,1,-6,5,-3,-3,-7,-6,2,-3,7,-10,-1,8,6,6,8,5,2,-10,-4,9,1,-4,-3,9,-2,-5,-6,4,8,10,-7,10,5,-6,4,-4,1,8,-4,4,-10,10,1,-7,-4,-4,2,-2,-5,-9,4,-4,9,8,10,1,2,-7,-5,-1,3,-2,1,-3,-4,7,-4,-2,-2,-8,4,-6,1,9,-8,-3,-8,-6,-10,-9,-9,-10,-7,8,5,-4,5,5,-3,-10,-5,-2,9,-3,-6,3,9,8,8,-5,-2,-1,-1,3,-9,7,-9,6,-8,6,1,7,3,-6,-3,-3,2,-4,-8,-8,-1,10,3,-7,-7,3,4,-10,9,-9,6,9,-6,10,-9,3,-3,-1,9,7,5,-3,-4,-8,-8,-7,-9,-4,-7,-10,-3,8,6,-4,9,10,-3,-3,-6,10,-7,10,-1,-2,-1,9,-3,-1,8,3,-7,1,6,7,-6,9,3,-9,6,-5,4,-10,-10,-6,-6,-8,-4,3,5,-3,6,-4,1,-3,5,-1,-3,5,-2,4,2,1,2,-5,1,-8,9,-5,-5,7,-8,8,9,-7,-4,1,2,1,3,3,-3,-4,-4,3,9,1,8,9,3,-2,-6,5,2,-3,2,-2,8,-7,5,2,-10,1,4,-9,7,-6,-4,-10,8,3,9,5,1,10,2,10,-6,-9,-3,7,-10,3,-5,2,6,1,1,5,-8,10,6,9,2,-2,10,5,4,6,10,-3,3,4,-2,8,3,-7,1,-10,-7,10,-4,7,-2,7,1,10,4,8,4,-5,7,7,9,10,4,3,-4,5,-5,10,2,10,6,1,5,1,7,9,-7,-2,-9,-8,7,-9,-6,8,8,4,-5,8,-9,3,6,2,1,6,-8,2,-8,8,1,1,-4,-5,-10,-9,1,6,4,-3,-1,4,3,-7,-9,6,2,9,-6,3,9,-6,-3,-10,-10,1,-1,-5,1,7,10,7,-5,-6,-5,7,-2,-6,5,4,-6,1,10,-2,-9,-10,-4,-8,7,10,-4,6,9,-5,-8,2,-5,-3,-6,-7,-8,1,-8,-3,-5,-7,-9,9,6,4,2,-10,-6,4,-10,8,-10,9,10,-10,-6,-5,1,5,-7,-1,-5,-8,6,5,-2,3,2,9,8,6,-8,9,5,7,4,-1,-1,-10,-10,2,-5,-6,-8,8,4,-10,-5,2,-4,3,8,-7,-2,3,-1,6,1,3,-8,-5,-5,7,-3,-1,9,-5,8,6,-8,-4,6,-6,-5,-6,10,-5,6,6,-10,-8,-10,6,4,10,2,2,-5,5,-7,-2,2,-1,-9,2,3,8,3,-1,-7,1,-4,-7,1,-10,-3,-5,-2,-5,-3,-10,-5,8,-4,6,-10,3,2,-5,-2,-3,-1,-9,-5,4,1,2,6,-5,1,-4,-10,6,-6,7,10,-4,8,-4,10,-9,-3,7,3,2,5,6,2,9,5,-8,8,5,-7,-7,6,8,-8,-1,-6,6,6,7,8,-7,-4,7,9,5,9,-3,-9,6,-5,-10,-7,-4,-10,10,9,-10,-2,-6,-9,7,5,2,-10,-1,3,-7,10,3,6,-10,1,-9,-7,4,-8,-5,-10,8,5,-6,-3,-1,6,-4,9,7,-4,-5,-3,-8,-9,-5,9,-8,5,5,8,-10,9,-9,8,8,-6,4,1,2,6,-8,1,1,1,1,-1,-6,-6,1,-2,-5,-2,-8,4,3,-7,5,9,1,4,-1,-6,-9,3,6,9,-4,2,-5,1,-6,-1,6,9,-9,-1,9,-9,-10,6,-8,7,7,-5,6,10,-6,-3,10,10,-4,-2,-10,1,3,-8,-4,6,-4,4,7,-2,-7,-5,-8,8,6,6,3,8,7,-1,-10,-2,6,-6,4,-7,-2,7,5,-1,7,9,-5,8,-2,2,-3,-6,-2,5,-2,-6,5,4,9,3,5,6,10,3,-2,-5,9,6,-6,9,5,2,-1,2,-4,-5,-8,2,-2,1,9,-9,-7,-4,-7,-9,-10,-2,7,-10,8,3,-4,-5,9,-2,-10,-9,-7,-10,-5,7,-1,-6,-3,-3,-10,7,1,-2,-8,1,7,-6,2,-9,6,2,-8,-8,-3,-2,-7,-8,-2,1,5,7,3,-6,-1,9,2,-7,-10,-1,-5,-5,-3,1,3,3,-5,10,6,-2,-1,-6,-4,5,-9,-4,-1,-7,2,-8,-1,3,-3,-5,-8,-8,9,4,-9,5,7,-4,-10,1,-10,-7,1,-8,-2,5,-9,-9,-7,2,-3,-5,8,-7,10,-10,-6,-1,8,3,-2,2,-1,2,5,5,-2,-6,-2,9,-2,2,8,7,1,-10,8,4,4,-7,-9,6,-5,-2,-7,8,-1,-7,4,-7,-4,9,8,9,-1,-2,1,4,2,4,8,-1,4,8,-4,4,3,-9,-9,4,7,2,3,5,-1,-5,2,-8,3,-4,-4,-4,10,-9,-4,-8,-8,-10,-6,6,-5,-8,8,3,5,-10,-4,5,8,2,8,2,10,-7,-8,-4,6,9,-8,-7,10,2,-2,7,-3,-1,-10,-7,6,7,-10,6,-2,2,-9,-3,-4,10,-1,5,7,7,10,8,-9,-1,-5,-7,-6,-5,5,4,4,-5,-3,1,8,-7,-1,8,1,10,-8,1,5,-9,-5,-10,-8,-3,7,1,3,6,5,5,5,2,-4,9,10,-10,2,-5,6,9,4,-1,-2,-1,2,4,5,1,8,1,-8,2,-10,10,5,-1,1,-5,10,-9,2,-10,-7,8,-10,3,7,-10,2,-5,9,-7,-10,3,-8,10,6,-9,-10,-10,4,-10,9,-2,-10,-2,-4,1,-3,2,7,-5,4,5,3,-6,-9,9,-8,2,9,9,5,5,-2,-3,-1,-1,-4,-8,-2,-3,-1,-10,5,-9,8,-8,-7,4,2,-10,6,9,-6,5,1,10,2,1,3,-4,7,5,6,5,5,-2], dtype = "int16")#candidate|254|(2016,)|const|int16
call_250 = relay.TupleGetItem(func_148_call(relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('bool'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_252.astype('float32'), []), relay.reshape(const_253.astype('float32'), [2,]), relay.reshape(const_254.astype('int16'), [1, 2016]), relay.reshape(const_251.astype('float32'), [5, 11]), ), 2)
call_255 = relay.TupleGetItem(func_161_call(relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('bool'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(const_252.astype('float32'), []), relay.reshape(const_253.astype('float32'), [2,]), relay.reshape(const_254.astype('int16'), [1, 2016]), relay.reshape(const_251.astype('float32'), [5, 11]), ), 2)
uop_256 = relay.cosh(bop_247.astype('float32')) # shape=(10, 7, 7)
bop_258 = relay.bitwise_xor(uop_256.astype('uint64'), relay.reshape(var_246.astype('uint64'), relay.shape_of(uop_256))) # shape=(10, 7, 7)
const_261 = relay.const([[[False,False,False,False,False,False,True],[False,False,True,False,True,True,True],[True,False,False,False,True,False,False],[True,False,False,True,False,True,True],[False,False,True,False,False,False,False],[False,False,True,True,True,False,True],[False,True,False,True,True,True,False]],[[False,True,False,True,True,False,False],[True,False,False,True,False,True,True],[False,True,True,True,False,True,False],[True,True,True,True,True,True,True],[False,False,True,True,True,True,False],[False,False,True,True,False,True,False],[False,False,True,False,False,True,True]],[[False,False,True,True,True,False,False],[False,False,True,True,False,False,False],[False,False,False,False,False,True,False],[False,False,False,False,True,False,False],[True,True,False,True,True,True,False],[False,False,False,True,False,False,False],[False,False,False,False,False,False,True]],[[False,False,True,False,False,True,False],[True,True,True,False,False,True,True],[True,False,True,False,False,False,True],[False,False,True,True,False,True,True],[True,True,False,True,True,True,False],[False,True,False,True,False,True,False],[True,False,True,False,True,True,False]],[[True,True,True,False,True,False,True],[True,True,False,False,True,True,False],[False,False,False,False,True,False,False],[False,False,False,True,False,True,True],[True,False,False,True,True,True,True],[False,False,False,False,True,False,False],[False,True,False,True,False,False,True]],[[True,True,True,True,True,True,False],[True,True,False,True,True,False,False],[True,True,True,True,True,True,True],[True,True,False,True,True,False,False],[True,True,False,False,False,True,False],[True,True,False,False,False,False,True],[True,False,True,False,False,False,False]],[[True,True,True,False,True,True,False],[False,False,False,True,True,False,True],[True,False,True,False,True,False,False],[True,False,False,False,True,True,True],[True,False,True,False,True,False,True],[False,False,True,False,False,True,True],[True,False,True,False,False,False,False]],[[True,False,False,False,True,False,False],[False,False,True,True,False,False,True],[True,True,False,False,True,True,False],[True,True,False,False,True,True,False],[True,True,False,False,False,False,False],[True,True,False,True,True,False,False],[True,False,True,False,True,False,False]],[[False,True,True,True,False,True,True],[True,False,True,False,True,True,False],[True,False,False,False,True,True,True],[False,True,True,True,True,True,True],[True,True,False,True,False,True,True],[True,False,False,False,False,False,True],[True,True,True,True,True,False,True]],[[False,True,True,False,False,False,True],[True,True,True,False,False,True,False],[True,False,True,False,False,True,False],[False,True,True,True,True,True,False],[False,True,False,True,False,True,True],[False,True,True,True,True,False,False],[True,True,False,False,True,True,False]]], dtype = "bool")#candidate|261|(10, 7, 7)|const|bool
bop_262 = relay.maximum(bop_247.astype('float32'), relay.reshape(const_261.astype('float32'), relay.shape_of(bop_247))) # shape=(10, 7, 7)
uop_265 = relay.rsqrt(var_237.astype('float32')) # shape=(10, 7, 7)
bop_267 = relay.greater_equal(bop_258.astype('bool'), uop_235.astype('bool')) # shape=(10, 7, 7)
uop_270 = relay.tan(uop_256.astype('float64')) # shape=(10, 7, 7)
uop_272 = relay.erf(uop_270.astype('float64')) # shape=(10, 7, 7)
uop_274 = relay.exp(uop_256.astype('float64')) # shape=(10, 7, 7)
bop_276 = relay.power(uop_272.astype('float32'), const_252.astype('float32')) # shape=(10, 7, 7)
var_279 = relay.var("var_279", dtype = "float64", shape = (10, 7, 7))#candidate|279|(10, 7, 7)|var|float64
bop_280 = relay.bitwise_xor(uop_270.astype('uint64'), relay.reshape(var_279.astype('uint64'), relay.shape_of(uop_270))) # shape=(10, 7, 7)
uop_283 = relay.rsqrt(bop_280.astype('float64')) # shape=(10, 7, 7)
func_148_call = mod.get_global_var('func_148')
func_161_call = mutated_mod.get_global_var('func_161')
call_285 = relay.TupleGetItem(func_148_call(relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(call_250.astype('bool'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(const_252.astype('float32'), []), relay.reshape(const_253.astype('float32'), [2,]), relay.reshape(const_254.astype('int16'), [1, 2016]), relay.reshape(call_250.astype('float32'), [5, 11]), ), 9)
call_286 = relay.TupleGetItem(func_161_call(relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(call_250.astype('bool'), [5, 11]), relay.reshape(const_251.astype('float32'), [5, 11]), relay.reshape(call_250.astype('float32'), [5, 11]), relay.reshape(const_252.astype('float32'), []), relay.reshape(const_253.astype('float32'), [2,]), relay.reshape(const_254.astype('int16'), [1, 2016]), relay.reshape(call_250.astype('float32'), [5, 11]), ), 9)
bop_287 = relay.right_shift(bop_280.astype('int32'), relay.reshape(var_246.astype('int32'), relay.shape_of(bop_280))) # shape=(10, 7, 7)
var_290 = relay.var("var_290", dtype = "bool", shape = (10, 7, 7))#candidate|290|(10, 7, 7)|var|bool
bop_291 = relay.mod(bop_267.astype('float32'), relay.reshape(var_290.astype('float32'), relay.shape_of(bop_267))) # shape=(10, 7, 7)
uop_294 = relay.sinh(uop_283.astype('float32')) # shape=(10, 7, 7)
bop_296 = relay.multiply(uop_272.astype('int16'), relay.reshape(uop_294.astype('int16'), relay.shape_of(uop_272))) # shape=(10, 7, 7)
bop_299 = relay.minimum(bop_267.astype('int32'), relay.reshape(uop_283.astype('int32'), relay.shape_of(bop_267))) # shape=(10, 7, 7)
uop_302 = relay.erf(uop_274.astype('float32')) # shape=(10, 7, 7)
var_304 = relay.var("var_304", dtype = "float32", shape = (10, 7, 7))#candidate|304|(10, 7, 7)|var|float32
bop_305 = relay.floor_divide(bop_291.astype('float32'), relay.reshape(var_304.astype('float32'), relay.shape_of(bop_291))) # shape=(10, 7, 7)
var_308 = relay.var("var_308", dtype = "float32", shape = (10, 7, 7))#candidate|308|(10, 7, 7)|var|float32
bop_309 = relay.mod(uop_265.astype('float64'), relay.reshape(var_308.astype('float64'), relay.shape_of(uop_265))) # shape=(10, 7, 7)
bop_312 = relay.greater_equal(uop_302.astype('bool'), uop_235.astype('bool')) # shape=(10, 7, 7)
uop_315 = relay.acosh(bop_296.astype('float32')) # shape=(10, 7, 7)
var_317 = relay.var("var_317", dtype = "float32", shape = (10, 7, 7))#candidate|317|(10, 7, 7)|var|float32
bop_318 = relay.subtract(uop_315.astype('float64'), relay.reshape(var_317.astype('float64'), relay.shape_of(uop_315))) # shape=(10, 7, 7)
bop_321 = relay.greater_equal(uop_283.astype('bool'), relay.reshape(bop_243.astype('bool'), relay.shape_of(uop_283))) # shape=(10, 7, 7)
bop_324 = relay.power(uop_315.astype('float64'), relay.reshape(var_237.astype('float64'), relay.shape_of(uop_315))) # shape=(10, 7, 7)
uop_327 = relay.acos(bop_296.astype('float32')) # shape=(10, 7, 7)
uop_329 = relay.log(uop_272.astype('float64')) # shape=(10, 7, 7)
output = relay.Tuple([uop_233,call_250,const_251,const_253,const_254,bop_262,bop_276,call_285,bop_287,bop_299,bop_305,bop_309,bop_312,bop_318,bop_321,bop_324,uop_327,uop_329,])
output2 = relay.Tuple([uop_233,call_255,const_251,const_253,const_254,bop_262,bop_276,call_286,bop_287,bop_299,bop_305,bop_309,bop_312,bop_318,bop_321,bop_324,uop_327,uop_329,])
func_331 = relay.Function([var_232,var_237,var_246,var_279,var_290,var_304,var_308,var_317,], output)
mod['func_331'] = func_331
mod = relay.transform.InferType()(mod)
mutated_mod['func_331'] = func_331
mutated_mod = relay.transform.InferType()(mutated_mod)
func_331_call = mutated_mod.get_global_var('func_331')
var_333 = relay.var("var_333", dtype = "float32", shape = (10, 7, 1))#candidate|333|(10, 7, 1)|var|float32
var_334 = relay.var("var_334", dtype = "float64", shape = (10, 7, 7))#candidate|334|(10, 7, 7)|var|float64
var_335 = relay.var("var_335", dtype = "float32", shape = (10, 7, 7))#candidate|335|(10, 7, 7)|var|float32
var_336 = relay.var("var_336", dtype = "float64", shape = (10, 7, 7))#candidate|336|(10, 7, 7)|var|float64
var_337 = relay.var("var_337", dtype = "bool", shape = (10, 7, 7))#candidate|337|(10, 7, 7)|var|bool
var_338 = relay.var("var_338", dtype = "float32", shape = (10, 7, 7))#candidate|338|(10, 7, 7)|var|float32
var_339 = relay.var("var_339", dtype = "float32", shape = (10, 7, 7))#candidate|339|(10, 7, 7)|var|float32
var_340 = relay.var("var_340", dtype = "float32", shape = (10, 7, 7))#candidate|340|(10, 7, 7)|var|float32
call_332 = func_331_call(var_333,var_334,var_335,var_336,var_337,var_338,var_339,var_340,)
output = call_332
func_341 = relay.Function([var_333,var_334,var_335,var_336,var_337,var_338,var_339,var_340,], output)
mutated_mod['func_341'] = func_341
mutated_mod = relay.transform.InferType()(mutated_mod)
var_343 = relay.var("var_343", dtype = "float32", shape = (5,))#candidate|343|(5,)|var|float32
uop_344 = relay.asin(var_343.astype('float32')) # shape=(5,)
uop_346 = relay.exp(uop_344.astype('float32')) # shape=(5,)
uop_348 = relay.atan(uop_346.astype('float32')) # shape=(5,)
uop_350 = relay.erf(uop_348.astype('float64')) # shape=(5,)
output = relay.Tuple([uop_350,])
output2 = relay.Tuple([uop_350,])
F = relay.Function([var_343,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_343,], output2)
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
input_343= np.array([4.731715,9.609326,9.490528,1.664205,5.717578], dtype='float32')
module1.set_input('var_343', input_343)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_343, )
res3 = intrp3.evaluate()(input_343, )
res4 = intrp4.evaluate()(input_343, )
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
module5.set_input('var_343', input_343)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_343, )
res7 = intrp7.evaluate()(input_343, )
res8 = intrp8.evaluate()(input_343, )
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
module9.set_input('var_343', input_343)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_343, )
res11 = intrp11.evaluate()(input_343, )
res12 = intrp12.evaluate()(input_343, )
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
module13.set_input('var_343', input_343)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_343, )
res15 = intrp15.evaluate()(input_343, )
res16 = intrp16.evaluate()(input_343, )
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
module17.set_input('var_343', input_343)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_343, )
res19 = intrp19.evaluate()(input_343, )
res20 = intrp20.evaluate()(input_343, )
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
module21.set_input('var_343', input_343)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_343, )
res23 = intrp23.evaluate()(input_343, )
res24 = intrp24.evaluate()(input_343, )
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