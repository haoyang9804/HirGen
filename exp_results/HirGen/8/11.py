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
var_0 = relay.var("var_0", dtype = "uint8", shape = (8, 3, 1))#candidate|0|(8, 3, 1)|var|uint8
var_1 = relay.var("var_1", dtype = "uint8", shape = (8, 3, 5))#candidate|1|(8, 3, 5)|var|uint8
bop_2 = relay.bitwise_and(var_0.astype('uint8'), var_1.astype('uint8')) # shape=(8, 3, 5)
uop_5 = relay.acosh(bop_2.astype('float64')) # shape=(8, 3, 5)
var_7 = relay.var("var_7", dtype = "uint8", shape = (8, 3, 5))#candidate|7|(8, 3, 5)|var|uint8
bop_8 = relay.right_shift(var_1.astype('int16'), relay.reshape(var_7.astype('int16'), relay.shape_of(var_1))) # shape=(8, 3, 5)
output = relay.Tuple([uop_5,bop_8,])
output2 = relay.Tuple([uop_5,bop_8,])
func_11 = relay.Function([var_0,var_1,var_7,], output)
mod['func_11'] = func_11
mod = relay.transform.InferType()(mod)
var_12 = relay.var("var_12", dtype = "uint8", shape = (8, 3, 1))#candidate|12|(8, 3, 1)|var|uint8
var_13 = relay.var("var_13", dtype = "uint8", shape = (8, 3, 5))#candidate|13|(8, 3, 5)|var|uint8
var_14 = relay.var("var_14", dtype = "uint8", shape = (8, 3, 5))#candidate|14|(8, 3, 5)|var|uint8
output = func_11(var_12,var_13,var_14,)
func_15 = relay.Function([var_12,var_13,var_14,], output)
mutated_mod['func_15'] = func_15
mutated_mod = relay.transform.InferType()(mutated_mod)
const_17 = relay.const([-7.671758], dtype = "float64")#candidate|17|(1,)|const|float64
uop_18 = relay.atan(const_17.astype('float64')) # shape=(1,)
func_11_call = mod.get_global_var('func_11')
func_15_call = mutated_mod.get_global_var('func_15')
var_21 = relay.var("var_21", dtype = "uint8", shape = (24,))#candidate|21|(24,)|var|uint8
const_22 = relay.const([-2,2,-10,5,6,6,-8,6,5,2,-2,2,6,9,4,3,-10,-3,5,-10,-7,-2,3,-6,-5,2,5,9,-8,9,6,-9,6,-9,-9,1,-3,-9,-2,7,-3,-5,3,-10,-2,-4,-1,-7,6,-10,-2,10,-6,6,-6,-7,3,7,-8,8,6,-8,-6,3,-10,-8,-6,10,7,-1,-8,-2,-2,7,-8,10,9,-10,-7,-9,-8,9,-5,-5,6,7,-6,-5,10,10,6,-1,-2,-10,-5,-5,-7,-6,-5,7,7,-5,7,-3,9,6,-5,-2,9,5,-2,7,-5,4,-4,-1,-6,-5,-4,9], dtype = "uint8")#candidate|22|(120,)|const|uint8
call_20 = relay.TupleGetItem(func_11_call(relay.reshape(var_21.astype('uint8'), [8, 3, 1]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), ), 0)
call_23 = relay.TupleGetItem(func_15_call(relay.reshape(var_21.astype('uint8'), [8, 3, 1]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), ), 0)
var_24 = relay.var("var_24", dtype = "uint8", shape = (120,))#candidate|24|(120,)|var|uint8
bop_25 = relay.floor_divide(const_22.astype('float64'), relay.reshape(var_24.astype('float64'), relay.shape_of(const_22))) # shape=(120,)
uop_28 = relay.sinh(var_21.astype('float64')) # shape=(24,)
var_30 = relay.var("var_30", dtype = "float64", shape = (1,))#candidate|30|(1,)|var|float64
bop_31 = relay.minimum(uop_18.astype('int32'), relay.reshape(var_30.astype('int32'), relay.shape_of(uop_18))) # shape=(1,)
uop_34 = relay.rsqrt(uop_28.astype('float32')) # shape=(24,)
uop_36 = relay.asin(uop_34.astype('float64')) # shape=(24,)
bop_38 = relay.less_equal(uop_34.astype('bool'), relay.reshape(uop_28.astype('bool'), relay.shape_of(uop_34))) # shape=(24,)
var_41 = relay.var("var_41", dtype = "float64", shape = (24,))#candidate|41|(24,)|var|float64
bop_42 = relay.floor_mod(uop_36.astype('float32'), relay.reshape(var_41.astype('float32'), relay.shape_of(uop_36))) # shape=(24,)
func_11_call = mod.get_global_var('func_11')
func_15_call = mutated_mod.get_global_var('func_15')
call_45 = relay.TupleGetItem(func_11_call(relay.reshape(var_21.astype('uint8'), [8, 3, 1]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), ), 1)
call_46 = relay.TupleGetItem(func_15_call(relay.reshape(var_21.astype('uint8'), [8, 3, 1]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), relay.reshape(const_22.astype('uint8'), [8, 3, 5]), ), 1)
uop_47 = relay.atanh(bop_42.astype('float32')) # shape=(24,)
bop_49 = relay.logical_and(uop_47.astype('bool'), const_17.astype('bool')) # shape=(24,)
uop_52 = relay.sin(bop_49.astype('float32')) # shape=(24,)
func_11_call = mod.get_global_var('func_11')
func_15_call = mutated_mod.get_global_var('func_15')
call_54 = relay.TupleGetItem(func_11_call(relay.reshape(var_21.astype('uint8'), [8, 3, 1]), relay.reshape(bop_25.astype('uint8'), [8, 3, 5]), relay.reshape(var_24.astype('uint8'), [8, 3, 5]), ), 0)
call_55 = relay.TupleGetItem(func_15_call(relay.reshape(var_21.astype('uint8'), [8, 3, 1]), relay.reshape(bop_25.astype('uint8'), [8, 3, 5]), relay.reshape(var_24.astype('uint8'), [8, 3, 5]), ), 0)
bop_56 = relay.power(uop_52.astype('float64'), relay.reshape(uop_36.astype('float64'), relay.shape_of(uop_52))) # shape=(24,)
var_59 = relay.var("var_59", dtype = "float32", shape = (24,))#candidate|59|(24,)|var|float32
bop_60 = relay.right_shift(uop_47.astype('uint8'), relay.reshape(var_59.astype('uint8'), relay.shape_of(uop_47))) # shape=(24,)
uop_63 = relay.atanh(uop_52.astype('float64')) # shape=(24,)
uop_65 = relay.cos(bop_56.astype('float64')) # shape=(24,)
uop_67 = relay.tan(bop_56.astype('float32')) # shape=(24,)
bop_69 = relay.not_equal(uop_52.astype('bool'), relay.reshape(uop_67.astype('bool'), relay.shape_of(uop_52))) # shape=(24,)
output = relay.Tuple([call_20,bop_25,bop_31,bop_38,call_45,call_54,bop_60,uop_63,uop_65,bop_69,])
output2 = relay.Tuple([call_23,bop_25,bop_31,bop_38,call_46,call_55,bop_60,uop_63,uop_65,bop_69,])
func_72 = relay.Function([var_21,var_24,var_30,var_41,var_59,], output)
mod['func_72'] = func_72
mod = relay.transform.InferType()(mod)
var_73 = relay.var("var_73", dtype = "uint8", shape = (24,))#candidate|73|(24,)|var|uint8
var_74 = relay.var("var_74", dtype = "uint8", shape = (120,))#candidate|74|(120,)|var|uint8
var_75 = relay.var("var_75", dtype = "float64", shape = (1,))#candidate|75|(1,)|var|float64
var_76 = relay.var("var_76", dtype = "float64", shape = (24,))#candidate|76|(24,)|var|float64
var_77 = relay.var("var_77", dtype = "float32", shape = (24,))#candidate|77|(24,)|var|float32
output = func_72(var_73,var_74,var_75,var_76,var_77,)
func_78 = relay.Function([var_73,var_74,var_75,var_76,var_77,], output)
mutated_mod['func_78'] = func_78
mutated_mod = relay.transform.InferType()(mutated_mod)
const_80 = relay.const(-2.892653, dtype = "float32")#candidate|80|()|const|float32
uop_81 = relay.acos(const_80.astype('float32')) # shape=()
func_72_call = mod.get_global_var('func_72')
func_78_call = mutated_mod.get_global_var('func_78')
var_84 = relay.var("var_84", dtype = "uint8", shape = (2, 12))#candidate|84|(2, 12)|var|uint8
const_85 = relay.const([[-7],[-4],[5],[2],[2],[-1],[-1],[4],[-8],[-4],[2],[3],[8],[10],[-9],[-8],[5],[5],[9],[10],[-5],[4],[6],[1],[-8],[4],[-9],[-7],[2],[8],[8],[-8],[-10],[-4],[2],[-8],[5],[-3],[9],[-6],[1],[6],[4],[-8],[3],[-10],[3],[-3],[-2],[-10],[1],[1],[2],[-8],[10],[-8],[3],[-2],[9],[5],[-7],[2],[-2],[-3],[-4],[-1],[2],[2],[6],[7],[-5],[8],[10],[4],[7],[1],[2],[-3],[10],[1],[8],[-8],[3],[-8],[-3],[6],[9],[5],[1],[9],[7],[-3],[-3],[7],[2],[-7],[7],[-5],[-9],[-1],[8],[-8],[9],[-4],[-3],[10],[5],[2],[2],[3],[-5],[-6],[-8],[-5],[4],[7],[3],[-8],[-1],[5]], dtype = "uint8")#candidate|85|(120, 1)|const|uint8
call_83 = relay.TupleGetItem(func_72_call(relay.reshape(var_84.astype('uint8'), [24,]), relay.reshape(const_85.astype('uint8'), [120,]), relay.reshape(const_80.astype('float64'), [1,]), relay.reshape(var_84.astype('float64'), [24,]), relay.reshape(var_84.astype('float32'), [24,]), ), 2)
call_86 = relay.TupleGetItem(func_78_call(relay.reshape(var_84.astype('uint8'), [24,]), relay.reshape(const_85.astype('uint8'), [120,]), relay.reshape(const_80.astype('float64'), [1,]), relay.reshape(var_84.astype('float64'), [24,]), relay.reshape(var_84.astype('float32'), [24,]), ), 2)
output = relay.Tuple([uop_81,call_83,var_84,const_85,])
output2 = relay.Tuple([uop_81,call_86,var_84,const_85,])
func_87 = relay.Function([var_84,], output)
mod['func_87'] = func_87
mod = relay.transform.InferType()(mod)
var_88 = relay.var("var_88", dtype = "uint8", shape = (2, 12))#candidate|88|(2, 12)|var|uint8
output = func_87(var_88)
func_89 = relay.Function([var_88], output)
mutated_mod['func_89'] = func_89
mutated_mod = relay.transform.InferType()(mutated_mod)
var_91 = relay.var("var_91", dtype = "float32", shape = ())#candidate|91|()|var|float32
uop_92 = relay.atan(var_91.astype('float32')) # shape=()
bop_94 = relay.greater(var_91.astype('bool'), uop_92.astype('bool')) # shape=()
var_97 = relay.var("var_97", dtype = "float32", shape = (10,))#candidate|97|(10,)|var|float32
bop_98 = relay.minimum(var_91.astype('int16'), var_97.astype('int16')) # shape=(10,)
uop_101 = relay.log(bop_98.astype('float32')) # shape=(10,)
uop_103 = relay.atan(uop_92.astype('float32')) # shape=()
var_105 = relay.var("var_105", dtype = "bool", shape = (2, 5, 11))#candidate|105|(2, 5, 11)|var|bool
bop_106 = relay.bitwise_and(bop_94.astype('int8'), var_105.astype('int8')) # shape=(2, 5, 11)
uop_109 = relay.acos(uop_103.astype('float32')) # shape=()
uop_111 = relay.acosh(uop_92.astype('float64')) # shape=()
uop_113 = relay.exp(uop_109.astype('float64')) # shape=()
uop_115 = relay.asin(bop_106.astype('float32')) # shape=(2, 5, 11)
uop_117 = relay.cos(uop_109.astype('float64')) # shape=()
bop_119 = relay.equal(uop_113.astype('bool'), bop_94.astype('bool')) # shape=()
bop_122 = relay.add(uop_113.astype('uint8'), bop_94.astype('uint8')) # shape=()
uop_125 = relay.log10(uop_101.astype('float32')) # shape=(10,)
uop_127 = relay.cos(uop_111.astype('float32')) # shape=()
uop_129 = relay.atan(var_91.astype('float32')) # shape=()
bop_131 = relay.logical_or(bop_119.astype('bool'), bop_94.astype('bool')) # shape=()
const_134 = relay.const([2.666759], dtype = "float32")#candidate|134|(1,)|const|float32
bop_135 = relay.minimum(uop_109.astype('int8'), const_134.astype('int8')) # shape=()
uop_138 = relay.asin(uop_113.astype('float32')) # shape=()
bop_140 = relay.greater_equal(uop_138.astype('bool'), uop_117.astype('bool')) # shape=()
uop_143 = relay.cos(uop_117.astype('float32')) # shape=()
bop_145 = relay.not_equal(uop_143.astype('bool'), uop_103.astype('bool')) # shape=()
uop_148 = relay.sin(uop_127.astype('float32')) # shape=()
bop_150 = relay.logical_xor(bop_135.astype('uint32'), uop_129.astype('uint32')) # shape=()
uop_153 = relay.log2(bop_145.astype('float32')) # shape=()
func_11_call = mod.get_global_var('func_11')
func_15_call = mutated_mod.get_global_var('func_15')
const_156 = relay.const([-8,5,1,-8,1,9,-2,2,3,7,9,-1,4,-2,-6,7,7,-1,3,7,3,3,6,-7], dtype = "uint8")#candidate|156|(24,)|const|uint8
const_157 = relay.const([8,7,4,3,-6,-8,4,-2,-4,-2,10,-8,3,10,3,-4,-7,2,-4,9,3,-2,3,-2,8,9,-3,3,-2,-8,8,8,4,-9,5,10,-7,7,-4,-8,-6,4,-5,5,-2,4,6,-9,4,-1,-6,-8,-1,3,-5,2,8,-9,3,-2,-9,-6,10,5,3,-2,-5,2,-5,-3,-7,-10,-9,10,4,-8,10,-8,-9,-6,-7,1,-9,7,-9,-6,5,3,-6,-7,-7,-5,-9,7,4,-2,2,-2,-8,8,-3,-6,-10,-4,2,-1,6,-5,-9,-7,-6,5,-2,-2,7,-2,2,-9,-6,3], dtype = "uint8")#candidate|157|(120,)|const|uint8
call_155 = relay.TupleGetItem(func_11_call(relay.reshape(const_156.astype('uint8'), [8, 3, 1]), relay.reshape(const_157.astype('uint8'), [8, 3, 5]), relay.reshape(const_157.astype('uint8'), [8, 3, 5]), ), 1)
call_158 = relay.TupleGetItem(func_15_call(relay.reshape(const_156.astype('uint8'), [8, 3, 1]), relay.reshape(const_157.astype('uint8'), [8, 3, 5]), relay.reshape(const_157.astype('uint8'), [8, 3, 5]), ), 1)
func_72_call = mod.get_global_var('func_72')
func_78_call = mutated_mod.get_global_var('func_78')
call_159 = relay.TupleGetItem(func_72_call(relay.reshape(const_156.astype('uint8'), [24,]), relay.reshape(const_157.astype('uint8'), [120,]), relay.reshape(bop_131.astype('float64'), [1,]), relay.reshape(const_156.astype('float64'), [24,]), relay.reshape(const_156.astype('float32'), [24,]), ), 1)
call_160 = relay.TupleGetItem(func_78_call(relay.reshape(const_156.astype('uint8'), [24,]), relay.reshape(const_157.astype('uint8'), [120,]), relay.reshape(bop_131.astype('float64'), [1,]), relay.reshape(const_156.astype('float64'), [24,]), relay.reshape(const_156.astype('float32'), [24,]), ), 1)
bop_161 = relay.power(uop_115.astype('float32'), bop_131.astype('float32')) # shape=(2, 5, 11)
const_164 = relay.const([[False,True,True,False,True,True],[False,True,False,False,True,False],[True,False,True,False,False,True],[False,False,False,False,True,True],[True,False,False,True,True,False],[True,True,True,True,True,False],[False,True,True,False,False,False],[False,True,False,True,True,False],[False,True,False,True,False,True],[False,True,False,False,True,True],[False,False,True,True,True,False],[False,True,False,True,True,False],[False,False,True,False,True,False],[True,False,False,True,True,False],[True,True,True,True,True,False],[False,False,True,True,True,False]], dtype = "bool")#candidate|164|(16, 6)|const|bool
bop_165 = relay.logical_xor(bop_140.astype('int32'), const_164.astype('int32')) # shape=(16, 6)
bop_168 = relay.multiply(bop_140.astype('uint64'), bop_122.astype('uint64')) # shape=()
bop_171 = relay.bitwise_or(bop_140.astype('uint64'), uop_127.astype('uint64')) # shape=()
uop_174 = relay.log(uop_153.astype('float64')) # shape=()
bop_176 = relay.divide(uop_153.astype('float64'), bop_168.astype('float64')) # shape=()
bop_179 = relay.add(uop_174.astype('uint32'), var_97.astype('uint32')) # shape=(10,)
uop_182 = relay.acos(uop_174.astype('float64')) # shape=()
bop_184 = relay.minimum(uop_174.astype('int32'), call_159.astype('int32')) # shape=(120,)
bop_187 = relay.minimum(uop_174.astype('int32'), call_160.astype('int32')) # shape=(120,)
output = relay.Tuple([uop_125,uop_148,bop_150,call_155,const_156,const_157,bop_161,bop_165,bop_171,bop_176,bop_179,uop_182,bop_184,])
output2 = relay.Tuple([uop_125,uop_148,bop_150,call_158,const_156,const_157,bop_161,bop_165,bop_171,bop_176,bop_179,uop_182,bop_187,])
func_188 = relay.Function([var_91,var_97,var_105,], output)
mod['func_188'] = func_188
mod = relay.transform.InferType()(mod)
var_189 = relay.var("var_189", dtype = "float32", shape = ())#candidate|189|()|var|float32
var_190 = relay.var("var_190", dtype = "float32", shape = (10,))#candidate|190|(10,)|var|float32
var_191 = relay.var("var_191", dtype = "bool", shape = (2, 5, 11))#candidate|191|(2, 5, 11)|var|bool
output = func_188(var_189,var_190,var_191,)
func_192 = relay.Function([var_189,var_190,var_191,], output)
mutated_mod['func_192'] = func_192
mutated_mod = relay.transform.InferType()(mutated_mod)
var_194 = relay.var("var_194", dtype = "float32", shape = ())#candidate|194|()|var|float32
uop_195 = relay.asin(var_194.astype('float32')) # shape=()
bop_197 = relay.logical_xor(var_194.astype('int64'), uop_195.astype('int64')) # shape=()
uop_200 = relay.sin(bop_197.astype('float64')) # shape=()
var_202 = relay.var("var_202", dtype = "float32", shape = (13,))#candidate|202|(13,)|var|float32
bop_203 = relay.logical_or(uop_195.astype('bool'), var_202.astype('bool')) # shape=(13,)
bop_206 = relay.multiply(uop_200.astype('int32'), var_202.astype('int32')) # shape=(13,)
uop_209 = relay.erf(uop_200.astype('float32')) # shape=()
bop_211 = relay.divide(uop_209.astype('float64'), bop_203.astype('float64')) # shape=(13,)
uop_214 = relay.erf(uop_209.astype('float64')) # shape=()
bop_216 = relay.less_equal(uop_200.astype('bool'), bop_211.astype('bool')) # shape=(13,)
var_219 = relay.var("var_219", dtype = "float64", shape = (5, 14, 9))#candidate|219|(5, 14, 9)|var|float64
bop_220 = relay.not_equal(uop_214.astype('bool'), var_219.astype('bool')) # shape=(5, 14, 9)
var_223 = relay.var("var_223", dtype = "int32", shape = (13,))#candidate|223|(13,)|var|int32
bop_224 = relay.power(bop_206.astype('float32'), relay.reshape(var_223.astype('float32'), relay.shape_of(bop_206))) # shape=(13,)
uop_227 = relay.sin(bop_216.astype('float64')) # shape=(13,)
uop_229 = relay.exp(uop_214.astype('float32')) # shape=()
bop_231 = relay.equal(uop_214.astype('bool'), var_202.astype('bool')) # shape=(13,)
output = relay.Tuple([bop_220,bop_224,uop_227,uop_229,bop_231,])
output2 = relay.Tuple([bop_220,bop_224,uop_227,uop_229,bop_231,])
func_234 = relay.Function([var_194,var_202,var_219,var_223,], output)
mod['func_234'] = func_234
mod = relay.transform.InferType()(mod)
var_235 = relay.var("var_235", dtype = "float32", shape = ())#candidate|235|()|var|float32
var_236 = relay.var("var_236", dtype = "float32", shape = (13,))#candidate|236|(13,)|var|float32
var_237 = relay.var("var_237", dtype = "float64", shape = (5, 14, 9))#candidate|237|(5, 14, 9)|var|float64
var_238 = relay.var("var_238", dtype = "int32", shape = (13,))#candidate|238|(13,)|var|int32
output = func_234(var_235,var_236,var_237,var_238,)
func_239 = relay.Function([var_235,var_236,var_237,var_238,], output)
mutated_mod['func_239'] = func_239
mutated_mod = relay.transform.InferType()(mutated_mod)
var_241 = relay.var("var_241", dtype = "float32", shape = (6, 6, 1))#candidate|241|(6, 6, 1)|var|float32
var_242 = relay.var("var_242", dtype = "float32", shape = (6, 6, 16))#candidate|242|(6, 6, 16)|var|float32
bop_243 = relay.floor_mod(var_241.astype('float32'), var_242.astype('float32')) # shape=(6, 6, 16)
func_72_call = mod.get_global_var('func_72')
func_78_call = mutated_mod.get_global_var('func_78')
var_247 = relay.var("var_247", dtype = "uint8", shape = (24, 1))#candidate|247|(24, 1)|var|uint8
var_248 = relay.var("var_248", dtype = "uint8", shape = (3, 40))#candidate|248|(3, 40)|var|uint8
var_249 = relay.var("var_249", dtype = "float64", shape = (1,))#candidate|249|(1,)|var|float64
call_246 = relay.TupleGetItem(func_72_call(relay.reshape(var_247.astype('uint8'), [24,]), relay.reshape(var_248.astype('uint8'), [120,]), relay.reshape(var_249.astype('float64'), [1,]), relay.reshape(var_247.astype('float64'), [24,]), relay.reshape(var_247.astype('float32'), [24,]), ), 1)
call_250 = relay.TupleGetItem(func_78_call(relay.reshape(var_247.astype('uint8'), [24,]), relay.reshape(var_248.astype('uint8'), [120,]), relay.reshape(var_249.astype('float64'), [1,]), relay.reshape(var_247.astype('float64'), [24,]), relay.reshape(var_247.astype('float32'), [24,]), ), 1)
output = relay.Tuple([bop_243,call_246,var_247,var_248,var_249,])
output2 = relay.Tuple([bop_243,call_250,var_247,var_248,var_249,])
func_251 = relay.Function([var_241,var_242,var_247,var_248,var_249,], output)
mod['func_251'] = func_251
mod = relay.transform.InferType()(mod)
mutated_mod['func_251'] = func_251
mutated_mod = relay.transform.InferType()(mutated_mod)
func_251_call = mutated_mod.get_global_var('func_251')
var_253 = relay.var("var_253", dtype = "float32", shape = (6, 6, 1))#candidate|253|(6, 6, 1)|var|float32
var_254 = relay.var("var_254", dtype = "float32", shape = (6, 6, 16))#candidate|254|(6, 6, 16)|var|float32
var_255 = relay.var("var_255", dtype = "uint8", shape = (24, 1))#candidate|255|(24, 1)|var|uint8
var_256 = relay.var("var_256", dtype = "uint8", shape = (3, 40))#candidate|256|(3, 40)|var|uint8
var_257 = relay.var("var_257", dtype = "float64", shape = (1,))#candidate|257|(1,)|var|float64
call_252 = func_251_call(var_253,var_254,var_255,var_256,var_257,)
output = call_252
func_258 = relay.Function([var_253,var_254,var_255,var_256,var_257,], output)
mutated_mod['func_258'] = func_258
mutated_mod = relay.transform.InferType()(mutated_mod)
const_260 = relay.const(-10, dtype = "uint32")#candidate|260|()|const|uint32
var_261 = relay.var("var_261", dtype = "uint32", shape = (10,))#candidate|261|(10,)|var|uint32
bop_262 = relay.subtract(const_260.astype('uint32'), var_261.astype('uint32')) # shape=(10,)
uop_265 = relay.log(const_260.astype('float32')) # shape=()
uop_267 = relay.sin(var_261.astype('float64')) # shape=(10,)
uop_269 = relay.sigmoid(var_261.astype('float32')) # shape=(10,)
uop_271 = relay.asinh(const_260.astype('float64')) # shape=()
output = relay.Tuple([bop_262,uop_265,uop_267,uop_269,uop_271,])
output2 = relay.Tuple([bop_262,uop_265,uop_267,uop_269,uop_271,])
func_273 = relay.Function([var_261,], output)
mod['func_273'] = func_273
mod = relay.transform.InferType()(mod)
var_274 = relay.var("var_274", dtype = "uint32", shape = (10,))#candidate|274|(10,)|var|uint32
output = func_273(var_274)
func_275 = relay.Function([var_274], output)
mutated_mod['func_275'] = func_275
mutated_mod = relay.transform.InferType()(mutated_mod)
const_277 = relay.const([[[-6,1,-5,-9,8,-1],[-5,-7,3,-1,6,-6]],[[-9,9,7,-2,-8,-5],[3,-4,-10,-10,-1,-6]],[[-7,-5,-5,1,-3,2],[4,4,-6,-4,5,-9]],[[-1,9,-8,-10,-6,-7],[-7,2,1,-7,7,-8]],[[2,-1,-10,-1,-9,-6],[-9,-3,-1,-5,1,8]],[[3,-8,9,-3,-7,-8],[-5,1,3,-5,4,-7]],[[-6,-5,2,-8,9,7],[-1,-8,7,7,-7,7]],[[6,2,10,6,4,3],[-5,2,9,1,-10,-8]],[[7,8,-6,5,9,-3],[-9,-4,-10,-10,2,-8]],[[-3,-10,-10,-1,-6,2],[-6,-5,6,-5,-4,-9]],[[-8,9,3,10,6,1],[10,4,1,-9,3,-3]],[[9,4,-8,10,-4,-7],[-4,-2,-7,-6,-5,3]],[[6,-4,-4,-10,6,10],[10,2,-7,-10,-4,-3]]], dtype = "int16")#candidate|277|(13, 2, 6)|const|int16
const_278 = relay.const([[[1,-1,9,-9,2,5],[-2,-6,10,-7,-4,6]],[[-1,-6,4,-1,4,-7],[-2,3,8,4,9,-8]],[[-3,-7,7,-7,-4,3],[1,-5,8,7,-10,3]],[[-10,-9,-9,9,-2,-9],[4,7,-10,-5,4,7]],[[-2,3,-7,-10,-5,10],[-6,8,-9,-8,4,8]],[[8,7,5,-8,-8,-9],[-7,-6,9,-10,6,6]],[[-4,6,-2,2,-2,-3],[-2,-4,-2,-7,-3,3]],[[10,-9,-3,10,9,-5],[-2,2,7,1,-8,2]],[[-7,9,4,3,9,-10],[7,8,-8,9,-4,-6]],[[-8,-2,1,6,-7,-1],[9,4,-4,-8,4,6]],[[-3,-4,8,9,6,-10],[-9,8,-1,6,-10,10]],[[5,-6,-6,-3,-9,-7],[4,3,-1,-10,9,-8]],[[-6,4,5,2,2,2],[-2,8,-9,-8,-4,6]]], dtype = "int16")#candidate|278|(13, 2, 6)|const|int16
bop_279 = relay.minimum(const_277.astype('int16'), relay.reshape(const_278.astype('int16'), relay.shape_of(const_277))) # shape=(13, 2, 6)
uop_282 = relay.erf(const_278.astype('float32')) # shape=(13, 2, 6)
bop_284 = relay.multiply(uop_282.astype('float32'), relay.reshape(const_278.astype('float32'), relay.shape_of(uop_282))) # shape=(13, 2, 6)
bop_287 = relay.bitwise_xor(bop_279.astype('uint64'), relay.reshape(bop_284.astype('uint64'), relay.shape_of(bop_279))) # shape=(13, 2, 6)
uop_290 = relay.exp(bop_287.astype('float32')) # shape=(13, 2, 6)
bop_292 = relay.equal(uop_282.astype('bool'), relay.reshape(const_278.astype('bool'), relay.shape_of(uop_282))) # shape=(13, 2, 6)
uop_295 = relay.exp(uop_290.astype('float64')) # shape=(13, 2, 6)
uop_297 = relay.sin(uop_290.astype('float64')) # shape=(13, 2, 6)
bop_299 = relay.floor_divide(bop_292.astype('float32'), relay.reshape(uop_297.astype('float32'), relay.shape_of(bop_292))) # shape=(13, 2, 6)
bop_302 = relay.left_shift(bop_292.astype('uint8'), relay.reshape(const_278.astype('uint8'), relay.shape_of(bop_292))) # shape=(13, 2, 6)
output = relay.Tuple([uop_295,bop_299,bop_302,])
output2 = relay.Tuple([uop_295,bop_299,bop_302,])
func_305 = relay.Function([], output)
mod['func_305'] = func_305
mod = relay.transform.InferType()(mod)
output = func_305()
func_306 = relay.Function([], output)
mutated_mod['func_306'] = func_306
mutated_mod = relay.transform.InferType()(mutated_mod)
var_307 = relay.var("var_307", dtype = "int8", shape = ())#candidate|307|()|var|int8
var_308 = relay.var("var_308", dtype = "int8", shape = (14, 6))#candidate|308|(14, 6)|var|int8
bop_309 = relay.less(var_307.astype('bool'), var_308.astype('bool')) # shape=(14, 6)
func_11_call = mod.get_global_var('func_11')
func_15_call = mutated_mod.get_global_var('func_15')
var_313 = relay.var("var_313", dtype = "uint8", shape = (24,))#candidate|313|(24,)|var|uint8
const_314 = relay.const([-7,-4,8,4,6,-2,-4,-4,-2,10,3,-8,-3,-9,5,5,-3,-6,10,10,-1,-3,-3,5,1,7,-1,-2,-8,-9,3,4,3,-6,-9,9,1,4,10,-1,7,9,-5,-4,2,-3,-9,10,6,-3,2,-4,5,-7,9,9,8,-8,7,-3,7,-5,-4,6,-10,-6,4,-7,7,-5,-2,4,10,-2,-4,-7,10,7,8,-5,-6,-3,10,-9,-4,5,7,-7,-1,-4,-1,-6,-5,-8,-10,-1,3,-7,4,9,-10,-3,4,-6,8,-10,2,-1,-2,-10,-1,9,7,-6,-5,8,-1,-4,10,10], dtype = "uint8")#candidate|314|(120,)|const|uint8
call_312 = relay.TupleGetItem(func_11_call(relay.reshape(var_313.astype('uint8'), [8, 3, 1]), relay.reshape(const_314.astype('uint8'), [8, 3, 5]), relay.reshape(const_314.astype('uint8'), [8, 3, 5]), ), 1)
call_315 = relay.TupleGetItem(func_15_call(relay.reshape(var_313.astype('uint8'), [8, 3, 1]), relay.reshape(const_314.astype('uint8'), [8, 3, 5]), relay.reshape(const_314.astype('uint8'), [8, 3, 5]), ), 1)
uop_316 = relay.acosh(const_314.astype('float32')) # shape=(120,)
var_318 = relay.var("var_318", dtype = "uint8", shape = (120,))#candidate|318|(120,)|var|uint8
bop_319 = relay.subtract(const_314.astype('float32'), relay.reshape(var_318.astype('float32'), relay.shape_of(const_314))) # shape=(120,)
uop_322 = relay.log2(uop_316.astype('float64')) # shape=(120,)
uop_324 = relay.acosh(uop_322.astype('float32')) # shape=(120,)
uop_326 = relay.sin(uop_324.astype('float32')) # shape=(120,)
uop_328 = relay.rsqrt(uop_324.astype('float32')) # shape=(120,)
bop_330 = relay.divide(uop_326.astype('float32'), relay.reshape(call_312.astype('float32'), relay.shape_of(uop_326))) # shape=(120,)
bop_333 = relay.divide(uop_326.astype('float32'), relay.reshape(call_315.astype('float32'), relay.shape_of(uop_326))) # shape=(120,)
uop_334 = relay.cosh(uop_326.astype('float32')) # shape=(120,)
uop_336 = relay.sinh(uop_326.astype('float32')) # shape=(120,)
output = relay.Tuple([bop_309,var_313,bop_319,uop_328,bop_330,uop_334,uop_336,])
output2 = relay.Tuple([bop_309,var_313,bop_319,uop_328,bop_333,uop_334,uop_336,])
func_338 = relay.Function([var_307,var_308,var_313,var_318,], output)
mod['func_338'] = func_338
mod = relay.transform.InferType()(mod)
var_339 = relay.var("var_339", dtype = "int8", shape = ())#candidate|339|()|var|int8
var_340 = relay.var("var_340", dtype = "int8", shape = (14, 6))#candidate|340|(14, 6)|var|int8
var_341 = relay.var("var_341", dtype = "uint8", shape = (24,))#candidate|341|(24,)|var|uint8
var_342 = relay.var("var_342", dtype = "uint8", shape = (120,))#candidate|342|(120,)|var|uint8
output = func_338(var_339,var_340,var_341,var_342,)
func_343 = relay.Function([var_339,var_340,var_341,var_342,], output)
mutated_mod['func_343'] = func_343
mutated_mod = relay.transform.InferType()(mutated_mod)
var_345 = relay.var("var_345", dtype = "float64", shape = ())#candidate|345|()|var|float64
uop_346 = relay.sqrt(var_345.astype('float64')) # shape=()
uop_348 = relay.log(var_345.astype('float32')) # shape=()
bop_350 = relay.divide(uop_346.astype('float32'), uop_348.astype('float32')) # shape=()
uop_353 = relay.sinh(uop_346.astype('float64')) # shape=()
bop_355 = relay.multiply(uop_353.astype('uint64'), var_345.astype('uint64')) # shape=()
output = relay.Tuple([bop_350,bop_355,])
output2 = relay.Tuple([bop_350,bop_355,])
F = relay.Function([var_345,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_345,], output2)
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
input_345= np.array(8.871134, dtype='float64')
module1.set_input('var_345', input_345)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_345, )
res3 = intrp3.evaluate()(input_345, )
res4 = intrp4.evaluate()(input_345, )
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
module5.set_input('var_345', input_345)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_345, )
res7 = intrp7.evaluate()(input_345, )
res8 = intrp8.evaluate()(input_345, )
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
module9.set_input('var_345', input_345)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_345, )
res11 = intrp11.evaluate()(input_345, )
res12 = intrp12.evaluate()(input_345, )
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
module13.set_input('var_345', input_345)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_345, )
res15 = intrp15.evaluate()(input_345, )
res16 = intrp16.evaluate()(input_345, )
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
module17.set_input('var_345', input_345)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_345, )
res19 = intrp19.evaluate()(input_345, )
res20 = intrp20.evaluate()(input_345, )
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
module21.set_input('var_345', input_345)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_345, )
res23 = intrp23.evaluate()(input_345, )
res24 = intrp24.evaluate()(input_345, )
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