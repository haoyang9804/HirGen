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
var_0 = relay.var("var_0", dtype = "bool", shape = (9, 1))#candidate|0|(9, 1)|var|bool
var_1 = relay.var("var_1", dtype = "bool", shape = (9, 1))#candidate|1|(9, 1)|var|bool
bop_2 = relay.logical_or(var_0.astype('bool'), relay.reshape(var_1.astype('bool'), relay.shape_of(var_0))) # shape=(9, 1)
bop_5 = relay.logical_and(var_0.astype('bool'), relay.reshape(bop_2.astype('bool'), relay.shape_of(var_0))) # shape=(9, 1)
bop_8 = relay.floor_divide(var_0.astype('float32'), relay.reshape(bop_2.astype('float32'), relay.shape_of(var_0))) # shape=(9, 1)
uop_11 = relay.asinh(var_0.astype('float64')) # shape=(9, 1)
bop_13 = relay.left_shift(uop_11.astype('uint32'), relay.reshape(var_1.astype('uint32'), relay.shape_of(uop_11))) # shape=(9, 1)
uop_16 = relay.erf(uop_11.astype('float64')) # shape=(9, 1)
uop_18 = relay.atan(uop_16.astype('float32')) # shape=(9, 1)
bop_20 = relay.bitwise_or(uop_18.astype('int8'), relay.reshape(bop_5.astype('int8'), relay.shape_of(uop_18))) # shape=(9, 1)
bop_23 = relay.floor_mod(bop_13.astype('float32'), relay.reshape(bop_5.astype('float32'), relay.shape_of(bop_13))) # shape=(9, 1)
bop_26 = relay.subtract(uop_16.astype('uint16'), relay.reshape(bop_8.astype('uint16'), relay.shape_of(uop_16))) # shape=(9, 1)
uop_29 = relay.log(bop_20.astype('float32')) # shape=(9, 1)
var_31 = relay.var("var_31", dtype = "float32", shape = (9, 16))#candidate|31|(9, 16)|var|float32
bop_32 = relay.add(uop_29.astype('int32'), var_31.astype('int32')) # shape=(9, 16)
uop_35 = relay.sinh(uop_29.astype('float32')) # shape=(9, 1)
output = relay.Tuple([bop_23,bop_26,bop_32,uop_35,])
output2 = relay.Tuple([bop_23,bop_26,bop_32,uop_35,])
func_37 = relay.Function([var_0,var_1,var_31,], output)
mod['func_37'] = func_37
mod = relay.transform.InferType()(mod)
mutated_mod['func_37'] = func_37
mutated_mod = relay.transform.InferType()(mutated_mod)
func_37_call = mutated_mod.get_global_var('func_37')
var_39 = relay.var("var_39", dtype = "bool", shape = (9, 1))#candidate|39|(9, 1)|var|bool
var_40 = relay.var("var_40", dtype = "bool", shape = (9, 1))#candidate|40|(9, 1)|var|bool
var_41 = relay.var("var_41", dtype = "float32", shape = (9, 16))#candidate|41|(9, 16)|var|float32
call_38 = func_37_call(var_39,var_40,var_41,)
output = call_38
func_42 = relay.Function([var_39,var_40,var_41,], output)
mutated_mod['func_42'] = func_42
mutated_mod = relay.transform.InferType()(mutated_mod)
const_44 = relay.const([[[6,-1],[3,-9],[2,9],[-5,6],[6,-6],[-7,8],[7,5],[-2,9]]], dtype = "uint8")#candidate|44|(1, 8, 2)|const|uint8
var_45 = relay.var("var_45", dtype = "uint8", shape = (15, 8, 2))#candidate|45|(15, 8, 2)|var|uint8
bop_46 = relay.minimum(const_44.astype('uint8'), var_45.astype('uint8')) # shape=(15, 8, 2)
uop_49 = relay.sinh(bop_46.astype('float32')) # shape=(15, 8, 2)
uop_51 = relay.log(uop_49.astype('float32')) # shape=(15, 8, 2)
bop_53 = relay.floor_divide(uop_51.astype('float32'), relay.reshape(uop_49.astype('float32'), relay.shape_of(uop_51))) # shape=(15, 8, 2)
uop_56 = relay.sin(bop_53.astype('float32')) # shape=(15, 8, 2)
output = uop_56
output2 = uop_56
func_58 = relay.Function([var_45,], output)
mod['func_58'] = func_58
mod = relay.transform.InferType()(mod)
var_59 = relay.var("var_59", dtype = "uint8", shape = (15, 8, 2))#candidate|59|(15, 8, 2)|var|uint8
output = func_58(var_59)
func_60 = relay.Function([var_59], output)
mutated_mod['func_60'] = func_60
mutated_mod = relay.transform.InferType()(mutated_mod)
var_62 = relay.var("var_62", dtype = "float32", shape = (11, 6, 4))#candidate|62|(11, 6, 4)|var|float32
uop_63 = relay.acos(var_62.astype('float32')) # shape=(11, 6, 4)
uop_65 = relay.sigmoid(uop_63.astype('float32')) # shape=(11, 6, 4)
bop_67 = relay.divide(uop_65.astype('float32'), relay.reshape(uop_63.astype('float32'), relay.shape_of(uop_65))) # shape=(11, 6, 4)
bop_70 = relay.equal(bop_67.astype('bool'), relay.reshape(var_62.astype('bool'), relay.shape_of(bop_67))) # shape=(11, 6, 4)
uop_73 = relay.atanh(uop_65.astype('float32')) # shape=(11, 6, 4)
uop_75 = relay.tan(uop_73.astype('float32')) # shape=(11, 6, 4)
uop_77 = relay.cosh(uop_75.astype('float32')) # shape=(11, 6, 4)
bop_79 = relay.logical_or(uop_77.astype('bool'), relay.reshape(uop_73.astype('bool'), relay.shape_of(uop_77))) # shape=(11, 6, 4)
uop_82 = relay.cosh(uop_73.astype('float64')) # shape=(11, 6, 4)
bop_84 = relay.bitwise_or(uop_73.astype('int64'), relay.reshape(uop_63.astype('int64'), relay.shape_of(uop_73))) # shape=(11, 6, 4)
bop_87 = relay.logical_and(uop_77.astype('bool'), relay.reshape(bop_84.astype('bool'), relay.shape_of(uop_77))) # shape=(11, 6, 4)
var_90 = relay.var("var_90", dtype = "float32", shape = (11, 6, 4))#candidate|90|(11, 6, 4)|var|float32
bop_91 = relay.subtract(uop_75.astype('uint64'), relay.reshape(var_90.astype('uint64'), relay.shape_of(uop_75))) # shape=(11, 6, 4)
output = relay.Tuple([bop_70,bop_79,uop_82,bop_87,bop_91,])
output2 = relay.Tuple([bop_70,bop_79,uop_82,bop_87,bop_91,])
func_94 = relay.Function([var_62,var_90,], output)
mod['func_94'] = func_94
mod = relay.transform.InferType()(mod)
mutated_mod['func_94'] = func_94
mutated_mod = relay.transform.InferType()(mutated_mod)
func_94_call = mutated_mod.get_global_var('func_94')
var_96 = relay.var("var_96", dtype = "float32", shape = (11, 6, 4))#candidate|96|(11, 6, 4)|var|float32
var_97 = relay.var("var_97", dtype = "float32", shape = (11, 6, 4))#candidate|97|(11, 6, 4)|var|float32
call_95 = func_94_call(var_96,var_97,)
output = call_95
func_98 = relay.Function([var_96,var_97,], output)
mutated_mod['func_98'] = func_98
mutated_mod = relay.transform.InferType()(mutated_mod)
var_100 = relay.var("var_100", dtype = "int32", shape = (16, 5, 9))#candidate|100|(16, 5, 9)|var|int32
var_101 = relay.var("var_101", dtype = "int32", shape = (16, 5, 9))#candidate|101|(16, 5, 9)|var|int32
bop_102 = relay.less_equal(var_100.astype('bool'), relay.reshape(var_101.astype('bool'), relay.shape_of(var_100))) # shape=(16, 5, 9)
bop_105 = relay.logical_and(var_100.astype('bool'), relay.reshape(var_101.astype('bool'), relay.shape_of(var_100))) # shape=(16, 5, 9)
output = relay.Tuple([bop_102,bop_105,])
output2 = relay.Tuple([bop_102,bop_105,])
func_108 = relay.Function([var_100,var_101,], output)
mod['func_108'] = func_108
mod = relay.transform.InferType()(mod)
mutated_mod['func_108'] = func_108
mutated_mod = relay.transform.InferType()(mutated_mod)
func_108_call = mutated_mod.get_global_var('func_108')
var_110 = relay.var("var_110", dtype = "int32", shape = (16, 5, 9))#candidate|110|(16, 5, 9)|var|int32
var_111 = relay.var("var_111", dtype = "int32", shape = (16, 5, 9))#candidate|111|(16, 5, 9)|var|int32
call_109 = func_108_call(var_110,var_111,)
output = call_109
func_112 = relay.Function([var_110,var_111,], output)
mutated_mod['func_112'] = func_112
mutated_mod = relay.transform.InferType()(mutated_mod)
var_114 = relay.var("var_114", dtype = "float32", shape = (15,))#candidate|114|(15,)|var|float32
var_115 = relay.var("var_115", dtype = "float32", shape = (15,))#candidate|115|(15,)|var|float32
bop_116 = relay.mod(var_114.astype('float32'), relay.reshape(var_115.astype('float32'), relay.shape_of(var_114))) # shape=(15,)
const_119 = relay.const([-2.525291,-7.170109,-4.472436,-4.219524,3.897365,6.953248,-8.434766,-9.058053,4.773385,9.598718,-9.465702,9.535519,-7.890485,-0.207923,0.445607], dtype = "float32")#candidate|119|(15,)|const|float32
bop_120 = relay.bitwise_and(var_115.astype('int64'), relay.reshape(const_119.astype('int64'), relay.shape_of(var_115))) # shape=(15,)
uop_123 = relay.log(const_119.astype('float32')) # shape=(15,)
uop_125 = relay.atan(var_114.astype('float64')) # shape=(15,)
uop_127 = relay.sinh(uop_123.astype('float64')) # shape=(15,)
bop_129 = relay.greater(uop_127.astype('bool'), relay.reshape(uop_123.astype('bool'), relay.shape_of(uop_127))) # shape=(15,)
uop_132 = relay.atanh(uop_125.astype('float64')) # shape=(15,)
uop_134 = relay.sqrt(bop_129.astype('float64')) # shape=(15,)
var_136 = relay.var("var_136", dtype = "float64", shape = (15,))#candidate|136|(15,)|var|float64
bop_137 = relay.logical_or(uop_132.astype('bool'), relay.reshape(var_136.astype('bool'), relay.shape_of(uop_132))) # shape=(15,)
uop_140 = relay.cosh(uop_134.astype('float32')) # shape=(15,)
var_142 = relay.var("var_142", dtype = "float32", shape = (15,))#candidate|142|(15,)|var|float32
bop_143 = relay.floor_divide(uop_140.astype('float32'), relay.reshape(var_142.astype('float32'), relay.shape_of(uop_140))) # shape=(15,)
var_146 = relay.var("var_146", dtype = "float64", shape = (15,))#candidate|146|(15,)|var|float64
bop_147 = relay.bitwise_and(uop_134.astype('int64'), relay.reshape(var_146.astype('int64'), relay.shape_of(uop_134))) # shape=(15,)
var_150 = relay.var("var_150", dtype = "int64", shape = (15,))#candidate|150|(15,)|var|int64
bop_151 = relay.power(bop_147.astype('float64'), relay.reshape(var_150.astype('float64'), relay.shape_of(bop_147))) # shape=(15,)
uop_154 = relay.cosh(uop_125.astype('float64')) # shape=(15,)
bop_156 = relay.less_equal(uop_140.astype('bool'), relay.reshape(var_142.astype('bool'), relay.shape_of(uop_140))) # shape=(15,)
bop_159 = relay.floor_divide(bop_147.astype('float32'), relay.reshape(bop_151.astype('float32'), relay.shape_of(bop_147))) # shape=(15,)
uop_162 = relay.sigmoid(uop_127.astype('float64')) # shape=(15,)
output = relay.Tuple([bop_116,bop_120,bop_137,bop_143,uop_154,bop_156,bop_159,uop_162,])
output2 = relay.Tuple([bop_116,bop_120,bop_137,bop_143,uop_154,bop_156,bop_159,uop_162,])
func_164 = relay.Function([var_114,var_115,var_136,var_142,var_146,var_150,], output)
mod['func_164'] = func_164
mod = relay.transform.InferType()(mod)
mutated_mod['func_164'] = func_164
mutated_mod = relay.transform.InferType()(mutated_mod)
func_164_call = mutated_mod.get_global_var('func_164')
var_166 = relay.var("var_166", dtype = "float32", shape = (15,))#candidate|166|(15,)|var|float32
var_167 = relay.var("var_167", dtype = "float32", shape = (15,))#candidate|167|(15,)|var|float32
var_168 = relay.var("var_168", dtype = "float64", shape = (15,))#candidate|168|(15,)|var|float64
var_169 = relay.var("var_169", dtype = "float32", shape = (15,))#candidate|169|(15,)|var|float32
var_170 = relay.var("var_170", dtype = "float64", shape = (15,))#candidate|170|(15,)|var|float64
var_171 = relay.var("var_171", dtype = "int64", shape = (15,))#candidate|171|(15,)|var|int64
call_165 = func_164_call(var_166,var_167,var_168,var_169,var_170,var_171,)
output = call_165
func_172 = relay.Function([var_166,var_167,var_168,var_169,var_170,var_171,], output)
mutated_mod['func_172'] = func_172
mutated_mod = relay.transform.InferType()(mutated_mod)
var_174 = relay.var("var_174", dtype = "uint8", shape = (16, 8))#candidate|174|(16, 8)|var|uint8
var_175 = relay.var("var_175", dtype = "uint8", shape = (16, 8))#candidate|175|(16, 8)|var|uint8
bop_176 = relay.equal(var_174.astype('bool'), relay.reshape(var_175.astype('bool'), relay.shape_of(var_174))) # shape=(16, 8)
uop_179 = relay.cos(bop_176.astype('float64')) # shape=(16, 8)
var_181 = relay.var("var_181", dtype = "uint8", shape = (16, 8))#candidate|181|(16, 8)|var|uint8
bop_182 = relay.logical_and(var_174.astype('bool'), relay.reshape(var_181.astype('bool'), relay.shape_of(var_174))) # shape=(16, 8)
uop_185 = relay.sqrt(uop_179.astype('float64')) # shape=(16, 8)
uop_187 = relay.acosh(uop_179.astype('float32')) # shape=(16, 8)
bop_189 = relay.floor_mod(uop_187.astype('float32'), relay.reshape(uop_185.astype('float32'), relay.shape_of(uop_187))) # shape=(16, 8)
uop_192 = relay.cosh(uop_179.astype('float32')) # shape=(16, 8)
var_194 = relay.var("var_194", dtype = "float32", shape = (16, 8))#candidate|194|(16, 8)|var|float32
bop_195 = relay.bitwise_and(uop_187.astype('uint16'), relay.reshape(var_194.astype('uint16'), relay.shape_of(uop_187))) # shape=(16, 8)
bop_198 = relay.floor_mod(bop_189.astype('float64'), relay.reshape(uop_185.astype('float64'), relay.shape_of(bop_189))) # shape=(16, 8)
func_164_call = mod.get_global_var('func_164')
func_172_call = mutated_mod.get_global_var('func_172')
const_202 = relay.const([-4.069024,-5.119782,-6.113243,-8.255911,-3.992515,0.188253,-9.979024,7.290586,-1.080696,1.948615,-0.814896,8.895846,-1.807524,1.405948,1.469348], dtype = "float32")#candidate|202|(15,)|const|float32
call_201 = relay.TupleGetItem(func_164_call(relay.reshape(const_202.astype('float32'), [15,]), relay.reshape(const_202.astype('float32'), [15,]), relay.reshape(const_202.astype('float64'), [15,]), relay.reshape(const_202.astype('float32'), [15,]), relay.reshape(const_202.astype('float64'), [15,]), relay.reshape(const_202.astype('int64'), [15,]), ), 7)
call_203 = relay.TupleGetItem(func_172_call(relay.reshape(const_202.astype('float32'), [15,]), relay.reshape(const_202.astype('float32'), [15,]), relay.reshape(const_202.astype('float64'), [15,]), relay.reshape(const_202.astype('float32'), [15,]), relay.reshape(const_202.astype('float64'), [15,]), relay.reshape(const_202.astype('int64'), [15,]), ), 7)
bop_204 = relay.bitwise_or(uop_179.astype('int64'), relay.reshape(var_175.astype('int64'), relay.shape_of(uop_179))) # shape=(16, 8)
bop_207 = relay.logical_or(bop_189.astype('bool'), relay.reshape(bop_198.astype('bool'), relay.shape_of(bop_189))) # shape=(16, 8)
var_210 = relay.var("var_210", dtype = "float32", shape = (16, 8))#candidate|210|(16, 8)|var|float32
bop_211 = relay.floor_mod(uop_187.astype('float64'), relay.reshape(var_210.astype('float64'), relay.shape_of(uop_187))) # shape=(16, 8)
bop_214 = relay.greater_equal(uop_185.astype('bool'), relay.reshape(bop_176.astype('bool'), relay.shape_of(uop_185))) # shape=(16, 8)
bop_217 = relay.equal(uop_185.astype('bool'), relay.reshape(bop_214.astype('bool'), relay.shape_of(uop_185))) # shape=(16, 8)
uop_220 = relay.sinh(bop_195.astype('float32')) # shape=(16, 8)
uop_222 = relay.log2(bop_189.astype('float32')) # shape=(16, 8)
uop_224 = relay.exp(bop_195.astype('float32')) # shape=(16, 8)
uop_226 = relay.log10(uop_222.astype('float64')) # shape=(16, 8)
bop_228 = relay.bitwise_xor(uop_226.astype('uint16'), relay.reshape(uop_187.astype('uint16'), relay.shape_of(uop_226))) # shape=(16, 8)
var_231 = relay.var("var_231", dtype = "float32", shape = (16, 8))#candidate|231|(16, 8)|var|float32
bop_232 = relay.bitwise_or(uop_224.astype('uint32'), relay.reshape(var_231.astype('uint32'), relay.shape_of(uop_224))) # shape=(16, 8)
uop_235 = relay.sigmoid(bop_228.astype('float64')) # shape=(16, 8)
bop_237 = relay.bitwise_xor(uop_235.astype('uint16'), relay.reshape(bop_214.astype('uint16'), relay.shape_of(uop_235))) # shape=(16, 8)
var_240 = relay.var("var_240", dtype = "float64", shape = (16, 8))#candidate|240|(16, 8)|var|float64
bop_241 = relay.maximum(uop_235.astype('uint16'), relay.reshape(var_240.astype('uint16'), relay.shape_of(uop_235))) # shape=(16, 8)
func_37_call = mod.get_global_var('func_37')
func_42_call = mutated_mod.get_global_var('func_42')
const_245 = relay.const([False,False,False,False,False,False,True,True,True], dtype = "bool")#candidate|245|(9,)|const|bool
var_246 = relay.var("var_246", dtype = "float32", shape = (144,))#candidate|246|(144,)|var|float32
call_244 = relay.TupleGetItem(func_37_call(relay.reshape(const_245.astype('bool'), [9, 1]), relay.reshape(const_245.astype('bool'), [9, 1]), relay.reshape(var_246.astype('float32'), [9, 16]), ), 1)
call_247 = relay.TupleGetItem(func_42_call(relay.reshape(const_245.astype('bool'), [9, 1]), relay.reshape(const_245.astype('bool'), [9, 1]), relay.reshape(var_246.astype('float32'), [9, 16]), ), 1)
uop_248 = relay.atanh(bop_232.astype('float64')) # shape=(16, 8)
var_250 = relay.var("var_250", dtype = "uint16", shape = (16, 8))#candidate|250|(16, 8)|var|uint16
bop_251 = relay.logical_or(bop_241.astype('bool'), relay.reshape(var_250.astype('bool'), relay.shape_of(bop_241))) # shape=(16, 8)
bop_254 = relay.logical_and(bop_198.astype('bool'), relay.reshape(bop_241.astype('bool'), relay.shape_of(bop_198))) # shape=(16, 8)
bop_257 = relay.logical_xor(bop_237.astype('int16'), relay.reshape(uop_187.astype('int16'), relay.shape_of(bop_237))) # shape=(16, 8)
var_260 = relay.var("var_260", dtype = "float64", shape = (16, 8))#candidate|260|(16, 8)|var|float64
bop_261 = relay.minimum(uop_226.astype('int32'), relay.reshape(var_260.astype('int32'), relay.shape_of(uop_226))) # shape=(16, 8)
func_37_call = mod.get_global_var('func_37')
func_42_call = mutated_mod.get_global_var('func_42')
call_264 = relay.TupleGetItem(func_37_call(relay.reshape(call_244.astype('bool'), [9, 1]), relay.reshape(call_244.astype('bool'), [9, 1]), relay.reshape(var_246.astype('float32'), [9, 16]), ), 3)
call_265 = relay.TupleGetItem(func_42_call(relay.reshape(call_244.astype('bool'), [9, 1]), relay.reshape(call_244.astype('bool'), [9, 1]), relay.reshape(var_246.astype('float32'), [9, 16]), ), 3)
var_266 = relay.var("var_266", dtype = "uint16", shape = (16, 8))#candidate|266|(16, 8)|var|uint16
bop_267 = relay.bitwise_and(bop_237.astype('int32'), relay.reshape(var_266.astype('int32'), relay.shape_of(bop_237))) # shape=(16, 8)
bop_270 = relay.floor_divide(uop_222.astype('float64'), relay.reshape(bop_189.astype('float64'), relay.shape_of(uop_222))) # shape=(16, 8)
func_94_call = mod.get_global_var('func_94')
func_98_call = mutated_mod.get_global_var('func_98')
var_274 = relay.var("var_274", dtype = "float32", shape = (264,))#candidate|274|(264,)|var|float32
call_273 = relay.TupleGetItem(func_94_call(relay.reshape(var_274.astype('float32'), [11, 6, 4]), relay.reshape(var_274.astype('float32'), [11, 6, 4]), ), 1)
call_275 = relay.TupleGetItem(func_98_call(relay.reshape(var_274.astype('float32'), [11, 6, 4]), relay.reshape(var_274.astype('float32'), [11, 6, 4]), ), 1)
bop_276 = relay.right_shift(bop_267.astype('uint32'), relay.reshape(bop_214.astype('uint32'), relay.shape_of(bop_267))) # shape=(16, 8)
uop_279 = relay.sin(bop_276.astype('float64')) # shape=(16, 8)
uop_281 = relay.log10(bop_267.astype('float32')) # shape=(16, 8)
bop_283 = relay.bitwise_or(bop_228.astype('int16'), relay.reshape(bop_270.astype('int16'), relay.shape_of(bop_228))) # shape=(16, 8)
bop_286 = relay.power(uop_279.astype('float32'), relay.reshape(bop_237.astype('float32'), relay.shape_of(uop_279))) # shape=(16, 8)
bop_289 = relay.subtract(bop_283.astype('float64'), relay.reshape(var_250.astype('float64'), relay.shape_of(bop_283))) # shape=(16, 8)
uop_292 = relay.sin(bop_251.astype('float64')) # shape=(16, 8)
uop_294 = relay.acosh(uop_235.astype('float32')) # shape=(16, 8)
func_58_call = mod.get_global_var('func_58')
func_60_call = mutated_mod.get_global_var('func_60')
const_297 = relay.const([-8,-7,-8,2,-7,-8,10,-4,9,8,5,-6,6,-4,-3,-8,-3,-4,10,-5,-5,-7,4,-9,-3,9,-9,-8,10,8,6,6,3,-2,-8,-1,-2,7,2,-7,5,3,-10,-10,3,-8,8,-10,-4,3,1,-6,-2,7,10,-9,1,-10,-4,-3,-10,6,5,-10,-8,2,9,-2,-2,7,4,-7,-9,-6,-7,8,10,-7,9,5,-10,-7,-6,2,-8,9,4,-7,-1,3,-3,-7,-4,9,2,1,4,3,-6,9,-3,-8,-4,-9,9,5,3,-9,8,2,-4,6,-10,-10,-5,2,7,4,-4,-10,-4,-7,9,-8,5,-7,-3,-9,-3,1,7,-3,-3,10,9,-2,-5,2,6,10,-9,2,-1,9,9,-9,3,-5,-2,-6,4,-6,9,-4,-9,6,-3,-2,-9,-8,10,7,3,2,6,8,7,1,-3,1,-9,-2,7,-7,-6,5,4,1,-5,-5,-2,2,-7,7,10,-2,7,9,9,-2,-3,-10,10,-6,-6,-7,2,-2,-4,8,7,3,5,1,10,-8,-10,-1,4,5,-8,10,4,10,-1,4,-8,5,10,9,9,6,8,2,-5,-1,1,-5,-5,5,-3,7,7,-7,8,-1,-3,-8,4,-2], dtype = "uint8")#candidate|297|(240,)|const|uint8
call_296 = func_58_call(relay.reshape(const_297.astype('uint8'), [15, 8, 2]))
call_298 = func_58_call(relay.reshape(const_297.astype('uint8'), [15, 8, 2]))
uop_299 = relay.sqrt(uop_279.astype('float64')) # shape=(16, 8)
uop_301 = relay.sqrt(uop_299.astype('float32')) # shape=(16, 8)
uop_303 = relay.acosh(uop_299.astype('float64')) # shape=(16, 8)
var_305 = relay.var("var_305", dtype = "float64", shape = (16, 8))#candidate|305|(16, 8)|var|float64
bop_306 = relay.add(uop_303.astype('float32'), relay.reshape(var_305.astype('float32'), relay.shape_of(uop_303))) # shape=(16, 8)
const_309 = relay.const([[2.593819,7.618228,-3.696850,-8.420377,9.529878,-7.452026,-8.487611,6.031119],[-7.265502,6.260059,-2.865525,-8.228111,-4.038535,-0.475137,-2.150479,-2.864835],[-2.737051,6.549446,7.368611,8.815657,1.699759,-1.952699,-5.847067,3.120189],[9.467153,-5.665495,-1.490233,5.109381,6.666614,-1.458202,5.257605,3.252983],[-7.868730,-5.037437,-4.747100,7.291686,9.490643,6.645522,8.710087,-6.908357],[2.002056,-2.296878,7.932110,-6.186866,3.718239,3.884511,-3.913232,-1.063366],[7.347998,-2.289424,-4.852655,3.497787,-7.156449,4.365185,7.219832,-2.276653],[5.396289,-8.260548,-6.208309,8.655078,-6.329912,-2.248815,1.352272,8.979645],[-5.722448,-4.349253,0.903186,-1.916491,5.112148,-2.553460,-5.910457,3.360210],[6.190884,-2.461338,1.710579,-4.049794,-5.046247,-4.286991,9.652567,6.688240],[0.879406,7.595463,2.337842,-0.719218,5.461254,-6.733023,-6.300838,9.417766],[8.167168,8.505411,6.378274,-3.379976,2.188346,6.974857,0.477407,-3.564596],[2.839760,7.693862,3.176779,4.680152,-4.914315,-3.626768,2.281574,2.436698],[-5.054105,2.324799,-3.306824,1.986994,-1.501958,-8.077356,5.007153,8.400042],[-4.312128,-9.738607,-4.948490,7.875794,8.181118,1.599603,0.892180,0.317191],[-7.949740,-5.106097,8.766518,-2.783392,-6.064146,4.070678,9.443664,-5.326640]], dtype = "float64")#candidate|309|(16, 8)|const|float64
bop_310 = relay.not_equal(uop_303.astype('bool'), relay.reshape(const_309.astype('bool'), relay.shape_of(uop_303))) # shape=(16, 8)
bop_313 = relay.power(uop_224.astype('float64'), relay.reshape(var_260.astype('float64'), relay.shape_of(uop_224))) # shape=(16, 8)
var_316 = relay.var("var_316", dtype = "bool", shape = (16, 8))#candidate|316|(16, 8)|var|bool
bop_317 = relay.multiply(bop_310.astype('uint8'), relay.reshape(var_316.astype('uint8'), relay.shape_of(bop_310))) # shape=(16, 8)
bop_320 = relay.maximum(bop_317.astype('int64'), relay.reshape(bop_267.astype('int64'), relay.shape_of(bop_317))) # shape=(16, 8)
uop_323 = relay.log2(bop_317.astype('float32')) # shape=(16, 8)
uop_325 = relay.exp(bop_286.astype('float32')) # shape=(16, 8)
output = relay.Tuple([bop_182,uop_192,call_201,const_202,bop_204,bop_207,bop_211,bop_217,uop_220,call_244,const_245,var_246,uop_248,bop_254,bop_257,bop_261,call_264,call_273,var_274,uop_281,bop_289,uop_292,uop_294,call_296,const_297,uop_301,bop_306,bop_313,bop_320,uop_323,uop_325,])
output2 = relay.Tuple([bop_182,uop_192,call_203,const_202,bop_204,bop_207,bop_211,bop_217,uop_220,call_247,const_245,var_246,uop_248,bop_254,bop_257,bop_261,call_265,call_275,var_274,uop_281,bop_289,uop_292,uop_294,call_298,const_297,uop_301,bop_306,bop_313,bop_320,uop_323,uop_325,])
func_327 = relay.Function([var_174,var_175,var_181,var_194,var_210,var_231,var_240,var_246,var_250,var_260,var_266,var_274,var_305,var_316,], output)
mod['func_327'] = func_327
mod = relay.transform.InferType()(mod)
var_328 = relay.var("var_328", dtype = "uint8", shape = (16, 8))#candidate|328|(16, 8)|var|uint8
var_329 = relay.var("var_329", dtype = "uint8", shape = (16, 8))#candidate|329|(16, 8)|var|uint8
var_330 = relay.var("var_330", dtype = "uint8", shape = (16, 8))#candidate|330|(16, 8)|var|uint8
var_331 = relay.var("var_331", dtype = "float32", shape = (16, 8))#candidate|331|(16, 8)|var|float32
var_332 = relay.var("var_332", dtype = "float32", shape = (16, 8))#candidate|332|(16, 8)|var|float32
var_333 = relay.var("var_333", dtype = "float32", shape = (16, 8))#candidate|333|(16, 8)|var|float32
var_334 = relay.var("var_334", dtype = "float64", shape = (16, 8))#candidate|334|(16, 8)|var|float64
var_335 = relay.var("var_335", dtype = "float32", shape = (144,))#candidate|335|(144,)|var|float32
var_336 = relay.var("var_336", dtype = "uint16", shape = (16, 8))#candidate|336|(16, 8)|var|uint16
var_337 = relay.var("var_337", dtype = "float64", shape = (16, 8))#candidate|337|(16, 8)|var|float64
var_338 = relay.var("var_338", dtype = "uint16", shape = (16, 8))#candidate|338|(16, 8)|var|uint16
var_339 = relay.var("var_339", dtype = "float32", shape = (264,))#candidate|339|(264,)|var|float32
var_340 = relay.var("var_340", dtype = "float64", shape = (16, 8))#candidate|340|(16, 8)|var|float64
var_341 = relay.var("var_341", dtype = "bool", shape = (16, 8))#candidate|341|(16, 8)|var|bool
output = func_327(var_328,var_329,var_330,var_331,var_332,var_333,var_334,var_335,var_336,var_337,var_338,var_339,var_340,var_341,)
func_342 = relay.Function([var_328,var_329,var_330,var_331,var_332,var_333,var_334,var_335,var_336,var_337,var_338,var_339,var_340,var_341,], output)
mutated_mod['func_342'] = func_342
mutated_mod = relay.transform.InferType()(mutated_mod)
const_344 = relay.const(9.152649, dtype = "float64")#candidate|344|()|const|float64
var_345 = relay.var("var_345", dtype = "float64", shape = ())#candidate|345|()|var|float64
bop_346 = relay.equal(const_344.astype('bool'), var_345.astype('bool')) # shape=()
bop_349 = relay.divide(const_344.astype('float64'), bop_346.astype('float64')) # shape=()
bop_352 = relay.divide(bop_346.astype('float32'), const_344.astype('float32')) # shape=()
output = relay.Tuple([bop_349,bop_352,])
output2 = relay.Tuple([bop_349,bop_352,])
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
input_345= np.array(-0.128114, dtype='float64')
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

'''55: TVMFuncCall
54: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
53: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
52: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
51: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
50: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
49: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
48: tvm::transform::Pass::operator()(tvm::IRModule) const
47: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
46: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
45: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
44: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
43: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
42: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
41: tvm::transform::Pass::operator()(tvm::IRModule) const
40: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
39: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
37: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
36: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
35: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
34: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
33: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
32: _ZN3tvm5relay9transform22Devic
31: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
30: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
29: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
28: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
27: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
26: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
25: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
24: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
23: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
22: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
21: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
20: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
19: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
18: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::te::Tensor, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
17: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
16: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
15: tvm::transform::Pass::operator()(tvm::IRModule) const
14: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
13: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
12: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
11: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
10: _ZNSt17_Function_handlerIFvN3tvm7
9: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::tir::transform::NarrowDataType(int)::{lambda(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::tir::transform::NarrowDataType(int)::{lambda(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
8: tvm::tir::DataTypeRewriter::operator()(tvm::tir::Stmt)
7: _ZZN3tvm3tir11StmtFunctorIFNS0_4StmtERKS
6: tvm::tir::DataTypeRewriter::VisitStmt_(tvm::tir::StoreNode const*)
5: tvm::tir::StmtExprMutator::VisitExpr(tvm::PrimExpr const&)
4: _ZZN3tvm3tir11ExprFunctorIFNS_8PrimExprE
3: _ZThn16_N3tvm3tir16DataTyp
2: tvm::tir::DataTypeRewriter::VisitExpr_(tvm::tir::DivNode const*)
1: tvm::div(tvm::PrimExpr, tvm::PrimExpr, tvm::Span)
0: tvm::PrimExpr tvm::arith::TryConstFold<tvm::tir::Div>(tvm::PrimExpr, tvm::PrimExpr)

'''