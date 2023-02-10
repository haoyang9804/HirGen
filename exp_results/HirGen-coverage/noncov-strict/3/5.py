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
var_0 = relay.var("var_0", dtype = "float32", shape = (9,))#candidate|0|(9,)|var|float32
uop_1 = relay.atan(var_0.astype('float32')) # shape=(9,)
var_3 = relay.var("var_3", dtype = "float32", shape = (9,))#candidate|3|(9,)|var|float32
bop_4 = relay.not_equal(uop_1.astype('bool'), relay.reshape(var_3.astype('bool'), relay.shape_of(uop_1))) # shape=(9,)
bop_7 = relay.floor_divide(uop_1.astype('float32'), relay.reshape(var_3.astype('float32'), relay.shape_of(uop_1))) # shape=(9,)
var_10 = relay.var("var_10", dtype = "float32", shape = (9,))#candidate|10|(9,)|var|float32
bop_11 = relay.logical_or(var_3.astype('bool'), relay.reshape(var_10.astype('bool'), relay.shape_of(var_3))) # shape=(9,)
uop_14 = relay.atan(var_10.astype('float32')) # shape=(9,)
uop_16 = relay.asinh(bop_7.astype('float64')) # shape=(9,)
uop_18 = relay.sigmoid(uop_16.astype('float32')) # shape=(9,)
uop_20 = relay.asin(uop_18.astype('float32')) # shape=(9,)
var_22 = relay.var("var_22", dtype = "float32", shape = (9,))#candidate|22|(9,)|var|float32
bop_23 = relay.logical_xor(uop_20.astype('int64'), relay.reshape(var_22.astype('int64'), relay.shape_of(uop_20))) # shape=(9,)
bop_26 = relay.left_shift(uop_18.astype('int16'), relay.reshape(var_10.astype('int16'), relay.shape_of(uop_18))) # shape=(9,)
output = relay.Tuple([bop_4,bop_11,uop_14,bop_23,bop_26,])
output2 = relay.Tuple([bop_4,bop_11,uop_14,bop_23,bop_26,])
func_29 = relay.Function([var_0,var_3,var_10,var_22,], output)
mod['func_29'] = func_29
mod = relay.transform.InferType()(mod)
var_30 = relay.var("var_30", dtype = "float32", shape = (9,))#candidate|30|(9,)|var|float32
var_31 = relay.var("var_31", dtype = "float32", shape = (9,))#candidate|31|(9,)|var|float32
var_32 = relay.var("var_32", dtype = "float32", shape = (9,))#candidate|32|(9,)|var|float32
var_33 = relay.var("var_33", dtype = "float32", shape = (9,))#candidate|33|(9,)|var|float32
output = func_29(var_30,var_31,var_32,var_33,)
func_34 = relay.Function([var_30,var_31,var_32,var_33,], output)
mutated_mod['func_34'] = func_34
mutated_mod = relay.transform.InferType()(mutated_mod)
var_36 = relay.var("var_36", dtype = "uint64", shape = (9, 4))#candidate|36|(9, 4)|var|uint64
var_37 = relay.var("var_37", dtype = "uint64", shape = (9, 4))#candidate|37|(9, 4)|var|uint64
bop_38 = relay.logical_xor(var_36.astype('uint64'), relay.reshape(var_37.astype('uint64'), relay.shape_of(var_36))) # shape=(9, 4)
const_41 = relay.const([[-9,1,-4,1],[6,2,-3,-3],[4,-3,-4,-7],[-2,-1,-4,9],[1,3,2,-5],[-10,-10,9,-1],[7,-7,2,9],[-9,-4,-9,-10],[-2,-6,-9,-3]], dtype = "uint64")#candidate|41|(9, 4)|const|uint64
bop_42 = relay.floor_mod(bop_38.astype('float32'), relay.reshape(const_41.astype('float32'), relay.shape_of(bop_38))) # shape=(9, 4)
var_45 = relay.var("var_45", dtype = "uint64", shape = (9, 4))#candidate|45|(9, 4)|var|uint64
bop_46 = relay.minimum(const_41.astype('int64'), relay.reshape(var_45.astype('int64'), relay.shape_of(const_41))) # shape=(9, 4)
uop_49 = relay.sin(bop_42.astype('float32')) # shape=(9, 4)
bop_51 = relay.less_equal(uop_49.astype('bool'), relay.reshape(var_45.astype('bool'), relay.shape_of(uop_49))) # shape=(9, 4)
output = relay.Tuple([bop_46,bop_51,])
output2 = relay.Tuple([bop_46,bop_51,])
func_54 = relay.Function([var_36,var_37,var_45,], output)
mod['func_54'] = func_54
mod = relay.transform.InferType()(mod)
mutated_mod['func_54'] = func_54
mutated_mod = relay.transform.InferType()(mutated_mod)
func_54_call = mutated_mod.get_global_var('func_54')
var_56 = relay.var("var_56", dtype = "uint64", shape = (9, 4))#candidate|56|(9, 4)|var|uint64
var_57 = relay.var("var_57", dtype = "uint64", shape = (9, 4))#candidate|57|(9, 4)|var|uint64
var_58 = relay.var("var_58", dtype = "uint64", shape = (9, 4))#candidate|58|(9, 4)|var|uint64
call_55 = func_54_call(var_56,var_57,var_58,)
output = call_55
func_59 = relay.Function([var_56,var_57,var_58,], output)
mutated_mod['func_59'] = func_59
mutated_mod = relay.transform.InferType()(mutated_mod)
var_61 = relay.var("var_61", dtype = "float32", shape = ())#candidate|61|()|var|float32
uop_62 = relay.asin(var_61.astype('float32')) # shape=()
uop_64 = relay.atan(uop_62.astype('float64')) # shape=()
func_29_call = mod.get_global_var('func_29')
func_34_call = mutated_mod.get_global_var('func_34')
const_67 = relay.const([[-6.692777],[-2.824424],[-9.383670],[9.943362],[-6.478070],[4.759568],[1.145280],[-9.360994],[-9.362038]], dtype = "float32")#candidate|67|(9, 1)|const|float32
call_66 = relay.TupleGetItem(func_29_call(relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), ), 3)
call_68 = relay.TupleGetItem(func_34_call(relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), ), 3)
var_69 = relay.var("var_69", dtype = "float32", shape = (10,))#candidate|69|(10,)|var|float32
bop_70 = relay.not_equal(uop_62.astype('bool'), var_69.astype('bool')) # shape=(10,)
uop_73 = relay.log2(uop_64.astype('float64')) # shape=()
bop_75 = relay.less_equal(uop_73.astype('bool'), uop_62.astype('bool')) # shape=()
uop_78 = relay.sin(uop_73.astype('float32')) # shape=()
uop_80 = relay.exp(uop_64.astype('float32')) # shape=()
uop_82 = relay.sqrt(bop_75.astype('float32')) # shape=()
bop_84 = relay.logical_or(uop_80.astype('bool'), uop_64.astype('bool')) # shape=()
bop_87 = relay.logical_and(uop_82.astype('bool'), uop_64.astype('bool')) # shape=()
uop_90 = relay.asin(uop_78.astype('float32')) # shape=()
func_29_call = mod.get_global_var('func_29')
func_34_call = mutated_mod.get_global_var('func_34')
call_92 = relay.TupleGetItem(func_29_call(relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(call_66.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(call_66.astype('float32'), [9,]), ), 2)
call_93 = relay.TupleGetItem(func_34_call(relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(call_66.astype('float32'), [9,]), relay.reshape(const_67.astype('float32'), [9,]), relay.reshape(call_66.astype('float32'), [9,]), ), 2)
bop_94 = relay.logical_xor(uop_90.astype('int64'), uop_80.astype('int64')) # shape=()
bop_97 = relay.less_equal(uop_64.astype('bool'), var_69.astype('bool')) # shape=(10,)
output = relay.Tuple([call_66,const_67,bop_70,bop_84,bop_87,call_92,bop_94,bop_97,])
output2 = relay.Tuple([call_68,const_67,bop_70,bop_84,bop_87,call_93,bop_94,bop_97,])
func_100 = relay.Function([var_61,var_69,], output)
mod['func_100'] = func_100
mod = relay.transform.InferType()(mod)
mutated_mod['func_100'] = func_100
mutated_mod = relay.transform.InferType()(mutated_mod)
func_100_call = mutated_mod.get_global_var('func_100')
var_102 = relay.var("var_102", dtype = "float32", shape = ())#candidate|102|()|var|float32
var_103 = relay.var("var_103", dtype = "float32", shape = (10,))#candidate|103|(10,)|var|float32
call_101 = func_100_call(var_102,var_103,)
output = call_101
func_104 = relay.Function([var_102,var_103,], output)
mutated_mod['func_104'] = func_104
mutated_mod = relay.transform.InferType()(mutated_mod)
var_106 = relay.var("var_106", dtype = "float64", shape = (10, 14, 13))#candidate|106|(10, 14, 13)|var|float64
uop_107 = relay.sigmoid(var_106.astype('float64')) # shape=(10, 14, 13)
bop_109 = relay.logical_or(uop_107.astype('bool'), relay.reshape(var_106.astype('bool'), relay.shape_of(uop_107))) # shape=(10, 14, 13)
var_112 = relay.var("var_112", dtype = "bool", shape = (10, 14, 13))#candidate|112|(10, 14, 13)|var|bool
bop_113 = relay.left_shift(bop_109.astype('int32'), relay.reshape(var_112.astype('int32'), relay.shape_of(bop_109))) # shape=(10, 14, 13)
output = relay.Tuple([bop_113,])
output2 = relay.Tuple([bop_113,])
func_116 = relay.Function([var_106,var_112,], output)
mod['func_116'] = func_116
mod = relay.transform.InferType()(mod)
mutated_mod['func_116'] = func_116
mutated_mod = relay.transform.InferType()(mutated_mod)
func_116_call = mutated_mod.get_global_var('func_116')
var_118 = relay.var("var_118", dtype = "float64", shape = (10, 14, 13))#candidate|118|(10, 14, 13)|var|float64
var_119 = relay.var("var_119", dtype = "bool", shape = (10, 14, 13))#candidate|119|(10, 14, 13)|var|bool
call_117 = func_116_call(var_118,var_119,)
output = call_117
func_120 = relay.Function([var_118,var_119,], output)
mutated_mod['func_120'] = func_120
mutated_mod = relay.transform.InferType()(mutated_mod)
var_122 = relay.var("var_122", dtype = "float64", shape = (7,))#candidate|122|(7,)|var|float64
uop_123 = relay.acosh(var_122.astype('float64')) # shape=(7,)
uop_125 = relay.asin(uop_123.astype('float32')) # shape=(7,)
func_29_call = mod.get_global_var('func_29')
func_34_call = mutated_mod.get_global_var('func_34')
var_128 = relay.var("var_128", dtype = "float32", shape = (3, 3))#candidate|128|(3, 3)|var|float32
call_127 = relay.TupleGetItem(func_29_call(relay.reshape(var_128.astype('float32'), [9,]), relay.reshape(var_128.astype('float32'), [9,]), relay.reshape(var_128.astype('float32'), [9,]), relay.reshape(var_128.astype('float32'), [9,]), ), 2)
call_129 = relay.TupleGetItem(func_34_call(relay.reshape(var_128.astype('float32'), [9,]), relay.reshape(var_128.astype('float32'), [9,]), relay.reshape(var_128.astype('float32'), [9,]), relay.reshape(var_128.astype('float32'), [9,]), ), 2)
uop_130 = relay.erf(uop_125.astype('float64')) # shape=(7,)
var_132 = relay.var("var_132", dtype = "float64", shape = (7,))#candidate|132|(7,)|var|float64
bop_133 = relay.multiply(uop_130.astype('int64'), relay.reshape(var_132.astype('int64'), relay.shape_of(uop_130))) # shape=(7,)
var_136 = relay.var("var_136", dtype = "float64", shape = (7,))#candidate|136|(7,)|var|float64
bop_137 = relay.minimum(uop_130.astype('uint16'), relay.reshape(var_136.astype('uint16'), relay.shape_of(uop_130))) # shape=(7,)
bop_140 = relay.divide(bop_133.astype('float32'), relay.reshape(var_136.astype('float32'), relay.shape_of(bop_133))) # shape=(7,)
uop_143 = relay.log(uop_123.astype('float64')) # shape=(7,)
output = relay.Tuple([call_127,var_128,bop_137,bop_140,uop_143,])
output2 = relay.Tuple([call_129,var_128,bop_137,bop_140,uop_143,])
func_145 = relay.Function([var_122,var_128,var_132,var_136,], output)
mod['func_145'] = func_145
mod = relay.transform.InferType()(mod)
mutated_mod['func_145'] = func_145
mutated_mod = relay.transform.InferType()(mutated_mod)
func_145_call = mutated_mod.get_global_var('func_145')
var_147 = relay.var("var_147", dtype = "float64", shape = (7,))#candidate|147|(7,)|var|float64
var_148 = relay.var("var_148", dtype = "float32", shape = (3, 3))#candidate|148|(3, 3)|var|float32
var_149 = relay.var("var_149", dtype = "float64", shape = (7,))#candidate|149|(7,)|var|float64
var_150 = relay.var("var_150", dtype = "float64", shape = (7,))#candidate|150|(7,)|var|float64
call_146 = func_145_call(var_147,var_148,var_149,var_150,)
output = call_146
func_151 = relay.Function([var_147,var_148,var_149,var_150,], output)
mutated_mod['func_151'] = func_151
mutated_mod = relay.transform.InferType()(mutated_mod)
var_153 = relay.var("var_153", dtype = "float64", shape = (1, 13))#candidate|153|(1, 13)|var|float64
uop_154 = relay.acos(var_153.astype('float64')) # shape=(1, 13)
uop_156 = relay.acosh(var_153.astype('float32')) # shape=(1, 13)
func_100_call = mod.get_global_var('func_100')
func_104_call = mutated_mod.get_global_var('func_104')
var_159 = relay.var("var_159", dtype = "float32", shape = ())#candidate|159|()|var|float32
var_160 = relay.var("var_160", dtype = "float32", shape = (10,))#candidate|160|(10,)|var|float32
call_158 = relay.TupleGetItem(func_100_call(relay.reshape(var_159.astype('float32'), []), relay.reshape(var_160.astype('float32'), [10,]), ), 6)
call_161 = relay.TupleGetItem(func_104_call(relay.reshape(var_159.astype('float32'), []), relay.reshape(var_160.astype('float32'), [10,]), ), 6)
var_162 = relay.var("var_162", dtype = "float32", shape = (10,))#candidate|162|(10,)|var|float32
bop_163 = relay.minimum(var_160.astype('uint16'), relay.reshape(var_162.astype('uint16'), relay.shape_of(var_160))) # shape=(10,)
var_166 = relay.var("var_166", dtype = "float64", shape = (3, 13))#candidate|166|(3, 13)|var|float64
bop_167 = relay.left_shift(uop_154.astype('uint8'), var_166.astype('uint8')) # shape=(3, 13)
uop_170 = relay.tan(bop_167.astype('float32')) # shape=(3, 13)
output = relay.Tuple([uop_156,call_158,var_159,bop_163,uop_170,])
output2 = relay.Tuple([uop_156,call_161,var_159,bop_163,uop_170,])
func_172 = relay.Function([var_153,var_159,var_160,var_162,var_166,], output)
mod['func_172'] = func_172
mod = relay.transform.InferType()(mod)
mutated_mod['func_172'] = func_172
mutated_mod = relay.transform.InferType()(mutated_mod)
func_172_call = mutated_mod.get_global_var('func_172')
var_174 = relay.var("var_174", dtype = "float64", shape = (1, 13))#candidate|174|(1, 13)|var|float64
var_175 = relay.var("var_175", dtype = "float32", shape = ())#candidate|175|()|var|float32
var_176 = relay.var("var_176", dtype = "float32", shape = (10,))#candidate|176|(10,)|var|float32
var_177 = relay.var("var_177", dtype = "float32", shape = (10,))#candidate|177|(10,)|var|float32
var_178 = relay.var("var_178", dtype = "float64", shape = (3, 13))#candidate|178|(3, 13)|var|float64
call_173 = func_172_call(var_174,var_175,var_176,var_177,var_178,)
output = call_173
func_179 = relay.Function([var_174,var_175,var_176,var_177,var_178,], output)
mutated_mod['func_179'] = func_179
mutated_mod = relay.transform.InferType()(mutated_mod)
var_181 = relay.var("var_181", dtype = "float32", shape = (15, 5, 2))#candidate|181|(15, 5, 2)|var|float32
uop_182 = relay.sigmoid(var_181.astype('float32')) # shape=(15, 5, 2)
uop_184 = relay.atanh(var_181.astype('float32')) # shape=(15, 5, 2)
output = relay.Tuple([uop_182,uop_184,])
output2 = relay.Tuple([uop_182,uop_184,])
func_186 = relay.Function([var_181,], output)
mod['func_186'] = func_186
mod = relay.transform.InferType()(mod)
var_187 = relay.var("var_187", dtype = "float32", shape = (15, 5, 2))#candidate|187|(15, 5, 2)|var|float32
output = func_186(var_187)
func_188 = relay.Function([var_187], output)
mutated_mod['func_188'] = func_188
mutated_mod = relay.transform.InferType()(mutated_mod)
var_190 = relay.var("var_190", dtype = "int64", shape = ())#candidate|190|()|var|int64
var_191 = relay.var("var_191", dtype = "int64", shape = (4, 9))#candidate|191|(4, 9)|var|int64
bop_192 = relay.greater_equal(var_190.astype('bool'), var_191.astype('bool')) # shape=(4, 9)
bop_195 = relay.right_shift(var_191.astype('int64'), relay.reshape(bop_192.astype('int64'), relay.shape_of(var_191))) # shape=(4, 9)
uop_198 = relay.sin(var_190.astype('float64')) # shape=()
uop_200 = relay.atan(uop_198.astype('float32')) # shape=()
uop_202 = relay.asin(uop_200.astype('float64')) # shape=()
uop_204 = relay.exp(uop_200.astype('float64')) # shape=()
bop_206 = relay.less(uop_204.astype('bool'), uop_198.astype('bool')) # shape=()
bop_209 = relay.floor_divide(uop_200.astype('float64'), bop_192.astype('float64')) # shape=(4, 9)
uop_212 = relay.sigmoid(bop_209.astype('float32')) # shape=(4, 9)
const_214 = relay.const(1.147549, dtype = "float64")#candidate|214|()|const|float64
bop_215 = relay.minimum(uop_204.astype('uint16'), const_214.astype('uint16')) # shape=()
uop_218 = relay.asinh(uop_200.astype('float32')) # shape=()
bop_220 = relay.maximum(uop_202.astype('uint64'), const_214.astype('uint64')) # shape=()
var_223 = relay.var("var_223", dtype = "uint16", shape = (10,))#candidate|223|(10,)|var|uint16
bop_224 = relay.equal(bop_215.astype('bool'), var_223.astype('bool')) # shape=(10,)
uop_227 = relay.rsqrt(bop_215.astype('float32')) # shape=()
var_229 = relay.var("var_229", dtype = "float64", shape = (4,))#candidate|229|(4,)|var|float64
bop_230 = relay.mod(uop_202.astype('float64'), var_229.astype('float64')) # shape=(4,)
output = relay.Tuple([bop_195,bop_206,uop_212,uop_218,bop_220,bop_224,uop_227,bop_230,])
output2 = relay.Tuple([bop_195,bop_206,uop_212,uop_218,bop_220,bop_224,uop_227,bop_230,])
func_233 = relay.Function([var_190,var_191,var_223,var_229,], output)
mod['func_233'] = func_233
mod = relay.transform.InferType()(mod)
var_234 = relay.var("var_234", dtype = "int64", shape = ())#candidate|234|()|var|int64
var_235 = relay.var("var_235", dtype = "int64", shape = (4, 9))#candidate|235|(4, 9)|var|int64
var_236 = relay.var("var_236", dtype = "uint16", shape = (10,))#candidate|236|(10,)|var|uint16
var_237 = relay.var("var_237", dtype = "float64", shape = (4,))#candidate|237|(4,)|var|float64
output = func_233(var_234,var_235,var_236,var_237,)
func_238 = relay.Function([var_234,var_235,var_236,var_237,], output)
mutated_mod['func_238'] = func_238
mutated_mod = relay.transform.InferType()(mutated_mod)
const_240 = relay.const(1, dtype = "int8")#candidate|240|()|const|int8
var_241 = relay.var("var_241", dtype = "int8", shape = (2,))#candidate|241|(2,)|var|int8
bop_242 = relay.less(const_240.astype('bool'), var_241.astype('bool')) # shape=(2,)
uop_245 = relay.rsqrt(bop_242.astype('float64')) # shape=(2,)
const_247 = relay.const([-10,-7], dtype = "int8")#candidate|247|(2,)|const|int8
bop_248 = relay.greater(var_241.astype('bool'), relay.reshape(const_247.astype('bool'), relay.shape_of(var_241))) # shape=(2,)
bop_251 = relay.power(uop_245.astype('float64'), relay.reshape(var_241.astype('float64'), relay.shape_of(uop_245))) # shape=(2,)
uop_254 = relay.cos(uop_245.astype('float32')) # shape=(2,)
uop_256 = relay.sigmoid(uop_254.astype('float64')) # shape=(2,)
uop_258 = relay.acos(uop_256.astype('float64')) # shape=(2,)
bop_260 = relay.less_equal(uop_258.astype('bool'), relay.reshape(bop_248.astype('bool'), relay.shape_of(uop_258))) # shape=(2,)
var_263 = relay.var("var_263", dtype = "float32", shape = (2,))#candidate|263|(2,)|var|float32
bop_264 = relay.subtract(uop_254.astype('int8'), relay.reshape(var_263.astype('int8'), relay.shape_of(uop_254))) # shape=(2,)
const_267 = relay.const([True,True], dtype = "bool")#candidate|267|(2,)|const|bool
bop_268 = relay.left_shift(bop_260.astype('uint16'), relay.reshape(const_267.astype('uint16'), relay.shape_of(bop_260))) # shape=(2,)
bop_271 = relay.subtract(uop_256.astype('uint32'), const_240.astype('uint32')) # shape=(2,)
var_274 = relay.var("var_274", dtype = "float64", shape = (2,))#candidate|274|(2,)|var|float64
bop_275 = relay.maximum(uop_258.astype('uint32'), relay.reshape(var_274.astype('uint32'), relay.shape_of(uop_258))) # shape=(2,)
bop_278 = relay.divide(bop_264.astype('float64'), relay.reshape(bop_260.astype('float64'), relay.shape_of(bop_264))) # shape=(2,)
func_233_call = mod.get_global_var('func_233')
func_238_call = mutated_mod.get_global_var('func_238')
var_282 = relay.var("var_282", dtype = "int64", shape = (36,))#candidate|282|(36,)|var|int64
const_283 = relay.const([-3,6,3,9,-9,4,1,9,3,8], dtype = "uint16")#candidate|283|(10,)|const|uint16
const_284 = relay.const([2.745731,0.429070,9.991451,-0.669422], dtype = "float64")#candidate|284|(4,)|const|float64
call_281 = relay.TupleGetItem(func_233_call(relay.reshape(const_240.astype('int64'), []), relay.reshape(var_282.astype('int64'), [4, 9]), relay.reshape(const_283.astype('uint16'), [10,]), relay.reshape(const_284.astype('float64'), [4,]), ), 1)
call_285 = relay.TupleGetItem(func_238_call(relay.reshape(const_240.astype('int64'), []), relay.reshape(var_282.astype('int64'), [4, 9]), relay.reshape(const_283.astype('uint16'), [10,]), relay.reshape(const_284.astype('float64'), [4,]), ), 1)
const_286 = relay.const([-7,1], dtype = "int8")#candidate|286|(2,)|const|int8
bop_287 = relay.greater(bop_264.astype('bool'), relay.reshape(const_286.astype('bool'), relay.shape_of(bop_264))) # shape=(2,)
output = relay.Tuple([bop_251,bop_268,bop_271,bop_275,bop_278,call_281,var_282,const_283,const_284,bop_287,])
output2 = relay.Tuple([bop_251,bop_268,bop_271,bop_275,bop_278,call_285,var_282,const_283,const_284,bop_287,])
func_290 = relay.Function([var_241,var_263,var_274,var_282,], output)
mod['func_290'] = func_290
mod = relay.transform.InferType()(mod)
mutated_mod['func_290'] = func_290
mutated_mod = relay.transform.InferType()(mutated_mod)
func_290_call = mutated_mod.get_global_var('func_290')
var_292 = relay.var("var_292", dtype = "int8", shape = (2,))#candidate|292|(2,)|var|int8
var_293 = relay.var("var_293", dtype = "float32", shape = (2,))#candidate|293|(2,)|var|float32
var_294 = relay.var("var_294", dtype = "float64", shape = (2,))#candidate|294|(2,)|var|float64
var_295 = relay.var("var_295", dtype = "int64", shape = (36,))#candidate|295|(36,)|var|int64
call_291 = func_290_call(var_292,var_293,var_294,var_295,)
output = call_291
func_296 = relay.Function([var_292,var_293,var_294,var_295,], output)
mutated_mod['func_296'] = func_296
mutated_mod = relay.transform.InferType()(mutated_mod)
const_298 = relay.const([[[8.391492,-8.186208,0.397475,2.479618,-0.679574,4.898275,-2.119656,8.198322],[6.860474,-5.665999,7.707096,-3.215093,7.024386,0.032187,-8.216708,-9.829313],[6.502614,-4.444581,-0.544041,-8.740944,-5.618255,-6.492113,-8.134686,-9.790083],[-3.742347,-0.614454,3.156852,-7.585561,-6.065462,-9.074757,-0.186885,9.504246],[-5.523779,-0.708635,1.928645,1.148078,3.825888,-8.258288,0.080944,3.619243],[-3.109705,9.398857,5.364431,-2.665915,9.807944,-7.101419,0.618939,5.673395],[-6.667979,-5.265728,8.668558,3.710699,8.201149,-4.689784,2.262113,-9.452217],[-1.339543,9.601982,6.289306,4.743155,4.856614,-4.920824,7.304749,-6.483592],[-9.930856,-2.494710,-4.681685,-9.011805,4.931724,-2.267500,-9.706037,-3.936486],[-3.421543,-5.894983,4.961454,1.731546,-1.139724,6.630987,5.843181,5.805901],[-6.228349,5.711279,9.174481,-9.116878,4.163247,6.710638,-5.027864,-8.206341],[-7.318061,-1.858602,-8.395027,-2.115990,-3.731653,3.829251,6.365682,6.159663],[-1.147979,-2.133868,-5.178488,-2.197111,1.077174,9.908266,-4.380174,-5.034779]],[[3.990834,-5.387824,-8.441894,-5.678487,2.201789,5.419017,-6.809141,-7.540180],[5.351221,-0.120738,-9.913964,4.138810,1.153512,3.696261,-4.977849,-4.072453],[-7.552646,-9.507980,6.909770,1.901181,-3.662594,2.891404,6.916709,1.022974],[-4.598588,-5.022943,-9.945817,-0.228633,7.495406,1.822848,0.090487,8.137796],[7.407443,2.921054,-9.318889,5.584118,0.762846,7.730787,-8.364089,9.513333],[-9.448076,3.286051,-6.767421,-2.794691,6.195958,8.575585,6.525488,4.114806],[-9.857655,9.324902,5.362671,5.500158,5.359918,-7.459614,3.776389,-4.746672],[9.939536,7.391479,1.159382,5.849870,7.002349,-5.504047,5.073765,0.170625],[-4.388966,-6.774598,9.523487,-8.732805,-7.670944,9.242069,-8.505627,-3.086694],[5.921641,7.161317,5.475279,-3.198209,-8.046314,-1.730660,-0.586744,6.367946],[-6.580436,-0.967023,-1.240027,8.066100,8.145284,-6.952240,-0.423560,-9.960721],[-7.294322,2.430791,-4.883060,5.934282,-1.717984,7.048859,3.144707,0.831554],[-5.489092,-1.520630,-0.297935,-6.513788,-1.288076,8.736452,9.217018,-7.839401]],[[-2.711033,8.883495,2.574068,8.529946,-6.192464,-6.661949,3.733810,-1.363451],[-4.234716,-1.871627,-7.408072,-7.524719,-5.451572,3.184540,7.635939,5.105050],[5.849188,2.346592,5.526812,-8.057817,5.214937,-2.641250,-0.963527,-8.498684],[9.994152,3.434781,5.679738,-7.071321,-8.676323,2.917983,0.291606,5.043003],[-0.746216,-3.865536,6.538595,7.753478,5.033158,9.448755,-6.498905,-9.035180],[4.506508,6.920144,-9.240764,-7.530218,-1.219215,-3.144636,-8.682715,-1.737059],[-4.028412,0.130465,-4.028474,-1.497900,-0.225695,0.997309,-7.236577,-3.566081],[-9.459522,-9.046981,1.978359,9.353736,7.744793,4.279925,7.644381,-4.931463],[-2.175834,9.887158,-2.079177,1.130301,7.176930,-7.246680,-4.713454,5.215222],[-7.992976,3.078409,-0.702269,5.375537,4.426424,-8.477718,7.082528,-5.714029],[-6.248416,-4.646485,-0.070745,9.134389,-4.715165,3.186201,-8.417333,9.901493],[-7.184606,-9.879238,8.821983,-9.245123,-4.056878,6.740231,1.267194,-3.079928],[6.628770,0.905784,-7.203971,-9.338491,-2.064477,8.792880,-9.054008,9.469753]],[[-6.043946,-8.966401,-1.616713,-1.543383,0.226717,-9.580681,2.562631,-8.703846],[7.239879,-4.482898,-0.776353,-6.690041,-5.033648,1.975893,4.849387,-0.624329],[-0.011613,5.865684,2.910077,9.219985,-7.321227,-1.080384,-0.176258,-8.832848],[-2.773944,8.998240,9.811498,4.697413,-1.611723,2.744105,3.009595,0.075793],[-9.428823,-1.049635,-0.276129,-7.108849,6.585015,-0.423716,4.849748,0.706579],[-2.296312,-0.269917,6.354979,3.473106,-6.901883,7.993399,-3.240929,7.379311],[5.311601,0.206534,2.821954,-9.594084,-2.010489,2.032390,-7.415700,9.243080],[9.374371,-7.603537,5.081658,-5.258302,-2.066023,3.242047,-9.346107,7.971000],[4.420638,8.013753,-3.399105,6.035478,9.778026,-0.508403,-3.093258,-8.062558],[9.978371,-6.176216,-7.333030,-0.975202,-9.086087,-2.670600,-9.665837,-9.614384],[-3.622750,8.042101,-9.647368,5.803629,0.866628,8.009136,9.671674,3.040997],[-2.246965,1.316469,4.789614,-9.972654,0.222145,-2.993193,-7.728747,5.027842],[2.860137,-1.499035,-8.815404,6.996930,4.116471,2.558057,-7.592606,-7.138249]],[[-3.570295,7.263027,5.862237,0.911827,-3.181687,6.243162,-9.727598,4.436388],[0.116347,7.925110,9.245475,6.460561,3.757596,2.378774,6.586090,7.698693],[8.130316,7.560797,-4.962075,-9.228664,7.937461,2.735063,-3.975834,0.149040],[-7.794937,9.004157,0.847256,6.125853,1.328737,-3.391252,-4.205789,-0.566355],[-6.080491,-4.174899,-0.333832,-2.685669,-6.818421,-7.640858,-0.984750,-4.006349],[-6.370412,1.535067,-3.880398,8.878265,3.212227,7.928107,6.286721,-9.102420],[8.584566,-8.842266,1.953149,7.881334,-1.138850,-9.425558,4.087636,-0.095079],[2.498541,1.180272,-2.398403,2.972403,-7.592769,9.863995,1.163465,-2.979728],[-9.106740,-8.552007,6.152845,-6.739909,5.838717,-4.713961,6.906965,2.155838],[-3.909278,6.413639,6.216527,5.980573,7.669234,2.161109,-1.390966,1.908660],[-0.527618,-7.246057,-3.328802,9.467297,7.525415,-3.351247,-3.071580,8.379426],[9.258564,3.303524,-0.430131,-0.817217,7.674303,-5.996916,7.076302,3.183543],[1.541144,5.129478,-4.540254,-3.575507,0.586163,-7.633429,-2.966692,-8.306175]],[[-1.394911,-7.235141,-7.859427,-0.254493,-6.448869,9.467774,4.885900,6.537549],[-8.810374,-8.146741,2.185216,7.250070,2.837386,6.115317,3.129606,1.057622],[4.779190,-3.219205,2.173743,-0.135484,4.042184,8.899578,-2.617726,-0.799408],[-2.425926,7.713298,8.458361,-3.286117,-9.115431,-2.962684,5.805359,-5.077927],[8.445565,2.600112,1.037449,-2.566715,3.064511,7.488746,-2.008273,4.861835],[-6.209322,-4.160189,6.308395,8.706275,-5.134827,1.185885,3.734690,5.925781],[4.470954,0.720541,-0.239186,6.367544,-5.014226,0.533248,-4.563705,7.188860],[7.847834,-1.589207,-8.193908,-1.919836,9.567615,-7.561276,9.294531,5.081662],[-4.888858,-8.844129,-3.782726,8.026550,-1.472501,1.989399,6.551322,-0.434945],[5.291489,1.612629,-2.311875,-0.862650,9.291072,-0.874017,-3.352275,2.608858],[6.481710,9.289440,-5.321914,-4.137160,7.287248,9.286097,-8.395470,8.605209],[5.323835,2.761048,1.565552,3.284480,-9.573711,7.031814,-6.822898,-9.544390],[-6.462435,6.599596,-8.441116,-3.106857,-0.938178,-2.228267,7.860291,-2.077488]],[[-4.615113,-5.503199,9.973351,-2.353892,1.776745,4.838553,2.123637,1.879540],[6.617407,-3.700181,-8.842308,-3.315320,-7.103034,8.606377,-2.958936,-9.922239],[5.700267,-6.208287,3.409711,6.885649,5.819686,-5.969699,-1.675991,-2.782139],[-8.536979,9.762051,1.379482,3.472920,-0.540811,5.127826,-2.983117,9.462325],[-7.645516,4.374067,-7.207290,5.320790,0.918244,-0.580190,9.226824,0.137633],[8.312878,0.687238,-1.061676,9.629783,1.680608,-6.249566,-1.173909,-4.531144],[6.256720,-1.926340,-6.904939,6.761362,6.957725,2.333224,2.803164,0.296430],[9.419370,5.735402,8.278959,-2.334000,-7.960323,9.914154,-5.731081,4.363143],[1.339117,9.033662,6.088404,9.818764,5.803429,-9.931824,7.293047,5.284548],[-3.655031,0.850659,3.934243,-7.449706,-0.311489,0.163876,3.009047,-5.357617],[-3.182744,-5.882725,0.355589,2.494184,-7.759322,7.060671,4.381844,-0.960194],[0.733908,0.368374,0.548480,-7.821400,7.254596,-8.994410,1.358643,2.906837],[9.595873,4.636752,3.487123,7.995417,-6.658999,-3.766229,4.398177,7.976993]],[[-7.098218,-4.190586,9.428442,2.632220,-9.848434,1.999795,-8.666090,1.020867],[6.736427,-7.048232,4.247447,-4.161416,-5.003518,9.717741,-8.455045,4.432748],[-3.442839,-1.677670,-8.178660,-5.292143,-1.832369,-0.891621,2.620307,-5.126905],[1.464450,6.519925,-0.358085,-5.775296,-7.923344,-9.587653,5.625263,8.772644],[6.696092,7.341823,0.546002,-5.138225,7.161608,8.528109,0.399220,1.919787],[-4.671946,2.273222,8.878634,-8.513181,-0.197651,-8.304398,-6.857364,5.061272],[1.819427,8.838403,-0.574903,-9.589857,7.728802,-9.510517,1.084502,4.025263],[1.379491,-3.213281,-5.344557,-5.850282,-0.751453,-8.373905,1.308676,-1.475071],[-5.422970,2.215614,-8.440891,4.125626,2.706324,-9.881937,4.755614,0.084100],[-0.966709,7.793522,-0.244692,8.297376,-7.381716,6.640540,6.935105,4.498022],[3.998822,4.050838,-4.261077,5.632665,-0.858216,9.109111,9.565865,-5.637349],[-4.379304,4.228935,-2.765843,2.963850,-8.634367,-1.716504,4.987307,-3.222013],[-6.044565,-8.108301,-3.407325,5.086896,4.784487,-6.040963,8.232645,-5.309551]]], dtype = "float64")#candidate|298|(8, 13, 8)|const|float64
var_299 = relay.var("var_299", dtype = "float64", shape = (8, 13, 8))#candidate|299|(8, 13, 8)|var|float64
bop_300 = relay.mod(const_298.astype('float64'), relay.reshape(var_299.astype('float64'), relay.shape_of(const_298))) # shape=(8, 13, 8)
func_233_call = mod.get_global_var('func_233')
func_238_call = mutated_mod.get_global_var('func_238')
const_304 = relay.const(-6, dtype = "int64")#candidate|304|()|const|int64
const_305 = relay.const([5,4,6,5,2,-10,-2,5,10,-8,-4,-9,-2,5,1,10,3,10,-2,1,8,8,-2,3,2,6,-2,7,7,2,9,-4,8,4,-8,-7], dtype = "int64")#candidate|305|(36,)|const|int64
var_306 = relay.var("var_306", dtype = "uint16", shape = (1, 10))#candidate|306|(1, 10)|var|uint16
const_307 = relay.const([7.409341,5.335520,-0.511752,4.076115], dtype = "float64")#candidate|307|(4,)|const|float64
call_303 = relay.TupleGetItem(func_233_call(relay.reshape(const_304.astype('int64'), []), relay.reshape(const_305.astype('int64'), [4, 9]), relay.reshape(var_306.astype('uint16'), [10,]), relay.reshape(const_307.astype('float64'), [4,]), ), 3)
call_308 = relay.TupleGetItem(func_238_call(relay.reshape(const_304.astype('int64'), []), relay.reshape(const_305.astype('int64'), [4, 9]), relay.reshape(var_306.astype('uint16'), [10,]), relay.reshape(const_307.astype('float64'), [4,]), ), 3)
uop_309 = relay.exp(const_298.astype('float32')) # shape=(8, 13, 8)
bop_311 = relay.power(uop_309.astype('float64'), relay.reshape(bop_300.astype('float64'), relay.shape_of(uop_309))) # shape=(8, 13, 8)
uop_314 = relay.atan(uop_309.astype('float64')) # shape=(8, 13, 8)
uop_316 = relay.sqrt(uop_314.astype('float32')) # shape=(8, 13, 8)
var_318 = relay.var("var_318", dtype = "float32", shape = (8, 13, 8))#candidate|318|(8, 13, 8)|var|float32
bop_319 = relay.multiply(uop_316.astype('int16'), relay.reshape(var_318.astype('int16'), relay.shape_of(uop_316))) # shape=(8, 13, 8)
var_322 = relay.var("var_322", dtype = "float32", shape = (8, 13, 8))#candidate|322|(8, 13, 8)|var|float32
bop_323 = relay.logical_or(uop_316.astype('bool'), relay.reshape(var_322.astype('bool'), relay.shape_of(uop_316))) # shape=(8, 13, 8)
uop_326 = relay.sigmoid(const_305.astype('float64')) # shape=(36,)
bop_328 = relay.divide(uop_316.astype('float64'), relay.reshape(uop_309.astype('float64'), relay.shape_of(uop_316))) # shape=(8, 13, 8)
bop_331 = relay.not_equal(bop_328.astype('bool'), relay.reshape(var_322.astype('bool'), relay.shape_of(bop_328))) # shape=(8, 13, 8)
uop_334 = relay.asinh(uop_309.astype('float64')) # shape=(8, 13, 8)
uop_336 = relay.sqrt(uop_334.astype('float32')) # shape=(8, 13, 8)
bop_338 = relay.less(bop_323.astype('bool'), relay.reshape(uop_314.astype('bool'), relay.shape_of(bop_323))) # shape=(8, 13, 8)
bop_341 = relay.subtract(uop_334.astype('float32'), call_303.astype('float32')) # shape=(8, 13, 8)
bop_344 = relay.subtract(uop_334.astype('float32'), call_308.astype('float32')) # shape=(8, 13, 8)
bop_345 = relay.bitwise_xor(bop_338.astype('int16'), relay.reshape(bop_341.astype('int16'), relay.shape_of(bop_338))) # shape=(8, 13, 8)
bop_348 = relay.bitwise_xor(bop_338.astype('int16'), relay.reshape(bop_344.astype('int16'), relay.shape_of(bop_338))) # shape=(8, 13, 8)
uop_349 = relay.atanh(bop_338.astype('float32')) # shape=(8, 13, 8)
uop_351 = relay.sigmoid(uop_309.astype('float64')) # shape=(8, 13, 8)
bop_353 = relay.logical_and(uop_349.astype('bool'), relay.reshape(var_322.astype('bool'), relay.shape_of(uop_349))) # shape=(8, 13, 8)
bop_356 = relay.minimum(bop_331.astype('int32'), call_303.astype('int32')) # shape=(8, 13, 8)
bop_359 = relay.minimum(bop_331.astype('int32'), call_308.astype('int32')) # shape=(8, 13, 8)
output = relay.Tuple([const_304,var_306,const_307,bop_311,bop_319,uop_326,uop_336,bop_345,uop_351,bop_353,bop_356,])
output2 = relay.Tuple([const_304,var_306,const_307,bop_311,bop_319,uop_326,uop_336,bop_348,uop_351,bop_353,bop_359,])
func_360 = relay.Function([var_299,var_306,var_318,var_322,], output)
mod['func_360'] = func_360
mod = relay.transform.InferType()(mod)
mutated_mod['func_360'] = func_360
mutated_mod = relay.transform.InferType()(mutated_mod)
func_360_call = mutated_mod.get_global_var('func_360')
var_362 = relay.var("var_362", dtype = "float64", shape = (8, 13, 8))#candidate|362|(8, 13, 8)|var|float64
var_363 = relay.var("var_363", dtype = "uint16", shape = (1, 10))#candidate|363|(1, 10)|var|uint16
var_364 = relay.var("var_364", dtype = "float32", shape = (8, 13, 8))#candidate|364|(8, 13, 8)|var|float32
var_365 = relay.var("var_365", dtype = "float32", shape = (8, 13, 8))#candidate|365|(8, 13, 8)|var|float32
call_361 = func_360_call(var_362,var_363,var_364,var_365,)
output = call_361
func_366 = relay.Function([var_362,var_363,var_364,var_365,], output)
mutated_mod['func_366'] = func_366
mutated_mod = relay.transform.InferType()(mutated_mod)
const_368 = relay.const([[[-1.340496,-6.585827],[3.228117,1.835519],[-2.185975,3.362726],[-8.361772,-2.508593],[1.355981,-9.872181],[4.644009,-6.172785],[-5.276940,4.649225],[2.560759,-5.868589],[-9.162121,5.388810]],[[-2.211409,5.481276],[-4.419329,5.146520],[-0.517480,-5.220584],[-7.796672,0.246648],[7.013198,-1.979377],[7.508131,-5.316846],[-1.220723,-7.628741],[-2.412295,-6.147104],[-4.617197,9.542810]],[[7.910162,-6.812843],[-0.428866,0.295073],[3.295310,2.796225],[-4.254104,-3.602394],[6.476404,8.061752],[7.139844,4.470166],[5.270409,7.831369],[8.946034,-6.624524],[5.673447,-3.151951]],[[3.353849,2.216459],[-2.385867,8.220035],[-3.728875,-2.138387],[5.708068,3.313705],[0.403593,-8.439223],[-8.644327,3.770343],[-1.239978,-5.229103],[-5.169237,3.985635],[-1.514085,-6.125087]],[[3.039451,-0.873583],[-4.272662,-4.718003],[-6.825359,-9.366933],[-9.927662,-0.775895],[-5.596394,-1.689871],[-7.198336,6.605989],[5.113579,-8.011192],[8.678045,-7.697115],[2.604458,-7.854606]],[[7.299742,8.379054],[-5.764564,5.051028],[-4.878923,5.789015],[6.586625,8.515825],[2.717223,-7.452724],[2.474914,-6.271691],[-8.263010,-4.548747],[5.142383,8.355456],[3.249894,-6.152016]],[[-1.762101,-8.367735],[-3.033934,-1.062065],[5.880621,0.917578],[-4.797323,6.568035],[-2.643205,-6.327452],[-0.180680,-1.172571],[4.978771,-8.966517],[-5.315556,-8.237054],[-8.030564,7.961938]],[[-5.626405,9.811373],[3.637676,-2.206927],[-4.529528,-5.721815],[-1.847855,8.043782],[3.842623,-9.823259],[8.858067,5.290689],[-8.302913,4.117061],[-7.440153,9.934905],[-1.556471,-5.306490]],[[8.225818,5.864806],[-4.640436,-6.288588],[-0.174307,4.060014],[-5.380206,-8.598828],[-4.217805,8.336255],[6.238273,8.210108],[-4.727458,-5.585587],[1.057667,-0.616143],[-8.045890,2.169562]],[[6.491051,5.981575],[2.241425,9.852786],[-7.696467,-5.248016],[2.116069,9.133048],[-4.169460,-4.947323],[6.695843,4.250195],[9.773184,1.891980],[0.839336,-4.656634],[4.232250,-0.366447]],[[-4.854456,-0.610901],[3.335100,0.616418],[-7.475261,3.486269],[-5.372724,8.110573],[-5.166803,-7.463320],[8.813142,2.695964],[-0.937400,9.352564],[3.940714,1.199340],[-8.381687,6.099004]],[[-7.998371,-4.772905],[3.472508,3.228622],[-7.327623,-4.952900],[3.543899,-7.171231],[-9.749301,2.805303],[-8.383372,-0.461374],[-6.477168,0.197533],[2.811589,-1.190664],[-5.798307,8.605622]],[[-4.191454,0.303928],[1.280438,9.514593],[-5.006031,6.367293],[-4.551752,-8.990838],[-7.292483,-7.971801],[-3.209911,4.808780],[-9.962135,5.331201],[-8.538493,0.040987],[2.667758,7.378830]],[[-1.358611,4.381196],[2.939972,3.553799],[8.382741,-6.463079],[3.921872,-8.374684],[-7.696746,8.431202],[4.336993,2.811540],[2.098515,-7.489637],[9.292428,-8.342513],[-2.510838,5.029027]]], dtype = "float32")#candidate|368|(14, 9, 2)|const|float32
uop_369 = relay.sqrt(const_368.astype('float32')) # shape=(14, 9, 2)
uop_371 = relay.sin(const_368.astype('float32')) # shape=(14, 9, 2)
bop_373 = relay.right_shift(uop_369.astype('uint16'), relay.reshape(const_368.astype('uint16'), relay.shape_of(uop_369))) # shape=(14, 9, 2)
bop_376 = relay.power(const_368.astype('float64'), relay.reshape(uop_369.astype('float64'), relay.shape_of(const_368))) # shape=(14, 9, 2)
uop_379 = relay.atanh(const_368.astype('float32')) # shape=(14, 9, 2)
bop_381 = relay.bitwise_or(uop_371.astype('int32'), relay.reshape(uop_369.astype('int32'), relay.shape_of(uop_371))) # shape=(14, 9, 2)
bop_384 = relay.less_equal(uop_379.astype('bool'), relay.reshape(uop_369.astype('bool'), relay.shape_of(uop_379))) # shape=(14, 9, 2)
output = relay.Tuple([bop_373,bop_376,bop_381,bop_384,])
output2 = relay.Tuple([bop_373,bop_376,bop_381,bop_384,])
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
	relay.transform.Legalize(),
	relay.transform.FoldConstant(),
	relay.transform.ToANormalForm(),
	relay.transform.ToGraphNormalForm(),
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

'''9: TVMFuncCall
8: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
7: tvm::transform::Pass::operator()(tvm::IRModule) const
6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
5: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::LazyGradientInit(tvm::RelayExpr const&, tvm::IRModule)
0: tvm::relay::CheckFeature(tvm::RelayExpr const&, tvm::relay::FeatureSet const&)

'''