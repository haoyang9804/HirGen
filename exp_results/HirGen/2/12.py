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
var_0 = relay.var("var_0", dtype = "float32", shape = (3, 2, 12))#candidate|0|(3, 2, 12)|var|float32
const_1 = relay.const([[[-7.997367,8.652309,6.758070,-7.673485,-9.019416,-8.530509,-0.391519,-6.497870,-3.989492,5.333754,-2.529445,-0.940964],[-3.616881,6.988435,6.332693,0.127444,2.450501,-4.769689,9.348229,3.099047,7.323495,-6.745430,-5.645373,5.538822]],[[-4.247633,7.644397,-0.053812,-1.560361,-3.628414,-9.914847,-2.790894,-7.741853,-8.148697,-6.289213,3.113011,5.834331],[-9.081057,-5.515311,-1.629981,8.973008,1.849537,5.912907,-3.365724,-0.517586,-3.276038,8.123975,-2.447672,-6.696353]],[[-3.060784,-9.132368,6.008446,6.310732,4.541112,-4.610754,8.958120,2.248533,-9.971239,5.089262,-6.209713,-9.543857],[-2.166336,-2.707541,3.591428,0.103852,5.494441,9.724151,7.512142,-4.112496,5.892132,-2.012630,4.484925,4.195910]]], dtype = "float32")#candidate|1|(3, 2, 12)|const|float32
bop_2 = relay.not_equal(var_0.astype('bool'), relay.reshape(const_1.astype('bool'), relay.shape_of(var_0))) # shape=(3, 2, 12)
var_5 = relay.var("var_5", dtype = "float32", shape = (3, 2, 12))#candidate|5|(3, 2, 12)|var|float32
bop_6 = relay.equal(const_1.astype('bool'), relay.reshape(var_5.astype('bool'), relay.shape_of(const_1))) # shape=(3, 2, 12)
bop_9 = relay.logical_and(var_5.astype('bool'), relay.reshape(const_1.astype('bool'), relay.shape_of(var_5))) # shape=(3, 2, 12)
bop_12 = relay.left_shift(const_1.astype('uint16'), relay.reshape(var_0.astype('uint16'), relay.shape_of(const_1))) # shape=(3, 2, 12)
uop_15 = relay.asin(bop_12.astype('float64')) # shape=(3, 2, 12)
uop_17 = relay.atan(const_1.astype('float64')) # shape=(3, 2, 12)
uop_19 = relay.cos(uop_17.astype('float32')) # shape=(3, 2, 12)
uop_21 = relay.asinh(uop_15.astype('float32')) # shape=(3, 2, 12)
bop_23 = relay.equal(uop_19.astype('bool'), relay.reshape(bop_12.astype('bool'), relay.shape_of(uop_19))) # shape=(3, 2, 12)
var_26 = relay.var("var_26", dtype = "float32", shape = (3, 2, 12))#candidate|26|(3, 2, 12)|var|float32
bop_27 = relay.equal(uop_21.astype('bool'), relay.reshape(var_26.astype('bool'), relay.shape_of(uop_21))) # shape=(3, 2, 12)
uop_30 = relay.acosh(uop_17.astype('float32')) # shape=(3, 2, 12)
var_32 = relay.var("var_32", dtype = "float32", shape = (3, 2, 12))#candidate|32|(3, 2, 12)|var|float32
bop_33 = relay.logical_or(uop_30.astype('bool'), relay.reshape(var_32.astype('bool'), relay.shape_of(uop_30))) # shape=(3, 2, 12)
uop_36 = relay.erf(uop_17.astype('float64')) # shape=(3, 2, 12)
bop_38 = relay.bitwise_xor(uop_19.astype('uint16'), relay.reshape(bop_2.astype('uint16'), relay.shape_of(uop_19))) # shape=(3, 2, 12)
uop_41 = relay.cos(uop_30.astype('float32')) # shape=(3, 2, 12)
uop_43 = relay.rsqrt(uop_36.astype('float64')) # shape=(3, 2, 12)
bop_45 = relay.not_equal(var_32.astype('bool'), relay.reshape(bop_12.astype('bool'), relay.shape_of(var_32))) # shape=(3, 2, 12)
uop_48 = relay.sin(bop_23.astype('float32')) # shape=(3, 2, 12)
bop_50 = relay.bitwise_xor(uop_43.astype('uint8'), relay.reshape(bop_9.astype('uint8'), relay.shape_of(uop_43))) # shape=(3, 2, 12)
var_53 = relay.var("var_53", dtype = "bool", shape = (3, 2, 12))#candidate|53|(3, 2, 12)|var|bool
bop_54 = relay.divide(bop_27.astype('float32'), relay.reshape(var_53.astype('float32'), relay.shape_of(bop_27))) # shape=(3, 2, 12)
bop_57 = relay.subtract(bop_27.astype('int64'), relay.reshape(bop_2.astype('int64'), relay.shape_of(bop_27))) # shape=(3, 2, 12)
uop_60 = relay.cosh(uop_17.astype('float32')) # shape=(3, 2, 12)
uop_62 = relay.atan(uop_15.astype('float64')) # shape=(3, 2, 12)
bop_64 = relay.right_shift(bop_50.astype('int16'), relay.reshape(uop_15.astype('int16'), relay.shape_of(bop_50))) # shape=(3, 2, 12)
bop_67 = relay.logical_and(uop_17.astype('bool'), relay.reshape(bop_6.astype('bool'), relay.shape_of(uop_17))) # shape=(3, 2, 12)
uop_70 = relay.atanh(bop_67.astype('float64')) # shape=(3, 2, 12)
output = relay.Tuple([bop_33,bop_38,uop_41,bop_45,uop_48,bop_54,bop_57,uop_60,uop_62,bop_64,uop_70,])
output2 = relay.Tuple([bop_33,bop_38,uop_41,bop_45,uop_48,bop_54,bop_57,uop_60,uop_62,bop_64,uop_70,])
func_72 = relay.Function([var_0,var_5,var_26,var_32,var_53,], output)
mod['func_72'] = func_72
mod = relay.transform.InferType()(mod)
var_73 = relay.var("var_73", dtype = "float32", shape = (3, 2, 12))#candidate|73|(3, 2, 12)|var|float32
var_74 = relay.var("var_74", dtype = "float32", shape = (3, 2, 12))#candidate|74|(3, 2, 12)|var|float32
var_75 = relay.var("var_75", dtype = "float32", shape = (3, 2, 12))#candidate|75|(3, 2, 12)|var|float32
var_76 = relay.var("var_76", dtype = "float32", shape = (3, 2, 12))#candidate|76|(3, 2, 12)|var|float32
var_77 = relay.var("var_77", dtype = "bool", shape = (3, 2, 12))#candidate|77|(3, 2, 12)|var|bool
output = func_72(var_73,var_74,var_75,var_76,var_77,)
func_78 = relay.Function([var_73,var_74,var_75,var_76,var_77,], output)
mutated_mod['func_78'] = func_78
mutated_mod = relay.transform.InferType()(mutated_mod)
var_80 = relay.var("var_80", dtype = "float64", shape = (14,))#candidate|80|(14,)|var|float64
uop_81 = relay.asin(var_80.astype('float64')) # shape=(14,)
uop_83 = relay.exp(var_80.astype('float32')) # shape=(14,)
bop_85 = relay.equal(uop_83.astype('bool'), relay.reshape(uop_81.astype('bool'), relay.shape_of(uop_83))) # shape=(14,)
var_88 = relay.var("var_88", dtype = "float32", shape = (14,))#candidate|88|(14,)|var|float32
bop_89 = relay.logical_or(uop_83.astype('bool'), relay.reshape(var_88.astype('bool'), relay.shape_of(uop_83))) # shape=(14,)
var_92 = relay.var("var_92", dtype = "float32", shape = (14,))#candidate|92|(14,)|var|float32
bop_93 = relay.less(uop_83.astype('bool'), relay.reshape(var_92.astype('bool'), relay.shape_of(uop_83))) # shape=(14,)
uop_96 = relay.acosh(bop_85.astype('float32')) # shape=(14,)
func_72_call = mod.get_global_var('func_72')
func_78_call = mutated_mod.get_global_var('func_78')
const_99 = relay.const([4.872337,2.122954,9.515767,-5.224197,4.761046,0.293460,2.583874,-7.432500,3.371415,4.283813,6.370601,-2.925344,-7.896913,0.910977,7.340849,-1.452788,-6.022044,-9.246176,6.251487,3.640697,-1.171939,9.966486,-6.820031,-7.945772,-6.195965,-6.952728,1.000523,-7.445341,-3.884102,0.553338,9.701174,8.902645,8.453987,5.590576,-4.819624,2.760993,-9.358650,3.641742,-7.295431,-4.197046,8.190279,9.380048,8.450271,7.881634,2.852718,9.939501,-6.663903,-7.387362,-5.160000,-2.736930,-5.909907,-4.610070,7.526728,4.060954,8.922727,-8.215867,3.461330,8.581002,2.230530,-1.163648,4.500193,7.286296,-7.726021,2.748899,-6.309276,-5.264092,9.745351,4.327931,-4.398393,6.379591,-7.669587,-5.804308], dtype = "float32")#candidate|99|(72,)|const|float32
call_98 = relay.TupleGetItem(func_72_call(relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('bool'), [3, 2, 12]), ), 7)
call_100 = relay.TupleGetItem(func_78_call(relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('float32'), [3, 2, 12]), relay.reshape(const_99.astype('bool'), [3, 2, 12]), ), 7)
bop_101 = relay.floor_mod(uop_96.astype('float64'), relay.reshape(var_92.astype('float64'), relay.shape_of(uop_96))) # shape=(14,)
uop_104 = relay.asinh(uop_96.astype('float64')) # shape=(14,)
output = relay.Tuple([bop_89,bop_93,call_98,const_99,bop_101,uop_104,])
output2 = relay.Tuple([bop_89,bop_93,call_100,const_99,bop_101,uop_104,])
func_106 = relay.Function([var_80,var_88,var_92,], output)
mod['func_106'] = func_106
mod = relay.transform.InferType()(mod)
mutated_mod['func_106'] = func_106
mutated_mod = relay.transform.InferType()(mutated_mod)
func_106_call = mutated_mod.get_global_var('func_106')
var_108 = relay.var("var_108", dtype = "float64", shape = (14,))#candidate|108|(14,)|var|float64
var_109 = relay.var("var_109", dtype = "float32", shape = (14,))#candidate|109|(14,)|var|float32
var_110 = relay.var("var_110", dtype = "float32", shape = (14,))#candidate|110|(14,)|var|float32
call_107 = func_106_call(var_108,var_109,var_110,)
output = call_107
func_111 = relay.Function([var_108,var_109,var_110,], output)
mutated_mod['func_111'] = func_111
mutated_mod = relay.transform.InferType()(mutated_mod)
var_113 = relay.var("var_113", dtype = "float64", shape = (9, 1))#candidate|113|(9, 1)|var|float64
var_114 = relay.var("var_114", dtype = "float64", shape = (9, 15))#candidate|114|(9, 15)|var|float64
bop_115 = relay.mod(var_113.astype('float64'), var_114.astype('float64')) # shape=(9, 15)
bop_118 = relay.logical_or(var_113.astype('bool'), var_114.astype('bool')) # shape=(9, 15)
uop_121 = relay.sqrt(var_113.astype('float64')) # shape=(9, 1)
uop_123 = relay.exp(uop_121.astype('float32')) # shape=(9, 1)
uop_125 = relay.atan(var_113.astype('float64')) # shape=(9, 1)
var_127 = relay.var("var_127", dtype = "float32", shape = (9, 15))#candidate|127|(9, 15)|var|float32
bop_128 = relay.logical_xor(uop_123.astype('int8'), var_127.astype('int8')) # shape=(9, 15)
uop_131 = relay.sigmoid(uop_123.astype('float64')) # shape=(9, 1)
bop_133 = relay.bitwise_and(uop_131.astype('int8'), var_127.astype('int8')) # shape=(9, 15)
uop_136 = relay.sqrt(uop_121.astype('float64')) # shape=(9, 1)
bop_138 = relay.power(uop_125.astype('float32'), var_114.astype('float32')) # shape=(9, 15)
uop_141 = relay.cos(bop_133.astype('float32')) # shape=(9, 15)
uop_143 = relay.sqrt(uop_131.astype('float32')) # shape=(9, 1)
uop_145 = relay.sinh(uop_143.astype('float64')) # shape=(9, 1)
var_147 = relay.var("var_147", dtype = "float64", shape = (9, 14))#candidate|147|(9, 14)|var|float64
bop_148 = relay.greater(uop_136.astype('bool'), var_147.astype('bool')) # shape=(9, 14)
bop_151 = relay.logical_or(uop_145.astype('bool'), uop_141.astype('bool')) # shape=(9, 15)
bop_154 = relay.less(uop_143.astype('bool'), bop_115.astype('bool')) # shape=(9, 15)
bop_157 = relay.left_shift(uop_136.astype('uint16'), bop_133.astype('uint16')) # shape=(9, 15)
var_160 = relay.var("var_160", dtype = "bool", shape = (9, 15))#candidate|160|(9, 15)|var|bool
bop_161 = relay.greater_equal(bop_154.astype('bool'), relay.reshape(var_160.astype('bool'), relay.shape_of(bop_154))) # shape=(9, 15)
bop_164 = relay.maximum(uop_141.astype('uint32'), var_113.astype('uint32')) # shape=(9, 15)
output = relay.Tuple([bop_118,bop_128,bop_138,bop_148,bop_151,bop_157,bop_161,bop_164,])
output2 = relay.Tuple([bop_118,bop_128,bop_138,bop_148,bop_151,bop_157,bop_161,bop_164,])
func_167 = relay.Function([var_113,var_114,var_127,var_147,var_160,], output)
mod['func_167'] = func_167
mod = relay.transform.InferType()(mod)
mutated_mod['func_167'] = func_167
mutated_mod = relay.transform.InferType()(mutated_mod)
func_167_call = mutated_mod.get_global_var('func_167')
var_169 = relay.var("var_169", dtype = "float64", shape = (9, 1))#candidate|169|(9, 1)|var|float64
var_170 = relay.var("var_170", dtype = "float64", shape = (9, 15))#candidate|170|(9, 15)|var|float64
var_171 = relay.var("var_171", dtype = "float32", shape = (9, 15))#candidate|171|(9, 15)|var|float32
var_172 = relay.var("var_172", dtype = "float64", shape = (9, 14))#candidate|172|(9, 14)|var|float64
var_173 = relay.var("var_173", dtype = "bool", shape = (9, 15))#candidate|173|(9, 15)|var|bool
call_168 = func_167_call(var_169,var_170,var_171,var_172,var_173,)
output = call_168
func_174 = relay.Function([var_169,var_170,var_171,var_172,var_173,], output)
mutated_mod['func_174'] = func_174
mutated_mod = relay.transform.InferType()(mutated_mod)
const_176 = relay.const([[-2,6,10,-10,4,8,-1,9,2,-2,-1,8,10,8,1],[3,-4,5,-8,6,6,-1,-9,-3,-4,-7,-1,9,-5,5],[8,1,8,5,1,-9,9,4,-2,1,1,1,5,3,6],[-7,-5,-8,4,-10,-7,2,9,-6,-2,-7,4,-2,5,-8],[3,-7,4,3,1,3,-2,-5,-2,4,-7,6,8,-10,-3],[-5,-6,7,10,-9,3,10,-8,8,6,-5,-2,4,-1,2],[7,-2,-2,2,-1,4,5,-9,8,5,9,5,-1,1,5]], dtype = "uint16")#candidate|176|(7, 15)|const|uint16
const_177 = relay.const([[1,-9,-2,-7,8,7,3,-3,8,2,2,-7,-3,4,-3],[9,8,9,7,-6,-5,8,4,-7,-5,-4,-8,-1,9,6],[4,-6,-5,6,-10,9,-7,3,2,6,7,-4,2,-10,3],[-1,-1,-3,1,-10,-2,-4,10,2,7,7,7,-3,-9,2],[-8,8,1,8,3,-10,-1,-5,4,-2,5,-9,3,3,-6],[1,-9,5,-4,-9,5,1,-8,6,4,-4,-7,1,-6,1],[9,-6,-5,-8,8,-6,-9,-8,1,-8,1,-4,-7,-8,9]], dtype = "uint16")#candidate|177|(7, 15)|const|uint16
bop_178 = relay.left_shift(const_176.astype('uint16'), relay.reshape(const_177.astype('uint16'), relay.shape_of(const_176))) # shape=(7, 15)
output = relay.Tuple([bop_178,])
output2 = relay.Tuple([bop_178,])
func_181 = relay.Function([], output)
mod['func_181'] = func_181
mod = relay.transform.InferType()(mod)
mutated_mod['func_181'] = func_181
mutated_mod = relay.transform.InferType()(mutated_mod)
func_181_call = mutated_mod.get_global_var('func_181')
call_182 = func_181_call()
output = call_182
func_183 = relay.Function([], output)
mutated_mod['func_183'] = func_183
mutated_mod = relay.transform.InferType()(mutated_mod)
var_184 = relay.var("var_184", dtype = "float64", shape = ())#candidate|184|()|var|float64
var_185 = relay.var("var_185", dtype = "float64", shape = (5, 12, 12))#candidate|185|(5, 12, 12)|var|float64
bop_186 = relay.divide(var_184.astype('float64'), var_185.astype('float64')) # shape=(5, 12, 12)
bop_189 = relay.greater_equal(var_184.astype('bool'), bop_186.astype('bool')) # shape=(5, 12, 12)
uop_192 = relay.asinh(bop_189.astype('float64')) # shape=(5, 12, 12)
bop_194 = relay.divide(uop_192.astype('float32'), relay.reshape(bop_186.astype('float32'), relay.shape_of(uop_192))) # shape=(5, 12, 12)
bop_197 = relay.divide(var_185.astype('float64'), var_184.astype('float64')) # shape=(5, 12, 12)
var_200 = relay.var("var_200", dtype = "float32", shape = (5, 12, 12))#candidate|200|(5, 12, 12)|var|float32
bop_201 = relay.multiply(bop_194.astype('float32'), relay.reshape(var_200.astype('float32'), relay.shape_of(bop_194))) # shape=(5, 12, 12)
uop_204 = relay.sqrt(bop_201.astype('float64')) # shape=(5, 12, 12)
bop_206 = relay.bitwise_and(uop_204.astype('uint8'), relay.reshape(bop_194.astype('uint8'), relay.shape_of(uop_204))) # shape=(5, 12, 12)
uop_209 = relay.acosh(uop_204.astype('float64')) # shape=(5, 12, 12)
func_72_call = mod.get_global_var('func_72')
func_78_call = mutated_mod.get_global_var('func_78')
const_212 = relay.const([-9.801376,0.163870,9.725832,-9.626242,5.688488,-8.686155,2.694939,-1.814293,-7.510829,-4.133517,-0.193058,-8.649872,7.488973,8.113464,-6.811699,-0.383560,0.283779,8.517698,0.927392,-0.941292,5.774040,-7.870338,-5.962374,5.413120,-0.644873,4.320907,-5.311441,-4.554600,6.728870,-4.439854,-2.939873,-5.147722,-0.378544,-3.877865,8.271194,1.706012,6.450370,8.823807,-0.065542,-5.454889,4.337106,-1.190363,-7.203955,5.962135,-3.980457,4.227708,2.953856,-1.744320,-0.290541,1.220779,1.183556,-9.429071,-8.093956,-0.903371,-8.318925,7.935209,8.870456,0.367037,0.642753,7.609835,-4.321737,-6.172086,-2.307357,-8.146503,5.986410,7.589446,-3.206874,6.774411,7.371273,5.049905,-6.952171,4.503781], dtype = "float32")#candidate|212|(72,)|const|float32
call_211 = relay.TupleGetItem(func_72_call(relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('bool'), [3, 2, 12]), ), 9)
call_213 = relay.TupleGetItem(func_78_call(relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('float32'), [3, 2, 12]), relay.reshape(const_212.astype('bool'), [3, 2, 12]), ), 9)
uop_214 = relay.cosh(uop_209.astype('float32')) # shape=(5, 12, 12)
bop_216 = relay.mod(uop_214.astype('float32'), relay.reshape(var_185.astype('float32'), relay.shape_of(uop_214))) # shape=(5, 12, 12)
bop_219 = relay.multiply(uop_209.astype('int32'), relay.reshape(bop_206.astype('int32'), relay.shape_of(uop_209))) # shape=(5, 12, 12)
bop_222 = relay.left_shift(bop_219.astype('uint64'), relay.reshape(uop_214.astype('uint64'), relay.shape_of(bop_219))) # shape=(5, 12, 12)
output = relay.Tuple([bop_197,call_211,const_212,bop_216,bop_222,])
output2 = relay.Tuple([bop_197,call_213,const_212,bop_216,bop_222,])
func_225 = relay.Function([var_184,var_185,var_200,], output)
mod['func_225'] = func_225
mod = relay.transform.InferType()(mod)
mutated_mod['func_225'] = func_225
mutated_mod = relay.transform.InferType()(mutated_mod)
func_225_call = mutated_mod.get_global_var('func_225')
var_227 = relay.var("var_227", dtype = "float64", shape = ())#candidate|227|()|var|float64
var_228 = relay.var("var_228", dtype = "float64", shape = (5, 12, 12))#candidate|228|(5, 12, 12)|var|float64
var_229 = relay.var("var_229", dtype = "float32", shape = (5, 12, 12))#candidate|229|(5, 12, 12)|var|float32
call_226 = func_225_call(var_227,var_228,var_229,)
output = call_226
func_230 = relay.Function([var_227,var_228,var_229,], output)
mutated_mod['func_230'] = func_230
mutated_mod = relay.transform.InferType()(mutated_mod)
func_181_call = mod.get_global_var('func_181')
func_183_call = mutated_mod.get_global_var('func_183')
call_232 = relay.TupleGetItem(func_181_call(), 0)
call_233 = relay.TupleGetItem(func_183_call(), 0)
const_234 = relay.const([[5,-8,-3,1,10,1,8,1,3,-6,4,4,-10,5,1],[9,4,-10,-10,8,4,9,-7,5,8,10,-9,-4,-9,5],[2,7,-1,-7,-7,-3,4,-3,10,-10,-9,10,7,7,6],[-7,5,-8,-2,-1,8,9,4,3,-10,-4,-5,9,2,-9],[7,7,-9,-10,10,1,-5,-1,2,5,-1,3,7,10,-5],[8,-3,5,1,-5,9,10,6,-5,5,2,-10,2,8,-7],[-8,-6,1,7,7,3,-9,9,2,6,3,5,-5,-1,8]], dtype = "uint16")#candidate|234|(7, 15)|const|uint16
bop_235 = relay.not_equal(call_232.astype('bool'), relay.reshape(const_234.astype('bool'), relay.shape_of(call_232))) # shape=(7, 15)
bop_238 = relay.not_equal(call_233.astype('bool'), relay.reshape(const_234.astype('bool'), relay.shape_of(call_233))) # shape=(7, 15)
output = relay.Tuple([bop_235,])
output2 = relay.Tuple([bop_238,])
func_239 = relay.Function([], output)
mod['func_239'] = func_239
mod = relay.transform.InferType()(mod)
output = func_239()
func_240 = relay.Function([], output)
mutated_mod['func_240'] = func_240
mutated_mod = relay.transform.InferType()(mutated_mod)
const_241 = relay.const([[[7.697069,-5.413622],[3.213494,-2.935537],[-4.838054,-1.504578],[-3.183559,4.581187],[-6.833431,1.272587],[5.535927,2.678182],[4.170670,-4.813511],[7.876719,3.741138],[1.434910,3.320293],[3.305993,2.572124],[8.304731,-4.499780]],[[-4.740127,-7.335811],[-8.460961,-0.345121],[0.161270,3.915264],[0.989696,-6.847613],[2.264050,6.263143],[-4.261172,-4.102581],[-7.540151,5.412127],[3.266962,5.774290],[9.633789,-3.562657],[6.223669,-2.379964],[-5.628162,-3.752395]],[[2.700242,-3.155028],[-6.300294,4.332688],[-4.710639,-2.405555],[7.343705,-4.507550],[2.710074,-8.296651],[5.184468,9.314046],[-5.212173,3.191363],[1.549608,-3.180171],[5.380249,5.241707],[-5.229506,-6.792736],[-3.958557,-5.945059]]], dtype = "float64")#candidate|241|(3, 11, 2)|const|float64
uop_242 = relay.log(const_241.astype('float64')) # shape=(3, 11, 2)
bop_244 = relay.bitwise_or(uop_242.astype('uint16'), relay.reshape(const_241.astype('uint16'), relay.shape_of(uop_242))) # shape=(3, 11, 2)
uop_247 = relay.sigmoid(const_241.astype('float64')) # shape=(3, 11, 2)
bop_249 = relay.logical_and(const_241.astype('bool'), relay.reshape(bop_244.astype('bool'), relay.shape_of(const_241))) # shape=(3, 11, 2)
uop_252 = relay.atan(const_241.astype('float64')) # shape=(3, 11, 2)
var_254 = relay.var("var_254", dtype = "float64", shape = (3, 11, 2))#candidate|254|(3, 11, 2)|var|float64
bop_255 = relay.add(const_241.astype('int8'), relay.reshape(var_254.astype('int8'), relay.shape_of(const_241))) # shape=(3, 11, 2)
bop_258 = relay.less_equal(uop_252.astype('bool'), relay.reshape(bop_249.astype('bool'), relay.shape_of(uop_252))) # shape=(3, 11, 2)
bop_261 = relay.right_shift(uop_242.astype('int8'), relay.reshape(bop_244.astype('int8'), relay.shape_of(uop_242))) # shape=(3, 11, 2)
uop_264 = relay.erf(uop_252.astype('float32')) # shape=(3, 11, 2)
uop_266 = relay.log2(uop_252.astype('float32')) # shape=(3, 11, 2)
var_268 = relay.var("var_268", dtype = "float64", shape = (3, 11, 2))#candidate|268|(3, 11, 2)|var|float64
bop_269 = relay.bitwise_and(var_254.astype('uint32'), relay.reshape(var_268.astype('uint32'), relay.shape_of(var_254))) # shape=(3, 11, 2)
bop_272 = relay.greater(bop_261.astype('bool'), relay.reshape(uop_252.astype('bool'), relay.shape_of(bop_261))) # shape=(3, 11, 2)
uop_275 = relay.tan(bop_255.astype('float64')) # shape=(3, 11, 2)
bop_277 = relay.floor_divide(bop_272.astype('float64'), relay.reshape(uop_275.astype('float64'), relay.shape_of(bop_272))) # shape=(3, 11, 2)
bop_280 = relay.right_shift(uop_242.astype('uint8'), relay.reshape(bop_249.astype('uint8'), relay.shape_of(uop_242))) # shape=(3, 11, 2)
var_283 = relay.var("var_283", dtype = "float32", shape = (3, 11, 2))#candidate|283|(3, 11, 2)|var|float32
bop_284 = relay.less_equal(uop_264.astype('bool'), relay.reshape(var_283.astype('bool'), relay.shape_of(uop_264))) # shape=(3, 11, 2)
var_287 = relay.var("var_287", dtype = "float32", shape = (3, 11, 2))#candidate|287|(3, 11, 2)|var|float32
bop_288 = relay.power(uop_264.astype('float64'), relay.reshape(var_287.astype('float64'), relay.shape_of(uop_264))) # shape=(3, 11, 2)
uop_291 = relay.tan(uop_266.astype('float32')) # shape=(3, 11, 2)
bop_293 = relay.less_equal(bop_258.astype('bool'), relay.reshape(uop_242.astype('bool'), relay.shape_of(bop_258))) # shape=(3, 11, 2)
output = relay.Tuple([uop_247,bop_269,bop_277,bop_280,bop_284,bop_288,uop_291,bop_293,])
output2 = relay.Tuple([uop_247,bop_269,bop_277,bop_280,bop_284,bop_288,uop_291,bop_293,])
func_296 = relay.Function([var_254,var_268,var_283,var_287,], output)
mod['func_296'] = func_296
mod = relay.transform.InferType()(mod)
var_297 = relay.var("var_297", dtype = "float64", shape = (3, 11, 2))#candidate|297|(3, 11, 2)|var|float64
var_298 = relay.var("var_298", dtype = "float64", shape = (3, 11, 2))#candidate|298|(3, 11, 2)|var|float64
var_299 = relay.var("var_299", dtype = "float32", shape = (3, 11, 2))#candidate|299|(3, 11, 2)|var|float32
var_300 = relay.var("var_300", dtype = "float32", shape = (3, 11, 2))#candidate|300|(3, 11, 2)|var|float32
output = func_296(var_297,var_298,var_299,var_300,)
func_301 = relay.Function([var_297,var_298,var_299,var_300,], output)
mutated_mod['func_301'] = func_301
mutated_mod = relay.transform.InferType()(mutated_mod)
var_303 = relay.var("var_303", dtype = "bool", shape = (6, 12, 13))#candidate|303|(6, 12, 13)|var|bool
var_304 = relay.var("var_304", dtype = "bool", shape = (6, 12, 13))#candidate|304|(6, 12, 13)|var|bool
bop_305 = relay.logical_or(var_303.astype('bool'), relay.reshape(var_304.astype('bool'), relay.shape_of(var_303))) # shape=(6, 12, 13)
uop_308 = relay.cosh(var_303.astype('float32')) # shape=(6, 12, 13)
uop_310 = relay.tan(uop_308.astype('float64')) # shape=(6, 12, 13)
uop_312 = relay.asin(uop_310.astype('float64')) # shape=(6, 12, 13)
bop_314 = relay.add(uop_312.astype('float64'), relay.reshape(var_304.astype('float64'), relay.shape_of(uop_312))) # shape=(6, 12, 13)
bop_317 = relay.divide(uop_312.astype('float64'), relay.reshape(uop_310.astype('float64'), relay.shape_of(uop_312))) # shape=(6, 12, 13)
uop_320 = relay.erf(uop_312.astype('float64')) # shape=(6, 12, 13)
bop_322 = relay.subtract(uop_320.astype('uint32'), relay.reshape(var_304.astype('uint32'), relay.shape_of(uop_320))) # shape=(6, 12, 13)
uop_325 = relay.atanh(uop_320.astype('float32')) # shape=(6, 12, 13)
output = relay.Tuple([bop_305,bop_314,bop_317,bop_322,uop_325,])
output2 = relay.Tuple([bop_305,bop_314,bop_317,bop_322,uop_325,])
func_327 = relay.Function([var_303,var_304,], output)
mod['func_327'] = func_327
mod = relay.transform.InferType()(mod)
mutated_mod['func_327'] = func_327
mutated_mod = relay.transform.InferType()(mutated_mod)
func_327_call = mutated_mod.get_global_var('func_327')
var_329 = relay.var("var_329", dtype = "bool", shape = (6, 12, 13))#candidate|329|(6, 12, 13)|var|bool
var_330 = relay.var("var_330", dtype = "bool", shape = (6, 12, 13))#candidate|330|(6, 12, 13)|var|bool
call_328 = func_327_call(var_329,var_330,)
output = call_328
func_331 = relay.Function([var_329,var_330,], output)
mutated_mod['func_331'] = func_331
mutated_mod = relay.transform.InferType()(mutated_mod)
var_333 = relay.var("var_333", dtype = "float32", shape = ())#candidate|333|()|var|float32
uop_334 = relay.sinh(var_333.astype('float32')) # shape=()
bop_336 = relay.less_equal(uop_334.astype('bool'), var_333.astype('bool')) # shape=()
uop_339 = relay.acosh(bop_336.astype('float64')) # shape=()
bop_341 = relay.multiply(uop_339.astype('uint8'), uop_334.astype('uint8')) # shape=()
output = bop_341
output2 = bop_341
F = relay.Function([var_333,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_333,], output2)
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
input_333= np.array(1.444853, dtype='float32')
module1.set_input('var_333', input_333)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_333, )
res3 = intrp3.evaluate()(input_333, )
res4 = intrp4.evaluate()(input_333, )
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
module5.set_input('var_333', input_333)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_333, )
res7 = intrp7.evaluate()(input_333, )
res8 = intrp8.evaluate()(input_333, )
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
module9.set_input('var_333', input_333)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_333, )
res11 = intrp11.evaluate()(input_333, )
res12 = intrp12.evaluate()(input_333, )
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
module13.set_input('var_333', input_333)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_333, )
res15 = intrp15.evaluate()(input_333, )
res16 = intrp16.evaluate()(input_333, )
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
module17.set_input('var_333', input_333)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_333, )
res19 = intrp19.evaluate()(input_333, )
res20 = intrp20.evaluate()(input_333, )
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
module21.set_input('var_333', input_333)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_333, )
res23 = intrp23.evaluate()(input_333, )
res24 = intrp24.evaluate()(input_333, )
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