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
var_0 = relay.var("var_0", dtype = "float64", shape = (11, 2, 3))#candidate|0|(11, 2, 3)|var|float64
var_1 = relay.var("var_1", dtype = "float64", shape = (11, 2, 3))#candidate|1|(11, 2, 3)|var|float64
bop_2 = relay.floor_mod(var_0.astype('float64'), relay.reshape(var_1.astype('float64'), relay.shape_of(var_0))) # shape=(11, 2, 3)
var_5 = relay.var("var_5", dtype = "float64", shape = (11, 2, 3))#candidate|5|(11, 2, 3)|var|float64
bop_6 = relay.subtract(var_1.astype('int16'), relay.reshape(var_5.astype('int16'), relay.shape_of(var_1))) # shape=(11, 2, 3)
bop_9 = relay.logical_and(var_5.astype('bool'), relay.reshape(bop_2.astype('bool'), relay.shape_of(var_5))) # shape=(11, 2, 3)
bop_12 = relay.not_equal(var_5.astype('bool'), relay.reshape(bop_9.astype('bool'), relay.shape_of(var_5))) # shape=(11, 2, 3)
bop_15 = relay.equal(bop_12.astype('bool'), relay.reshape(var_5.astype('bool'), relay.shape_of(bop_12))) # shape=(11, 2, 3)
output = relay.Tuple([bop_6,bop_15,])
output2 = relay.Tuple([bop_6,bop_15,])
func_18 = relay.Function([var_0,var_1,var_5,], output)
mod['func_18'] = func_18
mod = relay.transform.InferType()(mod)
mutated_mod['func_18'] = func_18
mutated_mod = relay.transform.InferType()(mutated_mod)
func_18_call = mutated_mod.get_global_var('func_18')
var_20 = relay.var("var_20", dtype = "float64", shape = (11, 2, 3))#candidate|20|(11, 2, 3)|var|float64
var_21 = relay.var("var_21", dtype = "float64", shape = (11, 2, 3))#candidate|21|(11, 2, 3)|var|float64
var_22 = relay.var("var_22", dtype = "float64", shape = (11, 2, 3))#candidate|22|(11, 2, 3)|var|float64
call_19 = func_18_call(var_20,var_21,var_22,)
output = call_19
func_23 = relay.Function([var_20,var_21,var_22,], output)
mutated_mod['func_23'] = func_23
mutated_mod = relay.transform.InferType()(mutated_mod)
const_25 = relay.const(False, dtype = "bool")#candidate|25|()|const|bool
var_26 = relay.var("var_26", dtype = "bool", shape = ())#candidate|26|()|var|bool
bop_27 = relay.logical_or(const_25.astype('bool'), var_26.astype('bool')) # shape=()
uop_30 = relay.log(const_25.astype('float64')) # shape=()
output = relay.Tuple([bop_27,uop_30,])
output2 = relay.Tuple([bop_27,uop_30,])
func_32 = relay.Function([var_26,], output)
mod['func_32'] = func_32
mod = relay.transform.InferType()(mod)
mutated_mod['func_32'] = func_32
mutated_mod = relay.transform.InferType()(mutated_mod)
var_33 = relay.var("var_33", dtype = "bool", shape = ())#candidate|33|()|var|bool
func_32_call = mutated_mod.get_global_var('func_32')
call_34 = func_32_call(var_33)
output = call_34
func_35 = relay.Function([var_33], output)
mutated_mod['func_35'] = func_35
mutated_mod = relay.transform.InferType()(mutated_mod)
var_37 = relay.var("var_37", dtype = "int16", shape = (7, 2, 16))#candidate|37|(7, 2, 16)|var|int16
const_38 = relay.const([[[10,2,6,-9,3,10,1,-10,6,8,1,6,9,-10,3,-4],[5,-8,10,-1,7,10,5,-6,3,1,-7,10,10,-8,-5,6]],[[10,9,1,10,-1,6,10,9,6,-9,-6,9,-10,-9,-1,3],[6,1,3,-5,-4,2,-3,-7,-2,1,7,4,-9,2,-5,-1]],[[-8,10,6,8,6,-3,-5,-9,-9,5,-2,-3,9,3,10,-6],[-8,2,4,3,-3,1,7,5,10,-1,-8,-5,8,2,6,8]],[[6,8,-1,-10,-7,4,-10,-9,-8,7,5,-2,2,2,-4,1],[-2,2,10,-2,4,3,3,-8,-2,-7,-7,-3,5,-4,-8,10]],[[4,5,-5,-4,-3,7,-9,-8,-10,-10,-4,-5,-8,9,-3,-8],[-5,4,-2,-1,5,10,4,5,-5,-10,-1,9,3,-7,-8,-7]],[[10,3,10,-8,5,-7,7,3,-10,-10,-8,-10,6,-9,-4,9],[8,7,-7,2,-2,1,4,1,6,7,-10,-5,6,-3,2,6]],[[4,-9,8,-5,2,-2,-9,-4,-4,-1,8,3,1,6,-7,-5],[-8,4,-8,8,4,-8,-5,-9,3,-5,7,-7,-5,-2,-7,-5]]], dtype = "int16")#candidate|38|(7, 2, 16)|const|int16
bop_39 = relay.bitwise_or(var_37.astype('int16'), relay.reshape(const_38.astype('int16'), relay.shape_of(var_37))) # shape=(7, 2, 16)
uop_42 = relay.cos(const_38.astype('float64')) # shape=(7, 2, 16)
uop_44 = relay.sinh(uop_42.astype('float32')) # shape=(7, 2, 16)
bop_46 = relay.bitwise_xor(const_38.astype('int8'), relay.reshape(uop_44.astype('int8'), relay.shape_of(const_38))) # shape=(7, 2, 16)
bop_49 = relay.equal(var_37.astype('bool'), relay.reshape(const_38.astype('bool'), relay.shape_of(var_37))) # shape=(7, 2, 16)
func_18_call = mod.get_global_var('func_18')
func_23_call = mutated_mod.get_global_var('func_23')
var_53 = relay.var("var_53", dtype = "float64", shape = (66,))#candidate|53|(66,)|var|float64
call_52 = relay.TupleGetItem(func_18_call(relay.reshape(var_53.astype('float64'), [11, 2, 3]), relay.reshape(var_53.astype('float64'), [11, 2, 3]), relay.reshape(var_53.astype('float64'), [11, 2, 3]), ), 0)
call_54 = relay.TupleGetItem(func_23_call(relay.reshape(var_53.astype('float64'), [11, 2, 3]), relay.reshape(var_53.astype('float64'), [11, 2, 3]), relay.reshape(var_53.astype('float64'), [11, 2, 3]), ), 0)
func_18_call = mod.get_global_var('func_18')
func_23_call = mutated_mod.get_global_var('func_23')
call_55 = relay.TupleGetItem(func_18_call(relay.reshape(var_53.astype('float64'), [11, 2, 3]), relay.reshape(call_52.astype('float64'), [11, 2, 3]), relay.reshape(call_52.astype('float64'), [11, 2, 3]), ), 1)
call_56 = relay.TupleGetItem(func_23_call(relay.reshape(var_53.astype('float64'), [11, 2, 3]), relay.reshape(call_52.astype('float64'), [11, 2, 3]), relay.reshape(call_52.astype('float64'), [11, 2, 3]), ), 1)
bop_57 = relay.power(uop_44.astype('float32'), relay.reshape(uop_42.astype('float32'), relay.shape_of(uop_44))) # shape=(7, 2, 16)
bop_60 = relay.mod(bop_46.astype('float32'), relay.reshape(const_38.astype('float32'), relay.shape_of(bop_46))) # shape=(7, 2, 16)
uop_63 = relay.sigmoid(uop_44.astype('float64')) # shape=(7, 2, 16)
uop_65 = relay.erf(uop_63.astype('float32')) # shape=(7, 2, 16)
uop_67 = relay.acosh(bop_60.astype('float64')) # shape=(7, 2, 16)
uop_69 = relay.asinh(uop_67.astype('float32')) # shape=(7, 2, 16)
bop_71 = relay.mod(uop_67.astype('float32'), relay.reshape(bop_39.astype('float32'), relay.shape_of(uop_67))) # shape=(7, 2, 16)
bop_74 = relay.mod(uop_69.astype('float64'), relay.reshape(const_38.astype('float64'), relay.shape_of(uop_69))) # shape=(7, 2, 16)
bop_77 = relay.add(uop_65.astype('int32'), relay.reshape(var_37.astype('int32'), relay.shape_of(uop_65))) # shape=(7, 2, 16)
const_80 = relay.const([[[8.141821,-9.160744,0.818629,4.474247,0.596983,-7.592123,7.597852,4.501917,3.845682,4.807203,1.865063,-8.618604,3.385325,8.300930,-2.472715,-8.487839],[4.411468,4.174045,1.954309,-3.335537,-3.750942,0.235841,-9.366927,0.431735,-9.927443,5.392260,-4.622836,-5.401894,-7.913871,-8.521239,-6.798874,-2.162957]],[[-8.777075,-3.252538,3.063420,-1.444597,-5.900267,2.405541,-5.460295,-7.335146,3.321728,6.529010,9.538352,-7.374172,1.362873,-1.936437,1.957688,7.834194],[7.158985,1.395272,6.864019,-6.692018,1.387802,-4.257310,-1.120568,-8.161428,-0.218102,-1.923989,9.859399,-5.981339,3.201025,2.927020,9.655210,-7.938005]],[[-1.719742,7.273255,-0.586389,5.222118,8.937870,3.378283,-0.032305,-0.185958,7.443434,9.941446,6.098269,-1.524983,-3.167707,9.145860,-5.751788,-2.107375],[8.466992,-3.242780,-0.374859,2.084197,4.041377,9.613398,-6.270134,2.505694,-5.440907,-3.941695,-8.603913,1.085047,-5.167144,-8.141351,-9.221626,-4.864080]],[[5.351186,4.616740,-7.375842,3.716255,-9.400953,-7.382207,-6.701059,-3.516190,-7.264781,-4.685471,4.545610,8.467018,-3.902509,0.663747,3.842357,-8.785587],[-6.960900,-3.605050,-4.882510,0.319561,-1.685805,8.031979,2.689254,-3.861188,7.818783,-0.099708,-5.965658,-6.557238,0.229209,5.089783,8.314598,6.990474]],[[-8.427171,2.341163,-6.664648,7.691282,-6.700888,4.163373,-5.051414,4.813028,7.811364,9.601429,7.083754,-1.105198,9.142006,1.369404,-0.473969,-5.089457],[-3.117204,8.790756,-0.348609,-2.766729,-6.646675,-3.621744,2.310686,-8.667220,2.718326,-9.074836,3.735440,-8.964515,-5.806078,-1.674103,-0.127184,3.641362]],[[-5.756831,0.303290,3.318152,5.929119,-0.210318,7.259990,7.116702,-1.039202,4.898368,8.549364,9.148441,6.306448,-1.187933,-0.303799,-7.221591,9.200263],[8.333514,-5.357780,-2.760229,-5.697984,0.975585,-7.843921,1.406407,-6.420052,7.657133,9.540154,4.780357,6.363190,-9.071103,-5.771632,8.118834,4.717525]],[[9.496586,-9.008070,6.974429,-3.776605,-3.111509,8.341359,5.307515,-7.021583,-3.652768,2.860471,-6.916657,9.089821,9.448114,0.749003,7.585895,0.925429],[-1.795437,7.330408,-7.121854,1.617129,-6.591127,-4.970623,-1.693274,1.261898,-4.380348,-6.236946,0.413419,-2.827016,9.456179,0.611054,-3.270110,2.154189]]], dtype = "float32")#candidate|80|(7, 2, 16)|const|float32
bop_81 = relay.divide(uop_69.astype('float32'), relay.reshape(const_80.astype('float32'), relay.shape_of(uop_69))) # shape=(7, 2, 16)
bop_84 = relay.maximum(uop_44.astype('uint64'), relay.reshape(bop_39.astype('uint64'), relay.shape_of(uop_44))) # shape=(7, 2, 16)
func_32_call = mod.get_global_var('func_32')
func_35_call = mutated_mod.get_global_var('func_35')
var_88 = relay.var("var_88", dtype = "bool", shape = ())#candidate|88|()|var|bool
call_87 = relay.TupleGetItem(func_32_call(relay.reshape(var_88.astype('bool'), [])), 0)
call_89 = relay.TupleGetItem(func_35_call(relay.reshape(var_88.astype('bool'), [])), 0)
uop_90 = relay.erf(uop_65.astype('float64')) # shape=(7, 2, 16)
var_92 = relay.var("var_92", dtype = "float64", shape = (7, 2, 16))#candidate|92|(7, 2, 16)|var|float64
bop_93 = relay.multiply(uop_63.astype('uint8'), relay.reshape(var_92.astype('uint8'), relay.shape_of(uop_63))) # shape=(7, 2, 16)
uop_96 = relay.acosh(bop_74.astype('float64')) # shape=(7, 2, 16)
var_98 = relay.var("var_98", dtype = "float64", shape = (7, 2, 16))#candidate|98|(7, 2, 16)|var|float64
bop_99 = relay.greater(uop_96.astype('bool'), relay.reshape(var_98.astype('bool'), relay.shape_of(uop_96))) # shape=(7, 2, 16)
bop_102 = relay.multiply(bop_74.astype('float64'), relay.reshape(bop_77.astype('float64'), relay.shape_of(bop_74))) # shape=(7, 2, 16)
uop_105 = relay.log(uop_44.astype('float32')) # shape=(7, 2, 16)
uop_107 = relay.rsqrt(uop_90.astype('float32')) # shape=(7, 2, 16)
uop_109 = relay.acosh(uop_107.astype('float64')) # shape=(7, 2, 16)
var_111 = relay.var("var_111", dtype = "float64", shape = (7, 2, 16))#candidate|111|(7, 2, 16)|var|float64
bop_112 = relay.minimum(uop_96.astype('int16'), relay.reshape(var_111.astype('int16'), relay.shape_of(uop_96))) # shape=(7, 2, 16)
bop_115 = relay.bitwise_and(uop_107.astype('uint8'), relay.reshape(bop_77.astype('uint8'), relay.shape_of(uop_107))) # shape=(7, 2, 16)
uop_118 = relay.sinh(bop_99.astype('float32')) # shape=(7, 2, 16)
bop_120 = relay.greater(uop_109.astype('bool'), relay.reshape(const_38.astype('bool'), relay.shape_of(uop_109))) # shape=(7, 2, 16)
uop_123 = relay.acos(uop_109.astype('float32')) # shape=(7, 2, 16)
uop_125 = relay.acosh(uop_109.astype('float32')) # shape=(7, 2, 16)
uop_127 = relay.sinh(uop_123.astype('float64')) # shape=(7, 2, 16)
output = relay.Tuple([bop_49,call_52,var_53,call_55,bop_57,bop_71,bop_81,bop_84,call_87,var_88,bop_93,bop_102,uop_105,bop_112,bop_115,uop_118,bop_120,uop_125,uop_127,])
output2 = relay.Tuple([bop_49,call_54,var_53,call_56,bop_57,bop_71,bop_81,bop_84,call_89,var_88,bop_93,bop_102,uop_105,bop_112,bop_115,uop_118,bop_120,uop_125,uop_127,])
func_129 = relay.Function([var_37,var_53,var_88,var_92,var_98,var_111,], output)
mod['func_129'] = func_129
mod = relay.transform.InferType()(mod)
var_130 = relay.var("var_130", dtype = "int16", shape = (7, 2, 16))#candidate|130|(7, 2, 16)|var|int16
var_131 = relay.var("var_131", dtype = "float64", shape = (66,))#candidate|131|(66,)|var|float64
var_132 = relay.var("var_132", dtype = "bool", shape = ())#candidate|132|()|var|bool
var_133 = relay.var("var_133", dtype = "float64", shape = (7, 2, 16))#candidate|133|(7, 2, 16)|var|float64
var_134 = relay.var("var_134", dtype = "float64", shape = (7, 2, 16))#candidate|134|(7, 2, 16)|var|float64
var_135 = relay.var("var_135", dtype = "float64", shape = (7, 2, 16))#candidate|135|(7, 2, 16)|var|float64
output = func_129(var_130,var_131,var_132,var_133,var_134,var_135,)
func_136 = relay.Function([var_130,var_131,var_132,var_133,var_134,var_135,], output)
mutated_mod['func_136'] = func_136
mutated_mod = relay.transform.InferType()(mutated_mod)
var_138 = relay.var("var_138", dtype = "int32", shape = (6, 9))#candidate|138|(6, 9)|var|int32
var_139 = relay.var("var_139", dtype = "int32", shape = (6, 9))#candidate|139|(6, 9)|var|int32
bop_140 = relay.not_equal(var_138.astype('bool'), relay.reshape(var_139.astype('bool'), relay.shape_of(var_138))) # shape=(6, 9)
const_143 = relay.const([[7,10,-3,7,4,-6,-9,-6,-9],[-5,-2,2,-4,4,-7,5,-1,1],[-3,7,-6,8,-7,4,-3,-6,-6],[7,-5,-3,3,4,2,8,-8,5],[-10,-9,-5,3,-4,5,1,-3,-4],[-9,-2,-3,-2,2,-9,-4,-9,-8]], dtype = "int32")#candidate|143|(6, 9)|const|int32
bop_144 = relay.bitwise_or(var_138.astype('int32'), relay.reshape(const_143.astype('int32'), relay.shape_of(var_138))) # shape=(6, 9)
uop_147 = relay.log(var_138.astype('float32')) # shape=(6, 9)
output = relay.Tuple([bop_140,bop_144,uop_147,])
output2 = relay.Tuple([bop_140,bop_144,uop_147,])
func_149 = relay.Function([var_138,var_139,], output)
mod['func_149'] = func_149
mod = relay.transform.InferType()(mod)
mutated_mod['func_149'] = func_149
mutated_mod = relay.transform.InferType()(mutated_mod)
func_149_call = mutated_mod.get_global_var('func_149')
var_151 = relay.var("var_151", dtype = "int32", shape = (6, 9))#candidate|151|(6, 9)|var|int32
var_152 = relay.var("var_152", dtype = "int32", shape = (6, 9))#candidate|152|(6, 9)|var|int32
call_150 = func_149_call(var_151,var_152,)
output = call_150
func_153 = relay.Function([var_151,var_152,], output)
mutated_mod['func_153'] = func_153
mutated_mod = relay.transform.InferType()(mutated_mod)
var_155 = relay.var("var_155", dtype = "float32", shape = (15, 7))#candidate|155|(15, 7)|var|float32
uop_156 = relay.cosh(var_155.astype('float32')) # shape=(15, 7)
uop_158 = relay.asin(uop_156.astype('float64')) # shape=(15, 7)
bop_160 = relay.minimum(uop_158.astype('int32'), relay.reshape(uop_156.astype('int32'), relay.shape_of(uop_158))) # shape=(15, 7)
uop_163 = relay.rsqrt(uop_158.astype('float64')) # shape=(15, 7)
bop_165 = relay.not_equal(uop_163.astype('bool'), relay.reshape(var_155.astype('bool'), relay.shape_of(uop_163))) # shape=(15, 7)
bop_168 = relay.multiply(var_155.astype('int64'), relay.reshape(uop_156.astype('int64'), relay.shape_of(var_155))) # shape=(15, 7)
uop_171 = relay.asin(uop_163.astype('float64')) # shape=(15, 7)
bop_173 = relay.left_shift(bop_165.astype('uint32'), relay.reshape(var_155.astype('uint32'), relay.shape_of(bop_165))) # shape=(15, 7)
output = relay.Tuple([bop_160,bop_168,uop_171,bop_173,])
output2 = relay.Tuple([bop_160,bop_168,uop_171,bop_173,])
func_176 = relay.Function([var_155,], output)
mod['func_176'] = func_176
mod = relay.transform.InferType()(mod)
var_177 = relay.var("var_177", dtype = "float32", shape = (15, 7))#candidate|177|(15, 7)|var|float32
output = func_176(var_177)
func_178 = relay.Function([var_177], output)
mutated_mod['func_178'] = func_178
mutated_mod = relay.transform.InferType()(mutated_mod)
var_180 = relay.var("var_180", dtype = "float32", shape = (4, 15))#candidate|180|(4, 15)|var|float32
uop_181 = relay.acosh(var_180.astype('float32')) # shape=(4, 15)
bop_183 = relay.add(uop_181.astype('int64'), relay.reshape(var_180.astype('int64'), relay.shape_of(uop_181))) # shape=(4, 15)
var_186 = relay.var("var_186", dtype = "float32", shape = (4, 15))#candidate|186|(4, 15)|var|float32
bop_187 = relay.multiply(uop_181.astype('uint8'), relay.reshape(var_186.astype('uint8'), relay.shape_of(uop_181))) # shape=(4, 15)
var_190 = relay.var("var_190", dtype = "float32", shape = (4, 15))#candidate|190|(4, 15)|var|float32
bop_191 = relay.greater_equal(uop_181.astype('bool'), relay.reshape(var_190.astype('bool'), relay.shape_of(uop_181))) # shape=(4, 15)
bop_194 = relay.floor_divide(var_186.astype('float32'), relay.reshape(bop_187.astype('float32'), relay.shape_of(var_186))) # shape=(4, 15)
var_197 = relay.var("var_197", dtype = "float32", shape = (4, 15))#candidate|197|(4, 15)|var|float32
bop_198 = relay.left_shift(uop_181.astype('uint32'), relay.reshape(var_197.astype('uint32'), relay.shape_of(uop_181))) # shape=(4, 15)
bop_201 = relay.mod(bop_183.astype('float64'), relay.reshape(var_197.astype('float64'), relay.shape_of(bop_183))) # shape=(4, 15)
uop_204 = relay.erf(var_186.astype('float64')) # shape=(4, 15)
uop_206 = relay.sigmoid(bop_187.astype('float64')) # shape=(4, 15)
uop_208 = relay.sigmoid(uop_206.astype('float32')) # shape=(4, 15)
bop_210 = relay.maximum(uop_208.astype('uint8'), relay.reshape(var_190.astype('uint8'), relay.shape_of(uop_208))) # shape=(4, 15)
uop_213 = relay.erf(uop_208.astype('float32')) # shape=(4, 15)
bop_215 = relay.greater(uop_213.astype('bool'), relay.reshape(uop_204.astype('bool'), relay.shape_of(uop_213))) # shape=(4, 15)
var_218 = relay.var("var_218", dtype = "float64", shape = (4, 15))#candidate|218|(4, 15)|var|float64
bop_219 = relay.less(bop_201.astype('bool'), relay.reshape(var_218.astype('bool'), relay.shape_of(bop_201))) # shape=(4, 15)
uop_222 = relay.acos(bop_210.astype('float32')) # shape=(4, 15)
uop_224 = relay.cosh(uop_222.astype('float32')) # shape=(4, 15)
var_226 = relay.var("var_226", dtype = "float32", shape = (4, 15))#candidate|226|(4, 15)|var|float32
bop_227 = relay.multiply(uop_222.astype('int64'), relay.reshape(var_226.astype('int64'), relay.shape_of(uop_222))) # shape=(4, 15)
bop_230 = relay.logical_and(uop_224.astype('bool'), relay.reshape(bop_210.astype('bool'), relay.shape_of(uop_224))) # shape=(4, 15)
bop_233 = relay.greater(var_218.astype('bool'), relay.reshape(var_197.astype('bool'), relay.shape_of(var_218))) # shape=(4, 15)
var_236 = relay.var("var_236", dtype = "float32", shape = (4, 15))#candidate|236|(4, 15)|var|float32
bop_237 = relay.equal(uop_224.astype('bool'), relay.reshape(var_236.astype('bool'), relay.shape_of(uop_224))) # shape=(4, 15)
uop_240 = relay.sinh(bop_230.astype('float64')) # shape=(4, 15)
bop_242 = relay.logical_or(uop_240.astype('bool'), relay.reshape(bop_230.astype('bool'), relay.shape_of(uop_240))) # shape=(4, 15)
bop_245 = relay.subtract(bop_237.astype('uint64'), relay.reshape(bop_198.astype('uint64'), relay.shape_of(bop_237))) # shape=(4, 15)
bop_248 = relay.less(uop_240.astype('bool'), relay.reshape(bop_183.astype('bool'), relay.shape_of(uop_240))) # shape=(4, 15)
uop_251 = relay.log(uop_206.astype('float64')) # shape=(4, 15)
bop_253 = relay.right_shift(bop_230.astype('uint32'), relay.reshape(uop_251.astype('uint32'), relay.shape_of(bop_230))) # shape=(4, 15)
var_256 = relay.var("var_256", dtype = "float32", shape = (4, 15))#candidate|256|(4, 15)|var|float32
bop_257 = relay.equal(uop_222.astype('bool'), relay.reshape(var_256.astype('bool'), relay.shape_of(uop_222))) # shape=(4, 15)
var_260 = relay.var("var_260", dtype = "bool", shape = (4, 15))#candidate|260|(4, 15)|var|bool
bop_261 = relay.logical_and(bop_248.astype('bool'), relay.reshape(var_260.astype('bool'), relay.shape_of(bop_248))) # shape=(4, 15)
uop_264 = relay.cosh(bop_245.astype('float32')) # shape=(4, 15)
var_266 = relay.var("var_266", dtype = "float64", shape = (4, 15))#candidate|266|(4, 15)|var|float64
bop_267 = relay.floor_divide(uop_240.astype('float64'), relay.reshape(var_266.astype('float64'), relay.shape_of(uop_240))) # shape=(4, 15)
func_176_call = mod.get_global_var('func_176')
func_178_call = mutated_mod.get_global_var('func_178')
var_271 = relay.var("var_271", dtype = "float32", shape = (7, 15))#candidate|271|(7, 15)|var|float32
call_270 = relay.TupleGetItem(func_176_call(relay.reshape(var_271.astype('float32'), [15, 7])), 2)
call_272 = relay.TupleGetItem(func_178_call(relay.reshape(var_271.astype('float32'), [15, 7])), 2)
uop_273 = relay.sin(bop_230.astype('float64')) # shape=(4, 15)
func_18_call = mod.get_global_var('func_18')
func_23_call = mutated_mod.get_global_var('func_23')
const_276 = relay.const([[2.282999],[-5.234638],[-3.690615],[4.233011],[-5.288443],[0.942716],[5.114718],[-3.160572],[-2.427077],[-6.103036],[-1.461850],[2.327129],[7.557522],[3.516437],[1.657184],[-0.947296],[3.023411],[7.276098],[2.508449],[3.554125],[-0.939370],[-3.992534],[9.648859],[4.861147],[6.961338],[3.020922],[3.164763],[4.818783],[1.055250],[4.425338],[-0.366863],[5.181979],[-0.668897],[4.855487],[0.118507],[-5.758324],[0.294734],[-4.833622],[4.334068],[-5.852630],[5.047068],[1.575064],[-6.578898],[0.579979],[1.961601],[-4.109622],[-8.314194],[8.606145],[6.376391],[8.391296],[3.876840],[-5.285701],[-5.098803],[-3.396788],[-1.361866],[-2.698639],[-2.106757],[5.439776],[-9.343954],[-8.391698],[5.173611],[6.508866],[-9.783054],[-4.082232],[1.972240],[6.765794]], dtype = "float64")#candidate|276|(66, 1)|const|float64
call_275 = relay.TupleGetItem(func_18_call(relay.reshape(const_276.astype('float64'), [11, 2, 3]), relay.reshape(const_276.astype('float64'), [11, 2, 3]), relay.reshape(const_276.astype('float64'), [11, 2, 3]), ), 1)
call_277 = relay.TupleGetItem(func_23_call(relay.reshape(const_276.astype('float64'), [11, 2, 3]), relay.reshape(const_276.astype('float64'), [11, 2, 3]), relay.reshape(const_276.astype('float64'), [11, 2, 3]), ), 1)
bop_278 = relay.floor_mod(bop_242.astype('float32'), relay.reshape(bop_261.astype('float32'), relay.shape_of(bop_242))) # shape=(4, 15)
uop_281 = relay.cos(bop_278.astype('float32')) # shape=(4, 15)
uop_283 = relay.cos(uop_281.astype('float32')) # shape=(4, 15)
uop_285 = relay.acosh(uop_283.astype('float32')) # shape=(4, 15)
output = relay.Tuple([bop_191,bop_194,bop_215,bop_219,bop_227,bop_233,bop_253,bop_257,uop_264,bop_267,call_270,var_271,uop_273,call_275,const_276,uop_285,])
output2 = relay.Tuple([bop_191,bop_194,bop_215,bop_219,bop_227,bop_233,bop_253,bop_257,uop_264,bop_267,call_272,var_271,uop_273,call_277,const_276,uop_285,])
func_287 = relay.Function([var_180,var_186,var_190,var_197,var_218,var_226,var_236,var_256,var_260,var_266,var_271,], output)
mod['func_287'] = func_287
mod = relay.transform.InferType()(mod)
var_288 = relay.var("var_288", dtype = "float32", shape = (4, 15))#candidate|288|(4, 15)|var|float32
var_289 = relay.var("var_289", dtype = "float32", shape = (4, 15))#candidate|289|(4, 15)|var|float32
var_290 = relay.var("var_290", dtype = "float32", shape = (4, 15))#candidate|290|(4, 15)|var|float32
var_291 = relay.var("var_291", dtype = "float32", shape = (4, 15))#candidate|291|(4, 15)|var|float32
var_292 = relay.var("var_292", dtype = "float64", shape = (4, 15))#candidate|292|(4, 15)|var|float64
var_293 = relay.var("var_293", dtype = "float32", shape = (4, 15))#candidate|293|(4, 15)|var|float32
var_294 = relay.var("var_294", dtype = "float32", shape = (4, 15))#candidate|294|(4, 15)|var|float32
var_295 = relay.var("var_295", dtype = "float32", shape = (4, 15))#candidate|295|(4, 15)|var|float32
var_296 = relay.var("var_296", dtype = "bool", shape = (4, 15))#candidate|296|(4, 15)|var|bool
var_297 = relay.var("var_297", dtype = "float64", shape = (4, 15))#candidate|297|(4, 15)|var|float64
var_298 = relay.var("var_298", dtype = "float32", shape = (7, 15))#candidate|298|(7, 15)|var|float32
output = func_287(var_288,var_289,var_290,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,)
func_299 = relay.Function([var_288,var_289,var_290,var_291,var_292,var_293,var_294,var_295,var_296,var_297,var_298,], output)
mutated_mod['func_299'] = func_299
mutated_mod = relay.transform.InferType()(mutated_mod)
const_301 = relay.const([[7,-8],[-2,6],[-3,-8],[3,5],[-7,-7],[4,-4],[7,2]], dtype = "int64")#candidate|301|(7, 2)|const|int64
var_302 = relay.var("var_302", dtype = "int64", shape = (7, 2))#candidate|302|(7, 2)|var|int64
bop_303 = relay.bitwise_and(const_301.astype('int64'), relay.reshape(var_302.astype('int64'), relay.shape_of(const_301))) # shape=(7, 2)
bop_306 = relay.logical_and(const_301.astype('bool'), relay.reshape(bop_303.astype('bool'), relay.shape_of(const_301))) # shape=(7, 2)
uop_309 = relay.sigmoid(var_302.astype('float64')) # shape=(7, 2)
func_287_call = mod.get_global_var('func_287')
func_299_call = mutated_mod.get_global_var('func_299')
const_312 = relay.const([-4.545147,-8.228162,-2.394057,-9.661225,2.372945,7.675462,-5.157016,-5.263883,1.162707,0.579356,8.792143,6.129430,-3.416294,-9.093061,5.742633,7.764206,-7.476368,-1.900673,-3.341248,0.229665,0.955547,7.440869,-1.474810,-8.957546,4.752325,-4.055131,7.621853,-7.567278,-8.614624,5.908931,-3.341624,2.274492,-7.275280,9.446154,3.816507,-9.628717,0.305460,-8.578353,2.836047,-4.704153,-0.576074,0.151925,-6.854431,7.465424,-2.346117,5.923207,3.069431,-7.600962,-9.683733,1.565664,-1.430759,-3.278469,0.605347,3.777286,7.876171,-2.420802,-2.271190,7.158249,1.179269,-0.511378], dtype = "float32")#candidate|312|(60,)|const|float32
const_313 = relay.const([[-1.959885,-3.439199,0.305920,2.594598,-4.205364,-6.323590,-6.766863,0.629374,-3.158421,2.824901,-7.873838,-2.897231,-3.644012,3.335025,8.832944,8.278433,6.299331,-4.031870,-7.720277,-2.412530,3.220121,-5.508523,7.873758,1.142466,7.939269,8.209265,-9.174717,-4.852813,5.560341,3.381493,-9.298318,-4.257157,-2.374331,-5.568749,1.836475,6.997609,4.611782,-0.005947,1.360887,-5.681655,7.194842,9.006517,-2.965711,-5.686618,6.094032,-0.391354,-7.350291,4.826937,5.017455,-0.295313,1.055688,-1.484449,8.006624,4.653723,-5.638350,3.119143,9.860891,7.652397,-6.744099,9.654461,-6.457723,5.113458,1.679978,9.953223,3.463890,0.681052,-1.365305,7.872103,-4.797362,-2.584647,7.127965,-6.523186,-9.867289,-7.059683,6.868558,-0.607427,9.062047,9.310333,8.060420,-9.921729,-8.998888,8.393939,8.105069,-9.958987,6.925883,-2.636617,-2.690857,1.281621,0.519350,-7.960334,9.433949,-6.256074,-3.074474,6.852006,-8.726343,-3.885669,2.586110,2.248849,4.987362,4.662396,1.036297,8.926484,-9.580693,-7.111953,-2.509655]], dtype = "float32")#candidate|313|(1, 105)|const|float32
call_311 = relay.TupleGetItem(func_287_call(relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float64'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('bool'), [4, 15]), relay.reshape(const_312.astype('float64'), [4, 15]), relay.reshape(const_313.astype('float32'), [7, 15]), ), 0)
call_314 = relay.TupleGetItem(func_299_call(relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float64'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('float32'), [4, 15]), relay.reshape(const_312.astype('bool'), [4, 15]), relay.reshape(const_312.astype('float64'), [4, 15]), relay.reshape(const_313.astype('float32'), [7, 15]), ), 0)
bop_315 = relay.bitwise_or(uop_309.astype('uint16'), relay.reshape(bop_303.astype('uint16'), relay.shape_of(uop_309))) # shape=(7, 2)
bop_318 = relay.subtract(uop_309.astype('uint64'), relay.reshape(var_302.astype('uint64'), relay.shape_of(uop_309))) # shape=(7, 2)
uop_321 = relay.exp(bop_303.astype('float32')) # shape=(7, 2)
func_149_call = mod.get_global_var('func_149')
func_153_call = mutated_mod.get_global_var('func_153')
const_324 = relay.const([10,-6,-3,2,7,-3,5,1,-2,-6,-5,-7,-2,10,-8,1,-10,9,-7,-3,7,1,-3,-1,-5,-3,1,-3,-10,2,7,6,-8,5,5,2,-3,5,-5,-10,-5,-9,2,-9,5,10,5,-1,2,10,1,9,-5,9], dtype = "int32")#candidate|324|(54,)|const|int32
call_323 = relay.TupleGetItem(func_149_call(relay.reshape(const_324.astype('int32'), [6, 9]), relay.reshape(const_324.astype('int32'), [6, 9]), ), 0)
call_325 = relay.TupleGetItem(func_153_call(relay.reshape(const_324.astype('int32'), [6, 9]), relay.reshape(const_324.astype('int32'), [6, 9]), ), 0)
uop_326 = relay.sigmoid(bop_303.astype('float32')) # shape=(7, 2)
uop_328 = relay.acosh(uop_309.astype('float32')) # shape=(7, 2)
bop_330 = relay.floor_divide(bop_315.astype('float64'), relay.reshape(uop_321.astype('float64'), relay.shape_of(bop_315))) # shape=(7, 2)
uop_333 = relay.tan(uop_328.astype('float32')) # shape=(7, 2)
bop_335 = relay.equal(uop_321.astype('bool'), relay.reshape(uop_333.astype('bool'), relay.shape_of(uop_321))) # shape=(7, 2)
output = relay.Tuple([bop_306,call_311,const_312,const_313,bop_318,call_323,const_324,uop_326,bop_330,bop_335,])
output2 = relay.Tuple([bop_306,call_314,const_312,const_313,bop_318,call_325,const_324,uop_326,bop_330,bop_335,])
func_338 = relay.Function([var_302,], output)
mod['func_338'] = func_338
mod = relay.transform.InferType()(mod)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
var_339 = relay.var("var_339", dtype = "int64", shape = (7, 2))#candidate|339|(7, 2)|var|int64
func_338_call = mutated_mod.get_global_var('func_338')
call_340 = func_338_call(var_339)
output = call_340
func_341 = relay.Function([var_339], output)
mutated_mod['func_341'] = func_341
mutated_mod = relay.transform.InferType()(mutated_mod)
var_343 = relay.var("var_343", dtype = "float64", shape = (5, 2, 8))#candidate|343|(5, 2, 8)|var|float64
uop_344 = relay.sin(var_343.astype('float64')) # shape=(5, 2, 8)
uop_346 = relay.log2(uop_344.astype('float64')) # shape=(5, 2, 8)
uop_348 = relay.acosh(uop_346.astype('float32')) # shape=(5, 2, 8)
uop_350 = relay.asin(uop_348.astype('float64')) # shape=(5, 2, 8)
uop_352 = relay.cosh(uop_348.astype('float64')) # shape=(5, 2, 8)
output = relay.Tuple([uop_350,uop_352,])
output2 = relay.Tuple([uop_350,uop_352,])
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
input_343= np.array([[[-7.588960,-2.098708,-6.493420,8.099414,-8.398202,9.187529,-7.220054,4.349658],[8.898498,6.613252,-6.146772,0.018634,2.294929,2.036835,3.615561,5.767531]],[[-0.136466,2.727288,3.908826,4.449047,-5.083481,0.264743,7.967254,4.093274],[1.695703,8.225188,6.215564,0.577811,1.774533,-1.061735,-9.834710,9.391271]],[[-9.214453,-9.918363,6.448135,7.983610,-6.410941,-2.964009,6.760905,-7.247660],[-5.730819,-6.690211,-8.296781,-9.978200,-7.931799,-7.862593,-9.691029,9.547641]],[[-2.387874,-4.301194,0.062482,0.966641,-1.943002,9.431728,7.439293,5.440263],[7.641970,7.619318,9.986835,-7.949487,5.314121,-7.380224,-6.720062,-4.446278]],[[6.491071,-9.264959,-4.240601,-3.030020,5.903190,-1.328844,5.829192,-7.179846],[6.337551,-0.969677,9.856139,-8.323579,4.531117,-3.778320,-9.913758,-6.186809]]], dtype='float64')
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