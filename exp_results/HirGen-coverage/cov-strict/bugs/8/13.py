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
var_0 = relay.var("var_0", dtype = "float32", shape = (4,))#candidate|0|(4,)|var|float32
uop_1 = relay.log10(var_0.astype('float32')) # shape=(4,)
const_3 = relay.const([-6.376421,7.101274,9.194248,-5.156253], dtype = "float32")#candidate|3|(4,)|const|float32
bop_4 = relay.power(var_0.astype('float32'), relay.reshape(const_3.astype('float32'), relay.shape_of(var_0))) # shape=(4,)
bop_7 = relay.greater(uop_1.astype('bool'), relay.reshape(var_0.astype('bool'), relay.shape_of(uop_1))) # shape=(4,)
bop_10 = relay.maximum(uop_1.astype('uint8'), relay.reshape(var_0.astype('uint8'), relay.shape_of(uop_1))) # shape=(4,)
var_13 = relay.var("var_13", dtype = "float32", shape = (4,))#candidate|13|(4,)|var|float32
bop_14 = relay.greater(const_3.astype('bool'), relay.reshape(var_13.astype('bool'), relay.shape_of(const_3))) # shape=(4,)
bop_17 = relay.less(bop_14.astype('bool'), relay.reshape(uop_1.astype('bool'), relay.shape_of(bop_14))) # shape=(4,)
const_20 = relay.const([3.097363,-7.908756,1.915030,-6.321959], dtype = "float32")#candidate|20|(4,)|const|float32
bop_21 = relay.less(var_0.astype('bool'), relay.reshape(const_20.astype('bool'), relay.shape_of(var_0))) # shape=(4,)
output = relay.Tuple([bop_4,bop_7,bop_10,bop_17,bop_21,])
output2 = relay.Tuple([bop_4,bop_7,bop_10,bop_17,bop_21,])
func_24 = relay.Function([var_0,var_13,], output)
mod['func_24'] = func_24
mod = relay.transform.InferType()(mod)
mutated_mod['func_24'] = func_24
mutated_mod = relay.transform.InferType()(mutated_mod)
func_24_call = mutated_mod.get_global_var('func_24')
var_26 = relay.var("var_26", dtype = "float32", shape = (4,))#candidate|26|(4,)|var|float32
var_27 = relay.var("var_27", dtype = "float32", shape = (4,))#candidate|27|(4,)|var|float32
call_25 = func_24_call(var_26,var_27,)
output = call_25
func_28 = relay.Function([var_26,var_27,], output)
mutated_mod['func_28'] = func_28
mutated_mod = relay.transform.InferType()(mutated_mod)
var_30 = relay.var("var_30", dtype = "bool", shape = (15, 7))#candidate|30|(15, 7)|var|bool
var_31 = relay.var("var_31", dtype = "bool", shape = (15, 7))#candidate|31|(15, 7)|var|bool
bop_32 = relay.logical_and(var_30.astype('bool'), relay.reshape(var_31.astype('bool'), relay.shape_of(var_30))) # shape=(15, 7)
func_24_call = mod.get_global_var('func_24')
func_28_call = mutated_mod.get_global_var('func_28')
const_36 = relay.const([6.685802,-8.782320,-1.525690,8.463414], dtype = "float32")#candidate|36|(4,)|const|float32
call_35 = relay.TupleGetItem(func_24_call(relay.reshape(const_36.astype('float32'), [4,]), relay.reshape(const_36.astype('float32'), [4,]), ), 3)
call_37 = relay.TupleGetItem(func_28_call(relay.reshape(const_36.astype('float32'), [4,]), relay.reshape(const_36.astype('float32'), [4,]), ), 3)
var_38 = relay.var("var_38", dtype = "bool", shape = (4,))#candidate|38|(4,)|var|bool
bop_39 = relay.add(call_35.astype('int16'), relay.reshape(var_38.astype('int16'), relay.shape_of(call_35))) # shape=(4,)
bop_42 = relay.add(call_37.astype('int16'), relay.reshape(var_38.astype('int16'), relay.shape_of(call_37))) # shape=(4,)
output = relay.Tuple([bop_32,const_36,bop_39,])
output2 = relay.Tuple([bop_32,const_36,bop_42,])
func_43 = relay.Function([var_30,var_31,var_38,], output)
mod['func_43'] = func_43
mod = relay.transform.InferType()(mod)
var_44 = relay.var("var_44", dtype = "bool", shape = (15, 7))#candidate|44|(15, 7)|var|bool
var_45 = relay.var("var_45", dtype = "bool", shape = (15, 7))#candidate|45|(15, 7)|var|bool
var_46 = relay.var("var_46", dtype = "bool", shape = (4,))#candidate|46|(4,)|var|bool
output = func_43(var_44,var_45,var_46,)
func_47 = relay.Function([var_44,var_45,var_46,], output)
mutated_mod['func_47'] = func_47
mutated_mod = relay.transform.InferType()(mutated_mod)
var_49 = relay.var("var_49", dtype = "int8", shape = ())#candidate|49|()|var|int8
const_50 = relay.const(4, dtype = "int8")#candidate|50|()|const|int8
bop_51 = relay.less_equal(var_49.astype('bool'), const_50.astype('bool')) # shape=()
const_54 = relay.const([[-5,6],[4,-3],[6,7],[-5,-3],[-2,-10],[-4,3],[8,-7],[-6,6],[-3,-5],[2,7],[6,-9],[10,5]], dtype = "int8")#candidate|54|(12, 2)|const|int8
bop_55 = relay.maximum(const_50.astype('float32'), const_54.astype('float32')) # shape=(12, 2)
uop_58 = relay.atan(const_50.astype('float32')) # shape=()
var_60 = relay.var("var_60", dtype = "float32", shape = (6,))#candidate|60|(6,)|var|float32
bop_61 = relay.floor_mod(uop_58.astype('float32'), var_60.astype('float32')) # shape=(6,)
func_43_call = mod.get_global_var('func_43')
func_47_call = mutated_mod.get_global_var('func_47')
const_65 = relay.const([[True,True,True,True,False,True,True,False,False,False,False,False,False,False,False],[True,False,True,False,False,True,False,False,False,False,True,True,True,True,False],[False,False,True,False,True,False,True,False,False,False,False,False,False,False,False],[False,True,True,False,False,True,True,False,True,True,False,False,False,True,True],[False,False,True,True,False,False,True,True,True,True,True,True,False,False,False],[False,False,True,True,False,True,True,True,True,False,False,False,True,False,True],[False,False,True,False,False,True,False,True,True,True,True,False,True,True,False]], dtype = "bool")#candidate|65|(7, 15)|const|bool
var_66 = relay.var("var_66", dtype = "bool", shape = (4,))#candidate|66|(4,)|var|bool
call_64 = relay.TupleGetItem(func_43_call(relay.reshape(const_65.astype('bool'), [15, 7]), relay.reshape(const_65.astype('bool'), [15, 7]), relay.reshape(var_66.astype('bool'), [4,]), ), 1)
call_67 = relay.TupleGetItem(func_47_call(relay.reshape(const_65.astype('bool'), [15, 7]), relay.reshape(const_65.astype('bool'), [15, 7]), relay.reshape(var_66.astype('bool'), [4,]), ), 1)
output = relay.Tuple([bop_51,bop_55,bop_61,call_64,const_65,var_66,])
output2 = relay.Tuple([bop_51,bop_55,bop_61,call_67,const_65,var_66,])
func_68 = relay.Function([var_49,var_60,var_66,], output)
mod['func_68'] = func_68
mod = relay.transform.InferType()(mod)
var_69 = relay.var("var_69", dtype = "int8", shape = ())#candidate|69|()|var|int8
var_70 = relay.var("var_70", dtype = "float32", shape = (6,))#candidate|70|(6,)|var|float32
var_71 = relay.var("var_71", dtype = "bool", shape = (4,))#candidate|71|(4,)|var|bool
output = func_68(var_69,var_70,var_71,)
func_72 = relay.Function([var_69,var_70,var_71,], output)
mutated_mod['func_72'] = func_72
mutated_mod = relay.transform.InferType()(mutated_mod)
var_74 = relay.var("var_74", dtype = "float64", shape = (12, 14))#candidate|74|(12, 14)|var|float64
uop_75 = relay.sqrt(var_74.astype('float64')) # shape=(12, 14)
func_43_call = mod.get_global_var('func_43')
func_47_call = mutated_mod.get_global_var('func_47')
var_78 = relay.var("var_78", dtype = "bool", shape = (105,))#candidate|78|(105,)|var|bool
const_79 = relay.const([False,False,False,False], dtype = "bool")#candidate|79|(4,)|const|bool
call_77 = relay.TupleGetItem(func_43_call(relay.reshape(var_78.astype('bool'), [15, 7]), relay.reshape(var_78.astype('bool'), [15, 7]), relay.reshape(const_79.astype('bool'), [4,]), ), 2)
call_80 = relay.TupleGetItem(func_47_call(relay.reshape(var_78.astype('bool'), [15, 7]), relay.reshape(var_78.astype('bool'), [15, 7]), relay.reshape(const_79.astype('bool'), [4,]), ), 2)
output = relay.Tuple([uop_75,call_77,var_78,const_79,])
output2 = relay.Tuple([uop_75,call_80,var_78,const_79,])
func_81 = relay.Function([var_74,var_78,], output)
mod['func_81'] = func_81
mod = relay.transform.InferType()(mod)
mutated_mod['func_81'] = func_81
mutated_mod = relay.transform.InferType()(mutated_mod)
func_81_call = mutated_mod.get_global_var('func_81')
var_83 = relay.var("var_83", dtype = "float64", shape = (12, 14))#candidate|83|(12, 14)|var|float64
var_84 = relay.var("var_84", dtype = "bool", shape = (105,))#candidate|84|(105,)|var|bool
call_82 = func_81_call(var_83,var_84,)
output = call_82
func_85 = relay.Function([var_83,var_84,], output)
mutated_mod['func_85'] = func_85
mutated_mod = relay.transform.InferType()(mutated_mod)
var_87 = relay.var("var_87", dtype = "float32", shape = (9, 13, 5))#candidate|87|(9, 13, 5)|var|float32
uop_88 = relay.log10(var_87.astype('float32')) # shape=(9, 13, 5)
bop_90 = relay.maximum(var_87.astype('int64'), relay.reshape(uop_88.astype('int64'), relay.shape_of(var_87))) # shape=(9, 13, 5)
bop_93 = relay.subtract(uop_88.astype('int32'), relay.reshape(var_87.astype('int32'), relay.shape_of(uop_88))) # shape=(9, 13, 5)
bop_96 = relay.power(uop_88.astype('float32'), relay.reshape(var_87.astype('float32'), relay.shape_of(uop_88))) # shape=(9, 13, 5)
bop_99 = relay.logical_xor(var_87.astype('uint8'), relay.reshape(bop_96.astype('uint8'), relay.shape_of(var_87))) # shape=(9, 13, 5)
uop_102 = relay.atanh(uop_88.astype('float64')) # shape=(9, 13, 5)
func_81_call = mod.get_global_var('func_81')
func_85_call = mutated_mod.get_global_var('func_85')
const_105 = relay.const([[-9.085299,5.876559],[-8.371464,2.290187],[-1.163813,-7.608776],[6.610697,5.847821],[2.052959,9.245300],[-7.898051,-1.998185],[1.567761,-7.537056],[2.956367,-2.023266],[4.688341,7.209293],[6.803926,3.748220],[0.991806,4.924014],[1.915280,6.807395],[-8.631703,-0.579535],[-9.487086,-4.044571],[0.434153,7.528996],[-5.167807,-7.764480],[-2.791761,7.473660],[1.627794,1.583698],[2.054331,9.568843],[0.311018,7.652621],[1.636512,-6.920393],[-2.210869,1.429595],[8.144475,-3.418854],[-3.032184,3.712057],[-4.166386,-2.196801],[9.199654,-9.407725],[-9.697711,-0.713477],[-9.390522,-4.434993],[2.086351,-2.233431],[6.632641,6.837632],[-2.181340,-1.571957],[0.238377,7.372297],[2.327898,9.193807],[4.842311,0.745481],[9.347934,5.960183],[3.994678,9.972663],[-9.718275,-9.163518],[8.852573,-7.390869],[7.805095,-2.476695],[0.671080,-5.205461],[8.392801,1.061255],[7.311009,-9.245121],[-6.928799,-1.743532],[2.890150,3.414940],[7.227986,8.188029],[9.023931,5.600901],[1.820621,7.475068],[2.045700,4.995506],[3.874344,-3.263325],[1.214104,5.132755],[-2.514443,2.608556],[2.852171,-2.693671],[-7.570860,5.616248],[1.262353,-0.280616],[-3.373769,8.752517],[8.896529,-4.486469],[-7.306384,0.227626],[-6.532193,0.177622],[-3.054228,-0.686438],[8.431825,9.374083],[4.686845,-0.615189],[7.255799,2.102698],[-3.935968,4.632489],[5.700451,6.216584],[-9.062139,6.990026],[9.419695,-2.213226],[5.880709,-7.445529],[6.698728,-2.944662],[-6.712931,2.980389],[4.711184,-5.200508],[-2.891285,3.957718],[2.270572,5.551056],[9.324392,-9.452675],[5.496540,-0.775904],[-5.214486,-7.100132],[-0.562199,-9.196104],[2.402185,-8.274035],[-5.008514,-2.629661],[8.856361,-9.128264],[-2.240517,-9.446003],[-8.911070,8.973274],[7.772739,-2.030312],[-4.578518,3.914380],[6.165260,6.889541]], dtype = "float64")#candidate|105|(84, 2)|const|float64
var_106 = relay.var("var_106", dtype = "bool", shape = (105,))#candidate|106|(105,)|var|bool
call_104 = relay.TupleGetItem(func_81_call(relay.reshape(const_105.astype('float64'), [12, 14]), relay.reshape(var_106.astype('bool'), [105,]), ), 3)
call_107 = relay.TupleGetItem(func_85_call(relay.reshape(const_105.astype('float64'), [12, 14]), relay.reshape(var_106.astype('bool'), [105,]), ), 3)
uop_108 = relay.log(uop_102.astype('float64')) # shape=(9, 13, 5)
var_110 = relay.var("var_110", dtype = "float64", shape = (9, 13, 5))#candidate|110|(9, 13, 5)|var|float64
bop_111 = relay.greater_equal(uop_108.astype('bool'), relay.reshape(var_110.astype('bool'), relay.shape_of(uop_108))) # shape=(9, 13, 5)
uop_114 = relay.sinh(call_104.astype('float32')) # shape=(4,)
uop_116 = relay.sinh(call_107.astype('float32')) # shape=(4,)
bop_117 = relay.power(uop_108.astype('float64'), relay.reshape(bop_90.astype('float64'), relay.shape_of(uop_108))) # shape=(9, 13, 5)
bop_120 = relay.subtract(bop_96.astype('uint64'), relay.reshape(bop_99.astype('uint64'), relay.shape_of(bop_96))) # shape=(9, 13, 5)
var_123 = relay.var("var_123", dtype = "float64", shape = (9, 13, 5))#candidate|123|(9, 13, 5)|var|float64
bop_124 = relay.greater(bop_117.astype('bool'), relay.reshape(var_123.astype('bool'), relay.shape_of(bop_117))) # shape=(9, 13, 5)
bop_127 = relay.logical_or(uop_102.astype('bool'), relay.reshape(bop_111.astype('bool'), relay.shape_of(uop_102))) # shape=(9, 13, 5)
bop_130 = relay.power(bop_127.astype('float64'), relay.reshape(bop_120.astype('float64'), relay.shape_of(bop_127))) # shape=(9, 13, 5)
uop_133 = relay.tan(bop_90.astype('float32')) # shape=(9, 13, 5)
bop_135 = relay.bitwise_or(uop_102.astype('int16'), relay.reshape(var_123.astype('int16'), relay.shape_of(uop_102))) # shape=(9, 13, 5)
var_138 = relay.var("var_138", dtype = "bool", shape = (9, 13, 5))#candidate|138|(9, 13, 5)|var|bool
bop_139 = relay.bitwise_and(bop_111.astype('uint16'), relay.reshape(var_138.astype('uint16'), relay.shape_of(bop_111))) # shape=(9, 13, 5)
var_142 = relay.var("var_142", dtype = "int16", shape = (9, 13, 5))#candidate|142|(9, 13, 5)|var|int16
bop_143 = relay.bitwise_and(bop_135.astype('uint64'), relay.reshape(var_142.astype('uint64'), relay.shape_of(bop_135))) # shape=(9, 13, 5)
uop_146 = relay.exp(uop_88.astype('float64')) # shape=(9, 13, 5)
uop_148 = relay.rsqrt(bop_117.astype('float64')) # shape=(9, 13, 5)
uop_150 = relay.asin(uop_148.astype('float32')) # shape=(9, 13, 5)
uop_152 = relay.asinh(uop_150.astype('float64')) # shape=(9, 13, 5)
bop_154 = relay.subtract(uop_152.astype('uint16'), relay.reshape(var_138.astype('uint16'), relay.shape_of(uop_152))) # shape=(9, 13, 5)
uop_157 = relay.asin(bop_154.astype('float64')) # shape=(9, 13, 5)
bop_159 = relay.subtract(uop_148.astype('int16'), relay.reshape(var_110.astype('int16'), relay.shape_of(uop_148))) # shape=(9, 13, 5)
uop_162 = relay.acos(uop_157.astype('float32')) # shape=(9, 13, 5)
var_164 = relay.var("var_164", dtype = "float64", shape = (9, 13, 5))#candidate|164|(9, 13, 5)|var|float64
bop_165 = relay.maximum(uop_157.astype('float32'), relay.reshape(var_164.astype('float32'), relay.shape_of(uop_157))) # shape=(9, 13, 5)
bop_168 = relay.maximum(uop_162.astype('uint64'), relay.reshape(bop_139.astype('uint64'), relay.shape_of(uop_162))) # shape=(9, 13, 5)
uop_171 = relay.log(uop_157.astype('float64')) # shape=(9, 13, 5)
uop_173 = relay.sqrt(uop_171.astype('float64')) # shape=(9, 13, 5)
bop_175 = relay.logical_and(uop_173.astype('bool'), relay.reshape(var_164.astype('bool'), relay.shape_of(uop_173))) # shape=(9, 13, 5)
output = relay.Tuple([bop_93,const_105,var_106,uop_114,bop_124,bop_130,uop_133,bop_143,uop_146,bop_159,bop_165,bop_168,bop_175,])
output2 = relay.Tuple([bop_93,const_105,var_106,uop_116,bop_124,bop_130,uop_133,bop_143,uop_146,bop_159,bop_165,bop_168,bop_175,])
func_178 = relay.Function([var_87,var_106,var_110,var_123,var_138,var_142,var_164,], output)
mod['func_178'] = func_178
mod = relay.transform.InferType()(mod)
var_179 = relay.var("var_179", dtype = "float32", shape = (9, 13, 5))#candidate|179|(9, 13, 5)|var|float32
var_180 = relay.var("var_180", dtype = "bool", shape = (105,))#candidate|180|(105,)|var|bool
var_181 = relay.var("var_181", dtype = "float64", shape = (9, 13, 5))#candidate|181|(9, 13, 5)|var|float64
var_182 = relay.var("var_182", dtype = "float64", shape = (9, 13, 5))#candidate|182|(9, 13, 5)|var|float64
var_183 = relay.var("var_183", dtype = "bool", shape = (9, 13, 5))#candidate|183|(9, 13, 5)|var|bool
var_184 = relay.var("var_184", dtype = "int16", shape = (9, 13, 5))#candidate|184|(9, 13, 5)|var|int16
var_185 = relay.var("var_185", dtype = "float64", shape = (9, 13, 5))#candidate|185|(9, 13, 5)|var|float64
output = func_178(var_179,var_180,var_181,var_182,var_183,var_184,var_185,)
func_186 = relay.Function([var_179,var_180,var_181,var_182,var_183,var_184,var_185,], output)
mutated_mod['func_186'] = func_186
mutated_mod = relay.transform.InferType()(mutated_mod)
var_188 = relay.var("var_188", dtype = "uint8", shape = ())#candidate|188|()|var|uint8
var_189 = relay.var("var_189", dtype = "uint8", shape = (5, 6, 11))#candidate|189|(5, 6, 11)|var|uint8
bop_190 = relay.equal(var_188.astype('bool'), var_189.astype('bool')) # shape=(5, 6, 11)
bop_193 = relay.mod(bop_190.astype('float32'), relay.reshape(var_189.astype('float32'), relay.shape_of(bop_190))) # shape=(5, 6, 11)
uop_196 = relay.log2(var_189.astype('float64')) # shape=(5, 6, 11)
bop_198 = relay.equal(var_189.astype('bool'), relay.reshape(bop_193.astype('bool'), relay.shape_of(var_189))) # shape=(5, 6, 11)
uop_201 = relay.log(uop_196.astype('float32')) # shape=(5, 6, 11)
var_203 = relay.var("var_203", dtype = "float64", shape = (5, 6, 11))#candidate|203|(5, 6, 11)|var|float64
bop_204 = relay.logical_xor(uop_196.astype('int16'), relay.reshape(var_203.astype('int16'), relay.shape_of(uop_196))) # shape=(5, 6, 11)
bop_207 = relay.maximum(bop_204.astype('int64'), var_188.astype('int64')) # shape=(5, 6, 11)
bop_210 = relay.left_shift(uop_201.astype('int32'), relay.reshape(bop_190.astype('int32'), relay.shape_of(uop_201))) # shape=(5, 6, 11)
uop_213 = relay.sigmoid(uop_201.astype('float64')) # shape=(5, 6, 11)
bop_215 = relay.bitwise_and(bop_207.astype('int16'), relay.reshape(uop_196.astype('int16'), relay.shape_of(bop_207))) # shape=(5, 6, 11)
output = relay.Tuple([bop_198,bop_210,uop_213,bop_215,])
output2 = relay.Tuple([bop_198,bop_210,uop_213,bop_215,])
func_218 = relay.Function([var_188,var_189,var_203,], output)
mod['func_218'] = func_218
mod = relay.transform.InferType()(mod)
var_219 = relay.var("var_219", dtype = "uint8", shape = ())#candidate|219|()|var|uint8
var_220 = relay.var("var_220", dtype = "uint8", shape = (5, 6, 11))#candidate|220|(5, 6, 11)|var|uint8
var_221 = relay.var("var_221", dtype = "float64", shape = (5, 6, 11))#candidate|221|(5, 6, 11)|var|float64
output = func_218(var_219,var_220,var_221,)
func_222 = relay.Function([var_219,var_220,var_221,], output)
mutated_mod['func_222'] = func_222
mutated_mod = relay.transform.InferType()(mutated_mod)
var_224 = relay.var("var_224", dtype = "int8", shape = (16, 10))#candidate|224|(16, 10)|var|int8
const_225 = relay.const([[5,3,10,6,9,-5,1,10,-5,7],[-9,-10,-7,8,-6,-1,8,-10,-8,-4],[6,-9,-5,5,4,-4,2,3,-4,9],[1,2,-6,2,9,-2,-5,-1,2,10],[2,8,-9,3,1,-5,2,4,10,-1],[-9,-3,5,-7,-2,2,3,7,-2,7],[-10,8,10,9,7,10,-2,2,-8,4],[-6,-3,-8,-2,1,-6,2,3,5,3],[-4,9,9,-5,2,7,8,8,-10,-6],[10,2,-3,3,8,1,-2,-2,-4,7],[6,4,4,10,2,5,-8,-3,1,-1],[5,-5,-2,9,-3,10,8,3,-7,6],[-1,-3,-5,3,4,8,-8,8,7,-6],[4,10,-1,4,10,-7,-8,-1,7,-2],[8,4,-3,5,-2,10,8,-5,-7,-7],[8,7,6,-10,1,-6,2,-4,2,4]], dtype = "int8")#candidate|225|(16, 10)|const|int8
bop_226 = relay.subtract(var_224.astype('int8'), relay.reshape(const_225.astype('int8'), relay.shape_of(var_224))) # shape=(16, 10)
uop_229 = relay.acos(const_225.astype('float32')) # shape=(16, 10)
bop_231 = relay.logical_and(var_224.astype('bool'), relay.reshape(bop_226.astype('bool'), relay.shape_of(var_224))) # shape=(16, 10)
var_234 = relay.var("var_234", dtype = "int8", shape = (16, 10))#candidate|234|(16, 10)|var|int8
bop_235 = relay.logical_and(bop_226.astype('bool'), relay.reshape(var_234.astype('bool'), relay.shape_of(bop_226))) # shape=(16, 10)
uop_238 = relay.asin(uop_229.astype('float64')) # shape=(16, 10)
bop_240 = relay.floor_divide(bop_231.astype('float64'), relay.reshape(uop_238.astype('float64'), relay.shape_of(bop_231))) # shape=(16, 10)
uop_243 = relay.asinh(bop_240.astype('float32')) # shape=(16, 10)
uop_245 = relay.log(uop_243.astype('float64')) # shape=(16, 10)
uop_247 = relay.log(uop_245.astype('float32')) # shape=(16, 10)
func_24_call = mod.get_global_var('func_24')
func_28_call = mutated_mod.get_global_var('func_28')
var_250 = relay.var("var_250", dtype = "float32", shape = (4,))#candidate|250|(4,)|var|float32
call_249 = relay.TupleGetItem(func_24_call(relay.reshape(var_250.astype('float32'), [4,]), relay.reshape(var_250.astype('float32'), [4,]), ), 2)
call_251 = relay.TupleGetItem(func_28_call(relay.reshape(var_250.astype('float32'), [4,]), relay.reshape(var_250.astype('float32'), [4,]), ), 2)
uop_252 = relay.sin(uop_247.astype('float64')) # shape=(16, 10)
bop_254 = relay.right_shift(uop_252.astype('uint32'), relay.reshape(bop_235.astype('uint32'), relay.shape_of(uop_252))) # shape=(16, 10)
uop_257 = relay.exp(uop_252.astype('float64')) # shape=(16, 10)
uop_259 = relay.sin(uop_257.astype('float64')) # shape=(16, 10)
uop_261 = relay.cosh(uop_257.astype('float64')) # shape=(16, 10)
output = relay.Tuple([call_249,var_250,bop_254,uop_259,uop_261,])
output2 = relay.Tuple([call_251,var_250,bop_254,uop_259,uop_261,])
func_263 = relay.Function([var_224,var_234,var_250,], output)
mod['func_263'] = func_263
mod = relay.transform.InferType()(mod)
var_264 = relay.var("var_264", dtype = "int8", shape = (16, 10))#candidate|264|(16, 10)|var|int8
var_265 = relay.var("var_265", dtype = "int8", shape = (16, 10))#candidate|265|(16, 10)|var|int8
var_266 = relay.var("var_266", dtype = "float32", shape = (4,))#candidate|266|(4,)|var|float32
output = func_263(var_264,var_265,var_266,)
func_267 = relay.Function([var_264,var_265,var_266,], output)
mutated_mod['func_267'] = func_267
mutated_mod = relay.transform.InferType()(mutated_mod)
var_269 = relay.var("var_269", dtype = "float64", shape = ())#candidate|269|()|var|float64
var_270 = relay.var("var_270", dtype = "float64", shape = ())#candidate|270|()|var|float64
bop_271 = relay.mod(var_269.astype('float64'), var_270.astype('float64')) # shape=()
const_274 = relay.const([[-1.353836,-4.746113,1.264594,8.128951,-4.094520,-7.322361],[8.195216,-0.567901,-8.326145,-8.865012,-6.117337,-6.557744],[-2.401988,-2.134526,-8.991514,2.144106,2.153654,-5.981871]], dtype = "float64")#candidate|274|(3, 6)|const|float64
bop_275 = relay.bitwise_xor(var_269.astype('int64'), const_274.astype('int64')) # shape=(3, 6)
uop_278 = relay.asin(var_270.astype('float32')) # shape=()
var_280 = relay.var("var_280", dtype = "float64", shape = (7, 13, 14))#candidate|280|(7, 13, 14)|var|float64
bop_281 = relay.logical_or(var_269.astype('bool'), var_280.astype('bool')) # shape=(7, 13, 14)
bop_284 = relay.mod(uop_278.astype('float64'), var_269.astype('float64')) # shape=()
uop_287 = relay.cosh(var_270.astype('float64')) # shape=()
bop_289 = relay.left_shift(uop_278.astype('int16'), const_274.astype('int16')) # shape=(3, 6)
bop_292 = relay.floor_mod(bop_289.astype('float64'), var_270.astype('float64')) # shape=(3, 6)
func_218_call = mod.get_global_var('func_218')
func_222_call = mutated_mod.get_global_var('func_222')
const_296 = relay.const([[-1,-10,7,-8,10,-6,2,-2,4,8,-9,-5,-4,4,4,-10,5,1,-8,10,4,10,2,1,-4,-2,-6,1,-9,-9,-7,5,9,-3,4,6,1,-4,-9,-7,-3,-7,-9,10,3,8,-6,-3,-10,3,-10,-3,-8,4,-3,-10,3,7,4,-9,-5,-7,9,-8,-6,2,-4,9,-6,-7,7,7,-5,-1,2,-5,4,-9,-2,1,-10,4,-1,-3,-7,6,-2,-3,7,-5,3,-5,-4,9,1,-3,-6,-2,1,4,10,1,2,6,-7,6,4,6,-6,-5,-8,8,7,-3,-5,-8,6,6,3,-7,4,-2,-7,-8,3,1,-2,-7,7,7,4,9,-5,10,-9,-10,-4,-3,9,5,-5,-5,-5,3,-10,-6,-6,-8,3,8,-9,-5,9,4,-1,1,-5,-10,1,-2,6,7,6,2,7,9,6,-7,-4,-4,9,-1,-6,-5,-6,7,-7,-5,-2,-2,7,-5,9,-10,2,-10,-3,-7,1,3,-1,3,-7,-2,-5,7,10,-7,-3,-2,4,10,-2,-4,9,4,10,-9,-6,-5,7,-5,6,-6,10,6,1,-1,2,-9,3,-4,-1,-9,6,-9,5,-10,7,-10,5,3,3,-10,-9,10,10,-6,-5,-9,-8,3,-6,4,10,-2,-3,10,1,3,-7,-6,4,-2,-7,-1,-1,10,-9,1,-8,4,-10,-7,4,-2,2,-10,2,-9,-2,5,7,9,8,5,4,9,-2,6,-10,-7,-8,1,8,10,10,-1,9,-3,3,6,7,-3,4,7,10,6,-10,10,9,-7,-5,8,-10,2,-10,1,4,8,-3,-3,8,3,-8,-5,-2,8,1,-1,8,6,-2,5,1,3,6,-7,7,2]], dtype = "uint8")#candidate|296|(1, 330)|const|uint8
call_295 = relay.TupleGetItem(func_218_call(relay.reshape(uop_287.astype('uint8'), []), relay.reshape(const_296.astype('uint8'), [5, 6, 11]), relay.reshape(const_296.astype('float64'), [5, 6, 11]), ), 1)
call_297 = relay.TupleGetItem(func_222_call(relay.reshape(uop_287.astype('uint8'), []), relay.reshape(const_296.astype('uint8'), [5, 6, 11]), relay.reshape(const_296.astype('float64'), [5, 6, 11]), ), 1)
uop_298 = relay.sigmoid(bop_284.astype('float32')) # shape=()
bop_300 = relay.greater_equal(var_270.astype('bool'), uop_287.astype('bool')) # shape=()
uop_303 = relay.cos(uop_298.astype('float32')) # shape=()
uop_305 = relay.acosh(uop_303.astype('float32')) # shape=()
uop_307 = relay.rsqrt(uop_305.astype('float64')) # shape=()
func_68_call = mod.get_global_var('func_68')
func_72_call = mutated_mod.get_global_var('func_72')
var_310 = relay.var("var_310", dtype = "float32", shape = (6,))#candidate|310|(6,)|var|float32
const_311 = relay.const([True,False,False,True], dtype = "bool")#candidate|311|(4,)|const|bool
call_309 = relay.TupleGetItem(func_68_call(relay.reshape(uop_305.astype('int8'), []), relay.reshape(var_310.astype('float32'), [6,]), relay.reshape(const_311.astype('bool'), [4,]), ), 0)
call_312 = relay.TupleGetItem(func_72_call(relay.reshape(uop_305.astype('int8'), []), relay.reshape(var_310.astype('float32'), [6,]), relay.reshape(const_311.astype('bool'), [4,]), ), 0)
uop_313 = relay.rsqrt(uop_305.astype('float32')) # shape=()
var_315 = relay.var("var_315", dtype = "float32", shape = (2,))#candidate|315|(2,)|var|float32
bop_316 = relay.minimum(uop_313.astype('uint8'), var_315.astype('uint8')) # shape=(2,)
uop_319 = relay.sigmoid(uop_303.astype('float32')) # shape=()
uop_321 = relay.asin(uop_307.astype('float64')) # shape=()
var_323 = relay.var("var_323", dtype = "float64", shape = (7,))#candidate|323|(7,)|var|float64
bop_324 = relay.power(uop_321.astype('float64'), var_323.astype('float64')) # shape=(7,)
uop_327 = relay.cos(bop_324.astype('float32')) # shape=(7,)
bop_329 = relay.mod(bop_324.astype('float64'), uop_307.astype('float64')) # shape=(7,)
bop_332 = relay.multiply(uop_321.astype('uint64'), bop_281.astype('uint64')) # shape=(7, 13, 14)
uop_335 = relay.sin(uop_327.astype('float32')) # shape=(7,)
bop_337 = relay.floor_divide(uop_335.astype('float64'), uop_305.astype('float64')) # shape=(7,)
uop_340 = relay.cos(bop_337.astype('float64')) # shape=(7,)
uop_342 = relay.sqrt(bop_337.astype('float64')) # shape=(7,)
bop_344 = relay.divide(uop_335.astype('float32'), relay.reshape(uop_327.astype('float32'), relay.shape_of(uop_335))) # shape=(7,)
output = relay.Tuple([bop_271,bop_275,bop_292,call_295,const_296,bop_300,call_309,var_310,const_311,bop_316,uop_319,bop_329,bop_332,uop_340,uop_342,bop_344,])
output2 = relay.Tuple([bop_271,bop_275,bop_292,call_297,const_296,bop_300,call_312,var_310,const_311,bop_316,uop_319,bop_329,bop_332,uop_340,uop_342,bop_344,])
func_347 = relay.Function([var_269,var_270,var_280,var_310,var_315,var_323,], output)
mod['func_347'] = func_347
mod = relay.transform.InferType()(mod)
mutated_mod['func_347'] = func_347
mutated_mod = relay.transform.InferType()(mutated_mod)
func_347_call = mutated_mod.get_global_var('func_347')
var_349 = relay.var("var_349", dtype = "float64", shape = ())#candidate|349|()|var|float64
var_350 = relay.var("var_350", dtype = "float64", shape = ())#candidate|350|()|var|float64
var_351 = relay.var("var_351", dtype = "float64", shape = (7, 13, 14))#candidate|351|(7, 13, 14)|var|float64
var_352 = relay.var("var_352", dtype = "float32", shape = (6,))#candidate|352|(6,)|var|float32
var_353 = relay.var("var_353", dtype = "float32", shape = (2,))#candidate|353|(2,)|var|float32
var_354 = relay.var("var_354", dtype = "float64", shape = (7,))#candidate|354|(7,)|var|float64
call_348 = func_347_call(var_349,var_350,var_351,var_352,var_353,var_354,)
output = call_348
func_355 = relay.Function([var_349,var_350,var_351,var_352,var_353,var_354,], output)
mutated_mod['func_355'] = func_355
mutated_mod = relay.transform.InferType()(mutated_mod)
var_357 = relay.var("var_357", dtype = "uint64", shape = ())#candidate|357|()|var|uint64
var_358 = relay.var("var_358", dtype = "uint64", shape = ())#candidate|358|()|var|uint64
bop_359 = relay.less_equal(var_357.astype('bool'), var_358.astype('bool')) # shape=()
var_362 = relay.var("var_362", dtype = "uint64", shape = ())#candidate|362|()|var|uint64
bop_363 = relay.subtract(var_357.astype('int16'), var_362.astype('int16')) # shape=()
const_366 = relay.const(-5, dtype = "uint64")#candidate|366|()|const|uint64
bop_367 = relay.left_shift(var_357.astype('int64'), const_366.astype('int64')) # shape=()
uop_370 = relay.acos(bop_359.astype('float64')) # shape=()
bop_372 = relay.less(var_357.astype('bool'), var_358.astype('bool')) # shape=()
output = relay.Tuple([bop_363,bop_367,uop_370,bop_372,])
output2 = relay.Tuple([bop_363,bop_367,uop_370,bop_372,])
F = relay.Function([var_357,var_358,var_362,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_357,var_358,var_362,], output2)
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
	relay.transform.FuseOps(3),
	relay.transform.DefuseOps(),
	relay.transform.SimplifyExpr(),
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
input_357= np.array(-4, dtype='uint64')
module1.set_input('var_357', input_357)
input_358= np.array(4, dtype='uint64')
module1.set_input('var_358', input_358)
input_362= np.array(-6, dtype='uint64')
module1.set_input('var_362', input_362)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_357, input_358, input_362, )
res3 = intrp3.evaluate()(input_357, input_358, input_362, )
res4 = intrp4.evaluate()(input_357, input_358, input_362, )
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
module5.set_input('var_357', input_357)
module5.set_input('var_358', input_358)
module5.set_input('var_362', input_362)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_357, input_358, input_362, )
res7 = intrp7.evaluate()(input_357, input_358, input_362, )
res8 = intrp8.evaluate()(input_357, input_358, input_362, )
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
module9.set_input('var_357', input_357)
module9.set_input('var_358', input_358)
module9.set_input('var_362', input_362)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_357, input_358, input_362, )
res11 = intrp11.evaluate()(input_357, input_358, input_362, )
res12 = intrp12.evaluate()(input_357, input_358, input_362, )
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
module13.set_input('var_357', input_357)
module13.set_input('var_358', input_358)
module13.set_input('var_362', input_362)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_357, input_358, input_362, )
res15 = intrp15.evaluate()(input_357, input_358, input_362, )
res16 = intrp16.evaluate()(input_357, input_358, input_362, )
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
module17.set_input('var_357', input_357)
module17.set_input('var_358', input_358)
module17.set_input('var_362', input_362)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_357, input_358, input_362, )
res19 = intrp19.evaluate()(input_357, input_358, input_362, )
res20 = intrp20.evaluate()(input_357, input_358, input_362, )
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
module21.set_input('var_357', input_357)
module21.set_input('var_358', input_358)
module21.set_input('var_362', input_362)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_357, input_358, input_362, )
res23 = intrp23.evaluate()(input_357, input_358, input_362, )
res24 = intrp24.evaluate()(input_357, input_358, input_362, )
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
2: tvm::tir::DataTypeRewriter::VisitExpr_(tvm::tir::CallNode const*)
1: tvm::operator<<(tvm::PrimExpr, tvm::PrimExpr)
0: tvm::left_shift(tvm::PrimExpr, tvm::PrimExpr, tvm::Span)

'''