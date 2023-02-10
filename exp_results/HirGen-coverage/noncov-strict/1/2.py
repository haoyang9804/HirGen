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
var_0 = relay.var("var_0", dtype = "uint16", shape = ())#candidate|0|()|var|uint16
var_1 = relay.var("var_1", dtype = "uint16", shape = (5, 7, 13))#candidate|1|(5, 7, 13)|var|uint16
bop_2 = relay.logical_xor(var_0.astype('uint16'), var_1.astype('uint16')) # shape=(5, 7, 13)
bop_5 = relay.less(var_0.astype('bool'), bop_2.astype('bool')) # shape=(5, 7, 13)
var_8 = relay.var("var_8", dtype = "uint16", shape = (5, 7, 13))#candidate|8|(5, 7, 13)|var|uint16
bop_9 = relay.mod(bop_2.astype('float32'), relay.reshape(var_8.astype('float32'), relay.shape_of(bop_2))) # shape=(5, 7, 13)
uop_12 = relay.atanh(bop_5.astype('float64')) # shape=(5, 7, 13)
var_14 = relay.var("var_14", dtype = "uint16", shape = (5, 7, 13))#candidate|14|(5, 7, 13)|var|uint16
bop_15 = relay.divide(bop_2.astype('float64'), relay.reshape(var_14.astype('float64'), relay.shape_of(bop_2))) # shape=(5, 7, 13)
var_18 = relay.var("var_18", dtype = "float64", shape = (5, 7, 13))#candidate|18|(5, 7, 13)|var|float64
bop_19 = relay.not_equal(uop_12.astype('bool'), relay.reshape(var_18.astype('bool'), relay.shape_of(uop_12))) # shape=(5, 7, 13)
uop_22 = relay.asinh(bop_19.astype('float32')) # shape=(5, 7, 13)
uop_24 = relay.log2(uop_22.astype('float64')) # shape=(5, 7, 13)
bop_26 = relay.floor_divide(uop_24.astype('float64'), relay.reshape(bop_2.astype('float64'), relay.shape_of(uop_24))) # shape=(5, 7, 13)
bop_29 = relay.left_shift(bop_26.astype('int64'), relay.reshape(bop_9.astype('int64'), relay.shape_of(bop_26))) # shape=(5, 7, 13)
uop_32 = relay.exp(uop_24.astype('float64')) # shape=(5, 7, 13)
bop_34 = relay.logical_xor(uop_32.astype('int32'), relay.reshape(var_14.astype('int32'), relay.shape_of(uop_32))) # shape=(5, 7, 13)
bop_37 = relay.add(uop_32.astype('uint8'), relay.reshape(uop_24.astype('uint8'), relay.shape_of(uop_32))) # shape=(5, 7, 13)
var_40 = relay.var("var_40", dtype = "float32", shape = (5, 7, 13))#candidate|40|(5, 7, 13)|var|float32
bop_41 = relay.subtract(uop_22.astype('int32'), relay.reshape(var_40.astype('int32'), relay.shape_of(uop_22))) # shape=(5, 7, 13)
const_44 = relay.const([[[7,5,3,-1,9,-5,-10,-3,8,9,-7,-9,-3],[-7,-6,-8,8,-8,9,3,7,-9,2,2,-5,-5],[3,1,9,-4,-2,10,6,-4,-7,-4,-3,-3,-3],[9,5,2,10,-1,1,5,-8,5,3,10,4,-4],[7,4,-10,1,8,1,-9,6,10,3,-6,-3,-8],[9,-6,-1,-8,-6,7,-4,-3,3,-1,-1,-2,-10],[-5,-7,3,6,-9,-6,-1,5,-2,-3,6,-6,-6]],[[-8,5,9,1,4,-2,-7,5,-7,-6,4,-1,-2],[6,-6,6,3,3,1,7,-10,-6,-1,-8,9,1],[3,-4,-1,-2,-3,2,-2,-7,9,-10,5,3,1],[4,-9,-5,9,5,10,-6,8,8,-2,-4,-10,-9],[-2,-9,-8,-5,8,-7,-1,-6,2,5,-1,-1,-9],[6,-3,4,-9,-8,-8,4,-5,4,3,-9,3,8],[-9,10,6,-7,3,4,-6,1,-7,9,-3,-8,-4]],[[-3,10,-4,-2,7,10,-9,-7,7,7,-7,-10,-8],[1,1,-4,-4,-8,1,3,-8,-4,-8,-9,-6,2],[3,-2,-4,7,-10,-10,5,-5,5,-6,2,9,-3],[-4,5,4,5,6,-9,-5,-4,-3,2,-5,5,2],[-3,7,7,7,-7,-9,7,4,-10,4,5,-5,7],[-3,-3,-2,-6,-6,-10,7,6,-2,1,10,4,-7],[3,-5,4,-5,-3,-5,5,3,6,5,-6,-6,10]],[[10,10,-4,-9,-3,8,-10,-4,1,-8,-7,4,8],[-6,7,-8,3,-4,1,5,-6,2,6,-3,-7,3],[5,3,-7,8,6,-1,-4,-5,10,-8,5,-4,-8],[-1,-8,-9,2,-8,3,-3,2,8,-10,-10,5,9],[-10,4,-10,4,-7,-4,-5,8,-6,8,6,5,-1],[8,-5,-7,-2,4,3,-1,-9,1,-5,5,-1,-1],[-1,7,-3,9,4,2,6,-5,4,-3,-10,8,-5]],[[-1,7,7,10,9,9,-1,3,-6,-1,-10,-10,7],[-7,-7,7,-1,2,2,2,-7,-1,-3,-4,-3,-1],[-8,-6,-2,10,-5,9,-3,2,7,1,4,-2,6],[7,1,-4,10,3,-1,1,-2,-2,3,2,-9,-4],[5,-1,4,-2,-7,-3,6,-8,1,-3,-4,-9,10],[-7,-6,-1,8,4,1,6,-2,-6,7,5,7,-2],[-4,8,-4,-2,-5,-3,-6,-5,-3,-6,10,-3,-8]]], dtype = "uint8")#candidate|44|(5, 7, 13)|const|uint8
bop_45 = relay.add(bop_37.astype('int8'), relay.reshape(const_44.astype('int8'), relay.shape_of(bop_37))) # shape=(5, 7, 13)
bop_48 = relay.floor_mod(bop_34.astype('float32'), relay.reshape(uop_24.astype('float32'), relay.shape_of(bop_34))) # shape=(5, 7, 13)
bop_51 = relay.logical_or(uop_24.astype('bool'), relay.reshape(bop_26.astype('bool'), relay.shape_of(uop_24))) # shape=(5, 7, 13)
bop_54 = relay.bitwise_or(uop_32.astype('int32'), relay.reshape(var_8.astype('int32'), relay.shape_of(uop_32))) # shape=(5, 7, 13)
var_57 = relay.var("var_57", dtype = "int8", shape = (5, 7, 13))#candidate|57|(5, 7, 13)|var|int8
bop_58 = relay.logical_and(bop_45.astype('bool'), relay.reshape(var_57.astype('bool'), relay.shape_of(bop_45))) # shape=(5, 7, 13)
uop_61 = relay.cosh(bop_41.astype('float64')) # shape=(5, 7, 13)
uop_63 = relay.asin(uop_32.astype('float64')) # shape=(5, 7, 13)
bop_65 = relay.floor_mod(uop_22.astype('float32'), relay.reshape(uop_61.astype('float32'), relay.shape_of(uop_22))) # shape=(5, 7, 13)
output = relay.Tuple([bop_15,bop_29,bop_48,bop_51,bop_54,bop_58,uop_63,bop_65,])
output2 = relay.Tuple([bop_15,bop_29,bop_48,bop_51,bop_54,bop_58,uop_63,bop_65,])
func_68 = relay.Function([var_0,var_1,var_8,var_14,var_18,var_40,var_57,], output)
mod['func_68'] = func_68
mod = relay.transform.InferType()(mod)
var_69 = relay.var("var_69", dtype = "uint16", shape = ())#candidate|69|()|var|uint16
var_70 = relay.var("var_70", dtype = "uint16", shape = (5, 7, 13))#candidate|70|(5, 7, 13)|var|uint16
var_71 = relay.var("var_71", dtype = "uint16", shape = (5, 7, 13))#candidate|71|(5, 7, 13)|var|uint16
var_72 = relay.var("var_72", dtype = "uint16", shape = (5, 7, 13))#candidate|72|(5, 7, 13)|var|uint16
var_73 = relay.var("var_73", dtype = "float64", shape = (5, 7, 13))#candidate|73|(5, 7, 13)|var|float64
var_74 = relay.var("var_74", dtype = "float32", shape = (5, 7, 13))#candidate|74|(5, 7, 13)|var|float32
var_75 = relay.var("var_75", dtype = "int8", shape = (5, 7, 13))#candidate|75|(5, 7, 13)|var|int8
output = func_68(var_69,var_70,var_71,var_72,var_73,var_74,var_75,)
func_76 = relay.Function([var_69,var_70,var_71,var_72,var_73,var_74,var_75,], output)
mutated_mod['func_76'] = func_76
mutated_mod = relay.transform.InferType()(mutated_mod)
var_78 = relay.var("var_78", dtype = "int16", shape = (13, 9))#candidate|78|(13, 9)|var|int16
var_79 = relay.var("var_79", dtype = "int16", shape = (13, 9))#candidate|79|(13, 9)|var|int16
bop_80 = relay.greater(var_78.astype('bool'), relay.reshape(var_79.astype('bool'), relay.shape_of(var_78))) # shape=(13, 9)
uop_83 = relay.exp(var_79.astype('float64')) # shape=(13, 9)
bop_85 = relay.less_equal(uop_83.astype('bool'), relay.reshape(bop_80.astype('bool'), relay.shape_of(uop_83))) # shape=(13, 9)
output = relay.Tuple([bop_85,])
output2 = relay.Tuple([bop_85,])
func_88 = relay.Function([var_78,var_79,], output)
mod['func_88'] = func_88
mod = relay.transform.InferType()(mod)
var_89 = relay.var("var_89", dtype = "int16", shape = (13, 9))#candidate|89|(13, 9)|var|int16
var_90 = relay.var("var_90", dtype = "int16", shape = (13, 9))#candidate|90|(13, 9)|var|int16
output = func_88(var_89,var_90,)
func_91 = relay.Function([var_89,var_90,], output)
mutated_mod['func_91'] = func_91
mutated_mod = relay.transform.InferType()(mutated_mod)
const_93 = relay.const([7,4,-5,1,-2,1,-3,-6,-3,-1,9], dtype = "uint32")#candidate|93|(11,)|const|uint32
var_94 = relay.var("var_94", dtype = "uint32", shape = (11,))#candidate|94|(11,)|var|uint32
bop_95 = relay.left_shift(const_93.astype('uint32'), relay.reshape(var_94.astype('uint32'), relay.shape_of(const_93))) # shape=(11,)
bop_98 = relay.greater(var_94.astype('bool'), relay.reshape(bop_95.astype('bool'), relay.shape_of(var_94))) # shape=(11,)
bop_101 = relay.logical_and(bop_95.astype('bool'), relay.reshape(const_93.astype('bool'), relay.shape_of(bop_95))) # shape=(11,)
output = relay.Tuple([bop_98,bop_101,])
output2 = relay.Tuple([bop_98,bop_101,])
func_104 = relay.Function([var_94,], output)
mod['func_104'] = func_104
mod = relay.transform.InferType()(mod)
var_105 = relay.var("var_105", dtype = "uint32", shape = (11,))#candidate|105|(11,)|var|uint32
output = func_104(var_105)
func_106 = relay.Function([var_105], output)
mutated_mod['func_106'] = func_106
mutated_mod = relay.transform.InferType()(mutated_mod)
var_108 = relay.var("var_108", dtype = "float64", shape = (14,))#candidate|108|(14,)|var|float64
uop_109 = relay.asin(var_108.astype('float64')) # shape=(14,)
uop_111 = relay.sigmoid(uop_109.astype('float64')) # shape=(14,)
var_113 = relay.var("var_113", dtype = "float64", shape = (14,))#candidate|113|(14,)|var|float64
bop_114 = relay.less_equal(uop_109.astype('bool'), relay.reshape(var_113.astype('bool'), relay.shape_of(uop_109))) # shape=(14,)
output = relay.Tuple([uop_111,bop_114,])
output2 = relay.Tuple([uop_111,bop_114,])
func_117 = relay.Function([var_108,var_113,], output)
mod['func_117'] = func_117
mod = relay.transform.InferType()(mod)
var_118 = relay.var("var_118", dtype = "float64", shape = (14,))#candidate|118|(14,)|var|float64
var_119 = relay.var("var_119", dtype = "float64", shape = (14,))#candidate|119|(14,)|var|float64
output = func_117(var_118,var_119,)
func_120 = relay.Function([var_118,var_119,], output)
mutated_mod['func_120'] = func_120
mutated_mod = relay.transform.InferType()(mutated_mod)
var_122 = relay.var("var_122", dtype = "float64", shape = (12, 6, 10))#candidate|122|(12, 6, 10)|var|float64
var_123 = relay.var("var_123", dtype = "float64", shape = (12, 6, 10))#candidate|123|(12, 6, 10)|var|float64
bop_124 = relay.floor_divide(var_122.astype('float64'), relay.reshape(var_123.astype('float64'), relay.shape_of(var_122))) # shape=(12, 6, 10)
bop_127 = relay.floor_mod(bop_124.astype('float32'), relay.reshape(var_123.astype('float32'), relay.shape_of(bop_124))) # shape=(12, 6, 10)
uop_130 = relay.tan(var_123.astype('float64')) # shape=(12, 6, 10)
var_132 = relay.var("var_132", dtype = "float64", shape = (12, 6, 10))#candidate|132|(12, 6, 10)|var|float64
bop_133 = relay.bitwise_and(uop_130.astype('uint16'), relay.reshape(var_132.astype('uint16'), relay.shape_of(uop_130))) # shape=(12, 6, 10)
uop_136 = relay.cosh(bop_133.astype('float32')) # shape=(12, 6, 10)
uop_138 = relay.acosh(var_122.astype('float64')) # shape=(12, 6, 10)
var_140 = relay.var("var_140", dtype = "float64", shape = (12, 6, 10))#candidate|140|(12, 6, 10)|var|float64
bop_141 = relay.not_equal(uop_130.astype('bool'), relay.reshape(var_140.astype('bool'), relay.shape_of(uop_130))) # shape=(12, 6, 10)
var_144 = relay.var("var_144", dtype = "float32", shape = (12, 6, 10))#candidate|144|(12, 6, 10)|var|float32
bop_145 = relay.subtract(uop_136.astype('uint64'), relay.reshape(var_144.astype('uint64'), relay.shape_of(uop_136))) # shape=(12, 6, 10)
const_148 = relay.const([[[-9,-8,-5,-5,-5,7,5,6,-7,-6],[8,6,10,5,6,1,-6,-6,-9,8],[-5,9,10,6,2,10,3,10,-3,3],[9,-7,-5,8,3,-6,-2,10,2,-1],[-7,-4,8,4,-9,3,10,-4,2,-7],[-7,-3,9,3,9,-6,7,-7,-4,-6]],[[-5,-2,8,10,-8,-5,3,8,-1,-5],[-6,-7,3,-6,4,-3,-1,-5,-10,9],[3,5,-3,-10,-5,-5,8,-6,7,-9],[3,9,6,5,9,6,-9,10,5,6],[-6,2,4,-6,-9,-1,8,10,-4,-10],[10,-6,6,-3,-7,2,10,9,2,-7]],[[10,-5,10,2,2,-8,7,-3,2,-5],[-5,-4,1,-10,-1,-5,-5,-9,6,8],[5,-10,1,-5,-9,6,7,8,-5,2],[5,7,8,6,9,-1,1,-7,7,7],[-4,2,5,1,10,6,-9,-7,8,2],[-1,-4,-6,-7,4,-3,-9,10,-6,-3]],[[-4,-2,-1,3,-4,9,-8,10,-1,-1],[2,-3,-8,6,8,10,8,9,1,-9],[5,-2,8,-10,-8,-7,-4,3,5,-8],[1,-7,8,-10,-5,5,8,6,-7,8],[7,8,-6,-5,-2,-4,-1,-6,6,-9],[7,-5,-1,-3,-7,-6,6,6,-4,-3]],[[-6,-1,-9,-5,5,9,9,-8,-5,1],[-4,8,10,2,-7,-4,1,9,4,-6],[-9,2,-9,8,-5,-5,7,-9,-9,3],[-5,8,-4,-5,-10,-7,7,7,-4,-9],[-4,-8,-10,-10,-7,3,-1,7,8,2],[-6,-5,10,7,6,3,-10,3,-7,9]],[[7,7,2,1,-7,-8,-4,5,4,10],[-8,2,-3,8,-7,5,8,1,-7,2],[7,-8,-2,7,-5,-5,-1,8,1,3],[4,5,-3,4,9,5,-5,-1,7,2],[-6,-2,-4,-8,-8,-7,2,-6,8,-4],[-7,7,3,8,5,-10,9,-5,-4,-1]],[[5,-7,-2,1,-3,9,3,4,3,3],[-4,-8,-5,-5,7,1,6,10,-1,7],[3,7,-3,1,3,-3,-7,9,3,2],[1,-3,-3,2,5,1,-5,-5,7,4],[10,-6,-8,2,10,3,-9,-6,-3,-8],[10,-6,8,-5,5,-7,-7,-3,-10,8]],[[9,9,2,-10,-8,1,10,2,9,4],[-7,1,6,10,-1,5,-2,7,4,6],[10,7,-7,9,-8,-1,6,8,-10,-7],[9,-10,-8,2,8,3,7,9,1,-9],[-1,-4,4,8,-2,-4,10,-9,-5,-2],[3,1,7,4,-8,-7,-5,7,-2,-5]],[[1,-5,-5,9,7,3,7,10,5,-2],[-8,1,9,-7,2,-7,-1,-4,-7,4],[-6,-6,10,9,-9,-8,1,4,-9,10],[1,6,10,-6,-7,-10,7,6,-1,5],[-5,1,-6,5,-4,-1,-4,8,-9,10],[-7,8,6,5,10,-9,5,-10,-5,-8]],[[-2,5,-6,-10,9,10,10,-2,-4,6],[-7,9,-7,2,-6,10,-9,-5,-9,-9],[-5,-6,9,-3,-7,6,-6,-7,-6,2],[9,-3,8,-3,-5,7,1,6,3,10],[-8,3,3,-4,9,2,-9,-8,3,6],[-2,-7,-6,1,-2,-1,-9,-4,-6,-3]],[[8,5,-8,-6,4,6,-3,-9,6,10],[2,10,-6,8,-5,-7,4,-2,-4,-3],[8,-6,-4,2,4,-10,3,-2,4,-3],[10,-4,1,-9,2,-5,3,3,-6,-9],[-1,1,-9,-5,4,2,7,8,-1,4],[9,8,-3,-1,-8,-1,9,4,-6,1]],[[-6,-2,10,-10,-5,6,4,-5,-7,-5],[-7,5,-10,-6,-6,4,1,-5,-6,-3],[9,-6,-5,9,-9,6,-3,-4,8,7],[10,-7,9,-3,7,7,-7,-9,4,-4],[-4,-1,-1,-10,-8,6,10,8,-7,4],[8,2,-4,2,10,-3,8,1,-10,10]]], dtype = "uint64")#candidate|148|(12, 6, 10)|const|uint64
bop_149 = relay.logical_or(bop_145.astype('bool'), relay.reshape(const_148.astype('bool'), relay.shape_of(bop_145))) # shape=(12, 6, 10)
uop_152 = relay.rsqrt(uop_136.astype('float64')) # shape=(12, 6, 10)
bop_154 = relay.maximum(uop_138.astype('uint16'), relay.reshape(bop_145.astype('uint16'), relay.shape_of(uop_138))) # shape=(12, 6, 10)
uop_157 = relay.cosh(bop_149.astype('float32')) # shape=(12, 6, 10)
bop_159 = relay.right_shift(uop_130.astype('uint64'), relay.reshape(var_140.astype('uint64'), relay.shape_of(uop_130))) # shape=(12, 6, 10)
uop_162 = relay.sin(bop_149.astype('float64')) # shape=(12, 6, 10)
var_164 = relay.var("var_164", dtype = "float64", shape = (12, 6, 10))#candidate|164|(12, 6, 10)|var|float64
bop_165 = relay.logical_xor(uop_162.astype('uint32'), relay.reshape(var_164.astype('uint32'), relay.shape_of(uop_162))) # shape=(12, 6, 10)
output = relay.Tuple([bop_127,bop_141,uop_152,bop_154,uop_157,bop_159,bop_165,])
output2 = relay.Tuple([bop_127,bop_141,uop_152,bop_154,uop_157,bop_159,bop_165,])
func_168 = relay.Function([var_122,var_123,var_132,var_140,var_144,var_164,], output)
mod['func_168'] = func_168
mod = relay.transform.InferType()(mod)
mutated_mod['func_168'] = func_168
mutated_mod = relay.transform.InferType()(mutated_mod)
func_168_call = mutated_mod.get_global_var('func_168')
var_170 = relay.var("var_170", dtype = "float64", shape = (12, 6, 10))#candidate|170|(12, 6, 10)|var|float64
var_171 = relay.var("var_171", dtype = "float64", shape = (12, 6, 10))#candidate|171|(12, 6, 10)|var|float64
var_172 = relay.var("var_172", dtype = "float64", shape = (12, 6, 10))#candidate|172|(12, 6, 10)|var|float64
var_173 = relay.var("var_173", dtype = "float64", shape = (12, 6, 10))#candidate|173|(12, 6, 10)|var|float64
var_174 = relay.var("var_174", dtype = "float32", shape = (12, 6, 10))#candidate|174|(12, 6, 10)|var|float32
var_175 = relay.var("var_175", dtype = "float64", shape = (12, 6, 10))#candidate|175|(12, 6, 10)|var|float64
call_169 = func_168_call(var_170,var_171,var_172,var_173,var_174,var_175,)
output = call_169
func_176 = relay.Function([var_170,var_171,var_172,var_173,var_174,var_175,], output)
mutated_mod['func_176'] = func_176
mutated_mod = relay.transform.InferType()(mutated_mod)
var_178 = relay.var("var_178", dtype = "float64", shape = (8, 12))#candidate|178|(8, 12)|var|float64
uop_179 = relay.sinh(var_178.astype('float64')) # shape=(8, 12)
uop_181 = relay.rsqrt(uop_179.astype('float64')) # shape=(8, 12)
uop_183 = relay.cos(uop_181.astype('float64')) # shape=(8, 12)
output = relay.Tuple([uop_183,])
output2 = relay.Tuple([uop_183,])
func_185 = relay.Function([var_178,], output)
mod['func_185'] = func_185
mod = relay.transform.InferType()(mod)
var_186 = relay.var("var_186", dtype = "float64", shape = (8, 12))#candidate|186|(8, 12)|var|float64
output = func_185(var_186)
func_187 = relay.Function([var_186], output)
mutated_mod['func_187'] = func_187
mutated_mod = relay.transform.InferType()(mutated_mod)
var_189 = relay.var("var_189", dtype = "float64", shape = (13, 10, 7))#candidate|189|(13, 10, 7)|var|float64
uop_190 = relay.atanh(var_189.astype('float64')) # shape=(13, 10, 7)
output = relay.Tuple([uop_190,])
output2 = relay.Tuple([uop_190,])
func_192 = relay.Function([var_189,], output)
mod['func_192'] = func_192
mod = relay.transform.InferType()(mod)
mutated_mod['func_192'] = func_192
mutated_mod = relay.transform.InferType()(mutated_mod)
var_193 = relay.var("var_193", dtype = "float64", shape = (13, 10, 7))#candidate|193|(13, 10, 7)|var|float64
func_192_call = mutated_mod.get_global_var('func_192')
call_194 = func_192_call(var_193)
output = call_194
func_195 = relay.Function([var_193], output)
mutated_mod['func_195'] = func_195
mutated_mod = relay.transform.InferType()(mutated_mod)
var_197 = relay.var("var_197", dtype = "float32", shape = (2, 1))#candidate|197|(2, 1)|var|float32
uop_198 = relay.log10(var_197.astype('float32')) # shape=(2, 1)
var_200 = relay.var("var_200", dtype = "float32", shape = (2, 7))#candidate|200|(2, 7)|var|float32
bop_201 = relay.divide(var_197.astype('float32'), var_200.astype('float32')) # shape=(2, 7)
uop_204 = relay.asin(uop_198.astype('float64')) # shape=(2, 1)
var_206 = relay.var("var_206", dtype = "float32", shape = (2, 1))#candidate|206|(2, 1)|var|float32
bop_207 = relay.bitwise_or(uop_198.astype('int32'), relay.reshape(var_206.astype('int32'), relay.shape_of(uop_198))) # shape=(2, 1)
uop_210 = relay.log(uop_204.astype('float64')) # shape=(2, 1)
bop_212 = relay.multiply(uop_210.astype('float64'), bop_201.astype('float64')) # shape=(2, 7)
uop_215 = relay.asin(bop_212.astype('float64')) # shape=(2, 7)
bop_217 = relay.less(uop_198.astype('bool'), bop_201.astype('bool')) # shape=(2, 7)
var_220 = relay.var("var_220", dtype = "float64", shape = (2, 7))#candidate|220|(2, 7)|var|float64
bop_221 = relay.subtract(uop_215.astype('int32'), relay.reshape(var_220.astype('int32'), relay.shape_of(uop_215))) # shape=(2, 7)
uop_224 = relay.tan(uop_204.astype('float64')) # shape=(2, 1)
uop_226 = relay.tan(uop_215.astype('float64')) # shape=(2, 7)
bop_228 = relay.not_equal(bop_221.astype('bool'), relay.reshape(bop_217.astype('bool'), relay.shape_of(bop_221))) # shape=(2, 7)
bop_231 = relay.power(bop_228.astype('float64'), relay.reshape(var_200.astype('float64'), relay.shape_of(bop_228))) # shape=(2, 7)
bop_234 = relay.logical_and(bop_221.astype('bool'), var_197.astype('bool')) # shape=(2, 7)
const_237 = relay.const([[-1.418975,-2.159833,6.133391,6.159754,-1.110283,8.528499,-8.352934],[5.654114,-6.451074,-1.222252,2.204701,4.760154,-7.425093,9.690345]], dtype = "float64")#candidate|237|(2, 7)|const|float64
bop_238 = relay.greater(uop_226.astype('bool'), relay.reshape(const_237.astype('bool'), relay.shape_of(uop_226))) # shape=(2, 7)
uop_241 = relay.log10(uop_204.astype('float64')) # shape=(2, 1)
uop_243 = relay.sinh(uop_215.astype('float32')) # shape=(2, 7)
output = relay.Tuple([bop_207,uop_224,bop_231,bop_234,bop_238,uop_241,uop_243,])
output2 = relay.Tuple([bop_207,uop_224,bop_231,bop_234,bop_238,uop_241,uop_243,])
func_245 = relay.Function([var_197,var_200,var_206,var_220,], output)
mod['func_245'] = func_245
mod = relay.transform.InferType()(mod)
mutated_mod['func_245'] = func_245
mutated_mod = relay.transform.InferType()(mutated_mod)
func_245_call = mutated_mod.get_global_var('func_245')
var_247 = relay.var("var_247", dtype = "float32", shape = (2, 1))#candidate|247|(2, 1)|var|float32
var_248 = relay.var("var_248", dtype = "float32", shape = (2, 7))#candidate|248|(2, 7)|var|float32
var_249 = relay.var("var_249", dtype = "float32", shape = (2, 1))#candidate|249|(2, 1)|var|float32
var_250 = relay.var("var_250", dtype = "float64", shape = (2, 7))#candidate|250|(2, 7)|var|float64
call_246 = func_245_call(var_247,var_248,var_249,var_250,)
output = call_246
func_251 = relay.Function([var_247,var_248,var_249,var_250,], output)
mutated_mod['func_251'] = func_251
mutated_mod = relay.transform.InferType()(mutated_mod)
const_253 = relay.const([[[-2.902030,8.399655,7.570457,-4.605867,-0.755973,5.061518,-3.728526,2.635056,-2.610345,-2.938907,-9.088113],[-5.533138,-5.597188,-4.164393,7.863682,-6.088242,-4.192978,-8.823159,5.441038,3.480634,5.650108,-3.459910]],[[8.478641,0.267622,7.313454,-3.551900,-1.913520,2.553722,8.580616,-7.650054,0.864384,-8.961528,-1.854070],[6.832602,-9.562101,4.684154,-1.158096,1.365988,9.297976,7.851351,-6.748556,-3.490799,-8.006639,4.598581]],[[7.135452,8.474529,8.483792,-4.834855,-9.375505,9.215792,-6.721428,-3.986428,3.550166,-5.849855,-6.500290],[0.923686,2.075467,7.855877,8.709944,2.564412,-9.092225,-7.855100,-6.724353,5.681798,-9.345675,7.757171]],[[9.528759,4.630319,1.755852,8.710615,9.905438,8.671361,-8.789393,9.387619,3.864172,-9.526895,8.118589],[1.069156,9.316617,-1.886371,-5.519515,-3.222225,8.940238,-8.361278,9.943384,-9.409339,-6.934796,4.034194]],[[-7.238880,3.728245,9.261143,7.745269,-7.836165,-6.040955,9.920924,9.527491,6.048101,4.768577,-4.915106],[0.776217,5.945704,-8.330967,2.039475,-8.351009,-6.067783,-9.090019,5.386856,-7.873650,-3.599233,9.765367]],[[2.427114,3.851746,-9.486869,4.999644,-6.587097,-6.204565,-8.724409,-8.504037,-2.486901,2.551482,-8.488683],[-9.020282,7.996713,0.325647,2.758808,6.875912,4.823832,4.028177,4.752686,-5.645113,-5.873605,3.514360]]], dtype = "float32")#candidate|253|(6, 2, 11)|const|float32
uop_254 = relay.rsqrt(const_253.astype('float32')) # shape=(6, 2, 11)
bop_256 = relay.logical_xor(const_253.astype('uint32'), relay.reshape(uop_254.astype('uint32'), relay.shape_of(const_253))) # shape=(6, 2, 11)
uop_259 = relay.exp(uop_254.astype('float64')) # shape=(6, 2, 11)
uop_261 = relay.cosh(bop_256.astype('float32')) # shape=(6, 2, 11)
bop_263 = relay.mod(uop_261.astype('float64'), relay.reshape(const_253.astype('float64'), relay.shape_of(uop_261))) # shape=(6, 2, 11)
uop_266 = relay.log10(uop_254.astype('float64')) # shape=(6, 2, 11)
const_268 = relay.const([[[-4.721883,-0.973326,7.499090,-9.687926,-9.012425,-1.856625,5.229221,2.020084,8.648430,-7.847655,4.531652],[-2.846119,-1.257948,-7.578704,5.839158,0.783022,-5.285678,-4.875928,9.594875,-9.832000,-1.676429,8.962330]],[[-3.930976,7.328770,1.310739,-5.088242,-8.032314,-6.822784,3.431640,5.882025,1.355260,3.925336,6.579115],[8.610324,-1.741964,-1.979029,-2.181150,-6.222317,-7.159372,-6.102317,3.952860,-6.683624,-1.678228,6.182049]],[[9.651942,-2.041629,7.103598,-4.191013,2.575386,2.460937,9.542156,6.918780,9.934477,4.162158,0.292196],[-9.686544,-2.273228,-4.034511,2.220473,2.501957,2.567933,5.232423,7.811421,6.319818,2.344066,-6.075660]],[[-8.483655,8.143748,-1.566002,-1.609108,-4.934716,-3.566164,2.897713,3.331430,-6.834619,-4.120063,-6.378861],[2.854581,7.052727,5.754688,3.072180,-0.339070,9.729881,-7.746664,-9.780947,0.438239,3.472123,9.977621]],[[-6.863376,6.142770,7.070395,-0.464171,-5.688424,5.503166,-1.987302,-0.041749,4.696569,-2.062781,-8.849268],[-6.776030,2.799631,-7.467014,5.921016,5.379078,-8.894863,1.903895,-3.290099,-0.299381,5.632534,4.015775]],[[-2.244410,0.264202,-9.534732,-1.526123,-7.175500,9.783552,-5.526211,-3.103341,6.954635,2.074284,3.622429],[-3.111031,1.260938,-3.952260,-8.090191,-8.145795,9.110916,-3.414461,5.935486,7.338228,4.677047,0.048680]]], dtype = "float64")#candidate|268|(6, 2, 11)|const|float64
bop_269 = relay.logical_or(uop_266.astype('bool'), relay.reshape(const_268.astype('bool'), relay.shape_of(uop_266))) # shape=(6, 2, 11)
var_272 = relay.var("var_272", dtype = "bool", shape = (6, 2, 11))#candidate|272|(6, 2, 11)|var|bool
bop_273 = relay.less(bop_269.astype('bool'), relay.reshape(var_272.astype('bool'), relay.shape_of(bop_269))) # shape=(6, 2, 11)
output = relay.Tuple([uop_259,bop_263,bop_273,])
output2 = relay.Tuple([uop_259,bop_263,bop_273,])
func_276 = relay.Function([var_272,], output)
mod['func_276'] = func_276
mod = relay.transform.InferType()(mod)
mutated_mod['func_276'] = func_276
mutated_mod = relay.transform.InferType()(mutated_mod)
var_277 = relay.var("var_277", dtype = "bool", shape = (6, 2, 11))#candidate|277|(6, 2, 11)|var|bool
func_276_call = mutated_mod.get_global_var('func_276')
call_278 = func_276_call(var_277)
output = call_278
func_279 = relay.Function([var_277], output)
mutated_mod['func_279'] = func_279
mutated_mod = relay.transform.InferType()(mutated_mod)
var_281 = relay.var("var_281", dtype = "float32", shape = (12,))#candidate|281|(12,)|var|float32
uop_282 = relay.cos(var_281.astype('float32')) # shape=(12,)
uop_284 = relay.exp(var_281.astype('float64')) # shape=(12,)
uop_286 = relay.cosh(var_281.astype('float32')) # shape=(12,)
bop_288 = relay.greater_equal(uop_282.astype('bool'), relay.reshape(uop_286.astype('bool'), relay.shape_of(uop_282))) # shape=(12,)
func_168_call = mod.get_global_var('func_168')
func_176_call = mutated_mod.get_global_var('func_176')
var_292 = relay.var("var_292", dtype = "float64", shape = (720,))#candidate|292|(720,)|var|float64
call_291 = relay.TupleGetItem(func_168_call(relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float32'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), ), 5)
call_293 = relay.TupleGetItem(func_176_call(relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), relay.reshape(var_292.astype('float32'), [12, 6, 10]), relay.reshape(var_292.astype('float64'), [12, 6, 10]), ), 5)
uop_294 = relay.cosh(var_292.astype('float32')) # shape=(720,)
bop_296 = relay.bitwise_and(bop_288.astype('int64'), relay.reshape(uop_282.astype('int64'), relay.shape_of(bop_288))) # shape=(12,)
uop_299 = relay.asinh(uop_286.astype('float64')) # shape=(12,)
uop_301 = relay.cos(uop_282.astype('float64')) # shape=(12,)
uop_303 = relay.asin(uop_294.astype('float32')) # shape=(720,)
uop_305 = relay.cos(uop_301.astype('float64')) # shape=(12,)
uop_307 = relay.log2(uop_305.astype('float64')) # shape=(12,)
func_117_call = mod.get_global_var('func_117')
func_120_call = mutated_mod.get_global_var('func_120')
const_310 = relay.const([7.417161,-8.505101,-0.751836,9.798609,-5.062481,-1.890731,-8.705744,0.424471,-4.573712,-1.180973,-7.447104,-0.256587,8.151739,2.323199], dtype = "float64")#candidate|310|(14,)|const|float64
call_309 = relay.TupleGetItem(func_117_call(relay.reshape(const_310.astype('float64'), [14,]), relay.reshape(const_310.astype('float64'), [14,]), ), 0)
call_311 = relay.TupleGetItem(func_120_call(relay.reshape(const_310.astype('float64'), [14,]), relay.reshape(const_310.astype('float64'), [14,]), ), 0)
bop_312 = relay.greater(uop_294.astype('bool'), relay.reshape(call_291.astype('bool'), relay.shape_of(uop_294))) # shape=(720,)
bop_315 = relay.greater(uop_294.astype('bool'), relay.reshape(call_293.astype('bool'), relay.shape_of(uop_294))) # shape=(720,)
output = relay.Tuple([uop_284,bop_296,uop_299,uop_303,uop_307,call_309,const_310,bop_312,])
output2 = relay.Tuple([uop_284,bop_296,uop_299,uop_303,uop_307,call_311,const_310,bop_315,])
func_316 = relay.Function([var_281,var_292,], output)
mod['func_316'] = func_316
mod = relay.transform.InferType()(mod)
mutated_mod['func_316'] = func_316
mutated_mod = relay.transform.InferType()(mutated_mod)
func_316_call = mutated_mod.get_global_var('func_316')
var_318 = relay.var("var_318", dtype = "float32", shape = (12,))#candidate|318|(12,)|var|float32
var_319 = relay.var("var_319", dtype = "float64", shape = (720,))#candidate|319|(720,)|var|float64
call_317 = func_316_call(var_318,var_319,)
output = call_317
func_320 = relay.Function([var_318,var_319,], output)
mutated_mod['func_320'] = func_320
mutated_mod = relay.transform.InferType()(mutated_mod)
var_322 = relay.var("var_322", dtype = "float64", shape = (10, 8, 15))#candidate|322|(10, 8, 15)|var|float64
uop_323 = relay.rsqrt(var_322.astype('float64')) # shape=(10, 8, 15)
func_88_call = mod.get_global_var('func_88')
func_91_call = mutated_mod.get_global_var('func_91')
const_326 = relay.const([5,-6,-1,9,8,-6,-3,-5,6,3,-10,-10,6,3,-4,-1,6,-9,10,9,-2,-8,-1,-9,-4,4,9,9,3,-3,-2,-3,-2,3,2,1,-6,-1,5,-4,2,2,-4,-7,3,-4,-6,3,-1,9,10,-7,10,-1,-6,3,3,-9,6,-5,-6,6,-6,-10,-7,10,1,-4,-5,-8,-2,-9,6,-3,4,3,-7,-1,-2,-1,10,-6,1,3,-5,-7,10,1,5,-10,3,8,4,4,8,5,-9,10,3,5,2,6,-7,-10,4,-1,3,7,-9,-5,4,-7,-5,2,10,-7,-9], dtype = "int16")#candidate|326|(117,)|const|int16
call_325 = relay.TupleGetItem(func_88_call(relay.reshape(const_326.astype('int16'), [13, 9]), relay.reshape(const_326.astype('int16'), [13, 9]), ), 0)
call_327 = relay.TupleGetItem(func_91_call(relay.reshape(const_326.astype('int16'), [13, 9]), relay.reshape(const_326.astype('int16'), [13, 9]), ), 0)
func_68_call = mod.get_global_var('func_68')
func_76_call = mutated_mod.get_global_var('func_76')
const_329 = relay.const(-4, dtype = "uint16")#candidate|329|()|const|uint16
const_330 = relay.const([2,-7,1,9,2,-5,-5,9,10,1,-8,10,6,7,6,4,-2,-1,6,-10,-1,-5,6,10,2,1,-8,-3,10,8,1,-6,-1,-8,8,-7,5,-5,8,6,-9,-7,7,3,3,5,-3,9,-9,-3,6,-8,-4,9,-10,-3,3,-10,-1,8,-4,9,6,-1,6,-4,4,3,-6,-3,9,-7,-5,3,-10,1,5,-3,-10,8,8,6,5,-3,4,-7,-9,5,1,-1,2,7,4,10,4,-7,1,6,5,-5,3,-1,-6,8,8,-10,-3,-8,5,-10,8,-1,5,5,5,10,-3,4,-9,7,-4,-6,4,-4,5,-7,-3,2,-8,-7,-8,-2,-8,-10,-1,-4,6,3,-7,-8,-3,5,-7,-2,-5,-7,6,4,8,-2,-1,-5,8,6,5,8,-10,1,-10,-10,6,8,9,-7,-5,5,5,2,5,9,-1,7,-9,-1,-3,-4,10,-2,-5,3,8,-6,7,-8,-8,-4,5,-2,-10,-3,-6,4,2,-3,5,-9,-6,-9,4,7,6,3,-3,-8,-8,9,-2,9,9,1,10,10,-4,2,6,-2,-3,6,9,-10,-6,2,4,-2,-3,-8,4,10,-4,-7,-4,-5,-1,-2,4,9,2,10,1,8,6,9,-5,7,3,-2,-3,1,-5,-6,8,-2,5,6,-3,-3,2,8,-3,7,-1,3,5,1,1,-10,3,7,2,-7,10,4,2,-9,9,10,-4,-5,9,8,3,8,2,5,1,-10,10,10,6,-6,5,4,-1,10,-4,9,-8,3,-2,-3,8,3,7,-1,-5,9,-3,3,3,3,2,3,8,-10,-4,-5,-6,2,4,-10,8,-10,4,4,-7,-3,10,-8,3,1,3,-2,6,4,-7,10,-4,8,-4,-5,1,9,-5,-10,-9,6,8,-2,-1,-6,-3,-6,9,-4,8,5,7,-1,-2,-1,-3,-2,-6,-7,1,7,-2,-2,5,-5,7,-2,-1,-9,3,4,8,7,10,-2,9,-6,10,-1,4,-10,3,5,5,9,-3,-4,-10,-6,-4,3,6,2,7,-3,-1,-2,-6,-10,-1,9,-5,7,-1,-8,-1,-8,-2,5,6,-10,2,3,9,9,4,3,8,-7,-3,5,5,1,6,7,-5,-10,7,4,-7,-5,10,2,-8,-8,5,-4,-7,-10,-10,-6,1,1,7,5,10,-9,-10,10,-2], dtype = "uint16")#candidate|330|(455,)|const|uint16
call_328 = relay.TupleGetItem(func_68_call(relay.reshape(const_329.astype('uint16'), []), relay.reshape(const_330.astype('uint16'), [5, 7, 13]), relay.reshape(const_330.astype('uint16'), [5, 7, 13]), relay.reshape(const_330.astype('uint16'), [5, 7, 13]), relay.reshape(const_330.astype('float64'), [5, 7, 13]), relay.reshape(const_330.astype('float32'), [5, 7, 13]), relay.reshape(const_330.astype('int8'), [5, 7, 13]), ), 0)
call_331 = relay.TupleGetItem(func_76_call(relay.reshape(const_329.astype('uint16'), []), relay.reshape(const_330.astype('uint16'), [5, 7, 13]), relay.reshape(const_330.astype('uint16'), [5, 7, 13]), relay.reshape(const_330.astype('uint16'), [5, 7, 13]), relay.reshape(const_330.astype('float64'), [5, 7, 13]), relay.reshape(const_330.astype('float32'), [5, 7, 13]), relay.reshape(const_330.astype('int8'), [5, 7, 13]), ), 0)
output = relay.Tuple([uop_323,call_325,const_326,call_328,const_329,const_330,])
output2 = relay.Tuple([uop_323,call_327,const_326,call_331,const_329,const_330,])
func_332 = relay.Function([var_322,], output)
mod['func_332'] = func_332
mod = relay.transform.InferType()(mod)
mutated_mod['func_332'] = func_332
mutated_mod = relay.transform.InferType()(mutated_mod)
var_333 = relay.var("var_333", dtype = "float64", shape = (10, 8, 15))#candidate|333|(10, 8, 15)|var|float64
func_332_call = mutated_mod.get_global_var('func_332')
call_334 = func_332_call(var_333)
output = call_334
func_335 = relay.Function([var_333], output)
mutated_mod['func_335'] = func_335
mutated_mod = relay.transform.InferType()(mutated_mod)
var_337 = relay.var("var_337", dtype = "uint16", shape = (4, 14))#candidate|337|(4, 14)|var|uint16
var_338 = relay.var("var_338", dtype = "uint16", shape = (4, 14))#candidate|338|(4, 14)|var|uint16
bop_339 = relay.bitwise_or(var_337.astype('uint16'), relay.reshape(var_338.astype('uint16'), relay.shape_of(var_337))) # shape=(4, 14)
output = bop_339
output2 = bop_339
func_342 = relay.Function([var_337,var_338,], output)
mod['func_342'] = func_342
mod = relay.transform.InferType()(mod)
var_343 = relay.var("var_343", dtype = "uint16", shape = (4, 14))#candidate|343|(4, 14)|var|uint16
var_344 = relay.var("var_344", dtype = "uint16", shape = (4, 14))#candidate|344|(4, 14)|var|uint16
output = func_342(var_343,var_344,)
func_345 = relay.Function([var_343,var_344,], output)
mutated_mod['func_345'] = func_345
mutated_mod = relay.transform.InferType()(mutated_mod)
const_347 = relay.const([9.569342,-6.766813,5.882459,-0.342835,-2.487728,-3.935142,7.373612], dtype = "float64")#candidate|347|(7,)|const|float64
uop_348 = relay.atanh(const_347.astype('float64')) # shape=(7,)
bop_350 = relay.bitwise_and(uop_348.astype('uint16'), relay.reshape(const_347.astype('uint16'), relay.shape_of(uop_348))) # shape=(7,)
output = bop_350
output2 = bop_350
func_353 = relay.Function([], output)
mod['func_353'] = func_353
mod = relay.transform.InferType()(mod)
output = func_353()
func_354 = relay.Function([], output)
mutated_mod['func_354'] = func_354
mutated_mod = relay.transform.InferType()(mutated_mod)
var_355 = relay.var("var_355", dtype = "int32", shape = (2, 10))#candidate|355|(2, 10)|var|int32
var_356 = relay.var("var_356", dtype = "int32", shape = (2, 10))#candidate|356|(2, 10)|var|int32
bop_357 = relay.equal(var_355.astype('bool'), relay.reshape(var_356.astype('bool'), relay.shape_of(var_355))) # shape=(2, 10)
output = relay.Tuple([bop_357,])
output2 = relay.Tuple([bop_357,])
func_360 = relay.Function([var_355,var_356,], output)
mod['func_360'] = func_360
mod = relay.transform.InferType()(mod)
var_361 = relay.var("var_361", dtype = "int32", shape = (2, 10))#candidate|361|(2, 10)|var|int32
var_362 = relay.var("var_362", dtype = "int32", shape = (2, 10))#candidate|362|(2, 10)|var|int32
output = func_360(var_361,var_362,)
func_363 = relay.Function([var_361,var_362,], output)
mutated_mod['func_363'] = func_363
mutated_mod = relay.transform.InferType()(mutated_mod)
var_365 = relay.var("var_365", dtype = "float64", shape = (8,))#candidate|365|(8,)|var|float64
uop_366 = relay.log10(var_365.astype('float64')) # shape=(8,)
uop_368 = relay.asin(var_365.astype('float32')) # shape=(8,)
var_370 = relay.var("var_370", dtype = "float64", shape = (8,))#candidate|370|(8,)|var|float64
bop_371 = relay.right_shift(var_365.astype('int64'), relay.reshape(var_370.astype('int64'), relay.shape_of(var_365))) # shape=(8,)
uop_374 = relay.sin(var_365.astype('float64')) # shape=(8,)
uop_376 = relay.cosh(uop_366.astype('float64')) # shape=(8,)
output = relay.Tuple([uop_368,bop_371,uop_374,uop_376,])
output2 = relay.Tuple([uop_368,bop_371,uop_374,uop_376,])
func_378 = relay.Function([var_365,var_370,], output)
mod['func_378'] = func_378
mod = relay.transform.InferType()(mod)
var_379 = relay.var("var_379", dtype = "float64", shape = (8,))#candidate|379|(8,)|var|float64
var_380 = relay.var("var_380", dtype = "float64", shape = (8,))#candidate|380|(8,)|var|float64
output = func_378(var_379,var_380,)
func_381 = relay.Function([var_379,var_380,], output)
mutated_mod['func_381'] = func_381
mutated_mod = relay.transform.InferType()(mutated_mod)
var_383 = relay.var("var_383", dtype = "uint8", shape = (9,))#candidate|383|(9,)|var|uint8
var_384 = relay.var("var_384", dtype = "uint8", shape = (9,))#candidate|384|(9,)|var|uint8
bop_385 = relay.left_shift(var_383.astype('uint8'), relay.reshape(var_384.astype('uint8'), relay.shape_of(var_383))) # shape=(9,)
uop_388 = relay.asin(bop_385.astype('float64')) # shape=(9,)
uop_390 = relay.log2(var_383.astype('float32')) # shape=(9,)
output = relay.Tuple([uop_388,uop_390,])
output2 = relay.Tuple([uop_388,uop_390,])
F = relay.Function([var_383,var_384,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_383,var_384,], output2)
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
input_383= np.array([-4,8,-5,-8,7,-1,6,7,-5], dtype='uint8')
module1.set_input('var_383', input_383)
input_384= np.array([-6,-7,-9,-2,-6,7,-5,-10,-7], dtype='uint8')
module1.set_input('var_384', input_384)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_383, input_384, )
res3 = intrp3.evaluate()(input_383, input_384, )
res4 = intrp4.evaluate()(input_383, input_384, )
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
module5.set_input('var_383', input_383)
module5.set_input('var_384', input_384)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_383, input_384, )
res7 = intrp7.evaluate()(input_383, input_384, )
res8 = intrp8.evaluate()(input_383, input_384, )
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
module9.set_input('var_383', input_383)
module9.set_input('var_384', input_384)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_383, input_384, )
res11 = intrp11.evaluate()(input_383, input_384, )
res12 = intrp12.evaluate()(input_383, input_384, )
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
module13.set_input('var_383', input_383)
module13.set_input('var_384', input_384)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_383, input_384, )
res15 = intrp15.evaluate()(input_383, input_384, )
res16 = intrp16.evaluate()(input_383, input_384, )
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
module17.set_input('var_383', input_383)
module17.set_input('var_384', input_384)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_383, input_384, )
res19 = intrp19.evaluate()(input_383, input_384, )
res20 = intrp20.evaluate()(input_383, input_384, )
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
module21.set_input('var_383', input_383)
module21.set_input('var_384', input_384)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_383, input_384, )
res23 = intrp23.evaluate()(input_383, input_384, )
res24 = intrp24.evaluate()(input_383, input_384, )
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

'''28: TVMFuncCall
27: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
26: tvm::transform::Pass::operator()(tvm::IRModule) const
25: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
24: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
23: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
22: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
21: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
20: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
19: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
18: _ZNSt17_Function_handlerIFSt10sha
17: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
16: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
15: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
14: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::TupleNode const*)
13: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''