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
var_0 = relay.var("var_0", dtype = "float64", shape = (3, 4, 6))#candidate|0|(3, 4, 6)|var|float64
uop_1 = relay.sin(var_0.astype('float64')) # shape=(3, 4, 6)
uop_3 = relay.sin(uop_1.astype('float32')) # shape=(3, 4, 6)
bop_5 = relay.power(uop_3.astype('float64'), relay.reshape(var_0.astype('float64'), relay.shape_of(uop_3))) # shape=(3, 4, 6)
var_8 = relay.var("var_8", dtype = "float64", shape = (3, 4, 6))#candidate|8|(3, 4, 6)|var|float64
bop_9 = relay.greater(uop_1.astype('bool'), relay.reshape(var_8.astype('bool'), relay.shape_of(uop_1))) # shape=(3, 4, 6)
var_12 = relay.var("var_12", dtype = "float64", shape = (3, 4, 6))#candidate|12|(3, 4, 6)|var|float64
bop_13 = relay.bitwise_or(uop_1.astype('int8'), relay.reshape(var_12.astype('int8'), relay.shape_of(uop_1))) # shape=(3, 4, 6)
uop_16 = relay.cos(bop_13.astype('float64')) # shape=(3, 4, 6)
uop_18 = relay.cos(bop_13.astype('float64')) # shape=(3, 4, 6)
uop_20 = relay.asinh(bop_9.astype('float32')) # shape=(3, 4, 6)
uop_22 = relay.log(bop_9.astype('float32')) # shape=(3, 4, 6)
bop_24 = relay.maximum(var_12.astype('int16'), relay.reshape(uop_3.astype('int16'), relay.shape_of(var_12))) # shape=(3, 4, 6)
var_27 = relay.var("var_27", dtype = "float64", shape = (3, 4, 6))#candidate|27|(3, 4, 6)|var|float64
bop_28 = relay.logical_and(bop_5.astype('bool'), relay.reshape(var_27.astype('bool'), relay.shape_of(bop_5))) # shape=(3, 4, 6)
bop_31 = relay.divide(bop_9.astype('float64'), relay.reshape(uop_20.astype('float64'), relay.shape_of(bop_9))) # shape=(3, 4, 6)
uop_34 = relay.log10(uop_22.astype('float64')) # shape=(3, 4, 6)
const_36 = relay.const([[[6.314025,6.751006,3.721743,-4.847263,8.080492,9.951873],[-5.336816,3.632841,-2.463959,7.801769,6.935568,5.713502],[-8.052632,2.633284,-9.142546,7.089808,5.823112,7.333780],[-3.871392,1.584693,9.250977,5.120530,-1.803392,-2.139500]],[[-8.964417,-1.910727,-6.209858,3.936932,-3.220768,-6.510321],[6.877680,-7.402432,-2.917712,9.603855,-7.749351,-3.422255],[9.806646,6.120222,7.086292,-1.046341,-8.656268,8.303259],[-1.226192,3.132723,9.671080,6.252117,2.175372,-1.466552]],[[1.673362,0.803985,-6.490805,1.625176,8.577196,6.781979],[6.003704,-7.704014,-3.835070,-6.563683,-4.396478,7.080587],[-2.583813,-6.843687,-8.401746,4.774419,-6.158137,0.659133],[3.593637,0.515754,-1.038762,2.696031,-6.959964,3.409794]]], dtype = "float64")#candidate|36|(3, 4, 6)|const|float64
bop_37 = relay.greater(uop_34.astype('bool'), relay.reshape(const_36.astype('bool'), relay.shape_of(uop_34))) # shape=(3, 4, 6)
uop_40 = relay.asinh(bop_37.astype('float64')) # shape=(3, 4, 6)
bop_42 = relay.equal(uop_40.astype('bool'), relay.reshape(uop_1.astype('bool'), relay.shape_of(uop_40))) # shape=(3, 4, 6)
uop_45 = relay.asinh(bop_5.astype('float32')) # shape=(3, 4, 6)
bop_47 = relay.logical_and(bop_24.astype('bool'), relay.reshape(uop_18.astype('bool'), relay.shape_of(bop_24))) # shape=(3, 4, 6)
bop_50 = relay.minimum(uop_40.astype('uint32'), relay.reshape(uop_1.astype('uint32'), relay.shape_of(uop_40))) # shape=(3, 4, 6)
uop_53 = relay.asinh(bop_28.astype('float32')) # shape=(3, 4, 6)
uop_55 = relay.asin(bop_42.astype('float32')) # shape=(3, 4, 6)
bop_57 = relay.left_shift(uop_55.astype('int64'), relay.reshape(bop_31.astype('int64'), relay.shape_of(uop_55))) # shape=(3, 4, 6)
uop_60 = relay.asinh(bop_42.astype('float32')) # shape=(3, 4, 6)
bop_62 = relay.floor_divide(uop_60.astype('float64'), relay.reshape(bop_42.astype('float64'), relay.shape_of(uop_60))) # shape=(3, 4, 6)
bop_65 = relay.not_equal(bop_62.astype('bool'), relay.reshape(bop_28.astype('bool'), relay.shape_of(bop_62))) # shape=(3, 4, 6)
output = relay.Tuple([uop_16,uop_45,bop_47,bop_50,uop_53,bop_57,bop_65,])
output2 = relay.Tuple([uop_16,uop_45,bop_47,bop_50,uop_53,bop_57,bop_65,])
func_68 = relay.Function([var_0,var_8,var_12,var_27,], output)
mod['func_68'] = func_68
mod = relay.transform.InferType()(mod)
var_69 = relay.var("var_69", dtype = "float64", shape = (3, 4, 6))#candidate|69|(3, 4, 6)|var|float64
var_70 = relay.var("var_70", dtype = "float64", shape = (3, 4, 6))#candidate|70|(3, 4, 6)|var|float64
var_71 = relay.var("var_71", dtype = "float64", shape = (3, 4, 6))#candidate|71|(3, 4, 6)|var|float64
var_72 = relay.var("var_72", dtype = "float64", shape = (3, 4, 6))#candidate|72|(3, 4, 6)|var|float64
output = func_68(var_69,var_70,var_71,var_72,)
func_73 = relay.Function([var_69,var_70,var_71,var_72,], output)
mutated_mod['func_73'] = func_73
mutated_mod = relay.transform.InferType()(mutated_mod)
var_75 = relay.var("var_75", dtype = "uint64", shape = (14, 10, 6))#candidate|75|(14, 10, 6)|var|uint64
const_76 = relay.const([[[3,-7,-6,-4,-10,-9],[-5,6,7,-7,-9,8],[5,8,-2,1,2,2],[4,2,6,4,-2,7],[-2,8,7,-3,-10,6],[-3,7,-4,3,6,-3],[-9,-2,-9,4,3,-7],[8,3,-2,9,-7,-3],[7,8,-3,6,9,-5],[-8,2,-1,-2,-10,6]],[[-5,9,3,-2,8,-6],[2,-3,1,-9,-8,-8],[1,-7,-4,-3,10,-10],[4,3,-1,-2,-10,-4],[-6,-10,6,-3,-10,8],[-5,-5,-6,8,-7,3],[10,3,-2,-1,-9,-5],[10,4,10,4,-4,10],[3,10,-9,-8,-10,-8],[-7,1,1,3,-5,9]],[[-8,1,-3,-3,-9,6],[8,4,4,-10,-3,9],[8,5,-8,9,1,-9],[10,-9,-7,-4,9,-2],[-2,-8,-3,10,10,5],[2,-5,1,-5,7,10],[1,8,-2,4,1,8],[8,5,3,-7,3,8],[-5,-5,-9,6,-8,9],[-5,3,-5,-9,-1,3]],[[6,9,-2,-3,9,1],[4,-7,4,-2,9,2],[-10,-8,-9,3,1,-3],[10,-1,-6,7,10,2],[-10,-10,2,6,6,1],[7,9,9,-2,-1,-5],[5,-5,-8,8,1,-10],[3,5,9,5,1,5],[-9,6,-8,-5,5,-10],[8,4,8,-4,-9,10]],[[-2,-4,-10,-2,6,7],[1,8,3,-10,6,2],[-9,4,-3,5,-1,10],[-10,10,-10,-9,-10,-6],[5,-6,1,4,4,7],[-4,5,-1,-7,-9,6],[-6,2,-4,-9,-6,9],[-4,-7,-2,-8,-7,2],[-1,-9,-7,-10,2,7],[10,-4,-4,6,-7,-6]],[[7,7,-8,4,-1,-10],[9,-9,3,4,5,-2],[2,-5,-8,10,-9,-3],[-8,-6,1,-7,-7,-1],[4,4,5,6,-1,-6],[6,-8,5,-1,10,1],[-5,-2,-5,-3,-2,8],[-2,-7,-9,-3,1,9],[-5,-10,-10,-6,1,5],[3,-7,-4,4,-1,-10]],[[6,-7,-7,4,9,-4],[-6,3,-8,-5,8,-10],[7,10,-7,-10,2,-1],[-6,-10,10,5,6,-5],[-10,-9,-10,-3,-3,7],[-1,-2,-4,3,-1,-6],[-9,-8,1,-1,-4,1],[7,1,5,9,-4,-1],[5,8,-7,6,-7,2],[8,9,2,-5,4,-9]],[[5,6,-1,8,1,2],[10,5,-4,5,10,3],[-6,3,6,10,1,1],[-3,8,4,-3,9,4],[-6,-5,-7,-10,-9,-4],[6,6,-1,7,-3,9],[4,4,1,7,8,-8],[10,-7,-1,-4,6,2],[7,1,-6,-9,-2,6],[-2,-5,-6,-5,-10,-7]],[[-7,8,1,3,1,1],[-4,-2,5,4,-2,10],[-1,-6,9,-2,4,-10],[-9,-4,6,-2,8,-1],[-3,-2,3,4,-5,4],[2,10,5,-1,-2,-7],[-2,1,2,-3,-6,-9],[-7,2,1,-5,2,-9],[7,1,8,-10,-6,2],[-4,10,-2,-8,10,-10]],[[-5,-6,-2,6,-8,1],[-5,-8,-3,-9,-5,-6],[8,5,-3,10,10,7],[-2,3,9,6,8,-7],[4,9,-6,8,8,-3],[8,1,9,4,-10,-1],[-3,-7,3,-2,-1,-6],[-6,1,10,7,7,-8],[5,-8,-8,-7,8,-4],[7,7,6,7,10,8]],[[-1,-7,-4,-5,-8,-5],[5,-1,-6,-5,-10,-3],[6,10,7,7,4,8],[-1,-6,-7,-7,-5,-7],[-3,5,5,-2,9,6],[-10,6,8,-2,7,10],[-1,6,7,-8,-6,-6],[3,1,-7,-10,1,-3],[-7,8,2,-8,6,10],[9,-4,3,5,5,7]],[[-4,-1,2,4,8,3],[-9,-5,-7,8,5,3],[3,3,4,2,-9,-3],[-8,10,10,-5,6,-5],[7,9,7,-9,1,-10],[1,5,4,1,-6,7],[-4,-4,-6,-3,4,4],[-5,-2,6,1,4,9],[3,-2,-1,-4,3,7],[-9,-2,-4,4,1,4]],[[-10,5,1,3,-4,-5],[-6,-1,-9,10,-1,-8],[-1,2,10,1,3,5],[-2,9,5,-2,-3,2],[3,9,-4,-9,-3,9],[9,3,-6,7,9,3],[-2,-5,-9,-8,-8,-8],[-5,-5,10,-2,-4,2],[-7,7,6,10,-8,-2],[1,-1,8,1,-8,4]],[[10,6,4,5,6,-7],[-3,-8,-9,10,-4,-1],[4,-8,8,-10,-1,-4],[6,-6,1,-5,-8,9],[-5,1,10,4,2,8],[-5,10,-7,-10,-4,-5],[2,3,-5,9,-9,-1],[-10,5,-6,-4,1,6],[-10,3,7,-1,10,-8],[4,4,8,6,-8,-2]]], dtype = "uint64")#candidate|76|(14, 10, 6)|const|uint64
bop_77 = relay.left_shift(var_75.astype('uint64'), relay.reshape(const_76.astype('uint64'), relay.shape_of(var_75))) # shape=(14, 10, 6)
func_68_call = mod.get_global_var('func_68')
func_73_call = mutated_mod.get_global_var('func_73')
const_81 = relay.const([0.327323,1.497125,-8.609397,4.896892,7.332795,-2.012407,0.324156,0.232129,9.545417,-4.661227,-7.332804,-0.995152,1.752136,8.316870,-0.794434,5.385063,9.559101,8.635010,7.119870,4.819578,-9.662560,-6.420407,-5.558669,1.313219,4.559554,4.574298,0.501888,-5.979164,0.776419,2.634281,-5.285474,1.319016,5.303918,8.948625,1.625793,-6.663471,-8.034926,2.803315,-5.496576,-9.146006,0.138436,8.594110,1.973098,7.843361,-6.352178,2.301196,6.039175,-7.018419,2.972136,5.599141,-9.380150,-0.738746,4.354784,-8.647774,-2.004215,6.742357,-7.375266,-8.917956,-4.525039,-3.250147,8.416606,3.083192,-4.200186,0.950823,5.310623,-2.836503,-8.035446,-1.461975,1.814805,-1.989472,9.402630,7.495106], dtype = "float64")#candidate|81|(72,)|const|float64
call_80 = relay.TupleGetItem(func_68_call(relay.reshape(const_81.astype('float64'), [3, 4, 6]), relay.reshape(const_81.astype('float64'), [3, 4, 6]), relay.reshape(const_81.astype('float64'), [3, 4, 6]), relay.reshape(const_81.astype('float64'), [3, 4, 6]), ), 1)
call_82 = relay.TupleGetItem(func_73_call(relay.reshape(const_81.astype('float64'), [3, 4, 6]), relay.reshape(const_81.astype('float64'), [3, 4, 6]), relay.reshape(const_81.astype('float64'), [3, 4, 6]), relay.reshape(const_81.astype('float64'), [3, 4, 6]), ), 1)
bop_83 = relay.minimum(call_80.astype('float32'), relay.reshape(const_81.astype('float32'), relay.shape_of(call_80))) # shape=(3, 4, 6)
bop_86 = relay.minimum(call_82.astype('float32'), relay.reshape(const_81.astype('float32'), relay.shape_of(call_82))) # shape=(3, 4, 6)
uop_87 = relay.asin(var_75.astype('float32')) # shape=(14, 10, 6)
bop_89 = relay.right_shift(const_81.astype('int64'), relay.reshape(call_80.astype('int64'), relay.shape_of(const_81))) # shape=(72,)
bop_92 = relay.right_shift(const_81.astype('int64'), relay.reshape(call_82.astype('int64'), relay.shape_of(const_81))) # shape=(72,)
uop_93 = relay.log10(uop_87.astype('float64')) # shape=(14, 10, 6)
bop_95 = relay.subtract(uop_93.astype('float64'), relay.reshape(bop_77.astype('float64'), relay.shape_of(uop_93))) # shape=(14, 10, 6)
bop_98 = relay.add(bop_95.astype('float64'), relay.reshape(var_75.astype('float64'), relay.shape_of(bop_95))) # shape=(14, 10, 6)
var_101 = relay.var("var_101", dtype = "float32", shape = (14, 10, 6))#candidate|101|(14, 10, 6)|var|float32
bop_102 = relay.divide(uop_87.astype('float32'), relay.reshape(var_101.astype('float32'), relay.shape_of(uop_87))) # shape=(14, 10, 6)
uop_105 = relay.acos(bop_95.astype('float64')) # shape=(14, 10, 6)
uop_107 = relay.cos(uop_105.astype('float32')) # shape=(14, 10, 6)
uop_109 = relay.log10(uop_107.astype('float32')) # shape=(14, 10, 6)
output = relay.Tuple([bop_83,bop_89,bop_98,bop_102,uop_109,])
output2 = relay.Tuple([bop_86,bop_92,bop_98,bop_102,uop_109,])
func_111 = relay.Function([var_75,var_101,], output)
mod['func_111'] = func_111
mod = relay.transform.InferType()(mod)
mutated_mod['func_111'] = func_111
mutated_mod = relay.transform.InferType()(mutated_mod)
func_111_call = mutated_mod.get_global_var('func_111')
var_113 = relay.var("var_113", dtype = "uint64", shape = (14, 10, 6))#candidate|113|(14, 10, 6)|var|uint64
var_114 = relay.var("var_114", dtype = "float32", shape = (14, 10, 6))#candidate|114|(14, 10, 6)|var|float32
call_112 = func_111_call(var_113,var_114,)
output = call_112
func_115 = relay.Function([var_113,var_114,], output)
mutated_mod['func_115'] = func_115
mutated_mod = relay.transform.InferType()(mutated_mod)
var_117 = relay.var("var_117", dtype = "float32", shape = (10, 14))#candidate|117|(10, 14)|var|float32
uop_118 = relay.log2(var_117.astype('float32')) # shape=(10, 14)
func_68_call = mod.get_global_var('func_68')
func_73_call = mutated_mod.get_global_var('func_73')
var_121 = relay.var("var_121", dtype = "float64", shape = (72,))#candidate|121|(72,)|var|float64
call_120 = relay.TupleGetItem(func_68_call(relay.reshape(var_121.astype('float64'), [3, 4, 6]), relay.reshape(var_121.astype('float64'), [3, 4, 6]), relay.reshape(var_121.astype('float64'), [3, 4, 6]), relay.reshape(var_121.astype('float64'), [3, 4, 6]), ), 1)
call_122 = relay.TupleGetItem(func_73_call(relay.reshape(var_121.astype('float64'), [3, 4, 6]), relay.reshape(var_121.astype('float64'), [3, 4, 6]), relay.reshape(var_121.astype('float64'), [3, 4, 6]), relay.reshape(var_121.astype('float64'), [3, 4, 6]), ), 1)
bop_123 = relay.multiply(uop_118.astype('uint32'), relay.reshape(var_117.astype('uint32'), relay.shape_of(uop_118))) # shape=(10, 14)
bop_126 = relay.power(uop_118.astype('float64'), relay.reshape(var_117.astype('float64'), relay.shape_of(uop_118))) # shape=(10, 14)
bop_129 = relay.add(bop_123.astype('float32'), relay.reshape(uop_118.astype('float32'), relay.shape_of(bop_123))) # shape=(10, 14)
var_132 = relay.var("var_132", dtype = "float32", shape = (3, 4, 6))#candidate|132|(3, 4, 6)|var|float32
bop_133 = relay.mod(call_120.astype('float32'), relay.reshape(var_132.astype('float32'), relay.shape_of(call_120))) # shape=(3, 4, 6)
bop_136 = relay.mod(call_122.astype('float32'), relay.reshape(var_132.astype('float32'), relay.shape_of(call_122))) # shape=(3, 4, 6)
bop_137 = relay.logical_xor(bop_126.astype('uint8'), relay.reshape(var_117.astype('uint8'), relay.shape_of(bop_126))) # shape=(10, 14)
bop_140 = relay.greater(call_120.astype('bool'), relay.reshape(bop_133.astype('bool'), relay.shape_of(call_120))) # shape=(3, 4, 6)
bop_143 = relay.greater(call_122.astype('bool'), relay.reshape(bop_136.astype('bool'), relay.shape_of(call_122))) # shape=(3, 4, 6)
uop_144 = relay.sigmoid(bop_126.astype('float32')) # shape=(10, 14)
bop_146 = relay.not_equal(uop_144.astype('bool'), relay.reshape(bop_123.astype('bool'), relay.shape_of(uop_144))) # shape=(10, 14)
output = relay.Tuple([var_121,bop_129,bop_137,bop_140,bop_146,])
output2 = relay.Tuple([var_121,bop_129,bop_137,bop_143,bop_146,])
func_149 = relay.Function([var_117,var_121,var_132,], output)
mod['func_149'] = func_149
mod = relay.transform.InferType()(mod)
var_150 = relay.var("var_150", dtype = "float32", shape = (10, 14))#candidate|150|(10, 14)|var|float32
var_151 = relay.var("var_151", dtype = "float64", shape = (72,))#candidate|151|(72,)|var|float64
var_152 = relay.var("var_152", dtype = "float32", shape = (3, 4, 6))#candidate|152|(3, 4, 6)|var|float32
output = func_149(var_150,var_151,var_152,)
func_153 = relay.Function([var_150,var_151,var_152,], output)
mutated_mod['func_153'] = func_153
mutated_mod = relay.transform.InferType()(mutated_mod)
var_155 = relay.var("var_155", dtype = "float32", shape = (4,))#candidate|155|(4,)|var|float32
uop_156 = relay.cos(var_155.astype('float32')) # shape=(4,)
uop_158 = relay.exp(uop_156.astype('float64')) # shape=(4,)
bop_160 = relay.logical_xor(uop_158.astype('int16'), relay.reshape(var_155.astype('int16'), relay.shape_of(uop_158))) # shape=(4,)
bop_163 = relay.less(uop_156.astype('bool'), relay.reshape(uop_158.astype('bool'), relay.shape_of(uop_156))) # shape=(4,)
const_166 = relay.const([1.835809,9.662063,-9.780495,0.451037], dtype = "float32")#candidate|166|(4,)|const|float32
bop_167 = relay.maximum(uop_156.astype('uint8'), relay.reshape(const_166.astype('uint8'), relay.shape_of(uop_156))) # shape=(4,)
bop_170 = relay.add(uop_158.astype('uint32'), relay.reshape(var_155.astype('uint32'), relay.shape_of(uop_158))) # shape=(4,)
uop_173 = relay.log10(bop_163.astype('float64')) # shape=(4,)
uop_175 = relay.acos(uop_173.astype('float64')) # shape=(4,)
bop_177 = relay.not_equal(const_166.astype('bool'), relay.reshape(bop_163.astype('bool'), relay.shape_of(const_166))) # shape=(4,)
bop_180 = relay.power(uop_175.astype('float64'), relay.reshape(bop_163.astype('float64'), relay.shape_of(uop_175))) # shape=(4,)
bop_183 = relay.bitwise_xor(bop_180.astype('uint16'), relay.reshape(bop_163.astype('uint16'), relay.shape_of(bop_180))) # shape=(4,)
uop_186 = relay.acosh(bop_180.astype('float32')) # shape=(4,)
uop_188 = relay.sqrt(bop_180.astype('float32')) # shape=(4,)
bop_190 = relay.greater(uop_188.astype('bool'), relay.reshape(bop_163.astype('bool'), relay.shape_of(uop_188))) # shape=(4,)
uop_193 = relay.tan(uop_186.astype('float32')) # shape=(4,)
bop_195 = relay.left_shift(bop_183.astype('uint32'), relay.reshape(bop_160.astype('uint32'), relay.shape_of(bop_183))) # shape=(4,)
bop_198 = relay.floor_divide(uop_193.astype('float64'), relay.reshape(bop_160.astype('float64'), relay.shape_of(uop_193))) # shape=(4,)
var_201 = relay.var("var_201", dtype = "float64", shape = (4,))#candidate|201|(4,)|var|float64
bop_202 = relay.bitwise_and(bop_198.astype('int8'), relay.reshape(var_201.astype('int8'), relay.shape_of(bop_198))) # shape=(4,)
bop_205 = relay.logical_xor(bop_190.astype('int64'), relay.reshape(bop_180.astype('int64'), relay.shape_of(bop_190))) # shape=(4,)
uop_208 = relay.log(uop_175.astype('float32')) # shape=(4,)
bop_210 = relay.multiply(bop_198.astype('int32'), relay.reshape(bop_160.astype('int32'), relay.shape_of(bop_198))) # shape=(4,)
var_213 = relay.var("var_213", dtype = "int32", shape = (4,))#candidate|213|(4,)|var|int32
bop_214 = relay.mod(bop_210.astype('float64'), relay.reshape(var_213.astype('float64'), relay.shape_of(bop_210))) # shape=(4,)
var_217 = relay.var("var_217", dtype = "int8", shape = (4,))#candidate|217|(4,)|var|int8
bop_218 = relay.floor_mod(bop_202.astype('float32'), relay.reshape(var_217.astype('float32'), relay.shape_of(bop_202))) # shape=(4,)
bop_221 = relay.less(bop_170.astype('bool'), relay.reshape(uop_175.astype('bool'), relay.shape_of(bop_170))) # shape=(4,)
const_224 = relay.const([-0.867876,0.812274,-5.439449,3.792103], dtype = "float32")#candidate|224|(4,)|const|float32
bop_225 = relay.greater_equal(uop_188.astype('bool'), relay.reshape(const_224.astype('bool'), relay.shape_of(uop_188))) # shape=(4,)
uop_228 = relay.exp(uop_193.astype('float32')) # shape=(4,)
bop_230 = relay.mod(bop_214.astype('float32'), relay.reshape(bop_183.astype('float32'), relay.shape_of(bop_214))) # shape=(4,)
bop_233 = relay.greater_equal(bop_214.astype('bool'), relay.reshape(bop_190.astype('bool'), relay.shape_of(bop_214))) # shape=(4,)
uop_236 = relay.tan(uop_228.astype('float32')) # shape=(4,)
bop_238 = relay.logical_or(uop_228.astype('bool'), relay.reshape(bop_170.astype('bool'), relay.shape_of(uop_228))) # shape=(4,)
uop_241 = relay.atanh(bop_238.astype('float32')) # shape=(4,)
bop_243 = relay.minimum(uop_241.astype('float64'), relay.reshape(var_213.astype('float64'), relay.shape_of(uop_241))) # shape=(4,)
var_246 = relay.var("var_246", dtype = "bool", shape = (4,))#candidate|246|(4,)|var|bool
bop_247 = relay.floor_mod(bop_238.astype('float64'), relay.reshape(var_246.astype('float64'), relay.shape_of(bop_238))) # shape=(4,)
var_250 = relay.var("var_250", dtype = "float64", shape = (4,))#candidate|250|(4,)|var|float64
bop_251 = relay.maximum(bop_243.astype('uint8'), relay.reshape(var_250.astype('uint8'), relay.shape_of(bop_243))) # shape=(4,)
output = relay.Tuple([bop_167,bop_177,bop_195,bop_205,uop_208,bop_218,bop_221,bop_225,bop_230,bop_233,uop_236,bop_247,bop_251,])
output2 = relay.Tuple([bop_167,bop_177,bop_195,bop_205,uop_208,bop_218,bop_221,bop_225,bop_230,bop_233,uop_236,bop_247,bop_251,])
func_254 = relay.Function([var_155,var_201,var_213,var_217,var_246,var_250,], output)
mod['func_254'] = func_254
mod = relay.transform.InferType()(mod)
var_255 = relay.var("var_255", dtype = "float32", shape = (4,))#candidate|255|(4,)|var|float32
var_256 = relay.var("var_256", dtype = "float64", shape = (4,))#candidate|256|(4,)|var|float64
var_257 = relay.var("var_257", dtype = "int32", shape = (4,))#candidate|257|(4,)|var|int32
var_258 = relay.var("var_258", dtype = "int8", shape = (4,))#candidate|258|(4,)|var|int8
var_259 = relay.var("var_259", dtype = "bool", shape = (4,))#candidate|259|(4,)|var|bool
var_260 = relay.var("var_260", dtype = "float64", shape = (4,))#candidate|260|(4,)|var|float64
output = func_254(var_255,var_256,var_257,var_258,var_259,var_260,)
func_261 = relay.Function([var_255,var_256,var_257,var_258,var_259,var_260,], output)
mutated_mod['func_261'] = func_261
mutated_mod = relay.transform.InferType()(mutated_mod)
var_263 = relay.var("var_263", dtype = "bool", shape = (1, 1, 12))#candidate|263|(1, 1, 12)|var|bool
var_264 = relay.var("var_264", dtype = "bool", shape = (11, 13, 12))#candidate|264|(11, 13, 12)|var|bool
bop_265 = relay.logical_or(var_263.astype('bool'), var_264.astype('bool')) # shape=(11, 13, 12)
uop_268 = relay.log(var_263.astype('float64')) # shape=(1, 1, 12)
var_270 = relay.var("var_270", dtype = "float64", shape = (2, 16, 12))#candidate|270|(2, 16, 12)|var|float64
bop_271 = relay.floor_mod(uop_268.astype('float32'), var_270.astype('float32')) # shape=(2, 16, 12)
uop_274 = relay.log2(uop_268.astype('float64')) # shape=(1, 1, 12)
uop_276 = relay.sqrt(uop_274.astype('float32')) # shape=(1, 1, 12)
uop_278 = relay.rsqrt(uop_274.astype('float64')) # shape=(1, 1, 12)
var_280 = relay.var("var_280", dtype = "float64", shape = (3, 3, 12))#candidate|280|(3, 3, 12)|var|float64
bop_281 = relay.floor_mod(uop_274.astype('float32'), var_280.astype('float32')) # shape=(3, 3, 12)
bop_284 = relay.equal(uop_276.astype('bool'), var_270.astype('bool')) # shape=(2, 16, 12)
bop_287 = relay.left_shift(bop_281.astype('int16'), uop_268.astype('int16')) # shape=(3, 3, 12)
var_290 = relay.var("var_290", dtype = "float64", shape = (15, 5, 12))#candidate|290|(15, 5, 12)|var|float64
bop_291 = relay.logical_and(uop_278.astype('bool'), var_290.astype('bool')) # shape=(15, 5, 12)
uop_294 = relay.cosh(uop_268.astype('float32')) # shape=(1, 1, 12)
uop_296 = relay.sigmoid(uop_276.astype('float32')) # shape=(1, 1, 12)
var_298 = relay.var("var_298", dtype = "float32", shape = (14, 4, 12))#candidate|298|(14, 4, 12)|var|float32
bop_299 = relay.power(uop_296.astype('float32'), var_298.astype('float32')) # shape=(14, 4, 12)
bop_302 = relay.add(bop_299.astype('int8'), uop_278.astype('int8')) # shape=(14, 4, 12)
func_68_call = mod.get_global_var('func_68')
func_73_call = mutated_mod.get_global_var('func_73')
var_306 = relay.var("var_306", dtype = "float64", shape = (72, 1))#candidate|306|(72, 1)|var|float64
call_305 = relay.TupleGetItem(func_68_call(relay.reshape(var_306.astype('float64'), [3, 4, 6]), relay.reshape(var_306.astype('float64'), [3, 4, 6]), relay.reshape(var_306.astype('float64'), [3, 4, 6]), relay.reshape(var_306.astype('float64'), [3, 4, 6]), ), 0)
call_307 = relay.TupleGetItem(func_73_call(relay.reshape(var_306.astype('float64'), [3, 4, 6]), relay.reshape(var_306.astype('float64'), [3, 4, 6]), relay.reshape(var_306.astype('float64'), [3, 4, 6]), relay.reshape(var_306.astype('float64'), [3, 4, 6]), ), 0)
bop_308 = relay.add(bop_299.astype('int32'), relay.reshape(var_298.astype('int32'), relay.shape_of(bop_299))) # shape=(14, 4, 12)
output = relay.Tuple([bop_265,bop_271,bop_284,bop_287,bop_291,uop_294,bop_302,call_305,var_306,bop_308,])
output2 = relay.Tuple([bop_265,bop_271,bop_284,bop_287,bop_291,uop_294,bop_302,call_307,var_306,bop_308,])
func_311 = relay.Function([var_263,var_264,var_270,var_280,var_290,var_298,var_306,], output)
mod['func_311'] = func_311
mod = relay.transform.InferType()(mod)
mutated_mod['func_311'] = func_311
mutated_mod = relay.transform.InferType()(mutated_mod)
func_311_call = mutated_mod.get_global_var('func_311')
var_313 = relay.var("var_313", dtype = "bool", shape = (1, 1, 12))#candidate|313|(1, 1, 12)|var|bool
var_314 = relay.var("var_314", dtype = "bool", shape = (11, 13, 12))#candidate|314|(11, 13, 12)|var|bool
var_315 = relay.var("var_315", dtype = "float64", shape = (2, 16, 12))#candidate|315|(2, 16, 12)|var|float64
var_316 = relay.var("var_316", dtype = "float64", shape = (3, 3, 12))#candidate|316|(3, 3, 12)|var|float64
var_317 = relay.var("var_317", dtype = "float64", shape = (15, 5, 12))#candidate|317|(15, 5, 12)|var|float64
var_318 = relay.var("var_318", dtype = "float32", shape = (14, 4, 12))#candidate|318|(14, 4, 12)|var|float32
var_319 = relay.var("var_319", dtype = "float64", shape = (72, 1))#candidate|319|(72, 1)|var|float64
call_312 = func_311_call(var_313,var_314,var_315,var_316,var_317,var_318,var_319,)
output = call_312
func_320 = relay.Function([var_313,var_314,var_315,var_316,var_317,var_318,var_319,], output)
mutated_mod['func_320'] = func_320
mutated_mod = relay.transform.InferType()(mutated_mod)
var_322 = relay.var("var_322", dtype = "float32", shape = (3,))#candidate|322|(3,)|var|float32
const_323 = relay.const([7.092687,5.204391,-8.026120], dtype = "float32")#candidate|323|(3,)|const|float32
bop_324 = relay.mod(var_322.astype('float32'), relay.reshape(const_323.astype('float32'), relay.shape_of(var_322))) # shape=(3,)
var_327 = relay.var("var_327", dtype = "float32", shape = (3,))#candidate|327|(3,)|var|float32
bop_328 = relay.left_shift(bop_324.astype('int32'), relay.reshape(var_327.astype('int32'), relay.shape_of(bop_324))) # shape=(3,)
output = relay.Tuple([bop_328,])
output2 = relay.Tuple([bop_328,])
func_331 = relay.Function([var_322,var_327,], output)
mod['func_331'] = func_331
mod = relay.transform.InferType()(mod)
mutated_mod['func_331'] = func_331
mutated_mod = relay.transform.InferType()(mutated_mod)
func_331_call = mutated_mod.get_global_var('func_331')
var_333 = relay.var("var_333", dtype = "float32", shape = (3,))#candidate|333|(3,)|var|float32
var_334 = relay.var("var_334", dtype = "float32", shape = (3,))#candidate|334|(3,)|var|float32
call_332 = func_331_call(var_333,var_334,)
output = call_332
func_335 = relay.Function([var_333,var_334,], output)
mutated_mod['func_335'] = func_335
mutated_mod = relay.transform.InferType()(mutated_mod)
var_337 = relay.var("var_337", dtype = "float32", shape = (5, 11))#candidate|337|(5, 11)|var|float32
uop_338 = relay.sinh(var_337.astype('float32')) # shape=(5, 11)
uop_340 = relay.sinh(var_337.astype('float32')) # shape=(5, 11)
uop_342 = relay.atanh(var_337.astype('float32')) # shape=(5, 11)
uop_344 = relay.sinh(uop_338.astype('float32')) # shape=(5, 11)
output = relay.Tuple([uop_340,uop_342,uop_344,])
output2 = relay.Tuple([uop_340,uop_342,uop_344,])
F = relay.Function([var_337,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_337,], output2)
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
input_337= np.array([[-8.358685,-6.979555,-9.104699,8.442044,-3.800543,0.618064,-1.275162,-3.657982,-8.080904,-8.335489,1.458808],[-1.579970,-5.060130,-3.398157,-1.707394,-8.521554,-2.652150,6.623760,6.983612,7.999266,-8.288459,-3.274396],[5.557470,6.852689,-2.841550,-7.278048,-7.192539,3.468763,8.597053,7.012292,-2.704075,-3.782645,-1.978191],[9.163296,-7.898595,2.224697,7.393660,-3.441991,-6.240642,-9.444148,0.345349,-3.079698,-6.335520,-1.850412],[0.846668,1.727801,7.405528,-0.520959,-6.635295,9.901745,0.730014,4.620004,9.678349,4.175850,-4.957090]], dtype='float32')
module1.set_input('var_337', input_337)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_337, )
res3 = intrp3.evaluate()(input_337, )
res4 = intrp4.evaluate()(input_337, )
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
module5.set_input('var_337', input_337)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_337, )
res7 = intrp7.evaluate()(input_337, )
res8 = intrp8.evaluate()(input_337, )
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
module9.set_input('var_337', input_337)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_337, )
res11 = intrp11.evaluate()(input_337, )
res12 = intrp12.evaluate()(input_337, )
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
module13.set_input('var_337', input_337)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_337, )
res15 = intrp15.evaluate()(input_337, )
res16 = intrp16.evaluate()(input_337, )
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
module17.set_input('var_337', input_337)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_337, )
res19 = intrp19.evaluate()(input_337, )
res20 = intrp20.evaluate()(input_337, )
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
module21.set_input('var_337', input_337)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_337, )
res23 = intrp23.evaluate()(input_337, )
res24 = intrp24.evaluate()(input_337, )
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