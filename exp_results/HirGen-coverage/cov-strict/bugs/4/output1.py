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
var_0 = relay.var("var_0", dtype = "float64", shape = (7, 1))#candidate|0|(7, 1)|var|float64
uop_1 = relay.sinh(var_0.astype('float64')) # shape=(7, 1)
bop_3 = relay.multiply(uop_1.astype('int16'), relay.reshape(var_0.astype('int16'), relay.shape_of(uop_1))) # shape=(7, 1)
bop_7 = relay.greater(uop_1.astype('bool'), relay.reshape(var_0.astype('bool'), relay.shape_of(uop_1))) # shape=(7, 1)
bop_10 = relay.logical_or(uop_1.astype('bool'), relay.reshape(bop_7.astype('bool'), relay.shape_of(uop_1))) # shape=(7, 1)
const_13 = relay.const([[-1.238746,-4.840959,-7.989714,-4.537047,-3.170296],[5.979677,9.018131,5.797227,-4.553127,2.818323],[1.674042,-8.410423,-1.795693,2.858729,6.296178],[8.098633,-9.638840,-2.956146,-0.244831,6.147703],[3.519628,6.870878,1.441126,8.729488,8.882809],[-4.832007,-0.718241,4.436689,-3.415935,-1.892099],[7.093747,5.217453,3.349467,-3.556552,-0.384000]], dtype = "float64")#candidate|13|(7, 5)|const|float64
bop_14 = relay.less(uop_1.astype('bool'), const_13.astype('bool')) # shape=(7, 5)
const_17 = relay.const([[-4,1,9,6,7,9,9,7,-8],[3,-9,-2,-2,8,7,-7,-10,-6],[-2,-4,2,-7,7,-8,-4,-2,-5],[-4,1,2,9,-9,-10,7,7,-10],[-7,9,-2,-6,-9,-1,-5,3,9],[4,-5,-9,5,-6,7,1,-3,-5],[9,-3,-4,-5,1,6,1,8,-2]], dtype = "int16")#candidate|17|(7, 9)|const|int16
bop_18 = relay.subtract(bop_3.astype('uint32'), const_17.astype('uint32')) # shape=(7, 9)
bop_21 = relay.logical_xor(bop_7.astype('int16'), const_13.astype('int16')) # shape=(7, 5)
var_28 = relay.var("var_28", dtype = "uint32", shape = (7, 9))#candidate|28|(7, 9)|var|uint32
bop_29 = relay.equal(bop_18.astype('bool'), relay.reshape(var_28.astype('bool'), relay.shape_of(bop_18))) # shape=(7, 9)
var_35 = relay.var("var_35", dtype = "bool", shape = (7, 3))#candidate|35|(7, 3)|var|bool
bop_36 = relay.divide(bop_7.astype('float32'), var_35.astype('float32')) # shape=(7, 3)
uop_40 = relay.acosh(var_0.astype('float32')) # shape=(7, 1)
uop_42 = relay.atan(bop_7.astype('float32')) # shape=(7, 1)
uop_44 = relay.log2(uop_1.astype('float32')) # shape=(7, 1)
output = relay.Tuple([bop_10,bop_14,bop_21,bop_29,bop_36,uop_40,uop_42,uop_44,])
output2 = relay.Tuple([bop_10,bop_14,bop_21,bop_29,bop_36,uop_40,uop_42,uop_44,])
func_46 = relay.Function([var_0,var_28,var_35,], output)
mod['func_46'] = func_46
mod = relay.transform.InferType()(mod)
mutated_mod['func_46'] = func_46
mutated_mod = relay.transform.InferType()(mutated_mod)
func_46_call = mutated_mod.get_global_var('func_46')
var_48 = relay.var("var_48", dtype = "float64", shape = (7, 1))#candidate|48|(7, 1)|var|float64
var_49 = relay.var("var_49", dtype = "uint32", shape = (7, 9))#candidate|49|(7, 9)|var|uint32
var_50 = relay.var("var_50", dtype = "bool", shape = (7, 3))#candidate|50|(7, 3)|var|bool
call_47 = func_46_call(var_48,var_49,var_50,)
output = call_47
func_51 = relay.Function([var_48,var_49,var_50,], output)
mutated_mod['func_51'] = func_51
mutated_mod = relay.transform.InferType()(mutated_mod)
var_56 = relay.var("var_56", dtype = "float64", shape = (2,))#candidate|56|(2,)|var|float64
uop_57 = relay.cosh(var_56.astype('float64')) # shape=(2,)
bop_61 = relay.divide(uop_57.astype('float32'), relay.reshape(var_56.astype('float32'), relay.shape_of(uop_57))) # shape=(2,)
func_46_call = mod.get_global_var('func_46')
func_51_call = mutated_mod.get_global_var('func_51')
const_65 = relay.const([-9.543042,1.705428,-7.757630,4.807142,-2.998378,-6.753330,-7.804552], dtype = "float64")#candidate|65|(7,)|const|float64
var_66 = relay.var("var_66", dtype = "uint32", shape = (63,))#candidate|66|(63,)|var|uint32
const_67 = relay.const([False,False,True,False,False,False,False,False,True,True,True,False,True,False,False,True,False,False,True,True,True], dtype = "bool")#candidate|67|(21,)|const|bool
call_64 = relay.TupleGetItem(func_46_call(relay.reshape(const_65.astype('float64'), [7, 1]), relay.reshape(var_66.astype('uint32'), [7, 9]), relay.reshape(const_67.astype('bool'), [7, 3]), ), 6)
call_68 = relay.TupleGetItem(func_51_call(relay.reshape(const_65.astype('float64'), [7, 1]), relay.reshape(var_66.astype('uint32'), [7, 9]), relay.reshape(const_67.astype('bool'), [7, 3]), ), 6)
uop_73 = relay.sinh(uop_57.astype('float32')) # shape=(2,)
uop_79 = relay.cos(uop_73.astype('float64')) # shape=(2,)
output = relay.Tuple([bop_61,call_64,const_65,var_66,const_67,uop_79,])
output2 = relay.Tuple([bop_61,call_68,const_65,var_66,const_67,uop_79,])
func_81 = relay.Function([var_56,var_66,], output)
mod['func_81'] = func_81
mod = relay.transform.InferType()(mod)
var_82 = relay.var("var_82", dtype = "float64", shape = (2,))#candidate|82|(2,)|var|float64
var_83 = relay.var("var_83", dtype = "uint32", shape = (63,))#candidate|83|(63,)|var|uint32
output = func_81(var_82,var_83,)
func_84 = relay.Function([var_82,var_83,], output)
mutated_mod['func_84'] = func_84
mutated_mod = relay.transform.InferType()(mutated_mod)
var_96 = relay.var("var_96", dtype = "uint16", shape = (2, 11))#candidate|96|(2, 11)|var|uint16
var_97 = relay.var("var_97", dtype = "uint16", shape = (2, 11))#candidate|97|(2, 11)|var|uint16
bop_98 = relay.bitwise_or(var_96.astype('uint16'), relay.reshape(var_97.astype('uint16'), relay.shape_of(var_96))) # shape=(2, 11)
var_101 = relay.var("var_101", dtype = "uint16", shape = (2, 11))#candidate|101|(2, 11)|var|uint16
bop_102 = relay.power(bop_98.astype('float64'), relay.reshape(var_101.astype('float64'), relay.shape_of(bop_98))) # shape=(2, 11)
uop_105 = relay.rsqrt(var_97.astype('float32')) # shape=(2, 11)
bop_108 = relay.mod(uop_105.astype('float32'), relay.reshape(bop_102.astype('float32'), relay.shape_of(uop_105))) # shape=(2, 11)
bop_113 = relay.right_shift(uop_105.astype('int32'), relay.reshape(bop_108.astype('int32'), relay.shape_of(uop_105))) # shape=(2, 11)
output = relay.Tuple([bop_113,])
output2 = relay.Tuple([bop_113,])
func_117 = relay.Function([var_96,var_97,var_101,], output)
mod['func_117'] = func_117
mod = relay.transform.InferType()(mod)
mutated_mod['func_117'] = func_117
mutated_mod = relay.transform.InferType()(mutated_mod)
func_117_call = mutated_mod.get_global_var('func_117')
var_119 = relay.var("var_119", dtype = "uint16", shape = (2, 11))#candidate|119|(2, 11)|var|uint16
var_120 = relay.var("var_120", dtype = "uint16", shape = (2, 11))#candidate|120|(2, 11)|var|uint16
var_121 = relay.var("var_121", dtype = "uint16", shape = (2, 11))#candidate|121|(2, 11)|var|uint16
call_118 = func_117_call(var_119,var_120,var_121,)
output = call_118
func_122 = relay.Function([var_119,var_120,var_121,], output)
mutated_mod['func_122'] = func_122
mutated_mod = relay.transform.InferType()(mutated_mod)
const_136 = relay.const([[[-4,-1,3,-6,5,-3,1,9,-2,6,-6,2,3,4,9,-7],[-10,1,-6,-4,-8,5,10,1,-5,-7,-8,-5,-5,6,9,-6],[-4,4,-3,-10,5,2,-1,9,-10,7,5,8,8,-3,-9,-8],[7,2,-10,1,-10,-3,6,7,8,3,-3,-5,-4,10,7,-6]],[[-9,4,-3,-5,10,3,9,1,3,3,6,9,-6,5,8,-8],[2,-4,-9,-7,8,4,-2,-5,2,1,5,7,-2,5,-6,4],[-6,-4,-9,5,-10,5,7,-7,9,6,1,-5,8,10,2,-5],[-6,7,-7,-3,5,-1,1,1,7,3,-3,-10,8,-8,-5,1]],[[9,7,2,4,-9,10,-2,-9,10,-6,10,-7,-5,-9,7,2],[-9,5,10,-1,-6,-10,-3,7,10,-8,9,-9,10,7,-9,5],[-9,-3,10,9,-6,4,9,-5,-9,9,10,-4,-7,2,-9,8],[-7,-8,-5,-5,-7,2,-4,-5,2,1,-4,-6,6,-9,-5,5]],[[-7,1,2,-5,-6,-3,4,3,1,-1,7,-8,-5,9,5,2],[-7,-4,-10,7,5,-8,7,-8,3,5,-7,4,3,2,7,10],[6,4,4,-3,1,-2,1,9,9,6,1,10,3,-2,3,-8],[-9,-6,-4,-3,-10,-5,5,7,-10,-2,2,4,3,8,1,6]],[[6,2,2,-7,2,-5,6,3,-4,-8,8,-2,2,-6,3,3],[-5,-1,2,-3,-10,3,2,-3,-10,1,-6,8,2,-1,5,4],[8,-1,1,-5,4,-6,-4,1,1,2,6,5,4,-2,-7,1],[2,-6,-7,4,8,6,-1,6,10,9,-9,-5,-10,2,-3,-5]]], dtype = "uint64")#candidate|136|(5, 4, 16)|const|uint64
var_137 = relay.var("var_137", dtype = "uint64", shape = (5, 4, 16))#candidate|137|(5, 4, 16)|var|uint64
bop_138 = relay.greater(const_136.astype('bool'), relay.reshape(var_137.astype('bool'), relay.shape_of(const_136))) # shape=(5, 4, 16)
bop_141 = relay.left_shift(const_136.astype('uint8'), relay.reshape(bop_138.astype('uint8'), relay.shape_of(const_136))) # shape=(5, 4, 16)
uop_148 = relay.cosh(const_136.astype('float64')) # shape=(5, 4, 16)
uop_150 = relay.sqrt(uop_148.astype('float32')) # shape=(5, 4, 16)
uop_153 = relay.sigmoid(uop_150.astype('float32')) # shape=(5, 4, 16)
const_155 = relay.const([[[2.222332,-6.539053,-9.926411,-8.433662,7.128973,8.934652,2.827665,4.491978,4.354870,0.048584,9.865352,6.776348,-2.151666,-3.919653,8.074825,-5.375081],[3.997779,-9.854652,7.134037,0.147995,7.454341,1.698795,-7.772118,8.481755,-1.835043,-0.352124,9.957120,-8.299024,9.618780,4.220636,-5.967297,0.388867],[-8.439483,-7.678105,-5.264711,-4.153239,-3.466464,-5.145521,9.637568,5.562808,2.945112,7.253126,3.065575,-2.525553,-5.791556,6.802103,7.373700,-3.093977],[-9.881992,-1.855845,-1.453351,-1.125289,-8.190561,1.894492,-7.742164,2.090305,-6.456154,-2.726913,6.772858,-6.635021,3.372124,-9.411228,-8.121793,-6.818403]],[[1.567508,-3.175150,-4.910522,6.392506,8.382115,-4.846257,8.662288,8.867082,-4.918241,0.362144,2.141852,5.609138,5.200745,4.410981,-9.878803,3.913643],[-2.628765,6.747766,-4.734996,6.640360,4.690150,-1.002200,-5.471584,-7.184553,5.167752,-7.842708,-8.396104,1.062844,0.849735,8.482953,9.550448,-2.481874],[-5.427681,-1.505897,-8.282059,2.332395,5.012551,-4.531746,0.531503,-1.679541,-1.201338,4.472863,-2.565734,-7.039870,-2.572297,8.509277,-2.856922,3.462287],[7.422792,-5.184312,-5.170282,1.371960,7.766519,9.787095,-3.997848,5.673004,-3.024300,5.299504,-3.153352,-8.643188,-4.724981,-3.674002,9.368052,1.358536]],[[0.870722,-1.529031,1.025661,1.971258,0.565556,-8.883150,-0.004800,-6.866222,-3.525814,3.813079,-0.283508,5.970413,3.041073,4.946005,-4.993749,2.097967],[-8.218541,5.849199,0.432721,-9.135632,7.770788,-3.590023,8.809151,-4.999113,0.599568,-3.071199,1.971897,-0.009876,6.150592,-4.784927,-7.594928,5.952049],[4.224137,4.031527,9.691144,-1.081784,-4.739680,-0.777259,9.554715,5.771491,-3.738832,-9.429428,-5.081572,-2.765317,7.314414,8.212801,-0.578225,-0.927022],[4.659744,-6.440213,-7.753561,0.106036,6.947638,-2.358516,-7.254566,9.490855,0.211430,-6.597131,-4.123536,2.905183,2.703484,-6.229245,1.671538,-0.783380]],[[-2.432496,1.737568,8.876757,0.186893,-1.659552,0.288708,4.775452,8.393093,2.162900,-4.611176,-8.997857,7.018711,-3.694879,7.371147,-0.484465,-0.303203],[-1.296633,1.528760,-3.989398,-6.426434,-5.923174,2.772724,1.089272,-6.555382,7.086577,-3.226056,-7.291740,6.538523,0.030239,-3.465430,-6.989357,-1.330126],[-5.394201,-9.026462,-9.558434,3.043765,8.925345,9.360294,4.386900,0.157196,-7.658584,-8.145433,6.936074,-2.698238,-6.450790,-3.750859,0.770142,2.853447],[0.269889,1.579546,1.443590,-4.161727,6.477548,2.570268,-9.420757,9.573134,-3.902534,8.926817,-5.140126,6.627076,0.808250,0.437844,-2.464058,-9.270877]],[[0.732945,-9.163489,6.773934,8.578357,-5.070705,4.079858,9.809429,6.487292,6.902987,3.645085,-8.651468,-0.346356,-1.347214,9.938433,6.393823,-0.112279],[6.997722,8.433293,-6.336573,7.635729,3.045953,2.484795,-4.234492,-3.421416,1.607388,7.792115,-4.327282,1.510420,4.023451,-9.209026,0.164308,8.422132],[4.742625,9.118515,7.600665,-3.083073,4.509192,3.078739,-9.150785,5.446301,6.449516,-4.572959,9.890451,8.969524,-6.817096,2.852798,-4.799873,-1.993007],[4.693248,3.036653,-6.058549,7.239969,0.143762,8.062009,-5.393136,-9.574995,5.203723,-5.634852,-8.154998,-5.028764,1.610275,-8.705346,-0.430927,6.587008]]], dtype = "float64")#candidate|155|(5, 4, 16)|const|float64
bop_156 = relay.equal(uop_148.astype('bool'), relay.reshape(const_155.astype('bool'), relay.shape_of(uop_148))) # shape=(5, 4, 16)
uop_159 = relay.cosh(uop_150.astype('float64')) # shape=(5, 4, 16)
var_161 = relay.var("var_161", dtype = "bool", shape = (5, 4, 16))#candidate|161|(5, 4, 16)|var|bool
bop_162 = relay.multiply(bop_156.astype('uint32'), relay.reshape(var_161.astype('uint32'), relay.shape_of(bop_156))) # shape=(5, 4, 16)
bop_166 = relay.greater(uop_159.astype('bool'), relay.reshape(uop_153.astype('bool'), relay.shape_of(uop_159))) # shape=(5, 4, 16)
bop_169 = relay.add(uop_153.astype('int64'), relay.reshape(const_136.astype('int64'), relay.shape_of(uop_153))) # shape=(5, 4, 16)
bop_174 = relay.logical_and(uop_159.astype('bool'), relay.reshape(bop_162.astype('bool'), relay.shape_of(uop_159))) # shape=(5, 4, 16)
uop_177 = relay.asinh(bop_166.astype('float64')) # shape=(5, 4, 16)
bop_179 = relay.left_shift(uop_177.astype('int64'), relay.reshape(uop_159.astype('int64'), relay.shape_of(uop_177))) # shape=(5, 4, 16)
uop_182 = relay.log10(bop_179.astype('float32')) # shape=(5, 4, 16)
bop_184 = relay.power(uop_148.astype('float32'), relay.reshape(uop_150.astype('float32'), relay.shape_of(uop_148))) # shape=(5, 4, 16)
var_188 = relay.var("var_188", dtype = "int64", shape = (5, 4, 16))#candidate|188|(5, 4, 16)|var|int64
bop_189 = relay.divide(bop_179.astype('float64'), relay.reshape(var_188.astype('float64'), relay.shape_of(bop_179))) # shape=(5, 4, 16)
bop_196 = relay.logical_or(uop_159.astype('bool'), relay.reshape(bop_166.astype('bool'), relay.shape_of(uop_159))) # shape=(5, 4, 16)
uop_199 = relay.atan(uop_182.astype('float32')) # shape=(5, 4, 16)
uop_202 = relay.sigmoid(uop_199.astype('float32')) # shape=(5, 4, 16)
uop_204 = relay.log2(uop_202.astype('float32')) # shape=(5, 4, 16)
uop_206 = relay.asin(bop_179.astype('float64')) # shape=(5, 4, 16)
func_117_call = mod.get_global_var('func_117')
func_122_call = mutated_mod.get_global_var('func_122')
const_209 = relay.const([[-2,-4],[-9,-10],[5,-5],[2,6],[10,-5],[3,-6],[5,7],[-2,1],[10,-2],[8,7],[-6,2]], dtype = "uint16")#candidate|209|(11, 2)|const|uint16
call_208 = relay.TupleGetItem(func_117_call(relay.reshape(const_209.astype('uint16'), [2, 11]), relay.reshape(const_209.astype('uint16'), [2, 11]), relay.reshape(const_209.astype('uint16'), [2, 11]), ), 0)
call_210 = relay.TupleGetItem(func_122_call(relay.reshape(const_209.astype('uint16'), [2, 11]), relay.reshape(const_209.astype('uint16'), [2, 11]), relay.reshape(const_209.astype('uint16'), [2, 11]), ), 0)
uop_211 = relay.sin(uop_204.astype('float32')) # shape=(5, 4, 16)
output = relay.Tuple([bop_141,bop_169,bop_174,bop_184,bop_189,bop_196,uop_206,call_208,const_209,uop_211,])
output2 = relay.Tuple([bop_141,bop_169,bop_174,bop_184,bop_189,bop_196,uop_206,call_210,const_209,uop_211,])
func_215 = relay.Function([var_137,var_161,var_188,], output)
mod['func_215'] = func_215
mod = relay.transform.InferType()(mod)
var_216 = relay.var("var_216", dtype = "uint64", shape = (5, 4, 16))#candidate|216|(5, 4, 16)|var|uint64
var_217 = relay.var("var_217", dtype = "bool", shape = (5, 4, 16))#candidate|217|(5, 4, 16)|var|bool
var_218 = relay.var("var_218", dtype = "int64", shape = (5, 4, 16))#candidate|218|(5, 4, 16)|var|int64
output = func_215(var_216,var_217,var_218,)
func_219 = relay.Function([var_216,var_217,var_218,], output)
mutated_mod['func_219'] = func_219
mutated_mod = relay.transform.InferType()(mutated_mod)
var_226 = relay.var("var_226", dtype = "uint64", shape = (12,))#candidate|226|(12,)|var|uint64
const_227 = relay.const([-1,-6,2,3,6,6,-1,10,-6,-10,6,-10], dtype = "uint64")#candidate|227|(12,)|const|uint64
bop_228 = relay.add(var_226.astype('uint64'), relay.reshape(const_227.astype('uint64'), relay.shape_of(var_226))) # shape=(12,)
output = bop_228
output2 = bop_228
func_233 = relay.Function([var_226,], output)
mod['func_233'] = func_233
mod = relay.transform.InferType()(mod)
mutated_mod['func_233'] = func_233
mutated_mod = relay.transform.InferType()(mutated_mod)
var_234 = relay.var("var_234", dtype = "uint64", shape = (12,))#candidate|234|(12,)|var|uint64
func_233_call = mutated_mod.get_global_var('func_233')
call_235 = func_233_call(var_234)
output = call_235
func_236 = relay.Function([var_234], output)
mutated_mod['func_236'] = func_236
mutated_mod = relay.transform.InferType()(mutated_mod)
var_266 = relay.var("var_266", dtype = "uint8", shape = (1, 1))#candidate|266|(1, 1)|var|uint8
var_267 = relay.var("var_267", dtype = "uint8", shape = (11, 4))#candidate|267|(11, 4)|var|uint8
bop_268 = relay.greater_equal(var_266.astype('bool'), var_267.astype('bool')) # shape=(11, 4)
uop_271 = relay.tan(bop_268.astype('float64')) # shape=(11, 4)
uop_273 = relay.acos(uop_271.astype('float32')) # shape=(11, 4)
uop_275 = relay.atanh(uop_271.astype('float32')) # shape=(11, 4)
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
var_278 = relay.var("var_278", dtype = "uint64", shape = (320,))#candidate|278|(320,)|var|uint64
call_277 = relay.TupleGetItem(func_215_call(relay.reshape(var_278.astype('uint64'), [5, 4, 16]), relay.reshape(var_278.astype('bool'), [5, 4, 16]), relay.reshape(var_278.astype('int64'), [5, 4, 16]), ), 8)
call_279 = relay.TupleGetItem(func_219_call(relay.reshape(var_278.astype('uint64'), [5, 4, 16]), relay.reshape(var_278.astype('bool'), [5, 4, 16]), relay.reshape(var_278.astype('int64'), [5, 4, 16]), ), 8)
uop_280 = relay.sinh(uop_271.astype('float64')) # shape=(11, 4)
bop_288 = relay.bitwise_xor(uop_271.astype('int64'), relay.reshape(uop_275.astype('int64'), relay.shape_of(uop_271))) # shape=(11, 4)
bop_291 = relay.multiply(uop_275.astype('uint32'), relay.reshape(bop_268.astype('uint32'), relay.shape_of(uop_275))) # shape=(11, 4)
bop_294 = relay.add(bop_291.astype('int64'), relay.reshape(uop_275.astype('int64'), relay.shape_of(bop_291))) # shape=(11, 4)
uop_297 = relay.log(uop_275.astype('float64')) # shape=(11, 4)
uop_300 = relay.log(uop_297.astype('float32')) # shape=(11, 4)
bop_304 = relay.subtract(uop_297.astype('float32'), relay.reshape(bop_291.astype('float32'), relay.shape_of(uop_297))) # shape=(11, 4)
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
call_307 = relay.TupleGetItem(func_215_call(relay.reshape(var_278.astype('uint64'), [5, 4, 16]), relay.reshape(var_278.astype('bool'), [5, 4, 16]), relay.reshape(var_278.astype('int64'), [5, 4, 16]), ), 7)
call_308 = relay.TupleGetItem(func_219_call(relay.reshape(var_278.astype('uint64'), [5, 4, 16]), relay.reshape(var_278.astype('bool'), [5, 4, 16]), relay.reshape(var_278.astype('int64'), [5, 4, 16]), ), 7)
bop_310 = relay.maximum(uop_275.astype('uint64'), relay.reshape(uop_297.astype('uint64'), relay.shape_of(uop_275))) # shape=(11, 4)
const_313 = relay.const([[-1.212724,-3.126565,9.346881,3.578457],[8.375353,6.770253,-9.257678,9.196196],[3.890667,5.230631,4.700489,-9.046197],[1.260657,-5.711856,-7.335484,-7.392164],[-7.268366,-8.995947,3.252762,-3.108407],[-8.973724,-0.168333,9.012420,-2.409027],[-7.703092,5.087142,-4.439559,-3.255488],[-0.042626,-8.568354,-3.642700,-4.090770],[5.131620,-0.803246,-6.335531,-1.116000],[6.508748,7.414252,-1.134874,7.643506],[3.047687,-0.290027,-7.326360,-6.312414]], dtype = "float64")#candidate|313|(11, 4)|const|float64
bop_314 = relay.logical_and(uop_271.astype('bool'), relay.reshape(const_313.astype('bool'), relay.shape_of(uop_271))) # shape=(11, 4)
bop_319 = relay.minimum(bop_310.astype('int32'), relay.reshape(uop_271.astype('int32'), relay.shape_of(bop_310))) # shape=(11, 4)
bop_322 = relay.less_equal(bop_310.astype('bool'), relay.reshape(bop_314.astype('bool'), relay.shape_of(bop_310))) # shape=(11, 4)
uop_325 = relay.atan(bop_310.astype('float32')) # shape=(11, 4)
uop_327 = relay.exp(uop_325.astype('float64')) # shape=(11, 4)
func_81_call = mod.get_global_var('func_81')
func_84_call = mutated_mod.get_global_var('func_84')
const_333 = relay.const([-7.726451,-0.743122], dtype = "float64")#candidate|333|(2,)|const|float64
var_334 = relay.var("var_334", dtype = "uint32", shape = (63,))#candidate|334|(63,)|var|uint32
call_332 = relay.TupleGetItem(func_81_call(relay.reshape(const_333.astype('float64'), [2,]), relay.reshape(var_334.astype('uint32'), [63,]), ), 2)
call_335 = relay.TupleGetItem(func_84_call(relay.reshape(const_333.astype('float64'), [2,]), relay.reshape(var_334.astype('uint32'), [63,]), ), 2)
var_345 = relay.var("var_345", dtype = "float64", shape = (11, 4))#candidate|345|(11, 4)|var|float64
bop_346 = relay.equal(uop_327.astype('bool'), relay.reshape(var_345.astype('bool'), relay.shape_of(uop_327))) # shape=(11, 4)
bop_349 = relay.less(uop_327.astype('bool'), relay.reshape(bop_291.astype('bool'), relay.shape_of(uop_327))) # shape=(11, 4)
uop_352 = relay.asinh(uop_327.astype('float32')) # shape=(11, 4)
output = relay.Tuple([uop_273,call_277,var_278,uop_280,bop_288,bop_294,uop_300,bop_304,call_307,bop_319,bop_322,call_332,const_333,var_334,bop_346,bop_349,uop_352,])
output2 = relay.Tuple([uop_273,call_279,var_278,uop_280,bop_288,bop_294,uop_300,bop_304,call_308,bop_319,bop_322,call_335,const_333,var_334,bop_346,bop_349,uop_352,])
func_358 = relay.Function([var_266,var_267,var_278,var_334,var_345,], output)
mod['func_358'] = func_358
mod = relay.transform.InferType()(mod)
var_359 = relay.var("var_359", dtype = "uint8", shape = (1, 1))#candidate|359|(1, 1)|var|uint8
var_360 = relay.var("var_360", dtype = "uint8", shape = (11, 4))#candidate|360|(11, 4)|var|uint8
var_361 = relay.var("var_361", dtype = "uint64", shape = (320,))#candidate|361|(320,)|var|uint64
var_362 = relay.var("var_362", dtype = "uint32", shape = (63,))#candidate|362|(63,)|var|uint32
var_363 = relay.var("var_363", dtype = "float64", shape = (11, 4))#candidate|363|(11, 4)|var|float64
output = func_358(var_359,var_360,var_361,var_362,var_363,)
func_364 = relay.Function([var_359,var_360,var_361,var_362,var_363,], output)
mutated_mod['func_364'] = func_364
mutated_mod = relay.transform.InferType()(mutated_mod)
var_380 = relay.var("var_380", dtype = "int16", shape = (9, 5))#candidate|380|(9, 5)|var|int16
var_381 = relay.var("var_381", dtype = "int16", shape = (9, 5))#candidate|381|(9, 5)|var|int16
bop_382 = relay.bitwise_xor(var_380.astype('int16'), relay.reshape(var_381.astype('int16'), relay.shape_of(var_380))) # shape=(9, 5)
bop_385 = relay.add(var_380.astype('uint16'), relay.reshape(bop_382.astype('uint16'), relay.shape_of(var_380))) # shape=(9, 5)
uop_388 = relay.sinh(bop_382.astype('float64')) # shape=(9, 5)
var_391 = relay.var("var_391", dtype = "float64", shape = (9, 5))#candidate|391|(9, 5)|var|float64
bop_392 = relay.left_shift(uop_388.astype('int16'), relay.reshape(var_391.astype('int16'), relay.shape_of(uop_388))) # shape=(9, 5)
uop_395 = relay.rsqrt(bop_392.astype('float64')) # shape=(9, 5)
bop_397 = relay.bitwise_xor(uop_395.astype('uint16'), relay.reshape(bop_382.astype('uint16'), relay.shape_of(uop_395))) # shape=(9, 5)
uop_400 = relay.log10(uop_388.astype('float64')) # shape=(9, 5)
bop_402 = relay.not_equal(uop_400.astype('bool'), relay.reshape(var_391.astype('bool'), relay.shape_of(uop_400))) # shape=(9, 5)
output = relay.Tuple([bop_385,bop_397,bop_402,])
output2 = relay.Tuple([bop_385,bop_397,bop_402,])
func_405 = relay.Function([var_380,var_381,var_391,], output)
mod['func_405'] = func_405
mod = relay.transform.InferType()(mod)
mutated_mod['func_405'] = func_405
mutated_mod = relay.transform.InferType()(mutated_mod)
func_405_call = mutated_mod.get_global_var('func_405')
var_407 = relay.var("var_407", dtype = "int16", shape = (9, 5))#candidate|407|(9, 5)|var|int16
var_408 = relay.var("var_408", dtype = "int16", shape = (9, 5))#candidate|408|(9, 5)|var|int16
var_409 = relay.var("var_409", dtype = "float64", shape = (9, 5))#candidate|409|(9, 5)|var|float64
call_406 = func_405_call(var_407,var_408,var_409,)
output = call_406
func_410 = relay.Function([var_407,var_408,var_409,], output)
mutated_mod['func_410'] = func_410
mutated_mod = relay.transform.InferType()(mutated_mod)
var_412 = relay.var("var_412", dtype = "uint8", shape = (4, 13, 16))#candidate|412|(4, 13, 16)|var|uint8
var_413 = relay.var("var_413", dtype = "uint8", shape = (4, 13, 16))#candidate|413|(4, 13, 16)|var|uint8
bop_414 = relay.multiply(var_412.astype('uint8'), relay.reshape(var_413.astype('uint8'), relay.shape_of(var_412))) # shape=(4, 13, 16)
var_417 = relay.var("var_417", dtype = "uint8", shape = (4, 13, 16))#candidate|417|(4, 13, 16)|var|uint8
bop_418 = relay.less(var_413.astype('bool'), relay.reshape(var_417.astype('bool'), relay.shape_of(var_413))) # shape=(4, 13, 16)
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
const_422 = relay.const([-7,-9,4,7,10,7,-10,-1,-4,-8,-1,7,10,-5,-3,8,-1,10,-5,-1,4,-6,-9,-3,-9,5,-6,7,2,-1,-4,6,-4,-1,5,7,-9,4,-7,4,9,-7,2,3,5,10,10,5,2,4,-9,7,-3,8,9,7,3,-3,7,-10,9,-2,-6,3,-4,-7,10,-10,6,-5,2,-5,8,-2,3,7,-2,10,-5,-7,-2,1,8,-8,-1,-1,7,-10,3,1,-3,-9,5,-4,2,9,-6,9,4,-8,2,-10,5,-5,-3,4,5,3,-3,-10,8,3,2,8,1,10,7,-6,3,1,-8,-4,-10,-4,-5,2,-6,1,-10,1,-7,7,-3,6,-4,2,4,6,-4,-1,-5,-6,9,-8,-6,-10,-1,9,-10,8,4,6,7,-3,4,-1,8,-9,-7,-6,7,-8,-3,-2,9,-4,1,2,6,10,1,9,1,6,7,10,8,-4,-2,1,-8,6,-9,-7,6,-4,-9,8,-3,9,-6,5,7,-2,-4,4,-2,-1,-1,-7,-8,-5,4,-10,-8,8,-7,8,10,5,-7,-8,-1,9,1,-10,8,7,-9,-10,5,-2,9,-6,-10,-5,-10,-5,3,-3,-1,-3,-8,-1,-7,-5,9,5,-7,-6,-10,5,3,-10,-5,1,8,5,4,5,8,-6,-9,-6,-9,-5,-5,-10,-10,5,5,-3,1,1,5,9,-7,-4,-5,3,3,7,-8,-8,-9,-4,-10,9,7,7,5,-7,10,-9,2,-2,-9,4,-6,-6,5,1,-8,-7,2,7,9,-10,-5,-5,-10,-3,6,4,10,-10,1,10,-4,4,-10,-5,-5,7,3,-7,1,-10,9,-10], dtype = "uint64")#candidate|422|(320,)|const|uint64
call_421 = relay.TupleGetItem(func_215_call(relay.reshape(const_422.astype('uint64'), [5, 4, 16]), relay.reshape(const_422.astype('bool'), [5, 4, 16]), relay.reshape(const_422.astype('int64'), [5, 4, 16]), ), 4)
call_423 = relay.TupleGetItem(func_219_call(relay.reshape(const_422.astype('uint64'), [5, 4, 16]), relay.reshape(const_422.astype('bool'), [5, 4, 16]), relay.reshape(const_422.astype('int64'), [5, 4, 16]), ), 4)
bop_429 = relay.logical_and(bop_418.astype('bool'), relay.reshape(bop_414.astype('bool'), relay.shape_of(bop_418))) # shape=(4, 13, 16)
bop_432 = relay.logical_xor(bop_429.astype('int8'), relay.reshape(var_412.astype('int8'), relay.shape_of(bop_429))) # shape=(4, 13, 16)
bop_440 = relay.less_equal(bop_429.astype('bool'), relay.reshape(var_412.astype('bool'), relay.shape_of(bop_429))) # shape=(4, 13, 16)
output = relay.Tuple([call_421,const_422,bop_432,bop_440,])
output2 = relay.Tuple([call_423,const_422,bop_432,bop_440,])
func_443 = relay.Function([var_412,var_413,var_417,], output)
mod['func_443'] = func_443
mod = relay.transform.InferType()(mod)
mutated_mod['func_443'] = func_443
mutated_mod = relay.transform.InferType()(mutated_mod)
func_443_call = mutated_mod.get_global_var('func_443')
var_445 = relay.var("var_445", dtype = "uint8", shape = (4, 13, 16))#candidate|445|(4, 13, 16)|var|uint8
var_446 = relay.var("var_446", dtype = "uint8", shape = (4, 13, 16))#candidate|446|(4, 13, 16)|var|uint8
var_447 = relay.var("var_447", dtype = "uint8", shape = (4, 13, 16))#candidate|447|(4, 13, 16)|var|uint8
call_444 = func_443_call(var_445,var_446,var_447,)
output = call_444
func_448 = relay.Function([var_445,var_446,var_447,], output)
mutated_mod['func_448'] = func_448
mutated_mod = relay.transform.InferType()(mutated_mod)
const_461 = relay.const([-3.538191,-3.726379,2.713962,-3.048012,-6.284381,6.433347,-0.980780,1.283308], dtype = "float64")#candidate|461|(8,)|const|float64
uop_462 = relay.sigmoid(const_461.astype('float64')) # shape=(8,)
bop_469 = relay.subtract(uop_462.astype('uint8'), relay.reshape(const_461.astype('uint8'), relay.shape_of(uop_462))) # shape=(8,)
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
var_477 = relay.var("var_477", dtype = "uint64", shape = (320,))#candidate|477|(320,)|var|uint64
call_476 = relay.TupleGetItem(func_215_call(relay.reshape(var_477.astype('uint64'), [5, 4, 16]), relay.reshape(var_477.astype('bool'), [5, 4, 16]), relay.reshape(var_477.astype('int64'), [5, 4, 16]), ), 1)
call_478 = relay.TupleGetItem(func_219_call(relay.reshape(var_477.astype('uint64'), [5, 4, 16]), relay.reshape(var_477.astype('bool'), [5, 4, 16]), relay.reshape(var_477.astype('int64'), [5, 4, 16]), ), 1)
bop_484 = relay.bitwise_xor(uop_462.astype('uint32'), relay.reshape(const_461.astype('uint32'), relay.shape_of(uop_462))) # shape=(8,)
output = relay.Tuple([bop_469,call_476,var_477,bop_484,])
output2 = relay.Tuple([bop_469,call_478,var_477,bop_484,])
func_488 = relay.Function([var_477,], output)
mod['func_488'] = func_488
mod = relay.transform.InferType()(mod)
mutated_mod['func_488'] = func_488
mutated_mod = relay.transform.InferType()(mutated_mod)
var_489 = relay.var("var_489", dtype = "uint64", shape = (320,))#candidate|489|(320,)|var|uint64
func_488_call = mutated_mod.get_global_var('func_488')
call_490 = func_488_call(var_489)
output = call_490
func_491 = relay.Function([var_489], output)
mutated_mod['func_491'] = func_491
mutated_mod = relay.transform.InferType()(mutated_mod)
const_493 = relay.const(-3, dtype = "int16")#candidate|493|()|const|int16
var_494 = relay.var("var_494", dtype = "int16", shape = (4,))#candidate|494|(4,)|var|int16
bop_495 = relay.left_shift(const_493.astype('int16'), var_494.astype('int16')) # shape=(4,)
bop_501 = relay.logical_xor(bop_495.astype('int64'), const_493.astype('int64')) # shape=(4,)
output = relay.Tuple([bop_501,])
output2 = relay.Tuple([bop_501,])
func_508 = relay.Function([var_494,], output)
mod['func_508'] = func_508
mod = relay.transform.InferType()(mod)
var_509 = relay.var("var_509", dtype = "int16", shape = (4,))#candidate|509|(4,)|var|int16
output = func_508(var_509)
func_510 = relay.Function([var_509], output)
mutated_mod['func_510'] = func_510
mutated_mod = relay.transform.InferType()(mutated_mod)
const_512 = relay.const([-2.198152,4.093703,-1.635876,-9.348376,0.737006,-9.068075], dtype = "float32")#candidate|512|(6,)|const|float32
var_513 = relay.var("var_513", dtype = "float32", shape = (6,))#candidate|513|(6,)|var|float32
bop_514 = relay.floor_mod(const_512.astype('float32'), relay.reshape(var_513.astype('float32'), relay.shape_of(const_512))) # shape=(6,)
uop_517 = relay.log10(const_512.astype('float64')) # shape=(6,)
bop_519 = relay.logical_or(uop_517.astype('bool'), relay.reshape(bop_514.astype('bool'), relay.shape_of(uop_517))) # shape=(6,)
var_522 = relay.var("var_522", dtype = "bool", shape = (6,))#candidate|522|(6,)|var|bool
bop_523 = relay.greater_equal(bop_519.astype('bool'), relay.reshape(var_522.astype('bool'), relay.shape_of(bop_519))) # shape=(6,)
uop_528 = relay.cosh(bop_523.astype('float32')) # shape=(6,)
output = relay.Tuple([uop_528,])
output2 = relay.Tuple([uop_528,])
func_531 = relay.Function([var_513,var_522,], output)
mod['func_531'] = func_531
mod = relay.transform.InferType()(mod)
mutated_mod['func_531'] = func_531
mutated_mod = relay.transform.InferType()(mutated_mod)
func_531_call = mutated_mod.get_global_var('func_531')
var_533 = relay.var("var_533", dtype = "float32", shape = (6,))#candidate|533|(6,)|var|float32
var_534 = relay.var("var_534", dtype = "bool", shape = (6,))#candidate|534|(6,)|var|bool
call_532 = func_531_call(var_533,var_534,)
output = call_532
func_535 = relay.Function([var_533,var_534,], output)
mutated_mod['func_535'] = func_535
mutated_mod = relay.transform.InferType()(mutated_mod)
var_588 = relay.var("var_588", dtype = "uint16", shape = ())#candidate|588|()|var|uint16
var_589 = relay.var("var_589", dtype = "uint16", shape = (9, 11, 13))#candidate|589|(9, 11, 13)|var|uint16
bop_590 = relay.minimum(var_588.astype('uint16'), var_589.astype('uint16')) # shape=(9, 11, 13)
bop_597 = relay.left_shift(var_589.astype('int32'), relay.reshape(bop_590.astype('int32'), relay.shape_of(var_589))) # shape=(9, 11, 13)
output = bop_597
output2 = bop_597
func_600 = relay.Function([var_588,var_589,], output)
mod['func_600'] = func_600
mod = relay.transform.InferType()(mod)
var_601 = relay.var("var_601", dtype = "uint16", shape = ())#candidate|601|()|var|uint16
var_602 = relay.var("var_602", dtype = "uint16", shape = (9, 11, 13))#candidate|602|(9, 11, 13)|var|uint16
output = func_600(var_601,var_602,)
func_603 = relay.Function([var_601,var_602,], output)
mutated_mod['func_603'] = func_603
mutated_mod = relay.transform.InferType()(mutated_mod)
var_605 = relay.var("var_605", dtype = "float64", shape = (9,))#candidate|605|(9,)|var|float64
uop_606 = relay.exp(var_605.astype('float64')) # shape=(9,)
bop_608 = relay.less_equal(var_605.astype('bool'), relay.reshape(uop_606.astype('bool'), relay.shape_of(var_605))) # shape=(9,)
func_233_call = mod.get_global_var('func_233')
func_236_call = mutated_mod.get_global_var('func_236')
const_612 = relay.const([7,3,2,-3,6,5,3,3,2,-10,9,4], dtype = "uint64")#candidate|612|(12,)|const|uint64
call_611 = func_233_call(relay.reshape(const_612.astype('uint64'), [12,]))
call_613 = func_233_call(relay.reshape(const_612.astype('uint64'), [12,]))
var_614 = relay.var("var_614", dtype = "float64", shape = (9,))#candidate|614|(9,)|var|float64
bop_615 = relay.less(uop_606.astype('bool'), relay.reshape(var_614.astype('bool'), relay.shape_of(uop_606))) # shape=(9,)
bop_619 = relay.minimum(bop_608.astype('int64'), relay.reshape(uop_606.astype('int64'), relay.shape_of(bop_608))) # shape=(9,)
bop_623 = relay.floor_divide(bop_608.astype('float32'), relay.reshape(uop_606.astype('float32'), relay.shape_of(bop_608))) # shape=(9,)
bop_628 = relay.add(var_605.astype('int16'), relay.reshape(bop_623.astype('int16'), relay.shape_of(var_605))) # shape=(9,)
bop_633 = relay.right_shift(var_605.astype('int8'), relay.reshape(bop_608.astype('int8'), relay.shape_of(var_605))) # shape=(9,)
bop_637 = relay.floor_mod(var_614.astype('float64'), relay.reshape(uop_606.astype('float64'), relay.shape_of(var_614))) # shape=(9,)
uop_640 = relay.asin(bop_619.astype('float64')) # shape=(9,)
uop_642 = relay.log10(bop_615.astype('float64')) # shape=(9,)
bop_645 = relay.greater_equal(uop_642.astype('bool'), relay.reshape(uop_640.astype('bool'), relay.shape_of(uop_642))) # shape=(9,)
output = relay.Tuple([call_611,const_612,bop_628,bop_633,bop_637,bop_645,])
output2 = relay.Tuple([call_613,const_612,bop_628,bop_633,bop_637,bop_645,])
F = relay.Function([var_605,var_614,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_605,var_614,], output2)
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
	relay.transform.InferType(),
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
input_605= np.array([-0.026501,-0.891943,-5.468154,7.789134,-2.383878,7.789397,2.558537,-6.203746,-1.473263], dtype='float64')
module1.set_input('var_605', input_605)
input_614= np.array([-4.716485,-3.666400,-9.018317,1.105730,-1.798189,-1.296804,-0.739016,7.520618,-9.432806], dtype='float64')
module1.set_input('var_614', input_614)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_605, input_614, )
res3 = intrp3.evaluate()(input_605, input_614, )
res4 = intrp4.evaluate()(input_605, input_614, )
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
res1_4 = module1.get_output(4).asnumpy()
res2_4 = res2[4].asnumpy()
res3_4 = res3[4].asnumpy()
res4_4 = res4[4].asnumpy()
np.testing.assert_allclose(res1_4 ,res2_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_4 ,res3_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_4 ,res4_4, atol=1e-3, rtol=1e-3)
(res1_4 == res2_4).all()
(res1_4 == res3_4).all()
(res1_4 == res4_4).all()
res1_5 = module1.get_output(5).asnumpy()
res2_5 = res2[5].asnumpy()
res3_5 = res3[5].asnumpy()
res4_5 = res4[5].asnumpy()
np.testing.assert_allclose(res1_5 ,res2_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_5 ,res3_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_5 ,res4_5, atol=1e-3, rtol=1e-3)
(res1_5 == res2_5).all()
(res1_5 == res3_5).all()
(res1_5 == res4_5).all()
module5.set_input('var_605', input_605)
module5.set_input('var_614', input_614)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_605, input_614, )
res7 = intrp7.evaluate()(input_605, input_614, )
res8 = intrp8.evaluate()(input_605, input_614, )
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
res5_4 = module5.get_output(4).asnumpy()
res6_4 = res6[4].asnumpy()
res7_4 = res7[4].asnumpy()
res8_4 = res8[4].asnumpy()
np.testing.assert_allclose(res5_4 ,res6_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_4 ,res7_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_4 ,res8_4, atol=1e-3, rtol=1e-3)
(res5_4 == res6_4).all()
(res5_4 == res7_4).all()
(res5_4 == res8_4).all()
res5_5 = module5.get_output(5).asnumpy()
res6_5 = res6[5].asnumpy()
res7_5 = res7[5].asnumpy()
res8_5 = res8[5].asnumpy()
np.testing.assert_allclose(res5_5 ,res6_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_5 ,res7_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_5 ,res8_5, atol=1e-3, rtol=1e-3)
(res5_5 == res6_5).all()
(res5_5 == res7_5).all()
(res5_5 == res8_5).all()
module9.set_input('var_605', input_605)
module9.set_input('var_614', input_614)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_605, input_614, )
res11 = intrp11.evaluate()(input_605, input_614, )
res12 = intrp12.evaluate()(input_605, input_614, )
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
res9_4 = module9.get_output(4).asnumpy()
res10_4 = res10[4].asnumpy()
res11_4 = res11[4].asnumpy()
res12_4 = res12[4].asnumpy()
np.testing.assert_allclose(res9_4 ,res10_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_4 ,res11_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_4 ,res12_4, atol=1e-3, rtol=1e-3)
(res9_4 == res10_4).all()
(res9_4 == res11_4).all()
(res9_4 == res12_4).all()
res9_5 = module9.get_output(5).asnumpy()
res10_5 = res10[5].asnumpy()
res11_5 = res11[5].asnumpy()
res12_5 = res12[5].asnumpy()
np.testing.assert_allclose(res9_5 ,res10_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_5 ,res11_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_5 ,res12_5, atol=1e-3, rtol=1e-3)
(res9_5 == res10_5).all()
(res9_5 == res11_5).all()
(res9_5 == res12_5).all()
module13.set_input('var_605', input_605)
module13.set_input('var_614', input_614)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_605, input_614, )
res15 = intrp15.evaluate()(input_605, input_614, )
res16 = intrp16.evaluate()(input_605, input_614, )
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
res13_4 = module13.get_output(4).asnumpy()
res14_4 = res14[4].asnumpy()
res15_4 = res15[4].asnumpy()
res16_4 = res16[4].asnumpy()
np.testing.assert_allclose(res13_4 ,res14_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_4 ,res15_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_4 ,res16_4, atol=1e-3, rtol=1e-3)
(res13_4 == res14_4).all()
(res13_4 == res15_4).all()
(res13_4 == res16_4).all()
res13_5 = module13.get_output(5).asnumpy()
res14_5 = res14[5].asnumpy()
res15_5 = res15[5].asnumpy()
res16_5 = res16[5].asnumpy()
np.testing.assert_allclose(res13_5 ,res14_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_5 ,res15_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_5 ,res16_5, atol=1e-3, rtol=1e-3)
(res13_5 == res14_5).all()
(res13_5 == res15_5).all()
(res13_5 == res16_5).all()
module17.set_input('var_605', input_605)
module17.set_input('var_614', input_614)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_605, input_614, )
res19 = intrp19.evaluate()(input_605, input_614, )
res20 = intrp20.evaluate()(input_605, input_614, )
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
res17_4 = module17.get_output(4).asnumpy()
res18_4 = res18[4].asnumpy()
res19_4 = res19[4].asnumpy()
res20_4 = res20[4].asnumpy()
np.testing.assert_allclose(res17_4 ,res18_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_4 ,res19_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_4 ,res20_4, atol=1e-3, rtol=1e-3)
(res17_4 == res18_4).all()
(res17_4 == res19_4).all()
(res17_4 == res20_4).all()
res17_5 = module17.get_output(5).asnumpy()
res18_5 = res18[5].asnumpy()
res19_5 = res19[5].asnumpy()
res20_5 = res20[5].asnumpy()
np.testing.assert_allclose(res17_5 ,res18_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_5 ,res19_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_5 ,res20_5, atol=1e-3, rtol=1e-3)
(res17_5 == res18_5).all()
(res17_5 == res19_5).all()
(res17_5 == res20_5).all()
module21.set_input('var_605', input_605)
module21.set_input('var_614', input_614)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_605, input_614, )
res23 = intrp23.evaluate()(input_605, input_614, )
res24 = intrp24.evaluate()(input_605, input_614, )
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
res21_4 = module21.get_output(4).asnumpy()
res22_4 = res22[4].asnumpy()
res23_4 = res23[4].asnumpy()
res24_4 = res24[4].asnumpy()
np.testing.assert_allclose(res21_4 ,res22_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_4 ,res23_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_4 ,res24_4, atol=1e-3, rtol=1e-3)
(res21_4 == res22_4).all()
(res21_4 == res23_4).all()
(res21_4 == res24_4).all()
res21_5 = module21.get_output(5).asnumpy()
res22_5 = res22[5].asnumpy()
res23_5 = res23[5].asnumpy()
res24_5 = res24[5].asnumpy()
np.testing.assert_allclose(res21_5 ,res22_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_5 ,res23_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_5 ,res24_5, atol=1e-3, rtol=1e-3)
(res21_5 == res22_5).all()
(res21_5 == res23_5).all()
(res21_5 == res24_5).all()

'''57: TVMFuncCall
56: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
55: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
54: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
53: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
52: tvm::transform::Pass::operator()(tvm::IRModule) const
51: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
50: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
49: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
48: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
47: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
46: tvm::relay::DynamicToStatic(tvm::relay::Function, tvm::IRModule)
45: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
44: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
43: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
42: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
41: _ZN3tvm5relay16MixedModeMutato
40: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
39: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
38: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
37: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
36: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
35: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
34: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
33: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
32: _ZN3tvm5relay16MixedModeMutato
31: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
30: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
29: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
28: tvm::relay::MixedModeMutator::VisitExpr_(tvm::relay::CallNode const*)
27: tvm::relay::DynamicToStaticMutator::Rewrite_(tvm::relay::CallNode const*, tvm::RelayExpr const&)
26: std::_Function_handler<tvm::RelayExpr (tvm::relay::CallNode const*), tvm::relay::DynamicToStaticMutator::DynamicToStaticMutator(tvm::IRModule, tvm::relay::Function)::{lambda(tvm::relay::CallNode const*)#1}>::_M_invoke(std::_Any_data const&, tvm::relay::CallNode const*&&)
25: tvm::relay::DynamicToStaticMutator::PrepareArgs(tvm::relay::CallNode const*)
24: tvm::relay::DynamicToStaticMutator::PrepareInput(tvm::RelayExpr const&)
23: tvm::transform::Pass::operator()(tvm::IRModule) const
22: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
21: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
20: tvm::transform::Pass::operator()(tvm::IRModule) const
19: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
18: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
17: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
16: tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}::operator()(tvm::IRModule, tvm::transform::PassContext const&) const [clone .isra.813]
15: tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)
14: tvm::relay::TypeInferencer::GetType(tvm::RelayExpr const&)
13: tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)
12: void tvm::relay::ExpandDataflow<tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}>(tvm::RelayExpr, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2})
11: void tvm::relay::ExpandDataflow<tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1})
10: _ZZN3tvm5relay11ExprFunctorIFNS_4TypeERK
9: tvm::relay::TypeInferencer::VisitExpr_(tvm::relay::FunctionNode const*)
8: tvm::relay::TypeInferencer::GetType(tvm::RelayExpr const&)
7: tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)
6: void tvm::relay::ExpandDataflow<tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}>(tvm::RelayExpr, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2})
5: void tvm::relay::ExpandDataflow<tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::TypeInferencer::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1})
4: _ZZN3tvm5relay11ExprFunctorIFNS_4TypeERK
3: tvm::relay::TypeInferencer::VisitExpr_(tvm::relay::CallNode const*)
2: tvm::relay::TypeInferencer::GeneralCall(tvm::relay::CallNode const*, tvm::runtime::Array<tvm::Type, void>)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''