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
var_0 = relay.var("var_0", dtype = "float64", shape = (1,))#candidate|0|(1,)|var|float64
uop_1 = relay.log10(var_0.astype('float64')) # shape=(1,)
uop_3 = relay.erf(var_0.astype('float32')) # shape=(1,)
bop_5 = relay.less_equal(var_0.astype('bool'), relay.reshape(uop_3.astype('bool'), relay.shape_of(var_0))) # shape=(1,)
const_8 = relay.const([-4.108927,-1.702729,-4.108586,3.564832,-8.600460,9.483706,-7.671987,6.177191,-2.777777], dtype = "float64")#candidate|8|(9,)|const|float64
bop_9 = relay.maximum(uop_1.astype('int8'), const_8.astype('int8')) # shape=(9,)
output = relay.Tuple([bop_5,bop_9,])
output2 = relay.Tuple([bop_5,bop_9,])
func_12 = relay.Function([var_0,], output)
mod['func_12'] = func_12
mod = relay.transform.InferType()(mod)
var_13 = relay.var("var_13", dtype = "float64", shape = (1,))#candidate|13|(1,)|var|float64
output = func_12(var_13)
func_14 = relay.Function([var_13], output)
mutated_mod['func_14'] = func_14
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16 = relay.var("var_16", dtype = "float32", shape = ())#candidate|16|()|var|float32
uop_17 = relay.erf(var_16.astype('float32')) # shape=()
uop_19 = relay.cos(uop_17.astype('float64')) # shape=()
var_21 = relay.var("var_21", dtype = "float32", shape = (2,))#candidate|21|(2,)|var|float32
bop_22 = relay.floor_divide(var_16.astype('float64'), var_21.astype('float64')) # shape=(2,)
uop_25 = relay.acos(uop_17.astype('float32')) # shape=()
bop_27 = relay.add(uop_19.astype('uint16'), var_21.astype('uint16')) # shape=(2,)
bop_30 = relay.logical_and(uop_19.astype('bool'), uop_17.astype('bool')) # shape=()
const_33 = relay.const([[8.429518],[-3.648878],[-2.452167],[-3.176037],[1.000818],[0.948588],[2.983722],[7.566768]], dtype = "float32")#candidate|33|(8, 1)|const|float32
bop_34 = relay.divide(uop_25.astype('float32'), const_33.astype('float32')) # shape=(8, 1)
bop_37 = relay.bitwise_and(uop_19.astype('int64'), bop_22.astype('int64')) # shape=(2,)
bop_40 = relay.less_equal(uop_25.astype('bool'), const_33.astype('bool')) # shape=(8, 1)
uop_43 = relay.sinh(bop_37.astype('float32')) # shape=(2,)
var_45 = relay.var("var_45", dtype = "float32", shape = (2,))#candidate|45|(2,)|var|float32
bop_46 = relay.floor_mod(uop_43.astype('float32'), relay.reshape(var_45.astype('float32'), relay.shape_of(uop_43))) # shape=(2,)
uop_49 = relay.asinh(bop_46.astype('float32')) # shape=(2,)
uop_51 = relay.cosh(uop_49.astype('float32')) # shape=(2,)
uop_53 = relay.atanh(uop_51.astype('float64')) # shape=(2,)
const_55 = relay.const([5.724713,-4.565158], dtype = "float64")#candidate|55|(2,)|const|float64
bop_56 = relay.not_equal(uop_53.astype('bool'), relay.reshape(const_55.astype('bool'), relay.shape_of(uop_53))) # shape=(2,)
uop_59 = relay.sqrt(uop_53.astype('float32')) # shape=(2,)
uop_61 = relay.log10(uop_59.astype('float32')) # shape=(2,)
bop_63 = relay.mod(uop_59.astype('float64'), relay.reshape(bop_27.astype('float64'), relay.shape_of(uop_59))) # shape=(2,)
uop_66 = relay.rsqrt(bop_63.astype('float64')) # shape=(2,)
var_68 = relay.var("var_68", dtype = "float32", shape = (2,))#candidate|68|(2,)|var|float32
bop_69 = relay.logical_and(uop_51.astype('bool'), relay.reshape(var_68.astype('bool'), relay.shape_of(uop_51))) # shape=(2,)
bop_72 = relay.floor_divide(uop_66.astype('float64'), var_16.astype('float64')) # shape=(2,)
const_75 = relay.const([3.327465,-9.034549], dtype = "float64")#candidate|75|(2,)|const|float64
bop_76 = relay.divide(bop_63.astype('float32'), relay.reshape(const_75.astype('float32'), relay.shape_of(bop_63))) # shape=(2,)
uop_79 = relay.asin(bop_56.astype('float64')) # shape=(2,)
const_81 = relay.const([9.690500,-3.175958], dtype = "float64")#candidate|81|(2,)|const|float64
bop_82 = relay.greater(uop_53.astype('bool'), relay.reshape(const_81.astype('bool'), relay.shape_of(uop_53))) # shape=(2,)
uop_85 = relay.cos(uop_66.astype('float32')) # shape=(2,)
bop_87 = relay.logical_or(uop_85.astype('bool'), bop_40.astype('bool')) # shape=(8, 2)
uop_90 = relay.acosh(bop_87.astype('float32')) # shape=(8, 2)
uop_92 = relay.log10(uop_90.astype('float64')) # shape=(8, 2)
bop_94 = relay.logical_and(uop_92.astype('bool'), uop_49.astype('bool')) # shape=(8, 2)
bop_97 = relay.add(bop_63.astype('uint32'), relay.reshape(bop_72.astype('uint32'), relay.shape_of(bop_63))) # shape=(2,)
uop_100 = relay.acosh(uop_92.astype('float32')) # shape=(8, 2)
uop_102 = relay.cosh(uop_85.astype('float64')) # shape=(2,)
uop_104 = relay.sin(bop_94.astype('float64')) # shape=(8, 2)
var_106 = relay.var("var_106", dtype = "float32", shape = (8, 2))#candidate|106|(8, 2)|var|float32
bop_107 = relay.bitwise_or(uop_100.astype('int32'), relay.reshape(var_106.astype('int32'), relay.shape_of(uop_100))) # shape=(8, 2)
bop_110 = relay.bitwise_xor(uop_100.astype('int16'), const_75.astype('int16')) # shape=(8, 2)
uop_113 = relay.tan(uop_102.astype('float64')) # shape=(2,)
bop_115 = relay.bitwise_or(uop_100.astype('int64'), bop_76.astype('int64')) # shape=(8, 2)
uop_118 = relay.sqrt(bop_107.astype('float64')) # shape=(8, 2)
bop_120 = relay.subtract(bop_107.astype('uint32'), relay.reshape(bop_94.astype('uint32'), relay.shape_of(bop_107))) # shape=(8, 2)
bop_123 = relay.greater(uop_118.astype('bool'), var_45.astype('bool')) # shape=(8, 2)
func_12_call = mod.get_global_var('func_12')
func_14_call = mutated_mod.get_global_var('func_14')
call_126 = relay.TupleGetItem(func_12_call(relay.reshape(bop_30.astype('float64'), [1,])), 0)
call_127 = relay.TupleGetItem(func_14_call(relay.reshape(bop_30.astype('float64'), [1,])), 0)
bop_128 = relay.equal(bop_94.astype('bool'), relay.reshape(bop_115.astype('bool'), relay.shape_of(bop_94))) # shape=(8, 2)
bop_131 = relay.greater_equal(bop_123.astype('bool'), uop_25.astype('bool')) # shape=(8, 2)
func_12_call = mod.get_global_var('func_12')
func_14_call = mutated_mod.get_global_var('func_14')
call_134 = relay.TupleGetItem(func_12_call(relay.reshape(var_16.astype('float64'), [1,])), 1)
call_135 = relay.TupleGetItem(func_14_call(relay.reshape(var_16.astype('float64'), [1,])), 1)
output = relay.Tuple([bop_30,bop_34,uop_61,bop_69,uop_79,bop_82,bop_97,uop_104,bop_110,uop_113,bop_120,call_126,bop_128,bop_131,call_134,])
output2 = relay.Tuple([bop_30,bop_34,uop_61,bop_69,uop_79,bop_82,bop_97,uop_104,bop_110,uop_113,bop_120,call_127,bop_128,bop_131,call_135,])
func_136 = relay.Function([var_16,var_21,var_45,var_68,var_106,], output)
mod['func_136'] = func_136
mod = relay.transform.InferType()(mod)
var_137 = relay.var("var_137", dtype = "float32", shape = ())#candidate|137|()|var|float32
var_138 = relay.var("var_138", dtype = "float32", shape = (2,))#candidate|138|(2,)|var|float32
var_139 = relay.var("var_139", dtype = "float32", shape = (2,))#candidate|139|(2,)|var|float32
var_140 = relay.var("var_140", dtype = "float32", shape = (2,))#candidate|140|(2,)|var|float32
var_141 = relay.var("var_141", dtype = "float32", shape = (8, 2))#candidate|141|(8, 2)|var|float32
output = func_136(var_137,var_138,var_139,var_140,var_141,)
func_142 = relay.Function([var_137,var_138,var_139,var_140,var_141,], output)
mutated_mod['func_142'] = func_142
mutated_mod = relay.transform.InferType()(mutated_mod)
var_144 = relay.var("var_144", dtype = "uint64", shape = (8, 3, 6))#candidate|144|(8, 3, 6)|var|uint64
const_145 = relay.const([[[-10,-2,-1,-2,-3,-6],[9,-3,8,-9,-1,-5],[-7,3,-4,7,-3,8]],[[4,-6,-6,6,-2,6],[8,4,4,-8,-2,6],[-9,3,4,5,3,8]],[[4,5,-3,-7,10,4],[9,-4,3,1,3,-2],[-5,-6,8,10,-8,-7]],[[-7,3,6,-1,7,7],[3,10,2,-2,8,-9],[-2,6,3,-2,-8,8]],[[-10,10,-4,7,-7,-3],[8,-7,-5,-3,-5,-8],[4,9,-5,-7,-7,2]],[[4,-9,6,-1,9,-3],[6,3,4,-2,-3,10],[9,3,-4,7,6,-10]],[[-8,-9,9,-1,-7,-6],[6,-9,8,-7,7,-5],[6,-1,-8,7,2,-2]],[[9,5,8,6,1,2],[-5,8,-10,-6,-5,-1],[8,8,5,8,-3,-9]]], dtype = "uint64")#candidate|145|(8, 3, 6)|const|uint64
bop_146 = relay.add(var_144.astype('uint64'), relay.reshape(const_145.astype('uint64'), relay.shape_of(var_144))) # shape=(8, 3, 6)
bop_149 = relay.subtract(bop_146.astype('int32'), relay.reshape(var_144.astype('int32'), relay.shape_of(bop_146))) # shape=(8, 3, 6)
func_136_call = mod.get_global_var('func_136')
func_142_call = mutated_mod.get_global_var('func_142')
const_153 = relay.const(-6.815268, dtype = "float32")#candidate|153|()|const|float32
var_154 = relay.var("var_154", dtype = "float32", shape = (2,))#candidate|154|(2,)|var|float32
var_155 = relay.var("var_155", dtype = "float32", shape = (16,))#candidate|155|(16,)|var|float32
call_152 = relay.TupleGetItem(func_136_call(relay.reshape(const_153.astype('float32'), []), relay.reshape(var_154.astype('float32'), [2,]), relay.reshape(var_154.astype('float32'), [2,]), relay.reshape(var_154.astype('float32'), [2,]), relay.reshape(var_155.astype('float32'), [8, 2]), ), 7)
call_156 = relay.TupleGetItem(func_142_call(relay.reshape(const_153.astype('float32'), []), relay.reshape(var_154.astype('float32'), [2,]), relay.reshape(var_154.astype('float32'), [2,]), relay.reshape(var_154.astype('float32'), [2,]), relay.reshape(var_155.astype('float32'), [8, 2]), ), 7)
bop_157 = relay.logical_or(var_154.astype('bool'), call_152.astype('bool')) # shape=(8, 2)
bop_160 = relay.logical_or(var_154.astype('bool'), call_156.astype('bool')) # shape=(8, 2)
uop_161 = relay.asinh(bop_149.astype('float32')) # shape=(8, 3, 6)
bop_163 = relay.logical_xor(uop_161.astype('uint16'), relay.reshape(var_144.astype('uint16'), relay.shape_of(uop_161))) # shape=(8, 3, 6)
bop_166 = relay.logical_xor(bop_163.astype('uint16'), relay.reshape(bop_146.astype('uint16'), relay.shape_of(bop_163))) # shape=(8, 3, 6)
bop_169 = relay.greater_equal(bop_163.astype('bool'), relay.reshape(uop_161.astype('bool'), relay.shape_of(bop_163))) # shape=(8, 3, 6)
uop_172 = relay.erf(bop_166.astype('float64')) # shape=(8, 3, 6)
var_174 = relay.var("var_174", dtype = "float32", shape = (8, 3, 6))#candidate|174|(8, 3, 6)|var|float32
bop_175 = relay.not_equal(uop_161.astype('bool'), relay.reshape(var_174.astype('bool'), relay.shape_of(uop_161))) # shape=(8, 3, 6)
output = relay.Tuple([const_153,var_155,bop_157,bop_169,uop_172,bop_175,])
output2 = relay.Tuple([const_153,var_155,bop_160,bop_169,uop_172,bop_175,])
func_178 = relay.Function([var_144,var_154,var_155,var_174,], output)
mod['func_178'] = func_178
mod = relay.transform.InferType()(mod)
mutated_mod['func_178'] = func_178
mutated_mod = relay.transform.InferType()(mutated_mod)
func_178_call = mutated_mod.get_global_var('func_178')
var_180 = relay.var("var_180", dtype = "uint64", shape = (8, 3, 6))#candidate|180|(8, 3, 6)|var|uint64
var_181 = relay.var("var_181", dtype = "float32", shape = (2,))#candidate|181|(2,)|var|float32
var_182 = relay.var("var_182", dtype = "float32", shape = (16,))#candidate|182|(16,)|var|float32
var_183 = relay.var("var_183", dtype = "float32", shape = (8, 3, 6))#candidate|183|(8, 3, 6)|var|float32
call_179 = func_178_call(var_180,var_181,var_182,var_183,)
output = call_179
func_184 = relay.Function([var_180,var_181,var_182,var_183,], output)
mutated_mod['func_184'] = func_184
mutated_mod = relay.transform.InferType()(mutated_mod)
var_186 = relay.var("var_186", dtype = "float32", shape = (15, 7))#candidate|186|(15, 7)|var|float32
uop_187 = relay.sinh(var_186.astype('float32')) # shape=(15, 7)
uop_189 = relay.sinh(uop_187.astype('float32')) # shape=(15, 7)
uop_191 = relay.asinh(uop_189.astype('float64')) # shape=(15, 7)
bop_193 = relay.logical_and(uop_189.astype('bool'), relay.reshape(uop_187.astype('bool'), relay.shape_of(uop_189))) # shape=(15, 7)
uop_196 = relay.sin(uop_187.astype('float64')) # shape=(15, 7)
func_136_call = mod.get_global_var('func_136')
func_142_call = mutated_mod.get_global_var('func_142')
const_199 = relay.const(-1.061274, dtype = "float32")#candidate|199|()|const|float32
const_200 = relay.const([[4.517584],[-1.829781]], dtype = "float32")#candidate|200|(2, 1)|const|float32
var_201 = relay.var("var_201", dtype = "float32", shape = (16,))#candidate|201|(16,)|var|float32
call_198 = relay.TupleGetItem(func_136_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 5)
call_202 = relay.TupleGetItem(func_142_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 5)
uop_203 = relay.tan(uop_191.astype('float64')) # shape=(15, 7)
uop_205 = relay.sin(uop_203.astype('float32')) # shape=(15, 7)
uop_207 = relay.atanh(uop_205.astype('float64')) # shape=(15, 7)
bop_209 = relay.less(uop_207.astype('bool'), relay.reshape(uop_189.astype('bool'), relay.shape_of(uop_207))) # shape=(15, 7)
bop_212 = relay.subtract(uop_207.astype('uint32'), relay.reshape(uop_187.astype('uint32'), relay.shape_of(uop_207))) # shape=(15, 7)
uop_215 = relay.asinh(bop_212.astype('float64')) # shape=(15, 7)
const_217 = relay.const([[False,True,False,False,True,True,True],[True,True,True,False,False,True,True],[True,True,True,False,False,False,False],[False,True,False,False,True,False,True],[False,True,False,False,False,False,False],[True,True,True,False,False,False,True],[True,True,False,False,True,True,True],[True,True,True,True,False,True,True],[False,True,False,False,False,False,False],[False,True,True,False,False,False,False],[True,True,True,False,False,True,False],[True,False,True,False,True,True,True],[False,False,False,False,False,True,False],[False,True,False,True,False,True,True],[True,False,True,False,True,True,False]], dtype = "bool")#candidate|217|(15, 7)|const|bool
bop_218 = relay.mod(bop_209.astype('float64'), relay.reshape(const_217.astype('float64'), relay.shape_of(bop_209))) # shape=(15, 7)
bop_221 = relay.greater_equal(uop_205.astype('bool'), relay.reshape(bop_218.astype('bool'), relay.shape_of(uop_205))) # shape=(15, 7)
uop_224 = relay.sqrt(bop_212.astype('float32')) # shape=(15, 7)
func_178_call = mod.get_global_var('func_178')
func_184_call = mutated_mod.get_global_var('func_184')
const_227 = relay.const([1,-6,9,-9,-9,-4,-1,-10,7,9,2,4,6,-1,7,9,-9,-2,-6,7,7,-4,2,-7,1,8,7,-5,10,-4,-10,-6,9,-7,-8,-2,-6,3,8,5,-5,-5,9,-5,-6,7,-5,2,8,1,-4,10,9,5,5,-9,7,5,3,-4,-1,4,-3,-2,-8,6,5,-9,1,-4,-8,-9,6,-10,3,8,6,-4,10,9,-8,-7,4,1,-4,4,-9,-6,6,-6,1,-7,-5,5,6,1,-2,10,6,1,-6,1,-7,-2,4,-4,-1,-1,9,-2,-1,-10,5,9,3,3,-7,7,-1,9,-5,10,1,2,7,-8,1,3,9,-7,-4,-2,-5,6,-7,3,7,8,7,1,-8,-4,6,4], dtype = "uint64")#candidate|227|(144,)|const|uint64
call_226 = relay.TupleGetItem(func_178_call(relay.reshape(const_227.astype('uint64'), [8, 3, 6]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [16,]), relay.reshape(const_227.astype('float32'), [8, 3, 6]), ), 5)
call_228 = relay.TupleGetItem(func_184_call(relay.reshape(const_227.astype('uint64'), [8, 3, 6]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [16,]), relay.reshape(const_227.astype('float32'), [8, 3, 6]), ), 5)
uop_229 = relay.log(uop_215.astype('float32')) # shape=(15, 7)
uop_231 = relay.erf(uop_229.astype('float64')) # shape=(15, 7)
bop_233 = relay.right_shift(uop_231.astype('int64'), relay.reshape(uop_205.astype('int64'), relay.shape_of(uop_231))) # shape=(15, 7)
uop_236 = relay.log(uop_229.astype('float32')) # shape=(15, 7)
uop_238 = relay.sigmoid(uop_229.astype('float64')) # shape=(15, 7)
func_136_call = mod.get_global_var('func_136')
func_142_call = mutated_mod.get_global_var('func_142')
call_240 = relay.TupleGetItem(func_136_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(call_198.astype('float32'), [2,]), relay.reshape(call_198.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 5)
call_241 = relay.TupleGetItem(func_142_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(call_198.astype('float32'), [2,]), relay.reshape(call_198.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 5)
const_242 = relay.const([[-4.623724,6.657989,2.675003,-5.133684,2.465584,3.573980,7.826142],[0.085984,1.185946,4.675772,-2.881031,-6.630100,7.475600,5.313985],[-9.706117,2.504991,1.938797,-3.461157,-2.161903,2.762150,-3.081994],[7.543720,-3.600396,-3.426044,4.570097,4.287462,7.005278,-4.545476],[6.735035,0.535728,-2.961208,-1.469734,0.756690,2.636610,-8.047373],[1.105182,4.777890,-9.685631,-7.583899,-5.082611,-2.798942,1.597251],[0.133756,6.766128,8.724490,-6.399488,5.793520,2.901634,5.465928],[-8.784008,0.187553,-6.356331,-7.609173,-8.854555,-7.191185,-7.057512],[-7.945814,-2.315760,4.405123,-0.341590,-3.137777,-5.833933,-4.838348],[7.569959,-0.292166,5.701862,9.170860,9.250790,-4.437581,-5.397996],[-6.221645,-9.823983,4.547116,-1.488308,-2.694745,6.216720,9.116315],[-0.982755,-4.893506,2.101292,2.210309,-0.586651,-2.742624,-0.314747],[-7.177092,4.118228,3.122995,2.458890,-8.032964,3.766695,9.592141],[-5.216081,-0.203596,-1.761483,-1.860151,-1.397577,-2.572331,-7.404110],[-1.233775,3.302482,-2.559719,2.183599,-7.674464,-5.552122,-6.903894]], dtype = "float32")#candidate|242|(15, 7)|const|float32
bop_243 = relay.floor_mod(uop_189.astype('float64'), relay.reshape(const_242.astype('float64'), relay.shape_of(uop_189))) # shape=(15, 7)
uop_246 = relay.atan(uop_231.astype('float32')) # shape=(15, 7)
uop_248 = relay.tan(uop_246.astype('float64')) # shape=(15, 7)
func_136_call = mod.get_global_var('func_136')
func_142_call = mutated_mod.get_global_var('func_142')
call_250 = relay.TupleGetItem(func_136_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(call_240.astype('float32'), [2,]), relay.reshape(call_240.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 6)
call_251 = relay.TupleGetItem(func_142_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(call_240.astype('float32'), [2,]), relay.reshape(call_240.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 6)
uop_252 = relay.sigmoid(bop_221.astype('float64')) # shape=(15, 7)
bop_254 = relay.add(uop_229.astype('uint32'), relay.reshape(bop_221.astype('uint32'), relay.shape_of(uop_229))) # shape=(15, 7)
uop_257 = relay.rsqrt(bop_254.astype('float32')) # shape=(15, 7)
var_259 = relay.var("var_259", dtype = "float64", shape = (15, 7))#candidate|259|(15, 7)|var|float64
bop_260 = relay.not_equal(uop_248.astype('bool'), relay.reshape(var_259.astype('bool'), relay.shape_of(uop_248))) # shape=(15, 7)
bop_263 = relay.floor_divide(uop_248.astype('float64'), relay.reshape(uop_246.astype('float64'), relay.shape_of(uop_248))) # shape=(15, 7)
bop_266 = relay.add(bop_263.astype('uint32'), relay.reshape(bop_260.astype('uint32'), relay.shape_of(bop_263))) # shape=(15, 7)
bop_269 = relay.greater_equal(bop_260.astype('bool'), relay.reshape(bop_209.astype('bool'), relay.shape_of(bop_260))) # shape=(15, 7)
func_136_call = mod.get_global_var('func_136')
func_142_call = mutated_mod.get_global_var('func_142')
call_272 = relay.TupleGetItem(func_136_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(call_250.astype('float32'), [2,]), relay.reshape(call_250.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 8)
call_273 = relay.TupleGetItem(func_142_call(relay.reshape(const_199.astype('float32'), []), relay.reshape(call_250.astype('float32'), [2,]), relay.reshape(call_250.astype('float32'), [2,]), relay.reshape(const_200.astype('float32'), [2,]), relay.reshape(var_201.astype('float32'), [8, 2]), ), 8)
uop_274 = relay.sinh(bop_260.astype('float32')) # shape=(15, 7)
var_276 = relay.var("var_276", dtype = "int64", shape = (15, 7))#candidate|276|(15, 7)|var|int64
bop_277 = relay.divide(bop_233.astype('float64'), relay.reshape(var_276.astype('float64'), relay.shape_of(bop_233))) # shape=(15, 7)
uop_280 = relay.atan(bop_233.astype('float64')) # shape=(15, 7)
bop_282 = relay.left_shift(uop_238.astype('uint64'), relay.reshape(bop_233.astype('uint64'), relay.shape_of(uop_238))) # shape=(15, 7)
bop_285 = relay.greater(uop_274.astype('bool'), relay.reshape(bop_263.astype('bool'), relay.shape_of(uop_274))) # shape=(15, 7)
uop_288 = relay.acos(bop_285.astype('float64')) # shape=(15, 7)
func_178_call = mod.get_global_var('func_178')
func_184_call = mutated_mod.get_global_var('func_184')
call_290 = relay.TupleGetItem(func_178_call(relay.reshape(call_226.astype('uint64'), [8, 3, 6]), relay.reshape(call_198.astype('float32'), [2,]), relay.reshape(call_272.astype('float32'), [16,]), relay.reshape(call_226.astype('float32'), [8, 3, 6]), ), 5)
call_291 = relay.TupleGetItem(func_184_call(relay.reshape(call_226.astype('uint64'), [8, 3, 6]), relay.reshape(call_198.astype('float32'), [2,]), relay.reshape(call_272.astype('float32'), [16,]), relay.reshape(call_226.astype('float32'), [8, 3, 6]), ), 5)
var_292 = relay.var("var_292", dtype = "float64", shape = (15, 7))#candidate|292|(15, 7)|var|float64
bop_293 = relay.logical_xor(uop_288.astype('int32'), relay.reshape(var_292.astype('int32'), relay.shape_of(uop_288))) # shape=(15, 7)
uop_296 = relay.acos(bop_293.astype('float32')) # shape=(15, 7)
var_298 = relay.var("var_298", dtype = "bool", shape = (15, 7))#candidate|298|(15, 7)|var|bool
bop_299 = relay.left_shift(bop_285.astype('uint16'), relay.reshape(var_298.astype('uint16'), relay.shape_of(bop_285))) # shape=(15, 7)
uop_302 = relay.tan(uop_288.astype('float64')) # shape=(15, 7)
bop_304 = relay.floor_mod(bop_218.astype('float32'), relay.reshape(const_217.astype('float32'), relay.shape_of(bop_218))) # shape=(15, 7)
bop_307 = relay.bitwise_or(bop_293.astype('int64'), relay.reshape(uop_296.astype('int64'), relay.shape_of(bop_293))) # shape=(15, 7)
uop_310 = relay.asin(uop_302.astype('float32')) # shape=(15, 7)
uop_312 = relay.cosh(uop_310.astype('float32')) # shape=(15, 7)
uop_314 = relay.acos(uop_288.astype('float64')) # shape=(15, 7)
output = relay.Tuple([bop_193,uop_196,call_198,const_199,const_200,var_201,uop_224,call_226,const_227,uop_236,call_240,bop_243,call_250,uop_252,uop_257,bop_266,bop_269,call_272,bop_277,uop_280,bop_282,call_290,bop_299,bop_304,bop_307,uop_312,uop_314,])
output2 = relay.Tuple([bop_193,uop_196,call_202,const_199,const_200,var_201,uop_224,call_228,const_227,uop_236,call_241,bop_243,call_251,uop_252,uop_257,bop_266,bop_269,call_273,bop_277,uop_280,bop_282,call_291,bop_299,bop_304,bop_307,uop_312,uop_314,])
func_316 = relay.Function([var_186,var_201,var_259,var_276,var_292,var_298,], output)
mod['func_316'] = func_316
mod = relay.transform.InferType()(mod)
var_317 = relay.var("var_317", dtype = "float32", shape = (15, 7))#candidate|317|(15, 7)|var|float32
var_318 = relay.var("var_318", dtype = "float32", shape = (16,))#candidate|318|(16,)|var|float32
var_319 = relay.var("var_319", dtype = "float64", shape = (15, 7))#candidate|319|(15, 7)|var|float64
var_320 = relay.var("var_320", dtype = "int64", shape = (15, 7))#candidate|320|(15, 7)|var|int64
var_321 = relay.var("var_321", dtype = "float64", shape = (15, 7))#candidate|321|(15, 7)|var|float64
var_322 = relay.var("var_322", dtype = "bool", shape = (15, 7))#candidate|322|(15, 7)|var|bool
output = func_316(var_317,var_318,var_319,var_320,var_321,var_322,)
func_323 = relay.Function([var_317,var_318,var_319,var_320,var_321,var_322,], output)
mutated_mod['func_323'] = func_323
mutated_mod = relay.transform.InferType()(mutated_mod)
var_325 = relay.var("var_325", dtype = "int16", shape = (5, 5, 5))#candidate|325|(5, 5, 5)|var|int16
var_326 = relay.var("var_326", dtype = "int16", shape = (5, 5, 5))#candidate|326|(5, 5, 5)|var|int16
bop_327 = relay.subtract(var_325.astype('int16'), relay.reshape(var_326.astype('int16'), relay.shape_of(var_325))) # shape=(5, 5, 5)
output = relay.Tuple([bop_327,])
output2 = relay.Tuple([bop_327,])
F = relay.Function([var_325,var_326,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_325,var_326,], output2)
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
input_325= np.array([[[1,4,-10,-5,5],[9,-9,-9,8,-7],[6,-3,3,-2,-5],[8,-10,10,-1,-2],[-10,10,-5,8,-8]],[[10,-10,-2,-4,-1],[-5,-4,5,-6,9],[-1,-10,-8,9,2],[10,3,4,2,-10],[-6,-4,-9,-4,-5]],[[4,2,8,2,-4],[5,4,-4,5,-8],[-9,-9,5,-3,-4],[4,4,-3,5,-10],[-10,5,-4,6,-4]],[[-4,9,5,-1,4],[7,-7,-3,-5,-8],[9,9,3,4,-5],[-2,-9,-2,6,5],[3,8,-4,10,1]],[[-2,7,-6,-9,10],[7,2,-4,4,-10],[-9,-10,-10,-8,-2],[10,9,9,6,6],[-1,9,-2,-8,2]]], dtype='int16')
module1.set_input('var_325', input_325)
input_326= np.array([[[-3,8,8,6,10],[-4,3,-8,1,-10],[1,1,-3,-2,9],[-8,2,6,3,5],[1,4,5,8,-5]],[[4,-10,-5,-7,-5],[-8,9,-6,-2,-8],[-2,8,-6,-7,10],[5,-7,10,7,6],[6,-4,6,1,-10]],[[-6,-1,-8,1,3],[7,-7,2,-6,-5],[-2,6,5,5,1],[4,-10,6,-1,-5],[-8,2,-7,-1,3]],[[7,6,-1,-7,-4],[4,8,4,-1,-10],[-1,9,6,-6,-10],[4,4,1,8,2],[10,3,-9,3,1]],[[-4,-4,4,10,-8],[10,-8,-9,-8,8],[-2,-5,6,9,3],[5,-2,10,-8,7],[1,-7,10,7,-4]]], dtype='int16')
module1.set_input('var_326', input_326)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_325, input_326, )
res3 = intrp3.evaluate()(input_325, input_326, )
res4 = intrp4.evaluate()(input_325, input_326, )
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
module5.set_input('var_325', input_325)
module5.set_input('var_326', input_326)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_325, input_326, )
res7 = intrp7.evaluate()(input_325, input_326, )
res8 = intrp8.evaluate()(input_325, input_326, )
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
module9.set_input('var_325', input_325)
module9.set_input('var_326', input_326)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_325, input_326, )
res11 = intrp11.evaluate()(input_325, input_326, )
res12 = intrp12.evaluate()(input_325, input_326, )
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
module13.set_input('var_325', input_325)
module13.set_input('var_326', input_326)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_325, input_326, )
res15 = intrp15.evaluate()(input_325, input_326, )
res16 = intrp16.evaluate()(input_325, input_326, )
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
module17.set_input('var_325', input_325)
module17.set_input('var_326', input_326)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_325, input_326, )
res19 = intrp19.evaluate()(input_325, input_326, )
res20 = intrp20.evaluate()(input_325, input_326, )
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
module21.set_input('var_325', input_325)
module21.set_input('var_326', input_326)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_325, input_326, )
res23 = intrp23.evaluate()(input_325, input_326, )
res24 = intrp24.evaluate()(input_325, input_326, )
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

'''47: TVMFuncCall
46: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
45: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
44: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
43: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
42: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
41: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
40: tvm::transform::Pass::operator()(tvm::IRModule) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
36: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
35: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
34: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
33: tvm::transform::Pass::operator()(tvm::IRModule) const
32: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
30: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
29: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
28: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
27: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
26: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
25: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
24: _ZN3tvm5relay9transform22Devic
23: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
22: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
21: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
20: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
18: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
17: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
15: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
14: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
11: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
9: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
8: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
7: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
6: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
5: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
3: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
2: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
1: tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
0: tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)

'''