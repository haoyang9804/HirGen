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
var_0 = relay.var("var_0", dtype = "float64", shape = (7, 9, 9))#candidate|0|(7, 9, 9)|var|float64
uop_1 = relay.sqrt(var_0.astype('float64')) # shape=(7, 9, 9)
output = uop_1
output2 = uop_1
func_3 = relay.Function([var_0,], output)
mod['func_3'] = func_3
mod = relay.transform.InferType()(mod)
mutated_mod['func_3'] = func_3
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4 = relay.var("var_4", dtype = "float64", shape = (7, 9, 9))#candidate|4|(7, 9, 9)|var|float64
func_3_call = mutated_mod.get_global_var('func_3')
call_5 = func_3_call(var_4)
output = call_5
func_6 = relay.Function([var_4], output)
mutated_mod['func_6'] = func_6
mutated_mod = relay.transform.InferType()(mutated_mod)
var_8 = relay.var("var_8", dtype = "float32", shape = (3,))#candidate|8|(3,)|var|float32
uop_9 = relay.log2(var_8.astype('float32')) # shape=(3,)
var_11 = relay.var("var_11", dtype = "float32", shape = (3,))#candidate|11|(3,)|var|float32
bop_12 = relay.maximum(uop_9.astype('uint16'), relay.reshape(var_11.astype('uint16'), relay.shape_of(uop_9))) # shape=(3,)
uop_15 = relay.tan(var_8.astype('float64')) # shape=(3,)
uop_17 = relay.sin(var_11.astype('float32')) # shape=(3,)
bop_19 = relay.not_equal(uop_9.astype('bool'), relay.reshape(var_11.astype('bool'), relay.shape_of(uop_9))) # shape=(3,)
uop_22 = relay.sinh(uop_9.astype('float32')) # shape=(3,)
var_24 = relay.var("var_24", dtype = "uint16", shape = (3,))#candidate|24|(3,)|var|uint16
bop_25 = relay.floor_mod(bop_12.astype('float32'), relay.reshape(var_24.astype('float32'), relay.shape_of(bop_12))) # shape=(3,)
func_3_call = mod.get_global_var('func_3')
func_6_call = mutated_mod.get_global_var('func_6')
var_29 = relay.var("var_29", dtype = "float64", shape = (567,))#candidate|29|(567,)|var|float64
call_28 = func_3_call(relay.reshape(var_29.astype('float64'), [7, 9, 9]))
call_30 = func_3_call(relay.reshape(var_29.astype('float64'), [7, 9, 9]))
uop_31 = relay.acosh(var_11.astype('float32')) # shape=(3,)
bop_33 = relay.maximum(uop_22.astype('uint16'), relay.reshape(uop_15.astype('uint16'), relay.shape_of(uop_22))) # shape=(3,)
bop_36 = relay.bitwise_or(bop_33.astype('int16'), relay.reshape(bop_19.astype('int16'), relay.shape_of(bop_33))) # shape=(3,)
var_39 = relay.var("var_39", dtype = "float32", shape = (3,))#candidate|39|(3,)|var|float32
bop_40 = relay.equal(uop_22.astype('bool'), relay.reshape(var_39.astype('bool'), relay.shape_of(uop_22))) # shape=(3,)
uop_43 = relay.acos(uop_9.astype('float64')) # shape=(3,)
uop_45 = relay.sin(uop_17.astype('float32')) # shape=(3,)
uop_47 = relay.asin(uop_45.astype('float32')) # shape=(3,)
uop_49 = relay.atan(uop_47.astype('float32')) # shape=(3,)
uop_51 = relay.sigmoid(uop_49.astype('float32')) # shape=(3,)
uop_53 = relay.sin(uop_51.astype('float64')) # shape=(3,)
uop_55 = relay.rsqrt(uop_51.astype('float64')) # shape=(3,)
uop_57 = relay.sinh(uop_55.astype('float32')) # shape=(3,)
uop_59 = relay.sinh(uop_57.astype('float64')) # shape=(3,)
bop_61 = relay.add(uop_59.astype('int8'), relay.reshape(uop_31.astype('int8'), relay.shape_of(uop_59))) # shape=(3,)
uop_64 = relay.tan(uop_59.astype('float64')) # shape=(3,)
uop_66 = relay.atan(uop_64.astype('float32')) # shape=(3,)
func_3_call = mod.get_global_var('func_3')
func_6_call = mutated_mod.get_global_var('func_6')
call_68 = func_3_call(relay.reshape(call_28.astype('float64'), [7, 9, 9]))
call_69 = func_3_call(relay.reshape(call_28.astype('float64'), [7, 9, 9]))
bop_70 = relay.bitwise_or(uop_66.astype('uint16'), relay.reshape(bop_61.astype('uint16'), relay.shape_of(uop_66))) # shape=(3,)
uop_73 = relay.sqrt(bop_70.astype('float32')) # shape=(3,)
bop_75 = relay.multiply(uop_73.astype('uint8'), relay.reshape(uop_47.astype('uint8'), relay.shape_of(uop_73))) # shape=(3,)
bop_78 = relay.add(bop_75.astype('uint32'), relay.reshape(uop_73.astype('uint32'), relay.shape_of(bop_75))) # shape=(3,)
uop_81 = relay.asin(uop_66.astype('float32')) # shape=(3,)
bop_83 = relay.less(bop_75.astype('bool'), relay.reshape(uop_22.astype('bool'), relay.shape_of(bop_75))) # shape=(3,)
uop_86 = relay.erf(uop_81.astype('float32')) # shape=(3,)
var_88 = relay.var("var_88", dtype = "bool", shape = (3,))#candidate|88|(3,)|var|bool
bop_89 = relay.floor_mod(bop_83.astype('float32'), relay.reshape(var_88.astype('float32'), relay.shape_of(bop_83))) # shape=(3,)
uop_92 = relay.sin(uop_86.astype('float64')) # shape=(3,)
bop_94 = relay.floor_mod(uop_92.astype('float64'), relay.reshape(bop_33.astype('float64'), relay.shape_of(uop_92))) # shape=(3,)
uop_97 = relay.sigmoid(uop_86.astype('float32')) # shape=(3,)
uop_99 = relay.acosh(uop_81.astype('float64')) # shape=(3,)
bop_101 = relay.maximum(uop_59.astype('uint64'), relay.reshape(uop_73.astype('uint64'), relay.shape_of(uop_59))) # shape=(3,)
uop_104 = relay.asin(uop_97.astype('float32')) # shape=(3,)
var_106 = relay.var("var_106", dtype = "float32", shape = (3,))#candidate|106|(3,)|var|float32
bop_107 = relay.maximum(uop_104.astype('uint32'), relay.reshape(var_106.astype('uint32'), relay.shape_of(uop_104))) # shape=(3,)
func_3_call = mod.get_global_var('func_3')
func_6_call = mutated_mod.get_global_var('func_6')
call_110 = func_3_call(relay.reshape(call_68.astype('float64'), [7, 9, 9]))
call_111 = func_3_call(relay.reshape(call_68.astype('float64'), [7, 9, 9]))
var_112 = relay.var("var_112", dtype = "float32", shape = (3,))#candidate|112|(3,)|var|float32
bop_113 = relay.logical_and(uop_97.astype('bool'), relay.reshape(var_112.astype('bool'), relay.shape_of(uop_97))) # shape=(3,)
uop_116 = relay.exp(uop_104.astype('float32')) # shape=(3,)
var_118 = relay.var("var_118", dtype = "float32", shape = (3,))#candidate|118|(3,)|var|float32
bop_119 = relay.bitwise_or(uop_86.astype('uint32'), relay.reshape(var_118.astype('uint32'), relay.shape_of(uop_86))) # shape=(3,)
bop_122 = relay.less_equal(bop_70.astype('bool'), relay.reshape(bop_94.astype('bool'), relay.shape_of(bop_70))) # shape=(3,)
bop_125 = relay.floor_mod(uop_116.astype('float32'), relay.reshape(uop_53.astype('float32'), relay.shape_of(uop_116))) # shape=(3,)
uop_128 = relay.rsqrt(bop_94.astype('float32')) # shape=(3,)
uop_130 = relay.tan(uop_104.astype('float32')) # shape=(3,)
uop_132 = relay.asin(bop_107.astype('float32')) # shape=(3,)
uop_134 = relay.sqrt(uop_104.astype('float64')) # shape=(3,)
uop_136 = relay.asin(uop_104.astype('float64')) # shape=(3,)
bop_138 = relay.mod(bop_125.astype('float32'), relay.reshape(bop_40.astype('float32'), relay.shape_of(bop_125))) # shape=(3,)
bop_141 = relay.logical_xor(uop_132.astype('uint8'), relay.reshape(bop_89.astype('uint8'), relay.shape_of(uop_132))) # shape=(3,)
bop_144 = relay.logical_or(bop_141.astype('bool'), relay.reshape(uop_130.astype('bool'), relay.shape_of(bop_141))) # shape=(3,)
bop_147 = relay.multiply(uop_130.astype('int32'), relay.reshape(uop_51.astype('int32'), relay.shape_of(uop_130))) # shape=(3,)
var_150 = relay.var("var_150", dtype = "float32", shape = (3,))#candidate|150|(3,)|var|float32
bop_151 = relay.less(uop_73.astype('bool'), relay.reshape(var_150.astype('bool'), relay.shape_of(uop_73))) # shape=(3,)
uop_154 = relay.sinh(uop_128.astype('float32')) # shape=(3,)
uop_156 = relay.log(uop_116.astype('float32')) # shape=(3,)
uop_158 = relay.sinh(bop_122.astype('float64')) # shape=(3,)
bop_160 = relay.mod(uop_156.astype('float64'), relay.reshape(bop_107.astype('float64'), relay.shape_of(uop_156))) # shape=(3,)
uop_163 = relay.acosh(uop_132.astype('float32')) # shape=(3,)
uop_165 = relay.log2(bop_160.astype('float64')) # shape=(3,)
output = relay.Tuple([bop_25,call_28,var_29,bop_36,uop_43,call_68,bop_78,uop_99,bop_101,call_110,bop_113,bop_119,uop_134,uop_136,bop_138,bop_144,bop_147,bop_151,uop_154,uop_158,uop_163,uop_165,])
output2 = relay.Tuple([bop_25,call_30,var_29,bop_36,uop_43,call_69,bop_78,uop_99,bop_101,call_111,bop_113,bop_119,uop_134,uop_136,bop_138,bop_144,bop_147,bop_151,uop_154,uop_158,uop_163,uop_165,])
func_167 = relay.Function([var_8,var_11,var_24,var_29,var_39,var_88,var_106,var_112,var_118,var_150,], output)
mod['func_167'] = func_167
mod = relay.transform.InferType()(mod)
mutated_mod['func_167'] = func_167
mutated_mod = relay.transform.InferType()(mutated_mod)
func_167_call = mutated_mod.get_global_var('func_167')
var_169 = relay.var("var_169", dtype = "float32", shape = (3,))#candidate|169|(3,)|var|float32
var_170 = relay.var("var_170", dtype = "float32", shape = (3,))#candidate|170|(3,)|var|float32
var_171 = relay.var("var_171", dtype = "uint16", shape = (3,))#candidate|171|(3,)|var|uint16
var_172 = relay.var("var_172", dtype = "float64", shape = (567,))#candidate|172|(567,)|var|float64
var_173 = relay.var("var_173", dtype = "float32", shape = (3,))#candidate|173|(3,)|var|float32
var_174 = relay.var("var_174", dtype = "bool", shape = (3,))#candidate|174|(3,)|var|bool
var_175 = relay.var("var_175", dtype = "float32", shape = (3,))#candidate|175|(3,)|var|float32
var_176 = relay.var("var_176", dtype = "float32", shape = (3,))#candidate|176|(3,)|var|float32
var_177 = relay.var("var_177", dtype = "float32", shape = (3,))#candidate|177|(3,)|var|float32
var_178 = relay.var("var_178", dtype = "float32", shape = (3,))#candidate|178|(3,)|var|float32
call_168 = func_167_call(var_169,var_170,var_171,var_172,var_173,var_174,var_175,var_176,var_177,var_178,)
output = call_168
func_179 = relay.Function([var_169,var_170,var_171,var_172,var_173,var_174,var_175,var_176,var_177,var_178,], output)
mutated_mod['func_179'] = func_179
mutated_mod = relay.transform.InferType()(mutated_mod)
var_181 = relay.var("var_181", dtype = "float64", shape = (15, 15, 16))#candidate|181|(15, 15, 16)|var|float64
uop_182 = relay.cosh(var_181.astype('float64')) # shape=(15, 15, 16)
uop_184 = relay.rsqrt(uop_182.astype('float64')) # shape=(15, 15, 16)
bop_186 = relay.power(uop_184.astype('float64'), relay.reshape(var_181.astype('float64'), relay.shape_of(uop_184))) # shape=(15, 15, 16)
bop_189 = relay.maximum(uop_184.astype('uint32'), relay.reshape(bop_186.astype('uint32'), relay.shape_of(uop_184))) # shape=(15, 15, 16)
bop_192 = relay.bitwise_and(uop_182.astype('int64'), relay.reshape(bop_189.astype('int64'), relay.shape_of(uop_182))) # shape=(15, 15, 16)
uop_195 = relay.cos(var_181.astype('float64')) # shape=(15, 15, 16)
uop_197 = relay.log2(uop_182.astype('float32')) # shape=(15, 15, 16)
uop_199 = relay.cos(uop_182.astype('float64')) # shape=(15, 15, 16)
bop_201 = relay.bitwise_and(uop_184.astype('uint64'), relay.reshape(bop_186.astype('uint64'), relay.shape_of(uop_184))) # shape=(15, 15, 16)
uop_204 = relay.erf(uop_195.astype('float32')) # shape=(15, 15, 16)
bop_206 = relay.equal(var_181.astype('bool'), relay.reshape(bop_189.astype('bool'), relay.shape_of(var_181))) # shape=(15, 15, 16)
bop_209 = relay.bitwise_or(uop_182.astype('int64'), relay.reshape(bop_192.astype('int64'), relay.shape_of(uop_182))) # shape=(15, 15, 16)
uop_212 = relay.log2(uop_182.astype('float64')) # shape=(15, 15, 16)
var_214 = relay.var("var_214", dtype = "float32", shape = (15, 15, 16))#candidate|214|(15, 15, 16)|var|float32
bop_215 = relay.greater_equal(uop_204.astype('bool'), relay.reshape(var_214.astype('bool'), relay.shape_of(uop_204))) # shape=(15, 15, 16)
uop_218 = relay.sigmoid(uop_195.astype('float64')) # shape=(15, 15, 16)
uop_220 = relay.cosh(uop_195.astype('float64')) # shape=(15, 15, 16)
bop_222 = relay.bitwise_or(uop_218.astype('int32'), relay.reshape(uop_199.astype('int32'), relay.shape_of(uop_218))) # shape=(15, 15, 16)
var_225 = relay.var("var_225", dtype = "float64", shape = (15, 15, 16))#candidate|225|(15, 15, 16)|var|float64
bop_226 = relay.logical_or(uop_195.astype('bool'), relay.reshape(var_225.astype('bool'), relay.shape_of(uop_195))) # shape=(15, 15, 16)
bop_229 = relay.multiply(bop_201.astype('float32'), relay.reshape(bop_226.astype('float32'), relay.shape_of(bop_201))) # shape=(15, 15, 16)
uop_232 = relay.asin(uop_182.astype('float32')) # shape=(15, 15, 16)
bop_234 = relay.bitwise_and(bop_222.astype('int32'), relay.reshape(var_225.astype('int32'), relay.shape_of(bop_222))) # shape=(15, 15, 16)
uop_237 = relay.sigmoid(bop_215.astype('float64')) # shape=(15, 15, 16)
uop_239 = relay.sigmoid(uop_237.astype('float32')) # shape=(15, 15, 16)
bop_241 = relay.power(uop_239.astype('float64'), relay.reshape(uop_199.astype('float64'), relay.shape_of(uop_239))) # shape=(15, 15, 16)
bop_244 = relay.add(bop_241.astype('int8'), relay.reshape(uop_218.astype('int8'), relay.shape_of(bop_241))) # shape=(15, 15, 16)
bop_247 = relay.divide(uop_237.astype('float32'), relay.reshape(uop_232.astype('float32'), relay.shape_of(uop_237))) # shape=(15, 15, 16)
uop_250 = relay.log2(bop_226.astype('float64')) # shape=(15, 15, 16)
uop_252 = relay.atan(bop_241.astype('float32')) # shape=(15, 15, 16)
bop_254 = relay.add(uop_239.astype('float64'), relay.reshape(bop_206.astype('float64'), relay.shape_of(uop_239))) # shape=(15, 15, 16)
uop_257 = relay.log10(uop_252.astype('float64')) # shape=(15, 15, 16)
bop_259 = relay.floor_divide(bop_209.astype('float32'), relay.reshape(uop_218.astype('float32'), relay.shape_of(bop_209))) # shape=(15, 15, 16)
uop_262 = relay.log10(uop_252.astype('float64')) # shape=(15, 15, 16)
bop_264 = relay.equal(uop_252.astype('bool'), relay.reshape(uop_204.astype('bool'), relay.shape_of(uop_252))) # shape=(15, 15, 16)
uop_267 = relay.tan(uop_262.astype('float64')) # shape=(15, 15, 16)
bop_269 = relay.less(uop_267.astype('bool'), relay.reshape(bop_201.astype('bool'), relay.shape_of(uop_267))) # shape=(15, 15, 16)
bop_272 = relay.bitwise_or(uop_257.astype('int64'), relay.reshape(uop_239.astype('int64'), relay.shape_of(uop_257))) # shape=(15, 15, 16)
uop_275 = relay.sigmoid(bop_269.astype('float32')) # shape=(15, 15, 16)
bop_277 = relay.right_shift(bop_269.astype('uint32'), relay.reshape(bop_222.astype('uint32'), relay.shape_of(bop_269))) # shape=(15, 15, 16)
output = relay.Tuple([uop_197,uop_212,uop_220,bop_229,bop_234,bop_244,bop_247,uop_250,bop_254,bop_259,bop_264,bop_272,uop_275,bop_277,])
output2 = relay.Tuple([uop_197,uop_212,uop_220,bop_229,bop_234,bop_244,bop_247,uop_250,bop_254,bop_259,bop_264,bop_272,uop_275,bop_277,])
func_280 = relay.Function([var_181,var_214,var_225,], output)
mod['func_280'] = func_280
mod = relay.transform.InferType()(mod)
var_281 = relay.var("var_281", dtype = "float64", shape = (15, 15, 16))#candidate|281|(15, 15, 16)|var|float64
var_282 = relay.var("var_282", dtype = "float32", shape = (15, 15, 16))#candidate|282|(15, 15, 16)|var|float32
var_283 = relay.var("var_283", dtype = "float64", shape = (15, 15, 16))#candidate|283|(15, 15, 16)|var|float64
output = func_280(var_281,var_282,var_283,)
func_284 = relay.Function([var_281,var_282,var_283,], output)
mutated_mod['func_284'] = func_284
mutated_mod = relay.transform.InferType()(mutated_mod)
var_286 = relay.var("var_286", dtype = "float64", shape = ())#candidate|286|()|var|float64
var_287 = relay.var("var_287", dtype = "float64", shape = ())#candidate|287|()|var|float64
bop_288 = relay.multiply(var_286.astype('float64'), var_287.astype('float64')) # shape=()
bop_291 = relay.divide(bop_288.astype('float32'), var_286.astype('float32')) # shape=()
output = relay.Tuple([bop_291,])
output2 = relay.Tuple([bop_291,])
F = relay.Function([var_286,var_287,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_286,var_287,], output2)
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
input_286= np.array(-8.148270, dtype='float64')
module1.set_input('var_286', input_286)
input_287= np.array(-4.120578, dtype='float64')
module1.set_input('var_287', input_287)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_286, input_287, )
res3 = intrp3.evaluate()(input_286, input_287, )
res4 = intrp4.evaluate()(input_286, input_287, )
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
module5.set_input('var_286', input_286)
module5.set_input('var_287', input_287)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_286, input_287, )
res7 = intrp7.evaluate()(input_286, input_287, )
res8 = intrp8.evaluate()(input_286, input_287, )
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
module9.set_input('var_286', input_286)
module9.set_input('var_287', input_287)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_286, input_287, )
res11 = intrp11.evaluate()(input_286, input_287, )
res12 = intrp12.evaluate()(input_286, input_287, )
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
module13.set_input('var_286', input_286)
module13.set_input('var_287', input_287)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_286, input_287, )
res15 = intrp15.evaluate()(input_286, input_287, )
res16 = intrp16.evaluate()(input_286, input_287, )
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
module17.set_input('var_286', input_286)
module17.set_input('var_287', input_287)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_286, input_287, )
res19 = intrp19.evaluate()(input_286, input_287, )
res20 = intrp20.evaluate()(input_286, input_287, )
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
module21.set_input('var_286', input_286)
module21.set_input('var_287', input_287)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_286, input_287, )
res23 = intrp23.evaluate()(input_286, input_287, )
res24 = intrp24.evaluate()(input_286, input_287, )
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