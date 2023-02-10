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
var_0 = relay.var("var_0", dtype = "float64", shape = (11, 7, 2))#candidate|0|(11, 7, 2)|var|float64
uop_1 = relay.rsqrt(var_0.astype('float64')) # shape=(11, 7, 2)
bop_3 = relay.multiply(var_0.astype('int8'), relay.reshape(uop_1.astype('int8'), relay.shape_of(var_0))) # shape=(11, 7, 2)
output = bop_3
output2 = bop_3
func_6 = relay.Function([var_0,], output)
mod['func_6'] = func_6
mod = relay.transform.InferType()(mod)
mutated_mod['func_6'] = func_6
mutated_mod = relay.transform.InferType()(mutated_mod)
var_7 = relay.var("var_7", dtype = "float64", shape = (11, 7, 2))#candidate|7|(11, 7, 2)|var|float64
func_6_call = mutated_mod.get_global_var('func_6')
call_8 = func_6_call(var_7)
output = call_8
func_9 = relay.Function([var_7], output)
mutated_mod['func_9'] = func_9
mutated_mod = relay.transform.InferType()(mutated_mod)
var_11 = relay.var("var_11", dtype = "float64", shape = (14, 4))#candidate|11|(14, 4)|var|float64
var_12 = relay.var("var_12", dtype = "float64", shape = (14, 4))#candidate|12|(14, 4)|var|float64
bop_13 = relay.mod(var_11.astype('float64'), relay.reshape(var_12.astype('float64'), relay.shape_of(var_11))) # shape=(14, 4)
bop_16 = relay.logical_xor(var_12.astype('int8'), relay.reshape(bop_13.astype('int8'), relay.shape_of(var_12))) # shape=(14, 4)
uop_19 = relay.asinh(bop_16.astype('float32')) # shape=(14, 4)
const_21 = relay.const([[-1.929922,6.601715,3.725286,-6.939798],[-2.660722,-6.962097,-4.182159,-6.794723],[3.315060,1.037744,-6.365711,0.123687],[3.455152,8.129619,3.709132,-4.017022],[-1.767958,2.535023,-4.043585,3.708562],[-4.507076,8.628555,3.489072,-0.928356],[6.156444,-2.396872,-4.929853,6.730874],[-3.447440,-9.499530,3.824602,5.000035],[9.265170,9.928800,-4.357355,-2.224292],[-4.981912,6.480034,2.526976,-4.371330],[-3.816155,-3.978019,-9.550800,-9.740852],[-6.988843,5.845640,-1.023877,5.020245],[3.182428,3.163751,-9.131730,-1.211862],[4.959433,-0.921244,6.955129,8.675929]], dtype = "float32")#candidate|21|(14, 4)|const|float32
bop_22 = relay.greater(uop_19.astype('bool'), relay.reshape(const_21.astype('bool'), relay.shape_of(uop_19))) # shape=(14, 4)
uop_25 = relay.log(bop_22.astype('float64')) # shape=(14, 4)
output = relay.Tuple([uop_25,])
output2 = relay.Tuple([uop_25,])
func_27 = relay.Function([var_11,var_12,], output)
mod['func_27'] = func_27
mod = relay.transform.InferType()(mod)
var_28 = relay.var("var_28", dtype = "float64", shape = (14, 4))#candidate|28|(14, 4)|var|float64
var_29 = relay.var("var_29", dtype = "float64", shape = (14, 4))#candidate|29|(14, 4)|var|float64
output = func_27(var_28,var_29,)
func_30 = relay.Function([var_28,var_29,], output)
mutated_mod['func_30'] = func_30
mutated_mod = relay.transform.InferType()(mutated_mod)
var_32 = relay.var("var_32", dtype = "float64", shape = ())#candidate|32|()|var|float64
var_33 = relay.var("var_33", dtype = "float64", shape = ())#candidate|33|()|var|float64
bop_34 = relay.less(var_32.astype('bool'), var_33.astype('bool')) # shape=()
var_37 = relay.var("var_37", dtype = "float64", shape = (5, 14))#candidate|37|(5, 14)|var|float64
bop_38 = relay.add(var_33.astype('int32'), var_37.astype('int32')) # shape=(5, 14)
uop_41 = relay.sinh(var_32.astype('float32')) # shape=()
bop_43 = relay.divide(var_37.astype('float64'), uop_41.astype('float64')) # shape=(5, 14)
bop_46 = relay.mod(bop_43.astype('float32'), uop_41.astype('float32')) # shape=(5, 14)
bop_49 = relay.multiply(bop_46.astype('int64'), relay.reshape(var_37.astype('int64'), relay.shape_of(bop_46))) # shape=(5, 14)
func_6_call = mod.get_global_var('func_6')
func_9_call = mutated_mod.get_global_var('func_9')
var_53 = relay.var("var_53", dtype = "float64", shape = (154,))#candidate|53|(154,)|var|float64
call_52 = func_6_call(relay.reshape(var_53.astype('float64'), [11, 7, 2]))
call_54 = func_6_call(relay.reshape(var_53.astype('float64'), [11, 7, 2]))
uop_55 = relay.asinh(uop_41.astype('float64')) # shape=()
bop_57 = relay.multiply(uop_41.astype('int32'), var_32.astype('int32')) # shape=()
const_60 = relay.const([-0.997243,3.852009,4.145698,-0.639328,2.215548,-2.996155,-6.442824,0.257896,8.227889,2.004304,-5.615719,0.206530,-2.983078,-2.322107,-6.578499,-5.549222,-2.737801,-1.438547,2.184030,4.558503,1.538312,5.588424,3.586535,8.632290,-4.915852,3.878424,-1.074851,5.958108,1.804036,5.349614,0.012865,-4.278381,8.818330,4.114351,0.943316,-3.889379,-1.565505,-3.433652,-8.826125,-5.658268,-3.483080,4.084022,6.729503,0.376096,-0.489271,-5.006644,8.103406,2.984426,4.999805,6.068846,6.279742,5.041592,-0.935626,-5.999269,-5.439111,0.580484,-0.981318,2.027265,9.151025,-5.430756,9.607102,4.381770,0.580125,-9.333403,2.956613,1.669826,9.572443,1.951498,-7.450763,-3.007258,-4.863622,-1.023698,-9.178671,-9.415844,-7.268558,5.891024,3.630443,-5.619392,-3.971729,-1.152500,2.273557,-7.058512,-6.159544,-7.788584,5.278355,4.497605,-2.213054,-9.477400,-0.144423,6.980070,0.381247,-0.104346,3.249445,3.946969,8.996183,-5.070675,7.549209,4.957904,-3.006203,7.626384,-6.687538,7.403025,-5.716323,3.215530,3.441963,5.497935,-4.474289,8.519179,2.611372,3.770314,-4.536091,0.351571,-6.791862,-3.013578,-6.983657,4.088819,-7.139025,5.241067,-1.067044,2.221516,9.126576,1.917949,1.431476,-6.649870,4.880560,6.717582,5.104774,-1.778814,-9.887769,-4.485965,-4.324647,3.165866,-2.473132,3.241451,-5.190183,-0.059901,4.855736,4.945564,3.526673,5.562870,4.080429,-0.531373,8.153925,-1.765566,8.621655,-0.684620,-8.051873,5.379903,5.698703,-5.440030,1.700166,1.475073,-5.082715,0.938680], dtype = "float64")#candidate|60|(154,)|const|float64
bop_61 = relay.not_equal(var_53.astype('bool'), relay.reshape(const_60.astype('bool'), relay.shape_of(var_53))) # shape=(154,)
bop_64 = relay.equal(uop_55.astype('bool'), bop_34.astype('bool')) # shape=()
bop_67 = relay.multiply(bop_64.astype('int16'), var_53.astype('int16')) # shape=(154,)
uop_70 = relay.log2(bop_67.astype('float64')) # shape=(154,)
uop_72 = relay.cos(uop_70.astype('float32')) # shape=(154,)
output = relay.Tuple([bop_38,bop_49,call_52,bop_57,bop_61,uop_72,])
output2 = relay.Tuple([bop_38,bop_49,call_54,bop_57,bop_61,uop_72,])
func_74 = relay.Function([var_32,var_33,var_37,var_53,], output)
mod['func_74'] = func_74
mod = relay.transform.InferType()(mod)
var_75 = relay.var("var_75", dtype = "float64", shape = ())#candidate|75|()|var|float64
var_76 = relay.var("var_76", dtype = "float64", shape = ())#candidate|76|()|var|float64
var_77 = relay.var("var_77", dtype = "float64", shape = (5, 14))#candidate|77|(5, 14)|var|float64
var_78 = relay.var("var_78", dtype = "float64", shape = (154,))#candidate|78|(154,)|var|float64
output = func_74(var_75,var_76,var_77,var_78,)
func_79 = relay.Function([var_75,var_76,var_77,var_78,], output)
mutated_mod['func_79'] = func_79
mutated_mod = relay.transform.InferType()(mutated_mod)
const_81 = relay.const(4.549021, dtype = "float32")#candidate|81|()|const|float32
uop_82 = relay.sqrt(const_81.astype('float32')) # shape=()
uop_84 = relay.tan(const_81.astype('float32')) # shape=()
bop_86 = relay.add(uop_82.astype('int64'), uop_84.astype('int64')) # shape=()
uop_89 = relay.rsqrt(uop_84.astype('float32')) # shape=()
output = relay.Tuple([bop_86,uop_89,])
output2 = relay.Tuple([bop_86,uop_89,])
func_91 = relay.Function([], output)
mod['func_91'] = func_91
mod = relay.transform.InferType()(mod)
mutated_mod['func_91'] = func_91
mutated_mod = relay.transform.InferType()(mutated_mod)
func_91_call = mutated_mod.get_global_var('func_91')
call_92 = func_91_call()
output = call_92
func_93 = relay.Function([], output)
mutated_mod['func_93'] = func_93
mutated_mod = relay.transform.InferType()(mutated_mod)
var_94 = relay.var("var_94", dtype = "uint16", shape = (4, 13))#candidate|94|(4, 13)|var|uint16
var_95 = relay.var("var_95", dtype = "uint16", shape = (4, 13))#candidate|95|(4, 13)|var|uint16
bop_96 = relay.less_equal(var_94.astype('bool'), relay.reshape(var_95.astype('bool'), relay.shape_of(var_94))) # shape=(4, 13)
bop_99 = relay.floor_mod(bop_96.astype('float32'), relay.reshape(var_94.astype('float32'), relay.shape_of(bop_96))) # shape=(4, 13)
var_102 = relay.var("var_102", dtype = "bool", shape = (4, 13))#candidate|102|(4, 13)|var|bool
bop_103 = relay.mod(bop_96.astype('float64'), relay.reshape(var_102.astype('float64'), relay.shape_of(bop_96))) # shape=(4, 13)
bop_106 = relay.bitwise_xor(var_95.astype('int64'), relay.reshape(var_102.astype('int64'), relay.shape_of(var_95))) # shape=(4, 13)
uop_109 = relay.sigmoid(var_95.astype('float64')) # shape=(4, 13)
bop_111 = relay.right_shift(uop_109.astype('uint32'), relay.reshape(bop_96.astype('uint32'), relay.shape_of(uop_109))) # shape=(4, 13)
uop_114 = relay.acosh(bop_111.astype('float32')) # shape=(4, 13)
uop_116 = relay.log(bop_96.astype('float32')) # shape=(4, 13)
bop_118 = relay.not_equal(uop_114.astype('bool'), relay.reshape(var_95.astype('bool'), relay.shape_of(uop_114))) # shape=(4, 13)
bop_121 = relay.equal(bop_118.astype('bool'), relay.reshape(bop_96.astype('bool'), relay.shape_of(bop_118))) # shape=(4, 13)
uop_124 = relay.cosh(bop_121.astype('float64')) # shape=(4, 13)
uop_126 = relay.sin(uop_124.astype('float64')) # shape=(4, 13)
bop_128 = relay.logical_and(uop_126.astype('bool'), relay.reshape(var_95.astype('bool'), relay.shape_of(uop_126))) # shape=(4, 13)
output = relay.Tuple([bop_99,bop_103,bop_106,uop_116,bop_128,])
output2 = relay.Tuple([bop_99,bop_103,bop_106,uop_116,bop_128,])
func_131 = relay.Function([var_94,var_95,var_102,], output)
mod['func_131'] = func_131
mod = relay.transform.InferType()(mod)
var_132 = relay.var("var_132", dtype = "uint16", shape = (4, 13))#candidate|132|(4, 13)|var|uint16
var_133 = relay.var("var_133", dtype = "uint16", shape = (4, 13))#candidate|133|(4, 13)|var|uint16
var_134 = relay.var("var_134", dtype = "bool", shape = (4, 13))#candidate|134|(4, 13)|var|bool
output = func_131(var_132,var_133,var_134,)
func_135 = relay.Function([var_132,var_133,var_134,], output)
mutated_mod['func_135'] = func_135
mutated_mod = relay.transform.InferType()(mutated_mod)
var_137 = relay.var("var_137", dtype = "uint32", shape = (3, 8))#candidate|137|(3, 8)|var|uint32
var_138 = relay.var("var_138", dtype = "uint32", shape = (3, 8))#candidate|138|(3, 8)|var|uint32
bop_139 = relay.greater_equal(var_137.astype('bool'), relay.reshape(var_138.astype('bool'), relay.shape_of(var_137))) # shape=(3, 8)
var_142 = relay.var("var_142", dtype = "uint32", shape = (3, 8))#candidate|142|(3, 8)|var|uint32
bop_143 = relay.less_equal(var_137.astype('bool'), relay.reshape(var_142.astype('bool'), relay.shape_of(var_137))) # shape=(3, 8)
bop_146 = relay.right_shift(bop_139.astype('uint64'), relay.reshape(var_142.astype('uint64'), relay.shape_of(bop_139))) # shape=(3, 8)
bop_149 = relay.logical_or(var_138.astype('bool'), relay.reshape(var_137.astype('bool'), relay.shape_of(var_138))) # shape=(3, 8)
uop_152 = relay.acosh(var_142.astype('float64')) # shape=(3, 8)
output = relay.Tuple([bop_143,bop_146,bop_149,uop_152,])
output2 = relay.Tuple([bop_143,bop_146,bop_149,uop_152,])
func_154 = relay.Function([var_137,var_138,var_142,], output)
mod['func_154'] = func_154
mod = relay.transform.InferType()(mod)
var_155 = relay.var("var_155", dtype = "uint32", shape = (3, 8))#candidate|155|(3, 8)|var|uint32
var_156 = relay.var("var_156", dtype = "uint32", shape = (3, 8))#candidate|156|(3, 8)|var|uint32
var_157 = relay.var("var_157", dtype = "uint32", shape = (3, 8))#candidate|157|(3, 8)|var|uint32
output = func_154(var_155,var_156,var_157,)
func_158 = relay.Function([var_155,var_156,var_157,], output)
mutated_mod['func_158'] = func_158
mutated_mod = relay.transform.InferType()(mutated_mod)
var_160 = relay.var("var_160", dtype = "float64", shape = (12, 12))#candidate|160|(12, 12)|var|float64
uop_161 = relay.sqrt(var_160.astype('float64')) # shape=(12, 12)
var_163 = relay.var("var_163", dtype = "float64", shape = (12, 12))#candidate|163|(12, 12)|var|float64
bop_164 = relay.logical_or(uop_161.astype('bool'), relay.reshape(var_163.astype('bool'), relay.shape_of(uop_161))) # shape=(12, 12)
uop_167 = relay.atan(bop_164.astype('float64')) # shape=(12, 12)
bop_169 = relay.minimum(var_160.astype('int8'), relay.reshape(uop_161.astype('int8'), relay.shape_of(var_160))) # shape=(12, 12)
uop_172 = relay.atan(bop_169.astype('float32')) # shape=(12, 12)
uop_174 = relay.asinh(var_160.astype('float32')) # shape=(12, 12)
bop_176 = relay.greater_equal(uop_167.astype('bool'), relay.reshape(var_163.astype('bool'), relay.shape_of(uop_167))) # shape=(12, 12)
uop_179 = relay.tan(bop_164.astype('float32')) # shape=(12, 12)
uop_181 = relay.erf(bop_164.astype('float64')) # shape=(12, 12)
uop_183 = relay.exp(uop_179.astype('float64')) # shape=(12, 12)
uop_185 = relay.tan(uop_183.astype('float32')) # shape=(12, 12)
func_6_call = mod.get_global_var('func_6')
func_9_call = mutated_mod.get_global_var('func_9')
var_188 = relay.var("var_188", dtype = "float64", shape = (154,))#candidate|188|(154,)|var|float64
call_187 = func_6_call(relay.reshape(var_188.astype('float64'), [11, 7, 2]))
call_189 = func_6_call(relay.reshape(var_188.astype('float64'), [11, 7, 2]))
bop_190 = relay.greater_equal(uop_179.astype('bool'), relay.reshape(uop_183.astype('bool'), relay.shape_of(uop_179))) # shape=(12, 12)
uop_193 = relay.rsqrt(uop_185.astype('float32')) # shape=(12, 12)
uop_195 = relay.sigmoid(uop_193.astype('float32')) # shape=(12, 12)
var_197 = relay.var("var_197", dtype = "float32", shape = (12, 12))#candidate|197|(12, 12)|var|float32
bop_198 = relay.mod(uop_185.astype('float32'), relay.reshape(var_197.astype('float32'), relay.shape_of(uop_185))) # shape=(12, 12)
bop_201 = relay.add(uop_193.astype('int8'), relay.reshape(bop_190.astype('int8'), relay.shape_of(uop_193))) # shape=(12, 12)
var_204 = relay.var("var_204", dtype = "float32", shape = (12, 12))#candidate|204|(12, 12)|var|float32
bop_205 = relay.floor_mod(uop_195.astype('float64'), relay.reshape(var_204.astype('float64'), relay.shape_of(uop_195))) # shape=(12, 12)
uop_208 = relay.rsqrt(bop_201.astype('float64')) # shape=(12, 12)
uop_210 = relay.rsqrt(bop_201.astype('float64')) # shape=(12, 12)
uop_212 = relay.asin(uop_185.astype('float32')) # shape=(12, 12)
uop_214 = relay.sqrt(bop_176.astype('float32')) # shape=(12, 12)
bop_216 = relay.multiply(bop_205.astype('uint64'), relay.reshape(uop_193.astype('uint64'), relay.shape_of(bop_205))) # shape=(12, 12)
const_219 = relay.const([[-3.718536,-8.180576,-3.904832,-4.048409,-3.672721,-0.291054,6.954746,-9.615851,0.371194,-8.625460,9.560485,2.062724],[-6.310498,-8.563167,-6.011071,2.708809,2.284401,-0.271404,-5.530597,-1.401760,-1.779586,2.648644,-8.200911,3.920318],[4.397302,7.816023,-7.237638,-2.448126,7.403515,-3.195786,-6.967267,-3.133011,2.153378,2.518582,2.322923,-3.433421],[2.567648,-6.048934,-4.483837,9.720550,6.716214,-2.944420,8.349586,9.979274,5.051861,-8.963045,6.703192,6.529107],[2.195859,-7.095991,-7.575048,0.511727,-7.328275,2.655613,6.370639,3.141288,-7.256126,-1.315313,-3.212272,-3.561489],[-9.251793,6.520233,-9.272397,-5.276468,-2.138759,-7.178935,-8.740263,-4.376878,3.845665,0.261072,-1.002016,-0.202858],[-4.676580,-5.858658,-7.149401,9.882893,9.750766,-5.994694,9.997807,1.241350,9.211145,4.505154,0.480569,-0.920996],[5.329105,9.572869,-4.234904,-6.532641,5.102986,6.485256,-6.177857,-0.153075,-3.453240,3.229826,9.104262,9.602859],[-6.687493,-8.088567,-0.734848,1.236613,-0.387521,2.186452,-6.919031,-5.594340,-2.905327,2.949996,-9.616535,-1.890900],[-4.401711,-3.461691,-4.544382,-2.021241,-2.930967,5.607087,5.607978,1.647464,9.244694,9.689074,-7.298087,0.091185],[5.583730,0.784599,-5.412082,-7.776545,1.413464,0.487414,-8.292352,3.588117,-9.912974,9.532503,9.348564,-1.228710],[1.776219,-6.361203,9.160965,-3.316976,-5.903694,-3.670001,4.488406,2.157242,2.272052,8.108912,-4.451907,9.078136]], dtype = "float32")#candidate|219|(12, 12)|const|float32
bop_220 = relay.right_shift(uop_195.astype('uint8'), relay.reshape(const_219.astype('uint8'), relay.shape_of(uop_195))) # shape=(12, 12)
bop_223 = relay.mod(uop_210.astype('float64'), relay.reshape(var_204.astype('float64'), relay.shape_of(uop_210))) # shape=(12, 12)
var_226 = relay.var("var_226", dtype = "float32", shape = (12, 12))#candidate|226|(12, 12)|var|float32
bop_227 = relay.subtract(uop_212.astype('int8'), relay.reshape(var_226.astype('int8'), relay.shape_of(uop_212))) # shape=(12, 12)
uop_230 = relay.acos(bop_205.astype('float64')) # shape=(12, 12)
bop_232 = relay.add(uop_230.astype('uint8'), relay.reshape(uop_172.astype('uint8'), relay.shape_of(uop_230))) # shape=(12, 12)
uop_235 = relay.log2(uop_230.astype('float64')) # shape=(12, 12)
var_237 = relay.var("var_237", dtype = "float64", shape = (12, 12))#candidate|237|(12, 12)|var|float64
bop_238 = relay.equal(uop_230.astype('bool'), relay.reshape(var_237.astype('bool'), relay.shape_of(uop_230))) # shape=(12, 12)
uop_241 = relay.asinh(uop_235.astype('float64')) # shape=(12, 12)
bop_243 = relay.right_shift(bop_232.astype('uint64'), relay.reshape(uop_212.astype('uint64'), relay.shape_of(bop_232))) # shape=(12, 12)
uop_246 = relay.exp(uop_235.astype('float64')) # shape=(12, 12)
var_248 = relay.var("var_248", dtype = "float64", shape = (12, 12))#candidate|248|(12, 12)|var|float64
bop_249 = relay.less_equal(uop_246.astype('bool'), relay.reshape(var_248.astype('bool'), relay.shape_of(uop_246))) # shape=(12, 12)
bop_252 = relay.multiply(uop_185.astype('int64'), relay.reshape(var_237.astype('int64'), relay.shape_of(uop_185))) # shape=(12, 12)
uop_255 = relay.cosh(bop_249.astype('float64')) # shape=(12, 12)
const_257 = relay.const([[-1.782879,3.066515,-4.386415,-3.520278,-8.019095,-1.167826,-3.980299,-4.325317,2.755423,-1.658830,-2.231348,8.175193],[-4.695196,-8.885283,4.618775,2.188062,7.757407,8.364610,3.797123,6.832476,9.079330,2.057719,-8.445849,7.872333],[0.783889,1.660976,2.133628,-2.676729,-2.604367,-8.442811,2.565027,4.042774,-1.761453,-9.376238,6.408823,9.146958],[8.342420,-1.084354,9.898004,-2.860855,-3.328342,-7.854202,-5.081194,4.094828,8.538216,-7.481861,9.428058,-1.814794],[-5.370267,9.491015,-2.966516,-3.981117,6.884342,8.690897,-5.679603,-3.274672,9.881741,-5.906614,3.151386,-8.688433],[9.917288,8.485569,7.651567,-7.753156,-9.747419,9.235680,5.555646,3.219542,-9.423210,0.883141,-3.540757,7.854783],[-5.496692,1.967381,-0.177489,-3.339928,-3.846306,-7.023361,5.120730,9.675981,1.871285,-6.567943,-2.922096,4.326350],[4.280160,2.967906,-5.314760,3.106304,0.376508,8.558205,1.626209,2.112174,6.471286,6.802601,0.334340,7.306404],[7.428993,3.171210,5.045141,-7.708412,2.782816,-9.180189,1.684067,-7.470473,-5.712920,-9.964491,9.298203,7.846151],[8.963570,-2.432276,9.227382,2.129032,1.946378,-2.698673,-9.451436,5.773552,-3.330559,5.635158,-7.000021,-2.470518],[9.013186,-4.373977,7.972076,-8.727883,-1.627846,7.532704,-5.749226,-7.541305,7.795666,-6.028423,0.417871,8.144338],[5.555506,0.785251,-0.861203,-5.076650,-7.309782,-8.935597,-8.657020,6.001302,5.824756,-4.872918,6.006398,-0.269686]], dtype = "float64")#candidate|257|(12, 12)|const|float64
bop_258 = relay.right_shift(uop_255.astype('uint16'), relay.reshape(const_257.astype('uint16'), relay.shape_of(uop_255))) # shape=(12, 12)
uop_261 = relay.acosh(uop_241.astype('float32')) # shape=(12, 12)
uop_263 = relay.log10(uop_235.astype('float64')) # shape=(12, 12)
bop_265 = relay.floor_mod(uop_263.astype('float32'), relay.reshape(uop_230.astype('float32'), relay.shape_of(uop_263))) # shape=(12, 12)
output = relay.Tuple([uop_174,uop_181,call_187,var_188,bop_198,uop_208,uop_214,bop_216,bop_220,bop_223,bop_227,bop_238,bop_243,bop_252,bop_258,uop_261,bop_265,])
output2 = relay.Tuple([uop_174,uop_181,call_189,var_188,bop_198,uop_208,uop_214,bop_216,bop_220,bop_223,bop_227,bop_238,bop_243,bop_252,bop_258,uop_261,bop_265,])
func_268 = relay.Function([var_160,var_163,var_188,var_197,var_204,var_226,var_237,var_248,], output)
mod['func_268'] = func_268
mod = relay.transform.InferType()(mod)
mutated_mod['func_268'] = func_268
mutated_mod = relay.transform.InferType()(mutated_mod)
func_268_call = mutated_mod.get_global_var('func_268')
var_270 = relay.var("var_270", dtype = "float64", shape = (12, 12))#candidate|270|(12, 12)|var|float64
var_271 = relay.var("var_271", dtype = "float64", shape = (12, 12))#candidate|271|(12, 12)|var|float64
var_272 = relay.var("var_272", dtype = "float64", shape = (154,))#candidate|272|(154,)|var|float64
var_273 = relay.var("var_273", dtype = "float32", shape = (12, 12))#candidate|273|(12, 12)|var|float32
var_274 = relay.var("var_274", dtype = "float32", shape = (12, 12))#candidate|274|(12, 12)|var|float32
var_275 = relay.var("var_275", dtype = "float32", shape = (12, 12))#candidate|275|(12, 12)|var|float32
var_276 = relay.var("var_276", dtype = "float64", shape = (12, 12))#candidate|276|(12, 12)|var|float64
var_277 = relay.var("var_277", dtype = "float64", shape = (12, 12))#candidate|277|(12, 12)|var|float64
call_269 = func_268_call(var_270,var_271,var_272,var_273,var_274,var_275,var_276,var_277,)
output = call_269
func_278 = relay.Function([var_270,var_271,var_272,var_273,var_274,var_275,var_276,var_277,], output)
mutated_mod['func_278'] = func_278
mutated_mod = relay.transform.InferType()(mutated_mod)
var_280 = relay.var("var_280", dtype = "float64", shape = ())#candidate|280|()|var|float64
const_281 = relay.const([1.167402,4.319356,-2.229606,1.348558,2.439371,5.977327,4.427408,9.263331,-3.209398], dtype = "float64")#candidate|281|(9,)|const|float64
bop_282 = relay.less(var_280.astype('bool'), const_281.astype('bool')) # shape=(9,)
uop_285 = relay.asin(bop_282.astype('float64')) # shape=(9,)
uop_287 = relay.tan(uop_285.astype('float32')) # shape=(9,)
uop_289 = relay.asin(uop_287.astype('float64')) # shape=(9,)
var_291 = relay.var("var_291", dtype = "float64", shape = (9,))#candidate|291|(9,)|var|float64
bop_292 = relay.less_equal(uop_289.astype('bool'), relay.reshape(var_291.astype('bool'), relay.shape_of(uop_289))) # shape=(9,)
bop_295 = relay.logical_or(uop_285.astype('bool'), relay.reshape(uop_287.astype('bool'), relay.shape_of(uop_285))) # shape=(9,)
uop_298 = relay.erf(bop_295.astype('float64')) # shape=(9,)
bop_300 = relay.bitwise_xor(uop_289.astype('uint16'), relay.reshape(bop_295.astype('uint16'), relay.shape_of(uop_289))) # shape=(9,)
bop_303 = relay.logical_xor(bop_300.astype('int16'), relay.reshape(bop_295.astype('int16'), relay.shape_of(bop_300))) # shape=(9,)
uop_306 = relay.sin(bop_300.astype('float64')) # shape=(9,)
bop_308 = relay.right_shift(uop_285.astype('int8'), relay.reshape(uop_306.astype('int8'), relay.shape_of(uop_285))) # shape=(9,)
uop_311 = relay.atanh(bop_308.astype('float32')) # shape=(9,)
uop_313 = relay.atan(uop_311.astype('float64')) # shape=(9,)
var_315 = relay.var("var_315", dtype = "float32", shape = (9,))#candidate|315|(9,)|var|float32
bop_316 = relay.bitwise_xor(uop_311.astype('int8'), relay.reshape(var_315.astype('int8'), relay.shape_of(uop_311))) # shape=(9,)
uop_319 = relay.rsqrt(bop_316.astype('float64')) # shape=(9,)
bop_321 = relay.minimum(bop_308.astype('int16'), relay.reshape(uop_298.astype('int16'), relay.shape_of(bop_308))) # shape=(9,)
const_324 = relay.const([-4.435592,-8.032141,-9.860766,-6.530356,-4.937891,1.822843,8.669895,-1.767773,-5.144524], dtype = "float32")#candidate|324|(9,)|const|float32
bop_325 = relay.maximum(uop_311.astype('uint8'), relay.reshape(const_324.astype('uint8'), relay.shape_of(uop_311))) # shape=(9,)
output = relay.Tuple([bop_292,bop_303,uop_313,uop_319,bop_321,bop_325,])
output2 = relay.Tuple([bop_292,bop_303,uop_313,uop_319,bop_321,bop_325,])
func_328 = relay.Function([var_280,var_291,var_315,], output)
mod['func_328'] = func_328
mod = relay.transform.InferType()(mod)
var_329 = relay.var("var_329", dtype = "float64", shape = ())#candidate|329|()|var|float64
var_330 = relay.var("var_330", dtype = "float64", shape = (9,))#candidate|330|(9,)|var|float64
var_331 = relay.var("var_331", dtype = "float32", shape = (9,))#candidate|331|(9,)|var|float32
output = func_328(var_329,var_330,var_331,)
func_332 = relay.Function([var_329,var_330,var_331,], output)
mutated_mod['func_332'] = func_332
mutated_mod = relay.transform.InferType()(mutated_mod)
var_334 = relay.var("var_334", dtype = "float32", shape = (1,))#candidate|334|(1,)|var|float32
uop_335 = relay.erf(var_334.astype('float32')) # shape=(1,)
uop_337 = relay.exp(uop_335.astype('float64')) # shape=(1,)
output = relay.Tuple([uop_337,])
output2 = relay.Tuple([uop_337,])
F = relay.Function([var_334,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_334,], output2)
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
input_334= np.array([6.521290], dtype='float32')
module1.set_input('var_334', input_334)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_334, )
res3 = intrp3.evaluate()(input_334, )
res4 = intrp4.evaluate()(input_334, )
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
module5.set_input('var_334', input_334)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_334, )
res7 = intrp7.evaluate()(input_334, )
res8 = intrp8.evaluate()(input_334, )
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
module9.set_input('var_334', input_334)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_334, )
res11 = intrp11.evaluate()(input_334, )
res12 = intrp12.evaluate()(input_334, )
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
module13.set_input('var_334', input_334)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_334, )
res15 = intrp15.evaluate()(input_334, )
res16 = intrp16.evaluate()(input_334, )
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
module17.set_input('var_334', input_334)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_334, )
res19 = intrp19.evaluate()(input_334, )
res20 = intrp20.evaluate()(input_334, )
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
module21.set_input('var_334', input_334)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_334, )
res23 = intrp23.evaluate()(input_334, )
res24 = intrp24.evaluate()(input_334, )
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

'''45: TVMFuncCall
44: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
43: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
42: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
41: tvm::IRModule::FromExpr(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&)
40: tvm::IRModule::FromExprInContext(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&, std::unordered_set<tvm::runtime::String, std::hash<tvm::runtime::String>, std::equal_to<tvm::runtime::String>, std::allocator<tvm::runtime::String> >)
39: tvm::IRModuleNode::Add(tvm::GlobalVar const&, tvm::BaseFunc const&, bool)
38: tvm::WarnIfMalformed(tvm::IRModule const&, tvm::relay::Function)
37: tvm::relay::FreeTypeVars(tvm::RelayExpr const&, tvm::IRModule const&)
36: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
35: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
34: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
33: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
32: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
31: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
30: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
29: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
28: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
27: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
26: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
25: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
24: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
23: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
22: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
21: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
20: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
19: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
16: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
15: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
14: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
13: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
12: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
11: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
10: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
9: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
8: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
7: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
6: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
5: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
4: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9RelayEx
1: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::ConstructorNode const*)
0: tvm::IRModuleNode::LookupTypeDef(tvm::GlobalTypeVar const&) const

'''