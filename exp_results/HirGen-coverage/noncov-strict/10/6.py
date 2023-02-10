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
var_0 = relay.var("var_0", dtype = "float64", shape = ())#candidate|0|()|var|float64
const_1 = relay.const(-5.757707, dtype = "float64")#candidate|1|()|const|float64
bop_2 = relay.divide(var_0.astype('float64'), const_1.astype('float64')) # shape=()
var_5 = relay.var("var_5", dtype = "float64", shape = ())#candidate|5|()|var|float64
bop_6 = relay.subtract(var_0.astype('int32'), var_5.astype('int32')) # shape=()
uop_9 = relay.sigmoid(bop_2.astype('float32')) # shape=()
uop_11 = relay.cosh(var_5.astype('float32')) # shape=()
uop_13 = relay.sigmoid(uop_11.astype('float64')) # shape=()
bop_15 = relay.subtract(uop_13.astype('uint64'), uop_11.astype('uint64')) # shape=()
uop_18 = relay.exp(bop_15.astype('float64')) # shape=()
uop_20 = relay.sinh(uop_18.astype('float32')) # shape=()
bop_22 = relay.right_shift(uop_18.astype('int64'), uop_20.astype('int64')) # shape=()
bop_25 = relay.minimum(bop_22.astype('uint16'), bop_2.astype('uint16')) # shape=()
bop_28 = relay.equal(bop_22.astype('bool'), uop_20.astype('bool')) # shape=()
bop_31 = relay.logical_xor(uop_20.astype('uint8'), bop_25.astype('uint8')) # shape=()
bop_34 = relay.logical_and(bop_22.astype('bool'), bop_25.astype('bool')) # shape=()
uop_37 = relay.asinh(bop_28.astype('float64')) # shape=()
bop_39 = relay.floor_divide(uop_37.astype('float64'), uop_13.astype('float64')) # shape=()
bop_42 = relay.bitwise_xor(uop_20.astype('uint8'), bop_34.astype('uint8')) # shape=()
bop_45 = relay.bitwise_or(bop_39.astype('int32'), uop_13.astype('int32')) # shape=()
var_48 = relay.var("var_48", dtype = "int32", shape = (6, 5, 13))#candidate|48|(6, 5, 13)|var|int32
bop_49 = relay.maximum(bop_45.astype('int8'), var_48.astype('int8')) # shape=(6, 5, 13)
var_52 = relay.var("var_52", dtype = "int32", shape = (5, 11, 4))#candidate|52|(5, 11, 4)|var|int32
bop_53 = relay.floor_mod(bop_45.astype('float32'), var_52.astype('float32')) # shape=(5, 11, 4)
uop_56 = relay.asin(bop_49.astype('float32')) # shape=(6, 5, 13)
bop_58 = relay.maximum(bop_39.astype('float64'), uop_13.astype('float64')) # shape=()
bop_61 = relay.bitwise_xor(uop_56.astype('uint16'), const_1.astype('uint16')) # shape=(6, 5, 13)
uop_64 = relay.sigmoid(bop_61.astype('float64')) # shape=(6, 5, 13)
uop_66 = relay.sin(uop_64.astype('float64')) # shape=(6, 5, 13)
bop_68 = relay.mod(uop_66.astype('float32'), uop_9.astype('float32')) # shape=(6, 5, 13)
uop_71 = relay.log2(uop_64.astype('float64')) # shape=(6, 5, 13)
bop_73 = relay.less_equal(uop_64.astype('bool'), uop_37.astype('bool')) # shape=(6, 5, 13)
uop_76 = relay.atanh(bop_68.astype('float32')) # shape=(6, 5, 13)
output = relay.Tuple([bop_6,bop_31,bop_42,bop_53,bop_58,uop_71,bop_73,uop_76,])
output2 = relay.Tuple([bop_6,bop_31,bop_42,bop_53,bop_58,uop_71,bop_73,uop_76,])
func_78 = relay.Function([var_0,var_5,var_48,var_52,], output)
mod['func_78'] = func_78
mod = relay.transform.InferType()(mod)
var_79 = relay.var("var_79", dtype = "float64", shape = ())#candidate|79|()|var|float64
var_80 = relay.var("var_80", dtype = "float64", shape = ())#candidate|80|()|var|float64
var_81 = relay.var("var_81", dtype = "int32", shape = (6, 5, 13))#candidate|81|(6, 5, 13)|var|int32
var_82 = relay.var("var_82", dtype = "int32", shape = (5, 11, 4))#candidate|82|(5, 11, 4)|var|int32
output = func_78(var_79,var_80,var_81,var_82,)
func_83 = relay.Function([var_79,var_80,var_81,var_82,], output)
mutated_mod['func_83'] = func_83
mutated_mod = relay.transform.InferType()(mutated_mod)
var_85 = relay.var("var_85", dtype = "float32", shape = ())#candidate|85|()|var|float32
const_86 = relay.const([[[-9.633796,5.432331,0.857864,1.082837,6.632910,-6.400290,4.016478,-3.060249,-4.344372,-2.919279,0.846433],[-7.965278,-7.384162,5.717808,5.937721,-0.970817,9.896301,-3.189838,-1.713138,-9.219094,-9.680792,-4.933317],[0.249116,9.099454,-6.993599,-1.274776,3.625707,9.276359,-1.807523,8.044640,-4.524675,7.800885,9.161408],[-1.294167,8.222186,5.952816,5.766345,4.520275,4.419907,-8.212610,6.825050,6.880324,-3.818190,-6.706459],[-0.824378,-1.048504,7.822731,-4.217108,2.772948,9.160590,9.630908,-9.968678,-6.964855,8.893076,4.226669],[6.213411,-7.132545,1.099481,0.174968,5.056118,2.493627,-9.520094,5.489578,-0.256131,9.408859,-4.353326],[-4.311723,9.724014,-9.661644,-4.088340,9.456848,-0.793581,0.379191,-7.933991,-9.443639,-0.631568,3.711858],[-7.472813,9.753850,9.008506,-8.261362,8.512741,-2.863742,8.357862,0.819124,-2.346714,0.893221,-8.330203]],[[-5.290658,-7.014588,2.431354,6.039657,7.350475,8.697569,-1.922895,-2.672386,-0.524030,6.663479,-5.629594],[-4.282009,2.548316,-4.436584,2.220861,0.257624,7.452391,-6.430761,-3.213157,5.850081,-1.112835,-8.182542],[3.111709,2.989587,-8.936744,6.658390,-4.443051,-2.491370,-0.194934,-3.874212,1.652947,-8.665626,3.189741],[2.606010,-7.975014,-4.286600,-2.994827,5.269696,8.583287,5.081253,3.894794,-9.756666,-2.053150,4.676064],[-3.988911,6.530972,1.770404,-3.763351,-5.388993,-6.243697,2.683984,0.529622,0.966970,2.193476,-3.746436],[-2.128789,8.667639,4.689646,0.719052,6.378354,-3.573491,-5.817090,-9.683914,5.357440,-4.753300,6.786119],[-3.075310,0.935737,2.554240,-5.599275,-2.659231,3.368801,4.230720,6.893023,-6.068768,-1.258112,1.393753],[-3.343911,-5.426411,1.702513,-1.470245,-8.432884,3.249692,-0.383934,-7.467950,-4.641621,-2.310148,1.688400]],[[-4.099580,-6.322511,8.963608,-2.216131,5.770813,-6.455507,-8.714513,1.673566,-9.244793,5.472983,8.178276],[-3.236278,-6.733308,-3.107811,-8.060256,-6.642712,3.686809,9.681900,-4.993748,0.962372,3.252444,7.041931],[2.096785,-2.355506,-9.075208,-7.591862,5.177422,9.784675,7.709237,-3.417230,3.635634,3.038330,-9.995558],[8.634210,-7.388573,-0.517321,-8.537667,3.304955,-0.547239,3.445845,2.138177,7.365312,-5.147371,-6.134319],[-0.101884,-5.939595,8.710020,-5.775434,-9.842696,-0.736205,9.258744,-0.709367,-1.222120,5.866042,8.388715],[3.980202,-1.868842,-9.181023,-9.269992,2.601804,7.160557,-1.276702,5.925081,-1.106408,2.622466,-3.790856],[-6.023633,-5.253941,-7.027542,7.227538,7.998703,-1.430725,3.003824,-0.455570,7.381970,0.972363,0.242587],[9.188483,-9.217175,9.336200,5.747588,-6.678275,-5.063341,-9.769802,3.556455,5.755707,6.578963,-6.763033]],[[-4.976979,-9.674553,-2.624683,-6.511723,5.435636,-2.307588,-5.613196,2.454063,3.293191,4.164712,6.899043],[-8.152718,0.027692,5.641928,1.286960,-2.406066,-9.403997,6.609297,-9.671032,6.564388,-7.545376,0.494140],[6.226006,-0.234509,1.503099,-8.492978,3.258847,9.602494,-8.007531,4.184149,-0.697065,4.743852,-1.595570],[-7.205395,7.009735,3.348187,3.869144,-7.482888,9.417910,-2.723700,-2.729576,9.115209,5.683486,-2.276043],[5.314025,-8.186494,8.107444,7.712285,-5.361168,-9.663917,9.894149,5.842576,6.846236,4.436347,-8.295090],[1.468901,-9.579231,8.377281,4.699543,-1.637052,7.733103,-5.741009,-9.688597,6.017608,5.958443,-6.754905],[9.220190,-1.018747,4.428693,8.489945,2.717116,-2.539337,-3.679012,6.115116,-5.138120,-0.982420,-8.803713],[-9.965718,-1.364064,-3.313159,-4.211327,-5.248804,-0.267747,-2.487904,-2.314452,-0.149885,-4.971228,-4.274438]]], dtype = "float32")#candidate|86|(4, 8, 11)|const|float32
bop_87 = relay.multiply(var_85.astype('float32'), const_86.astype('float32')) # shape=(4, 8, 11)
bop_90 = relay.divide(const_86.astype('float32'), relay.reshape(bop_87.astype('float32'), relay.shape_of(const_86))) # shape=(4, 8, 11)
var_93 = relay.var("var_93", dtype = "float32", shape = (9, 12))#candidate|93|(9, 12)|var|float32
bop_94 = relay.logical_xor(var_85.astype('uint64'), var_93.astype('uint64')) # shape=(9, 12)
uop_97 = relay.sinh(bop_87.astype('float64')) # shape=(4, 8, 11)
output = relay.Tuple([bop_90,bop_94,uop_97,])
output2 = relay.Tuple([bop_90,bop_94,uop_97,])
func_99 = relay.Function([var_85,var_93,], output)
mod['func_99'] = func_99
mod = relay.transform.InferType()(mod)
var_100 = relay.var("var_100", dtype = "float32", shape = ())#candidate|100|()|var|float32
var_101 = relay.var("var_101", dtype = "float32", shape = (9, 12))#candidate|101|(9, 12)|var|float32
output = func_99(var_100,var_101,)
func_102 = relay.Function([var_100,var_101,], output)
mutated_mod['func_102'] = func_102
mutated_mod = relay.transform.InferType()(mutated_mod)
var_104 = relay.var("var_104", dtype = "float32", shape = (12, 6))#candidate|104|(12, 6)|var|float32
uop_105 = relay.asinh(var_104.astype('float32')) # shape=(12, 6)
bop_107 = relay.greater(uop_105.astype('bool'), relay.reshape(var_104.astype('bool'), relay.shape_of(uop_105))) # shape=(12, 6)
output = relay.Tuple([bop_107,])
output2 = relay.Tuple([bop_107,])
func_110 = relay.Function([var_104,], output)
mod['func_110'] = func_110
mod = relay.transform.InferType()(mod)
var_111 = relay.var("var_111", dtype = "float32", shape = (12, 6))#candidate|111|(12, 6)|var|float32
output = func_110(var_111)
func_112 = relay.Function([var_111], output)
mutated_mod['func_112'] = func_112
mutated_mod = relay.transform.InferType()(mutated_mod)
var_114 = relay.var("var_114", dtype = "bool", shape = (8, 3, 14))#candidate|114|(8, 3, 14)|var|bool
var_115 = relay.var("var_115", dtype = "bool", shape = (8, 3, 14))#candidate|115|(8, 3, 14)|var|bool
bop_116 = relay.logical_and(var_114.astype('bool'), relay.reshape(var_115.astype('bool'), relay.shape_of(var_114))) # shape=(8, 3, 14)
bop_119 = relay.multiply(bop_116.astype('uint8'), relay.reshape(var_114.astype('uint8'), relay.shape_of(bop_116))) # shape=(8, 3, 14)
uop_122 = relay.log2(bop_116.astype('float64')) # shape=(8, 3, 14)
bop_124 = relay.logical_and(uop_122.astype('bool'), relay.reshape(bop_116.astype('bool'), relay.shape_of(uop_122))) # shape=(8, 3, 14)
uop_127 = relay.atan(uop_122.astype('float64')) # shape=(8, 3, 14)
bop_129 = relay.floor_divide(uop_127.astype('float32'), relay.reshape(bop_119.astype('float32'), relay.shape_of(uop_127))) # shape=(8, 3, 14)
uop_132 = relay.sinh(uop_127.astype('float32')) # shape=(8, 3, 14)
output = relay.Tuple([bop_124,bop_129,uop_132,])
output2 = relay.Tuple([bop_124,bop_129,uop_132,])
func_134 = relay.Function([var_114,var_115,], output)
mod['func_134'] = func_134
mod = relay.transform.InferType()(mod)
var_135 = relay.var("var_135", dtype = "bool", shape = (8, 3, 14))#candidate|135|(8, 3, 14)|var|bool
var_136 = relay.var("var_136", dtype = "bool", shape = (8, 3, 14))#candidate|136|(8, 3, 14)|var|bool
output = func_134(var_135,var_136,)
func_137 = relay.Function([var_135,var_136,], output)
mutated_mod['func_137'] = func_137
mutated_mod = relay.transform.InferType()(mutated_mod)
var_139 = relay.var("var_139", dtype = "uint64", shape = (10, 9))#candidate|139|(10, 9)|var|uint64
const_140 = relay.const([[-8,7,-5,-4,1,-7,-6,-3,-10],[9,-5,-3,-7,-2,-1,-6,-4,-1],[-3,-9,-8,3,9,-4,-1,5,1],[4,6,-8,10,6,-7,5,3,-6],[-10,-8,7,8,-4,-3,-2,3,-2],[5,-3,8,-9,1,6,-4,8,-5],[-3,-2,-5,-9,3,-5,3,7,-2],[-7,7,-2,7,7,-9,10,-5,-8],[-9,-9,1,5,5,-1,8,-5,-7],[-8,-6,-10,9,-8,2,7,5,2]], dtype = "uint64")#candidate|140|(10, 9)|const|uint64
bop_141 = relay.left_shift(var_139.astype('uint64'), relay.reshape(const_140.astype('uint64'), relay.shape_of(var_139))) # shape=(10, 9)
func_134_call = mod.get_global_var('func_134')
func_137_call = mutated_mod.get_global_var('func_137')
var_145 = relay.var("var_145", dtype = "bool", shape = (336,))#candidate|145|(336,)|var|bool
call_144 = relay.TupleGetItem(func_134_call(relay.reshape(var_145.astype('bool'), [8, 3, 14]), relay.reshape(var_145.astype('bool'), [8, 3, 14]), ), 0)
call_146 = relay.TupleGetItem(func_137_call(relay.reshape(var_145.astype('bool'), [8, 3, 14]), relay.reshape(var_145.astype('bool'), [8, 3, 14]), ), 0)
func_99_call = mod.get_global_var('func_99')
func_102_call = mutated_mod.get_global_var('func_102')
var_148 = relay.var("var_148", dtype = "float32", shape = ())#candidate|148|()|var|float32
var_149 = relay.var("var_149", dtype = "float32", shape = (108,))#candidate|149|(108,)|var|float32
call_147 = relay.TupleGetItem(func_99_call(relay.reshape(var_148.astype('float32'), []), relay.reshape(var_149.astype('float32'), [9, 12]), ), 1)
call_150 = relay.TupleGetItem(func_102_call(relay.reshape(var_148.astype('float32'), []), relay.reshape(var_149.astype('float32'), [9, 12]), ), 1)
uop_151 = relay.sigmoid(call_144.astype('float64')) # shape=(8, 3, 14)
uop_153 = relay.sigmoid(call_146.astype('float64')) # shape=(8, 3, 14)
uop_154 = relay.tan(uop_151.astype('float64')) # shape=(8, 3, 14)
uop_156 = relay.tan(uop_153.astype('float64')) # shape=(8, 3, 14)
uop_157 = relay.sinh(uop_151.astype('float64')) # shape=(8, 3, 14)
uop_159 = relay.sinh(uop_153.astype('float64')) # shape=(8, 3, 14)
bop_160 = relay.logical_and(uop_157.astype('bool'), relay.reshape(uop_154.astype('bool'), relay.shape_of(uop_157))) # shape=(8, 3, 14)
bop_163 = relay.logical_and(uop_159.astype('bool'), relay.reshape(uop_156.astype('bool'), relay.shape_of(uop_159))) # shape=(8, 3, 14)
bop_164 = relay.bitwise_or(uop_151.astype('int64'), relay.reshape(uop_154.astype('int64'), relay.shape_of(uop_151))) # shape=(8, 3, 14)
bop_167 = relay.bitwise_or(uop_153.astype('int64'), relay.reshape(uop_156.astype('int64'), relay.shape_of(uop_153))) # shape=(8, 3, 14)
var_168 = relay.var("var_168", dtype = "uint64", shape = (10, 9))#candidate|168|(10, 9)|var|uint64
bop_169 = relay.greater(var_139.astype('bool'), relay.reshape(var_168.astype('bool'), relay.shape_of(var_139))) # shape=(10, 9)
var_172 = relay.var("var_172", dtype = "float64", shape = (8, 3, 14))#candidate|172|(8, 3, 14)|var|float64
bop_173 = relay.less(uop_154.astype('bool'), relay.reshape(var_172.astype('bool'), relay.shape_of(uop_154))) # shape=(8, 3, 14)
bop_176 = relay.less(uop_156.astype('bool'), relay.reshape(var_172.astype('bool'), relay.shape_of(uop_156))) # shape=(8, 3, 14)
bop_177 = relay.logical_xor(call_144.astype('int32'), relay.reshape(var_145.astype('int32'), relay.shape_of(call_144))) # shape=(8, 3, 14)
bop_180 = relay.logical_xor(call_146.astype('int32'), relay.reshape(var_145.astype('int32'), relay.shape_of(call_146))) # shape=(8, 3, 14)
var_181 = relay.var("var_181", dtype = "float64", shape = (8, 3, 14))#candidate|181|(8, 3, 14)|var|float64
bop_182 = relay.power(uop_151.astype('float64'), relay.reshape(var_181.astype('float64'), relay.shape_of(uop_151))) # shape=(8, 3, 14)
bop_185 = relay.power(uop_153.astype('float64'), relay.reshape(var_181.astype('float64'), relay.shape_of(uop_153))) # shape=(8, 3, 14)
uop_186 = relay.rsqrt(bop_173.astype('float32')) # shape=(8, 3, 14)
uop_188 = relay.rsqrt(bop_176.astype('float32')) # shape=(8, 3, 14)
uop_189 = relay.cos(uop_186.astype('float64')) # shape=(8, 3, 14)
uop_191 = relay.cos(uop_188.astype('float64')) # shape=(8, 3, 14)
uop_192 = relay.sin(uop_189.astype('float64')) # shape=(8, 3, 14)
uop_194 = relay.sin(uop_191.astype('float64')) # shape=(8, 3, 14)
bop_195 = relay.bitwise_and(uop_192.astype('int8'), relay.reshape(uop_151.astype('int8'), relay.shape_of(uop_192))) # shape=(8, 3, 14)
bop_198 = relay.bitwise_and(uop_194.astype('int8'), relay.reshape(uop_153.astype('int8'), relay.shape_of(uop_194))) # shape=(8, 3, 14)
uop_199 = relay.atanh(bop_141.astype('float32')) # shape=(10, 9)
bop_201 = relay.logical_and(uop_189.astype('bool'), relay.reshape(var_181.astype('bool'), relay.shape_of(uop_189))) # shape=(8, 3, 14)
bop_204 = relay.logical_and(uop_191.astype('bool'), relay.reshape(var_181.astype('bool'), relay.shape_of(uop_191))) # shape=(8, 3, 14)
bop_205 = relay.bitwise_and(uop_192.astype('uint32'), relay.reshape(bop_182.astype('uint32'), relay.shape_of(uop_192))) # shape=(8, 3, 14)
bop_208 = relay.bitwise_and(uop_194.astype('uint32'), relay.reshape(bop_185.astype('uint32'), relay.shape_of(uop_194))) # shape=(8, 3, 14)
uop_209 = relay.atan(bop_205.astype('float32')) # shape=(8, 3, 14)
uop_211 = relay.atan(bop_208.astype('float32')) # shape=(8, 3, 14)
func_78_call = mod.get_global_var('func_78')
func_83_call = mutated_mod.get_global_var('func_83')
var_213 = relay.var("var_213", dtype = "int32", shape = (390,))#candidate|213|(390,)|var|int32
var_214 = relay.var("var_214", dtype = "int32", shape = (220,))#candidate|214|(220,)|var|int32
call_212 = relay.TupleGetItem(func_78_call(relay.reshape(var_148.astype('float64'), []), relay.reshape(var_148.astype('float64'), []), relay.reshape(var_213.astype('int32'), [6, 5, 13]), relay.reshape(var_214.astype('int32'), [5, 11, 4]), ), 3)
call_215 = relay.TupleGetItem(func_83_call(relay.reshape(var_148.astype('float64'), []), relay.reshape(var_148.astype('float64'), []), relay.reshape(var_213.astype('int32'), [6, 5, 13]), relay.reshape(var_214.astype('int32'), [5, 11, 4]), ), 3)
bop_216 = relay.less(uop_209.astype('bool'), relay.reshape(uop_192.astype('bool'), relay.shape_of(uop_209))) # shape=(8, 3, 14)
bop_219 = relay.less(uop_211.astype('bool'), relay.reshape(uop_194.astype('bool'), relay.shape_of(uop_211))) # shape=(8, 3, 14)
uop_220 = relay.sqrt(bop_216.astype('float32')) # shape=(8, 3, 14)
uop_222 = relay.sqrt(bop_219.astype('float32')) # shape=(8, 3, 14)
uop_223 = relay.atanh(uop_220.astype('float64')) # shape=(8, 3, 14)
uop_225 = relay.atanh(uop_222.astype('float64')) # shape=(8, 3, 14)
uop_226 = relay.log10(uop_189.astype('float64')) # shape=(8, 3, 14)
uop_228 = relay.log10(uop_191.astype('float64')) # shape=(8, 3, 14)
uop_229 = relay.acosh(uop_209.astype('float64')) # shape=(8, 3, 14)
uop_231 = relay.acosh(uop_211.astype('float64')) # shape=(8, 3, 14)
const_232 = relay.const([[[True,False,True,True,False,True,False,False,True,True,True,True,False,True],[True,True,True,True,False,True,False,False,True,False,False,True,False,True],[True,True,False,False,True,True,True,False,False,True,False,True,True,False]],[[True,True,True,False,True,False,True,True,True,True,True,False,False,False],[False,False,True,True,True,True,True,True,False,False,True,True,True,True],[False,False,True,True,False,False,False,True,False,True,False,False,True,True]],[[False,True,True,False,True,True,True,False,False,False,True,True,False,False],[False,False,False,False,False,True,False,False,False,False,True,False,True,True],[False,False,True,True,True,False,True,False,True,True,True,False,True,False]],[[True,False,True,True,False,True,True,False,False,True,True,False,True,False],[True,True,False,True,True,True,False,True,True,False,True,True,True,False],[True,False,True,False,False,False,True,False,True,False,True,True,False,False]],[[False,True,False,True,False,False,False,False,True,True,True,True,True,False],[False,False,True,True,False,False,True,True,False,False,True,True,False,False],[False,False,False,False,False,True,True,False,True,False,False,True,True,True]],[[False,False,False,False,False,True,True,False,True,False,True,True,False,True],[False,False,True,False,True,False,True,True,True,False,True,False,False,False],[True,True,True,True,True,True,True,True,False,False,False,True,False,True]],[[False,False,False,False,True,False,True,False,False,False,True,True,False,False],[True,True,False,True,False,False,False,False,True,False,True,False,False,True],[True,True,True,False,True,True,False,False,True,True,False,True,True,True]],[[False,False,False,False,True,False,True,True,False,True,True,False,True,True],[False,False,False,True,True,True,True,False,True,False,True,False,True,True],[False,True,True,False,True,True,False,False,True,True,True,False,True,True]]], dtype = "bool")#candidate|232|(8, 3, 14)|const|bool
bop_233 = relay.bitwise_and(bop_216.astype('uint64'), relay.reshape(const_232.astype('uint64'), relay.shape_of(bop_216))) # shape=(8, 3, 14)
bop_236 = relay.bitwise_and(bop_219.astype('uint64'), relay.reshape(const_232.astype('uint64'), relay.shape_of(bop_219))) # shape=(8, 3, 14)
uop_237 = relay.asin(uop_223.astype('float64')) # shape=(8, 3, 14)
uop_239 = relay.asin(uop_225.astype('float64')) # shape=(8, 3, 14)
uop_240 = relay.cosh(uop_237.astype('float32')) # shape=(8, 3, 14)
uop_242 = relay.cosh(uop_239.astype('float32')) # shape=(8, 3, 14)
bop_243 = relay.bitwise_xor(uop_240.astype('uint64'), relay.reshape(bop_195.astype('uint64'), relay.shape_of(uop_240))) # shape=(8, 3, 14)
bop_246 = relay.bitwise_xor(uop_242.astype('uint64'), relay.reshape(bop_198.astype('uint64'), relay.shape_of(uop_242))) # shape=(8, 3, 14)
bop_247 = relay.minimum(uop_240.astype('uint64'), relay.reshape(uop_220.astype('uint64'), relay.shape_of(uop_240))) # shape=(8, 3, 14)
bop_250 = relay.minimum(uop_242.astype('uint64'), relay.reshape(uop_222.astype('uint64'), relay.shape_of(uop_242))) # shape=(8, 3, 14)
bop_251 = relay.bitwise_xor(bop_233.astype('int16'), relay.reshape(uop_154.astype('int16'), relay.shape_of(bop_233))) # shape=(8, 3, 14)
bop_254 = relay.bitwise_xor(bop_236.astype('int16'), relay.reshape(uop_156.astype('int16'), relay.shape_of(bop_236))) # shape=(8, 3, 14)
bop_255 = relay.power(bop_243.astype('float64'), relay.reshape(uop_154.astype('float64'), relay.shape_of(bop_243))) # shape=(8, 3, 14)
bop_258 = relay.power(bop_246.astype('float64'), relay.reshape(uop_156.astype('float64'), relay.shape_of(bop_246))) # shape=(8, 3, 14)
bop_259 = relay.not_equal(bop_247.astype('bool'), relay.reshape(uop_209.astype('bool'), relay.shape_of(bop_247))) # shape=(8, 3, 14)
bop_262 = relay.not_equal(bop_250.astype('bool'), relay.reshape(uop_211.astype('bool'), relay.shape_of(bop_250))) # shape=(8, 3, 14)
uop_263 = relay.rsqrt(bop_247.astype('float64')) # shape=(8, 3, 14)
uop_265 = relay.rsqrt(bop_250.astype('float64')) # shape=(8, 3, 14)
func_78_call = mod.get_global_var('func_78')
func_83_call = mutated_mod.get_global_var('func_83')
call_266 = relay.TupleGetItem(func_78_call(relay.reshape(var_148.astype('float64'), []), relay.reshape(var_148.astype('float64'), []), relay.reshape(var_213.astype('int32'), [6, 5, 13]), relay.reshape(call_212.astype('int32'), [5, 11, 4]), ), 3)
call_267 = relay.TupleGetItem(func_83_call(relay.reshape(var_148.astype('float64'), []), relay.reshape(var_148.astype('float64'), []), relay.reshape(var_213.astype('int32'), [6, 5, 13]), relay.reshape(call_212.astype('int32'), [5, 11, 4]), ), 3)
uop_268 = relay.asin(uop_263.astype('float32')) # shape=(8, 3, 14)
uop_270 = relay.asin(uop_265.astype('float32')) # shape=(8, 3, 14)
bop_271 = relay.left_shift(uop_268.astype('int64'), relay.reshape(call_144.astype('int64'), relay.shape_of(uop_268))) # shape=(8, 3, 14)
bop_274 = relay.left_shift(uop_270.astype('int64'), relay.reshape(call_146.astype('int64'), relay.shape_of(uop_270))) # shape=(8, 3, 14)
bop_275 = relay.less(uop_263.astype('bool'), relay.reshape(uop_151.astype('bool'), relay.shape_of(uop_263))) # shape=(8, 3, 14)
bop_278 = relay.less(uop_265.astype('bool'), relay.reshape(uop_153.astype('bool'), relay.shape_of(uop_265))) # shape=(8, 3, 14)
uop_279 = relay.asin(bop_275.astype('float64')) # shape=(8, 3, 14)
uop_281 = relay.asin(bop_278.astype('float64')) # shape=(8, 3, 14)
bop_282 = relay.left_shift(bop_271.astype('uint64'), relay.reshape(uop_192.astype('uint64'), relay.shape_of(bop_271))) # shape=(8, 3, 14)
bop_285 = relay.left_shift(bop_274.astype('uint64'), relay.reshape(uop_194.astype('uint64'), relay.shape_of(bop_274))) # shape=(8, 3, 14)
uop_286 = relay.atanh(bop_282.astype('float64')) # shape=(8, 3, 14)
uop_288 = relay.atanh(bop_285.astype('float64')) # shape=(8, 3, 14)
uop_289 = relay.log10(uop_286.astype('float64')) # shape=(8, 3, 14)
uop_291 = relay.log10(uop_288.astype('float64')) # shape=(8, 3, 14)
uop_292 = relay.atanh(uop_289.astype('float32')) # shape=(8, 3, 14)
uop_294 = relay.atanh(uop_291.astype('float32')) # shape=(8, 3, 14)
uop_295 = relay.log(uop_289.astype('float64')) # shape=(8, 3, 14)
uop_297 = relay.log(uop_291.astype('float64')) # shape=(8, 3, 14)
uop_298 = relay.asin(uop_295.astype('float32')) # shape=(8, 3, 14)
uop_300 = relay.asin(uop_297.astype('float32')) # shape=(8, 3, 14)
bop_301 = relay.multiply(uop_298.astype('uint16'), relay.reshape(call_144.astype('uint16'), relay.shape_of(uop_298))) # shape=(8, 3, 14)
bop_304 = relay.multiply(uop_300.astype('uint16'), relay.reshape(call_146.astype('uint16'), relay.shape_of(uop_300))) # shape=(8, 3, 14)
uop_305 = relay.sinh(bop_301.astype('float32')) # shape=(8, 3, 14)
uop_307 = relay.sinh(bop_304.astype('float32')) # shape=(8, 3, 14)
uop_308 = relay.sinh(uop_292.astype('float64')) # shape=(8, 3, 14)
uop_310 = relay.sinh(uop_294.astype('float64')) # shape=(8, 3, 14)
uop_311 = relay.tan(uop_305.astype('float32')) # shape=(8, 3, 14)
uop_313 = relay.tan(uop_307.astype('float32')) # shape=(8, 3, 14)
const_314 = relay.const([[[7.722585,-2.064515,-4.035823,-0.408076,3.402027,8.614615,-8.447962,-0.671239,-8.930215,1.367753,3.519600,-1.191460,-1.777185,-4.227591],[0.885525,-9.524727,-5.490324,-1.934666,-4.225284,4.940987,4.356738,-4.937992,-2.990313,1.320508,-6.663280,-2.138310,-1.678231,-2.640807],[6.419195,-8.417735,-2.284673,2.607553,6.504404,6.341481,2.154732,7.078579,-3.097507,-6.321765,8.401631,7.691857,0.997702,5.753202]],[[-1.680580,-0.221062,-7.630743,-3.779793,3.408262,1.973874,-7.173745,5.054334,9.817522,-0.273063,-0.631749,-1.585442,-2.388609,7.820750],[-1.949230,-1.298373,-0.961730,-3.846376,6.055984,-1.922046,-5.113251,-9.990340,-7.445460,-4.192522,-5.964943,3.776102,-4.286039,-4.706127],[-3.176008,-2.895259,-2.856184,5.334076,9.457038,7.637254,4.166986,-5.143558,4.053858,-0.493644,-0.954304,4.403160,8.513448,4.636580]],[[0.758981,-0.312410,7.914172,-0.069596,1.601088,-2.521458,7.256246,-0.926395,4.231674,-4.857481,1.337555,8.622385,-1.954158,-2.710659],[9.620167,9.724366,-0.915379,-0.190007,-5.856137,9.425865,4.362658,-9.630025,-9.008064,6.314675,2.445275,-6.599504,-6.581392,-3.373616],[7.744037,-5.366784,-6.860468,6.627942,2.794098,-1.340052,-9.139268,-1.650385,-1.094407,-1.104751,7.485948,-3.033025,8.570358,4.005209]],[[-2.055697,-2.812803,-0.247650,3.077643,-8.465282,5.706240,-3.347390,2.325435,-5.921779,-1.560018,5.417676,4.077809,-7.711140,0.288699],[-1.419660,2.721745,8.466810,-4.190318,-9.041687,-5.249901,-4.379196,1.931362,-4.175630,5.794916,-0.648632,7.240227,-7.319651,-2.013786],[-8.623884,2.760172,-3.749709,-2.141773,-2.477284,0.539387,-5.255467,-4.179670,-0.825861,0.126866,-1.707913,0.846915,2.851456,-3.986579]],[[2.288741,-7.865382,2.568575,-7.764152,4.042889,7.461507,-4.105009,-6.932930,-6.990586,1.056512,-0.238748,1.601198,-6.310504,9.724200],[6.814533,5.004596,-7.546068,5.362636,0.033501,0.325931,-9.497383,6.087926,-5.816184,7.619918,-8.453179,5.288265,4.585572,8.114355],[6.563369,3.408468,9.227299,5.680627,5.131541,0.215753,-2.768606,1.609071,5.535662,4.481453,-3.376491,-9.901788,-6.709364,-5.256158]],[[4.777814,2.968725,-7.903501,2.347845,8.948095,-4.271186,7.272143,-5.711553,-8.613300,-3.891952,-6.453087,9.476401,-5.039694,4.071784],[3.528168,4.435122,0.289827,7.649842,8.802772,0.727211,6.178098,5.058387,2.163359,4.989588,6.231556,8.379820,1.858940,3.094733],[7.680102,-4.384815,-5.283677,5.221609,-6.114681,8.874356,2.688516,7.563554,-2.991176,-4.739265,-5.424021,-1.123189,5.947542,8.646747]],[[-1.242121,1.832717,9.326619,7.081087,1.539324,8.663804,3.884523,9.082482,-2.196012,-5.526571,-4.410049,-5.064556,-6.536637,-4.069745],[2.563488,-4.445811,-1.598454,4.103670,-8.958540,4.546468,5.210974,7.228310,-3.362542,4.920757,9.387800,7.291643,-3.562313,-8.881103],[-0.976731,0.257580,-5.254978,-1.293749,9.174119,-3.003686,-4.667232,5.479171,8.272953,-4.832879,6.560428,-8.457282,8.356083,-7.645045]],[[2.670902,-7.093267,-9.029215,6.509530,-5.783610,7.420500,8.315993,9.736394,-1.341746,2.824271,9.514836,0.617755,-3.949985,-6.988343],[0.916019,1.721600,3.561463,5.559944,2.997984,0.672079,-7.467125,7.862288,-0.219230,2.759610,7.305825,-3.192491,-3.014911,-8.369397],[5.803462,3.501941,0.484237,-5.748732,-9.423416,-0.729260,-1.104089,1.147420,2.235946,0.097541,4.136616,-9.681989,7.987894,-2.404431]]], dtype = "float32")#candidate|314|(8, 3, 14)|const|float32
bop_315 = relay.power(uop_311.astype('float32'), relay.reshape(const_314.astype('float32'), relay.shape_of(uop_311))) # shape=(8, 3, 14)
bop_318 = relay.power(uop_313.astype('float32'), relay.reshape(const_314.astype('float32'), relay.shape_of(uop_313))) # shape=(8, 3, 14)
var_319 = relay.var("var_319", dtype = "float32", shape = (8, 3, 14))#candidate|319|(8, 3, 14)|var|float32
bop_320 = relay.less(uop_305.astype('bool'), relay.reshape(var_319.astype('bool'), relay.shape_of(uop_305))) # shape=(8, 3, 14)
bop_323 = relay.less(uop_307.astype('bool'), relay.reshape(var_319.astype('bool'), relay.shape_of(uop_307))) # shape=(8, 3, 14)
const_324 = relay.const([[[-1.178019,-6.477126,9.376891,6.534621,6.130248,1.356993,-2.980863,-0.857682,-0.504438,9.456938,0.795390,3.351656,8.629849,7.516966],[0.217781,-2.487296,-0.460098,-5.426238,4.586692,2.234310,-0.288654,3.965532,6.290219,6.809948,-1.532747,-2.132244,5.844864,2.052114],[2.101616,1.377029,7.587137,-1.560016,8.164978,6.390523,1.437356,4.891972,2.214899,-6.922955,-1.221456,2.880491,-3.024415,-2.994468]],[[9.635621,0.575545,7.457012,0.940134,-3.753442,-4.964002,8.671666,1.947088,-8.659893,-4.105475,9.257191,4.510125,-8.336138,-1.859639],[-3.516770,8.968046,-3.899162,-2.358551,2.218484,4.654292,3.046003,8.708130,9.316365,0.848929,8.386814,-1.984520,-3.941936,-7.855609],[-1.854767,-9.985462,-7.009831,-9.698480,-1.328172,-4.806965,-9.427898,4.810900,1.649936,-2.099189,3.969306,-0.089999,-7.946941,6.418547]],[[-1.581261,9.008746,3.366230,1.450459,2.210416,-3.577731,-4.696310,-0.576569,-4.711730,1.836926,-7.358427,8.997105,7.770995,5.373730],[-1.785265,9.786353,-5.255422,5.454684,-2.308649,2.995078,-5.020042,6.868294,-8.470470,7.356536,9.037709,-6.792974,2.002625,-1.029634],[-2.653748,-0.034655,-9.809332,7.583262,-2.459955,-1.122371,-5.612546,0.745937,-1.393881,2.538823,-0.491635,5.546031,4.520411,5.871890]],[[1.522761,-8.394424,4.245488,-5.779161,-3.352649,1.947645,-1.231358,7.963085,4.740669,9.816385,9.033093,0.625678,-7.510296,7.605145],[-1.076275,-5.009542,-2.873782,4.952015,8.349212,-6.383308,-3.859506,-5.399671,1.945083,9.634634,-7.395276,-4.094430,5.698203,0.091478],[5.959110,-3.668469,-9.907437,1.043692,2.633052,-4.099135,1.068144,-6.916803,-6.449109,-5.900942,-9.392178,1.579020,9.203087,-2.180470]],[[8.538782,-3.779736,-2.043999,-3.948489,8.261163,-3.531352,0.850357,9.200812,1.430856,-0.544314,-1.763846,1.356572,-1.108826,-9.843672],[6.689749,-2.973909,-1.170004,0.814225,4.845758,4.325479,4.445730,6.152585,0.930238,4.495541,-3.690290,-3.667594,7.823693,9.029081],[-9.269588,0.086904,-7.967350,-8.036640,-9.729939,-5.470596,-0.910210,-2.353001,9.974337,-3.181432,-2.168472,3.996052,5.181306,2.952738]],[[3.414914,-0.139914,-9.843179,9.381263,8.620226,-0.949866,8.124531,-1.714734,-3.491047,-6.371419,-1.553299,-0.114703,-7.232443,-2.584838],[-1.454355,-9.500518,-4.850230,6.092082,-6.721255,-3.179442,0.026458,7.089362,4.287031,-1.925499,-6.354294,9.900488,0.218826,6.840049],[7.095681,6.865155,-4.165708,0.763128,-8.530026,6.816354,-6.151188,-6.880069,-1.971641,-9.436346,3.772341,-4.194009,3.255048,4.521762]],[[3.868299,3.219320,-4.491195,-7.934877,-0.647592,4.702995,0.359828,-7.290980,-5.939786,7.449274,-4.599851,-2.926249,-0.321256,-8.581357],[-7.583418,-5.753836,-5.614428,-5.907608,-3.033913,4.454182,4.166315,4.247674,-8.250962,-0.932077,-1.159413,3.227828,1.578169,2.688985],[-3.849543,-7.717896,4.416176,8.832034,-6.106792,-9.806383,-4.404738,1.469338,-8.341779,-3.633890,-6.606811,-7.571637,-8.236610,-5.386754]],[[3.113561,2.171896,8.962027,1.055554,-4.325952,-1.461056,-6.136884,8.774135,6.511292,-5.195522,0.319819,-4.373868,7.631997,3.263845],[-8.402970,-1.461325,3.345317,-4.774826,6.923487,3.635022,-2.267281,-8.058926,-4.925590,7.594552,-9.716966,5.432683,4.453172,-8.694453],[-4.461962,-8.882358,7.811079,8.603391,6.408493,0.558411,8.040474,-7.305884,-8.850783,-5.937373,4.603899,-8.142744,-1.978683,6.694662]]], dtype = "float32")#candidate|324|(8, 3, 14)|const|float32
bop_325 = relay.power(bop_315.astype('float64'), relay.reshape(const_324.astype('float64'), relay.shape_of(bop_315))) # shape=(8, 3, 14)
bop_328 = relay.power(bop_318.astype('float64'), relay.reshape(const_324.astype('float64'), relay.shape_of(bop_318))) # shape=(8, 3, 14)
var_329 = relay.var("var_329", dtype = "float32", shape = (8, 3, 14))#candidate|329|(8, 3, 14)|var|float32
bop_330 = relay.less_equal(uop_305.astype('bool'), relay.reshape(var_329.astype('bool'), relay.shape_of(uop_305))) # shape=(8, 3, 14)
bop_333 = relay.less_equal(uop_307.astype('bool'), relay.reshape(var_329.astype('bool'), relay.shape_of(uop_307))) # shape=(8, 3, 14)
uop_334 = relay.sqrt(uop_305.astype('float64')) # shape=(8, 3, 14)
uop_336 = relay.sqrt(uop_307.astype('float64')) # shape=(8, 3, 14)
uop_337 = relay.tan(bop_320.astype('float64')) # shape=(8, 3, 14)
uop_339 = relay.tan(bop_323.astype('float64')) # shape=(8, 3, 14)
uop_340 = relay.log10(uop_311.astype('float64')) # shape=(8, 3, 14)
uop_342 = relay.log10(uop_313.astype('float64')) # shape=(8, 3, 14)
var_343 = relay.var("var_343", dtype = "float64", shape = (8, 3, 14))#candidate|343|(8, 3, 14)|var|float64
bop_344 = relay.maximum(uop_334.astype('float32'), relay.reshape(var_343.astype('float32'), relay.shape_of(uop_334))) # shape=(8, 3, 14)
bop_347 = relay.maximum(uop_336.astype('float32'), relay.reshape(var_343.astype('float32'), relay.shape_of(uop_336))) # shape=(8, 3, 14)
bop_348 = relay.less(uop_298.astype('bool'), relay.reshape(bop_325.astype('bool'), relay.shape_of(uop_298))) # shape=(8, 3, 14)
bop_351 = relay.less(uop_300.astype('bool'), relay.reshape(bop_328.astype('bool'), relay.shape_of(uop_300))) # shape=(8, 3, 14)
uop_352 = relay.cosh(uop_298.astype('float32')) # shape=(8, 3, 14)
uop_354 = relay.cosh(uop_300.astype('float32')) # shape=(8, 3, 14)
uop_355 = relay.asinh(bop_348.astype('float64')) # shape=(8, 3, 14)
uop_357 = relay.asinh(bop_351.astype('float64')) # shape=(8, 3, 14)
func_134_call = mod.get_global_var('func_134')
func_137_call = mutated_mod.get_global_var('func_137')
call_358 = relay.TupleGetItem(func_134_call(relay.reshape(uop_340.astype('bool'), [8, 3, 14]), relay.reshape(uop_223.astype('bool'), [8, 3, 14]), ), 2)
call_359 = relay.TupleGetItem(func_137_call(relay.reshape(uop_340.astype('bool'), [8, 3, 14]), relay.reshape(uop_223.astype('bool'), [8, 3, 14]), ), 2)
bop_360 = relay.equal(uop_355.astype('bool'), relay.reshape(bop_301.astype('bool'), relay.shape_of(uop_355))) # shape=(8, 3, 14)
bop_363 = relay.equal(uop_357.astype('bool'), relay.reshape(bop_304.astype('bool'), relay.shape_of(uop_357))) # shape=(8, 3, 14)
output = relay.Tuple([call_147,var_148,var_149,bop_160,bop_164,bop_169,bop_177,uop_199,bop_201,call_212,var_213,var_214,uop_226,uop_229,bop_251,bop_255,bop_259,call_266,uop_279,uop_308,bop_330,uop_337,uop_340,bop_344,uop_352,call_358,bop_360,])
output2 = relay.Tuple([call_150,var_148,var_149,bop_163,bop_167,bop_169,bop_180,uop_199,bop_204,call_215,var_213,var_214,uop_228,uop_231,bop_254,bop_258,bop_262,call_267,uop_281,uop_310,bop_333,uop_339,uop_342,bop_347,uop_354,call_359,bop_363,])
func_364 = relay.Function([var_139,var_145,var_148,var_149,var_168,var_172,var_181,var_213,var_214,var_319,var_329,var_343,], output)
mod['func_364'] = func_364
mod = relay.transform.InferType()(mod)
mutated_mod['func_364'] = func_364
mutated_mod = relay.transform.InferType()(mutated_mod)
func_364_call = mutated_mod.get_global_var('func_364')
var_366 = relay.var("var_366", dtype = "uint64", shape = (10, 9))#candidate|366|(10, 9)|var|uint64
var_367 = relay.var("var_367", dtype = "bool", shape = (336,))#candidate|367|(336,)|var|bool
var_368 = relay.var("var_368", dtype = "float32", shape = ())#candidate|368|()|var|float32
var_369 = relay.var("var_369", dtype = "float32", shape = (108,))#candidate|369|(108,)|var|float32
var_370 = relay.var("var_370", dtype = "uint64", shape = (10, 9))#candidate|370|(10, 9)|var|uint64
var_371 = relay.var("var_371", dtype = "float64", shape = (8, 3, 14))#candidate|371|(8, 3, 14)|var|float64
var_372 = relay.var("var_372", dtype = "float64", shape = (8, 3, 14))#candidate|372|(8, 3, 14)|var|float64
var_373 = relay.var("var_373", dtype = "int32", shape = (390,))#candidate|373|(390,)|var|int32
var_374 = relay.var("var_374", dtype = "int32", shape = (220,))#candidate|374|(220,)|var|int32
var_375 = relay.var("var_375", dtype = "float32", shape = (8, 3, 14))#candidate|375|(8, 3, 14)|var|float32
var_376 = relay.var("var_376", dtype = "float32", shape = (8, 3, 14))#candidate|376|(8, 3, 14)|var|float32
var_377 = relay.var("var_377", dtype = "float64", shape = (8, 3, 14))#candidate|377|(8, 3, 14)|var|float64
call_365 = func_364_call(var_366,var_367,var_368,var_369,var_370,var_371,var_372,var_373,var_374,var_375,var_376,var_377,)
output = call_365
func_378 = relay.Function([var_366,var_367,var_368,var_369,var_370,var_371,var_372,var_373,var_374,var_375,var_376,var_377,], output)
mutated_mod['func_378'] = func_378
mutated_mod = relay.transform.InferType()(mutated_mod)
var_380 = relay.var("var_380", dtype = "float64", shape = (10, 15))#candidate|380|(10, 15)|var|float64
uop_381 = relay.sqrt(var_380.astype('float64')) # shape=(10, 15)
output = uop_381
output2 = uop_381
F = relay.Function([var_380,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_380,], output2)
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
input_380= np.array([[-5.224986,6.493848,-8.965785,-5.639791,6.959342,0.287344,1.935478,-0.240385,-2.875306,1.303951,0.056000,2.411474,-9.565634,-6.700605,9.371212],[-6.067920,5.903300,-9.443821,6.126858,-1.105914,1.676073,2.069692,8.986265,4.602650,-3.955311,-7.239444,8.398936,9.379309,7.976577,7.825448],[3.224986,-1.784903,-4.240275,-7.804855,2.453339,-4.914101,6.790966,-2.688408,-9.569000,-6.799772,-5.049317,-0.428624,-8.567375,2.716507,-6.003049],[-1.842089,-8.392493,-7.368646,-8.165555,-9.029550,6.667136,-1.108315,2.138437,-1.277211,-1.255897,-3.211297,1.482942,1.950249,-2.877526,7.705374],[6.173902,-9.460066,-4.808255,-0.412711,4.619290,4.710754,-3.518197,4.540159,-7.498521,4.923288,5.669607,5.520272,-6.685419,-8.719106,8.790478],[3.865283,-1.883648,1.087415,9.382029,9.787327,-6.608046,5.520742,-6.795373,-0.125200,-0.222066,0.428288,4.444442,3.029229,1.652039,-3.207006],[8.737456,-4.280738,8.012436,2.300901,0.107691,-8.189902,5.367272,-2.167648,-4.276331,-6.617981,-1.391169,-6.250706,-9.683615,-9.596532,-7.742944],[-7.751763,-5.799707,2.814603,-5.164845,-1.180101,6.007321,-8.238693,7.532992,-9.922047,0.797442,-9.866972,-4.253300,-7.428760,-0.049509,6.557241],[6.182649,3.670202,-3.086376,-7.156953,2.371129,-2.441853,7.818158,9.235459,-2.931225,-0.302057,-3.546179,3.832465,-0.949917,9.737670,-0.090856],[1.766124,-3.658634,-8.032308,3.003161,-7.184244,0.412675,3.241294,2.914811,7.001422,-9.130910,6.188109,2.104565,2.468046,-4.778790,-0.177651]], dtype='float64')
module1.set_input('var_380', input_380)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_380, )
res3 = intrp3.evaluate()(input_380, )
res4 = intrp4.evaluate()(input_380, )
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
module5.set_input('var_380', input_380)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_380, )
res7 = intrp7.evaluate()(input_380, )
res8 = intrp8.evaluate()(input_380, )
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
module9.set_input('var_380', input_380)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_380, )
res11 = intrp11.evaluate()(input_380, )
res12 = intrp12.evaluate()(input_380, )
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
module13.set_input('var_380', input_380)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_380, )
res15 = intrp15.evaluate()(input_380, )
res16 = intrp16.evaluate()(input_380, )
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
module17.set_input('var_380', input_380)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_380, )
res19 = intrp19.evaluate()(input_380, )
res20 = intrp20.evaluate()(input_380, )
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
module21.set_input('var_380', input_380)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_380, )
res23 = intrp23.evaluate()(input_380, )
res24 = intrp24.evaluate()(input_380, )
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

'''43: TVMFuncCall
42: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
41: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
40: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
39: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
38: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
37: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
36: tvm::transform::Pass::operator()(tvm::IRModule) const
35: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
34: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
30: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
29: tvm::transform::Pass::operator()(tvm::IRModule) const
28: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
27: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
26: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
25: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
24: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
22: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
21: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
20: _ZN3tvm5relay9transform22Devic
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
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