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
var_0 = relay.var("var_0", dtype = "float64", shape = (1, 12))#candidate|0|(1, 12)|var|float64
uop_1 = relay.atan(var_0.astype('float64')) # shape=(1, 12)
var_3 = relay.var("var_3", dtype = "float64", shape = (8, 12))#candidate|3|(8, 12)|var|float64
bop_4 = relay.not_equal(uop_1.astype('bool'), var_3.astype('bool')) # shape=(8, 12)
bop_7 = relay.power(var_0.astype('float64'), bop_4.astype('float64')) # shape=(8, 12)
uop_10 = relay.erf(var_3.astype('float32')) # shape=(8, 12)
uop_12 = relay.log(var_0.astype('float32')) # shape=(1, 12)
bop_14 = relay.right_shift(uop_1.astype('int32'), var_3.astype('int32')) # shape=(8, 12)
bop_17 = relay.mod(var_3.astype('float64'), uop_1.astype('float64')) # shape=(8, 12)
bop_20 = relay.logical_xor(bop_7.astype('uint16'), uop_1.astype('uint16')) # shape=(8, 12)
uop_23 = relay.atan(bop_7.astype('float32')) # shape=(8, 12)
uop_25 = relay.log2(uop_23.astype('float32')) # shape=(8, 12)
var_27 = relay.var("var_27", dtype = "float32", shape = (8, 12))#candidate|27|(8, 12)|var|float32
bop_28 = relay.not_equal(uop_25.astype('bool'), relay.reshape(var_27.astype('bool'), relay.shape_of(uop_25))) # shape=(8, 12)
uop_31 = relay.acosh(bop_28.astype('float64')) # shape=(8, 12)
const_33 = relay.const([[2.541403,6.604917,-3.849030,-0.736100,-1.976232,-3.450538,-5.643512,-3.600952,-9.149281,2.640506,-3.251203,9.795528],[5.108277,-3.054892,-9.250766,-0.734996,-5.802463,8.028037,8.104914,1.981269,5.954181,2.014885,-0.058530,4.134976],[7.636642,4.064752,2.635114,3.707301,-3.493654,5.926386,8.634390,-5.198375,2.043380,-8.587177,6.807443,-1.943937],[9.809538,-7.807323,-6.751275,5.980591,-2.786794,4.406463,2.614221,-9.835573,1.969091,-6.933210,0.086980,-0.322097],[-7.729280,2.111661,-9.440710,5.463866,5.208995,-2.985758,2.444828,-2.588512,-2.391588,2.601565,-5.963101,7.138768],[1.395503,-6.488494,-8.032641,-1.763579,9.458252,-5.052304,3.876566,4.048924,6.518343,-9.142605,8.896541,-7.000329],[2.422626,6.579861,2.204727,7.914781,-0.095046,6.015813,-3.185134,-0.764787,8.644784,2.012153,-4.409322,-4.435820],[-4.359183,-3.492337,4.524659,-8.343838,-5.759148,-8.769221,-8.463618,-7.128197,4.964537,-5.269723,9.811936,0.847481]], dtype = "float32")#candidate|33|(8, 12)|const|float32
bop_34 = relay.power(uop_25.astype('float64'), relay.reshape(const_33.astype('float64'), relay.shape_of(uop_25))) # shape=(8, 12)
bop_37 = relay.right_shift(uop_31.astype('int32'), relay.reshape(uop_23.astype('int32'), relay.shape_of(uop_31))) # shape=(8, 12)
bop_40 = relay.power(bop_28.astype('float64'), relay.reshape(var_27.astype('float64'), relay.shape_of(bop_28))) # shape=(8, 12)
uop_43 = relay.sqrt(bop_40.astype('float32')) # shape=(8, 12)
output = relay.Tuple([uop_10,uop_12,bop_14,bop_17,bop_20,bop_34,bop_37,uop_43,])
output2 = relay.Tuple([uop_10,uop_12,bop_14,bop_17,bop_20,bop_34,bop_37,uop_43,])
func_45 = relay.Function([var_0,var_3,var_27,], output)
mod['func_45'] = func_45
mod = relay.transform.InferType()(mod)
mutated_mod['func_45'] = func_45
mutated_mod = relay.transform.InferType()(mutated_mod)
func_45_call = mutated_mod.get_global_var('func_45')
var_47 = relay.var("var_47", dtype = "float64", shape = (1, 12))#candidate|47|(1, 12)|var|float64
var_48 = relay.var("var_48", dtype = "float64", shape = (8, 12))#candidate|48|(8, 12)|var|float64
var_49 = relay.var("var_49", dtype = "float32", shape = (8, 12))#candidate|49|(8, 12)|var|float32
call_46 = func_45_call(var_47,var_48,var_49,)
output = call_46
func_50 = relay.Function([var_47,var_48,var_49,], output)
mutated_mod['func_50'] = func_50
mutated_mod = relay.transform.InferType()(mutated_mod)
var_52 = relay.var("var_52", dtype = "float64", shape = (11, 11))#candidate|52|(11, 11)|var|float64
uop_53 = relay.log(var_52.astype('float64')) # shape=(11, 11)
uop_55 = relay.sqrt(uop_53.astype('float32')) # shape=(11, 11)
bop_57 = relay.divide(uop_55.astype('float64'), relay.reshape(uop_53.astype('float64'), relay.shape_of(uop_55))) # shape=(11, 11)
bop_60 = relay.not_equal(uop_55.astype('bool'), relay.reshape(var_52.astype('bool'), relay.shape_of(uop_55))) # shape=(11, 11)
uop_63 = relay.atanh(bop_60.astype('float64')) # shape=(11, 11)
var_65 = relay.var("var_65", dtype = "float64", shape = (11, 11))#candidate|65|(11, 11)|var|float64
bop_66 = relay.floor_mod(uop_63.astype('float64'), relay.reshape(var_65.astype('float64'), relay.shape_of(uop_63))) # shape=(11, 11)
uop_69 = relay.sigmoid(uop_55.astype('float32')) # shape=(11, 11)
var_71 = relay.var("var_71", dtype = "bool", shape = (11, 11))#candidate|71|(11, 11)|var|bool
bop_72 = relay.logical_xor(bop_60.astype('int8'), relay.reshape(var_71.astype('int8'), relay.shape_of(bop_60))) # shape=(11, 11)
uop_75 = relay.sin(var_71.astype('float32')) # shape=(11, 11)
uop_77 = relay.sigmoid(bop_60.astype('float32')) # shape=(11, 11)
const_79 = relay.const([[8.147765,0.572661,-1.908707,-5.186168,-9.615668,-0.669484,-4.644419,2.613602,2.343989,0.536666,6.122843],[7.862747,1.048670,8.092025,7.413268,-2.925109,-6.664519,9.329991,0.484226,-9.631939,-3.219215,-0.907712],[-3.486619,-7.556389,-9.606544,-8.458362,-8.655938,-8.360938,7.875235,-6.109983,-6.337863,4.732104,8.124001],[-5.790687,-2.700749,-9.987615,5.861814,1.878605,-5.061977,7.330574,-4.370355,-5.544057,-9.060673,-5.338078],[-7.461783,-8.225287,-3.675437,3.713230,-2.914849,-8.402190,7.116324,4.569900,5.103121,-7.773458,1.648242],[-2.794771,2.383211,-9.573610,-5.605351,-3.035826,4.845542,8.568600,-2.588482,7.982513,-4.714322,-0.687740],[-4.198229,-0.690863,-5.949535,-1.779405,9.773072,9.002232,5.642189,-8.287727,4.075648,2.270549,6.709600],[-7.423974,4.664429,-5.157098,5.197857,7.372201,6.001175,-5.313773,-5.064808,6.482403,8.185248,2.510926],[-6.102880,-5.495764,-2.806427,-4.257582,5.630971,2.886005,5.695388,-2.968925,-2.258853,-6.725448,-1.506182],[-3.532876,0.233305,-1.325074,-4.984303,-6.409978,1.511624,-7.013881,-9.078916,4.861481,4.691174,-9.698246],[-3.703496,-4.840153,0.187524,-4.953813,0.504246,-5.576832,-4.126390,7.305751,-6.205769,3.321756,1.686736]], dtype = "float64")#candidate|79|(11, 11)|const|float64
bop_80 = relay.right_shift(bop_66.astype('uint32'), relay.reshape(const_79.astype('uint32'), relay.shape_of(bop_66))) # shape=(11, 11)
func_45_call = mod.get_global_var('func_45')
func_50_call = mutated_mod.get_global_var('func_50')
var_84 = relay.var("var_84", dtype = "float64", shape = (12,))#candidate|84|(12,)|var|float64
var_85 = relay.var("var_85", dtype = "float64", shape = (96,))#candidate|85|(96,)|var|float64
call_83 = relay.TupleGetItem(func_45_call(relay.reshape(var_84.astype('float64'), [1, 12]), relay.reshape(var_85.astype('float64'), [8, 12]), relay.reshape(var_85.astype('float32'), [8, 12]), ), 4)
call_86 = relay.TupleGetItem(func_50_call(relay.reshape(var_84.astype('float64'), [1, 12]), relay.reshape(var_85.astype('float64'), [8, 12]), relay.reshape(var_85.astype('float32'), [8, 12]), ), 4)
uop_87 = relay.asin(uop_75.astype('float32')) # shape=(11, 11)
var_89 = relay.var("var_89", dtype = "bool", shape = (11, 11))#candidate|89|(11, 11)|var|bool
bop_90 = relay.multiply(var_71.astype('uint64'), relay.reshape(var_89.astype('uint64'), relay.shape_of(var_71))) # shape=(11, 11)
uop_93 = relay.tan(uop_75.astype('float32')) # shape=(11, 11)
bop_95 = relay.floor_divide(uop_87.astype('float32'), relay.reshape(var_52.astype('float32'), relay.shape_of(uop_87))) # shape=(11, 11)
bop_98 = relay.maximum(uop_87.astype('int32'), relay.reshape(bop_95.astype('int32'), relay.shape_of(uop_87))) # shape=(11, 11)
var_101 = relay.var("var_101", dtype = "float64", shape = (11, 11))#candidate|101|(11, 11)|var|float64
bop_102 = relay.bitwise_xor(uop_53.astype('int32'), relay.reshape(var_101.astype('int32'), relay.shape_of(uop_53))) # shape=(11, 11)
bop_105 = relay.power(bop_80.astype('float32'), relay.reshape(uop_69.astype('float32'), relay.shape_of(bop_80))) # shape=(11, 11)
func_45_call = mod.get_global_var('func_45')
func_50_call = mutated_mod.get_global_var('func_50')
call_108 = relay.TupleGetItem(func_45_call(relay.reshape(var_84.astype('float64'), [1, 12]), relay.reshape(call_83.astype('float64'), [8, 12]), relay.reshape(var_85.astype('float32'), [8, 12]), ), 4)
call_109 = relay.TupleGetItem(func_50_call(relay.reshape(var_84.astype('float64'), [1, 12]), relay.reshape(call_83.astype('float64'), [8, 12]), relay.reshape(var_85.astype('float32'), [8, 12]), ), 4)
bop_110 = relay.floor_mod(uop_93.astype('float32'), relay.reshape(bop_95.astype('float32'), relay.shape_of(uop_93))) # shape=(11, 11)
var_113 = relay.var("var_113", dtype = "int32", shape = (11, 11))#candidate|113|(11, 11)|var|int32
bop_114 = relay.not_equal(bop_102.astype('bool'), relay.reshape(var_113.astype('bool'), relay.shape_of(bop_102))) # shape=(11, 11)
uop_117 = relay.exp(var_89.astype('float64')) # shape=(11, 11)
output = relay.Tuple([bop_57,bop_72,uop_77,call_83,var_84,var_85,bop_90,bop_98,bop_105,call_108,bop_110,bop_114,uop_117,])
output2 = relay.Tuple([bop_57,bop_72,uop_77,call_86,var_84,var_85,bop_90,bop_98,bop_105,call_109,bop_110,bop_114,uop_117,])
func_119 = relay.Function([var_52,var_65,var_71,var_84,var_85,var_89,var_101,var_113,], output)
mod['func_119'] = func_119
mod = relay.transform.InferType()(mod)
var_120 = relay.var("var_120", dtype = "float64", shape = (11, 11))#candidate|120|(11, 11)|var|float64
var_121 = relay.var("var_121", dtype = "float64", shape = (11, 11))#candidate|121|(11, 11)|var|float64
var_122 = relay.var("var_122", dtype = "bool", shape = (11, 11))#candidate|122|(11, 11)|var|bool
var_123 = relay.var("var_123", dtype = "float64", shape = (12,))#candidate|123|(12,)|var|float64
var_124 = relay.var("var_124", dtype = "float64", shape = (96,))#candidate|124|(96,)|var|float64
var_125 = relay.var("var_125", dtype = "bool", shape = (11, 11))#candidate|125|(11, 11)|var|bool
var_126 = relay.var("var_126", dtype = "float64", shape = (11, 11))#candidate|126|(11, 11)|var|float64
var_127 = relay.var("var_127", dtype = "int32", shape = (11, 11))#candidate|127|(11, 11)|var|int32
output = func_119(var_120,var_121,var_122,var_123,var_124,var_125,var_126,var_127,)
func_128 = relay.Function([var_120,var_121,var_122,var_123,var_124,var_125,var_126,var_127,], output)
mutated_mod['func_128'] = func_128
mutated_mod = relay.transform.InferType()(mutated_mod)
const_130 = relay.const(-8.614006, dtype = "float32")#candidate|130|()|const|float32
var_131 = relay.var("var_131", dtype = "float32", shape = (15, 6))#candidate|131|(15, 6)|var|float32
bop_132 = relay.multiply(const_130.astype('float32'), var_131.astype('float32')) # shape=(15, 6)
var_135 = relay.var("var_135", dtype = "float32", shape = (15, 6))#candidate|135|(15, 6)|var|float32
bop_136 = relay.add(var_131.astype('uint32'), relay.reshape(var_135.astype('uint32'), relay.shape_of(var_131))) # shape=(15, 6)
var_139 = relay.var("var_139", dtype = "float32", shape = (15, 6))#candidate|139|(15, 6)|var|float32
bop_140 = relay.logical_xor(var_135.astype('int64'), relay.reshape(var_139.astype('int64'), relay.shape_of(var_135))) # shape=(15, 6)
uop_143 = relay.erf(var_139.astype('float64')) # shape=(15, 6)
bop_145 = relay.add(uop_143.astype('int8'), const_130.astype('int8')) # shape=(15, 6)
bop_148 = relay.bitwise_xor(uop_143.astype('uint8'), relay.reshape(bop_136.astype('uint8'), relay.shape_of(uop_143))) # shape=(15, 6)
bop_151 = relay.not_equal(bop_148.astype('bool'), relay.reshape(var_139.astype('bool'), relay.shape_of(bop_148))) # shape=(15, 6)
uop_154 = relay.asin(const_130.astype('float32')) # shape=()
uop_156 = relay.cos(bop_132.astype('float32')) # shape=(15, 6)
bop_158 = relay.mod(bop_132.astype('float32'), relay.reshape(var_135.astype('float32'), relay.shape_of(bop_132))) # shape=(15, 6)
uop_161 = relay.exp(uop_143.astype('float32')) # shape=(15, 6)
bop_163 = relay.less_equal(uop_161.astype('bool'), relay.reshape(bop_140.astype('bool'), relay.shape_of(uop_161))) # shape=(15, 6)
bop_166 = relay.minimum(bop_163.astype('int16'), relay.reshape(uop_143.astype('int16'), relay.shape_of(bop_163))) # shape=(15, 6)
uop_169 = relay.sin(uop_161.astype('float32')) # shape=(15, 6)
var_171 = relay.var("var_171", dtype = "float32", shape = (15, 6))#candidate|171|(15, 6)|var|float32
bop_172 = relay.multiply(uop_169.astype('int16'), relay.reshape(var_171.astype('int16'), relay.shape_of(uop_169))) # shape=(15, 6)
func_119_call = mod.get_global_var('func_119')
func_128_call = mutated_mod.get_global_var('func_128')
const_176 = relay.const([-4.479218,8.347220,0.226171,-3.255614,-0.278146,-2.260178,-0.793491,-5.486208,6.555786,0.048133,0.336087,-6.373917,-1.506323,-2.708087,2.125977,-5.855373,3.356426,-7.880488,6.791677,-5.322103,-8.045596,4.704158,-2.106322,-6.563998,-1.031680,-7.525337,-9.545774,8.408562,3.212672,9.869802,1.101930,-5.452958,1.877989,-8.201603,2.637682,-6.604210,2.129715,3.480887,-4.597304,1.391323,0.072349,-9.011755,-3.246144,-6.293578,9.457263,-7.669036,-0.576386,-9.027895,1.722683,-5.149451,8.223062,7.242590,-8.772065,-0.356032,-0.129046,4.929745,-2.757863,-7.192271,-6.287235,-6.117096,-9.688188,-8.768113,-9.385515,-0.222054,6.077581,-2.080714,-9.489428,0.187853,-7.802532,5.527977,-8.177979,-9.245950,-3.591085,2.607185,-5.191672,-3.439920,3.654785,6.628144,4.142137,0.875293,-2.300584,-8.120629,8.993784,3.503348,5.576659,0.561775,4.418355,-7.105217,-1.857113,-1.871959,-7.264591,4.044592,-4.395862,0.380847,-0.456494,-6.015775,-4.916637,9.608189,0.081431,-7.861274,7.516534,-8.842028,0.875901,-2.561694,-6.772068,-6.536347,-7.743930,-2.985978,-1.342639,-6.883686,6.174171,0.921083,1.624857,0.544213,5.008421,-7.733878,7.180528,-3.730749,0.024813,-3.893980,4.970453], dtype = "float64")#candidate|176|(121,)|const|float64
var_177 = relay.var("var_177", dtype = "float64", shape = (3, 4))#candidate|177|(3, 4)|var|float64
var_178 = relay.var("var_178", dtype = "float64", shape = (16, 6))#candidate|178|(16, 6)|var|float64
call_175 = relay.TupleGetItem(func_119_call(relay.reshape(const_176.astype('float64'), [11, 11]), relay.reshape(const_176.astype('float64'), [11, 11]), relay.reshape(const_176.astype('bool'), [11, 11]), relay.reshape(var_177.astype('float64'), [12,]), relay.reshape(var_178.astype('float64'), [96,]), relay.reshape(const_176.astype('bool'), [11, 11]), relay.reshape(const_176.astype('float64'), [11, 11]), relay.reshape(const_176.astype('int32'), [11, 11]), ), 6)
call_179 = relay.TupleGetItem(func_128_call(relay.reshape(const_176.astype('float64'), [11, 11]), relay.reshape(const_176.astype('float64'), [11, 11]), relay.reshape(const_176.astype('bool'), [11, 11]), relay.reshape(var_177.astype('float64'), [12,]), relay.reshape(var_178.astype('float64'), [96,]), relay.reshape(const_176.astype('bool'), [11, 11]), relay.reshape(const_176.astype('float64'), [11, 11]), relay.reshape(const_176.astype('int32'), [11, 11]), ), 6)
bop_180 = relay.less_equal(bop_172.astype('bool'), const_130.astype('bool')) # shape=(15, 6)
output = relay.Tuple([bop_145,bop_151,uop_154,uop_156,bop_158,bop_166,call_175,const_176,var_177,var_178,bop_180,])
output2 = relay.Tuple([bop_145,bop_151,uop_154,uop_156,bop_158,bop_166,call_179,const_176,var_177,var_178,bop_180,])
func_183 = relay.Function([var_131,var_135,var_139,var_171,var_177,var_178,], output)
mod['func_183'] = func_183
mod = relay.transform.InferType()(mod)
mutated_mod['func_183'] = func_183
mutated_mod = relay.transform.InferType()(mutated_mod)
func_183_call = mutated_mod.get_global_var('func_183')
var_185 = relay.var("var_185", dtype = "float32", shape = (15, 6))#candidate|185|(15, 6)|var|float32
var_186 = relay.var("var_186", dtype = "float32", shape = (15, 6))#candidate|186|(15, 6)|var|float32
var_187 = relay.var("var_187", dtype = "float32", shape = (15, 6))#candidate|187|(15, 6)|var|float32
var_188 = relay.var("var_188", dtype = "float32", shape = (15, 6))#candidate|188|(15, 6)|var|float32
var_189 = relay.var("var_189", dtype = "float64", shape = (3, 4))#candidate|189|(3, 4)|var|float64
var_190 = relay.var("var_190", dtype = "float64", shape = (16, 6))#candidate|190|(16, 6)|var|float64
call_184 = func_183_call(var_185,var_186,var_187,var_188,var_189,var_190,)
output = call_184
func_191 = relay.Function([var_185,var_186,var_187,var_188,var_189,var_190,], output)
mutated_mod['func_191'] = func_191
mutated_mod = relay.transform.InferType()(mutated_mod)
var_193 = relay.var("var_193", dtype = "float64", shape = ())#candidate|193|()|var|float64
var_194 = relay.var("var_194", dtype = "float64", shape = (12, 13))#candidate|194|(12, 13)|var|float64
bop_195 = relay.floor_divide(var_193.astype('float64'), var_194.astype('float64')) # shape=(12, 13)
var_198 = relay.var("var_198", dtype = "float64", shape = (14,))#candidate|198|(14,)|var|float64
bop_199 = relay.logical_and(var_193.astype('bool'), var_198.astype('bool')) # shape=(14,)
bop_202 = relay.greater_equal(bop_195.astype('bool'), var_193.astype('bool')) # shape=(12, 13)
uop_205 = relay.exp(var_198.astype('float32')) # shape=(14,)
bop_207 = relay.maximum(uop_205.astype('int8'), var_193.astype('int8')) # shape=(14,)
var_210 = relay.var("var_210", dtype = "float32", shape = (14,))#candidate|210|(14,)|var|float32
bop_211 = relay.subtract(uop_205.astype('int64'), relay.reshape(var_210.astype('int64'), relay.shape_of(uop_205))) # shape=(14,)
output = relay.Tuple([bop_199,bop_202,bop_207,bop_211,])
output2 = relay.Tuple([bop_199,bop_202,bop_207,bop_211,])
func_214 = relay.Function([var_193,var_194,var_198,var_210,], output)
mod['func_214'] = func_214
mod = relay.transform.InferType()(mod)
mutated_mod['func_214'] = func_214
mutated_mod = relay.transform.InferType()(mutated_mod)
func_214_call = mutated_mod.get_global_var('func_214')
var_216 = relay.var("var_216", dtype = "float64", shape = ())#candidate|216|()|var|float64
var_217 = relay.var("var_217", dtype = "float64", shape = (12, 13))#candidate|217|(12, 13)|var|float64
var_218 = relay.var("var_218", dtype = "float64", shape = (14,))#candidate|218|(14,)|var|float64
var_219 = relay.var("var_219", dtype = "float32", shape = (14,))#candidate|219|(14,)|var|float32
call_215 = func_214_call(var_216,var_217,var_218,var_219,)
output = call_215
func_220 = relay.Function([var_216,var_217,var_218,var_219,], output)
mutated_mod['func_220'] = func_220
mutated_mod = relay.transform.InferType()(mutated_mod)
var_222 = relay.var("var_222", dtype = "float64", shape = (6, 6))#candidate|222|(6, 6)|var|float64
uop_223 = relay.exp(var_222.astype('float64')) # shape=(6, 6)
var_225 = relay.var("var_225", dtype = "float64", shape = (6, 6))#candidate|225|(6, 6)|var|float64
bop_226 = relay.logical_xor(uop_223.astype('uint8'), relay.reshape(var_225.astype('uint8'), relay.shape_of(uop_223))) # shape=(6, 6)
bop_229 = relay.right_shift(uop_223.astype('int32'), relay.reshape(var_225.astype('int32'), relay.shape_of(uop_223))) # shape=(6, 6)
uop_232 = relay.acosh(uop_223.astype('float64')) # shape=(6, 6)
var_234 = relay.var("var_234", dtype = "int32", shape = (6, 6))#candidate|234|(6, 6)|var|int32
bop_235 = relay.not_equal(bop_229.astype('bool'), relay.reshape(var_234.astype('bool'), relay.shape_of(bop_229))) # shape=(6, 6)
uop_238 = relay.cos(var_222.astype('float32')) # shape=(6, 6)
bop_240 = relay.divide(uop_232.astype('float64'), relay.reshape(bop_235.astype('float64'), relay.shape_of(uop_232))) # shape=(6, 6)
uop_243 = relay.log(bop_235.astype('float64')) # shape=(6, 6)
output = relay.Tuple([bop_226,uop_238,bop_240,uop_243,])
output2 = relay.Tuple([bop_226,uop_238,bop_240,uop_243,])
func_245 = relay.Function([var_222,var_225,var_234,], output)
mod['func_245'] = func_245
mod = relay.transform.InferType()(mod)
var_246 = relay.var("var_246", dtype = "float64", shape = (6, 6))#candidate|246|(6, 6)|var|float64
var_247 = relay.var("var_247", dtype = "float64", shape = (6, 6))#candidate|247|(6, 6)|var|float64
var_248 = relay.var("var_248", dtype = "int32", shape = (6, 6))#candidate|248|(6, 6)|var|int32
output = func_245(var_246,var_247,var_248,)
func_249 = relay.Function([var_246,var_247,var_248,], output)
mutated_mod['func_249'] = func_249
mutated_mod = relay.transform.InferType()(mutated_mod)
var_251 = relay.var("var_251", dtype = "float32", shape = (16, 15, 12))#candidate|251|(16, 15, 12)|var|float32
uop_252 = relay.acos(var_251.astype('float32')) # shape=(16, 15, 12)
var_254 = relay.var("var_254", dtype = "float32", shape = (16, 15, 12))#candidate|254|(16, 15, 12)|var|float32
bop_255 = relay.floor_mod(uop_252.astype('float32'), relay.reshape(var_254.astype('float32'), relay.shape_of(uop_252))) # shape=(16, 15, 12)
uop_258 = relay.acos(var_251.astype('float64')) # shape=(16, 15, 12)
bop_260 = relay.greater(bop_255.astype('bool'), relay.reshape(var_251.astype('bool'), relay.shape_of(bop_255))) # shape=(16, 15, 12)
bop_263 = relay.greater_equal(var_251.astype('bool'), relay.reshape(var_254.astype('bool'), relay.shape_of(var_251))) # shape=(16, 15, 12)
uop_266 = relay.log(bop_260.astype('float32')) # shape=(16, 15, 12)
uop_268 = relay.cosh(bop_255.astype('float64')) # shape=(16, 15, 12)
bop_270 = relay.subtract(uop_268.astype('uint64'), relay.reshape(bop_263.astype('uint64'), relay.shape_of(uop_268))) # shape=(16, 15, 12)
uop_273 = relay.sin(var_254.astype('float64')) # shape=(16, 15, 12)
output = relay.Tuple([uop_258,uop_266,bop_270,uop_273,])
output2 = relay.Tuple([uop_258,uop_266,bop_270,uop_273,])
func_275 = relay.Function([var_251,var_254,], output)
mod['func_275'] = func_275
mod = relay.transform.InferType()(mod)
var_276 = relay.var("var_276", dtype = "float32", shape = (16, 15, 12))#candidate|276|(16, 15, 12)|var|float32
var_277 = relay.var("var_277", dtype = "float32", shape = (16, 15, 12))#candidate|277|(16, 15, 12)|var|float32
output = func_275(var_276,var_277,)
func_278 = relay.Function([var_276,var_277,], output)
mutated_mod['func_278'] = func_278
mutated_mod = relay.transform.InferType()(mutated_mod)
var_280 = relay.var("var_280", dtype = "float64", shape = ())#candidate|280|()|var|float64
uop_281 = relay.acosh(var_280.astype('float64')) # shape=()
bop_283 = relay.floor_divide(uop_281.astype('float64'), var_280.astype('float64')) # shape=()
bop_286 = relay.bitwise_and(var_280.astype('uint32'), uop_281.astype('uint32')) # shape=()
bop_289 = relay.right_shift(bop_283.astype('int64'), var_280.astype('int64')) # shape=()
func_245_call = mod.get_global_var('func_245')
func_249_call = mutated_mod.get_global_var('func_249')
var_293 = relay.var("var_293", dtype = "float64", shape = (36,))#candidate|293|(36,)|var|float64
call_292 = relay.TupleGetItem(func_245_call(relay.reshape(var_293.astype('float64'), [6, 6]), relay.reshape(var_293.astype('float64'), [6, 6]), relay.reshape(var_293.astype('int32'), [6, 6]), ), 0)
call_294 = relay.TupleGetItem(func_249_call(relay.reshape(var_293.astype('float64'), [6, 6]), relay.reshape(var_293.astype('float64'), [6, 6]), relay.reshape(var_293.astype('int32'), [6, 6]), ), 0)
uop_295 = relay.sin(uop_281.astype('float32')) # shape=()
bop_297 = relay.not_equal(uop_295.astype('bool'), var_293.astype('bool')) # shape=(36,)
uop_300 = relay.sigmoid(uop_295.astype('float32')) # shape=()
func_119_call = mod.get_global_var('func_119')
func_128_call = mutated_mod.get_global_var('func_128')
const_303 = relay.const([9.886336,5.806264,-0.112410,-6.669828,0.212549,-4.088347,-7.194960,-1.817301,8.357571,-3.729783,-9.539880,8.632683,9.539329,5.226195,-1.461970,9.603451,5.653051,-5.847504,7.691847,3.447964,-7.878840,-6.251499,-5.048050,6.940605,3.755838,-5.404987,-1.417393,-6.370596,2.418079,3.006434,9.489170,3.056552,3.576539,-8.826073,8.775333,3.381754,-5.846955,-2.068081,-5.956607,-4.806181,3.535660,3.860325,6.118230,1.592849,-5.374954,-5.955137,4.336423,-5.400645,3.054576,3.154302,0.287141,1.004851,1.890023,-1.822423,6.288479,1.428955,9.514357,7.857082,-2.193079,2.847798,6.829099,-9.857197,-7.232002,4.918272,3.214990,-0.257098,6.367594,-8.785812,5.297548,-7.047764,-1.441051,-6.538209,-6.857579,-6.788078,-0.423907,0.105500,2.586198,6.366134,2.423259,4.103898,1.797547,3.349908,8.976870,9.844549,-0.044602,-6.963704,9.183194,-0.631637,-4.080065,-0.563228,5.460278,-8.359659,4.522980,-8.260158,-5.496432,-7.552756,5.033501,-8.590752,7.958424,7.986647,4.715357,-5.694262,-0.966221,-9.769074,5.252685,-9.616695,8.286322,5.062191,-2.879531,7.925679,-4.836460,-8.145732,0.483193,-8.297277,3.152412,-9.593896,1.446481,6.473776,3.070463,-6.798206,8.273182], dtype = "float64")#candidate|303|(121,)|const|float64
const_304 = relay.const([-5.872616,-2.787735,-5.257023,7.271084,3.751836,3.153175,6.931611,-6.277380,-4.053061,3.557681,2.220960,5.967093], dtype = "float64")#candidate|304|(12,)|const|float64
const_305 = relay.const([-4.729890,0.405988,-8.501304,-1.039283,7.752325,-5.324705,-2.700063,-6.490317,-7.961268,-6.109132,-6.253629,-1.133458,-3.387970,0.399910,-0.010972,4.369784,-6.543481,-7.391030,-3.875934,-0.262550,-7.807273,-3.887060,-7.963927,3.728010,-4.300441,8.596708,4.687343,-6.878510,-4.257615,3.323514,-0.455045,5.573149,0.577573,-5.060108,-1.612082,1.786996,1.420729,4.772481,-3.149582,-4.607979,7.988328,3.558752,3.088414,-2.789988,6.987413,9.545301,-2.077028,3.199819,-5.740582,-9.609873,-9.096188,5.154248,5.266668,8.419068,0.617235,-1.838580,6.647209,9.388824,0.350931,-6.953064,2.126825,5.398069,6.791926,3.955917,-1.444980,-7.484274,0.312623,-8.393365,1.595557,-8.787525,7.628741,8.908735,0.137846,-7.224619,-5.953116,7.647286,-1.528923,-0.872242,7.160137,3.782595,5.032288,-0.235262,3.412197,5.396356,-4.902875,-8.929432,0.604873,6.637960,-4.613810,7.444708,-0.268604,9.980859,3.963696,-9.589282,3.330535,-0.158849], dtype = "float64")#candidate|305|(96,)|const|float64
call_302 = relay.TupleGetItem(func_119_call(relay.reshape(const_303.astype('float64'), [11, 11]), relay.reshape(const_303.astype('float64'), [11, 11]), relay.reshape(const_303.astype('bool'), [11, 11]), relay.reshape(const_304.astype('float64'), [12,]), relay.reshape(const_305.astype('float64'), [96,]), relay.reshape(const_303.astype('bool'), [11, 11]), relay.reshape(const_303.astype('float64'), [11, 11]), relay.reshape(const_303.astype('int32'), [11, 11]), ), 1)
call_306 = relay.TupleGetItem(func_128_call(relay.reshape(const_303.astype('float64'), [11, 11]), relay.reshape(const_303.astype('float64'), [11, 11]), relay.reshape(const_303.astype('bool'), [11, 11]), relay.reshape(const_304.astype('float64'), [12,]), relay.reshape(const_305.astype('float64'), [96,]), relay.reshape(const_303.astype('bool'), [11, 11]), relay.reshape(const_303.astype('float64'), [11, 11]), relay.reshape(const_303.astype('int32'), [11, 11]), ), 1)
bop_307 = relay.bitwise_or(bop_289.astype('int64'), uop_281.astype('int64')) # shape=()
uop_310 = relay.atanh(uop_300.astype('float32')) # shape=()
bop_312 = relay.maximum(uop_300.astype('uint16'), bop_297.astype('uint16')) # shape=(36,)
uop_315 = relay.erf(uop_310.astype('float64')) # shape=()
func_245_call = mod.get_global_var('func_245')
func_249_call = mutated_mod.get_global_var('func_249')
call_317 = relay.TupleGetItem(func_245_call(relay.reshape(bop_312.astype('float64'), [6, 6]), relay.reshape(bop_297.astype('float64'), [6, 6]), relay.reshape(var_293.astype('int32'), [6, 6]), ), 1)
call_318 = relay.TupleGetItem(func_249_call(relay.reshape(bop_312.astype('float64'), [6, 6]), relay.reshape(bop_297.astype('float64'), [6, 6]), relay.reshape(var_293.astype('int32'), [6, 6]), ), 1)
var_319 = relay.var("var_319", dtype = "float64", shape = (15, 7))#candidate|319|(15, 7)|var|float64
bop_320 = relay.bitwise_and(uop_315.astype('uint32'), var_319.astype('uint32')) # shape=(15, 7)
bop_323 = relay.logical_and(uop_310.astype('bool'), bop_320.astype('bool')) # shape=(15, 7)
const_326 = relay.const(0.076315, dtype = "float32")#candidate|326|()|const|float32
bop_327 = relay.logical_or(uop_300.astype('bool'), const_326.astype('bool')) # shape=()
uop_330 = relay.acos(uop_315.astype('float32')) # shape=()
bop_332 = relay.floor_mod(bop_323.astype('float64'), relay.reshape(bop_320.astype('float64'), relay.shape_of(bop_323))) # shape=(15, 7)
bop_335 = relay.greater_equal(uop_330.astype('bool'), bop_286.astype('bool')) # shape=()
bop_338 = relay.minimum(uop_330.astype('float64'), const_304.astype('float64')) # shape=(12,)
uop_341 = relay.log10(bop_338.astype('float64')) # shape=(12,)
output = relay.Tuple([call_292,call_302,const_303,const_305,bop_307,bop_312,call_317,bop_327,bop_332,bop_335,uop_341,])
output2 = relay.Tuple([call_294,call_306,const_303,const_305,bop_307,bop_312,call_318,bop_327,bop_332,bop_335,uop_341,])
func_343 = relay.Function([var_280,var_293,var_319,], output)
mod['func_343'] = func_343
mod = relay.transform.InferType()(mod)
var_344 = relay.var("var_344", dtype = "float64", shape = ())#candidate|344|()|var|float64
var_345 = relay.var("var_345", dtype = "float64", shape = (36,))#candidate|345|(36,)|var|float64
var_346 = relay.var("var_346", dtype = "float64", shape = (15, 7))#candidate|346|(15, 7)|var|float64
output = func_343(var_344,var_345,var_346,)
func_347 = relay.Function([var_344,var_345,var_346,], output)
mutated_mod['func_347'] = func_347
mutated_mod = relay.transform.InferType()(mutated_mod)
var_349 = relay.var("var_349", dtype = "float64", shape = ())#candidate|349|()|var|float64
uop_350 = relay.asin(var_349.astype('float64')) # shape=()
bop_352 = relay.not_equal(uop_350.astype('bool'), var_349.astype('bool')) # shape=()
uop_355 = relay.sin(bop_352.astype('float32')) # shape=()
uop_357 = relay.exp(uop_355.astype('float64')) # shape=()
uop_359 = relay.sqrt(uop_355.astype('float32')) # shape=()
output = relay.Tuple([uop_357,uop_359,])
output2 = relay.Tuple([uop_357,uop_359,])
F = relay.Function([var_349,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_349,], output2)
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
input_349= np.array(-5.343248, dtype='float64')
module1.set_input('var_349', input_349)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_349, )
res3 = intrp3.evaluate()(input_349, )
res4 = intrp4.evaluate()(input_349, )
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
module5.set_input('var_349', input_349)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_349, )
res7 = intrp7.evaluate()(input_349, )
res8 = intrp8.evaluate()(input_349, )
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
module9.set_input('var_349', input_349)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_349, )
res11 = intrp11.evaluate()(input_349, )
res12 = intrp12.evaluate()(input_349, )
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
module13.set_input('var_349', input_349)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_349, )
res15 = intrp15.evaluate()(input_349, )
res16 = intrp16.evaluate()(input_349, )
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
module17.set_input('var_349', input_349)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_349, )
res19 = intrp19.evaluate()(input_349, )
res20 = intrp20.evaluate()(input_349, )
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
module21.set_input('var_349', input_349)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_349, )
res23 = intrp23.evaluate()(input_349, )
res24 = intrp24.evaluate()(input_349, )
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