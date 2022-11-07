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
var_0 = relay.var("var_0", dtype = "uint8", shape = (9, 4))#candidate|0|(9, 4)|var|uint8
var_1 = relay.var("var_1", dtype = "uint8", shape = (9, 4))#candidate|1|(9, 4)|var|uint8
bop_2 = relay.subtract(var_0.astype('uint8'), relay.reshape(var_1.astype('uint8'), relay.shape_of(var_0))) # shape=(9, 4)
uop_5 = relay.erf(var_1.astype('float64')) # shape=(9, 4)
uop_7 = relay.sqrt(uop_5.astype('float32')) # shape=(9, 4)
var_9 = relay.var("var_9", dtype = "float32", shape = (9, 4))#candidate|9|(9, 4)|var|float32
bop_10 = relay.mod(uop_7.astype('float64'), relay.reshape(var_9.astype('float64'), relay.shape_of(uop_7))) # shape=(9, 4)
output = relay.Tuple([bop_2,bop_10,])
output2 = relay.Tuple([bop_2,bop_10,])
func_13 = relay.Function([var_0,var_1,var_9,], output)
mod['func_13'] = func_13
mod = relay.transform.InferType()(mod)
mutated_mod['func_13'] = func_13
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13_call = mutated_mod.get_global_var('func_13')
var_15 = relay.var("var_15", dtype = "uint8", shape = (9, 4))#candidate|15|(9, 4)|var|uint8
var_16 = relay.var("var_16", dtype = "uint8", shape = (9, 4))#candidate|16|(9, 4)|var|uint8
var_17 = relay.var("var_17", dtype = "float32", shape = (9, 4))#candidate|17|(9, 4)|var|float32
call_14 = func_13_call(var_15,var_16,var_17,)
output = call_14
func_18 = relay.Function([var_15,var_16,var_17,], output)
mutated_mod['func_18'] = func_18
mutated_mod = relay.transform.InferType()(mutated_mod)
var_20 = relay.var("var_20", dtype = "uint8", shape = (5,))#candidate|20|(5,)|var|uint8
const_21 = relay.const([-1,-10,1,-4,8], dtype = "uint8")#candidate|21|(5,)|const|uint8
bop_22 = relay.multiply(var_20.astype('uint8'), relay.reshape(const_21.astype('uint8'), relay.shape_of(var_20))) # shape=(5,)
const_25 = relay.const([-7,1,-3,7,-2], dtype = "uint8")#candidate|25|(5,)|const|uint8
bop_26 = relay.subtract(var_20.astype('float32'), relay.reshape(const_25.astype('float32'), relay.shape_of(var_20))) # shape=(5,)
bop_29 = relay.greater(const_25.astype('bool'), relay.reshape(bop_22.astype('bool'), relay.shape_of(const_25))) # shape=(5,)
output = relay.Tuple([bop_26,bop_29,])
output2 = relay.Tuple([bop_26,bop_29,])
func_32 = relay.Function([var_20,], output)
mod['func_32'] = func_32
mod = relay.transform.InferType()(mod)
mutated_mod['func_32'] = func_32
mutated_mod = relay.transform.InferType()(mutated_mod)
var_33 = relay.var("var_33", dtype = "uint8", shape = (5,))#candidate|33|(5,)|var|uint8
func_32_call = mutated_mod.get_global_var('func_32')
call_34 = func_32_call(var_33)
output = call_34
func_35 = relay.Function([var_33], output)
mutated_mod['func_35'] = func_35
mutated_mod = relay.transform.InferType()(mutated_mod)
const_37 = relay.const(-3.689174, dtype = "float32")#candidate|37|()|const|float32
const_38 = relay.const([-2.605301,-3.890442,2.545759,4.928028,-5.185562,6.051323,9.520842,-6.902895,6.573458,8.900631,6.820309,-5.242921], dtype = "float32")#candidate|38|(12,)|const|float32
bop_39 = relay.floor_divide(const_37.astype('float32'), const_38.astype('float32')) # shape=(12,)
var_42 = relay.var("var_42", dtype = "float32", shape = (12,))#candidate|42|(12,)|var|float32
bop_43 = relay.greater(bop_39.astype('bool'), relay.reshape(var_42.astype('bool'), relay.shape_of(bop_39))) # shape=(12,)
bop_46 = relay.divide(var_42.astype('float32'), relay.reshape(const_38.astype('float32'), relay.shape_of(var_42))) # shape=(12,)
func_32_call = mod.get_global_var('func_32')
func_35_call = mutated_mod.get_global_var('func_35')
const_50 = relay.const([-5,-1,8,-1,4], dtype = "uint8")#candidate|50|(5,)|const|uint8
call_49 = relay.TupleGetItem(func_32_call(relay.reshape(const_50.astype('uint8'), [5,])), 0)
call_51 = relay.TupleGetItem(func_35_call(relay.reshape(const_50.astype('uint8'), [5,])), 0)
uop_52 = relay.asinh(const_38.astype('float32')) # shape=(12,)
const_54 = relay.const([True,False,True,False,False,False,False,True,True,True,True,False], dtype = "bool")#candidate|54|(12,)|const|bool
bop_55 = relay.mod(bop_43.astype('float32'), relay.reshape(const_54.astype('float32'), relay.shape_of(bop_43))) # shape=(12,)
func_13_call = mod.get_global_var('func_13')
func_18_call = mutated_mod.get_global_var('func_18')
const_59 = relay.const([-8,-5,3,9,-3,1,4,-10,-2,3,-10,4,-5,7,9,-5,-6,-5,2,-2,-5,-2,9,-9,5,-6,4,-1,6,8,7,-2,4,1,-3,1], dtype = "uint8")#candidate|59|(36,)|const|uint8
call_58 = relay.TupleGetItem(func_13_call(relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('float32'), [9, 4]), ), 1)
call_60 = relay.TupleGetItem(func_18_call(relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('float32'), [9, 4]), ), 1)
var_61 = relay.var("var_61", dtype = "float32", shape = (12,))#candidate|61|(12,)|var|float32
bop_62 = relay.mod(uop_52.astype('float32'), relay.reshape(var_61.astype('float32'), relay.shape_of(uop_52))) # shape=(12,)
uop_65 = relay.cos(uop_52.astype('float32')) # shape=(12,)
var_67 = relay.var("var_67", dtype = "float32", shape = (12,))#candidate|67|(12,)|var|float32
bop_68 = relay.divide(uop_65.astype('float64'), relay.reshape(var_67.astype('float64'), relay.shape_of(uop_65))) # shape=(12,)
bop_71 = relay.bitwise_or(uop_65.astype('uint8'), relay.reshape(bop_43.astype('uint8'), relay.shape_of(uop_65))) # shape=(12,)
uop_74 = relay.erf(bop_68.astype('float64')) # shape=(12,)
func_13_call = mod.get_global_var('func_13')
func_18_call = mutated_mod.get_global_var('func_18')
call_76 = relay.TupleGetItem(func_13_call(relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('float32'), [9, 4]), ), 0)
call_77 = relay.TupleGetItem(func_18_call(relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('uint8'), [9, 4]), relay.reshape(const_59.astype('float32'), [9, 4]), ), 0)
const_78 = relay.const([-6.230589,-6.707345,2.652548,-1.500145,1.647328,-1.268326,-6.331481,8.616891,8.476383,3.769695,9.339668,3.444557], dtype = "float32")#candidate|78|(12,)|const|float32
bop_79 = relay.subtract(uop_52.astype('uint32'), relay.reshape(const_78.astype('uint32'), relay.shape_of(uop_52))) # shape=(12,)
bop_82 = relay.logical_or(uop_74.astype('bool'), relay.reshape(bop_71.astype('bool'), relay.shape_of(uop_74))) # shape=(12,)
output = relay.Tuple([bop_46,call_49,const_50,bop_55,call_58,const_59,bop_62,call_76,bop_79,bop_82,])
output2 = relay.Tuple([bop_46,call_51,const_50,bop_55,call_60,const_59,bop_62,call_77,bop_79,bop_82,])
func_85 = relay.Function([var_42,var_61,var_67,], output)
mod['func_85'] = func_85
mod = relay.transform.InferType()(mod)
var_86 = relay.var("var_86", dtype = "float32", shape = (12,))#candidate|86|(12,)|var|float32
var_87 = relay.var("var_87", dtype = "float32", shape = (12,))#candidate|87|(12,)|var|float32
var_88 = relay.var("var_88", dtype = "float32", shape = (12,))#candidate|88|(12,)|var|float32
output = func_85(var_86,var_87,var_88,)
func_89 = relay.Function([var_86,var_87,var_88,], output)
mutated_mod['func_89'] = func_89
mutated_mod = relay.transform.InferType()(mutated_mod)
var_91 = relay.var("var_91", dtype = "float64", shape = (1,))#candidate|91|(1,)|var|float64
uop_92 = relay.exp(var_91.astype('float64')) # shape=(1,)
bop_94 = relay.logical_and(uop_92.astype('bool'), relay.reshape(var_91.astype('bool'), relay.shape_of(uop_92))) # shape=(1,)
uop_97 = relay.sin(var_91.astype('float32')) # shape=(1,)
bop_99 = relay.bitwise_or(bop_94.astype('int8'), relay.reshape(uop_92.astype('int8'), relay.shape_of(bop_94))) # shape=(1,)
bop_102 = relay.logical_and(uop_97.astype('bool'), relay.reshape(uop_92.astype('bool'), relay.shape_of(uop_97))) # shape=(1,)
bop_105 = relay.bitwise_or(bop_102.astype('int8'), relay.reshape(uop_92.astype('int8'), relay.shape_of(bop_102))) # shape=(1,)
output = relay.Tuple([bop_99,bop_105,])
output2 = relay.Tuple([bop_99,bop_105,])
func_108 = relay.Function([var_91,], output)
mod['func_108'] = func_108
mod = relay.transform.InferType()(mod)
mutated_mod['func_108'] = func_108
mutated_mod = relay.transform.InferType()(mutated_mod)
var_109 = relay.var("var_109", dtype = "float64", shape = (1,))#candidate|109|(1,)|var|float64
func_108_call = mutated_mod.get_global_var('func_108')
call_110 = func_108_call(var_109)
output = call_110
func_111 = relay.Function([var_109], output)
mutated_mod['func_111'] = func_111
mutated_mod = relay.transform.InferType()(mutated_mod)
var_113 = relay.var("var_113", dtype = "float32", shape = (10, 6))#candidate|113|(10, 6)|var|float32
uop_114 = relay.rsqrt(var_113.astype('float32')) # shape=(10, 6)
bop_116 = relay.less_equal(var_113.astype('bool'), relay.reshape(uop_114.astype('bool'), relay.shape_of(var_113))) # shape=(10, 6)
func_13_call = mod.get_global_var('func_13')
func_18_call = mutated_mod.get_global_var('func_18')
const_120 = relay.const([-3,4,-4,10,-5,5,-8,1,-3,-4,-7,-7,-4,-5,2,-10,1,-10,-6,-3,7,4,8,-7,6,-3,6,5,7,-6,6,-5,-1,-1,7,-9], dtype = "uint8")#candidate|120|(36,)|const|uint8
call_119 = relay.TupleGetItem(func_13_call(relay.reshape(const_120.astype('uint8'), [9, 4]), relay.reshape(const_120.astype('uint8'), [9, 4]), relay.reshape(const_120.astype('float32'), [9, 4]), ), 1)
call_121 = relay.TupleGetItem(func_18_call(relay.reshape(const_120.astype('uint8'), [9, 4]), relay.reshape(const_120.astype('uint8'), [9, 4]), relay.reshape(const_120.astype('float32'), [9, 4]), ), 1)
bop_122 = relay.bitwise_xor(const_120.astype('uint32'), relay.reshape(call_119.astype('uint32'), relay.shape_of(const_120))) # shape=(36,)
bop_125 = relay.bitwise_xor(const_120.astype('uint32'), relay.reshape(call_121.astype('uint32'), relay.shape_of(const_120))) # shape=(36,)
var_126 = relay.var("var_126", dtype = "uint8", shape = (36,))#candidate|126|(36,)|var|uint8
bop_127 = relay.power(const_120.astype('float64'), relay.reshape(var_126.astype('float64'), relay.shape_of(const_120))) # shape=(36,)
bop_130 = relay.less(uop_114.astype('bool'), relay.reshape(var_113.astype('bool'), relay.shape_of(uop_114))) # shape=(10, 6)
uop_133 = relay.erf(uop_114.astype('float32')) # shape=(10, 6)
var_135 = relay.var("var_135", dtype = "uint8", shape = (36,))#candidate|135|(36,)|var|uint8
bop_136 = relay.floor_divide(const_120.astype('float64'), relay.reshape(var_135.astype('float64'), relay.shape_of(const_120))) # shape=(36,)
func_32_call = mod.get_global_var('func_32')
func_35_call = mutated_mod.get_global_var('func_35')
const_140 = relay.const([-10,-9,1,-5,10], dtype = "uint8")#candidate|140|(5,)|const|uint8
call_139 = relay.TupleGetItem(func_32_call(relay.reshape(const_140.astype('uint8'), [5,])), 1)
call_141 = relay.TupleGetItem(func_35_call(relay.reshape(const_140.astype('uint8'), [5,])), 1)
bop_142 = relay.right_shift(uop_133.astype('uint64'), relay.reshape(bop_130.astype('uint64'), relay.shape_of(uop_133))) # shape=(10, 6)
bop_145 = relay.right_shift(bop_127.astype('int16'), relay.reshape(var_135.astype('int16'), relay.shape_of(bop_127))) # shape=(36,)
var_148 = relay.var("var_148", dtype = "float32", shape = (10, 6))#candidate|148|(10, 6)|var|float32
bop_149 = relay.bitwise_xor(uop_114.astype('int8'), relay.reshape(var_148.astype('int8'), relay.shape_of(uop_114))) # shape=(10, 6)
uop_152 = relay.rsqrt(uop_133.astype('float32')) # shape=(10, 6)
uop_154 = relay.cosh(bop_142.astype('float32')) # shape=(10, 6)
bop_156 = relay.right_shift(uop_154.astype('int16'), relay.reshape(uop_152.astype('int16'), relay.shape_of(uop_154))) # shape=(10, 6)
output = relay.Tuple([bop_116,bop_122,bop_136,call_139,const_140,bop_145,bop_149,bop_156,])
output2 = relay.Tuple([bop_116,bop_125,bop_136,call_141,const_140,bop_145,bop_149,bop_156,])
func_159 = relay.Function([var_113,var_126,var_135,var_148,], output)
mod['func_159'] = func_159
mod = relay.transform.InferType()(mod)
var_160 = relay.var("var_160", dtype = "float32", shape = (10, 6))#candidate|160|(10, 6)|var|float32
var_161 = relay.var("var_161", dtype = "uint8", shape = (36,))#candidate|161|(36,)|var|uint8
var_162 = relay.var("var_162", dtype = "uint8", shape = (36,))#candidate|162|(36,)|var|uint8
var_163 = relay.var("var_163", dtype = "float32", shape = (10, 6))#candidate|163|(10, 6)|var|float32
output = func_159(var_160,var_161,var_162,var_163,)
func_164 = relay.Function([var_160,var_161,var_162,var_163,], output)
mutated_mod['func_164'] = func_164
mutated_mod = relay.transform.InferType()(mutated_mod)
var_166 = relay.var("var_166", dtype = "float32", shape = (3, 8, 3))#candidate|166|(3, 8, 3)|var|float32
const_167 = relay.const([[[-2.921255,3.713119,4.728276],[-7.933399,-3.967595,-0.766369],[-9.458503,-9.162308,-9.213290],[5.953693,9.617855,3.951999],[3.231324,0.868338,5.068411],[-1.513275,-8.744829,0.420321],[7.061861,1.891469,-4.683996],[6.041427,-7.020169,6.141994]],[[2.554138,-6.882529,4.154988],[-8.851257,-6.176395,4.339441],[-8.262194,2.944425,4.447550],[-9.061459,-6.996205,3.651816],[6.312970,7.965023,-6.644989],[-9.696244,2.565714,-6.962032],[9.487686,1.985797,-8.945670],[6.196191,-1.057420,-0.944721]],[[2.327821,-6.451655,-9.783785],[2.361973,4.231386,-8.535749],[3.881593,0.832043,7.313090],[9.995664,-3.738940,-1.262870],[9.457099,6.053053,-3.175279],[-9.678078,-5.484958,1.460853],[-6.429951,-5.361200,8.909171],[5.956603,-1.968194,9.813731]]], dtype = "float32")#candidate|167|(3, 8, 3)|const|float32
bop_168 = relay.greater_equal(var_166.astype('bool'), relay.reshape(const_167.astype('bool'), relay.shape_of(var_166))) # shape=(3, 8, 3)
var_171 = relay.var("var_171", dtype = "float32", shape = (3, 8, 3))#candidate|171|(3, 8, 3)|var|float32
bop_172 = relay.right_shift(var_166.astype('int32'), relay.reshape(var_171.astype('int32'), relay.shape_of(var_166))) # shape=(3, 8, 3)
const_175 = relay.const([[[5.391822,3.471660,0.200710],[-6.244473,-2.750235,-4.573362],[-9.979944,1.070856,6.196479],[7.142377,4.827336,-3.818457],[3.931584,9.637091,-2.718636],[-5.941044,5.339894,9.292914],[7.235980,2.862174,4.624174],[-5.585525,-3.665007,9.921289]],[[-2.854100,-2.896750,-6.611312],[-3.782109,5.708856,-4.274869],[-5.793696,4.492813,-4.586393],[8.468705,8.643532,2.483391],[-4.856391,-8.326291,2.244435],[-5.177431,1.035207,1.377013],[5.417437,0.787517,4.114734],[-8.668877,-1.825108,7.689229]],[[-4.764630,-8.872900,5.688574],[-4.849865,4.018834,-1.334841],[-4.275944,7.758946,7.359936],[5.043610,-0.553348,9.825756],[-5.673299,-1.505859,-5.179675],[0.513660,-5.867683,-7.509146],[0.126690,3.775488,3.008912],[3.615197,-8.133941,0.088741]]], dtype = "float32")#candidate|175|(3, 8, 3)|const|float32
bop_176 = relay.minimum(const_167.astype('uint8'), relay.reshape(const_175.astype('uint8'), relay.shape_of(const_167))) # shape=(3, 8, 3)
uop_179 = relay.acos(bop_172.astype('float64')) # shape=(3, 8, 3)
uop_181 = relay.sigmoid(uop_179.astype('float32')) # shape=(3, 8, 3)
bop_183 = relay.less(uop_179.astype('bool'), relay.reshape(var_166.astype('bool'), relay.shape_of(uop_179))) # shape=(3, 8, 3)
bop_186 = relay.minimum(uop_181.astype('float64'), relay.reshape(bop_168.astype('float64'), relay.shape_of(uop_181))) # shape=(3, 8, 3)
bop_189 = relay.equal(bop_172.astype('bool'), relay.reshape(bop_183.astype('bool'), relay.shape_of(bop_172))) # shape=(3, 8, 3)
bop_192 = relay.power(bop_186.astype('float64'), relay.reshape(uop_181.astype('float64'), relay.shape_of(bop_186))) # shape=(3, 8, 3)
bop_195 = relay.left_shift(bop_183.astype('uint8'), relay.reshape(bop_172.astype('uint8'), relay.shape_of(bop_183))) # shape=(3, 8, 3)
uop_198 = relay.cos(bop_168.astype('float64')) # shape=(3, 8, 3)
bop_200 = relay.divide(bop_186.astype('float64'), relay.reshape(var_166.astype('float64'), relay.shape_of(bop_186))) # shape=(3, 8, 3)
uop_203 = relay.atanh(uop_181.astype('float32')) # shape=(3, 8, 3)
uop_205 = relay.acos(bop_200.astype('float32')) # shape=(3, 8, 3)
uop_207 = relay.sinh(uop_198.astype('float64')) # shape=(3, 8, 3)
uop_209 = relay.rsqrt(uop_203.astype('float32')) # shape=(3, 8, 3)
uop_211 = relay.cos(uop_209.astype('float32')) # shape=(3, 8, 3)
bop_213 = relay.logical_or(uop_211.astype('bool'), relay.reshape(bop_183.astype('bool'), relay.shape_of(uop_211))) # shape=(3, 8, 3)
bop_216 = relay.right_shift(uop_207.astype('uint16'), relay.reshape(uop_211.astype('uint16'), relay.shape_of(uop_207))) # shape=(3, 8, 3)
uop_219 = relay.erf(bop_216.astype('float32')) # shape=(3, 8, 3)
func_13_call = mod.get_global_var('func_13')
func_18_call = mutated_mod.get_global_var('func_18')
const_222 = relay.const([8,8,10,10,6,5,8,1,-5,-5,5,-10,8,7,10,-1,-1,-8,2,2,-1,-3,7,8,9,5,-1,-8,-4,4,-2,-2,-3,3,-10,5], dtype = "uint8")#candidate|222|(36,)|const|uint8
call_221 = relay.TupleGetItem(func_13_call(relay.reshape(const_222.astype('uint8'), [9, 4]), relay.reshape(const_222.astype('uint8'), [9, 4]), relay.reshape(const_222.astype('float32'), [9, 4]), ), 1)
call_223 = relay.TupleGetItem(func_18_call(relay.reshape(const_222.astype('uint8'), [9, 4]), relay.reshape(const_222.astype('uint8'), [9, 4]), relay.reshape(const_222.astype('float32'), [9, 4]), ), 1)
var_224 = relay.var("var_224", dtype = "float32", shape = (3, 8, 3))#candidate|224|(3, 8, 3)|var|float32
bop_225 = relay.power(uop_219.astype('float32'), relay.reshape(var_224.astype('float32'), relay.shape_of(uop_219))) # shape=(3, 8, 3)
uop_228 = relay.acosh(uop_209.astype('float32')) # shape=(3, 8, 3)
uop_230 = relay.erf(uop_211.astype('float32')) # shape=(3, 8, 3)
bop_232 = relay.equal(uop_219.astype('bool'), relay.reshape(uop_203.astype('bool'), relay.shape_of(uop_219))) # shape=(3, 8, 3)
output = relay.Tuple([bop_176,bop_189,bop_192,bop_195,uop_205,bop_213,call_221,const_222,bop_225,uop_228,uop_230,bop_232,])
output2 = relay.Tuple([bop_176,bop_189,bop_192,bop_195,uop_205,bop_213,call_223,const_222,bop_225,uop_228,uop_230,bop_232,])
func_235 = relay.Function([var_166,var_171,var_224,], output)
mod['func_235'] = func_235
mod = relay.transform.InferType()(mod)
var_236 = relay.var("var_236", dtype = "float32", shape = (3, 8, 3))#candidate|236|(3, 8, 3)|var|float32
var_237 = relay.var("var_237", dtype = "float32", shape = (3, 8, 3))#candidate|237|(3, 8, 3)|var|float32
var_238 = relay.var("var_238", dtype = "float32", shape = (3, 8, 3))#candidate|238|(3, 8, 3)|var|float32
output = func_235(var_236,var_237,var_238,)
func_239 = relay.Function([var_236,var_237,var_238,], output)
mutated_mod['func_239'] = func_239
mutated_mod = relay.transform.InferType()(mutated_mod)
var_241 = relay.var("var_241", dtype = "int64", shape = (11, 8))#candidate|241|(11, 8)|var|int64
var_242 = relay.var("var_242", dtype = "int64", shape = (11, 8))#candidate|242|(11, 8)|var|int64
bop_243 = relay.less_equal(var_241.astype('bool'), relay.reshape(var_242.astype('bool'), relay.shape_of(var_241))) # shape=(11, 8)
uop_246 = relay.log2(var_242.astype('float32')) # shape=(11, 8)
output = relay.Tuple([bop_243,uop_246,])
output2 = relay.Tuple([bop_243,uop_246,])
func_248 = relay.Function([var_241,var_242,], output)
mod['func_248'] = func_248
mod = relay.transform.InferType()(mod)
var_249 = relay.var("var_249", dtype = "int64", shape = (11, 8))#candidate|249|(11, 8)|var|int64
var_250 = relay.var("var_250", dtype = "int64", shape = (11, 8))#candidate|250|(11, 8)|var|int64
output = func_248(var_249,var_250,)
func_251 = relay.Function([var_249,var_250,], output)
mutated_mod['func_251'] = func_251
mutated_mod = relay.transform.InferType()(mutated_mod)
const_253 = relay.const([[7.805717,1.016425,2.083808,-0.641193,-4.319989,7.859942,-0.319788,-6.561244,9.493343],[-3.048534,3.540847,3.870701,-9.526043,5.192910,0.448395,-8.786964,9.893476,6.806867],[9.159731,-6.305595,7.214503,0.553071,-0.587896,-3.706235,-1.533121,-8.217397,-5.824783],[6.271963,-5.156368,-5.180930,2.532719,-4.523590,0.606165,2.911544,7.540805,-0.341435],[-1.741871,4.769034,1.278334,-8.968938,-7.774049,-7.973031,8.800290,4.862541,-9.719247],[9.792835,-3.948938,-9.562783,0.074267,-6.596447,2.699525,9.168251,7.026263,6.970788],[-6.527404,-4.358245,-9.835537,8.198030,-0.276797,7.083899,4.532699,1.770686,-8.006049],[6.211535,0.019830,-1.191508,-4.885891,-7.429621,-5.606437,-2.257287,-2.393591,3.876640],[6.267413,-3.196230,-6.514949,0.644604,-3.449245,7.699910,-4.893044,-1.505031,8.268575],[5.734913,7.973233,3.462977,8.622247,-3.456631,-8.261144,-8.380154,2.931819,-8.577641],[-1.155876,6.961579,-3.683432,8.133561,7.956475,-0.900081,-4.369881,-7.468083,-8.099845],[-6.489708,2.233849,1.128781,-5.420450,-0.445876,-9.556018,2.698926,-4.190317,9.289902],[-3.789098,7.517937,-0.075111,-6.573057,9.153280,-2.699176,-6.076353,9.206728,-0.925942],[1.605823,-2.691423,-8.714857,-6.293386,9.068008,-7.368604,1.114162,1.133189,-6.268172],[3.474786,0.732290,3.252638,-4.175001,-1.261914,4.097643,-5.525798,1.548676,-0.710936]], dtype = "float64")#candidate|253|(15, 9)|const|float64
var_254 = relay.var("var_254", dtype = "float64", shape = (15, 9))#candidate|254|(15, 9)|var|float64
bop_255 = relay.floor_mod(const_253.astype('float64'), relay.reshape(var_254.astype('float64'), relay.shape_of(const_253))) # shape=(15, 9)
bop_258 = relay.equal(bop_255.astype('bool'), relay.reshape(var_254.astype('bool'), relay.shape_of(bop_255))) # shape=(15, 9)
var_261 = relay.var("var_261", dtype = "float64", shape = (15, 9))#candidate|261|(15, 9)|var|float64
bop_262 = relay.power(const_253.astype('float32'), relay.reshape(var_261.astype('float32'), relay.shape_of(const_253))) # shape=(15, 9)
const_265 = relay.const([[-4.989866,4.034092,5.215728,-1.098576,-7.448028,-2.951832,-5.432294,4.822945,9.765219],[-8.455475,2.969179,-5.430100,-5.990772,0.122487,-4.596563,9.593250,7.176365,5.864879],[5.522203,-5.900684,0.757356,1.021786,-3.672593,4.978462,-3.068971,2.571681,5.836455],[-1.238447,0.902336,-7.522433,7.620145,-6.064143,-2.573873,0.443739,1.851713,-6.784284],[4.453150,-3.159863,-9.652763,3.278702,5.257953,7.458413,-5.682179,7.734471,0.380613],[8.806101,3.317954,6.625144,3.857188,8.352927,-3.249194,8.501974,8.879162,8.017872],[1.083818,8.621581,3.182047,-3.134199,-0.128675,1.477880,8.404564,-0.383164,-9.053359],[-3.283795,3.706592,-3.572609,3.266815,2.431137,-3.521018,3.614614,3.414682,-3.334303],[5.692728,-8.126188,-0.334318,6.954218,-5.773803,8.941830,-3.607533,-0.140861,2.218857],[0.525671,-7.032695,-6.382974,-1.856534,-8.240103,-3.798918,-6.869784,3.712268,3.826095],[4.368032,4.045750,0.319090,-6.151333,0.468963,5.592399,-8.530722,2.157410,0.580206],[1.810886,7.406082,6.146251,-3.927167,2.453120,-8.604563,-6.858727,-2.150370,0.755687],[-4.564080,2.392344,-7.822810,-1.218377,1.614728,7.785280,5.227379,-5.896155,-3.459600],[-6.290341,-3.330050,5.230117,1.525823,-8.057913,0.493180,2.883025,4.911232,5.438770],[3.607957,-3.615671,-7.417381,3.996912,-5.509175,-7.860355,8.210085,-2.145350,5.790839]], dtype = "float64")#candidate|265|(15, 9)|const|float64
bop_266 = relay.not_equal(var_261.astype('bool'), relay.reshape(const_265.astype('bool'), relay.shape_of(var_261))) # shape=(15, 9)
bop_269 = relay.not_equal(var_254.astype('bool'), relay.reshape(const_253.astype('bool'), relay.shape_of(var_254))) # shape=(15, 9)
output = relay.Tuple([bop_258,bop_262,bop_266,bop_269,])
output2 = relay.Tuple([bop_258,bop_262,bop_266,bop_269,])
func_272 = relay.Function([var_254,var_261,], output)
mod['func_272'] = func_272
mod = relay.transform.InferType()(mod)
mutated_mod['func_272'] = func_272
mutated_mod = relay.transform.InferType()(mutated_mod)
func_272_call = mutated_mod.get_global_var('func_272')
var_274 = relay.var("var_274", dtype = "float64", shape = (15, 9))#candidate|274|(15, 9)|var|float64
var_275 = relay.var("var_275", dtype = "float64", shape = (15, 9))#candidate|275|(15, 9)|var|float64
call_273 = func_272_call(var_274,var_275,)
output = call_273
func_276 = relay.Function([var_274,var_275,], output)
mutated_mod['func_276'] = func_276
mutated_mod = relay.transform.InferType()(mutated_mod)
var_278 = relay.var("var_278", dtype = "int16", shape = (14,))#candidate|278|(14,)|var|int16
const_279 = relay.const([-3,-5,3,10,-9,2,-5,-1,-10,10,9,10,4,7], dtype = "int16")#candidate|279|(14,)|const|int16
bop_280 = relay.bitwise_and(var_278.astype('int16'), relay.reshape(const_279.astype('int16'), relay.shape_of(var_278))) # shape=(14,)
uop_283 = relay.sin(const_279.astype('float64')) # shape=(14,)
const_285 = relay.const([4.423001,-8.275701,-0.474979,3.909961,-0.988683,0.582873,-6.977435,-3.175202,-6.209571,0.812075,1.363017,-8.302754,-0.455797,-3.207567], dtype = "float64")#candidate|285|(14,)|const|float64
bop_286 = relay.right_shift(uop_283.astype('int32'), relay.reshape(const_285.astype('int32'), relay.shape_of(uop_283))) # shape=(14,)
output = relay.Tuple([bop_280,bop_286,])
output2 = relay.Tuple([bop_280,bop_286,])
func_289 = relay.Function([var_278,], output)
mod['func_289'] = func_289
mod = relay.transform.InferType()(mod)
mutated_mod['func_289'] = func_289
mutated_mod = relay.transform.InferType()(mutated_mod)
var_290 = relay.var("var_290", dtype = "int16", shape = (14,))#candidate|290|(14,)|var|int16
func_289_call = mutated_mod.get_global_var('func_289')
call_291 = func_289_call(var_290)
output = call_291
func_292 = relay.Function([var_290], output)
mutated_mod['func_292'] = func_292
mutated_mod = relay.transform.InferType()(mutated_mod)
const_294 = relay.const(0.892578, dtype = "float64")#candidate|294|()|const|float64
uop_295 = relay.cos(const_294.astype('float64')) # shape=()
output = uop_295
output2 = uop_295
func_297 = relay.Function([], output)
mod['func_297'] = func_297
mod = relay.transform.InferType()(mod)
output = func_297()
func_298 = relay.Function([], output)
mutated_mod['func_298'] = func_298
mutated_mod = relay.transform.InferType()(mutated_mod)
var_299 = relay.var("var_299", dtype = "float64", shape = (16,))#candidate|299|(16,)|var|float64
var_300 = relay.var("var_300", dtype = "float64", shape = (16,))#candidate|300|(16,)|var|float64
bop_301 = relay.less(var_299.astype('bool'), relay.reshape(var_300.astype('bool'), relay.shape_of(var_299))) # shape=(16,)
var_304 = relay.var("var_304", dtype = "float64", shape = (16,))#candidate|304|(16,)|var|float64
bop_305 = relay.not_equal(var_299.astype('bool'), relay.reshape(var_304.astype('bool'), relay.shape_of(var_299))) # shape=(16,)
output = relay.Tuple([bop_301,bop_305,])
output2 = relay.Tuple([bop_301,bop_305,])
func_308 = relay.Function([var_299,var_300,var_304,], output)
mod['func_308'] = func_308
mod = relay.transform.InferType()(mod)
mutated_mod['func_308'] = func_308
mutated_mod = relay.transform.InferType()(mutated_mod)
func_308_call = mutated_mod.get_global_var('func_308')
var_310 = relay.var("var_310", dtype = "float64", shape = (16,))#candidate|310|(16,)|var|float64
var_311 = relay.var("var_311", dtype = "float64", shape = (16,))#candidate|311|(16,)|var|float64
var_312 = relay.var("var_312", dtype = "float64", shape = (16,))#candidate|312|(16,)|var|float64
call_309 = func_308_call(var_310,var_311,var_312,)
output = call_309
func_313 = relay.Function([var_310,var_311,var_312,], output)
mutated_mod['func_313'] = func_313
mutated_mod = relay.transform.InferType()(mutated_mod)
const_315 = relay.const(2.406652, dtype = "float32")#candidate|315|()|const|float32
uop_316 = relay.sin(const_315.astype('float32')) # shape=()
uop_318 = relay.log2(const_315.astype('float64')) # shape=()
uop_320 = relay.asin(const_315.astype('float32')) # shape=()
bop_322 = relay.logical_and(const_315.astype('bool'), uop_316.astype('bool')) # shape=()
bop_325 = relay.subtract(const_315.astype('int64'), uop_318.astype('int64')) # shape=()
output = relay.Tuple([uop_320,bop_322,bop_325,])
output2 = relay.Tuple([uop_320,bop_322,bop_325,])
func_328 = relay.Function([], output)
mod['func_328'] = func_328
mod = relay.transform.InferType()(mod)
output = func_328()
func_329 = relay.Function([], output)
mutated_mod['func_329'] = func_329
mutated_mod = relay.transform.InferType()(mutated_mod)
var_330 = relay.var("var_330", dtype = "uint8", shape = (2, 8))#candidate|330|(2, 8)|var|uint8
var_331 = relay.var("var_331", dtype = "uint8", shape = (2, 8))#candidate|331|(2, 8)|var|uint8
bop_332 = relay.greater_equal(var_330.astype('bool'), relay.reshape(var_331.astype('bool'), relay.shape_of(var_330))) # shape=(2, 8)
output = bop_332
output2 = bop_332
func_335 = relay.Function([var_330,var_331,], output)
mod['func_335'] = func_335
mod = relay.transform.InferType()(mod)
var_336 = relay.var("var_336", dtype = "uint8", shape = (2, 8))#candidate|336|(2, 8)|var|uint8
var_337 = relay.var("var_337", dtype = "uint8", shape = (2, 8))#candidate|337|(2, 8)|var|uint8
output = func_335(var_336,var_337,)
func_338 = relay.Function([var_336,var_337,], output)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
var_340 = relay.var("var_340", dtype = "float64", shape = (8,))#candidate|340|(8,)|var|float64
uop_341 = relay.log10(var_340.astype('float64')) # shape=(8,)
var_343 = relay.var("var_343", dtype = "float64", shape = (8,))#candidate|343|(8,)|var|float64
bop_344 = relay.floor_mod(var_340.astype('float32'), relay.reshape(var_343.astype('float32'), relay.shape_of(var_340))) # shape=(8,)
bop_347 = relay.less_equal(var_340.astype('bool'), relay.reshape(bop_344.astype('bool'), relay.shape_of(var_340))) # shape=(8,)
func_289_call = mod.get_global_var('func_289')
func_292_call = mutated_mod.get_global_var('func_292')
var_351 = relay.var("var_351", dtype = "int16", shape = (14,))#candidate|351|(14,)|var|int16
call_350 = relay.TupleGetItem(func_289_call(relay.reshape(var_351.astype('int16'), [14,])), 0)
call_352 = relay.TupleGetItem(func_292_call(relay.reshape(var_351.astype('int16'), [14,])), 0)
var_353 = relay.var("var_353", dtype = "int16", shape = (14,))#candidate|353|(14,)|var|int16
bop_354 = relay.logical_and(var_351.astype('bool'), relay.reshape(var_353.astype('bool'), relay.shape_of(var_351))) # shape=(14,)
uop_357 = relay.cosh(uop_341.astype('float32')) # shape=(8,)
bop_359 = relay.bitwise_or(uop_341.astype('int32'), relay.reshape(uop_357.astype('int32'), relay.shape_of(uop_341))) # shape=(8,)
uop_362 = relay.acos(bop_347.astype('float32')) # shape=(8,)
uop_364 = relay.rsqrt(uop_362.astype('float64')) # shape=(8,)
bop_366 = relay.divide(uop_362.astype('float64'), relay.reshape(var_343.astype('float64'), relay.shape_of(uop_362))) # shape=(8,)
func_297_call = mod.get_global_var('func_297')
func_298_call = mutated_mod.get_global_var('func_298')
call_369 = func_297_call()
call_370 = func_297_call()
const_371 = relay.const([-1.205121,-8.407867,-8.481278,6.617301,0.588832,1.840693,6.177590,-9.663840], dtype = "float64")#candidate|371|(8,)|const|float64
bop_372 = relay.less(uop_341.astype('bool'), relay.reshape(const_371.astype('bool'), relay.shape_of(uop_341))) # shape=(8,)
var_375 = relay.var("var_375", dtype = "float32", shape = (8,))#candidate|375|(8,)|var|float32
bop_376 = relay.equal(uop_357.astype('bool'), relay.reshape(var_375.astype('bool'), relay.shape_of(uop_357))) # shape=(8,)
uop_379 = relay.erf(uop_362.astype('float32')) # shape=(8,)
uop_381 = relay.sin(uop_379.astype('float32')) # shape=(8,)
bop_383 = relay.divide(uop_357.astype('float64'), relay.reshape(bop_359.astype('float64'), relay.shape_of(uop_357))) # shape=(8,)
uop_386 = relay.cos(uop_379.astype('float32')) # shape=(8,)
var_388 = relay.var("var_388", dtype = "float64", shape = (8,))#candidate|388|(8,)|var|float64
bop_389 = relay.greater(bop_366.astype('bool'), relay.reshape(var_388.astype('bool'), relay.shape_of(bop_366))) # shape=(8,)
uop_392 = relay.log10(uop_386.astype('float64')) # shape=(8,)
bop_394 = relay.divide(uop_386.astype('float32'), relay.reshape(bop_383.astype('float32'), relay.shape_of(uop_386))) # shape=(8,)
func_297_call = mod.get_global_var('func_297')
func_298_call = mutated_mod.get_global_var('func_298')
call_397 = func_297_call()
call_398 = func_297_call()
output = relay.Tuple([call_350,bop_354,uop_364,call_369,bop_372,bop_376,uop_381,bop_389,uop_392,bop_394,call_397,])
output2 = relay.Tuple([call_352,bop_354,uop_364,call_370,bop_372,bop_376,uop_381,bop_389,uop_392,bop_394,call_398,])
func_399 = relay.Function([var_340,var_343,var_351,var_353,var_375,var_388,], output)
mod['func_399'] = func_399
mod = relay.transform.InferType()(mod)
var_400 = relay.var("var_400", dtype = "float64", shape = (8,))#candidate|400|(8,)|var|float64
var_401 = relay.var("var_401", dtype = "float64", shape = (8,))#candidate|401|(8,)|var|float64
var_402 = relay.var("var_402", dtype = "int16", shape = (14,))#candidate|402|(14,)|var|int16
var_403 = relay.var("var_403", dtype = "int16", shape = (14,))#candidate|403|(14,)|var|int16
var_404 = relay.var("var_404", dtype = "float32", shape = (8,))#candidate|404|(8,)|var|float32
var_405 = relay.var("var_405", dtype = "float64", shape = (8,))#candidate|405|(8,)|var|float64
output = func_399(var_400,var_401,var_402,var_403,var_404,var_405,)
func_406 = relay.Function([var_400,var_401,var_402,var_403,var_404,var_405,], output)
mutated_mod['func_406'] = func_406
mutated_mod = relay.transform.InferType()(mutated_mod)
var_408 = relay.var("var_408", dtype = "float32", shape = (13,))#candidate|408|(13,)|var|float32
uop_409 = relay.log10(var_408.astype('float32')) # shape=(13,)
output = uop_409
output2 = uop_409
F = relay.Function([var_408,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_408,], output2)
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
input_408= np.array([3.260094,-4.193477,4.189731,-2.611360,-9.392389,-4.266680,9.556022,-9.786790,6.736009,-6.099499,-8.875350,-6.429450,1.838979], dtype='float32')
module1.set_input('var_408', input_408)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_408, )
res3 = intrp3.evaluate()(input_408, )
res4 = intrp4.evaluate()(input_408, )
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
module5.set_input('var_408', input_408)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_408, )
res7 = intrp7.evaluate()(input_408, )
res8 = intrp8.evaluate()(input_408, )
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
module9.set_input('var_408', input_408)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_408, )
res11 = intrp11.evaluate()(input_408, )
res12 = intrp12.evaluate()(input_408, )
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
module13.set_input('var_408', input_408)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_408, )
res15 = intrp15.evaluate()(input_408, )
res16 = intrp16.evaluate()(input_408, )
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
module17.set_input('var_408', input_408)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_408, )
res19 = intrp19.evaluate()(input_408, )
res20 = intrp20.evaluate()(input_408, )
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
module21.set_input('var_408', input_408)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_408, )
res23 = intrp23.evaluate()(input_408, )
res24 = intrp24.evaluate()(input_408, )
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