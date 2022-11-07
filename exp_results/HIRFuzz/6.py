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
var_16 = relay.var("var_16", dtype = "float32", shape = (13,))#candidate|16|(13,)|var|float32
var_17 = relay.var("var_17", dtype = "float32", shape = (13,))#candidate|17|(13,)|var|float32
bop_18 = relay.floor_mod(var_16.astype('float32'), relay.reshape(var_17.astype('float32'), relay.shape_of(var_16))) # shape=(13,)
output = relay.Tuple([bop_18,])
output2 = relay.Tuple([bop_18,])
func_26 = relay.Function([var_16,var_17,], output)
mod['func_26'] = func_26
mod = relay.transform.InferType()(mod)
mutated_mod['func_26'] = func_26
mutated_mod = relay.transform.InferType()(mutated_mod)
func_26_call = mutated_mod.get_global_var('func_26')
var_28 = relay.var("var_28", dtype = "float32", shape = (13,))#candidate|28|(13,)|var|float32
var_29 = relay.var("var_29", dtype = "float32", shape = (13,))#candidate|29|(13,)|var|float32
call_27 = func_26_call(var_28,var_29,)
output = call_27
func_30 = relay.Function([var_28,var_29,], output)
mutated_mod['func_30'] = func_30
mutated_mod = relay.transform.InferType()(mutated_mod)
var_55 = relay.var("var_55", dtype = "float64", shape = (1, 10))#candidate|55|(1, 10)|var|float64
uop_56 = relay.sigmoid(var_55.astype('float64')) # shape=(1, 10)
output = relay.Tuple([uop_56,])
output2 = relay.Tuple([uop_56,])
func_65 = relay.Function([var_55,], output)
mod['func_65'] = func_65
mod = relay.transform.InferType()(mod)
var_66 = relay.var("var_66", dtype = "float64", shape = (1, 10))#candidate|66|(1, 10)|var|float64
output = func_65(var_66)
func_67 = relay.Function([var_66], output)
mutated_mod['func_67'] = func_67
mutated_mod = relay.transform.InferType()(mutated_mod)
var_69 = relay.var("var_69", dtype = "float64", shape = (7, 6, 7))#candidate|69|(7, 6, 7)|var|float64
const_70 = relay.const([[[5.753847,-4.365862,-9.253936,9.393334,7.465176,7.788119,2.002341],[7.724741,-5.788465,3.812058,-4.293234,-4.937889,-6.255220,-4.625071],[-0.217257,6.849780,8.395496,-6.428453,5.630607,8.507085,-1.938625],[-6.121739,-8.721192,-1.316602,3.980138,9.780710,6.487348,-4.315823],[9.099586,-7.845961,-9.267034,-0.023664,-7.124120,-5.683262,-1.301552],[-5.530478,-8.028397,-1.291892,-9.154075,1.677848,-8.369331,-8.871322]],[[6.401708,-9.547308,0.101886,-3.916110,-2.863181,-3.315941,9.430602],[-3.056237,-4.225715,-3.245566,-8.028855,4.270783,-0.336880,-7.546161],[3.916535,-1.837331,3.681983,-6.367161,1.122315,2.357706,1.838058],[-4.801651,-3.379835,-0.015896,-4.384662,5.421132,3.726095,6.597507],[8.501723,5.674298,-1.103812,3.340980,-0.866028,9.670960,7.610450],[0.778296,-8.588898,9.422481,-2.284819,3.486415,5.782645,2.019777]],[[8.948230,-1.684688,-7.388191,3.476321,1.354700,9.885885,-5.882136],[6.338650,8.739468,-0.724806,7.551559,-4.261499,8.662128,-6.687948],[-9.663238,-7.576020,-4.166136,-5.024316,-2.961278,-9.191516,7.567158],[5.248126,-6.449856,0.544840,9.240452,6.168431,9.663803,-4.552350],[5.339267,2.332096,2.089111,2.723392,-4.499711,-1.989099,6.002486],[-9.997132,-3.427491,2.137482,8.771732,-2.350905,-6.281141,2.726784]],[[2.381857,4.643009,-1.103300,-7.012585,-7.543881,5.068212,-2.140419],[5.488825,-3.019818,-1.956718,2.254158,2.581869,-6.971928,4.472265],[6.631130,-8.168204,-5.351490,3.442011,4.735803,-5.245432,1.473235],[-3.082274,-7.454148,-7.906525,-5.811370,1.117904,-7.178398,-8.344619],[9.751041,8.568887,-9.285178,7.647315,7.048094,0.576282,6.253307],[-4.281886,4.530142,-0.704322,-2.083205,5.312415,2.617985,-1.992296]],[[9.131019,-3.575107,-6.839004,-6.494706,3.862084,-5.452074,-4.705308],[7.206058,4.955893,4.156195,0.122952,-2.858936,-0.163133,-2.566700],[2.131899,9.681387,9.070830,1.840491,7.720700,-9.044633,5.772529],[-6.007261,-5.081106,5.461780,0.645892,-0.987398,-6.607599,0.014869],[-1.659653,-1.076589,-4.978563,6.278144,8.328468,-2.254077,-2.503829],[-5.307950,8.210227,3.148209,-6.985704,2.457253,8.091512,-7.308638]],[[7.636054,-6.461531,3.153280,9.424891,-8.928489,7.550406,1.957888],[9.812295,-3.415779,8.083522,-3.064292,3.110716,-7.762156,-2.818115],[1.348500,6.104388,-1.739047,-4.902171,3.275005,-5.367366,-3.833538],[8.447322,2.161556,9.072633,-7.599538,4.719868,-0.110952,-2.697147],[6.798947,-2.465795,1.046322,3.080468,-0.155497,-6.853436,9.693207],[3.896226,-9.314153,5.593310,1.933319,-0.614175,-5.461714,-0.709463]],[[-8.691375,6.204505,-2.313709,-1.629096,6.530365,8.144236,-1.311365],[9.104111,-1.079362,5.321033,-4.476228,-1.206277,-8.202354,7.907584],[-1.927017,0.334660,-5.861813,8.275997,-6.452591,6.713738,2.049409],[7.910455,-3.386496,-7.556925,-1.537795,5.127004,-9.962950,-3.783512],[-1.234207,9.774011,-5.343400,-1.442271,8.150954,4.613406,9.010680],[-3.547200,9.452013,-8.278441,-6.951478,5.764827,4.987993,1.318968]]], dtype = "float64")#candidate|70|(7, 6, 7)|const|float64
bop_71 = relay.divide(var_69.astype('float64'), relay.reshape(const_70.astype('float64'), relay.shape_of(var_69))) # shape=(7, 6, 7)
bop_76 = relay.right_shift(const_70.astype('int16'), relay.reshape(bop_71.astype('int16'), relay.shape_of(const_70))) # shape=(7, 6, 7)
bop_79 = relay.logical_and(bop_71.astype('bool'), relay.reshape(var_69.astype('bool'), relay.shape_of(bop_71))) # shape=(7, 6, 7)
uop_82 = relay.log(bop_71.astype('float32')) # shape=(7, 6, 7)
func_65_call = mod.get_global_var('func_65')
func_67_call = mutated_mod.get_global_var('func_67')
const_88 = relay.const([8.891591,-7.128205,-6.670062,3.223308,-1.402293,-2.734908,8.676293,-0.505070,-3.962529,-5.257020], dtype = "float64")#candidate|88|(10,)|const|float64
call_87 = relay.TupleGetItem(func_65_call(relay.reshape(const_88.astype('float64'), [1, 10])), 0)
call_89 = relay.TupleGetItem(func_67_call(relay.reshape(const_88.astype('float64'), [1, 10])), 0)
uop_90 = relay.log(uop_82.astype('float32')) # shape=(7, 6, 7)
bop_92 = relay.logical_xor(uop_90.astype('uint32'), relay.reshape(bop_71.astype('uint32'), relay.shape_of(uop_90))) # shape=(7, 6, 7)
uop_98 = relay.tan(uop_82.astype('float32')) # shape=(7, 6, 7)
bop_100 = relay.mod(uop_90.astype('float32'), relay.reshape(bop_71.astype('float32'), relay.shape_of(uop_90))) # shape=(7, 6, 7)
output = relay.Tuple([bop_76,bop_79,call_87,const_88,bop_92,uop_98,bop_100,])
output2 = relay.Tuple([bop_76,bop_79,call_89,const_88,bop_92,uop_98,bop_100,])
func_103 = relay.Function([var_69,], output)
mod['func_103'] = func_103
mod = relay.transform.InferType()(mod)
mutated_mod['func_103'] = func_103
mutated_mod = relay.transform.InferType()(mutated_mod)
var_104 = relay.var("var_104", dtype = "float64", shape = (7, 6, 7))#candidate|104|(7, 6, 7)|var|float64
func_103_call = mutated_mod.get_global_var('func_103')
call_105 = func_103_call(var_104)
output = call_105
func_106 = relay.Function([var_104], output)
mutated_mod['func_106'] = func_106
mutated_mod = relay.transform.InferType()(mutated_mod)
var_149 = relay.var("var_149", dtype = "bool", shape = ())#candidate|149|()|var|bool
var_150 = relay.var("var_150", dtype = "bool", shape = (7, 16, 5))#candidate|150|(7, 16, 5)|var|bool
bop_151 = relay.logical_and(var_149.astype('bool'), var_150.astype('bool')) # shape=(7, 16, 5)
func_26_call = mod.get_global_var('func_26')
func_30_call = mutated_mod.get_global_var('func_30')
var_161 = relay.var("var_161", dtype = "float32", shape = (1, 13))#candidate|161|(1, 13)|var|float32
call_160 = relay.TupleGetItem(func_26_call(relay.reshape(var_161.astype('float32'), [13,]), relay.reshape(var_161.astype('float32'), [13,]), ), 0)
call_162 = relay.TupleGetItem(func_30_call(relay.reshape(var_161.astype('float32'), [13,]), relay.reshape(var_161.astype('float32'), [13,]), ), 0)
func_65_call = mod.get_global_var('func_65')
func_67_call = mutated_mod.get_global_var('func_67')
const_171 = relay.const([-0.644850,-4.948948,1.160033,9.613424,-5.317224,-4.426205,6.446681,6.891821,6.228125,-6.693228], dtype = "float64")#candidate|171|(10,)|const|float64
call_170 = relay.TupleGetItem(func_65_call(relay.reshape(const_171.astype('float64'), [1, 10])), 0)
call_172 = relay.TupleGetItem(func_67_call(relay.reshape(const_171.astype('float64'), [1, 10])), 0)
bop_184 = relay.bitwise_and(bop_151.astype('int32'), relay.reshape(var_150.astype('int32'), relay.shape_of(bop_151))) # shape=(7, 16, 5)
output = relay.Tuple([call_160,var_161,call_170,const_171,bop_184,])
output2 = relay.Tuple([call_162,var_161,call_172,const_171,bop_184,])
func_187 = relay.Function([var_149,var_150,var_161,], output)
mod['func_187'] = func_187
mod = relay.transform.InferType()(mod)
mutated_mod['func_187'] = func_187
mutated_mod = relay.transform.InferType()(mutated_mod)
func_187_call = mutated_mod.get_global_var('func_187')
var_189 = relay.var("var_189", dtype = "bool", shape = ())#candidate|189|()|var|bool
var_190 = relay.var("var_190", dtype = "bool", shape = (7, 16, 5))#candidate|190|(7, 16, 5)|var|bool
var_191 = relay.var("var_191", dtype = "float32", shape = (1, 13))#candidate|191|(1, 13)|var|float32
call_188 = func_187_call(var_189,var_190,var_191,)
output = call_188
func_192 = relay.Function([var_189,var_190,var_191,], output)
mutated_mod['func_192'] = func_192
mutated_mod = relay.transform.InferType()(mutated_mod)
var_202 = relay.var("var_202", dtype = "int64", shape = (4, 2, 11))#candidate|202|(4, 2, 11)|var|int64
var_203 = relay.var("var_203", dtype = "int64", shape = (4, 2, 11))#candidate|203|(4, 2, 11)|var|int64
bop_204 = relay.maximum(var_202.astype('int64'), relay.reshape(var_203.astype('int64'), relay.shape_of(var_202))) # shape=(4, 2, 11)
output = bop_204
output2 = bop_204
func_208 = relay.Function([var_202,var_203,], output)
mod['func_208'] = func_208
mod = relay.transform.InferType()(mod)
var_209 = relay.var("var_209", dtype = "int64", shape = (4, 2, 11))#candidate|209|(4, 2, 11)|var|int64
var_210 = relay.var("var_210", dtype = "int64", shape = (4, 2, 11))#candidate|210|(4, 2, 11)|var|int64
output = func_208(var_209,var_210,)
func_211 = relay.Function([var_209,var_210,], output)
mutated_mod['func_211'] = func_211
mutated_mod = relay.transform.InferType()(mutated_mod)
var_232 = relay.var("var_232", dtype = "uint16", shape = ())#candidate|232|()|var|uint16
var_233 = relay.var("var_233", dtype = "uint16", shape = (4, 14))#candidate|233|(4, 14)|var|uint16
bop_234 = relay.greater_equal(var_232.astype('bool'), var_233.astype('bool')) # shape=(4, 14)
output = bop_234
output2 = bop_234
func_243 = relay.Function([var_232,var_233,], output)
mod['func_243'] = func_243
mod = relay.transform.InferType()(mod)
var_244 = relay.var("var_244", dtype = "uint16", shape = ())#candidate|244|()|var|uint16
var_245 = relay.var("var_245", dtype = "uint16", shape = (4, 14))#candidate|245|(4, 14)|var|uint16
output = func_243(var_244,var_245,)
func_246 = relay.Function([var_244,var_245,], output)
mutated_mod['func_246'] = func_246
mutated_mod = relay.transform.InferType()(mutated_mod)
var_269 = relay.var("var_269", dtype = "float32", shape = (3, 13, 11))#candidate|269|(3, 13, 11)|var|float32
uop_270 = relay.asin(var_269.astype('float32')) # shape=(3, 13, 11)
uop_272 = relay.atan(uop_270.astype('float32')) # shape=(3, 13, 11)
output = uop_272
output2 = uop_272
func_277 = relay.Function([var_269,], output)
mod['func_277'] = func_277
mod = relay.transform.InferType()(mod)
var_278 = relay.var("var_278", dtype = "float32", shape = (3, 13, 11))#candidate|278|(3, 13, 11)|var|float32
output = func_277(var_278)
func_279 = relay.Function([var_278], output)
mutated_mod['func_279'] = func_279
mutated_mod = relay.transform.InferType()(mutated_mod)
var_339 = relay.var("var_339", dtype = "float64", shape = (9, 1, 9))#candidate|339|(9, 1, 9)|var|float64
uop_340 = relay.sinh(var_339.astype('float64')) # shape=(9, 1, 9)
uop_350 = relay.sinh(uop_340.astype('float32')) # shape=(9, 1, 9)
bop_355 = relay.greater_equal(uop_350.astype('bool'), relay.reshape(uop_340.astype('bool'), relay.shape_of(uop_350))) # shape=(9, 1, 9)
bop_362 = relay.not_equal(uop_350.astype('bool'), relay.reshape(uop_340.astype('bool'), relay.shape_of(uop_350))) # shape=(9, 1, 9)
bop_365 = relay.logical_and(uop_340.astype('bool'), relay.reshape(bop_362.astype('bool'), relay.shape_of(uop_340))) # shape=(9, 1, 9)
var_368 = relay.var("var_368", dtype = "float64", shape = (9, 15, 9))#candidate|368|(9, 15, 9)|var|float64
bop_369 = relay.bitwise_xor(uop_340.astype('uint64'), var_368.astype('uint64')) # shape=(9, 15, 9)
bop_372 = relay.subtract(bop_362.astype('uint64'), relay.reshape(bop_365.astype('uint64'), relay.shape_of(bop_362))) # shape=(9, 1, 9)
bop_376 = relay.less_equal(uop_350.astype('bool'), var_368.astype('bool')) # shape=(9, 15, 9)
uop_380 = relay.log10(bop_376.astype('float64')) # shape=(9, 15, 9)
uop_388 = relay.atan(uop_380.astype('float32')) # shape=(9, 15, 9)
bop_395 = relay.mod(bop_369.astype('float64'), bop_372.astype('float64')) # shape=(9, 15, 9)
bop_398 = relay.logical_or(bop_355.astype('bool'), uop_380.astype('bool')) # shape=(9, 15, 9)
output = relay.Tuple([uop_388,bop_395,bop_398,])
output2 = relay.Tuple([uop_388,bop_395,bop_398,])
func_403 = relay.Function([var_339,var_368,], output)
mod['func_403'] = func_403
mod = relay.transform.InferType()(mod)
mutated_mod['func_403'] = func_403
mutated_mod = relay.transform.InferType()(mutated_mod)
func_403_call = mutated_mod.get_global_var('func_403')
var_405 = relay.var("var_405", dtype = "float64", shape = (9, 1, 9))#candidate|405|(9, 1, 9)|var|float64
var_406 = relay.var("var_406", dtype = "float64", shape = (9, 15, 9))#candidate|406|(9, 15, 9)|var|float64
call_404 = func_403_call(var_405,var_406,)
output = call_404
func_407 = relay.Function([var_405,var_406,], output)
mutated_mod['func_407'] = func_407
mutated_mod = relay.transform.InferType()(mutated_mod)
var_430 = relay.var("var_430", dtype = "float32", shape = (1,))#candidate|430|(1,)|var|float32
uop_431 = relay.cosh(var_430.astype('float32')) # shape=(1,)
bop_436 = relay.less(uop_431.astype('bool'), relay.reshape(var_430.astype('bool'), relay.shape_of(uop_431))) # shape=(1,)
bop_447 = relay.left_shift(var_430.astype('int8'), relay.reshape(bop_436.astype('int8'), relay.shape_of(var_430))) # shape=(1,)
output = relay.Tuple([bop_447,])
output2 = relay.Tuple([bop_447,])
func_456 = relay.Function([var_430,], output)
mod['func_456'] = func_456
mod = relay.transform.InferType()(mod)
var_457 = relay.var("var_457", dtype = "float32", shape = (1,))#candidate|457|(1,)|var|float32
output = func_456(var_457)
func_458 = relay.Function([var_457], output)
mutated_mod['func_458'] = func_458
mutated_mod = relay.transform.InferType()(mutated_mod)
var_511 = relay.var("var_511", dtype = "float32", shape = (3,))#candidate|511|(3,)|var|float32
uop_512 = relay.log2(var_511.astype('float32')) # shape=(3,)
bop_520 = relay.multiply(uop_512.astype('int16'), relay.reshape(var_511.astype('int16'), relay.shape_of(uop_512))) # shape=(3,)
bop_526 = relay.greater(var_511.astype('bool'), relay.reshape(uop_512.astype('bool'), relay.shape_of(var_511))) # shape=(3,)
func_187_call = mod.get_global_var('func_187')
func_192_call = mutated_mod.get_global_var('func_192')
const_535 = relay.const(True, dtype = "bool")#candidate|535|()|const|bool
const_536 = relay.const([False,False,False,False,False,True,True,True,True,True,False,False,True,True,False,True,False,False,False,False,False,False,False,False,False,True,True,False,False,True,False,False,False,False,True,False,False,False,True,True,True,True,True,True,True,True,False,True,True,False,True,True,False,False,False,False,True,True,True,True,True,True,False,True,False,True,True,False,True,False,True,False,False,False,True,True,True,True,False,True,True,True,False,False,True,False,False,False,False,True,False,True,True,False,False,True,True,True,True,False,True,False,False,True,False,False,False,True,True,False,False,True,False,True,True,True,True,True,False,True,True,False,False,False,False,False,True,True,True,False,True,True,False,True,False,False,True,True,True,True,True,False,False,True,True,True,True,False,False,True,False,True,True,False,True,True,True,False,False,False,False,True,True,False,False,False,False,False,True,False,True,False,False,True,False,True,False,True,True,False,False,True,False,True,False,True,False,True,False,False,True,False,True,True,True,True,True,True,True,False,True,False,False,True,True,False,False,True,True,False,False,True,True,False,False,True,True,False,False,True,False,False,False,True,True,True,True,False,False,False,False,False,True,False,True,False,True,False,False,False,False,False,False,True,False,False,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,True,True,False,False,False,False,False,False,False,False,False,True,False,True,False,True,False,False,False,True,True,True,False,False,True,False,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,True,False,False,False,False,True,False,False,True,True,True,False,False,True,False,False,True,True,True,False,True,True,True,True,False,True,False,False,True,False,False,False,False,False,False,False,False,True,True,True,False,False,True,False,False,False,True,True,True,False,True,False,True,False,False,True,True,False,True,False,False,False,True,True,False,True,True,False,False,False,False,True,True,True,True,True,True,False,False,False,False,False,True,False,False,True,True,False,True,True,False,True,True,True,False,True,True,False,False,True,False,False,False,True,True,False,False,True,False,True,True,True,True,False,True,True,True,False,True,False,True,False,False,False,True,False,False,False,False,False,False,True,False,False,False,True,False,True,False,True,False,False,False,True,False,True,False,False,True,False,False,True,False,False,True,True,True,True,False,True,True,False,False,True,False,True,True,True,False,True,False,False,True,False,True,False,True,True,False,False,True,False,True,True,True,True,True,False,False,True,True,False,True,False,True,True,True,False,False,True,False,False,True,True,False,False,True,True,True,True,False,True,False,True,False,True,False,True,True,True,False,False,True,True,False,False,True,True,True,True,False,True,False,True,False,False,True,False,False,True,True,False,False,True,True,False,False], dtype = "bool")#candidate|536|(560,)|const|bool
const_537 = relay.const([-7.328512,-8.343395,7.460872,9.823009,3.021287,0.713580,7.570286,5.358531,-2.341574,6.794621,-4.414780,5.961269,9.149776], dtype = "float32")#candidate|537|(13,)|const|float32
call_534 = relay.TupleGetItem(func_187_call(relay.reshape(const_535.astype('bool'), []), relay.reshape(const_536.astype('bool'), [7, 16, 5]), relay.reshape(const_537.astype('float32'), [1, 13]), ), 3)
call_538 = relay.TupleGetItem(func_192_call(relay.reshape(const_535.astype('bool'), []), relay.reshape(const_536.astype('bool'), [7, 16, 5]), relay.reshape(const_537.astype('float32'), [1, 13]), ), 3)
var_540 = relay.var("var_540", dtype = "int16", shape = (3,))#candidate|540|(3,)|var|int16
bop_541 = relay.divide(bop_520.astype('float32'), relay.reshape(var_540.astype('float32'), relay.shape_of(bop_520))) # shape=(3,)
uop_547 = relay.exp(bop_541.astype('float32')) # shape=(3,)
uop_550 = relay.acosh(uop_547.astype('float64')) # shape=(3,)
uop_557 = relay.sigmoid(uop_550.astype('float64')) # shape=(3,)
func_456_call = mod.get_global_var('func_456')
func_458_call = mutated_mod.get_global_var('func_458')
call_561 = relay.TupleGetItem(func_456_call(relay.reshape(const_535.astype('float32'), [1,])), 0)
call_562 = relay.TupleGetItem(func_458_call(relay.reshape(const_535.astype('float32'), [1,])), 0)
uop_567 = relay.atan(uop_557.astype('float32')) # shape=(3,)
uop_574 = relay.rsqrt(uop_567.astype('float32')) # shape=(3,)
var_593 = relay.var("var_593", dtype = "float32", shape = (3,))#candidate|593|(3,)|var|float32
bop_594 = relay.less(uop_574.astype('bool'), relay.reshape(var_593.astype('bool'), relay.shape_of(uop_574))) # shape=(3,)
bop_607 = relay.power(uop_557.astype('float32'), call_561.astype('float32')) # shape=(3,)
bop_610 = relay.power(uop_557.astype('float32'), call_562.astype('float32')) # shape=(3,)
uop_616 = relay.log10(bop_594.astype('float64')) # shape=(3,)
uop_619 = relay.acos(uop_616.astype('float32')) # shape=(3,)
bop_623 = relay.not_equal(uop_616.astype('bool'), relay.reshape(uop_512.astype('bool'), relay.shape_of(uop_616))) # shape=(3,)
bop_627 = relay.bitwise_and(uop_616.astype('uint8'), relay.reshape(uop_619.astype('uint8'), relay.shape_of(uop_616))) # shape=(3,)
func_208_call = mod.get_global_var('func_208')
func_211_call = mutated_mod.get_global_var('func_211')
const_637 = relay.const([-3,-10,-1,-6,-6,-3,-3,-3,4,10,-5,2,4,-5,3,5,-2,1,-5,-8,-5,8,-3,-2,2,8,-4,5,6,1,1,7,3,7,-6,-1,-8,-8,9,-10,2,10,10,-4,-5,-6,-4,-2,-3,-9,5,-10,8,-6,10,9,3,9,-7,8,9,10,8,-2,-2,8,10,2,7,-6,1,-8,7,8,10,5,1,-5,-7,8,6,9,-5,4,7,9,-7,2], dtype = "int64")#candidate|637|(88,)|const|int64
call_636 = func_208_call(relay.reshape(const_637.astype('int64'), [4, 2, 11]), relay.reshape(const_637.astype('int64'), [4, 2, 11]), )
call_638 = func_208_call(relay.reshape(const_637.astype('int64'), [4, 2, 11]), relay.reshape(const_637.astype('int64'), [4, 2, 11]), )
bop_640 = relay.add(bop_594.astype('uint64'), relay.reshape(bop_526.astype('uint64'), relay.shape_of(bop_594))) # shape=(3,)
bop_648 = relay.right_shift(bop_627.astype('uint32'), relay.reshape(bop_640.astype('uint32'), relay.shape_of(bop_627))) # shape=(3,)
uop_656 = relay.asinh(bop_623.astype('float64')) # shape=(3,)
func_456_call = mod.get_global_var('func_456')
func_458_call = mutated_mod.get_global_var('func_458')
call_660 = relay.TupleGetItem(func_456_call(relay.reshape(call_561.astype('float32'), [1,])), 0)
call_661 = relay.TupleGetItem(func_458_call(relay.reshape(call_561.astype('float32'), [1,])), 0)
output = relay.Tuple([call_534,const_535,const_536,const_537,bop_607,call_636,const_637,bop_648,uop_656,call_660,])
output2 = relay.Tuple([call_538,const_535,const_536,const_537,bop_610,call_638,const_637,bop_648,uop_656,call_661,])
func_666 = relay.Function([var_511,var_540,var_593,], output)
mod['func_666'] = func_666
mod = relay.transform.InferType()(mod)
mutated_mod['func_666'] = func_666
mutated_mod = relay.transform.InferType()(mutated_mod)
func_666_call = mutated_mod.get_global_var('func_666')
var_668 = relay.var("var_668", dtype = "float32", shape = (3,))#candidate|668|(3,)|var|float32
var_669 = relay.var("var_669", dtype = "int16", shape = (3,))#candidate|669|(3,)|var|int16
var_670 = relay.var("var_670", dtype = "float32", shape = (3,))#candidate|670|(3,)|var|float32
call_667 = func_666_call(var_668,var_669,var_670,)
output = call_667
func_671 = relay.Function([var_668,var_669,var_670,], output)
mutated_mod['func_671'] = func_671
mutated_mod = relay.transform.InferType()(mutated_mod)
var_681 = relay.var("var_681", dtype = "bool", shape = (9, 4, 4))#candidate|681|(9, 4, 4)|var|bool
var_682 = relay.var("var_682", dtype = "bool", shape = (9, 4, 4))#candidate|682|(9, 4, 4)|var|bool
bop_683 = relay.logical_and(var_681.astype('bool'), relay.reshape(var_682.astype('bool'), relay.shape_of(var_681))) # shape=(9, 4, 4)
uop_686 = relay.log(bop_683.astype('float64')) # shape=(9, 4, 4)
uop_688 = relay.asin(uop_686.astype('float32')) # shape=(9, 4, 4)
bop_692 = relay.multiply(uop_686.astype('uint64'), relay.reshape(uop_688.astype('uint64'), relay.shape_of(uop_686))) # shape=(9, 4, 4)
var_696 = relay.var("var_696", dtype = "uint64", shape = (9, 4, 4))#candidate|696|(9, 4, 4)|var|uint64
bop_697 = relay.minimum(bop_692.astype('float32'), relay.reshape(var_696.astype('float32'), relay.shape_of(bop_692))) # shape=(9, 4, 4)
var_704 = relay.var("var_704", dtype = "uint64", shape = (9, 4, 4))#candidate|704|(9, 4, 4)|var|uint64
bop_705 = relay.less(bop_692.astype('bool'), relay.reshape(var_704.astype('bool'), relay.shape_of(bop_692))) # shape=(9, 4, 4)
bop_708 = relay.add(uop_686.astype('float64'), relay.reshape(var_682.astype('float64'), relay.shape_of(uop_686))) # shape=(9, 4, 4)
bop_714 = relay.equal(uop_688.astype('bool'), relay.reshape(bop_705.astype('bool'), relay.shape_of(uop_688))) # shape=(9, 4, 4)
func_65_call = mod.get_global_var('func_65')
func_67_call = mutated_mod.get_global_var('func_67')
var_718 = relay.var("var_718", dtype = "float64", shape = (10,))#candidate|718|(10,)|var|float64
call_717 = relay.TupleGetItem(func_65_call(relay.reshape(var_718.astype('float64'), [1, 10])), 0)
call_719 = relay.TupleGetItem(func_67_call(relay.reshape(var_718.astype('float64'), [1, 10])), 0)
func_403_call = mod.get_global_var('func_403')
func_407_call = mutated_mod.get_global_var('func_407')
const_722 = relay.const([6.196801,-5.951778,4.062040,-9.855737,0.588089,-9.781804,-3.858865,-7.631992,-2.584610,-4.349602,-7.986161,-9.711495,5.529365,9.733655,-4.481394,5.179404,1.623294,1.894771,-9.651707,3.331791,-1.409221,-1.296793,-5.764145,0.042900,-1.792217,-0.201557,-6.382347,7.603304,3.418122,-0.576190,0.035107,0.293509,8.508241,3.768122,9.669160,-5.806625,9.882423,-2.010803,1.360732,-3.518738,3.103769,-2.597588,-8.995936,-6.240515,6.324212,2.736792,-6.514825,7.363669,2.582343,3.576174,-1.814039,9.840571,-6.724615,8.691012,9.135063,6.238354,-0.921177,0.038952,5.312161,-9.434001,-9.501760,4.464092,8.258617,0.159020,1.117638,4.586687,-8.153989,9.912480,-4.749614,-6.567795,2.687467,-6.455658,-3.552537,-1.690682,8.030741,-4.010179,-2.552696,-6.582043,-9.308756,-2.635566,-9.114554], dtype = "float64")#candidate|722|(81,)|const|float64
const_723 = relay.const([1.445836,-4.471695,5.051767,6.701689,5.284457,5.340493,8.022713,0.152181,-8.498013,6.845899,2.213056,-2.460004,-3.366216,8.863111,1.261629,-7.974985,-0.350290,2.789284,-4.454188,3.991756,-4.420662,4.289935,-9.682693,6.456375,-7.496524,6.740867,-8.266034,-2.140617,5.973172,2.832502,0.447684,4.128698,-6.643436,9.696093,-0.297092,-5.969113,6.090584,-0.553339,-2.964808,5.027888,6.177458,-2.304297,-3.040134,-0.785921,-9.879893,8.369721,5.655974,6.519192,1.496231,5.014769,-3.434153,-3.065926,6.871325,0.981901,-6.198177,1.760816,1.187951,2.935853,-1.407989,3.913203,3.779569,-2.961683,-8.907930,-1.296890,1.745838,0.779590,-5.816401,4.595002,-8.329127,8.724438,2.417272,-2.403785,7.835319,0.917472,8.919579,-7.989955,6.381190,-9.171389,3.399052,6.274730,-6.089387,-7.051123,-5.108293,-7.611519,-9.546284,1.938754,-3.984603,-7.610759,0.512585,8.870151,1.125520,-2.866182,3.128124,-4.994982,5.280228,8.254900,-7.977281,-6.849473,7.007823,6.017913,5.731253,-3.301031,2.517073,5.962562,-9.444698,3.549558,-6.623375,-4.076380,-0.570948,-4.139174,-4.107151,7.166531,4.870875,5.849930,1.452893,-7.993459,3.555953,-4.416609,-1.632529,5.342470,4.826754,-9.807084,-4.924578,9.260478,6.644596,1.850127,3.552441,-7.820893,-2.139316,2.693400,-0.085382,-2.066405,-0.938077,-8.121606,-3.015636,2.944956,8.510970,-9.998280,-5.690486,-4.414806,9.251962,2.244007,1.515527,-5.002143,1.506321,9.002756,6.752960,1.334970,-7.126594,-1.326899,-2.548645,-6.602977,7.327956,1.584142,-0.610584,1.646050,9.474435,-4.466213,-6.798596,2.935868,5.732289,-7.540884,-1.302890,-6.389850,7.757544,-3.740040,0.317719,7.483103,0.898606,-7.506552,3.645686,7.455322,7.317748,3.782549,-1.350981,6.795651,4.100285,4.723355,-3.397233,-3.520335,-8.885388,5.018824,-6.949732,1.575899,2.691832,3.489978,5.563642,-8.824793,0.304687,8.616633,-9.382415,3.343482,-4.536037,1.092053,-9.220329,-3.954165,-7.189345,-4.220579,0.391974,-2.197921,-4.754302,4.704351,3.010227,-3.661409,6.406643,-1.649094,-9.206154,5.526632,-5.675583,7.511980,-9.981315,-2.298837,0.048657,5.228350,-4.017068,-3.579254,-9.828040,-0.405633,7.506577,-0.236955,-1.826382,-5.069555,-7.997002,6.692582,-3.669363,3.273467,5.449026,-7.460636,5.631990,3.710278,5.931032,-1.901460,1.132046,8.440970,2.875619,7.059583,3.132712,-3.546884,9.903109,-9.121715,7.436866,-1.053431,0.420467,8.035850,5.983248,-6.203551,-4.291128,-3.640006,-2.645782,-8.763079,5.354156,8.995765,6.106658,-6.320870,-0.252387,6.877254,-0.789153,-6.630816,-6.183920,-0.283942,-7.095115,-1.794475,9.951481,-3.744236,9.040022,6.454025,-7.225833,9.649662,2.281021,-1.228734,-6.050516,6.252809,9.117276,0.822320,0.770260,9.308781,-9.646267,0.126530,-5.551429,2.751170,-6.073845,-1.576647,-0.222419,6.868341,9.213996,7.034520,1.335373,6.454523,-2.239390,-5.185022,-6.090289,1.079397,3.047860,7.560279,-6.520502,2.670859,0.510687,2.093998,7.974445,0.653927,-3.939831,4.037080,1.740471,-9.461075,6.976632,-7.013089,0.550716,-4.190165,-3.441936,-5.515316,0.114592,3.171628,-1.807062,-0.752907,5.374526,-6.653195,-7.790086,5.721889,-7.669237,5.236663,-8.919626,-3.900568,7.425164,6.342317,-6.856855,-1.216260,9.648464,-4.332822,-6.151090,-3.699936,9.420687,3.484511,-1.531423,2.309543,-0.744708,-4.745738,-3.710640,-4.674537,8.127961,3.698035,-7.671003,2.490771,-1.312724,5.241098,-4.214682,-7.275185,-6.149471,2.869015,9.188861,-1.782477,6.365831,-0.859307,6.259227,7.664284,-9.950241,-4.782730,-5.918847,5.344829,9.027056,0.235578,-3.999714,0.171002,3.597192,5.880958,-4.558152,5.900200,-3.605833,2.413352,-6.339440,-1.444394,-9.448312,0.172126,1.453251,3.943204,1.246999,-7.918234,-5.961798,5.508726,8.788636,-1.117752,-1.498664,8.820799,-6.728425,-8.426379,-1.134684,4.567593,3.893998,2.446202,9.720444,0.634411,-9.849697,-3.499374,-5.020208,-5.473566,-6.393024,1.668760,-7.420150,2.986226,9.036493,-3.928189,7.367493,-4.939977,-5.502920,9.313774,-6.238383,5.167257,-0.948058,-1.391086,4.574336,-2.394220,7.693234,-6.230690,0.926332,-0.899926,2.729311,-6.710691,8.537700,5.976745,-5.596013,2.221616,4.946024,8.923296,3.539830,-3.921439,9.142470,-9.433730,-9.838070,-0.972306,8.803396,1.427792,3.621054,8.473610,-5.009942,0.854805,9.219177,-5.596439,3.371453,-8.799753,8.760381,0.301560,-6.728947,8.797642,-1.240500,-0.397233,5.521491,-4.501610,8.530892,7.451437,-9.901651,-4.370893,0.556623,-4.257091,8.580308,4.145363,-8.273121,1.455865,1.049164,8.514210,-4.562664,8.956974,1.448109,8.139017,8.583662,8.304865,6.376323,2.456902,-5.184136,5.108675,-0.167939,-6.511051,2.042041,4.189169,-4.983931,-9.027634,-1.115507,-9.536842,0.497158,-0.993329,-6.216346,-3.880665,-3.973589,6.401082,5.347396,-0.951933,0.168595,-7.578085,-2.984665,0.731036,4.538723,5.598557,7.030338,8.333084,3.645607,6.472497,8.068982,0.342519,4.858929,-5.804099,5.002582,2.220827,8.744527,1.020938,8.253594,2.408335,-6.192906,2.612912,-7.836231,-5.606027,-4.778695,-9.647146,8.451159,4.135926,-3.125162,-9.093955,-8.629130,-4.892101,4.214464,0.638703,1.356612,5.619568,-7.999618,5.328277,-8.922334,-6.887036,-2.988805,-9.180688,8.552858,-9.001459,-3.106408,6.135779,-6.962681,-5.326542,-4.158857,3.786879,-7.439335,6.012159,1.014483,-2.272455,1.643796,-6.770728,6.787564,8.303603,5.564818,-1.844102,-2.875577,-4.450984,8.067059,3.985117,2.662581,3.031464,0.100119,-6.389540,-9.359099,-4.643077,7.941590,2.438279,-7.315360,8.927976,-0.620895,3.686206,-0.593272,-1.473251,-4.661185,8.108106,-8.266955,-9.222756,1.478795,-9.388471,7.589982,2.899408,8.368974,6.124722,-7.454866,7.586031,-2.567403,2.592955,4.012639,-9.579181,6.632381,6.953788,4.749098,-9.261510,-8.491106,-2.391035,3.967022,-6.186530,-4.253971,-8.820477,4.461676,1.474423,-8.037816,-7.387368,-4.542444,-3.825155,-3.754485,9.337369,7.913182,-5.140462,-7.988542,3.660344,1.988884,1.211707,1.103772,-7.042938,3.415126,-4.491767,-1.500465,9.216821,5.182287,5.329375,4.097793,2.826161,-1.879230,-3.847631,-7.189205,-2.866734,-2.492524,3.628538,-7.083141,-6.865297,-9.891123,6.106428,0.432764,2.969882,-2.326584,-1.816268,-1.224215,-4.872762,2.838205,0.351142,-7.135768,7.078527,7.564188,-0.514197,-4.372832,-1.045648,7.928363,-2.078651,-7.635214,-1.916719,-4.091703,9.608445,-4.275106,9.160771,-5.093274,-0.217456,-5.613343,9.568245,-2.552203,5.227720,-9.610157,4.822614,-7.340942,-9.483363,-4.534798,5.779831,5.978080,3.134431,3.692805,-7.468481,2.628600,-8.786615,-7.234516,-1.739348,-2.683972,-8.696507,2.015111,-9.575596,4.162309,-6.416446,3.638330,-4.629365,-2.821707,0.485882,-1.762641,-2.011635,-4.260774,7.703203,-2.465963,3.149712,-4.449878,-4.523892,-5.739950,8.541582,2.278398,8.965942,0.196493,-7.353243,6.030701,2.684100,1.877275,3.253899,3.349012,-3.248173,-7.258931,-1.969648,6.906399,-5.920738,1.250499,4.975537,-5.111930,-2.067863,-5.080772,3.219084,-6.351907,9.245226,-0.985416,8.098101,-9.041786,-0.030623,3.950824,3.106090,6.821777,-0.191302,3.802529,5.579958,-0.655971,7.726409,-2.159451,-5.927555,-9.213326,-4.357974,6.003975,-0.532849,-3.292228,-6.283961,8.611987,-5.549171,-2.390882,0.145357,0.905552,9.518086,-7.211080,-6.952989,6.888624,9.381584,7.347795,9.569760,5.401614,-8.888387,-5.376401,-7.801376,-0.620865,-0.436161,6.875314,-6.307198,-7.741300,4.698896,-2.018953,-7.756056,-0.007768,1.467602,3.480446,-8.787646,-3.698365,-3.062828,-4.684801,-2.641394,-9.186992,5.817186,4.038045,-8.354163,1.814938,-2.166489,3.084450,5.922801,5.638662,2.418534,6.073543,3.332779,-2.790310,-7.995670,-7.370560,3.216887,1.495499,3.426211,-0.559164,9.863359,2.144284,-1.217601,-5.505455,-6.614558,-5.732686,-9.433969,-1.298815,-3.144537,-0.633466,2.852712,0.509929,9.245422,5.485228,6.619315,-6.851724,5.335002,3.653860,-1.722035,7.475149,0.903550,5.855530,-6.735041,8.613090,-3.747100,0.730378,2.503002,2.421081,-9.760296,4.348315,-4.884965,-8.580384,0.104935,9.605438,3.779955,-1.086107,4.602248,-0.905041,9.812898,-7.084937,0.846625,2.632057,-4.913512,2.520266,7.277926,3.018295,-2.820739,2.242398,4.645484,-1.799112,-2.788219,8.277605,-8.876165,2.202111,3.976829,6.832377,2.616846,4.346664,2.675932,-5.044284,2.092486,7.958928,3.843036,2.654097,5.682803,1.443385,0.891838,7.921628,8.136708,9.887970,-1.252747,1.878296,-7.019007,-8.532377,-1.394325,-2.236707,-0.663144,5.813724,4.392937,-6.280732,-1.011803,-4.720037,-9.900095,4.986818,-5.712770,-8.661454,-0.339849,-3.403263,8.386917,-1.966182,-8.219743,7.235318,-5.423397,-6.815955,-0.190979,2.586279,1.680974,-7.981545,8.802513,2.802934,-8.277349,9.545186,8.658103,2.803178,-3.930093,-1.506405,9.303297,2.359531,4.976564,7.273006,-8.468282,-8.335350,-2.156792,3.487719,8.502029,0.815924,-7.341861,-4.957338,-8.018114,-3.133756,3.361174,1.787282,8.064817,7.233801,2.640218,-7.296750,4.503932,2.797755,3.703349,8.224710,3.855407,-1.418657,2.906604,9.699126,8.411751,-5.686040,5.806850,1.638579,-7.882027,4.493181,4.093559,0.330294,-8.822845,-5.011842,5.976120,-1.478626,-3.782960,5.875925,2.271393,5.484999,-7.104539,-2.835958,-6.625801,-2.556706,-6.415064,-6.249470,4.451216,-9.461250,-8.516827,1.032727,-9.414354,-5.164374,9.783134,6.596181,-7.673354,1.419123,-8.672560,-2.213828,6.991568,-7.283763,-6.693272,9.549035,-5.966604,-0.306517,-3.724000,9.655547,3.809623,6.697953,5.053747,-2.137932,8.011730,-0.397260,8.229479,-9.097972,-7.161140,-4.472508,6.822822,-7.710703,9.055620,-4.192591,2.229631,-8.048866,5.697770,5.939134,-7.293556,9.014764,9.489305,-9.540767,-9.475561,1.170903,9.424946,-6.163283,9.967791,-2.172580,0.182555,-5.632018,-3.064677,2.161959,-4.922886,-2.398724,-1.016465,-7.382472,-9.635505,-3.143414,-2.858431,7.095236,5.129497,-0.481004,-8.748668,5.196716,-1.807241,-2.140501,0.933475,-8.631119,-0.097999,-7.860796,-1.568331,-8.874462,1.998757,-2.920749,-9.229649,2.603735,3.253225,-1.144368,1.872517,5.465947,6.275388,-1.794563,0.625606,-8.401153,-1.140700,7.598312,-2.069671,1.767816,-7.420230,2.660844,1.387604,8.784173,-6.759969,-7.826734,5.154546,2.229280,-6.917717,8.611762,0.067340,3.086008,2.140872,-2.244416,4.347649,3.517672,2.778783,-3.545188,-5.561836,7.595172,-1.156240,-1.371419,5.022836,-8.301536,2.990981,-4.503434,-9.546418,-5.438201,-5.542294,1.382446,-8.593084,2.587716,7.551344,-3.786017,3.526026,-3.603370,-7.050342,-9.597606,8.617103,3.776802,-0.101043,0.507347,3.998206,-3.869926,-2.700478,6.379731,5.376616,7.988768,1.629878,-1.357734,9.021836,-0.633524,5.451840,-8.997994,6.802441,9.852478,7.876706,-0.579928,6.260483,-6.819757,-9.280774,-5.776597,-4.329914,-8.662368,-7.649017,0.931918,-0.498044,6.124068,-7.897130,-9.286741,0.576059,0.726309,-1.531296,6.422423,5.068398,7.917749,-6.152538,0.359263,6.107395,-0.758472,-7.672168,-3.191315,2.906279,-1.679277,2.884869,2.090594,-6.042108,2.215615,9.303756,-2.164056,-1.759750,-9.549229,-9.282258,9.231974,-5.906838,8.572101,6.042513,7.764241,1.157534,-7.370744,-3.791540,0.393098,3.377975,3.948549,1.571837,7.277037,4.021165,-6.999697,9.385177,-5.823060,5.175034,-3.626397,3.841258,1.066752,7.574156,3.828351,-2.021217,-3.668394,4.024398,-5.329061,-7.432500,-5.115292,9.777621,6.355056,-0.106581,-5.052913,9.299346,2.205516,-2.260967,8.451762,-2.291842,1.270986,5.632125,-6.775482,2.788198,8.505124,-8.076759,-9.652355,1.172711,-7.216622,9.856503,7.378003,9.869846,8.603494,-1.387847,2.808511,6.021628,5.741123,8.458223,4.376903,-2.831075,-1.703083,-4.765483,-7.246539,-6.315176,8.907337,-2.598513,-6.684739,1.697591,9.131575,-0.610771,4.470182,-7.757219,-5.472738,-3.084335,-4.524536,-7.719242,-2.596426,6.689680,-5.373821,0.891893,9.897166,0.289780,-8.985008,0.042149,-0.936969,-5.094590,8.755659,-2.234726,-9.432833,6.176968,1.352460,1.714864,-7.397176,-3.405464,-7.682316,7.698325,9.196048,5.797117,0.974881,-8.427022,-9.315054,4.112739,8.558907], dtype = "float64")#candidate|723|(1215,)|const|float64
call_721 = relay.TupleGetItem(func_403_call(relay.reshape(const_722.astype('float64'), [9, 1, 9]), relay.reshape(const_723.astype('float64'), [9, 15, 9]), ), 2)
call_724 = relay.TupleGetItem(func_407_call(relay.reshape(const_722.astype('float64'), [9, 1, 9]), relay.reshape(const_723.astype('float64'), [9, 15, 9]), ), 2)
output = relay.Tuple([bop_697,bop_708,bop_714,call_717,var_718,call_721,const_722,const_723,])
output2 = relay.Tuple([bop_697,bop_708,bop_714,call_719,var_718,call_724,const_722,const_723,])
func_730 = relay.Function([var_681,var_682,var_696,var_704,var_718,], output)
mod['func_730'] = func_730
mod = relay.transform.InferType()(mod)
mutated_mod['func_730'] = func_730
mutated_mod = relay.transform.InferType()(mutated_mod)
func_730_call = mutated_mod.get_global_var('func_730')
var_732 = relay.var("var_732", dtype = "bool", shape = (9, 4, 4))#candidate|732|(9, 4, 4)|var|bool
var_733 = relay.var("var_733", dtype = "bool", shape = (9, 4, 4))#candidate|733|(9, 4, 4)|var|bool
var_734 = relay.var("var_734", dtype = "uint64", shape = (9, 4, 4))#candidate|734|(9, 4, 4)|var|uint64
var_735 = relay.var("var_735", dtype = "uint64", shape = (9, 4, 4))#candidate|735|(9, 4, 4)|var|uint64
var_736 = relay.var("var_736", dtype = "float64", shape = (10,))#candidate|736|(10,)|var|float64
call_731 = func_730_call(var_732,var_733,var_734,var_735,var_736,)
output = call_731
func_737 = relay.Function([var_732,var_733,var_734,var_735,var_736,], output)
mutated_mod['func_737'] = func_737
mutated_mod = relay.transform.InferType()(mutated_mod)
var_779 = relay.var("var_779", dtype = "int8", shape = (14, 9))#candidate|779|(14, 9)|var|int8
var_780 = relay.var("var_780", dtype = "int8", shape = (14, 9))#candidate|780|(14, 9)|var|int8
bop_781 = relay.equal(var_779.astype('bool'), relay.reshape(var_780.astype('bool'), relay.shape_of(var_779))) # shape=(14, 9)
uop_788 = relay.acosh(var_780.astype('float32')) # shape=(14, 9)
bop_791 = relay.logical_and(uop_788.astype('bool'), relay.reshape(bop_781.astype('bool'), relay.shape_of(uop_788))) # shape=(14, 9)
bop_797 = relay.mod(bop_791.astype('float32'), relay.reshape(uop_788.astype('float32'), relay.shape_of(bop_791))) # shape=(14, 9)
bop_800 = relay.bitwise_xor(bop_797.astype('uint8'), relay.reshape(var_779.astype('uint8'), relay.shape_of(bop_797))) # shape=(14, 9)
var_803 = relay.var("var_803", dtype = "float32", shape = (14, 9))#candidate|803|(14, 9)|var|float32
bop_804 = relay.less(uop_788.astype('bool'), relay.reshape(var_803.astype('bool'), relay.shape_of(uop_788))) # shape=(14, 9)
uop_812 = relay.acos(bop_804.astype('float32')) # shape=(14, 9)
var_820 = relay.var("var_820", dtype = "float32", shape = (14, 9))#candidate|820|(14, 9)|var|float32
bop_821 = relay.logical_or(uop_812.astype('bool'), relay.reshape(var_820.astype('bool'), relay.shape_of(uop_812))) # shape=(14, 9)
func_243_call = mod.get_global_var('func_243')
func_246_call = mutated_mod.get_global_var('func_246')
const_825 = relay.const(-3, dtype = "uint16")#candidate|825|()|const|uint16
var_826 = relay.var("var_826", dtype = "uint16", shape = (56,))#candidate|826|(56,)|var|uint16
call_824 = func_243_call(relay.reshape(const_825.astype('uint16'), []), relay.reshape(var_826.astype('uint16'), [4, 14]), )
call_827 = func_243_call(relay.reshape(const_825.astype('uint16'), []), relay.reshape(var_826.astype('uint16'), [4, 14]), )
func_403_call = mod.get_global_var('func_403')
func_407_call = mutated_mod.get_global_var('func_407')
var_829 = relay.var("var_829", dtype = "float64", shape = (81,))#candidate|829|(81,)|var|float64
const_830 = relay.const([[-1.833157,1.933489,-9.288908,-8.263840,5.692572,-2.789628,3.144948,-5.310603,8.867602,5.325207,6.236209,0.015620,-4.297948,-2.477444,7.299861],[-8.024745,-6.733263,8.364988,-1.568047,-8.146669,1.514920,-1.402613,5.874177,6.297480,-6.333895,2.540077,-4.852309,-5.367499,1.526107,7.066641],[1.949507,-4.012313,6.363529,-7.006412,-5.522694,-0.784385,9.234773,-7.189105,7.279321,7.725388,-9.979799,4.573464,3.918851,-4.780147,6.214272],[-9.454982,1.501721,-7.697597,-5.548690,1.124151,-5.883358,-6.613476,-1.343781,-7.565455,-8.865434,2.974811,-3.610534,5.903842,-2.880079,-6.461355],[-5.840947,-0.791417,5.523372,-5.005900,3.815638,-1.900233,1.379961,7.472614,-4.570098,4.694354,5.368423,0.465162,-6.834412,-9.792795,-0.795404],[-2.702609,2.793295,6.799974,7.200222,-1.781239,6.167625,-2.360636,-3.455900,-6.951646,2.700591,9.117145,-7.647719,-1.310961,-0.376385,9.841007],[2.361097,6.968644,-0.875596,-4.839208,-3.705461,-6.518496,7.826373,7.087611,-2.763180,-9.897715,4.035647,-5.174197,8.523924,-2.227028,6.865708],[-2.439636,6.919068,3.668511,-7.365782,-5.421552,8.473626,-2.004943,7.068503,-5.149301,8.409873,-1.399059,-1.374122,-8.614785,1.051608,0.747952],[-1.408434,-5.480725,6.947439,0.900321,2.625345,1.197821,-7.265619,2.946547,0.853296,8.325866,8.600780,1.318000,0.768679,6.403012,7.153919],[2.675196,-3.726468,7.789152,-1.983913,-6.718683,-2.530629,-0.110503,6.887860,3.348386,1.737590,-1.175403,-6.643015,-2.864664,9.143777,-7.869471],[9.077806,5.406874,1.812670,2.861110,7.025236,2.059963,4.777157,1.473356,-4.191677,2.805204,3.824143,-7.026902,-7.149628,4.368409,-3.949365],[-5.940262,-1.262661,5.327431,-7.554288,7.788875,-0.740163,-3.592188,-3.485710,0.258224,-6.342382,-2.576131,-3.198496,4.288564,1.267217,-2.944218],[-4.200138,0.360142,-5.758607,5.424725,-7.355712,0.287023,9.857075,-2.459385,1.304329,-4.233063,4.258243,0.366113,6.068745,-3.989402,-3.850118],[-7.404220,7.160870,5.407948,3.313752,-0.203868,5.021666,-2.669422,9.082870,-7.988237,-3.257543,-1.502580,5.065595,9.382231,-8.317121,5.720640],[7.080048,-7.067284,-3.851377,-1.674388,-4.432782,6.408123,-7.130158,-8.305125,-5.301458,-4.948786,-2.769478,-1.277744,3.545366,-8.036850,3.948920],[-9.070053,-4.559030,6.798248,-1.040423,-9.690603,1.855515,-7.061008,3.500309,6.199699,6.378049,7.694080,1.239520,8.599461,1.613396,-2.089273],[9.699804,2.313949,0.925803,-8.059822,2.908389,-4.609508,-0.942730,-2.460321,-2.642784,-9.761912,-3.414212,5.604382,7.651017,-6.586898,-9.513329],[-9.419996,2.295171,0.674025,-3.054312,-7.212089,-2.564301,-1.851459,5.426251,4.263524,4.680729,3.774143,3.042514,6.624963,5.882090,6.943844],[-6.360137,6.396971,-1.774890,8.766633,3.716032,-6.014265,-5.477897,-5.064507,4.549593,2.901461,9.284603,-6.533487,7.977925,-3.647751,1.258774],[9.832201,0.481556,2.139470,6.874997,-6.262721,-0.625176,8.532674,-1.933360,-3.063454,9.546422,-7.600350,-5.596643,9.358263,-2.797009,-2.489119],[7.668184,0.659794,0.876318,-2.061502,8.412467,2.730484,6.126386,7.495277,-8.778940,-7.122069,9.855144,3.629135,6.351820,-8.450011,-7.795333],[-5.437574,-6.509379,-0.418606,-7.041172,-4.296524,2.071340,-0.013911,-1.412328,-0.517043,7.785132,-0.539540,9.809169,2.152893,-2.060090,-6.473534],[-9.575301,4.544501,4.449398,-5.199599,-2.852149,9.493474,-7.697844,9.688926,2.421039,-3.457051,-3.274740,-2.513059,-3.700900,-9.052074,0.800001],[9.382003,7.203833,-5.819118,-8.675741,1.557173,5.871946,-3.639807,-8.705005,7.203854,-2.759264,-9.722127,-3.923260,-3.487859,-5.490239,-6.124664],[-5.443233,-9.403155,-2.333003,-2.855842,6.070511,9.807497,-4.848619,9.500835,2.412732,2.216111,-2.009504,-0.550368,-1.863219,7.919157,-9.081693],[-8.725911,3.876735,-5.779987,5.089846,-2.905463,-1.797671,8.062878,4.635068,2.773790,6.198092,-0.144803,7.659465,9.440077,-0.143449,-3.085608],[-2.671178,8.262145,1.398711,8.135290,-2.803793,-9.067433,6.972125,5.561432,-6.542149,-0.506413,4.253802,4.669644,2.243827,-9.652875,6.015926],[-0.799718,9.160536,3.468116,-0.701483,9.802178,-0.256210,2.507984,-1.179173,-2.672852,-9.058935,-8.761089,-5.766210,-6.874609,-1.662845,9.153708],[8.231131,1.192360,3.890682,-9.961538,1.218758,-1.196310,0.553906,-4.088588,-6.255474,-8.050177,8.980101,5.950340,-3.791421,6.595874,-6.176141],[-8.047921,9.870355,6.821989,-0.995843,-6.545487,9.284874,7.050751,-3.926564,-5.908977,-1.146119,1.057215,-2.312096,9.459681,-5.517587,6.600057],[8.175963,6.327428,-8.055832,5.998126,-7.491755,-5.210317,-0.573026,-7.834406,6.859263,-1.600695,3.916560,7.219678,-4.289181,-5.639436,-8.209537],[2.997128,0.420432,-7.739105,1.313441,-9.634612,9.175809,-9.496815,-6.054639,-0.156303,2.867783,3.446579,8.746555,-1.917193,-4.754023,-7.443567],[3.182660,-0.833770,-5.965052,-3.359214,2.098037,8.545310,3.616500,-1.995399,-6.105846,-7.568910,6.123744,-6.698323,-1.886942,-1.729808,5.763810],[7.731807,1.189177,1.506956,-6.663707,8.008136,-0.976997,-4.312249,-5.672216,7.475951,-8.311268,6.747101,-0.157007,-8.431502,-1.296431,0.501623],[2.235411,-5.963197,-9.409576,7.136973,5.689577,-0.751829,9.178461,-8.455665,-2.373688,3.804260,7.192565,-5.213206,-6.734272,-7.874145,-1.705026],[5.238766,1.192189,-9.809599,1.437147,-8.719736,-3.025143,7.154805,-5.823534,-5.625311,2.883240,2.069520,-8.760197,-2.536478,-7.738449,7.915987],[-0.856706,-4.258537,7.941017,-2.382855,-7.617474,0.838781,0.287377,3.051458,-4.492535,-5.094974,4.128134,-2.858873,-2.021548,-7.687492,-5.670269],[0.699379,-0.855020,-6.237403,-7.074758,3.100302,4.259903,0.580118,6.439882,-5.549433,-4.041905,2.646563,-1.641579,1.782831,9.391335,2.772660],[-8.872973,-9.182040,7.382302,-2.206975,0.086649,-3.290275,-3.639825,-3.796319,-9.009263,-1.033201,6.167805,-0.460359,2.748810,-4.207345,-4.228185],[-5.791620,-2.964609,0.524834,0.627357,6.057328,9.487173,-8.058251,-8.590246,2.204876,0.961438,3.121668,-8.426733,6.421908,-3.184561,-1.596672],[7.355701,6.114106,3.845602,6.374765,-9.775118,-3.307966,6.586867,7.864644,9.103086,-4.835955,-4.360808,1.390511,-7.025083,3.882987,3.762299],[1.277214,-1.011811,-6.935578,8.564687,7.919351,4.120221,-4.041079,-0.171679,-7.101877,-5.109503,-4.038804,-7.298387,3.936693,-0.314904,3.750450],[-6.815297,2.700793,7.813537,4.093077,8.721680,-1.209719,-0.276134,5.380318,8.418927,-2.182239,3.943763,-7.781098,-9.132265,-7.785583,-9.239716],[-9.880571,-0.684491,-8.554313,-1.024882,-7.020356,-5.518945,-9.280193,0.265089,1.044936,5.602104,2.990862,-8.446209,-6.787739,5.605463,-5.629432],[2.223044,4.847742,-9.900281,5.706414,7.780197,-5.019151,2.631397,-9.526665,2.119324,9.583736,2.146463,7.037009,-9.028193,-3.378335,-9.477956],[-2.291079,-1.816811,4.166788,3.064147,8.237402,-7.650892,-5.435469,-4.327746,5.420793,-7.701910,-9.081144,8.336945,2.619124,9.741262,5.372760],[-9.730145,9.330420,3.098041,3.302704,-6.793069,7.930762,2.224690,-6.004139,-8.781584,5.496333,-0.468233,-5.367160,3.071422,6.170645,-0.873717],[-0.640747,-6.077390,4.661392,-5.284941,-6.852145,7.103633,0.695987,9.939711,-1.642285,3.562157,-8.984292,0.371544,4.299814,2.486690,8.615048],[-6.927582,-7.418533,-0.020966,-9.912409,7.706872,2.120497,6.513847,6.325404,-9.707689,-3.712239,6.771831,4.619902,-7.049978,6.481897,4.064358],[4.509193,-6.869972,0.159205,3.821204,4.135285,-5.218051,6.872704,2.016825,8.837493,-9.475611,-4.316085,-2.924944,-5.654480,-5.803339,-1.020621],[-2.179106,6.467610,-1.499901,0.284849,9.585047,-7.461897,7.223757,2.438302,-3.599212,-1.254109,-9.529477,-8.554589,-9.430533,2.251977,-8.853989],[8.719715,3.234964,7.799281,8.515836,0.041573,6.038938,-5.451741,5.061812,0.700725,-8.100669,-6.340189,-4.449223,-6.049493,4.427376,-5.014152],[2.954780,0.079299,-4.445683,-7.705375,7.282708,-3.502533,4.830859,-2.012817,-5.912631,-3.623952,1.967197,-1.481547,2.486512,0.500597,-8.132031],[-2.823451,2.164564,0.395158,-4.831724,8.472731,6.945030,-0.726000,-7.286959,-8.482931,5.710137,-1.191187,9.865587,-2.667991,2.370506,-4.838411],[9.149480,3.346661,-8.454116,0.862905,1.902295,-1.158203,-6.333828,5.799457,2.665235,-8.831570,1.198031,-9.031489,2.567350,4.068600,-6.065531],[-7.124374,7.560787,-8.497178,-8.821983,-5.458004,-0.519520,4.508700,-5.814316,8.414385,-1.536714,-3.618542,-4.922206,-4.225822,3.584062,-6.116239],[2.578693,-3.927902,0.409804,-2.573876,-2.288603,7.403361,-9.008703,3.103575,2.029946,-6.194479,-6.536775,1.655856,7.511644,-3.834409,-0.532928],[-4.299513,-2.257800,-4.754111,5.054718,7.408731,9.469506,-5.633405,2.088497,-1.599997,9.262161,-7.427289,4.458322,-0.744603,-2.370541,-8.463981],[-7.253697,-1.104638,9.080871,-7.743104,-1.487169,-2.744648,-9.699018,1.572602,-1.375039,-7.002328,-9.453261,-6.793311,0.518691,6.446682,7.212639],[-0.845535,-4.726774,-2.314391,4.442172,0.353126,3.760492,7.344936,-9.559481,3.495811,9.622955,-3.425959,4.162418,2.109408,-5.599022,-9.829371],[-2.478912,9.910190,6.707387,-5.435233,-2.606036,-0.846844,4.180001,0.953378,-8.776623,1.100927,-9.196253,-4.442727,-3.464823,-4.468891,-1.551117],[-7.598554,0.763350,9.435898,-4.966461,1.671299,7.148134,-4.536561,-3.867577,8.915807,-4.161554,-8.405435,-7.869191,-3.771386,-7.293122,3.022633],[5.898836,-3.584273,2.647735,-5.697449,4.641934,-0.855115,2.777475,-6.406455,-9.965569,8.664377,-1.373277,-0.690302,-6.430037,0.411851,4.823873],[7.051535,-9.937770,-9.322221,4.005826,7.157675,-7.374642,-3.834726,-4.747853,3.426193,-7.820023,-7.430685,-1.952512,8.898844,4.598280,-0.448558],[-6.639406,-6.756710,6.381292,-3.710275,0.230681,5.579471,-3.258837,2.867376,-3.627633,3.111861,3.418386,0.786053,-0.842622,-4.262191,7.348311],[4.876807,-5.445164,0.996195,-8.756665,8.211392,1.509499,4.006082,-4.131779,6.803649,-8.418573,6.856501,-3.397196,3.201647,5.244552,-9.904368],[-2.035306,-3.870029,0.932516,-4.785556,-1.275106,6.410120,7.424509,5.425403,-4.088272,-6.750334,1.357448,-6.176691,9.615262,0.238924,-3.713003],[5.150008,1.337222,1.155342,2.943257,-0.283655,0.071278,-8.814121,-8.550774,-6.303540,-6.267744,3.809048,-0.877276,0.776257,-3.394224,8.282608],[-0.838877,6.496976,-4.636073,4.819906,5.871309,1.677775,4.246055,5.004106,9.027796,-8.279633,-7.959161,-0.024763,-0.328119,-0.324502,-0.979913],[5.917396,4.718493,-5.792627,8.495976,8.403094,8.057377,-9.200904,-0.062872,-3.423727,6.477780,9.470004,9.623549,1.111077,0.982964,-1.724181],[7.375204,6.841182,-8.334516,-3.743206,8.043901,-2.432721,0.728596,2.934866,-2.729260,8.112243,6.769705,-1.525996,-0.345170,-3.742449,4.342197],[3.550768,-2.720329,5.049382,-5.789253,-9.297752,-4.304913,-5.453368,9.722807,0.459484,-4.856629,-3.074106,-9.618362,-0.216642,-4.928271,-7.310474],[9.632482,-5.989081,-8.478412,-3.025949,7.772867,-8.420368,-0.870913,8.716464,7.701350,7.577861,-8.423217,1.394793,-6.575791,1.208988,-3.485140],[0.806149,-4.394461,8.837465,9.963301,1.951794,-8.826024,-9.499288,3.640861,0.590276,1.111341,-9.327303,-2.313024,-6.302114,4.464637,7.935166],[2.467977,4.020060,-8.767908,7.654889,-4.419716,-2.767506,-8.695970,0.831082,6.173050,-2.086774,-5.655523,-1.336508,-8.861026,6.732166,-3.918453],[6.502387,0.136855,7.253703,5.191587,-0.761054,-0.091816,2.956508,4.513710,-3.923135,0.971593,5.600435,5.646472,-6.313475,-5.156293,3.153113],[-4.840047,4.025670,3.981470,2.777935,9.668079,-1.010631,-4.734412,-6.688834,3.679871,-3.288264,-8.818155,-4.307011,9.102393,1.062844,-3.268969],[0.439748,-9.657372,7.764335,3.810476,-1.606411,-0.508567,-9.156869,-9.837626,-4.911617,6.800764,-4.503172,5.552769,9.220560,0.291638,-8.643306],[1.629942,-9.256120,-1.338340,5.256687,-7.113820,-0.917739,2.759354,-1.850661,-5.605957,9.721912,6.818365,0.548400,-0.616704,-9.170537,0.435940],[2.520182,-5.987281,-5.356925,8.885220,-5.841906,-8.560854,-6.301731,-4.279276,3.468442,-0.155820,-7.176704,7.771736,-5.842420,9.964839,3.051625],[8.524919,0.257301,2.776495,8.066233,-7.814557,-0.182076,-2.766030,-2.954516,9.178563,2.857789,-8.349721,-3.450175,3.757239,0.547140,1.935629]], dtype = "float64")#candidate|830|(81, 15)|const|float64
call_828 = relay.TupleGetItem(func_403_call(relay.reshape(var_829.astype('float64'), [9, 1, 9]), relay.reshape(const_830.astype('float64'), [9, 15, 9]), ), 0)
call_831 = relay.TupleGetItem(func_407_call(relay.reshape(var_829.astype('float64'), [9, 1, 9]), relay.reshape(const_830.astype('float64'), [9, 15, 9]), ), 0)
output = relay.Tuple([bop_800,bop_821,call_824,const_825,var_826,call_828,var_829,const_830,])
output2 = relay.Tuple([bop_800,bop_821,call_827,const_825,var_826,call_831,var_829,const_830,])
func_832 = relay.Function([var_779,var_780,var_803,var_820,var_826,var_829,], output)
mod['func_832'] = func_832
mod = relay.transform.InferType()(mod)
mutated_mod['func_832'] = func_832
mutated_mod = relay.transform.InferType()(mutated_mod)
func_832_call = mutated_mod.get_global_var('func_832')
var_834 = relay.var("var_834", dtype = "int8", shape = (14, 9))#candidate|834|(14, 9)|var|int8
var_835 = relay.var("var_835", dtype = "int8", shape = (14, 9))#candidate|835|(14, 9)|var|int8
var_836 = relay.var("var_836", dtype = "float32", shape = (14, 9))#candidate|836|(14, 9)|var|float32
var_837 = relay.var("var_837", dtype = "float32", shape = (14, 9))#candidate|837|(14, 9)|var|float32
var_838 = relay.var("var_838", dtype = "uint16", shape = (56,))#candidate|838|(56,)|var|uint16
var_839 = relay.var("var_839", dtype = "float64", shape = (81,))#candidate|839|(81,)|var|float64
call_833 = func_832_call(var_834,var_835,var_836,var_837,var_838,var_839,)
output = call_833
func_840 = relay.Function([var_834,var_835,var_836,var_837,var_838,var_839,], output)
mutated_mod['func_840'] = func_840
mutated_mod = relay.transform.InferType()(mutated_mod)
const_867 = relay.const([[-5.113208,1.053476,-3.113852,9.251557,8.842057],[9.433312,9.949846,-0.907726,5.955683,5.357303],[-9.005083,4.586385,4.066424,6.268159,-7.104027],[5.643764,-6.832242,-2.199865,-9.730417,-0.520762],[1.260039,0.919814,-2.196629,-6.935605,-8.347490],[7.518448,5.974508,-0.311358,4.616882,8.783576],[-5.029949,8.932502,-1.250043,6.228695,7.079494]], dtype = "float64")#candidate|867|(7, 5)|const|float64
uop_868 = relay.sin(const_867.astype('float64')) # shape=(7, 5)
func_103_call = mod.get_global_var('func_103')
func_106_call = mutated_mod.get_global_var('func_106')
const_873 = relay.const([6.286338,-5.042814,-8.576054,-3.914673,-5.085478,2.819579,-4.366141,4.731075,9.647471,-1.336705,-1.426110,5.623342,-7.210559,2.542794,-0.312732,8.319558,6.322733,-6.212877,9.226567,-6.310295,-4.639558,-2.135881,5.094616,1.426153,9.815779,3.370215,0.618399,3.146547,-7.766871,1.826025,6.478122,-4.779351,-0.758010,-9.054721,7.883615,-2.937572,9.838934,9.135182,-1.903624,5.297291,-1.436963,4.012496,2.382694,-4.302705,3.832113,1.850773,-5.576900,7.718889,-2.639047,-5.894051,-3.060707,1.322469,-0.215515,-7.197314,1.622906,-2.631987,-3.027948,3.875422,9.845277,2.522559,7.438418,8.895788,-0.852109,-3.841828,2.493050,7.694324,7.284026,0.418564,9.156857,-8.082730,3.070666,-4.006759,-5.984982,2.704984,4.937281,-0.779617,8.303813,-0.563652,-0.952861,-1.370986,-6.175090,3.826943,-0.903775,7.889721,-1.869193,-7.302521,-1.189839,9.339308,-1.050243,-6.968690,8.491743,-1.054423,5.892790,1.567007,-3.488812,0.209908,3.311599,-3.587236,1.384165,2.487073,-2.045301,5.107476,-0.631789,2.452620,5.287007,0.149989,4.467938,-7.764534,-5.112563,-6.527192,9.121369,6.079600,3.247073,7.363533,2.955822,-0.403052,-2.921009,9.120897,1.654397,-0.320643,-2.586457,6.948318,8.970825,-2.380373,4.532316,-8.466868,-5.084461,6.040960,-0.793330,-7.160834,1.673725,2.662450,-3.442676,3.020082,0.843385,7.690221,8.634386,4.518255,-8.739874,-2.991847,3.553233,-5.050821,4.981030,6.738998,-3.122788,4.527405,-6.903050,-2.434576,8.406089,-8.063291,0.538455,6.042153,6.832295,0.119675,-4.779159,0.281587,0.816288,-2.108890,-3.454852,-3.201556,2.923797,-2.124796,-0.698064,6.750312,-7.452029,-6.462647,1.714341,-8.018853,1.883014,-0.944440,-3.084711,5.029377,7.653494,-8.907574,6.336528,9.485262,-1.376188,-0.192071,-2.225417,8.724257,-2.161366,-5.994474,5.225104,-7.157487,-8.933252,-3.755916,-4.801837,-5.741295,6.819558,7.488684,8.166552,-8.691653,9.856312,-0.174960,-1.002227,-0.415381,-5.940427,-9.204983,-1.472963,-2.986479,9.318331,6.357725,-6.940422,2.460064,-5.639686,-2.230558,1.070950,6.674139,3.516615,-0.640066,9.352232,-9.080793,2.749086,6.522989,1.399467,5.910896,7.857766,4.125676,-3.475466,4.162796,1.874712,6.107760,-3.022589,5.870495,-3.654709,1.368090,8.866166,5.300788,-4.672129,-2.284755,-5.749835,-3.842690,-1.738409,0.527928,9.684675,6.289153,-1.800712,8.842746,0.977923,1.754535,1.091867,-3.777944,-6.592823,0.175045,-2.973212,5.272427,7.484587,-4.831909,2.653044,9.565948,9.083335,7.039214,3.785617,-0.855340,-7.660583,-8.791386,2.956306,-7.349050,2.951115,-7.612806,4.589938,-2.129258,-4.623226,-0.260473,6.572970,-8.534311,-2.696040,-2.581019,-9.265952,-5.749908,-4.564388,-0.214238,2.023050,-7.231804,1.789073,3.160216,-4.681082,7.380208,-6.839851,-5.911525,5.771960,-1.978886,-7.694996,6.833788,-0.339932,3.223680,5.845539,-7.345558,-9.141979,2.417874,0.847022,-3.139361,-0.013705,-7.368541], dtype = "float64")#candidate|873|(294,)|const|float64
call_872 = relay.TupleGetItem(func_103_call(relay.reshape(const_873.astype('float64'), [7, 6, 7])), 6)
call_874 = relay.TupleGetItem(func_106_call(relay.reshape(const_873.astype('float64'), [7, 6, 7])), 6)
output = relay.Tuple([uop_868,call_872,const_873,])
output2 = relay.Tuple([uop_868,call_874,const_873,])
func_877 = relay.Function([], output)
mod['func_877'] = func_877
mod = relay.transform.InferType()(mod)
output = func_877()
func_878 = relay.Function([], output)
mutated_mod['func_878'] = func_878
mutated_mod = relay.transform.InferType()(mutated_mod)
var_897 = relay.var("var_897", dtype = "float32", shape = (8, 5, 1))#candidate|897|(8, 5, 1)|var|float32
uop_898 = relay.acosh(var_897.astype('float32')) # shape=(8, 5, 1)
uop_902 = relay.log2(uop_898.astype('float64')) # shape=(8, 5, 1)
bop_905 = relay.bitwise_xor(uop_902.astype('uint32'), relay.reshape(uop_898.astype('uint32'), relay.shape_of(uop_902))) # shape=(8, 5, 1)
bop_908 = relay.maximum(var_897.astype('float64'), relay.reshape(uop_902.astype('float64'), relay.shape_of(var_897))) # shape=(8, 5, 1)
bop_913 = relay.mod(bop_908.astype('float32'), relay.reshape(uop_898.astype('float32'), relay.shape_of(bop_908))) # shape=(8, 5, 1)
output = relay.Tuple([bop_905,bop_913,])
output2 = relay.Tuple([bop_905,bop_913,])
func_918 = relay.Function([var_897,], output)
mod['func_918'] = func_918
mod = relay.transform.InferType()(mod)
mutated_mod['func_918'] = func_918
mutated_mod = relay.transform.InferType()(mutated_mod)
var_919 = relay.var("var_919", dtype = "float32", shape = (8, 5, 1))#candidate|919|(8, 5, 1)|var|float32
func_918_call = mutated_mod.get_global_var('func_918')
call_920 = func_918_call(var_919)
output = call_920
func_921 = relay.Function([var_919], output)
mutated_mod['func_921'] = func_921
mutated_mod = relay.transform.InferType()(mutated_mod)
var_931 = relay.var("var_931", dtype = "float64", shape = (4, 3))#candidate|931|(4, 3)|var|float64
var_932 = relay.var("var_932", dtype = "float64", shape = (4, 3))#candidate|932|(4, 3)|var|float64
bop_933 = relay.mod(var_931.astype('float64'), relay.reshape(var_932.astype('float64'), relay.shape_of(var_931))) # shape=(4, 3)
func_730_call = mod.get_global_var('func_730')
func_737_call = mutated_mod.get_global_var('func_737')
const_938 = relay.const([False,False,True,True,True,True,True,True,True,False,False,False,False,True,False,False,False,True,True,False,False,True,True,True,False,False,False,False,False,True,True,False,True,False,True,False,False,False,True,True,False,True,True,False,False,False,False,True,True,True,True,True,True,False,False,True,False,True,False,False,False,True,False,True,True,True,True,True,True,True,True,True,False,False,True,True,False,True,False,True,False,True,True,True,True,True,True,True,False,True,True,False,False,True,True,True,False,True,True,True,False,False,False,False,False,True,True,True,False,True,False,False,False,True,False,True,True,True,False,True,False,True,False,False,False,True,True,False,False,False,True,False,False,True,True,True,False,False,False,False,False,False,True,False], dtype = "bool")#candidate|938|(144,)|const|bool
var_939 = relay.var("var_939", dtype = "float64", shape = (10,))#candidate|939|(10,)|var|float64
call_937 = relay.TupleGetItem(func_730_call(relay.reshape(const_938.astype('bool'), [9, 4, 4]), relay.reshape(const_938.astype('bool'), [9, 4, 4]), relay.reshape(const_938.astype('uint64'), [9, 4, 4]), relay.reshape(const_938.astype('uint64'), [9, 4, 4]), relay.reshape(var_939.astype('float64'), [10,]), ), 0)
call_940 = relay.TupleGetItem(func_737_call(relay.reshape(const_938.astype('bool'), [9, 4, 4]), relay.reshape(const_938.astype('bool'), [9, 4, 4]), relay.reshape(const_938.astype('uint64'), [9, 4, 4]), relay.reshape(const_938.astype('uint64'), [9, 4, 4]), relay.reshape(var_939.astype('float64'), [10,]), ), 0)
func_243_call = mod.get_global_var('func_243')
func_246_call = mutated_mod.get_global_var('func_246')
var_943 = relay.var("var_943", dtype = "uint16", shape = ())#candidate|943|()|var|uint16
const_944 = relay.const([-9,1,-7,9,-8,3,-5,-3,-2,-2,-10,6,6,7,10,-9,-8,-10,3,1,8,-3,10,7,-2,7,-7,-5,8,6,10,5,-8,-3,4,2,-9,4,-10,3,3,6,-3,-4,-7,-9,8,2,-8,-8,-10,4,-7,1,1,-1], dtype = "uint16")#candidate|944|(56,)|const|uint16
call_942 = func_243_call(relay.reshape(var_943.astype('uint16'), []), relay.reshape(const_944.astype('uint16'), [4, 14]), )
call_945 = func_243_call(relay.reshape(var_943.astype('uint16'), []), relay.reshape(const_944.astype('uint16'), [4, 14]), )
func_26_call = mod.get_global_var('func_26')
func_30_call = mutated_mod.get_global_var('func_30')
var_950 = relay.var("var_950", dtype = "float32", shape = (13,))#candidate|950|(13,)|var|float32
call_949 = relay.TupleGetItem(func_26_call(relay.reshape(var_950.astype('float32'), [13,]), relay.reshape(var_950.astype('float32'), [13,]), ), 0)
call_951 = relay.TupleGetItem(func_30_call(relay.reshape(var_950.astype('float32'), [13,]), relay.reshape(var_950.astype('float32'), [13,]), ), 0)
output = relay.Tuple([bop_933,call_937,const_938,var_939,call_942,var_943,const_944,call_949,var_950,])
output2 = relay.Tuple([bop_933,call_940,const_938,var_939,call_945,var_943,const_944,call_951,var_950,])
func_962 = relay.Function([var_931,var_932,var_939,var_943,var_950,], output)
mod['func_962'] = func_962
mod = relay.transform.InferType()(mod)
mutated_mod['func_962'] = func_962
mutated_mod = relay.transform.InferType()(mutated_mod)
func_962_call = mutated_mod.get_global_var('func_962')
var_964 = relay.var("var_964", dtype = "float64", shape = (4, 3))#candidate|964|(4, 3)|var|float64
var_965 = relay.var("var_965", dtype = "float64", shape = (4, 3))#candidate|965|(4, 3)|var|float64
var_966 = relay.var("var_966", dtype = "float64", shape = (10,))#candidate|966|(10,)|var|float64
var_967 = relay.var("var_967", dtype = "uint16", shape = ())#candidate|967|()|var|uint16
var_968 = relay.var("var_968", dtype = "float32", shape = (13,))#candidate|968|(13,)|var|float32
call_963 = func_962_call(var_964,var_965,var_966,var_967,var_968,)
output = call_963
func_969 = relay.Function([var_964,var_965,var_966,var_967,var_968,], output)
mutated_mod['func_969'] = func_969
mutated_mod = relay.transform.InferType()(mutated_mod)
var_977 = relay.var("var_977", dtype = "float32", shape = (16, 12, 3))#candidate|977|(16, 12, 3)|var|float32
uop_978 = relay.sin(var_977.astype('float32')) # shape=(16, 12, 3)
uop_983 = relay.exp(uop_978.astype('float64')) # shape=(16, 12, 3)
output = uop_983
output2 = uop_983
func_985 = relay.Function([var_977,], output)
mod['func_985'] = func_985
mod = relay.transform.InferType()(mod)
mutated_mod['func_985'] = func_985
mutated_mod = relay.transform.InferType()(mutated_mod)
var_986 = relay.var("var_986", dtype = "float32", shape = (16, 12, 3))#candidate|986|(16, 12, 3)|var|float32
func_985_call = mutated_mod.get_global_var('func_985')
call_987 = func_985_call(var_986)
output = call_987
func_988 = relay.Function([var_986], output)
mutated_mod['func_988'] = func_988
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1017 = relay.var("var_1017", dtype = "float64", shape = (12, 9))#candidate|1017|(12, 9)|var|float64
uop_1018 = relay.asinh(var_1017.astype('float64')) # shape=(12, 9)
func_918_call = mod.get_global_var('func_918')
func_921_call = mutated_mod.get_global_var('func_921')
var_1023 = relay.var("var_1023", dtype = "float32", shape = (40,))#candidate|1023|(40,)|var|float32
call_1022 = relay.TupleGetItem(func_918_call(relay.reshape(var_1023.astype('float32'), [8, 5, 1])), 0)
call_1024 = relay.TupleGetItem(func_921_call(relay.reshape(var_1023.astype('float32'), [8, 5, 1])), 0)
func_456_call = mod.get_global_var('func_456')
func_458_call = mutated_mod.get_global_var('func_458')
var_1039 = relay.var("var_1039", dtype = "float32", shape = (1,))#candidate|1039|(1,)|var|float32
call_1038 = relay.TupleGetItem(func_456_call(relay.reshape(var_1039.astype('float32'), [1,])), 0)
call_1040 = relay.TupleGetItem(func_458_call(relay.reshape(var_1039.astype('float32'), [1,])), 0)
uop_1041 = relay.exp(uop_1018.astype('float64')) # shape=(12, 9)
bop_1043 = relay.bitwise_or(uop_1041.astype('int16'), var_1039.astype('int16')) # shape=(12, 9)
func_403_call = mod.get_global_var('func_403')
func_407_call = mutated_mod.get_global_var('func_407')
const_1047 = relay.const([2.823572,0.797330,4.975781,0.526690,8.874319,8.947126,5.500150,7.310094,-4.422878,-4.964700,-7.528347,-1.160165,8.879043,6.776079,-9.419464,-9.312549,9.938672,2.565694,-5.501817,5.586459,-6.988445,8.141541,-9.439697,6.320040,7.374084,3.276877,2.917428,-3.602157,9.027060,4.661592,-4.732813,-5.510053,-7.591843,-2.295623,-1.994242,-1.152300,-2.023128,5.929485,-2.298823,6.613697,-7.740955,9.706548,-3.156864,-1.623428,-3.564415,-4.752791,4.009270,-2.021548,-7.704842,4.023853,9.489248,1.246786,4.447062,-3.265730,9.947023,8.883186,-4.293791,9.265667,-1.867325,0.030014,-1.905202,4.213974,-0.677823,8.584690,-1.362288,9.801336,-3.720203,6.999748,2.091865,3.229318,0.459458,-7.021232,-3.079460,6.468692,-5.498171,1.011286,-6.243947,-1.889027,-1.213841,4.226417,-0.838137], dtype = "float64")#candidate|1047|(81,)|const|float64
var_1048 = relay.var("var_1048", dtype = "float64", shape = (1, 1215))#candidate|1048|(1, 1215)|var|float64
call_1046 = relay.TupleGetItem(func_403_call(relay.reshape(const_1047.astype('float64'), [9, 1, 9]), relay.reshape(var_1048.astype('float64'), [9, 15, 9]), ), 1)
call_1049 = relay.TupleGetItem(func_407_call(relay.reshape(const_1047.astype('float64'), [9, 1, 9]), relay.reshape(var_1048.astype('float64'), [9, 15, 9]), ), 1)
const_1050 = relay.const([[5.436671,-9.956452,1.120091,-5.994644,-6.236832,-8.042596,3.349764,1.313676,-8.972941],[1.696406,6.628927,4.208649,9.924362,-2.682620,0.026681,-1.456509,-0.684626,-1.066719],[7.470245,-0.188110,-2.675315,3.218816,3.126461,0.426888,-5.802156,-2.413770,5.245787],[-8.778435,6.772626,8.092585,-2.000293,7.883153,-0.744777,1.754365,1.638143,6.420009],[-7.320700,3.791665,5.456411,-9.896965,4.620842,4.690322,-9.071832,9.842034,2.479659],[-1.441032,-4.238868,-0.402163,7.492179,8.330127,5.639399,-9.581603,1.316157,-8.612744],[3.951480,-5.571256,8.500475,1.212812,7.161489,7.040895,4.541834,0.438806,-7.090633],[-5.090422,1.323286,-9.770441,-5.459067,-6.442489,-5.020783,-0.073711,4.172631,-6.556263],[-9.318825,-0.178014,6.089107,-2.149358,2.932678,-4.830690,-6.120207,2.500238,-4.792847],[-3.347351,4.126711,-3.397691,5.248185,4.424538,-1.686947,-5.615589,0.674789,-7.963359],[-7.496150,7.381810,-3.709233,7.292455,-6.229354,-8.284004,-8.388257,3.426945,-1.837799],[5.756130,-2.531363,-4.018643,2.121695,0.716571,-3.340827,9.840668,2.544255,-4.228433]], dtype = "float64")#candidate|1050|(12, 9)|const|float64
bop_1051 = relay.power(uop_1041.astype('float32'), relay.reshape(const_1050.astype('float32'), relay.shape_of(uop_1041))) # shape=(12, 9)
bop_1058 = relay.multiply(bop_1051.astype('float64'), relay.reshape(const_1050.astype('float64'), relay.shape_of(bop_1051))) # shape=(12, 9)
func_456_call = mod.get_global_var('func_456')
func_458_call = mutated_mod.get_global_var('func_458')
call_1061 = relay.TupleGetItem(func_456_call(relay.reshape(call_1038.astype('float32'), [1,])), 0)
call_1062 = relay.TupleGetItem(func_458_call(relay.reshape(call_1038.astype('float32'), [1,])), 0)
bop_1064 = relay.minimum(bop_1051.astype('uint16'), call_1061.astype('uint16')) # shape=(12, 9)
bop_1067 = relay.minimum(bop_1051.astype('uint16'), call_1062.astype('uint16')) # shape=(12, 9)
output = relay.Tuple([call_1022,var_1023,call_1038,bop_1043,call_1046,const_1047,var_1048,bop_1058,bop_1064,])
output2 = relay.Tuple([call_1024,var_1023,call_1040,bop_1043,call_1049,const_1047,var_1048,bop_1058,bop_1067,])
func_1072 = relay.Function([var_1017,var_1023,var_1039,var_1048,], output)
mod['func_1072'] = func_1072
mod = relay.transform.InferType()(mod)
var_1073 = relay.var("var_1073", dtype = "float64", shape = (12, 9))#candidate|1073|(12, 9)|var|float64
var_1074 = relay.var("var_1074", dtype = "float32", shape = (40,))#candidate|1074|(40,)|var|float32
var_1075 = relay.var("var_1075", dtype = "float32", shape = (1,))#candidate|1075|(1,)|var|float32
var_1076 = relay.var("var_1076", dtype = "float64", shape = (1, 1215))#candidate|1076|(1, 1215)|var|float64
output = func_1072(var_1073,var_1074,var_1075,var_1076,)
func_1077 = relay.Function([var_1073,var_1074,var_1075,var_1076,], output)
mutated_mod['func_1077'] = func_1077
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1079 = relay.var("var_1079", dtype = "int32", shape = (9, 14, 7))#candidate|1079|(9, 14, 7)|var|int32
var_1080 = relay.var("var_1080", dtype = "int32", shape = (9, 14, 7))#candidate|1080|(9, 14, 7)|var|int32
bop_1081 = relay.bitwise_and(var_1079.astype('int32'), relay.reshape(var_1080.astype('int32'), relay.shape_of(var_1079))) # shape=(9, 14, 7)
bop_1093 = relay.multiply(var_1079.astype('int16'), relay.reshape(bop_1081.astype('int16'), relay.shape_of(var_1079))) # shape=(9, 14, 7)
output = relay.Tuple([bop_1093,])
output2 = relay.Tuple([bop_1093,])
func_1100 = relay.Function([var_1079,var_1080,], output)
mod['func_1100'] = func_1100
mod = relay.transform.InferType()(mod)
var_1101 = relay.var("var_1101", dtype = "int32", shape = (9, 14, 7))#candidate|1101|(9, 14, 7)|var|int32
var_1102 = relay.var("var_1102", dtype = "int32", shape = (9, 14, 7))#candidate|1102|(9, 14, 7)|var|int32
output = func_1100(var_1101,var_1102,)
func_1103 = relay.Function([var_1101,var_1102,], output)
mutated_mod['func_1103'] = func_1103
mutated_mod = relay.transform.InferType()(mutated_mod)
func_877_call = mod.get_global_var('func_877')
func_878_call = mutated_mod.get_global_var('func_878')
call_1117 = relay.TupleGetItem(func_877_call(), 2)
call_1118 = relay.TupleGetItem(func_878_call(), 2)
output = relay.Tuple([call_1117,])
output2 = relay.Tuple([call_1118,])
func_1128 = relay.Function([], output)
mod['func_1128'] = func_1128
mod = relay.transform.InferType()(mod)
mutated_mod['func_1128'] = func_1128
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1128_call = mutated_mod.get_global_var('func_1128')
call_1129 = func_1128_call()
output = call_1129
func_1130 = relay.Function([], output)
mutated_mod['func_1130'] = func_1130
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1128_call = mod.get_global_var('func_1128')
func_1130_call = mutated_mod.get_global_var('func_1130')
call_1141 = relay.TupleGetItem(func_1128_call(), 0)
call_1142 = relay.TupleGetItem(func_1130_call(), 0)
var_1143 = relay.var("var_1143", dtype = "float64", shape = (294,))#candidate|1143|(294,)|var|float64
bop_1144 = relay.mod(call_1141.astype('float32'), relay.reshape(var_1143.astype('float32'), relay.shape_of(call_1141))) # shape=(294,)
bop_1147 = relay.mod(call_1142.astype('float32'), relay.reshape(var_1143.astype('float32'), relay.shape_of(call_1142))) # shape=(294,)
bop_1161 = relay.floor_divide(call_1141.astype('float32'), relay.reshape(var_1143.astype('float32'), relay.shape_of(call_1141))) # shape=(294,)
bop_1164 = relay.floor_divide(call_1142.astype('float32'), relay.reshape(var_1143.astype('float32'), relay.shape_of(call_1142))) # shape=(294,)
bop_1165 = relay.less_equal(bop_1161.astype('bool'), relay.reshape(bop_1144.astype('bool'), relay.shape_of(bop_1161))) # shape=(294,)
bop_1168 = relay.less_equal(bop_1164.astype('bool'), relay.reshape(bop_1147.astype('bool'), relay.shape_of(bop_1164))) # shape=(294,)
func_730_call = mod.get_global_var('func_730')
func_737_call = mutated_mod.get_global_var('func_737')
const_1173 = relay.const([False,False,False,False,True,False,False,True,False,False,True,False,True,True,False,False,False,True,False,False,False,True,False,True,False,False,True,False,False,True,True,False,True,True,True,False,True,True,True,True,False,False,False,True,False,False,False,False,False,False,False,False,True,True,True,True,True,False,False,True,True,True,True,True,False,False,True,True,False,True,False,False,True,False,True,True,True,True,True,True,False,False,True,True,True,False,True,False,True,True,True,False,False,False,True,False,True,True,True,True,False,True,True,True,False,False,True,True,False,False,False,False,False,True,True,True,True,False,True,False,True,False,True,True,True,False,True,False,True,False,True,True,False,False,True,False,False,False,True,False,False,True,False,True], dtype = "bool")#candidate|1173|(144,)|const|bool
var_1174 = relay.var("var_1174", dtype = "float64", shape = (10,))#candidate|1174|(10,)|var|float64
call_1172 = relay.TupleGetItem(func_730_call(relay.reshape(const_1173.astype('bool'), [9, 4, 4]), relay.reshape(const_1173.astype('bool'), [9, 4, 4]), relay.reshape(const_1173.astype('uint64'), [9, 4, 4]), relay.reshape(const_1173.astype('uint64'), [9, 4, 4]), relay.reshape(var_1174.astype('float64'), [10,]), ), 0)
call_1175 = relay.TupleGetItem(func_737_call(relay.reshape(const_1173.astype('bool'), [9, 4, 4]), relay.reshape(const_1173.astype('bool'), [9, 4, 4]), relay.reshape(const_1173.astype('uint64'), [9, 4, 4]), relay.reshape(const_1173.astype('uint64'), [9, 4, 4]), relay.reshape(var_1174.astype('float64'), [10,]), ), 0)
const_1190 = relay.const([-3.579657,8.043584,0.914624,-3.602732,9.023864,3.646382,-5.019933,-2.646776,-8.934166,7.869698,1.168908,9.920998,-4.959332,3.749258,-7.552944,0.173587,9.879006,-7.975754,-3.628086,3.012502,6.215757,7.030466,1.999344,2.616232,5.716948,1.655117,6.878313,-0.989591,2.279302,6.717477,1.464333,-6.307860,9.767535,3.090733,-5.456855,5.131581,3.608870,2.448695,-3.152653,8.991247,-9.978252,-9.050751,7.254024,9.888194,-9.600656,-7.217494,-3.376779,-1.823260,3.774315,2.055417,-7.793325,8.369333,7.445015,7.101267,9.112120,-1.278874,6.844588,4.633160,9.733010,1.736365,-0.877126,7.488747,2.041373,-1.144476,-9.879531,-7.560289,2.780299,-0.523784,-9.347748,-8.042999,-5.160690,9.351226,-9.943862,-3.654650,-8.508746,4.856332,7.568263,0.293521,4.545302,8.168906,9.691367,-9.424878,2.240132,0.233141,-3.369378,8.916415,-2.078981,4.566760,-2.988994,0.529224,3.675487,4.370564,3.209433,0.130036,-0.685792,-6.079061,-7.359476,-8.064175,-9.364355,-7.013491,-8.518252,4.221687,-5.548171,7.936174,3.530186,5.708673,4.127755,9.308576,2.475088,5.127303,-0.313554,9.793017,9.453874,-8.617489,-4.886435,-5.040323,3.644870,7.620275,-7.151780,-1.740987,8.494287,3.422875,-2.630265,-5.955105,2.608411,6.855492,2.610997,5.423950,1.022076,-4.160794,-5.201071,-2.688438,3.424502,0.384116,-0.984277,0.914378,-2.777978,-0.418352,3.000193,-8.929682,8.055380,-5.185575,6.707361,4.094899,-6.523799,-6.395045,-1.171934,-2.352721,-6.886160,-5.834703,0.639371,2.123840,-7.052890,-1.599032,-9.696471,-1.579479,-7.106068,-1.709248,-7.753841,-9.784192,-0.043490,4.718781,3.709674,-2.411296,-2.646234,-1.068737,-1.697837,9.526451,-7.190062,-0.281743,-6.624695,-3.582387,-6.983486,-8.670585,-0.049104,9.117515,4.469127,6.467032,-8.760730,0.615604,-8.560639,-4.670356,1.829850,2.925713,9.785537,-7.691110,3.169612,-3.523201,7.749326,-0.456270,-7.652124,-9.252117,6.080922,-2.995551,8.280338,-4.727859,-8.626439,-6.636061,-9.655609,4.762000,-6.150336,3.144302,-5.134607,5.424440,6.623805,7.786658,0.103974,3.705696,7.464034,-8.829898,8.002710,7.671036,0.174824,1.345096,2.802856,6.499456,-8.234062,8.609826,-0.425210,0.195728,9.556643,6.658868,-3.166191,6.571706,5.838997,7.001785,-1.448300,-7.944182,-3.325199,8.209981,-5.877413,8.854210,9.650655,0.517520,7.974842,6.815051,9.337407,5.256728,-8.657698,2.037211,-9.997004,-3.942060,4.952517,-5.437616,0.293051,-7.419236,-8.294436,-6.678468,-7.442999,-9.350142,-8.786621,-3.884131,-7.560429,-8.363146,-0.364236,-4.597839,-9.432451,4.943436,1.408038,9.394205,7.671080,-3.604929,-3.616343,-5.118921,-8.673000,-4.876180,-6.095484,-8.539299,0.230776,2.168128,5.488729,-7.569915,6.593478,6.181431,-1.436365,1.403412,-5.696623,9.029003,-7.699070,-6.820819,-8.546026,-8.608243,0.063604,-6.227187,5.121309,5.925954,1.682974,-0.579764,1.345456,-0.380014,-6.026616,-6.822084,9.427352,4.559249], dtype = "float64")#candidate|1190|(294,)|const|float64
bop_1191 = relay.greater(var_1143.astype('bool'), relay.reshape(const_1190.astype('bool'), relay.shape_of(var_1143))) # shape=(294,)
func_962_call = mod.get_global_var('func_962')
func_969_call = mutated_mod.get_global_var('func_969')
var_1195 = relay.var("var_1195", dtype = "float64", shape = (12,))#candidate|1195|(12,)|var|float64
var_1196 = relay.var("var_1196", dtype = "uint16", shape = ())#candidate|1196|()|var|uint16
const_1197 = relay.const([9.707862,8.373237,-0.974416,-8.037753,-9.860454,9.431506,6.834782,5.611514,-8.766108,-5.871732,0.938725,-9.986556,1.185220], dtype = "float32")#candidate|1197|(13,)|const|float32
call_1194 = relay.TupleGetItem(func_962_call(relay.reshape(var_1195.astype('float64'), [4, 3]), relay.reshape(var_1195.astype('float64'), [4, 3]), relay.reshape(var_1174.astype('float64'), [10,]), relay.reshape(var_1196.astype('uint16'), []), relay.reshape(const_1197.astype('float32'), [13,]), ), 4)
call_1198 = relay.TupleGetItem(func_969_call(relay.reshape(var_1195.astype('float64'), [4, 3]), relay.reshape(var_1195.astype('float64'), [4, 3]), relay.reshape(var_1174.astype('float64'), [10,]), relay.reshape(var_1196.astype('uint16'), []), relay.reshape(const_1197.astype('float32'), [13,]), ), 4)
output = relay.Tuple([bop_1165,call_1172,const_1173,var_1174,bop_1191,call_1194,var_1195,var_1196,const_1197,])
output2 = relay.Tuple([bop_1168,call_1175,const_1173,var_1174,bop_1191,call_1198,var_1195,var_1196,const_1197,])
func_1201 = relay.Function([var_1143,var_1174,var_1195,var_1196,], output)
mod['func_1201'] = func_1201
mod = relay.transform.InferType()(mod)
mutated_mod['func_1201'] = func_1201
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1201_call = mutated_mod.get_global_var('func_1201')
var_1203 = relay.var("var_1203", dtype = "float64", shape = (294,))#candidate|1203|(294,)|var|float64
var_1204 = relay.var("var_1204", dtype = "float64", shape = (10,))#candidate|1204|(10,)|var|float64
var_1205 = relay.var("var_1205", dtype = "float64", shape = (12,))#candidate|1205|(12,)|var|float64
var_1206 = relay.var("var_1206", dtype = "uint16", shape = ())#candidate|1206|()|var|uint16
call_1202 = func_1201_call(var_1203,var_1204,var_1205,var_1206,)
output = call_1202
func_1207 = relay.Function([var_1203,var_1204,var_1205,var_1206,], output)
mutated_mod['func_1207'] = func_1207
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1209 = relay.var("var_1209", dtype = "int32", shape = ())#candidate|1209|()|var|int32
var_1210 = relay.var("var_1210", dtype = "int32", shape = (3, 3, 13))#candidate|1210|(3, 3, 13)|var|int32
bop_1211 = relay.not_equal(var_1209.astype('bool'), var_1210.astype('bool')) # shape=(3, 3, 13)
bop_1215 = relay.mod(bop_1211.astype('float64'), relay.reshape(var_1210.astype('float64'), relay.shape_of(bop_1211))) # shape=(3, 3, 13)
output = bop_1215
output2 = bop_1215
func_1218 = relay.Function([var_1209,var_1210,], output)
mod['func_1218'] = func_1218
mod = relay.transform.InferType()(mod)
var_1219 = relay.var("var_1219", dtype = "int32", shape = ())#candidate|1219|()|var|int32
var_1220 = relay.var("var_1220", dtype = "int32", shape = (3, 3, 13))#candidate|1220|(3, 3, 13)|var|int32
output = func_1218(var_1219,var_1220,)
func_1221 = relay.Function([var_1219,var_1220,], output)
mutated_mod['func_1221'] = func_1221
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1281 = relay.var("var_1281", dtype = "float32", shape = (16, 10))#candidate|1281|(16, 10)|var|float32
uop_1282 = relay.asinh(var_1281.astype('float32')) # shape=(16, 10)
func_962_call = mod.get_global_var('func_962')
func_969_call = mutated_mod.get_global_var('func_969')
const_1286 = relay.const([3.516498,-4.621916,7.013422,7.902202,-5.897655,-3.550096,-0.226620,-9.717246,-9.368965,4.688600,-4.403054,-1.620715], dtype = "float64")#candidate|1286|(12,)|const|float64
var_1287 = relay.var("var_1287", dtype = "float64", shape = (10,))#candidate|1287|(10,)|var|float64
var_1288 = relay.var("var_1288", dtype = "uint16", shape = ())#candidate|1288|()|var|uint16
var_1289 = relay.var("var_1289", dtype = "float32", shape = (13,))#candidate|1289|(13,)|var|float32
call_1285 = relay.TupleGetItem(func_962_call(relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(var_1287.astype('float64'), [10,]), relay.reshape(var_1288.astype('uint16'), []), relay.reshape(var_1289.astype('float32'), [13,]), ), 8)
call_1290 = relay.TupleGetItem(func_969_call(relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(var_1287.astype('float64'), [10,]), relay.reshape(var_1288.astype('uint16'), []), relay.reshape(var_1289.astype('float32'), [13,]), ), 8)
func_65_call = mod.get_global_var('func_65')
func_67_call = mutated_mod.get_global_var('func_67')
call_1291 = relay.TupleGetItem(func_65_call(relay.reshape(var_1287.astype('float64'), [1, 10])), 0)
call_1292 = relay.TupleGetItem(func_67_call(relay.reshape(var_1287.astype('float64'), [1, 10])), 0)
func_962_call = mod.get_global_var('func_962')
func_969_call = mutated_mod.get_global_var('func_969')
call_1293 = relay.TupleGetItem(func_962_call(relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(call_1291.astype('float64'), [10,]), relay.reshape(var_1288.astype('uint16'), []), relay.reshape(call_1285.astype('float32'), [13,]), ), 0)
call_1294 = relay.TupleGetItem(func_969_call(relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(const_1286.astype('float64'), [4, 3]), relay.reshape(call_1291.astype('float64'), [10,]), relay.reshape(var_1288.astype('uint16'), []), relay.reshape(call_1285.astype('float32'), [13,]), ), 0)
func_877_call = mod.get_global_var('func_877')
func_878_call = mutated_mod.get_global_var('func_878')
call_1297 = relay.TupleGetItem(func_877_call(), 2)
call_1298 = relay.TupleGetItem(func_878_call(), 2)
output = relay.Tuple([uop_1282,call_1285,const_1286,var_1287,var_1288,var_1289,call_1291,call_1293,call_1297,])
output2 = relay.Tuple([uop_1282,call_1290,const_1286,var_1287,var_1288,var_1289,call_1292,call_1294,call_1298,])
func_1305 = relay.Function([var_1281,var_1287,var_1288,var_1289,], output)
mod['func_1305'] = func_1305
mod = relay.transform.InferType()(mod)
var_1306 = relay.var("var_1306", dtype = "float32", shape = (16, 10))#candidate|1306|(16, 10)|var|float32
var_1307 = relay.var("var_1307", dtype = "float64", shape = (10,))#candidate|1307|(10,)|var|float64
var_1308 = relay.var("var_1308", dtype = "uint16", shape = ())#candidate|1308|()|var|uint16
var_1309 = relay.var("var_1309", dtype = "float32", shape = (13,))#candidate|1309|(13,)|var|float32
output = func_1305(var_1306,var_1307,var_1308,var_1309,)
func_1310 = relay.Function([var_1306,var_1307,var_1308,var_1309,], output)
mutated_mod['func_1310'] = func_1310
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1128_call = mod.get_global_var('func_1128')
func_1130_call = mutated_mod.get_global_var('func_1130')
call_1365 = relay.TupleGetItem(func_1128_call(), 0)
call_1366 = relay.TupleGetItem(func_1130_call(), 0)
uop_1372 = relay.asinh(call_1365.astype('float64')) # shape=(294,)
uop_1374 = relay.asinh(call_1366.astype('float64')) # shape=(294,)
func_730_call = mod.get_global_var('func_730')
func_737_call = mutated_mod.get_global_var('func_737')
const_1380 = relay.const([[False,True,False,False,False,False,False,False,False,False,False,True,True,False,False,True,False,True,False,True,True,True,False,True,False,False,True,True,False,False,False,False,False,False,False,False,False,True,False,True,True,False,False,False,False,False,False,True,False,False,False,True,True,False,False,False,False,True,True,True,False,True,True,False,False,False,False,False,True,False,True,False],[False,False,True,True,False,True,False,False,True,False,False,False,False,False,False,False,False,True,True,False,True,True,False,True,True,False,True,False,False,True,False,False,True,True,True,True,False,True,False,True,True,False,False,True,False,False,False,False,False,True,False,True,False,False,False,True,False,True,True,False,False,False,True,True,True,False,True,False,False,True,True,True]], dtype = "bool")#candidate|1380|(2, 72)|const|bool
var_1381 = relay.var("var_1381", dtype = "float64", shape = (10,))#candidate|1381|(10,)|var|float64
call_1379 = relay.TupleGetItem(func_730_call(relay.reshape(const_1380.astype('bool'), [9, 4, 4]), relay.reshape(const_1380.astype('bool'), [9, 4, 4]), relay.reshape(const_1380.astype('uint64'), [9, 4, 4]), relay.reshape(const_1380.astype('uint64'), [9, 4, 4]), relay.reshape(var_1381.astype('float64'), [10,]), ), 2)
call_1382 = relay.TupleGetItem(func_737_call(relay.reshape(const_1380.astype('bool'), [9, 4, 4]), relay.reshape(const_1380.astype('bool'), [9, 4, 4]), relay.reshape(const_1380.astype('uint64'), [9, 4, 4]), relay.reshape(const_1380.astype('uint64'), [9, 4, 4]), relay.reshape(var_1381.astype('float64'), [10,]), ), 2)
func_456_call = mod.get_global_var('func_456')
func_458_call = mutated_mod.get_global_var('func_458')
var_1388 = relay.var("var_1388", dtype = "float32", shape = (1, 1))#candidate|1388|(1, 1)|var|float32
call_1387 = relay.TupleGetItem(func_456_call(relay.reshape(var_1388.astype('float32'), [1,])), 0)
call_1389 = relay.TupleGetItem(func_458_call(relay.reshape(var_1388.astype('float32'), [1,])), 0)
output = relay.Tuple([uop_1372,call_1379,const_1380,var_1381,call_1387,var_1388,])
output2 = relay.Tuple([uop_1374,call_1382,const_1380,var_1381,call_1389,var_1388,])
func_1391 = relay.Function([var_1381,var_1388,], output)
mod['func_1391'] = func_1391
mod = relay.transform.InferType()(mod)
mutated_mod['func_1391'] = func_1391
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1391_call = mutated_mod.get_global_var('func_1391')
var_1393 = relay.var("var_1393", dtype = "float64", shape = (10,))#candidate|1393|(10,)|var|float64
var_1394 = relay.var("var_1394", dtype = "float32", shape = (1, 1))#candidate|1394|(1, 1)|var|float32
call_1392 = func_1391_call(var_1393,var_1394,)
output = call_1392
func_1395 = relay.Function([var_1393,var_1394,], output)
mutated_mod['func_1395'] = func_1395
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1128_call = mod.get_global_var('func_1128')
func_1130_call = mutated_mod.get_global_var('func_1130')
call_1403 = relay.TupleGetItem(func_1128_call(), 0)
call_1404 = relay.TupleGetItem(func_1130_call(), 0)
func_730_call = mod.get_global_var('func_730')
func_737_call = mutated_mod.get_global_var('func_737')
var_1414 = relay.var("var_1414", dtype = "bool", shape = (144,))#candidate|1414|(144,)|var|bool
var_1415 = relay.var("var_1415", dtype = "float64", shape = (10,))#candidate|1415|(10,)|var|float64
call_1413 = relay.TupleGetItem(func_730_call(relay.reshape(var_1414.astype('bool'), [9, 4, 4]), relay.reshape(var_1414.astype('bool'), [9, 4, 4]), relay.reshape(var_1414.astype('uint64'), [9, 4, 4]), relay.reshape(var_1414.astype('uint64'), [9, 4, 4]), relay.reshape(var_1415.astype('float64'), [10,]), ), 5)
call_1416 = relay.TupleGetItem(func_737_call(relay.reshape(var_1414.astype('bool'), [9, 4, 4]), relay.reshape(var_1414.astype('bool'), [9, 4, 4]), relay.reshape(var_1414.astype('uint64'), [9, 4, 4]), relay.reshape(var_1414.astype('uint64'), [9, 4, 4]), relay.reshape(var_1415.astype('float64'), [10,]), ), 5)
func_985_call = mod.get_global_var('func_985')
func_988_call = mutated_mod.get_global_var('func_988')
var_1420 = relay.var("var_1420", dtype = "float32", shape = (576,))#candidate|1420|(576,)|var|float32
call_1419 = func_985_call(relay.reshape(var_1420.astype('float32'), [16, 12, 3]))
call_1421 = func_985_call(relay.reshape(var_1420.astype('float32'), [16, 12, 3]))
func_1072_call = mod.get_global_var('func_1072')
func_1077_call = mutated_mod.get_global_var('func_1077')
const_1423 = relay.const([2.878728,-0.683277,-4.546164,7.725074,1.994865,-0.779312,-9.112883,7.680402,-0.408413,6.925349,-4.846305,-4.613018,5.375076,8.718681,-1.409063,1.000020,7.216364,1.197376,-4.019624,-8.545149,-2.423015,-6.687161,-6.425062,-2.156277,1.536720,-6.813602,7.913044,2.023562,-9.566296,6.499860,2.198917,0.886036,-3.712323,-9.390052,0.962621,-3.062372,3.681012,5.417898,-4.637329,-8.837669,9.166508,2.794070,6.731645,9.635837,-6.391693,4.749193,4.744877,-3.962351,0.223130,-8.820222,6.965215,-4.777724,-0.870051,-5.016455,3.174051,-5.391663,8.871193,9.067127,-1.351098,1.537798,7.871938,8.083172,6.629271,-4.288990,-3.809889,0.597666,7.931517,4.258931,9.217027,-9.818582,-4.965194,0.205148,0.559964,5.682997,8.181572,-2.936971,5.316792,-0.591795,5.863793,2.515499,6.305780,1.561984,2.636708,-1.386004,-3.626674,0.392376,-5.041179,8.258045,-6.034581,3.312340,9.132018,-1.694015,6.373178,6.483410,-5.190195,9.888819,3.592604,-0.641435,5.926481,-8.199294,-9.340022,-0.241821,-2.954118,-9.332622,-8.977084,8.661255,8.426417,-0.505036], dtype = "float64")#candidate|1423|(108,)|const|float64
var_1424 = relay.var("var_1424", dtype = "float32", shape = (40,))#candidate|1424|(40,)|var|float32
const_1425 = relay.const([2.384524], dtype = "float32")#candidate|1425|(1,)|const|float32
call_1422 = relay.TupleGetItem(func_1072_call(relay.reshape(const_1423.astype('float64'), [12, 9]), relay.reshape(var_1424.astype('float32'), [40,]), relay.reshape(const_1425.astype('float32'), [1,]), relay.reshape(call_1413.astype('float64'), [1, 1215]), ), 1)
call_1426 = relay.TupleGetItem(func_1077_call(relay.reshape(const_1423.astype('float64'), [12, 9]), relay.reshape(var_1424.astype('float32'), [40,]), relay.reshape(const_1425.astype('float32'), [1,]), relay.reshape(call_1413.astype('float64'), [1, 1215]), ), 1)
uop_1429 = relay.sin(var_1414.astype('float32')) # shape=(144,)
output = relay.Tuple([call_1403,call_1413,var_1415,call_1419,var_1420,call_1422,const_1423,var_1424,const_1425,uop_1429,])
output2 = relay.Tuple([call_1404,call_1416,var_1415,call_1421,var_1420,call_1426,const_1423,var_1424,const_1425,uop_1429,])
func_1431 = relay.Function([var_1414,var_1415,var_1420,var_1424,], output)
mod['func_1431'] = func_1431
mod = relay.transform.InferType()(mod)
mutated_mod['func_1431'] = func_1431
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1431_call = mutated_mod.get_global_var('func_1431')
var_1433 = relay.var("var_1433", dtype = "bool", shape = (144,))#candidate|1433|(144,)|var|bool
var_1434 = relay.var("var_1434", dtype = "float64", shape = (10,))#candidate|1434|(10,)|var|float64
var_1435 = relay.var("var_1435", dtype = "float32", shape = (576,))#candidate|1435|(576,)|var|float32
var_1436 = relay.var("var_1436", dtype = "float32", shape = (40,))#candidate|1436|(40,)|var|float32
call_1432 = func_1431_call(var_1433,var_1434,var_1435,var_1436,)
output = call_1432
func_1437 = relay.Function([var_1433,var_1434,var_1435,var_1436,], output)
mutated_mod['func_1437'] = func_1437
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1444 = relay.const([[[False,False,False,False,True,True,False,True,True,False,True]],[[True,False,False,True,False,False,True,True,True,False,False]],[[True,False,True,True,False,True,True,False,True,True,False]]], dtype = "bool")#candidate|1444|(3, 1, 11)|const|bool
var_1445 = relay.var("var_1445", dtype = "bool", shape = (3, 1, 11))#candidate|1445|(3, 1, 11)|var|bool
bop_1446 = relay.logical_or(const_1444.astype('bool'), relay.reshape(var_1445.astype('bool'), relay.shape_of(const_1444))) # shape=(3, 1, 11)
uop_1450 = relay.acos(bop_1446.astype('float64')) # shape=(3, 1, 11)
output = relay.Tuple([uop_1450,])
output2 = relay.Tuple([uop_1450,])
func_1458 = relay.Function([var_1445,], output)
mod['func_1458'] = func_1458
mod = relay.transform.InferType()(mod)
var_1459 = relay.var("var_1459", dtype = "bool", shape = (3, 1, 11))#candidate|1459|(3, 1, 11)|var|bool
output = func_1458(var_1459)
func_1460 = relay.Function([var_1459], output)
mutated_mod['func_1460'] = func_1460
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1468 = relay.var("var_1468", dtype = "bool", shape = (6, 10, 1))#candidate|1468|(6, 10, 1)|var|bool
var_1469 = relay.var("var_1469", dtype = "bool", shape = (6, 10, 16))#candidate|1469|(6, 10, 16)|var|bool
bop_1470 = relay.logical_or(var_1468.astype('bool'), var_1469.astype('bool')) # shape=(6, 10, 16)
func_985_call = mod.get_global_var('func_985')
func_988_call = mutated_mod.get_global_var('func_988')
var_1474 = relay.var("var_1474", dtype = "float32", shape = (576,))#candidate|1474|(576,)|var|float32
call_1473 = func_985_call(relay.reshape(var_1474.astype('float32'), [16, 12, 3]))
call_1475 = func_985_call(relay.reshape(var_1474.astype('float32'), [16, 12, 3]))
uop_1476 = relay.asin(call_1473.astype('float64')) # shape=(16, 12, 3)
uop_1478 = relay.asin(call_1475.astype('float64')) # shape=(16, 12, 3)
uop_1480 = relay.acos(var_1474.astype('float32')) # shape=(576,)
output = relay.Tuple([bop_1470,uop_1476,uop_1480,])
output2 = relay.Tuple([bop_1470,uop_1478,uop_1480,])
func_1483 = relay.Function([var_1468,var_1469,var_1474,], output)
mod['func_1483'] = func_1483
mod = relay.transform.InferType()(mod)
var_1484 = relay.var("var_1484", dtype = "bool", shape = (6, 10, 1))#candidate|1484|(6, 10, 1)|var|bool
var_1485 = relay.var("var_1485", dtype = "bool", shape = (6, 10, 16))#candidate|1485|(6, 10, 16)|var|bool
var_1486 = relay.var("var_1486", dtype = "float32", shape = (576,))#candidate|1486|(576,)|var|float32
output = func_1483(var_1484,var_1485,var_1486,)
func_1487 = relay.Function([var_1484,var_1485,var_1486,], output)
mutated_mod['func_1487'] = func_1487
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1128_call = mod.get_global_var('func_1128')
func_1130_call = mutated_mod.get_global_var('func_1130')
call_1527 = relay.TupleGetItem(func_1128_call(), 0)
call_1528 = relay.TupleGetItem(func_1130_call(), 0)
func_1305_call = mod.get_global_var('func_1305')
func_1310_call = mutated_mod.get_global_var('func_1310')
var_1530 = relay.var("var_1530", dtype = "float32", shape = (160,))#candidate|1530|(160,)|var|float32
var_1531 = relay.var("var_1531", dtype = "float64", shape = (10,))#candidate|1531|(10,)|var|float64
var_1532 = relay.var("var_1532", dtype = "uint16", shape = ())#candidate|1532|()|var|uint16
const_1533 = relay.const([-1.047923,-5.670475,5.361317,-5.197690,9.896387,0.327105,0.683628,-4.201837,8.113034,4.724155,3.452358,-7.115226,0.682228], dtype = "float32")#candidate|1533|(13,)|const|float32
call_1529 = relay.TupleGetItem(func_1305_call(relay.reshape(var_1530.astype('float32'), [16, 10]), relay.reshape(var_1531.astype('float64'), [10,]), relay.reshape(var_1532.astype('uint16'), []), relay.reshape(const_1533.astype('float32'), [13,]), ), 1)
call_1534 = relay.TupleGetItem(func_1310_call(relay.reshape(var_1530.astype('float32'), [16, 10]), relay.reshape(var_1531.astype('float64'), [10,]), relay.reshape(var_1532.astype('uint16'), []), relay.reshape(const_1533.astype('float32'), [13,]), ), 1)
func_103_call = mod.get_global_var('func_103')
func_106_call = mutated_mod.get_global_var('func_106')
call_1551 = relay.TupleGetItem(func_103_call(relay.reshape(call_1527.astype('float64'), [7, 6, 7])), 0)
call_1552 = relay.TupleGetItem(func_106_call(relay.reshape(call_1527.astype('float64'), [7, 6, 7])), 0)
output = relay.Tuple([call_1527,call_1529,var_1530,var_1531,var_1532,const_1533,call_1551,])
output2 = relay.Tuple([call_1528,call_1534,var_1530,var_1531,var_1532,const_1533,call_1552,])
func_1553 = relay.Function([var_1530,var_1531,var_1532,], output)
mod['func_1553'] = func_1553
mod = relay.transform.InferType()(mod)
mutated_mod['func_1553'] = func_1553
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1553_call = mutated_mod.get_global_var('func_1553')
var_1555 = relay.var("var_1555", dtype = "float32", shape = (160,))#candidate|1555|(160,)|var|float32
var_1556 = relay.var("var_1556", dtype = "float64", shape = (10,))#candidate|1556|(10,)|var|float64
var_1557 = relay.var("var_1557", dtype = "uint16", shape = ())#candidate|1557|()|var|uint16
call_1554 = func_1553_call(var_1555,var_1556,var_1557,)
output = call_1554
func_1558 = relay.Function([var_1555,var_1556,var_1557,], output)
mutated_mod['func_1558'] = func_1558
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1571 = relay.const([[3.789642,-9.609513,3.320961,2.373517,-2.925693],[-7.692379,2.091219,-2.432208,-2.069442,3.626546],[-2.071298,-5.254214,1.145757,-7.402314,5.037088],[0.089356,-6.304996,-7.961370,-4.792225,-8.237810],[-0.535743,-3.773691,5.832404,1.007503,5.621404],[-2.093760,7.408915,2.184380,-6.156622,7.280573],[1.517035,5.187034,-5.652473,5.775579,4.150890]], dtype = "float32")#candidate|1571|(7, 5)|const|float32
uop_1572 = relay.acosh(const_1571.astype('float32')) # shape=(7, 5)
uop_1577 = relay.acos(uop_1572.astype('float64')) # shape=(7, 5)
bop_1582 = relay.floor_divide(uop_1577.astype('float32'), relay.reshape(uop_1572.astype('float32'), relay.shape_of(uop_1577))) # shape=(7, 5)
const_1586 = relay.const([[-6.778046,-4.554609,-4.086355,3.400962,-4.388503],[-0.163704,2.485349,-4.797171,7.559396,4.120195],[2.506130,3.995383,-7.337355,2.149731,-2.779754],[3.625403,8.908318,-8.702238,2.236838,-0.017454],[0.912448,1.635735,-6.119633,-8.828912,3.523875],[-1.190128,-4.218995,-9.128359,-2.784648,5.004038],[9.079224,9.945821,-7.970427,-4.494312,8.658614]], dtype = "float64")#candidate|1586|(7, 5)|const|float64
bop_1587 = relay.floor_mod(uop_1577.astype('float64'), relay.reshape(const_1586.astype('float64'), relay.shape_of(uop_1577))) # shape=(7, 5)
output = relay.Tuple([bop_1582,bop_1587,])
output2 = relay.Tuple([bop_1582,bop_1587,])
F = relay.Function([], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([], output2)
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
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()()
res3 = intrp3.evaluate()()
res4 = intrp4.evaluate()()
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
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()()
res7 = intrp7.evaluate()()
res8 = intrp8.evaluate()()
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
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()()
res11 = intrp11.evaluate()()
res12 = intrp12.evaluate()()
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
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()()
res15 = intrp15.evaluate()()
res16 = intrp16.evaluate()()
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
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()()
res19 = intrp19.evaluate()()
res20 = intrp20.evaluate()()
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
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()()
res23 = intrp23.evaluate()()
res24 = intrp24.evaluate()()
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

'''9: TVMFuncCall
8: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
7: tvm::transform::Pass::operator()(tvm::IRModule) const
6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
5: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::LazyGradientInit(tvm::RelayExpr const&, tvm::IRModule)
0: tvm::relay::CheckFeature(tvm::RelayExpr const&, tvm::relay::FeatureSet const&)

'''