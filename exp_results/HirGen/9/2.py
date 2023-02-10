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
var_0 = relay.var("var_0", dtype = "float64", shape = (1, 10))#candidate|0|(1, 10)|var|float64
uop_1 = relay.exp(var_0.astype('float64')) # shape=(1, 10)
bop_3 = relay.maximum(var_0.astype('uint16'), relay.reshape(uop_1.astype('uint16'), relay.shape_of(var_0))) # shape=(1, 10)
bop_6 = relay.multiply(bop_3.astype('int8'), relay.reshape(uop_1.astype('int8'), relay.shape_of(bop_3))) # shape=(1, 10)
var_9 = relay.var("var_9", dtype = "float64", shape = (15, 10))#candidate|9|(15, 10)|var|float64
bop_10 = relay.minimum(var_0.astype('int8'), var_9.astype('int8')) # shape=(15, 10)
bop_15 = relay.right_shift(uop_1.astype('uint8'), bop_10.astype('uint8')) # shape=(15, 10)
uop_18 = relay.sin(bop_15.astype('float64')) # shape=(15, 10)
uop_20 = relay.rsqrt(uop_1.astype('float32')) # shape=(1, 10)
bop_22 = relay.bitwise_and(uop_18.astype('uint8'), relay.reshape(bop_10.astype('uint8'), relay.shape_of(uop_18))) # shape=(15, 10)
uop_25 = relay.tan(uop_18.astype('float32')) # shape=(15, 10)
uop_27 = relay.cos(uop_25.astype('float64')) # shape=(15, 10)
bop_29 = relay.bitwise_or(uop_25.astype('uint8'), bop_6.astype('uint8')) # shape=(15, 10)
uop_32 = relay.cos(bop_29.astype('float32')) # shape=(15, 10)
bop_34 = relay.power(uop_25.astype('float64'), relay.reshape(uop_27.astype('float64'), relay.shape_of(uop_25))) # shape=(15, 10)
const_40 = relay.const([[-0.309950,4.908749,2.971534,-8.404302,-3.574328,2.690153,7.624489,8.925378,-1.982063,3.066968],[-2.455364,0.766252,8.565874,4.785660,-4.891581,-1.296679,-0.823202,8.992147,5.706434,-3.937079],[1.842399,-7.470251,8.693507,1.216047,3.480644,4.124841,-5.149110,-9.192129,0.160413,-5.313123],[-9.797752,-8.057815,3.300486,-0.068978,-5.990305,-1.381195,-0.774640,-6.393794,-4.604860,-4.852992],[-4.083201,2.040223,-5.866759,-9.272066,-7.582038,2.978359,8.316890,7.372701,4.259771,-4.115400],[8.721756,1.010387,7.214782,7.316182,1.393246,0.633023,0.586412,4.705294,-9.216158,-2.626614],[-2.825898,6.539513,0.573753,-6.710958,-3.782901,-3.485570,8.568652,1.931120,-7.702764,3.559728],[-9.993877,9.740593,2.796048,2.920821,4.142635,-3.150557,4.635079,1.775954,5.971471,1.613527],[-6.655679,5.709572,-2.110877,-6.491858,1.451843,-3.072416,-5.202559,-4.809642,-9.870685,4.939664],[5.013471,-4.738533,8.357461,1.664306,8.789176,-0.727048,7.901020,-5.859489,0.290325,-1.716611],[-8.066683,-3.723228,-2.961255,-1.874838,3.671394,-9.004239,-1.332354,0.035893,9.910993,-1.958813],[-2.432888,-3.617627,-7.306220,8.844679,-5.665436,-9.854215,-1.683797,6.467687,9.428410,-7.804798],[-2.475749,8.937859,4.325379,-5.458264,-9.265588,3.877499,-8.200644,-5.563524,-6.661969,-3.069834],[1.592324,4.239646,4.952568,-7.810270,4.689547,4.627921,-5.824231,-1.693641,-3.533101,-1.037652],[8.417232,5.204944,9.264193,-7.324772,0.216184,-8.711446,3.810679,-0.087088,1.229212,-0.969680]], dtype = "float32")#candidate|40|(15, 10)|const|float32
bop_41 = relay.divide(uop_32.astype('float32'), relay.reshape(const_40.astype('float32'), relay.shape_of(uop_32))) # shape=(15, 10)
bop_47 = relay.floor_mod(uop_32.astype('float32'), uop_20.astype('float32')) # shape=(15, 10)
output = relay.Tuple([bop_22,bop_34,bop_41,bop_47,])
output2 = relay.Tuple([bop_22,bop_34,bop_41,bop_47,])
func_51 = relay.Function([var_0,var_9,], output)
mod['func_51'] = func_51
mod = relay.transform.InferType()(mod)
mutated_mod['func_51'] = func_51
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mutated_mod.get_global_var('func_51')
var_53 = relay.var("var_53", dtype = "float64", shape = (1, 10))#candidate|53|(1, 10)|var|float64
var_54 = relay.var("var_54", dtype = "float64", shape = (15, 10))#candidate|54|(15, 10)|var|float64
call_52 = func_51_call(var_53,var_54,)
output = call_52
func_55 = relay.Function([var_53,var_54,], output)
mutated_mod['func_55'] = func_55
mutated_mod = relay.transform.InferType()(mutated_mod)
var_59 = relay.var("var_59", dtype = "float32", shape = (14, 4))#candidate|59|(14, 4)|var|float32
var_60 = relay.var("var_60", dtype = "float32", shape = (14, 4))#candidate|60|(14, 4)|var|float32
bop_61 = relay.minimum(var_59.astype('float32'), relay.reshape(var_60.astype('float32'), relay.shape_of(var_59))) # shape=(14, 4)
uop_64 = relay.atanh(bop_61.astype('float32')) # shape=(14, 4)
bop_70 = relay.greater(var_60.astype('bool'), relay.reshape(uop_64.astype('bool'), relay.shape_of(var_60))) # shape=(14, 4)
bop_73 = relay.logical_or(bop_70.astype('bool'), relay.reshape(uop_64.astype('bool'), relay.shape_of(bop_70))) # shape=(14, 4)
var_76 = relay.var("var_76", dtype = "float32", shape = (14, 4))#candidate|76|(14, 4)|var|float32
bop_77 = relay.logical_and(uop_64.astype('bool'), relay.reshape(var_76.astype('bool'), relay.shape_of(uop_64))) # shape=(14, 4)
output = relay.Tuple([bop_73,bop_77,])
output2 = relay.Tuple([bop_73,bop_77,])
func_80 = relay.Function([var_59,var_60,var_76,], output)
mod['func_80'] = func_80
mod = relay.transform.InferType()(mod)
var_81 = relay.var("var_81", dtype = "float32", shape = (14, 4))#candidate|81|(14, 4)|var|float32
var_82 = relay.var("var_82", dtype = "float32", shape = (14, 4))#candidate|82|(14, 4)|var|float32
var_83 = relay.var("var_83", dtype = "float32", shape = (14, 4))#candidate|83|(14, 4)|var|float32
output = func_80(var_81,var_82,var_83,)
func_84 = relay.Function([var_81,var_82,var_83,], output)
mutated_mod['func_84'] = func_84
mutated_mod = relay.transform.InferType()(mutated_mod)
var_179 = relay.var("var_179", dtype = "int32", shape = (6, 14, 4))#candidate|179|(6, 14, 4)|var|int32
var_180 = relay.var("var_180", dtype = "int32", shape = (6, 14, 4))#candidate|180|(6, 14, 4)|var|int32
bop_181 = relay.equal(var_179.astype('bool'), relay.reshape(var_180.astype('bool'), relay.shape_of(var_179))) # shape=(6, 14, 4)
uop_184 = relay.sin(bop_181.astype('float32')) # shape=(6, 14, 4)
bop_188 = relay.add(uop_184.astype('uint64'), relay.reshape(var_180.astype('uint64'), relay.shape_of(uop_184))) # shape=(6, 14, 4)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
const_192 = relay.const([8.608456,6.461381,-0.779081,9.557781,-9.711088,2.672937,4.896328,-3.855709,3.329075,3.871299,-3.189055,4.448218,-2.395716,2.823480,-0.647635,9.780754,2.847593,3.393394,-9.126367,6.125413,-3.078451,-9.670933,-0.666232,-9.375393,-5.049402,-2.216968,3.550396,2.761046,3.756106,5.567012,-6.526565,2.602529,4.652190,6.509586,8.590824,-6.006811,-6.924402,-7.771824,8.868609,-6.524956,-0.427133,-3.100334,-0.356892,0.086528,0.807906,0.832737,7.565620,5.336340,1.542880,-2.915725,3.455933,-7.291181,9.325644,3.826750,9.889526,-8.753334], dtype = "float32")#candidate|192|(56,)|const|float32
call_191 = relay.TupleGetItem(func_80_call(relay.reshape(const_192.astype('float32'), [14, 4]), relay.reshape(const_192.astype('float32'), [14, 4]), relay.reshape(const_192.astype('float32'), [14, 4]), ), 1)
call_193 = relay.TupleGetItem(func_84_call(relay.reshape(const_192.astype('float32'), [14, 4]), relay.reshape(const_192.astype('float32'), [14, 4]), relay.reshape(const_192.astype('float32'), [14, 4]), ), 1)
var_194 = relay.var("var_194", dtype = "uint64", shape = (6, 14, 4))#candidate|194|(6, 14, 4)|var|uint64
bop_195 = relay.greater_equal(bop_188.astype('bool'), relay.reshape(var_194.astype('bool'), relay.shape_of(bop_188))) # shape=(6, 14, 4)
bop_199 = relay.logical_xor(bop_188.astype('int64'), relay.reshape(var_180.astype('int64'), relay.shape_of(bop_188))) # shape=(6, 14, 4)
uop_202 = relay.acosh(bop_181.astype('float64')) # shape=(6, 14, 4)
uop_204 = relay.erf(bop_181.astype('float64')) # shape=(6, 14, 4)
bop_211 = relay.not_equal(var_194.astype('bool'), relay.reshape(bop_199.astype('bool'), relay.shape_of(var_194))) # shape=(6, 14, 4)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
call_214 = relay.TupleGetItem(func_80_call(relay.reshape(call_191.astype('float32'), [14, 4]), relay.reshape(call_191.astype('float32'), [14, 4]), relay.reshape(call_191.astype('float32'), [14, 4]), ), 1)
call_215 = relay.TupleGetItem(func_84_call(relay.reshape(call_191.astype('float32'), [14, 4]), relay.reshape(call_191.astype('float32'), [14, 4]), relay.reshape(call_191.astype('float32'), [14, 4]), ), 1)
const_216 = relay.const([[[2.469742,5.035114,3.161668,-5.318428],[0.988156,-7.328439,7.804141,2.119365],[-6.635227,-2.797632,-6.378898,0.988923],[-8.815171,-9.211178,6.241884,-5.174815],[2.207470,2.530120,-7.256753,6.383249],[-9.029168,0.444218,-8.917288,4.390711],[-1.532970,8.342542,9.527483,-6.369982],[9.470573,-2.116834,-0.275825,4.207840],[-3.707895,5.806137,9.816532,-5.223200],[-1.547279,7.655166,3.423930,9.309608],[-2.410810,-4.117354,1.513841,7.724727],[2.198198,7.578785,-3.141291,-2.258722],[1.029673,9.775539,-6.486040,1.937620],[3.946925,-4.449624,6.106771,-1.540069]],[[-1.551469,7.507945,-0.658536,-7.601099],[6.972913,4.737166,6.117285,-9.252645],[5.245787,5.065935,-2.026763,-2.470803],[4.768298,-8.600360,6.588335,0.815943],[9.013188,6.416094,7.666093,-7.459410],[-9.093633,1.494613,5.306074,-0.098487],[4.432645,-6.221746,-7.087831,-0.921127],[0.888527,8.554249,2.789547,-4.189028],[0.607497,-3.812729,-7.868904,0.191409],[5.730069,-8.277683,6.628929,1.929510],[-0.289484,-0.726201,2.634785,-6.745813],[4.905530,6.544266,-8.781917,1.223310],[3.717261,0.578541,-8.679185,-9.993387],[7.784076,-2.596479,-5.179976,3.175577]],[[-6.192133,4.197733,-9.966150,-1.074128],[9.190489,1.273428,-3.230983,-2.580727],[-8.960600,1.115169,-0.647417,-8.364378],[-6.997895,-8.834841,0.842415,2.955753],[2.576770,5.728233,-5.331330,0.503089],[4.374289,-8.682695,-5.061188,7.051239],[-1.198809,0.633027,2.791932,-7.033242],[0.551649,-4.639297,-5.225096,-7.839020],[-9.562660,-2.570869,8.820742,9.396823],[-1.055984,-5.509983,-3.848251,5.437812],[-8.972760,9.928855,1.988916,9.916664],[4.066789,3.688026,7.277031,6.526577],[7.879410,1.858036,-4.176523,2.339165],[-7.983659,5.660789,1.099926,1.058783]],[[3.415988,8.945852,-7.769480,9.975942],[-1.351558,4.831440,-3.328989,-5.075340],[-8.772510,7.162633,-2.554739,9.430827],[-2.801851,9.197920,-6.018642,-8.931308],[1.672779,-5.177034,-5.140306,6.676145],[-4.032254,3.025864,-3.290137,-8.213282],[6.164055,7.200918,-0.959744,-3.501186],[7.545336,-5.740107,-5.623902,8.265532],[1.342681,3.046824,-0.017696,-1.091912],[4.868618,8.217748,6.936548,0.046174],[8.859911,-7.183775,9.576370,2.038401],[-4.262856,-1.027213,-8.965853,6.248686],[9.514745,8.130156,5.947621,-8.967061],[3.325397,-4.313518,7.905807,7.633155]],[[-7.910802,-2.067172,-9.726719,-7.343113],[-0.106248,1.046040,6.402818,-4.871500],[0.425079,-9.977782,8.956646,-6.508974],[-9.642911,-3.968282,-0.620134,3.165892],[4.635189,-4.971932,5.694661,8.742494],[-0.763503,-3.070679,1.871236,-2.984844],[-6.173241,1.961482,-1.005043,3.668777],[8.925905,-2.228984,7.846930,-0.080334],[-1.270571,7.828955,-1.114152,3.478793],[3.919452,-5.316932,-4.743205,-3.475183],[2.383682,-0.601066,8.965016,-6.709182],[9.516734,-8.435579,-8.535609,-1.133961],[4.045162,-0.950253,-2.507982,-6.674879],[9.354413,-5.533480,-9.629902,5.085117]],[[-5.917973,9.722948,3.655676,-1.568913],[-9.310382,4.330516,-4.835382,1.861518],[3.435444,7.530307,9.615919,-3.683880],[-3.474344,-0.557016,4.355725,-1.628664],[-0.257851,3.881581,7.505471,-1.789273],[-4.480030,3.000160,6.605769,-5.651452],[-3.229023,-6.327852,-1.895963,9.203027],[0.581223,6.920829,8.556876,3.750302],[8.999960,7.498935,-4.914777,-0.174571],[9.495044,4.585971,1.081596,-9.227823],[2.469391,2.101725,6.991601,-3.536672],[0.911062,9.960248,0.739209,-5.668295],[2.613724,-5.271212,9.951899,-7.347971],[-1.174708,-4.665827,2.689138,5.660563]]], dtype = "float64")#candidate|216|(6, 14, 4)|const|float64
bop_217 = relay.less(uop_202.astype('bool'), relay.reshape(const_216.astype('bool'), relay.shape_of(uop_202))) # shape=(6, 14, 4)
bop_220 = relay.floor_divide(bop_199.astype('float64'), relay.reshape(bop_217.astype('float64'), relay.shape_of(bop_199))) # shape=(6, 14, 4)
uop_225 = relay.sigmoid(bop_211.astype('float64')) # shape=(6, 14, 4)
uop_227 = relay.log10(uop_225.astype('float32')) # shape=(6, 14, 4)
bop_229 = relay.right_shift(bop_181.astype('uint16'), relay.reshape(bop_217.astype('uint16'), relay.shape_of(bop_181))) # shape=(6, 14, 4)
bop_232 = relay.power(uop_227.astype('float32'), relay.reshape(uop_202.astype('float32'), relay.shape_of(uop_227))) # shape=(6, 14, 4)
bop_236 = relay.less(bop_220.astype('bool'), relay.reshape(const_216.astype('bool'), relay.shape_of(bop_220))) # shape=(6, 14, 4)
uop_240 = relay.asinh(uop_227.astype('float64')) # shape=(6, 14, 4)
bop_243 = relay.less_equal(uop_225.astype('bool'), relay.reshape(const_216.astype('bool'), relay.shape_of(uop_225))) # shape=(6, 14, 4)
const_247 = relay.const([[[-0.627209,-4.377044,7.838328,0.269893],[-1.508158,3.150157,2.588864,0.777039],[9.936722,-9.056491,0.920536,-2.421514],[2.275714,0.091069,8.117943,-2.503859],[-9.504810,-6.160739,-8.278166,2.872827],[-7.296739,-9.406082,8.004987,-5.360279],[2.970507,-2.442752,-6.290332,4.897544],[-9.383298,8.226630,4.427725,1.646701],[5.491478,-1.656885,4.061475,-2.064324],[-4.446779,-2.925220,-6.858334,0.968443],[8.437240,-1.079729,-9.237018,3.806750],[-6.137860,-9.831297,-1.543358,0.874949],[-6.319767,5.197969,-2.227634,0.606133],[-4.763336,-4.782409,-2.716517,1.649620]],[[3.120014,1.505968,7.029238,-4.478771],[-8.484346,-0.067275,-2.827592,-2.671904],[5.510752,1.350074,4.777529,-7.998070],[-5.750164,-7.392194,-0.285348,-5.370897],[5.848118,3.839878,6.700414,-3.454520],[7.746769,-7.472148,4.672212,-1.876084],[2.321969,-5.380759,-2.494250,-3.918944],[5.719113,0.571105,-3.855853,8.452838],[3.286715,8.937593,-2.896207,-1.961751],[-3.689180,2.788452,9.423461,-5.390793],[7.960335,0.594892,7.603019,-6.610754],[-3.993604,7.582439,-8.243790,-3.442421],[9.991908,-6.940031,-5.470460,6.020472],[3.208999,-7.996848,-1.702199,8.277002]],[[8.762507,0.160497,8.059697,1.043562],[6.815694,-6.878394,5.776397,-5.118019],[-0.368734,-2.429056,2.365407,6.566478],[-2.451562,-2.617894,-9.381412,7.352491],[-0.821042,8.716719,-6.528976,-7.127663],[7.350389,9.266575,5.504725,4.776282],[-7.227158,-6.572722,-1.336167,1.227050],[5.913441,4.958152,2.045177,7.441577],[-4.740922,3.137297,1.400280,-1.923390],[1.615997,3.205901,9.683668,-2.449819],[-0.061055,-0.391370,2.436024,-9.786664],[0.001947,-0.276138,-0.318377,-8.870652],[-4.674991,2.333167,3.484176,-4.333032],[-6.159930,-8.091205,4.975459,6.014195]],[[6.535991,-2.142069,9.403111,-8.472245],[-7.003319,-1.717738,-2.473462,1.944744],[-3.074015,6.620721,1.828678,-5.715600],[6.218317,9.858842,-1.439293,6.934776],[-9.646035,-7.265233,-3.413172,-1.957093],[0.601556,1.286517,-6.677852,9.789254],[5.649634,-0.447101,-2.604840,-8.877838],[9.951009,-5.658538,2.247996,-4.843784],[5.481436,9.012366,2.692187,3.571082],[-6.240571,-1.864473,7.242532,-8.720133],[-8.792390,0.245015,6.040319,6.812301],[-6.134263,-2.670375,2.776109,7.507426],[2.472889,-5.000821,4.708588,-4.251113],[-0.001752,-8.127130,-2.180007,-7.575181]],[[2.261814,8.262870,-4.296235,-6.036479],[4.437245,-8.165668,2.290739,5.385046],[-5.727051,3.072840,2.540172,-2.731638],[-5.572783,-6.606810,4.510079,-8.185530],[-4.396825,9.720447,-4.078136,7.972985],[0.409531,-1.178505,-8.869746,3.738396],[0.211843,-2.488432,-2.381014,-9.251987],[6.820593,7.128431,6.685204,7.894194],[-8.101190,-1.703157,-5.120885,-8.790146],[9.539912,-5.046808,-8.959118,2.650273],[-3.603675,7.094185,-2.876533,-3.308951],[-8.951024,-0.106178,4.975593,0.555082],[-8.592588,-5.423011,-2.247222,-2.769681],[4.804033,6.815819,1.131383,3.651635]],[[-0.485908,-8.736463,-1.539167,-1.432525],[2.973610,-4.699150,-9.727612,-9.845719],[-3.753728,-7.602649,9.529504,-9.701310],[7.133568,-7.967757,-7.243279,3.680117],[-1.666938,0.418266,-8.318641,3.221013],[-4.631359,-8.167347,-7.439077,-7.658858],[6.645761,8.409824,3.904361,8.445943],[2.145506,7.733651,4.812951,-4.369816],[3.221717,-0.746015,5.199171,5.164002],[4.156434,-8.951381,-5.613965,-7.015971],[2.266176,-9.182206,-2.875782,-7.783025],[1.459162,6.215300,-0.511985,0.711763],[-4.251861,0.815224,6.852887,6.298270],[-2.109762,-3.297467,-5.441458,-4.811786]]], dtype = "float64")#candidate|247|(6, 14, 4)|const|float64
bop_248 = relay.less_equal(uop_240.astype('bool'), relay.reshape(const_247.astype('bool'), relay.shape_of(uop_240))) # shape=(6, 14, 4)
bop_251 = relay.bitwise_and(uop_240.astype('uint64'), relay.reshape(uop_202.astype('uint64'), relay.shape_of(uop_240))) # shape=(6, 14, 4)
uop_254 = relay.cosh(uop_225.astype('float64')) # shape=(6, 14, 4)
var_256 = relay.var("var_256", dtype = "uint64", shape = (6, 14, 4))#candidate|256|(6, 14, 4)|var|uint64
bop_257 = relay.divide(bop_251.astype('float32'), relay.reshape(var_256.astype('float32'), relay.shape_of(bop_251))) # shape=(6, 14, 4)
output = relay.Tuple([call_191,const_192,bop_195,uop_204,call_214,bop_229,bop_232,bop_236,bop_243,bop_248,uop_254,bop_257,])
output2 = relay.Tuple([call_193,const_192,bop_195,uop_204,call_215,bop_229,bop_232,bop_236,bop_243,bop_248,uop_254,bop_257,])
func_261 = relay.Function([var_179,var_180,var_194,var_256,], output)
mod['func_261'] = func_261
mod = relay.transform.InferType()(mod)
var_262 = relay.var("var_262", dtype = "int32", shape = (6, 14, 4))#candidate|262|(6, 14, 4)|var|int32
var_263 = relay.var("var_263", dtype = "int32", shape = (6, 14, 4))#candidate|263|(6, 14, 4)|var|int32
var_264 = relay.var("var_264", dtype = "uint64", shape = (6, 14, 4))#candidate|264|(6, 14, 4)|var|uint64
var_265 = relay.var("var_265", dtype = "uint64", shape = (6, 14, 4))#candidate|265|(6, 14, 4)|var|uint64
output = func_261(var_262,var_263,var_264,var_265,)
func_266 = relay.Function([var_262,var_263,var_264,var_265,], output)
mutated_mod['func_266'] = func_266
mutated_mod = relay.transform.InferType()(mutated_mod)
var_293 = relay.var("var_293", dtype = "float32", shape = (3, 14, 13))#candidate|293|(3, 14, 13)|var|float32
var_294 = relay.var("var_294", dtype = "float32", shape = (3, 14, 13))#candidate|294|(3, 14, 13)|var|float32
bop_295 = relay.less(var_293.astype('bool'), relay.reshape(var_294.astype('bool'), relay.shape_of(var_293))) # shape=(3, 14, 13)
bop_298 = relay.not_equal(bop_295.astype('bool'), relay.reshape(var_294.astype('bool'), relay.shape_of(bop_295))) # shape=(3, 14, 13)
uop_301 = relay.sinh(bop_295.astype('float64')) # shape=(3, 14, 13)
output = relay.Tuple([bop_298,uop_301,])
output2 = relay.Tuple([bop_298,uop_301,])
func_303 = relay.Function([var_293,var_294,], output)
mod['func_303'] = func_303
mod = relay.transform.InferType()(mod)
var_304 = relay.var("var_304", dtype = "float32", shape = (3, 14, 13))#candidate|304|(3, 14, 13)|var|float32
var_305 = relay.var("var_305", dtype = "float32", shape = (3, 14, 13))#candidate|305|(3, 14, 13)|var|float32
output = func_303(var_304,var_305,)
func_306 = relay.Function([var_304,var_305,], output)
mutated_mod['func_306'] = func_306
mutated_mod = relay.transform.InferType()(mutated_mod)
var_313 = relay.var("var_313", dtype = "float64", shape = (3, 14, 9))#candidate|313|(3, 14, 9)|var|float64
uop_314 = relay.sinh(var_313.astype('float64')) # shape=(3, 14, 9)
uop_316 = relay.atan(uop_314.astype('float32')) # shape=(3, 14, 9)
uop_318 = relay.cosh(uop_316.astype('float32')) # shape=(3, 14, 9)
const_321 = relay.const([[[6.467053,0.001534,-5.887931,7.955965,6.515060,1.363183,0.071269,5.263659,9.950741],[5.862255,1.157649,-1.338471,-0.873437,6.840735,4.677993,9.242228,-5.209323,-1.217385],[9.158128,-7.708655,5.664517,-6.760106,-6.703228,0.280470,7.026593,3.388962,6.002992],[-9.724360,-4.798545,-0.283930,0.712522,-9.940835,-8.739645,-6.833030,-1.486144,0.303432],[4.886641,6.437012,6.054651,4.995802,2.145034,4.595966,7.353871,-1.059662,6.704726],[0.220624,-0.844188,-2.354440,-1.644242,-9.290757,3.542060,9.645042,6.878051,-7.523197],[5.566650,5.180433,-7.456616,2.405915,7.347893,7.074540,-4.078718,8.562782,3.216722],[-0.546718,-4.796106,-2.419212,-9.009811,-5.453692,-8.593345,-6.822512,8.667644,-4.854511],[-4.151110,0.977257,-0.861997,-3.980301,3.058587,2.327080,5.957929,-2.985659,6.123298],[-0.657562,-4.435672,9.308704,7.147696,7.237496,-0.227763,5.989224,4.851565,-2.197049],[8.244591,2.026392,-6.090111,4.441168,-6.913794,2.029923,5.576589,3.778688,5.355175],[-8.199155,-8.578346,7.119774,-0.237935,5.011346,9.444618,2.155392,-3.199986,0.923384],[1.879981,-6.838867,-5.471831,0.087567,6.129359,-1.680163,-2.596584,-4.090705,-1.430459],[8.525881,-1.234373,-7.674061,-7.807703,-3.441799,-0.162995,0.548702,2.635732,5.921268]],[[6.796384,0.967054,5.606255,4.827884,-4.453120,-5.884068,9.227954,3.099507,-7.301786],[2.784406,4.092321,1.112328,8.708492,-0.417746,3.360400,6.257825,-9.017917,8.904059],[-3.051238,0.346243,-0.854387,3.479502,1.608391,9.953575,-5.527652,-4.653210,3.710122],[-0.980766,-1.577156,2.595050,-2.590901,1.371948,2.792070,1.614289,3.344219,-5.092354],[-8.469986,-4.908424,8.223730,-5.637926,-9.770925,8.514317,-7.508616,5.595525,-5.605280],[-5.021112,7.387439,-7.228664,5.498234,-9.809727,-7.162392,-2.434086,-7.811794,9.400946],[7.737956,9.250219,-6.183416,3.294163,-5.036934,-4.557233,6.599616,-7.905369,-6.883642],[1.597551,-7.137859,-9.509877,-7.978362,3.600844,4.426443,-5.327392,-9.987493,2.553709],[6.197948,-4.801786,6.705447,-2.615247,9.762601,1.364043,4.454647,3.937543,-0.462439],[6.333894,3.879926,5.163249,6.164948,3.600402,3.753270,-2.693886,6.938804,8.979366],[2.215198,5.910299,-9.810407,-8.704353,4.922422,1.559581,-8.353350,-5.232489,9.783883],[-0.034517,-0.813789,7.599073,-9.207113,-9.642589,-7.129601,-5.224174,7.811373,-1.191467],[-6.920149,-1.767351,1.898207,7.070710,-8.402059,5.492384,1.153734,-7.094651,5.895491],[-0.591647,2.571077,2.002357,-1.557302,-3.460587,7.984374,0.420187,0.676668,4.991868]],[[2.290514,-4.233672,-0.429229,6.519066,-8.537155,5.226908,9.282130,7.773655,-2.847960],[-2.241759,2.625838,9.015896,-9.844902,-8.907400,-7.245643,-4.421109,9.482272,9.922371],[-2.533844,2.100769,-8.835861,-8.844086,-9.793302,-9.294060,-5.734915,9.781416,1.905083],[-0.001058,6.373338,-1.148205,-1.336885,9.109795,4.528969,7.990137,-2.027412,-0.961614],[-2.837431,-7.744780,3.980119,6.314296,-8.499127,5.117552,5.725511,0.021817,-8.948853],[-6.917022,-4.447917,-3.252205,-3.970213,3.053911,9.592763,0.681621,-6.360150,-0.443761],[7.450142,9.683492,3.860223,1.922411,-2.879955,7.562479,-3.060022,5.869879,-7.963130],[5.202010,3.353086,5.388856,-5.096773,-5.001603,-7.946830,-0.644834,0.605773,9.572103],[-5.903222,1.088288,6.945533,-7.684521,2.441093,-9.455280,7.168715,-8.127706,-4.985931],[2.182372,1.424760,0.201832,7.896520,1.685121,4.497297,5.254058,1.022185,-2.599512],[2.357438,0.382199,-8.676235,-5.925344,7.642435,8.497302,-0.290550,-0.968988,-2.121044],[-9.776192,-7.784310,-2.417001,-0.517402,7.574124,1.482748,6.850820,-5.214800,2.100714],[-4.097444,7.740067,-8.225420,-7.315631,1.684161,-2.647523,-6.124566,2.168666,-8.360578],[8.396529,9.779142,6.947790,-2.319366,-1.661251,0.015495,-7.507945,7.197071,4.968150]]], dtype = "float32")#candidate|321|(3, 14, 9)|const|float32
bop_322 = relay.subtract(uop_318.astype('uint8'), relay.reshape(const_321.astype('uint8'), relay.shape_of(uop_318))) # shape=(3, 14, 9)
const_325 = relay.const([[[-7,9,1,7,-2,-8,-7,3,5],[-7,-9,6,-5,7,3,-9,8,2],[-1,5,-1,-10,-4,-2,-3,3,-5],[8,-4,-6,-6,-1,5,6,-4,9],[-6,4,4,-10,-5,2,9,-10,6],[-2,-10,-7,-6,-3,3,-6,3,-8],[7,5,-10,2,9,-2,7,-5,7],[-4,-6,3,-10,-10,-2,3,6,-1],[-5,8,6,-8,6,-10,7,7,-4],[5,-1,9,3,1,10,-8,9,-10],[5,-6,1,-5,-3,-4,-3,10,-3],[-7,1,-6,2,-7,-1,3,-4,9],[8,-8,6,-9,10,-3,10,-7,-10],[3,10,-2,-8,-7,8,10,1,9]],[[9,-8,-9,9,-1,-2,3,2,9],[4,-10,10,8,4,-9,7,-10,6],[-6,-9,-5,-9,-8,2,4,5,-7],[-10,6,-4,-6,4,6,5,4,10],[-7,-8,9,-2,-3,6,3,-7,-1],[-5,-2,5,6,-1,-4,-2,7,5],[9,8,4,4,-2,-6,1,-2,-10],[-10,-4,9,-1,10,10,-5,-3,-10],[7,-6,-6,3,-6,10,-1,2,-3],[-5,1,-2,1,9,10,-4,-5,9],[-7,-1,-7,-10,-3,-4,-3,-1,-6],[-5,-10,-2,-4,-1,-1,2,-1,2],[6,-6,6,5,1,-6,-9,-4,5],[-1,-6,-10,1,6,9,-3,2,1]],[[-1,9,-2,-1,-6,-10,-4,-2,5],[9,7,9,-5,-8,4,-7,6,1],[3,-9,-6,3,-1,-10,8,2,5],[1,6,9,9,-6,-5,2,-8,10],[9,-6,8,-5,-7,4,-10,4,-4],[-7,2,-9,2,-7,-2,6,2,-2],[-8,-9,-7,7,-7,3,-5,-4,6],[5,5,-8,-9,-8,8,1,10,3],[2,5,2,-1,-10,3,8,-4,10],[-10,8,-1,-10,4,-4,-9,6,8],[-4,4,-5,2,5,-6,9,5,4],[6,10,-7,-5,-5,-2,9,-3,2],[10,-1,9,1,1,-7,-5,10,3],[4,-4,10,-1,4,2,1,-1,8]]], dtype = "uint8")#candidate|325|(3, 14, 9)|const|uint8
bop_326 = relay.less(bop_322.astype('bool'), relay.reshape(const_325.astype('bool'), relay.shape_of(bop_322))) # shape=(3, 14, 9)
bop_329 = relay.power(bop_322.astype('float32'), relay.reshape(const_325.astype('float32'), relay.shape_of(bop_322))) # shape=(3, 14, 9)
bop_334 = relay.bitwise_and(uop_318.astype('uint64'), relay.reshape(bop_322.astype('uint64'), relay.shape_of(uop_318))) # shape=(3, 14, 9)
bop_339 = relay.not_equal(uop_316.astype('bool'), relay.reshape(bop_329.astype('bool'), relay.shape_of(uop_316))) # shape=(3, 14, 9)
var_343 = relay.var("var_343", dtype = "float32", shape = (3, 14, 9))#candidate|343|(3, 14, 9)|var|float32
bop_344 = relay.multiply(bop_329.astype('int32'), relay.reshape(var_343.astype('int32'), relay.shape_of(bop_329))) # shape=(3, 14, 9)
bop_351 = relay.left_shift(bop_326.astype('uint64'), relay.reshape(bop_339.astype('uint64'), relay.shape_of(bop_326))) # shape=(3, 14, 9)
bop_355 = relay.left_shift(const_321.astype('uint8'), relay.reshape(bop_344.astype('uint8'), relay.shape_of(const_321))) # shape=(3, 14, 9)
bop_359 = relay.equal(uop_316.astype('bool'), relay.reshape(var_343.astype('bool'), relay.shape_of(uop_316))) # shape=(3, 14, 9)
bop_362 = relay.equal(bop_351.astype('bool'), relay.reshape(var_343.astype('bool'), relay.shape_of(bop_351))) # shape=(3, 14, 9)
bop_365 = relay.bitwise_xor(bop_339.astype('uint16'), relay.reshape(var_313.astype('uint16'), relay.shape_of(bop_339))) # shape=(3, 14, 9)
bop_369 = relay.greater_equal(bop_326.astype('bool'), relay.reshape(var_313.astype('bool'), relay.shape_of(bop_326))) # shape=(3, 14, 9)
bop_372 = relay.less_equal(bop_359.astype('bool'), relay.reshape(uop_318.astype('bool'), relay.shape_of(bop_359))) # shape=(3, 14, 9)
uop_375 = relay.asinh(bop_334.astype('float32')) # shape=(3, 14, 9)
output = relay.Tuple([bop_355,bop_362,bop_365,bop_369,bop_372,uop_375,])
output2 = relay.Tuple([bop_355,bop_362,bop_365,bop_369,bop_372,uop_375,])
func_377 = relay.Function([var_313,var_343,], output)
mod['func_377'] = func_377
mod = relay.transform.InferType()(mod)
mutated_mod['func_377'] = func_377
mutated_mod = relay.transform.InferType()(mutated_mod)
func_377_call = mutated_mod.get_global_var('func_377')
var_379 = relay.var("var_379", dtype = "float64", shape = (3, 14, 9))#candidate|379|(3, 14, 9)|var|float64
var_380 = relay.var("var_380", dtype = "float32", shape = (3, 14, 9))#candidate|380|(3, 14, 9)|var|float32
call_378 = func_377_call(var_379,var_380,)
output = call_378
func_381 = relay.Function([var_379,var_380,], output)
mutated_mod['func_381'] = func_381
mutated_mod = relay.transform.InferType()(mutated_mod)
const_388 = relay.const([[-9,9,-4,4,-4,-9,-8,-6,6],[-3,-7,1,-9,1,8,-6,-2,-5],[-5,6,-5,6,7,8,-5,-8,-5],[-10,-1,-10,2,-8,-3,3,-3,-9],[-5,2,9,7,2,4,4,4,4],[-8,10,-1,2,10,4,-6,6,-2],[9,-9,-5,-8,7,-8,-4,7,4],[-8,1,-2,9,5,5,8,-5,2],[-10,-10,-10,9,-5,-1,-6,3,-7],[2,10,-1,5,8,-6,1,-1,-6],[9,7,2,1,7,5,1,-9,-7],[8,10,1,6,-4,2,8,-5,-5],[2,-7,4,-2,3,5,-2,3,9],[-1,-7,10,7,3,2,1,6,3],[-3,-4,-8,6,4,-3,-3,-7,-9]], dtype = "int64")#candidate|388|(15, 9)|const|int64
var_389 = relay.var("var_389", dtype = "int64", shape = (15, 9))#candidate|389|(15, 9)|var|int64
bop_390 = relay.bitwise_and(const_388.astype('int64'), relay.reshape(var_389.astype('int64'), relay.shape_of(const_388))) # shape=(15, 9)
uop_397 = relay.sqrt(bop_390.astype('float32')) # shape=(15, 9)
bop_400 = relay.greater_equal(uop_397.astype('bool'), relay.reshape(var_389.astype('bool'), relay.shape_of(uop_397))) # shape=(15, 9)
uop_403 = relay.asinh(bop_400.astype('float64')) # shape=(15, 9)
output = relay.Tuple([uop_403,])
output2 = relay.Tuple([uop_403,])
func_405 = relay.Function([var_389,], output)
mod['func_405'] = func_405
mod = relay.transform.InferType()(mod)
mutated_mod['func_405'] = func_405
mutated_mod = relay.transform.InferType()(mutated_mod)
var_406 = relay.var("var_406", dtype = "int64", shape = (15, 9))#candidate|406|(15, 9)|var|int64
func_405_call = mutated_mod.get_global_var('func_405')
call_407 = func_405_call(var_406)
output = call_407
func_408 = relay.Function([var_406], output)
mutated_mod['func_408'] = func_408
mutated_mod = relay.transform.InferType()(mutated_mod)
var_438 = relay.var("var_438", dtype = "uint16", shape = (13, 2, 2))#candidate|438|(13, 2, 2)|var|uint16
var_439 = relay.var("var_439", dtype = "uint16", shape = (13, 2, 2))#candidate|439|(13, 2, 2)|var|uint16
bop_440 = relay.greater(var_438.astype('bool'), relay.reshape(var_439.astype('bool'), relay.shape_of(var_438))) # shape=(13, 2, 2)
bop_445 = relay.power(bop_440.astype('float64'), relay.reshape(var_439.astype('float64'), relay.shape_of(bop_440))) # shape=(13, 2, 2)
const_448 = relay.const([[[-5.391023,-9.640560],[4.715088,0.709860]],[[9.423259,-9.446243],[3.886445,-9.853123]],[[-7.930508,9.011482],[9.792102,-8.970636]],[[-0.850805,-9.626904],[-4.560978,6.165149]],[[-8.598658,-9.826359],[5.382153,-7.633461]],[[0.439951,4.456098],[-8.646534,-9.219520]],[[7.074437,7.706818],[2.537816,4.016310]],[[-6.215880,9.535943],[9.427430,4.893203]],[[-9.989948,6.316730],[-0.083502,-2.010926]],[[-2.930376,-0.814493],[3.036755,8.049134]],[[-1.715726,-3.689362],[-0.211219,6.046413]],[[4.609209,-9.383388],[7.576149,-7.635324]],[[3.497315,-8.848714],[-0.562093,1.151042]]], dtype = "float64")#candidate|448|(13, 2, 2)|const|float64
bop_449 = relay.logical_xor(bop_445.astype('int32'), relay.reshape(const_448.astype('int32'), relay.shape_of(bop_445))) # shape=(13, 2, 2)
bop_453 = relay.mod(const_448.astype('float64'), relay.reshape(bop_445.astype('float64'), relay.shape_of(const_448))) # shape=(13, 2, 2)
uop_456 = relay.erf(bop_449.astype('float32')) # shape=(13, 2, 2)
bop_459 = relay.not_equal(uop_456.astype('bool'), relay.reshape(var_438.astype('bool'), relay.shape_of(uop_456))) # shape=(13, 2, 2)
func_261_call = mod.get_global_var('func_261')
func_266_call = mutated_mod.get_global_var('func_266')
const_465 = relay.const([7,5,6,-10,2,10,7,-3,7,-9,-1,4,9,10,1,7,-1,9,6,6,5,6,-9,1,9,2,-5,1,10,-8,-1,1,7,-9,-9,3,-10,-8,-7,7,-8,-10,-5,-9,-2,-8,-1,-7,8,1,3,-3,-2,2,-1,7,-9,-4,-10,1,-9,-5,9,9,7,7,4,3,-1,7,-7,1,3,-10,10,5,10,10,-1,9,8,8,-5,10,-9,8,-1,-1,6,-9,3,-8,1,4,-6,10,8,8,2,-9,-6,6,-8,4,-8,-2,5,6,7,-8,5,-7,-7,-6,4,1,-4,9,-3,6,6,-4,8,-8,2,-8,-5,3,3,-2,-7,-8,7,-1,-10,4,6,6,-8,1,-10,10,-7,7,2,-6,2,10,-1,2,9,-7,3,-3,10,2,4,-5,-5,-10,4,6,9,-5,-6,-2,-2,1,4,6,-9,-4,-2,-5,-3,-6,10,5,9,-10,7,-1,2,-6,-8,-2,1,-10,-7,10,10,1,7,9,-5,10,-2,-3,-4,3,-9,9,-1,3,-3,-9,10,-3,5,-2,1,-10,-4,9,2,-10,-5,3,9,5,10,7,6,-8,1,9,3,-7,4,6,6,-7,6,-10,-7,-7,10,8,3,5,5,8,-5,6,10,5,8,10,1,1,4,-6,3,10,-9,9,-9,1,-1,3,-7,-9,10,2,-7,1,10,-6,1,-10,1,4,6,5,-10,-1,7,-5,9,-10,10,-10,9,5,1,10,-5,-5,-8,10,-9,9,1,4,-3,6,6,-6,-1,-4,-6,-6,9,-4,2,-6,-5,10,1,5,8,7,-7,-10,-4,3,-9,3,-2,6,8,-6,9,10,10,3,-7,4,5,-2,-8,-1,4,6,10,2], dtype = "int32")#candidate|465|(336,)|const|int32
call_464 = relay.TupleGetItem(func_261_call(relay.reshape(const_465.astype('int32'), [6, 14, 4]), relay.reshape(const_465.astype('int32'), [6, 14, 4]), relay.reshape(const_465.astype('uint64'), [6, 14, 4]), relay.reshape(const_465.astype('uint64'), [6, 14, 4]), ), 8)
call_466 = relay.TupleGetItem(func_266_call(relay.reshape(const_465.astype('int32'), [6, 14, 4]), relay.reshape(const_465.astype('int32'), [6, 14, 4]), relay.reshape(const_465.astype('uint64'), [6, 14, 4]), relay.reshape(const_465.astype('uint64'), [6, 14, 4]), ), 8)
var_467 = relay.var("var_467", dtype = "bool", shape = (13, 2, 2))#candidate|467|(13, 2, 2)|var|bool
bop_468 = relay.maximum(bop_459.astype('int32'), relay.reshape(var_467.astype('int32'), relay.shape_of(bop_459))) # shape=(13, 2, 2)
uop_474 = relay.asinh(bop_468.astype('float32')) # shape=(13, 2, 2)
bop_476 = relay.multiply(uop_474.astype('float64'), relay.reshape(var_438.astype('float64'), relay.shape_of(uop_474))) # shape=(13, 2, 2)
uop_479 = relay.exp(uop_456.astype('float32')) # shape=(13, 2, 2)
var_481 = relay.var("var_481", dtype = "int32", shape = (13, 2, 2))#candidate|481|(13, 2, 2)|var|int32
bop_482 = relay.greater(bop_468.astype('bool'), relay.reshape(var_481.astype('bool'), relay.shape_of(bop_468))) # shape=(13, 2, 2)
uop_487 = relay.tan(uop_474.astype('float32')) # shape=(13, 2, 2)
bop_492 = relay.logical_or(uop_487.astype('bool'), relay.reshape(bop_482.astype('bool'), relay.shape_of(uop_487))) # shape=(13, 2, 2)
uop_495 = relay.log(bop_492.astype('float32')) # shape=(13, 2, 2)
bop_497 = relay.right_shift(uop_487.astype('int16'), relay.reshape(bop_449.astype('int16'), relay.shape_of(uop_487))) # shape=(13, 2, 2)
output = relay.Tuple([bop_453,call_464,const_465,bop_476,uop_479,uop_495,bop_497,])
output2 = relay.Tuple([bop_453,call_466,const_465,bop_476,uop_479,uop_495,bop_497,])
func_500 = relay.Function([var_438,var_439,var_467,var_481,], output)
mod['func_500'] = func_500
mod = relay.transform.InferType()(mod)
mutated_mod['func_500'] = func_500
mutated_mod = relay.transform.InferType()(mutated_mod)
func_500_call = mutated_mod.get_global_var('func_500')
var_502 = relay.var("var_502", dtype = "uint16", shape = (13, 2, 2))#candidate|502|(13, 2, 2)|var|uint16
var_503 = relay.var("var_503", dtype = "uint16", shape = (13, 2, 2))#candidate|503|(13, 2, 2)|var|uint16
var_504 = relay.var("var_504", dtype = "bool", shape = (13, 2, 2))#candidate|504|(13, 2, 2)|var|bool
var_505 = relay.var("var_505", dtype = "int32", shape = (13, 2, 2))#candidate|505|(13, 2, 2)|var|int32
call_501 = func_500_call(var_502,var_503,var_504,var_505,)
output = call_501
func_506 = relay.Function([var_502,var_503,var_504,var_505,], output)
mutated_mod['func_506'] = func_506
mutated_mod = relay.transform.InferType()(mutated_mod)
var_526 = relay.var("var_526", dtype = "float64", shape = ())#candidate|526|()|var|float64
uop_527 = relay.log(var_526.astype('float64')) # shape=()
bop_530 = relay.greater_equal(uop_527.astype('bool'), var_526.astype('bool')) # shape=()
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
var_535 = relay.var("var_535", dtype = "float32", shape = (56,))#candidate|535|(56,)|var|float32
call_534 = relay.TupleGetItem(func_80_call(relay.reshape(var_535.astype('float32'), [14, 4]), relay.reshape(var_535.astype('float32'), [14, 4]), relay.reshape(var_535.astype('float32'), [14, 4]), ), 0)
call_536 = relay.TupleGetItem(func_84_call(relay.reshape(var_535.astype('float32'), [14, 4]), relay.reshape(var_535.astype('float32'), [14, 4]), relay.reshape(var_535.astype('float32'), [14, 4]), ), 0)
func_377_call = mod.get_global_var('func_377')
func_381_call = mutated_mod.get_global_var('func_381')
var_541 = relay.var("var_541", dtype = "float64", shape = (378,))#candidate|541|(378,)|var|float64
call_540 = relay.TupleGetItem(func_377_call(relay.reshape(var_541.astype('float64'), [3, 14, 9]), relay.reshape(var_541.astype('float32'), [3, 14, 9]), ), 0)
call_542 = relay.TupleGetItem(func_381_call(relay.reshape(var_541.astype('float64'), [3, 14, 9]), relay.reshape(var_541.astype('float32'), [3, 14, 9]), ), 0)
output = relay.Tuple([bop_530,call_534,var_535,call_540,var_541,])
output2 = relay.Tuple([bop_530,call_536,var_535,call_542,var_541,])
func_547 = relay.Function([var_526,var_535,var_541,], output)
mod['func_547'] = func_547
mod = relay.transform.InferType()(mod)
mutated_mod['func_547'] = func_547
mutated_mod = relay.transform.InferType()(mutated_mod)
func_547_call = mutated_mod.get_global_var('func_547')
var_549 = relay.var("var_549", dtype = "float64", shape = ())#candidate|549|()|var|float64
var_550 = relay.var("var_550", dtype = "float32", shape = (56,))#candidate|550|(56,)|var|float32
var_551 = relay.var("var_551", dtype = "float64", shape = (378,))#candidate|551|(378,)|var|float64
call_548 = func_547_call(var_549,var_550,var_551,)
output = call_548
func_552 = relay.Function([var_549,var_550,var_551,], output)
mutated_mod['func_552'] = func_552
mutated_mod = relay.transform.InferType()(mutated_mod)
var_561 = relay.var("var_561", dtype = "int64", shape = ())#candidate|561|()|var|int64
var_562 = relay.var("var_562", dtype = "int64", shape = (5,))#candidate|562|(5,)|var|int64
bop_563 = relay.greater_equal(var_561.astype('bool'), var_562.astype('bool')) # shape=(5,)
bop_568 = relay.divide(bop_563.astype('float32'), relay.reshape(var_562.astype('float32'), relay.shape_of(bop_563))) # shape=(5,)
bop_571 = relay.logical_and(var_561.astype('bool'), bop_568.astype('bool')) # shape=(5,)
bop_575 = relay.multiply(bop_568.astype('int64'), var_561.astype('int64')) # shape=(5,)
func_547_call = mod.get_global_var('func_547')
func_552_call = mutated_mod.get_global_var('func_552')
var_581 = relay.var("var_581", dtype = "float32", shape = (2, 28))#candidate|581|(2, 28)|var|float32
var_582 = relay.var("var_582", dtype = "float64", shape = (378,))#candidate|582|(378,)|var|float64
call_580 = relay.TupleGetItem(func_547_call(relay.reshape(var_561.astype('float64'), []), relay.reshape(var_581.astype('float32'), [56,]), relay.reshape(var_582.astype('float64'), [378,]), ), 0)
call_583 = relay.TupleGetItem(func_552_call(relay.reshape(var_561.astype('float64'), []), relay.reshape(var_581.astype('float32'), [56,]), relay.reshape(var_582.astype('float64'), [378,]), ), 0)
var_584 = relay.var("var_584", dtype = "float32", shape = (2, 28))#candidate|584|(2, 28)|var|float32
bop_585 = relay.minimum(var_581.astype('uint16'), relay.reshape(var_584.astype('uint16'), relay.shape_of(var_581))) # shape=(2, 28)
const_593 = relay.const([False,True,False,False,False], dtype = "bool")#candidate|593|(5,)|const|bool
bop_594 = relay.greater(bop_563.astype('bool'), relay.reshape(const_593.astype('bool'), relay.shape_of(bop_563))) # shape=(5,)
var_597 = relay.var("var_597", dtype = "uint16", shape = (2, 28))#candidate|597|(2, 28)|var|uint16
bop_598 = relay.minimum(bop_585.astype('int64'), relay.reshape(var_597.astype('int64'), relay.shape_of(bop_585))) # shape=(2, 28)
output = relay.Tuple([bop_571,bop_575,call_580,var_582,bop_594,bop_598,])
output2 = relay.Tuple([bop_571,bop_575,call_583,var_582,bop_594,bop_598,])
func_604 = relay.Function([var_561,var_562,var_581,var_582,var_584,var_597,], output)
mod['func_604'] = func_604
mod = relay.transform.InferType()(mod)
var_605 = relay.var("var_605", dtype = "int64", shape = ())#candidate|605|()|var|int64
var_606 = relay.var("var_606", dtype = "int64", shape = (5,))#candidate|606|(5,)|var|int64
var_607 = relay.var("var_607", dtype = "float32", shape = (2, 28))#candidate|607|(2, 28)|var|float32
var_608 = relay.var("var_608", dtype = "float64", shape = (378,))#candidate|608|(378,)|var|float64
var_609 = relay.var("var_609", dtype = "float32", shape = (2, 28))#candidate|609|(2, 28)|var|float32
var_610 = relay.var("var_610", dtype = "uint16", shape = (2, 28))#candidate|610|(2, 28)|var|uint16
output = func_604(var_605,var_606,var_607,var_608,var_609,var_610,)
func_611 = relay.Function([var_605,var_606,var_607,var_608,var_609,var_610,], output)
mutated_mod['func_611'] = func_611
mutated_mod = relay.transform.InferType()(mutated_mod)
var_625 = relay.var("var_625", dtype = "float32", shape = ())#candidate|625|()|var|float32
var_626 = relay.var("var_626", dtype = "float32", shape = (9,))#candidate|626|(9,)|var|float32
bop_627 = relay.subtract(var_625.astype('float32'), var_626.astype('float32')) # shape=(9,)
uop_630 = relay.erf(bop_627.astype('float64')) # shape=(9,)
uop_632 = relay.log(uop_630.astype('float64')) # shape=(9,)
bop_635 = relay.bitwise_or(var_625.astype('int16'), uop_630.astype('int16')) # shape=(9,)
const_639 = relay.const([-1.802302,-2.445423,-8.996919,5.827768,-6.043092,7.091967,7.588223,9.540537,-2.505215], dtype = "float64")#candidate|639|(9,)|const|float64
bop_640 = relay.mod(uop_632.astype('float64'), relay.reshape(const_639.astype('float64'), relay.shape_of(uop_632))) # shape=(9,)
func_80_call = mod.get_global_var('func_80')
func_84_call = mutated_mod.get_global_var('func_84')
const_644 = relay.const([[-6.906044,-8.419148],[3.453875,-7.134728],[-0.335869,-7.640156],[5.976751,-2.984665],[-1.669454,-0.959311],[-2.867047,4.377319],[6.609356,7.672367],[7.610396,1.916471],[4.807454,1.989702],[6.683625,-2.894308],[2.774790,-0.397885],[-8.919066,-3.676610],[9.161114,-0.603841],[-4.680807,-5.303967],[-2.047455,5.716518],[-5.414064,4.279226],[5.353686,-9.695297],[-7.059481,-0.751492],[9.911919,9.693161],[7.611759,-5.582232],[-0.359925,-8.477555],[-3.718384,7.788809],[-9.560865,2.243274],[8.010833,-4.538825],[5.865339,4.024745],[5.593374,4.977189],[-8.077306,5.060515],[5.089068,-3.558309]], dtype = "float32")#candidate|644|(28, 2)|const|float32
call_643 = relay.TupleGetItem(func_80_call(relay.reshape(const_644.astype('float32'), [14, 4]), relay.reshape(const_644.astype('float32'), [14, 4]), relay.reshape(const_644.astype('float32'), [14, 4]), ), 1)
call_645 = relay.TupleGetItem(func_84_call(relay.reshape(const_644.astype('float32'), [14, 4]), relay.reshape(const_644.astype('float32'), [14, 4]), relay.reshape(const_644.astype('float32'), [14, 4]), ), 1)
bop_647 = relay.equal(bop_640.astype('bool'), relay.reshape(var_626.astype('bool'), relay.shape_of(bop_640))) # shape=(9,)
bop_650 = relay.bitwise_or(bop_647.astype('uint64'), relay.reshape(uop_630.astype('uint64'), relay.shape_of(bop_647))) # shape=(9,)
const_653 = relay.const([-2,1,-3,10,-2,-9,1,-7,-4], dtype = "int16")#candidate|653|(9,)|const|int16
bop_654 = relay.mod(bop_635.astype('float32'), relay.reshape(const_653.astype('float32'), relay.shape_of(bop_635))) # shape=(9,)
output = relay.Tuple([call_643,const_644,bop_650,bop_654,])
output2 = relay.Tuple([call_645,const_644,bop_650,bop_654,])
func_657 = relay.Function([var_625,var_626,], output)
mod['func_657'] = func_657
mod = relay.transform.InferType()(mod)
var_658 = relay.var("var_658", dtype = "float32", shape = ())#candidate|658|()|var|float32
var_659 = relay.var("var_659", dtype = "float32", shape = (9,))#candidate|659|(9,)|var|float32
output = func_657(var_658,var_659,)
func_660 = relay.Function([var_658,var_659,], output)
mutated_mod['func_660'] = func_660
mutated_mod = relay.transform.InferType()(mutated_mod)
var_680 = relay.var("var_680", dtype = "int8", shape = (11, 8, 3))#candidate|680|(11, 8, 3)|var|int8
const_681 = relay.const([[[-7,-5,4],[4,10,5],[5,4,-3],[3,6,1],[-3,-7,-2],[-10,-6,4],[10,3,1],[7,7,3]],[[10,-9,9],[8,-6,-2],[6,9,-7],[-1,10,-9],[1,6,5],[10,-7,-2],[1,-6,-6],[9,-10,-3]],[[1,6,-9],[2,-10,2],[2,-3,-2],[-3,6,4],[-8,4,-1],[-7,-3,4],[7,-6,6],[4,-9,-5]],[[3,-8,7],[-1,-6,5],[3,10,4],[-2,9,9],[3,1,10],[6,4,-2],[-7,6,-9],[-3,-2,2]],[[6,-8,-9],[8,2,-7],[5,-8,9],[-10,5,2],[-8,8,-5],[-7,-2,5],[3,7,-3],[8,10,-5]],[[-7,7,-2],[6,3,9],[4,-10,2],[-2,-9,10],[3,-9,5],[-4,-10,4],[2,-9,6],[3,-5,-8]],[[10,-2,7],[10,10,-5],[-5,7,1],[-7,-1,-8],[-6,7,-5],[8,3,4],[-2,3,8],[-8,8,8]],[[7,2,4],[3,10,1],[4,-3,-1],[-10,3,-6],[-1,4,10],[-1,-7,-4],[-5,-1,6],[-2,-3,-9]],[[2,-1,3],[6,4,5],[1,-10,1],[1,1,-2],[-6,-9,-1],[-5,-10,-10],[4,6,5],[1,-7,8]],[[4,-5,1],[-4,-4,-6],[-3,-10,9],[-3,5,2],[-1,7,5],[10,-5,8],[4,1,-7],[5,-9,-3]],[[4,-1,9],[10,-8,10],[4,-5,-8],[-3,5,2],[-2,-1,3],[-9,-4,-5],[-6,3,-4],[9,6,5]]], dtype = "int8")#candidate|681|(11, 8, 3)|const|int8
bop_682 = relay.right_shift(var_680.astype('int8'), relay.reshape(const_681.astype('int8'), relay.shape_of(var_680))) # shape=(11, 8, 3)
var_686 = relay.var("var_686", dtype = "int8", shape = (11, 8, 3))#candidate|686|(11, 8, 3)|var|int8
bop_687 = relay.less(var_680.astype('bool'), relay.reshape(var_686.astype('bool'), relay.shape_of(var_680))) # shape=(11, 8, 3)
output = relay.Tuple([bop_682,bop_687,])
output2 = relay.Tuple([bop_682,bop_687,])
func_691 = relay.Function([var_680,var_686,], output)
mod['func_691'] = func_691
mod = relay.transform.InferType()(mod)
var_692 = relay.var("var_692", dtype = "int8", shape = (11, 8, 3))#candidate|692|(11, 8, 3)|var|int8
var_693 = relay.var("var_693", dtype = "int8", shape = (11, 8, 3))#candidate|693|(11, 8, 3)|var|int8
output = func_691(var_692,var_693,)
func_694 = relay.Function([var_692,var_693,], output)
mutated_mod['func_694'] = func_694
mutated_mod = relay.transform.InferType()(mutated_mod)
var_706 = relay.var("var_706", dtype = "uint64", shape = (2,))#candidate|706|(2,)|var|uint64
var_707 = relay.var("var_707", dtype = "uint64", shape = (2,))#candidate|707|(2,)|var|uint64
bop_708 = relay.less_equal(var_706.astype('bool'), relay.reshape(var_707.astype('bool'), relay.shape_of(var_706))) # shape=(2,)
output = relay.Tuple([bop_708,])
output2 = relay.Tuple([bop_708,])
F = relay.Function([var_706,var_707,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_706,var_707,], output2)
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
input_706= np.array([10,-3], dtype='uint64')
module1.set_input('var_706', input_706)
input_707= np.array([-4,7], dtype='uint64')
module1.set_input('var_707', input_707)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_706, input_707, )
res3 = intrp3.evaluate()(input_706, input_707, )
res4 = intrp4.evaluate()(input_706, input_707, )
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
module5.set_input('var_706', input_706)
module5.set_input('var_707', input_707)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_706, input_707, )
res7 = intrp7.evaluate()(input_706, input_707, )
res8 = intrp8.evaluate()(input_706, input_707, )
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
module9.set_input('var_706', input_706)
module9.set_input('var_707', input_707)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_706, input_707, )
res11 = intrp11.evaluate()(input_706, input_707, )
res12 = intrp12.evaluate()(input_706, input_707, )
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
module13.set_input('var_706', input_706)
module13.set_input('var_707', input_707)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_706, input_707, )
res15 = intrp15.evaluate()(input_706, input_707, )
res16 = intrp16.evaluate()(input_706, input_707, )
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
module17.set_input('var_706', input_706)
module17.set_input('var_707', input_707)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_706, input_707, )
res19 = intrp19.evaluate()(input_706, input_707, )
res20 = intrp20.evaluate()(input_706, input_707, )
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
module21.set_input('var_706', input_706)
module21.set_input('var_707', input_707)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_706, input_707, )
res23 = intrp23.evaluate()(input_706, input_707, )
res24 = intrp24.evaluate()(input_706, input_707, )
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

'''49: TVMFuncCall
48: _ZNSt17_Function_handlerIFvN3tvm7run
47: tvm::runtime::TypedPackedFunc<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>::AssignTypedLambda<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>(tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target), std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const
46: tvm::relay::EvalFunction(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)
45: tvm::relay::Prepare(tvm::IRModule, tvm::CompilationConfig)
44: tvm::transform::Pass::operator()(tvm::IRModule) const
43: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
42: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
41: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
40: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
36: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
35: tvm::transform::Pass::operator()(tvm::IRModule) const
34: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
31: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
30: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
29: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
28: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
27: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
26: _ZN3tvm5relay9transform22Devic
25: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
24: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
23: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
22: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
21: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::LetNode const*)
20: tvm::relay::tec::LowerTensorExprMutator::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
19: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
16: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
15: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
14: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
13: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
12: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
11: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
9: tvm::relay::tec::TECompilerImpl::LowerShapeFunc(tvm::relay::tec::CCacheKey const&)
8: tvm::relay::tec::TECompilerImpl::LowerShapeFuncInternal(tvm::relay::tec::CCacheKey const&)
7: tvm::relay::tec::ShapeFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
6: tvm::relay::tec::MakeShapeFunc::Create(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
5: tvm::relay::tec::MakeShapeFunc::VisitExpr(tvm::RelayExpr const&)
4: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFNS_7runtime
1: tvm::relay::tec::MakeShapeFunc::VisitExpr_(tvm::relay::CallNode const*)
0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
3: TVMFuncCall
2: _ZNSt17_Function_handlerIFvN3tvm7run
1: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
0: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
4: TVMFuncCall
3: _ZNSt17_Function_handlerIFvN3tvm7run
2: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
1: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
0: tvm::runtime::Array<tvm::PrimExpr, void> tvm::runtime::TVMPODValue_::AsObjectRef<tvm::runtime::Array<tvm::PrimExpr, void> >() const

'''