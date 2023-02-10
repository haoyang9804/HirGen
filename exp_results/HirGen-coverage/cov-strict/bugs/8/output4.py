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
var_8 = relay.var("var_8", dtype = "int64", shape = ())#candidate|8|()|var|int64
const_9 = relay.const([[[-2,6,-1,-5,4,-10,8],[-5,-9,-2,2,-2,3,-5],[6,6,-8,-4,-9,-1,-4],[3,-7,2,-7,-5,-10,3],[-6,-2,-4,-10,3,3,-2],[6,-9,7,5,2,-6,2],[5,7,6,4,10,-6,10],[8,6,-10,-10,1,-8,9]],[[9,8,-1,-3,-6,8,-3],[-2,6,6,3,6,-3,-9],[7,7,-1,10,-1,8,-10],[-7,3,-5,-9,-10,7,6],[-5,2,-4,8,5,-7,10],[6,-6,-4,-2,9,-2,-4],[6,-10,6,3,-3,7,-8],[6,4,10,5,-8,-5,-10]],[[7,-7,-3,-4,-10,-1,7],[-3,-4,4,1,7,-10,-4],[-3,3,6,1,-6,10,-3],[1,1,-6,10,-5,8,6],[-9,6,-5,-10,-10,2,-7],[3,5,1,-5,3,8,6],[-9,7,-5,10,1,-2,-2],[-8,9,10,-5,-2,-7,-8]],[[2,-3,10,-9,-1,6,-8],[5,-1,8,6,3,-8,-2],[9,-9,5,2,-2,-10,-1],[-6,-3,8,6,5,3,-3],[-5,5,6,7,-6,7,-1],[2,3,10,1,-3,3,10],[10,-8,-10,-10,-6,10,-5],[9,-5,-7,-1,8,6,-10]],[[1,-6,5,7,8,-6,4],[8,4,-5,-10,-9,3,4],[1,8,-6,-3,10,-8,7],[5,-9,-1,2,-2,-2,5],[10,-3,-3,-3,5,2,7],[-6,-6,-6,-5,10,4,-10],[-6,-10,-4,7,3,-10,5],[-4,4,9,-2,-8,1,-2]],[[1,9,-9,-4,-3,-4,-1],[7,-6,10,-2,1,1,10],[-9,-7,8,-10,9,2,4],[-10,-10,-1,-9,-8,8,-4],[-1,1,9,-2,5,4,3],[6,3,10,-5,2,2,5],[-9,8,-5,2,8,1,-4],[-9,-2,-6,-3,10,7,-1]],[[10,-8,-9,3,-7,10,-8],[1,7,-1,-7,3,3,-1],[5,-1,10,-6,-6,-2,3],[-10,-10,-3,3,5,-2,3],[4,-8,-5,-6,10,1,2],[-1,-3,5,2,-2,5,6],[-8,3,-10,-10,9,10,-7],[-5,4,3,-7,-3,-10,5]],[[-2,1,9,-1,4,-5,-7],[-4,6,5,-7,7,9,-6],[-5,-4,-3,7,8,2,9],[-9,-2,3,2,8,-8,-5],[7,6,2,8,-5,-7,1],[7,7,-8,-8,-9,-9,-2],[-3,4,10,-9,1,8,-10],[-9,4,-3,-3,-4,-2,-3]],[[-9,9,-6,10,-4,8,8],[-9,3,-1,7,2,7,-3],[6,-1,3,-8,3,-4,-9],[-10,10,-5,-9,-2,2,-8],[-9,-9,7,1,-10,-10,7],[7,8,-2,1,3,-6,8],[-10,-9,5,3,10,5,-3],[2,-1,-5,-8,-9,-6,9]],[[-9,2,-3,-6,10,4,5],[9,-4,-7,10,2,-3,-7],[5,-1,-6,2,3,-10,9],[-5,3,4,3,-10,5,2],[-5,7,8,-4,10,-4,6],[-7,7,-3,4,-8,9,1],[-2,-2,-2,8,-6,4,8],[10,-1,-3,9,-9,-1,3]],[[4,4,6,4,-4,10,5],[-7,-10,-9,3,9,5,-6],[-4,-6,2,-6,-5,-8,5],[8,-2,10,-1,-4,10,8],[-3,2,5,8,8,-8,8],[-2,6,10,-7,1,-1,4],[-10,7,-9,3,-8,3,-8],[-1,-1,-7,4,-6,-8,-1]],[[-3,10,-9,1,9,4,-6],[5,10,-8,8,-7,-3,-5],[9,1,3,9,10,10,3],[-7,10,1,-8,10,-4,-10],[9,9,-3,4,-10,8,1],[9,9,-3,10,9,-2,-5],[6,-5,-3,5,-3,-2,9],[4,7,6,-1,-6,-2,8]],[[4,4,9,6,-4,-9,-7],[-1,-10,-10,2,4,-6,6],[-5,-4,8,-10,-7,-10,-7],[-9,10,3,-7,-9,-9,-4],[-3,-5,-7,-7,8,3,-4],[-8,3,-7,9,5,3,7],[-8,8,-3,10,-3,6,-6],[7,-3,-1,2,9,8,-9]],[[9,6,9,9,-7,6,5],[1,1,1,3,-2,2,6],[7,-2,3,4,-1,8,1],[10,-5,7,-7,2,2,4],[-5,-3,-8,-9,-4,7,-7],[3,-8,-5,-4,-9,-3,-10],[2,9,4,10,-1,-5,5],[8,6,9,6,4,-6,-10]]], dtype = "int64")#candidate|9|(14, 8, 7)|const|int64
bop_10 = relay.less_equal(var_8.astype('bool'), const_9.astype('bool')) # shape=(14, 8, 7)
output = relay.Tuple([bop_10,])
output2 = relay.Tuple([bop_10,])
func_15 = relay.Function([var_8,], output)
mod['func_15'] = func_15
mod = relay.transform.InferType()(mod)
var_16 = relay.var("var_16", dtype = "int64", shape = ())#candidate|16|()|var|int64
output = func_15(var_16)
func_17 = relay.Function([var_16], output)
mutated_mod['func_17'] = func_17
mutated_mod = relay.transform.InferType()(mutated_mod)
var_27 = relay.var("var_27", dtype = "float64", shape = (16,))#candidate|27|(16,)|var|float64
uop_28 = relay.asinh(var_27.astype('float64')) # shape=(16,)
bop_32 = relay.logical_xor(uop_28.astype('int8'), relay.reshape(var_27.astype('int8'), relay.shape_of(uop_28))) # shape=(16,)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
const_37 = relay.const(5, dtype = "int64")#candidate|37|()|const|int64
call_36 = relay.TupleGetItem(func_15_call(relay.reshape(const_37.astype('int64'), [])), 0)
call_38 = relay.TupleGetItem(func_17_call(relay.reshape(const_37.astype('int64'), [])), 0)
var_40 = relay.var("var_40", dtype = "float64", shape = (16,))#candidate|40|(16,)|var|float64
bop_41 = relay.multiply(uop_28.astype('float32'), relay.reshape(var_40.astype('float32'), relay.shape_of(uop_28))) # shape=(16,)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
call_46 = relay.TupleGetItem(func_15_call(relay.reshape(const_37.astype('int64'), [])), 0)
call_47 = relay.TupleGetItem(func_17_call(relay.reshape(const_37.astype('int64'), [])), 0)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
call_48 = relay.TupleGetItem(func_15_call(relay.reshape(const_37.astype('int64'), [])), 0)
call_49 = relay.TupleGetItem(func_17_call(relay.reshape(const_37.astype('int64'), [])), 0)
bop_53 = relay.right_shift(var_27.astype('int8'), relay.reshape(uop_28.astype('int8'), relay.shape_of(var_27))) # shape=(16,)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
call_56 = relay.TupleGetItem(func_15_call(relay.reshape(const_37.astype('int64'), [])), 0)
call_57 = relay.TupleGetItem(func_17_call(relay.reshape(const_37.astype('int64'), [])), 0)
bop_60 = relay.less_equal(bop_41.astype('bool'), relay.reshape(var_40.astype('bool'), relay.shape_of(bop_41))) # shape=(16,)
uop_64 = relay.log2(call_46.astype('float32')) # shape=(14, 8, 7)
uop_66 = relay.log2(call_47.astype('float32')) # shape=(14, 8, 7)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
call_67 = relay.TupleGetItem(func_15_call(relay.reshape(const_37.astype('int64'), [])), 0)
call_68 = relay.TupleGetItem(func_17_call(relay.reshape(const_37.astype('int64'), [])), 0)
output = relay.Tuple([bop_32,call_36,const_37,call_48,bop_53,call_56,bop_60,uop_64,call_67,])
output2 = relay.Tuple([bop_32,call_38,const_37,call_49,bop_53,call_57,bop_60,uop_66,call_68,])
func_72 = relay.Function([var_27,var_40,], output)
mod['func_72'] = func_72
mod = relay.transform.InferType()(mod)
var_73 = relay.var("var_73", dtype = "float64", shape = (16,))#candidate|73|(16,)|var|float64
var_74 = relay.var("var_74", dtype = "float64", shape = (16,))#candidate|74|(16,)|var|float64
output = func_72(var_73,var_74,)
func_75 = relay.Function([var_73,var_74,], output)
mutated_mod['func_75'] = func_75
mutated_mod = relay.transform.InferType()(mutated_mod)
var_77 = relay.var("var_77", dtype = "int8", shape = (16,))#candidate|77|(16,)|var|int8
var_78 = relay.var("var_78", dtype = "int8", shape = (16,))#candidate|78|(16,)|var|int8
bop_79 = relay.greater(var_77.astype('bool'), relay.reshape(var_78.astype('bool'), relay.shape_of(var_77))) # shape=(16,)
bop_105 = relay.greater(bop_79.astype('bool'), relay.reshape(var_77.astype('bool'), relay.shape_of(bop_79))) # shape=(16,)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
var_117 = relay.var("var_117", dtype = "int64", shape = ())#candidate|117|()|var|int64
call_116 = relay.TupleGetItem(func_15_call(relay.reshape(var_117.astype('int64'), [])), 0)
call_118 = relay.TupleGetItem(func_17_call(relay.reshape(var_117.astype('int64'), [])), 0)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
call_123 = relay.TupleGetItem(func_15_call(relay.reshape(var_117.astype('int64'), [])), 0)
call_124 = relay.TupleGetItem(func_17_call(relay.reshape(var_117.astype('int64'), [])), 0)
output = relay.Tuple([bop_105,call_116,var_117,call_123,])
output2 = relay.Tuple([bop_105,call_118,var_117,call_124,])
func_132 = relay.Function([var_77,var_78,var_117,], output)
mod['func_132'] = func_132
mod = relay.transform.InferType()(mod)
mutated_mod['func_132'] = func_132
mutated_mod = relay.transform.InferType()(mutated_mod)
func_132_call = mutated_mod.get_global_var('func_132')
var_134 = relay.var("var_134", dtype = "int8", shape = (16,))#candidate|134|(16,)|var|int8
var_135 = relay.var("var_135", dtype = "int8", shape = (16,))#candidate|135|(16,)|var|int8
var_136 = relay.var("var_136", dtype = "int64", shape = ())#candidate|136|()|var|int64
call_133 = func_132_call(var_134,var_135,var_136,)
output = call_133
func_137 = relay.Function([var_134,var_135,var_136,], output)
mutated_mod['func_137'] = func_137
mutated_mod = relay.transform.InferType()(mutated_mod)
var_181 = relay.var("var_181", dtype = "float64", shape = (7,))#candidate|181|(7,)|var|float64
uop_182 = relay.log(var_181.astype('float64')) # shape=(7,)
output = relay.Tuple([uop_182,])
output2 = relay.Tuple([uop_182,])
func_186 = relay.Function([var_181,], output)
mod['func_186'] = func_186
mod = relay.transform.InferType()(mod)
var_187 = relay.var("var_187", dtype = "float64", shape = (7,))#candidate|187|(7,)|var|float64
output = func_186(var_187)
func_188 = relay.Function([var_187], output)
mutated_mod['func_188'] = func_188
mutated_mod = relay.transform.InferType()(mutated_mod)
var_234 = relay.var("var_234", dtype = "int16", shape = (2,))#candidate|234|(2,)|var|int16
var_235 = relay.var("var_235", dtype = "int16", shape = (2,))#candidate|235|(2,)|var|int16
bop_236 = relay.right_shift(var_234.astype('int16'), relay.reshape(var_235.astype('int16'), relay.shape_of(var_234))) # shape=(2,)
bop_239 = relay.floor_mod(var_234.astype('float32'), relay.reshape(bop_236.astype('float32'), relay.shape_of(var_234))) # shape=(2,)
bop_246 = relay.not_equal(bop_239.astype('bool'), relay.reshape(var_234.astype('bool'), relay.shape_of(bop_239))) # shape=(2,)
uop_256 = relay.sin(bop_236.astype('float32')) # shape=(2,)
bop_258 = relay.equal(uop_256.astype('bool'), relay.reshape(bop_239.astype('bool'), relay.shape_of(uop_256))) # shape=(2,)
var_261 = relay.var("var_261", dtype = "bool", shape = (2,))#candidate|261|(2,)|var|bool
bop_262 = relay.maximum(bop_258.astype('uint8'), relay.reshape(var_261.astype('uint8'), relay.shape_of(bop_258))) # shape=(2,)
bop_267 = relay.add(bop_258.astype('float64'), relay.reshape(bop_262.astype('float64'), relay.shape_of(bop_258))) # shape=(2,)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
const_273 = relay.const(-3, dtype = "int64")#candidate|273|()|const|int64
call_272 = relay.TupleGetItem(func_15_call(relay.reshape(const_273.astype('int64'), [])), 0)
call_274 = relay.TupleGetItem(func_17_call(relay.reshape(const_273.astype('int64'), [])), 0)
output = relay.Tuple([bop_246,bop_267,call_272,const_273,])
output2 = relay.Tuple([bop_246,bop_267,call_274,const_273,])
func_275 = relay.Function([var_234,var_235,var_261,], output)
mod['func_275'] = func_275
mod = relay.transform.InferType()(mod)
var_276 = relay.var("var_276", dtype = "int16", shape = (2,))#candidate|276|(2,)|var|int16
var_277 = relay.var("var_277", dtype = "int16", shape = (2,))#candidate|277|(2,)|var|int16
var_278 = relay.var("var_278", dtype = "bool", shape = (2,))#candidate|278|(2,)|var|bool
output = func_275(var_276,var_277,var_278,)
func_279 = relay.Function([var_276,var_277,var_278,], output)
mutated_mod['func_279'] = func_279
mutated_mod = relay.transform.InferType()(mutated_mod)
var_494 = relay.var("var_494", dtype = "float32", shape = (15,))#candidate|494|(15,)|var|float32
uop_495 = relay.atanh(var_494.astype('float32')) # shape=(15,)
bop_501 = relay.equal(uop_495.astype('bool'), relay.reshape(var_494.astype('bool'), relay.shape_of(uop_495))) # shape=(15,)
output = relay.Tuple([bop_501,])
output2 = relay.Tuple([bop_501,])
func_504 = relay.Function([var_494,], output)
mod['func_504'] = func_504
mod = relay.transform.InferType()(mod)
var_505 = relay.var("var_505", dtype = "float32", shape = (15,))#candidate|505|(15,)|var|float32
output = func_504(var_505)
func_506 = relay.Function([var_505], output)
mutated_mod['func_506'] = func_506
mutated_mod = relay.transform.InferType()(mutated_mod)
var_535 = relay.var("var_535", dtype = "float32", shape = (16, 7))#candidate|535|(16, 7)|var|float32
uop_536 = relay.cosh(var_535.astype('float32')) # shape=(16, 7)
func_186_call = mod.get_global_var('func_186')
func_188_call = mutated_mod.get_global_var('func_188')
var_543 = relay.var("var_543", dtype = "float64", shape = (7,))#candidate|543|(7,)|var|float64
call_542 = relay.TupleGetItem(func_186_call(relay.reshape(var_543.astype('float64'), [7,])), 0)
call_544 = relay.TupleGetItem(func_188_call(relay.reshape(var_543.astype('float64'), [7,])), 0)
output = relay.Tuple([uop_536,call_542,var_543,])
output2 = relay.Tuple([uop_536,call_544,var_543,])
func_548 = relay.Function([var_535,var_543,], output)
mod['func_548'] = func_548
mod = relay.transform.InferType()(mod)
mutated_mod['func_548'] = func_548
mutated_mod = relay.transform.InferType()(mutated_mod)
func_548_call = mutated_mod.get_global_var('func_548')
var_550 = relay.var("var_550", dtype = "float32", shape = (16, 7))#candidate|550|(16, 7)|var|float32
var_551 = relay.var("var_551", dtype = "float64", shape = (7,))#candidate|551|(7,)|var|float64
call_549 = func_548_call(var_550,var_551,)
output = call_549
func_552 = relay.Function([var_550,var_551,], output)
mutated_mod['func_552'] = func_552
mutated_mod = relay.transform.InferType()(mutated_mod)
const_575 = relay.const([[[-2.086873,1.373065,-5.499282,5.972452,7.969102,-0.262049,5.600078,-4.759153,4.089096,-2.628152,-7.245807,1.022005,4.178329],[8.033498,-5.148919,-9.930491,0.499596,-5.811792,-3.112025,-9.800993,-9.952414,-5.828818,-3.588618,7.549620,-5.071641,8.622487],[-4.620074,2.704658,9.707222,-7.573173,5.896136,5.450941,-1.205147,8.644453,4.447153,-9.681240,-6.979026,4.277494,7.134178],[-2.634444,1.989598,-9.755852,-1.338406,6.709148,2.925715,0.317885,9.222555,7.318046,-7.374625,-0.968745,-9.927706,1.962859],[3.135922,7.951125,1.604482,5.777373,-4.799873,-2.706884,5.199419,-1.041084,0.429793,-8.589259,-0.564551,-7.053201,-3.353633]],[[-7.969382,9.374120,9.069825,7.310945,7.426191,-0.711643,5.852881,1.097781,-6.061427,2.942148,1.701219,-0.972677,2.800253],[8.896162,-3.215137,-1.666595,3.747922,-7.744623,9.005888,1.544133,1.198687,8.397314,-6.626926,-8.774634,7.082566,-8.148940],[6.794571,-1.123663,0.759054,-9.102761,3.072100,4.081538,6.162729,0.954467,0.649215,0.839875,5.069084,3.087406,5.301511],[-4.676145,7.948702,2.876798,3.432736,-8.940981,-6.283114,-3.792537,7.701010,-4.744919,-8.358677,9.018236,-3.000244,-2.277514],[0.930868,-2.632618,3.959691,-9.982719,3.326492,8.162363,5.048713,1.485083,8.871446,-3.346580,5.674028,3.663437,-6.580794]],[[1.003508,3.231477,-7.924872,3.353048,-7.998456,9.386533,-7.485060,-5.510698,-4.641394,4.974938,4.029348,4.837089,9.921660],[-1.431958,-2.716325,-1.189130,-3.470669,9.574084,-8.366551,2.863574,-7.377201,3.884915,4.389057,-8.456088,-0.888566,0.951147],[-4.972860,5.491682,-5.999951,-4.027686,-6.347684,-5.752446,3.513315,-1.646471,1.341595,-1.274435,-2.743694,-2.023387,0.697226],[3.776229,-9.525555,-5.805638,-2.030053,-8.144215,5.748884,7.826039,4.028424,-3.911086,-2.459387,8.738076,-9.117563,8.235226],[7.236670,-6.491204,-8.672213,-2.856188,4.633998,-7.409800,-9.696554,-2.193374,4.078968,-7.981903,0.629319,4.098784,2.407569]],[[7.053072,-6.574644,-6.424661,-6.298301,0.170494,6.937520,0.764773,-0.359701,-5.483072,6.901952,-5.234903,-7.457339,9.190212],[-2.679159,1.576317,9.606272,5.092426,5.467917,-9.500857,5.775374,0.586997,3.097733,1.720487,-8.035739,2.032024,-1.442733],[-5.105460,-2.448081,-0.536884,3.379199,-0.245473,9.869830,-6.619560,5.963498,-7.185387,-7.394350,8.742034,6.000138,2.191343],[-0.794698,2.832595,9.435504,-8.720106,6.102481,-2.650353,-7.230960,-9.121117,-8.618868,1.467382,9.988662,-3.762562,4.220156],[4.324773,-4.298538,-6.449305,0.698277,6.179898,0.486805,-5.469932,6.993177,7.936784,5.006562,8.512045,8.473341,-2.441154]],[[-4.122210,-4.926632,-7.550404,-8.405884,1.075067,-6.314303,-5.990234,8.915337,3.835064,4.695757,-0.611569,-2.533705,-5.785692],[2.069222,-9.984678,-7.678776,7.729284,-6.151160,-9.344840,-7.880267,8.415582,7.976113,6.667627,-9.374456,-0.923800,8.185499],[-8.757934,6.172649,0.201495,-5.857213,-8.790171,-1.292506,-4.612449,1.833514,-4.405912,-4.396023,-5.816553,1.816056,-6.002038],[-9.824570,9.572298,6.310076,-8.837823,-1.423382,2.548331,4.464277,2.170141,4.620513,0.067635,5.645141,7.227873,6.479736],[3.069667,5.711072,-5.639927,-0.025536,2.975168,-7.301098,-8.996236,2.666221,5.817015,4.973044,-3.176564,-9.572024,-2.972304]],[[7.474208,6.791181,0.728655,-7.638395,-0.299292,2.165011,-8.437316,-5.291178,-0.959499,-0.677988,1.693779,-9.977042,-0.496752],[5.881044,-8.649427,-0.731365,7.770899,2.712847,-6.949452,5.776516,-2.364166,-0.984955,6.592513,8.378169,7.160741,8.482654],[-3.916729,-6.524774,3.422777,3.759174,1.751146,8.964995,-9.342156,9.984712,-9.343098,4.362843,8.261005,-9.124152,0.266027],[-6.901066,7.467289,3.317028,-3.528345,-6.506458,1.694281,9.868550,-9.018659,-2.635795,0.773641,7.252031,-8.960701,0.983098],[-4.673561,6.091206,7.384871,7.319159,3.221679,-8.514107,-0.891121,4.983489,1.306926,-1.671160,-4.946182,-6.088707,-6.368470]]], dtype = "float32")#candidate|575|(6, 5, 13)|const|float32
var_576 = relay.var("var_576", dtype = "float32", shape = (6, 5, 13))#candidate|576|(6, 5, 13)|var|float32
bop_577 = relay.floor_divide(const_575.astype('float32'), relay.reshape(var_576.astype('float32'), relay.shape_of(const_575))) # shape=(6, 5, 13)
bop_583 = relay.divide(bop_577.astype('float64'), relay.reshape(const_575.astype('float64'), relay.shape_of(bop_577))) # shape=(6, 5, 13)
func_275_call = mod.get_global_var('func_275')
func_279_call = mutated_mod.get_global_var('func_279')
var_587 = relay.var("var_587", dtype = "int16", shape = (2,))#candidate|587|(2,)|var|int16
call_586 = relay.TupleGetItem(func_275_call(relay.reshape(var_587.astype('int16'), [2,]), relay.reshape(var_587.astype('int16'), [2,]), relay.reshape(var_587.astype('bool'), [2,]), ), 1)
call_588 = relay.TupleGetItem(func_279_call(relay.reshape(var_587.astype('int16'), [2,]), relay.reshape(var_587.astype('int16'), [2,]), relay.reshape(var_587.astype('bool'), [2,]), ), 1)
func_275_call = mod.get_global_var('func_275')
func_279_call = mutated_mod.get_global_var('func_279')
call_591 = relay.TupleGetItem(func_275_call(relay.reshape(var_587.astype('int16'), [2,]), relay.reshape(call_586.astype('int16'), [2,]), relay.reshape(var_587.astype('bool'), [2,]), ), 0)
call_592 = relay.TupleGetItem(func_279_call(relay.reshape(var_587.astype('int16'), [2,]), relay.reshape(call_586.astype('int16'), [2,]), relay.reshape(var_587.astype('bool'), [2,]), ), 0)
bop_595 = relay.maximum(bop_583.astype('float32'), relay.reshape(bop_577.astype('float32'), relay.shape_of(bop_583))) # shape=(6, 5, 13)
var_599 = relay.var("var_599", dtype = "float32", shape = (6, 5, 13))#candidate|599|(6, 5, 13)|var|float32
bop_600 = relay.left_shift(bop_577.astype('uint32'), relay.reshape(var_599.astype('uint32'), relay.shape_of(bop_577))) # shape=(6, 5, 13)
func_504_call = mod.get_global_var('func_504')
func_506_call = mutated_mod.get_global_var('func_506')
var_604 = relay.var("var_604", dtype = "float32", shape = (15,))#candidate|604|(15,)|var|float32
call_603 = relay.TupleGetItem(func_504_call(relay.reshape(var_604.astype('float32'), [15,])), 0)
call_605 = relay.TupleGetItem(func_506_call(relay.reshape(var_604.astype('float32'), [15,])), 0)
uop_606 = relay.sinh(bop_583.astype('float64')) # shape=(6, 5, 13)
uop_616 = relay.atan(uop_606.astype('float64')) # shape=(6, 5, 13)
var_618 = relay.var("var_618", dtype = "float64", shape = (6, 5, 13))#candidate|618|(6, 5, 13)|var|float64
bop_619 = relay.bitwise_or(uop_616.astype('int8'), relay.reshape(var_618.astype('int8'), relay.shape_of(uop_616))) # shape=(6, 5, 13)
output = relay.Tuple([call_586,var_587,call_591,bop_595,bop_600,call_603,var_604,bop_619,])
output2 = relay.Tuple([call_588,var_587,call_592,bop_595,bop_600,call_605,var_604,bop_619,])
func_623 = relay.Function([var_576,var_587,var_599,var_604,var_618,], output)
mod['func_623'] = func_623
mod = relay.transform.InferType()(mod)
var_624 = relay.var("var_624", dtype = "float32", shape = (6, 5, 13))#candidate|624|(6, 5, 13)|var|float32
var_625 = relay.var("var_625", dtype = "int16", shape = (2,))#candidate|625|(2,)|var|int16
var_626 = relay.var("var_626", dtype = "float32", shape = (6, 5, 13))#candidate|626|(6, 5, 13)|var|float32
var_627 = relay.var("var_627", dtype = "float32", shape = (15,))#candidate|627|(15,)|var|float32
var_628 = relay.var("var_628", dtype = "float64", shape = (6, 5, 13))#candidate|628|(6, 5, 13)|var|float64
output = func_623(var_624,var_625,var_626,var_627,var_628,)
func_629 = relay.Function([var_624,var_625,var_626,var_627,var_628,], output)
mutated_mod['func_629'] = func_629
mutated_mod = relay.transform.InferType()(mutated_mod)
var_730 = relay.var("var_730", dtype = "float32", shape = (7, 2))#candidate|730|(7, 2)|var|float32
var_731 = relay.var("var_731", dtype = "float32", shape = (7, 2))#candidate|731|(7, 2)|var|float32
bop_732 = relay.floor_mod(var_730.astype('float32'), relay.reshape(var_731.astype('float32'), relay.shape_of(var_730))) # shape=(7, 2)
uop_735 = relay.sigmoid(bop_732.astype('float64')) # shape=(7, 2)
func_548_call = mod.get_global_var('func_548')
func_552_call = mutated_mod.get_global_var('func_552')
const_744 = relay.const([4.497122,0.649193,-4.034676,5.448389,-7.249056,0.096365,-8.981224,-8.774669,6.508500,0.829800,-7.069346,-9.614255,-3.865182,-9.777697,-6.206112,-3.729871,-1.634357,6.027434,-2.148036,-6.363737,-8.270279,4.313907,7.947612,-7.337326,8.652263,-4.414310,3.029890,4.845197,9.916275,-0.899006,-9.345979,3.387443,-7.909554,7.980899,6.556503,2.566187,9.323920,9.013075,-5.209455,-6.656004,3.172760,-4.164311,5.802202,-4.781428,3.148358,9.504477,8.659354,-8.127949,8.333454,3.327764,9.881222,4.995043,9.446728,2.389834,-6.923830,6.397701,-4.212670,6.071672,-7.859785,-6.309008,-8.477588,8.569995,4.983630,1.674682,7.355775,8.756792,6.509012,8.207997,9.196160,2.219433,-0.734850,2.518874,-2.937240,6.296213,2.883351,-3.606784,2.917485,8.380556,4.395121,5.473799,6.334695,1.641427,1.633534,-1.553422,6.587644,2.157441,-2.233197,-1.705001,4.380627,0.615191,-2.688951,-0.027581,9.348389,-0.111954,-6.100418,6.469966,-7.868835,-7.771222,-0.528776,6.147522,-4.849122,4.503884,3.321871,7.806329,2.328715,0.232951,-5.725198,6.039442,-8.277182,-4.997940,-3.366176,-0.914221], dtype = "float32")#candidate|744|(112,)|const|float32
var_745 = relay.var("var_745", dtype = "float64", shape = (7,))#candidate|745|(7,)|var|float64
call_743 = relay.TupleGetItem(func_548_call(relay.reshape(const_744.astype('float32'), [16, 7]), relay.reshape(var_745.astype('float64'), [7,]), ), 1)
call_746 = relay.TupleGetItem(func_552_call(relay.reshape(const_744.astype('float32'), [16, 7]), relay.reshape(var_745.astype('float64'), [7,]), ), 1)
var_747 = relay.var("var_747", dtype = "float64", shape = (7, 2))#candidate|747|(7, 2)|var|float64
bop_748 = relay.maximum(uop_735.astype('uint8'), relay.reshape(var_747.astype('uint8'), relay.shape_of(uop_735))) # shape=(7, 2)
bop_752 = relay.logical_or(uop_735.astype('bool'), relay.reshape(var_731.astype('bool'), relay.shape_of(uop_735))) # shape=(7, 2)
output = relay.Tuple([call_743,const_744,var_745,bop_748,bop_752,])
output2 = relay.Tuple([call_746,const_744,var_745,bop_748,bop_752,])
func_755 = relay.Function([var_730,var_731,var_745,var_747,], output)
mod['func_755'] = func_755
mod = relay.transform.InferType()(mod)
var_756 = relay.var("var_756", dtype = "float32", shape = (7, 2))#candidate|756|(7, 2)|var|float32
var_757 = relay.var("var_757", dtype = "float32", shape = (7, 2))#candidate|757|(7, 2)|var|float32
var_758 = relay.var("var_758", dtype = "float64", shape = (7,))#candidate|758|(7,)|var|float64
var_759 = relay.var("var_759", dtype = "float64", shape = (7, 2))#candidate|759|(7, 2)|var|float64
output = func_755(var_756,var_757,var_758,var_759,)
func_760 = relay.Function([var_756,var_757,var_758,var_759,], output)
mutated_mod['func_760'] = func_760
mutated_mod = relay.transform.InferType()(mutated_mod)
const_793 = relay.const([[0.773406,9.520527,-1.736440,-3.746990],[0.512858,2.382423,-7.073468,-1.645920],[7.126268,8.759310,-1.063594,3.330250],[-8.404872,4.343869,9.146102,-9.910636],[2.847541,-2.095668,9.762108,4.246138]], dtype = "float32")#candidate|793|(5, 4)|const|float32
uop_794 = relay.asinh(const_793.astype('float32')) # shape=(5, 4)
func_548_call = mod.get_global_var('func_548')
func_552_call = mutated_mod.get_global_var('func_552')
var_797 = relay.var("var_797", dtype = "float32", shape = (112,))#candidate|797|(112,)|var|float32
var_798 = relay.var("var_798", dtype = "float64", shape = (7,))#candidate|798|(7,)|var|float64
call_796 = relay.TupleGetItem(func_548_call(relay.reshape(var_797.astype('float32'), [16, 7]), relay.reshape(var_798.astype('float64'), [7,]), ), 0)
call_799 = relay.TupleGetItem(func_552_call(relay.reshape(var_797.astype('float32'), [16, 7]), relay.reshape(var_798.astype('float64'), [7,]), ), 0)
output = relay.Tuple([uop_794,call_796,var_797,var_798,])
output2 = relay.Tuple([uop_794,call_799,var_797,var_798,])
func_800 = relay.Function([var_797,var_798,], output)
mod['func_800'] = func_800
mod = relay.transform.InferType()(mod)
mutated_mod['func_800'] = func_800
mutated_mod = relay.transform.InferType()(mutated_mod)
func_800_call = mutated_mod.get_global_var('func_800')
var_802 = relay.var("var_802", dtype = "float32", shape = (112,))#candidate|802|(112,)|var|float32
var_803 = relay.var("var_803", dtype = "float64", shape = (7,))#candidate|803|(7,)|var|float64
call_801 = func_800_call(var_802,var_803,)
output = call_801
func_804 = relay.Function([var_802,var_803,], output)
mutated_mod['func_804'] = func_804
mutated_mod = relay.transform.InferType()(mutated_mod)
var_873 = relay.var("var_873", dtype = "uint32", shape = ())#candidate|873|()|var|uint32
const_874 = relay.const([[[-3,-4,8,-4,-8],[-5,-1,-7,-1,5],[-6,-8,10,-1,1],[4,10,10,8,7],[9,7,4,-8,6],[4,8,2,-6,-1],[-2,-10,5,-10,-5],[8,-1,-6,6,-2]],[[9,-6,-6,5,1],[8,10,-8,9,7],[7,8,10,-4,7],[-3,-2,3,7,-9],[-3,6,6,-10,1],[10,5,-5,1,-3],[7,-7,5,4,-8],[-2,1,-6,-1,-1]],[[4,-8,9,-3,-7],[3,10,-1,-1,-7],[-10,-2,9,8,-2],[2,-10,4,6,-4],[-6,4,-8,-3,2],[10,2,-2,7,1],[-9,-7,3,-2,1],[-1,4,4,-10,3]]], dtype = "uint32")#candidate|874|(3, 8, 5)|const|uint32
bop_875 = relay.less_equal(var_873.astype('bool'), const_874.astype('bool')) # shape=(3, 8, 5)
func_15_call = mod.get_global_var('func_15')
func_17_call = mutated_mod.get_global_var('func_17')
call_883 = relay.TupleGetItem(func_15_call(relay.reshape(var_873.astype('int64'), [])), 0)
call_884 = relay.TupleGetItem(func_17_call(relay.reshape(var_873.astype('int64'), [])), 0)
func_800_call = mod.get_global_var('func_800')
func_804_call = mutated_mod.get_global_var('func_804')
const_896 = relay.const([-4.266757,2.303947,-9.686751,4.600902,9.759487,-0.636859,-9.094416,-8.262576,-5.810519,1.959136,9.236877,9.892301,-4.452807,-9.846191,-0.108937,9.330212,0.157574,3.928281,-0.179283,8.906020,-1.865218,6.035731,-5.902281,6.440445,2.776682,6.625655,8.433911,-4.231931,5.310536,2.619638,-9.731396,-9.411667,6.532246,5.588510,-5.765307,-0.407367,-1.935873,0.145056,7.595718,6.889555,-7.295551,0.780623,9.240471,7.451131,0.154405,-9.162051,-0.279530,8.400407,3.302133,2.965259,2.928227,-1.414083,0.430684,-3.655040,-2.472833,2.889638,-1.377305,-6.713376,-0.265355,-3.913651,-8.310129,5.658743,4.865519,1.387609,9.755514,7.284060,-8.877549,-0.938695,5.444563,-0.939280,-0.261543,-0.889480,-6.559173,9.435175,4.206612,-9.221511,-8.452813,-1.753879,-4.296478,-0.308887,-6.578832,3.964461,9.856310,1.772258,-4.749153,7.759282,0.711852,-8.759708,-8.697985,3.737151,8.195948,-9.645735,5.623188,-3.184575,-3.863569,-5.014798,-6.218363,8.270275,8.221987,-6.799791,-1.488438,1.169756,-5.760705,4.217620,6.420057,-5.380813,5.848890,-1.441114,9.228903,-5.918873,-1.554409,-9.174539], dtype = "float32")#candidate|896|(112,)|const|float32
var_897 = relay.var("var_897", dtype = "float64", shape = (7,))#candidate|897|(7,)|var|float64
call_895 = relay.TupleGetItem(func_800_call(relay.reshape(const_896.astype('float32'), [112,]), relay.reshape(var_897.astype('float64'), [7,]), ), 2)
call_898 = relay.TupleGetItem(func_804_call(relay.reshape(const_896.astype('float32'), [112,]), relay.reshape(var_897.astype('float64'), [7,]), ), 2)
output = relay.Tuple([bop_875,call_883,call_895,const_896,var_897,])
output2 = relay.Tuple([bop_875,call_884,call_898,const_896,var_897,])
func_901 = relay.Function([var_873,var_897,], output)
mod['func_901'] = func_901
mod = relay.transform.InferType()(mod)
mutated_mod['func_901'] = func_901
mutated_mod = relay.transform.InferType()(mutated_mod)
func_901_call = mutated_mod.get_global_var('func_901')
var_903 = relay.var("var_903", dtype = "uint32", shape = ())#candidate|903|()|var|uint32
var_904 = relay.var("var_904", dtype = "float64", shape = (7,))#candidate|904|(7,)|var|float64
call_902 = func_901_call(var_903,var_904,)
output = call_902
func_905 = relay.Function([var_903,var_904,], output)
mutated_mod['func_905'] = func_905
mutated_mod = relay.transform.InferType()(mutated_mod)
var_946 = relay.var("var_946", dtype = "float64", shape = (9, 6))#candidate|946|(9, 6)|var|float64
uop_947 = relay.cosh(var_946.astype('float64')) # shape=(9, 6)
bop_949 = relay.bitwise_or(var_946.astype('int8'), relay.reshape(uop_947.astype('int8'), relay.shape_of(var_946))) # shape=(9, 6)
bop_952 = relay.power(bop_949.astype('float64'), relay.reshape(uop_947.astype('float64'), relay.shape_of(bop_949))) # shape=(9, 6)
bop_957 = relay.right_shift(var_946.astype('int8'), relay.reshape(bop_949.astype('int8'), relay.shape_of(var_946))) # shape=(9, 6)
bop_963 = relay.bitwise_and(bop_957.astype('int8'), relay.reshape(bop_949.astype('int8'), relay.shape_of(bop_957))) # shape=(9, 6)
output = relay.Tuple([bop_952,bop_963,])
output2 = relay.Tuple([bop_952,bop_963,])
func_972 = relay.Function([var_946,], output)
mod['func_972'] = func_972
mod = relay.transform.InferType()(mod)
var_973 = relay.var("var_973", dtype = "float64", shape = (9, 6))#candidate|973|(9, 6)|var|float64
output = func_972(var_973)
func_974 = relay.Function([var_973], output)
mutated_mod['func_974'] = func_974
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1004 = relay.const([9,-6,4,-1,-1,-6,7,8,-10,-7,1,8,7,-5,-8,2], dtype = "int64")#candidate|1004|(16,)|const|int64
var_1005 = relay.var("var_1005", dtype = "int64", shape = (16,))#candidate|1005|(16,)|var|int64
bop_1006 = relay.multiply(const_1004.astype('int64'), relay.reshape(var_1005.astype('int64'), relay.shape_of(const_1004))) # shape=(16,)
output = relay.Tuple([bop_1006,])
output2 = relay.Tuple([bop_1006,])
func_1012 = relay.Function([var_1005,], output)
mod['func_1012'] = func_1012
mod = relay.transform.InferType()(mod)
mutated_mod['func_1012'] = func_1012
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1013 = relay.var("var_1013", dtype = "int64", shape = (16,))#candidate|1013|(16,)|var|int64
func_1012_call = mutated_mod.get_global_var('func_1012')
call_1014 = func_1012_call(var_1013)
output = call_1014
func_1015 = relay.Function([var_1013], output)
mutated_mod['func_1015'] = func_1015
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1022 = relay.var("var_1022", dtype = "float64", shape = (14,))#candidate|1022|(14,)|var|float64
const_1023 = relay.const([-8.939882,-3.346726,2.103244,8.781032,7.189381,-9.486903,-5.113130,1.262889,-3.220625,-8.910090,-0.227708,-4.150113,-9.013966,-1.567383], dtype = "float64")#candidate|1023|(14,)|const|float64
bop_1024 = relay.minimum(var_1022.astype('float64'), relay.reshape(const_1023.astype('float64'), relay.shape_of(var_1022))) # shape=(14,)
uop_1031 = relay.rsqrt(bop_1024.astype('float32')) # shape=(14,)
bop_1033 = relay.maximum(uop_1031.astype('uint8'), relay.reshape(const_1023.astype('uint8'), relay.shape_of(uop_1031))) # shape=(14,)
bop_1041 = relay.left_shift(const_1023.astype('int32'), relay.reshape(uop_1031.astype('int32'), relay.shape_of(const_1023))) # shape=(14,)
output = relay.Tuple([bop_1033,bop_1041,])
output2 = relay.Tuple([bop_1033,bop_1041,])
func_1063 = relay.Function([var_1022,], output)
mod['func_1063'] = func_1063
mod = relay.transform.InferType()(mod)
mutated_mod['func_1063'] = func_1063
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1064 = relay.var("var_1064", dtype = "float64", shape = (14,))#candidate|1064|(14,)|var|float64
func_1063_call = mutated_mod.get_global_var('func_1063')
call_1065 = func_1063_call(var_1064)
output = call_1065
func_1066 = relay.Function([var_1064], output)
mutated_mod['func_1066'] = func_1066
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1086 = relay.var("var_1086", dtype = "uint32", shape = (2,))#candidate|1086|(2,)|var|uint32
var_1087 = relay.var("var_1087", dtype = "uint32", shape = (2,))#candidate|1087|(2,)|var|uint32
bop_1088 = relay.bitwise_xor(var_1086.astype('uint32'), relay.reshape(var_1087.astype('uint32'), relay.shape_of(var_1086))) # shape=(2,)
var_1093 = relay.var("var_1093", dtype = "uint32", shape = (2,))#candidate|1093|(2,)|var|uint32
bop_1094 = relay.greater(bop_1088.astype('bool'), relay.reshape(var_1093.astype('bool'), relay.shape_of(bop_1088))) # shape=(2,)
func_901_call = mod.get_global_var('func_901')
func_905_call = mutated_mod.get_global_var('func_905')
const_1102 = relay.const(-1, dtype = "uint32")#candidate|1102|()|const|uint32
var_1103 = relay.var("var_1103", dtype = "float64", shape = (7,))#candidate|1103|(7,)|var|float64
call_1101 = relay.TupleGetItem(func_901_call(relay.reshape(const_1102.astype('uint32'), []), relay.reshape(var_1103.astype('float64'), [7,]), ), 3)
call_1104 = relay.TupleGetItem(func_905_call(relay.reshape(const_1102.astype('uint32'), []), relay.reshape(var_1103.astype('float64'), [7,]), ), 3)
bop_1105 = relay.bitwise_or(bop_1094.astype('int32'), relay.reshape(var_1086.astype('int32'), relay.shape_of(bop_1094))) # shape=(2,)
func_1063_call = mod.get_global_var('func_1063')
func_1066_call = mutated_mod.get_global_var('func_1066')
var_1114 = relay.var("var_1114", dtype = "float64", shape = (14,))#candidate|1114|(14,)|var|float64
call_1113 = relay.TupleGetItem(func_1063_call(relay.reshape(var_1114.astype('float64'), [14,])), 1)
call_1115 = relay.TupleGetItem(func_1066_call(relay.reshape(var_1114.astype('float64'), [14,])), 1)
uop_1120 = relay.sigmoid(call_1101.astype('float32')) # shape=(112,)
uop_1122 = relay.sigmoid(call_1104.astype('float32')) # shape=(112,)
uop_1124 = relay.atanh(uop_1120.astype('float32')) # shape=(112,)
uop_1126 = relay.atanh(uop_1122.astype('float32')) # shape=(112,)
bop_1128 = relay.mod(uop_1124.astype('float64'), relay.reshape(call_1101.astype('float64'), relay.shape_of(uop_1124))) # shape=(112,)
bop_1131 = relay.mod(uop_1126.astype('float64'), relay.reshape(call_1104.astype('float64'), relay.shape_of(uop_1126))) # shape=(112,)
bop_1132 = relay.less_equal(uop_1124.astype('bool'), relay.reshape(bop_1128.astype('bool'), relay.shape_of(uop_1124))) # shape=(112,)
bop_1135 = relay.less_equal(uop_1126.astype('bool'), relay.reshape(bop_1131.astype('bool'), relay.shape_of(uop_1126))) # shape=(112,)
bop_1138 = relay.logical_or(var_1103.astype('bool'), const_1102.astype('bool')) # shape=(7,)
var_1141 = relay.var("var_1141", dtype = "float32", shape = (112,))#candidate|1141|(112,)|var|float32
bop_1142 = relay.logical_and(uop_1120.astype('bool'), relay.reshape(var_1141.astype('bool'), relay.shape_of(uop_1120))) # shape=(112,)
bop_1145 = relay.logical_and(uop_1122.astype('bool'), relay.reshape(var_1141.astype('bool'), relay.shape_of(uop_1122))) # shape=(112,)
bop_1150 = relay.less_equal(bop_1128.astype('bool'), relay.reshape(bop_1132.astype('bool'), relay.shape_of(bop_1128))) # shape=(112,)
bop_1153 = relay.less_equal(bop_1131.astype('bool'), relay.reshape(bop_1135.astype('bool'), relay.shape_of(bop_1131))) # shape=(112,)
const_1157 = relay.const([True,False,False,True,False,False,False,True,False,False,False,True,False,False,False,True,True,True,False,False,False,False,True,False,False,False,True,True,True,True,True,False,True,False,True,False,False,True,True,True,True,True,False,True,True,True,False,True,False,False,True,False,True,False,True,True,False,False,False,True,True,False,False,True,False,True,True,False,True,False,True,False,True,False,False,True,True,False,False,True,True,True,False,False,False,True,True,False,True,True,False,True,True,False,False,True,True,True,False,False,True,True,True,False,True,True,True,False,True,True,False,False], dtype = "bool")#candidate|1157|(112,)|const|bool
bop_1158 = relay.floor_mod(bop_1142.astype('float32'), relay.reshape(const_1157.astype('float32'), relay.shape_of(bop_1142))) # shape=(112,)
bop_1161 = relay.floor_mod(bop_1145.astype('float32'), relay.reshape(const_1157.astype('float32'), relay.shape_of(bop_1145))) # shape=(112,)
output = relay.Tuple([bop_1105,call_1113,var_1114,bop_1138,bop_1150,bop_1158,])
output2 = relay.Tuple([bop_1105,call_1115,var_1114,bop_1138,bop_1153,bop_1161,])
func_1162 = relay.Function([var_1086,var_1087,var_1093,var_1103,var_1114,var_1141,], output)
mod['func_1162'] = func_1162
mod = relay.transform.InferType()(mod)
mutated_mod['func_1162'] = func_1162
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1162_call = mutated_mod.get_global_var('func_1162')
var_1164 = relay.var("var_1164", dtype = "uint32", shape = (2,))#candidate|1164|(2,)|var|uint32
var_1165 = relay.var("var_1165", dtype = "uint32", shape = (2,))#candidate|1165|(2,)|var|uint32
var_1166 = relay.var("var_1166", dtype = "uint32", shape = (2,))#candidate|1166|(2,)|var|uint32
var_1167 = relay.var("var_1167", dtype = "float64", shape = (7,))#candidate|1167|(7,)|var|float64
var_1168 = relay.var("var_1168", dtype = "float64", shape = (14,))#candidate|1168|(14,)|var|float64
var_1169 = relay.var("var_1169", dtype = "float32", shape = (112,))#candidate|1169|(112,)|var|float32
call_1163 = func_1162_call(var_1164,var_1165,var_1166,var_1167,var_1168,var_1169,)
output = call_1163
func_1170 = relay.Function([var_1164,var_1165,var_1166,var_1167,var_1168,var_1169,], output)
mutated_mod['func_1170'] = func_1170
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1220 = relay.var("var_1220", dtype = "float32", shape = (7, 3))#candidate|1220|(7, 3)|var|float32
var_1221 = relay.var("var_1221", dtype = "float32", shape = (7, 3))#candidate|1221|(7, 3)|var|float32
bop_1222 = relay.less(var_1220.astype('bool'), relay.reshape(var_1221.astype('bool'), relay.shape_of(var_1220))) # shape=(7, 3)
output = relay.Tuple([bop_1222,])
output2 = relay.Tuple([bop_1222,])
func_1228 = relay.Function([var_1220,var_1221,], output)
mod['func_1228'] = func_1228
mod = relay.transform.InferType()(mod)
mutated_mod['func_1228'] = func_1228
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1228_call = mutated_mod.get_global_var('func_1228')
var_1230 = relay.var("var_1230", dtype = "float32", shape = (7, 3))#candidate|1230|(7, 3)|var|float32
var_1231 = relay.var("var_1231", dtype = "float32", shape = (7, 3))#candidate|1231|(7, 3)|var|float32
call_1229 = func_1228_call(var_1230,var_1231,)
output = call_1229
func_1232 = relay.Function([var_1230,var_1231,], output)
mutated_mod['func_1232'] = func_1232
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1305 = relay.const([[[-4.386828],[-2.279440],[4.574859],[2.168057],[-5.115521],[-9.310932],[-1.887902]],[[7.186727],[-3.094578],[8.529114],[0.517634],[8.396159],[9.259711],[-0.613373]],[[-4.402584],[9.263952],[4.485605],[-9.799780],[4.473807],[-0.695115],[7.359519]],[[4.850289],[-5.086573],[-8.015905],[-8.672957],[-4.350390],[7.376199],[0.338837]]], dtype = "float64")#candidate|1305|(4, 7, 1)|const|float64
uop_1306 = relay.asinh(const_1305.astype('float64')) # shape=(4, 7, 1)
func_132_call = mod.get_global_var('func_132')
func_137_call = mutated_mod.get_global_var('func_137')
const_1309 = relay.const([-6,6,9,9,9,-6,-10,-9,-3,10,7,-4,-5,5,1,10], dtype = "int8")#candidate|1309|(16,)|const|int8
const_1310 = relay.const(-3, dtype = "int64")#candidate|1310|()|const|int64
call_1308 = relay.TupleGetItem(func_132_call(relay.reshape(const_1309.astype('int8'), [16,]), relay.reshape(const_1309.astype('int8'), [16,]), relay.reshape(const_1310.astype('int64'), []), ), 3)
call_1311 = relay.TupleGetItem(func_137_call(relay.reshape(const_1309.astype('int8'), [16,]), relay.reshape(const_1309.astype('int8'), [16,]), relay.reshape(const_1310.astype('int64'), []), ), 3)
func_901_call = mod.get_global_var('func_901')
func_905_call = mutated_mod.get_global_var('func_905')
const_1318 = relay.const([4.471007,0.294624,-8.543443,-1.049130,-6.625204,3.823801,-2.680857], dtype = "float64")#candidate|1318|(7,)|const|float64
call_1317 = relay.TupleGetItem(func_901_call(relay.reshape(const_1310.astype('uint32'), []), relay.reshape(const_1318.astype('float64'), [7,]), ), 4)
call_1319 = relay.TupleGetItem(func_905_call(relay.reshape(const_1310.astype('uint32'), []), relay.reshape(const_1318.astype('float64'), [7,]), ), 4)
output = relay.Tuple([uop_1306,call_1308,const_1309,const_1310,call_1317,const_1318,])
output2 = relay.Tuple([uop_1306,call_1311,const_1309,const_1310,call_1319,const_1318,])
func_1321 = relay.Function([], output)
mod['func_1321'] = func_1321
mod = relay.transform.InferType()(mod)
mutated_mod['func_1321'] = func_1321
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1321_call = mutated_mod.get_global_var('func_1321')
call_1322 = func_1321_call()
output = call_1322
func_1323 = relay.Function([], output)
mutated_mod['func_1323'] = func_1323
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1321_call = mod.get_global_var('func_1321')
func_1323_call = mutated_mod.get_global_var('func_1323')
call_1338 = relay.TupleGetItem(func_1321_call(), 4)
call_1339 = relay.TupleGetItem(func_1323_call(), 4)
uop_1344 = relay.rsqrt(call_1338.astype('float64')) # shape=(7,)
uop_1346 = relay.rsqrt(call_1339.astype('float64')) # shape=(7,)
func_72_call = mod.get_global_var('func_72')
func_75_call = mutated_mod.get_global_var('func_75')
var_1350 = relay.var("var_1350", dtype = "float64", shape = (16, 1))#candidate|1350|(16, 1)|var|float64
call_1349 = relay.TupleGetItem(func_72_call(relay.reshape(var_1350.astype('float64'), [16,]), relay.reshape(var_1350.astype('float64'), [16,]), ), 6)
call_1351 = relay.TupleGetItem(func_75_call(relay.reshape(var_1350.astype('float64'), [16,]), relay.reshape(var_1350.astype('float64'), [16,]), ), 6)
func_623_call = mod.get_global_var('func_623')
func_629_call = mutated_mod.get_global_var('func_629')
var_1356 = relay.var("var_1356", dtype = "float32", shape = (1, 390))#candidate|1356|(1, 390)|var|float32
const_1357 = relay.const([-3,1], dtype = "int16")#candidate|1357|(2,)|const|int16
var_1358 = relay.var("var_1358", dtype = "float32", shape = (1, 15))#candidate|1358|(1, 15)|var|float32
call_1355 = relay.TupleGetItem(func_623_call(relay.reshape(var_1356.astype('float32'), [6, 5, 13]), relay.reshape(const_1357.astype('int16'), [2,]), relay.reshape(var_1356.astype('float32'), [6, 5, 13]), relay.reshape(var_1358.astype('float32'), [15,]), relay.reshape(var_1356.astype('float64'), [6, 5, 13]), ), 2)
call_1359 = relay.TupleGetItem(func_629_call(relay.reshape(var_1356.astype('float32'), [6, 5, 13]), relay.reshape(const_1357.astype('int16'), [2,]), relay.reshape(var_1356.astype('float32'), [6, 5, 13]), relay.reshape(var_1358.astype('float32'), [15,]), relay.reshape(var_1356.astype('float64'), [6, 5, 13]), ), 2)
output = relay.Tuple([uop_1344,call_1349,var_1350,call_1355,var_1356,const_1357,var_1358,])
output2 = relay.Tuple([uop_1346,call_1351,var_1350,call_1359,var_1356,const_1357,var_1358,])
func_1360 = relay.Function([var_1350,var_1356,var_1358,], output)
mod['func_1360'] = func_1360
mod = relay.transform.InferType()(mod)
var_1361 = relay.var("var_1361", dtype = "float64", shape = (16, 1))#candidate|1361|(16, 1)|var|float64
var_1362 = relay.var("var_1362", dtype = "float32", shape = (1, 390))#candidate|1362|(1, 390)|var|float32
var_1363 = relay.var("var_1363", dtype = "float32", shape = (1, 15))#candidate|1363|(1, 15)|var|float32
output = func_1360(var_1361,var_1362,var_1363,)
func_1364 = relay.Function([var_1361,var_1362,var_1363,], output)
mutated_mod['func_1364'] = func_1364
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1321_call = mod.get_global_var('func_1321')
func_1323_call = mutated_mod.get_global_var('func_1323')
call_1373 = relay.TupleGetItem(func_1321_call(), 4)
call_1374 = relay.TupleGetItem(func_1323_call(), 4)
func_186_call = mod.get_global_var('func_186')
func_188_call = mutated_mod.get_global_var('func_188')
call_1385 = relay.TupleGetItem(func_186_call(relay.reshape(call_1373.astype('float64'), [7,])), 0)
call_1386 = relay.TupleGetItem(func_188_call(relay.reshape(call_1373.astype('float64'), [7,])), 0)
bop_1391 = relay.logical_xor(call_1373.astype('uint16'), relay.reshape(call_1385.astype('uint16'), relay.shape_of(call_1373))) # shape=(7,)
bop_1394 = relay.logical_xor(call_1374.astype('uint16'), relay.reshape(call_1386.astype('uint16'), relay.shape_of(call_1374))) # shape=(7,)
output = bop_1391
output2 = bop_1394
func_1400 = relay.Function([], output)
mod['func_1400'] = func_1400
mod = relay.transform.InferType()(mod)
mutated_mod['func_1400'] = func_1400
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1400_call = mutated_mod.get_global_var('func_1400')
call_1401 = func_1400_call()
output = call_1401
func_1402 = relay.Function([], output)
mutated_mod['func_1402'] = func_1402
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1403 = relay.var("var_1403", dtype = "float64", shape = ())#candidate|1403|()|var|float64
var_1404 = relay.var("var_1404", dtype = "float64", shape = (1, 12, 5))#candidate|1404|(1, 12, 5)|var|float64
bop_1405 = relay.floor_divide(var_1403.astype('float64'), var_1404.astype('float64')) # shape=(1, 12, 5)
func_504_call = mod.get_global_var('func_504')
func_506_call = mutated_mod.get_global_var('func_506')
const_1409 = relay.const([-4.117572,-9.770759,-7.744328,-3.795955,9.389895,7.812649,3.153229,-1.347057,-9.353445,-6.266284,-2.321799,9.849977,6.695882,4.312283,3.859278], dtype = "float32")#candidate|1409|(15,)|const|float32
call_1408 = relay.TupleGetItem(func_504_call(relay.reshape(const_1409.astype('float32'), [15,])), 0)
call_1410 = relay.TupleGetItem(func_506_call(relay.reshape(const_1409.astype('float32'), [15,])), 0)
const_1420 = relay.const([1.416270,-0.183574,7.913444,4.751835,3.867895,8.477097,-2.218660,-3.697279,1.794991,2.524227,5.615498,0.774742,-2.689635,-1.657555,-1.699284], dtype = "float32")#candidate|1420|(15,)|const|float32
bop_1421 = relay.mod(const_1409.astype('float32'), relay.reshape(const_1420.astype('float32'), relay.shape_of(const_1409))) # shape=(15,)
bop_1425 = relay.bitwise_or(var_1404.astype('uint8'), relay.reshape(bop_1405.astype('uint8'), relay.shape_of(var_1404))) # shape=(1, 12, 5)
func_800_call = mod.get_global_var('func_800')
func_804_call = mutated_mod.get_global_var('func_804')
var_1430 = relay.var("var_1430", dtype = "float32", shape = (112,))#candidate|1430|(112,)|var|float32
var_1431 = relay.var("var_1431", dtype = "float64", shape = (7,))#candidate|1431|(7,)|var|float64
call_1429 = relay.TupleGetItem(func_800_call(relay.reshape(var_1430.astype('float32'), [112,]), relay.reshape(var_1431.astype('float64'), [7,]), ), 1)
call_1432 = relay.TupleGetItem(func_804_call(relay.reshape(var_1430.astype('float32'), [112,]), relay.reshape(var_1431.astype('float64'), [7,]), ), 1)
bop_1437 = relay.minimum(const_1420.astype('float64'), relay.reshape(const_1409.astype('float64'), relay.shape_of(const_1420))) # shape=(15,)
output = relay.Tuple([call_1408,bop_1421,bop_1425,call_1429,var_1430,var_1431,bop_1437,])
output2 = relay.Tuple([call_1410,bop_1421,bop_1425,call_1432,var_1430,var_1431,bop_1437,])
func_1441 = relay.Function([var_1403,var_1404,var_1430,var_1431,], output)
mod['func_1441'] = func_1441
mod = relay.transform.InferType()(mod)
mutated_mod['func_1441'] = func_1441
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1441_call = mutated_mod.get_global_var('func_1441')
var_1443 = relay.var("var_1443", dtype = "float64", shape = ())#candidate|1443|()|var|float64
var_1444 = relay.var("var_1444", dtype = "float64", shape = (1, 12, 5))#candidate|1444|(1, 12, 5)|var|float64
var_1445 = relay.var("var_1445", dtype = "float32", shape = (112,))#candidate|1445|(112,)|var|float32
var_1446 = relay.var("var_1446", dtype = "float64", shape = (7,))#candidate|1446|(7,)|var|float64
call_1442 = func_1441_call(var_1443,var_1444,var_1445,var_1446,)
output = call_1442
func_1447 = relay.Function([var_1443,var_1444,var_1445,var_1446,], output)
mutated_mod['func_1447'] = func_1447
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1495 = relay.var("var_1495", dtype = "float32", shape = (1, 11, 15))#candidate|1495|(1, 11, 15)|var|float32
uop_1496 = relay.atan(var_1495.astype('float32')) # shape=(1, 11, 15)
var_1499 = relay.var("var_1499", dtype = "float32", shape = (16, 11, 15))#candidate|1499|(16, 11, 15)|var|float32
bop_1500 = relay.not_equal(uop_1496.astype('bool'), var_1499.astype('bool')) # shape=(16, 11, 15)
uop_1503 = relay.erf(bop_1500.astype('float64')) # shape=(16, 11, 15)
bop_1508 = relay.less_equal(uop_1503.astype('bool'), var_1495.astype('bool')) # shape=(16, 11, 15)
bop_1512 = relay.maximum(var_1499.astype('int32'), relay.reshape(uop_1503.astype('int32'), relay.shape_of(var_1499))) # shape=(16, 11, 15)
bop_1515 = relay.mod(uop_1496.astype('float32'), uop_1503.astype('float32')) # shape=(16, 11, 15)
func_800_call = mod.get_global_var('func_800')
func_804_call = mutated_mod.get_global_var('func_804')
var_1519 = relay.var("var_1519", dtype = "float32", shape = (112,))#candidate|1519|(112,)|var|float32
const_1520 = relay.const([7.210310,-1.012671,-2.939994,3.591070,-8.785553,-8.473875,8.342218], dtype = "float64")#candidate|1520|(7,)|const|float64
call_1518 = relay.TupleGetItem(func_800_call(relay.reshape(var_1519.astype('float32'), [112,]), relay.reshape(const_1520.astype('float64'), [7,]), ), 0)
call_1521 = relay.TupleGetItem(func_804_call(relay.reshape(var_1519.astype('float32'), [112,]), relay.reshape(const_1520.astype('float64'), [7,]), ), 0)
uop_1522 = relay.sqrt(bop_1515.astype('float64')) # shape=(16, 11, 15)
output = relay.Tuple([bop_1508,bop_1512,call_1518,var_1519,const_1520,uop_1522,])
output2 = relay.Tuple([bop_1508,bop_1512,call_1521,var_1519,const_1520,uop_1522,])
func_1525 = relay.Function([var_1495,var_1499,var_1519,], output)
mod['func_1525'] = func_1525
mod = relay.transform.InferType()(mod)
mutated_mod['func_1525'] = func_1525
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1525_call = mutated_mod.get_global_var('func_1525')
var_1527 = relay.var("var_1527", dtype = "float32", shape = (1, 11, 15))#candidate|1527|(1, 11, 15)|var|float32
var_1528 = relay.var("var_1528", dtype = "float32", shape = (16, 11, 15))#candidate|1528|(16, 11, 15)|var|float32
var_1529 = relay.var("var_1529", dtype = "float32", shape = (112,))#candidate|1529|(112,)|var|float32
call_1526 = func_1525_call(var_1527,var_1528,var_1529,)
output = call_1526
func_1530 = relay.Function([var_1527,var_1528,var_1529,], output)
mutated_mod['func_1530'] = func_1530
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1321_call = mod.get_global_var('func_1321')
func_1323_call = mutated_mod.get_global_var('func_1323')
call_1537 = relay.TupleGetItem(func_1321_call(), 5)
call_1538 = relay.TupleGetItem(func_1323_call(), 5)
uop_1539 = relay.log10(call_1537.astype('float64')) # shape=(7,)
uop_1541 = relay.log10(call_1538.astype('float64')) # shape=(7,)
func_755_call = mod.get_global_var('func_755')
func_760_call = mutated_mod.get_global_var('func_760')
var_1546 = relay.var("var_1546", dtype = "float32", shape = (14,))#candidate|1546|(14,)|var|float32
call_1545 = relay.TupleGetItem(func_755_call(relay.reshape(var_1546.astype('float32'), [7, 2]), relay.reshape(var_1546.astype('float32'), [7, 2]), relay.reshape(call_1537.astype('float64'), [7,]), relay.reshape(var_1546.astype('float64'), [7, 2]), ), 0)
call_1547 = relay.TupleGetItem(func_760_call(relay.reshape(var_1546.astype('float32'), [7, 2]), relay.reshape(var_1546.astype('float32'), [7, 2]), relay.reshape(call_1537.astype('float64'), [7,]), relay.reshape(var_1546.astype('float64'), [7, 2]), ), 0)
bop_1551 = relay.mod(uop_1539.astype('float64'), relay.reshape(call_1537.astype('float64'), relay.shape_of(uop_1539))) # shape=(7,)
bop_1554 = relay.mod(uop_1541.astype('float64'), relay.reshape(call_1538.astype('float64'), relay.shape_of(uop_1541))) # shape=(7,)
bop_1557 = relay.power(bop_1551.astype('float64'), relay.reshape(uop_1539.astype('float64'), relay.shape_of(bop_1551))) # shape=(7,)
bop_1560 = relay.power(bop_1554.astype('float64'), relay.reshape(uop_1541.astype('float64'), relay.shape_of(bop_1554))) # shape=(7,)
output = relay.Tuple([call_1545,var_1546,bop_1557,])
output2 = relay.Tuple([call_1547,var_1546,bop_1560,])
func_1562 = relay.Function([var_1546,], output)
mod['func_1562'] = func_1562
mod = relay.transform.InferType()(mod)
mutated_mod['func_1562'] = func_1562
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1563 = relay.var("var_1563", dtype = "float32", shape = (14,))#candidate|1563|(14,)|var|float32
func_1562_call = mutated_mod.get_global_var('func_1562')
call_1564 = func_1562_call(var_1563)
output = call_1564
func_1565 = relay.Function([var_1563], output)
mutated_mod['func_1565'] = func_1565
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1321_call = mod.get_global_var('func_1321')
func_1323_call = mutated_mod.get_global_var('func_1323')
call_1605 = relay.TupleGetItem(func_1321_call(), 2)
call_1606 = relay.TupleGetItem(func_1323_call(), 2)
func_1162_call = mod.get_global_var('func_1162')
func_1170_call = mutated_mod.get_global_var('func_1170')
const_1611 = relay.const([1,-8], dtype = "uint32")#candidate|1611|(2,)|const|uint32
const_1612 = relay.const([[9.608293,-4.267507,2.781695,-6.899128,-9.252396,5.415276,-3.722691]], dtype = "float64")#candidate|1612|(1, 7)|const|float64
const_1613 = relay.const([2.425596,-1.830624,-0.536319,-3.421010,8.357253,-1.773285,-8.859837,0.062883,8.082003,-1.329416,-0.664713,0.135921,-8.725801,-2.459999], dtype = "float64")#candidate|1613|(14,)|const|float64
const_1614 = relay.const([5.416642,2.498433,6.109205,2.607982,-5.507235,5.758560,-2.429659,-9.810845,0.567907,6.540805,3.850638,4.482694,-2.323509,7.601955,-3.886491,-3.458365,-1.217505,8.789283,0.746383,1.851456,6.423340,-9.739461,4.215464,6.418972,3.976938,-4.235344,8.978438,-6.310177,9.602424,0.443557,-2.671692,-7.469796,-4.233227,-4.290814,-9.921503,9.065492,6.521273,9.321659,-7.817248,4.088886,5.566857,8.827909,-7.064356,2.591580,4.977606,3.695152,-4.590627,3.541770,-1.179779,9.062799,5.463791,-5.268619,-0.476525,0.600966,-5.170029,4.744249,9.512629,-8.181503,-4.932616,0.469559,-4.402677,-0.367933,-5.854849,9.534159,0.555918,-4.665573,-2.665211,3.031601,-1.953550,3.642758,-0.705233,7.546984,8.496978,6.177931,8.697224,-6.865836,3.721013,-1.331615,-5.450107,-2.908497,-0.526186,-0.026180,7.915664,-4.476920,-4.478477,-3.643190,-5.871965,-7.190146,1.399590,6.691169,2.889275,4.607636,-3.869577,-0.714639,6.174723,5.718308,7.783984,-2.862986,8.200289,4.949810,5.818016,-8.488310,1.185376,0.055678,-0.613872,9.770751,-9.607953,5.713949,-9.201715,5.393010,-8.500250,-8.107745], dtype = "float32")#candidate|1614|(112,)|const|float32
call_1610 = relay.TupleGetItem(func_1162_call(relay.reshape(const_1611.astype('uint32'), [2,]), relay.reshape(const_1611.astype('uint32'), [2,]), relay.reshape(const_1611.astype('uint32'), [2,]), relay.reshape(const_1612.astype('float64'), [7,]), relay.reshape(const_1613.astype('float64'), [14,]), relay.reshape(const_1614.astype('float32'), [112,]), ), 0)
call_1615 = relay.TupleGetItem(func_1170_call(relay.reshape(const_1611.astype('uint32'), [2,]), relay.reshape(const_1611.astype('uint32'), [2,]), relay.reshape(const_1611.astype('uint32'), [2,]), relay.reshape(const_1612.astype('float64'), [7,]), relay.reshape(const_1613.astype('float64'), [14,]), relay.reshape(const_1614.astype('float32'), [112,]), ), 0)
func_548_call = mod.get_global_var('func_548')
func_552_call = mutated_mod.get_global_var('func_552')
call_1617 = relay.TupleGetItem(func_548_call(relay.reshape(const_1614.astype('float32'), [16, 7]), relay.reshape(const_1612.astype('float64'), [7,]), ), 2)
call_1618 = relay.TupleGetItem(func_552_call(relay.reshape(const_1614.astype('float32'), [16, 7]), relay.reshape(const_1612.astype('float64'), [7,]), ), 2)
func_901_call = mod.get_global_var('func_901')
func_905_call = mutated_mod.get_global_var('func_905')
var_1630 = relay.var("var_1630", dtype = "uint32", shape = ())#candidate|1630|()|var|uint32
call_1629 = relay.TupleGetItem(func_901_call(relay.reshape(var_1630.astype('uint32'), []), relay.reshape(call_1617.astype('float64'), [7,]), ), 4)
call_1631 = relay.TupleGetItem(func_905_call(relay.reshape(var_1630.astype('uint32'), []), relay.reshape(call_1617.astype('float64'), [7,]), ), 4)
var_1633 = relay.var("var_1633", dtype = "float32", shape = (112,))#candidate|1633|(112,)|var|float32
bop_1634 = relay.not_equal(const_1614.astype('bool'), relay.reshape(var_1633.astype('bool'), relay.shape_of(const_1614))) # shape=(112,)
uop_1637 = relay.log10(bop_1634.astype('float32')) # shape=(112,)
bop_1639 = relay.left_shift(uop_1637.astype('uint8'), relay.reshape(bop_1634.astype('uint8'), relay.shape_of(uop_1637))) # shape=(112,)
func_1400_call = mod.get_global_var('func_1400')
func_1402_call = mutated_mod.get_global_var('func_1402')
call_1642 = func_1400_call()
call_1643 = func_1400_call()
uop_1646 = relay.log10(uop_1637.astype('float64')) # shape=(112,)
uop_1649 = relay.cosh(bop_1639.astype('float32')) # shape=(112,)
func_1360_call = mod.get_global_var('func_1360')
func_1364_call = mutated_mod.get_global_var('func_1364')
const_1652 = relay.const([[5.113495,-6.741579],[-9.248057,-5.756107],[-8.566538,4.024646],[8.805694,7.977135],[3.286833,5.190424],[0.369828,-2.200593],[-0.510053,-1.434728],[9.665158,5.128303],[1.836617,-3.626183],[8.726761,4.782161],[-0.352837,6.553323],[-3.567764,-6.935895],[2.743632,-9.466162],[1.324706,-5.366618],[-2.152228,-4.852837],[-8.107769,-7.336815],[0.143088,1.667431],[-6.591257,7.407505],[-3.458215,-7.076217],[1.645655,-9.121322],[1.937092,7.695947],[5.829930,-1.541836],[-5.024316,-1.620207],[5.825250,-0.722860],[8.140556,8.637130],[3.138060,-5.208179],[6.392538,-5.067204],[3.732602,-5.372763],[3.921212,-6.251926],[9.567629,0.258111],[0.414655,-8.865387],[-6.443327,-5.071498],[2.763479,7.106317],[-4.874716,-4.884354],[3.493714,-6.624545],[4.001816,7.717853],[-7.307322,-6.490770],[-6.893612,2.992263],[7.438426,-1.480121],[-8.984697,4.616441],[-3.730786,-2.851509],[9.293272,5.418973],[-9.350596,2.970431],[-3.887721,-8.432828],[-8.500044,-2.817387],[-5.228666,0.101333],[-5.136770,-7.908991],[-4.026412,-1.508043],[-7.937085,0.412552],[1.435662,-7.586052],[-8.599966,6.279435],[-4.600710,5.052755],[-4.005484,-6.631228],[5.131656,-3.901932],[7.749280,-3.460277],[1.531482,8.013780],[-3.385418,3.827170],[1.114777,9.482369],[1.344525,-7.618022],[-5.813259,0.687287],[9.697269,-3.783161],[2.785746,-3.746876],[7.521464,-4.459299],[8.372018,-8.385567],[-0.070236,9.776741],[-5.391521,7.066058],[-3.273098,7.121993],[6.536793,0.451996],[-0.820987,2.091544],[5.734607,3.010161],[-6.964526,6.233051],[0.723321,1.061291],[1.668002,4.689325],[6.022660,8.362356],[-5.617948,0.093011],[5.082883,-4.027993],[-0.781880,5.895036],[-3.407156,-7.172105],[-4.035621,-4.548003],[4.434520,5.350762],[1.267136,-9.331988],[3.222521,-5.162174],[6.653459,5.638832],[-9.346059,-0.447542],[6.909104,-3.625389],[-3.662203,9.937370],[-1.165226,-5.996227],[4.795528,-8.344912],[-9.000102,-1.233638],[6.078724,3.234360],[-8.721165,5.963806],[2.179495,0.284583],[-7.954124,-4.666915],[-6.638887,4.033772],[8.454579,-6.069382],[-8.396314,1.389576],[-4.509218,-8.744162],[-4.668746,-1.469051],[9.960523,-8.652003],[0.270042,9.656258],[9.607935,5.694750],[9.801601,9.709858],[-5.264052,-4.824166],[-3.804295,-1.743810],[3.718006,-4.603827],[-0.280262,-6.880221],[6.297202,-5.908334],[-7.118789,-3.753907],[8.146756,9.518503],[-0.231599,-3.355872],[-1.236950,8.641748],[-7.769733,1.715445],[4.059609,-2.260975],[-3.483196,3.016602],[7.701282,5.475610],[0.770398,6.650313],[8.464953,-3.181277],[2.248049,-8.908400],[1.735099,0.904913],[8.821744,1.799871],[-8.800718,9.455990],[-0.066677,-8.770209],[-9.362774,9.504420],[5.206696,4.267174],[5.005207,7.319918],[-7.564855,-1.349968],[1.654241,4.107929],[2.384072,1.070460],[-2.105986,-7.878875],[0.930038,-9.023034],[-1.586944,-3.643787],[6.407920,-1.171623],[0.174624,6.754097],[-3.818019,4.960408],[9.191974,-0.502822],[1.007477,2.756810],[-2.514996,-7.850833],[-6.147822,-2.017120],[-2.652326,3.193501],[-4.899909,-5.500036],[5.331218,8.225041],[3.835615,4.505302],[4.436123,7.433011],[-7.119206,1.113553],[8.028374,-5.429397],[-1.209610,-2.219804],[5.977990,-4.811917],[-9.583677,3.735062],[9.394665,5.360439],[3.884542,2.731705],[-0.658840,7.468398],[6.326195,-7.009236],[-9.872762,3.376533],[9.154173,8.481853],[3.351759,-7.797372],[-9.921248,3.353897],[-6.603574,4.591520],[1.437525,-5.641997],[4.759572,-5.159906],[-3.857819,7.915873],[-3.728612,-6.433495],[7.520783,-6.655705],[9.282230,-1.176598],[1.339836,7.899985],[-5.501773,2.533846],[-1.355864,9.320956],[-1.021533,-9.477345],[-4.484475,7.355968],[-1.776461,-5.908960],[-9.905686,-1.568596],[8.223858,7.095340],[-3.182100,-6.512683],[0.890742,7.700548],[-2.360218,-2.188632],[-6.432243,4.083195],[-1.754638,3.488896],[-5.894077,-1.183814],[5.710505,1.574755],[7.342501,3.817699],[1.452554,-6.316571],[2.646148,-6.662717],[6.662971,-4.006556],[-8.237128,-9.201830],[0.666926,3.749275],[-3.125302,-8.200054],[4.739649,-4.449233],[-7.511047,-4.320014],[-8.939843,1.989594],[-6.817684,1.123326],[-0.435196,-5.872797],[7.070078,0.003039],[-0.982236,2.072324],[0.471766,-6.791684],[0.446585,1.950733],[-9.336137,8.503533]], dtype = "float32")#candidate|1652|(195, 2)|const|float32
const_1653 = relay.const([6.177284,8.161904,-0.200148,-1.479754,-0.053568,-3.592933,-8.816523,5.621065,6.314794,-9.896251,8.541892,7.134691,-9.480313,-6.136619,8.304347], dtype = "float32")#candidate|1653|(15,)|const|float32
call_1651 = relay.TupleGetItem(func_1360_call(relay.reshape(call_1605.astype('float64'), [16, 1]), relay.reshape(const_1652.astype('float32'), [1, 390]), relay.reshape(const_1653.astype('float32'), [1, 15]), ), 1)
call_1654 = relay.TupleGetItem(func_1364_call(relay.reshape(call_1605.astype('float64'), [16, 1]), relay.reshape(const_1652.astype('float32'), [1, 390]), relay.reshape(const_1653.astype('float32'), [1, 15]), ), 1)
var_1655 = relay.var("var_1655", dtype = "float32", shape = (112,))#candidate|1655|(112,)|var|float32
bop_1656 = relay.logical_or(uop_1649.astype('bool'), relay.reshape(var_1655.astype('bool'), relay.shape_of(uop_1649))) # shape=(112,)
uop_1661 = relay.sin(uop_1637.astype('float32')) # shape=(112,)
uop_1670 = relay.log(uop_1637.astype('float64')) # shape=(112,)
var_1674 = relay.var("var_1674", dtype = "float32", shape = (112,))#candidate|1674|(112,)|var|float32
bop_1675 = relay.less(uop_1637.astype('bool'), relay.reshape(var_1674.astype('bool'), relay.shape_of(uop_1637))) # shape=(112,)
bop_1678 = relay.mod(uop_1661.astype('float64'), relay.reshape(uop_1637.astype('float64'), relay.shape_of(uop_1661))) # shape=(112,)
bop_1682 = relay.greater(uop_1670.astype('bool'), relay.reshape(uop_1649.astype('bool'), relay.shape_of(uop_1670))) # shape=(112,)
var_1689 = relay.var("var_1689", dtype = "bool", shape = (112,))#candidate|1689|(112,)|var|bool
bop_1690 = relay.mod(bop_1656.astype('float32'), relay.reshape(var_1689.astype('float32'), relay.shape_of(bop_1656))) # shape=(112,)
uop_1696 = relay.rsqrt(bop_1639.astype('float32')) # shape=(112,)
const_1699 = relay.const([-8.574088,-3.300898,1.396837,-6.463193,2.212426,-4.760547,2.735074,5.791902,5.851568,-7.213445,2.687971,0.614852,6.385042,2.034866,-3.723577,1.142838,6.383524,8.378615,2.693202,8.209810,3.568601,9.321512,8.407482,-4.121763,8.321389,-2.629936,8.734987,4.121793,7.375355,5.265176,5.442057,7.493196,3.680664,-2.508962,-0.528228,6.938867,9.864740,2.760775,9.456720,-2.086534,-8.719348,-6.421539,-5.371154,0.898325,-9.909764,8.146611,7.678797,-4.410692,6.154896,7.845280,7.766708,-9.637835,6.526706,9.611978,3.295432,-4.402919,-3.427758,5.015250,-9.103569,1.697884,-1.392307,-3.735725,3.693409,-8.477110,6.173900,-9.877725,-5.839741,-3.587491,-4.550400,-9.583432,6.178933,-1.055533,2.794131,-6.165851,-5.422506,-2.601950,-2.846235,-6.876310,-5.387993,3.759081,9.396152,-4.876941,-0.982577,7.700741,-9.785203,-3.298423,-2.158587,7.924937,-6.588202,8.081911,7.200182,0.886346,5.939459,4.641961,-8.590595,1.867246,6.717000,4.229194,-3.750545,-6.091965,7.346074,-1.729170,8.419764,5.335317,3.804011,-6.763663,5.936407,3.436577,-9.347608,-7.187131,6.693559,-6.422840], dtype = "float32")#candidate|1699|(112,)|const|float32
bop_1700 = relay.multiply(uop_1649.astype('uint8'), relay.reshape(const_1699.astype('uint8'), relay.shape_of(uop_1649))) # shape=(112,)
func_504_call = mod.get_global_var('func_504')
func_506_call = mutated_mod.get_global_var('func_506')
call_1705 = relay.TupleGetItem(func_504_call(relay.reshape(const_1653.astype('float32'), [15,])), 0)
call_1706 = relay.TupleGetItem(func_506_call(relay.reshape(const_1653.astype('float32'), [15,])), 0)
var_1707 = relay.var("var_1707", dtype = "bool", shape = (112,))#candidate|1707|(112,)|var|bool
bop_1708 = relay.not_equal(bop_1675.astype('bool'), relay.reshape(var_1707.astype('bool'), relay.shape_of(bop_1675))) # shape=(112,)
var_1716 = relay.var("var_1716", dtype = "float64", shape = (112,))#candidate|1716|(112,)|var|float64
bop_1717 = relay.equal(bop_1678.astype('bool'), relay.reshape(var_1716.astype('bool'), relay.shape_of(bop_1678))) # shape=(112,)
uop_1725 = relay.cos(bop_1678.astype('float32')) # shape=(112,)
uop_1729 = relay.atanh(uop_1725.astype('float32')) # shape=(112,)
bop_1732 = relay.divide(uop_1725.astype('float64'), relay.reshape(uop_1696.astype('float64'), relay.shape_of(uop_1725))) # shape=(112,)
bop_1739 = relay.less(uop_1729.astype('bool'), relay.reshape(bop_1717.astype('bool'), relay.shape_of(uop_1729))) # shape=(112,)
uop_1744 = relay.log2(uop_1725.astype('float32')) # shape=(112,)
output = relay.Tuple([call_1605,call_1610,const_1611,const_1612,const_1613,call_1617,call_1629,var_1630,call_1642,uop_1646,call_1651,const_1652,const_1653,bop_1682,bop_1690,bop_1700,call_1705,bop_1708,bop_1732,bop_1739,uop_1744,])
output2 = relay.Tuple([call_1606,call_1615,const_1611,const_1612,const_1613,call_1618,call_1631,var_1630,call_1643,uop_1646,call_1654,const_1652,const_1653,bop_1682,bop_1690,bop_1700,call_1706,bop_1708,bop_1732,bop_1739,uop_1744,])
func_1753 = relay.Function([var_1630,var_1633,var_1655,var_1674,var_1689,var_1707,var_1716,], output)
mod['func_1753'] = func_1753
mod = relay.transform.InferType()(mod)
mutated_mod['func_1753'] = func_1753
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1753_call = mutated_mod.get_global_var('func_1753')
var_1755 = relay.var("var_1755", dtype = "uint32", shape = ())#candidate|1755|()|var|uint32
var_1756 = relay.var("var_1756", dtype = "float32", shape = (112,))#candidate|1756|(112,)|var|float32
var_1757 = relay.var("var_1757", dtype = "float32", shape = (112,))#candidate|1757|(112,)|var|float32
var_1758 = relay.var("var_1758", dtype = "float32", shape = (112,))#candidate|1758|(112,)|var|float32
var_1759 = relay.var("var_1759", dtype = "bool", shape = (112,))#candidate|1759|(112,)|var|bool
var_1760 = relay.var("var_1760", dtype = "bool", shape = (112,))#candidate|1760|(112,)|var|bool
var_1761 = relay.var("var_1761", dtype = "float64", shape = (112,))#candidate|1761|(112,)|var|float64
call_1754 = func_1753_call(var_1755,var_1756,var_1757,var_1758,var_1759,var_1760,var_1761,)
output = call_1754
func_1762 = relay.Function([var_1755,var_1756,var_1757,var_1758,var_1759,var_1760,var_1761,], output)
mutated_mod['func_1762'] = func_1762
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1400_call = mod.get_global_var('func_1400')
func_1402_call = mutated_mod.get_global_var('func_1402')
call_1774 = func_1400_call()
call_1775 = func_1400_call()
output = relay.Tuple([call_1774,])
output2 = relay.Tuple([call_1775,])
func_1779 = relay.Function([], output)
mod['func_1779'] = func_1779
mod = relay.transform.InferType()(mod)
output = func_1779()
func_1780 = relay.Function([], output)
mutated_mod['func_1780'] = func_1780
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1781 = relay.const(9, dtype = "uint32")#candidate|1781|()|const|uint32
var_1782 = relay.var("var_1782", dtype = "uint32", shape = (7, 6))#candidate|1782|(7, 6)|var|uint32
bop_1783 = relay.bitwise_and(const_1781.astype('uint32'), var_1782.astype('uint32')) # shape=(7, 6)
output = bop_1783
output2 = bop_1783
func_1787 = relay.Function([var_1782,], output)
mod['func_1787'] = func_1787
mod = relay.transform.InferType()(mod)
var_1788 = relay.var("var_1788", dtype = "uint32", shape = (7, 6))#candidate|1788|(7, 6)|var|uint32
output = func_1787(var_1788)
func_1789 = relay.Function([var_1788], output)
mutated_mod['func_1789'] = func_1789
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1791 = relay.var("var_1791", dtype = "int16", shape = ())#candidate|1791|()|var|int16
var_1792 = relay.var("var_1792", dtype = "int16", shape = (7,))#candidate|1792|(7,)|var|int16
bop_1793 = relay.greater(var_1791.astype('bool'), var_1792.astype('bool')) # shape=(7,)
uop_1806 = relay.sin(bop_1793.astype('float32')) # shape=(7,)
uop_1811 = relay.sqrt(uop_1806.astype('float64')) # shape=(7,)
bop_1813 = relay.less_equal(uop_1811.astype('bool'), relay.reshape(var_1792.astype('bool'), relay.shape_of(uop_1811))) # shape=(7,)
output = relay.Tuple([bop_1813,])
output2 = relay.Tuple([bop_1813,])
func_1816 = relay.Function([var_1791,var_1792,], output)
mod['func_1816'] = func_1816
mod = relay.transform.InferType()(mod)
mutated_mod['func_1816'] = func_1816
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1816_call = mutated_mod.get_global_var('func_1816')
var_1818 = relay.var("var_1818", dtype = "int16", shape = ())#candidate|1818|()|var|int16
var_1819 = relay.var("var_1819", dtype = "int16", shape = (7,))#candidate|1819|(7,)|var|int16
call_1817 = func_1816_call(var_1818,var_1819,)
output = call_1817
func_1820 = relay.Function([var_1818,var_1819,], output)
mutated_mod['func_1820'] = func_1820
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1826 = relay.const([[9,-7,-8,-4,-7,1,7,3,6,5,8,5,-1],[2,-9,-2,-3,-3,-7,-1,2,-10,-1,-4,-10,6],[-4,3,10,-9,-8,-7,2,-3,9,-1,-7,7,-3],[-4,-4,1,5,-5,-9,2,-5,-10,1,4,-9,-10],[-6,-4,-1,-4,-1,5,10,9,-4,7,-1,4,2],[3,-3,-8,1,-9,-10,-7,9,5,2,7,6,-3]], dtype = "uint64")#candidate|1826|(6, 13)|const|uint64
var_1827 = relay.var("var_1827", dtype = "uint64", shape = (6, 13))#candidate|1827|(6, 13)|var|uint64
bop_1828 = relay.bitwise_xor(const_1826.astype('uint64'), relay.reshape(var_1827.astype('uint64'), relay.shape_of(const_1826))) # shape=(6, 13)
func_1779_call = mod.get_global_var('func_1779')
func_1780_call = mutated_mod.get_global_var('func_1780')
call_1831 = relay.TupleGetItem(func_1779_call(), 0)
call_1832 = relay.TupleGetItem(func_1780_call(), 0)
func_275_call = mod.get_global_var('func_275')
func_279_call = mutated_mod.get_global_var('func_279')
var_1840 = relay.var("var_1840", dtype = "int16", shape = (2,))#candidate|1840|(2,)|var|int16
call_1839 = relay.TupleGetItem(func_275_call(relay.reshape(var_1840.astype('int16'), [2,]), relay.reshape(var_1840.astype('int16'), [2,]), relay.reshape(var_1840.astype('bool'), [2,]), ), 3)
call_1841 = relay.TupleGetItem(func_279_call(relay.reshape(var_1840.astype('int16'), [2,]), relay.reshape(var_1840.astype('int16'), [2,]), relay.reshape(var_1840.astype('bool'), [2,]), ), 3)
uop_1857 = relay.cosh(bop_1828.astype('float32')) # shape=(6, 13)
output = relay.Tuple([call_1831,call_1839,var_1840,uop_1857,])
output2 = relay.Tuple([call_1832,call_1841,var_1840,uop_1857,])
F = relay.Function([var_1827,var_1840,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1827,var_1840,], output2)
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
input_1827= np.array([[-9,-2,5,-1,-4,7,8,8,-9,-1,5,-9,4],[-7,10,-10,10,-3,4,10,-9,-6,3,4,10,-2],[-1,10,-1,-10,8,1,5,-5,-9,7,-1,4,-8],[-8,8,-3,-3,4,2,6,-7,-10,-10,-9,1,9],[-10,-6,3,-7,5,10,5,-1,-10,-8,3,-9,-4],[-4,-6,-7,-7,-5,10,-3,-1,3,1,-2,-3,7]], dtype='uint64')
module1.set_input('var_1827', input_1827)
input_1840= np.array([-8,-1], dtype='int16')
module1.set_input('var_1840', input_1840)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1827, input_1840, )
res3 = intrp3.evaluate()(input_1827, input_1840, )
res4 = intrp4.evaluate()(input_1827, input_1840, )
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
module5.set_input('var_1827', input_1827)
module5.set_input('var_1840', input_1840)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1827, input_1840, )
res7 = intrp7.evaluate()(input_1827, input_1840, )
res8 = intrp8.evaluate()(input_1827, input_1840, )
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
module9.set_input('var_1827', input_1827)
module9.set_input('var_1840', input_1840)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1827, input_1840, )
res11 = intrp11.evaluate()(input_1827, input_1840, )
res12 = intrp12.evaluate()(input_1827, input_1840, )
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
module13.set_input('var_1827', input_1827)
module13.set_input('var_1840', input_1840)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1827, input_1840, )
res15 = intrp15.evaluate()(input_1827, input_1840, )
res16 = intrp16.evaluate()(input_1827, input_1840, )
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
module17.set_input('var_1827', input_1827)
module17.set_input('var_1840', input_1840)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1827, input_1840, )
res19 = intrp19.evaluate()(input_1827, input_1840, )
res20 = intrp20.evaluate()(input_1827, input_1840, )
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
module21.set_input('var_1827', input_1827)
module21.set_input('var_1840', input_1840)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1827, input_1840, )
res23 = intrp23.evaluate()(input_1827, input_1840, )
res24 = intrp24.evaluate()(input_1827, input_1840, )
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