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
var_16 = relay.var("var_16", dtype = "float32", shape = (10, 7, 5))#candidate|16|(10, 7, 5)|var|float32
var_17 = relay.var("var_17", dtype = "float32", shape = (10, 7, 5))#candidate|17|(10, 7, 5)|var|float32
bop_18 = relay.mod(var_16.astype('float32'), relay.reshape(var_17.astype('float32'), relay.shape_of(var_16))) # shape=(10, 7, 5)
bop_23 = relay.maximum(var_17.astype('uint16'), relay.reshape(bop_18.astype('uint16'), relay.shape_of(var_17))) # shape=(10, 7, 5)
output = relay.Tuple([bop_23,])
output2 = relay.Tuple([bop_23,])
func_30 = relay.Function([var_16,var_17,], output)
mod['func_30'] = func_30
mod = relay.transform.InferType()(mod)
mutated_mod['func_30'] = func_30
mutated_mod = relay.transform.InferType()(mutated_mod)
func_30_call = mutated_mod.get_global_var('func_30')
var_32 = relay.var("var_32", dtype = "float32", shape = (10, 7, 5))#candidate|32|(10, 7, 5)|var|float32
var_33 = relay.var("var_33", dtype = "float32", shape = (10, 7, 5))#candidate|33|(10, 7, 5)|var|float32
call_31 = func_30_call(var_32,var_33,)
output = call_31
func_34 = relay.Function([var_32,var_33,], output)
mutated_mod['func_34'] = func_34
mutated_mod = relay.transform.InferType()(mutated_mod)
var_111 = relay.var("var_111", dtype = "float64", shape = (3,))#candidate|111|(3,)|var|float64
uop_112 = relay.asin(var_111.astype('float64')) # shape=(3,)
func_30_call = mod.get_global_var('func_30')
func_34_call = mutated_mod.get_global_var('func_34')
const_115 = relay.const([-4.645459,-0.659417,9.639499,6.392241,1.298599,2.474773,-7.815805,-2.002224,0.795514,4.968580,9.039607,5.607344,-1.626555,-5.275611,7.292446,1.844738,-4.830843,-4.806238,-0.518199,3.114987,-9.024665,-5.892830,-5.339469,-9.285011,-4.160111,0.845542,7.505775,1.490170,-2.765594,1.346482,-3.982480,6.774027,-7.807279,-7.962678,-8.848531,8.338633,8.456567,7.192361,-9.956545,9.836668,6.105472,-0.740568,1.334323,3.246336,8.250839,1.987324,0.753850,-4.443653,-5.767363,-4.591538,-1.991049,0.958022,-1.168332,8.169231,8.835922,4.369455,6.422186,7.817006,8.146790,6.142442,6.392868,-9.640167,0.857941,8.707643,-1.129852,-2.769039,1.903295,-3.678433,4.061442,-9.398224,-5.773308,8.563644,3.115778,7.431184,6.237545,-1.805909,4.821179,-7.858969,0.155185,4.113696,0.609294,1.013618,6.102490,-4.354649,1.483810,9.681967,-1.406075,-1.596676,3.925399,6.108186,-8.753969,-8.736237,-7.357804,-6.246865,-9.499004,2.998024,-2.537495,-4.299266,-1.382375,-5.457963,6.088135,-9.445254,-3.324479,4.462245,-4.082834,4.150406,-5.862289,-0.629904,-8.687559,1.984100,-7.857580,-4.392097,8.453256,3.077997,-5.242184,7.988768,8.954171,6.448328,-3.754419,-4.262017,0.221459,0.286833,4.892813,1.094339,1.774784,-5.311500,5.455458,-0.121081,0.359706,6.636710,6.059575,6.529640,-3.440893,-8.408469,1.414903,3.501078,-0.456065,7.593044,-0.350861,2.945382,2.015356,-9.043980,3.106237,5.185173,-1.055069,-8.951328,0.866717,-4.938547,-8.994688,2.849363,5.290699,8.883259,3.991630,2.506116,4.709373,3.469121,-1.408342,2.103231,-6.851549,-6.079040,7.967333,-5.102713,-0.045646,-2.802107,-2.995797,1.472811,-9.200519,9.264989,6.195034,-5.166365,-8.813133,9.166473,5.707658,-5.877491,7.671671,-5.351365,0.817328,5.722126,6.472987,-8.284439,5.495886,-0.897561,-3.661130,-9.815332,-0.205845,-0.843243,-6.764506,-4.151501,7.939454,-9.238243,2.496465,0.988298,7.376064,6.731122,1.187508,-7.691362,8.444611,-5.944041,2.984586,7.311069,3.776346,-5.823054,3.650632,8.146308,6.471687,8.862413,-0.047327,1.580814,8.397394,5.984220,-7.953948,-2.546061,-6.140321,-2.344142,1.966188,0.120703,8.857861,-9.781629,-8.194049,1.926707,4.765170,7.214268,-8.401678,-5.623955,7.764860,-2.889334,6.590919,7.661412,2.255273,4.043224,6.573980,-2.067579,-8.794406,6.479418,-4.706316,-8.016663,5.634228,-8.900249,-9.798765,1.689073,-6.475037,-8.609926,-7.623804,4.875877,4.110173,-4.975646,8.400869,2.975319,-3.606468,8.707185,-1.518636,-0.134053,5.964154,8.288796,4.887293,7.038726,-3.114064,-1.484132,4.029163,-2.367779,-0.257790,7.895133,0.945055,-3.457725,7.667822,3.912831,2.120894,2.581429,-4.336424,8.691526,9.175427,8.891640,-4.795824,1.893574,8.146880,-0.873660,-0.931272,0.027576,-4.709080,7.101504,2.417611,8.644464,4.412970,9.670732,9.112245,5.842014,2.484130,-6.818274,-9.491198,9.089170,-2.582091,-8.129403,0.163262,-4.214045,5.435294,0.282899,-6.930921,3.961233,0.611073,6.782108,0.666872,-8.768016,3.715614,-0.601408,-2.312043,5.968360,1.335149,7.106688,-7.370619,9.130880,-7.251923,-8.189728,6.873089,-7.624030,6.830868,-9.220463,-7.396322,2.225719,9.026158,-0.436045,-7.691224,-5.856313,4.073119,-0.065435,-4.361642,7.567072,4.221360,1.795799,9.687154,7.114744,-0.489423,-6.401773,-2.576388,-6.080801,2.757185,5.804428,3.392988,-6.607997,-0.097650,-4.029542,-9.102200,4.817403,2.866568,-0.768870,-6.060330,8.585009,1.058182,1.456876,-2.115172,-6.747286], dtype = "float32")#candidate|115|(350,)|const|float32
call_114 = relay.TupleGetItem(func_30_call(relay.reshape(const_115.astype('float32'), [10, 7, 5]), relay.reshape(const_115.astype('float32'), [10, 7, 5]), ), 0)
call_116 = relay.TupleGetItem(func_34_call(relay.reshape(const_115.astype('float32'), [10, 7, 5]), relay.reshape(const_115.astype('float32'), [10, 7, 5]), ), 0)
uop_120 = relay.sin(const_115.astype('float64')) # shape=(350,)
uop_128 = relay.log10(uop_120.astype('float64')) # shape=(350,)
bop_135 = relay.floor_mod(const_115.astype('float64'), relay.reshape(uop_120.astype('float64'), relay.shape_of(const_115))) # shape=(350,)
const_138 = relay.const([2.622403,-9.474753,8.123852,-8.592280,-1.047149,-4.162008,-2.619395,2.393623,-7.951771,6.263381,-1.689289,-6.577481,-0.121732,-2.619698,3.946123,9.452580,5.813809,5.921087,-1.030371,9.053871,1.207399,-9.147318,7.350675,1.600457,1.934516,2.396741,-9.791788,-4.343437,-1.218705,8.304192,-5.847107,-0.870127,6.301836,0.626307,4.050347,-8.875466,-6.780446,-2.223472,-5.158004,6.123576,8.434095,-4.541812,2.316672,-4.687231,1.063618,5.745062,-1.217099,4.262664,-4.035326,6.090856,3.622416,-9.916021,-1.487889,5.487807,-9.828895,8.052585,-3.558553,-9.184346,-3.540189,5.678660,-2.766257,3.053777,-4.076864,-1.377691,2.686942,7.592691,-1.681891,-0.399664,-1.873954,5.667137,2.780995,-6.969247,-3.683650,6.600198,7.140772,2.614259,5.694525,-9.917519,0.852376,-8.500660,1.037733,-5.579707,-3.489785,-0.701984,-5.486413,1.891465,-4.816484,9.496497,-4.585125,2.593866,1.485645,-7.237291,-0.107904,0.374772,4.898936,-1.276597,9.884903,-3.295205,-1.439029,-2.056815,7.494175,-9.582444,-8.974187,-2.437811,-7.964362,-5.028144,-7.783714,-2.324447,-1.230391,0.976983,-0.281282,-8.897035,-2.529029,0.825645,3.728169,5.877070,1.395793,1.097830,-0.944399,1.480319,6.361750,-9.368466,-9.668754,-2.713153,-5.565587,-8.205964,-3.752464,-8.442801,-3.525568,0.926654,-0.296111,-1.212093,-7.459407,5.280560,6.532297,0.886708,-3.468129,9.014974,-0.124879,5.177679,1.655323,5.212440,-0.758313,-0.048226,3.209227,8.840466,-2.544191,-4.522778,5.904396,4.722048,-9.376643,0.137239,5.676537,1.758916,4.965011,-6.634911,-3.759759,-5.696677,-3.013530,2.726091,-9.156794,5.111302,-0.315106,9.957895,8.415423,9.067679,3.893676,-4.940985,5.021664,-1.086612,0.383785,3.945092,3.485908,-2.068780,-1.209967,1.934533,-3.247038,2.299313,-3.961681,-3.423331,-5.876179,3.792390,3.786201,-4.707758,9.123598,8.034129,1.656862,3.125584,-7.611778,-8.785390,1.716009,5.689834,-7.172905,9.349895,7.678228,-4.192251,7.847911,4.602640,-7.033528,1.405702,-4.564212,-9.482644,6.197013,-8.291271,9.468064,-6.225089,8.338060,3.786913,-8.301958,-5.686548,-3.817988,7.306175,5.360405,-0.707589,-4.085710,3.166716,-8.339102,9.906477,8.551673,7.335870,-2.270129,-1.241239,7.863250,-9.086710,-9.846539,9.701952,6.681415,-5.346736,9.332735,-0.170438,-2.938936,0.170945,9.503838,-2.796660,-0.863249,-3.683361,3.357500,-0.622728,-5.368984,-7.754567,-0.240105,3.346369,4.192543,-8.372928,0.587980,-4.362356,-3.952389,-2.558810,-5.510259,7.813725,5.756411,-9.838365,-2.033685,8.498053,8.056011,8.737097,-8.470490,5.572261,-9.035029,-7.514299,9.679343,3.177348,-2.011291,7.515257,-1.460477,-4.665527,-3.358613,-4.491429,2.722010,2.483397,7.757566,-4.199755,-7.212681,4.439634,0.869591,-4.158741,3.909505,6.872839,5.433944,-5.033856,4.150134,0.668072,2.917475,-2.345504,-8.184041,3.296932,-1.494332,6.315117,-5.381146,6.593465,-1.229516,-1.602390,4.980410,-2.288845,-6.477078,-2.670877,8.273029,-5.226737,6.176933,3.159689,-7.657966,5.339373,1.470225,1.146306,-4.007416,4.191967,-5.688940,-4.088907,0.447401,0.715285,1.494404,4.824477,7.927510,-4.094054,-8.175426,-7.204573,-6.333506,6.496488,-8.209543,0.441482,7.736007,-1.520330,-7.590519,5.913930,3.805677,-3.718785,-2.273839,3.621222,1.087335,-4.421384,8.787889,-5.363474,9.682871,6.959159,4.665110,-1.308653,-8.939551,-5.750691,9.238329,-2.127154,-5.731642,1.977735,2.058698,3.562491,-4.516522,-4.274483,-4.186235,-7.872113,3.441276,-2.928302], dtype = "float64")#candidate|138|(350,)|const|float64
bop_139 = relay.bitwise_or(uop_128.astype('uint8'), relay.reshape(const_138.astype('uint8'), relay.shape_of(uop_128))) # shape=(350,)
bop_148 = relay.greater_equal(uop_120.astype('bool'), relay.reshape(uop_128.astype('bool'), relay.shape_of(uop_120))) # shape=(350,)
func_30_call = mod.get_global_var('func_30')
func_34_call = mutated_mod.get_global_var('func_34')
call_159 = relay.TupleGetItem(func_30_call(relay.reshape(const_115.astype('float32'), [10, 7, 5]), relay.reshape(const_115.astype('float32'), [10, 7, 5]), ), 0)
call_160 = relay.TupleGetItem(func_34_call(relay.reshape(const_115.astype('float32'), [10, 7, 5]), relay.reshape(const_115.astype('float32'), [10, 7, 5]), ), 0)
output = relay.Tuple([uop_112,call_114,bop_135,bop_139,bop_148,call_159,])
output2 = relay.Tuple([uop_112,call_116,bop_135,bop_139,bop_148,call_160,])
func_165 = relay.Function([var_111,], output)
mod['func_165'] = func_165
mod = relay.transform.InferType()(mod)
mutated_mod['func_165'] = func_165
mutated_mod = relay.transform.InferType()(mutated_mod)
var_166 = relay.var("var_166", dtype = "float64", shape = (3,))#candidate|166|(3,)|var|float64
func_165_call = mutated_mod.get_global_var('func_165')
call_167 = func_165_call(var_166)
output = call_167
func_168 = relay.Function([var_166], output)
mutated_mod['func_168'] = func_168
mutated_mod = relay.transform.InferType()(mutated_mod)
var_265 = relay.var("var_265", dtype = "uint8", shape = (9, 14))#candidate|265|(9, 14)|var|uint8
var_266 = relay.var("var_266", dtype = "uint8", shape = (9, 14))#candidate|266|(9, 14)|var|uint8
bop_267 = relay.bitwise_and(var_265.astype('uint8'), relay.reshape(var_266.astype('uint8'), relay.shape_of(var_265))) # shape=(9, 14)
func_30_call = mod.get_global_var('func_30')
func_34_call = mutated_mod.get_global_var('func_34')
const_285 = relay.const([-4.870783,-4.848454,-8.542554,-1.421583,1.830477,6.577735,-4.965004,0.865404,-7.642300,-9.558220,6.812249,6.482999,2.709028,-9.133357,-6.528420,-8.703519,-5.911208,1.956258,6.737860,0.211670,5.960440,-7.046397,-3.392083,7.434772,3.836986,-7.265886,-9.841053,8.846298,-9.977453,2.489473,3.646952,5.488507,-2.087733,-2.318078,-6.404427,-4.769860,6.045277,2.567797,-3.485263,-3.304312,-3.543329,-8.130159,-4.188093,-1.636522,0.800989,6.298039,2.517841,5.220644,5.392310,9.516362,2.569659,3.644939,-2.068562,0.490667,0.554485,9.004490,-3.142462,6.798441,4.475825,1.178241,5.129788,0.876307,5.326172,-6.014584,7.394170,-4.757556,5.744759,4.967416,8.686756,-5.269394,-6.256873,-8.152231,5.157694,6.706349,-8.106637,-9.745493,9.200501,-2.409165,-3.023099,7.903281,8.173407,-8.809136,0.137723,-1.300965,-9.969841,-7.961735,-9.202795,3.095221,-7.217921,8.179971,0.597107,-8.228856,8.134281,7.696097,5.881183,1.655076,8.599803,7.209053,2.855678,9.770216,-1.441412,-8.062485,-6.692429,-0.081403,-9.348307,5.092378,9.060074,-5.294895,3.375399,-6.365856,-8.157002,3.557895,-4.784595,4.646283,-9.657916,2.178772,-9.407344,-5.583355,-0.754682,7.094818,-5.415085,-1.957176,-1.842195,-9.173838,5.929510,-1.266854,1.373943,9.807661,-8.025260,1.955904,4.905120,8.162855,6.527258,-6.868163,-6.655895,-9.693931,3.744198,9.833483,-7.956298,3.012018,-7.947334,2.843748,5.284206,5.790437,2.650428,-9.607558,-3.086536,-2.750150,0.332555,5.975854,-9.387593,-0.239612,-4.910773,7.259125,-9.793619,-7.464830,-4.998133,-3.913475,-8.969231,8.798930,-4.312247,-2.432529,4.551249,-5.360189,-9.736276,4.149357,-3.617310,-9.685704,5.891763,1.552055,-4.632030,-2.475851,7.789365,-4.156025,1.862904,-3.311464,5.028434,-9.185303,3.991408,-9.031816,-2.423390,4.529398,7.871095,5.093274,1.747980,-4.240409,4.061377,3.653868,4.651966,-2.026056,7.231083,7.957819,-8.845366,-2.880084,-6.522550,6.387588,-6.889682,7.494054,-0.622105,-0.648302,1.351764,3.450186,8.043419,-5.492192,-8.446140,3.827221,8.409274,-6.175451,-3.212083,6.597269,8.363597,0.279365,6.907407,-3.408095,-9.177014,-4.204397,0.236423,-0.182670,-3.930842,-1.265064,2.117985,-3.582384,-4.325416,-4.057773,3.839023,7.758854,-6.703459,-3.643629,-3.658224,0.245088,3.919660,-8.290220,4.546074,-6.663858,5.432093,-7.349642,-0.105429,5.622897,-0.489707,-0.399836,-8.167898,-5.682233,-9.973443,-5.809143,7.731147,6.215141,1.306659,-3.867860,-9.004679,-9.945201,5.639706,9.122907,-1.015236,-1.223908,-0.583450,-2.891425,-0.053593,-2.451321,-9.822272,7.402308,-1.997392,-4.970105,3.708623,8.148664,7.295396,6.272995,-5.077810,5.474150,-9.603715,2.089312,-0.252756,8.237466,3.117670,7.196949,-8.411641,7.027228,-1.856954,4.158944,-0.183239,2.750166,-7.624248,-5.051702,5.655422,5.597781,2.043501,8.430447,0.616539,3.934305,-7.114304,-5.215444,4.333748,7.669638,5.086847,5.490358,4.862176,-0.818032,-4.102545,0.390883,-5.160091,0.147970,8.149738,8.662146,3.579952,1.974562,7.045957,-3.575679,-5.188362,-1.268868,-4.643551,1.632524,3.122283,-8.261066,1.080644,7.193702,-4.479638,7.861746,2.515440,-2.074720,6.688743,-7.959897,0.795157,-1.020544,9.143698,-7.410992,-3.080440,3.079152,7.467029,5.696578,2.188982,-3.265611,5.670167,-1.804320,-9.051217,4.683947,-9.293400,-2.785336,6.147224,8.248371,-1.098622,-4.443581,-9.502120,-9.857423,3.683286,-1.148894,-8.639768,0.731272,-5.988005,9.433933,-1.905202,9.773551], dtype = "float32")#candidate|285|(350,)|const|float32
call_284 = relay.TupleGetItem(func_30_call(relay.reshape(const_285.astype('float32'), [10, 7, 5]), relay.reshape(const_285.astype('float32'), [10, 7, 5]), ), 0)
call_286 = relay.TupleGetItem(func_34_call(relay.reshape(const_285.astype('float32'), [10, 7, 5]), relay.reshape(const_285.astype('float32'), [10, 7, 5]), ), 0)
func_165_call = mod.get_global_var('func_165')
func_168_call = mutated_mod.get_global_var('func_168')
var_288 = relay.var("var_288", dtype = "float64", shape = (3,))#candidate|288|(3,)|var|float64
call_287 = relay.TupleGetItem(func_165_call(relay.reshape(var_288.astype('float64'), [3,])), 3)
call_289 = relay.TupleGetItem(func_168_call(relay.reshape(var_288.astype('float64'), [3,])), 3)
output = relay.Tuple([bop_267,call_284,const_285,call_287,var_288,])
output2 = relay.Tuple([bop_267,call_286,const_285,call_289,var_288,])
func_292 = relay.Function([var_265,var_266,var_288,], output)
mod['func_292'] = func_292
mod = relay.transform.InferType()(mod)
var_293 = relay.var("var_293", dtype = "uint8", shape = (9, 14))#candidate|293|(9, 14)|var|uint8
var_294 = relay.var("var_294", dtype = "uint8", shape = (9, 14))#candidate|294|(9, 14)|var|uint8
var_295 = relay.var("var_295", dtype = "float64", shape = (3,))#candidate|295|(3,)|var|float64
output = func_292(var_293,var_294,var_295,)
func_296 = relay.Function([var_293,var_294,var_295,], output)
mutated_mod['func_296'] = func_296
mutated_mod = relay.transform.InferType()(mutated_mod)
var_347 = relay.var("var_347", dtype = "uint16", shape = (1, 6))#candidate|347|(1, 6)|var|uint16
var_348 = relay.var("var_348", dtype = "uint16", shape = (4, 6))#candidate|348|(4, 6)|var|uint16
bop_349 = relay.left_shift(var_347.astype('uint16'), var_348.astype('uint16')) # shape=(4, 6)
bop_353 = relay.greater(var_348.astype('bool'), var_347.astype('bool')) # shape=(4, 6)
func_165_call = mod.get_global_var('func_165')
func_168_call = mutated_mod.get_global_var('func_168')
var_360 = relay.var("var_360", dtype = "float64", shape = (3,))#candidate|360|(3,)|var|float64
call_359 = relay.TupleGetItem(func_165_call(relay.reshape(var_360.astype('float64'), [3,])), 2)
call_361 = relay.TupleGetItem(func_168_call(relay.reshape(var_360.astype('float64'), [3,])), 2)
uop_362 = relay.atanh(call_359.astype('float64')) # shape=(350,)
uop_364 = relay.atanh(call_361.astype('float64')) # shape=(350,)
uop_365 = relay.acosh(uop_362.astype('float64')) # shape=(350,)
uop_367 = relay.acosh(uop_364.astype('float64')) # shape=(350,)
bop_368 = relay.add(uop_362.astype('uint64'), relay.reshape(uop_365.astype('uint64'), relay.shape_of(uop_362))) # shape=(350,)
bop_371 = relay.add(uop_364.astype('uint64'), relay.reshape(uop_367.astype('uint64'), relay.shape_of(uop_364))) # shape=(350,)
bop_379 = relay.floor_divide(bop_368.astype('float32'), relay.reshape(uop_365.astype('float32'), relay.shape_of(bop_368))) # shape=(350,)
bop_382 = relay.floor_divide(bop_371.astype('float32'), relay.reshape(uop_367.astype('float32'), relay.shape_of(bop_371))) # shape=(350,)
func_165_call = mod.get_global_var('func_165')
func_168_call = mutated_mod.get_global_var('func_168')
call_387 = relay.TupleGetItem(func_165_call(relay.reshape(var_360.astype('float64'), [3,])), 1)
call_388 = relay.TupleGetItem(func_168_call(relay.reshape(var_360.astype('float64'), [3,])), 1)
output = relay.Tuple([bop_349,bop_353,var_360,bop_379,call_387,])
output2 = relay.Tuple([bop_349,bop_353,var_360,bop_382,call_388,])
func_390 = relay.Function([var_347,var_348,var_360,], output)
mod['func_390'] = func_390
mod = relay.transform.InferType()(mod)
mutated_mod['func_390'] = func_390
mutated_mod = relay.transform.InferType()(mutated_mod)
func_390_call = mutated_mod.get_global_var('func_390')
var_392 = relay.var("var_392", dtype = "uint16", shape = (1, 6))#candidate|392|(1, 6)|var|uint16
var_393 = relay.var("var_393", dtype = "uint16", shape = (4, 6))#candidate|393|(4, 6)|var|uint16
var_394 = relay.var("var_394", dtype = "float64", shape = (3,))#candidate|394|(3,)|var|float64
call_391 = func_390_call(var_392,var_393,var_394,)
output = call_391
func_395 = relay.Function([var_392,var_393,var_394,], output)
mutated_mod['func_395'] = func_395
mutated_mod = relay.transform.InferType()(mutated_mod)
var_411 = relay.var("var_411", dtype = "int64", shape = (15, 13))#candidate|411|(15, 13)|var|int64
var_412 = relay.var("var_412", dtype = "int64", shape = (15, 13))#candidate|412|(15, 13)|var|int64
bop_413 = relay.bitwise_or(var_411.astype('int64'), relay.reshape(var_412.astype('int64'), relay.shape_of(var_411))) # shape=(15, 13)
uop_425 = relay.atan(bop_413.astype('float64')) # shape=(15, 13)
bop_430 = relay.greater(uop_425.astype('bool'), relay.reshape(bop_413.astype('bool'), relay.shape_of(uop_425))) # shape=(15, 13)
output = bop_430
output2 = bop_430
func_439 = relay.Function([var_411,var_412,], output)
mod['func_439'] = func_439
mod = relay.transform.InferType()(mod)
var_440 = relay.var("var_440", dtype = "int64", shape = (15, 13))#candidate|440|(15, 13)|var|int64
var_441 = relay.var("var_441", dtype = "int64", shape = (15, 13))#candidate|441|(15, 13)|var|int64
output = func_439(var_440,var_441,)
func_442 = relay.Function([var_440,var_441,], output)
mutated_mod['func_442'] = func_442
mutated_mod = relay.transform.InferType()(mutated_mod)
var_488 = relay.var("var_488", dtype = "uint64", shape = ())#candidate|488|()|var|uint64
const_489 = relay.const([[9,8,6,9,9,-10],[-8,-1,-5,-5,-8,-4],[-5,-2,7,2,7,3],[5,3,-6,1,3,8]], dtype = "uint64")#candidate|489|(4, 6)|const|uint64
bop_490 = relay.bitwise_xor(var_488.astype('uint64'), const_489.astype('uint64')) # shape=(4, 6)
bop_500 = relay.bitwise_or(const_489.astype('uint8'), relay.reshape(bop_490.astype('uint8'), relay.shape_of(const_489))) # shape=(4, 6)
output = relay.Tuple([bop_500,])
output2 = relay.Tuple([bop_500,])
func_511 = relay.Function([var_488,], output)
mod['func_511'] = func_511
mod = relay.transform.InferType()(mod)
var_512 = relay.var("var_512", dtype = "uint64", shape = ())#candidate|512|()|var|uint64
output = func_511(var_512)
func_513 = relay.Function([var_512], output)
mutated_mod['func_513'] = func_513
mutated_mod = relay.transform.InferType()(mutated_mod)
const_518 = relay.const(-5, dtype = "int8")#candidate|518|()|const|int8
var_519 = relay.var("var_519", dtype = "int8", shape = (16, 13, 2))#candidate|519|(16, 13, 2)|var|int8
bop_520 = relay.multiply(const_518.astype('int8'), var_519.astype('int8')) # shape=(16, 13, 2)
func_390_call = mod.get_global_var('func_390')
func_395_call = mutated_mod.get_global_var('func_395')
const_524 = relay.const([[6],[2],[7],[-6],[5],[-7]], dtype = "uint16")#candidate|524|(6, 1)|const|uint16
const_525 = relay.const([7,5,-2,-10,-9,-7,-1,-3,-5,-4,-8,8,-3,-3,9,10,-3,-7,-3,-2,-3,1,1,4], dtype = "uint16")#candidate|525|(24,)|const|uint16
var_526 = relay.var("var_526", dtype = "float64", shape = (3,))#candidate|526|(3,)|var|float64
call_523 = relay.TupleGetItem(func_390_call(relay.reshape(const_524.astype('uint16'), [1, 6]), relay.reshape(const_525.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 3)
call_527 = relay.TupleGetItem(func_395_call(relay.reshape(const_524.astype('uint16'), [1, 6]), relay.reshape(const_525.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 3)
uop_537 = relay.erf(bop_520.astype('float64')) # shape=(16, 13, 2)
uop_544 = relay.cos(uop_537.astype('float32')) # shape=(16, 13, 2)
bop_550 = relay.maximum(uop_544.astype('int64'), relay.reshape(bop_520.astype('int64'), relay.shape_of(uop_544))) # shape=(16, 13, 2)
func_390_call = mod.get_global_var('func_390')
func_395_call = mutated_mod.get_global_var('func_395')
call_554 = relay.TupleGetItem(func_390_call(relay.reshape(const_524.astype('uint16'), [1, 6]), relay.reshape(const_525.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 1)
call_555 = relay.TupleGetItem(func_395_call(relay.reshape(const_524.astype('uint16'), [1, 6]), relay.reshape(const_525.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 1)
bop_557 = relay.power(uop_537.astype('float64'), relay.reshape(var_519.astype('float64'), relay.shape_of(uop_537))) # shape=(16, 13, 2)
bop_568 = relay.add(uop_537.astype('uint8'), relay.reshape(bop_550.astype('uint8'), relay.shape_of(uop_537))) # shape=(16, 13, 2)
bop_573 = relay.minimum(uop_544.astype('uint32'), relay.reshape(uop_537.astype('uint32'), relay.shape_of(uop_544))) # shape=(16, 13, 2)
bop_579 = relay.equal(const_524.astype('bool'), const_518.astype('bool')) # shape=(6, 1)
bop_583 = relay.left_shift(bop_550.astype('uint64'), relay.reshape(bop_520.astype('uint64'), relay.shape_of(bop_550))) # shape=(16, 13, 2)
func_292_call = mod.get_global_var('func_292')
func_296_call = mutated_mod.get_global_var('func_296')
const_588 = relay.const([-8,-1,2,7,-3,-3,-7,4,8,7,-7,-9,-8,-7,7,10,1,-8,-3,-6,-6,-7,-6,5,8,-5,-4,9,-7,10,6,-3,-1,-10,8,3,7,-6,-2,4,-6,6,-7,9,-3,10,-7,6,6,6,-4,-5,-2,1,-4,5,-5,-8,-8,-8,4,-2,-3,-2,-6,2,-9,10,-9,-5,-7,-3,-3,-7,-7,-2,5,3,10,-8,6,-8,-6,-6,9,-8,-4,-1,9,6,6,-7,9,10,7,-8,5,-8,4,6,7,-2,-4,-8,4,10,-3,1,-9,3,-7,9,-9,-5,-2,6,4,2,2,2,7,4,-1,2,-5,5], dtype = "uint8")#candidate|588|(126,)|const|uint8
call_587 = relay.TupleGetItem(func_292_call(relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(var_526.astype('float64'), [3,]), ), 0)
call_589 = relay.TupleGetItem(func_296_call(relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(var_526.astype('float64'), [3,]), ), 0)
bop_590 = relay.mod(bop_557.astype('float32'), relay.reshape(bop_583.astype('float32'), relay.shape_of(bop_557))) # shape=(16, 13, 2)
bop_605 = relay.logical_and(bop_557.astype('bool'), relay.reshape(uop_537.astype('bool'), relay.shape_of(bop_557))) # shape=(16, 13, 2)
func_292_call = mod.get_global_var('func_292')
func_296_call = mutated_mod.get_global_var('func_296')
call_608 = relay.TupleGetItem(func_292_call(relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(call_587.astype('uint8'), [9, 14]), relay.reshape(var_526.astype('float64'), [3,]), ), 2)
call_609 = relay.TupleGetItem(func_296_call(relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(call_587.astype('uint8'), [9, 14]), relay.reshape(var_526.astype('float64'), [3,]), ), 2)
bop_619 = relay.less(uop_544.astype('bool'), relay.reshape(bop_605.astype('bool'), relay.shape_of(uop_544))) # shape=(16, 13, 2)
func_165_call = mod.get_global_var('func_165')
func_168_call = mutated_mod.get_global_var('func_168')
call_623 = relay.TupleGetItem(func_165_call(relay.reshape(var_526.astype('float64'), [3,])), 4)
call_624 = relay.TupleGetItem(func_168_call(relay.reshape(var_526.astype('float64'), [3,])), 4)
uop_638 = relay.rsqrt(bop_557.astype('float32')) # shape=(16, 13, 2)
func_511_call = mod.get_global_var('func_511')
func_513_call = mutated_mod.get_global_var('func_513')
call_644 = relay.TupleGetItem(func_511_call(relay.reshape(const_518.astype('uint64'), [])), 0)
call_645 = relay.TupleGetItem(func_513_call(relay.reshape(const_518.astype('uint64'), [])), 0)
func_390_call = mod.get_global_var('func_390')
func_395_call = mutated_mod.get_global_var('func_395')
call_648 = relay.TupleGetItem(func_390_call(relay.reshape(bop_579.astype('uint16'), [1, 6]), relay.reshape(call_554.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 3)
call_649 = relay.TupleGetItem(func_395_call(relay.reshape(bop_579.astype('uint16'), [1, 6]), relay.reshape(call_554.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 3)
bop_678 = relay.subtract(uop_638.astype('int16'), relay.reshape(var_519.astype('int16'), relay.shape_of(uop_638))) # shape=(16, 13, 2)
bop_686 = relay.not_equal(bop_550.astype('bool'), relay.reshape(bop_520.astype('bool'), relay.shape_of(bop_550))) # shape=(16, 13, 2)
func_292_call = mod.get_global_var('func_292')
func_296_call = mutated_mod.get_global_var('func_296')
call_694 = relay.TupleGetItem(func_292_call(relay.reshape(call_587.astype('uint8'), [9, 14]), relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(var_526.astype('float64'), [3,]), ), 2)
call_695 = relay.TupleGetItem(func_296_call(relay.reshape(call_587.astype('uint8'), [9, 14]), relay.reshape(const_588.astype('uint8'), [9, 14]), relay.reshape(var_526.astype('float64'), [3,]), ), 2)
bop_699 = relay.logical_and(bop_583.astype('bool'), relay.reshape(bop_686.astype('bool'), relay.shape_of(bop_583))) # shape=(16, 13, 2)
bop_722 = relay.floor_divide(bop_557.astype('float64'), relay.reshape(bop_583.astype('float64'), relay.shape_of(bop_557))) # shape=(16, 13, 2)
func_390_call = mod.get_global_var('func_390')
func_395_call = mutated_mod.get_global_var('func_395')
call_731 = relay.TupleGetItem(func_390_call(relay.reshape(const_524.astype('uint16'), [1, 6]), relay.reshape(call_644.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 3)
call_732 = relay.TupleGetItem(func_395_call(relay.reshape(const_524.astype('uint16'), [1, 6]), relay.reshape(call_644.astype('uint16'), [4, 6]), relay.reshape(var_526.astype('float64'), [3,]), ), 3)
bop_735 = relay.floor_mod(uop_544.astype('float32'), relay.reshape(bop_678.astype('float32'), relay.shape_of(uop_544))) # shape=(16, 13, 2)
uop_738 = relay.sinh(call_644.astype('float64')) # shape=(4, 6)
uop_740 = relay.sinh(call_645.astype('float64')) # shape=(4, 6)
uop_743 = relay.asin(bop_583.astype('float32')) # shape=(16, 13, 2)
func_30_call = mod.get_global_var('func_30')
func_34_call = mutated_mod.get_global_var('func_34')
call_748 = relay.TupleGetItem(func_30_call(relay.reshape(call_608.astype('float32'), [10, 7, 5]), relay.reshape(call_623.astype('float32'), [10, 7, 5]), ), 0)
call_749 = relay.TupleGetItem(func_34_call(relay.reshape(call_608.astype('float32'), [10, 7, 5]), relay.reshape(call_623.astype('float32'), [10, 7, 5]), ), 0)
bop_754 = relay.bitwise_xor(bop_722.astype('uint64'), relay.reshape(bop_686.astype('uint64'), relay.shape_of(bop_722))) # shape=(16, 13, 2)
bop_763 = relay.right_shift(uop_743.astype('int64'), relay.reshape(bop_699.astype('int64'), relay.shape_of(uop_743))) # shape=(16, 13, 2)
uop_767 = relay.log10(uop_537.astype('float32')) # shape=(16, 13, 2)
bop_779 = relay.bitwise_or(uop_743.astype('int16'), relay.reshape(uop_544.astype('int16'), relay.shape_of(uop_743))) # shape=(16, 13, 2)
uop_784 = relay.sinh(bop_779.astype('float64')) # shape=(16, 13, 2)
output = relay.Tuple([call_523,const_525,var_526,call_554,bop_568,bop_573,bop_579,call_587,const_588,bop_590,call_608,bop_619,call_623,call_648,call_694,call_731,bop_735,uop_738,call_748,bop_754,bop_763,uop_767,uop_784,])
output2 = relay.Tuple([call_527,const_525,var_526,call_555,bop_568,bop_573,bop_579,call_589,const_588,bop_590,call_609,bop_619,call_624,call_649,call_695,call_732,bop_735,uop_740,call_749,bop_754,bop_763,uop_767,uop_784,])
func_792 = relay.Function([var_519,var_526,], output)
mod['func_792'] = func_792
mod = relay.transform.InferType()(mod)
mutated_mod['func_792'] = func_792
mutated_mod = relay.transform.InferType()(mutated_mod)
func_792_call = mutated_mod.get_global_var('func_792')
var_794 = relay.var("var_794", dtype = "int8", shape = (16, 13, 2))#candidate|794|(16, 13, 2)|var|int8
var_795 = relay.var("var_795", dtype = "float64", shape = (3,))#candidate|795|(3,)|var|float64
call_793 = func_792_call(var_794,var_795,)
output = call_793
func_796 = relay.Function([var_794,var_795,], output)
mutated_mod['func_796'] = func_796
mutated_mod = relay.transform.InferType()(mutated_mod)
const_798 = relay.const([[-0.680513,6.431442,5.548486,7.882542,-7.409377],[2.769639,-5.696333,-5.974539,-6.214660,-8.722584],[7.234368,4.990075,5.189926,-9.783003,-4.488783],[-5.429490,-9.136763,-2.096823,8.950772,-4.506374],[-4.613094,5.166453,-2.714746,-0.809056,2.981067],[2.995061,6.044686,-5.336117,-5.272084,-7.933512],[-9.236507,-2.776519,-9.690723,-2.196630,-4.490090],[1.925702,6.525235,6.409228,-8.896073,-8.376013]], dtype = "float32")#candidate|798|(8, 5)|const|float32
uop_799 = relay.atanh(const_798.astype('float32')) # shape=(8, 5)
output = uop_799
output2 = uop_799
func_804 = relay.Function([], output)
mod['func_804'] = func_804
mod = relay.transform.InferType()(mod)
output = func_804()
func_805 = relay.Function([], output)
mutated_mod['func_805'] = func_805
mutated_mod = relay.transform.InferType()(mutated_mod)
var_820 = relay.var("var_820", dtype = "float32", shape = (10, 1))#candidate|820|(10, 1)|var|float32
uop_821 = relay.sqrt(var_820.astype('float32')) # shape=(10, 1)
output = uop_821
output2 = uop_821
func_823 = relay.Function([var_820,], output)
mod['func_823'] = func_823
mod = relay.transform.InferType()(mod)
mutated_mod['func_823'] = func_823
mutated_mod = relay.transform.InferType()(mutated_mod)
var_824 = relay.var("var_824", dtype = "float32", shape = (10, 1))#candidate|824|(10, 1)|var|float32
func_823_call = mutated_mod.get_global_var('func_823')
call_825 = func_823_call(var_824)
output = call_825
func_826 = relay.Function([var_824], output)
mutated_mod['func_826'] = func_826
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_893 = func_804_call()
call_894 = func_804_call()
func_511_call = mod.get_global_var('func_511')
func_513_call = mutated_mod.get_global_var('func_513')
var_896 = relay.var("var_896", dtype = "uint64", shape = ())#candidate|896|()|var|uint64
call_895 = relay.TupleGetItem(func_511_call(relay.reshape(var_896.astype('uint64'), [])), 0)
call_897 = relay.TupleGetItem(func_513_call(relay.reshape(var_896.astype('uint64'), [])), 0)
output = relay.Tuple([call_893,call_895,var_896,])
output2 = relay.Tuple([call_894,call_897,var_896,])
func_898 = relay.Function([var_896,], output)
mod['func_898'] = func_898
mod = relay.transform.InferType()(mod)
mutated_mod['func_898'] = func_898
mutated_mod = relay.transform.InferType()(mutated_mod)
var_899 = relay.var("var_899", dtype = "uint64", shape = ())#candidate|899|()|var|uint64
func_898_call = mutated_mod.get_global_var('func_898')
call_900 = func_898_call(var_899)
output = call_900
func_901 = relay.Function([var_899], output)
mutated_mod['func_901'] = func_901
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_918 = func_804_call()
call_919 = func_804_call()
output = call_918
output2 = call_919
func_922 = relay.Function([], output)
mod['func_922'] = func_922
mod = relay.transform.InferType()(mod)
output = func_922()
func_923 = relay.Function([], output)
mutated_mod['func_923'] = func_923
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_937 = func_804_call()
call_938 = func_804_call()
var_941 = relay.var("var_941", dtype = "float32", shape = (8, 5))#candidate|941|(8, 5)|var|float32
bop_942 = relay.subtract(call_937.astype('uint64'), relay.reshape(var_941.astype('uint64'), relay.shape_of(call_937))) # shape=(8, 5)
bop_945 = relay.subtract(call_938.astype('uint64'), relay.reshape(var_941.astype('uint64'), relay.shape_of(call_938))) # shape=(8, 5)
output = bop_942
output2 = bop_945
func_946 = relay.Function([var_941,], output)
mod['func_946'] = func_946
mod = relay.transform.InferType()(mod)
var_947 = relay.var("var_947", dtype = "float32", shape = (8, 5))#candidate|947|(8, 5)|var|float32
output = func_946(var_947)
func_948 = relay.Function([var_947], output)
mutated_mod['func_948'] = func_948
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_965 = func_804_call()
call_966 = func_804_call()
var_969 = relay.var("var_969", dtype = "float32", shape = (8, 5))#candidate|969|(8, 5)|var|float32
bop_970 = relay.logical_and(call_965.astype('bool'), relay.reshape(var_969.astype('bool'), relay.shape_of(call_965))) # shape=(8, 5)
bop_973 = relay.logical_and(call_966.astype('bool'), relay.reshape(var_969.astype('bool'), relay.shape_of(call_966))) # shape=(8, 5)
output = relay.Tuple([bop_970,])
output2 = relay.Tuple([bop_973,])
func_974 = relay.Function([var_969,], output)
mod['func_974'] = func_974
mod = relay.transform.InferType()(mod)
mutated_mod['func_974'] = func_974
mutated_mod = relay.transform.InferType()(mutated_mod)
var_975 = relay.var("var_975", dtype = "float32", shape = (8, 5))#candidate|975|(8, 5)|var|float32
func_974_call = mutated_mod.get_global_var('func_974')
call_976 = func_974_call(var_975)
output = call_976
func_977 = relay.Function([var_975], output)
mutated_mod['func_977'] = func_977
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_979 = func_804_call()
call_980 = func_804_call()
output = call_979
output2 = call_980
func_981 = relay.Function([], output)
mod['func_981'] = func_981
mod = relay.transform.InferType()(mod)
output = func_981()
func_982 = relay.Function([], output)
mutated_mod['func_982'] = func_982
mutated_mod = relay.transform.InferType()(mutated_mod)
var_997 = relay.var("var_997", dtype = "bool", shape = ())#candidate|997|()|var|bool
var_998 = relay.var("var_998", dtype = "bool", shape = (3, 1))#candidate|998|(3, 1)|var|bool
bop_999 = relay.logical_and(var_997.astype('bool'), var_998.astype('bool')) # shape=(3, 1)
bop_1013 = relay.bitwise_xor(var_998.astype('int8'), relay.reshape(bop_999.astype('int8'), relay.shape_of(var_998))) # shape=(3, 1)
output = relay.Tuple([bop_1013,])
output2 = relay.Tuple([bop_1013,])
func_1021 = relay.Function([var_997,var_998,], output)
mod['func_1021'] = func_1021
mod = relay.transform.InferType()(mod)
var_1022 = relay.var("var_1022", dtype = "bool", shape = ())#candidate|1022|()|var|bool
var_1023 = relay.var("var_1023", dtype = "bool", shape = (3, 1))#candidate|1023|(3, 1)|var|bool
output = func_1021(var_1022,var_1023,)
func_1024 = relay.Function([var_1022,var_1023,], output)
mutated_mod['func_1024'] = func_1024
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_1050 = func_804_call()
call_1051 = func_804_call()
uop_1052 = relay.asin(call_1050.astype('float32')) # shape=(8, 5)
uop_1054 = relay.asin(call_1051.astype('float32')) # shape=(8, 5)
output = uop_1052
output2 = uop_1054
func_1056 = relay.Function([], output)
mod['func_1056'] = func_1056
mod = relay.transform.InferType()(mod)
mutated_mod['func_1056'] = func_1056
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1056_call = mutated_mod.get_global_var('func_1056')
call_1057 = func_1056_call()
output = call_1057
func_1058 = relay.Function([], output)
mutated_mod['func_1058'] = func_1058
mutated_mod = relay.transform.InferType()(mutated_mod)
func_922_call = mod.get_global_var('func_922')
func_923_call = mutated_mod.get_global_var('func_923')
call_1083 = func_922_call()
call_1084 = func_922_call()
func_511_call = mod.get_global_var('func_511')
func_513_call = mutated_mod.get_global_var('func_513')
var_1086 = relay.var("var_1086", dtype = "uint64", shape = ())#candidate|1086|()|var|uint64
call_1085 = relay.TupleGetItem(func_511_call(relay.reshape(var_1086.astype('uint64'), [])), 0)
call_1087 = relay.TupleGetItem(func_513_call(relay.reshape(var_1086.astype('uint64'), [])), 0)
func_898_call = mod.get_global_var('func_898')
func_901_call = mutated_mod.get_global_var('func_901')
call_1104 = relay.TupleGetItem(func_898_call(relay.reshape(var_1086.astype('uint64'), [])), 0)
call_1105 = relay.TupleGetItem(func_901_call(relay.reshape(var_1086.astype('uint64'), [])), 0)
uop_1106 = relay.sqrt(call_1083.astype('float64')) # shape=(8, 5)
uop_1108 = relay.sqrt(call_1084.astype('float64')) # shape=(8, 5)
bop_1116 = relay.equal(uop_1106.astype('bool'), var_1086.astype('bool')) # shape=(8, 5)
bop_1119 = relay.equal(uop_1108.astype('bool'), var_1086.astype('bool')) # shape=(8, 5)
uop_1120 = relay.exp(call_1104.astype('float32')) # shape=(8, 5)
uop_1122 = relay.exp(call_1105.astype('float32')) # shape=(8, 5)
bop_1125 = relay.floor_divide(bop_1116.astype('float64'), relay.reshape(call_1083.astype('float64'), relay.shape_of(bop_1116))) # shape=(8, 5)
bop_1128 = relay.floor_divide(bop_1119.astype('float64'), relay.reshape(call_1084.astype('float64'), relay.shape_of(bop_1119))) # shape=(8, 5)
bop_1129 = relay.add(bop_1125.astype('int64'), relay.reshape(uop_1106.astype('int64'), relay.shape_of(bop_1125))) # shape=(8, 5)
bop_1132 = relay.add(bop_1128.astype('int64'), relay.reshape(uop_1108.astype('int64'), relay.shape_of(bop_1128))) # shape=(8, 5)
uop_1140 = relay.tan(bop_1125.astype('float64')) # shape=(8, 5)
uop_1142 = relay.tan(bop_1128.astype('float64')) # shape=(8, 5)
uop_1147 = relay.log2(uop_1140.astype('float32')) # shape=(8, 5)
uop_1149 = relay.log2(uop_1142.astype('float32')) # shape=(8, 5)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_1150 = func_804_call()
call_1151 = func_804_call()
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_1157 = func_804_call()
call_1158 = func_804_call()
bop_1159 = relay.divide(uop_1147.astype('float64'), relay.reshape(call_1150.astype('float64'), relay.shape_of(uop_1147))) # shape=(8, 5)
bop_1162 = relay.divide(uop_1149.astype('float64'), relay.reshape(call_1151.astype('float64'), relay.shape_of(uop_1149))) # shape=(8, 5)
func_1056_call = mod.get_global_var('func_1056')
func_1058_call = mutated_mod.get_global_var('func_1058')
call_1167 = func_1056_call()
call_1168 = func_1056_call()
output = relay.Tuple([call_1085,uop_1120,bop_1129,call_1157,bop_1159,call_1167,])
output2 = relay.Tuple([call_1087,uop_1122,bop_1132,call_1158,bop_1162,call_1168,])
func_1171 = relay.Function([var_1086,], output)
mod['func_1171'] = func_1171
mod = relay.transform.InferType()(mod)
var_1172 = relay.var("var_1172", dtype = "uint64", shape = ())#candidate|1172|()|var|uint64
output = func_1171(var_1172)
func_1173 = relay.Function([var_1172], output)
mutated_mod['func_1173'] = func_1173
mutated_mod = relay.transform.InferType()(mutated_mod)
func_981_call = mod.get_global_var('func_981')
func_982_call = mutated_mod.get_global_var('func_982')
call_1184 = func_981_call()
call_1185 = func_981_call()
output = call_1184
output2 = call_1185
func_1188 = relay.Function([], output)
mod['func_1188'] = func_1188
mod = relay.transform.InferType()(mod)
mutated_mod['func_1188'] = func_1188
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1188_call = mutated_mod.get_global_var('func_1188')
call_1189 = func_1188_call()
output = call_1189
func_1190 = relay.Function([], output)
mutated_mod['func_1190'] = func_1190
mutated_mod = relay.transform.InferType()(mutated_mod)
func_981_call = mod.get_global_var('func_981')
func_982_call = mutated_mod.get_global_var('func_982')
call_1295 = func_981_call()
call_1296 = func_981_call()
func_30_call = mod.get_global_var('func_30')
func_34_call = mutated_mod.get_global_var('func_34')
var_1303 = relay.var("var_1303", dtype = "float32", shape = (350,))#candidate|1303|(350,)|var|float32
call_1302 = relay.TupleGetItem(func_30_call(relay.reshape(var_1303.astype('float32'), [10, 7, 5]), relay.reshape(var_1303.astype('float32'), [10, 7, 5]), ), 0)
call_1304 = relay.TupleGetItem(func_34_call(relay.reshape(var_1303.astype('float32'), [10, 7, 5]), relay.reshape(var_1303.astype('float32'), [10, 7, 5]), ), 0)
bop_1306 = relay.less_equal(call_1302.astype('bool'), relay.reshape(var_1303.astype('bool'), relay.shape_of(call_1302))) # shape=(10, 7, 5)
bop_1309 = relay.less_equal(call_1304.astype('bool'), relay.reshape(var_1303.astype('bool'), relay.shape_of(call_1304))) # shape=(10, 7, 5)
output = relay.Tuple([call_1295,bop_1306,])
output2 = relay.Tuple([call_1296,bop_1309,])
func_1313 = relay.Function([var_1303,], output)
mod['func_1313'] = func_1313
mod = relay.transform.InferType()(mod)
var_1314 = relay.var("var_1314", dtype = "float32", shape = (350,))#candidate|1314|(350,)|var|float32
output = func_1313(var_1314)
func_1315 = relay.Function([var_1314], output)
mutated_mod['func_1315'] = func_1315
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1188_call = mod.get_global_var('func_1188')
func_1190_call = mutated_mod.get_global_var('func_1190')
call_1365 = func_1188_call()
call_1366 = func_1188_call()
func_1171_call = mod.get_global_var('func_1171')
func_1173_call = mutated_mod.get_global_var('func_1173')
var_1392 = relay.var("var_1392", dtype = "uint64", shape = ())#candidate|1392|()|var|uint64
call_1391 = relay.TupleGetItem(func_1171_call(relay.reshape(var_1392.astype('uint64'), [])), 3)
call_1393 = relay.TupleGetItem(func_1173_call(relay.reshape(var_1392.astype('uint64'), [])), 3)
output = relay.Tuple([call_1365,call_1391,var_1392,])
output2 = relay.Tuple([call_1366,call_1393,var_1392,])
func_1397 = relay.Function([var_1392,], output)
mod['func_1397'] = func_1397
mod = relay.transform.InferType()(mod)
mutated_mod['func_1397'] = func_1397
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1398 = relay.var("var_1398", dtype = "uint64", shape = ())#candidate|1398|()|var|uint64
func_1397_call = mutated_mod.get_global_var('func_1397')
call_1399 = func_1397_call(var_1398)
output = call_1399
func_1400 = relay.Function([var_1398], output)
mutated_mod['func_1400'] = func_1400
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1402 = relay.var("var_1402", dtype = "float32", shape = (13, 11))#candidate|1402|(13, 11)|var|float32
uop_1403 = relay.log(var_1402.astype('float32')) # shape=(13, 11)
uop_1405 = relay.sinh(uop_1403.astype('float64')) # shape=(13, 11)
uop_1409 = relay.asin(uop_1403.astype('float32')) # shape=(13, 11)
uop_1413 = relay.acosh(uop_1409.astype('float32')) # shape=(13, 11)
func_981_call = mod.get_global_var('func_981')
func_982_call = mutated_mod.get_global_var('func_982')
call_1422 = func_981_call()
call_1423 = func_981_call()
var_1426 = relay.var("var_1426", dtype = "float32", shape = (13, 11))#candidate|1426|(13, 11)|var|float32
bop_1427 = relay.greater_equal(var_1402.astype('bool'), relay.reshape(var_1426.astype('bool'), relay.shape_of(var_1402))) # shape=(13, 11)
output = relay.Tuple([uop_1405,uop_1413,call_1422,bop_1427,])
output2 = relay.Tuple([uop_1405,uop_1413,call_1423,bop_1427,])
func_1431 = relay.Function([var_1402,var_1426,], output)
mod['func_1431'] = func_1431
mod = relay.transform.InferType()(mod)
mutated_mod['func_1431'] = func_1431
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1431_call = mutated_mod.get_global_var('func_1431')
var_1433 = relay.var("var_1433", dtype = "float32", shape = (13, 11))#candidate|1433|(13, 11)|var|float32
var_1434 = relay.var("var_1434", dtype = "float32", shape = (13, 11))#candidate|1434|(13, 11)|var|float32
call_1432 = func_1431_call(var_1433,var_1434,)
output = call_1432
func_1435 = relay.Function([var_1433,var_1434,], output)
mutated_mod['func_1435'] = func_1435
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_1446 = func_804_call()
call_1447 = func_804_call()
func_439_call = mod.get_global_var('func_439')
func_442_call = mutated_mod.get_global_var('func_442')
var_1452 = relay.var("var_1452", dtype = "int64", shape = (195,))#candidate|1452|(195,)|var|int64
call_1451 = func_439_call(relay.reshape(var_1452.astype('int64'), [15, 13]), relay.reshape(var_1452.astype('int64'), [15, 13]), )
call_1453 = func_439_call(relay.reshape(var_1452.astype('int64'), [15, 13]), relay.reshape(var_1452.astype('int64'), [15, 13]), )
output = relay.Tuple([call_1446,call_1451,var_1452,])
output2 = relay.Tuple([call_1447,call_1453,var_1452,])
func_1463 = relay.Function([var_1452,], output)
mod['func_1463'] = func_1463
mod = relay.transform.InferType()(mod)
var_1464 = relay.var("var_1464", dtype = "int64", shape = (195,))#candidate|1464|(195,)|var|int64
output = func_1463(var_1464)
func_1465 = relay.Function([var_1464], output)
mutated_mod['func_1465'] = func_1465
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1056_call = mod.get_global_var('func_1056')
func_1058_call = mutated_mod.get_global_var('func_1058')
call_1467 = func_1056_call()
call_1468 = func_1056_call()
var_1471 = relay.var("var_1471", dtype = "float32", shape = (8, 5))#candidate|1471|(8, 5)|var|float32
bop_1472 = relay.bitwise_xor(call_1467.astype('uint16'), relay.reshape(var_1471.astype('uint16'), relay.shape_of(call_1467))) # shape=(8, 5)
bop_1475 = relay.bitwise_xor(call_1468.astype('uint16'), relay.reshape(var_1471.astype('uint16'), relay.shape_of(call_1468))) # shape=(8, 5)
func_439_call = mod.get_global_var('func_439')
func_442_call = mutated_mod.get_global_var('func_442')
const_1484 = relay.const([4,-6,-4,7,9,-2,-8,10,-8,8,2,-4,-1,6,-10,4,2,-2,-2,-8,1,-6,10,-8,-4,-4,-4,-5,9,-5,-10,-2,8,9,-2,-2,8,-4,3,9,-2,-6,-3,1,4,-10,4,7,5,3,4,4,7,-2,-4,10,4,-10,10,-1,-5,9,5,5,-1,-1,8,-8,-3,-5,-1,-1,-2,-4,-2,7,5,1,1,10,-4,8,6,5,6,6,-7,-2,8,1,8,9,3,3,-6,5,-6,2,-9,-1,8,-10,-4,4,-8,-4,-5,4,-3,-10,-2,8,8,-9,-6,-9,8,9,-6,4,10,-2,10,-2,3,-3,7,-4,-1,1,3,-4,-1,-3,-8,1,-2,2,1,7,6,-10,1,-7,6,2,7,-2,-8,5,9,-3,-1,5,9,-5,-3,-7,8,9,-9,5,4,4,2,10,-5,6,-8,-7,4,1,8,4,-1,2,-3,-7,6,-2,1,2,3,-4,-9,-7,3,-4,7,9,-2,-7,7,5,5], dtype = "int64")#candidate|1484|(195,)|const|int64
call_1483 = func_439_call(relay.reshape(const_1484.astype('int64'), [15, 13]), relay.reshape(const_1484.astype('int64'), [15, 13]), )
call_1485 = func_439_call(relay.reshape(const_1484.astype('int64'), [15, 13]), relay.reshape(const_1484.astype('int64'), [15, 13]), )
func_390_call = mod.get_global_var('func_390')
func_395_call = mutated_mod.get_global_var('func_395')
var_1491 = relay.var("var_1491", dtype = "uint16", shape = (6,))#candidate|1491|(6,)|var|uint16
var_1492 = relay.var("var_1492", dtype = "uint16", shape = (24,))#candidate|1492|(24,)|var|uint16
const_1493 = relay.const([-7.167512,5.718063,3.952191], dtype = "float64")#candidate|1493|(3,)|const|float64
call_1490 = relay.TupleGetItem(func_390_call(relay.reshape(var_1491.astype('uint16'), [1, 6]), relay.reshape(var_1492.astype('uint16'), [4, 6]), relay.reshape(const_1493.astype('float64'), [3,]), ), 1)
call_1494 = relay.TupleGetItem(func_395_call(relay.reshape(var_1491.astype('uint16'), [1, 6]), relay.reshape(var_1492.astype('uint16'), [4, 6]), relay.reshape(const_1493.astype('float64'), [3,]), ), 1)
const_1495 = relay.const([[5.087204,1.024368,9.572284,0.094938,6.241492],[-6.647168,-3.805421,7.201849,-2.462161,-9.759195],[6.463877,3.715873,1.254194,3.883085,-7.285846],[5.481107,-3.989891,-4.521966,-7.264793,2.934199],[-9.418701,-5.097331,8.605662,3.644433,9.744411],[-3.797692,-6.094732,-2.609723,-2.672893,-0.567031],[-4.250580,6.440634,-9.630835,-6.521700,5.170720],[-5.822570,0.144271,-8.759858,6.616119,-2.459487]], dtype = "float32")#candidate|1495|(8, 5)|const|float32
bop_1496 = relay.minimum(call_1467.astype('uint64'), relay.reshape(const_1495.astype('uint64'), relay.shape_of(call_1467))) # shape=(8, 5)
bop_1499 = relay.minimum(call_1468.astype('uint64'), relay.reshape(const_1495.astype('uint64'), relay.shape_of(call_1468))) # shape=(8, 5)
output = relay.Tuple([bop_1472,call_1483,const_1484,call_1490,var_1491,var_1492,const_1493,bop_1496,])
output2 = relay.Tuple([bop_1475,call_1485,const_1484,call_1494,var_1491,var_1492,const_1493,bop_1499,])
func_1510 = relay.Function([var_1471,var_1491,var_1492,], output)
mod['func_1510'] = func_1510
mod = relay.transform.InferType()(mod)
var_1511 = relay.var("var_1511", dtype = "float32", shape = (8, 5))#candidate|1511|(8, 5)|var|float32
var_1512 = relay.var("var_1512", dtype = "uint16", shape = (6,))#candidate|1512|(6,)|var|uint16
var_1513 = relay.var("var_1513", dtype = "uint16", shape = (24,))#candidate|1513|(24,)|var|uint16
output = func_1510(var_1511,var_1512,var_1513,)
func_1514 = relay.Function([var_1511,var_1512,var_1513,], output)
mutated_mod['func_1514'] = func_1514
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1516 = relay.var("var_1516", dtype = "float32", shape = ())#candidate|1516|()|var|float32
var_1517 = relay.var("var_1517", dtype = "float32", shape = (5, 10, 8))#candidate|1517|(5, 10, 8)|var|float32
bop_1518 = relay.divide(var_1516.astype('float32'), var_1517.astype('float32')) # shape=(5, 10, 8)
uop_1521 = relay.acosh(bop_1518.astype('float64')) # shape=(5, 10, 8)
func_792_call = mod.get_global_var('func_792')
func_796_call = mutated_mod.get_global_var('func_796')
const_1533 = relay.const([[-5,-6,-6,-9,-5,5,5,-7,9,4,-7,-9,7,-7,-7,3,-7,-4,-1,-6,-5,10,-8,-10,10,7,5,8,-9,6,1,-9,8,9,-4,6,7,9,4,6,-5,1,7,1,2,4,-9,-2,-6,-6,6,-10,-7,-10,-8,5,9,3,-3,4,4,-1,-2,-9,3,-9,-7,-4,9,7,10,1,8,-7,-10,10,-10,-10,3,-3,7,5,-2,-10,3,-8,-9,-4,5,-10,4,-3,-1,1,2,-2,5,-3,-8,-7,10,6,6,-8,5,8,-8,-6,-3,6,-10,-4,-8,5,10,9,-9,1,8,4,7,-3,-2,7,7,-1,-2,5,10,5,1,-1,-8,-6,2,-10,-4,8,-3,1,6,10,-5,-7,-4,-6,-8,-7,-9,-5,-5,-2,7,2,2,-3,-5,8,7,-4,10,9,-7,4,-1,7,-3,4,-6,3,2,-8,-8,-5,-2,7,-1,9,5,3,4,4,6,9,-6,2,6,-3,4,1,-10,-9,-5,6,4,7,-9,5,-9,-9,-5,-3,1,-7,-7,-3,7,6],[2,3,5,-6,-1,10,-3,6,-10,-3,6,-7,10,-10,4,10,-1,-4,3,1,-8,7,-1,-3,4,3,-8,5,-10,5,-2,-10,6,-9,1,-4,-8,4,5,-2,3,-9,4,7,8,8,4,2,1,10,9,2,-4,9,-2,-8,-4,4,-10,2,2,7,5,8,9,7,-4,7,5,-7,5,-2,4,-7,-8,4,6,-9,-1,8,-5,-9,-5,-8,10,-9,-3,-2,-10,1,7,-9,-7,7,-5,9,1,10,-2,-8,4,-7,-8,-3,-9,8,10,-8,-6,-6,-9,3,-2,-8,3,-7,3,5,-5,2,3,-10,5,-3,-5,4,-3,-4,1,-9,7,-1,-2,-4,5,10,3,-4,-10,-1,-5,6,1,-4,5,-9,1,-3,-6,-1,10,-10,3,-3,-10,-5,9,-4,-5,-5,4,-3,3,10,-2,10,-4,8,4,-8,4,-1,5,3,4,10,-1,-2,8,4,5,-5,6,-6,-3,-9,8,-9,-5,-4,-2,8,-5,10,4,-3,-1,8,2,-2,-8,3,10,8,-4,-6,-7,2]], dtype = "int8")#candidate|1533|(2, 208)|const|int8
const_1534 = relay.const([5.562596,-4.583208,4.791302], dtype = "float64")#candidate|1534|(3,)|const|float64
call_1532 = relay.TupleGetItem(func_792_call(relay.reshape(const_1533.astype('int8'), [16, 13, 2]), relay.reshape(const_1534.astype('float64'), [3,]), ), 16)
call_1535 = relay.TupleGetItem(func_796_call(relay.reshape(const_1533.astype('int8'), [16, 13, 2]), relay.reshape(const_1534.astype('float64'), [3,]), ), 16)
func_1397_call = mod.get_global_var('func_1397')
func_1400_call = mutated_mod.get_global_var('func_1400')
call_1541 = relay.TupleGetItem(func_1397_call(relay.reshape(var_1516.astype('uint64'), [])), 0)
call_1542 = relay.TupleGetItem(func_1400_call(relay.reshape(var_1516.astype('uint64'), [])), 0)
func_1431_call = mod.get_global_var('func_1431')
func_1435_call = mutated_mod.get_global_var('func_1435')
const_1545 = relay.const([[1.303618,3.108595,-3.958014,5.099473,3.683335,-5.478619,7.200398,-5.965603,2.233653,-4.666016,8.775742],[-8.002841,-7.491711,-5.041411,-6.607689,-6.076500,-5.114246,-4.455815,-5.511891,3.318292,1.261099,4.856349],[-6.367360,-5.195701,-2.924394,-9.328054,-5.415344,-6.268856,-9.872178,0.485329,6.212832,-1.271527,4.553614],[-4.186051,8.964314,2.348870,0.751586,1.058185,-8.082242,-7.282308,2.996204,-2.768825,4.824690,-3.943059],[-3.162693,-2.326388,-7.377014,7.002287,-4.261892,3.130050,-2.192088,-6.110946,6.159829,1.438591,-4.388444],[-8.498186,8.824602,2.533927,-3.779858,-2.037764,6.862231,-7.944924,-1.359513,-0.404668,-1.812248,-5.758022],[-7.504847,-0.173838,-6.727802,-9.773825,0.897484,-4.546847,5.495513,4.571224,-4.975803,-8.830597,0.401964],[5.096744,0.113143,-4.703447,8.534554,-8.955985,7.040942,-6.845983,-7.891337,-2.248105,0.791726,-8.828455],[-9.193273,-6.796041,-6.064166,8.291344,-9.213281,1.995082,-4.306801,3.649395,-7.944399,9.701923,-6.850838],[8.088311,-9.534996,7.218665,5.482766,8.547771,1.409100,-2.556908,-0.816030,3.405220,-0.248562,-8.762225],[-8.322375,-9.005904,5.269193,1.485688,0.670585,5.741964,-6.476432,-4.017768,-1.645151,-2.455385,7.882457],[5.523980,-5.327435,-9.355885,5.587198,6.836740,-4.372670,-7.176715,2.243203,-8.620213,6.911151,3.434350],[-3.060063,4.424145,8.002932,-4.092831,-3.810343,-6.618186,4.662484,4.979666,-9.677657,-2.567055,-4.489954]], dtype = "float32")#candidate|1545|(13, 11)|const|float32
call_1544 = relay.TupleGetItem(func_1431_call(relay.reshape(const_1545.astype('float32'), [13, 11]), relay.reshape(const_1545.astype('float32'), [13, 11]), ), 0)
call_1546 = relay.TupleGetItem(func_1435_call(relay.reshape(const_1545.astype('float32'), [13, 11]), relay.reshape(const_1545.astype('float32'), [13, 11]), ), 0)
output = relay.Tuple([uop_1521,call_1532,const_1533,const_1534,call_1541,call_1544,const_1545,])
output2 = relay.Tuple([uop_1521,call_1535,const_1533,const_1534,call_1542,call_1546,const_1545,])
func_1567 = relay.Function([var_1516,var_1517,], output)
mod['func_1567'] = func_1567
mod = relay.transform.InferType()(mod)
mutated_mod['func_1567'] = func_1567
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1567_call = mutated_mod.get_global_var('func_1567')
var_1569 = relay.var("var_1569", dtype = "float32", shape = ())#candidate|1569|()|var|float32
var_1570 = relay.var("var_1570", dtype = "float32", shape = (5, 10, 8))#candidate|1570|(5, 10, 8)|var|float32
call_1568 = func_1567_call(var_1569,var_1570,)
output = call_1568
func_1571 = relay.Function([var_1569,var_1570,], output)
mutated_mod['func_1571'] = func_1571
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1056_call = mod.get_global_var('func_1056')
func_1058_call = mutated_mod.get_global_var('func_1058')
call_1579 = func_1056_call()
call_1580 = func_1056_call()
output = relay.Tuple([call_1579,])
output2 = relay.Tuple([call_1580,])
func_1588 = relay.Function([], output)
mod['func_1588'] = func_1588
mod = relay.transform.InferType()(mod)
mutated_mod['func_1588'] = func_1588
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1588_call = mutated_mod.get_global_var('func_1588')
call_1589 = func_1588_call()
output = call_1589
func_1590 = relay.Function([], output)
mutated_mod['func_1590'] = func_1590
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1588_call = mod.get_global_var('func_1588')
func_1590_call = mutated_mod.get_global_var('func_1590')
call_1625 = relay.TupleGetItem(func_1588_call(), 0)
call_1626 = relay.TupleGetItem(func_1590_call(), 0)
func_1588_call = mod.get_global_var('func_1588')
func_1590_call = mutated_mod.get_global_var('func_1590')
call_1629 = relay.TupleGetItem(func_1588_call(), 0)
call_1630 = relay.TupleGetItem(func_1590_call(), 0)
func_981_call = mod.get_global_var('func_981')
func_982_call = mutated_mod.get_global_var('func_982')
call_1643 = func_981_call()
call_1644 = func_981_call()
output = relay.Tuple([call_1625,call_1629,call_1643,])
output2 = relay.Tuple([call_1626,call_1630,call_1644,])
func_1646 = relay.Function([], output)
mod['func_1646'] = func_1646
mod = relay.transform.InferType()(mod)
mutated_mod['func_1646'] = func_1646
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1646_call = mutated_mod.get_global_var('func_1646')
call_1647 = func_1646_call()
output = call_1647
func_1648 = relay.Function([], output)
mutated_mod['func_1648'] = func_1648
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1188_call = mod.get_global_var('func_1188')
func_1190_call = mutated_mod.get_global_var('func_1190')
call_1651 = func_1188_call()
call_1652 = func_1188_call()
func_1567_call = mod.get_global_var('func_1567')
func_1571_call = mutated_mod.get_global_var('func_1571')
const_1657 = relay.const(-5.949638, dtype = "float32")#candidate|1657|()|const|float32
const_1658 = relay.const([5.163414,7.809974,8.736646,-5.538334,-3.578616,-4.601291,6.542511,-5.487242,-6.394846,4.189544,-9.881859,8.823536,9.128470,-8.782752,6.185234,-0.918607,4.710702,-7.413610,4.881735,-3.771550,-5.538983,5.804150,9.121034,-5.527677,-2.664335,2.662147,-3.652124,2.443373,-1.795863,-5.366586,-8.919438,-1.269805,-0.548243,-1.978509,3.914802,4.227776,6.490234,0.913118,3.183294,-5.527358,-9.334718,-8.413324,9.106704,-3.373202,0.768257,8.223389,1.503959,-9.679671,9.865800,5.818104,7.048752,-3.034025,-9.111745,-1.212603,-7.150744,4.891289,-6.036063,2.646947,5.313812,3.636975,-2.334288,-7.739233,3.999800,2.891109,-3.007967,0.261259,2.656846,-7.139601,-5.854511,4.768141,-3.374443,0.198861,3.602750,-5.430610,-4.733063,8.614822,-0.849058,-1.260206,-6.802971,-3.663842,-8.679545,2.551361,-5.804075,2.145917,-3.577714,0.047314,-2.514974,9.344041,2.283097,-4.113093,4.922963,-9.391015,-0.836765,7.829099,-6.598155,2.424083,-0.362994,-8.406229,-0.601267,7.147246,1.732779,0.988926,9.630947,0.887575,-8.993238,-3.944349,9.157586,-9.444561,2.607567,6.166596,3.737373,-3.458484,2.707090,-4.442259,2.793780,-9.429983,-9.982204,-4.347007,-6.278109,-7.768687,8.003621,-9.507507,0.645602,-8.281965,2.832371,-2.125824,-6.428253,-1.367767,-5.258131,-7.212400,-5.314384,-2.104796,-2.563478,-7.980028,-1.659552,-7.452587,1.539169,-9.393231,0.714176,4.086145,-0.254606,3.111518,-9.797818,-2.903901,8.497427,-0.906189,2.576163,-7.526544,8.319369,-1.988503,-4.953976,-7.708698,-1.659607,-3.772234,5.127125,-5.729381,-2.339461,2.305092,8.715487,5.592718,-9.889691,-6.065373,-6.056067,3.494920,1.307995,8.541796,-9.380224,8.966487,-6.058654,-2.788607,8.858029,-2.974552,7.217804,-8.037331,0.341591,3.134384,-7.645937,-6.861795,-3.529097,-0.177811,1.372554,4.980780,7.419769,-0.512275,2.422702,-2.669340,1.320129,-2.919909,-0.125539,5.049543,-8.560758,2.142237,-0.651526,8.261599,3.467077,-5.927122,8.474375,5.505948,-8.514249,-6.409274,-7.010539,-4.713875,7.462643,-9.180803,-3.029446,-2.157364,-5.544950,2.631647,8.089471,-7.790633,-3.299510,4.559001,-5.456959,3.394706,0.499441,-4.668843,-5.523199,1.375312,7.085602,1.307459,-8.949359,-9.479963,0.099558,6.201283,-9.013800,1.937915,4.296096,-1.275574,9.576492,-5.437970,-6.503555,8.134744,-3.728993,-9.320298,-7.088246,2.565821,4.786793,-5.412019,8.951571,2.110898,6.684114,-3.129325,-0.194495,-1.163943,-4.976809,-7.307350,-4.098178,7.168121,-1.076065,-4.977838,4.505684,8.240250,-4.565146,9.862018,-8.093722,4.334834,-4.294686,-4.653874,5.341154,-9.172384,0.904776,-1.547832,-8.493691,-3.680794,7.421851,-0.047195,-2.594168,-5.023766,9.020741,-2.569138,-6.252815,5.007684,-6.668416,6.504988,-2.959139,7.993727,-5.074050,3.901013,-3.469085,-4.683543,6.760545,-5.581090,-7.915223,-9.982169,9.442461,7.635950,-0.602625,-2.694404,-3.835963,-9.617923,4.015179,4.509287,2.407917,2.105686,-7.185474,-5.036436,-4.444685,-4.181205,7.445865,-5.237213,2.353304,-1.092412,-9.621625,-5.033814,5.576046,1.449637,6.396720,-5.053091,-7.157817,-2.133198,2.075958,-4.234584,8.281997,0.543306,2.812830,1.040097,1.428333,-1.224978,-6.187266,1.905924,-8.452384,3.953124,6.299250,-8.571353,8.046193,-8.499918,4.702185,-6.198909,7.898452,5.517361,9.191846,-9.856355,0.388112,-5.145520,5.794762,7.262727,-9.563385,2.010583,-0.321748,5.475897,-0.522813,6.624274,-7.047404,6.811806,2.007145,-8.531767,7.243215,7.794461,-6.556127,8.053677,8.510377,6.734929,4.877816,7.523174,-3.425844,6.383767,4.694952,7.729515,-6.771006,1.644855,5.673297,-7.105401,-2.106583,6.534596,5.986012,0.556454,9.532024,-8.001931,4.789829,4.485385,4.008381,0.635559,2.623413,-4.256267,4.387365,8.009481,-0.888717,8.289086,7.116464,4.970102,2.906599,-2.832371,4.717293,6.617956,-8.960784,-9.032025,3.426996,-1.630130,-3.154592,-3.177156,6.538767,7.366737,3.005127,-0.428988,5.725802,6.514673,-0.977155,-6.028675,5.032569,5.084951], dtype = "float32")#candidate|1658|(400,)|const|float32
call_1656 = relay.TupleGetItem(func_1567_call(relay.reshape(const_1657.astype('float32'), []), relay.reshape(const_1658.astype('float32'), [5, 10, 8]), ), 4)
call_1659 = relay.TupleGetItem(func_1571_call(relay.reshape(const_1657.astype('float32'), []), relay.reshape(const_1658.astype('float32'), [5, 10, 8]), ), 4)
output = relay.Tuple([call_1651,call_1656,const_1657,const_1658,])
output2 = relay.Tuple([call_1652,call_1659,const_1657,const_1658,])
func_1673 = relay.Function([], output)
mod['func_1673'] = func_1673
mod = relay.transform.InferType()(mod)
output = func_1673()
func_1674 = relay.Function([], output)
mutated_mod['func_1674'] = func_1674
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1646_call = mod.get_global_var('func_1646')
func_1648_call = mutated_mod.get_global_var('func_1648')
call_1684 = relay.TupleGetItem(func_1646_call(), 0)
call_1685 = relay.TupleGetItem(func_1648_call(), 0)
output = call_1684
output2 = call_1685
func_1694 = relay.Function([], output)
mod['func_1694'] = func_1694
mod = relay.transform.InferType()(mod)
output = func_1694()
func_1695 = relay.Function([], output)
mutated_mod['func_1695'] = func_1695
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1646_call = mod.get_global_var('func_1646')
func_1648_call = mutated_mod.get_global_var('func_1648')
call_1696 = relay.TupleGetItem(func_1646_call(), 0)
call_1697 = relay.TupleGetItem(func_1648_call(), 0)
func_898_call = mod.get_global_var('func_898')
func_901_call = mutated_mod.get_global_var('func_901')
const_1727 = relay.const(3, dtype = "uint64")#candidate|1727|()|const|uint64
call_1726 = relay.TupleGetItem(func_898_call(relay.reshape(const_1727.astype('uint64'), [])), 0)
call_1728 = relay.TupleGetItem(func_901_call(relay.reshape(const_1727.astype('uint64'), [])), 0)
output = relay.Tuple([call_1696,call_1726,const_1727,])
output2 = relay.Tuple([call_1697,call_1728,const_1727,])
func_1731 = relay.Function([], output)
mod['func_1731'] = func_1731
mod = relay.transform.InferType()(mod)
output = func_1731()
func_1732 = relay.Function([], output)
mutated_mod['func_1732'] = func_1732
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1056_call = mod.get_global_var('func_1056')
func_1058_call = mutated_mod.get_global_var('func_1058')
call_1768 = func_1056_call()
call_1769 = func_1056_call()
var_1775 = relay.var("var_1775", dtype = "float32", shape = (8, 5))#candidate|1775|(8, 5)|var|float32
bop_1776 = relay.bitwise_and(call_1768.astype('uint32'), relay.reshape(var_1775.astype('uint32'), relay.shape_of(call_1768))) # shape=(8, 5)
bop_1779 = relay.bitwise_and(call_1769.astype('uint32'), relay.reshape(var_1775.astype('uint32'), relay.shape_of(call_1769))) # shape=(8, 5)
func_1313_call = mod.get_global_var('func_1313')
func_1315_call = mutated_mod.get_global_var('func_1315')
var_1787 = relay.var("var_1787", dtype = "float32", shape = (350,))#candidate|1787|(350,)|var|float32
call_1786 = relay.TupleGetItem(func_1313_call(relay.reshape(var_1787.astype('float32'), [350,])), 1)
call_1788 = relay.TupleGetItem(func_1315_call(relay.reshape(var_1787.astype('float32'), [350,])), 1)
bop_1800 = relay.maximum(bop_1776.astype('uint16'), relay.reshape(call_1768.astype('uint16'), relay.shape_of(bop_1776))) # shape=(8, 5)
bop_1803 = relay.maximum(bop_1779.astype('uint16'), relay.reshape(call_1769.astype('uint16'), relay.shape_of(bop_1779))) # shape=(8, 5)
uop_1805 = relay.atan(bop_1800.astype('float64')) # shape=(8, 5)
uop_1807 = relay.atan(bop_1803.astype('float64')) # shape=(8, 5)
bop_1810 = relay.subtract(uop_1805.astype('int8'), relay.reshape(bop_1776.astype('int8'), relay.shape_of(uop_1805))) # shape=(8, 5)
bop_1813 = relay.subtract(uop_1807.astype('int8'), relay.reshape(bop_1779.astype('int8'), relay.shape_of(uop_1807))) # shape=(8, 5)
output = relay.Tuple([call_1786,var_1787,bop_1810,])
output2 = relay.Tuple([call_1788,var_1787,bop_1813,])
func_1837 = relay.Function([var_1775,var_1787,], output)
mod['func_1837'] = func_1837
mod = relay.transform.InferType()(mod)
mutated_mod['func_1837'] = func_1837
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1837_call = mutated_mod.get_global_var('func_1837')
var_1839 = relay.var("var_1839", dtype = "float32", shape = (8, 5))#candidate|1839|(8, 5)|var|float32
var_1840 = relay.var("var_1840", dtype = "float32", shape = (350,))#candidate|1840|(350,)|var|float32
call_1838 = func_1837_call(var_1839,var_1840,)
output = call_1838
func_1841 = relay.Function([var_1839,var_1840,], output)
mutated_mod['func_1841'] = func_1841
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1845 = relay.var("var_1845", dtype = "uint64", shape = (16, 4, 5))#candidate|1845|(16, 4, 5)|var|uint64
var_1846 = relay.var("var_1846", dtype = "uint64", shape = (16, 4, 5))#candidate|1846|(16, 4, 5)|var|uint64
bop_1847 = relay.bitwise_xor(var_1845.astype('uint64'), relay.reshape(var_1846.astype('uint64'), relay.shape_of(var_1845))) # shape=(16, 4, 5)
func_1731_call = mod.get_global_var('func_1731')
func_1732_call = mutated_mod.get_global_var('func_1732')
call_1853 = relay.TupleGetItem(func_1731_call(), 1)
call_1854 = relay.TupleGetItem(func_1732_call(), 1)
bop_1856 = relay.add(var_1845.astype('uint32'), relay.reshape(bop_1847.astype('uint32'), relay.shape_of(var_1845))) # shape=(16, 4, 5)
bop_1859 = relay.floor_divide(var_1846.astype('float32'), relay.reshape(bop_1847.astype('float32'), relay.shape_of(var_1846))) # shape=(16, 4, 5)
output = relay.Tuple([call_1853,bop_1856,bop_1859,])
output2 = relay.Tuple([call_1854,bop_1856,bop_1859,])
func_1862 = relay.Function([var_1845,var_1846,], output)
mod['func_1862'] = func_1862
mod = relay.transform.InferType()(mod)
var_1863 = relay.var("var_1863", dtype = "uint64", shape = (16, 4, 5))#candidate|1863|(16, 4, 5)|var|uint64
var_1864 = relay.var("var_1864", dtype = "uint64", shape = (16, 4, 5))#candidate|1864|(16, 4, 5)|var|uint64
output = func_1862(var_1863,var_1864,)
func_1865 = relay.Function([var_1863,var_1864,], output)
mutated_mod['func_1865'] = func_1865
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1894 = relay.var("var_1894", dtype = "int8", shape = (15,))#candidate|1894|(15,)|var|int8
var_1895 = relay.var("var_1895", dtype = "int8", shape = (15,))#candidate|1895|(15,)|var|int8
bop_1896 = relay.equal(var_1894.astype('bool'), relay.reshape(var_1895.astype('bool'), relay.shape_of(var_1894))) # shape=(15,)
output = relay.Tuple([bop_1896,])
output2 = relay.Tuple([bop_1896,])
func_1904 = relay.Function([var_1894,var_1895,], output)
mod['func_1904'] = func_1904
mod = relay.transform.InferType()(mod)
mutated_mod['func_1904'] = func_1904
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1904_call = mutated_mod.get_global_var('func_1904')
var_1906 = relay.var("var_1906", dtype = "int8", shape = (15,))#candidate|1906|(15,)|var|int8
var_1907 = relay.var("var_1907", dtype = "int8", shape = (15,))#candidate|1907|(15,)|var|int8
call_1905 = func_1904_call(var_1906,var_1907,)
output = call_1905
func_1908 = relay.Function([var_1906,var_1907,], output)
mutated_mod['func_1908'] = func_1908
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1056_call = mod.get_global_var('func_1056')
func_1058_call = mutated_mod.get_global_var('func_1058')
call_1943 = func_1056_call()
call_1944 = func_1056_call()
uop_1969 = relay.log(call_1943.astype('float32')) # shape=(8, 5)
uop_1971 = relay.log(call_1944.astype('float32')) # shape=(8, 5)
const_1973 = relay.const([[-3.827179,5.915282,7.467094,-8.898543,-3.899113],[-9.560405,-7.153954,-3.383496,7.362218,4.934290],[0.556274,-5.179657,-3.736389,4.664377,2.018849],[4.187059,4.798767,5.258532,-0.863958,-6.998614],[4.752675,5.565760,8.925543,1.209473,7.410315],[-4.491494,-0.241054,5.119869,-6.917226,7.270132],[1.266108,0.598635,2.750299,4.648519,-3.112652],[2.500286,-4.788898,8.834924,-2.649693,-0.207501]], dtype = "float32")#candidate|1973|(8, 5)|const|float32
bop_1974 = relay.logical_or(uop_1969.astype('bool'), relay.reshape(const_1973.astype('bool'), relay.shape_of(uop_1969))) # shape=(8, 5)
bop_1977 = relay.logical_or(uop_1971.astype('bool'), relay.reshape(const_1973.astype('bool'), relay.shape_of(uop_1971))) # shape=(8, 5)
output = bop_1974
output2 = bop_1977
func_1991 = relay.Function([], output)
mod['func_1991'] = func_1991
mod = relay.transform.InferType()(mod)
mutated_mod['func_1991'] = func_1991
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1991_call = mutated_mod.get_global_var('func_1991')
call_1992 = func_1991_call()
output = call_1992
func_1993 = relay.Function([], output)
mutated_mod['func_1993'] = func_1993
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1994 = relay.var("var_1994", dtype = "float64", shape = (12, 12))#candidate|1994|(12, 12)|var|float64
var_1995 = relay.var("var_1995", dtype = "float64", shape = (12, 12))#candidate|1995|(12, 12)|var|float64
bop_1996 = relay.divide(var_1994.astype('float64'), relay.reshape(var_1995.astype('float64'), relay.shape_of(var_1994))) # shape=(12, 12)
uop_1999 = relay.atan(bop_1996.astype('float32')) # shape=(12, 12)
bop_2003 = relay.not_equal(uop_1999.astype('bool'), relay.reshape(bop_1996.astype('bool'), relay.shape_of(uop_1999))) # shape=(12, 12)
output = relay.Tuple([bop_2003,])
output2 = relay.Tuple([bop_2003,])
func_2006 = relay.Function([var_1994,var_1995,], output)
mod['func_2006'] = func_2006
mod = relay.transform.InferType()(mod)
mutated_mod['func_2006'] = func_2006
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2006_call = mutated_mod.get_global_var('func_2006')
var_2008 = relay.var("var_2008", dtype = "float64", shape = (12, 12))#candidate|2008|(12, 12)|var|float64
var_2009 = relay.var("var_2009", dtype = "float64", shape = (12, 12))#candidate|2009|(12, 12)|var|float64
call_2007 = func_2006_call(var_2008,var_2009,)
output = call_2007
func_2010 = relay.Function([var_2008,var_2009,], output)
mutated_mod['func_2010'] = func_2010
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1694_call = mod.get_global_var('func_1694')
func_1695_call = mutated_mod.get_global_var('func_1695')
call_2018 = func_1694_call()
call_2019 = func_1694_call()
output = relay.Tuple([call_2018,])
output2 = relay.Tuple([call_2019,])
func_2020 = relay.Function([], output)
mod['func_2020'] = func_2020
mod = relay.transform.InferType()(mod)
output = func_2020()
func_2021 = relay.Function([], output)
mutated_mod['func_2021'] = func_2021
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1673_call = mod.get_global_var('func_1673')
func_1674_call = mutated_mod.get_global_var('func_1674')
call_2032 = relay.TupleGetItem(func_1673_call(), 3)
call_2033 = relay.TupleGetItem(func_1674_call(), 3)
func_1463_call = mod.get_global_var('func_1463')
func_1465_call = mutated_mod.get_global_var('func_1465')
var_2039 = relay.var("var_2039", dtype = "int64", shape = (195,))#candidate|2039|(195,)|var|int64
call_2038 = relay.TupleGetItem(func_1463_call(relay.reshape(var_2039.astype('int64'), [195,])), 0)
call_2040 = relay.TupleGetItem(func_1465_call(relay.reshape(var_2039.astype('int64'), [195,])), 0)
func_1991_call = mod.get_global_var('func_1991')
func_1993_call = mutated_mod.get_global_var('func_1993')
call_2055 = func_1991_call()
call_2056 = func_1991_call()
func_1510_call = mod.get_global_var('func_1510')
func_1514_call = mutated_mod.get_global_var('func_1514')
var_2060 = relay.var("var_2060", dtype = "uint16", shape = (6,))#candidate|2060|(6,)|var|uint16
var_2061 = relay.var("var_2061", dtype = "uint16", shape = (24,))#candidate|2061|(24,)|var|uint16
call_2059 = relay.TupleGetItem(func_1510_call(relay.reshape(call_2038.astype('float32'), [8, 5]), relay.reshape(var_2060.astype('uint16'), [6,]), relay.reshape(var_2061.astype('uint16'), [24,]), ), 5)
call_2062 = relay.TupleGetItem(func_1514_call(relay.reshape(call_2038.astype('float32'), [8, 5]), relay.reshape(var_2060.astype('uint16'), [6,]), relay.reshape(var_2061.astype('uint16'), [24,]), ), 5)
output = relay.Tuple([call_2032,call_2038,var_2039,call_2055,call_2059,var_2060,var_2061,])
output2 = relay.Tuple([call_2033,call_2040,var_2039,call_2056,call_2062,var_2060,var_2061,])
func_2064 = relay.Function([var_2039,var_2060,var_2061,], output)
mod['func_2064'] = func_2064
mod = relay.transform.InferType()(mod)
mutated_mod['func_2064'] = func_2064
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2064_call = mutated_mod.get_global_var('func_2064')
var_2066 = relay.var("var_2066", dtype = "int64", shape = (195,))#candidate|2066|(195,)|var|int64
var_2067 = relay.var("var_2067", dtype = "uint16", shape = (6,))#candidate|2067|(6,)|var|uint16
var_2068 = relay.var("var_2068", dtype = "uint16", shape = (24,))#candidate|2068|(24,)|var|uint16
call_2065 = func_2064_call(var_2066,var_2067,var_2068,)
output = call_2065
func_2069 = relay.Function([var_2066,var_2067,var_2068,], output)
mutated_mod['func_2069'] = func_2069
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1673_call = mod.get_global_var('func_1673')
func_1674_call = mutated_mod.get_global_var('func_1674')
call_2078 = relay.TupleGetItem(func_1673_call(), 3)
call_2079 = relay.TupleGetItem(func_1674_call(), 3)
func_946_call = mod.get_global_var('func_946')
func_948_call = mutated_mod.get_global_var('func_948')
const_2086 = relay.const([-7.032074,-2.854317,-6.778249,-1.546690,2.372384,-8.021720,-8.130266,-1.244932,0.394714,3.991634,-9.755824,2.853759,7.035511,-2.091428,-3.812897,-5.770846,-2.666613,-6.090744,9.731038,6.239885,-7.426732,-0.642985,-9.904408,-1.944261,4.162034,-4.293483,-8.694016,-3.892685,-0.098467,-1.103862,-8.574296,4.837746,0.548164,-4.520468,-9.943297,8.421232,7.928794,-4.978058,-3.284641,-0.658726], dtype = "float32")#candidate|2086|(40,)|const|float32
call_2085 = func_946_call(relay.reshape(const_2086.astype('float32'), [8, 5]))
call_2087 = func_946_call(relay.reshape(const_2086.astype('float32'), [8, 5]))
func_823_call = mod.get_global_var('func_823')
func_826_call = mutated_mod.get_global_var('func_826')
var_2092 = relay.var("var_2092", dtype = "float32", shape = (10,))#candidate|2092|(10,)|var|float32
call_2091 = func_823_call(relay.reshape(var_2092.astype('float32'), [10, 1]))
call_2093 = func_823_call(relay.reshape(var_2092.astype('float32'), [10, 1]))
var_2101 = relay.var("var_2101", dtype = "float32", shape = (10, 11))#candidate|2101|(10, 11)|var|float32
bop_2102 = relay.power(call_2091.astype('float32'), var_2101.astype('float32')) # shape=(10, 11)
bop_2105 = relay.power(call_2093.astype('float32'), var_2101.astype('float32')) # shape=(10, 11)
func_439_call = mod.get_global_var('func_439')
func_442_call = mutated_mod.get_global_var('func_442')
const_2110 = relay.const([-3,5,5,10,-8,-10,8,1,-6,-9,-7,9,7,-7,5,9,3,4,8,5,8,7,5,-10,-4,-3,7,7,5,-4,-7,8,7,-10,-5,4,-10,-9,-8,-5,-10,-5,4,3,-5,-10,2,-4,10,-7,-9,-6,10,4,-8,2,9,9,9,-3,-8,9,-2,10,-9,5,-6,8,10,-5,1,8,-3,3,6,-8,-10,-6,-2,-2,-1,-10,2,2,10,-10,-5,10,5,1,1,8,8,4,8,2,7,-6,-10,8,-10,-8,7,4,-3,-3,-6,-8,5,-4,-9,4,-8,2,-3,-5,-1,5,-5,9,-5,2,5,4,1,-3,-4,-6,-9,2,1,2,-2,-9,7,-5,-2,2,9,7,5,10,-9,4,7,8,3,7,9,4,-3,8,8,-1,7,-6,-10,-2,5,6,1,-7,6,-2,9,2,-5,-7,2,9,-8,2,3,6,1,9,8,8,1,10,8,-4,-4,-5,9,-8,10,4,5,3,-3,-9,2,6,1], dtype = "int64")#candidate|2110|(195,)|const|int64
call_2109 = func_439_call(relay.reshape(const_2110.astype('int64'), [15, 13]), relay.reshape(const_2110.astype('int64'), [15, 13]), )
call_2111 = func_439_call(relay.reshape(const_2110.astype('int64'), [15, 13]), relay.reshape(const_2110.astype('int64'), [15, 13]), )
output = relay.Tuple([call_2078,call_2085,const_2086,var_2092,bop_2102,call_2109,const_2110,])
output2 = relay.Tuple([call_2079,call_2087,const_2086,var_2092,bop_2105,call_2111,const_2110,])
func_2112 = relay.Function([var_2092,var_2101,], output)
mod['func_2112'] = func_2112
mod = relay.transform.InferType()(mod)
var_2113 = relay.var("var_2113", dtype = "float32", shape = (10,))#candidate|2113|(10,)|var|float32
var_2114 = relay.var("var_2114", dtype = "float32", shape = (10, 11))#candidate|2114|(10, 11)|var|float32
output = func_2112(var_2113,var_2114,)
func_2115 = relay.Function([var_2113,var_2114,], output)
mutated_mod['func_2115'] = func_2115
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1056_call = mod.get_global_var('func_1056')
func_1058_call = mutated_mod.get_global_var('func_1058')
call_2127 = func_1056_call()
call_2128 = func_1056_call()
const_2135 = relay.const([[-4.466125,-0.269490,-8.192643,9.037415,-7.677573],[-9.609777,-5.893708,-5.507421,-2.467859,8.529156],[-1.740235,-0.714244,5.463852,-2.490946,-3.617382],[2.663814,-9.513906,-7.725976,9.592629,-6.654847],[1.319597,7.027611,6.971548,-5.216053,-4.492211],[-5.550230,-4.024513,3.238801,-3.858275,-7.267355],[7.440933,3.217425,1.585771,-3.154309,9.666956],[-4.553472,-7.621279,-7.512898,2.635311,8.089086]], dtype = "float32")#candidate|2135|(8, 5)|const|float32
bop_2136 = relay.not_equal(call_2127.astype('bool'), relay.reshape(const_2135.astype('bool'), relay.shape_of(call_2127))) # shape=(8, 5)
bop_2139 = relay.not_equal(call_2128.astype('bool'), relay.reshape(const_2135.astype('bool'), relay.shape_of(call_2128))) # shape=(8, 5)
output = relay.Tuple([bop_2136,])
output2 = relay.Tuple([bop_2139,])
func_2143 = relay.Function([], output)
mod['func_2143'] = func_2143
mod = relay.transform.InferType()(mod)
output = func_2143()
func_2144 = relay.Function([], output)
mutated_mod['func_2144'] = func_2144
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2020_call = mod.get_global_var('func_2020')
func_2021_call = mutated_mod.get_global_var('func_2021')
call_2327 = relay.TupleGetItem(func_2020_call(), 0)
call_2328 = relay.TupleGetItem(func_2021_call(), 0)
func_292_call = mod.get_global_var('func_292')
func_296_call = mutated_mod.get_global_var('func_296')
var_2339 = relay.var("var_2339", dtype = "uint8", shape = (1, 126))#candidate|2339|(1, 126)|var|uint8
const_2340 = relay.const([1.058633,2.466089,-4.335482], dtype = "float64")#candidate|2340|(3,)|const|float64
call_2338 = relay.TupleGetItem(func_292_call(relay.reshape(var_2339.astype('uint8'), [9, 14]), relay.reshape(var_2339.astype('uint8'), [9, 14]), relay.reshape(const_2340.astype('float64'), [3,]), ), 2)
call_2341 = relay.TupleGetItem(func_296_call(relay.reshape(var_2339.astype('uint8'), [9, 14]), relay.reshape(var_2339.astype('uint8'), [9, 14]), relay.reshape(const_2340.astype('float64'), [3,]), ), 2)
func_2006_call = mod.get_global_var('func_2006')
func_2010_call = mutated_mod.get_global_var('func_2010')
var_2348 = relay.var("var_2348", dtype = "float64", shape = (144,))#candidate|2348|(144,)|var|float64
call_2347 = relay.TupleGetItem(func_2006_call(relay.reshape(var_2348.astype('float64'), [12, 12]), relay.reshape(var_2348.astype('float64'), [12, 12]), ), 0)
call_2349 = relay.TupleGetItem(func_2010_call(relay.reshape(var_2348.astype('float64'), [12, 12]), relay.reshape(var_2348.astype('float64'), [12, 12]), ), 0)
const_2358 = relay.const([[4,4,5,5,-2,4,10,2,-8,-6,7,3,-5,-4,-8,4,2,-9,2,2,-9,-9,-8,4,-3,-2,2,8,-6,8,-1,7,-2,2,-2,-9,-1,-1,9,-7,-10,5,-6,-7,7,-8,2,2,4,10,-8,-9,7,-2,-9,-1,7,9,2,10,5,-6,3,-1,-3,5,9,-4,5,-5,4,3,5,-7,-10,-4,4,2,-9,-6,-3,6,6,10,-4,8,-2,-1,3,-2,4,-9,-8,-8,3,1,-5,-10,7,-4,5,-10,1,4,9,8,-1,-2,8,-6,-3,9,-4,-1,-10,-3,-7,4,4,3,-5,7,-7,-7,-10,-8],[-9,-1,7,-5,8,-4,3,-2,-7,10,-7,-9,7,-1,-8,10,-5,2,4,9,-7,7,-2,-7,-4,-10,-6,-7,-5,-2,5,-10,-2,10,-10,-7,1,-5,3,-1,-8,9,-10,-9,-3,7,10,8,-10,9,9,-6,-10,-3,-4,-9,7,-8,-4,-1,-7,-7,-6,10,-7,-5,-5,-7,2,5,6,8,-3,10,5,-8,-2,1,4,-3,2,-7,8,1,4,8,3,-3,4,6,5,-2,-1,-6,-6,-5,-8,5,-6,10,4,2,-3,4,8,-5,-5,3,-1,2,7,-8,-2,5,-1,8,10,-3,-4,-10,-10,-9,-3,-3,-9,-4],[6,5,4,6,4,1,3,-7,-4,3,-4,-8,-5,3,-8,2,8,-2,9,10,-1,5,6,4,1,-3,-10,-3,-2,8,-5,5,-10,-8,6,-10,10,4,-9,5,4,6,8,8,8,-9,-6,8,3,5,-5,-8,-2,10,-1,-3,4,-8,1,-10,1,1,6,-5,5,-10,-5,-3,3,10,10,-8,7,5,-3,6,-6,3,-4,3,-6,-10,-3,2,8,-9,-7,1,4,-1,-2,-3,2,2,-2,-9,-8,-3,-4,9,7,10,-7,-8,-8,-10,-6,1,-8,-7,-6,1,3,-6,-9,1,-5,1,-4,2,10,4,-6,-5,6,5],[6,7,-1,10,-10,-7,-2,4,9,-7,2,6,-6,10,-3,-5,2,2,-5,-5,-1,-9,-1,8,10,1,-9,-10,8,-7,7,-3,4,-4,8,-5,10,-4,6,2,-9,8,10,5,7,2,-5,-5,-10,1,6,7,-9,-10,-4,3,-5,-3,8,-4,1,-1,1,-9,-4,-9,6,8,-6,2,-1,5,-5,-3,-8,10,-5,8,-6,8,-7,10,-2,1,-9,10,-10,-7,3,10,7,1,-8,9,-7,-3,2,10,-6,-4,7,-6,-8,4,3,1,1,-3,2,-4,-8,5,-2,6,7,-2,3,-3,-1,2,-8,-5,4,9,-1,-7],[10,1,-9,10,8,-2,10,-1,2,-4,-6,-3,-6,-10,-7,-5,1,9,-2,9,8,10,9,-10,7,-5,-2,5,1,8,4,-5,9,6,10,3,7,8,7,6,2,8,10,-6,1,-8,-7,-8,6,2,-9,6,10,-9,3,-2,-7,-1,-6,-9,-2,9,-10,10,3,4,6,-6,-3,5,2,-1,10,7,2,2,-3,-5,-10,-10,6,-4,3,9,7,-2,-3,-10,-6,8,5,1,-9,-10,1,1,-6,-10,-5,5,-6,2,1,-2,-3,6,-8,-3,-5,-7,6,6,3,1,6,3,-6,-1,9,5,-2,7,9,-9,6,3],[-6,-4,-4,10,-2,1,-7,4,-4,5,1,7,3,-6,-4,1,-7,-2,9,-10,2,-8,4,5,2,-3,4,-2,-8,8,-7,-3,-5,5,6,4,7,-6,-5,-8,-3,-7,-10,7,-9,-9,5,1,6,3,-1,9,2,-8,10,-1,6,-1,-1,4,5,-6,-7,2,5,7,-4,-8,4,3,-2,7,6,-6,-8,7,-6,-4,2,-8,3,4,2,4,-8,9,3,9,6,6,-6,5,-6,6,-7,4,8,7,-2,5,10,-1,10,4,-5,-9,-3,-4,-5,6,5,-2,-10,4,-2,1,2,4,-3,-9,-9,1,3,1,-8,-2],[9,-7,-4,2,-9,-7,6,-5,2,-7,-10,-2,-4,3,-10,-6,-10,7,-3,10,-1,-4,-8,5,-3,-10,-1,-8,-5,-8,7,-3,6,-8,10,-2,5,-9,-5,5,-9,10,7,-6,10,-7,-7,-3,2,-3,-1,5,-2,-1,1,4,-10,1,-4,-10,-10,-3,-9,3,8,-6,-2,7,7,3,-5,-9,3,4,8,4,-3,6,-1,-3,-6,10,-10,-4,-6,-7,3,-7,6,4,9,2,2,4,-9,-6,8,3,-4,6,-1,2,9,-10,3,10,9,4,2,-9,-6,5,9,-5,4,-5,-4,-9,-8,-9,1,3,-3,-5,4,5],[-7,8,-8,-10,-4,-7,4,5,-5,1,6,-7,-10,4,-6,-5,-9,10,-8,2,-7,-2,5,1,4,5,7,-8,6,-2,-1,5,2,9,-8,-1,10,-6,-5,-2,-2,-10,10,-7,-10,-9,9,-2,9,-10,4,2,9,8,4,2,8,-4,-9,-5,1,3,5,-1,-1,-3,-6,-9,-2,10,1,-1,-3,-7,9,5,-5,-10,-6,-9,5,1,5,-8,-1,3,10,10,3,10,-1,-10,5,-9,4,-7,-9,10,-2,-3,7,-4,-1,10,4,-4,5,3,7,-1,1,-9,8,-4,-10,-1,4,3,10,9,-3,10,10,-8,-7,-7],[-8,-9,-10,5,2,6,-3,-3,-4,8,-7,7,-4,-6,5,1,2,10,-5,-9,-4,6,5,2,-2,-7,9,-8,-1,5,-8,1,-6,-9,7,1,6,-9,-7,-10,-1,9,3,2,4,1,-4,3,5,-3,1,-4,-1,7,3,-6,6,-4,-7,-4,4,2,-10,-9,6,-9,-8,-3,5,-9,-7,3,-2,-4,-3,-9,-6,6,7,-2,7,-1,2,6,-6,-8,-1,8,-9,-2,1,-3,2,2,5,2,3,9,-9,-7,-1,8,7,5,8,-10,9,-3,-3,2,-9,1,10,-1,6,-2,2,-1,7,5,-3,4,8,-5,-4,-8]], dtype = "uint8")#candidate|2358|(9, 126)|const|uint8
bop_2359 = relay.logical_xor(var_2339.astype('uint16'), const_2358.astype('uint16')) # shape=(9, 126)
func_898_call = mod.get_global_var('func_898')
func_901_call = mutated_mod.get_global_var('func_901')
var_2372 = relay.var("var_2372", dtype = "uint64", shape = ())#candidate|2372|()|var|uint64
call_2371 = relay.TupleGetItem(func_898_call(relay.reshape(var_2372.astype('uint64'), [])), 1)
call_2373 = relay.TupleGetItem(func_901_call(relay.reshape(var_2372.astype('uint64'), [])), 1)
func_946_call = mod.get_global_var('func_946')
func_948_call = mutated_mod.get_global_var('func_948')
call_2380 = func_946_call(relay.reshape(call_2327.astype('float32'), [8, 5]))
call_2381 = func_946_call(relay.reshape(call_2327.astype('float32'), [8, 5]))
output = relay.Tuple([call_2327,call_2338,const_2340,call_2347,var_2348,bop_2359,call_2371,var_2372,call_2380,])
output2 = relay.Tuple([call_2328,call_2341,const_2340,call_2349,var_2348,bop_2359,call_2373,var_2372,call_2381,])
func_2382 = relay.Function([var_2339,var_2348,var_2372,], output)
mod['func_2382'] = func_2382
mod = relay.transform.InferType()(mod)
var_2383 = relay.var("var_2383", dtype = "uint8", shape = (1, 126))#candidate|2383|(1, 126)|var|uint8
var_2384 = relay.var("var_2384", dtype = "float64", shape = (144,))#candidate|2384|(144,)|var|float64
var_2385 = relay.var("var_2385", dtype = "uint64", shape = ())#candidate|2385|()|var|uint64
output = func_2382(var_2383,var_2384,var_2385,)
func_2386 = relay.Function([var_2383,var_2384,var_2385,], output)
mutated_mod['func_2386'] = func_2386
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2396 = relay.var("var_2396", dtype = "int32", shape = (13, 5, 8))#candidate|2396|(13, 5, 8)|var|int32
var_2397 = relay.var("var_2397", dtype = "int32", shape = (13, 5, 8))#candidate|2397|(13, 5, 8)|var|int32
bop_2398 = relay.multiply(var_2396.astype('int32'), relay.reshape(var_2397.astype('int32'), relay.shape_of(var_2396))) # shape=(13, 5, 8)
output = relay.Tuple([bop_2398,])
output2 = relay.Tuple([bop_2398,])
func_2408 = relay.Function([var_2396,var_2397,], output)
mod['func_2408'] = func_2408
mod = relay.transform.InferType()(mod)
mutated_mod['func_2408'] = func_2408
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2408_call = mutated_mod.get_global_var('func_2408')
var_2410 = relay.var("var_2410", dtype = "int32", shape = (13, 5, 8))#candidate|2410|(13, 5, 8)|var|int32
var_2411 = relay.var("var_2411", dtype = "int32", shape = (13, 5, 8))#candidate|2411|(13, 5, 8)|var|int32
call_2409 = func_2408_call(var_2410,var_2411,)
output = call_2409
func_2412 = relay.Function([var_2410,var_2411,], output)
mutated_mod['func_2412'] = func_2412
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2425 = relay.var("var_2425", dtype = "float64", shape = (15, 4, 13))#candidate|2425|(15, 4, 13)|var|float64
uop_2426 = relay.asinh(var_2425.astype('float64')) # shape=(15, 4, 13)
uop_2436 = relay.log(uop_2426.astype('float32')) # shape=(15, 4, 13)
bop_2443 = relay.subtract(uop_2436.astype('uint64'), relay.reshape(uop_2426.astype('uint64'), relay.shape_of(uop_2436))) # shape=(15, 4, 13)
uop_2450 = relay.cos(uop_2436.astype('float64')) # shape=(15, 4, 13)
func_1646_call = mod.get_global_var('func_1646')
func_1648_call = mutated_mod.get_global_var('func_1648')
call_2458 = relay.TupleGetItem(func_1646_call(), 2)
call_2459 = relay.TupleGetItem(func_1648_call(), 2)
uop_2466 = relay.sinh(bop_2443.astype('float32')) # shape=(15, 4, 13)
bop_2468 = relay.multiply(bop_2443.astype('int8'), relay.reshape(uop_2450.astype('int8'), relay.shape_of(bop_2443))) # shape=(15, 4, 13)
bop_2477 = relay.bitwise_or(uop_2436.astype('uint64'), relay.reshape(uop_2426.astype('uint64'), relay.shape_of(uop_2436))) # shape=(15, 4, 13)
uop_2480 = relay.cos(uop_2466.astype('float32')) # shape=(15, 4, 13)
bop_2483 = relay.equal(uop_2480.astype('bool'), relay.reshape(bop_2443.astype('bool'), relay.shape_of(uop_2480))) # shape=(15, 4, 13)
bop_2494 = relay.greater(bop_2483.astype('bool'), relay.reshape(uop_2436.astype('bool'), relay.shape_of(bop_2483))) # shape=(15, 4, 13)
func_1673_call = mod.get_global_var('func_1673')
func_1674_call = mutated_mod.get_global_var('func_1674')
call_2502 = relay.TupleGetItem(func_1673_call(), 0)
call_2503 = relay.TupleGetItem(func_1674_call(), 0)
bop_2507 = relay.left_shift(uop_2450.astype('int64'), relay.reshape(var_2425.astype('int64'), relay.shape_of(uop_2450))) # shape=(15, 4, 13)
bop_2511 = relay.maximum(bop_2494.astype('uint32'), relay.reshape(uop_2426.astype('uint32'), relay.shape_of(bop_2494))) # shape=(15, 4, 13)
output = relay.Tuple([call_2458,bop_2468,bop_2477,call_2502,bop_2507,bop_2511,])
output2 = relay.Tuple([call_2459,bop_2468,bop_2477,call_2503,bop_2507,bop_2511,])
func_2514 = relay.Function([var_2425,], output)
mod['func_2514'] = func_2514
mod = relay.transform.InferType()(mod)
mutated_mod['func_2514'] = func_2514
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2515 = relay.var("var_2515", dtype = "float64", shape = (15, 4, 13))#candidate|2515|(15, 4, 13)|var|float64
func_2514_call = mutated_mod.get_global_var('func_2514')
call_2516 = func_2514_call(var_2515)
output = call_2516
func_2517 = relay.Function([var_2515], output)
mutated_mod['func_2517'] = func_2517
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1646_call = mod.get_global_var('func_1646')
func_1648_call = mutated_mod.get_global_var('func_1648')
call_2574 = relay.TupleGetItem(func_1646_call(), 2)
call_2575 = relay.TupleGetItem(func_1648_call(), 2)
output = call_2574
output2 = call_2575
func_2584 = relay.Function([], output)
mod['func_2584'] = func_2584
mod = relay.transform.InferType()(mod)
output = func_2584()
func_2585 = relay.Function([], output)
mutated_mod['func_2585'] = func_2585
mutated_mod = relay.transform.InferType()(mutated_mod)
func_804_call = mod.get_global_var('func_804')
func_805_call = mutated_mod.get_global_var('func_805')
call_2609 = func_804_call()
call_2610 = func_804_call()
output = relay.Tuple([call_2609,])
output2 = relay.Tuple([call_2610,])
func_2632 = relay.Function([], output)
mod['func_2632'] = func_2632
mod = relay.transform.InferType()(mod)
output = func_2632()
func_2633 = relay.Function([], output)
mutated_mod['func_2633'] = func_2633
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2634 = relay.const(-10, dtype = "int8")#candidate|2634|()|const|int8
const_2635 = relay.const([[[2,8,6,-6,-8,-8,4,10,3,1,10],[-3,3,7,10,9,-3,-8,9,-4,6,7]],[[-9,-8,-6,10,-2,-4,-6,-1,9,10,5],[5,-6,4,-8,4,9,2,10,3,6,-7]],[[10,-1,-5,1,1,-7,-4,10,3,-1,-3],[-8,-8,-5,8,-9,7,7,2,-6,-5,-1]]], dtype = "int8")#candidate|2635|(3, 2, 11)|const|int8
bop_2636 = relay.bitwise_or(const_2634.astype('int8'), const_2635.astype('int8')) # shape=(3, 2, 11)
output = bop_2636
output2 = bop_2636
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